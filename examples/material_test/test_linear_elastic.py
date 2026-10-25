from pathlib import Path

import torch
import torchfea


torch.set_default_device(
    torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
torch.set_default_dtype(torch.float64)


MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'C3D4.inp'


def _scalar(value):
    return float(value.detach().reshape(-1)[0].cpu())


def _equivalent_linear_parameters(neo_material):
    """Convert the Neo-Hookean initial shear/bulk moduli to E and nu."""
    shear_modulus = _scalar(neo_material.mu)
    bulk_modulus = _scalar(neo_material.kappa)
    denominator = 3.0 * bulk_modulus + shear_modulus
    youngs_modulus = 9.0 * bulk_modulus * shear_modulus / denominator
    poisson_ratio = (
        3.0 * bulk_modulus - 2.0 * shear_modulus
    ) / (2.0 * denominator)
    return youngs_modulus, poisson_ratio


def _arruda_boyce_mu(shear_modulus, N):
    """Scale the five-term Arruda--Boyce coefficient to the target G."""
    coefficients = (
        0.5,
        1.0 / (20.0 * N),
        11.0 / (1050.0 * N**2),
        19.0 / (7000.0 * N**3),
        519.0 / (673750.0 * N**4),
    )
    initial_W1 = sum(
        power * coefficient * 3.0**(power - 1)
        for power, coefficient in enumerate(coefficients, start=1)
    )
    # For W(I1bar), the infinitesimal shear modulus is 2*dW/dI1bar.
    return shear_modulus / (2.0 * initial_W1)


def _build_c3d4_pressure_model():
    fem = torchfea.FEA_INP()
    fem.read_inp(MODEL_PATH)
    fe = torchfea.from_inp(fem)

    fe.assembly.add_load(
        torchfea.loads.Pressure(
            instance_name='final_model',
            surface_set='surface_1_All',
            pressure=0.06,
        ),
        name='pressure-1',
    )
    fe.assembly.add_boundary(
        torchfea.boundarys.Boundary_Condition(
            instance_name='final_model',
            set_nodes_name='surface_0_Bottom',
        )
    )

    rp = fe.assembly.add_reference_point(torchfea.ReferencePoint([0, 0, 80]))
    fe.assembly.add_constraint(
        torchfea.constraints.Couple(
            instance_name='final_model',
            set_nodes_name='surface_0_Head',
            rp_name=rp,
        )
    )

    fe.solver = torchfea.solver.StaticImplicitSolver(tol_error=1e-8)
    return fe


def _material_cases(fe):
    element = fe.assembly.get_part('final_model').elems['C3D4']
    neo_material = next(iter(element.materials.values()))
    shear_modulus = _scalar(neo_material.mu)
    bulk_modulus = _scalar(neo_material.kappa)
    youngs_modulus, poisson_ratio = _equivalent_linear_parameters(neo_material)

    # All cases have the same infinitesimal shear and bulk moduli.  The
    # higher-order parameters are intentionally different so their finite
    # deformation responses can be compared on the same pressure model.
    arruda_N = 5.0
    cases = (
        ('Neo-Hookean', neo_material),
        ('Linear elastic', torchfea.materials.LinearElastic(
            E=youngs_modulus, nu=poisson_ratio)),
        ('Mooney-Rivlin', torchfea.materials.MooneyRivlin(
            c10=shear_modulus / 4.0,
            c01=shear_modulus / 4.0,
            kappa=bulk_modulus)),
        ('Yeoh', torchfea.materials.Yeoh(
            c1=shear_modulus / 2.0,
            c2=0.01,
            c3=0.001,
            kappa=bulk_modulus)),
        ('Gent', torchfea.materials.Gent(
            mu=shear_modulus, Jm=20.0, kappa=bulk_modulus)),
        ('Arruda-Boyce', torchfea.materials.ArrudaBoyce(
            mu=_arruda_boyce_mu(shear_modulus, arruda_N),
            N=arruda_N,
            kappa=bulk_modulus)),
        ('Ogden', torchfea.materials.Ogden(
            mu=[shear_modulus / 2.0, shear_modulus / 2.0],
            alpha=[2.0, 5.0],
            kappa=bulk_modulus)),
    )
    return element, cases


def _check_small_strain_equivalence(cases):
    F0 = torch.eye(3).reshape(1, 1, 3, 3)
    reference_material = cases[0][1]
    reference_stress, reference_tangent = (
        reference_material.material_Constitutive_C3(F0))

    for name, material in cases:
        stress, tangent = material.material_Constitutive_C3(F0)
        assert torch.allclose(stress, reference_stress, atol=1e-7), name
        assert torch.allclose(
            tangent, reference_tangent, rtol=1e-5, atol=1e-6), name


def _run_cases():
    fe = _build_c3d4_pressure_model()
    element, cases = _material_cases(fe)
    _check_small_strain_equivalence(cases)

    results = []
    for index, (name, material) in enumerate(cases):
        element.set_materials({name: material})
        if index == 0:
            result = fe.solve()
            first_solution = result.GC.detach().clone()
        else:
            # Reuse the first Neo-Hookean equilibrium as the initial guess for
            # every subsequent material while keeping the same model/load.
            result = fe.solve(GC0=first_solution, if_initialize=False)
        results.append((name, result))
    return fe, results


def _deformed_mesh(fe, result):
    instance = fe.assembly.get_instance('final_model')
    rgc = fe.GC2RGC(result.GC)
    mesh = instance.get_mesh(RGC=rgc, surf_name='surface_0_All')
    displacement = rgc[instance._RGC_index].norm(dim=1).cpu().numpy()
    mesh['displacement'] = displacement
    return mesh


def _plot_cases(fe, results):
    import math
    import pyvista as pv

    meshes = [_deformed_mesh(fe, result) for _, result in results]
    max_displacement = max(
        float(mesh['displacement'].max()) for mesh in meshes)
    clim = (0.0, max_displacement if max_displacement > 0.0 else 1.0)

    columns = 3
    rows = math.ceil(len(results) / columns)
    plotter = pv.Plotter(
        shape=(rows, columns),
        window_size=(600 * columns, 550 * rows),
    )
    try:
        for index, ((name, _), mesh) in enumerate(zip(results, meshes)):
            plotter.subplot(index // columns, index % columns)
            plotter.add_mesh(
                mesh,
                scalars='displacement',
                clim=clim,
                show_edges=True,
            )
            plotter.add_text(name, font_size=14)
            plotter.view_isometric()

        for index in range(len(results), rows * columns):
            plotter.subplot(index // columns, index % columns)
            plotter.add_text('')

        plotter.link_views()
        plotter.show()
    finally:
        plotter.close()


def test_c3d4_pressure_materials_small_strain_equivalent():
    fe, results = _run_cases()

    displacement_summary = {}
    for name, result in results:
        assert result.converged, name
        assert torch.isfinite(result.GC).all()
        assert result.GC.numel() > 0

        mesh = _deformed_mesh(fe, result)
        displacement_summary[name] = float(mesh['displacement'].max())
        assert displacement_summary[name] > 0.0

    print('C3D4 pressure displacement:', displacement_summary)


if __name__ == '__main__':
    _plot_cases(*_run_cases())
