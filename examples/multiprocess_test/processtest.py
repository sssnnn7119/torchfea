
def solvefea(fe):
    import torch
    torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float64)
    fe.change_device(torch.device('cuda'))
    return fe.solve(tol_error=0.01)


if __name__ == '__main__':
    import os
    import time
    import sys

    import numpy as np
    import torch
    import torchfea
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    current_path = os.path.dirname(os.path.abspath(__file__))

    torch.set_default_device(torch.device('cuda'))
    torch.set_default_dtype(torch.float64)

    fem = torchfea.FEA_INP()
    # fem.Read_INP(
    #     'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/2024Arm/WorkspaceCase/CAE/TopOptRun.inp'
    # )

    # fem.Read_INP(
    #     'Z:\RESULT\T20240325195025_\Cache/TopOptRun.inp'
    # )

    fem.read_inp(current_path + '/../models/C3D4Less.inp')

    fe = torchfea.from_inp(fem)
    fe.solver = torchfea.solver.StaticImplicitSolver()
    # elems = torch_fea.materials.initialize_materials(2, torch.tensor([[1.44, 0.45]]))
    # fe.elems['element-0'].set_materials(elems)

    # torch_fea.add_load(Loads.Body_Force_Undeformed(force_volumn_density=[1e-5, 0.0, 0.0], elem_index=torch_fea.elems['C3D4']._elems_index))

    fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06),
                    name='pressure-1')

    bc_name = fe.assembly.add_boundary(
        torchfea.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))

    rp = fe.assembly.add_reference_point(torchfea.ReferencePoint([0, 0, 80]))

    fe.assembly.add_constraint(torchfea.constraints.Couple(instance_name='final_model', set_nodes_name='surface_0_Head', rp_name=rp))



    t1 = time.time()
    fe.change_device(torch.device('cpu'))



    import multiprocessing as mp

    pools = mp.Pool(processes=2)
    ps = pools.apply_async(solvefea, args=(fe,))
    fearesult = ps.get()
    # fearesult = solvefea(fe)

    print(fe.solver.GC)
    print('ok')


    # extern_surf = fe.loads['pressure-1'].surface_element.cpu().numpy()
    extern_surf = fe.assembly.get_instance('final_model').surfaces.get_elements('surface_0_All')[0]._elems[:, :3].cpu().numpy()
    # extern_surf = fem.part['final_model'].surfaces['surface_1_All']

    import pyvista as pv

    # Get the deformed surface coordinates
    U = fe.assembly._GC2RGC(fe.solver.GC)[0].cpu().numpy()
    undeformed_surface = fem.part['final_model'].nodes[:,1:]
    deformed_surface = undeformed_surface + U

    Unorm = (U**2).sum(axis=1)**0.5

    # Plot the deformed surface
    faces = np.column_stack([np.full(len(extern_surf), 3), extern_surf])
    mesh = pv.PolyData(deformed_surface, faces)
    mesh['displacement'] = Unorm
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars='displacement')
    plotter.show()