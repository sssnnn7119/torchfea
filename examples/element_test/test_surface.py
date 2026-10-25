import os

import numpy as np
import torch

import torchfea


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)

fem = torchfea.FEA_INP()
fem.read_inp(os.path.join(current_path, 'FreeHex.inp'))

fe = torchfea.from_inp(fem)
part = fe.assembly.get_part('Part-2')
instance = fe.assembly.get_instance('Part-2')

part.nodes[part.elems['C3D8']._elems.flatten().unique(), 0] += 40

fe.solver = torchfea.solver.StaticImplicitSolver(tol_error=1e-3)
fe.assembly.add_load(
    torchfea.loads.Pressure(
        instance_name='Part-2',
        surface_set='surfacepressure',
        pressure=0.0005,
    ),
    name='pressure-1',
)
fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(
        instance_name='final_model',
        set_nodes_name='surface_0_Bottom',
    )
)
fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(
        instance_name='Part-2',
        set_nodes_name='fix',
    )
)

result = fe.solve()
RGC = fe.GC2RGC(result.GC)
U = RGC[instance._RGC_index].cpu().numpy()
undeformed_surface = part.nodes.cpu().numpy()
deformed_surface = undeformed_surface + U

print(result.GC[-6:])
print(U[:, 0].max())
print('ok')
np.savetxt(os.path.join(current_path, 'U.txt'), U, delimiter=',')

extern_surf = instance.surfaces.get_elements('surfaceblock')[0]._elems.cpu().numpy()
from mayavi import mlab

Unorm = (U**2).sum(axis=1)**0.5
mesh2 = mlab.triangular_mesh(
    deformed_surface[:, 0],
    deformed_surface[:, 1],
    deformed_surface[:, 2],
    extern_surf[:, [0, 1, 2]],
    scalars=Unorm,
)
mesh2.actor.property.edge_visibility = True
mesh2.actor.property.line_width = 1.0
mesh2.actor.property.edge_color = (0, 0, 0)

if extern_surf.shape[1] > 3:
    mesh3 = mlab.triangular_mesh(
        deformed_surface[:, 0],
        deformed_surface[:, 1],
        deformed_surface[:, 2],
        extern_surf[:, [0, 2, 3]],
        scalars=Unorm,
    )
    mesh3.actor.property.edge_visibility = True
    mesh3.actor.property.line_width = 1.0
    mesh3.actor.property.edge_color = (0, 0, 0)

mlab.show()
