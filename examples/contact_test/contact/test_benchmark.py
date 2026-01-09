import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch

import numpy as np
import time
import sys
sys.path.append('.')

import torchfea

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
 
fem.read_inp(current_path + '/Free.inp')

fe = torchfea.from_inp(fem)
fe.solver = torchfea.solver.StaticImplicitSolver()
fe.assembly.get_instance('Part-2')._translation = torch.tensor([0, 0, 40.])
fe.assembly.get_instance('Part-2')._rotation = torch.tensor([0., 0., 0.1])

# elems = torch_fea.materials.initialize_materials(2, torch.tensor([[1.44, 0.45]]))
# fe.elems['element-0'].set_materials(elems)

# torch_fea.add_load(Loads.Body_Force_Undeformed(force_volumn_density=[1e-5, 0.0, 0.0], elem_index=torch_fea.elems['C3D4']._elems_index))

fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.02),
                name='pressure-1')

bc_name = fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))
                                    


bc_name = fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='Part-2', set_nodes_name='fix',))


rp = fe.assembly.add_reference_point(torchfea.ReferencePoint([0, 0, 80]))

fe.assembly.add_constraint(torchfea.constraints.Couple(instance_name='final_model', set_nodes_name='surface_0_Head', rp_name=rp))

fe.assembly.add_load(torchfea.loads.Contact(instance_name1='final_model', instance_name2='Part-2', surface_name1='surface_0_All', surface_name2='surfaceblock'))

t1 = time.time()

fe.solve(tol_error=0.001)


print(fe.assembly.GC[-6:])
print('ok')


# extern_surf = fe.loads['pressure-1'].surface_element.cpu().numpy()
ins1 = fe.assembly.get_instance('final_model')
ins2 = fe.assembly.get_instance('Part-2')
extern_surf = ins1.surfaces.get_elements('surface_0_All')[0]._elems.cpu().numpy()
extern_surf2 = ins2.surfaces.get_elements('surfaceblock')[0]._elems.cpu().numpy()
# extern_surf = fem.part['final_model'].surfaces['surface_1_All']

import pyvista as pv

# Get the deformed surface coordinates
U1 = fe.assembly.RGC[ins1._RGC_index].cpu().numpy()
U2 = fe.assembly.RGC[ins2._RGC_index].cpu().numpy()
undeformed_surface1 = ins1.nodes.cpu().numpy()
undeformed_surface2 = ins2.nodes.cpu().numpy()
deformed_surface1 = undeformed_surface1 + U1
deformed_surface2 = undeformed_surface2 + U2

Unorm1 = (U1**2).sum(axis=1)**0.5
Unorm2 = (U2**2).sum(axis=1)**0.5

# Plot the deformed surface
faces1 = np.column_stack([np.full(len(extern_surf), 3), extern_surf])
mesh1 = pv.PolyData(deformed_surface1, faces1)
mesh1['displacement'] = Unorm1

faces2 = np.column_stack([np.full(len(extern_surf2), 3), extern_surf2[:, [0,1,2]]])
mesh2 = pv.PolyData(deformed_surface2, faces2)
mesh2['displacement'] = Unorm2

plotter = pv.Plotter()
plotter.add_mesh(mesh1, scalars='displacement', show_edges=True, line_width=1.0, edge_color='black')
plotter.add_mesh(mesh2, scalars='displacement', show_edges=True, line_width=1.0, edge_color='black')

if extern_surf2.shape[1] > 3:
    faces3 = np.column_stack([np.full(len(extern_surf2), 3), extern_surf2[:, [0,2,3]]])
    mesh3 = pv.PolyData(deformed_surface2, faces3)
    mesh3['displacement'] = Unorm2
    plotter.add_mesh(mesh3, scalars='displacement', show_edges=True, line_width=1.0, edge_color='black')

plotter.show()