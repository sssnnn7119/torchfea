import torch
import os
import numpy as np
import time
import sys
sys.path.append('.')

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
 
fem.read_inp(current_path + '/C3D4.inp')

fe = torchfea.from_inp(fem)

ins1 = fe.assembly.get_instance('final_model')

ins2 = torchfea.Instance(fe.assembly.get_part('final_model'))
fe.assembly.add_instance(ins2, name='final_model2')
ins2._translation = torch.tensor([0.0, -20.0, -20.0])
ins2._rotation = torch.tensor([0.0, 0.0, np.pi/2])

fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model2', surface_set='surface_1_All', pressure=0.06),
                name='pressure-1')

bc_name = fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='final_model2', set_nodes_name='surface_0_Bottom'))
bc_name = fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))

rp = fe.assembly.add_reference_point(torchfea.ReferencePoint([0, -20, 60]))

fe.assembly.add_constraint(torchfea.constraints.Couple(instance_name='final_model2', set_nodes_name='surface_0_Head', rp_name=rp))

fe.assembly.add_load(torchfea.loads.Contact(instance_name1='final_model', instance_name2='final_model2', surface_name1='surface_0_All', surface_name2='surface_0_All'))


t1 = time.time()


fe.solve(tol_error=0.01)


print(fe.assembly.GC)
print('ok')


# extern_surf = fe.loads['pressure-1'].surface_element.cpu().numpy()
ins1 = fe.assembly.get_instance('final_model')
ins2 = fe.assembly.get_instance('final_model2')

extern_surf = ins1.surfaces.get_elements('surface_0_All')[0]._elems[:, :3].cpu().numpy()
extern_surf2 = ins2.surfaces.get_elements('surface_0_All')[0]._elems[:, :3].cpu().numpy()

import pyvista as pv

# Get the deformed surface coordinates
U = fe.assembly.RGC[ins1._RGC_index].cpu().numpy()
undeformed_surface = ins1.nodes.cpu().numpy()
deformed_surface = undeformed_surface + U
Unorm = (U**2).sum(axis=1)**0.5

# Plot the deformed surface
faces = np.column_stack([np.full(len(extern_surf), 3), extern_surf])
mesh = pv.PolyData(deformed_surface, faces)
mesh['displacement'] = Unorm

U2 = fe.assembly.RGC[ins2._RGC_index].cpu().numpy()
undeformed_surface2 = ins2.nodes.cpu().numpy()
deformed_surface2 = undeformed_surface2 + U2
Unorm2 = (U2**2).sum(axis=1)**0.5

faces2 = np.column_stack([np.full(len(extern_surf2), 3), extern_surf2])
mesh2 = pv.PolyData(deformed_surface2, faces2)
mesh2['displacement'] = Unorm2

plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='displacement')
plotter.add_mesh(mesh2, scalars='displacement')
plotter.show()