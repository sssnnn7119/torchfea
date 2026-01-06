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

from mayavi import mlab
import vtk
from mayavi import mlab
coo=extern_surf

# Get the deformed surface coordinates
U1 = fe.assembly.RGC[ins1._RGC_index].cpu().numpy()
U2 = fe.assembly.RGC[ins2._RGC_index].cpu().numpy()
undeformed_surface1 = ins1.nodes.cpu().numpy()
undeformed_surface2 = ins2.nodes.cpu().numpy()
deformed_surface1 = undeformed_surface1 + U1
deformed_surface2 = undeformed_surface2 + U2

r1=deformed_surface1.transpose()
r2=deformed_surface2.transpose()

Unorm1 = (U1**2).sum(axis=1)**0.5
Unorm2 = (U2**2).sum(axis=1)**0.5

# surface = mlab.pipeline.triangular_mesh_source(r[0], r[1], r[2], coo)
# surface_vtk = surface.outputs[0]._vtk_obj
# stlWriter = vtk.vtkSTLWriter()
# stlWriter.SetFileName('test.stl')
# stlWriter.SetInputConnection(surface_vtk.GetOutputPort())
# stlWriter.Write()
# mlab.close()

# Plot the deformed surface
mesh1=mlab.triangular_mesh(deformed_surface1[:, 0], deformed_surface1[:, 1], deformed_surface1[:, 2], extern_surf, scalars=Unorm1)
mesh2=mlab.triangular_mesh(deformed_surface2[:, 0], deformed_surface2[:, 1], deformed_surface2[:, 2], extern_surf2[:, [0,1,2]], scalars=Unorm2)

mesh1.actor.property.edge_visibility = True
mesh1.actor.property.line_width = 1.0
mesh1.actor.property.edge_color = (0, 0, 0)  # Black edges

mesh2.actor.property.edge_visibility = True
mesh2.actor.property.line_width = 1.0
mesh2.actor.property.edge_color = (0, 0, 0)  # Black edges

if extern_surf2.shape[1] > 3:
    mesh3=mlab.triangular_mesh(deformed_surface2[:, 0], deformed_surface2[:, 1], deformed_surface2[:, 2], extern_surf2[:, [0,2,3]], scalars=Unorm2)
    mesh3.actor.property.edge_visibility = True
    mesh3.actor.property.line_width = 1.0
    mesh3.actor.property.edge_color = (0, 0, 0)  # Black edges

mlab.show()