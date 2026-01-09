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

fem = torchfea.inp()
# fem.Read_INP(
#     'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/2024Arm/WorkspaceCase/CAE/TopOptRun.inp'
# )

# fem.Read_INP(
#     'Z:\RESULT\T20240325195025_\Cache/TopOptRun.inp'
# )
 
fem.Read_INP(current_path + '/FreeHex.inp')

fe = torchfea.from_inp(fem)

# fe.delete_element('element-1')

fe.nodes[fe.elems['element-0']._elems.flatten().unique(), 0] += 40

# elems = torch_fea.materials.initialize_materials(2, torch.tensor([[1.44, 0.45]]))
# fe.elems['element-0'].set_materials(elems)

# torch_fea.add_load(Loads.Body_Force_Undeformed(force_volumn_density=[1e-5, 0.0, 0.0], elem_index=torch_fea.elems['C3D4']._elems_index))

fe.add_load(torchfea.loads.Pressure(surface_set='surfacepressure', pressure=0.0005),
                name='pressure-1')

bc_dof = fe.node_sets['surface_0_Bottom'] * 3
bc_dof = np.concatenate([bc_dof, bc_dof + 1, bc_dof + 2])
bc_name = fe.add_constraint(
    torchfea.constraints.Boundary_Condition(indexDOF=bc_dof,
                                    dispValue=torch.zeros(bc_dof.size)))
                                    


bc_dof = fe.node_sets['fix'] * 3
bc_dof = np.concatenate([bc_dof, bc_dof + 1, bc_dof + 2])
bc_name = fe.add_constraint(
    torchfea.constraints.Boundary_Condition(indexDOF=bc_dof,
                                    dispValue=torch.zeros(bc_dof.size)))


# rp = fe.add_reference_point(torch_fea.ReferencePoint([0, 0, 80]))

indexNodes = fe.node_sets['surface_0_Head']
# torch_fea.add_constraint(
#     Constraints.Couple(
#         indexNodes=indexNodes,
#         rp_index=2))
# fe.add_constraint(torch_fea.constraints.Couple(indexNodes=indexNodes, rp_name=rp))

# fe.add_load(torch_fea.loads.Contact(surface_name1='surface_0_All', surface_name2='surfaceblock'))

t1 = time.time()

fe.solve(tol_error=0.001)


print(fe.GC[-6:])
print(fe.RGC[0][:, 0].max())
print('ok')
np.savetxt(current_path + '/U.txt', fe.RGC[0].cpu().numpy(), delimiter=',')

# extern_surf = fe.loads['pressure-1'].surface_element.cpu().numpy()
# extern_surf = fe.get_surface_elements('surface_0_All')[0]._elems.cpu().numpy()
extern_surf2 = fe.get_surface_elements('surfaceblock')[0]._elems.cpu().numpy()
# extern_surf = fem.part['final_model'].surfaces['surface_1_All']

from mayavi import mlab
import vtk
from mayavi import mlab
# coo=extern_surf

# Get the deformed surface coordinates
U = fe.RGC[0].cpu().numpy()
undeformed_surface = (fe.nodes).cpu().numpy()
deformed_surface = undeformed_surface + U

r=deformed_surface.transpose()


Unorm = (U**2).sum(axis=1)**0.5

# surface = mlab.pipeline.triangular_mesh_source(r[0], r[1], r[2], coo)
# surface_vtk = surface.outputs[0]._vtk_obj
# stlWriter = vtk.vtkSTLWriter()
# stlWriter.SetFileName('test.stl')
# stlWriter.SetInputConnection(surface_vtk.GetOutputPort())
# stlWriter.Write()
# mlab.close()

# Plot the deformed surface
# mesh1=mlab.triangular_mesh(deformed_surface[:, 0], deformed_surface[:, 1], deformed_surface[:, 2], extern_surf, scalars=Unorm)
mesh2=mlab.triangular_mesh(deformed_surface[:, 0], deformed_surface[:, 1], deformed_surface[:, 2], extern_surf2[:, [0,1,2]], scalars=Unorm)


# mesh1.actor.property.edge_visibility = True
# mesh1.actor.property.line_width = 1.0
# mesh1.actor.property.edge_color = (0, 0, 0)  # Black edges

mesh2.actor.property.edge_visibility = True
mesh2.actor.property.line_width = 1.0
mesh2.actor.property.edge_color = (0, 0, 0)  # Black edges

if extern_surf2.shape[1] > 3:
    mesh3=mlab.triangular_mesh(deformed_surface[:, 0], deformed_surface[:, 1], deformed_surface[:, 2], extern_surf2[:, [0,2,3]], scalars=Unorm)
    mesh3.actor.property.edge_visibility = True
    mesh3.actor.property.line_width = 1.0
    mesh3.actor.property.edge_color = (0, 0, 0)  # Black edges

mlab.show()