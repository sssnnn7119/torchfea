from tkinter.font import names
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
 
fem.Read_INP(current_path + '/C3D4Less.inp')

fe = torchfea.from_inp(fem)
fe.maximum_iteration = 1000


# # convert to the second order elements
str_now = 'element-0'
elems0 = fe.elems[str_now]
material0 = fe.elems[str_now].materials
ind_surf = 0
elems_name_list = []
surf_name_list = []
while True:
    surf_name = 'surface_%d_All'%ind_surf
    if not surf_name in fe.surface_sets.keys():
        break
    elems_surface, elems_other = torchfea.elements.divide_surface_elements(fe=fe, name_element=str_now, name_surface=surf_name)

    fe.delete_element(str_now)
    fe.add_element(elems_other, name='element-0')
    fe.add_element(elems_surface, name='element-surf-%d'%ind_surf)
    elems_name_list.append('element-surf-%d'%ind_surf)
    surf_name_list.append(surf_name)

    ind_surf+=1

fe.merge_elements(element_name_list=elems_name_list, element_name_new='element-sensitivity')

fe = torchfea.elements.convert_to_second_order(fe, ['element-sensitivity'])

element: torchfea.elements.Element_3D = fe.elems['element-sensitivity']
element.surf_order = torch.ones([element._elems.shape[0], 4], dtype=torch.int8, device='cpu')



elems_1order: torchfea.elements.Element_3D = fe.elems['element-0']
elems_2order: torchfea.elements.Element_3D = fe.elems['element-sensitivity']
elems_1order.set_materials(material0)
elems_2order.set_materials(material0)

fe.add_load(torchfea.loads.Pressure(surface_set='surface_1_All', pressure=0.06),
                name='pressure-1')

bc_dof = np.array(
    list(fem.part['final_model'].sets_nodes['surface_0_Bottom'])) * 3
bc_dof = np.concatenate([bc_dof, bc_dof + 1, bc_dof + 2])
bc_name = fe.add_constraint(
    torchfea.constraints.Boundary_Condition(indexDOF=bc_dof,
                                    dispValue=torch.zeros(bc_dof.size)))

rp = fe.add_reference_point(torchfea.ReferencePoint([0, 0, 80]))

indexNodes = np.where((abs(fe.nodes[:, 2] - 80)
                        < 0.1).cpu().numpy())[0]

fe.add_constraint(torchfea.constraints.Couple(indexNodes=indexNodes, rp_name=rp))



t1 = time.time()


fe.solve(tol_error=0.01)


print(fe.GC)
print(fe.GC.shape)
print('ok')

for surf_name in surf_name_list:
    fe.elems['element-sensitivity'] = torchfea.elements.set_surface_2order(fe=fe, name_elems='element-sensitivity', name_surface=surf_name)


fe.solve(RGC0=fe.RGC, tol_error=0.01)


print(fe.GC)
print(fe.GC.shape)
print('ok')


# extern_surf = fe.loads['pressure-1'].surface_element.cpu().numpy()
extern_surf = fem.Find_Surface(['surface_0_All'])[1]
# extern_surf = fem.part['final_model'].surfaces['surface_1_All']

import pyvista as pv

# Get the deformed surface coordinates
U = fe.RGC[0].cpu().numpy()
undeformed_surface = (fe.nodes).cpu().numpy()
deformed_surface = undeformed_surface + U

Unorm = (U**2).sum(axis=1)**0.5

# Plot the deformed surface
faces = np.column_stack([np.full(len(extern_surf), 3), extern_surf])
mesh = pv.PolyData(deformed_surface, faces)
mesh['displacement'] = Unorm
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='displacement')
plotter.show()

assert False