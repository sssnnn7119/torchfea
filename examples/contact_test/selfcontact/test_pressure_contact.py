
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
 
fem.read_inp(current_path + '/C3D4.inp')

fe = torchfea.from_inp(fem)
fe.solver = torchfea.solver.StaticImplicitSolver()
# fe._maximum_step_length = 0.3
# elems = torch_fea.materials.initialize_materials(2, torch.tensor([[1.44, 0.45]]))
# fe.elems['element-0'].set_materials(elems)

# torch_fea.add_load(Loads.Body_Force_Undeformed(force_volumn_density=[1e-5, 0.0, 0.0], elem_index=torch_fea.elems['C3D4']._elems_index))

fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06),
                name='pressure-1')
# fe.assembly.add_load(torch_fea.loads.ContactSelf(instance_name='final_model',surface_name='surface_0_All', penalty_distance_g=10, penalty_threshold_h=5.5))
fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_0_All'))
fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_1_All'))
fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_2_All'))
fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_3_All'))

bc_name = fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))

rp = fe.assembly.add_reference_point(torchfea.ReferencePoint([0, 0, 70]))

# torch_fea.add_constraint(
#     Constraints.Couple(
#         set_nodes_name='surface_0_Head',
#         rp_index=2))
fe.assembly.add_constraint(torchfea.constraints.Couple(instance_name='final_model', set_nodes_name='surface_0_Head', rp_name=rp))




t1 = time.time()

fe.solve(tol_error=0.001)


print(fe.assembly.GC[-6:])
print('ok')


# extern_surf = fe.loads['pressure-1'].surface_element.cpu().numpy()
ins1 = fe.assembly.get_instance('final_model')

extern_surf = ins1.surfaces.get_elements('surface_0_All')[0]._elems[:, :3].cpu().numpy()
intern_surf = ins1.surfaces.get_elements('surface_1_All')[0]._elems[:, :3].cpu().numpy()
intern_surf2 = ins1.surfaces.get_elements('surface_2_All')[0]._elems[:, :3].cpu().numpy()

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
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='displacement', show_edges=True, edge_color='black', line_width=0.5, opacity=1.0)
plotter.show()