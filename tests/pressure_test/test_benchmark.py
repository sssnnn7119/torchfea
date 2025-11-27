import torch
import os
import numpy as np
import time
import sys
sys.path.append('.')

import FEA
os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)

fem = FEA.FEA_INP()
# fem.Read_INP(
#     'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/2024Arm/WorkspaceCase/CAE/TopOptRun.inp'
# )

# fem.Read_INP(
#     'Z:\RESULT\T20240325195025_\Cache/TopOptRun.inp'
# )
 
fem.read_inp(current_path + '/C3D4.inp')

fe = FEA.from_inp(fem)
fe.solver = FEA.solver.StaticImplicitSolver()
# elems = FEA.materials.initialize_materials(2, torch.tensor([[1.44, 0.45]]))
# fe.elems['element-0'].set_materials(elems)

# FEA.add_load(Loads.Body_Force_Undeformed(force_volumn_density=[1e-5, 0.0, 0.0], elem_index=FEA.elems['C3D4']._elems_index))

fe.assembly.add_load(FEA.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06),
                name='pressure-1')

bc_name = fe.assembly.add_boundary(
    FEA.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))

rp = fe.assembly.add_reference_point(FEA.ReferencePoint([0, 0, 80]))

fe.assembly.add_constraint(FEA.constraints.Couple(instance_name='final_model', set_nodes_name='surface_0_Head', rp_name=rp))



t1 = time.time()


fe.solve(tol_error=0.01)


print(fe.solver.GC)
print('ok')


# extern_surf = fe.loads['pressure-1'].surface_element.cpu().numpy()
extern_surf = fe.assembly.get_instance('final_model').surfaces.get_elements('surface_0_All')[0]._elems[:, :3].cpu().numpy()
# extern_surf = fem.part['final_model'].surfaces['surface_1_All']

from mayavi import mlab
import vtk
from mayavi import mlab
coo=extern_surf

# Get the deformed surface coordinates
U = fe.assembly._GC2RGC(fe.solver.GC)[0].cpu().numpy()
undeformed_surface = fem.part['final_model'].nodes[:,1:]
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
mlab.triangular_mesh(deformed_surface[:, 0], deformed_surface[:, 1], deformed_surface[:, 2], extern_surf, scalars=Unorm)
mlab.show()