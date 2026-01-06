import torch
import os
import numpy as np
import sys
sys.path.append('.')

import torchfea

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
current_path = os.path.dirname(os.path.abspath(__file__))

# Device and dtype (align with other tests)
torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)

# 1) Read INP
fem = torchfea.FEA_INP()
fem.read_inp(current_path + '/C3D4.inp')

# 2) Build controller
fe = torchfea.from_inp(fem)
fe.solver = torchfea.solver.StaticImplicitSolver()

# 3) Load: (optional) keep it simple: no external pressure so spring effect is visible
# If desired, you can uncomment to add pressure
fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06))

# 4) Boundary: fix bottom nodes by set name
fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom')
)

# 5) Reference point at the top and couple with head nodes
rp = fe.assembly.add_reference_point(torchfea.ReferencePoint([0, 0, 80]))
fe.assembly.add_constraint(
    torchfea.constraints.Couple(instance_name='final_model', set_nodes_name='surface_0_Head', rp_name=rp)
)

# 6) Spring from RP to fixed space point (0,0,100)
# Note: rest_length=None uses the initial distance (no initial force).
#       For a visible effect, set rest_length smaller/larger than initial.
#       Here we choose rest_length=0 to pull the RP toward the target point.
spring_load = torchfea.loads.Spring_RP_Point(rp_name=rp, point=[0, 0, 80], k=1e2, rest_length=0.0)
fe.assembly.add_load(spring_load)

# 7) Solve
fe.solve(tol_error=1e-3)

# 8) Report RP translation (first 3 DOFs)
GC = fe.solver.GC
# Map GC back to RGC blocks
RGC = fe.assembly._GC2RGC(GC)
# RP block index
rp_obj = fe.assembly.get_reference_point(rp)
rp_block = RGC[rp_obj._RGC_index]
print('RP translation:', rp_block[:3].detach().cpu().numpy())


# extern_surf = fe.loads['pressure-1'].surface_element.cpu().numpy()
extern_surf = fe.assembly.get_instance('final_model').surfaces.get_elements('surface_0_All')[0]._elems[:, :3].cpu().numpy()
# extern_surf = fem.part['final_model'].surfaces['surface_1_All']

from mayavi import mlab
import vtk
from mayavi import mlab
coo=extern_surf

# Get the deformed surface coordinates
U = fe.assembly._GC2RGC(fe.solver.GC)[0].cpu().numpy()
undeformed_surface = (fem.part['final_model'].nodes[:,1:]).cpu().numpy()
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