import torch
import os
import numpy as np
import time
import sys
sys.path.append('.')

import torchfea
os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cpu'))
torch.set_default_dtype(torch.float64)

fem = torchfea.FEA_INP()
# fem.Read_INP(
#     'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/2024Arm/WorkspaceCase/CAE/TopOptRun.inp'
# )

# fem.Read_INP(
#     'Z:\RESULT\T20240325195025_\Cache/TopOptRun.inp'
# )
 
fem.read_inp(current_path + '/C3D4Less.inp')

fe = torchfea.from_inp(fem)
fe.solver = torchfea.solver.DynamicImplicitSolver(deltaT=1e-2, time_end=0.5)

fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06))

bc_name = fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))

# rp = fe.assembly.add_reference_point(torch_fea.ReferencePoint([0, 0, 80]))

# indexNodes = fem.part['final_model'].sets_nodes['surface_0_Head']

# fe.assembly.add_constraint(torch_fea.constraints.Couple(instance_name='final_model', indexNodes=indexNodes, rp_name=rp))

fe.initialize()

t1 = time.time()


benchmark_data = np.load('Z:/temp/benchmark_result.npy').astype(np.float64)
v_list = np.load('Z:/temp/velocity.npy').astype(np.float64)

# extern_surf = fe.loads['pressure-1'].surface_element.cpu().numpy()
extern_surf = fe.assembly.get_instance('final_model').surfaces.get_elements('surface_0_All')[0]._elems.cpu().numpy()

# Ensure extern_surf is a triangle index array (n, 3)
if extern_surf.shape[1] > 3:
    extern_surf = extern_surf[:, :3]
elif extern_surf.shape[1] < 3:
    raise ValueError("Surface element array does not have enough vertices per face for triangles.")

import pyvista as pv
from time import sleep
import threading
coo=extern_surf

# Get the deformed surface coordinates

# Prepare undeformed surface
undeformed_surface = (fem.part['final_model'].nodes[:,1:]).cpu().numpy()
Unorm_list = []
deformed_list = []

for i in range(len(benchmark_data)):
    U = fe.assembly._GC2RGC(torch.from_numpy(benchmark_data[i]))[0].cpu().numpy()
    deformed_surface = undeformed_surface + U
    Unorm = (U**2).sum(axis=1)**0.5
    deformed_list.append(deformed_surface.copy())
    Unorm_list.append(Unorm.copy())

plotter = pv.Plotter()
plotter.set_background('white')
plotter.enable_parallel_projection()

faces = np.column_stack([np.full(len(extern_surf), 3), extern_surf])
mesh = pv.PolyData(deformed_list[0], faces)
mesh['displacement'] = Unorm_list[0]
actor = plotter.add_mesh(mesh, scalars='displacement')
plotter.view_azimuth(90)
plotter.view_elevation(90)
plotter.camera.parallel_scale *= 2

plotter.show(auto_close=False)

for idx in range(len(benchmark_data)):
    deformed_surface = deformed_list[idx]
    Unorm = Unorm_list[idx]
    mesh.points = deformed_surface
    mesh['displacement'] = Unorm
    plotter.update_scalars(mesh['displacement'])
    plotter.update()
    sleep(0.1)  # adjust speed

plotter.close()
