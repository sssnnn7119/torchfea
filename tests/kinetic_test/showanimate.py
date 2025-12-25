
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
import time
import sys
sys.path.append('.')

import FEA
current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cpu'))
torch.set_default_dtype(torch.float64)

fem = FEA.FEA_INP()
# fem.Read_INP(
#     'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/2024Arm/WorkspaceCase/CAE/TopOptRun.inp'
# )

# fem.Read_INP(
#     'Z:\RESULT\T20240325195025_\Cache/TopOptRun.inp'
# )
 
fem.read_inp(current_path + '/C3D4Less.inp')

fe = FEA.from_inp(fem)
fe.solver = FEA.solver.DynamicImplicitSolver(deltaT=1e-2, time_end=0.5)

fe.assembly.add_load(FEA.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06))

bc_name = fe.assembly.add_boundary(
    FEA.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))

# rp = fe.assembly.add_reference_point(FEA.ReferencePoint([0, 0, 80]))

# indexNodes = fem.part['final_model'].sets_nodes['surface_0_Head']

# fe.assembly.add_constraint(FEA.constraints.Couple(instance_name='final_model', indexNodes=indexNodes, rp_name=rp))

fe.initialize()

t1 = time.time()

path0='Z:/temp/implicit_high_gamma_'
benchmark_data = np.load(path0 + 'GC.npy').astype(np.float64)
v_list = np.load(path0 + 'GV.npy').astype(np.float64)

# extern_surf = fe.loads['pressure-1'].surface_element.cpu().numpy()
extern_surf = fe.assembly.get_instance('final_model').surfaces.get_elements('surface_0_All')[0]._elems.cpu().numpy()

# Ensure extern_surf is a triangle index array (n, 3)
if extern_surf.shape[1] > 3:
    extern_surf = extern_surf[:, :3]
elif extern_surf.shape[1] < 3:
    raise ValueError("Surface element array does not have enough vertices per face for triangles.")

from mayavi import mlab
import vtk
from mayavi import mlab
from time import sleep
import threading
coo=extern_surf

# Get the deformed surface coordinates

# Prepare undeformed surface
undeformed_surface = (fem.part['final_model'].nodes[:,1:])
Unorm_list = []
deformed_list = []

for i in range(len(benchmark_data)):
    U = fe.assembly._GC2RGC(torch.from_numpy(benchmark_data[i]))[0].cpu().numpy()
    deformed_surface = undeformed_surface + U
    Unorm = (U**2).sum(axis=1)**0.5
    deformed_list.append(deformed_surface.copy())
    Unorm_list.append(Unorm.copy())

mlab.figure(bgcolor=(1,1,1), size=(800, 1000))
fig = mlab.gcf()
scene = fig.scene


mesh = mlab.triangular_mesh(
    deformed_list[0][:, 0], deformed_list[0][:, 1], deformed_list[0][:, 2],
    extern_surf, scalars=Unorm_list[0]
)

mlab.view(azimuth=90, elevation=90)
scene.camera.parallel_projection = True
scene.camera.parallel_scale *= 2.
scene.camera.focal_point = (0, -50, 00)

# 添加帧编号文本，设置为黑色
frame_text = mlab.text(0.01, 0.95, 'Frame: 0', width=0.2, color=(0,0,0))

@mlab.animate(delay=100)
def anim():
    while True:
        for idx, (deformed_surface, Unorm) in enumerate(zip(deformed_list, Unorm_list)):
            mesh.mlab_source.set(
                x=deformed_surface[:, 0],
                y=deformed_surface[:, 1],
                z=deformed_surface[:, 2],
                scalars=Unorm
            )
            frame_text.text = f'Frame: {idx}'
            yield

anim()
mlab.show()