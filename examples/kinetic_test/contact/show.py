from __future__ import annotations
import os

from torchvision.models.detection import rpn

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import numpy as np
import time
import sys
sys.path.append('.')

import torchfea
current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cpu'))
torch.set_default_dtype(torch.float64)

fem = torchfea.FEA_INP()

 
fem.read_inp(current_path + '/rec.inp')

fe = torchfea.from_inp(fem)
fe.solver = torchfea.solver.DynamicImplicitSolver(deltaT=1e-3, time_end=0.1)

fem2 = torchfea.FEA_INP()
fem2.read_inp(current_path + '/base.inp')

fe2 = torchfea.from_inp(fem2)

fe.assembly.add_part(fe2.assembly.get_part('base'), name='base')
fe.assembly.add_instance(torchfea.Instance(part=fe.assembly.get_part('base')), name='base')

fe.assembly.get_part('rec').elems['element-0'].density = 1e-9
fe.assembly.get_part('base').elems['element-0'].density = 1e-9

fe.assembly.add_load(torchfea.loads.BodyForce(instance_name='rec', element_name='element-0', force_density=[0, 0, -9.81e-6, ]))
fe.assembly.add_load(torchfea.loads.Contact(instance_name1='rec', instance_name2='base', surface_name1='contact', surface_name2='contact'))

fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='base', set_nodes_name='base'))

t1 = time.time()

fe.initialize()

t1 = time.time()

path0='Z:/temp/implicit_high_gamma_'
benchmark_data = np.load(path0 + 'GC.npy').astype(np.float64)
v_list = np.load(path0 + 'GV.npy').astype(np.float64)

# extern_surf = fe.loads['pressure-1'].surface_element.cpu().numpy()
extern_surf = fe.assembly.get_instance('rec').surfaces.get_elements('contact')[0]._elems.cpu().numpy()

# Ensure extern_surf is a triangle index array (n, 3)
if extern_surf.shape[1] > 3:
    extern_surf = extern_surf[:, :3]
elif extern_surf.shape[1] < 3:
    raise ValueError("Surface element array does not have enough vertices per face for triangles.")

import pyvista as pv
from time import sleep

# Prepare undeformed surface
undeformed_surface = (fem.part['rec'].nodes[:,1:])
Unorm_list = []
deformed_list = []

for i in range(len(benchmark_data)):
    U = fe.assembly._GC2RGC(torch.from_numpy(benchmark_data[i]))[1].cpu().numpy()
    deformed_surface = undeformed_surface + U
    Unorm = (U**2).sum(axis=1)**0.5
    deformed_list.append(deformed_surface.copy())
    Unorm_list.append(Unorm.copy())

# Create faces for PyVista
faces = np.column_stack([np.full(len(extern_surf), 3), extern_surf])

# Create initial mesh
mesh = pv.PolyData(deformed_list[0], faces)
mesh['displacement'] = Unorm_list[0]

# Create plotter
plotter = pv.Plotter()
actor = plotter.add_mesh(mesh, scalars='displacement')

plotter.camera.parallel_projection = True

# Add frame text
text_actor = plotter.add_text('Frame: 0', position='upper_left', color='black')

def callback(step):
    mesh.points = deformed_list[step]
    mesh['displacement'] = Unorm_list[step]
    text_actor.SetText(0, f'Frame: {step}')

plotter.add_timer_event(max_steps=100, duration=10, callback=callback)

# Show plotter without closing
plotter.show()

plotter.close()