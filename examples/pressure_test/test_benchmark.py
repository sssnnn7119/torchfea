
import os
import time
import sys

import numpy as np
import torch
import pyvista as pv
os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))
import pathlib
current_path = pathlib.Path(current_path)
torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)


import torchfea



fem = torchfea.FEA_INP()

name = 'C3D4'
torchfea.enable_logging(level=torchfea.__logging.INFO, log_file=current_path.parent / 'models' / f'{name}.log', file_log_level=torchfea.__logging.DEBUG)
fem.read_inp(current_path.parent / 'models' / f'{name}.inp')

fe = torchfea.from_inp(fem)
fe.solver = torchfea.solver.StaticImplicitSolver(tol_error=1e-8)

fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06),
                name='pressure-1')

bc_name = fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))

rp = fe.assembly.add_reference_point(torchfea.ReferencePoint([0, 0, 80]))

fe.assembly.add_constraint(torchfea.constraints.Couple(instance_name='final_model', set_nodes_name='surface_0_Head', rp_name=rp))

# fe.assembly.add_load(torchfea.loads.Penalty_DoF(s=2, k=1e4, target=20.0, obj_name=rp), name='penalty-1')

t1 = time.time()

fe.save_model(current_path.parent / 'models' / f'{name}_model.npz', if_save_source_code=True)
result = fe.solve()
result.save(current_path.parent / 'models' / f'{name}_results.npz')
# result = torchfea.solver.StaticResult.load('temp.npz')
# os.remove('temp.npz')

fe.assembly.get_part('final_model').exterior_surface = 'surface_0_All'
fe.assembly.show_all(GC=result.GC)

part = fe.assembly.get_part('final_model')
mesh = part.get_mesh()
plotter = pv.Plotter()
try:
    plotter.add_mesh(mesh, show_edges=True)
    plotter.show()
finally:
    plotter.close()
