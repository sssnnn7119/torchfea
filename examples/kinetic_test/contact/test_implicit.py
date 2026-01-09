
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

torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)

fem = torchfea.FEA_INP()

 
fem.read_inp(current_path + '/rec.inp')

fe = torchfea.from_inp(fem)
fe.solver = torchfea.solver.DynamicImplicitSolver(deltaT=1e-3, time_end=1.0)

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


fe.solve(tol_error=1e-6)

current_path1 = "Z:/temp/"
np.save(current_path1 + '/implicit_high_gamma_GC.npy', np.array([fe.solver._GC_list[i].tolist() for i in range(len(fe.solver._GC_list))], dtype=np.float32))
np.save(current_path1 + '/implicit_high_gamma_GV.npy', np.array([fe.solver._GV_list[i].tolist() for i in range(len(fe.solver._GV_list))], dtype=np.float32))
np.save(current_path1 + '/implicit_high_gamma_GA.npy', np.array([fe.solver._GA_list[i].tolist() for i in range(len(fe.solver._GA_list))], dtype=np.float32))