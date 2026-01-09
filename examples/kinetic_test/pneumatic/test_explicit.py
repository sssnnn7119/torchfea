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

fem = torchfea.FEA_INP()
# fem.Read_INP(
#     'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/2024Arm/WorkspaceCase/CAE/TopOptRun.inp'
# )

# fem.Read_INP(
#     'Z:\RESULT\T20240325195025_\Cache/TopOptRun.inp'
# )
 
fem.read_inp(current_path + '/C3D4Less.inp')

fe = torchfea.from_inp(fem)
fe.solver = torchfea.solver.DynamicExplicitSolver(time_end=0.1, time_per_storage=1e-4)

fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06))
# fe.assembly.add_load(torch_fea.loads.BodyForce(instance_name='final_model', element_name='element-0', force_density=[-9.81e-6, 0, 0, ]))

bc_name = fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))

# rp = fe.assembly.add_reference_point(torch_fea.ReferencePoint([0, 0, 80]))

# indexNodes = fem.part['final_model'].sets_nodes['surface_0_Head']

# fe.assembly.add_constraint(torch_fea.constraints.Couple(instance_name='final_model', indexNodes=indexNodes, rp_name=rp))



t1 = time.time()


fe.solve()

current_path1 = "Z:/temp/"
np.save(current_path1 + '/explicitGC.npy', np.array([fe.solver._GC_list[i].tolist() for i in range(len(fe.solver._GC_list))], dtype=np.float32))
np.save(current_path1 + '/explicitGV.npy', np.array([fe.solver._GV_list[i].tolist() for i in range(len(fe.solver._GV_list))], dtype=np.float32))
np.save(current_path1 + '/explicitGA.npy', np.array([fe.solver._GA_list[i].tolist() for i in range(len(fe.solver._GA_list))], dtype=np.float32))