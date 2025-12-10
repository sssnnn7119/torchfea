
import os
import time
import sys

sys.path.append('.')

import numpy as np
import torch
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
 
fem.read_inp(current_path + '/C3D4Less.inp')

fe = FEA.from_inp(fem)
fe.solver = FEA.solver.StaticImplicitSolver()
# elems = FEA.materials.initialize_materials(2, torch.tensor([[1.44, 0.45]]))
# fe.elems['element-0'].set_materials(elems)

# FEA.add_load(Loads.Body_Force_Undeformed(force_volumn_density=[1e-5, 0.0, 0.0], elem_index=FEA.elems['C3D4']._elems_index))
pressure_load = FEA.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06)
fe.assembly.add_load(pressure_load,
                name='pressure-1')

bc_name = fe.assembly.add_boundary(
    FEA.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))

rp = fe.assembly.add_reference_point(FEA.ReferencePoint([0, 0, 80]))

fe.assembly.add_constraint(FEA.constraints.Couple(instance_name='final_model', set_nodes_name='surface_0_Head', rp_name=rp))



t1 = time.time()


fe.solve(tol_error=0.01)

jacobian = fe.solver.get_jacobian(GC_now=fe.assembly.GC)

GC0 = fe.assembly.GC.clone()

perturbation = 1e-4

pressure_load.pressure += perturbation

fe.solve(GC0=GC0, if_initialize=False)
GC1 = fe.assembly.GC.clone()

# check jacobian
GC_diff = (GC1 - GC0).cpu().numpy() / perturbation
jacobian_np = jacobian.cpu().numpy()

error = np.linalg.norm(GC_diff - jacobian_np) / np.linalg.norm(GC_diff)

assert False
