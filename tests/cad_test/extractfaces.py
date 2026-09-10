
import os
import time
import sys
import pathlib
import numpy as np
import torch
import pyvista as pv
os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))
current_path = pathlib.Path(current_path)
torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)


import torchfea

fem = torchfea.FEA_INP()
# fem.Read_INP(
#     'C:/Users/24391/OneDrive - sjtu.edu.cn/MineData/Learning/Publications/2024Arm/WorkspaceCase/CAE/TopOptRun.inp'
# )

# fem.Read_INP(
#     'Z:\RESULT\T20240325195025_\Cache/TopOptRun.inp'
# )

fem.read_inp(current_path.parent / 'models' / 'C3D4Less.inp')

fe = torchfea.from_inp(fem)
fe.solver = torchfea.solver.StaticImplicitSolver()
# elems = torch_fea.materials.initialize_materials(2, torch.tensor([[1.44, 0.45]]))
# fe.elems['element-0'].set_materials(elems)

# torch_fea.add_load(Loads.Body_Force_Undeformed(force_volumn_density=[1e-5, 0.0, 0.0], elem_index=torch_fea.elems['C3D4']._elems_index))

fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06),
                name='pressure-1')

bc_name = fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))

rp = fe.assembly.add_reference_point(torchfea.ReferencePoint([0, 0, 80]))

fe.assembly.add_constraint(torchfea.constraints.Couple(instance_name='final_model', set_nodes_name='surface_0_Head', rp_name=rp))

# fe.assembly.add_load(torchfea.loads.Penalty_DoF(s=2, k=1e4, target=20.0, obj_name=rp), name='penalty-1')

t1 = time.time()

result = fe.solve(tol_error=0.01)
# result.save('temp.npz')
# result = torchfea.solver.StaticResult.load('temp.npz')
# os.remove('temp.npz')


torch.cuda.synchronize()          # 确保 GPU 操作完成
# 3. 保存快照文件
torch.cuda.memory._dump_snapshot("my_snapshot.pickle")

# 4. 关闭记录，避免影响性能
torch.cuda.memory._record_memory_history(enabled=None)

print('ok')
print(result.GC[-6:])
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
