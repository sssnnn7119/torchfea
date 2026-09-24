
import os
import tempfile
import time
import sys
import pathlib
import numpy as np
import torch

os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))

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

path0 = pathlib.Path(current_path).parent / 'models' / 'C3D4Less.inp'
dir_path = pathlib.Path(current_path).parent / 'models' / 'C3D4_model.npz'

fe = torchfea.load_model(dir_path)

# torchfea.retrieve_source_code(dir_path, os.path.join(current_path, 'retrieved_source_code.py'))

fe.solver.tol_error = 0.01
result = fe.solve()
# result.save('temp.npz')
# result = torchfea.solver.StaticResult.load('temp.npz')
# os.remove('temp.npz')

print('ok')
print(result.GC[-6:])




# extern_surf = fe.loads['pressure-1'].surface_element.cpu().numpy()
instance = fe.assembly.get_instance('final_model')
extern_surf = instance.surfaces.get_elements('extern')[0]._elems[:, :3].cpu().numpy()
# extern_surf = fem.part['final_model'].surfaces['surface_1_All']

import pyvista as pv

# Get the deformed surface coordinates
U = fe.GC2RGC(result.GC)[instance._RGC_index].cpu().numpy()
undeformed_surface = instance.nodes.cpu().numpy()
deformed_surface = undeformed_surface + U

Unorm = (U**2).sum(axis=1)**0.5

# Plot the deformed surface
faces = np.column_stack([np.full(len(extern_surf), 3), extern_surf])
mesh = pv.PolyData(deformed_surface, faces)
mesh['displacement'] = Unorm
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='displacement')
plotter.show()
plotter.close()
