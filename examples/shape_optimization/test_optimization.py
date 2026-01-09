import pypardiso
import torch
import os
import numpy as np
import time
import sys
sys.path.append('.')
import scipy.sparse as sp
import torchfea

import pyvista as pv
os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)

fem = torchfea.FEA_INP()

fem.Read_INP(current_path + '/TopOptRun.inp')

fe = torchfea.from_inp(fem)

fe.add_load(torchfea.loads.Pressure(surface_set='surface_1_All', pressure=0.06),
                name='pressure-1')

bc_dof = np.array(
    list(fem.part['final_model'].sets_nodes['surface_0_Bottom'])) * 3
bc_dof = np.concatenate([bc_dof, bc_dof + 1, bc_dof + 2])
bc_name = fe.add_constraint(
    torchfea.constraints.Boundary_Condition(indexDOF=bc_dof,
                                    dispValue=torch.zeros(bc_dof.size)))

rp = fe.add_reference_point(torchfea.ReferencePoint([0, 0, 80]))

indexNodes = np.where((abs(fe.nodes[:, 2] - 80)
                        < 0.1).cpu().numpy())[0]
# torch_fea.add_constraint(
#     Constraints.Couple(
#         indexNodes=indexNodes,
#         rp_index=2))
fe.add_constraint(torchfea.constraints.Couple(indexNodes=indexNodes, rp_name=rp))

extern_surf = fe.get_surface_elements('surface_0_All')[0]._elems[:, :3].cpu().numpy()
intern_surf = fe.get_surface_elements('surface_1_All')[0]._elems[:, :3].cpu().numpy()

nodes_design = fe.nodes.clone().detach()
index_update = torch.where((fe.nodes[:, 2] > 0) & (fe.nodes[:, 2] < 80))[0]

fe.initialize()

history = []

plotter = pv.Plotter(window_size=(1080, 1920), off_screen=True)
plotter.set_background('white')
plotter.enable_parallel_projection()
for i in range(100):

    fe.nodes = nodes_design.clone().detach()
    fe.solve(tol_error=0.0001, RGC0=fe.RGC)

    GC0 = fe.GC.clone().detach()
    RGC0 = fe._GC2RGC(GC0)

    K_indices, K_values = fe._assemble_Stiffness_Matrix(
        RGC=RGC0)[1:]

    K_sp = sp.coo_matrix(
        (K_values.cpu().numpy(),
            (K_indices[0].cpu().numpy(), K_indices[1].cpu().numpy())),
        shape=(fe.GC.shape[0], fe.GC.shape[0])).tocsr()
    K_solver = pypardiso.PyPardisoSolver()
    K_solver.factorize(K_sp)

    ADJFu = torch.zeros_like(GC0).cpu().numpy()
    ADJFu[-2] = -1
    ADJu = K_solver.solve(K_sp, ADJFu)
    ADJu = torch.from_numpy(ADJu).to(GC0.device).type(GC0.dtype)

    nodes0 = fe.nodes.clone().detach().requires_grad_(True)

    fe.nodes = nodes0
    fe.initialize(RGC0=RGC0)

    R = fe._assemble_Stiffness_Matrix(RGC=RGC0)[0]

    work = (R*ADJu).sum()

    grad_pos = torch.autograd.grad(work, nodes0)[0]

    update_pos = grad_pos / grad_pos.flatten().abs().max() * 0.05

    nodes_design[index_update] += update_pos[index_update]

    print(f'Iteration {i}: displacement = {fe.GC[-2]}')
    history.append(fe.GC[-2].item())



    # extern_surf = fem.part['final_model'].surfaces['surface_1_All']


    coo=extern_surf

    # Get the deformed surface coordinates
    U = fe.RGC[0].cpu().numpy()
    undeformed_surface = (fem.part['final_model'].nodes[:,1:]).cpu().numpy()
    deformed_surface = undeformed_surface + U

    r=deformed_surface.transpose()


    Unorm = (U**2).sum(axis=1)**0.5


    # Plot the deformed surface
    plotter.clear()
    faces_extern = np.column_stack([np.full(len(extern_surf), 3), extern_surf])
    mesh_extern = pv.PolyData(deformed_surface, faces_extern)
    mesh_extern['displacement'] = Unorm
    plotter.add_mesh(mesh_extern, scalars='displacement', opacity=0.5)

    faces_intern = np.column_stack([np.full(len(intern_surf), 3), intern_surf])
    mesh_intern = pv.PolyData(deformed_surface, faces_intern)
    mesh_intern['displacement'] = Unorm
    plotter.add_mesh(mesh_intern, scalars='displacement', opacity=1.0)

    plotter.view_azimuth(90)
    plotter.view_elevation(90)

    plotter.screenshot(f'Z:/temp/iteration_{i:03d}.png')
