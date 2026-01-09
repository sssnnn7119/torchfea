import pypardiso
import torch
import os
import numpy as np
import time
import sys
sys.path.append('.')
import scipy.sparse as sp
import torchfea
os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)

def show_quiver3d(R, N, hold=False):
    import pyvista as pv
    r = R.detach().cpu().numpy()
    n = N.detach().cpu().numpy()
    points = pv.PolyData(r.T)
    vectors = n.T
    plotter = pv.Plotter()
    plotter.add_arrows(points, vectors)
    if not hold:
        plotter.show()

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


fe.solve(tol_error=0.001)

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


index_remain = fe.get_surface_elements('surface_1_All')[0]._elems.flatten().unique()

show_quiver3d(nodes0[index_remain].T, grad_pos[index_remain].T)

epsilon = 1e-3
test_pair = ((2, 1), (10, 0), (5, 1))

# index_test = torch.where(grad_pos.abs() > 0.0001)

# for i in range(index_test[0].shape[0]):
#     indtest1 = index_test[0][i].item()
#     indtest2 = index_test[1][i].item()
#     if (nodes0[indtest1, 2] == 0) or (nodes0[indtest1, 2] == 80):
#         continue
#     fe.nodes = nodes0.detach().clone()
#     fe.nodes[indtest1, indtest2] += epsilon
#     fe.solve(tol_error=0.001, RGC0=RGC0)
#     GC1 = fe.GC.clone().detach()

#     diff = (GC1 - GC0)[-2] / epsilon

#     print('ind:', (indtest1, indtest2))
#     print('diff:', diff.item())
#     print('grad_pos:', grad_pos[indtest1, indtest2].item())
#     print('error:', abs(diff - grad_pos[indtest1, indtest2].item()) / abs(diff))
#     print('\n\n')

