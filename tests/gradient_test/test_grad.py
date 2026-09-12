import pypardiso
import torch
import os
import numpy as np
import time
import sys
sys.path.append('.')
import scipy.sparse as sp

os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))
import pathlib
current_path = pathlib.Path(current_path)
torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)
import torchfea

name = 'C3D10'

fe = torchfea.load_model(current_path.parent / 'models' / f'{name}_model.npz')
fe.solver.tol_error = 1e-6
fe.initialize()
if not os.path.exists(current_path.parent / 'models' / f'{name}_results.npz'):
    
    feresult = fe.solve()
    feresult.save(current_path.parent / 'models' / f'{name}_results.npz')
else:
    feresult = torchfea.solver.StaticResult.load(current_path.parent / 'models' / f'{name}_results.npz')

feresult.GC = feresult.GC.to(fe.assembly.device)
fe.initialize()
GC0 = feresult.GC.clone().detach()
RGC0 = fe.assembly._GC2RGC(GC0)

K_indices, K_values = fe.assembly.assemble_Stiffness_Matrix(
    RGC=RGC0)[1:]

K_sp = sp.coo_matrix(
    (K_values.cpu().numpy(),
        (K_indices[0].cpu().numpy(), K_indices[1].cpu().numpy())),
    shape=(GC0.shape[0], GC0.shape[0])).tocsr()
K_solver = pypardiso.PyPardisoSolver()
K_solver.factorize(K_sp)

ADJFu = torch.zeros_like(GC0).cpu().numpy()
index_disp = -2
ADJFu[index_disp] = -1
ADJu = K_solver.solve(K_sp, ADJFu)
ADJu = torch.from_numpy(ADJu).to(GC0.device).type(GC0.dtype)


part = fe.assembly.get_part('final_model')

def closure_work(nodes_diff: torch.Tensor):
    part.nodes = nodes_diff
    fe.initialize()

    # compute the sensitivity of the displacement
    work = torch.tensor(0.0).to(part.nodes.device)
    R = fe.assembly.assemble_Stiffness_Matrix(GC=GC0)[0]
    work += (R*ADJu.detach()).sum()
    return work

grad_pos = torch.autograd.functional.jacobian(closure_work, part.nodes)

def apply_design_vars(assembly: torchfea.Assembly,
                        design_vars: torch.Tensor,
                        ) -> None:
    part = assembly.get_part('final_model')
    part.nodes = design_vars.reshape(part.nodes.shape)

def compute_objective(fe_result: torchfea.solver.StaticResult,
                        assembly: torchfea.Assembly,
                        ) -> torch.Tensor:
    # compute the sensitivity of the displacement
    return fe_result.GC[-2]
grad_sensi = fe.solver.get_sensitivity(
    fe_result=feresult,
    design_vars=part.nodes.reshape(-1),
    apply_func=apply_design_vars,
    compute_objective_func=compute_objective,
    )
# show_quiver3d(nodes0[index_remain].T, grad_pos[index_remain].T)

epsilon = 1e-2
test_pair = ((35021, 0), (10, 0), (5, 1))

nodes0 = part.nodes.clone().detach()
R0 = fe.assembly.assemble_Stiffness_Matrix(RGC=fe.assembly._GC2RGC(GC0))[0]


for i in range(len(test_pair)):
    indtest1 = test_pair[i][0]
    indtest2 = test_pair[i][1]
    # if (nodes0[indtest1, 2] != 70):
    #     continue
    part.nodes = nodes0.detach().clone()
    part.nodes[indtest1, indtest2] += epsilon
    fe.initialize()
    feresult1 = fe.solve(GC0=GC0, if_initialize=False)
    GC1 = feresult1.GC.clone().detach()
    R1partial = fe.assembly.assemble_Stiffness_Matrix(RGC=fe.assembly._GC2RGC(GC0))[0]

    diff = (GC1 - GC0)[index_disp] / epsilon
    diff1 = ((R1partial - R0)*ADJu / epsilon).sum()

    UdN = (GC1 - GC0) / epsilon
    RdN = (R1partial - R0) / epsilon
    K = torch.sparse_coo_tensor(K_indices, K_values, size=(GC0.shape[0], GC0.shape[0]))

    print('ind:', (indtest1, indtest2))
    print('nodes:', nodes0[indtest1].cpu().numpy())
    
    print('diff_R:', diff1.item())
    print('diff_U:', diff.item())
    print('grad_pos:', grad_pos[indtest1, indtest2].item())
    print('error:', abs(diff - grad_pos[indtest1, indtest2].item()) / abs(diff))
    print('\n\n')


    

    
