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


fem = torchfea.FEA_INP()
name = 'C3D4'
fem.read_inp(current_path + '/C3D4.inp')

fe = torchfea.from_inp(fem)
fe.solver = torchfea.solver.StaticImplicitSolver()
# fe._maximum_step_length = 0.3
# elems = torch_fea.materials.initialize_materials(2, torch.tensor([[1.44, 0.45]]))
# fe.elems['element-0'].set_materials(elems)

# torch_fea.add_load(Loads.Body_Force_Undeformed(force_volumn_density=[1e-5, 0.0, 0.0], elem_index=torch_fea.elems['C3D4']._elems_index))

fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06),
                name='pressure-1')
                
# fe.assembly.add_load(torch_fea.loads.ContactSelf(surface_name='surface_0_All', penalty_distance_g=10, penalty_threshold_h=5.5))
fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_0_All'))
fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_1_All'))
fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_2_All'))
fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_3_All'))

bc_name = fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))

rp = fe.assembly.add_reference_point(torchfea.ReferencePoint([0, 0, 70]))

fe.assembly.add_constraint(torchfea.constraints.Couple(instance_name='final_model', set_nodes_name='surface_0_Head', rp_name=rp))




t1 = time.time()


t1 = time.time()
fe.initialize()
if not os.path.exists(current_path + '/%s_results.npy' % name):
    fe.solve(tol_error=1e-6)
    np.save(current_path + '/%s_results' % name, fe.assembly.GC.cpu().numpy())
else:
    fe.assembly.GC = torch.from_numpy(
        np.load(current_path + '/%s_results.npy' % name)).to(fe.assembly.GC.device).type(fe.assembly.GC.dtype)

GC0 = fe.assembly.GC.clone().detach()
RGC0 = fe.assembly._GC2RGC(GC0)

K_indices, K_values = fe.assembly.assemble_Stiffness_Matrix(
    RGC=RGC0)[1:]

K_sp = sp.coo_matrix(
    (K_values.cpu().numpy(),
        (K_indices[0].cpu().numpy(), K_indices[1].cpu().numpy())),
    shape=(fe.assembly.GC.shape[0], fe.assembly.GC.shape[0])).tocsr()
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

# show_quiver3d(nodes0[index_remain].T, grad_pos[index_remain].T)

epsilon = 1e-3
test_pair = ((2, 1), (10, 0), (5, 1))

nodes0 = part.nodes.clone().detach()
R0 = fe.assembly.assemble_Stiffness_Matrix(RGC=fe.assembly._GC2RGC(GC0))[0]

index_test = torch.where(grad_pos.abs() > 0.000001)

for i in range(index_test[0].shape[0]):
    indtest1 = index_test[0][i].item()
    indtest2 = index_test[1][i].item()
    # if (nodes0[indtest1, 2] != 70):
    #     continue
    part.nodes = nodes0.detach().clone()
    part.nodes[indtest1, indtest2] += epsilon
    fe.solve(tol_error=1e-6, RGC0=RGC0)
    GC1 = fe.assembly.GC.clone().detach()
    R1 = fe.assembly.assemble_Stiffness_Matrix(RGC=fe.assembly._GC2RGC(GC0))[0]

    diff = (GC1 - GC0)[index_disp] / epsilon
    diff1 = ((R1 - R0)*ADJu / epsilon).sum()

    UdN = (GC1 - GC0) / epsilon
    RdN = (R1 - R0) / epsilon
    K = torch.sparse_coo_tensor(K_indices, K_values, size=(GC0.shape[0], GC0.shape[0]))

    print('ind:', (indtest1, indtest2))
    print('nodes:', nodes0[indtest1].cpu().numpy())
    
    print('diff_R:', diff1.item())
    print('diff_U:', diff.item())
    print('grad_pos:', grad_pos[indtest1, indtest2].item())
    print('error:', abs(diff - grad_pos[indtest1, indtest2].item()) / abs(diff))
    print('\n\n')


    

    

