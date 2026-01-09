import torch
import os
import numpy as np
import time
import sys
sys.path.append('.')
import scipy.sparse as sp
import pypardiso
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
name = 'multi_case'
fem.read_inp(current_path + '/%s.inp'%name)

fe = torchfea.from_inp(fem)
fe.solver = torchfea.solver.StaticImplicitSolver()

# elems = torch_fea.materials.initialize_materials(2, torch.tensor([[1.44, 0.45]]))
# fe.elems['element-0'].set_materials(elems)

# torch_fea.add_load(Loads.Body_Force_Undeformed(force_volumn_density=[1e-5, 0.0, 0.0], elem_index=torch_fea.elems['C3D4']._elems_index))

fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.02),
                name='pressure-1')
fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_2_All', pressure=0.02),
                name='pressure-2')
fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_3_All', pressure=0.02),
                name='pressure-3')

bc_name = fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))
                                    

rp = fe.assembly.add_reference_point(torchfea.ReferencePoint([0, 0, 70]))

fe.assembly.add_constraint(torchfea.constraints.Couple(instance_name='final_model', set_nodes_name='surface_0_Head', rp_name=rp))

fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_0_All'))
fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_1_All'))
fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_2_All'))
fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_3_All'))
fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_4_All'))

t1 = time.time()

fe.initialize()

GC0_list = []
pressure_list = [[0.00, 0.06, 0.06], [0.06, 0.06, 0.06]]

if not os.path.exists(current_path + '/%s_results.npy' % name):
    for pind in range(len(pressure_list[0])):
        fe.assembly._loads['pressure-%d'%(pind+1)].pressure = pressure_list[0][pind]
    fe.solve(tol_error=1e-6)
    GC0_list.append(fe.assembly.GC.clone().detach())

    for pind in range(len(pressure_list[0])):
        fe.assembly._loads['pressure-%d'%(pind+1)].pressure = pressure_list[1][pind]
    fe.solve(tol_error=1e-6)
    GC0_list.append(fe.assembly.GC.clone().detach())

    np.save(current_path + '/%s_results' % name, np.array([gc.cpu().numpy() for gc in GC0_list]))
else:
    GC0_list_np = np.load(current_path + '/%s_results.npy' % name)
    GC0_list = [torch.from_numpy(gc).to(fe.assembly.GC.device).type(fe.assembly.GC.dtype) for gc in GC0_list_np]

GC0_list = torch.stack(GC0_list, dim=0)

def closure_adj(GC_now: torch.Tensor):
    loss1 = (GC_now[0][-2]-2.0)**2*100
    loss2 = -GC_now[1][-4]

    return loss1 + loss2

def closure_work(nodes_diff: torch.Tensor):
    nodes0 = part.nodes
    part.nodes = nodes_diff
    fe.initialize()

    # compute the sensitivity of the displacement
    work = closure_adj(GC0_list).to(part.nodes.device)
    for i in range(GC0_list.shape[0]):
        GC0 = GC0_list[i].to(part.nodes.device)
        fe.assembly.GC = GC0
        fe.assembly.RGC = fe.assembly._GC2RGC(GC0)

        for pind in range(len(pressure_list[0])):
            fe.assembly._loads['pressure-%d'%(pind+1)].pressure = pressure_list[i][pind]

        
        R = fe.assembly.assemble_Stiffness_Matrix(GC=GC0)[0]
        ADJu = ADJu_list[i].to(part.nodes.device)
        work = work + (R*ADJu).sum()
    part.nodes = nodes0
    fe.initialize()
    return work

ADJFu = -torch.autograd.functional.jacobian(closure_adj, GC0_list)
ADJu_list = []

for i in range(len(GC0_list)):
    GC0 = GC0_list[i]
    RGC0 = fe.assembly._GC2RGC(GC0)
    for pind in range(len(pressure_list[0])):
        fe.assembly._loads['pressure-%d'%(pind+1)].pressure = pressure_list[i][pind]
    K_indices, K_values = fe.assembly.assemble_Stiffness_Matrix(
        RGC=RGC0)[1:]

    K_sp = sp.coo_matrix(
        (K_values.cpu().numpy(),
            (K_indices[0].cpu().numpy(), K_indices[1].cpu().numpy())),
        shape=(fe.assembly.GC.shape[0], fe.assembly.GC.shape[0])).tocsr()
    K_solver = pypardiso.PyPardisoSolver()
    K_solver.factorize(K_sp)


    part = fe.assembly.get_part('final_model')

    ADJu = K_solver.solve(K_sp, ADJFu[i].cpu().numpy())
    ADJu = torch.from_numpy(ADJu).to(GC0.device).type(GC0.dtype)
    ADJu_list.append(ADJu)

grad_pos = torch.autograd.functional.jacobian(closure_work, part.nodes)

# show_quiver3d(nodes0[index_remain].T, grad_pos[index_remain].T)

epsilon = 1e-3
test_pair = ((2, 1), (10, 0), (5, 1))

nodes0 = part.nodes.clone().detach()
R0 = fe.assembly.assemble_Stiffness_Matrix(RGC=fe.assembly._GC2RGC(GC0))[0]
Rf0 = closure_adj(GC0_list)

index_test = torch.where(grad_pos.abs() > 0.000001)

for i in range(index_test[0].shape[0]):
    indtest1 = index_test[0][i].item()
    indtest2 = index_test[1][i].item()
    # if (nodes0[indtest1, 2] != 70):
    #     continue
    part.nodes = nodes0.detach().clone()
    part.nodes[indtest1, indtest2] += epsilon

    GC1_list = []
    for pind in range(len(pressure_list[0])):
        fe.assembly._loads['pressure-%d'%(pind+1)].pressure = pressure_list[0][pind]
    fe.solve(tol_error=1e-6, GC0=GC0_list[0])
    GC1 = fe.assembly.GC.clone().detach()
    GC1_list.append(GC1)
    
    for pind in range(len(pressure_list[0])):
        fe.assembly._loads['pressure-%d'%(pind+1)].pressure = pressure_list[1][pind]
    fe.solve(tol_error=1e-6, GC0=GC0_list[1])
    GC1 = fe.assembly.GC.clone().detach()
    GC1_list.append(GC1)

    diff = (closure_adj(GC1_list) - Rf0) / epsilon


    print('ind:', (indtest1, indtest2))
    print('nodes:', nodes0[indtest1].cpu().numpy())
    
    print('diff_U:', diff.item())
    print('grad_pos:', grad_pos[indtest1, indtest2].item())
    print('error:', abs(diff - grad_pos[indtest1, indtest2].item()) / abs(diff))
    print('\n\n')