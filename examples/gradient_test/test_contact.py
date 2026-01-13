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
name = 'contact_each'
fem.read_inp(current_path + '/%s.inp'%name)

fe = torchfea.from_inp(fem)
fe.solver = torchfea.solver.StaticImplicitSolver()
fe.assembly.get_instance('Part-2')._translation = torch.tensor([0, 0, 40.])

# elems = torch_fea.materials.initialize_materials(2, torch.tensor([[1.44, 0.45]]))
# fe.elems['element-0'].set_materials(elems)

# torch_fea.add_load(Loads.Body_Force_Undeformed(force_volumn_density=[1e-5, 0.0, 0.0], elem_index=torch_fea.elems['C3D4']._elems_index))

fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.02),
                name='pressure-1')

bc_name = fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))
                                    


bc_name = fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='Part-2', set_nodes_name='fix'))
bc_dof2 = fe.assembly.get_instance('Part-2').set_nodes['fix']


rp = fe.assembly.add_reference_point(torchfea.ReferencePoint([0, 0, 80]))

fe.assembly.add_constraint(torchfea.constraints.Couple(instance_name='final_model', set_nodes_name='surface_0_Head', rp_name=rp))

fe.assembly.add_load(torchfea.loads.Contact(instance_name1='final_model', instance_name2='Part-2', surface_name1='surface_0_All', surface_name2='surfaceblock'))

t1 = time.time()

fe.initialize()
if not os.path.exists('Z:/temp/%s_results.npz' % name):
    fearesult = fe.solve(tol_error=1e-6)
    fearesult.save('Z:/temp/%s_results.npz' % name)
else:
    fearesult = torchfea.solver.StaticResult.load('Z:/temp/%s_results.npz' % name)


GC0 = fearesult.GC.clone().detach()
RGC0 = fe.assembly._GC2RGC(GC0)

K_indices, K_values = fe.assembly.assemble_Stiffness_Matrix(
    RGC=RGC0)[1:]

K_sp = sp.coo_matrix(
    (K_values.cpu().numpy(),
        (K_indices[0].cpu().numpy(), K_indices[1].cpu().numpy())),
    shape=(fe.assembly.GC.shape[0], fe.assembly.GC.shape[0])).tocsr()
K_solver = pypardiso.PyPardisoSolver()
K_solver.factorize(K_sp)


part = fe.assembly.get_part('final_model')

def closure_adj(GC_now: torch.Tensor):
    R = fe.assembly._assemble_generalized_Matrix(RGC=fe.assembly._GC2RGC(GC_now))[0]
    R_now = R[:fe.assembly.RGC_list_indexStart[1]].reshape([-1, 3])
    R_bc = R_now[bc_dof2]
    return R_bc.sum(0)[1]

ADJF = -torch.autograd.functional.jacobian(closure_adj, GC0).cpu().numpy()
ADJu = K_solver.solve(K_sp, ADJF)
ADJu = torch.from_numpy(ADJu).to(GC0.device).type(GC0.dtype)

def closure_work(nodes_diff: torch.Tensor):
    part.nodes = nodes_diff
    fe.initialize()

    # compute the sensitivity of the displacement
    work = torch.tensor(0.0).to(part.nodes.device)
    R = fe.assembly.assemble_Stiffness_Matrix(GC=GC0)[0]
    work += (closure_adj(GC0) + R*ADJu).sum()
    return work

grad_pos = torch.autograd.functional.jacobian(closure_work, part.nodes)

# show_quiver3d(nodes0[index_remain].T, grad_pos[index_remain].T)

def apply_design_vars(assembly: torchfea.Assembly,
                        design_vars: torch.Tensor,
                        ) -> None:
    part = assembly.get_part('final_model')
    part.nodes = design_vars.reshape(part.nodes.shape)

def compute_objective(GC: torch.Tensor,
                        assembly: torchfea.Assembly,
                        ) -> torch.Tensor:
    # compute the sensitivity of the displacement
    R = assembly._assemble_generalized_Matrix(RGC=fe.assembly._GC2RGC(GC))[0]
    R_now = R[:fe.assembly.RGC_list_indexStart[1]].reshape([-1, 3])
    R_bc = R_now[bc_dof2]
    return R_bc.sum(0)[1]
    
grad_sensi = torchfea.solver.get_sensitivity_static(
    fe_result=fearesult,
    assembly=fe.assembly,
    design_vars=part.nodes.reshape(-1),
    apply_func=apply_design_vars,
    compute_objective_func=compute_objective,
    )

print('Gradient check for node position:')
print('Autograd gradient:')

raise False