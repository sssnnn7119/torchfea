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
if not os.path.exists('Z:/temp/%s_results.npz' % name):
    feresult = fe.solve(tol_error=1e-6)
    feresult.save('Z:/temp/%s_results.npz' % name)
else:
    feresult = torchfea.solver.StaticResult.load('Z:/temp/%s_results.npz' % name)
GC0 = feresult.GC.clone().detach()
RGC0 = fe.assembly._GC2RGC(GC0)

K_indices, K_values = fe.assembly.assemble_Stiffness_Matrix(
    RGC=RGC0)[1:]

K_sp = sp.coo_matrix(
    (K_values.cpu().numpy(),
        (K_indices[0].cpu().numpy(), K_indices[1].cpu().numpy())),
    shape=(fe.assembly.GC.shape[0], fe.assembly.GC.shape[0])).tocsr()
K_solver = pypardiso.PyPardisoSolver()
K_solver.factorize(K_sp)

objective = GC0[-2]

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

def apply_design_vars(assembly: torchfea.Assembly,
                        design_vars: torch.Tensor,
                        ) -> None:
    part = assembly.get_part('final_model')
    part.nodes = design_vars.reshape(part.nodes.shape)

def compute_objective(GC: torch.Tensor,
                        assembly: torchfea.Assembly,
                        ) -> torch.Tensor:
    # compute the sensitivity of the displacement
    return GC[-2]
    
grad_sensi = torchfea.solver.get_sensitivity_static(
    fe_result=feresult,
    assembly=fe.assembly,
    design_vars=part.nodes.reshape(-1),
    apply_func=apply_design_vars,
    compute_objective_func=compute_objective,
    )

print('Gradient check for node position:')
print('Autograd gradient:')


assert False