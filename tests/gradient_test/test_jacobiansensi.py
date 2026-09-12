import pypardiso
import torch
import os
import numpy as np
import time
import sys

from torch.nn import grad
sys.path.append('.')
import scipy.sparse as sp
import torchfea
os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))

torch.set_default_device(torch.device('cpu'))
torch.set_default_dtype(torch.float64)


fem = torchfea.FEA_INP()
name = 'C3D4'
fem.read_inp(current_path + '/C3D4.inp')

fe = torchfea.from_inp(fem)
fe.solver = torchfea.solver.StaticImplicitSolver(tol_error=1e-6)
# fe._maximum_step_length = 0.3
# elems = torch_fea.materials.initialize_materials(2, torch.tensor([[1.44, 0.45]]))
# fe.elems['element-0'].set_materials(elems)

# torch_fea.add_load(Loads.Body_Force_Undeformed(force_volumn_density=[1e-5, 0.0, 0.0], elem_index=torch_fea.elems['C3D4']._elems_index))

fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06),
                name='pressure-1')
load2 = torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_2_All', pressure=0.02)
fe.assembly.add_load(load2,
                name='pressure-2')
                
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
fe.initialize()
if not os.path.exists('Z:/temp/%s_results1.npz' % name):
    load2.pressure = 0.02
    feresult1 = fe.solve()
    feresult1.save('Z:/temp/%s_results1.npz' % name)
else:
    feresult1 = torchfea.solver.StaticResult.load('Z:/temp/%s_results1.npz' % name)

if not os.path.exists('Z:/temp/%s_results2.npz' % name):
    load2.pressure = 0.05
    feresult2 = fe.solve()
    feresult2.save('Z:/temp/%s_results2.npz' % name)
else:
    feresult2 = torchfea.solver.StaticResult.load('Z:/temp/%s_results2.npz' % name)

fe.initialize()

part = fe.assembly.get_part('final_model')
solver: torchfea.solver.StaticImplicitSolver = fe.solver
def apply_design_vars(assembly: torchfea.Assembly,
                        design_vars: torch.Tensor,
                        ) -> None:
    part = assembly.get_part('final_model')
    part.nodes = design_vars.reshape(part.nodes.shape)

jacobian1 = solver.get_jacobian(feresult1, load_names=['pressure-1', 'pressure-2'])
jacobian2 = solver.get_jacobian(feresult2, load_names=['pressure-1', 'pressure-2'])
feresult1.jacobian = jacobian1
feresult2.jacobian = jacobian2

def compute_objective_jacobian(fe_results: list[torchfea.solver.StaticResult],
                        assembly: torchfea.Assembly,
                        ) -> torch.Tensor:
    # compute the sensitivity of the displacement
    assembly.set_work_conditions(feresult1.work_conditions)
    energy1 = assembly._total_Potential_Energy(GC=fe_results[1].GC)
    assembly.set_work_conditions(feresult2.work_conditions)
    energy0 = assembly._total_Potential_Energy(GC=fe_results[0].GC)
    print('E1=%f, E0=%f' % (energy1.item(), energy0.item()))
    return fe_results[0].jacobian['pressure-1'][-2, 0] * fe_results[0].GC[-2] * fe_results[0].jacobian['pressure-2'][-2, 0] * \
        (energy1 - energy0)

def compute_objective_jacobian0(fe_result: torchfea.solver.StaticResult,
                        assembly: torchfea.Assembly,
                        ) -> torch.Tensor:
    # compute the sensitivity of the displacement
    return fe_result.jacobian['pressure-1'][-2, 0] * fe_result.GC[-2] * fe_result.jacobian['pressure-2'][-2, 0]

grad_sensi_jacobian_batch = solver.get_jacobian_sensitivity_multistep(
    fe_results=[feresult1, feresult2],
    design_vars=part.nodes.reshape(-1),
    load_names=['pressure-1', 'pressure-2'],
    apply_func=apply_design_vars,
    compute_objective_funcs=compute_objective_jacobian,
    )

grad_sensi_jacobian = solver.get_jacobian_sensitivity(
    fe_result=feresult1,
    design_vars=part.nodes.reshape(-1),
    load_names=['pressure-1', 'pressure-2'],
    apply_func=apply_design_vars,
    compute_objective_func=compute_objective_jacobian0,
    )
grad_sensi_jacobian_single = grad_sensi_jacobian.reshape(part.nodes.shape)
grad_sensi_jacobian_batch = grad_sensi_jacobian_batch.reshape(part.nodes.shape)
nodes0 = part.nodes.clone().detach()
epsilon = 1e-2
test_pair = ((2, 1), (10, 0), (5, 1))

index_test = torch.where(grad_sensi_jacobian_batch.abs() > 0.000001)

feresult1.jacobian = jacobian1
feresult2.jacobian = jacobian2
obj0 = compute_objective_jacobian([feresult1, feresult2], fe.assembly)

for i in range(index_test[0].shape[0]):
    indtest1 = index_test[0][i].item()
    indtest2 = index_test[1][i].item()
    # if (nodes0[indtest1, 2] != 70):
    #     continue
    part.nodes = nodes0.detach().clone()
    part.nodes[indtest1, indtest2] += epsilon

    load2.pressure = 0.02
    newresult1 = fe.solve(GC0=feresult1.GC)
    GC1 = newresult1.GC.clone().detach()
    jacobian1 = solver.get_jacobian(newresult1, load_names=['pressure-1', 'pressure-2'])
    newresult1.jacobian = jacobian1
    
    load2.pressure = 0.05
    newresult2 = fe.solve(GC0=feresult2.GC)
    GC2 = newresult2.GC.clone().detach()
    jacobian2 = solver.get_jacobian(newresult2, load_names=['pressure-1', 'pressure-2'])
    newresult2.jacobian = jacobian2

    obj1 = compute_objective_jacobian([newresult1, newresult2], fe.assembly)

    diff = (obj1 - obj0) / epsilon
    print('Testing node (%d, %d):' % (indtest1, indtest2))
    print('Finite difference Jacobian sensitivity:', diff.item())
    print('Autograd Jacobian sensitivity:', grad_sensi_jacobian_batch[indtest1, indtest2].item())
    print('Error:', abs(diff - grad_sensi_jacobian_batch[indtest1, indtest2].item()) / abs(diff))
    print('\n\n')

print('Gradient check for node position:')
print('Autograd gradient:')


assert False
