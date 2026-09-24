
import torch
import os
import time
import sys

sys.path.append('.')
import torchfea
os.environ['KMP_DUPLICATE_LIB_OK']='True'
current_path = os.path.dirname(os.path.abspath(__file__))
import pathlib
current_path = pathlib.Path(current_path)

torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)

from typing import Optional, TYPE_CHECKING, Callable


fem = torchfea.FEA_INP()
name = 'C3D10'
fem.read_inp(current_path.parent / 'models' / f'{name}.inp')

fe = torchfea.from_inp(fem)
fe.solver = torchfea.solver.StaticImplicitSolver(tol_error=1e-6)
# fe._maximum_step_length = 0.3
# elems = torch_fea.materials.initialize_materials(2, torch.tensor([[1.44, 0.45]]))
# fe.elems['element-0'].set_materials(elems)

# torch_fea.add_load(Loads.Body_Force_Undeformed(force_volumn_density=[1e-5, 0.0, 0.0], elem_index=torch_fea.elems['C3D4']._elems_index))

fe.assembly.add_load(torchfea.loads.Pressure(instance_name='final_model', surface_set='surface_1_All', pressure=0.06),
                name='pressure-1')
                
# fe.assembly.add_load(torch_fea.loads.ContactSelf(surface_name='surface_0_All', penalty_distance_g=10, penalty_threshold_h=5.5))
# fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_0_All'))
# fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_1_All'))
# fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_2_All'))
# fe.assembly.add_load(torchfea.loads.ContactSelf(instance_name='final_model',surface_name='surface_3_All'))

bc_name = fe.assembly.add_boundary(
    torchfea.boundarys.Boundary_Condition(instance_name='final_model', set_nodes_name='surface_0_Bottom'))

rp = fe.assembly.add_reference_point(torchfea.ReferencePoint([0, 0, 70]))

fe.assembly.add_constraint(torchfea.constraints.Couple(instance_name='final_model', set_nodes_name='surface_0_Head', rp_name=rp))


t1 = time.time()
fe.initialize()
if not os.path.exists('/run/media/song/缓存/temp/%s_results.npz' % name):
    feresult = fe.solve()
    feresult.save('/run/media/song/缓存/temp/%s_results.npz' % name)
else:
    feresult = torchfea.solver.StaticResult.load('/run/media/song/缓存/temp/%s_results.npz' % name)
GC0 = feresult.GC.clone().detach()
RGC0 = fe.assembly._GC2RGC(GC0)



fe.initialize()

part = fe.assembly.get_part('final_model')
solver: torchfea.solver.StaticImplicitSolver = fe.solver
def apply_design_vars(assembly: torchfea.Assembly,
                        design_vars: torch.Tensor,
                        ) -> None:
    part = assembly.get_part('final_model')
    part.nodes = design_vars.reshape_as(part.nodes)

def compute_objective(fe_result: torchfea.solver.StaticResult,
                        assembly: torchfea.Assembly,
                        ) -> torch.Tensor:
    # compute the sensitivity of the displacement
    return fe_result.GC[-2]
    
grad_sensi = solver.get_sensitivity(
    fe_result=feresult,
    design_vars=part.nodes.reshape(-1),
    apply_func=apply_design_vars,
    compute_objective_func=compute_objective,
    )

jacobian = solver.get_jacobian(feresult, load_names=['pressure-1'])

def compute_objective_jacobian(fe_result: torchfea.solver.StaticResult,
                        assembly: torchfea.Assembly,
                        ) -> torch.Tensor:
    # compute the sensitivity of the displacement
    return fe_result.GC[-2] + fe_result.jacobian['pressure-1'][-2, 0]

grad_sensi_jacobian = solver.get_jacobian_sensitivity(
    fe_result=feresult,
    design_vars=part.nodes.reshape(-1),
    load_names=['pressure-1'],
    apply_func=apply_design_vars,
    compute_objective_func=compute_objective_jacobian,
    )
grad_sensi_jacobian = grad_sensi_jacobian.reshape_as(part.nodes)
jacobian = solver.get_jacobian(feresult, load_names=['pressure-1'])
jacobian0 = torch.cat([jacobian[load_name] for load_name in ['pressure-1']], dim=1)
nodes0 = part.nodes.clone().detach()
epsilon = 1e-1
test_pair = ((2, 1), (10, 0), (5, 1))


feresult.jacobian = {'pressure-1': jacobian0}
obj0 = compute_objective_jacobian(feresult, fe.assembly)

indtest1 = 15716
indtest2 = 0
# if (nodes0[indtest1, 2] != 70):
#     continue
part.nodes = nodes0.detach().clone()
part.nodes[indtest1, indtest2] += epsilon
result1 = fe.solve(GC0=GC0)
GC1 = result1.GC.clone().detach()
jacobian1 = solver.get_jacobian(result1, load_names=['pressure-1'])
jacobian1 = torch.cat([jacobian1[load_name] for load_name in ['pressure-1']], dim=1)
result1.jacobian = {'pressure-1': jacobian1}
obj1 = compute_objective_jacobian(result1, fe.assembly)


diff = (obj1 - obj0) / epsilon
print('Testing node (%d, %d):' % (indtest1, indtest2))
print('Finite difference Jacobian sensitivity:', diff.item())
print('Autograd Jacobian sensitivity:', grad_sensi_jacobian[indtest1, indtest2].item())
print('Error:', abs(diff - grad_sensi_jacobian[indtest1, indtest2].item()) / abs(diff))
print('\n\n')

print('Gradient check for node position:')
print('Autograd gradient:')


assert False
