import os
import sys

import numpy as np
import torch

sys.path.append(os.getcwd())

import torchfea


current_path = os.path.dirname(os.path.abspath(__file__))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.set_default_device(torch.device('cuda'))
torch.set_default_dtype(torch.float64)


def fea_inp(element_name: str):
    fem = torchfea.FEA_INP()
    fem.read_inp(os.path.join(current_path, element_name + '.inp'))

    fe = torchfea.from_inp(fem)
    part_name = 'Part-1'
    part = fe.assembly.get_part(part_name)
    instance = fe.assembly.get_instance(part_name)
    element = part.elems[element_name]

    bottom_nodes = np.where((abs(part.nodes[:, 2] - 0) < 0.1).cpu().numpy())[0]
    top_nodes = np.where((abs(part.nodes[:, 2] - 100) < 0.1).cpu().numpy())[0]
    part.add_node_set('bottom', bottom_nodes)
    part.add_node_set('top', top_nodes)

    fe.assembly.add_boundary(
        torchfea.boundarys.Boundary_Condition(
            instance_name=part_name,
            set_nodes_name='bottom',
        )
    )

    rp = fe.assembly.add_reference_point(torchfea.ReferencePoint([0, 0, 100]))
    fe.assembly.add_constraint(
        torchfea.constraints.Couple(
            instance_name=part_name,
            set_nodes_name='top',
            rp_name=rp,
        )
    )
    fe.assembly.add_load(
        torchfea.loads.BodyForce(
            instance_name=part_name,
            element_name=element_name,
            force_density=[9.8e-6, 0, 0],
        )
    )

    fe.solver = torchfea.solver.StaticImplicitSolver(tol_error=1e-6)
    result = fe.solve()

    U = fe.GC2RGC(result.GC)[instance._RGC_index].cpu().numpy()
    undeformed = part.nodes.cpu().numpy()
    _deformed = undeformed + U

    print(result.GC[-6:])
    print('ok')
    return result.GC[-6:].tolist()


if __name__ == '__main__':
    element_list = ['C3D10', 'C3D15', 'C3D20', 'C3D4', 'C3D6', 'C3D8']

    results = {}
    for element_name in element_list:
        print(f'Running test for element: {element_name}')
        results[element_name] = fea_inp(element_name)

    print('All tests completed.')
    print('Results:')
    for element_name, result in results.items():
        print(f'{element_name}\t: {result}')
