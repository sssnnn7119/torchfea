from io import StringIO
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from torchfea import FEA_INP, Instance
from torchfea.model.constraints import Couple
from torchfea.ui.workbench import Workbench
from torchfea.utils import skew_symmetric


@pytest.mark.parametrize("rotvec", [[0, 0, 0], [0.2, -0.3, 0.4], [0, 0, math.pi]])
def test_model_coupling_and_preview_use_the_same_rotation(rotvec):
    z = torch.tensor(rotvec, dtype=torch.float64)
    points = torch.tensor([[1., 2., -1.], [0.5, -0.2, 1.]])
    matrix = torch.matrix_exp(skew_symmetric(z))
    expected = points @ matrix.T
    instance = Instance("unused", rotation=rotvec)
    assert torch.allclose(Instance._transform(points, z), expected, atol=1e-12)
    if instance.rotation_matrix is not None:
        assert torch.allclose(instance.rotation_matrix, matrix, atol=1e-12)
    couple = Couple("unused", "all", "rp")
    couple._couple_index, couple._rp_index = 0, 1
    couple._indexNodes, couple._ref_location = np.arange(2), points
    coordinates = [torch.zeros_like(points), torch.cat((torch.zeros(3), z))]
    assert torch.allclose(couple.modify_RGC(coordinates)[0], expected - points, atol=1e-12)
    preview = Workbench._transform_points(points.numpy(), [1, 2, 3], rotvec)
    assert np.allclose(preview, expected.numpy() + [1, 2, 3], atol=1e-12)


@pytest.mark.parametrize("rotvec", [[0.2, -0.3, 0.4], [math.pi, 0, 0], [0, math.pi, 0]])
@pytest.mark.parametrize("translation", [[0., 0., 0.], [1., -2., 3.]])
def test_inp_instance_transform_round_trip(rotvec, translation):
    transform = np.eye(4)
    transform[:3, :3] = torch.matrix_exp(skew_symmetric(torch.tensor(rotvec))).numpy()
    transform[:3, 3] = translation
    assembly = FEA_INP.Assembly()
    assembly.instances = {"instance": ("part", transform)}
    stream = StringIO()
    assembly.write_inp(stream)
    loaded = FEA_INP.Assembly()
    loaded.read(stream.getvalue().splitlines(), 0)
    assert np.allclose(loaded.instances["instance"][1], transform, atol=1e-12)


def test_inp_rotation_about_an_offset_axis():
    loaded = FEA_INP.Assembly()
    loaded.read([
        "*Assembly, name=Assembly", "*Instance, name=instance, part=part",
        "0, 0, 0", "1, 2, 0, 1, 2, 1, 90", "*End Instance", "*End Assembly",
    ], 0)
    transform = loaded.instances["instance"][1]
    axis_start = np.array([1., 2., 0.])
    assert np.allclose(transform[:3, :3] @ axis_start + transform[:3, 3], axis_start)
    assert np.allclose(transform[:3, :3] @ [1., 0., 0.], [0., 1., 0.], atol=1e-12)


@pytest.mark.parametrize("rotvec", [[0., 0., 0.], [1e-9, -2e-9, 3e-9], [0.4, -0.7, 0.9]])
def test_couple_residual_and_tangent_with_shared_rotation_derivatives(rotvec):
    points = torch.tensor([[1., 2., -1.], [0.5, -0.2, 1.]])
    weights = torch.arange(1., 7.)
    couple = Couple("unused", "all", "rp")
    couple._assembly = SimpleNamespace(_RGC_list_indexStart=[0, 6, 12])
    couple._couple_index = couple._instance_RGC_index = 0
    couple._rp_index = 1
    couple._indexNodes, couple._ref_location = np.arange(2), points
    diagonal = torch.arange(6)
    nodal_indices = torch.stack((diagonal, diagonal))

    def evaluate(q, tangent=False):
        # Independent matrix exponential defines a quadratic nodal energy.
        displacement = q[:3] + points @ torch.matrix_exp(skew_symmetric(q[3:])).T - points
        residual = torch.cat((weights * displacement.flatten(), torch.zeros(6)))
        coordinates = [displacement, q]
        if not tangent:
            return couple.modify_R_K(coordinates, residual, if_onlyforce=True)[6:]
        force, indices, values = couple.modify_R_K(coordinates, residual, nodal_indices, weights)
        matrix = torch.sparse_coo_tensor(indices, values, (12, 12)).coalesce().to_dense()
        return force[6:], matrix[6:, 6:]

    def energy(q):
        displacement = q[:3] + points @ torch.matrix_exp(skew_symmetric(q[3:])).T - points
        return 0.5 * (weights * displacement.flatten()**2).sum()

    q = torch.tensor([0.1, -0.2, 0.3, *rotvec])
    residual, tangent = evaluate(q, tangent=True)
    assert torch.isfinite(tangent).all()
    assert torch.allclose(tangent, tangent.T, atol=1e-10)
    step = 1e-6
    for col in range(6):
        delta = torch.eye(6)[col] * step
        energy_difference = (energy(q + delta) - energy(q - delta)) / (2 * step)
        force_difference = (evaluate(q + delta) - evaluate(q - delta)) / (2 * step)
        assert residual[col].item() == pytest.approx(energy_difference.item(), abs=1e-7, rel=1e-6)
        assert torch.allclose(tangent[:, col], force_difference, atol=1e-7, rtol=1e-6)
