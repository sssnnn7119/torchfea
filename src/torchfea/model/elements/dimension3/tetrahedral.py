import torch

from .C3base import Element_3D


class C3D4(Element_3D):
    """
    Local coordinates:
        origin: 0-th nodal
        ksi_0: 0-1 vector
        ksi_1: 0-2 vector
        ksi_2: 0-3 vector

    face nodal always point at the void
        face0: 021
        face1: 013
        face2: 123
        face3: 032

    shape_funtion:
        N_i = ksi_i * ksi_i, i<=3
    """

    shape_function = [
        torch.tensor(
            [
                [1.0, -1.0, -1.0, -1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ]

    num_nodes_per_elem = 4
    _num_gaussian = 1

    gaussian_weight_ref = torch.tensor([1 / 6])

    gaussian_coordinates = torch.tensor([[0.25, 0.25, 0.25]])

    num_surfaces = 4

    surfaceid_map = {0: [0, 2, 1], 1: [0, 1, 3], 2: [1, 2, 3], 3: [0, 3, 2]}


class C3D10(Element_3D):
    """
    Local coordinates:
        origin: 0-th nodal
        ksi_0: 0-1 vector
        ksi_1: 0-2 vector
        ksi_2: 0-3 vector

    face nodal always point at the void
        face0: 0(6)2(5)1(4)
        face1: 0(4)1(8)3(7)
        face2: 1(5)2(9)3(8)
        face3: 0(7)3(9)2(6)

    2-nd element extra nodals:
        4(01) 5(12) 6(02) 7(03) 8(13) 9(23)

    shape_funtion:
        N_i = (2 ksi_i - 1) * ksi_i, i<=2 \n
        N_i = 4 ksi_j ksi_k, i>2 and jk is the neighbor nodals fo i-th nodal
    """

    shape_function = [
        torch.tensor(
            [
                [1.0, -3.0, -3.0, -3.0, 4.0, 4.0, 4.0, 2.0, 2.0, 2.0],
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0],
                [0.0, 4.0, 0.0, 0.0, -4.0, 0.0, -4.0, -4.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0, -4.0, -4.0, 0.0, 0.0, -4.0, 0.0],
                [0.0, 0.0, 0.0, 4.0, 0.0, -4.0, -4.0, 0.0, 0.0, -4.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    ]
    num_nodes_per_elem = 10
    _num_gaussian = 4
    gaussian_weight_ref = torch.tensor([1 / 24, 1 / 24, 1 / 24, 1 / 24])
    gaussian_coordinates = torch.tensor(
        [
            [0.13819660, 0.13819660, 0.13819660],
            [0.58541020, 0.13819660, 0.13819660],
            [0.13819660, 0.58541020, 0.13819660],
            [0.13819660, 0.13819660, 0.58541020],
        ]
    )
    num_surfaces = 4

    surfaceid_map = {
        0: [0, 2, 1, 6, 5, 4],
        1: [0, 1, 3, 4, 8, 7],
        2: [1, 2, 3, 5, 9, 8],
        3: [0, 3, 2, 7, 9, 6],
    }
