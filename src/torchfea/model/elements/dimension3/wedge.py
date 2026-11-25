import numpy as np
import torch

from .C3base import Element_3D


class C3D6(Element_3D):
    """
    # Local coordinates:
        origin: 0-th nodal
        ksi_0: 0-1 vector
        ksi_1: 0-2 vector
        ksi_2: 0-3 vector

    # face nodal always point at the void
        face0: 021 (Triangle)
        face1: 345 (Triangle)
        face2: 0143 (Rectangle)
        face3: 1254 (Rectangle)
        face4: 2035 (Rectangle)

    # shape_funtion:
        N_0 = 0.5 * (1 - ksi_0 - ksi_1) * (1 - ksi_2) \n
        N_1 = 0.5 * ksi_0 * (1 - ksi_2) \n
        N_2 = 0.5 * ksi_1 * (1 - ksi_2) \n
        N_3 = 0.5 * (1 - ksi_0 - ksi_1) * (1 + ksi_2) \n
        N_4 = 0.5 * ksi_0 * (1 + ksi_2) \n
        N_5 = 0.5 * ksi_1 * (1 + ksi_2) \n
    """

    shape_function = [
        torch.tensor(
            [
                [0.5, -0.5, -0.5, -0.5, 0.0, 0.5, 0.5],
                [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -0.5],
                [0.0, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0],
                [0.5, -0.5, -0.5, 0.5, 0.0, -0.5, -0.5],
                [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0],
            ]
        ),
    ]
    num_nodes_per_elem = 6
    _num_gaussian = 2
    gaussian_weight_ref = torch.tensor([1 / 2, 1 / 2])
    gaussian_coordinates = torch.tensor(
        [
            [1 / 3, 1 / 3, 1 / np.sqrt(3)],
            [1 / 3, 1 / 3, -1 / np.sqrt(3)],
        ]
    )
    num_surfaces = 5

    surfaceid_map = {
        0: [0, 2, 1],
        1: [3, 4, 5],
        2: [0, 1, 4, 3],
        3: [1, 2, 5, 4],
        4: [2, 0, 3, 5],
    }


class C3D15(Element_3D):
    """
    # Local coordinates:
        origin: bottom triangle center
        g, h: coordinates in triangle base
        r: coordinate along prism height

    # Node numbering:
        - Bottom face (r=-1): 0, 1, 2 (vertices), 6, 7, 8 (mid-edge)
        - Top face (r=1): 3, 4, 5 (vertices), 9, 10, 11 (mid-edge)
        - Middle nodes (r=0): 12, 13, 14 (on vertical edges)

    # Face description:
        face0: 0(8)2(7)1(6) (Triangle)
        face1: 3(9)4(10)5(11) (Triangle)
        face2: 0(6)1(13)4(9)3(12) (Rectangle)
        face3: 1(7)2(14)5(10)4(13) (Rectangle)
        face4: 2(8)0(12)3(11)5(14) (Rectangle)

    # Shape functions:
        Quadratic interpolation in all directions
        Combines triangular base shape functions with prismatic extrusion
    """

    shape_function = [
        torch.tensor(
            [
                [
                    0,
                    -1.0,
                    -1.0,
                    -0.5,
                    2.0,
                    1.5,
                    1.5,
                    1.0,
                    1.0,
                    0.5,
                    0,
                    0,
                    -1.0,
                    -0.5,
                    -0.5,
                    -1.0,
                    -2.0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    -1.0,
                    0,
                    0,
                    0,
                    0,
                    0.5,
                    1.0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.5,
                    -1.0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    -1.0,
                    0,
                    0,
                    0.5,
                    0,
                    0,
                    1.0,
                    0,
                    0,
                    0,
                    -1.0,
                    0.5,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    -1.0,
                    -1.0,
                    0.5,
                    2.0,
                    -1.5,
                    -1.5,
                    1.0,
                    1.0,
                    0.5,
                    0,
                    0,
                    1.0,
                    -0.5,
                    -0.5,
                    1.0,
                    2.0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    -1.0,
                    0,
                    0,
                    0,
                    0,
                    -0.5,
                    1.0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.5,
                    1.0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    0,
                    -1.0,
                    0,
                    0,
                    -0.5,
                    0,
                    0,
                    1.0,
                    0,
                    0,
                    0,
                    1.0,
                    0.5,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    2.0,
                    0,
                    0,
                    -2.0,
                    0,
                    -2.0,
                    -2.0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    2.0,
                    2.0,
                    0,
                    0,
                    0,
                ],
                [0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.0, 0, 0, 0],
                [
                    0,
                    0,
                    2.0,
                    0,
                    -2.0,
                    -2.0,
                    0,
                    0,
                    -2.0,
                    0,
                    0,
                    0,
                    2.0,
                    0,
                    0,
                    0,
                    2.0,
                    0,
                    0,
                    0,
                ],
                [
                    0,
                    2.0,
                    0,
                    0,
                    -2.0,
                    0,
                    2.0,
                    -2.0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -2.0,
                    -2.0,
                    0,
                    0,
                    0,
                ],
                [0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.0, 0, 0, 0],
                [
                    0,
                    0,
                    2.0,
                    0,
                    -2.0,
                    2.0,
                    0,
                    0,
                    -2.0,
                    0,
                    0,
                    0,
                    -2.0,
                    0,
                    0,
                    0,
                    -2.0,
                    0,
                    0,
                    0,
                ],
                [
                    1.0,
                    -1.0,
                    -1.0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    -1.0,
                    0,
                    0,
                    0,
                    1.0,
                    1.0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0, 0],
                [0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1.0, 0, 0, 0, 0, 0, 0],
            ]
        ),
    ]
    num_nodes_per_elem = 15
    _num_gaussian = 9
    gaussian_weight_ref = torch.tensor(
        [
            5.0 / 54.0,
            5.0 / 54.0,
            5.0 / 54.0,
            8.0 / 54.0,
            8.0 / 54.0,
            8.0 / 54.0,
            5.0 / 54.0,
            5.0 / 54.0,
            5.0 / 54.0,
        ]
    )
    gaussian_coordinates = torch.tensor(
        [
            [1 / 6, 1 / 6, -np.sqrt(3 / 5)],
            [2 / 3, 1 / 6, -np.sqrt(3 / 5)],
            [1 / 6, 2 / 3, -np.sqrt(3 / 5)],
            [1 / 6, 1 / 6, 0.0],
            [2 / 3, 1 / 6, 0.0],
            [1 / 6, 2 / 3, 0.0],
            [1 / 6, 1 / 6, np.sqrt(3 / 5)],
            [2 / 3, 1 / 6, np.sqrt(3 / 5)],
            [1 / 6, 2 / 3, np.sqrt(3 / 5)],
        ]
    )
    num_surfaces = 5

    surfaceid_map = {
        0: [0, 2, 1, 6, 7, 8],
        1: [3, 4, 5, 9, 10, 11],
        2: [0, 1, 4, 3, 6, 13, 9, 12],
        3: [1, 2, 5, 4, 7, 14, 10, 13],
        4: [2, 0, 3, 5, 8, 12, 11, 14],
    }
