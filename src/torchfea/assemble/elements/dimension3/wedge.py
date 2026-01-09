import numpy as np
import torch
from .C3base import Element_3D
from .surfaces import initialize_surfaces

class C3D6(Element_3D):
    """
    # Local coordinates:
        origin: 0-th nodal
        \ksi_0: 0-1 vector
        \ksi_1: 0-2 vector
        \ksi_2: 0-3 vector

    # face nodal always point at the void
        face0: 021 (Triangle)
        face1: 345 (Triangle)
        face2: 0143 (Rectangle)
        face3: 1254 (Rectangle)
        face4: 2035 (Rectangle)
    
    # shape_funtion:
        N_0 = 0.5 * (1 - \ksi_0 - \ksi_1) * (1 - \ksi_2) \n
        N_1 = 0.5 * \ksi_0 * (1 - \ksi_2) \n
        N_2 = 0.5 * \ksi_1 * (1 - \ksi_2) \n
        N_3 = 0.5 * (1 - \ksi_0 - \ksi_1) * (1 + \ksi_2) \n
        N_4 = 0.5 * \ksi_0 * (1 + \ksi_2) \n
        N_5 = 0.5 * \ksi_1 * (1 + \ksi_2) \n
    """
    
    def __init__(self, elems: torch.Tensor = None, elems_index: torch.Tensor = None, part = None):
        super().__init__(elems=elems, elems_index=elems_index, part=part)
        self.num_surfaces = 5
        
    
    def initialize(self, *args, **kwargs):
        
        # Shape function coefficients in format aligned with your other elements
        self.shape_function = [
            torch.tensor([
                [0.5, -0.5, -0.5, -0.5, 0.0, 0.5, 0.5],
                [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -0.5],
                [0.0, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0],
                [0.5, -0.5, -0.5, 0.5, 0.0, -0.5, -0.5],
                [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0]]),]
        self.num_nodes_per_elem = 6
        self._num_gaussian = 2
        self.gaussian_weight = torch.tensor([1 / 2, 1 / 2, ])

        # get the interpolation coordinates of the guass_points
        p0 = torch.tensor([[1/3, 1/3, 1 / np.sqrt(3)],
                           [1/3, 1/3, -1 / np.sqrt(3)]])

        # Gauss weights
        # gaussian_weight_triangle = torch.tensor([1 / 6, 1 / 6, 1 / 6])
        # gaussian_points_triangle = torch.tensor([[1 / 6, 1 / 6],
        #                                             [2 / 3, 1 / 6],
        #                                             [1 / 6, 2 / 3]])

        # gaussian_weight_height = torch.tensor([5 / 9, 8 / 9, 5 / 9])
        # gaussian_points_height = torch.tensor(
        #     [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])

        # # Combine weights and points for 3D integration
        # self.gaussian_weight = torch.einsum(
        #     'i,j->ij', gaussian_weight_triangle,
        #     gaussian_weight_height).flatten()
        # p0 = torch.cat([
        #     gaussian_points_triangle,
        #     torch.zeros([gaussian_points_triangle.shape[0], 1])
        # ],
        #                 dim=1)
        # p0 = p0.reshape([-1, 1, 3
        #                     ]).repeat([1, gaussian_points_height.shape[0], 1])
        # p0[:, :, 2] = gaussian_points_height.reshape([1, -1])

        # p0 = p0.reshape([-1, 3])

        # # Gauss integration points setup
        # self._num_gaussian = 9
        
        self._pre_load_gaussian(p0, nodes=self.part.nodes)
        super().initialize(*args, **kwargs)
        
    def extract_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        index_now = np.where(np.isin(self._elems_index, elems_ind))[0]
        
        if index_now.shape[0] == 0:
            tri_elems = torch.empty([0, 3], dtype=torch.long, device=self._elems.device)
            return []
        
        if surface_ind in [0, 1]:
            if surface_ind == 0:
                tri_elems = self._elems[index_now][:, [0, 2, 1]]
            elif surface_ind == 1:
                tri_elems = self._elems[index_now][:, [3, 4, 5]]
            return [initialize_surfaces(tri_elems)]
        elif surface_ind in [2, 3, 4]:
            if surface_ind == 2:
                quad_elems = self._elems[index_now][:, [0, 1, 4, 3]]
            elif surface_ind == 3:
                quad_elems = self._elems[index_now][:, [1, 2, 5, 4]]
            elif surface_ind == 4:
                quad_elems = self._elems[index_now][:, [2, 0, 3, 5]]
            return [initialize_surfaces(quad_elems)]

        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")


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

    def __init__(self,
                 elems: torch.Tensor = None,
                 elems_index: torch.Tensor = None,
                 part = None):
        super().__init__(elems=elems, elems_index=elems_index, part=part)
        self.num_surfaces = 5
        # Shape function coefficients and derivatives
        # Format: [shape_function, derivatives]
        # These matrices represent the shape functions and their derivatives
        # for the 15-node prismatic element

    def initialize(self, *args, **kwargs):

        # Shape function matrix (coefficients for each node's shape function)
        self.shape_function = [
            # Shape functions for all 15 nodes
            torch.tensor([[
                0, -1.0, -1.0, -0.5, 2.0, 1.5, 1.5, 1.0, 1.0, 0.5, 0, 0,
                -1.0, -0.5, -0.5, -1.0, -2.0, 0, 0, 0
            ],
                            [
                                0, -1.0, 0, 0, 0, 0, 0.5, 1.0, 0, 0, 0, 0, 0,
                                0, 0.5, -1.0, 0, 0, 0, 0
                            ],
                            [
                                0, 0, -1.0, 0, 0, 0.5, 0, 0, 1.0, 0, 0, 0,
                                -1.0, 0.5, 0, 0, 0, 0, 0, 0
                            ],
                            [
                                0, -1.0, -1.0, 0.5, 2.0, -1.5, -1.5, 1.0,
                                1.0, 0.5, 0, 0, 1.0, -0.5, -0.5, 1.0, 2.0, 0,
                                0, 0
                            ],
                            [
                                0, -1.0, 0, 0, 0, 0, -0.5, 1.0, 0, 0, 0, 0,
                                0, 0, 0.5, 1.0, 0, 0, 0, 0
                            ],
                            [
                                0, 0, -1.0, 0, 0, -0.5, 0, 0, 1.0, 0, 0, 0,
                                1.0, 0.5, 0, 0, 0, 0, 0, 0
                            ],
                            [
                                0, 2.0, 0, 0, -2.0, 0, -2.0, -2.0, 0, 0, 0,
                                0, 0, 0, 0, 2.0, 2.0, 0, 0, 0
                            ],
                            [
                                0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, -2.0, 0, 0, 0
                            ],
                            [
                                0, 0, 2.0, 0, -2.0, -2.0, 0, 0, -2.0, 0, 0,
                                0, 2.0, 0, 0, 0, 2.0, 0, 0, 0
                            ],
                            [
                                0, 2.0, 0, 0, -2.0, 0, 2.0, -2.0, 0, 0, 0, 0,
                                0, 0, 0, -2.0, -2.0, 0, 0, 0
                            ],
                            [
                                0, 0, 0, 0, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 2.0, 0, 0, 0
                            ],
                            [
                                0, 0, 2.0, 0, -2.0, 2.0, 0, 0, -2.0, 0, 0, 0,
                                -2.0, 0, 0, 0, -2.0, 0, 0, 0
                            ],
                            [
                                1.0, -1.0, -1.0, 0, 0, 0, 0, 0, 0, -1.0, 0,
                                0, 0, 1.0, 1.0, 0, 0, 0, 0, 0
                            ],
                            [
                                0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                -1.0, 0, 0, 0, 0, 0
                            ],
                            [
                                0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                -1.0, 0, 0, 0, 0, 0, 0
                            ]]),
        ]

        # Gauss weights
        gaussian_weight_triangle = torch.tensor([1 / 6, 1 / 6, 1 / 6])
        gaussian_points_triangle = torch.tensor([[1 / 6, 1 / 6],
                                                    [2 / 3, 1 / 6],
                                                    [1 / 6, 2 / 3]])

        gaussian_weight_height = torch.tensor([5 / 9, 8 / 9, 5 / 9])
        gaussian_points_height = torch.tensor(
            [-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)])

        # Combine weights and points for 3D integration
        self.gaussian_weight = torch.einsum(
            'i,j->ij', gaussian_weight_triangle,
            gaussian_weight_height).flatten()
        p0 = torch.cat([
            gaussian_points_triangle,
            torch.zeros([gaussian_points_triangle.shape[0], 1])
        ],
                        dim=1)
        p0 = p0.reshape([-1, 1, 3
                            ]).repeat([1, gaussian_points_height.shape[0], 1])
        p0[:, :, 2] = gaussian_points_height.reshape([1, -1])

        # Gauss integration points setup
        self.num_nodes_per_elem = 15
        self._num_gaussian = 9

        # Load the Gaussian points for integration
        self._pre_load_gaussian(p0.reshape([-1, 3]), nodes=self.part.nodes)
        super().initialize(*args, **kwargs)

    def extract_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        index_now = np.where(np.isin(self._elems_index, elems_ind))[0]
        
        if index_now.shape[0] == 0:
            return []
        
        ind_1order = torch.where(self.surf_order[index_now, surface_ind] == 1)[0]
        ind_2order = torch.where(self.surf_order[index_now, surface_ind] != 1)[0]

        if surface_ind == 0:
            # Bottom triangular face: 0(8)2(7)1(6) -> T6 elements
            T3_elems = self._elems[index_now][ind_1order][:, [0, 2, 1]]
            T6_elems = self._elems[index_now][ind_2order][:, [0, 2, 1, 8, 7, 6]]
            
        elif surface_ind == 1:
            # Top triangular face: 3(9)4(10)5(11) -> T6 elements

            T3_elems = self._elems[index_now][ind_1order][:, [3, 4, 5]]
            T6_elems = self._elems[index_now][ind_2order][:, [3, 4, 5, 9, 10, 11]]
            
        elif surface_ind == 2:
            # Rectangular face: 0(6)1(13)4(9)3(12) -> Q8 elements

            quad_elems = self._elems[index_now][:, [0, 1, 4, 3]]

            quad_elems = self._elems[index_now][:, [0, 1, 4, 3, 6, 13, 9, 12]]
            
        elif surface_ind == 3:
            # Rectangular face: 1(7)2(14)5(10)4(13) -> Q8 elements

            quad_elems = self._elems[index_now][:, [1, 2, 5, 4]]

            quad_elems = self._elems[index_now][:, [1, 2, 5, 4, 7, 14, 10, 13]]
            
        elif surface_ind == 4:
            # Rectangular face: 2(8)0(12)3(11)5(14) -> Q8 elements
            quad_elems = self._elems[index_now][:, [2, 0, 3, 5]]

            quad_elems = self._elems[index_now][:, [2, 0, 3, 5, 8, 12, 11, 14]]
            
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")
        
        result = []
        if T3_elems.shape[0] > 0:
            result.append(initialize_surfaces(T3_elems))
        if T6_elems.shape[0] > 0:
            result.append(initialize_surfaces(T6_elems))
        return result
    
    def get_2nd_order_point_index_surface(self, surface_ind: int):
        """
        Get the 2nd order point index for the specified surface.
        This is used to identify the mid-edge nodes for the surface elements.
        """
        if surface_ind == 0:
            return torch.tensor([[8, 0, 2],
                                    [7, 1, 2],
                                    [6, 0, 1]], dtype=torch.long, device='cpu')
        if surface_ind == 1:
            return torch.tensor([[9, 3, 4],
                                    [10, 4, 5],
                                    [11, 3, 5]], dtype=torch.long, device='cpu')
        if surface_ind == 2:
            return torch.tensor([[6, 0, 1],
                                    [13, 1, 4],
                                    [9, 3, 4],
                                    [12, 0, 3]], dtype=torch.long, device='cpu')
        if surface_ind == 3:
            return torch.tensor([[7, 1, 2],
                                    [14, 2, 5],
                                    [10, 4, 5],
                                    [13, 1, 4]], dtype=torch.long, device='cpu')
        if surface_ind == 4:
            return torch.tensor([[8, 0, 2],
                                    [12, 0, 3],
                                    [11, 3, 5],
                                    [14, 2, 5]], dtype=torch.long, device='cpu')
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")
