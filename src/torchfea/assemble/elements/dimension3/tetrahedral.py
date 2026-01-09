import numpy as np
import torch
from .C3base import Element_3D
from .surfaces import initialize_surfaces

class C3D4(Element_3D):
    """
        Local coordinates:
            origin: 0-th nodal
            \ksi_0: 0-1 vector
            \ksi_1: 0-2 vector
            \ksi_2: 0-3 vector

        face nodal always point at the void
            face0: 021
            face1: 013
            face2: 123
            face3: 032

        shape_funtion:
            N_i = \ksi_i * \ksi_i, i<=3
    """

    def __init__(self, elems: torch.Tensor = None, elems_index: torch.Tensor = None, part = None):
        super().__init__(elems=elems, elems_index=elems_index, part=part)
        self.num_surfaces = 4

    def initialize(self, *args, **kwargs):
        
        self.shape_function = [
            torch.tensor([[1.0, -1.0, -1.0, -1.0], [0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]),
        ]

        self.num_nodes_per_elem = 4
        self._num_gaussian = 1
        
        self.gaussian_weight = torch.tensor([1 / 6])

        p0 = torch.tensor([[0.25, 0.25, 0.25]])
        self._pre_load_gaussian(p0, nodes=self.part.nodes)
        super().initialize(*args, **kwargs)
        
    
    def extract_surface(self, surface_ind: int,
                           elems_ind: torch.Tensor):

        index_now = np.where(np.isin(self._elems_index, elems_ind))[0]

        if index_now.shape[0] == 0:
            tri_elems = torch.empty([0, 3], dtype=torch.long, device=self._elems.device)

        elif surface_ind == 0:
            tri_elems = self._elems[index_now][:, [0, 2, 1]]
        elif surface_ind == 1:
            tri_elems = self._elems[index_now][:, [0, 1, 3]]
        elif surface_ind == 2:
            tri_elems = self._elems[index_now][:, [1, 2, 3]]
        elif surface_ind == 3:
            tri_elems = self._elems[index_now][:, [0, 3, 2]]
        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")

        return [initialize_surfaces(tri_elems)]
    
class C3D10(Element_3D):
    """
        Local coordinates:
            origin: 0-th nodal
            \ksi_0: 0-1 vector
            \ksi_1: 0-2 vector
            \ksi_2: 0-3 vector

        face nodal always point at the void
            face0: 0(6)2(5)1(4)
            face1: 0(4)1(8)3(7)
            face2: 1(5)2(9)3(8)
            face3: 0(7)3(9)2(6)

        2-nd element extra nodals:
            4(01) 5(12) 6(02) 7(03) 8(13) 9(23)

        shape_funtion:
            N_i = (2 \ksi_i - 1) * \ksi_i, i<=2 \n
            N_i = 4 \ksi_j \ksi_k, i>2 and jk is the neighbor nodals fo i-th nodal
    """

    def __init__(self, elems: torch.Tensor = None, elems_index: torch.Tensor = None, part = None):
        super().__init__(elems=elems, elems_index=elems_index, part=part)
        self.num_surfaces = 4
        
    def initialize(self, *args, **kwargs):

        self.shape_function = [
            torch.tensor([[1., -3., -3., -3., 4., 4., 4., 2., 2., 2.],
                        [0., -1., 0., 0., 0., 0., 0., 2., 0., 0.],
                        [0., 0., -1., 0., 0., 0., 0., 0., 2., 0.],
                        [0., 0., 0., -1., 0., 0., 0., 0., 0., 2.],
                        [0., 4., 0., 0., -4., 0., -4., -4., 0., 0.],
                        [0., 0., 0., 0., 4., 0., 0., 0., 0., 0.],
                        [0., 0., 4., 0., -4., -4., 0., 0., -4., 0.],
                        [0., 0., 0., 4., 0., -4., -4., 0., 0., -4.],
                        [0., 0., 0., 0., 0., 0., 4., 0., 0., 0.],
                        [0., 0., 0., 0., 0., 4., 0., 0., 0., 0.]]),  
        ]

        self.gaussian_weight = torch.tensor([1 / 24, 1 / 24, 1 / 24, 1 / 24])

        self.num_nodes_per_elem = 10
        self._num_gaussian = 4
        
        alpha = 0.58541020
        beta = 0.13819660

        p0 = torch.tensor([[beta, beta, beta], [alpha, beta, beta],
                        [beta, alpha, beta], [beta, beta, alpha]])
            
            
        self._pre_load_gaussian(p0, nodes=self.part.nodes)
        super().initialize(*args, **kwargs)

    def extract_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        index_now = np.where(np.isin(self._elems_index, elems_ind))[0]
        
        if index_now.shape[0] == 0:
            return []
        
        if self.surf_order.dim() == 1:
            self.surf_order = self.surf_order.unsqueeze(0).repeat(self._elems.shape[0], 1)

        ind_1order = torch.where(self.surf_order[index_now, surface_ind] == 1)[0]
        ind_2order = torch.where(self.surf_order[index_now, surface_ind] != 1)[0]

        if surface_ind == 0:
            
            T3_elems = self._elems[index_now][ind_1order][:, [0, 2, 1]]
            T6_elems = self._elems[index_now][ind_2order][:, [0, 2, 1, 6, 5, 4]]

        elif surface_ind == 1:
            T3_elems = self._elems[index_now][ind_1order][:, [0, 1, 3]]
            T6_elems = self._elems[index_now][ind_2order][:, [0, 1, 3, 4, 8, 7]]
        elif surface_ind == 2:
            T3_elems = self._elems[index_now][ind_1order][:, [1, 2, 3]]
            T6_elems = self._elems[index_now][ind_2order][:, [1, 2, 3, 5, 9, 8]]
        elif surface_ind == 3:
            T3_elems = self._elems[index_now][ind_1order][:, [0, 3, 2]]
            T6_elems = self._elems[index_now][ind_2order][:, [0, 3, 2, 7, 9, 6]]
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
            return torch.tensor([[6, 0, 2],
                                    [5, 1, 2],
                                    [4, 0, 1]], dtype=torch.long, device='cpu')
        elif surface_ind == 1:
            return torch.tensor([[4, 0, 1],
                    [8, 1, 3],
                    [7, 0, 3]], dtype=torch.long, device='cpu')
        elif surface_ind == 2:
            return torch.tensor([[5, 1, 2],
                    [9, 2, 3],
                    [8, 1, 3]], dtype=torch.long, device='cpu')
        elif surface_ind == 3:
            return torch.tensor([[7, 0, 3],
                    [9, 2, 3],
                    [6, 0, 2]], dtype=torch.long, device='cpu')

        else:
            raise ValueError(f"Invalid surface index: {surface_ind}")
