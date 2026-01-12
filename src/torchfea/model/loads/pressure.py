import numpy as np
import torch
from .base import BaseLoad
from ..elements import BaseSurface
class Pressure(BaseLoad):

    def __init__(self, instance_name: str, surface_set: str, pressure: float) -> None:
        """
        initialize the pressure load on the surface element
        
        Args:
            surface_element (list[tuple[int, np.ndarray]]): the element index and the surface element index
            pressure (float): the pressure value
        """
        super().__init__()
        self.surface_name = surface_set
        self.instance_name = instance_name
        """Record the instance name and surface name
        """

        self.surface_element: list[BaseSurface]
        """
            the surface element
        """

        self._parameters = torch.tensor([pressure], dtype=torch.float64).flatten()
        """
        The pressure value applied to the surface element.
        """

        self._Vdot_indices: torch.Tensor
        """
        Indices for the force vector, used to apply the pressure load.
        """

        self._Vdot_2_indices: torch.Tensor
        """
        Indices for the second-order stiffness matrix, used to compute the stiffness contributions.
        """

        self._load_index: int

    @property
    def pressure(self) -> float:
        """
        Get the pressure value.
        """
        return self._parameters[0]
    
    @pressure.setter
    def pressure(self, value: float) -> None:
        """
        Set the pressure value.
        """
        self._parameters = torch.tensor([value], dtype=torch.float64).flatten()

    def initialize(self, assembly):
        super().initialize(assembly)

        self._load_index = self._assembly.get_instance(self.instance_name)._RGC_index
        index_offset = self._assembly.RGC_list_indexStart[self._load_index]

        self.surface_element = assembly.get_instance(self.instance_name).surfaces.get_elements(self.surface_name)

        _Vdot_indices = []
        _Vdot_2_indices = []

        for surf_ind in range(len(self.surface_element)):
            surf_elem = self.surface_element[surf_ind]
            if surf_elem._elems.shape[0] == 0:
                continue

            Vdot_indices = torch.stack([surf_elem._elems*3, surf_elem._elems*3+1, surf_elem._elems*3+2], dim=1).to(torch.int64).to(self._assembly.device)
            
            
            Vdot_2_indices = torch.stack([
                    surf_elem._elems.reshape([surf_elem._elems.shape[0], 1, surf_elem._elems.shape[1], 1, 1]).repeat([1, 3, 1, 3, surf_elem._elems.shape[1]]).flatten(),
                    torch.arange(3, device=surf_elem._elems.device).reshape([1, 3, 1, 1, 1]).repeat([surf_elem._elems.shape[0], 1, surf_elem._elems.shape[1], 3, surf_elem._elems.shape[1]]).flatten(),
                    surf_elem._elems.reshape([surf_elem._elems.shape[0], 1, 1, 1, surf_elem._elems.shape[1]]).repeat([1, 3, surf_elem._elems.shape[1], 3, 1]).flatten(),
                    torch.arange(3, device=surf_elem._elems.device).reshape([1, 1, 1, 3, 1]).repeat([surf_elem._elems.shape[0], 3, surf_elem._elems.shape[1], 1, surf_elem._elems.shape[1]]).flatten()
                ], dim=0).to(torch.int64).to(self._assembly.device)
            
            Vdot_2_indices = torch.stack([
                Vdot_2_indices[0] * 3 + Vdot_2_indices[1],
                Vdot_2_indices[2] * 3 + Vdot_2_indices[3]
            ], dim=0).to(torch.int64)

            _Vdot_indices.append(Vdot_indices)
            _Vdot_2_indices.append(Vdot_2_indices)
        
        self._Vdot_indices = torch.cat(_Vdot_indices, dim=0) + index_offset
        self._Vdot_2_indices = torch.cat(_Vdot_2_indices, dim=1) + index_offset
        
    def get_stiffness(self,
                RGC: list[torch.Tensor], if_onlyforce=False, *args, **kwargs):
        
        node_pos = self._assembly.get_instance(self.instance_name).nodes + RGC[self._load_index]

        # node_pos.requires_grad_()

        # V = self.get_potential_energy(RGC) / self.pressure * (-1)

        for surf_ind in range(len(self.surface_element)):
            surf_elem = self.surface_element[surf_ind]
            if surf_elem._elems.shape[0] == 0:
                continue
            
            # evaluate the deformed position and its derivatives at the Gaussian points
            shape_fun_added = torch.cat([surf_elem.shape_function_gaussian[0].unsqueeze(1),
                surf_elem.shape_function_gaussian[1]], dim=1)
            
            # V = [r, rdg, rdr] the mixed product of the deformed position and its derivatives
            # r_added_gaussian = [g, e, i, m] the deformed
            r_added_gaussian = torch.zeros([surf_elem._num_gaussian, surf_elem._elems.shape[0], 3, 3],
                device=self._assembly.device)
            
            for a in range(surf_elem.num_nodes_per_elem):
                r_added_gaussian += torch.einsum('gm, ei->geim', shape_fun_added[:, :, a],
                                           node_pos[surf_elem._elems[:, a]])
            
            

            det_ra = r_added_gaussian.det()
            inv_ra = r_added_gaussian.inverse()

            det_radra = torch.einsum('ge, gemi->geim', det_ra, inv_ra)

            Vdra = 1 / 3 * torch.einsum('g, geim->geim', surf_elem.gaussian_weight, det_radra)



            Vdot_values = torch.einsum('gma, geim->eia', shape_fun_added, Vdra)

            if if_onlyforce:
                continue

            part_I = torch.einsum('geim, genj->geimjn', det_radra, inv_ra)
            part_II = part_I.permute([0, 1, 2, 5, 4, 3])

            Vdra_2 = 1 / 3 * torch.einsum('g, geimjn->geimjn', surf_elem.gaussian_weight, part_I - part_II)
            Vdot_2_values = torch.einsum('gma, gnb, geimjn->eiajb', shape_fun_added, shape_fun_added, Vdra_2)
        
        if if_onlyforce:
            return self._Vdot_indices.flatten(), -self.pressure * Vdot_values.flatten()

        return self._Vdot_indices.flatten(), -self.pressure * Vdot_values.flatten(), self._Vdot_2_indices, -self.pressure * Vdot_2_values.flatten()

    def get_potential_energy(self, RGC: list[torch.Tensor]) -> torch.Tensor:

        node_pos = self._assembly.get_instance(self.instance_name).nodes + RGC[self._load_index]

        V = torch.scalar_tensor(0, device=self._assembly.device)
        for surf_ind in range(len(self.surface_element)):
            surf_elem = self.surface_element[surf_ind]
            if surf_elem._elems.shape[0] == 0:
                continue
            # evaluate the deformed position and its derivatives at the Gaussian points
            shape_fun_added = torch.cat([surf_elem.shape_function_gaussian[0].unsqueeze(1),
                surf_elem.shape_function_gaussian[1]], dim=1)
            
            # V = [r, rdg, rdr] the mixed product of the deformed position and its derivatives
            # r_added_gaussian = [g, e, i, m] the deformed
            r_added_gaussian = torch.zeros([surf_elem._num_gaussian, surf_elem._elems.shape[0], 3, 3], device=self._assembly.device)
            
            for a in range(surf_elem.num_nodes_per_elem):
                r_added_gaussian += torch.einsum('gm, ei->geim', shape_fun_added[:, :, a],
                                           node_pos[surf_elem._elems[:, a]])
            
            # evaluate the volume of the closed shell
            V_now = surf_elem.gaussian_weight.view([-1, 1]) * r_added_gaussian.det() 
            V += V_now.sum() / 3.

        return -self.pressure * V

    def set_required_DoFs(
            self, RGC_remain_index: list[np.ndarray]) -> list[np.ndarray]:
        """
        Modify the RGC_remain_index
        """
        for surf_ind in range(len(self.surface_element)):

            RGC_remain_index[self._load_index][self.surface_element[surf_ind]._elems.flatten().unique().cpu()] = True
        return RGC_remain_index
