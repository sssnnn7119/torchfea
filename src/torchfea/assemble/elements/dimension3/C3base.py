from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ... import Part


import re
import time
import pandas as pd
import torch
import numpy as np
from ..base import BaseElement


class Element_3D(BaseElement):

    def __init__(self, elems_index: torch.Tensor,
                 elems: torch.Tensor, part: Part) -> None:
        super().__init__(elems_index, elems, part)

        self.shape_function_d1_gaussian: torch.Tensor
        """
            the shape functions of each guassian point
                [
                    g: guassian point
                    e: element
                    i: derivative
                    a: a-th node
                ]
        """

        self.shape_function_d0_gaussian: torch.Tensor
        """the shape functions of each guassian point [guassian, element, node]"""

        self.shape_function: list[torch.Tensor]
        """
            the shape functions of the element

            # coordinates: (g,h,r) in the local coordinates
                0: constant,\n
                1: g,\n
                2: h,\n
                3: r,\n
                4: g*h,\n
                5: h*r,\n
                6: r*g,\n
                7: g^2,\n
                8: h^2,\n
                9: r^2,\n
                10: g^2*h,\n
                11: h^2*g,\n
                12: h^2*r,\n
                13: r^2*h,\n
                14: r^2*g,\n
                15: g^2*r,\n
                16: g*h*r,\n
                17: g^3,\n
                18: h^3,\n
                19: r^3,\n

            # the shape of shape_function 
                
                a-th func,\n
                b-th coordinates
                
            # its derivative:
                
                m-th derivative,\n
                a-th func,\n
                b-th coordinates
                
            # and its 2-nd derivative
                
                m-th derivative,\n
                n-th derivative,\n
                a-th func,\n
                b-th coordinates
                
        """
                
        self.surf_order: torch.Tensor = torch.LongTensor([0, 0, 0, 0, 0, 0], device='cpu').to(torch.int8)
        """
            whether to reduce the order of the element, for the first order element, this parameter is not used,\n
            size: [surface] or [element, surface]

            0: slave surface, i.e., the node on the surface is determined by other surfaces,\n
            1: force reduce the order of the element on the surface,\n
            2: force remain the order of the element on the surface,\n

            priority: 2>1>0
        """

        self._num_gaussian: int
        """
            the number of guassian points
        """
    
        self.num_surfaces: int
        """
            the number of surfaces of the element
        """

        self.gaussian_coordinates: torch.Tensor
        """the coordinates of gaussian points in the reference space"""



    def initialize(self, *args, **kwargs) -> None:

        super().initialize(*args, **kwargs)
        # coo index of the stiffness matricx of structural stress

        index0_ = torch.stack([
                self._elems.T.reshape([self.num_nodes_per_elem, 1, 1, 1, -1]).repeat([1, 3, self.num_nodes_per_elem, 3, 1]),
                torch.arange(3, device='cpu').reshape([1, 3, 1, 1, 1]).repeat([self.num_nodes_per_elem, 1, self.num_nodes_per_elem, 3, self._elems.shape[0]]),
                self._elems.T.reshape([1, 1, self.num_nodes_per_elem, 1, -1]).repeat([self.num_nodes_per_elem, 3, 1, 3, 1]),
                torch.arange(3, device='cpu').reshape([1, 1, 1, 3, 1]).repeat([self.num_nodes_per_elem, 3, self.num_nodes_per_elem, 1, self._elems.shape[0]])
            ], dim=0).reshape([4, -1])
        index0 = torch.zeros([2, index0_.shape[1]], dtype=torch.int64, device='cpu')
        index0[0] = index0_[0] * 3 + index0_[1]
        index0[1] = index0_[2] * 3 + index0_[3]

        # some trick to get the unique index and accelerate the calculation
        scaler = index0.max() + 1
        index1 = index0[0] * scaler + index0[1]
        index_sorted_matrix = index1.argsort()
        index2 = index1[index_sorted_matrix]
        index_unique, self._index_matrix_coalesce = torch.unique_consecutive(
            index2, return_inverse=True)

        inverse_index = torch.zeros_like(index_sorted_matrix,
                                         device='cpu',
                                         dtype=torch.int64)
        inverse_index[index_sorted_matrix] = torch.arange(
            0, index_sorted_matrix.max() + 1, device='cpu', dtype=torch.int64)

        default_device = torch.zeros([1]).device

        self._index_matrix_coalesce = self._index_matrix_coalesce[inverse_index].to(
            default_device)
        self._indices_matrix = torch.zeros([2, index_unique.shape[0]],
                                          dtype=torch.int64)
        self._indices_matrix[1] = index_unique % scaler
        self._indices_matrix[0] = index_unique // scaler

        # coo index of the force vector of structural stress
        self._indices_force = self._elems[:, :self.num_nodes_per_elem].transpose(0, 1).unsqueeze(1).repeat(
            1, 3, 1)
        self._indices_force *= 3
        self._indices_force[:, 1, :] += 1
        self._indices_force[:, 2, :] += 2
        self._indices_force = self._indices_force.flatten().to(default_device)

    def _pre_load_gaussian(self, gauss_coordinates: torch.Tensor, nodes: torch.Tensor):
        """
        load the guassian points and its weight

        Args:
            gauss_coordinates: [g, 3], the local coordinates of the element
            nodes: [p, 3], the global coordinates of the element
        """

        # get the coordinates of the guassian points
        self.gaussian_coordinates = gauss_coordinates.cpu()
        pp = self._get_interpolation_coordinates(gauss_coordinates)

        # get the possible surface order
        if self.surf_order.ndim == 1:
            self.surf_order = self.surf_order.unsqueeze(0).repeat(self._elems.shape[0], 1)
        self.surf_order = self.surf_order[:, :self.num_surfaces]
        surf_order_all = self._get_all_possible_surface_order()

        # prepare the information for the FEA
        shapeFun1 = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, self.num_nodes_per_elem])
        shapeFun0 = torch.zeros([self._num_gaussian, self._elems.shape[0], self.num_nodes_per_elem])
        det_Jacobian = torch.zeros([self._num_gaussian, self._elems.shape[0]])
        for order_ind in range(surf_order_all.shape[0]):
            surf_order_now = surf_order_all[order_ind]

            elem_index = torch.where((self.surf_order - surf_order_now).abs().sum(1) == 0)[0]
            if elem_index.shape[0] == 0:
                continue
            elem_now = self._elems[elem_index]

            # process the shape function for the reduced order elements
            shape0_now = self._reduce_order_shape_function(self.shape_function[0], surf_order_now)

            # get the derivative of the shape function
            shape1_now = torch.stack([
                    self._shape_function_derivative(shape0_now, 0),
                    self._shape_function_derivative(shape0_now, 1),
                    self._shape_function_derivative(shape0_now, 2),
                ],
                            dim=0)

            # calculate the Jacobian at the guassian points
            Jacobian = torch.zeros([self._num_gaussian, elem_index.shape[0], 3, 3])
            temp_ = torch.einsum('gb, mab->gma', pp, shape1_now)
            for i in range(self.num_nodes_per_elem):
                Jacobian  += torch.einsum('gm,ei->geim', temp_[:, :, i],
                                        nodes[elem_now[:, i]])

            # Jacobian_Function
            # J: g(Gaussian) * e * 3(ref) * 3(rest)
            det_Jacobian[:, elem_index] = Jacobian.det()
            inv_Jacobian = Jacobian.inverse()
            shapeFun1[:, elem_index] = torch.einsum('gemi,gb,mab->geia', inv_Jacobian, pp,
                                    shape1_now)
            
            shapeFun0[:, elem_index] = torch.einsum('ab, gb->ga', shape0_now,
                                      pp).unsqueeze(1)

        self.gaussian_weight = torch.einsum('ge, g->ge', det_Jacobian, self.gaussian_weight)
        self.shape_function_d1_gaussian = shapeFun1
        self.shape_function_d0_gaussian = shapeFun0
        
    def _shape_function_derivative(self, shape_function: torch.Tensor, ind: int):
        """
        get the derivative of the shape function

        Args:
            shape_function: [i, m], the shape function of the element
            ind: the index of the derivative

        Returns:
            torch.Tensor: the derivative of the shape function
        """

        # (1,x,y,z,xy,yz,zx,xx,yy,zz)
        result = torch.zeros_like(shape_function)
        if ind == 0:
            result[:, 0] = shape_function[:, 1]
            if shape_function.shape[1] > 4:
                result[:, 2] = shape_function[:, 4]
                result[:, 3] = shape_function[:, 6]
            if shape_function.shape[1] > 7:
                result[:, 1] = 2 * shape_function[:, 7]
            if shape_function.shape[1] > 10:
                result[:, 4] = 2 * shape_function[:, 10]
                result[:, 8] = shape_function[:, 11]
                result[:, 9] = shape_function[:, 14]
                result[:, 6] = 2 * shape_function[:, 15]
                result[:, 5] = shape_function[:, 16]
            if shape_function.shape[1] > 17:
                result[:, 7] = 3 * shape_function[:, 17]

        if ind == 1:
            result[:, 0] = shape_function[:, 2]
            if shape_function.shape[1] > 4:
                result[:, 1] = shape_function[:, 4]
                result[:, 3] = shape_function[:, 5]
            if shape_function.shape[1] > 7:
                result[:, 2] = 2 * shape_function[:, 8]
            if shape_function.shape[1] > 10:
                result[:, 7] = shape_function[:, 10]
                result[:, 4] = 2 * shape_function[:, 11]
                result[:, 5] = 2 * shape_function[:, 12]
                result[:, 9] = shape_function[:, 13]
                result[:, 6] = shape_function[:, 16]
            if shape_function.shape[1] > 17:
                result[:, 8] = 3 * shape_function[:, 18]
                

        if ind == 2:
            result[:, 0] = shape_function[:, 3]
            if shape_function.shape[1] > 4:
                result[:, 1] = shape_function[:, 6]
                result[:, 2] = shape_function[:, 5]
            if shape_function.shape[1] > 7:
                result[:, 3] = 2 * shape_function[:, 9]
            if shape_function.shape[1] > 10:
                result[:, 8] = shape_function[:, 12]
                result[:, 5] = 2 * shape_function[:, 13]
                result[:, 6] = 2 * shape_function[:, 14]
                result[:, 7] = shape_function[:, 15]
                result[:, 4] = shape_function[:, 16]
            if shape_function.shape[1] > 17:
                result[:, 9] = 3 * shape_function[:, 19]

        return result
    
    def _get_interpolation_coordinates(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        Generate interpolation coordinates for shape functions.
        This method constructs a matrix of polynomial terms used for shape function interpolation
        in a 3D element. It builds terms based on the shape function's complexity, supporting
        constant, linear, quadratic, and cubic terms along with mixed terms.

        Args:
            nodes(torch.Tensor):
                Gaussian integration points with shape [num_gaussian, 3],
                containing the (x,y,z) coordinates of each point.

        Returns:
            torch.Tensor: 
                Matrix of polynomial terms with shape [num_gaussian, num_terms],
                where num_terms depends on the polynomial order of the shape functions:
                - 4 terms for linear (constant + x, y, z)
                - 7 terms for bilinear (adds xy, yz, zx)
                - 10 terms for quadratic (adds x², y², z²)
                - 17 terms for cubic without full terms (adds mixed quadratic terms + xyz)
                - 20 terms for full cubic (adds x³, y³, z³)
        """
        

        pp = torch.zeros([self._num_gaussian, self.shape_function[0].shape[1]], device=nodes.device)
        pp[:, 0] = 1
        pp[:, 1] = nodes[:, 0]
        pp[:, 2] = nodes[:, 1]
        pp[:, 3] = nodes[:, 2]
        if self.shape_function[0].shape[1] > 4:
            pp[:, 4] = nodes[:, 0] * nodes[:, 1]
            pp[:, 5] = nodes[:, 1] * nodes[:, 2]
            pp[:, 6] = nodes[:, 2] * nodes[:, 0]
        if self.shape_function[0].shape[1] > 7:
            pp[:, 7] = nodes[:, 0]**2
            pp[:, 8] = nodes[:, 1]**2
            pp[:, 9] = nodes[:, 2]**2
        if self.shape_function[0].shape[1] > 10:
            pp[:, 10] = nodes[:, 0]**2 * nodes[:, 1]
            pp[:, 11] = nodes[:, 1]**2 * nodes[:, 0]
            pp[:, 12] = nodes[:, 1]**2 * nodes[:, 2]
            pp[:, 13] = nodes[:, 2]**2 * nodes[:, 1]
            pp[:, 14] = nodes[:, 2]**2 * nodes[:, 0]
            pp[:, 15] = nodes[:, 0]**2 * nodes[:, 2]
            pp[:, 16] = nodes[:, 0] * nodes[:, 1] * \
                        nodes[:, 2]
        if self.shape_function[0].shape[1] > 17:
            pp[:, 17] = nodes[:, 0]**3
            pp[:, 18] = nodes[:, 1]**3
            pp[:, 19] = nodes[:, 2]**3
        
        return pp

    def get_gaussian_points(self, nodes: torch.Tensor):
        pp = self._get_interpolation_coordinates(self.gaussian_coordinates)
        shapeFun0 = torch.einsum('ab, gb->ga', self.shape_function[0].cpu(),
                                      pp)
        gaussian_position = torch.zeros(
            [self._num_gaussian, self._elems.shape[0], 3], device='cpu')
        for i in range(self._elems.shape[1]):
            gaussian_position = gaussian_position + torch.einsum(
                'g,eI->geI', shapeFun0[:, i], nodes[self._elems[:,
                                                                         i]].cpu())
        return gaussian_position.to(nodes.device)
    
    def get_mass_matrix(self,rotation_matrix:torch.Tensor=None):
        """
        Assemble the consistent mass matrix for the element.
        Returns:
            indices_force: torch.Tensor, indices for the force vector (flattened)
            Melement: torch.Tensor, element mass vector (flattened)
            indices_matrix: torch.Tensor, indices for the mass matrix (COO format)
            values: torch.Tensor, values for the global mass matrix (flattened)
        """
        # Consistent mass matrix: M_ij = ∫_Ω ρ N_i N_j dΩ
        # For each element, integrate N_i * N_j over the domain using Gaussian quadrature

        # shape_function_d0_gaussian: [num_gauss, num_elem, num_nodes_per_elem]
        N = self.shape_function_d0_gaussian  # [g, e, a]
        rho = self.density

        # Compute element mass matrix at each Gaussian point: [g, e, a, b]
        # M_ij = ∑_g N_i(g) * N_j(g) * w_g * detJ_g * ρ
        M_elem = torch.einsum('gea,geb,ge->abe', N, N, self.gaussian_weight * rho)

        # Expand to 3D (for vector-valued DoFs): [e, a, b] -> [a, j, b, k, e]
        # Only diagonal blocks are nonzero for lumped mass (consistent mass: block diagonal)
        num_elems = M_elem.shape[2]
        num_nodes = self.num_nodes_per_elem
        M_elem_full = torch.zeros([num_nodes, 3, num_nodes, 3, num_elems], device=M_elem.device, dtype=M_elem.dtype)
        for d in range(3):
            M_elem_full[:, d, :, d, :] = M_elem  # [a, b, e]

        # consider the rotation of the instance
        if rotation_matrix is not None:
            M_elem_full = torch.einsum('mj,ajbke,nk->ambne', rotation_matrix, M_elem_full, rotation_matrix)

        # Assemble into global matrix (same pattern as stiffness)
        values = torch.zeros([self._indices_matrix.shape[1]], device=M_elem.device, dtype=M_elem.dtype)
        values = values.scatter_add(0, self._index_matrix_coalesce, M_elem_full.flatten())

        return self._indices_matrix, values
        
    def potential_Energy(self, RGC: torch.Tensor):
        
        U = RGC
        Ugrad = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad = Ugrad + torch.einsum('gki,kI->gkIi',
                                         self.shape_function_d1_gaussian[:, :, :, i],
                                         U[self._elems[:, i]])

        F = Ugrad.clone()
        F[:, :, 0, 0] += 1
        F[:, :, 1, 1] += 1
        F[:, :, 2, 2] += 1

        J = F.det()
        I1 = (F**2).sum([-1, -2]) * J**(-2 / 3)

        W = torch.zeros([self._num_gaussian, self._elems.shape[0]])
        W = self.materials.strain_energy_density_C3(F=F,)
        
        Ea = torch.einsum(
            'ge,ge->',W,
            self.gaussian_weight)

        return Ea

    def structural_Force(self, RGC: torch.Tensor, rotation_matrix: Optional[torch.Tensor] = None, if_onlyforce: bool = False, *args, **kwargs):
        
        U = RGC

        if rotation_matrix is not None:
            U = torch.einsum('ij,aj->ai', rotation_matrix.T, U)

        DG, I1, J, invF, s, C = self.components_Solid(U=U)
        
        
        # calculate the element residual force
        Relement = torch.einsum('geij,geia,ge->aje', s,
                                self.shape_function_d1_gaussian,
                                self.gaussian_weight)
        
        if if_onlyforce:
            if rotation_matrix is not None:
                Relement = torch.einsum('mj,aje->ame', rotation_matrix, Relement)
            return self._indices_force, Relement.flatten()
                                
        
        # calculate the element tangential stiffness matrix
        Ka_element = torch.einsum('geijkl,gelb,geia,ge->ajbke',
                                   C,
                                  self.shape_function_d1_gaussian,
                                  self.shape_function_d1_gaussian,
                                  self.gaussian_weight)
        
        if rotation_matrix is not None:
            Relement = torch.einsum('mj,aje->ame', rotation_matrix, Relement)
            Ka_element = torch.einsum('mj,ajbke,nk->ambne', rotation_matrix, Ka_element, rotation_matrix)
        
        # assembly the stiffness matrix and residual force                 
        
        ## stiffness matrix

        values = torch.zeros([self._indices_matrix.shape[1]]).scatter_add(0, self._index_matrix_coalesce, Ka_element.flatten())
        

        return self._indices_force, Relement.flatten(), self._indices_matrix, values

    def components_Solid(self, U: torch.Tensor):
        Ugrad = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad = Ugrad + torch.einsum('gki,kI->gkIi',
                                         self.shape_function_d1_gaussian[:, :, :, i],
                                         U[self._elems[:, i]])

        F = Ugrad.clone()
        F[:, :, 0, 0] += 1
        F[:, :, 1, 1] += 1
        F[:, :, 2, 2] += 1

        invF = F.inverse()
        J = F.det()
        Jneg = J**(-2 / 3)
        I1 = (F**2).sum([-1, -2]) * Jneg
        
        s = torch.zeros_like(F)
        C = torch.zeros([s.shape[0], s.shape[1], 3, 3, 3, 3])

        s, C = self.materials.material_Constitutive_C3(F=F,
                                                    J=J,
                                                    Jneg=Jneg,
                                                    invF=invF,
                                                    I1=I1)

        return F, I1, J, invF, s, C

    def get_volumn(self, U: torch.Tensor = None):
        if U is None:
            return self.gaussian_weight.sum()
        else:
            Ugrad = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3])
            for i in range(self.num_nodes_per_elem):
                Ugrad = Ugrad + torch.einsum('gki,kI->gkIi',
                                            self.shape_function_d1_gaussian[:, :, :, i],
                                            U[self._elems[:, i]])
            F = Ugrad.clone()
            F[:, :, 0, 0] += 1
            F[:, :, 1, 1] += 1
            F[:, :, 2, 2] += 1
            J = F.det()
 
            return (self.gaussian_weight * J).sum()

    def set_required_DoFs(
            self, RGC_remain_index: np.ndarray) -> np.ndarray:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[self._elems.unique()] = True
        
        mid_nodes_index = self.get_2nd_order_point_index(order_required=1)
        if mid_nodes_index.shape[0] > 0:
            # set the mid nodes to be not required DoFs
            RGC_remain_index[mid_nodes_index[:, 0]] = False

        mid_nodes_index = self.get_2nd_order_point_index(order_required=2)
        if mid_nodes_index.shape[0] > 0:
            # set the mid nodes to be not required DoFs
            RGC_remain_index[mid_nodes_index[:, 0]] = True

        return RGC_remain_index
    
    # region second order methods

    def get_2nd_order_point_index(self, order_required = 1) -> torch.Tensor:
        """
        The absolute point index of the element that lies in the middle of the element

        get the 2-nd order point index of the element that lies in the middle of the element
        only for the first order faces of the second order element
        
        Returns:
            torch.Tensor: the 2-nd order point index of the element \n
                size: [point_index, 3]\n
                [0]: the index of the middle node of the element\n
                [1]: the index of the neighbor node of the middle node of the element\n
                [2]: the index of the other neighbor node of the middle node of the element\n
        """
        if self.surf_order.dim() == 1:
            # if the surf_order is a 1D tensor, it means that the same order is applied to all surfaces
            surf_order = self.surf_order.unsqueeze(0).repeat([self._elems.shape[0], 1])
        else:
            # if the surf_order is a 2D tensor, it means that different orders are applied to different surfaces
            surf_order = self.surf_order

        # get the mid node index
        mid_nodes_index = []
        for surf_ind in range(self.num_surfaces):
            # find which element surface to reduce
            ind_reduce_now = torch.where(surf_order[:, surf_ind] == order_required)[0]
            if ind_reduce_now.shape[0] == 0:
                # if there is no element to reduce, continue
                continue

            # get the relative point index of the element that lies in the middle of the element
            mid_nodes_index_now = self.get_2nd_order_point_index_surface(surf_ind)
            if mid_nodes_index_now.shape[0] == 0:
                # if there is no mid node, continue
                continue

            # add the mid node index to the list
            mid_nodes_index.append(self._elems[ind_reduce_now][:, mid_nodes_index_now].reshape([-1, 3]))

        if len(mid_nodes_index) == 0:
            # if there is no mid node, return an empty tensor
            return torch.zeros([0, 3], dtype=torch.int64, device='cpu')
        
        mid_nodes_index = torch.cat(mid_nodes_index, dim=0)
        mid_nodes_index = mid_nodes_index.unique(dim=0)

        return mid_nodes_index
    
    def get_2nd_order_point_index_surface(self, surface_ind: int) -> torch.Tensor:
        """
        The relative point index of the element that lies in the middle of the element

        get the 2-nd order point index of the element that lies in the middle of the element
        only for the first order faces of the second order element
        
        Args:
            surface_ind: the index of the surface, 0 for the first surface, 1 for the second surface, etc.
        
        Returns:
            torch.Tensor: the 2-nd order point index of the element \n
                size: [point_index, 3]\n
                    [0]: the index of the middle node of the element\n
                    [1]: the index of the neighbor node of the middle node of the element\n
                    [2]: the index of the other neighbor node of the middle node of the element\n
        """
        return torch.zeros([0, 3], dtype=torch.int64, device='cpu')
    
    def _get_all_possible_surface_order(self) -> torch.Tensor:
        """
        Get all possible surface orders for the element.

        Returns:
            torch.Tensor: A tensor containing the possible surface orders.
        """

        # For a second order element, the possible surface orders are 1 and 2
        # 1 means the surface is reduced to a mid node, 2 means the surface

        result = torch.ones([3**self.num_surfaces, self.num_surfaces], dtype=torch.int8, device='cpu')
        for i in range(self.num_surfaces):
            # set the i-th column to 1 or 2 based on the binary representation of the row index
            result[:, i] = (torch.arange(0, 3**self.num_surfaces, device='cpu') // (3**i)) % 3

        return result

    def _reduce_order_shape_function(self, shape_function: torch.Tensor, surf_order: torch.Tensor) -> torch.Tensor:
        """
        Reduce the order of the shape function by averaging the values of the neighboring nodes.

        Args:
            shape_function: [a, b], the shape function of the element
        Returns:
            torch.Tensor: [a, b], the reduced order shape function of the element
        """
    
        # get the mid node index
        mid_nodes_index_list = []
        for i in range(self.num_surfaces):
            if surf_order[i] == 1:
                # reduce the order of the shape function
                mid_nodes_index_list.append(
                    self.get_2nd_order_point_index_surface(i).cpu())
        if len(mid_nodes_index_list) == 0:
            # if there is no mid node, return an empty tensor
            return shape_function.clone()
        
        mid_nodes_index = torch.cat(mid_nodes_index_list, dim=0)

        # unique the mid_nodes_index
        mid_nodes_index: torch.Tensor = torch.unique(mid_nodes_index, dim=0)

        
        # if the mid point belong to a surface with order 2, then the node will not be delete
        mid2_nodes_index_list = []
        for i in range(self.num_surfaces):
            if surf_order[i] == 2:
                # reduce the order of the shape function
                mid2_nodes_index_list.append(
                    self.get_2nd_order_point_index_surface(i).cpu())

        if len(mid2_nodes_index_list) != 0:
            mid2_nodes_index = torch.cat(mid2_nodes_index_list, dim=0)
            mid2_nodes_index: torch.Tensor = torch.unique(mid2_nodes_index, dim=0)

            matches = (mid_nodes_index.unsqueeze(1) == mid2_nodes_index.unsqueeze(0)).all(dim=2)
            mask = ~matches.any(dim=1)
            mid_nodes_index = mid_nodes_index[mask]

        if mid_nodes_index.shape[0] == 0:
            # if there is no mid node, return the original shape function
            return shape_function.clone()

        # reduce the order of the shape function
        shape_function_reduced = shape_function.clone()
        for i in range(mid_nodes_index.shape[0]):
            mid_node = mid_nodes_index[i, 0]
            neighbor1 = mid_nodes_index[i, 1]
            neighbor2 = mid_nodes_index[i, 2]

            shape_function_reduced[mid_node] = 0.
            shape_function_reduced[neighbor1] += shape_function[mid_node] / 2
            shape_function_reduced[neighbor2] += shape_function[mid_node] / 2

        return shape_function_reduced

    def refine_RGC(self, RGC: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        """
        Refine the first order surface's nodes, to make them the middle nodes of the neighboring nodes.

        This method will not create new RGC, directly modify the input RGC.
        
        Args:
            RGC: List of Reference Grid Coordinates
            nodes: Node coordinates
            
        Returns:
            Updated RGC
        """
        mid_nodes_index = self.get_2nd_order_point_index(order_required=1)
        mid2_nodes_index = self.get_2nd_order_point_index(order_required=2)

        if mid_nodes_index.shape[0] > 0 and mid2_nodes_index.shape[0] > 0:
            # Remove from mid_nodes_index any nodes that are also in mid2_nodes_index
            # First convert to node indices only (first column of each tensor)
            mid_node_ids = mid_nodes_index[:, 0]
            mid2_node_ids = mid2_nodes_index[:, 0]
            
            # Find which mid nodes should be kept (not in mid2)
            mask = torch.where(~torch.isin(mid_node_ids, mid2_node_ids))[0]
            
            # Filter the mid_nodes_index to only include nodes not in mid2
            mid_nodes_index = mid_nodes_index[mask]
        
        RGC[mid_nodes_index[:, 0]] = (RGC[mid_nodes_index[:, 1]] + RGC[mid_nodes_index[:, 2]]) / 2 + (nodes[mid_nodes_index[:, 1]] + nodes[mid_nodes_index[:, 2]] - 2 * nodes[mid_nodes_index[:, 0]]) / 2
        
        return RGC
    
    # endregion second order methods