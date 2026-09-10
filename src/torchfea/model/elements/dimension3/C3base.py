from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ... import Part


import torch
import numpy as np
from ..base import BaseElement
from .surfaces import initialize_surfaces

class Element_3D(BaseElement):

    _serialized_attributes_exclude = ['shape_function_d0_gaussian', 'shape_function_d1_gaussian', 'shape_function_d2_gaussian', '_dNW', '_dNdNW', '_indices_matrix', '_indices_force', '_index_matrix_coalesce', 'gaussian_weight', 'shape_function_gaussian']

    num_surfaces: int
    """
        the number of surfaces of the element
    """

    surfaceid_map: dict[int, list[int]]
    """
        the mapping that maps the surface id to the node index of the element, which is used to extract the surface elements.
    """

    def __init_subclass__(cls):
        super().__init_subclass__()

        if hasattr(cls, 'shape_function'):
            cls.shape_function[0] = cls.shape_function[0].to(torch.get_default_device()).to(torch.get_default_dtype())
            cls.shape_function.append(torch.stack([
                    cls._shape_function_derivative(cls.shape_function[0], 0),
                    cls._shape_function_derivative(cls.shape_function[0], 1),
                    cls._shape_function_derivative(cls.shape_function[0], 2),
                ],
                            dim=0).to(torch.get_default_device()).to(torch.get_default_dtype()))
            
            cls.shape_function.append(torch.zeros(
                [3, 3, cls.shape_function[0].shape[0], cls.shape_function[0].shape[1]]))
            for i in range(3):
                for j in range(3):
                    cls.shape_function[2][i, j] = cls._shape_function_derivative(cls.shape_function[1][i], j).to(torch.get_default_device()).to(torch.get_default_dtype())

    def __init__(self, elems_index: torch.Tensor,
                 elems: torch.Tensor) -> None:
        super().__init__(elems_index, elems)

        self.shape_function_d2_gaussian: torch.Tensor
        """
            the second derivative of the shape function of each guassian point
                [
                    g: guassian point
                    e: element
                    i: derivative index 0
                    j: derivative index 1
                    a: a-th node
                ]
        """

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


        self._indices_matrix: torch.Tensor
        """
            the coo index of the stiffness matricx of structural stress
        """

        self._indices_force: torch.Tensor
        """
            the coo index of the tructural stress
        """

        self._index_matrix_coalesce: torch.Tensor = None
        """
            the start index of the stiffness matricx of structural stress
        """

        self._dNW: torch.Tensor
        """the derivative of the shape function multiplied by the guassian weight [guassian, element, derivative, node]"""

        self._dNdNW: torch.Tensor
        """the derivative of the shape function multiplied by the guassian weight [guassian, element, derivative, node, derivative, node]"""

        if hasattr(self, 'shape_function'):
            self.shape_function[0] = self.shape_function[0].to(torch.get_default_device()).to(torch.get_default_dtype())
            self.shape_function.append(torch.stack([
                    self._shape_function_derivative(self.shape_function[0], 0),
                    self._shape_function_derivative(self.shape_function[0], 1),
                    self._shape_function_derivative(self.shape_function[0], 2),
                ],
                            dim=0).to(torch.get_default_device()).to(torch.get_default_dtype()))
            
            self.shape_function.append(torch.zeros(
                [3, 3, self.shape_function[0].shape[0], self.shape_function[0].shape[1]]))
            for i in range(3):
                for j in range(3):
                    self.shape_function[2][i, j] = self._shape_function_derivative(self.shape_function[1][i], j).to(torch.get_default_device()).to(torch.get_default_dtype())


    def initialize(self, nodes: torch.Tensor, *args, **kwargs) -> None:

        super().initialize(nodes, *args, **kwargs)

        # pre load the gaussian points and its weight for the element, which will be used in the FEA calculation
        self._pre_load_gaussian(nodes=nodes)

        # coo index of the stiffness matricx of structural stress

        index0_ = torch.stack([
                self._elems.cpu().T.reshape([self.num_nodes_per_elem, 1, 1, 1, -1]).repeat([1, 3, self.num_nodes_per_elem, 3, 1]),
                torch.arange(3, device='cpu').reshape([1, 3, 1, 1, 1]).repeat([self.num_nodes_per_elem, 1, self.num_nodes_per_elem, 3, self._elems.shape[0]]),
                self._elems.cpu().T.reshape([1, 1, self.num_nodes_per_elem, 1, -1]).repeat([self.num_nodes_per_elem, 3, 1, 3, 1]),
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
                                         dtype=torch.int64, device='cpu')
        inverse_index[index_sorted_matrix] = torch.arange(
            0, index_sorted_matrix.max() + 1, dtype=torch.int64, device='cpu')


        self._index_matrix_coalesce = self._index_matrix_coalesce[inverse_index].to(
            torch.get_default_device())
        self._indices_matrix = torch.zeros([2, index_unique.shape[0]],
                                          dtype=torch.int64, device='cpu')
        self._indices_matrix[1] = index_unique % scaler
        self._indices_matrix[0] = index_unique // scaler
        self._indices_matrix = self._indices_matrix.to(torch.get_default_device())

        # coo index of the force vector of structural stress
        self._indices_force = self._elems.cpu()[:, :self.num_nodes_per_elem].transpose(0, 1).unsqueeze(1).repeat(
            1, 3, 1)
        self._indices_force *= 3
        self._indices_force[:, 1, :] += 1
        self._indices_force[:, 2, :] += 2
        self._indices_force = self._indices_force.flatten().to(torch.get_default_device())

    def _pre_load_gaussian(self, nodes: torch.Tensor):
        """
        Pre-compute shape function values & derivatives at Gaussian points.
        Uses isoparametric mapping: ξ (reference) → x (physical).

        Key steps:
          1. N_a(ξ)   : shape function values at Gauss points
          2. ∂N_a/∂ξ  : shape function gradients w.r.t. reference coords
          3. J = ∂x/∂ξ : Jacobian of isoparametric mapping
          4. ∂N_a/∂x = J^{-1} · ∂N_a/∂ξ : push-forward to physical gradients
          5. dV = det(J) · dξ : integration weight in physical space

        Args:
            nodes: [p, 3], global nodal coordinates x_i^a in physical space
        """

        # —— Step 1: polynomial basis p(ξ) at Gaussian points ——
        # pp[g, m]: m-th monomial term evaluated at Gauss point ξ_g
        pp = self._get_interpolation_coordinates(self.gaussian_coordinates)

        # ∂N_a/∂x_i [g, e, i, a],  N_a [g, e, a],  det(J) [g, e]
        det_Jacobian = torch.zeros([self._num_gaussian, self._elems.shape[0]])

        elem_now = self._elems

        # —— Step 2: Jacobian J_{ij} = ∂x_i/∂ξ_j ——
        # isoparametric mapping: x_i(ξ) = Σ_a N_a(ξ) · x_i^a
        # => J_{ij} = Σ_a (∂N_a/∂ξ_j) · x_i^a
        Jacobian = torch.zeros([self._num_gaussian, elem_now.shape[0], 3, 3])
        shape1_gaussian = torch.einsum('gb, mab->gma', pp, self.shape_function[1])
        # shape1_gaussian[g, m, a] = p_m(ξ_g) · C_{am}  (intermediate for ∂N_a/∂ξ)
        for i in range(self.num_nodes_per_elem):
            # J_{geij} += ∂N_a/∂ξ_j|_{ξ_g} · x_i^a
            Jacobian  += torch.einsum('gm,ei->geim', shape1_gaussian[:, :, i],
                                    nodes[elem_now[:, i]])

        # —— Step 3: det(J) and inverse J^{-1} ——
        det_Jacobian = Jacobian.det()
        inv_Jacobian = Jacobian.inverse()

        # —— Step 4: push-forward ∂N_a/∂x = J^{-1} · ∂N_a/∂ξ ——
        # ∂N_a/∂x_i = (J^{-1})_{ij} · ∂N_a/∂ξ_j
        self.shape_function_d1_gaussian = torch.einsum('gemi,gma->geia', inv_Jacobian, shape1_gaussian)

        # N_a(ξ) at Gaussian points: N_a = C_{am} · p_m(ξ_g)
        self.shape_function_d0_gaussian = torch.einsum('ab, gb->ga', self.shape_function[0],
                                    pp).unsqueeze(1)

        # —— Step 5: integration weight in physical space ——
        # w_g = w_g^ref · det(J),  dΩ = det(J) dξ
        self.gaussian_weight = torch.einsum('ge, g->ge', det_Jacobian, self.gaussian_weight_ref)

        # —— Step 6: second derivative of shape function at Gaussian points ——
        # shape2_gaussian[g, i, j, a] = ∂²N_a/∂ξ_i∂ξ_j|_{ξ_g}
        #   = C_{am} · ∂²p_m/∂ξ_i∂ξ_j|_{ξ_g}  =  p_m(ξ_g) · shape2_now[i, j, a, m]
        shape2_gaussian = torch.einsum('gb,mnab->gmna', pp, self.shape_function[2])

        # —— Step 7: second derivative of isoparametric mapping ——
        # Jacobian2: H_{ijk} = ∂²x_i/∂ξ_j∂ξ_k = Σ_a (∂²N_a/∂ξ_j∂ξ_k) · x_i^a
        #   second derivative of the isoparametric mapping
        Jacobian2 = torch.zeros([self._num_gaussian, len(self._elems), 3, 3, 3])
        for i in range(self.num_nodes_per_elem):
            # Jacobian2[g, e, i, j, k] += ∂²N_a/∂ξ_j∂ξ_k|_{ξ_g} · x_i^a
            Jacobian2 += torch.einsum('gmn,ei->geimn', shape2_gaussian[:, :, :, i],
                                        nodes[elem_now[:, i]])

        # inv_Jacobian2: ∂(J^{-1})_{ij}/∂x_k
        #   differentiate J·J^{-1}=I  →  ∂J^{-1}/∂x = -J^{-1}·(∂J/∂x)·J^{-1}
        #   in tensor index form:
        #   ∂(J^{-1})_{ml}/∂x_k = -(J^{-1})_{mj}·(J^{-1})_{pk}·(J^{-1})_{nl}·(∂²x_p/∂ξ_j∂ξ_n)·(J^{-1})_{?}
        #   simplified via chain rule → -J^{-1}·J^{-1}·J^{-1}·H
        inv_Jacobian2 = -torch.einsum(
            'gemj,gepk,genl,gejnp->gemlk', inv_Jacobian,
            inv_Jacobian, inv_Jacobian, Jacobian2)

        # —— Step 8: second derivative in physical space ∂²N_a/∂x_i∂x_j ——
        # Chain rule for second derivatives:
        #   ∂²N_a/∂x_i∂x_j = (J^{-1})_{im}·(J^{-1})_{jn}·(∂²N_a/∂ξ_m∂ξ_n)
        #                   + ∂(J^{-1})_{im}/∂x_j · (∂N_a/∂ξ_m)
        #
        # Term 1 (from isoparametric mapping of Hessian):
        #   [g, e, i, j, a] += (J^{-1})_{im}·(J^{-1})_{jn} · ∂²N_a/∂ξ_m∂ξ_n
        # Term 2 (correction from the derivative of J^{-1}):
        #   [g, e, i, j, a] += ∂(J^{-1})_{im}/∂x_j · ∂N_a/∂ξ_m
        self.shape_function_d2_gaussian = torch.einsum(
                'gemi, genj,gmna->geija',
                inv_Jacobian, inv_Jacobian, shape2_gaussian) + torch.einsum(
                    'gemij, gma->geija', inv_Jacobian2, shape1_gaussian)


        self._dNW = torch.einsum('geia,ge->geia',
                                self.shape_function_d1_gaussian,
                                self.gaussian_weight)


    @staticmethod
    def _shape_function_derivative(shape_function: torch.Tensor, ind: int):
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
        """
        Get the physical coordinates of the Gaussian points for the element.

        Args:
            nodes: [p, 3], global nodal coordinates x_i^a in physical space

        Returns:
            torch.Tensor: [num_gaussian, num_elem, 3], physical coordinates of Gaussian points for each element
        """
        pp = self._get_interpolation_coordinates(self.gaussian_coordinates)
        shapeFun0 = torch.einsum('ab, gb->ga', self.shape_function[0],
                                      pp)
        gaussian_position = torch.zeros(
            [self._num_gaussian, self._elems.shape[0], 3])
        for i in range(self._elems.shape[1]):
            gaussian_position = gaussian_position + torch.einsum(
                'g,eI->geI', shapeFun0[:, i], nodes[self._elems[:,
                                                                         i]])
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
    
    
    def get_deformation_gradient(self, U: torch.Tensor):
        """
        Calculate the deformation gradient F at Gaussian points for the element.

        Args:
            U: [num_nodes, 3], the nodal displacements

        Returns:
            torch.Tensor: [num_gaussian, num_elem, 3, 3], deformation gradient F at each Gaussian point
        """

        Ugrad = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 3])
        for i in range(self.num_nodes_per_elem):
            Ugrad = Ugrad + torch.einsum('gki,kI->gkIi',
                                         self.shape_function_d1_gaussian[:, :, :, i],
                                         U[self._elems[:, i]])

        F = Ugrad.clone()
        F[:, :, 0, 0] += 1
        F[:, :, 1, 1] += 1
        F[:, :, 2, 2] += 1

        return F

    def get_potential_energy_density(self, U: torch.Tensor):
        """
        Calculate the potential energy density at Gaussian points for the element.

        Args:
            U: [num_nodes, 3], the nodal displacements

        Returns:
            torch.Tensor: [num_gaussian, num_elem], potential energy density at each Gaussian point
        """

        F = self.get_deformation_gradient(U=U)

        W = torch.zeros([self._num_gaussian, self._elems.shape[0]], device=F.device, dtype=F.dtype)
        for mat_now, element_indices in self._iter_material_contributions():
            if element_indices is None:
                W = W + mat_now.strain_energy_density_C3(F=F,)
                continue
            indices = torch.as_tensor(element_indices, dtype=torch.long, device=F.device)
            contribution = mat_now.strain_energy_density_C3(F=F[:, indices],)
            W = W + torch.zeros_like(W).index_copy(1, indices, contribution)

        return W

    def potential_Energy(self, RGC: torch.Tensor, rotation_matrix: Optional[torch.Tensor] = None):
        
        U = RGC

        if rotation_matrix is not None:
            U = torch.einsum('ij,aj->ai', rotation_matrix.T, U)

        W = self.get_potential_energy_density(U=U)

        Ea = torch.einsum(
            'ge,ge->',W,
            self.gaussian_weight)

        return Ea
    
    def _get_EpdUe_EpdUe2(self, U: torch.Tensor, if_onlyforce: bool = False):
        """
        Calculate the first and second derivatives of the potential energy with respect to the nodal displacements U.
        Which are the residual force and the stiffness matrix of the element, respectively.

        Args:
            U: [num_nodes, 3], the nodal displacements
            if_onlyforce: whether to only calculate the residual force, if True, only return the residual force, otherwise return both the residual force and the stiffness matrix
        
        Returns:
            If if_onlyforce is True, returns the residual force as a flattened tensor.
            Otherwise, returns a tuple of the residual force and the stiffness matrix.

        Relement = ∂Ea/∂U with shape [num_nodes_per_elem, 3, num_elems]
        Ka_element = ∂²Ea/∂U² with shape [num_nodes_per_elem, 3, num_nodes_per_elem, 3, num_elems]
        """
        
        s, C = self.components_Solid(U=U)

        # calculate the element residual force
        Relement = torch.einsum('geij,geia->aje', s,
                                self._dNW)
        
        if if_onlyforce:
            return Relement
        
        # calculate the element tangential stiffness matrix
        Ka_element = torch.einsum('geijkl,gelb,geia->ajbke',
                                   C,
                                  self.shape_function_d1_gaussian,
                                  self._dNW)
        
        return Relement, Ka_element
        
    def structural_Force(self, RGC: torch.Tensor, rotation_matrix: Optional[torch.Tensor] = None, if_onlyforce: bool = False):
        
        U = RGC

        if rotation_matrix is not None:
            U = torch.einsum('ij,aj->ai', rotation_matrix.T, U)

        result = self._get_EpdUe_EpdUe2(U=U, if_onlyforce=if_onlyforce)
        
        
        if if_onlyforce:
            Relement = result
            if rotation_matrix is not None:
                Relement = torch.einsum('mj,aje->ame', rotation_matrix, Relement)
            return self._indices_force, Relement.flatten()

        
        if rotation_matrix is not None:
            Relement = torch.einsum('mj,aje->ame', rotation_matrix, result[0])
            Ka_element = torch.einsum('mj,ajbke,nk->ambne', rotation_matrix, result[1], rotation_matrix)
        else:
            Relement = result[0]
            Ka_element = result[1]
        
        # assembly the stiffness matrix and residual force                 
        values = torch.zeros([self._indices_matrix.shape[1]]).scatter_add(0, self._index_matrix_coalesce, Ka_element.flatten())
        
        return self._indices_force, Relement.flatten(), self._indices_matrix, values

    def components_Solid(self, U: torch.Tensor):
        
        F = self.get_deformation_gradient(U=U)

        invF = F.inverse()
        J = F.det()
        Jneg = J**(-2 / 3)
        I1 = (F**2).sum([-1, -2]) * Jneg
        
        s = torch.zeros_like(F)
        C = torch.zeros([s.shape[0], s.shape[1], 3, 3, 3, 3], device=F.device, dtype=F.dtype)

        for mat_now, element_indices in self._iter_material_contributions():
            if element_indices is None:
                s_now, C_now = mat_now.material_Constitutive_C3(
                    F=F,
                    J=J,
                    Jneg=Jneg,
                    invF=invF,
                    I1=I1,
                )
                s = s + s_now
                C = C + C_now
                continue
            indices = torch.as_tensor(element_indices, dtype=torch.long, device=F.device)
            s_now, C_now = mat_now.material_Constitutive_C3(
                F=F[:, indices],
                J=J[:, indices],
                Jneg=Jneg[:, indices],
                invF=invF[:, indices],
                I1=I1[:, indices],
            )
            # Scatter each material contribution into the global element axis.
            # ``index_add`` is intentional here: it preserves contributions
            # if material regions overlap or contain repeated element ids.
            s = s.index_add(1, indices, s_now)
            C = C.index_add(1, indices, C_now)

        return s, C

    def get_volumn(self, U: torch.Tensor = None):
        if U is None:
            return self.gaussian_weight.sum()
        else:
            F = self.get_deformation_gradient(U=U)
            J = F.det()
            return (self.gaussian_weight * J).sum()

    def set_required_DoFs(
            self, RGC_remain_index: np.ndarray) -> np.ndarray:
        """
        Modify the RGC_remain_index
        """
        RGC_remain_index[self._elems.unique().cpu().numpy()] = True

        return RGC_remain_index

    def extract_boundary_surface_set(
        self
    ) -> list[tuple[torch.Tensor, int]]:
        """Extract boundary face groups from tet connectivity.

        Returns a list of (element_tags, face_index) tuples for each
        tetrahedron face that lies on the exterior boundary.
        """
        face_defs = self.surfaceid_map
        num_surfaces = self.num_surfaces
        surface_counter: dict[tuple[int], int] = {}

        for sfidx in range(num_surfaces):
            face_nodes = self._elems[:, face_defs[sfidx]]
            for face in face_nodes:
                sorted_face = tuple(sorted(face.cpu().numpy()))
                surface_counter[sorted_face] = surface_counter.get(sorted_face, 0) + 1
        
        surface_sets: list[tuple[torch.Tensor, int]] = []
        for sfidx in range(num_surfaces):
            face_nodes = self._elems[:, face_defs[sfidx]]

            surf_now = []
            for elemidx, face in enumerate(face_nodes):
                sorted_face = tuple(sorted(face.cpu().numpy()))
                if surface_counter[sorted_face] == 1:
                    surf_now.append(elemidx)
            
            if len(surf_now) > 0:
                surface_sets.append((np.array(surf_now, dtype=np.int64), sfidx))

        return surface_sets
    

    def extract_surface(self, surface_ind: int, elems_ind: torch.Tensor):
        index_now = np.where(np.isin(self._elems_index.cpu().numpy(), elems_ind))[0]

        if index_now.shape[0] == 0:
            return []

        if surface_ind in self.surfaceid_map:
            face_elems = self._elems[index_now][:, self.surfaceid_map[surface_ind]]
            return [initialize_surfaces(face_elems)]
        else:
            raise ValueError(f"Surface index {surface_ind} not found in surfaceid_map.")
