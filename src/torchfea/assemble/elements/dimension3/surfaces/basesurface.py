
from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .... import Part
import torch

class BaseSurface():
    """
    Base class for surface elements in the C3D family.
    This class is intended to be inherited by specific surface element classes.
    """
    _subclasses: dict[str, 'BaseSurface'] = {}

    def __init_subclass__(cls):
        """Register subclasses in the class registry for factory method."""
        cls._subclasses[cls.__name__] = cls

    def __init__(self, elems: torch.Tensor) -> None:
        self._elems = elems
        """
        The element connectivity.
        Shape: [num_elements, num_nodes_per_elem]
        """
        
        self.num_nodes_per_elem: int
        """
        The number of nodes per element for surface elements.
        """

        self._num_gaussian: int
        """
        The number of Gaussian points used for integration.
        """

        self._part: Part
        """
        The Part object that this surface belongs to.
        """
        
        self.shape_function_gaussian: list[torch.Tensor] = []
        """
            the shape functions of each guassian point 
            [
                [
                    g: guassian point
                    a: a-th node
                ],
                [
                    g: guassian point
                    m: derivative (reference to the local coordinates)
                    a: a-th node
                ]
            ]
        """

        self.shape_function: list[torch.Tensor]
        """
            the shape functions of the element

            # coordinates: (g,h) in the local coordinates
                0: constant,\n
                1: g,\n
                2: h,\n
                3: g*h,\n
                4: g^2,\n
                5: h^2,\n
                6: g^2*h,\n
                7: g*h^2,\n
                
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

        self.gaussian_weight: torch.Tensor
        """
        The weights of each Gaussian point.
        Shape: [num_gaussian_points]
        """

        self.det_Jacobian: torch.Tensor
        """
        The determinant of the Jacobian matrix at each Gaussian point.
        Used for numerical integration at the undeformed surface.
        Shape: [num_gaussian_points, num_elements]
        """

    def initialize(self, part: Part) -> None:
        """
        Initialize the surface element with the Part object.
        Args:
            fea (FEA_Main): The FEA_Main object that this surface belongs to.
        """
        self._part = part

    def _pre_load_gaussian(self, gauss_coordinates: torch.Tensor, nodes: torch.Tensor):
        """
        load the guassian points and its weight

        Args:
            gauss_coordinates: [g, 2], the local coordinates of the element
            nodes: [p, 2], the global coordinates of the element
        """

        pp = torch.zeros([self._num_gaussian, self.shape_function[0].shape[1]])
        pp[:, 0] = 1
        pp[:, 1] = gauss_coordinates[:, 0]
        pp[:, 2] = gauss_coordinates[:, 1]
        if self.shape_function[0].shape[1] > 3:
            pp[:, 3] = gauss_coordinates[:, 0] * gauss_coordinates[:, 1]
        if self.shape_function[0].shape[1] > 4:
            pp[:, 4] = gauss_coordinates[:, 0] ** 2
            pp[:, 5] = gauss_coordinates[:, 1] ** 2
        if self.shape_function[0].shape[1] > 6:
            pp[:, 6] = gauss_coordinates[:, 0] ** 2 * gauss_coordinates[:, 1]
            pp[:, 7] = gauss_coordinates[:, 0] * gauss_coordinates[:, 1] ** 2
        

        shapeFun0 = torch.einsum('ab, gb->ga', self.shape_function[0],
                                      pp)
        shapeFun1 = torch.einsum('mab, gb->gma', self.shape_function[1],
                                        pp)
        
        self.shape_function_gaussian = [shapeFun0, shapeFun1]


        Jacobian = torch.zeros([self._num_gaussian, len(self._elems), 3, 2])
        shape_now = self.shape_function[1]
        for i in range(self.num_nodes_per_elem):
            Jacobian += torch.einsum('gb,mb,ei->geim', pp, shape_now[:, i],
                                     nodes[self._elems[:, i]])
            
        self.det_Jacobian = torch.einsum('geim, gein->gemn', Jacobian, Jacobian).det().sqrt()
            

    def _shape_function_derivative(self, shape_function: torch.Tensor, ind: int):
        """
        get the derivative of the shape function

        Args:
            shape_function: [i, m], the shape function of the element
            ind: the index of the derivative

        Returns:
            torch.Tensor: the derivative of the shape function
        """

        """
        0: constant,\n
        1: g,\n
        2: h,\n
        3: g*h,\n
        4: g^2,\n
        5: h^2,\n
        6: g^2*h,\n
        7: g*h^2,\n
        """

        result = torch.zeros_like(shape_function)
        if ind == 0:
            result[:, 0] = shape_function[:, 1]
            if shape_function.shape[1] > 3:
                result[:, 2] = shape_function[:, 3]
            if shape_function.shape[1] > 4:
                result[:, 1] = 2 * shape_function[:, 4]
            if shape_function.shape[1] > 6:
                result[:, 3] = 2 * shape_function[:, 6]
                result[:, 5] = shape_function[:, 7]

        elif ind == 1:
            result[:, 0] = shape_function[:, 2]
            if shape_function.shape[1] > 3:
                result[:, 1] = shape_function[:, 3]
            if shape_function.shape[1] > 4:
                result[:, 2] = 2 * shape_function[:, 5]
            if shape_function.shape[1] > 6:
                result[:, 4] = shape_function[:, 6]
                result[:, 3] = 2 * shape_function[:, 7]

        return result


    def gaussian_points_position(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        Get the positions of the Gaussian points on the surface.
        Returns:
            torch.Tensor: The positions of the Gaussian points.
        """
        gaussian_points = torch.zeros([self._num_gaussian, self._elems.shape[0], 3],
                dtype=self._part.nodes.dtype, device=self._part.nodes.device)
            
        for a in range(self.num_nodes_per_elem):
            gaussian_points += torch.einsum('g, ei->gei', self.shape_function_gaussian[0][:, a],
                                        nodes[self._elems[:, a]])
            
        return gaussian_points

    def get_gaussian_normal(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        Get the normals of the Gaussian points on the surface.
        Returns:
            torch.Tensor: The normals of the Gaussian points.
        """

        rdu = torch.zeros([self._num_gaussian, self._elems.shape[0], 3, 2], dtype=self._part.nodes.dtype, device=self._part.nodes.device)

        for a in range(self.num_nodes_per_elem):
            rdu += torch.einsum('gm, ei->geim', self.shape_function_gaussian[1][:, :, a],
                                 nodes[self._elems[:, a]])
        gaussian_normals = torch.cross(rdu[:, :, :, 0], rdu[:, :, :, 1], dim=-1)
        return gaussian_normals

    @property
    def surf_elems_circ(self) -> torch.Tensor:
        """
        Get the circular elements of the surface.
        This property should be overridden in subclasses if needed.
        """
        return self._elems
    
