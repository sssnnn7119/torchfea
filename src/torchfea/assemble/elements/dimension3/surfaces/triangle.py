import torch
from .basesurface import BaseSurface
class T3(BaseSurface):
    """
    Class for triangular surface elements in the C3D family.
    This class inherits from BaseSurface and implements specific properties and methods for T3 surface.
    """

    def __init__(self, elems: torch.Tensor) -> None:
        super().__init__(elems)
        


    def initialize(self, part) -> None:
        """
        Initialize the T3 surface element with the FEA_Main object.
        This method sets up the shape functions and Gaussian points for the T3 element.
        """
        super().initialize(part)

        # Define shape functions for T3 surface
        self.shape_function = [
            torch.tensor([[1.0, -1.0, -1.0], 
                          [0.0, 1.0, 0.0], 
                          [0.0, 0.0, 1.0]])
        ]

        shape_i = torch.zeros([2, 3, 3])
        for i in range(2):
            shape_i[i] = self._shape_function_derivative(
                self.shape_function[0], i)
        self.shape_function.append(shape_i)
        
        # Define Gaussian points for T3 surface
        pp = torch.tensor([[1 / 3, 1 / 3]])
        self.gaussian_weight = torch.tensor([0.5])
        # pp = torch.tensor([[1/6, 1/6],
        #                   [2/3, 1/6],
        #                   [1/6, 2/3]])
        # self.gaussian_weight = torch.tensor([1/6, 1/6, 1/6])

        self.num_nodes_per_elem = 3
        self._num_gaussian = 1
        
        # Pre-load Gaussian points for T3 surface
        self._pre_load_gaussian(pp, self._part.nodes)

class T6(BaseSurface):
    """
    Class for 6-node triangular surface elements in the C3D family.
    This class inherits from BaseSurface and implements specific properties and methods for T6 surface.
    """

    def __init__(self, elems: torch.Tensor) -> None:
        super().__init__(elems)

    def initialize(self, part) -> None:
        """
        Initialize the T6 surface element with the FEA_Main object.
        This method sets up the shape functions and Gaussian points for the T6 element.
        """
        super().initialize(part)

        # Define shape functions for T6 surface (6-node triangular element)
        # N1 = (1-ξ-η)(1-2ξ-2η), N2 = ξ(2ξ-1), N3 = η(2η-1)
        # N4 = 4ξ(1-ξ-η), N5 = 4ξη, N6 = 4η(1-ξ-η)
        self.shape_function = [
            torch.tensor([[1.0, -3.0, -3.0, 4.0, 2.0, 2.0],
                          [0.0, -1.0, 0.0, 0.0, 2.0, 0.0],
                          [0.0, 0.0, -1.0, 0.0, 0.0, 2.0],
                          [0.0, 4.0, 0.0, -4.0, -4.0, 0.0],
                          [0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
                          [0.0, 0.0, 4.0, -4.0, 0.0, -4.0]])
        ]

        shape_i = torch.zeros([2, 6, 6])
        for i in range(2):
            shape_i[i] = self._shape_function_derivative(
                self.shape_function[0], i)
        self.shape_function.append(shape_i)
        
        # Define Gaussian points for T6 surface (3-point integration)
        pp = torch.tensor([[1/6, 1/6],
                          [2/3, 1/6],
                          [1/6, 2/3]])
        self.gaussian_weight = torch.tensor([1/6, 1/6, 1/6])

        self.num_nodes_per_elem = 6
        self._num_gaussian = 3
        
        # Pre-load Gaussian points for T6 surface
        self._pre_load_gaussian(pp, self._part.nodes)

    @property
    def surf_elems_circ(self) -> torch.Tensor:
        """
        Get the circular elements of the surface.
        This property returns the element connectivity for T6 surface elements.
        """
        return self._elems[:, [0, 3, 1, 4, 2, 5]]