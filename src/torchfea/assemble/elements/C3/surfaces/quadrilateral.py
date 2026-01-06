import torch
from .basesurface import BaseSurface

class Q4(BaseSurface):
    """
    Class for 4-node quadrilateral surface elements in the C3D family.
    This class inherits from BaseSurface and implements specific properties and methods for Q4 surface.
    """

    def __init__(self, elems: torch.Tensor) -> None:
        super().__init__(elems)

    def initialize(self, fea) -> None:
        """
        Initialize the Q4 surface element with the FEA_Main object.
        This method sets up the shape functions and Gaussian points for the Q4 element.
        """
        super().initialize(fea)

        # Define shape functions for Q4 surface (4-node quadrilateral element)
        # N1 = (1-ξ)(1-η)/4, N2 = (1+ξ)(1-η)/4, N3 = (1+ξ)(1+η)/4, N4 = (1-ξ)(1+η)/4
        self.shape_function = [
            torch.tensor([[0.25, -0.25, -0.25, 0.25],
                          [0.25, 0.25, -0.25, -0.25],
                          [0.25, 0.25, 0.25, 0.25],
                          [0.25, -0.25, 0.25, -0.25]])
        ]

        shape_i = torch.zeros([2, 4, 4])
        for i in range(2):
            shape_i[i] = self._shape_function_derivative(
                self.shape_function[0], i)
        self.shape_function.append(shape_i)
        
        # Define Gaussian points for Q4 surface (2x2 integration)
        g = 1.0 / torch.sqrt(torch.tensor(3.0))
        pp = torch.tensor([[-g, -g],
                          [g, -g],
                          [g, g],
                          [-g, g]])
        self.gaussian_weight = torch.tensor([1.0, 1.0, 1.0, 1.0])

        # Define Gaussian points for Q8 surface (3x3 integration)
        # g = torch.sqrt(torch.tensor(3.0/5.0))
        # pp = torch.tensor([[-g, -g], [0.0, -g], [g, -g],
        #                   [-g, 0.0], [0.0, 0.0], [g, 0.0],
        #                   [-g, g], [0.0, g], [g, g]])
        # w1, w2 = 5.0/9.0, 8.0/9.0
        # self.gaussian_weight = torch.tensor([w1*w1, w2*w1, w1*w1,
        #                                    w1*w2, w2*w2, w1*w2,
        #                                    w1*w1, w2*w1, w1*w1])

        self.num_nodes_per_elem = 4
        self._num_gaussian = 4
        
        # Pre-load Gaussian points for Q4 surface
        self._pre_load_gaussian(pp, fea.nodes)

class Q8(BaseSurface):
    """
    Class for 8-node quadrilateral surface elements in the C3D family.
    This class inherits from BaseSurface and implements specific properties and methods for Q8 surface.
    """

    def __init__(self, elems: torch.Tensor) -> None:
        super().__init__(elems)

    def initialize(self, fea) -> None:
        """
        Initialize the Q8 surface element with the FEA_Main object.
        This method sets up the shape functions and Gaussian points for the Q8 element.
        """
        super().initialize(fea)

        # Define shape functions for Q8 surface (8-node quadrilateral element)
        # 手工展开给定的Q8形函数表达式为基函数[1, g, h, g*h, g², h², g²*h, g*h²]的系数
        self.shape_function = [
            torch.tensor([
                [-0.25, 0.25, 0.25, -0.25, 0.25, 0.25, -0.25, -0.25],  # N1 = -1/4(1-g)(1-h)(1+g+h)
                [-0.25, -0.25, 0.25, 0.25, 0.25, 0.25, 0.25, -0.25],   # N2 = -1/4(1+g)(1-h)(1-g+h)  
                [-0.25, -0.25, -0.25, -0.25, 0.25, 0.25, 0.25, 0.25],  # N3 = -1/4(1+g)(1+h)(1-g-h)
                [-0.25, 0.25, -0.25, 0.25, 0.25, 0.25, -0.25, 0.25],   # N4 = -1/4(1-g)(1+h)(1+g-h)
                [0.5, 0.0, -0.5, 0.0, -0.5, 0.5, 0.0, 0.0],            # N5 = 1/2(1-g²)(1-h)
                [0.5, 0.5, 0.0, 0.0, -0.5, -0.5, 0.0, 0.5],            # N6 = 1/2(1-h²)(1+g)
                [0.5, 0.0, 0.5, 0.0, -0.5, -0.5, 0.0, 0.0],            # N7 = 1/2(1-g²)(1+h)
                [0.5, -0.5, 0.0, 0.0, -0.5, 0.5, 0.0, -0.5]            # N8 = 1/2(1-h²)(1-g)
            ])
        ]

        shape_i = torch.zeros([2, 8, 8])
        for i in range(2):
            shape_i[i] = self._shape_function_derivative(
                self.shape_function[0], i)
        self.shape_function.append(shape_i)
        
        # Define Gaussian points for Q8 surface (3x3 integration)
        g = torch.sqrt(torch.tensor(3.0/5.0))
        pp = torch.tensor([[-g, -g], [0.0, -g], [g, -g],
                          [-g, 0.0], [0.0, 0.0], [g, 0.0],
                          [-g, g], [0.0, g], [g, g]])
        w1, w2 = 5.0/9.0, 8.0/9.0
        self.gaussian_weight = torch.tensor([w1*w1, w2*w1, w1*w1,
                                           w1*w2, w2*w2, w1*w2,
                                           w1*w1, w2*w1, w1*w1])

        self.num_nodes_per_elem = 8
        self._num_gaussian = 9
        
        # Pre-load Gaussian points for Q8 surface
        self._pre_load_gaussian(pp, fea.nodes)

    @property
    def surf_elems_circ(self) -> torch.Tensor:
        """
        Get the circular elements of the surface.
        This property returns the element connectivity for Q8 surface elements.
        """
        return self._elems[:, [0, 4, 1, 5, 2, 6, 3, 7]]