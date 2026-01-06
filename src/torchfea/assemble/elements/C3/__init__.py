
from .C3base import Element_3D
from .brick import C3D8, C3D8R, C3D20
from .wedge import C3D6, C3D15
from .tetrahedral import C3D4, C3D10

from .utils.generate_shell import generate_shell_from_surface, add_shell_elements_to_model
from .utils.convert_elements import convert_to_second_order
from .utils.surface_operation import divide_surface_elements, set_surface_2order