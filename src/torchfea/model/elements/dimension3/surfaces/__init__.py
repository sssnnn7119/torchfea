import torch
from .triangle import T3, T6
from .quadrilateral import Q4, Q8
from .basesurface import BaseSurface

def initialize_surfaces(surface_elems: torch.Tensor) -> BaseSurface:
    """
    Initialize surface elements based on their type.
    
    Args:
        surface_elems (torch.Tensor): Tensor containing surface element data.
        
    Returns:
        BaseSurface: initialized surface elements.
    """
    if not isinstance(surface_elems, torch.Tensor):
        raise TypeError("surface_elems must be a torch.Tensor.")

    elem_type = surface_elems.shape[1]
    if elem_type == 3:  # T3
        surfaces = T3(surface_elems)
    elif elem_type == 6:  # T6
        surfaces = T6(surface_elems)
    elif elem_type == 4:  # Q4
        surfaces = Q4(surface_elems)
    elif elem_type == 8:  # Q8
        surfaces = Q8(surface_elems)
    else:
        raise ValueError(f"Unsupported surface element type: {elem_type}.")
    
    return surfaces

def merge_surfaces(surfaces: list[T3 | T6 | Q4 | Q8]) -> list[T3 | T6 | Q4 | Q8]:
    """
    Merge a list of surface elements into a single list.
    
    Args:
        surfaces (list[T3 | T6 | Q4 | Q8]): List of surface elements to merge.
        
    Returns:
        tuple: A tuple containing merged surface elements of types T3, T6, Q4, and Q8.
            - T3: Merged T3 surface elements.
            - T6: Merged T6 surface elements.
            - Q4: Merged Q4 surface elements.
            - Q8: Merged Q8 surface elements.
    """
    if len(surfaces) <= 1:
        return surfaces

    merged_elems_T3 = []
    merged_elems_T6 = []
    merged_elems_Q4 = []
    merged_elems_Q8 = []

    for i in range(len(surfaces)):
        if surfaces[i].__class__.__name__ == 'T3':
            merged_elems_T3.append(surfaces[i]._elems)
        elif surfaces[i].__class__.__name__ == 'T6':
            merged_elems_T6.append(surfaces[i]._elems)
        elif surfaces[i].__class__.__name__ == 'Q4':
            merged_elems_Q4.append(surfaces[i]._elems)
        elif surfaces[i].__class__.__name__ == 'Q8':
            merged_elems_Q8.append(surfaces[i]._elems)
        else:
            raise ValueError(f"Unsupported surface element type: {surfaces[i].__class__.__name__}.")

    # Concatenate the elements of each type
    merged_elems_T3 = torch.cat(merged_elems_T3, dim=0) if merged_elems_T3 else torch.empty((0, 3), dtype=torch.int64)
    merged_elems_T6 = torch.cat(merged_elems_T6, dim=0) if merged_elems_T6 else torch.empty((0, 6), dtype=torch.int64)
    merged_elems_Q4 = torch.cat(merged_elems_Q4, dim=0) if merged_elems_Q4 else torch.empty((0, 4), dtype=torch.int64)
    merged_elems_Q8 = torch.cat(merged_elems_Q8, dim=0) if merged_elems_Q8 else torch.empty((0, 8), dtype=torch.int64)

    # Discard empty surfaces
    surfaces = []
    if merged_elems_T3.numel() > 0:
        surfaces.append(T3(merged_elems_T3))
    if merged_elems_T6.numel() > 0:
        surfaces.append(T6(merged_elems_T6))
    if merged_elems_Q4.numel() > 0:
        surfaces.append(Q4(merged_elems_Q4))
    if merged_elems_Q8.numel() > 0:
        surfaces.append(Q8(merged_elems_Q8))
    
    return surfaces