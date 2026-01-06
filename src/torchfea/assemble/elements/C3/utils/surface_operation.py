from __future__ import annotations
from calendar import c
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ....part import Part

import torch
import numpy as np
import torchfea


def divide_surface_elements(part: Part, name_element: str, name_surface: str):

    elems_now: torchfea.elements.Element_3D = part.elems[name_element]
    surface = part.surfaces.get_elements(name_surface)

    # find the elements that contain the surface node
    ind_surface_element = []
    for i in range(len(surface)):
        elem_ind_now = surface[i]._elems.flatten()
        ind_surface_element += elem_ind_now.tolist()
    ind_surface_element = np.array(ind_surface_element, dtype=np.int64)
    ind_surface_element = np.unique(ind_surface_element)
    ind_other_element = np.setdiff1d(elems_now._elems_index.numpy(), ind_surface_element)

    ind_other_element = torch.tensor(ind_other_element, dtype=torch.int64, device=elems_now._elems.device)
    ind_surface_element = torch.tensor(ind_surface_element, dtype=torch.int64, device=elems_now._elems.device)

    elem_index_surface = torch.where(torch.isin(elems_now._elems_index, ind_surface_element))[0]
    elem_index_other = torch.where(torch.isin(elems_now._elems_index, ind_other_element))[0]
    

    elems_surface = torchfea.elements.C3.C3D4(elems=elems_now._elems[elem_index_surface],
                                        elems_index=elems_now._elems_index[elem_index_surface])
    elems_other = torchfea.elements.C3.C3D4(elems=elems_now._elems[elem_index_other],
                                        elems_index=elems_now._elems_index[elem_index_other])

    return elems_surface, elems_other

def set_surface_2order(fe: torchfea.FEAController, name_elems: str, name_surface: str):
    
    element0: torchfea.elements.Element_3D = fe.elems[name_elems]

    element = element0.__class__(elems_index=element0._elems_index, elems=element0._elems)
    surface = fe.surface_sets[name_surface]
    element.surf_order = element0.surf_order.clone()
    if element.surf_order.sum() == 0:
        element.surf_order = torch.ones([element._elems.shape[0], 4], dtype=torch.int8, device='cpu')
    for surf_ind in range(len(surface)):
        elem_ind_now = torch.where(torch.isin(element._elems_index, torch.from_numpy(surface[surf_ind][0])))[0]
        element.surf_order[elem_ind_now, surface[surf_ind][1]] = 2

    try:
        element.set_materials(element0.materials)
    except:
        pass
    
    return element