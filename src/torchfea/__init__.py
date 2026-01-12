import numpy as np
import torch
from .inp import FEA_INP
from .controller import FEAController
from .model import Part, Instance, ReferencePoint, Assembly
from .model import materials, elements, loads, constraints, surfaces, boundarys
from . import solver

def from_inp(inp: FEA_INP, create_instance=True) -> FEAController:
    """
    Load a FEA model from an INP file.

    Args:
        inp (FEA_INP): An instance of the FEA_INP class.

    Returns:
        FEA_Main: An instance of the FEA_Main class with imported elements and sets.
    """

    assembly_now = Assembly()

    part_name = list(inp.part.keys())[0]

    temp = torch.tensor([1.])
    default_device = temp.device
    default_dtype = temp.dtype

    for i in range(len(inp.part)):
        part_name = list(inp.part.keys())[i]
        part_nodes = torch.from_numpy(inp.part[part_name].nodes).to(device=default_device, dtype=default_dtype)

        part_now = Part(part_nodes[:, 1:])

        # define the set of nodes
        for set_name, node_indices in inp.part[part_name].sets_nodes.items():
            part_now.set_nodes[set_name] = np.unique(np.array(list(node_indices)))

        assembly_now.add_part(part=part_now, name=part_name)
        if create_instance:
            assembly_now.add_instance(instance=Instance(part_now), name=part_name)

        elems = inp.part[part_name].elems
        elems_num_now = 0
        
        
        for key in list(elems.keys()):

            materials_type = np.unique(inp.part[part_name].elems_material[elems[key][:, 0], 2].astype(int))

            elems_num_now += elems[key].shape[0]
            for mat_type in materials_type:
                index_now = np.where(inp.part[part_name].elems_material[elems[key][:, 0], 2].astype(int) == mat_type)

                
                materials_now = materials.initialize_materials(
                    materials_type=mat_type,
                    materials_params=torch.from_numpy(inp.part[part_name].elems_material[elems[key][:, 0]][index_now][:, 3:]).to(device=default_device, dtype=default_dtype)
                )

                element_name = key

                elems_now = elements.initialize_element(
                            element_type=element_name,
                            elems_index=torch.from_numpy(elems[key][:, 0]),
                            elems=torch.from_numpy(elems[key][:, 1:]),
                            part=part_now
                            )
                
                elems_now.set_materials(materials_now)
                elems_now.density = (inp.part[part_name].elems_material[elems[key][:, 0]][index_now][:, 1])
                
                part_now.add_element(elems_now)
 
        # Import surface sets from each part
        for surface_name, surface in inp.part[part_name].surfaces.items():
            full_name = f"{surface_name}"
            sf_now = []
            for sf in surface:
                sf_now.append((sf[0], sf[1]))
            part_now.add_surface_set(full_name, sf_now)


    fe = FEAController()
    fe.assembly = assembly_now
    return fe