"""
torchfea: A Python library for Finite Element Analysis (FEA) using PyTorch.
This library provides a framework for defining and solving finite element models using PyTorch tensors. It includes
modules for defining parts, instances, assemblies, materials, elements, loads, constraints, and surfaces. The library also supports importing models from INP files and provides a base solver class for implementing various finite element analysis solvers.
"""



# region: Logging Configuration
import logging as __logging
__logger = __logging.getLogger(__name__)
__logger.addHandler(__logging.NullHandler())

def enable_logging(level=__logging.INFO, log_file=None, file_log_level=__logging.INFO):
    """
    Enable logging for the FEA package.

    Parameters
    ----------
    level : int
        the logging level (e.g., logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL)
    log_file : str, optional
        the path to a log file where logs will be written. If None, logs will only be printed to the console.
    file_log_level : int
        the logging level for the log file. Default is logging.INFO.
    
    Examples
    --------
    >>> import torchfea
    >>> torchfea.enable_logging(level=logging.DEBUG, log_file='fem.log')
    """
    logger = __logging.getLogger(__name__)
    logger.setLevel(min(level, file_log_level))
    
    # clear existing handlers to avoid duplicate logs
    logger.handlers.clear()
    
    # logging to console
    console = __logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(__logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(console)
    
    # logging to file if log_file is provided
    if log_file:
        file_handler = __logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(__logging.Formatter(
            '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'
        ))
        logger.addHandler(file_handler)

    return logger
# endregion

# the fea must use double precision for convergence and accuracy
import torch as __torch
__torch.set_default_dtype(__torch.float64)

# ignore warnings from torch about device placement and other non-critical issues
import warnings as __warnings
__warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')



from .interfaces import Serializable
from .inp import FEA_INP
from .controller import FEAController, load_model, retrieve_source_code
from .model import Part, Instance, ReferencePoint, Assembly, WorkCondition
from .model import materials, elements, loads, constraints, surfaces, boundarys
from . import cad
from .cad import CADModel, CADHistory, CADOperation
from . import solver

def from_inp(inp: FEA_INP, create_instance=True) -> FEAController:
    """
    Load a FEA model from an INP file.

    Args:
        inp (FEA_INP): An instance of the FEA_INP class.

    Returns:
        FEA_Main: An instance of the FEA_Main class with imported elements and sets.
    """

    import numpy as np
    import torch

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
            assembly_now.add_instance(instance=Instance(part_name=part_name), name=part_name)

        elems = inp.part[part_name].elems
        elems_num_now = 0
        
        
        for key in list(elems.keys()):

            materials_type = np.unique(inp.part[part_name].elems_material[elems[key][:, 0], 2].astype(int))

            elems_num_now += elems[key].shape[0]

            element_name = key
            elems_now = elements.initialize_element(
                        element_type=element_name,
                        elems_index=torch.from_numpy(elems[key][:, 0]).to(torch.get_default_device()),
                        elems=torch.from_numpy(elems[key][:, 1:]).to(torch.get_default_device()),
                        part=part_now
                        )

            for mat_type in materials_type:
                index_now = np.where(inp.part[part_name].elems_material[elems[key][:, 0], 2].astype(int) == mat_type)

                
                materials_now = materials.initialize_materials(
                    materials_type=mat_type,
                    materials_params=torch.from_numpy(inp.part[part_name].elems_material[elems[key][:, 0]][index_now][:, 3:]).to(device=default_device, dtype=default_dtype)
                )

                elems_now.set_materials(materials_now, name=f"material-type-{int(mat_type)}")

            # Density is stored per element row; keep full vector for this element block.
            elems_now.density = inp.part[part_name].elems_material[elems[key][:, 0], 1]
            part_now.add_element(elems_now, name=element_name)
 
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
