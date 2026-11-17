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
    console.setFormatter(
        __logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(console)

    # logging to file if log_file is provided
    if log_file:
        file_handler = __logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(
            __logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
            )
        )
        logger.addHandler(file_handler)

    return logger


# endregion

# the fea must use double precision for convergence and accuracy
import torch as __torch

__torch.set_default_dtype(__torch.float64)

# ignore warnings from torch about device placement and other non-critical issues
import warnings as __warnings

__warnings.filterwarnings("ignore", ".*Sparse CSR tensor support is in beta state.*")


from . import cad, solver
from .cad import (
    Box,
    CADHistory,
    CADModel,
    CADOperation,
    CADSetDefinition,
    Cone,
    Cut,
    Cylinder,
    Dilate,
    Fragment,
    Fuse,
    Intersect,
    Mirror,
    Rotate,
    Sphere,
    StepImport,
    Torus,
    Translate,
    Wedge,
)
from .controller import FEAController, load_model, retrieve_source_code
from .converters import inp_to_controller as from_inp
from .inp import FEA_INP
from .interfaces import Serializable
from .model import (
    Assembly,
    Instance,
    Part,
    ReferencePoint,
    WorkCondition,
    boundarys,
    constraints,
    elements,
    loads,
    materials,
    surfaces,
)

__all__ = [
    "FEA_INP",
    "Assembly",
    "Box",
    "CADHistory",
    "CADModel",
    "CADOperation",
    "CADSetDefinition",
    "Cone",
    "Cut",
    "Cylinder",
    "Dilate",
    "FEAController",
    "Fragment",
    "Fuse",
    "Instance",
    "Intersect",
    "Mirror",
    "Part",
    "ReferencePoint",
    "Rotate",
    "Serializable",
    "Sphere",
    "StepImport",
    "Torus",
    "Translate",
    "Wedge",
    "WorkCondition",
    "boundarys",
    "cad",
    "constraints",
    "elements",
    "from_inp",
    "load_model",
    "loads",
    "materials",
    "retrieve_source_code",
    "solver",
    "surfaces",
]
