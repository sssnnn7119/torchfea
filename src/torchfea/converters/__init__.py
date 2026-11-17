"""Conversions between native finite-element models and external data models.

Format modules own semantic mappings. File parsing and writing remain with
the corresponding format model (for example, torchfea.inp.FEA_INP).
"""

from .inp import controller_to_inp, inp_to_controller

__all__ = ["controller_to_inp", "inp_to_controller"]
