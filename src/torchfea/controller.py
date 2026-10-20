import numpy as np
import torch
from .model import Assembly
from .cad import CADModel
from .solver import BaseSolver
from .interfaces import Serializable
from .model_hash import controller_model_hash, controller_result_compatibility_hash


class FEAController(Serializable):
    _serialized_attributes_exclude = ["_model_hash"]

    def __init__(self, maximum_iteration: int = 10000) -> None:
        """
        Initialize the FEA class.

        Args:
            nodes (torch.Tensor): The nodes of the finite element model.
        """

        self.assembly: Assembly = None
        """The assembly containing instances, elements, and reference points."""

        self.cad: CADModel = CADModel()
        """Discrete CAD construction histories used to build the assembly."""

        self.solver: BaseSolver = None
        """The solver used for finite element analysis."""

        self._model_hash: str | None = None
        """Stable identity of the persistent model definition."""

    def initialize(self, *args, **kwargs):
        """
        Initialize the finite element model.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.

        Returns:
            None
        """
        self.assembly.initialize(*args, **kwargs)
        self.solver.initialize(assembly=self.assembly, *args, **kwargs)

    def solve(
        self, GC0: torch.Tensor = None, if_initialize: bool = True, *args, **kwargs
    ):
        """
        Solves the finite element analysis problem.

        Args:
            GC0 (torch.Tensor, optional): Initial generalized coordinates. Defaults to an empty tensor.
            tol_error (float, optional): Tolerance error for convergence. Defaults to 1e-7.

        Returns:
            bool: True if the solution converged, False otherwise.
        """
        # Capture the identity before initialization creates solver caches.
        # The result must remain paired with the persistent model definition,
        # not with those transient arrays.
        model_hash = self.get_model_hash()
        result_compatibility_hash = self.get_result_compatibility_hash()
        if if_initialize:
            self.initialize()
        else:
            self.assembly.define_required_DoFs()

        result = self.solver.solve(GC0=GC0, *args, **kwargs)
        result.model_hash = model_hash
        result.compatibility_hash = result_compatibility_hash
        return result

    def get_model_hash(self) -> str:
        """Compute the current identity of this model definition.

        Hashing is intentionally explicit and uncached.  Callers should use
        it at model solve, persistence, or result-validation boundaries rather
        than from frequently refreshed UI state.
        """
        self._model_hash = controller_model_hash(self)
        return self._model_hash

    def get_result_compatibility_hash(self) -> str:
        """Compute an identity for viewing results across load value changes."""
        return controller_result_compatibility_hash(self)

    @property
    def model_hash(self) -> str:
        """Public model identity used to pair native results with a model."""
        return self.get_model_hash()

    def change_device(self, device: torch.device) -> None:
        """
        Recursively change the device of the finite element model and all nested objects.

        Args:
            device (torch.device): The target device.

        Returns:
            None
        """
        self._change_device_recursive(self, device)

    def _change_device_recursive(self, obj, device, visited=None):
        """
        Recursively move tensors to the target device.
        """
        if visited is None:
            visited = set()

        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)

        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, torch.Tensor):
                    obj[k] = v.to(device)
                else:
                    self._change_device_recursive(v, device, visited)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, torch.Tensor):
                    obj[i] = v.to(device)
                else:
                    self._change_device_recursive(v, device, visited)
        elif isinstance(obj, tuple):
            for v in obj:
                self._change_device_recursive(v, device, visited)
        elif hasattr(obj, "__dict__"):
            for k, v in list(obj.__dict__.items()):
                if isinstance(v, torch.Tensor):
                    setattr(obj, k, v.to(device))
                else:
                    self._change_device_recursive(v, device, visited)

    def save_model(self, path: str, if_save_source_code: bool = True) -> None:
        """
        Save the finite element model to a file.

        Args:
            path (str): The file path to save the model.
            if_save_source_code (bool): Whether to save the source code of subclasses.

        Returns:
            None
        """
        model_hash = self.get_model_hash()
        save_data = {
            "data": self._serialize(),
            "model_hash": model_hash,
        }
        if if_save_source_code:
            save_data["source_code"] = self._subclass_source_code
        np.savez_compressed(path, **save_data)

    def GC2RGC(self, GC: torch.Tensor) -> list[torch.Tensor]:
        """
        Convert global coordinates (GC) to reduced global coordinates (RGC).

        Args:
            GC (torch.Tensor): The global coordinates.

        Returns:
            list[torch.Tensor]: The reduced global coordinates.
        """
        return self.assembly._GC2RGC(GC)

    def RGC2GC(self, RGC: list[torch.Tensor]) -> torch.Tensor:
        """
        Convert reduced global coordinates (RGC) to global coordinates (GC).

        Args:
            RGC (list[torch.Tensor]): The reduced global coordinates.

        Returns:
            torch.Tensor: The global coordinates.
        """
        return self.assembly._RGC2GC(RGC)

    def set_work_conditions(self, work_conditions: dict[str, torch.Tensor]) -> None:
        """
        Set the work conditions for the finite element analysis.

        Args:
            work_conditions (WorkConditions): The work conditions to set.

        """
        self.assembly.set_work_conditions(work_conditions)


def load_model(path: str) -> FEAController:
    """
    Load a finite element model from a file.

    Args:
        path (str): The file path to load the model from.

    Returns:
        FEAController: The loaded finite element model.
    """
    with np.load(path, allow_pickle=True) as loaded:
        serialized_data = loaded["data"]
        controller = FEAController._deserialize(serialized_data)
        stored_hash = str(np.asarray(loaded["model_hash"]).reshape(-1)[0])

    # Deserialization restores only persistent fields.  Rebuild all links,
    # coordinate blocks, and enabled-object lists before the caller can use
    # the model.  Assembly.initialize() normalizes enabled=None to True.
    assembly = controller.assembly
    for part in assembly._parts.values():
        if part.exterior_surface == "extern" and "extern" not in part.surfaces:
            boundary = []
            for element in part.elems.values():
                boundary.extend(element.extract_boundary_surface_set())
            if boundary:
                part.surfaces[part.exterior_surface] = boundary

    if controller.solver is None:
        assembly.initialize()
    else:
        controller.initialize()

    # The rebuilt model is the source of truth for the current identity.
    # This also prevents runtime state restored from an archive from leaving
    # a stale cached hash behind.
    controller._model_hash = controller.get_model_hash()
    if not stored_hash:
        raise ValueError("The model archive contains an empty model hash.")
    return controller


def retrieve_source_code(path: str, path_out: str):
    """
    Retrieve the source code of subclasses from a saved model file for debugging purposes.

    Args:
        path (str): The file path to load the model from.
        path_out (str): The file path to save the source code to.
    """

    loaded = np.load(path, allow_pickle=True)

    if "source_code" not in loaded:
        raise ValueError("The .npz file does not contain 'source_code' key.")
    source_code = loaded[
        "source_code"
    ].item()  # Assuming source_code is stored as a dictionary in the .npz file

    result = ""
    for class_name, code in source_code.items():
        result += f"#========= Source code for {class_name} =========#\n"
        result += code + "\n\n"

    with open(path_out, "w") as f:
        f.write(result)
