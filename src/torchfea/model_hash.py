"""Compact model check codes for native models and calculation results."""

from __future__ import annotations

import json
import zlib
from collections.abc import Mapping

import numpy as np
import torch

from .interfaces import Serializable


_SCALAR_TAGS = {"bool", "int", "float", "str", "NoneType"}
_UNORDERED_MAPPING_KEYS = {
    "set_nodes",
    "set_elements",
    "material_element_indices",
    "_gmsh_vertex_nodes",
    "_gmsh_surface_nodes",
    "_gmsh_volume_elements",
}
_SURFACE_MAPPING_KEYS = {"_surface_dict", "_gmsh_surface_sets"}


def _tagged_payload(value, tag: str):
    """Return the payload of a Serializable wire-value with ``tag``."""
    if isinstance(value, tuple) and len(value) == 2 and value[1] == tag:
        return value[0]
    return None


def _is_generated_node_set(name: object) -> bool:
    name = str(name)
    return (
        name in {"all", "surface_all"}
        or name.startswith("gmsh_nodes_")
        or (name.startswith("surface_") and name.endswith("_nodes"))
    )


def _canonical_array(value: np.ndarray, unordered: bool = False,
                     numeric_coordinates: bool = False):
    """Normalize dtype/shape details that do not change model meaning."""
    if value.dtype == object:
        return tuple(_canonical_serialized(item) for item in value.tolist())
    if value.size == 1:
        # A scalar material parameter may be serialized as ``()`` or ``(1,)``.
        # Reduce both to the same Python scalar representation.
        return value.reshape(-1)[0].item()
    if numeric_coordinates and np.issubdtype(value.dtype, np.number):
        # Reference-point/translation coordinates may be loaded as integer
        # arrays and rebuilt as float arrays.  They represent the same point;
        # use one numeric representation for the identity calculation.
        value = value.astype(np.float64, copy=False)
    if unordered and value.ndim == 1:
        if np.issubdtype(value.dtype, np.integer):
            value = value.astype(np.int64, copy=False)
        value = np.sort(value.reshape(-1))
    return np.ascontiguousarray(value)


def _surface_collection(value):
    """Canonicalize a surface's unordered ``(element_ids, side)`` pairs."""
    raw = _tagged_payload(value, "list")
    if raw is None:
        raw = _tagged_payload(value, "tuple")
    if raw is None and isinstance(value, (list, tuple)):
        raw = value
    if raw is None:
        return _canonical_serialized(value)

    faces = []
    for face in raw:
        pair = _tagged_payload(face, "list")
        if pair is None:
            pair = _tagged_payload(face, "tuple")
        if pair is None and isinstance(face, (list, tuple)):
            pair = face
        if pair is None or len(pair) != 2:
            faces.append(_canonical_serialized(face))
            continue

        element_ids = _tagged_payload(pair[0], "ndarray")
        if element_ids is not None:
            element_ids = _canonical_array(element_ids, unordered=True)
        else:
            element_ids = _canonical_serialized(pair[0], unordered=True)
        side = _canonical_serialized(pair[1])
        faces.append((element_ids, side))

    def sort_key(face):
        if isinstance(face, tuple) and len(face) == 2:
            ids, side = face
            return (repr(side), repr(ids))
        return (repr(face),)

    return tuple(sorted(faces, key=sort_key))


def _canonical_mapping(value: Mapping, key: str | None = None):
    """Canonicalize serialized dictionaries, including set-like mappings."""
    result = {}
    for item_key, item_value in value.items():
        if key == "set_nodes" and _is_generated_node_set(item_key):
            # These are generated from surfaces/mesh topology and are not an
            # independent model definition.
            continue
        if key == "set_elements" and str(item_key) == "all":
            continue
        if key in _SURFACE_MAPPING_KEYS and str(item_key) == "extern":
            # The Part exterior is an automatic all-boundary surface.
            continue
        if key in _SURFACE_MAPPING_KEYS:
            result[item_key] = _surface_collection(item_value)
        else:
            result[item_key] = _canonical_serialized(
                item_value,
                unordered=key in _UNORDERED_MAPPING_KEYS,
            )
    return result


def _update_checksum(checksum: int, value) -> int:
    if isinstance(value, torch.Tensor):
        array = np.ascontiguousarray(value.detach().cpu().numpy())
        checksum = zlib.crc32(b"tensor:", checksum)
        return _update_checksum(checksum, array)
    if isinstance(value, np.ndarray):
        checksum = zlib.crc32(b"ndarray:", checksum)
        checksum = zlib.crc32(str(value.dtype).encode("utf-8"), checksum)
        checksum = zlib.crc32(json.dumps(value.shape).encode("ascii"), checksum)
        if value.dtype == object:
            return _update_checksum(checksum, value.tolist())
        else:
            return zlib.crc32(np.ascontiguousarray(value).tobytes(), checksum)
    if isinstance(value, Mapping):
        checksum = zlib.crc32(b"dict{", checksum)
        for item_key in sorted(value, key=lambda item: repr(item)):
            checksum = _update_checksum(checksum, item_key)
            checksum = _update_checksum(checksum, value[item_key])
        return zlib.crc32(b"}", checksum)
    if (isinstance(value, (list, tuple)) and len(value) == 2
            and isinstance(value[1], str) and value[1] in {"list", "tuple"}):
        # Serializable stores a container type tag, but deserialization
        # normalizes tuples to lists.  Ignore that representation detail.
        checksum = zlib.crc32(b"sequence-tagged[", checksum)
        checksum = _update_checksum(checksum, value[0])
        return zlib.crc32(b"]", checksum)
    if isinstance(value, (list, tuple)):
        checksum = zlib.crc32(b"sequence[", checksum)
        for item in value:
            checksum = _update_checksum(checksum, item)
        return zlib.crc32(b"]", checksum)
    if value is None or isinstance(value, (bool, int, float, str)):
        data = json.dumps(value, sort_keys=True, ensure_ascii=False, default=repr).encode("utf-8")
        return zlib.crc32(data + b";", checksum)
    checksum = zlib.crc32(
        f"{type(value).__module__}.{type(value).__qualname__}:".encode("utf-8"),
        checksum,
    )
    return zlib.crc32(repr(value).encode("utf-8"), checksum)


def stable_hash(value) -> str:
    """Return an 8-character CRC32 check code for a serialized value."""
    return f"{_update_checksum(0, value):08x}"


def _canonical_serialized(value, *, key: str | None = None,
                          unordered: bool = False):
    """Normalize a ``Serializable._serialize`` tree for identity hashing.

    Follow the same rules as ``Serializable``: an explicit
    ``_serialized_attributes`` list is authoritative; otherwise all object
    attributes are considered and ``_serialized_attributes_exclude`` removes
    runtime fields.  Unselected attributes are represented by a serializer
    marker and are omitted from the identity tree.
    """
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
        tag = value[1]
        if tag in _SCALAR_TAGS:
            return value[0]
        if tag in {"ndarray", "Tensor"} and isinstance(value[0], np.ndarray):
            return _canonical_array(
                value[0],
                unordered=unordered,
                numeric_coordinates=key in {
                    "node", "point", "_translation", "_rotation", "_ref_location",
                },
            )
        if tag == "dict" and isinstance(value[0], dict):
            return _canonical_mapping(value[0], key=key)
        if tag in {"list", "tuple"} and isinstance(value[0], (list, tuple)):
            return tuple(_canonical_serialized(item, key=key, unordered=unordered)
                         for item in value[0])

    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], dict):
        class_name = value[1]
        if class_name in Serializable._subclasses:
            klass = Serializable._subclasses[class_name]
            excluded = set(klass._serialized_attributes_exclude)
            selected = set(klass._serialized_attributes)
            payload = {}
            for item_key, item in value[0].items():
                if item_key in excluded or (selected and item_key not in selected):
                    continue
                # The allocation size is derived from the part mesh and is
                # allowed to change when initialize() rebuilds runtime state.
                if item_key == "_RGC_requirements":
                    continue
                # Assembly treats an unspecified enabled flag as enabled.
                # Normalize both spellings so semantically identical models
                # retain the same identity after loading and rebuilding.
                if (item_key == "enabled" and isinstance(item, tuple)
                        and len(item) == 2 and item[1] == "NoneType"):
                    payload[item_key] = True
                    continue
                # Serializable marks unselected attributes as ``NoneType``.
                if (isinstance(item, tuple) and len(item) == 2
                        and item[1] == "NoneType"):
                    continue
                # Lamé parameters are derived from E and nu and can differ by
                # a float32 round-trip even when the material is identical.
                if "LinearElastic" in class_name and item_key in {"lambda_", "mu"}:
                    continue
                if "FEAController" in class_name and item_key == "cad":
                    cad_parts = _tagged_payload(item, "dict")
                    if cad_parts is None:
                        cad_object = item[0] if (
                            isinstance(item, tuple) and len(item) == 2
                            and isinstance(item[0], dict)) else None
                        cad_parts = _tagged_payload(
                            cad_object.get("parts") if isinstance(cad_object, dict) else None,
                            "dict")
                    if not cad_parts:
                        continue
                # A single full-block material assignment is derived from the
                # material itself.
                if item_key == "material_element_indices":
                    assignments = _tagged_payload(item, "dict")
                    if assignments is not None and not assignments:
                        continue
                    element_data = value[0].get("_elems")
                    element_data_payload = _tagged_payload(element_data, "ndarray")
                    if element_data_payload is None:
                        element_data_payload = _tagged_payload(element_data, "Tensor")
                    element_data = element_data_payload
                    if assignments and len(assignments) == 1 and element_data is not None:
                        indices = next(iter(assignments.values()))
                        indices_payload = _tagged_payload(indices, "ndarray")
                        if indices_payload is None:
                            indices_payload = _tagged_payload(indices, "Tensor")
                        indices = indices_payload
                        if indices is not None:
                            expected = np.arange(len(element_data), dtype=indices.dtype)
                            if np.array_equal(np.sort(indices.reshape(-1)), expected):
                                continue
                payload[item_key] = _canonical_serialized(
                    item,
                    key=item_key,
                    unordered=item_key in _UNORDERED_MAPPING_KEYS,
                )
            return payload, class_name
    if isinstance(value, dict):
        return _canonical_mapping(value, key=key)
    if isinstance(value, list):
        return tuple(_canonical_serialized(item, key=key, unordered=unordered)
                     for item in value)
    if isinstance(value, tuple):
        return tuple(_canonical_serialized(item, key=key, unordered=unordered)
                     for item in value)
    if isinstance(value, np.ndarray):
        return _canonical_array(
            value,
            unordered=unordered,
            numeric_coordinates=key in {
                "node", "point", "_translation", "_rotation", "_ref_location",
            },
        )
    return value


def controller_model_hash(controller) -> str:
    """Compute the controller's identity hash from its current model.

    ``FEAController._serialize()`` is the same representation used by
    ``save_model``/``load_model``.  Its canonical form keeps CAD, assembly,
    elements, sets, loads, constraints and solver settings in one source of
    truth while ignoring only container order and generated/derived fields.
    This function deliberately does not cache: a direct Python edit is
    visible on the next explicit hash check.
    """
    serialized_model = _canonical_serialized(controller._serialize())
    return stable_hash(serialized_model)
