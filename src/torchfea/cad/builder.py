"""Replay CAD operation objects into one Gmsh OCC model."""

from __future__ import annotations

import logging
from typing import Iterable

from .operations import CADOperation, DimTag


LOGGER = logging.getLogger("torchfea.cae")


class OCCBuilder:
    """Coordinate history dependencies and OCC entity lifetimes."""

    def __init__(self, gmsh) -> None:
        self.occ = gmsh.model.occ
        self._operation_outputs: dict[str, list[DimTag]] = {}
        self._available_operation_ids: set[str] = set()
        self._live_entities: list[DimTag] = []

    def entities(self, operation_id: str) -> list[DimTag]:
        """Return the OCC entities produced by an earlier operation."""
        return list(self._operation_outputs[operation_id])

    def replay(self, operations: Iterable[CADOperation]) -> list[int]:
        """Apply an ordered history and return its final volume tags."""
        operations = list(operations)
        if not operations:
            raise ValueError("A Part needs at least one OCC feature.")

        for operation in operations:
            self._validate_operation(operation)
            input_entities = {
                entity
                for input_id in operation.input_ids
                for entity in self._operation_outputs[input_id]
            }
            LOGGER.debug(
                "OCC operation: id=%s type=%s inputs=%s",
                operation.id,
                type(operation).__name__,
                list(operation.input_ids),
            )

            output = self._normalize_output(operation, operation.apply_occ(self))
            self.occ.synchronize()

            self._live_entities = [
                entity
                for entity in self._live_entities
                if entity not in input_entities
            ]
            self._live_entities.extend(output)
            self._operation_outputs[operation.id] = output
            self._available_operation_ids.difference_update(operation.input_ids)
            self._available_operation_ids.add(operation.id)

        final_volumes = [
            tag
            for dimension, tag in dict.fromkeys(self._live_entities)
            if dimension == 3
        ]
        if not final_volumes:
            raise ValueError("The CAD history produced no final solid volume.")
        return final_volumes

    def _validate_operation(self, operation: CADOperation) -> None:
        if not isinstance(operation, CADOperation):
            raise TypeError(
                "OCC history entries must inherit CADOperation, got "
                f"{type(operation).__name__}.")

        operation_id = str(operation.id).strip()
        if not operation_id:
            raise ValueError("Every OCC feature needs a non-empty id.")
        if operation_id in self._operation_outputs:
            raise ValueError(f"Duplicate OCC feature id '{operation_id}'.")
        if operation_id != operation.id:
            raise ValueError(
                f"OCC feature id '{operation.id}' contains surrounding whitespace.")

        unknown_inputs = [
            input_id
            for input_id in operation.input_ids
            if input_id not in self._operation_outputs
        ]
        if unknown_inputs:
            raise ValueError(
                f"OCC feature '{operation_id}' references unknown or future "
                f"feature(s): {unknown_inputs}.")

        unavailable_inputs = [
            input_id
            for input_id in operation.input_ids
            if input_id not in self._available_operation_ids
        ]
        if unavailable_inputs:
            raise ValueError(
                f"OCC feature '{operation_id}' references consumed "
                f"feature(s): {unavailable_inputs}.")

        if len(operation.input_ids) != len(set(operation.input_ids)):
            raise ValueError(
                f"OCC feature '{operation_id}' contains duplicate inputs.")

    @staticmethod
    def _normalize_output(
        operation: CADOperation,
        output: list[DimTag],
    ) -> list[DimTag]:
        entities = list(dict.fromkeys(
            (int(dimension), int(tag))
            for dimension, tag in output
        ))
        if not entities:
            raise ValueError(
                f"OCC feature '{operation.id}' produced no solid volume.")
        if any(dimension != 3 for dimension, _tag in entities):
            raise ValueError(
                f"OCC feature '{operation.id}' produced a non-volume entity.")
        return entities
