# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Field layout and resolution: the bridge between Op declarations and IR data.

Op declarations (dsl.py) describe operations structurally: named operands,
results, successors, attributes, regions. Operations in the IR (ir.py) store
data positionally: operand value IDs in an array, result value IDs in an array,
successor block references in an array, attributes in a dict, regions in a list.

This module bridges the two: given an Op declaration, it computes a
FieldLayout that maps field names to positional indices. Given a
FieldLayout and an Operation instance, it resolves field names to
actual values.

The FieldLayout is computed once per Op declaration and cached. It
is the shared contract between the printer, parser, builder, and
verifier — all of which need to translate between field names (from
format specs and constraints) and positional data.

C equivalent: the code generator produces static field descriptor
tables where field names are replaced by integer indices at
generation time. The C format spec references field indices, not
strings. Same logical structure, zero runtime string lookups.

Most ops use the compact fixed-plus-trailing-variadic operand layout. Ops with
multiple named operand spans opt into segmented operands: each operand field has
one structural segment count in the C op's trailing storage, while the values
remain one flat operand array for use-def and generic pass infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Any, Protocol, runtime_checkable

from loom.dsl import Op
from loom.ir import (
    Block,
    Module,
    Operation,
    Region,
    TiedResult,
    Type,
    Value,
)

__all__ = [
    "FieldKind",
    "FieldDesc",
    "FieldLayout",
    "FormatFields",
    "ResolvedFields",
    "compute_layout",
    "resolve_fields",
]


# ============================================================================
# FormatFields protocol — the interface the format walker uses
# ============================================================================


@runtime_checkable
class FormatFields(Protocol):
    """Interface for the format walker's field access.

    ResolvedFields satisfies this protocol for all ops — both body ops
    and top-level symbol-defining ops (func.def, func.decl, etc.).
    The format walker is duck-typed over this interface.
    """

    def attr(self, name: str) -> Any: ...
    def is_present(self, name: str) -> bool: ...
    def successor(self, name: str) -> Block: ...
    def successors(self, name: str) -> list[Block]: ...
    def region(self, name: str) -> Region | None: ...
    def regions(self, name: str) -> list[Region]: ...
    def tied_result_map(self) -> dict[int, TiedResult]: ...
    def operand_name_for_tied(self, tied: TiedResult) -> str: ...
    def value_id(self, name: str) -> int: ...
    def value_ids(self, name: str) -> list[int]: ...
    def type_of(self, name: str) -> Type: ...
    def types_of(self, name: str) -> list[Type]: ...


# ============================================================================
# Field descriptor
# ============================================================================


@unique
class FieldKind(IntEnum):
    """Kind of field in an operation.

    Matches the C enum that the code generator will produce.
    """

    OPERAND = 0
    RESULT = 1
    ATTR = 2
    REGION = 3
    SUCCESSOR = 4


@dataclass(frozen=True, slots=True)
class FieldDesc:
    """Descriptor for a single named field.

    Maps a field name to its position in the operation's data arrays.

    kind: Whether this is an operand, result, successor, attribute, or region.
    index: Starting index in the relevant array.
    variadic: If True, this field spans from index to the end of the
        array (for the trailing variadic operand/result pattern).
    optional: If True, this fixed field may be absent from the operation's
        positional array. Optional fields must be trailing in their array.
    """

    kind: FieldKind
    index: int
    variadic: bool = False
    optional: bool = False


# ============================================================================
# Field layout
# ============================================================================


@dataclass(frozen=True, slots=True)
class FieldLayout:
    """Maps field names to their structural positions for an op kind.

    Computed once per Op declaration. The fields dict maps every
    field name (operands, results, attributes, regions) to a
    FieldDesc describing where to find it in the operation data.

    The fixed_operand_count is the number of required non-variadic operands.
    Optional operands may follow it as a bounded trailing run. The
    fixed_result_count is the number of non-variadic results. If there is a
    trailing variadic field, that field spans from the fixed count to the
    actual operation's operand/result count.

    When segmented_operands is true, operand FieldDesc.index values are
    descriptor/segment ordinals instead of flat operand offsets. The operation
    still stores operands in one flat array; generated builders, parsers, and C
    accessors maintain the segment-count tail metadata.
    """

    fields: dict[str, FieldDesc]
    fixed_operand_count: int
    fixed_result_count: int
    fixed_successor_count: int
    required_region_count: int
    fixed_region_count: int
    variadic_operand: str | None  # Name of the variadic operand, if any.
    variadic_result: str | None  # Name of the variadic result, if any.
    variadic_successor: str | None  # Name of the variadic successor, if any.
    variadic_region: str | None  # Name of the variadic region, if any.
    segmented_operands: bool = False


def compute_layout(op_decl: Op) -> FieldLayout:
    """Compute the field layout from an Op declaration.

    Raises ValueError if the declaration violates layout constraints
    (e.g., variadic operand is not trailing).
    """
    fields: dict[str, FieldDesc] = {}
    fixed_operand_count = 0
    fixed_result_count = 0
    fixed_successor_count = 0
    required_region_count = 0
    fixed_region_count = 0
    variadic_operand: str | None = None
    variadic_result: str | None = None
    variadic_successor: str | None = None
    variadic_region: str | None = None

    # Operands: use the legacy compact layout when possible, and segmented
    # counts when optional/variadic fields need independent spans.
    segmented_operands = False
    saw_optional_operand = False
    saw_variadic_operand = False
    for i, operand in enumerate(op_decl.operands):
        if operand.optional and operand.variadic:
            raise ValueError(
                f"Op '{op_decl.name}': operand '{operand.name}' cannot be "
                f"both optional and variadic."
            )
        if operand.variadic:
            if saw_variadic_operand or i != len(op_decl.operands) - 1:
                segmented_operands = True
            if saw_optional_operand:
                segmented_operands = True
            saw_variadic_operand = True
        elif saw_optional_operand and not operand.optional:
            segmented_operands = True
        if operand.optional:
            saw_optional_operand = True

    if segmented_operands:
        for i, operand in enumerate(op_decl.operands):
            if operand.variadic and variadic_operand is None:
                variadic_operand = operand.name
            fields[operand.name] = FieldDesc(
                FieldKind.OPERAND,
                i,
                variadic=operand.variadic,
                optional=operand.optional,
            )
    else:
        saw_optional_operand = False
        for i, operand in enumerate(op_decl.operands):
            if operand.variadic:
                variadic_operand = operand.name
                fields[operand.name] = FieldDesc(FieldKind.OPERAND, i, variadic=True)
            else:
                if operand.optional:
                    saw_optional_operand = True
                    fields[operand.name] = FieldDesc(
                        FieldKind.OPERAND,
                        i,
                        optional=True,
                    )
                else:
                    if saw_optional_operand:
                        raise ValueError(
                            f"Op '{op_decl.name}': required operand "
                            f"'{operand.name}' cannot follow an optional operand."
                        )
                    fields[operand.name] = FieldDesc(FieldKind.OPERAND, i)
                    fixed_operand_count += 1

    # Results: same pattern.
    for i, result in enumerate(op_decl.results):
        if result.variadic:
            if variadic_result is not None:
                raise ValueError(
                    f"Op '{op_decl.name}': multiple variadic results "
                    f"('{variadic_result}' and '{result.name}')."
                )
            if i != len(op_decl.results) - 1:
                raise ValueError(
                    f"Op '{op_decl.name}': variadic result '{result.name}' "
                    f"must be the last result."
                )
            variadic_result = result.name
            fields[result.name] = FieldDesc(FieldKind.RESULT, i, variadic=True)
        else:
            fields[result.name] = FieldDesc(FieldKind.RESULT, i)
            fixed_result_count += 1

    # Attributes: keyed by name (dict access, not positional).
    for i, attr in enumerate(op_decl.attrs):
        fields[attr.name] = FieldDesc(FieldKind.ATTR, i)

    # Successors: sequential indices, variadic must be last.
    for i, successor in enumerate(op_decl.successors):
        if successor.variadic:
            if variadic_successor is not None:
                raise ValueError(
                    f"Op '{op_decl.name}': multiple variadic successors "
                    f"('{variadic_successor}' and '{successor.name}')."
                )
            if i != len(op_decl.successors) - 1:
                raise ValueError(
                    f"Op '{op_decl.name}': variadic successor "
                    f"'{successor.name}' must be the last successor."
                )
            variadic_successor = successor.name
            fields[successor.name] = FieldDesc(FieldKind.SUCCESSOR, i, variadic=True)
        else:
            fields[successor.name] = FieldDesc(FieldKind.SUCCESSOR, i)
            fixed_successor_count += 1

    # Regions: sequential indices. Variadic regions and optional regions are
    # both trailing forms: a region list may have a variadic tail or a bounded
    # run of optional tail slots, but not both.
    saw_optional_region = False
    for i, region in enumerate(op_decl.regions):
        if region.variadic:
            if variadic_region is not None:
                raise ValueError(
                    f"Op '{op_decl.name}': multiple variadic regions "
                    f"('{variadic_region}' and '{region.name}')."
                )
            if region.optional:
                raise ValueError(
                    f"Op '{op_decl.name}': region '{region.name}' cannot be "
                    f"both optional and variadic."
                )
            if saw_optional_region:
                raise ValueError(
                    f"Op '{op_decl.name}': variadic region '{region.name}' "
                    f"cannot follow an optional region."
                )
            if i != len(op_decl.regions) - 1:
                raise ValueError(
                    f"Op '{op_decl.name}': variadic region '{region.name}' "
                    f"must be the last region."
                )
            variadic_region = region.name
            fields[region.name] = FieldDesc(FieldKind.REGION, i, variadic=True)
        else:
            if region.optional:
                saw_optional_region = True
                fields[region.name] = FieldDesc(
                    FieldKind.REGION,
                    i,
                    optional=True,
                )
            else:
                if saw_optional_region:
                    raise ValueError(
                        f"Op '{op_decl.name}': required region "
                        f"'{region.name}' cannot follow an optional region."
                    )
                fields[region.name] = FieldDesc(FieldKind.REGION, i)
                required_region_count += 1
            fixed_region_count += 1

    return FieldLayout(
        fields=fields,
        fixed_operand_count=fixed_operand_count,
        fixed_result_count=fixed_result_count,
        fixed_successor_count=fixed_successor_count,
        required_region_count=required_region_count,
        fixed_region_count=fixed_region_count,
        variadic_operand=variadic_operand,
        variadic_result=variadic_result,
        variadic_successor=variadic_successor,
        variadic_region=variadic_region,
        segmented_operands=segmented_operands,
    )


# ============================================================================
# Field resolution
# ============================================================================


class ResolvedFields:
    """Resolved field values for a specific operation instance.

    Created by resolve_fields() from a FieldLayout and an Operation.
    Provides typed access to field data by name, translating between
    the format spec's field names and the operation's positional data.

    The printer calls these methods while walking the format spec.
    The verifier calls them to evaluate constraints.
    """

    __slots__ = ("_layout", "_op", "_module")

    def __init__(self, layout: FieldLayout, op: Operation, module: Module) -> None:
        self._layout = layout
        self._op = op
        self._module = module

    def _desc(self, name: str) -> FieldDesc:
        """Look up the field descriptor, raising on unknown fields."""
        desc = self._layout.fields.get(name)
        if desc is None:
            raise KeyError(
                f"Unknown field '{name}' on op '{self._op.name}'. "
                f"Known fields: {sorted(self._layout.fields.keys())}"
            )
        return desc

    def _operand_segment(self, name: str, desc: FieldDesc) -> list[int]:
        """Return the flat values owned by one segmented operand field."""
        operand_field_count = sum(
            1
            for field in self._layout.fields.values()
            if field.kind == FieldKind.OPERAND
        )
        counts = self._op.operand_segment_counts
        if len(counts) != operand_field_count:
            raise ValueError(
                f"Op '{self._op.name}' uses segmented operands but has "
                f"{len(counts)} segment counts."
            )
        if sum(counts) != len(self._op.operands):
            raise ValueError(
                f"Op '{self._op.name}' operand segment counts do not sum to "
                "the flat operand count."
            )
        start = sum(counts[: desc.index])
        count = counts[desc.index]
        end = start + count
        if end > len(self._op.operands):
            raise ValueError(
                f"Op '{self._op.name}' operand segment '{name}' extends past "
                f"the flat operand list."
            )
        return self._op.operands[start:end]

    # --- Value references ---

    def value_id(self, name: str) -> int:
        """Get the single value ID for a non-variadic operand or result."""
        desc = self._desc(name)
        if desc.kind == FieldKind.OPERAND:
            if self._layout.segmented_operands:
                values = self._operand_segment(name, desc)
                if not values:
                    raise IndexError(
                        f"Operand field '{name}' is absent on op '{self._op.name}'."
                    )
                return values[0]
            if desc.index >= len(self._op.operands):
                raise IndexError(
                    f"Operand field '{name}' index {desc.index} is absent "
                    f"on op '{self._op.name}'."
                )
            value_id: int = self._op.operands[desc.index]
            return value_id
        if desc.kind == FieldKind.RESULT:
            result_id: int = self._op.results[desc.index]
            return result_id
        raise TypeError(
            f"Field '{name}' is {desc.kind.name}, not an operand or result."
        )

    def value_ids(self, name: str) -> list[int]:
        """Get value IDs for a variadic operand or result."""
        desc = self._desc(name)
        if desc.kind == FieldKind.OPERAND:
            if self._layout.segmented_operands:
                return self._operand_segment(name, desc)
            if desc.variadic:
                ids: list[int] = self._op.operands[desc.index :]
                return ids
            if desc.index >= len(self._op.operands):
                if desc.optional:
                    return []
                raise IndexError(
                    f"Operand field '{name}' index {desc.index} is absent "
                    f"on op '{self._op.name}'."
                )
            return [self._op.operands[desc.index]]
        if desc.kind == FieldKind.RESULT:
            if desc.variadic:
                ids = self._op.results[desc.index :]
                return ids
            return [self._op.results[desc.index]]
        raise TypeError(
            f"Field '{name}' is {desc.kind.name}, not an operand or result."
        )

    def value(self, name: str) -> Value:
        """Get the Value object for a non-variadic operand or result."""
        return self._module.values[self.value_id(name)]

    def values(self, name: str) -> list[Value]:
        """Get Value objects for a variadic operand or result."""
        return [self._module.values[vid] for vid in self.value_ids(name)]

    # --- Value names ---

    def value_name(self, name: str) -> str:
        """Get the SSA name of a non-variadic operand or result."""
        return self.value(name).name

    def value_names(self, name: str) -> list[str]:
        """Get SSA names for a variadic operand or result."""
        return [v.name for v in self.values(name)]

    # --- Types ---

    def type_of(self, name: str) -> Type:
        """Get the type of a non-variadic operand or result."""
        return self.value(name).type

    def types_of(self, name: str) -> list[Type]:
        """Get types for a variadic operand or result."""
        return [v.type for v in self.values(name)]

    # --- Attributes ---

    def attr(self, name: str) -> Any:
        """Get an attribute value by name."""
        desc = self._desc(name)
        if desc.kind != FieldKind.ATTR:
            raise TypeError(f"Field '{name}' is {desc.kind.name}, not an attribute.")
        return self._op.attributes.get(name)

    # --- Successors ---

    def successor(self, name: str) -> Block:
        """Get a successor block by name."""
        desc = self._desc(name)
        if desc.kind != FieldKind.SUCCESSOR:
            raise TypeError(f"Field '{name}' is {desc.kind.name}, not a successor.")
        return self._op.successors[desc.index]

    def successors(self, name: str) -> list[Block]:
        """Get blocks for a variadic successor field."""
        desc = self._desc(name)
        if desc.kind != FieldKind.SUCCESSOR:
            raise TypeError(f"Field '{name}' is {desc.kind.name}, not a successor.")
        if desc.variadic:
            return self._op.successors[desc.index :]
        return [self._op.successors[desc.index]]

    # --- Regions ---

    def region(self, name: str) -> Region | None:
        """Get a region by name."""
        desc = self._desc(name)
        if desc.kind != FieldKind.REGION:
            raise TypeError(f"Field '{name}' is {desc.kind.name}, not a region.")
        if desc.index >= len(self._op.regions):
            if desc.optional:
                return None
            raise IndexError(
                f"Region field '{name}' index {desc.index} is absent "
                f"on op '{self._op.name}'."
            )
        return self._op.regions[desc.index]

    def regions(self, name: str) -> list[Region]:
        """Get regions for a variadic region field."""
        desc = self._desc(name)
        if desc.kind != FieldKind.REGION:
            raise TypeError(f"Field '{name}' is {desc.kind.name}, not a region.")
        if desc.variadic:
            return self._op.regions[desc.index :]
        if desc.index >= len(self._op.regions):
            if desc.optional:
                return []
            raise IndexError(
                f"Region field '{name}' index {desc.index} is absent "
                f"on op '{self._op.name}'."
            )
        return [self._op.regions[desc.index]]

    # --- Tied results ---

    def tied_result_map(self) -> dict[int, TiedResult]:
        """Build a map from result index to TiedResult.

        Used by ResultTypeList to determine which results are tied.
        Returns {result_index: TiedResult} for all tied results.
        """
        return {tr.result_index: tr for tr in self._op.tied_results}

    def operand_name_for_tied(self, tied: TiedResult) -> str:
        """Get the SSA name of the operand or func arg a result is tied to.

        For body ops, tied.operand_index indexes into op.operands. For
        func-like ops with a body region (func.def, func.template), there
        are no op-level operands — tied.operand_index indexes into the
        entry block's arguments instead.
        """
        if tied.operand_index < len(self._op.operands):
            value_id = self._op.operands[tied.operand_index]
            name = self._module.values[value_id].name
        elif self._op.regions:
            entry = (
                self._op.regions[0].blocks[0] if self._op.regions[0].blocks else None
            )
            if entry and tied.operand_index < len(entry.arg_ids):
                value_id = entry.arg_ids[tied.operand_index]
                name = self._module.values[value_id].name
            else:
                return f"%arg{tied.operand_index}"
        else:
            return f"%arg{tied.operand_index}"
        return "%" + name

    # --- FuncArgs support for func-like ops ---

    def func_args(self) -> tuple[list[str], list[Type], list[int]]:
        """Get function argument data for the FuncArgs format element.

        For ops with a body region (func.def, func.template), args are the
        entry block's arguments. For declaration-style ops (func.decl,
        func.ukernel), args are the op's operands.
        """
        if self._op.regions:
            entry = (
                self._op.regions[0].blocks[0] if self._op.regions[0].blocks else None
            )
            arg_ids = list(entry.arg_ids) if entry else []
        else:
            arg_ids = list(self._op.operands)
        names = [self._module.values[vid].name for vid in arg_ids]
        types = [self._module.values[vid].type for vid in arg_ids]
        return (names, types, arg_ids)

    # --- Presence checks (for OptionalGroup anchors) ---

    def is_present(self, name: str) -> bool:
        """Check if an optional/variadic field has data.

        Used by OptionalGroup to decide whether to emit its elements.
        Returns False for: None attributes, empty variadic lists,
        absent optional operands, absent regions.
        """
        desc = self._layout.fields.get(name)
        if desc is None:
            # Implicit fields (iv, args, predicates) — check operation attrs.
            return name in self._op.attributes

        if desc.kind == FieldKind.OPERAND:
            if self._layout.segmented_operands:
                return bool(self._operand_segment(name, desc))
            if desc.variadic:
                return len(self._op.operands) > desc.index
            return desc.index < len(self._op.operands)
        if desc.kind == FieldKind.RESULT:
            if desc.variadic:
                return len(self._op.results) > desc.index
            return desc.index < len(self._op.results)
        if desc.kind == FieldKind.ATTR:
            value = self._op.attributes.get(name)
            return value is not None
        if desc.kind == FieldKind.SUCCESSOR:
            if desc.variadic:
                return len(self._op.successors) > desc.index
            return desc.index < len(self._op.successors)
        if desc.kind == FieldKind.REGION:
            if desc.variadic:
                return len(self._op.regions) > desc.index
            return desc.index < len(self._op.regions) and (
                len(self._op.regions[desc.index].blocks) > 0
            )
        return False


def resolve_fields(
    layout: FieldLayout, op: Operation, module: Module
) -> ResolvedFields:
    """Create a ResolvedFields for a specific operation instance.

    This is the main entry point. The layout is computed once per op
    kind (from Op declaration) and reused. ResolvedFields is created
    per-instance when printing, verifying, etc.
    """
    return ResolvedFields(layout, op, module)
