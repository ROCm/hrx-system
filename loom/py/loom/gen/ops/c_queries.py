# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C-visible op layout and metadata queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from loom.assembly import Clause, FormatElement, FuncArgs, OptionalGroup, Scope
from loom.dsl import (
    ATTR_TYPE_FLAGS,
    AttrDef,
    FuncLikeInterface,
    Op,
    Operand,
    RegionDef,
    TypeConstraint,
)
from loom.fields import FieldKind, FieldLayout
from loom.gen.ops.c_enums import CONSTRAINT_MAP


def find_interface[T](op: Op, iface_class: type[T]) -> T | None:
    """Returns the interface instance from op.interfaces matching |iface_class|."""
    for iface in op.interfaces:
        if isinstance(iface, iface_class):
            return iface
    return None


def func_args_are_operands(op: Op) -> bool:
    """Returns true if FuncArgs should be stored in the op's operand array."""
    func_like_iface = find_interface(op, FuncLikeInterface)
    return bool(func_like_iface and func_like_iface.args_as_operands)


def func_args_field_name(op: Op) -> str:
    """Returns the FuncArgs field name declared by a func-like op format."""

    def walk(elements: Sequence[FormatElement]) -> str | None:
        for element in elements:
            match element:
                case FuncArgs(field=name):
                    return name
                case OptionalGroup(elements=inner) | Scope(elements=inner) | Clause(elements=inner):
                    nested = walk(inner)
                    if nested is not None:
                        return nested
                case _:
                    continue
        return None

    return walk(op.format) or "args"


def func_args_field_names(op: Op) -> set[str]:
    """Returns FuncArgs field names explicitly declared by an op format."""
    names: set[str] = set()

    def walk(elements: Sequence[FormatElement]) -> None:
        for element in elements:
            match element:
                case FuncArgs(field=name):
                    names.add(name)
                case OptionalGroup(elements=inner) | Scope(elements=inner) | Clause(elements=inner):
                    walk(inner)
                case _:
                    continue

    walk(op.format)
    return names


def explicit_func_args_operand(op: Op) -> Operand | None:
    """Returns the operand descriptor that backs FuncArgs, if declared."""
    if not func_args_are_operands(op):
        return None
    field_name = func_args_field_name(op)
    for operand in op.operands:
        if operand.name != field_name:
            continue
        if not operand.variadic:
            raise ValueError(f"Op '{op.name}': FuncArgs field '{field_name}' must be a variadic operand when declared explicitly")
        return operand
    return None


def has_flags_attr(op: Op) -> bool:
    """Returns true if the op has any flags-typed attributes."""
    return any(attr_def.attr_type == ATTR_TYPE_FLAGS for attr_def in op.attrs)


def non_flags_attrs(op: Op) -> list[AttrDef]:
    """Returns attributes stored in the regular attribute array."""
    return [attr_def for attr_def in op.attrs if attr_def.attr_type != ATTR_TYPE_FLAGS]


def resolve_attr_index(op: Op, attr_name: str | None, interface_name: str = "interface") -> int:
    """Resolves an attr name to its non-flags attr index.

    Returns 0xFF (LOOM_ATTR_INDEX_NONE) if attr_name is None.
    Raises ValueError if the attr name is not found on the op.
    |interface_name| is used in error messages to identify which
    interface declaration is referencing this attr.
    """
    if attr_name is None:
        return 0xFF
    index = 0
    for attr_def in op.attrs:
        if attr_def.attr_type == ATTR_TYPE_FLAGS:
            continue
        if attr_def.name == attr_name:
            return index
        index += 1
    raise ValueError(f"{interface_name} on {op.name!r}: attr {attr_name!r} not found. Available: {[attr.name for attr in op.attrs if attr.attr_type != ATTR_TYPE_FLAGS]}")


def resolve_region_index(op: Op, region_name: str | None, interface_name: str = "interface") -> int:
    """Resolves a region name to its region index.

    Returns 0xFF (LOOM_REGION_INDEX_NONE) if region_name is None.
    Raises ValueError if the region name is not found on the op.
    |interface_name| is used in error messages.
    """
    if region_name is None:
        return 0xFF
    for i, region_def in enumerate(op.regions):
        if region_def.name == region_name:
            return i
    raise ValueError(f"{interface_name} on {op.name!r}: region {region_name!r} not found. Available: {[region.name for region in op.regions]}")


def resolve_operand_index(op: Op, operand_name: str | None, interface_name: str = "interface") -> int:
    """Resolves an operand name to its index in the op's operand list.

    Returns 0xFF (LOOM_OPERAND_INDEX_NONE) if operand_name is None.
    Raises ValueError if the operand name is not found on the op.
    |interface_name| is used in error messages.

    The returned index is the position in op.operands. This includes
    fixed operands and the variadic operand tail. For interfaces that
    need the offset of a variadic operand specifically, the returned
    index is exactly that offset.
    """
    if operand_name is None:
        return 0xFF
    for i, operand in enumerate(op.operands):
        if operand.name == operand_name:
            return i
    raise ValueError(f"{interface_name} on {op.name!r}: operand {operand_name!r} not found. Available: {[operand.name for operand in op.operands]}")


def resolve_successor_selector_operand_index(op: Op) -> int | None:
    """Resolves the op-level successor selector operand index, if present."""
    if op.successor_selector is None:
        return None
    index = resolve_operand_index(op, op.successor_selector, "successor_selector")
    if index > 0xFFFF:
        raise ValueError(f"Op '{op.name}': successor_selector operand index {index} exceeds uint16_t range")
    if len(op.successors) < 2:
        raise ValueError(f"Op '{op.name}': successor_selector requires at least two successors")
    return index


def resolve_result_index(op: Op, result_name: str | None, interface_name: str = "interface") -> int:
    """Resolves a result name to its index in the op's result list.

    Returns 0xFF (LOOM_RESULT_INDEX_NONE) if result_name is None.
    Raises ValueError if the result name is not found on the op.
    |interface_name| is used in error messages.

    The returned index is the position in op.results. This includes
    fixed results and the variadic result tail.
    """
    if result_name is None:
        return 0xFF
    for i, result in enumerate(op.results):
        if result.name == result_name:
            return i
    raise ValueError(f"{interface_name} on {op.name!r}: result {result_name!r} not found. Available: {[result.name for result in op.results]}")


def resolve_ownership_source_operand_index(op: Op, operand_name: str) -> int:
    index = resolve_operand_index(op, operand_name, "ownership effect")
    if index > 0xFE:
        raise ValueError(f"ownership effect on {op.name!r}: source operand {operand_name!r} index {index} exceeds uint8_t range")
    return index


def resolve_block_arg_index(
    op: Op,
    region_name: str,
    arg_name: str | None,
    interface_name: str = "interface",
) -> int:
    """Resolves a block argument name to its index in the entry block.

    The lookup considers a region's implicit block arguments first
    (declared via implicit_args=(("name", "type"),) on RegionDef),
    then any block args derived from BindingList format elements
    (which the format pipeline appends after implicit args).

    Returns 0xFF (LOOM_BLOCK_ARG_INDEX_NONE) if arg_name is None.
    Raises ValueError if the region or arg name is not found.
    """
    if arg_name is None:
        return 0xFF
    region_def: RegionDef | None = None
    for candidate in op.regions:
        if candidate.name == region_name:
            region_def = candidate
            break
    if region_def is None:
        raise ValueError(f"{interface_name} on {op.name!r}: region {region_name!r} not found. Available: {[region.name for region in op.regions]}")
    for i, (name, _type) in enumerate(region_def.implicit_args):
        if name == arg_name:
            return i
    raise ValueError(f"{interface_name} on {op.name!r}: block arg {arg_name!r} not found in region {region_name!r}. Available implicit_args: {[name for name, _ in region_def.implicit_args]}")


TYPE_PROPAGATION_REFINABLE_CONSTRAINTS = frozenset(
    {
        TypeConstraint.TILE,
        TypeConstraint.TENSOR,
        TypeConstraint.VECTOR,
        TypeConstraint.RANK_ONE_VECTOR,
        TypeConstraint.VIEW,
        TypeConstraint.INTEGER_ELEMENT,
        TypeConstraint.FLOAT_ELEMENT,
        TypeConstraint.INDEX_OR_NON_I1_INTEGER_ELEMENT,
        TypeConstraint.I1_ELEMENT,
        TypeConstraint.I8_ELEMENT,
        TypeConstraint.I32_ELEMENT,
        TypeConstraint.F16_OR_BF16_ELEMENT,
        TypeConstraint.F32_ELEMENT,
        TypeConstraint.ANY,
        TypeConstraint.ANY_ENCODING,
        TypeConstraint.ENCODING_LAYOUT,
        TypeConstraint.ENCODING_SCHEMA,
        TypeConstraint.ENCODING_STORAGE,
        TypeConstraint.ENCODING_TRANSFORM,
        TypeConstraint.POOL,
    }
)

TYPE_PROPAGATION_RELATIONS = frozenset(
    {
        "LOOM_RELATION_PAIRWISE_EQ",
        "LOOM_RELATION_ALL_SAME",
        "LOOM_RELATION_REGION_ARG_MATCH",
        "LOOM_RELATION_YIELD_MATCH",
        "LOOM_RELATION_VARIADIC_MATCH",
    }
)

TYPE_PROPAGATION_PROPERTIES = frozenset(
    {
        "LOOM_PROPERTY_TYPE",
        "LOOM_PROPERTY_ENCODING",
        "LOOM_PROPERTY_SHAPE",
    }
)

TYPE_PROPAGATION_TYPE_NOOP_RELATIONS = frozenset(
    {
        "LOOM_RELATION_YIELD_MATCH",
        "LOOM_RELATION_VARIADIC_MATCH",
    }
)


def _op_value_type_constraints(op: Op) -> dict[str, TypeConstraint]:
    constraints: dict[str, TypeConstraint] = {}
    for operand in op.operands:
        constraints[operand.name] = operand.type_constraint
    if func_args_are_operands(op) and explicit_func_args_operand(op) is None:
        constraints[func_args_field_name(op)] = TypeConstraint.ANY
    for result in op.results:
        constraints[result.name] = result.type_constraint
    return constraints


def _constraint_arg_can_refine_type(
    op: Op,
    layout: FieldLayout,
    value_type_constraints: Mapping[str, TypeConstraint],
    arg_name: str,
) -> bool:
    field = layout.fields.get(arg_name)
    if field is None:
        return False
    if field.kind == FieldKind.REGION:
        return True
    if field.kind not in (FieldKind.OPERAND, FieldKind.RESULT):
        return False
    type_constraint = value_type_constraints.get(arg_name)
    return type_constraint in TYPE_PROPAGATION_REFINABLE_CONSTRAINTS


def op_has_type_propagation_candidate(op: Op, layout: FieldLayout) -> bool:
    if op.type_transfer:
        return True
    value_type_constraints = _op_value_type_constraints(op)
    for constraint in op.constraints:
        constraint_entry = CONSTRAINT_MAP.get(constraint.name)
        if constraint_entry is None:
            continue
        relation_name, property_name = constraint_entry
        if property_name == "$data" or property_name == "$type_constraint_data":
            continue
        if relation_name not in TYPE_PROPAGATION_RELATIONS:
            continue
        if property_name not in TYPE_PROPAGATION_PROPERTIES:
            continue
        if property_name == "LOOM_PROPERTY_TYPE" and relation_name in TYPE_PROPAGATION_TYPE_NOOP_RELATIONS:
            continue
        if any(_constraint_arg_can_refine_type(op, layout, value_type_constraints, arg_name) for arg_name in constraint.args):
            return True
    return False
