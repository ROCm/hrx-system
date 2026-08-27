# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python predicates for declarative operation constraints.

The op declaration DSL stores these predicates beside the same relation data
projected into C verifier tables. Predicates operate on resolved field values
and deliberately avoid importing the declaration or IR modules so the DSL can
use them without a dependency cycle.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

type ValidateFn = Callable[[dict[str, Any]], tuple[bool, str]]
type TypeSatisfiesFn = Callable[[Any, Any], bool]


def _flatten_field(name: str, value: Any) -> list[tuple[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [(f"{name}[{i}]", item) for i, item in enumerate(value)]
    return [(name, value)]


def _field_value_type(item: Any) -> Any:
    return item.type if hasattr(item, "type") else item


def _field_element_type(item: Any) -> Any:
    value_type = _field_value_type(item)
    if hasattr(value_type, "element_type"):
        return value_type.element_type
    if hasattr(value_type, "dtype"):
        return value_type.dtype
    if hasattr(value_type, "type_kind") and hasattr(value_type, "kind"):
        return value_type
    return None


def _region_entry_args(region_value: Any) -> Sequence[Any] | None:
    return getattr(region_value, "entry_args", None)


def _region_terminator_operands(region_value: Any) -> Sequence[Any] | None:
    return getattr(region_value, "terminator_operands", None)


def _region_or_field_items(name: str, value: Any) -> list[tuple[str, Any]] | None:
    if hasattr(value, "entry_args"):
        entry_args = _region_entry_args(value)
        if entry_args is None:
            return None
        return [(f"{name}[{i}]", item) for i, item in enumerate(entry_args)]
    return _flatten_field(name, value)


def _validate_positional_types(
    lhs_name: str,
    lhs_items: list[tuple[str, Any]],
    rhs_name: str,
    rhs_items: list[tuple[str, Any]],
    *,
    element_types: bool = False,
) -> tuple[bool, str]:
    if len(lhs_items) != len(rhs_items):
        return (
            False,
            f"'{lhs_name}' count {len(lhs_items)} != "
            f"'{rhs_name}' count {len(rhs_items)}",
        )
    for (lhs_display_name, lhs_item), (rhs_display_name, rhs_item) in zip(
        lhs_items, rhs_items, strict=True
    ):
        # Invalid value IDs are diagnosed structurally by the verifier. Leave
        # their dependent type relationships unevaluated.
        if lhs_item is None or rhs_item is None:
            continue
        lhs_type = (
            _field_element_type(lhs_item)
            if element_types
            else _field_value_type(lhs_item)
        )
        rhs_type = (
            _field_element_type(rhs_item)
            if element_types
            else _field_value_type(rhs_item)
        )
        if lhs_type != rhs_type:
            return (
                False,
                f"'{lhs_display_name}' type {lhs_type} != "
                f"'{rhs_display_name}' type {rhs_type}",
            )
    return (True, "")


def same_encoding(fields: tuple[str, ...]) -> ValidateFn:
    """Builds a predicate requiring equal encodings across value fields."""

    def validate(values: dict[str, Any]) -> tuple[bool, str]:
        encodings: list[tuple[str, Any]] = []
        for name in fields:
            for display_name, item in _flatten_field(name, values.get(name)):
                encoding = getattr(_field_value_type(item), "encoding", None)
                encodings.append((display_name, encoding))
        if len(encodings) < 2:
            return (True, "")
        first_name, first_encoding = encodings[0]
        for entry_name, encoding in encodings[1:]:
            if encoding != first_encoding:
                return (
                    False,
                    f"'{entry_name}' encoding {encoding} != "
                    f"'{first_name}' encoding {first_encoding}",
                )
        return (True, "")

    return validate


def block_arg_count(region: str, inputs: str) -> ValidateFn:
    """Builds a predicate relating entry block args to another field."""

    def validate(values: dict[str, Any]) -> tuple[bool, str]:
        block_args = _region_entry_args(values.get(region))
        input_items = _region_or_field_items(inputs, values.get(inputs))
        if block_args is None or input_items is None:
            return (True, "")
        if len(block_args) == len(input_items):
            return (True, "")
        return (
            False,
            f"'{region}' block argument count {len(block_args)} != "
            f"'{inputs}' count {len(input_items)}",
        )

    return validate


def block_args_satisfy(
    region: str,
    constraint: Any,
    type_satisfies: TypeSatisfiesFn,
) -> ValidateFn:
    """Builds a predicate applying a type constraint to all block args."""

    def validate(values: dict[str, Any]) -> tuple[bool, str]:
        block_args = _region_entry_args(values.get(region))
        if block_args is None:
            return (True, "")
        for index, block_arg in enumerate(block_args):
            if block_arg is None:
                continue
            block_arg_type = _field_value_type(block_arg)
            if type_satisfies(block_arg_type, constraint):
                continue
            return (
                False,
                f"'{region}[{index}]' type {block_arg_type} does not satisfy "
                f"{constraint.value}",
            )
        return (True, "")

    return validate


def block_args_match(
    region: str, inputs: str, *, element_types: bool = False
) -> ValidateFn:
    """Builds a predicate matching block args to value or region fields."""

    def validate(values: dict[str, Any]) -> tuple[bool, str]:
        block_args = _region_entry_args(values.get(region))
        input_items = _region_or_field_items(inputs, values.get(inputs))
        if block_args is None or input_items is None:
            return (True, "")
        block_items = [(f"{region}[{i}]", item) for i, item in enumerate(block_args)]
        return _validate_positional_types(
            region,
            block_items,
            inputs,
            input_items,
            element_types=element_types,
        )

    return validate


def _valid_target_args(
    values: dict[str, Any], target_region: str, target_inputs: str
) -> tuple[Sequence[Any], list[tuple[str, Any]]] | None:
    target_args = _region_entry_args(values.get(target_region))
    target_input_items = _region_or_field_items(
        target_inputs, values.get(target_inputs)
    )
    if target_args is None or target_input_items is None:
        return None
    target_items = [
        (f"{target_region}[{i}]", item) for i, item in enumerate(target_args)
    ]
    target_args_valid, _ = _validate_positional_types(
        target_region,
        target_items,
        target_inputs,
        target_input_items,
    )
    return (target_args, target_items) if target_args_valid else None


def condition_forwarded_count(
    condition_region: str, target_region: str, target_inputs: str
) -> ValidateFn:
    """Builds a predicate matching condition forwarding and target arity."""

    def validate(values: dict[str, Any]) -> tuple[bool, str]:
        terminator_operands = _region_terminator_operands(values.get(condition_region))
        target = _valid_target_args(values, target_region, target_inputs)
        if terminator_operands is None or not terminator_operands or target is None:
            return (True, "")
        target_args, _ = target
        forwarded_count = len(terminator_operands) - 1
        if forwarded_count == len(target_args):
            return (True, "")
        return (
            False,
            f"'{condition_region}' forwarded count {forwarded_count} != "
            f"'{target_region}' block argument count {len(target_args)}",
        )

    return validate


def condition_forwarded_types(
    condition_region: str, target_region: str, target_inputs: str
) -> ValidateFn:
    """Builds a predicate matching condition forwarding and target types."""

    def validate(values: dict[str, Any]) -> tuple[bool, str]:
        terminator_operands = _region_terminator_operands(values.get(condition_region))
        target = _valid_target_args(values, target_region, target_inputs)
        if terminator_operands is None or not terminator_operands or target is None:
            return (True, "")
        target_args, target_items = target
        forwarded_items = [
            (f"forwarded[{i}]", item) for i, item in enumerate(terminator_operands[1:])
        ]
        if len(forwarded_items) != len(target_args):
            # The paired count relation owns this diagnostic.
            return (True, "")
        return _validate_positional_types(
            "forwarded", forwarded_items, target_region, target_items
        )

    return validate


def yield_count(region: str, results: str) -> ValidateFn:
    """Builds a predicate matching terminator and result counts."""

    def validate(values: dict[str, Any]) -> tuple[bool, str]:
        yielded = _region_terminator_operands(values.get(region))
        if yielded is None:
            return (True, "")
        result_items = _flatten_field(results, values.get(results))
        if len(yielded) == len(result_items):
            return (True, "")
        return (
            False,
            f"'{region}' terminator operand count {len(yielded)} != "
            f"'{results}' count {len(result_items)}",
        )

    return validate


def yield_types(
    region: str, results: str, *, element_types: bool = False
) -> ValidateFn:
    """Builds a predicate matching terminator and result types."""

    def validate(values: dict[str, Any]) -> tuple[bool, str]:
        yielded = _region_terminator_operands(values.get(region))
        if yielded is None:
            return (True, "")
        yielded_items = [
            (f"{region}.yield[{i}]", item) for i, item in enumerate(yielded)
        ]
        result_items = _flatten_field(results, values.get(results))
        if len(yielded_items) != len(result_items):
            # The paired count relation owns this diagnostic.
            return (True, "")
        return _validate_positional_types(
            f"{region}.yield",
            yielded_items,
            results,
            result_items,
            element_types=element_types,
        )

    return validate


def variadic_values_match(lhs: str, rhs: str) -> ValidateFn:
    """Builds a predicate matching two variadic value fields."""

    def validate(values: dict[str, Any]) -> tuple[bool, str]:
        return _validate_positional_types(
            lhs,
            _flatten_field(lhs, values.get(lhs)),
            rhs,
            _flatten_field(rhs, values.get(rhs)),
        )

    return validate
