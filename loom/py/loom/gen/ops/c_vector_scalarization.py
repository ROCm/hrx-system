# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: vector/scalar op relations -> C scalarization table rows."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dsl import ATTR_TYPE_FLAGS, Op
from loom.gen.ops.c_names import c_enum_name
from loom.gen.support.files import write_text_file
from loom.gen.support.generated_file import line_comment_header

_GENERATOR = "loom.gen.ops.c_vector_scalarization"
_NO_SEED_OPERAND = 0xFF

# Elementwise vector ops without a scalar-dialect counterpart. These are
# handled by explicit vector-to-scalar lane programs.
_SCALAR_COUNTERPART_EXCLUSIONS = frozenset(
    {
        "vector.bitfield.insert",
        "vector.select",
    }
)

# These ops have a semantic scalar counterpart, but the reference lowering
# intentionally expands their behavior into primitive scalar operations.
_MECHANICAL_LOWERING_EXCLUSIONS = frozenset(
    {
        "vector.bitfield.extracts",
        "vector.bitfield.extractu",
    }
)

# Accumulator-style operations prefer their addend as the dynamic aggregate
# seed. Other multi-operand elementwise ops use their first result-compatible
# operand, while unary ops deliberately use no source seed so producer chains
# can be rematerialized into one loop.
_PREFERRED_SEED_OPERANDS = {
    "vector.fmaf": "c",
    "vector.fmai": "c",
}


@dataclass(frozen=True)
class VectorScalarizationRow:
    """Validated scalar lane relation for one vector operation."""

    vector_op: Op
    scalar_op: Op
    mechanical: bool
    seed_operand_index: int


def _has_trait(op: Op, name: str) -> bool:
    return any(trait.name == name for trait in op.traits)


def _op_suffix(op: Op) -> str:
    return op.name.split(".", 1)[1]


def _non_flag_attr_signature(op: Op) -> tuple[tuple[object, ...], ...]:
    return tuple((attr.name, attr.attr_type, attr.optional, attr.enum_def) for attr in op.attrs if attr.attr_type != ATTR_TYPE_FLAGS)


def _flag_attr_signature(op: Op) -> tuple[tuple[object, ...], ...]:
    return tuple((attr.name, attr.enum_def) for attr in op.attrs if attr.attr_type == ATTR_TYPE_FLAGS)


def _same_type_field_names(op: Op, result_name: str) -> frozenset[str]:
    """Returns fields transitively constrained to the named result type."""

    edges: dict[str, set[str]] = {}
    groups = [constraint.args for constraint in op.constraints if constraint.name == "SameType"]
    groups.extend(trait.args for trait in op.traits if trait.name == "AllTypesMatch")
    for group in groups:
        for field_name in group:
            edges.setdefault(field_name, set()).update(group)

    reachable = {result_name}
    worklist = [result_name]
    while worklist:
        field_name = worklist.pop()
        for neighbor in edges.get(field_name, ()):
            if neighbor in reachable:
                continue
            reachable.add(neighbor)
            worklist.append(neighbor)
    return frozenset(reachable)


def _seed_operand_index(op: Op) -> int:
    if len(op.results) != 1 or len(op.operands) < 2:
        return _NO_SEED_OPERAND
    same_type_fields = _same_type_field_names(op, op.results[0].name)
    preferred_operand_name = _PREFERRED_SEED_OPERANDS.get(op.name)
    if preferred_operand_name is not None:
        for index, operand in enumerate(op.operands):
            if operand.name != preferred_operand_name:
                continue
            if operand.name not in same_type_fields:
                raise ValueError(f"{op.name}: preferred seed operand {preferred_operand_name!r} does not match the result type")
            return index
        raise ValueError(f"{op.name}: preferred seed operand {preferred_operand_name!r} is missing")
    for index, operand in enumerate(op.operands):
        if operand.name in same_type_fields:
            return index
    return _NO_SEED_OPERAND


def _validate_pair(vector_op: Op, scalar_op: Op) -> None:
    if len(vector_op.results) != 1 or len(scalar_op.results) != 1:
        raise ValueError(f"{vector_op.name}: scalarization requires one vector and scalar result")
    if any(operand.variadic or operand.optional for op in (vector_op, scalar_op) for operand in op.operands):
        raise ValueError(f"{vector_op.name}: mechanical scalarization requires fixed operands")
    if any(getattr(result, "variadic", False) for op in (vector_op, scalar_op) for result in op.results):
        raise ValueError(f"{vector_op.name}: mechanical scalarization requires fixed results")
    if len(vector_op.operands) > 4:
        raise ValueError(f"{vector_op.name}: mechanical scalarization supports at most four operands")

    vector_operand_names = tuple(operand.name for operand in vector_op.operands)
    scalar_operand_names = tuple(operand.name for operand in scalar_op.operands)
    if vector_operand_names != scalar_operand_names:
        raise ValueError(f"{vector_op.name}: vector operands {vector_operand_names} do not match {scalar_op.name} operands {scalar_operand_names}")
    vector_result_names = tuple(result.name for result in vector_op.results)
    scalar_result_names = tuple(result.name for result in scalar_op.results)
    if vector_result_names != scalar_result_names:
        raise ValueError(f"{vector_op.name}: vector results {vector_result_names} do not match {scalar_op.name} results {scalar_result_names}")
    vector_attrs = _non_flag_attr_signature(vector_op)
    scalar_attrs = _non_flag_attr_signature(scalar_op)
    if vector_attrs != scalar_attrs:
        raise ValueError(f"{vector_op.name}: vector attributes do not match {scalar_op.name}")
    vector_flags = _flag_attr_signature(vector_op)
    scalar_flags = _flag_attr_signature(scalar_op)
    if vector_flags != scalar_flags:
        raise ValueError(f"{vector_op.name}: vector instance flags do not match {scalar_op.name}")


def collect_vector_scalarization_rows(
    vector_ops: Sequence[Op] = ALL_VECTOR_OPS,
    scalar_ops: Sequence[Op] = ALL_SCALAR_OPS,
) -> tuple[VectorScalarizationRow, ...]:
    """Collects and validates vector operations with scalar counterparts."""

    scalar_ops_by_suffix = {_op_suffix(op): op for op in scalar_ops}
    vector_op_names = {op.name for op in vector_ops}
    stale_counterpart_exclusions = sorted(_SCALAR_COUNTERPART_EXCLUSIONS - vector_op_names)
    stale_mechanical_exclusions = sorted(_MECHANICAL_LOWERING_EXCLUSIONS - vector_op_names)
    stale_seed_preferences = sorted(set(_PREFERRED_SEED_OPERANDS) - vector_op_names)
    if stale_counterpart_exclusions or stale_mechanical_exclusions or stale_seed_preferences:
        raise ValueError(f"stale vector scalarization exclusions: {stale_counterpart_exclusions + stale_mechanical_exclusions}; stale seed preferences: {stale_seed_preferences}")

    rows: list[VectorScalarizationRow] = []
    missing_counterparts: list[str] = []
    for vector_op in vector_ops:
        if not _has_trait(vector_op, "Elementwise"):
            continue
        scalar_op = scalar_ops_by_suffix.get(_op_suffix(vector_op))
        if scalar_op is None:
            if vector_op.name not in _SCALAR_COUNTERPART_EXCLUSIONS:
                missing_counterparts.append(vector_op.name)
            continue
        if vector_op.name in _SCALAR_COUNTERPART_EXCLUSIONS:
            raise ValueError(f"{vector_op.name}: scalar counterpart exclusion is stale")
        _validate_pair(vector_op, scalar_op)
        rows.append(
            VectorScalarizationRow(
                vector_op=vector_op,
                scalar_op=scalar_op,
                mechanical=vector_op.name not in _MECHANICAL_LOWERING_EXCLUSIONS,
                seed_operand_index=_seed_operand_index(vector_op),
            )
        )

    if missing_counterparts:
        raise ValueError(f"elementwise vector ops missing scalar counterparts: {sorted(missing_counterparts)}")
    return tuple(rows)


def generate_vector_scalarization_rows() -> str:
    lines = [
        *line_comment_header("//", generator=_GENERATOR),
        "// clang-format off",
        "",
    ]
    for row in collect_vector_scalarization_rows():
        flags = "LOOM_VECTOR_SCALARIZATION_FLAG_MECHANICAL" if row.mechanical else "LOOM_VECTOR_SCALARIZATION_FLAG_NONE"
        seed_operand_index = "UINT8_MAX" if row.seed_operand_index == _NO_SEED_OPERAND else str(row.seed_operand_index)
        lines.append(f"LOOM_VECTOR_SCALARIZATION_ROW({c_enum_name(row.vector_op)}, {c_enum_name(row.scalar_op)}, {flags}, {seed_operand_index})")
    lines.extend(["", "// clang-format on", ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generates vector scalarization relation rows.")
    parser.add_argument("--rows", type=Path, help="Generated C row include.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate scalarization relations without writing output.",
    )
    args = parser.parse_args(argv)
    if args.check == (args.rows is not None):
        parser.error("select exactly one of --check or --rows")
    contents = generate_vector_scalarization_rows()
    if args.rows is not None:
        write_text_file(args.rows, contents)
    return 0


if __name__ == "__main__":
    sys.exit(main())
