# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU lower-dispatch ownership checks."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, unique

from loom.dsl import Op
from loom.target.contract_fragments import (
    ContractFragmentRegistration,
    iter_contract_fragment_registrations,
)
from loom.target.contracts import LOWER_RULE_FLAG_CONTRACT_ONLY, compile_lower_rule_set


@unique
class DispatchRowRole(Enum):
    """Ownership class declared by an AMDGPU dispatch row macro."""

    STRUCTURAL = "structural"
    VALUE = "value"
    MEMORY = "memory"
    RECIPE = "recipe"
    GENERATED_PRESELECT = "generated_preselect"
    LEGALITY = "legality"


@dataclass(frozen=True, slots=True)
class DispatchRow:
    """Parsed dispatch row from the AMDGPU registry table."""

    op_kind: str
    role: DispatchRowRole
    macro_name: str
    arguments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RowMacroSignature:
    argument_count: int
    storage_policy_argument: int | None = None
    source_count_argument: int | None = None
    preselect_policy_argument: int | None = None
    report_key_argument: int | None = None
    capability_argument: int | None = None


_ROW_RE = re.compile(
    r"\[LOOM_AMDGPU_OP_INDEX\((LOOM_OP_[A-Z0-9_]+)\)\]\s*=\s*"
    r"LOOM_AMDGPU_([A-Z0-9_]+ROW)\("
)

_ROW_MACRO_SIGNATURES = {
    "STRUCTURAL_DIRECT_STORAGE_ROW": _RowMacroSignature(
        argument_count=5, storage_policy_argument=4
    ),
    "STRUCTURAL_DATA_STORAGE_REPORT_KEY_ROW": _RowMacroSignature(
        argument_count=7, storage_policy_argument=5, report_key_argument=6
    ),
    "VALUE_STRUCTURAL_DIRECT_STORAGE_ROW": _RowMacroSignature(
        argument_count=5, storage_policy_argument=4
    ),
    "VALUE_STRUCTURAL_DIRECT_POLICY_ROW": _RowMacroSignature(
        argument_count=6, storage_policy_argument=4, preselect_policy_argument=5
    ),
    "VALUE_STRUCTURAL_DATA_STORAGE_ROW": _RowMacroSignature(
        argument_count=6, storage_policy_argument=5
    ),
    "VALUE_DATA_STORAGE_ROW": _RowMacroSignature(
        argument_count=6, storage_policy_argument=5
    ),
    "VALUE_DATA_STORAGE_REPORT_KEY_ROW": _RowMacroSignature(
        argument_count=7, storage_policy_argument=5, report_key_argument=6
    ),
    "VALUE_DATA_SOURCE_ROW": _RowMacroSignature(
        argument_count=6, source_count_argument=5
    ),
    "VALUE_DATA_SOURCE_POLICY_ROW": _RowMacroSignature(
        argument_count=7, source_count_argument=5, preselect_policy_argument=6
    ),
    "MEMORY_DATA_STORAGE_ROW": _RowMacroSignature(
        argument_count=6, storage_policy_argument=5
    ),
    "MEMORY_DATA_STORAGE_REPORT_KEY_ROW": _RowMacroSignature(
        argument_count=7, storage_policy_argument=5, report_key_argument=6
    ),
    "RECIPE_DIRECT_STORAGE_ROW": _RowMacroSignature(
        argument_count=5, storage_policy_argument=4
    ),
    "RECIPE_DATA_ROW": _RowMacroSignature(argument_count=5),
    "RECIPE_DATA_STORAGE_ROW": _RowMacroSignature(
        argument_count=6, storage_policy_argument=5
    ),
    "RECIPE_DATA_STORAGE_REPORT_KEY_ROW": _RowMacroSignature(
        argument_count=7, storage_policy_argument=5, report_key_argument=6
    ),
    "RECIPE_CAPABILITY_DATA_STORAGE_ROW": _RowMacroSignature(
        argument_count=7,
        storage_policy_argument=5,
        capability_argument=6,
    ),
    "RECIPE_CAPABILITY_DATA_STORAGE_REPORT_KEY_ROW": _RowMacroSignature(
        argument_count=8,
        storage_policy_argument=5,
        report_key_argument=6,
        capability_argument=7,
    ),
    "RECIPE_DATA_SOURCE_ROW": _RowMacroSignature(
        argument_count=6, source_count_argument=5
    ),
    "RECIPE_DATA_SOURCE_REPORT_KEY_ROW": _RowMacroSignature(
        argument_count=7, source_count_argument=5, report_key_argument=6
    ),
    "GENERATED_PRESELECT_DIRECT_POLICY_ROW": _RowMacroSignature(
        argument_count=6, storage_policy_argument=4, preselect_policy_argument=5
    ),
    "GENERATED_PRESELECT_DATA_POLICY_ROW": _RowMacroSignature(
        argument_count=7, storage_policy_argument=5, preselect_policy_argument=6
    ),
    "GENERATED_PRESELECT_DATA_SOURCE_POLICY_ROW": _RowMacroSignature(
        argument_count=7, source_count_argument=5, preselect_policy_argument=6
    ),
    "LEGALITY_ROW": _RowMacroSignature(argument_count=2),
}

_STORAGE_POLICY_NAMES = frozenset(
    {
        "LOOM_AMDGPU_STORAGE_VALUE_PLAN",
        "LOOM_AMDGPU_STORAGE_VECTOR_REGISTER_MAP_PLAN",
        "LOOM_AMDGPU_STORAGE_MEMORY_PLAN",
        "LOOM_AMDGPU_STORAGE_ATOMIC",
        "LOOM_AMDGPU_STORAGE_PREFETCH",
        "LOOM_AMDGPU_STORAGE_FRAGMENT_MEMORY",
        "LOOM_AMDGPU_STORAGE_SUBGROUP_BROADCAST",
        "LOOM_AMDGPU_STORAGE_SUBGROUP_SHUFFLE",
        "LOOM_AMDGPU_STORAGE_NONE",
        "LOOM_AMDGPU_STORAGE_ASYNC_GATHER",
        "LOOM_AMDGPU_STORAGE_ASYNC_CLUSTER",
        "LOOM_AMDGPU_STORAGE_SANITIZER_ACCESS",
        "LOOM_AMDGPU_STORAGE_VECTOR_CONSTRUCT_PLAN",
        "LOOM_AMDGPU_STORAGE_ASYNC_TENSOR",
    }
)

_LOWER_CAPABILITY_RE = re.compile(r"LOOM_AMDGPU_LOWER_CAPABILITY_[A-Z0-9_]+\Z")

_PRESELECT_POLICY_NAMES = frozenset(
    {
        "LOOM_AMDGPU_PRESELECT_VECTOR_CONSTRUCT_PLAN",
        "LOOM_AMDGPU_PRESELECT_TARGET_PLAN",
        "LOOM_AMDGPU_PRESELECT_TARGET_PLAN_FMA_DIAGNOSTIC",
    }
)

_REPORT_KEY_NAMES = frozenset(
    {
        "LOOM_AMDGPU_REPORT_KEY_FRAGMENT_REPACK_STRATEGY",
        "LOOM_AMDGPU_REPORT_KEY_FRAGMENT_MEMORY_STRATEGY",
        "LOOM_AMDGPU_REPORT_KEY_TABLE_LOOKUP_STRATEGY",
        "LOOM_AMDGPU_REPORT_KEY_KERNEL_BARRIER_STRATEGY",
        "LOOM_AMDGPU_REPORT_KEY_SUBGROUP_REDUCE_STRATEGY",
        "LOOM_AMDGPU_REPORT_KEY_SUBGROUP_BROADCAST_STRATEGY",
        "LOOM_AMDGPU_REPORT_KEY_VECTOR_16BIT_FLOAT_CONVERSION_STRATEGY",
        "LOOM_AMDGPU_REPORT_KEY_VECTOR_TRANSFORM_STRATEGY",
        "LOOM_AMDGPU_REPORT_KEY_WORKGROUP_REDUCE_PUBLICATION",
        "LOOM_AMDGPU_REPORT_KEY_TENSOR_MEMORY_PACKET",
    }
)

_ROW_TAG_NAMES_BY_KIND = {
    "storage": _STORAGE_POLICY_NAMES,
    "preselect": _PRESELECT_POLICY_NAMES,
    "report_key": _REPORT_KEY_NAMES,
}

_ROW_TAG_DESCRIPTIONS = {
    "storage": "storage policy",
    "preselect": "preselect policy",
    "report_key": "report key",
}

_ROW_TAG_PREFIXES = (
    "LOOM_AMDGPU_STORAGE_",
    "LOOM_AMDGPU_PRESELECT_",
    "LOOM_AMDGPU_REPORT_KEY_",
)
_SOURCE_COUNT_NAMES = frozenset({"1", "2", "3"})


def amdgpu_generated_lower_rule_op_kinds(
    *,
    registrations: Iterable[ContractFragmentRegistration] | None = None,
) -> frozenset[str]:
    """Returns C op-kind names owned by generated AMDGPU lower-rule rows."""

    op_kinds: set[str] = set()
    if registrations is None:
        registrations = iter_contract_fragment_registrations()
    for registration in registrations:
        if not registration.key.startswith("amdgpu."):
            continue
        fragment = registration.load()
        compiled = compile_lower_rule_set(
            fragment, dialect_ops=registration.load_dialect_ops()
        )
        for rule in compiled.rules:
            if rule.flags & LOWER_RULE_FLAG_CONTRACT_ONLY:
                continue
            op_kinds.add(_op_kind_c_name(rule.source_op))
    return frozenset(op_kinds)


def parse_dispatch_rows(source: str) -> tuple[DispatchRow, ...]:
    """Parses classified AMDGPU dispatch rows from registry source."""

    rows: list[DispatchRow] = []
    for match in _ROW_RE.finditer(source):
        macro_name = match.group(2)
        arguments = _parse_dispatch_row_arguments(source, match.end())
        rows.append(
            DispatchRow(
                op_kind=match.group(1),
                role=_dispatch_row_role(macro_name),
                macro_name=macro_name,
                arguments=arguments,
            )
        )
    return tuple(rows)


def dispatch_row_tag_names_by_kind() -> dict[str, frozenset[str]]:
    """Returns explicit row-tag tokens accepted by dispatch rows."""

    return dict(_ROW_TAG_NAMES_BY_KIND)


def dispatch_row_macro_names() -> frozenset[str]:
    """Returns public dispatch row macros accepted by the validator."""

    return frozenset(_ROW_MACRO_SIGNATURES)


def validate_dispatch_rows(
    rows: Iterable[DispatchRow],
    *,
    generated_lower_rule_op_kinds: Iterable[str],
) -> None:
    """Validates dispatch rows against Python-authored generated ownership."""

    rows = tuple(rows)
    for row in rows:
        _validate_dispatch_row_shape(row)

    generated_op_kinds = frozenset(generated_lower_rule_op_kinds)
    bad_generated_rows = tuple(
        row
        for row in rows
        if row.op_kind in generated_op_kinds
        and row.role
        not in (
            DispatchRowRole.GENERATED_PRESELECT,
            DispatchRowRole.LEGALITY,
        )
    )
    if bad_generated_rows:
        raise ValueError(
            "AMDGPU dispatch rows for generated lower-rule ops must be "
            "generated-preselect or legality-only rows: "
            + _format_rows(bad_generated_rows)
        )

    bad_preselect_rows = tuple(
        row
        for row in rows
        if row.role == DispatchRowRole.GENERATED_PRESELECT
        and row.op_kind not in generated_op_kinds
    )
    if bad_preselect_rows:
        raise ValueError(
            "AMDGPU generated-preselect dispatch rows need a generated lower-rule "
            "fallback owner: " + _format_rows(bad_preselect_rows)
        )


def _dispatch_row_role(macro_name: str) -> DispatchRowRole:
    if macro_name == "LEGALITY_ROW":
        return DispatchRowRole.LEGALITY
    if macro_name.startswith("STRUCTURAL_"):
        return DispatchRowRole.STRUCTURAL
    if macro_name.startswith("GENERATED_PRESELECT_"):
        return DispatchRowRole.GENERATED_PRESELECT
    if macro_name.startswith("VALUE_"):
        return DispatchRowRole.VALUE
    if macro_name.startswith("MEMORY_"):
        return DispatchRowRole.MEMORY
    if macro_name.startswith("RECIPE_"):
        return DispatchRowRole.RECIPE
    raise ValueError(f"AMDGPU dispatch row macro '{macro_name}' has no role prefix")


def _validate_dispatch_row_shape(row: DispatchRow) -> None:
    signature = _ROW_MACRO_SIGNATURES.get(row.macro_name)
    if signature is None:
        raise ValueError(
            f"AMDGPU dispatch row macro '{row.macro_name}' has no registered "
            "dispatch-row schema"
        )
    if len(row.arguments) != signature.argument_count:
        raise ValueError(
            f"AMDGPU dispatch row {row.op_kind} via {row.macro_name} has "
            f"{len(row.arguments)} arguments, expected {signature.argument_count}"
        )
    if row.arguments[0] != row.op_kind:
        raise ValueError(
            f"AMDGPU dispatch row index {row.op_kind} does not match macro "
            f"op-kind argument {row.arguments[0]}"
        )

    expected_row_tag_arguments: dict[int, str] = {}
    if signature.storage_policy_argument is not None:
        expected_row_tag_arguments[signature.storage_policy_argument] = "storage"
    if signature.preselect_policy_argument is not None:
        expected_row_tag_arguments[signature.preselect_policy_argument] = "preselect"
    if signature.report_key_argument is not None:
        expected_row_tag_arguments[signature.report_key_argument] = "report_key"
    if (
        signature.source_count_argument is not None
        and row.arguments[signature.source_count_argument] not in _SOURCE_COUNT_NAMES
    ):
        raise ValueError(
            f"AMDGPU dispatch row {row.op_kind} via {row.macro_name} expects "
            f"a leading source count at argument "
            f"{signature.source_count_argument + 1}, got "
            f"{row.arguments[signature.source_count_argument]}"
        )
    if signature.capability_argument is not None:
        capability = row.arguments[signature.capability_argument]
        if _LOWER_CAPABILITY_RE.fullmatch(capability) is None:
            raise ValueError(
                f"AMDGPU dispatch row {row.op_kind} via {row.macro_name} "
                f"expects a generated lower capability at argument "
                f"{signature.capability_argument + 1}, got {capability}"
            )

    for argument_index, row_tag_kind in expected_row_tag_arguments.items():
        argument = row.arguments[argument_index]
        if argument not in _ROW_TAG_NAMES_BY_KIND[row_tag_kind]:
            raise ValueError(
                f"AMDGPU dispatch row {row.op_kind} via {row.macro_name} "
                f"expects a {_ROW_TAG_DESCRIPTIONS[row_tag_kind]} at argument "
                f"{argument_index + 1}, got {argument}"
            )

    for argument_index, argument in enumerate(row.arguments):
        if not argument.startswith(_ROW_TAG_PREFIXES):
            continue
        if argument_index in expected_row_tag_arguments:
            continue
        raise ValueError(
            f"AMDGPU dispatch row {row.op_kind} via {row.macro_name} has row tag "
            f"token {argument} in non-tag argument {argument_index + 1}"
        )


def _parse_dispatch_row_arguments(source: str, argument_start: int) -> tuple[str, ...]:
    arguments: list[str] = []
    depth = 0
    current_start = argument_start
    index = argument_start
    while index < len(source):
        char = source[index]
        if char in ("'", '"'):
            index = _skip_c_quoted_literal(source, index)
            continue
        if char == "/" and index + 1 < len(source):
            next_char = source[index + 1]
            if next_char == "/":
                newline_index = source.find("\n", index + 2)
                index = len(source) if newline_index == -1 else newline_index + 1
                continue
            if next_char == "*":
                comment_end = source.find("*/", index + 2)
                if comment_end == -1:
                    raise ValueError("unterminated C block comment in dispatch row")
                index = comment_end + 2
                continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                if char != ")":
                    raise ValueError(
                        f"unexpected '{char}' while parsing dispatch row arguments"
                    )
                arguments.append(source[current_start:index].strip())
                return tuple(arguments)
            depth -= 1
        elif char == "," and depth == 0:
            arguments.append(source[current_start:index].strip())
            current_start = index + 1
        index += 1
    raise ValueError("unterminated AMDGPU dispatch row macro invocation")


def _skip_c_quoted_literal(source: str, quote_index: int) -> int:
    quote = source[quote_index]
    index = quote_index + 1
    while index < len(source):
        if source[index] == "\\":
            index += 2
            continue
        if source[index] == quote:
            return index + 1
        index += 1
    raise ValueError("unterminated C quoted literal in dispatch row")


def _format_rows(rows: Iterable[DispatchRow]) -> str:
    return ", ".join(f"{row.op_kind} via {row.macro_name}" for row in rows)


def _op_kind_c_name(op: Op) -> str:
    return "LOOM_OP_" + op.name.replace(".", "_").upper()
