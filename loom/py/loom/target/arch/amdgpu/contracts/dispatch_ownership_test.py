# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for AMDGPU lower-dispatch ownership guardrails."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from loom.target.arch.amdgpu.contracts.dispatch_ownership import (
    DispatchRow,
    DispatchRowRole,
    amdgpu_generated_lower_rule_op_kinds,
    dispatch_row_macro_names,
    dispatch_row_tag_names_by_kind,
    parse_dispatch_rows,
    validate_dispatch_rows,
)

_REGISTRY_SOURCE_PATH = Path("loom/src/loom/target/arch/amdgpu/lower/registry.c")
_REGISTRY_TABLE_PATH = Path(
    "loom/src/loom/target/arch/amdgpu/lower/registry_tables.inl"
)

_ROW_TAG_ENUMS_BY_KIND = {
    "storage": re.compile(
        r"enum loom_amdgpu_storage_policy_e\s*\{(?P<body>.*?)\};", re.DOTALL
    ),
    "preselect": re.compile(
        r"enum loom_amdgpu_preselect_policy_e\s*\{(?P<body>.*?)\};", re.DOTALL
    ),
    "report_key": re.compile(
        r"enum loom_amdgpu_report_key_kind_e\s*\{(?P<body>.*?)\};", re.DOTALL
    ),
}
_ROW_TAG_NAME_RE = re.compile(
    r"\b(LOOM_AMDGPU_(?:STORAGE|PRESELECT|REPORT_KEY)_[A-Z0-9_]+)\b"
)
_PUBLIC_ROW_MACRO_RE = re.compile(r"#define LOOM_AMDGPU_([A-Z0-9_]+ROW)\b")
_ROW_TAG_SENTINELS_BY_KIND = {
    "storage": frozenset(
        {
            "LOOM_AMDGPU_STORAGE_SOURCE_OPERANDS",
            "LOOM_AMDGPU_STORAGE_PLAN_LEADING_SOURCES",
            "LOOM_AMDGPU_STORAGE_MAX",
        }
    ),
    "preselect": frozenset({"LOOM_AMDGPU_PRESELECT_NONE", "LOOM_AMDGPU_PRESELECT_MAX"}),
    "report_key": frozenset(
        {"LOOM_AMDGPU_REPORT_KEY_NONE", "LOOM_AMDGPU_REPORT_KEY_MAX"}
    ),
}


def test_parse_dispatch_rows_requires_role_prefix() -> None:
    source = """
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_BITPACK)] =
            LOOM_AMDGPU_DATA_ROW(LOOM_OP_VECTOR_BITPACK, plan_t, select, emit, NULL),
    """

    with pytest.raises(ValueError, match="has no role prefix"):
        parse_dispatch_rows(source)


def test_parse_dispatch_rows_captures_arguments() -> None:
    source = """
        [LOOM_AMDGPU_OP_INDEX(LOOM_OP_VECTOR_FMAF)] =
            LOOM_AMDGPU_GENERATED_PRESELECT_DATA_SOURCE_POLICY_ROW(
                LOOM_OP_VECTOR_FMAF, loom_amdgpu_packed_ternary_plan_t,
                loom_amdgpu_select_vector_packed_fmaf_dispatch,
                loom_amdgpu_emit_vector_packed_ternary_dispatch, NULL,
                3,
                LOOM_AMDGPU_PRESELECT_TARGET_PLAN_FMA_DIAGNOSTIC),
    """

    rows = parse_dispatch_rows(source)

    assert rows == (
        DispatchRow(
            op_kind="LOOM_OP_VECTOR_FMAF",
            role=DispatchRowRole.GENERATED_PRESELECT,
            macro_name="GENERATED_PRESELECT_DATA_SOURCE_POLICY_ROW",
            arguments=(
                "LOOM_OP_VECTOR_FMAF",
                "loom_amdgpu_packed_ternary_plan_t",
                "loom_amdgpu_select_vector_packed_fmaf_dispatch",
                "loom_amdgpu_emit_vector_packed_ternary_dispatch",
                "NULL",
                "3",
                "LOOM_AMDGPU_PRESELECT_TARGET_PLAN_FMA_DIAGNOSTIC",
            ),
        ),
    )


def test_validate_dispatch_rows_rejects_unknown_schema() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_VECTOR_BITPACK",
            role=DispatchRowRole.VALUE,
            macro_name="VALUE_UNKNOWN_ROW",
            arguments=("LOOM_OP_VECTOR_BITPACK",),
        ),
    )

    with pytest.raises(ValueError, match="no registered dispatch-row schema"):
        validate_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_validate_dispatch_rows_rejects_op_kind_mismatch() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_VECTOR_BITPACK",
            role=DispatchRowRole.RECIPE,
            macro_name="RECIPE_DATA_ROW",
            arguments=(
                "LOOM_OP_VECTOR_BITUNPACKU",
                "loom_amdgpu_bitpack_plan_t",
                "select",
                "emit",
                "verify",
            ),
        ),
    )

    with pytest.raises(ValueError, match="does not match macro op-kind argument"):
        validate_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_validate_dispatch_rows_rejects_wrong_policy_namespace() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_VECTOR_FMAF",
            role=DispatchRowRole.GENERATED_PRESELECT,
            macro_name="GENERATED_PRESELECT_DATA_POLICY_ROW",
            arguments=(
                "LOOM_OP_VECTOR_FMAF",
                "loom_amdgpu_packed_ternary_plan_t",
                "select",
                "emit",
                "verify",
                "LOOM_AMDGPU_PRESELECT_TARGET_PLAN",
                "LOOM_AMDGPU_PRESELECT_TARGET_PLAN_FMA_DIAGNOSTIC",
            ),
        ),
    )

    with pytest.raises(ValueError, match="expects a storage policy"):
        validate_dispatch_rows(
            rows, generated_lower_rule_op_kinds={"LOOM_OP_VECTOR_FMAF"}
        )


def test_validate_dispatch_rows_rejects_wrong_report_key_namespace() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_KERNEL_WORKGROUP_REDUCE",
            role=DispatchRowRole.RECIPE,
            macro_name="RECIPE_DATA_SOURCE_REPORT_KEY_ROW",
            arguments=(
                "LOOM_OP_KERNEL_WORKGROUP_REDUCE",
                "loom_amdgpu_workgroup_reduce_plan_t",
                "select",
                "emit",
                "verify",
                "1",
                "LOOM_AMDGPU_STORAGE_VALUE_PLAN",
            ),
        ),
    )

    with pytest.raises(ValueError, match="expects a report key"):
        validate_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_validate_dispatch_rows_accepts_report_key() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_KERNEL_WORKGROUP_REDUCE",
            role=DispatchRowRole.RECIPE,
            macro_name="RECIPE_DATA_SOURCE_REPORT_KEY_ROW",
            arguments=(
                "LOOM_OP_KERNEL_WORKGROUP_REDUCE",
                "loom_amdgpu_workgroup_reduce_plan_t",
                "select",
                "emit",
                "verify",
                "1",
                "LOOM_AMDGPU_REPORT_KEY_WORKGROUP_REDUCE_PUBLICATION",
            ),
        ),
    )

    validate_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_validate_dispatch_rows_accepts_memory_report_key() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_VECTOR_FRAGMENT_LOAD",
            role=DispatchRowRole.MEMORY,
            macro_name="MEMORY_DATA_STORAGE_REPORT_KEY_ROW",
            arguments=(
                "LOOM_OP_VECTOR_FRAGMENT_LOAD",
                "loom_amdgpu_fragment_memory_plan_t",
                "select",
                "emit",
                "verify",
                "LOOM_AMDGPU_STORAGE_FRAGMENT_MEMORY",
                "LOOM_AMDGPU_REPORT_KEY_FRAGMENT_MEMORY_STRATEGY",
            ),
        ),
    )

    validate_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_validate_dispatch_rows_rejects_bad_source_count() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_VECTOR_BITPACK",
            role=DispatchRowRole.RECIPE,
            macro_name="RECIPE_DATA_SOURCE_ROW",
            arguments=(
                "LOOM_OP_VECTOR_BITPACK",
                "loom_amdgpu_bitpack_plan_t",
                "select",
                "emit",
                "verify",
                "LOOM_AMDGPU_STORAGE_VALUE_PLAN",
            ),
        ),
    )

    with pytest.raises(ValueError, match="expects a leading source count"):
        validate_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_dispatch_row_tag_names_match_registry_enums() -> None:
    registry_source = _read_repo_file(_REGISTRY_SOURCE_PATH)

    assert dispatch_row_tag_names_by_kind() == _parse_registry_row_tag_enums(
        registry_source
    )


def test_dispatch_row_macro_schemas_match_registry_macros() -> None:
    registry_source = _read_repo_file(_REGISTRY_SOURCE_PATH)

    assert dispatch_row_macro_names() == _parse_public_row_macros(registry_source)


def test_validate_dispatch_rows_rejects_policy_in_non_policy_slot() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_VECTOR_CONCAT",
            role=DispatchRowRole.RECIPE,
            macro_name="RECIPE_DATA_ROW",
            arguments=(
                "LOOM_OP_VECTOR_CONCAT",
                "LOOM_AMDGPU_STORAGE_VALUE_PLAN",
                "select",
                "emit",
                "verify",
            ),
        ),
    )

    with pytest.raises(ValueError, match=r"row tag token .* in non-tag argument"):
        validate_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_validate_dispatch_rows_rejects_removed_direct_default_macros() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_VECTOR_CONSTANT",
            role=DispatchRowRole.VALUE,
            macro_name="VALUE_DIRECT_ROW",
            arguments=(
                "LOOM_OP_VECTOR_CONSTANT",
                "select",
                "emit",
                "verify",
            ),
        ),
        DispatchRow(
            op_kind="LOOM_OP_KERNEL_BARRIER",
            role=DispatchRowRole.RECIPE,
            macro_name="RECIPE_DIRECT_ROW",
            arguments=(
                "LOOM_OP_KERNEL_BARRIER",
                "select",
                "emit",
                "verify",
            ),
        ),
    )

    with pytest.raises(ValueError, match="no registered dispatch-row schema"):
        validate_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_validate_dispatch_rows_rejects_generated_recipe() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_VECTOR_BITFIELD_EXTRACTU",
            role=DispatchRowRole.RECIPE,
            macro_name="RECIPE_DATA_ROW",
            arguments=(
                "LOOM_OP_VECTOR_BITFIELD_EXTRACTU",
                "loom_amdgpu_bitfield_extract_plan_t",
                "select",
                "emit",
                "verify",
            ),
        ),
    )

    with pytest.raises(ValueError, match="generated-preselect or legality-only"):
        validate_dispatch_rows(
            rows,
            generated_lower_rule_op_kinds={"LOOM_OP_VECTOR_BITFIELD_EXTRACTU"},
        )


def test_validate_dispatch_rows_accepts_vector_construct_policy() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_VECTOR_INSERT",
            role=DispatchRowRole.VALUE,
            macro_name="VALUE_STRUCTURAL_DIRECT_POLICY_ROW",
            arguments=(
                "LOOM_OP_VECTOR_INSERT",
                "select",
                "emit",
                "verify",
                "LOOM_AMDGPU_STORAGE_VECTOR_CONSTRUCT_PLAN",
                "LOOM_AMDGPU_PRESELECT_VECTOR_CONSTRUCT_PLAN",
            ),
        ),
    )

    validate_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_validate_dispatch_rows_accepts_structural_storage_policy() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_KERNEL_BARRIER",
            role=DispatchRowRole.STRUCTURAL,
            macro_name="STRUCTURAL_DIRECT_STORAGE_ROW",
            arguments=(
                "LOOM_OP_KERNEL_BARRIER",
                "select",
                "emit",
                "verify",
                "LOOM_AMDGPU_STORAGE_NONE",
            ),
        ),
    )

    validate_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_validate_dispatch_rows_accepts_generated_preselect() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_VECTOR_FMAF",
            role=DispatchRowRole.GENERATED_PRESELECT,
            macro_name="GENERATED_PRESELECT_DATA_SOURCE_POLICY_ROW",
            arguments=(
                "LOOM_OP_VECTOR_FMAF",
                "loom_amdgpu_packed_ternary_plan_t",
                "select",
                "emit",
                "verify",
                "3",
                "LOOM_AMDGPU_PRESELECT_TARGET_PLAN",
            ),
        ),
    )

    validate_dispatch_rows(
        rows,
        generated_lower_rule_op_kinds={"LOOM_OP_VECTOR_FMAF"},
    )


def test_validate_dispatch_rows_accepts_generated_direct_preselect() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_INDEX_ADD",
            role=DispatchRowRole.GENERATED_PRESELECT,
            macro_name="GENERATED_PRESELECT_DIRECT_POLICY_ROW",
            arguments=(
                "LOOM_OP_INDEX_ADD",
                "select",
                "emit",
                "verify",
                "LOOM_AMDGPU_STORAGE_VALUE_PLAN",
                "LOOM_AMDGPU_PRESELECT_TARGET_PLAN",
            ),
        ),
    )

    validate_dispatch_rows(
        rows,
        generated_lower_rule_op_kinds={"LOOM_OP_INDEX_ADD"},
    )


def test_validate_dispatch_rows_accepts_bounded_recipe() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_VECTOR_BITPACK",
            role=DispatchRowRole.RECIPE,
            macro_name="RECIPE_DATA_ROW",
            arguments=(
                "LOOM_OP_VECTOR_BITPACK",
                "loom_amdgpu_bitpack_plan_t",
                "select",
                "emit",
                "verify",
            ),
        ),
    )

    validate_dispatch_rows(
        rows,
        generated_lower_rule_op_kinds={"LOOM_OP_VECTOR_FMAF"},
    )


def test_validate_dispatch_rows_accepts_recipe_data_storage_policy() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_SANITIZER_ASSERT_ACCESS",
            role=DispatchRowRole.RECIPE,
            macro_name="RECIPE_DATA_STORAGE_ROW",
            arguments=(
                "LOOM_OP_SANITIZER_ASSERT_ACCESS",
                "loom_amdgpu_sanitizer_access_plan_t",
                "select",
                "emit",
                "verify",
                "LOOM_AMDGPU_STORAGE_SANITIZER_ACCESS",
            ),
        ),
    )

    validate_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_validate_dispatch_rows_accepts_value_data_source_policy() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_SCALAR_TRUNCI",
            role=DispatchRowRole.VALUE,
            macro_name="VALUE_DATA_SOURCE_POLICY_ROW",
            arguments=(
                "LOOM_OP_SCALAR_TRUNCI",
                "loom_amdgpu_scalar_conversion_plan_t",
                "select",
                "emit",
                "verify",
                "1",
                "LOOM_AMDGPU_PRESELECT_TARGET_PLAN",
            ),
        ),
    )

    validate_dispatch_rows(rows, generated_lower_rule_op_kinds=())


def test_validate_dispatch_rows_rejects_unowned_preselect() -> None:
    rows = (
        DispatchRow(
            op_kind="LOOM_OP_VECTOR_BITPACK",
            role=DispatchRowRole.GENERATED_PRESELECT,
            macro_name="GENERATED_PRESELECT_DATA_SOURCE_POLICY_ROW",
            arguments=(
                "LOOM_OP_VECTOR_BITPACK",
                "loom_amdgpu_bitpack_plan_t",
                "select",
                "emit",
                "verify",
                "3",
                "LOOM_AMDGPU_PRESELECT_TARGET_PLAN",
            ),
        ),
    )

    with pytest.raises(ValueError, match="fallback owner"):
        validate_dispatch_rows(
            rows,
            generated_lower_rule_op_kinds={"LOOM_OP_VECTOR_FMAF"},
        )


def test_amdgpu_registry_dispatch_rows_have_generated_ownership() -> None:
    source = _read_repo_file(_REGISTRY_TABLE_PATH)
    rows = parse_dispatch_rows(source)

    validate_dispatch_rows(
        rows,
        generated_lower_rule_op_kinds=amdgpu_generated_lower_rule_op_kinds(),
    )


def _read_repo_file(path: Path) -> str:
    roots = (
        Path.cwd(),
        *Path.cwd().parents,
        Path(__file__).resolve(),
        *Path(__file__).resolve().parents,
    )
    for root in roots:
        candidate = root / path
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(path)


def _parse_registry_row_tag_enums(source: str) -> dict[str, frozenset[str]]:
    row_tag_names_by_kind: dict[str, frozenset[str]] = {}
    for kind, enum_re in _ROW_TAG_ENUMS_BY_KIND.items():
        match = enum_re.search(source)
        assert match is not None
        row_tag_names = frozenset(_ROW_TAG_NAME_RE.findall(match.group("body")))
        row_tag_names_by_kind[kind] = row_tag_names - _ROW_TAG_SENTINELS_BY_KIND[kind]
    return row_tag_names_by_kind


def _parse_public_row_macros(source: str) -> frozenset[str]:
    dispatch_macro_region = source.split(
        '#include "loom/target/arch/amdgpu/lower/registry_tables.inl"', maxsplit=1
    )[0]
    return frozenset(
        macro_name
        for macro_name in _PUBLIC_ROW_MACRO_RE.findall(dispatch_macro_region)
        if not macro_name.startswith("INTERNAL_")
    )
