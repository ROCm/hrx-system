# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.error.target import ERR_TARGET_003
from loom.gen.target.contracts.lower_rule_rows import (
    attr_copy_row,
    descriptor_ref_keys,
    diagnostic_has_implicit_target_context,
    diagnostic_param_row,
    diagnostic_stored_params,
    guard_row,
    source_memory_row,
    value_ref_row,
)
from loom.gen.target.contracts.lower_rules import (
    _validate_c_table_shape,
    generate_lower_rule_set,
)
from loom.target.contracts import (
    CompiledLowerRuleSet,
    ContractFragment,
    DescriptorAccumulatorSeed,
    DescriptorAccumulatorTree,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardKind,
    LowerAttrCopy,
    LowerAttrCopyKind,
    LowerDiagnostic,
    LowerDiagnosticParam,
    LowerEmit,
    LowerEmitKind,
    LowerGuard,
    LowerRule,
    LowerRuleSpan,
    LowerSourceMemory,
    LowerTypePattern,
    LowerValueRef,
    RecipeRule,
    Scalar,
    SourceMemoryAddressBase,
    SourceMemoryAddressCoordinateType,
    SourceMemoryAddressLayout,
    SourceMemoryAddressMaterializer,
    SourceMemoryConstraint,
    SourceMemoryDynamicIndexSource,
    SourceMemoryOperation,
    SourceMemoryRootKind,
    SourceOpProject,
    SourceValueKind,
    ValueProject,
    ValueRef,
    Vector,
    View,
)
from loom.target.contracts.diagnostics import DiagnosticParamKind
from loom.target.low_descriptors import Immediate, ImmediateKind
from loom.target.test.descriptors import (
    TEST_LOW_ADD_F32_DESCRIPTOR,
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_CONST_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
    TEST_LOW_MUL_I32_DESCRIPTOR,
)

_TEST_PUBLIC_HEADER = "loom/target/test/contracts/generated.h"


def _expect_value_error(callable_obj: Callable[[], object], message: str) -> None:
    error: ValueError | None = None
    try:
        callable_obj()
    except ValueError as exc:
        error = exc
    assert error is not None
    assert message in str(error)


def _compiled_lower_rule_set(
    *,
    rules: tuple[LowerRule, ...] = (),
    spans: tuple[LowerRuleSpan, ...] = (),
    type_patterns: tuple[LowerTypePattern, ...] = (),
    value_refs: tuple[LowerValueRef, ...] = (),
    source_memories: tuple[LowerSourceMemory, ...] = (),
    guards: tuple[LowerGuard, ...] = (),
    attr_copies: tuple[LowerAttrCopy, ...] = (),
    emits: tuple[LowerEmit, ...] = (),
) -> CompiledLowerRuleSet:
    return CompiledLowerRuleSet(
        name="test.low.generated_c_shape",
        authored_case_indices=(),
        rules=rules,
        spans=spans,
        type_patterns=type_patterns,
        value_refs=value_refs,
        source_memories=source_memories,
        guards=guards,
        attr_copies=attr_copies,
        tied_results=(),
        emits=emits,
        diagnostics=(),
    )


def _c_shape_contract() -> ContractFragment:
    return ContractFragment(
        name="test.low.generated_c_shape",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        public_header=_TEST_PUBLIC_HEADER,
    )


def _add_f32_flags_descriptor_set():
    descriptor = replace(
        TEST_LOW_ADD_F32_DESCRIPTOR,
        key="test.add.f32.flags",
        immediates=(
            Immediate(
                "fast_math_flags",
                ImmediateKind.UNSIGNED,
                bit_width=7,
                unsigned_max=0x7F,
            ),
        ),
    )
    return descriptor, replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(*TEST_LOW_CORE_DESCRIPTOR_SET.descriptors, descriptor),
    )


def test_validate_c_table_shape_rejects_oversized_table_count() -> None:
    table = _compiled_lower_rule_set(
        type_patterns=(LowerTypePattern(Scalar("i32")),) * 0x10000,
    )

    _expect_value_error(
        lambda: _validate_c_table_shape(table, _c_shape_contract(), ()),
        "lower-rule set 'test.low.generated_c_shape' type-pattern count exceeds uint16_t",
    )


def test_validate_c_table_shape_rejects_oversized_rule_field() -> None:
    table = _compiled_lower_rule_set(
        rules=(
            LowerRule(
                source_op=scalar_arithmetic.scalar_addi,
                temporary_count=0x10000,
                guard_start=0,
                guard_count=0,
                emit_start=0,
                emit_count=0,
            ),
        ),
    )

    _expect_value_error(
        lambda: _validate_c_table_shape(table, _c_shape_contract(), ()),
        "lower-rule set 'test.low.generated_c_shape' rule 0 temporary count exceeds uint16_t",
    )


def test_validate_c_table_shape_rejects_oversized_type_payload() -> None:
    table = _compiled_lower_rule_set(
        type_patterns=(LowerTypePattern(Vector("i32", lanes=2**63)),),
    )

    _expect_value_error(
        lambda: _validate_c_table_shape(table, _c_shape_contract(), ()),
        "lower-rule set 'test.low.generated_c_shape' type-pattern 0 static lanes exceeds int64_t",
    )


def test_validate_c_table_shape_rejects_oversized_view_dimension() -> None:
    table = _compiled_lower_rule_set(
        type_patterns=(LowerTypePattern(View("i32", dims=(2**63, 4))),),
    )

    _expect_value_error(
        lambda: _validate_c_table_shape(table, _c_shape_contract(), ()),
        "lower-rule set 'test.low.generated_c_shape' type-pattern 0 static dim 0 exceeds int64_t",
    )


def test_validate_c_table_shape_rejects_rule_guard_range_oob() -> None:
    table = _compiled_lower_rule_set(
        rules=(
            LowerRule(
                source_op=scalar_arithmetic.scalar_addi,
                temporary_count=0,
                guard_start=1,
                guard_count=1,
                emit_start=0,
                emit_count=0,
            ),
        ),
    )

    _expect_value_error(
        lambda: _validate_c_table_shape(table, _c_shape_contract(), ()),
        "lower-rule set 'test.low.generated_c_shape' rule 0 guard range exceeds guard table",
    )


def test_validate_c_table_shape_rejects_emit_operand_range_oob() -> None:
    table = _compiled_lower_rule_set(
        emits=(
            LowerEmit(
                kind=LowerEmitKind.DESCRIPTOR_OP,
                descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                operand_ref_start=1,
                operand_ref_count=1,
            ),
        ),
    )

    _expect_value_error(
        lambda: _validate_c_table_shape(table, _c_shape_contract(), ()),
        "lower-rule set 'test.low.generated_c_shape' emit 0 operand-ref range exceeds value-ref table",
    )


def test_validate_c_table_shape_rejects_emit_source_memory_ordinal_oob() -> None:
    table = _compiled_lower_rule_set(
        emits=(
            LowerEmit(
                kind=LowerEmitKind.DESCRIPTOR_OP,
                descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                source_memory_ordinal=1,
            ),
        ),
    )

    _expect_value_error(
        lambda: _validate_c_table_shape(table, _c_shape_contract(), ()),
        "lower-rule set 'test.low.generated_c_shape' emit 0 source-memory ordinal references missing source-memory row",
    )


def test_validate_c_table_shape_rejects_span_rule_range_mismatch() -> None:
    table = _compiled_lower_rule_set(
        rules=(
            LowerRule(
                source_op=scalar_arithmetic.scalar_addi,
                temporary_count=0,
                guard_start=0,
                guard_count=0,
                emit_start=0,
                emit_count=0,
            ),
        ),
        spans=(
            LowerRuleSpan(
                source_op=scalar_arithmetic.scalar_mulf,
                rule_start=0,
                rule_count=1,
            ),
        ),
    )

    _expect_value_error(
        lambda: _validate_c_table_shape(table, _c_shape_contract(), ()),
        "span 0 rule range contains rule 0 for source op 'scalar.addi', expected 'scalar.mulf'",
    )


def test_validate_c_table_shape_rejects_guard_type_pattern_index_oob() -> None:
    table = _compiled_lower_rule_set(
        value_refs=(LowerValueRef(kind=SourceValueKind.OPERAND, index=0),),
        guards=(LowerGuard(kind=GuardKind.VALUE_TYPE),),
    )

    _expect_value_error(
        lambda: _validate_c_table_shape(table, _c_shape_contract(), ()),
        "lower-rule set 'test.low.generated_c_shape' guard 0 type-pattern index references missing type-pattern row",
    )


def test_validate_c_table_shape_rejects_guard_value_ref_index_oob() -> None:
    table = _compiled_lower_rule_set(
        guards=(LowerGuard(kind=GuardKind.VALUE_NO_USES),),
    )

    _expect_value_error(
        lambda: _validate_c_table_shape(table, _c_shape_contract(), ()),
        "lower-rule set 'test.low.generated_c_shape' guard 0 value-ref index references missing value-ref row",
    )


def test_validate_c_table_shape_rejects_guard_other_value_ref_index_oob() -> None:
    table = _compiled_lower_rule_set(
        value_refs=(LowerValueRef(kind=SourceValueKind.OPERAND, index=0),),
        guards=(
            LowerGuard(
                kind=GuardKind.VALUE_I64_RANGE_LE,
                value_ref_index=0,
                other_value_ref_index=1,
            ),
        ),
    )

    _expect_value_error(
        lambda: _validate_c_table_shape(table, _c_shape_contract(), ()),
        "lower-rule set 'test.low.generated_c_shape' guard 0 other value-ref index references missing value-ref row",
    )


def test_validate_c_table_shape_rejects_attr_copy_value_ref_index_oob() -> None:
    table = _compiled_lower_rule_set(
        attr_copies=(
            LowerAttrCopy(
                kind=LowerAttrCopyKind.VALUE_EXACT_I64,
                target_name="i32_value",
            ),
        ),
    )

    _expect_value_error(
        lambda: _validate_c_table_shape(table, _c_shape_contract(), ()),
        "lower-rule set 'test.low.generated_c_shape' attr-copy 0 value-ref index references missing value-ref row",
    )


def test_validate_c_table_shape_rejects_value_ref_materializer_index_oob() -> None:
    table = _compiled_lower_rule_set(
        value_refs=(
            LowerValueRef(
                kind=SourceValueKind.OPERAND,
                index=0,
                materializer_index=1,
            ),
        ),
    )

    _expect_value_error(
        lambda: _validate_c_table_shape(table, _c_shape_contract(), ()),
        "lower-rule set 'test.low.generated_c_shape' value-ref 0 materializer index references missing materializer row",
    )


def test_value_ref_row_emits_element_indices() -> None:
    row = value_ref_row(
        LowerValueRef(
            kind=SourceValueKind.OPERAND,
            index=1,
            element_index=2,
        )
    )

    assert row == [
        ".kind = LOOM_LOW_LOWER_VALUE_REF_OPERAND",
        ".index = 1",
        ".element_index = 2",
    ]


def test_generate_lower_rule_set_emits_report_key_ordinals() -> None:
    table = ContractFragment(
        name="test.low.report_keys",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        public_header=_TEST_PUBLIC_HEADER,
        cases=[
            DescriptorRule(
                source_op=scalar_arithmetic.scalar_mulf,
                descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                report_key="test.scalar_mulf.strategy.native",
                guards=(
                    Guard.value_type("lhs", Scalar("f32")),
                    Guard.value_type("rhs", Scalar("f32")),
                    Guard.value_type("result", Scalar("f32")),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                        operands={
                            "lhs": ValueRef.operand("lhs"),
                            "rhs": ValueRef.operand("rhs"),
                        },
                        results={"dst": ValueRef.result("result")},
                    ),
                ),
            )
        ],
    )

    generated = generate_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert 'LOOM_BSTRING_LITERAL(32, "test.scalar_mulf.strategy.native")' in generated.source
    assert "static const loom_bstring_table_offset_t" in generated.source
    assert ".report_key_ordinal = 1," in generated.source
    assert ".report_key_string_offsets = " in generated.source
    assert ".report_key_count = IREE_ARRAYSIZE(" in generated.source


def test_validate_c_table_shape_rejects_invalid_report_key() -> None:
    table = _compiled_lower_rule_set(
        rules=(
            LowerRule(
                source_op=scalar_arithmetic.scalar_addi,
                temporary_count=0,
                guard_start=0,
                guard_count=0,
                emit_start=0,
                emit_count=0,
                report_key="test scalar_addi",
            ),
        ),
    )

    _expect_value_error(
        lambda: _validate_c_table_shape(table, _c_shape_contract(), ()),
        "lower-rule set 'test.low.generated_c_shape' rule 0 report key must not contain whitespace",
    )


def test_generate_lower_rule_set_emits_value_ref_for_float_equals_guard() -> None:
    table = ContractFragment(
        name="test.low.float_equals",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        public_header=_TEST_PUBLIC_HEADER,
        cases=[
            DescriptorRule(
                source_op=scalar_arithmetic.scalar_mulf,
                descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                guards=(
                    Guard.value_type("lhs", Scalar("f32")),
                    Guard.value_float_equals("rhs", 0.0),
                    Guard.value_type("rhs", Scalar("f32")),
                    Guard.value_type("result", Scalar("f32")),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                        operands={
                            "lhs": ValueRef.operand("lhs"),
                            "rhs": ValueRef.operand("rhs"),
                        },
                        results={"dst": ValueRef.result("result")},
                    ),
                ),
            )
        ],
    )

    generated = generate_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    guard_start = generated.source.index("LOOM_LOW_LOWER_GUARD_VALUE_FLOAT_EQUALS")
    guard_end = generated.source.index("},", guard_start)
    guard_text = generated.source[guard_start:guard_end]
    assert ".value_ref_index = 1," in guard_text
    assert ".u64 = UINT64_C(0)," in guard_text


def test_generate_lower_rule_set_emits_storage_element_format_guard() -> None:
    table = ContractFragment(
        name="test.low.storage_schema",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        public_header=_TEST_PUBLIC_HEADER,
        cases=[
            DescriptorRule(
                source_op=vector.vector_fragment_load,
                descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                guards=(
                    Guard.value_storage_element_format(
                        "view",
                        "LOOM_VALUE_FACT_NUMERIC_FORMAT_U8",
                    ),
                    Guard.value_type("result", Vector("f32", lanes=4)),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                        operands={
                            "lhs": ValueRef.result("result"),
                            "rhs": ValueRef.result("result"),
                        },
                        results={"dst": ValueRef.result("result")},
                    ),
                ),
            )
        ],
    )

    generated = generate_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    guard_start = generated.source.index("LOOM_LOW_LOWER_GUARD_VALUE_STORAGE_ELEMENT_FORMAT")
    guard_end = generated.source.index("},", guard_start)
    guard_text = generated.source[guard_start:guard_end]
    assert ".value_ref_index = 0," in guard_text
    assert ".u64 = LOOM_VALUE_FACT_NUMERIC_FORMAT_U8," in guard_text


def test_generate_lower_rule_set_emits_packed_integer_storage_guard() -> None:
    table = ContractFragment(
        name="test.low.packed_integer_storage",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        public_header=_TEST_PUBLIC_HEADER,
        cases=[
            RecipeRule(
                source_op=vector.vector_bitunpacku,
                guards=(
                    Guard.value_packed_integer_lanes_from_payload(
                        "source",
                        "result",
                        "width",
                        storage_unit_bit_count=32,
                        maximum_storage_unit_count=16,
                        maximum_lane_count=32,
                    ),
                ),
            )
        ],
    )

    generated = generate_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    guard_start = generated.source.index("LOOM_LOW_LOWER_GUARD_VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD")
    guard_end = generated.source.index("},", guard_start)
    guard_text = generated.source[guard_start:guard_end]
    assert ".value_ref_index = 0," in guard_text
    assert ".other_value_ref_index = 1," in guard_text
    assert ".attr_index = 0," in guard_text
    assert ".u64 = UINT64_C(16)," in guard_text
    assert ".minimum_i64 = INT64_C(32)," in guard_text
    assert ".maximum_i64 = INT64_C(32)," in guard_text


def test_guard_row_emits_portable_signed_i64_bounds() -> None:
    fields = guard_row(
        {},
        LowerGuard(
            kind=GuardKind.I64_RANGE,
            minimum_i64=-(1 << 31),
            maximum_i64=(1 << 31) - 1,
        ),
    )

    assert ".minimum_i64 = (-INT64_C(2147483648))" in fields
    assert ".maximum_i64 = INT64_C(2147483647)" in fields


def test_guard_row_emits_value_memory_space_mask() -> None:
    fields = guard_row(
        {},
        LowerGuard(
            kind=GuardKind.VALUE_MEMORY_SPACE,
            memory_spaces=("unknown", "global", "descriptor"),
        ),
    )

    assert ".value_ref_index = 0" in fields
    assert (".u64 = LOOM_LOW_LOWER_MEMORY_SPACE_UNKNOWN | LOOM_LOW_LOWER_MEMORY_SPACE_GLOBAL | LOOM_LOW_LOWER_MEMORY_SPACE_DESCRIPTOR") in fields


def test_attr_copy_row_emits_portable_signed_i64_literal() -> None:
    fields = attr_copy_row(
        LowerAttrCopy(
            kind=LowerAttrCopyKind.I64_LITERAL,
            target_name="value",
            literal_i64=-(1 << 31),
        ),
        target_name_string_offset="TEST_STRING_VALUE",
    )

    assert ".target_name_string_offset = TEST_STRING_VALUE" in fields
    assert ".literal_i64 = (-INT64_C(2147483648))" in fields


def test_diagnostic_param_row_emits_portable_signed_i64_literal() -> None:
    fields = diagnostic_param_row(
        LowerDiagnosticParam(
            name="value",
            kind=DiagnosticParamKind.I64_LITERAL,
            i64_value=-(1 << 31),
        )
    )

    assert ".value = {.i64_value = (-INT64_C(2147483648))}" in fields


def test_diagnostic_rows_use_recorded_target_context() -> None:
    target_context_params = (
        LowerDiagnosticParam("target_key", DiagnosticParamKind.TARGET_KEY),
        LowerDiagnosticParam("export_name", DiagnosticParamKind.EXPORT_NAME),
        LowerDiagnosticParam("config_key", DiagnosticParamKind.CONFIG_KEY),
        LowerDiagnosticParam("function_name", DiagnosticParamKind.FUNCTION_NAME),
        LowerDiagnosticParam("op_name", DiagnosticParamKind.SOURCE_OP_NAME),
    )
    literal_params = (
        LowerDiagnosticParam(
            "subject_role",
            DiagnosticParamKind.STRING_LITERAL,
            string_value="operand",
        ),
        LowerDiagnosticParam(
            "subject_name",
            DiagnosticParamKind.STRING_LITERAL,
            string_value="lhs",
        ),
        LowerDiagnosticParam(
            "constraint_key",
            DiagnosticParamKind.STRING_LITERAL,
            string_value="type",
        ),
    )
    canonical = LowerDiagnostic(
        ERR_TARGET_003,
        (*target_context_params, *literal_params),
        target_context_param_count=5,
    )
    unmarked = LowerDiagnostic(
        ERR_TARGET_003,
        (*target_context_params, *literal_params),
    )

    assert diagnostic_has_implicit_target_context(canonical)
    assert diagnostic_stored_params(canonical) == literal_params
    assert not diagnostic_has_implicit_target_context(unmarked)
    assert diagnostic_stored_params(unmarked) == unmarked.params


def test_generate_lower_rule_set_elides_authored_target_context_rows() -> None:
    table = ContractFragment(
        name="test.low.diagnostic_context",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        public_header=_TEST_PUBLIC_HEADER,
        cases=[
            RecipeRule(
                source_op=scalar_arithmetic.scalar_addi,
                guards=(Guard.value_type("lhs", Scalar("i32")),),
            )
        ],
    )

    generated = generate_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert "LOOM_LOW_LOWER_DIAGNOSTIC_FLAG_IMPLICIT_TARGET_CONTEXT" in generated.source
    assert ".param_count = 8," in generated.source
    assert "LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_TARGET_KEY" not in generated.source
    assert "LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_EXPORT_NAME" not in generated.source
    assert "LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_CONFIG_KEY" not in generated.source
    assert "LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_FUNCTION_NAME" not in generated.source
    assert "LOOM_LOW_LOWER_DIAGNOSTIC_PARAM_SOURCE_OP_NAME" not in generated.source


def test_generate_lower_rule_set_emits_static_element_count_type_pattern() -> None:
    table = ContractFragment(
        name="test.low.vector_extract_shape",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        public_header=_TEST_PUBLIC_HEADER,
        cases=[
            RecipeRule(
                source_op=vector.vector_extract,
                guards=(
                    Guard.value_type(
                        "source",
                        Vector(
                            "f32",
                            minimum_static_elements=1,
                            maximum_static_elements=8,
                        ),
                    ),
                    Guard.value_type("result", Scalar("f32")),
                    Guard.vector_extract_shape("source", "result", "static_indices"),
                ),
            )
        ],
    )

    generated = generate_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    type_pattern_start = generated.source.index("LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_ELEMENT_COUNT_RANGE")
    type_pattern_end = generated.source.index("},", type_pattern_start)
    type_pattern_text = generated.source[type_pattern_start:type_pattern_end]
    assert ".static_element_count_min = 1," in type_pattern_text
    assert ".static_element_count_max = 8," in type_pattern_text

    guard_start = generated.source.index("LOOM_LOW_LOWER_GUARD_VECTOR_EXTRACT_SHAPE")
    guard_end = generated.source.index("},", guard_start)
    guard_text = generated.source[guard_start:guard_end]
    assert ".value_ref_index = 0," in guard_text
    assert ".other_value_ref_index = 1," in guard_text
    assert ".attr_index = 0," in guard_text


def test_generate_lower_rule_set_emits_exact_view_shape_type_pattern() -> None:
    table = ContractFragment(
        name="test.low.view_shape",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        public_header=_TEST_PUBLIC_HEADER,
        cases=[
            RecipeRule(
                source_op=vector.vector_load,
                guards=(Guard.value_type("view", View("i32", dims=(4, 8))),),
            )
        ],
    )

    generated = generate_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    type_pattern_start = generated.source.index("LOOM_LOW_LOWER_TYPE_PATTERN_FLAG_STATIC_DIM1")
    type_pattern_end = generated.source.index("},", type_pattern_start)
    type_pattern_text = generated.source[type_pattern_start:type_pattern_end]
    assert ".type_kind = LOOM_TYPE_VIEW," in type_pattern_text
    assert ".rank = 2," in type_pattern_text
    assert ".static_dim0 = 4," in type_pattern_text
    assert ".static_dim1 = 8," in type_pattern_text


def test_generate_lower_rule_set_emits_source_instance_flags_projection() -> None:
    descriptor, descriptor_set = _add_f32_flags_descriptor_set()
    table = ContractFragment(
        name="test.low.flags",
        descriptor_set=descriptor_set,
        public_header=_TEST_PUBLIC_HEADER,
        cases=[
            DescriptorRule(
                source_op=scalar_arithmetic.scalar_divf,
                descriptor=descriptor,
                guards=(
                    Guard.value_type("lhs", Scalar("f32")),
                    Guard.value_type("rhs", Scalar("f32")),
                    Guard.value_type("result", Scalar("f32")),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=descriptor,
                        operands={
                            "lhs": ValueRef.operand("lhs"),
                            "rhs": ValueRef.operand("rhs"),
                        },
                        results={"dst": ValueRef.result("result")},
                        immediates={"fast_math_flags": SourceOpProject.instance_flags()},
                    ),
                ),
            )
        ],
    )

    generated = generate_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert "LOOM_LOW_LOWER_ATTR_COPY_SOURCE_OP_INSTANCE_FLAGS" in generated.source
    assert 'LOOM_BSTRING_LITERAL(15, "fast_math_flags")' in generated.source
    assert ".target_name_string_offset = " in generated.source


def test_generate_lower_rule_set_emits_balanced_accumulator_flag() -> None:
    table = ContractFragment(
        name="test.low.accumulate_tree",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        public_header=_TEST_PUBLIC_HEADER,
        cases=[
            DescriptorRule(
                source_op=vector.vector_reduce,
                descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                guards=(
                    Guard.value_type("input", Vector("f32", lanes=4)),
                    Guard.value_type("init", Scalar("f32")),
                    Guard.value_type("result", Scalar("f32")),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                        operands={
                            "lhs": ValueRef.operand("input"),
                            "rhs": ValueRef.operand("input"),
                        },
                        results={"dst": ValueRef.result("result")},
                        form=DescriptorEmitForm.ACCUMULATE_LANES,
                        accumulator="lhs",
                        accumulator_seed=DescriptorAccumulatorSeed.FIRST_LANE,
                        accumulator_tree=DescriptorAccumulatorTree.BALANCED,
                    ),
                ),
            )
        ],
    )

    generated = generate_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert "LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_ACCUMULATE_LANES" in generated.source
    assert "LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE" in generated.source
    assert "LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_TREE_BALANCED" in generated.source


def test_generate_lower_rule_set_emits_balanced_operand_seed() -> None:
    table = ContractFragment(
        name="test.low.accumulate_operand_tree",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        public_header=_TEST_PUBLIC_HEADER,
        cases=[
            DescriptorRule(
                source_op=vector.vector_reduce,
                descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                guards=(
                    Guard.value_type("input", Vector("f32", lanes=4)),
                    Guard.value_type("init", Scalar("f32")),
                    Guard.value_type("result", Scalar("f32")),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                        operands={
                            "lhs": ValueRef.operand("init"),
                            "rhs": ValueRef.operand("input"),
                        },
                        results={"dst": ValueRef.result("result")},
                        form=DescriptorEmitForm.ACCUMULATE_LANES,
                        accumulator="lhs",
                        accumulator_tree=DescriptorAccumulatorTree.BALANCED,
                    ),
                ),
            )
        ],
    )

    generated = generate_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert "LOOM_LOW_LOWER_EMIT_DESCRIPTOR_OP_ACCUMULATE_LANES" in generated.source
    assert "LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_TREE_BALANCED" in generated.source
    assert "LOOM_LOW_LOWER_EMIT_FLAG_ACCUMULATE_SEED_FIRST_LANE" not in generated.source


def test_generate_lower_rule_set_emits_divisor_magic_projection() -> None:
    table = ContractFragment(
        name="test.low.divisor_magic",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        public_header=_TEST_PUBLIC_HEADER,
        cases=[
            DescriptorRule(
                source_op=scalar_arithmetic.scalar_addi,
                descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                guards=(
                    Guard.value_type("lhs", Scalar("i32")),
                    Guard.value_u32_divisor_magic_is_add("rhs", True),
                    Guard.value_type("rhs", Scalar("i32")),
                    Guard.value_type("result", Scalar("i32")),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                        results={"dst": ValueRef.result("result")},
                        immediates={"i32_value": ValueProject.u32_divisor_magic_multiplier("rhs")},
                    ),
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                        results={"dst": ValueRef.temporary("shift")},
                        result_types={"dst": ValueRef.result("result")},
                        immediates={"i32_value": ValueProject.u32_divisor_magic_shift("rhs")},
                    ),
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                        results={"dst": ValueRef.temporary("mask")},
                        result_types={"dst": ValueRef.result("result")},
                        immediates={"i32_value": ValueProject.exact_i64_minus_one("rhs")},
                    ),
                ),
            )
        ],
    )

    generated = generate_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert "LOOM_LOW_LOWER_GUARD_VALUE_U32_DIVISOR_MAGIC_IS_ADD" in generated.source
    assert "LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_MULTIPLIER" in generated.source
    assert "LOOM_LOW_LOWER_ATTR_COPY_VALUE_U32_DIVISOR_MAGIC_SHIFT" in generated.source
    assert "LOOM_LOW_LOWER_ATTR_COPY_VALUE_EXACT_I64_MINUS_ONE" in generated.source


def test_source_memory_row_emits_dynamic_byte_stride_any_flag() -> None:
    row = LowerSourceMemory(
        constraint=SourceMemoryConstraint(
            operation=SourceMemoryOperation.LOAD,
            memory_spaces=("global",),
            element_byte_count=4,
            vector_lane_count=1,
            vector_lane_byte_stride=4,
            static_byte_offset_minimum=0,
            static_byte_offset_maximum=128,
            dynamic_term_count=1,
            dynamic_view_base_term_count=0,
            dynamic_index_source=SourceMemoryDynamicIndexSource.VALUE,
            dynamic_byte_stride=None,
        ),
        diagnostic_index=3,
        dynamic_offset_diagnostic_index=4,
    )

    fields = source_memory_row({}, row)

    assert ".flags = LOOM_LOW_LOWER_SOURCE_MEMORY_FLAG_DYNAMIC_BYTE_STRIDE_ANY" in fields
    assert ".dynamic_term_count = 1" in fields
    assert ".dynamic_view_base_term_count = 0" in fields
    assert ".dynamic_byte_stride = " not in "\n".join(fields)


def test_source_memory_row_emits_compact_address_layout() -> None:
    row = LowerSourceMemory(
        constraint=SourceMemoryConstraint(
            operation=SourceMemoryOperation.LOAD,
            address_layout=SourceMemoryAddressLayout.COMPACT_ROW_MAJOR,
            memory_spaces=("global",),
            element_byte_count=4,
            vector_lane_count=1,
            vector_lane_byte_stride=4,
            static_byte_offset=0,
        ),
        diagnostic_index=3,
        dynamic_offset_diagnostic_index=4,
        address_layout_diagnostic_index=5,
    )

    fields = source_memory_row({}, row)

    assert (".address_layout = LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_LAYOUT_COMPACT_ROW_MAJOR") in fields
    assert ".address_layout_diagnostic_index = 5" in fields


def test_source_memory_row_emits_preserve_source_index_flag() -> None:
    row = LowerSourceMemory(
        constraint=SourceMemoryConstraint(
            operation=SourceMemoryOperation.LOAD,
            memory_spaces=("global",),
            element_byte_count=4,
            vector_lane_count=1,
            vector_lane_byte_stride=4,
            static_byte_offset_minimum=0,
            static_byte_offset_maximum=128,
            dynamic_term_count=None,
            dynamic_term_count_minimum=1,
            dynamic_view_base_term_count=0,
            preserve_source_index=True,
        ),
        diagnostic_index=3,
        dynamic_offset_diagnostic_index=4,
    )

    fields = source_memory_row({}, row)

    assert ".flags = LOOM_LOW_LOWER_SOURCE_MEMORY_FLAG_PRESERVE_SOURCE_INDEX" in fields


def test_source_memory_row_emits_any_positive_dynamic_term_count() -> None:
    row = LowerSourceMemory(
        constraint=SourceMemoryConstraint(
            operation=SourceMemoryOperation.LOAD,
            memory_spaces=("global",),
            element_byte_count=4,
            vector_lane_count=1,
            vector_lane_byte_stride=4,
            static_byte_offset_minimum=0,
            static_byte_offset_maximum=128,
            dynamic_term_count=None,
            dynamic_term_count_minimum=1,
        ),
        diagnostic_index=3,
        dynamic_offset_diagnostic_index=4,
    )

    fields = source_memory_row({}, row)

    assert (".dynamic_term_count = LOOM_LOW_LOWER_SOURCE_MEMORY_DYNAMIC_TERM_COUNT_ANY") in fields
    assert ".dynamic_term_count_minimum = 1" in fields
    assert (".dynamic_view_base_term_count = LOOM_LOW_LOWER_SOURCE_MEMORY_DYNAMIC_VIEW_BASE_TERM_COUNT_ANY") in fields


def test_source_memory_row_emits_portable_signed_i64_values() -> None:
    row = LowerSourceMemory(
        constraint=SourceMemoryConstraint(
            operation=SourceMemoryOperation.LOAD,
            memory_spaces=("global",),
            element_byte_count=4,
            vector_lane_count=1,
            vector_lane_byte_stride=-(1 << 31),
            static_byte_offset_minimum=-(1 << 63),
            static_byte_offset_maximum=(1 << 63) - 1,
            dynamic_term_count=1,
            dynamic_index_source=SourceMemoryDynamicIndexSource.VALUE,
            dynamic_byte_stride=-(1 << 31),
        ),
        diagnostic_index=0xFFFF,
        dynamic_offset_diagnostic_index=0xFFFF,
    )

    fields = source_memory_row({}, row)

    assert ".vector_lane_byte_stride = (-INT64_C(2147483648))" in fields
    assert ".static_byte_offset_minimum = INT64_MIN" in fields
    assert ".static_byte_offset_maximum = INT64_C(9223372036854775807)" in fields
    assert ".dynamic_byte_stride = (-INT64_C(2147483648))" in fields


def test_source_memory_row_emits_dynamic_stride_values_flag() -> None:
    row = LowerSourceMemory(
        constraint=SourceMemoryConstraint(
            operation=SourceMemoryOperation.LOAD,
            memory_spaces=("global",),
            element_byte_count=4,
            vector_lane_count=1,
            vector_lane_byte_stride=4,
            static_byte_offset_minimum=0,
            static_byte_offset_maximum=128,
            dynamic_term_count=1,
            dynamic_index_source=SourceMemoryDynamicIndexSource.VALUE,
            dynamic_byte_stride=None,
            allow_dynamic_stride_values=True,
        ),
        diagnostic_index=3,
        dynamic_offset_diagnostic_index=4,
    )

    fields = source_memory_row({}, row)

    assert (".flags = LOOM_LOW_LOWER_SOURCE_MEMORY_FLAG_DYNAMIC_BYTE_STRIDE_ANY | LOOM_LOW_LOWER_SOURCE_MEMORY_FLAG_DYNAMIC_STRIDE_VALUES") in fields


def test_source_memory_row_emits_complete_address_materializer() -> None:
    materializer = SourceMemoryAddressMaterializer(
        const_coordinate=TEST_LOW_CONST_I32_DESCRIPTOR,
        add_coordinate=TEST_LOW_ADD_I32_DESCRIPTOR,
        mul_coordinate=TEST_LOW_MUL_I32_DESCRIPTOR,
        shl_coordinate=None,
        address=TEST_LOW_ADD_I32_DESCRIPTOR,
        base=SourceMemoryAddressBase.BASE_VIEW,
        coordinate_type=SourceMemoryAddressCoordinateType.INDEX,
        coordinate_unit_byte_count=4,
        coordinate_minimum=0,
        coordinate_maximum=(2**31) - 1,
        const_coordinate_immediate="i32_value",
    )
    row = LowerSourceMemory(
        constraint=SourceMemoryConstraint(
            operation=SourceMemoryOperation.LOAD,
            root_kind=SourceMemoryRootKind.ALLOCA,
            memory_spaces=("global",),
            element_byte_count=4,
            vector_lane_count=1,
            vector_lane_byte_stride=4,
            static_byte_offset_minimum=0,
            static_byte_offset_maximum=(2**31) - 1,
            dynamic_term_count=None,
        ),
        diagnostic_index=3,
        dynamic_offset_diagnostic_index=4,
        address_diagnostic_index=5,
        address_materializer=materializer,
    )
    descriptor_refs = {
        TEST_LOW_CONST_I32_DESCRIPTOR.key: 0,
        TEST_LOW_ADD_I32_DESCRIPTOR.key: 1,
        TEST_LOW_MUL_I32_DESCRIPTOR.key: 2,
    }

    fields = source_memory_row(
        descriptor_refs,
        row,
        address_immediate_string_offset="TEST_STRING_I32_VALUE",
    )

    assert ".address_diagnostic_index = 5" in fields
    assert ".root_kind = LOOM_LOW_LOWER_SOURCE_MEMORY_ROOT_ALLOCA" in fields
    assert (".address_base_kind = LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_BASE_VIEW") in fields
    assert (".address_coordinate_type = LOOM_LOW_LOWER_SOURCE_MEMORY_ADDRESS_COORDINATE_INDEX") in fields
    assert ".address_coordinate_unit_byte_count = 4" in fields
    assert ".address_coordinate_minimum = INT64_C(0)" in fields
    assert ".address_coordinate_maximum = INT64_C(2147483647)" in fields
    assert ".address_const_coordinate_descriptor_ref = 0" in fields
    assert (".address_const_coordinate_immediate_string_offset = TEST_STRING_I32_VALUE") in fields
    assert ".address_add_coordinate_descriptor_ref = 1" in fields
    assert ".address_shl_coordinate_descriptor_ref = 65535" in fields
    assert ".address_index_to_coordinate_input_descriptor_ref = 65535" in fields
    assert ".address_index_to_coordinate_descriptor_ref = 65535" in fields
    assert ".address_descriptor_ref = 1" in fields

    table = _compiled_lower_rule_set(source_memories=(row,))
    keys = descriptor_ref_keys(table, _c_shape_contract())
    assert set(keys) == set(descriptor_refs)
