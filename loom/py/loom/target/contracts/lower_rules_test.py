# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import analysis as scalar_analysis
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import bitwise as scalar_bitwise
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.contracts import (
    LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS,
    LOWER_EMIT_FLAG_RESULT_DESCRIPTOR_TYPE,
    LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN,
    LOWER_RULE_FLAG_CONTRACT_ONLY,
    LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS,
    AttrProject,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    DirectDescriptorCase,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    GuardDiagnostic,
    GuardKind,
    LowerAttrCopyKind,
    LowerEmitKind,
    OrdinalValueAliasRule,
    RecipeRule,
    Scalar,
    SourceMemoryAddressBase,
    SourceMemoryAddressCoordinateType,
    SourceMemoryAddressLayout,
    SourceMemoryAddressMaterializer,
    SourceMemoryByteOffsetMaterializer,
    SourceMemoryConstraint,
    SourceMemoryDynamicIndexSource,
    SourceMemoryOperation,
    SourceMemoryProject,
    SourceMemoryRootKind,
    SourceOpProject,
    SourceValueKind,
    TypePattern,
    ValueAliasRule,
    ValueElideRule,
    ValueProject,
    ValueRef,
    Vector,
    binary_descriptor_rules,
    compile_lower_rule_set,
)
from loom.target.low_descriptors import EnumDomain, EnumValue, Immediate, ImmediateKind
from loom.target.test.descriptors import (
    TEST_LOW_ADD_F32_DESCRIPTOR,
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_AMBIGUOUS_DESCRIPTOR,
    TEST_LOW_CONST_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
    TEST_LOW_FROM_ELEMENTS_V4I32_DESCRIPTOR,
    TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR,
    TEST_LOW_LOAD_V4I32_DESCRIPTOR,
    TEST_LOW_MUL_I32_DESCRIPTOR,
    TEST_LOW_REMATERIALIZE_I32_DESCRIPTOR,
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


def test_compile_structural_register_emits() -> None:
    i32 = Scalar("i32")
    v2i32 = Vector("i32", lanes=2)
    fragment = ContractFragment(
        name="test.structural-register-emits",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=(
            DescriptorRule(
                source_op=vector.vector_from_elements,
                guards=(
                    Guard.operand_segment_count("elements", 2),
                    Guard.value_type("result", v2i32),
                ),
                emit=(
                    EmitRegisterConcat(
                        sources=(
                            ValueRef.operand("elements", element=0),
                            ValueRef.operand("elements", element=1),
                        ),
                        result=ValueRef.result("result"),
                    ),
                ),
            ),
            DescriptorRule(
                source_op=vector.vector_extract,
                guards=(
                    Guard.i64_array_count("static_indices", 1),
                    Guard.i64_array_element_range("static_indices", 0, 1, 1),
                    Guard.value_type("source", v2i32),
                    Guard.value_type("result", i32),
                ),
                emit=(
                    EmitRegisterSlice(
                        source=ValueRef.operand("source"),
                        result=ValueRef.temporary("element"),
                        unit_offset=1,
                        result_type=i32,
                    ),
                    EmitRegisterSlice(
                        source=ValueRef.temporary("element"),
                        result=ValueRef.result("result"),
                    ),
                ),
            ),
        ),
    )

    compiled = compile_lower_rule_set(
        fragment,
        dialect_ops={"vector": ALL_VECTOR_OPS},
    )

    assert tuple(emit.kind for emit in compiled.emits) == (
        LowerEmitKind.REGISTER_CONCAT,
        LowerEmitKind.REGISTER_SLICE,
        LowerEmitKind.REGISTER_SLICE,
    )
    assert all(emit.descriptor is None for emit in compiled.emits)
    assert compiled.emits[0].operand_ref_count == 2
    assert compiled.emits[1].operand_ref_count == 1
    assert compiled.emits[1].structural_offset == 1
    assert compiled.emits[1].flags == (
        LOWER_EMIT_FLAG_BIND_RESULTS_TO_REFS | LOWER_EMIT_FLAG_RESULT_TYPE_PATTERN
    )
    typed_slice_result = compiled.value_refs[compiled.emits[1].result_bind_ref_start]
    assert typed_slice_result.kind is SourceValueKind.TEMPORARY
    assert compiled.emits[2].operand_ref_count == 1
    assert compiled.emits[2].structural_offset == 0


def _expect_value_error(callable_obj: Callable[[], object], message: str) -> None:
    error: ValueError | None = None
    try:
        callable_obj()
    except ValueError as exc:
        error = exc
    assert error is not None
    assert message in str(error)


def _source_index_rule(
    *,
    preserve: bool,
    use_original_index: bool,
    additional_preserve: bool | None = None,
) -> ContractFragment:
    result_type = Vector("i32", lanes=4)
    index = (
        ValueRef.operand("indices")
        if use_original_index
        else ValueRef.source_memory_dynamic_term()
    )

    def source_memory(preserve_source_index: bool) -> SourceMemoryConstraint:
        return SourceMemoryConstraint(
            operation=SourceMemoryOperation.LOAD,
            memory_spaces=("global",),
            element_byte_count=4,
            vector_lane_count=4,
            vector_lane_byte_stride=4,
            static_byte_offset_minimum=0,
            static_byte_offset_maximum=128,
            dynamic_term_count=1,
            dynamic_view_base_term_count=0,
            dynamic_index_source=SourceMemoryDynamicIndexSource.VALUE,
            dynamic_byte_stride=4,
            preserve_source_index=preserve_source_index,
        )

    emits = []
    if additional_preserve is not None:
        emits.append(
            EmitDescriptorOp(
                descriptor=TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR,
                operands={"address": ValueRef.operand("view"), "index": index},
                results={"dst": ValueRef.temporary("additional_result")},
                result_types={"dst": result_type},
                source_memory=source_memory(additional_preserve),
            )
        )
    emits.append(
        EmitDescriptorOp(
            descriptor=TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR,
            operands={"address": ValueRef.operand("view"), "index": index},
            results={"dst": ValueRef.result("result")},
            source_memory=source_memory(preserve),
        )
    )
    return ContractFragment(
        name="test.source-index",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_load,
                descriptor=TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR,
                guards=(
                    Guard.operand_segment_count("indices", 1),
                    Guard.value_type("result", result_type),
                ),
                emit=tuple(emits),
            )
        ],
    )


def _source_memory_address_rule(
    *,
    materializer: SourceMemoryAddressMaterializer | None,
) -> ContractFragment:
    return ContractFragment(
        name="test.source-memory-address",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_load,
                descriptor=TEST_LOW_LOAD_V4I32_DESCRIPTOR,
                guards=(Guard.value_type("result", Vector("i32", lanes=4)),),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_LOAD_V4I32_DESCRIPTOR,
                        operands={"address": ValueRef.source_memory_address()},
                        results={"dst": ValueRef.result("result")},
                        source_memory=SourceMemoryConstraint(
                            operation=SourceMemoryOperation.LOAD,
                            memory_spaces=("global",),
                            element_byte_count=4,
                            vector_lane_count=4,
                            vector_lane_byte_stride=4,
                            static_byte_offset_minimum=0,
                            static_byte_offset_maximum=(2**31) - 1,
                            dynamic_term_count=None,
                            dynamic_term_count_minimum=0,
                        ),
                        source_memory_address_materializer=materializer,
                    ),
                ),
            )
        ],
    )


def _i32_source_memory_address_materializer() -> SourceMemoryAddressMaterializer:
    return SourceMemoryAddressMaterializer(
        const_coordinate=TEST_LOW_CONST_I32_DESCRIPTOR,
        add_coordinate=TEST_LOW_ADD_I32_DESCRIPTOR,
        mul_coordinate=TEST_LOW_MUL_I32_DESCRIPTOR,
        index_to_coordinate_input=TEST_LOW_REMATERIALIZE_I32_DESCRIPTOR,
        index_to_coordinate=TEST_LOW_REMATERIALIZE_I32_DESCRIPTOR,
        address=TEST_LOW_ADD_I32_DESCRIPTOR,
        const_coordinate_immediate="i32_value",
    )


def _binary_rule(
    *,
    source_op: Op,
    type_pattern: TypePattern,
) -> DescriptorRule:
    return binary_descriptor_rules(
        (
            DirectDescriptorCase(
                source_op=source_op,
                descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                type_patterns=type_pattern,
            ),
        )
    )[0]


def test_compile_lower_rule_set_compiles_direct_scalar_rule() -> None:
    table = ContractFragment(
        name="test.scalar",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            replace(
                _binary_rule(
                    source_op=scalar_arithmetic.scalar_addi,
                    type_pattern=Scalar("i32"),
                ),
                report_key="test.scalar_addi.strategy.direct",
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert compiled.authored_case_indices == (0,)
    assert len(compiled.spans) == 1
    assert compiled.spans[0].source_op is scalar_arithmetic.scalar_addi
    assert compiled.spans[0].rule_start == 0
    assert compiled.spans[0].rule_count == 1

    assert len(compiled.rules) == 1
    assert compiled.rules[0].source_op is scalar_arithmetic.scalar_addi
    assert compiled.rules[0].report_key == "test.scalar_addi.strategy.direct"
    assert compiled.rules[0].guard_count == 3
    assert compiled.rules[0].emit_count == 1

    assert len(compiled.emits) == 1
    assert compiled.emits[0].kind == LowerEmitKind.DESCRIPTOR_OP
    assert compiled.emits[0].descriptor is TEST_LOW_ADD_I32_DESCRIPTOR
    assert compiled.emits[0].operand_ref_count == 2
    assert compiled.emits[0].result_ref_count == 1


def test_compile_lower_rule_set_interns_exact_rule_programs() -> None:
    table = ContractFragment(
        name="test.scalar",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=(
            _binary_rule(
                source_op=scalar_arithmetic.scalar_addi,
                type_pattern=Scalar("i32"),
            ),
            _binary_rule(
                source_op=scalar_arithmetic.scalar_addi,
                type_pattern=Scalar("i32"),
            ),
        ),
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert len(compiled.rules) == 2
    assert len(compiled.guards) == 3
    assert len(compiled.emits) == 1
    assert compiled.rules[0].guard_start == compiled.rules[1].guard_start
    assert compiled.rules[0].emit_start == compiled.rules[1].emit_start
    assert compiled.rules[0].guard_count == compiled.rules[1].guard_count == 3
    assert compiled.rules[0].emit_count == compiled.rules[1].emit_count == 1


def test_compile_lower_rule_set_compiles_descriptor_result_type_binding() -> None:
    table = ContractFragment(
        name="test.scalar",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=scalar_arithmetic.scalar_addi,
                descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                guards=(
                    Guard.value_type("lhs", Scalar("i32")),
                    Guard.value_type("rhs", Scalar("i32")),
                    Guard.value_type("result", Scalar("i32")),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                        operands={
                            "lhs": ValueRef.operand("lhs"),
                            "rhs": ValueRef.operand("rhs"),
                        },
                        results={"dst": ValueRef.temporary("sum")},
                        result_types={"dst": DescriptorResultType()},
                    ),
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                        operands={
                            "lhs": ValueRef.temporary("sum"),
                            "rhs": ValueRef.operand("rhs"),
                        },
                        results={"dst": ValueRef.result("result")},
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert len(compiled.emits) == 2
    emit = compiled.emits[0]
    assert emit.flags & LOWER_EMIT_FLAG_RESULT_DESCRIPTOR_TYPE
    assert emit.result_ref_count == 1
    assert compiled.value_refs[emit.result_ref_start].kind == SourceValueKind.TEMPORARY


def test_compile_lower_rule_set_infers_vector_per_lane_emit() -> None:
    table = ContractFragment(
        name="test.vector",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            _binary_rule(
                source_op=vector.vector_addi,
                type_pattern=Vector("i32", lanes=4),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert len(compiled.rules) == 1
    assert compiled.rules[0].source_op is vector.vector_addi
    assert len(compiled.emits) == 1
    assert compiled.emits[0].kind == LowerEmitKind.DESCRIPTOR_OP_PER_LANE
    assert compiled.emits[0].descriptor is TEST_LOW_ADD_I32_DESCRIPTOR


def test_compile_lower_rule_set_compiles_per_lane_sequence_emit() -> None:
    table = ContractFragment(
        name="test.vector",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_addi,
                descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                guards=(
                    Guard.value_type("lhs", Vector("i32", lanes=4)),
                    Guard.value_type("rhs", Vector("i32", lanes=4)),
                    Guard.value_type("result", Vector("i32", lanes=4)),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                        operands={
                            "lhs": ValueRef.operand("lhs"),
                            "rhs": ValueRef.operand("rhs"),
                        },
                        results={"dst": ValueRef.temporary("partial")},
                        result_types={"dst": ValueRef.result("result")},
                        form=DescriptorEmitForm.PER_LANE_SEQUENCE,
                    ),
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                        operands={
                            "lhs": ValueRef.temporary("partial"),
                            "rhs": ValueRef.operand("rhs"),
                        },
                        results={"dst": ValueRef.result("result")},
                        form=DescriptorEmitForm.PER_LANE_SEQUENCE,
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert len(compiled.rules) == 1
    assert compiled.rules[0].emit_count == 2
    assert compiled.rules[0].temporary_count == 1
    assert tuple(emit.kind for emit in compiled.emits) == (
        LowerEmitKind.DESCRIPTOR_OP_PER_LANE_SEQUENCE,
        LowerEmitKind.DESCRIPTOR_OP_PER_LANE_SEQUENCE,
    )


def test_descriptor_rule_rejects_mixed_per_lane_sequence_emit() -> None:
    _expect_value_error(
        lambda: DescriptorRule(
            source_op=vector.vector_addi,
            descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
            emit=(
                EmitDescriptorOp(
                    descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                    operands={
                        "lhs": ValueRef.operand("lhs"),
                        "rhs": ValueRef.operand("rhs"),
                    },
                    results={"dst": ValueRef.temporary("partial")},
                    result_types={"dst": ValueRef.result("result")},
                    form=DescriptorEmitForm.PER_LANE_SEQUENCE,
                ),
                EmitDescriptorOp(
                    descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                    operands={
                        "lhs": ValueRef.temporary("partial"),
                        "rhs": ValueRef.operand("rhs"),
                    },
                    results={"dst": ValueRef.result("result")},
                    form=DescriptorEmitForm.PER_LANE,
                ),
            ),
        ).validate(TEST_LOW_CORE_DESCRIPTOR_SET),
        "per-lane-sequence emit programs cannot mix emission forms",
    )


def test_descriptor_rule_rejects_per_lane_sequence_without_final_result() -> None:
    _expect_value_error(
        lambda: DescriptorRule(
            source_op=vector.vector_addi,
            descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
            emit=(
                EmitDescriptorOp(
                    descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                    operands={
                        "lhs": ValueRef.operand("lhs"),
                        "rhs": ValueRef.operand("rhs"),
                    },
                    results={"dst": ValueRef.temporary("partial")},
                    result_types={"dst": ValueRef.result("result")},
                    form=DescriptorEmitForm.PER_LANE_SEQUENCE,
                ),
                EmitDescriptorOp(
                    descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                    operands={
                        "lhs": ValueRef.temporary("partial"),
                        "rhs": ValueRef.operand("rhs"),
                    },
                    results={"dst": ValueRef.temporary("discarded")},
                    result_types={"dst": ValueRef.result("result")},
                    form=DescriptorEmitForm.PER_LANE_SEQUENCE,
                ),
            ),
        ).validate(TEST_LOW_CORE_DESCRIPTOR_SET),
        "per-lane-sequence final emit must bind a source result",
    )


def test_compile_lower_rule_set_compiles_value_alias_cases() -> None:
    table = ContractFragment(
        name="test.alias",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            ValueAliasRule(
                source_op=vector.vector_fragment,
                source=ValueRef.operand("data"),
                result=ValueRef.result("result"),
                guards=(Guard.value_i64_range("rows", 0, 0),),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert compiled.authored_case_indices == (0,)
    assert len(compiled.rules) == 1
    assert compiled.rules[0].source_op is vector.vector_fragment
    assert compiled.rules[0].guard_count == 1
    assert compiled.rules[0].emit_count == 0
    assert compiled.rules[0].alias_ref_count == 1
    assert len(compiled.value_refs) == 3
    assert compiled.spans[0].source_op is vector.vector_fragment


def test_compile_lower_rule_set_compiles_ordinal_value_alias_cases() -> None:
    table = ContractFragment(
        name="test.ordinal_alias",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            OrdinalValueAliasRule(
                source_op=scalar_analysis.scalar_assume,
                source=ValueRef.operand("values"),
                result=ValueRef.result("results"),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert compiled.authored_case_indices == (0,)
    assert len(compiled.rules) == 1
    assert compiled.rules[0].source_op is scalar_analysis.scalar_assume
    assert compiled.rules[0].flags == LOWER_RULE_FLAG_ORDINAL_VALUE_ALIAS
    assert compiled.rules[0].alias_ref_count == 1
    assert compiled.value_refs[0].kind == SourceValueKind.OPERAND
    assert compiled.value_refs[1].kind == SourceValueKind.RESULT


def test_compile_lower_rule_set_compiles_value_elide_cases() -> None:
    table = ContractFragment(
        name="test.elide",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            ValueElideRule(
                source_op=vector.vector_extract,
                values=(ValueRef.result("result"),),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert compiled.authored_case_indices == (0,)
    assert len(compiled.rules) == 1
    assert compiled.rules[0].source_op is vector.vector_extract
    assert compiled.rules[0].emit_count == 0
    assert compiled.rules[0].elide_ref_count == 1
    assert len(compiled.value_refs) == 1
    assert compiled.spans[0].source_op is vector.vector_extract


def test_compile_lower_rule_set_compiles_guarded_value_elide_cases() -> None:
    table = ContractFragment(
        name="test.elide",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            ValueElideRule(
                source_op=vector.vector_extract,
                values=(ValueRef.result("result"),),
                guards=(Guard.value_no_uses("result"),),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert compiled.authored_case_indices == (0,)
    assert len(compiled.rules) == 1
    assert compiled.rules[0].source_op is vector.vector_extract
    assert compiled.rules[0].guard_count == 1
    assert compiled.rules[0].emit_count == 0
    assert compiled.rules[0].elide_ref_count == 1
    assert len(compiled.guards) == 1
    assert compiled.guards[0].kind == GuardKind.VALUE_NO_USES
    assert compiled.guards[0].value_ref_index == compiled.rules[0].elide_ref_start
    assert len(compiled.value_refs) == 1
    assert compiled.spans[0].source_op is vector.vector_extract


def test_compile_lower_rule_set_compiles_recipe_cases() -> None:
    table = ContractFragment(
        name="test.recipe",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            RecipeRule(
                source_op=vector.vector_addi,
                guards=(
                    Guard.value_type("lhs", Vector("i32", lanes=4)),
                    Guard.value_type("rhs", Vector("i32", lanes=4)),
                    Guard.value_type("result", Vector("i32", lanes=4)),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert compiled.authored_case_indices == (0,)
    assert len(compiled.rules) == 1
    assert compiled.rules[0].source_op is vector.vector_addi
    assert compiled.rules[0].flags == LOWER_RULE_FLAG_CONTRACT_ONLY
    assert compiled.rules[0].guard_count == 3
    assert compiled.rules[0].emit_count == 0
    assert compiled.rules[0].alias_ref_count == 0
    assert compiled.rules[0].elide_ref_count == 0
    assert compiled.spans[0].source_op is vector.vector_addi


def test_compile_lower_rule_set_offsets_variadic_operand_elements() -> None:
    table = ContractFragment(
        name="test.from-elements",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_from_elements,
                descriptor=TEST_LOW_FROM_ELEMENTS_V4I32_DESCRIPTOR,
                guards=(
                    Guard.operand_segment_count("elements", 4),
                    Guard.value_type("result", Vector("i32", lanes=4)),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_FROM_ELEMENTS_V4I32_DESCRIPTOR,
                        operands={
                            "lane0": ValueRef.operand("elements", element=0),
                            "lane1": ValueRef.operand("elements", element=1),
                            "lane2": ValueRef.operand("elements", element=2),
                            "lane3": ValueRef.operand("elements", element=3),
                        },
                        results={"dst": ValueRef.result("result")},
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    emit = compiled.emits[0]
    value_refs = compiled.value_refs[
        emit.operand_ref_start : emit.operand_ref_start + emit.operand_ref_count
    ]
    assert tuple(value_ref.index for value_ref in value_refs) == (0, 0, 0, 0)
    assert tuple(value_ref.element_index for value_ref in value_refs) == (0, 1, 2, 3)


def test_compile_lower_rule_set_compiles_source_memory_dynamic_term_operand() -> None:
    table = ContractFragment(
        name="test.source-memory-term",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_load,
                descriptor=TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR,
                guards=(
                    Guard.operand_segment_count("indices", 0),
                    Guard.value_type("result", Vector("i32", lanes=4)),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR,
                        operands={
                            "address": ValueRef.operand("view"),
                            "index": ValueRef.source_memory_dynamic_term(),
                        },
                        results={"dst": ValueRef.result("result")},
                        source_memory=SourceMemoryConstraint(
                            operation=SourceMemoryOperation.LOAD,
                            root_kind=SourceMemoryRootKind.ANY,
                            memory_spaces=("unknown", "global"),
                            element_byte_count=4,
                            vector_lane_count=4,
                            vector_lane_byte_stride=4,
                            static_byte_offset_minimum=-(2**63),
                            static_byte_offset_maximum=(2**63) - 1,
                            dynamic_term_count=1,
                            dynamic_view_base_term_count=0,
                            dynamic_index_source=SourceMemoryDynamicIndexSource.VALUE,
                            dynamic_byte_stride=None,
                        ),
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    emit = compiled.emits[0]
    value_refs = compiled.value_refs[
        emit.operand_ref_start : emit.operand_ref_start + emit.operand_ref_count
    ]
    assert tuple(value_ref.kind for value_ref in value_refs) == (
        SourceValueKind.OPERAND,
        SourceValueKind.SOURCE_MEMORY_DYNAMIC_TERM,
    )
    assert tuple(value_ref.index for value_ref in value_refs) == (0, 0)
    source_memory = compiled.source_memories[emit.source_memory_ordinal - 1]
    assert source_memory.constraint is table.cases[0].emit[0].source_memory
    assert source_memory.constraint.dynamic_view_base_term_count == 0


def test_compile_lower_rule_set_compiles_preserved_source_index() -> None:
    table = _source_index_rule(preserve=True, use_original_index=True)

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    source_memory = compiled.source_memories[0]
    assert source_memory.constraint.preserve_source_index


def test_compile_lower_rule_set_requires_source_index_preservation() -> None:
    _expect_value_error(
        lambda: compile_lower_rule_set(
            _source_index_rule(
                preserve=False,
                use_original_index=True,
            ),
            dialect_ops={"vector": ALL_VECTOR_OPS},
        ),
        "vector.load: source-memory rules that consume the original "
        "'indices' operand must preserve the source index",
    )


def test_compile_lower_rule_set_rejects_unused_source_index_preservation() -> None:
    _expect_value_error(
        lambda: compile_lower_rule_set(
            _source_index_rule(
                preserve=True,
                use_original_index=False,
            ),
            dialect_ops={"vector": ALL_VECTOR_OPS},
        ),
        "vector.load: source-index preservation requires the original "
        "'indices' operand",
    )


def test_compile_lower_rule_set_rejects_any_unused_source_index_preservation() -> None:
    _expect_value_error(
        lambda: compile_lower_rule_set(
            _source_index_rule(
                preserve=False,
                use_original_index=False,
                additional_preserve=True,
            ),
            dialect_ops={"vector": ALL_VECTOR_OPS},
        ),
        "vector.load: source-index preservation requires the original "
        "'indices' operand",
    )


def test_source_memory_constraint_rejects_static_source_index_preservation() -> None:
    _expect_value_error(
        lambda: SourceMemoryConstraint(
            operation=SourceMemoryOperation.LOAD,
            memory_spaces=("global",),
            element_byte_count=4,
            vector_lane_count=1,
            vector_lane_byte_stride=4,
            static_byte_offset=0,
            preserve_source_index=True,
        ),
        "source-index preservation requires dynamic source memory",
    )


def test_source_memory_constraint_rejects_dynamic_view_base_preservation() -> None:
    _expect_value_error(
        lambda: SourceMemoryConstraint(
            operation=SourceMemoryOperation.LOAD,
            memory_spaces=("global",),
            element_byte_count=4,
            vector_lane_count=1,
            vector_lane_byte_stride=4,
            static_byte_offset=0,
            dynamic_term_count=1,
            dynamic_index_source=SourceMemoryDynamicIndexSource.VALUE,
            dynamic_byte_stride=4,
            preserve_source_index=True,
        ),
        "source-index preservation requires zero dynamic view-base terms",
    )


def test_source_memory_constraint_accepts_any_dynamic_stride_value_terms() -> None:
    constraint = SourceMemoryConstraint(
        operation=SourceMemoryOperation.LOAD,
        memory_spaces=("global",),
        element_byte_count=4,
        vector_lane_count=1,
        vector_lane_byte_stride=4,
        static_byte_offset=0,
        dynamic_term_count=None,
        dynamic_term_count_minimum=1,
        allow_dynamic_stride_values=True,
    )

    assert constraint.dynamic_term_count is None
    assert constraint.dynamic_term_count_minimum == 1
    assert constraint.allow_dynamic_stride_values


def test_source_memory_constraint_rejects_stride_values_without_dynamic_terms() -> None:
    _expect_value_error(
        lambda: SourceMemoryConstraint(
            operation=SourceMemoryOperation.LOAD,
            memory_spaces=("global",),
            element_byte_count=4,
            vector_lane_count=1,
            vector_lane_byte_stride=4,
            static_byte_offset=0,
            dynamic_term_count=None,
            allow_dynamic_stride_values=True,
        ),
        "dynamic source memory stride values require at least one dynamic term",
    )


def test_source_memory_constraint_rejects_unknown_address_layout() -> None:
    _expect_value_error(
        lambda: SourceMemoryConstraint(
            operation=SourceMemoryOperation.LOAD,
            address_layout="compact",
            memory_spaces=("global",),
            element_byte_count=4,
            vector_lane_count=1,
            vector_lane_byte_stride=4,
            static_byte_offset=0,
        ),
        "source memory address layout must be a SourceMemoryAddressLayout",
    )


def test_source_memory_constraint_rejects_unused_address_layout_diagnostic() -> None:
    diagnostic = GuardDiagnostic(
        subject_role="value",
        subject_name="view",
        constraint_key="layout.compact_row_major",
    )
    _expect_value_error(
        lambda: SourceMemoryConstraint(
            operation=SourceMemoryOperation.LOAD,
            address_layout=SourceMemoryAddressLayout.ANY,
            address_layout_diagnostic=diagnostic,
            memory_spaces=("global",),
            element_byte_count=4,
            vector_lane_count=1,
            vector_lane_byte_stride=4,
            static_byte_offset=0,
        ),
        "unconstrained source memory cannot have an address-layout diagnostic",
    )


def test_compile_lower_rule_set_compiles_complete_source_memory_address() -> None:
    materializer = _i32_source_memory_address_materializer()

    compiled = compile_lower_rule_set(
        _source_memory_address_rule(materializer=materializer),
        dialect_ops={"vector": ALL_VECTOR_OPS},
    )

    emit = compiled.emits[0]
    value_refs = compiled.value_refs[
        emit.operand_ref_start : emit.operand_ref_start + emit.operand_ref_count
    ]
    assert tuple(value_ref.kind for value_ref in value_refs) == (
        SourceValueKind.SOURCE_MEMORY_ADDRESS,
    )
    source_memory = compiled.source_memories[emit.source_memory_ordinal - 1]
    assert source_memory.address_materializer is materializer


def test_complete_address_compiles_element_coordinate_policy() -> None:
    materializer = replace(
        _i32_source_memory_address_materializer(),
        base=SourceMemoryAddressBase.BASE_VIEW,
        coordinate_type=SourceMemoryAddressCoordinateType.INDEX,
        coordinate_unit_byte_count=4,
        coordinate_minimum=0,
        coordinate_maximum=(2**31) - 1,
        index_to_coordinate_input=None,
        index_to_coordinate=None,
    )

    compiled = compile_lower_rule_set(
        _source_memory_address_rule(materializer=materializer),
        dialect_ops={"vector": ALL_VECTOR_OPS},
    )

    source_memory = compiled.source_memories[0]
    assert source_memory.address_materializer is materializer
    assert materializer.base == SourceMemoryAddressBase.BASE_VIEW
    assert materializer.coordinate_type == SourceMemoryAddressCoordinateType.INDEX
    assert materializer.coordinate_unit_byte_count == 4
    assert materializer.coordinate_minimum == 0
    assert materializer.coordinate_maximum == (2**31) - 1


def test_complete_address_rejects_invalid_coordinate_policy() -> None:
    materializer = _i32_source_memory_address_materializer()

    _expect_value_error(
        lambda: replace(materializer, coordinate_unit_byte_count=0),
        "coordinate unit byte count must fit in a positive u32",
    )
    _expect_value_error(
        lambda: replace(
            materializer,
            coordinate_minimum=1,
            coordinate_maximum=0,
        ),
        "coordinate range is empty",
    )
    _expect_value_error(
        lambda: replace(materializer, coordinate_unit_byte_count=4),
        "offset-coordinate source-memory addresses use byte units",
    )
    _expect_value_error(
        lambda: replace(
            materializer,
            coordinate_type=SourceMemoryAddressCoordinateType.INDEX,
            index_to_coordinate=None,
        ),
        "index-coordinate source-memory addresses use the mapped index carrier",
    )
    _expect_value_error(
        lambda: replace(
            materializer,
            index_to_coordinate_input=None,
            index_to_coordinate=None,
        ),
        "offset-coordinate source-memory addresses need an index conversion",
    )


def test_compile_lower_rule_set_requires_complete_address_materializer() -> None:
    _expect_value_error(
        lambda: compile_lower_rule_set(
            _source_memory_address_rule(materializer=None),
            dialect_ops={"vector": ALL_VECTOR_OPS},
        ),
        "operand 'address' needs a source-memory address materializer",
    )


def test_complete_address_rejects_non_unary_index_conversion() -> None:
    materializer = replace(
        _i32_source_memory_address_materializer(),
        index_to_coordinate=TEST_LOW_ADD_I32_DESCRIPTOR,
    )

    _expect_value_error(
        lambda: compile_lower_rule_set(
            _source_memory_address_rule(materializer=materializer),
            dialect_ops={"vector": ALL_VECTOR_OPS},
        ),
        "source-memory address-coordinate index conversion descriptor "
        "'test.add.i32' "
        "must declare exactly 1 packet inputs",
    )


def test_complete_address_rejects_mixed_arithmetic_carriers() -> None:
    materializer = replace(
        _i32_source_memory_address_materializer(),
        add_coordinate=TEST_LOW_ADD_F32_DESCRIPTOR,
    )

    _expect_value_error(
        lambda: compile_lower_rule_set(
            _source_memory_address_rule(materializer=materializer),
            dialect_ops={"vector": ALL_VECTOR_OPS},
        ),
        "source-memory address-coordinate add descriptor 'test.add.f32' "
        "result does not use the materializer carrier",
    )


def test_source_memory_address_value_rejects_source_field() -> None:
    _expect_value_error(
        lambda: ValueRef(
            kind=SourceValueKind.SOURCE_MEMORY_ADDRESS,
            field="view",
        ).validate(vector.vector_load, "test value"),
        "source-memory address must not name a source field",
    )


def test_source_memory_address_value_rejects_element() -> None:
    _expect_value_error(
        lambda: ValueRef(
            kind=SourceValueKind.SOURCE_MEMORY_ADDRESS,
            field="",
            element=1,
        ).validate(vector.vector_load, "test value"),
        "source-memory address must not select an element",
    )


def test_compile_lower_rule_set_compiles_any_positive_dynamic_byte_offset() -> None:
    table = ContractFragment(
        name="test.source-memory-byte-offset",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_load,
                descriptor=TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR,
                guards=(Guard.value_type("result", Vector("i32", lanes=4)),),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR,
                        operands={
                            "address": ValueRef.operand("view"),
                            "index": ValueRef.source_memory_dynamic_byte_offset(),
                        },
                        results={"dst": ValueRef.result("result")},
                        source_memory=SourceMemoryConstraint(
                            operation=SourceMemoryOperation.LOAD,
                            memory_spaces=("unknown", "global"),
                            element_byte_count=4,
                            vector_lane_count=4,
                            vector_lane_byte_stride=4,
                            static_byte_offset_minimum=-(2**63),
                            static_byte_offset_maximum=(2**63) - 1,
                            dynamic_term_count=None,
                            dynamic_term_count_minimum=1,
                        ),
                        source_memory_byte_offset_materializer=(
                            SourceMemoryByteOffsetMaterializer(
                                const_i64=TEST_LOW_CONST_I32_DESCRIPTOR,
                                add_i64=TEST_LOW_ADD_I32_DESCRIPTOR,
                                mul_i64=TEST_LOW_MUL_I32_DESCRIPTOR,
                                shl_i64=None,
                                const_i64_immediate="i32_value",
                            )
                        ),
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    emit = compiled.emits[0]
    value_refs = compiled.value_refs[
        emit.operand_ref_start : emit.operand_ref_start + emit.operand_ref_count
    ]
    assert tuple(value_ref.kind for value_ref in value_refs) == (
        SourceValueKind.OPERAND,
        SourceValueKind.SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET,
    )
    source_memory = compiled.source_memories[emit.source_memory_ordinal - 1]
    assert source_memory.constraint.dynamic_term_count is None
    assert source_memory.constraint.dynamic_term_count_minimum == 1
    assert source_memory.constraint.dynamic_view_base_term_count is None


def test_compile_lower_rule_set_compiles_source_memory_static_offset_projects() -> None:
    descriptor = replace(
        TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR,
        key="test.load.index.v4i32.offsets",
        immediates=(
            Immediate("offset_plus", ImmediateKind.SIGNED, bit_width=32),
            Immediate("offset_quotient", ImmediateKind.SIGNED, bit_width=32),
            Immediate("offset_remainder", ImmediateKind.SIGNED, bit_width=32),
        ),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(*TEST_LOW_CORE_DESCRIPTOR_SET.descriptors, descriptor),
    )
    table = ContractFragment(
        name="test.source-memory-offset-projects",
        descriptor_set=descriptor_set,
        cases=[
            DescriptorRule(
                source_op=vector.vector_load,
                descriptor=descriptor,
                guards=(
                    Guard.operand_segment_count("indices", 0),
                    Guard.value_type("result", Vector("i32", lanes=4)),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=descriptor,
                        operands={
                            "address": ValueRef.operand("view"),
                            "index": ValueRef.source_memory_dynamic_term(),
                        },
                        results={"dst": ValueRef.result("result")},
                        immediates={
                            "offset_plus": SourceMemoryProject.static_byte_offset_plus(
                                64
                            ),
                            "offset_quotient": (
                                SourceMemoryProject.static_byte_offset_quotient(4)
                            ),
                            "offset_remainder": (
                                SourceMemoryProject.static_byte_offset_remainder(4)
                            ),
                        },
                        source_memory=SourceMemoryConstraint(
                            operation=SourceMemoryOperation.LOAD,
                            root_kind=SourceMemoryRootKind.ANY,
                            memory_spaces=("unknown", "global"),
                            element_byte_count=4,
                            vector_lane_count=4,
                            vector_lane_byte_stride=4,
                            static_byte_offset_minimum=-(2**31),
                            static_byte_offset_maximum=(2**31) - 1,
                            dynamic_term_count=1,
                            dynamic_index_source=SourceMemoryDynamicIndexSource.VALUE,
                            dynamic_byte_stride=16,
                        ),
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert tuple(attr_copy.kind for attr_copy in compiled.attr_copies) == (
        LowerAttrCopyKind.SOURCE_MEMORY_STATIC_BYTE_OFFSET_PLUS_LITERAL,
        LowerAttrCopyKind.SOURCE_MEMORY_STATIC_BYTE_OFFSET_QUOTIENT,
        LowerAttrCopyKind.SOURCE_MEMORY_STATIC_BYTE_OFFSET_REMAINDER,
    )
    assert tuple(attr_copy.literal_i64 for attr_copy in compiled.attr_copies) == (
        64,
        4,
        4,
    )

    source_emit = table.cases[0].emit[0]
    assert isinstance(source_emit, EmitDescriptorOp)
    assert source_emit.source_memory is not None
    overflowing_emit = replace(
        source_emit,
        source_memory=replace(
            source_emit.source_memory,
            static_byte_offset_minimum=-(2**63),
            static_byte_offset_maximum=(2**63) - 1,
        ),
    )
    _expect_value_error(
        lambda: replace(
            table,
            cases=(replace(table.cases[0], emit=(overflowing_emit,)),),
        ),
        "static byte offset range plus 64 must fit in signed i64",
    )


def test_compile_lower_rule_set_rejects_descriptor_rule_without_emit() -> None:
    table = ContractFragment(
        name="test.no-emit",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=scalar_arithmetic.scalar_addi,
                descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
            )
        ],
    )

    _expect_value_error(
        lambda: compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS}),
        "scalar.addi: descriptor-rule contracts must author their emit",
    )


def test_compile_lower_rule_set_compiles_const_immediate_emit() -> None:
    table = ContractFragment(
        name="test.immediate",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=scalar_conversion.scalar_constant,
                descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                guards=(Guard.value_type("result", Scalar("i32")),),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                        results={"dst": ValueRef.result("result")},
                        immediates={"i32_value": 0},
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert len(compiled.emits) == 1
    assert compiled.emits[0].kind == LowerEmitKind.DESCRIPTOR_CONST
    assert compiled.emits[0].attr_copy_count == 1
    assert len(compiled.attr_copies) == 1
    assert compiled.attr_copies[0].kind == LowerAttrCopyKind.I64_LITERAL
    assert compiled.attr_copies[0].literal_i64 == 0


def test_compile_lower_rule_set_keeps_operandless_op_emit() -> None:
    table = ContractFragment(
        name="test.operandless-op",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=scalar_conversion.scalar_constant,
                descriptor=TEST_LOW_AMBIGUOUS_DESCRIPTOR,
                guards=(Guard.value_type("result", Scalar("i32")),),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_AMBIGUOUS_DESCRIPTOR,
                        results={"dst": ValueRef.result("result")},
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert len(compiled.emits) == 1
    assert compiled.emits[0].kind == LowerEmitKind.DESCRIPTOR_OP


def test_contract_rejects_op_form_for_const_descriptor() -> None:
    _expect_value_error(
        lambda: ContractFragment(
            name="test.op-form-for-const",
            descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
            cases=[
                DescriptorRule(
                    source_op=scalar_conversion.scalar_constant,
                    descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                    guards=(Guard.value_type("result", Scalar("i32")),),
                    emit=(
                        EmitDescriptorOp(
                            descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                            results={"dst": ValueRef.result("result")},
                            immediates={"i32_value": 0},
                            form=DescriptorEmitForm.OP,
                        ),
                    ),
                )
            ],
        ),
        "scalar.constant: descriptor 'test.const.i32' uses low.const but the "
        "contract requests a low.op emission form",
    )


def test_contract_rejects_const_form_for_op_descriptor() -> None:
    _expect_value_error(
        lambda: ContractFragment(
            name="test.const-form-for-op",
            descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
            cases=[
                DescriptorRule(
                    source_op=scalar_conversion.scalar_constant,
                    descriptor=TEST_LOW_AMBIGUOUS_DESCRIPTOR,
                    guards=(Guard.value_type("result", Scalar("i32")),),
                    emit=(
                        EmitDescriptorOp(
                            descriptor=TEST_LOW_AMBIGUOUS_DESCRIPTOR,
                            results={"dst": ValueRef.result("result")},
                            form=DescriptorEmitForm.CONST,
                        ),
                    ),
                )
            ],
        ),
        "scalar.constant: descriptor 'test.ambiguous' uses low.op but the "
        "contract requests low.const",
    )


def test_compile_lower_rule_set_compiles_consecutive_i64_attr_pack() -> None:
    table = ContractFragment(
        name="test.attr-pack",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=scalar_bitwise.scalar_bitfield_extractu,
                descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                guards=(Guard.value_type("result", Scalar("i32")),),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                        results={"dst": ValueRef.result("result")},
                        immediates={
                            "i32_value": AttrProject.i64_attrs_pack_consecutive(
                                "offset",
                                count=2,
                                bit_width=8,
                            )
                        },
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert len(compiled.attr_copies) == 1
    attr_copy = compiled.attr_copies[0]
    assert attr_copy.kind == LowerAttrCopyKind.I64_ATTRS_PACK_CONSECUTIVE
    assert attr_copy.source_attr_index == 0
    assert attr_copy.source_element_count == 2
    assert attr_copy.source_element_bit_width == 8

    _expect_value_error(
        lambda: ContractFragment(
            name="test.bad-attr-pack",
            descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
            cases=[
                DescriptorRule(
                    source_op=scalar_bitwise.scalar_shli,
                    descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                    guards=(Guard.value_type("result", Scalar("i32")),),
                    emit=(
                        EmitDescriptorOp(
                            descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                            results={"dst": ValueRef.result("result")},
                            immediates={
                                "i32_value": AttrProject.i64_attrs_pack_consecutive(
                                    "overflow",
                                    count=1,
                                    bit_width=8,
                                )
                            },
                        ),
                    ),
                )
            ],
        ),
        "source attr 'overflow' must be an i64 attr",
    )


def test_compile_lower_rule_set_compiles_i64_bit_mask_attr_projection() -> None:
    descriptor = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        key="test.const.u32-mask",
        immediates=(
            Immediate("low", ImmediateKind.UNSIGNED, bit_width=32),
            Immediate("target", ImmediateKind.UNSIGNED, bit_width=32),
            Immediate("clear", ImmediateKind.UNSIGNED, bit_width=32),
            Immediate("shift", ImmediateKind.UNSIGNED, bit_width=8),
            Immediate("align", ImmediateKind.UNSIGNED, bit_width=8),
            Immediate("reverse_shift", ImmediateKind.SIGNED, bit_width=8),
        ),
    )
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(*TEST_LOW_CORE_DESCRIPTOR_SET.descriptors, descriptor),
    )
    table = ContractFragment(
        name="test.attr-bitmask",
        descriptor_set=descriptor_set,
        cases=[
            DescriptorRule(
                source_op=scalar_bitwise.scalar_bitfield_extractu,
                descriptor=descriptor,
                guards=(Guard.value_type("result", Scalar("i32")),),
                emit=(
                    EmitDescriptorOp(
                        descriptor=descriptor,
                        results={"dst": ValueRef.result("result")},
                        immediates={
                            "low": AttrProject.i64_low_bit_mask("width"),
                            "target": AttrProject.i64_shifted_low_bit_mask(
                                "width",
                                offset_attr="offset",
                            ),
                            "clear": AttrProject.i64_shifted_low_bit_clear_mask(
                                "width",
                                offset_attr="offset",
                            ),
                            "shift": AttrProject.i64_literal_minus_attr(
                                "width",
                                literal=32,
                            ),
                            "align": AttrProject.i64_literal_minus_attrs(
                                "offset",
                                other_source_attr="width",
                                literal=32,
                            ),
                            "reverse_shift": AttrProject.i64_attr_minus_literal(
                                "width",
                                literal=32,
                            ),
                        },
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert len(compiled.attr_copies) == 6
    low, target, clear, shift, align, reverse_shift = compiled.attr_copies
    assert low.kind == LowerAttrCopyKind.I64_LOW_BIT_MASK
    assert low.source_attr_index == 1
    assert target.kind == LowerAttrCopyKind.I64_SHIFTED_LOW_BIT_MASK
    assert target.source_attr_index == 1
    assert target.other_source_attr_index == 0
    assert clear.kind == LowerAttrCopyKind.I64_SHIFTED_LOW_BIT_CLEAR_MASK
    assert clear.source_attr_index == 1
    assert clear.other_source_attr_index == 0
    assert shift.kind == LowerAttrCopyKind.I64_LITERAL_MINUS_ATTR
    assert shift.source_attr_index == 1
    assert shift.literal_i64 == 32
    assert align.kind == LowerAttrCopyKind.I64_LITERAL_MINUS_ATTRS
    assert align.source_attr_index == 0
    assert align.other_source_attr_index == 1
    assert align.literal_i64 == 32
    assert reverse_shift.kind == LowerAttrCopyKind.I64_ATTR_MINUS_LITERAL
    assert reverse_shift.source_attr_index == 1
    assert reverse_shift.literal_i64 == 32

    _expect_value_error(
        lambda: ContractFragment(
            name="test.bad-attr-bitmask",
            descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
            cases=[
                DescriptorRule(
                    source_op=scalar_bitwise.scalar_bitfield_extractu,
                    descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                    guards=(Guard.value_type("result", Scalar("i32")),),
                    emit=(
                        EmitDescriptorOp(
                            descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                            results={"dst": ValueRef.result("result")},
                            immediates={
                                "i32_value": AttrProject.i64_low_bit_mask("width")
                            },
                        ),
                    ),
                )
            ],
        ),
        "descriptor immediate 'i32_value' must be an unsigned immediate",
    )


def test_compile_lower_rule_set_validates_enum_immediate_literal() -> None:
    immediate = Immediate(
        "mode",
        ImmediateKind.ENUM,
        bit_width=8,
        enum_domain="test.enum_mode",
    )
    descriptor = replace(TEST_LOW_CONST_I32_DESCRIPTOR, immediates=(immediate,))
    descriptor_set = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=tuple(
            descriptor
            if existing_descriptor == TEST_LOW_CONST_I32_DESCRIPTOR
            else existing_descriptor
            for existing_descriptor in TEST_LOW_CORE_DESCRIPTOR_SET.descriptors
        ),
        enum_domains=(
            *TEST_LOW_CORE_DESCRIPTOR_SET.enum_domains,
            EnumDomain("test.enum_mode", values=(EnumValue("seven", 7),)),
        ),
    )

    table = ContractFragment(
        name="test.enum-immediate",
        descriptor_set=descriptor_set,
        cases=[
            DescriptorRule(
                source_op=scalar_conversion.scalar_constant,
                descriptor=descriptor,
                guards=(Guard.value_type("result", Scalar("i32")),),
                emit=(
                    EmitDescriptorOp(
                        descriptor=descriptor,
                        results={"dst": ValueRef.result("result")},
                        immediates={"mode": 7},
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert len(compiled.attr_copies) == 1
    assert compiled.attr_copies[0].kind == LowerAttrCopyKind.I64_LITERAL
    assert compiled.attr_copies[0].literal_i64 == 7

    _expect_value_error(
        lambda: ContractFragment(
            name="test.bad-enum-immediate",
            descriptor_set=descriptor_set,
            cases=[
                DescriptorRule(
                    source_op=scalar_conversion.scalar_constant,
                    descriptor=descriptor,
                    guards=(Guard.value_type("result", Scalar("i32")),),
                    emit=(
                        EmitDescriptorOp(
                            descriptor=descriptor,
                            results={"dst": ValueRef.result("result")},
                            immediates={"mode": 5},
                        ),
                    ),
                )
            ],
        ),
        "literal 5 is not in enum domain 'test.enum_mode'",
    )


def test_compile_lower_rule_set_compiles_instance_flags_guard() -> None:
    table = ContractFragment(
        name="test.flags",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=scalar_arithmetic.scalar_divf,
                descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                guards=(
                    Guard.instance_flags_has_all("fastmath", "arcp"),
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

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert compiled.rules[0].guard_count == 4
    assert compiled.guards[0].kind == GuardKind.INSTANCE_FLAGS_HAS_ALL
    assert compiled.guards[0].u64 == 16


def test_compile_lower_rule_set_projects_source_instance_flags() -> None:
    descriptor, descriptor_set = _add_f32_flags_descriptor_set()
    table = ContractFragment(
        name="test.flags",
        descriptor_set=descriptor_set,
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
                        immediates={
                            "fast_math_flags": SourceOpProject.instance_flags()
                        },
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert compiled.emits[0].attr_copy_count == 1
    assert len(compiled.attr_copies) == 1
    assert compiled.attr_copies[0].kind == LowerAttrCopyKind.SOURCE_OP_INSTANCE_FLAGS


def test_compile_lower_rule_set_compiles_float_equals_guard() -> None:
    table = ContractFragment(
        name="test.f64-equals",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=scalar_arithmetic.scalar_mulf,
                descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
                guards=(
                    Guard.value_float_equals("lhs", 1.0),
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

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert compiled.rules[0].guard_count == 4
    assert compiled.guards[0].kind == GuardKind.VALUE_FLOAT_EQUALS
    assert compiled.guards[0].value_ref_index == 0
    assert compiled.guards[0].u64 == 0x3FF0000000000000


def test_compile_lower_rule_set_compiles_storage_element_format_guard() -> None:
    table = ContractFragment(
        name="test.storage-schema",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=vector.vector_fragment_load,
                descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                guards=(
                    Guard.value_storage_element_format(
                        "view",
                        "LOOM_VALUE_FACT_NUMERIC_FORMAT_U8",
                    ),
                    Guard.value_type("result", Vector("i32", lanes=4)),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
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

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert compiled.guards[0].kind == GuardKind.VALUE_STORAGE_ELEMENT_FORMAT
    assert compiled.guards[0].value_ref_index == 0
    assert compiled.guards[0].u64_c_expression == "LOOM_VALUE_FACT_NUMERIC_FORMAT_U8"


def test_compile_lower_rule_set_compiles_value_memory_space_guard() -> None:
    table = ContractFragment(
        name="test.value-memory-space",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            RecipeRule(
                source_op=vector.vector_fragment_load,
                guards=(
                    Guard.value_memory_space(
                        "view",
                        ("unknown", "generic", "global", "descriptor"),
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert compiled.guards[0].kind == GuardKind.VALUE_MEMORY_SPACE
    assert compiled.guards[0].value_ref_index == 0
    assert compiled.guards[0].memory_spaces == (
        "unknown",
        "generic",
        "global",
        "descriptor",
    )


def test_compile_lower_rule_set_compiles_packed_integer_storage_guards() -> None:
    table = ContractFragment(
        name="test.packed-integer-storage",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            RecipeRule(
                source_op=vector.vector_bitpack,
                guards=(
                    Guard.value_packed_integer_payload_from_lanes(
                        "source",
                        "result",
                        "width",
                        storage_unit_bit_count=32,
                        storage_payload_multiple=32,
                    ),
                ),
            ),
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
            ),
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert compiled.rules[0].flags == LOWER_RULE_FLAG_CONTRACT_ONLY
    assert compiled.guards[0].kind == GuardKind.VALUE_PACKED_INTEGER_PAYLOAD_FROM_LANES
    assert compiled.guards[0].attr_index == 0
    assert compiled.guards[0].u64 == 32
    assert compiled.guards[0].minimum_i64 == 32
    assert compiled.rules[1].flags == LOWER_RULE_FLAG_CONTRACT_ONLY
    assert compiled.guards[1].kind == GuardKind.VALUE_PACKED_INTEGER_LANES_FROM_PAYLOAD
    assert compiled.guards[1].attr_index == 0
    assert compiled.guards[1].u64 == 16
    assert compiled.guards[1].minimum_i64 == 32
    assert compiled.guards[1].maximum_i64 == 32


def test_compile_lower_rule_set_compiles_value_range_relation_guard() -> None:
    table = ContractFragment(
        name="test.value-range-relation",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=scalar_arithmetic.scalar_minsi,
                descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                guards=(
                    Guard.value_i64_range_le("lhs", "rhs"),
                    Guard.value_type("lhs", Scalar("i32")),
                    Guard.value_type("rhs", Scalar("i32")),
                    Guard.value_type("result", Scalar("i32")),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
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

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert compiled.rules[0].guard_count == 4
    assert compiled.guards[0].kind == GuardKind.VALUE_I64_RANGE_LE
    assert compiled.guards[0].value_ref_index == 0
    assert compiled.guards[0].other_value_ref_index == 1


def test_compile_lower_rule_set_compiles_static_element_count_relation_guard() -> None:
    table = ContractFragment(
        name="test.static-element-count-relation",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            RecipeRule(
                source_op=vector.vector_extf,
                guards=(
                    Guard.value_static_element_count_eq("input", "result"),
                    Guard.value_type("input", Vector("f16", lanes=4)),
                    Guard.value_type("result", Vector("f32", lanes=4)),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"vector": ALL_VECTOR_OPS})

    assert compiled.rules[0].flags == LOWER_RULE_FLAG_CONTRACT_ONLY
    assert compiled.rules[0].guard_count == 3
    assert compiled.guards[0].kind == GuardKind.VALUE_STATIC_ELEMENT_COUNT_EQ
    assert compiled.guards[0].value_ref_index == 0
    assert compiled.guards[0].other_value_ref_index == 1


def test_compile_lower_rule_set_compiles_value_fact_immediate_emit() -> None:
    table = ContractFragment(
        name="test.value-immediate",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=scalar_arithmetic.scalar_addi,
                descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                guards=(Guard.value_type("result", Scalar("i32")),),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                        results={"dst": ValueRef.result("result")},
                        immediates={
                            "i32_value": ValueProject.i32_as_u32_bits("lhs"),
                        },
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert len(compiled.attr_copies) == 1
    assert compiled.attr_copies[0].kind == LowerAttrCopyKind.VALUE_I32_AS_U32_BITS
    value_ref = compiled.value_refs[compiled.attr_copies[0].value_ref_index]
    assert value_ref.index == 0


def test_compile_lower_rule_set_compiles_power_of_two_log2_immediate() -> None:
    table = ContractFragment(
        name="test.power-of-two-log2",
        descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
        cases=[
            DescriptorRule(
                source_op=scalar_arithmetic.scalar_addi,
                descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                guards=(
                    Guard.value_exact_power_of_two_i64("lhs"),
                    Guard.value_type("result", Scalar("i32")),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                        results={"dst": ValueRef.result("result")},
                        immediates={
                            "i32_value": ValueProject.exact_i64_log2("lhs"),
                        },
                    ),
                ),
            )
        ],
    )

    compiled = compile_lower_rule_set(table, dialect_ops={"scalar": ALL_SCALAR_OPS})

    assert compiled.guards[0].kind == GuardKind.VALUE_EXACT_POWER_OF_TWO_I64
    assert compiled.guards[0].value_ref_index == 0
    assert len(compiled.attr_copies) == 1
    assert compiled.attr_copies[0].kind == LowerAttrCopyKind.VALUE_EXACT_I64_LOG2
    value_ref = compiled.value_refs[compiled.attr_copies[0].value_ref_index]
    assert value_ref.index == 0
