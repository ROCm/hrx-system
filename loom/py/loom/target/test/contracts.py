# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target contract fragments for the backend-independent test-low target."""

from __future__ import annotations

from loom.dialect.buffer import ALL_BUFFER_OPS
from loom.dialect.buffer import defs as buffer
from loom.dialect.index import ALL_INDEX_OPS
from loom.dialect.index import defs as index
from loom.dialect.scalar import ALL_SCALAR_OPS
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import ALL_VECTOR_OPS
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.contracts import (
    AttrProject,
    ContractFragment,
    DescriptorEmitForm,
    DescriptorMatrixRule,
    DescriptorRule,
    DirectDescriptorCase,
    DirectTypePatterns,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    PredicateDescriptorCase,
    Scalar,
    SelectDescriptorCase,
    SourceMemoryConstraint,
    SourceMemoryDynamicIndexSource,
    SourceMemoryOperation,
    TypePattern,
    ValueAliasRule,
    ValueRef,
    Vector,
    binary_descriptor_rules,
    compare_descriptor_rules,
    select_descriptor_rules,
)
from loom.target.low_descriptors import Descriptor
from loom.target.test.descriptors import (
    TEST_LOW_ADD_F32_DESCRIPTOR,
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_CMP_EQ_I32_DESCRIPTOR,
    TEST_LOW_CONST_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
    TEST_LOW_DOT4I_S8S8_DESCRIPTOR,
    TEST_LOW_FROM_ELEMENTS_V4I32_DESCRIPTOR,
    TEST_LOW_LOAD_INDEX_V4F32_DESCRIPTOR,
    TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR,
    TEST_LOW_LOAD_V4F32_DESCRIPTOR,
    TEST_LOW_LOAD_V4I32_DESCRIPTOR,
    TEST_LOW_MUL_F32_DESCRIPTOR,
    TEST_LOW_MUL_I32_DESCRIPTOR,
    TEST_LOW_SELECT_I32_DESCRIPTOR,
    TEST_LOW_SHUFFLE_V4I32_DESCRIPTOR,
    TEST_LOW_STORE_INDEX_V4F32_DESCRIPTOR,
    TEST_LOW_STORE_INDEX_V4I32_DESCRIPTOR,
    TEST_LOW_STORE_V4F32_DESCRIPTOR,
    TEST_LOW_STORE_V4I32_DESCRIPTOR,
    TEST_LOW_SUB_F32_DESCRIPTOR,
    TEST_LOW_TIED_ANY_DESCRIPTOR,
)


def _binary_rule(
    source_op: Op,
    descriptor: Descriptor,
    type_patterns: DirectTypePatterns,
    *,
    semantic_tag: str | None = None,
) -> DescriptorRule:
    return binary_descriptor_rules(
        (
            DirectDescriptorCase(
                source_op,
                descriptor,
                type_patterns,
                semantic_tag=semantic_tag,
            ),
        )
    )[0]


def _select_rule(
    source_op: Op,
    descriptor: Descriptor,
    *,
    condition_type: TypePattern,
    value_type: TypePattern,
    semantic_tag: str | None = None,
) -> DescriptorRule:
    return select_descriptor_rules(
        (
            SelectDescriptorCase(
                source_op,
                descriptor,
                condition_type=condition_type,
                value_type=value_type,
                semantic_tag=semantic_tag,
            ),
        )
    )[0]


def _compare_rule(
    source_op: Op,
    descriptor: Descriptor,
    *,
    predicate: str,
    operand_type: TypePattern,
    result_type: TypePattern,
    semantic_tag: str | None = None,
) -> DescriptorRule:
    return compare_descriptor_rules(
        source_op,
        (
            PredicateDescriptorCase(
                predicate,
                descriptor,
                semantic_tag=semantic_tag,
            ),
        ),
        operand_type=operand_type,
        result_type=result_type,
    )[0]


_I32 = Scalar("i32")
_F32 = Scalar("f32")
_INDEX = Scalar("index")
_OFFSET = Scalar("offset")
_V4I1 = Vector("i1", lanes=4)
_V16I8 = Vector("i8", lanes=16)
_V4I32 = Vector("i32", lanes=4)
_V4F32 = Vector("f32", lanes=4)


TEST_LOW_CORE_CONTRACT_DIALECT_OPS = {
    "buffer": ALL_BUFFER_OPS,
    "index": ALL_INDEX_OPS,
    "scalar": ALL_SCALAR_OPS,
    "vector": ALL_VECTOR_OPS,
}


def _const_i32_rule(source_op: Op, result_type: TypePattern) -> DescriptorRule:
    return DescriptorRule(
        source_op=source_op,
        descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
        guards=(
            Guard.attr_kind("value", "i64"),
            Guard.value_type("result", result_type),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                results={"dst": ValueRef.result("result")},
                immediates={"i32_value": AttrProject.direct("value")},
                form=DescriptorEmitForm.CONST,
            ),
        ),
    )


def _vector_reduce_rule(
    *,
    kind: str,
    input_type: TypePattern,
    accumulator_type: TypePattern,
    descriptor: Descriptor,
) -> DescriptorRule:
    return DescriptorRule(
        source_op=vector.vector_reduce,
        descriptor=descriptor,
        guards=(
            Guard.enum_attr_equals("kind", kind),
            Guard.value_type("input", input_type),
            Guard.value_type("init", accumulator_type),
            Guard.value_type("result", accumulator_type),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "lhs": ValueRef.operand("input"),
                    "rhs": ValueRef.operand("init"),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.ACCUMULATE_LANES,
                accumulator="rhs",
            ),
        ),
    )


_TEST_LOW_SOURCE_MEMORY_DIAGNOSTIC = GuardDiagnostic(
    subject_role="source-memory",
    subject_name="test-low",
    constraint_key="test.source_memory.v4i32",
)


def _source_memory_constraint(
    operation: SourceMemoryOperation,
    *,
    dynamic: bool,
) -> SourceMemoryConstraint:
    return SourceMemoryConstraint(
        operation=operation,
        memory_spaces=("unknown", "generic", "global"),
        element_byte_count=4,
        vector_lane_count=4,
        vector_lane_byte_stride=4,
        static_byte_offset=0,
        dynamic_term_count=1 if dynamic else 0,
        dynamic_index_source=(
            SourceMemoryDynamicIndexSource.VALUE
            if dynamic
            else SourceMemoryDynamicIndexSource.NONE
        ),
        dynamic_byte_stride=4 if dynamic else 0,
        diagnostic=_TEST_LOW_SOURCE_MEMORY_DIAGNOSTIC,
    )


def _vector_load_rule(
    descriptor: Descriptor,
    result_type: TypePattern,
    *,
    dynamic: bool,
) -> DescriptorRule:
    operands = {"address": ValueRef.operand("view")}
    if dynamic:
        operands["index"] = ValueRef.operand("indices")
    return DescriptorRule(
        source_op=vector.vector_load,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", 1 if dynamic else 0),
            Guard.value_type("result", result_type),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands=operands,
                results={"dst": ValueRef.result("result")},
                source_memory=_source_memory_constraint(
                    SourceMemoryOperation.LOAD,
                    dynamic=dynamic,
                ),
            ),
        ),
    )


def _vector_store_rule(
    descriptor: Descriptor,
    value_type: TypePattern,
    *,
    dynamic: bool,
) -> DescriptorRule:
    operands = {
        "address": ValueRef.operand("view"),
        "value": ValueRef.operand("value"),
    }
    if dynamic:
        operands["index"] = ValueRef.operand("indices")
    return DescriptorRule(
        source_op=vector.vector_store,
        descriptor=descriptor,
        guards=(
            Guard.operand_segment_count("indices", 1 if dynamic else 0),
            Guard.value_type("value", value_type),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=descriptor,
                operands=operands,
                source_memory=_source_memory_constraint(
                    SourceMemoryOperation.STORE,
                    dynamic=dynamic,
                ),
            ),
        ),
    )


TEST_LOW_CORE_CONTRACT_FRAGMENT = ContractFragment(
    name="test.low.core",
    descriptor_set=TEST_LOW_CORE_DESCRIPTOR_SET,
    cases=[
        ValueAliasRule(
            source_op=buffer.buffer_view,
            source=ValueRef.operand("buffer"),
            result=ValueRef.result("result"),
            guards=(
                Guard.value_i64_range(
                    "byte_offset",
                    0,
                    0,
                    diagnostic=GuardDiagnostic(
                        subject_role="value",
                        subject_name="byte_offset",
                        constraint_key="test.buffer_view.zero_offset",
                    ),
                ),
            ),
        ),
        _binary_rule(
            scalar_arithmetic.scalar_addi,
            TEST_LOW_ADD_I32_DESCRIPTOR,
            _I32,
            semantic_tag="integer.add.i32",
        ),
        _binary_rule(
            scalar_arithmetic.scalar_muli,
            TEST_LOW_MUL_I32_DESCRIPTOR,
            _I32,
            semantic_tag="integer.mul.i32",
        ),
        _binary_rule(
            scalar_arithmetic.scalar_addf,
            TEST_LOW_ADD_F32_DESCRIPTOR,
            _F32,
            semantic_tag="float.add.f32",
        ),
        _binary_rule(
            scalar_arithmetic.scalar_subf,
            TEST_LOW_SUB_F32_DESCRIPTOR,
            _F32,
            semantic_tag="float.sub.f32",
        ),
        _binary_rule(
            scalar_arithmetic.scalar_mulf,
            TEST_LOW_MUL_F32_DESCRIPTOR,
            _F32,
            semantic_tag="float.mul.f32",
        ),
        _select_rule(
            vector.vector_select,
            TEST_LOW_SELECT_I32_DESCRIPTOR,
            condition_type=_V4I1,
            value_type=_V4I32,
            semantic_tag="integer.select.i32",
        ),
        _compare_rule(
            vector.vector_cmpi,
            TEST_LOW_CMP_EQ_I32_DESCRIPTOR,
            predicate="eq",
            operand_type=_V4I32,
            result_type=_V4I1,
            semantic_tag="integer.cmp.eq.i32",
        ),
        _binary_rule(
            vector.vector_addf,
            TEST_LOW_ADD_F32_DESCRIPTOR,
            _V4F32,
            semantic_tag="float.add.f32",
        ),
        _binary_rule(
            vector.vector_subf,
            TEST_LOW_SUB_F32_DESCRIPTOR,
            _V4F32,
            semantic_tag="float.sub.f32",
        ),
        _binary_rule(
            vector.vector_mulf,
            TEST_LOW_MUL_F32_DESCRIPTOR,
            _V4F32,
            semantic_tag="float.mul.f32",
        ),
        _binary_rule(
            vector.vector_addi,
            TEST_LOW_ADD_I32_DESCRIPTOR,
            _V4I32,
            semantic_tag="integer.add.i32",
        ),
        _binary_rule(
            vector.vector_muli,
            TEST_LOW_MUL_I32_DESCRIPTOR,
            _V4I32,
            semantic_tag="integer.mul.i32",
        ),
        _vector_load_rule(
            TEST_LOW_LOAD_V4I32_DESCRIPTOR,
            _V4I32,
            dynamic=False,
        ),
        _vector_load_rule(
            TEST_LOW_LOAD_V4F32_DESCRIPTOR,
            _V4F32,
            dynamic=False,
        ),
        _vector_load_rule(
            TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR,
            _V4I32,
            dynamic=True,
        ),
        _vector_load_rule(
            TEST_LOW_LOAD_INDEX_V4F32_DESCRIPTOR,
            _V4F32,
            dynamic=True,
        ),
        _vector_store_rule(
            TEST_LOW_STORE_V4I32_DESCRIPTOR,
            _V4I32,
            dynamic=False,
        ),
        _vector_store_rule(
            TEST_LOW_STORE_V4F32_DESCRIPTOR,
            _V4F32,
            dynamic=False,
        ),
        _vector_store_rule(
            TEST_LOW_STORE_INDEX_V4I32_DESCRIPTOR,
            _V4I32,
            dynamic=True,
        ),
        _vector_store_rule(
            TEST_LOW_STORE_INDEX_V4F32_DESCRIPTOR,
            _V4F32,
            dynamic=True,
        ),
        DescriptorRule(
            source_op=scalar_arithmetic.scalar_subi,
            descriptor=TEST_LOW_TIED_ANY_DESCRIPTOR,
            guards=(
                Guard.value_type("lhs", _I32),
                Guard.value_type("rhs", _I32),
                Guard.value_type("result", _I32),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=TEST_LOW_TIED_ANY_DESCRIPTOR,
                    operands={"src": ValueRef.operand("lhs")},
                    results={"dst": ValueRef.result("result")},
                ),
            ),
        ),
        _const_i32_rule(scalar_conversion.scalar_constant, _I32),
        _const_i32_rule(index.index_constant, _INDEX),
        _const_i32_rule(index.index_constant, _OFFSET),
        DescriptorRule(
            source_op=vector.vector_extract,
            descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
            guards=(
                Guard.i64_array_count("static_indices", 1),
                Guard.i64_array_element_range("static_indices", 0, 0, 3),
                Guard.value_type("source", _V4I32),
                Guard.value_type("result", _I32),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=TEST_LOW_CONST_I32_DESCRIPTOR,
                    results={"dst": ValueRef.result("result")},
                    immediates={
                        "i32_value": AttrProject.i64_array_element(
                            "static_indices",
                            element=0,
                        )
                    },
                    form=DescriptorEmitForm.CONST,
                ),
            ),
        ),
        DescriptorRule(
            source_op=vector.vector_shuffle,
            descriptor=TEST_LOW_SHUFFLE_V4I32_DESCRIPTOR,
            guards=(
                Guard.i64_array_count("source_lanes", 4),
                Guard.i64_array_elements_range("source_lanes", 0, 3),
                Guard.value_type("source", _V4I32),
                Guard.value_type("result", _V4I32),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=TEST_LOW_SHUFFLE_V4I32_DESCRIPTOR,
                    operands={"source": ValueRef.operand("source")},
                    results={"dst": ValueRef.result("result")},
                    immediates={
                        "shuffle_control": AttrProject.i64_array_pack_elements(
                            "source_lanes",
                            element=0,
                            count=4,
                            bit_width=2,
                        )
                    },
                ),
            ),
        ),
        DescriptorRule(
            source_op=vector.vector_from_elements,
            descriptor=TEST_LOW_FROM_ELEMENTS_V4I32_DESCRIPTOR,
            guards=(
                Guard.operand_segment_count("elements", 4),
                Guard.value_type("result", _V4I32),
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
        ),
        DescriptorRule(
            source_op=vector.vector_subi,
            descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
            guards=(
                Guard.value_type("lhs", _V4I32),
                Guard.value_static_dim0_multiple("lhs", 4),
                Guard.value_type("rhs", _V4I32),
                Guard.low_value_register_unit_count_eq("lhs", "rhs"),
                Guard.value_type("result", _V4I32),
                Guard.low_value_register_unit_count_eq("lhs", "result"),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                    operands={
                        "lhs": ValueRef.operand("lhs"),
                        "rhs": ValueRef.operand("rhs"),
                    },
                    results={"dst": ValueRef.result("result")},
                    swap_first_two_operands=True,
                ),
            ),
        ),
        _vector_reduce_rule(
            kind="addi",
            input_type=_V4I32,
            accumulator_type=_I32,
            descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
        ),
        _vector_reduce_rule(
            kind="addf",
            input_type=_V4F32,
            accumulator_type=_F32,
            descriptor=TEST_LOW_ADD_F32_DESCRIPTOR,
        ),
        DescriptorMatrixRule(
            source_op=vector.vector_mma,
            source="vector_mma",
        ),
        DescriptorRule(
            source_op=vector.vector_dot4i,
            descriptor=TEST_LOW_DOT4I_S8S8_DESCRIPTOR,
            guards=(
                Guard.enum_attr_equals("kind", "s8s8"),
                Guard.value_type("lhs", _V16I8),
                Guard.value_type("rhs", _V16I8),
                Guard.value_type("acc", _V4I32),
                Guard.value_type("result", _V4I32),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=TEST_LOW_DOT4I_S8S8_DESCRIPTOR,
                    operands={
                        "lhs": ValueRef.operand("lhs"),
                        "rhs": ValueRef.operand("rhs"),
                        "acc": ValueRef.operand("acc"),
                    },
                    results={"dst": ValueRef.result("result")},
                ),
            ),
        ),
        DescriptorRule(
            source_op=index.index_madd,
            descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
            guards=(
                Guard.value_type("a", _INDEX),
                Guard.value_type("b", _INDEX),
                Guard.value_type("c", _INDEX),
                Guard.value_type("result", _INDEX),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                    operands={
                        "lhs": ValueRef.operand("a"),
                        "rhs": ValueRef.operand("b"),
                    },
                    results={"dst": ValueRef.temporary("product")},
                    result_types={"dst": ValueRef.result("result")},
                ),
                EmitDescriptorOp(
                    descriptor=TEST_LOW_ADD_I32_DESCRIPTOR,
                    operands={
                        "lhs": ValueRef.temporary("product"),
                        "rhs": ValueRef.operand("c"),
                    },
                    results={"dst": ValueRef.result("result")},
                    copy_operands=("rhs",),
                ),
            ),
        ),
    ],
)
