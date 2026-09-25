# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import replace

import pytest

from loom.dialect.vector import defs as vector
from loom.target.contracts import (
    EmitDescriptorOp,
    SourceMemoryAddressMaterializer,
    SourceMemoryByteOffsetMaterializer,
    SourceMemoryConstraint,
    SourceMemoryIntegerConversion,
    SourceMemoryOperation,
    ValueRef,
)
from loom.target.low_descriptors import DescriptorOpKind, Immediate, ImmediateKind
from loom.target.test.descriptors import (
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_CONST_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
    TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR,
    TEST_LOW_LOAD_V4I32_DESCRIPTOR,
    TEST_LOW_MUL_I32_DESCRIPTOR,
    TEST_LOW_REMATERIALIZE_I32_DESCRIPTOR,
    TEST_LOW_SELECT_I32_DESCRIPTOR,
)


def _validate(
    conversions,
    *,
    descriptors=TEST_LOW_CORE_DESCRIPTOR_SET,
    index_ref=None,
    multiply_add=None,
    static_bias=None,
):
    if index_ref is None:
        index_ref = ValueRef.source_memory_dynamic_byte_offset()
    emit = EmitDescriptorOp(
        descriptor=TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR,
        operands={
            "address": ValueRef.operand("view"),
            "index": index_ref,
        },
        results={"dst": ValueRef.result("result")},
        source_memory=SourceMemoryConstraint(
            operation=SourceMemoryOperation.LOAD,
            memory_spaces=("global",),
            element_byte_count=4,
            vector_lane_count=4,
            vector_lane_byte_stride=4,
            static_byte_offset=0,
            dynamic_term_count=None,
            dynamic_term_count_minimum=1,
        ),
        source_memory_byte_offset_materializer=SourceMemoryByteOffsetMaterializer(
            constant=TEST_LOW_CONST_I32_DESCRIPTOR,
            add=TEST_LOW_ADD_I32_DESCRIPTOR,
            multiply=TEST_LOW_MUL_I32_DESCRIPTOR,
            shift_left=None,
            multiply_add=multiply_add,
            static_bias=static_bias,
            constant_immediate="i32_value",
            integer_conversions=conversions,
        ),
    )
    emit.validate(vector.vector_load, descriptors, set())


def test_complete_byte_offset_validates_fused_arithmetic_descriptors():
    multiply_add = replace(
        TEST_LOW_ADD_I32_DESCRIPTOR,
        key="test.multiply_add.i32",
        mnemonic="test.multiply_add.i32",
        semantic_tag="integer.multiply_add.i32",
        operands=(
            TEST_LOW_ADD_I32_DESCRIPTOR.operands[0],
            replace(TEST_LOW_ADD_I32_DESCRIPTOR.operands[1], field_name="accumulator"),
            TEST_LOW_ADD_I32_DESCRIPTOR.operands[1],
            TEST_LOW_ADD_I32_DESCRIPTOR.operands[2],
        ),
        asm_forms=(),
    )
    static_bias = replace(
        TEST_LOW_CONST_I32_DESCRIPTOR,
        key="test.static_bias.i32",
        mnemonic="test.static_bias.i32",
        semantic_tag="integer.static_bias.i32",
        op_kind=DescriptorOpKind.OP,
    )
    descriptors = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(
            *TEST_LOW_CORE_DESCRIPTOR_SET.descriptors,
            multiply_add,
            static_bias,
        ),
    )
    _validate(
        (),
        descriptors=descriptors,
        index_ref=ValueRef.source_memory_byte_offset(),
        multiply_add=multiply_add,
        static_bias=static_bias,
    )

    with pytest.raises(ValueError, match="exactly 3 packet inputs"):
        _validate((), multiply_add=TEST_LOW_ADD_I32_DESCRIPTOR)
    with pytest.raises(ValueError, match=r"must use low\.op"):
        _validate((), static_bias=TEST_LOW_CONST_I32_DESCRIPTOR)


def test_integer_conversion_domains_are_unique_fixed_width_integers():
    conversion = SourceMemoryIntegerConversion(
        "i8", TEST_LOW_REMATERIALIZE_I32_DESCRIPTOR
    )
    _validate((conversion,))
    with pytest.raises(ValueError, match="duplicate address conversion"):
        _validate((conversion, conversion))
    for source_type in ("index", "offset", "f32", "i128"):
        with pytest.raises(ValueError, match="requires a fixed-width integer"):
            _validate((replace(conversion, source_type=source_type),))


def test_integer_conversion_requires_unary_instruction_and_matching_carrier():
    for descriptor, message in (
        (TEST_LOW_CONST_I32_DESCRIPTOR, "must use low.op"),
        (TEST_LOW_ADD_I32_DESCRIPTOR, "exactly 1 packet inputs"),
    ):
        with pytest.raises(ValueError, match=message):
            _validate((SourceMemoryIntegerConversion("i8", descriptor),))
    descriptor = TEST_LOW_REMATERIALIZE_I32_DESCRIPTOR
    wide = replace(
        descriptor,
        key="test.convert.wide",
        operands=(
            replace(descriptor.operands[0], unit_count=2),
            *descriptor.operands[1:],
        ),
    )
    descriptors = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(*TEST_LOW_CORE_DESCRIPTOR_SET.descriptors, wide),
    )
    with pytest.raises(
        ValueError, match="result does not use the materializer carrier"
    ):
        _validate((SourceMemoryIntegerConversion("i8", wide),), descriptors=descriptors)


def test_integer_conversion_selector_is_bound_and_range_checked():
    descriptor = replace(
        TEST_LOW_REMATERIALIZE_I32_DESCRIPTOR,
        key="test.convert.selected",
        immediates=(
            Immediate(
                "selector", ImmediateKind.UNSIGNED, bit_width=8, unsigned_max=255
            ),
        ),
    )
    descriptors = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(*TEST_LOW_CORE_DESCRIPTOR_SET.descriptors, descriptor),
    )
    conversion = SourceMemoryIntegerConversion("i8", descriptor, ("selector", 255))
    _validate((conversion,), descriptors=descriptors)
    with pytest.raises(ValueError, match="additional required immediates"):
        _validate((replace(conversion, immediate=None),), descriptors=descriptors)
    for value in (-1, 256):
        with pytest.raises(ValueError, match="out of range"):
            _validate(
                (replace(conversion, immediate=("selector", value)),),
                descriptors=descriptors,
            )


def test_integer_conversion_features_fit_the_runtime_feature_word():
    descriptor = replace(
        TEST_LOW_REMATERIALIZE_I32_DESCRIPTOR,
        key="test.convert.wide_features",
        feature_mask_words=(0, 1),
    )
    descriptors = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(*TEST_LOW_CORE_DESCRIPTOR_SET.descriptors, descriptor),
    )
    with pytest.raises(ValueError, match="features must fit in one u64"):
        _validate(
            (SourceMemoryIntegerConversion("i8", descriptor),), descriptors=descriptors
        )


def test_predicate_conversion_selects_numeric_carrier_values():
    conversion = SourceMemoryIntegerConversion("i1", TEST_LOW_SELECT_I32_DESCRIPTOR)
    assert conversion.input_count == 3
    _validate((conversion,))
    with pytest.raises(ValueError, match="exactly 1 packet inputs"):
        _validate((replace(conversion, source_type="i8"),))
    descriptor = replace(
        TEST_LOW_SELECT_I32_DESCRIPTOR,
        key="test.select.wide_true",
        operands=(
            *TEST_LOW_SELECT_I32_DESCRIPTOR.operands[:2],
            replace(TEST_LOW_SELECT_I32_DESCRIPTOR.operands[2], unit_count=2),
            TEST_LOW_SELECT_I32_DESCRIPTOR.operands[3],
        ),
    )
    descriptors = replace(
        TEST_LOW_CORE_DESCRIPTOR_SET,
        descriptors=(*TEST_LOW_CORE_DESCRIPTOR_SET.descriptors, descriptor),
    )
    with pytest.raises(ValueError, match="does not accept the materializer carrier"):
        _validate(
            (replace(conversion, descriptor=descriptor),), descriptors=descriptors
        )


def test_complete_address_validates_its_canonical_integer_conversions():
    conversion = SourceMemoryIntegerConversion("i1", TEST_LOW_SELECT_I32_DESCRIPTOR)
    materializer = SourceMemoryAddressMaterializer(
        const_coordinate=TEST_LOW_CONST_I32_DESCRIPTOR,
        add_coordinate=TEST_LOW_ADD_I32_DESCRIPTOR,
        mul_coordinate=TEST_LOW_MUL_I32_DESCRIPTOR,
        address=TEST_LOW_ADD_I32_DESCRIPTOR,
        index_to_coordinate=TEST_LOW_REMATERIALIZE_I32_DESCRIPTOR,
        const_coordinate_immediate="i32_value",
        integer_conversions=(conversion,),
    )
    emit = EmitDescriptorOp(
        descriptor=TEST_LOW_LOAD_V4I32_DESCRIPTOR,
        operands={"address": ValueRef.source_memory_address()},
        results={"dst": ValueRef.result("result")},
        source_memory=SourceMemoryConstraint(
            operation=SourceMemoryOperation.LOAD,
            memory_spaces=("global",),
            element_byte_count=4,
            vector_lane_count=4,
            vector_lane_byte_stride=4,
            static_byte_offset=0,
            dynamic_term_count=None,
        ),
        source_memory_address_materializer=materializer,
    )
    emit.validate(vector.vector_load, TEST_LOW_CORE_DESCRIPTOR_SET, set())
    for conversions, message in (
        ((conversion, conversion), "duplicate address conversion"),
        ((replace(conversion, source_type="offset"),), "fixed-width integer"),
        ((replace(conversion, source_type="i32"),), "exactly 1 packet inputs"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(
                emit,
                source_memory_address_materializer=replace(
                    materializer, integer_conversions=conversions
                ),
            ).validate(vector.vector_load, TEST_LOW_CORE_DESCRIPTOR_SET, set())


@pytest.mark.parametrize(
    "field", ["byte_offset_unsigned_bit_count", "dynamic_offset_unsigned_bit_count"]
)
def test_byte_offset_widths_cover_the_integer_fact_domain(field):
    constraint = SourceMemoryConstraint(
        operation=SourceMemoryOperation.LOAD,
        memory_spaces=("global",),
        element_byte_count=4,
        vector_lane_count=1,
        vector_lane_byte_stride=4,
        static_byte_offset=0,
    )
    for width in (0, 32, 64):
        replace(constraint, **{field: width})
    for width in (-1, 65):
        with pytest.raises(ValueError, match="width must be in"):
            replace(constraint, **{field: width})
