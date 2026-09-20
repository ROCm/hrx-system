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
    SourceMemoryByteOffsetMaterializer,
    SourceMemoryConstraint,
    SourceMemoryIntegerConversion,
    SourceMemoryOperation,
    ValueRef,
)
from loom.target.low_descriptors import Immediate, ImmediateKind
from loom.target.test.descriptors import (
    TEST_LOW_ADD_I32_DESCRIPTOR,
    TEST_LOW_CONST_I32_DESCRIPTOR,
    TEST_LOW_CORE_DESCRIPTOR_SET,
    TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR,
    TEST_LOW_MUL_I32_DESCRIPTOR,
    TEST_LOW_REMATERIALIZE_I32_DESCRIPTOR,
)


def _validate(conversions, *, descriptors=TEST_LOW_CORE_DESCRIPTOR_SET):
    emit = EmitDescriptorOp(
        descriptor=TEST_LOW_LOAD_INDEX_V4I32_DESCRIPTOR,
        operands={
            "address": ValueRef.operand("view"),
            "index": ValueRef.source_memory_dynamic_byte_offset(),
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
            constant_immediate="i32_value",
            integer_conversions=conversions,
        ),
    )
    emit.validate(vector.vector_load, descriptors, set())


def test_integer_conversion_domains_are_unique_fixed_width_integers():
    conversion = SourceMemoryIntegerConversion(
        "i8", TEST_LOW_REMATERIALIZE_I32_DESCRIPTOR
    )
    _validate((conversion,))
    with pytest.raises(ValueError, match="duplicate byte-offset conversion"):
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
