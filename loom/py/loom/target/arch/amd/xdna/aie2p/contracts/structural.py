# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMD XDNA AIE2P vector structural selection rules."""

from loom.dialect.vector import defs as vector
from loom.target.arch.amd.xdna.aie2p.core_descriptors import (
    AIE2P_CORE_DESCRIPTOR_SET,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    ValueAliasRule,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor

_I8X32_VECTOR = Vector("i8", lanes=32)
_I8X64_VECTOR = Vector("i8", lanes=64)
_I32X8_VECTOR = Vector("i32", lanes=8)
_I32X16_VECTOR = Vector("i32", lanes=16)

# AIE2P VSHUFFLE modes that select the even and odd byte lanes from the first
# 512-bit source. Each logical 32-byte result retains the target's 512-bit X
# carrier, with the remaining lanes outside the source vector's value domain.
_I8_DEINTERLEAVE_CONTROLS = (0, 1)

# Moving the upper eight i32 lanes into the low half of AIE2P's 512-bit X
# carrier is a 32-byte VSHIFT. The upper half of the result lies outside the
# logical vector<8xi32> value domain.
_I32_SLICE_HIGH_BYTE_OFFSET = 32


def _descriptor(key: str) -> Descriptor:
    return descriptor_by_key(AIE2P_CORE_DESCRIPTOR_SET, key)


def _vector_deinterleave_i8x64_rule() -> DescriptorRule:
    control_constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shuffle = _descriptor("amd.xdna.aie2p.shuffle.x.configured")

    emits = []
    for result_index, result_name in enumerate(("even", "odd")):
        control_name = f"{result_name}_control"
        emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=control_constant,
                    results={"dst": ValueRef.temporary(control_name)},
                    result_types={"dst": DescriptorResultType()},
                    immediates={"i": _I8_DEINTERLEAVE_CONTROLS[result_index]},
                    form=DescriptorEmitForm.CONST,
                ),
                EmitDescriptorOp(
                    descriptor=shuffle,
                    operands={
                        "s1": ValueRef.operand("source"),
                        "s2": ValueRef.operand("source"),
                        "mod": ValueRef.temporary(control_name),
                    },
                    results={"dst": ValueRef.result("results", element=result_index)},
                    form=DescriptorEmitForm.OP,
                ),
            )
        )

    return DescriptorRule(
        source_op=vector.vector_deinterleave,
        descriptor=shuffle,
        guards=(
            Guard.value_type("source", _I8X64_VECTOR),
            Guard.value_type("results", _I8X32_VECTOR),
            Guard.attr_kind("axis", "i64"),
            Guard.i64_range("axis", 0, 0),
        ),
        emit=tuple(emits),
    )


def _i32x16_to_i32x8_slice_guards(offset: int) -> tuple[Guard, ...]:
    return (
        Guard.value_type("source", _I32X16_VECTOR),
        Guard.value_type("result", _I32X8_VECTOR),
        Guard.operand_segment_count("offsets", 0),
        Guard.i64_array_count("static_offsets", 1),
        Guard.i64_array_element_range(
            "static_offsets", element=0, minimum=offset, maximum=offset
        ),
    )


def _vector_slice_i32x16_low_rule() -> ValueAliasRule:
    # A narrow ordinary vector retains the same 512-bit X carrier as its
    # source, so the low aligned half is a value alias.
    return ValueAliasRule(
        source_op=vector.vector_slice,
        source=ValueRef.operand("source"),
        result=ValueRef.result("result"),
        guards=_i32x16_to_i32x8_slice_guards(0),
    )


def _vector_slice_i32x16_high_rule() -> DescriptorRule:
    constant = _descriptor("amd.xdna.aie2p.constant.i32.mova")
    shift = _descriptor("amd.xdna.aie2p.shift.bytes.x.configured")
    return DescriptorRule(
        source_op=vector.vector_slice,
        descriptor=shift,
        guards=_i32x16_to_i32x8_slice_guards(8),
        emit=(
            EmitDescriptorOp(
                descriptor=constant,
                results={"dst": ValueRef.temporary("byte_offset")},
                result_types={"dst": DescriptorResultType()},
                immediates={"i": _I32_SLICE_HIGH_BYTE_OFFSET},
                form=DescriptorEmitForm.CONST,
            ),
            EmitDescriptorOp(
                descriptor=shift,
                operands={
                    "s1": ValueRef.operand("source"),
                    "s2": ValueRef.operand("source"),
                    "shift": ValueRef.temporary("byte_offset"),
                },
                results={"d": ValueRef.result("result")},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _vector_concat_i8x32_pair_rule() -> DescriptorRule:
    # Each input carries its 256 value bits in the low W unit of an ordinary X
    # carrier. Joining those two units gives the exact 512-bit result without
    # scalar lane extraction or insertion.
    return DescriptorRule(
        source_op=vector.vector_concat,
        guards=(
            Guard.i64_range("axis", 0, 0),
            Guard.operand_segment_count("inputs", 2),
            Guard.value_type("inputs", _I8X32_VECTOR),
            Guard.value_type("result", _I8X64_VECTOR),
        ),
        emit=(
            EmitRegisterSlice(
                source=ValueRef.operand("inputs", element=0),
                result=ValueRef.temporary("low"),
                unit_count=1,
            ),
            EmitRegisterSlice(
                source=ValueRef.operand("inputs", element=1),
                result=ValueRef.temporary("high"),
                unit_count=1,
            ),
            EmitRegisterConcat(
                sources=(
                    ValueRef.temporary("low"),
                    ValueRef.temporary("high"),
                ),
                result=ValueRef.result("result"),
            ),
        ),
    )


AIE2P_STRUCTURAL_RULES = (
    _vector_slice_i32x16_low_rule(),
    _vector_slice_i32x16_high_rule(),
    _vector_concat_i8x32_pair_rule(),
    _vector_deinterleave_i8x64_rule(),
)
