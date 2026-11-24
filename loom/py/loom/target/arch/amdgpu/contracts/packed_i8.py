# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Word-wise arithmetic rules for packed byte vectors."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

from loom.dialect.vector import defs as vector
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    ValueAliasRule,
    ValueRef,
    Vector,
    descriptor_by_key,
)
from loom.target.low_descriptors import DescriptorSet

PACKED_I8_TYPE = Vector(
    "i8",
    minimum_lanes=1,
    maximum_lanes="LOOM_AMDGPU_MAX_PACKED_I8_LANES",
)
_TYPE_GUARDS = tuple(
    Guard.value_type(field, PACKED_I8_TYPE) for field in ("lhs", "rhs", "result")
)
_PACKED_I8_LOW7_MASK = 0x7F7F7F7F
_PACKED_I8_SIGN_MASK = 0x80808080


def _register_bank_rules(
    descriptors: DescriptorSet,
    rule: DescriptorRule,
    operands: tuple[str, ...],
    *,
    setup: tuple[EmitDescriptorOp, ...] = (),
) -> tuple[DescriptorRule, ...]:
    """Broadcasts scalar words only where packed VALU arithmetic requires them."""
    move = descriptor_by_key(descriptors, "amdgpu.v_mov_b32_copy")
    rules = []
    for banks in product(("amdgpu.vgpr", "amdgpu.sgpr"), repeat=len(operands)):
        replacements = {
            ValueRef.operand(operand): ValueRef.temporary(operand + "_vgpr")
            for operand, bank in zip(operands, banks, strict=True)
            if bank == "amdgpu.sgpr"
        }
        broadcasts = tuple(
            EmitDescriptorOp(
                descriptor=move,
                operands={"src": source},
                results={"dst": result},
                result_types={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            )
            for source, result in replacements.items()
        )
        rules.append(
            replace(
                rule,
                guards=(
                    *rule.guards,
                    *(
                        Guard.low_value_register_class(operand, bank)
                        for operand, bank in zip(operands, banks, strict=True)
                    ),
                    *((Guard.descriptor_available(move),) if broadcasts else ()),
                ),
                emit=(
                    *setup,
                    *broadcasts,
                    *(
                        replace(
                            step,
                            operands={
                                name: replacements.get(value, value)
                                for name, value in step.operands.items()
                            },
                        )
                        for step in rule.emit
                    ),
                ),
            )
        )
    return tuple(rules)


def packed_i8_add_rules(descriptors: DescriptorSet) -> tuple[DescriptorRule, ...]:
    and_literal = descriptor_by_key(descriptors, "amdgpu.v_and_b32.lit")
    add = descriptor_by_key(descriptors, "amdgpu.v_add_u32")
    xor_bits = descriptor_by_key(descriptors, "amdgpu.v_xor_b32")
    result_type = {"dst": ValueRef.result("result")}
    rule = DescriptorRule(
        source_op=vector.vector_addi,
        descriptor=add,
        guards=(
            *_TYPE_GUARDS,
            Guard.descriptor_available(and_literal),
            Guard.descriptor_available(add),
            Guard.descriptor_available(xor_bits),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=and_literal,
                operands={"rhs": ValueRef.operand("lhs")},
                results={"dst": ValueRef.temporary("lhs_low")},
                result_types=result_type,
                immediates={"imm32": _PACKED_I8_LOW7_MASK},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=and_literal,
                operands={"rhs": ValueRef.operand("rhs")},
                results={"dst": ValueRef.temporary("rhs_low")},
                result_types=result_type,
                immediates={"imm32": _PACKED_I8_LOW7_MASK},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=add,
                operands={
                    "lhs": ValueRef.temporary("lhs_low"),
                    "rhs": ValueRef.temporary("rhs_low"),
                },
                results={"dst": ValueRef.temporary("low_sum")},
                result_types=result_type,
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=xor_bits,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.temporary("high_xor")},
                result_types=result_type,
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=and_literal,
                operands={"rhs": ValueRef.temporary("high_xor")},
                results={"dst": ValueRef.temporary("high_bits")},
                result_types=result_type,
                immediates={"imm32": _PACKED_I8_SIGN_MASK},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=xor_bits,
                operands={
                    "lhs": ValueRef.temporary("low_sum"),
                    "rhs": ValueRef.temporary("high_bits"),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
        ),
    )

    return _register_bank_rules(descriptors, rule, ("lhs", "rhs"))


def packed_i8_sub_rules(descriptors: DescriptorSet) -> tuple[DescriptorRule, ...]:
    and_literal = descriptor_by_key(descriptors, "amdgpu.v_and_b32.lit")
    or_literal = descriptor_by_key(descriptors, "amdgpu.v_or_b32.lit")
    sub = descriptor_by_key(descriptors, "amdgpu.v_sub_u32")
    xor_bits = descriptor_by_key(descriptors, "amdgpu.v_xor_b32")
    xor_literal = descriptor_by_key(descriptors, "amdgpu.v_xor_b32.lit")
    result_type = {"dst": ValueRef.result("result")}
    rule = DescriptorRule(
        source_op=vector.vector_subi,
        descriptor=sub,
        guards=(
            *_TYPE_GUARDS,
            Guard.descriptor_available(and_literal),
            Guard.descriptor_available(or_literal),
            Guard.descriptor_available(sub),
            Guard.descriptor_available(xor_bits),
            Guard.descriptor_available(xor_literal),
        ),
        emit=(
            EmitDescriptorOp(
                descriptor=or_literal,
                operands={"rhs": ValueRef.operand("lhs")},
                results={"dst": ValueRef.temporary("lhs_guard")},
                result_types=result_type,
                immediates={"imm32": _PACKED_I8_SIGN_MASK},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=and_literal,
                operands={"rhs": ValueRef.operand("rhs")},
                results={"dst": ValueRef.temporary("rhs_low")},
                result_types=result_type,
                immediates={"imm32": _PACKED_I8_LOW7_MASK},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=sub,
                operands={
                    "lhs": ValueRef.temporary("lhs_guard"),
                    "rhs": ValueRef.temporary("rhs_low"),
                },
                results={"dst": ValueRef.temporary("low_diff")},
                result_types=result_type,
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=xor_bits,
                operands={
                    "lhs": ValueRef.operand("lhs"),
                    "rhs": ValueRef.operand("rhs"),
                },
                results={"dst": ValueRef.temporary("high_xor")},
                result_types=result_type,
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=xor_literal,
                operands={"rhs": ValueRef.temporary("high_xor")},
                results={"dst": ValueRef.temporary("high_toggled")},
                result_types=result_type,
                immediates={"imm32": _PACKED_I8_SIGN_MASK},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=and_literal,
                operands={"rhs": ValueRef.temporary("high_toggled")},
                results={"dst": ValueRef.temporary("high_bits")},
                result_types=result_type,
                immediates={"imm32": _PACKED_I8_SIGN_MASK},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
            EmitDescriptorOp(
                descriptor=xor_bits,
                operands={
                    "lhs": ValueRef.temporary("low_diff"),
                    "rhs": ValueRef.temporary("high_bits"),
                },
                results={"dst": ValueRef.result("result")},
                form=DescriptorEmitForm.PER_LANE_SEQUENCE,
            ),
        ),
    )

    return _register_bank_rules(descriptors, rule, ("lhs", "rhs"))


def _constant_logical_shift_rules(
    descriptors: DescriptorSet,
) -> tuple[DescriptorRule | ValueAliasRule, ...]:
    mask_descriptor = descriptor_by_key(descriptors, "amdgpu.v_and_b32.lit")
    rules: list[DescriptorRule | ValueAliasRule] = []
    for source_op, shift_key in (
        (vector.vector_shli, "amdgpu.v_lshlrev_b32.src0_inline"),
        (vector.vector_shrui, "amdgpu.v_lshrrev_b32.src0_inline"),
    ):
        shift_descriptor = descriptor_by_key(descriptors, shift_key)
        rules.append(
            ValueAliasRule(
                source_op=source_op,
                source=ValueRef.operand("lhs"),
                result=ValueRef.result("result"),
                guards=(*_TYPE_GUARDS, Guard.value_i64_range("rhs", 0, 0)),
            )
        )
        for amount in range(1, 8):
            byte_mask = (
                (0xFF << amount) & 0xFF
                if source_op is vector.vector_shli
                else 0xFF >> amount
            )
            rule = DescriptorRule(
                source_op=source_op,
                descriptor=shift_descriptor,
                guards=(
                    *_TYPE_GUARDS,
                    # A singleton element range proves equal runtime counts;
                    # a common non-singleton range does not.
                    Guard.value_i64_range("rhs", amount, amount),
                    Guard.descriptor_available(shift_descriptor),
                    Guard.descriptor_available(mask_descriptor),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=shift_descriptor,
                        operands={"value": ValueRef.operand("lhs")},
                        results={"dst": ValueRef.temporary("shifted")},
                        result_types={"dst": ValueRef.result("result")},
                        immediates={"imm32": amount},
                        form=DescriptorEmitForm.PER_LANE_SEQUENCE,
                    ),
                    EmitDescriptorOp(
                        descriptor=mask_descriptor,
                        operands={"rhs": ValueRef.temporary("shifted")},
                        results={"dst": ValueRef.result("result")},
                        immediates={"imm32": byte_mask * 0x01010101},
                        form=DescriptorEmitForm.PER_LANE_SEQUENCE,
                    ),
                ),
            )
            rules.extend(_register_bank_rules(descriptors, rule, ("lhs",)))
    return tuple(rules)


def _shared_logical_shift_rules(
    descriptors: DescriptorSet,
) -> tuple[DescriptorRule, ...]:
    # One shared count lets every word use the same byte mask. A left shift
    # masks the input and a right shift masks the output. In either case
    # each byte of the mask is 0xff >> count. The scalar origin can vary
    # between workitems, so this setup runs in vector registers.
    scalar_count = ValueRef.uniform_element_origin_operand("rhs")
    byte = ValueRef.temporary("byte")
    ones = ValueRef.temporary("ones")
    mask = ValueRef.temporary("mask")
    word = ValueRef.operand("lhs")
    partial = ValueRef.temporary("partial")
    result = ValueRef.result("result")
    literal = descriptor_by_key(descriptors, "amdgpu.v_mov_b32")
    scalar_literal = descriptor_by_key(descriptors, "amdgpu.s_mov_b32")
    multiply = descriptor_by_key(descriptors, "amdgpu.v_mul_lo_u32")
    and_bits = descriptor_by_key(descriptors, "amdgpu.v_and_b32")
    right = descriptor_by_key(descriptors, "amdgpu.v_lshrrev_b32")
    shared_setup = (
        EmitDescriptorOp(
            descriptor=literal,
            results={"dst": byte},
            result_types={"dst": DescriptorResultType()},
            immediates={"imm32": 255},
        ),
        EmitDescriptorOp(
            descriptor=scalar_literal,
            results={"dst": ones},
            result_types={"dst": DescriptorResultType()},
            immediates={"imm32": 0x01010101},
        ),
        EmitDescriptorOp(
            descriptor=right,
            operands={"shift": scalar_count, "value": byte},
            results={"dst": ValueRef.temporary("byte_mask")},
            result_types={"dst": DescriptorResultType()},
        ),
        EmitDescriptorOp(
            descriptor=multiply,
            operands={"lhs": ones, "rhs": ValueRef.temporary("byte_mask")},
            results={"dst": mask},
            result_types={"dst": DescriptorResultType()},
        ),
    )
    rules = []
    for source_op, shift_descriptor in (
        (vector.vector_shli, descriptor_by_key(descriptors, "amdgpu.v_lshlrev_b32")),
        (vector.vector_shrui, right),
    ):
        left = source_op is vector.vector_shli
        shift_op = EmitDescriptorOp(
            descriptor=shift_descriptor,
            operands={"shift": scalar_count, "value": partial if left else word},
            results={"dst": result if left else partial},
            result_types={"dst": result},
            form=DescriptorEmitForm.PER_LANE_SEQUENCE,
        )
        mask_op = EmitDescriptorOp(
            descriptor=and_bits,
            operands={"lhs": mask, "rhs": word if left else partial},
            results={"dst": partial if left else result},
            result_types={"dst": result},
            form=DescriptorEmitForm.PER_LANE_SEQUENCE,
        )
        rule = DescriptorRule(
            source_op=source_op,
            descriptor=shift_descriptor,
            guards=(
                *_TYPE_GUARDS,
                Guard.uniform_element_origin_type("rhs", Scalar("i8")),
                *(
                    Guard.descriptor_available(op)
                    for op in (
                        literal,
                        scalar_literal,
                        multiply,
                        and_bits,
                        right,
                        shift_descriptor,
                    )
                ),
            ),
            emit=(mask_op, shift_op) if left else (shift_op, mask_op),
        )
        rules.extend(
            _register_bank_rules(descriptors, rule, ("lhs",), setup=shared_setup)
        )
    return tuple(rules)


def _independent_byte_shift_rules(
    descriptors: DescriptorSet,
) -> tuple[DescriptorRule, ...]:
    # A packed halfword shift takes the low four bits of each count.
    # On defined byte shifts the low byte already supplies a count in 0..7.
    # Shifting the count word right by eight exposes the odd byte counts.
    align_right = descriptor_by_key(descriptors, "amdgpu.v_lshrrev_b32.src0_inline")
    align_left = descriptor_by_key(descriptors, "amdgpu.v_lshlrev_b32.src0_inline")
    permute = descriptor_by_key(descriptors, "amdgpu.v_perm_b32")
    move = descriptor_by_key(descriptors, "amdgpu.s_mov_b32")
    values = ValueRef.operand("lhs")
    counts = ValueRef.operand("rhs")
    aligned = ValueRef.temporary("aligned")
    odd_counts = ValueRef.temporary("odd_counts")
    even_values = ValueRef.temporary("even")
    odd_values = ValueRef.temporary("odd")
    selectors = ValueRef.temporary("selectors")
    result_type = {"dst": ValueRef.result("result")}
    packet = DescriptorEmitForm.PER_LANE_SEQUENCE
    rules = []
    for source_op, shift_key in (
        (vector.vector_shli, "amdgpu.v_pk_lshlrev_b16"),
        (vector.vector_shrui, "amdgpu.v_pk_lshrrev_b16"),
        (vector.vector_shrsi, "amdgpu.v_pk_ashrrev_i16"),
    ):
        shift = descriptor_by_key(descriptors, shift_key)
        left = source_op is vector.vector_shli
        align = align_right if left else align_left
        # Place each byte at the edge that provides the zero or sign fill:
        # the low byte for left shifts and the high byte for right shifts.
        # This also puts the byte sign in the halfword sign bit for ASHR.
        # V_PERM byte IDs 4..7 address src0; IDs 0..3 address src1.
        selector = 0x02060004 if left else 0x03070105
        selector_op = EmitDescriptorOp(
            descriptor=move,
            results={"dst": selectors},
            result_types={"dst": DescriptorResultType()},
            immediates={"imm32": selector},
        )
        rule = DescriptorRule(
            source_op=source_op,
            descriptor=shift,
            guards=(
                *_TYPE_GUARDS,
                *(
                    Guard.descriptor_available(op)
                    for op in (align, align_right, shift, permute, move)
                ),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=align,
                    operands={"value": values},
                    results={"dst": aligned},
                    result_types=result_type,
                    immediates={"imm32": 8},
                    form=packet,
                ),
                EmitDescriptorOp(
                    descriptor=align_right,
                    operands={"value": counts},
                    results={"dst": odd_counts},
                    result_types=result_type,
                    immediates={"imm32": 8},
                    form=packet,
                ),
                EmitDescriptorOp(
                    descriptor=shift,
                    operands={"shift": counts, "value": values if left else aligned},
                    results={"dst": even_values},
                    result_types=result_type,
                    form=packet,
                ),
                EmitDescriptorOp(
                    descriptor=shift,
                    operands={
                        "shift": odd_counts,
                        "value": aligned if left else values,
                    },
                    results={"dst": odd_values},
                    result_types=result_type,
                    form=packet,
                ),
                EmitDescriptorOp(
                    descriptor=permute,
                    operands={
                        "src0": even_values,
                        "src1": odd_values,
                        "selectors": selectors,
                    },
                    results={"dst": ValueRef.result("result")},
                    form=packet,
                ),
            ),
        )
        rules.extend(
            _register_bank_rules(
                descriptors, rule, ("lhs", "rhs"), setup=(selector_op,)
            )
        )
    return tuple(rules)


def packed_i8_shift_rules(
    descriptors: DescriptorSet,
) -> tuple[DescriptorRule | ValueAliasRule, ...]:
    """Selects the narrowest lowering proved by the shift-count operand."""
    return (
        *_constant_logical_shift_rules(descriptors),
        *_shared_logical_shift_rules(descriptors),
        *_independent_byte_shift_rules(descriptors),
    )
