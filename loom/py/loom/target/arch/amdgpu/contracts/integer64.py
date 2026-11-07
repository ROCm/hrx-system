# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Fixed-width wide shifts and disjoint-word register transport."""

from loom.dialect.scalar import bitwise
from loom.target.arch.amdgpu.contracts.materializers import I64_VGPR_MATERIALIZER
from loom.target.contracts import (
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterConcat,
    EmitRegisterSlice,
    Guard,
    Scalar,
    ValueProject,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import DescriptorSet


def integer64_rules(descriptor_set: DescriptorSet) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = list(_word_join_rules())
    result = ValueRef.result("result")
    for source_op, scalar_name, vector_name in (
        (bitwise.scalar_shli, "s_lshl_b64", "v_lshlrev_b64"),
        (bitwise.scalar_shrui, "s_lshr_b64", "v_lshrrev_b64"),
        (bitwise.scalar_shrsi, "s_ashr_i64", "v_ashrrev_i64"),
    ):
        for bank, descriptor_name in (("sgpr", scalar_name), ("vgpr", vector_name)):
            register_class = f"amdgpu.{bank}"
            is_scalar = bank == "sgpr"
            descriptor = descriptor_by_key(descriptor_set, f"amdgpu.{descriptor_name}")
            guards = (
                *(
                    Guard.value_type(field, Scalar("i64"))
                    for field in ("lhs", "rhs", "result")
                ),
                Guard.low_value_register_class("result", register_class),
                *(
                    (
                        Guard.low_value_register_class("lhs", register_class),
                        Guard.low_value_register_class("rhs", register_class),
                    )
                    if is_scalar
                    else (
                        Guard.value_materializable("lhs", I64_VGPR_MATERIALIZER.name),
                    )
                ),
            )
            value = ValueRef.operand(
                "lhs", materializer=None if is_scalar else I64_VGPR_MATERIALIZER.name
            )
            if source_op != bitwise.scalar_shrsi:
                zero_descriptor = descriptor_by_key(
                    descriptor_set, f"amdgpu.{bank[0]}_mov_b32"
                )
                word = ValueRef.temporary("word")
                zero = ValueRef.temporary("zero")
                is_left = source_op == bitwise.scalar_shli
                variants = (
                    ((32, False),)
                    if is_scalar
                    else (
                        (32, False),
                        (63, True),
                        (63, False),
                    )
                )
                for maximum_count, inline_shift in variants:
                    descriptors = [zero_descriptor]
                    emits = [
                        EmitRegisterSlice(
                            source=value,
                            result=word,
                            unit_offset=0 if is_left else 1,
                            unit_count=1,
                        ),
                        EmitDescriptorOp(
                            descriptor=zero_descriptor,
                            immediates={"imm32": 0},
                            results={"dst": zero},
                            result_types={"dst": DescriptorResultType()},
                        ),
                    ]
                    shifted_word = word
                    if maximum_count == 63:
                        # Word shifts mask the count to five bits. The retained
                        # range selects the source word and proves the zero half.
                        word_key = f"amdgpu.{vector_name.removesuffix('_b64')}_b32"
                        word_descriptor = descriptor_by_key(
                            descriptor_set,
                            word_key + ".src0_inline" if inline_shift else word_key,
                        )
                        descriptors.append(word_descriptor)
                        shifted_word = ValueRef.temporary("shifted_word")
                        operands = {"value": word}
                        immediates = {}
                        if inline_shift:
                            immediates["imm32"] = ValueProject.exact_i64("rhs")
                        else:
                            shift = ValueRef.temporary("shift")
                            emits.append(
                                EmitRegisterSlice(
                                    source=ValueRef.operand("rhs"),
                                    result=shift,
                                    unit_count=1,
                                )
                            )
                            operands["shift"] = shift
                        emits.append(
                            EmitDescriptorOp(
                                descriptor=word_descriptor,
                                operands=operands,
                                immediates=immediates,
                                results={"dst": shifted_word},
                                result_types={"dst": DescriptorResultType()},
                            )
                        )
                    emits.append(
                        EmitRegisterConcat(
                            sources=(zero, shifted_word)
                            if is_left
                            else (shifted_word, zero),
                            result=result,
                        )
                    )
                    rules.append(
                        DescriptorRule(
                            source_op=source_op,
                            guards=(
                                *guards,
                                Guard.value_i64_range("rhs", 32, maximum_count),
                                *(
                                    (Guard.value_exact_i64("rhs"),)
                                    if inline_shift
                                    else ()
                                ),
                                *(
                                    Guard.descriptor_available(required_descriptor)
                                    for required_descriptor in descriptors
                                ),
                            ),
                            emit=tuple(emits),
                        )
                    )
            shift = ValueRef.temporary("shift")
            rules.append(
                DescriptorRule(
                    source_op=source_op,
                    descriptor=descriptor,
                    guards=(*guards, Guard.descriptor_available(descriptor)),
                    emit=(
                        EmitRegisterSlice(
                            source=ValueRef.operand("rhs"),
                            result=shift,
                            unit_count=1,
                        ),
                        EmitDescriptorOp(
                            descriptor=descriptor,
                            operands={"value": value, "shift": shift},
                            results={"dst": result},
                        ),
                    ),
                )
            )
    return tuple(rules)


def _word_join_rules() -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for source_op in (bitwise.scalar_ori, bitwise.scalar_xori):
        for bank in ("sgpr", "vgpr"):
            register_class = f"amdgpu.{bank}"
            for low, high in (("lhs", "rhs"), ("rhs", "lhs")):
                guards = (
                    *(
                        Guard.value_type(field, Scalar("i64"))
                        for field in ("lhs", "rhs", "result")
                    ),
                    Guard.value_i64_range(low, 0, 2**32 - 1),
                    Guard.value_divisible_by(high, 2**32),
                    Guard.low_value_register_class("result", register_class),
                    *(
                        Guard.low_value_register_class(field, register_class)
                        if bank == "sgpr"
                        else Guard.value_materializable(
                            field, I64_VGPR_MATERIALIZER.name
                        )
                        for field in ("lhs", "rhs")
                    ),
                )
                words = tuple(ValueRef.temporary(name) for name in ("low", "high"))
                rules.append(
                    DescriptorRule(
                        source_op=source_op,
                        guards=guards,
                        emit=(
                            *(
                                EmitRegisterSlice(
                                    source=ValueRef.operand(
                                        field,
                                        materializer=None
                                        if bank == "sgpr"
                                        else I64_VGPR_MATERIALIZER.name,
                                    ),
                                    result=word,
                                    unit_offset=offset,
                                    unit_count=1,
                                )
                                for offset, (field, word) in enumerate(
                                    zip((low, high), words, strict=True)
                                )
                            ),
                            EmitRegisterConcat(
                                sources=words, result=ValueRef.result("result")
                            ),
                        ),
                    )
                )
    return tuple(rules)
