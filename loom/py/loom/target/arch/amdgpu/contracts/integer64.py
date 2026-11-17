# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Wide integer operations and disjoint-word register transport."""

from loom.dialect.index import defs as index
from loom.dialect.scalar import bitwise
from loom.dsl import Op
from loom.target.arch.amdgpu.contracts.materializers import (
    ADDRESS_SGPR64_MATERIALIZER,
    ADDRESS_SGPR_MATERIALIZER,
    ADDRESS_VGPR64_MATERIALIZER,
    ADDRESS_VGPR_MATERIALIZER,
    I64_VGPR_MATERIALIZER,
    REGISTERS_SGPR_MATERIALIZER,
    WIDE_INDEX_SGPR_MATERIALIZER,
    WIDE_INDEX_VGPR_MATERIALIZER,
)
from loom.target.contracts import (
    DescriptorEmitForm,
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
                        Guard.value_materializable(
                            "lhs", REGISTERS_SGPR_MATERIALIZER.name
                        ),
                        Guard.value_materializable(
                            "rhs", REGISTERS_SGPR_MATERIALIZER.name
                        ),
                    )
                    if is_scalar
                    else (
                        Guard.value_materializable("lhs", I64_VGPR_MATERIALIZER.name),
                    )
                ),
            )
            value = ValueRef.operand(
                "lhs",
                materializer=(
                    REGISTERS_SGPR_MATERIALIZER.name
                    if is_scalar
                    else I64_VGPR_MATERIALIZER.name
                ),
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
                            source=ValueRef.operand(
                                "rhs",
                                materializer=(
                                    REGISTERS_SGPR_MATERIALIZER.name
                                    if is_scalar
                                    else None
                                ),
                            ),
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


def index64_shift_rules(descriptor_set: DescriptorSet) -> tuple[DescriptorRule, ...]:
    """Right shifts consume the full input even when the result fits one word."""
    rules: list[DescriptorRule] = []
    for source_op, scalar_name, vector_name in (
        (index.index_shrui, "s_lshr_b64", "v_lshrrev_b64"),
        (index.index_shrsi, "s_ashr_i64", "v_ashrrev_i64"),
    ):
        for bank, descriptor_name, value_materializer, count_materializer in (
            (
                "sgpr",
                scalar_name,
                WIDE_INDEX_SGPR_MATERIALIZER,
                ADDRESS_SGPR_MATERIALIZER,
            ),
            (
                "vgpr",
                vector_name,
                WIDE_INDEX_VGPR_MATERIALIZER,
                ADDRESS_VGPR_MATERIALIZER,
            ),
        ):
            descriptor = descriptor_by_key(descriptor_set, f"amdgpu.{descriptor_name}")
            for result_units in (1, 2):
                result = ValueRef.result("result")
                shifted = result if result_units == 2 else ValueRef.temporary("shifted")
                count = ValueRef.temporary("count")
                rules.append(
                    DescriptorRule(
                        source_op=source_op,
                        descriptor=descriptor,
                        guards=(
                            *(
                                Guard.value_type(field, Scalar("index"))
                                for field in ("lhs", "rhs", "result")
                            ),
                            Guard.low_value_register_class("result", f"amdgpu.{bank}"),
                            Guard.low_value_register_unit_count("result", result_units),
                            Guard.value_materializable("lhs", value_materializer.name),
                            Guard.value_materializable("lhs", count_materializer.name),
                            Guard.value_materializable("rhs", count_materializer.name),
                            Guard.descriptor_available(descriptor),
                        ),
                        emit=(
                            EmitRegisterSlice(
                                source=ValueRef.operand(
                                    "rhs", materializer=count_materializer.name
                                ),
                                result=count,
                                unit_count=1,
                            ),
                            EmitDescriptorOp(
                                descriptor=descriptor,
                                operands={
                                    "value": ValueRef.operand(
                                        "lhs", materializer=value_materializer.name
                                    ),
                                    "shift": count,
                                },
                                results={"dst": shifted},
                                result_types={"dst": DescriptorResultType()},
                            ),
                            *(
                                (EmitRegisterSlice(source=shifted, result=result),)
                                if result_units == 1
                                else ()
                            ),
                        ),
                    )
                )
    return tuple(rules)


def integer64_bitwise_rules(
    descriptor_set: DescriptorSet,
    source_op: Op,
    mnemonic: str,
    *,
    element_type: str = "i64",
) -> tuple[DescriptorRule, ...]:
    scalar_materializer, vector_materializer = {
        "i64": (REGISTERS_SGPR_MATERIALIZER, I64_VGPR_MATERIALIZER),
        "index": (ADDRESS_SGPR64_MATERIALIZER, ADDRESS_VGPR64_MATERIALIZER),
    }[element_type]
    # Fixed i64 sources already occupy two words. An index source can still
    # occupy one word when its consumer requires a wide result.
    uniform_materializer = scalar_materializer if element_type == "index" else None
    rules: list[DescriptorRule] = []
    result = ValueRef.result("result")
    typed_guards = tuple(
        Guard.value_type(field, Scalar(element_type))
        for field in ("lhs", "rhs", "result")
    )
    scalar_descriptor = descriptor_by_key(descriptor_set, f"amdgpu.s_{mnemonic}_b64")
    rules.append(
        DescriptorRule(
            source_op=source_op,
            descriptor=scalar_descriptor,
            guards=(
                *typed_guards,
                Guard.low_value_register_class("result", "amdgpu.sgpr"),
                Guard.low_value_register_unit_count("result", 2),
                Guard.value_materializable("lhs", scalar_materializer.name),
                Guard.value_materializable("rhs", scalar_materializer.name),
                Guard.descriptor_available(scalar_descriptor),
            ),
            emit=(
                EmitDescriptorOp(
                    descriptor=scalar_descriptor,
                    operands={
                        field: ValueRef.operand(
                            field, materializer=scalar_materializer.name
                        )
                        for field in ("lhs", "rhs")
                    },
                    results={"dst": result},
                ),
            ),
        )
    )
    vector_descriptor = descriptor_by_key(descriptor_set, f"amdgpu.v_{mnemonic}_b32")
    # Only the first vector source accepts SGPRs or encoded literals. Keep
    # either uniform operand there so word slicing preserves those choices.
    # Index operands may have different carrier widths; their materializers
    # extend a narrow operand in its original register bank before slicing.
    for first, second, preserve_uniform in (
        ("lhs", "rhs", True),
        ("rhs", "lhs", True),
        ("lhs", "rhs", False),
    ):
        first_materializer = (
            uniform_materializer if preserve_uniform else vector_materializer
        )
        first_guards = (
            (Guard.low_value_register_class(first, "amdgpu.sgpr"),)
            if preserve_uniform
            else ()
        )
        if first_materializer is not None:
            first_guards += (
                Guard.value_materializable(first, first_materializer.name),
            )
        rules.append(
            DescriptorRule(
                source_op=source_op,
                descriptor=vector_descriptor,
                guards=(
                    *typed_guards,
                    Guard.low_value_register_class("result", "amdgpu.vgpr"),
                    Guard.low_value_register_unit_count("result", 2),
                    *first_guards,
                    Guard.value_materializable(second, vector_materializer.name),
                    Guard.descriptor_available(vector_descriptor),
                ),
                emit=(
                    EmitDescriptorOp(
                        descriptor=vector_descriptor,
                        operands={
                            "lhs": ValueRef.operand(
                                first,
                                materializer=first_materializer.name
                                if first_materializer is not None
                                else None,
                            ),
                            "rhs": ValueRef.operand(
                                second, materializer=vector_materializer.name
                            ),
                        },
                        results={"dst": result},
                        result_types={"dst": result},
                        form=DescriptorEmitForm.PER_LANE,
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
                        Guard.value_materializable(
                            field, REGISTERS_SGPR_MATERIALIZER.name
                        )
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
                                        materializer=(
                                            REGISTERS_SGPR_MATERIALIZER.name
                                            if bank == "sgpr"
                                            else I64_VGPR_MATERIALIZER.name
                                        ),
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
