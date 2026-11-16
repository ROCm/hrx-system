# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact constant and runtime division source-to-low rules."""

from __future__ import annotations

from loom.dialect.index import defs as index
from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dsl import Op
from loom.target.arch.amdgpu.contracts.materializers import (
    ADDRESS_SGPR_MATERIALIZER,
    ADDRESS_VGPR_MATERIALIZER,
    I32_VGPR_MATERIALIZER,
    REGISTERS_SGPR_MATERIALIZER,
)
from loom.target.contracts import (
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    GuardDiagnostic,
    RecipeRule,
    Scalar,
    SignedDivisorMagicKind,
    TypePattern,
    UnsignedDivisorMagicKind,
    ValueMaterializer,
    ValueProject,
    ValueRef,
    descriptor_by_key,
)
from loom.target.low_descriptors import Descriptor, DescriptorSet

_INDEX = Scalar("index")
_I32 = Scalar("i32")
_RESULT = ValueRef.result("result")
_UINT32_MAX = (2**32) - 1
_INT32_MAX = (2**31) - 1
_ADDRESS_U32_DIAGNOSTIC = GuardDiagnostic(
    subject_role="address-width",
    subject_name="u32",
    constraint_key="amdgpu.address.u32",
)
_POSITIVE_U32_DIVISOR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="divisor",
    subject_name="u32",
    constraint_key="amdgpu.divisor.positive_u32",
)
_SIGNED_I32_DIVISOR_DIAGNOSTIC = GuardDiagnostic(
    subject_role="divisor",
    subject_name="i32",
    constraint_key="amdgpu.divisor.constant_signed_i32",
)


def _materialized_operand(field: str, materializer: ValueMaterializer) -> ValueRef:
    return ValueRef.operand(field, materializer=materializer.name)


def _sgpr_operand(field: str, type_pattern: TypePattern) -> ValueRef:
    return (
        _materialized_operand(field, ADDRESS_SGPR_MATERIALIZER)
        if type_pattern == _INDEX
        else _materialized_operand(field, REGISTERS_SGPR_MATERIALIZER)
    )


def _descriptor_available_guards(*descriptors: Descriptor) -> tuple[Guard, ...]:
    unique = {descriptor.key: descriptor for descriptor in descriptors}
    return tuple(
        Guard.descriptor_available(descriptor) for descriptor in unique.values()
    )


def _magic_division_guards(
    type_pattern: TypePattern,
    *,
    register_class: str,
    kind: UnsignedDivisorMagicKind,
    divisor_guards: tuple[Guard, ...] = (),
) -> tuple[Guard, ...]:
    type_guards = tuple(
        Guard.value_type(field, type_pattern) for field in ("lhs", "rhs", "result")
    )
    numerator_guards: tuple[Guard, ...] = ()
    if type_pattern == _INDEX:
        type_guards += (
            Guard.value_unsigned_bit_count(
                "result", 32, diagnostic=_ADDRESS_U32_DIAGNOSTIC
            ),
        )
        numerator_guards = (
            Guard.value_unsigned_bit_count(
                "lhs", 32, diagnostic=_ADDRESS_U32_DIAGNOSTIC
            ),
        )
    if register_class == "amdgpu.sgpr":
        materializer = (
            ADDRESS_SGPR_MATERIALIZER
            if type_pattern == _INDEX
            else REGISTERS_SGPR_MATERIALIZER
        )
        value_guards = tuple(
            Guard.value_materializable(field, materializer.name)
            for field in ("lhs", "rhs")
        )
    else:
        materializer = (
            ADDRESS_VGPR_MATERIALIZER
            if type_pattern == _INDEX
            else I32_VGPR_MATERIALIZER
        )
        value_guards = (
            Guard.value_materializable("lhs", materializer.name),
            Guard.value_materializable("rhs", materializer.name),
        )
    return (
        *type_guards,
        *divisor_guards,
        Guard.low_value_register_class("result", register_class),
        *numerator_guards,
        Guard.value_exact_i64("rhs", diagnostic=_POSITIVE_U32_DIVISOR_DIAGNOSTIC),
        Guard.value_i64_range(
            "rhs",
            2,
            _UINT32_MAX if type_pattern == _INDEX else _INT32_MAX,
            diagnostic=_POSITIVE_U32_DIVISOR_DIAGNOSTIC,
        ),
        Guard.value_u32_divisor_magic_kind("lhs", "rhs", kind),
        *value_guards,
    )


def _magic_division_sgpr_emits(
    descriptor_set: DescriptorSet,
    *,
    type_pattern: TypePattern,
    kind: UnsignedDivisorMagicKind,
    result: ValueRef,
) -> tuple[EmitDescriptorOp, ...]:
    move = descriptor_by_key(descriptor_set, "amdgpu.s_mov_b32")
    multiply_hi = descriptor_by_key(descriptor_set, "amdgpu.s_mul_hi_u32")
    subtract = descriptor_by_key(descriptor_set, "amdgpu.s_sub_u32")
    shift = descriptor_by_key(descriptor_set, "amdgpu.s_lshr_b32")
    add = descriptor_by_key(descriptor_set, "amdgpu.s_add_u32")
    quotient_value = (
        ValueRef.temporary("adjusted_quotient")
        if kind.has_add
        else ValueRef.temporary("quotient")
    )
    emits = [
        EmitDescriptorOp(
            descriptor=move,
            results={"dst": ValueRef.temporary("magic")},
            result_types={"dst": _RESULT},
            immediates={
                "imm32": ValueProject.u32_divisor_magic_multiplier("lhs", "rhs")
            },
        ),
        EmitDescriptorOp(
            descriptor=multiply_hi,
            operands={
                "lhs": _sgpr_operand("lhs", type_pattern),
                "rhs": ValueRef.temporary("magic"),
            },
            results={
                "dst": ValueRef.temporary("quotient") if kind.has_shift else result
            },
            result_types={"dst": _RESULT},
        ),
    ]
    if kind.has_add:
        emits.extend(
            [
                EmitDescriptorOp(
                    descriptor=subtract,
                    operands={
                        "lhs": _sgpr_operand("lhs", type_pattern),
                        "rhs": ValueRef.temporary("quotient"),
                    },
                    results={"dst": ValueRef.temporary("npq")},
                    result_types={"dst": _RESULT},
                ),
                EmitDescriptorOp(
                    descriptor=move,
                    results={"dst": ValueRef.temporary("one")},
                    result_types={"dst": _RESULT},
                    immediates={"imm32": 1},
                ),
                EmitDescriptorOp(
                    descriptor=shift,
                    operands={
                        "lhs": ValueRef.temporary("npq"),
                        "rhs": ValueRef.temporary("one"),
                    },
                    results={"dst": ValueRef.temporary("npq_half")},
                    result_types={"dst": _RESULT},
                ),
                EmitDescriptorOp(
                    descriptor=add,
                    operands={
                        "lhs": ValueRef.temporary("npq_half"),
                        "rhs": ValueRef.temporary("quotient"),
                    },
                    results={"dst": ValueRef.temporary("adjusted_quotient")},
                    result_types={"dst": _RESULT},
                ),
            ]
        )
    if not kind.has_shift:
        return tuple(emits)
    emits.extend(
        [
            EmitDescriptorOp(
                descriptor=move,
                results={"dst": ValueRef.temporary("post_shift")},
                result_types={"dst": _RESULT},
                immediates={
                    "imm32": ValueProject.u32_divisor_magic_shift(
                        "lhs", "rhs", product_bit_width=32
                    )
                },
            ),
            EmitDescriptorOp(
                descriptor=shift,
                operands={
                    "lhs": quotient_value,
                    "rhs": ValueRef.temporary("post_shift"),
                },
                results={"dst": result},
                result_types={"dst": _RESULT},
            ),
        ]
    )
    return tuple(emits)


def _magic_division_sgpr_descriptors(
    descriptor_set: DescriptorSet, *, kind: UnsignedDivisorMagicKind
) -> tuple[Descriptor, ...]:
    move = descriptor_by_key(descriptor_set, "amdgpu.s_mov_b32")
    multiply_hi = descriptor_by_key(descriptor_set, "amdgpu.s_mul_hi_u32")
    subtract = descriptor_by_key(descriptor_set, "amdgpu.s_sub_u32")
    shift = descriptor_by_key(descriptor_set, "amdgpu.s_lshr_b32")
    add = descriptor_by_key(descriptor_set, "amdgpu.s_add_u32")
    return (
        (move, multiply_hi, subtract, shift, add)
        if kind.has_add
        else (move, multiply_hi, shift)
        if kind.has_shift
        else (move, multiply_hi)
    )


def _magic_division_vgpr_emits(
    descriptor_set: DescriptorSet,
    *,
    materializer: ValueMaterializer,
    kind: UnsignedDivisorMagicKind,
    result: ValueRef,
) -> tuple[EmitDescriptorOp, ...]:
    move = descriptor_by_key(descriptor_set, "amdgpu.v_mov_b32")
    multiply_hi = descriptor_by_key(descriptor_set, "amdgpu.v_mul_hi_u32")
    subtract = descriptor_by_key(descriptor_set, "amdgpu.v_sub_u32")
    shift = descriptor_by_key(descriptor_set, "amdgpu.v_lshrrev_b32.src0_inline")
    add = descriptor_by_key(descriptor_set, "amdgpu.v_add_u32")
    quotient_value = (
        ValueRef.temporary("adjusted_quotient")
        if kind.has_add
        else ValueRef.temporary("quotient")
    )
    emits = [
        EmitDescriptorOp(
            descriptor=move,
            results={"dst": ValueRef.temporary("magic")},
            result_types={"dst": _RESULT},
            immediates={
                "imm32": ValueProject.u32_divisor_magic_multiplier("lhs", "rhs")
            },
        ),
        EmitDescriptorOp(
            descriptor=multiply_hi,
            operands={
                "lhs": _materialized_operand("lhs", materializer),
                "rhs": ValueRef.temporary("magic"),
            },
            results={
                "dst": ValueRef.temporary("quotient") if kind.has_shift else result
            },
            result_types={"dst": _RESULT},
            form=DescriptorEmitForm.OP,
        ),
    ]
    if kind.has_add:
        emits.extend(
            [
                EmitDescriptorOp(
                    descriptor=subtract,
                    operands={
                        "lhs": _materialized_operand(
                            "lhs",
                            materializer,
                        ),
                        "rhs": ValueRef.temporary("quotient"),
                    },
                    results={"dst": ValueRef.temporary("npq")},
                    result_types={"dst": _RESULT},
                    form=DescriptorEmitForm.OP,
                ),
                EmitDescriptorOp(
                    descriptor=shift,
                    operands={"value": ValueRef.temporary("npq")},
                    results={"dst": ValueRef.temporary("npq_half")},
                    result_types={"dst": _RESULT},
                    immediates={"imm32": 1},
                    form=DescriptorEmitForm.OP,
                ),
                EmitDescriptorOp(
                    descriptor=add,
                    operands={
                        "lhs": ValueRef.temporary("npq_half"),
                        "rhs": ValueRef.temporary("quotient"),
                    },
                    results={"dst": ValueRef.temporary("adjusted_quotient")},
                    result_types={"dst": _RESULT},
                    form=DescriptorEmitForm.OP,
                ),
            ]
        )
    if not kind.has_shift:
        return tuple(emits)
    emits.append(
        EmitDescriptorOp(
            descriptor=shift,
            operands={"value": quotient_value},
            results={"dst": result},
            result_types={"dst": _RESULT},
            immediates={
                "imm32": ValueProject.u32_divisor_magic_shift(
                    "lhs", "rhs", product_bit_width=32
                )
            },
            form=DescriptorEmitForm.OP,
        )
    )
    return tuple(emits)


def _magic_division_vgpr_descriptors(
    descriptor_set: DescriptorSet, *, kind: UnsignedDivisorMagicKind
) -> tuple[Descriptor, ...]:
    move = descriptor_by_key(descriptor_set, "amdgpu.v_mov_b32")
    multiply_hi = descriptor_by_key(descriptor_set, "amdgpu.v_mul_hi_u32")
    subtract = descriptor_by_key(descriptor_set, "amdgpu.v_sub_u32")
    shift = descriptor_by_key(descriptor_set, "amdgpu.v_lshrrev_b32.src0_inline")
    add = descriptor_by_key(descriptor_set, "amdgpu.v_add_u32")
    return (
        (move, multiply_hi, subtract, shift, add)
        if kind.has_add
        else (move, multiply_hi, shift)
        if kind.has_shift
        else (move, multiply_hi)
    )


def _magic_division_sgpr_rule(
    descriptor_set: DescriptorSet,
    source_op: Op,
    type_pattern: TypePattern,
    *,
    kind: UnsignedDivisorMagicKind,
) -> DescriptorRule:
    multiply_hi = descriptor_by_key(descriptor_set, "amdgpu.s_mul_hi_u32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=multiply_hi,
        guards=(
            *_magic_division_guards(
                type_pattern,
                register_class="amdgpu.sgpr",
                kind=kind,
            ),
            *_descriptor_available_guards(
                *_magic_division_sgpr_descriptors(descriptor_set, kind=kind)
            ),
        ),
        emit=_magic_division_sgpr_emits(
            descriptor_set,
            type_pattern=type_pattern,
            kind=kind,
            result=_RESULT,
        ),
    )


def _magic_division_vgpr_rule(
    descriptor_set: DescriptorSet,
    source_op: Op,
    type_pattern: TypePattern,
    *,
    kind: UnsignedDivisorMagicKind,
) -> DescriptorRule:
    materializer = (
        ADDRESS_VGPR_MATERIALIZER if type_pattern == _INDEX else I32_VGPR_MATERIALIZER
    )
    multiply_hi = descriptor_by_key(descriptor_set, "amdgpu.v_mul_hi_u32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=multiply_hi,
        guards=(
            *_magic_division_guards(
                type_pattern,
                register_class="amdgpu.vgpr",
                kind=kind,
            ),
            *_descriptor_available_guards(
                *_magic_division_vgpr_descriptors(descriptor_set, kind=kind)
            ),
        ),
        emit=_magic_division_vgpr_emits(
            descriptor_set, materializer=materializer, kind=kind, result=_RESULT
        ),
    )


def _magic_remainder_sgpr_rule(
    descriptor_set: DescriptorSet,
    source_op: Op,
    type_pattern: TypePattern,
    *,
    kind: UnsignedDivisorMagicKind,
) -> DescriptorRule:
    multiply_lo = descriptor_by_key(descriptor_set, "amdgpu.s_mul_i32")
    subtract = descriptor_by_key(descriptor_set, "amdgpu.s_sub_u32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=multiply_lo,
        guards=(
            *_magic_division_guards(
                type_pattern,
                register_class="amdgpu.sgpr",
                kind=kind,
            ),
            *_descriptor_available_guards(
                *_magic_division_sgpr_descriptors(descriptor_set, kind=kind),
                multiply_lo,
                subtract,
            ),
        ),
        emit=(
            *_magic_division_sgpr_emits(
                descriptor_set,
                type_pattern=type_pattern,
                kind=kind,
                result=ValueRef.temporary("quotient_final"),
            ),
            EmitDescriptorOp(
                descriptor=multiply_lo,
                operands={
                    "lhs": ValueRef.temporary("quotient_final"),
                    "rhs": _sgpr_operand("rhs", type_pattern),
                },
                results={"dst": ValueRef.temporary("product")},
                result_types={"dst": _RESULT},
            ),
            EmitDescriptorOp(
                descriptor=subtract,
                operands={
                    "lhs": _sgpr_operand("lhs", type_pattern),
                    "rhs": ValueRef.temporary("product"),
                },
                results={"dst": _RESULT},
            ),
        ),
    )


def _magic_remainder_vgpr_rule(
    descriptor_set: DescriptorSet,
    source_op: Op,
    type_pattern: TypePattern,
    *,
    kind: UnsignedDivisorMagicKind,
    product_shift: int | None = None,
) -> DescriptorRule:
    materializer = (
        ADDRESS_VGPR_MATERIALIZER if type_pattern == _INDEX else I32_VGPR_MATERIALIZER
    )
    quotient = ValueRef.temporary("quotient_final")
    product_guards: tuple[Guard, ...] = ()
    product_immediates: dict[str, int] = {}
    if product_shift is None:
        product_descriptor = descriptor_by_key(descriptor_set, "amdgpu.v_mul_lo_u32")
        product_operands = {
            "lhs": quotient,
            "rhs": _materialized_operand("rhs", materializer),
        }
    else:
        # q * (2^shift + 1) is (q << shift) + q modulo 2^32.
        divisor = (1 << product_shift) + 1
        product_guards = (Guard.value_i64_range("rhs", divisor, divisor),)
        product_descriptor = descriptor_by_key(
            descriptor_set, "amdgpu.v_lshl_add_u32.shift_imm"
        )
        product_operands = {"value": quotient, "addend": quotient}
        product_immediates = {"shift": product_shift}
    subtract = descriptor_by_key(descriptor_set, "amdgpu.v_sub_u32")
    return DescriptorRule(
        source_op=source_op,
        descriptor=product_descriptor,
        guards=(
            *_magic_division_guards(
                type_pattern,
                register_class="amdgpu.vgpr",
                kind=kind,
                divisor_guards=product_guards,
            ),
            *_descriptor_available_guards(
                *_magic_division_vgpr_descriptors(descriptor_set, kind=kind),
                product_descriptor,
                subtract,
            ),
        ),
        emit=(
            *_magic_division_vgpr_emits(
                descriptor_set,
                materializer=materializer,
                kind=kind,
                result=quotient,
            ),
            EmitDescriptorOp(
                descriptor=product_descriptor,
                operands=product_operands,
                results={"dst": ValueRef.temporary("product")},
                result_types={"dst": _RESULT},
                immediates=product_immediates,
                form=DescriptorEmitForm.OP,
            ),
            EmitDescriptorOp(
                descriptor=subtract,
                operands={
                    "lhs": _materialized_operand("lhs", materializer),
                    "rhs": ValueRef.temporary("product"),
                },
                results={"dst": _RESULT},
                form=DescriptorEmitForm.OP,
            ),
        ),
    )


def _high_bit_division_rule(
    descriptor_set: DescriptorSet,
    source_op: Op,
    register_class: str,
    divisor_range: tuple[int, int],
) -> DescriptorRule:
    scalar_register = register_class == "amdgpu.sgpr"
    operands = (
        tuple(
            _materialized_operand(field, REGISTERS_SGPR_MATERIALIZER)
            for field in ("lhs", "rhs")
        )
        if scalar_register
        else tuple(
            _materialized_operand(field, I32_VGPR_MATERIALIZER)
            for field in ("lhs", "rhs")
        )
    )
    register_guards = (
        tuple(
            Guard.value_materializable(field, REGISTERS_SGPR_MATERIALIZER.name)
            for field in ("lhs", "rhs")
        )
        if scalar_register
        else tuple(
            Guard.value_materializable(field, I32_VGPR_MATERIALIZER.name)
            for field in ("lhs", "rhs")
        )
    )
    prefix = "s" if scalar_register else "v"
    if source_op is scalar_arithmetic.scalar_divui:
        # For d >= 2^31, the quotient is exactly the unsigned predicate n >= d.
        compare = descriptor_by_key(
            descriptor_set,
            "amdgpu.s_cmp_ge_u32" if scalar_register else "amdgpu.v_cmp_uge_u32",
        )
        move = descriptor_by_key(descriptor_set, f"amdgpu.{prefix}_mov_b32")
        descriptor = descriptor_by_key(
            descriptor_set,
            "amdgpu.s_cselect_b32" if scalar_register else "amdgpu.v_cndmask_b32",
        )
        condition_result = "scc" if scalar_register else "mask"
        condition_operand = "condition" if scalar_register else "mask"
        descriptors = (compare, move, descriptor)
        emit = (
            EmitDescriptorOp(
                descriptor=compare,
                operands={"lhs": operands[0], "rhs": operands[1]},
                results={condition_result: ValueRef.temporary("condition")},
                result_types={condition_result: DescriptorResultType()},
            ),
            EmitDescriptorOp(
                descriptor=move,
                results={"dst": ValueRef.temporary("zero")},
                result_types={"dst": _RESULT},
                immediates={"imm32": 0},
            ),
            EmitDescriptorOp(
                descriptor=move,
                results={"dst": ValueRef.temporary("one")},
                result_types={"dst": _RESULT},
                immediates={"imm32": 1},
            ),
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={
                    "true_value": ValueRef.temporary("one"),
                    "false_value": ValueRef.temporary("zero"),
                    condition_operand: ValueRef.temporary("condition"),
                },
                results={"dst": _RESULT},
            ),
        )
    elif divisor_range[0] == divisor_range[1]:
        descriptor = descriptor_by_key(descriptor_set, f"amdgpu.{prefix}_and_b32.lit")
        descriptors = (descriptor,)
        emit = (
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"lhs" if scalar_register else "rhs": operands[0]},
                results={"dst": _RESULT},
                immediates={"imm32": 0x7FFFFFFF},
            ),
        )
    else:
        # For d >= 2^31, n/d is zero or one. The wrapped subtraction is
        # greater than n precisely when n < d, so unsigned min selects n%d.
        subtract = descriptor_by_key(descriptor_set, f"amdgpu.{prefix}_sub_u32")
        descriptor = descriptor_by_key(descriptor_set, f"amdgpu.{prefix}_min_u32")
        descriptors = (subtract, descriptor)
        emit = (
            EmitDescriptorOp(
                descriptor=subtract,
                operands={"lhs": operands[0], "rhs": operands[1]},
                results={"dst": ValueRef.temporary("difference")},
                result_types={"dst": _RESULT},
            ),
            EmitDescriptorOp(
                descriptor=descriptor,
                operands={"lhs": operands[0], "rhs": ValueRef.temporary("difference")},
                results={"dst": _RESULT},
            ),
        )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            *(Guard.value_type(field, _I32) for field in ("lhs", "rhs", "result")),
            Guard.low_value_register_class("result", register_class),
            Guard.value_exact_i64("rhs"),
            Guard.value_i64_range("rhs", *divisor_range),
            *register_guards,
            *_descriptor_available_guards(*descriptors),
        ),
        emit=emit,
    )


def _signed_magic_guards(
    *,
    register_class: str,
    kind: SignedDivisorMagicKind,
    needs_divisor_value: bool,
) -> tuple[Guard, ...]:
    if register_class == "amdgpu.sgpr":
        value_guards = [
            Guard.value_materializable("lhs", REGISTERS_SGPR_MATERIALIZER.name)
        ]
        if needs_divisor_value:
            value_guards.append(
                Guard.value_materializable("rhs", REGISTERS_SGPR_MATERIALIZER.name)
            )
    else:
        value_guards = [Guard.value_materializable("lhs", I32_VGPR_MATERIALIZER.name)]
        if needs_divisor_value:
            value_guards.append(
                Guard.value_materializable("rhs", I32_VGPR_MATERIALIZER.name)
            )
    return (
        *(Guard.value_type(field, _I32) for field in ("lhs", "rhs", "result")),
        Guard.low_value_register_class("result", register_class),
        Guard.value_i32_divisor_magic_kind(
            "rhs", kind, diagnostic=_SIGNED_I32_DIVISOR_DIAGNOSTIC
        ),
        *value_guards,
    )


def _signed_magic_descriptors(
    descriptor_set: DescriptorSet,
    *,
    register_class: str,
    kind: SignedDivisorMagicKind,
    is_remainder: bool,
) -> tuple[Descriptor, ...]:
    prefix = "s" if register_class == "amdgpu.sgpr" else "v"
    descriptors = [
        descriptor_by_key(descriptor_set, f"amdgpu.{prefix}_mov_b32"),
        descriptor_by_key(descriptor_set, f"amdgpu.{prefix}_mul_hi_i32"),
        descriptor_by_key(
            descriptor_set,
            "amdgpu.s_lshr_b32.rhs_inline"
            if prefix == "s"
            else "amdgpu.v_lshrrev_b32.src0_inline",
        ),
        descriptor_by_key(descriptor_set, f"amdgpu.{prefix}_add_u32"),
    ]
    if kind.numerator_factor != 0:
        descriptors.append(
            descriptor_by_key(
                descriptor_set,
                f"amdgpu.{prefix}_{'add' if kind.numerator_factor > 0 else 'sub'}_u32",
            )
        )
    if kind.has_shift:
        descriptors.append(
            descriptor_by_key(
                descriptor_set,
                "amdgpu.s_ashr_i32.rhs_inline"
                if prefix == "s"
                else "amdgpu.v_ashrrev_i32.src0_inline",
            )
        )
    if is_remainder:
        descriptors.extend(
            (
                descriptor_by_key(descriptor_set, f"amdgpu.{prefix}_mul_i32")
                if prefix == "s"
                else descriptor_by_key(descriptor_set, "amdgpu.v_mul_lo_u32"),
                descriptor_by_key(descriptor_set, f"amdgpu.{prefix}_sub_u32"),
            )
        )
    return tuple(descriptors)


def _signed_magic_emits(
    descriptor_set: DescriptorSet,
    *,
    register_class: str,
    kind: SignedDivisorMagicKind,
    result: ValueRef,
) -> tuple[EmitDescriptorOp, ...]:
    scalar_register = register_class == "amdgpu.sgpr"
    prefix = "s" if scalar_register else "v"
    form = DescriptorEmitForm.AUTO if scalar_register else DescriptorEmitForm.OP
    lhs = (
        _materialized_operand("lhs", REGISTERS_SGPR_MATERIALIZER)
        if scalar_register
        else _materialized_operand("lhs", I32_VGPR_MATERIALIZER)
    )
    move = descriptor_by_key(descriptor_set, f"amdgpu.{prefix}_mov_b32")
    multiply_hi = descriptor_by_key(descriptor_set, f"amdgpu.{prefix}_mul_hi_i32")
    emits = [
        EmitDescriptorOp(
            descriptor=move,
            results={"dst": ValueRef.temporary("magic")},
            result_types={"dst": _RESULT},
            immediates={"imm32": ValueProject.i32_divisor_magic_multiplier("rhs")},
        ),
        EmitDescriptorOp(
            descriptor=multiply_hi,
            operands={"lhs": lhs, "rhs": ValueRef.temporary("magic")},
            results={"dst": ValueRef.temporary("high_product")},
            result_types={"dst": _RESULT},
            form=form,
        ),
    ]
    quotient = ValueRef.temporary("high_product")
    if kind.numerator_factor != 0:
        adjust = descriptor_by_key(
            descriptor_set,
            f"amdgpu.{prefix}_{'add' if kind.numerator_factor > 0 else 'sub'}_u32",
        )
        emits.append(
            EmitDescriptorOp(
                descriptor=adjust,
                operands={"lhs": quotient, "rhs": lhs},
                results={"dst": ValueRef.temporary("adjusted_quotient")},
                result_types={"dst": _RESULT},
                form=form,
            )
        )
        quotient = ValueRef.temporary("adjusted_quotient")
    if kind.has_shift:
        shift = descriptor_by_key(
            descriptor_set,
            "amdgpu.s_ashr_i32.rhs_inline"
            if scalar_register
            else "amdgpu.v_ashrrev_i32.src0_inline",
        )
        emits.append(
            EmitDescriptorOp(
                descriptor=shift,
                operands={"lhs" if scalar_register else "value": quotient},
                results={"dst": ValueRef.temporary("shifted_quotient")},
                result_types={"dst": _RESULT},
                immediates={"imm32": ValueProject.i32_divisor_magic_shift("rhs")},
                form=form,
            )
        )
        quotient = ValueRef.temporary("shifted_quotient")
    sign = descriptor_by_key(
        descriptor_set,
        "amdgpu.s_lshr_b32.rhs_inline"
        if scalar_register
        else "amdgpu.v_lshrrev_b32.src0_inline",
    )
    add = descriptor_by_key(descriptor_set, f"amdgpu.{prefix}_add_u32")
    emits.extend(
        (
            EmitDescriptorOp(
                descriptor=sign,
                operands={"lhs" if scalar_register else "value": quotient},
                results={"dst": ValueRef.temporary("sign_correction")},
                result_types={"dst": _RESULT},
                immediates={"imm32": 31},
                form=form,
            ),
            EmitDescriptorOp(
                descriptor=add,
                operands={
                    "lhs": quotient,
                    "rhs": ValueRef.temporary("sign_correction"),
                },
                results={"dst": result},
                result_types={"dst": _RESULT},
                form=form,
            ),
        )
    )
    return tuple(emits)


def _signed_magic_rule(
    descriptor_set: DescriptorSet,
    source_op: Op,
    *,
    register_class: str,
    kind: SignedDivisorMagicKind,
) -> DescriptorRule:
    is_remainder = source_op is scalar_arithmetic.scalar_remsi
    quotient = ValueRef.temporary("quotient") if is_remainder else _RESULT
    emits = list(
        _signed_magic_emits(
            descriptor_set,
            register_class=register_class,
            kind=kind,
            result=quotient,
        )
    )
    if is_remainder:
        scalar_register = register_class == "amdgpu.sgpr"
        form = DescriptorEmitForm.AUTO if scalar_register else DescriptorEmitForm.OP
        multiply_lo = descriptor_by_key(
            descriptor_set,
            "amdgpu.s_mul_i32" if scalar_register else "amdgpu.v_mul_lo_u32",
        )
        subtract = descriptor_by_key(
            descriptor_set,
            "amdgpu.s_sub_u32" if scalar_register else "amdgpu.v_sub_u32",
        )
        lhs = (
            _materialized_operand("lhs", REGISTERS_SGPR_MATERIALIZER)
            if scalar_register
            else _materialized_operand("lhs", I32_VGPR_MATERIALIZER)
        )
        rhs = (
            _materialized_operand("rhs", REGISTERS_SGPR_MATERIALIZER)
            if scalar_register
            else _materialized_operand("rhs", I32_VGPR_MATERIALIZER)
        )
        emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=multiply_lo,
                    operands={"lhs": quotient, "rhs": rhs},
                    results={"dst": ValueRef.temporary("quotient_product")},
                    result_types={"dst": _RESULT},
                    form=form,
                ),
                EmitDescriptorOp(
                    descriptor=subtract,
                    operands={
                        "lhs": lhs,
                        "rhs": ValueRef.temporary("quotient_product"),
                    },
                    results={"dst": _RESULT},
                    form=form,
                ),
            )
        )
    multiply_hi = descriptor_by_key(
        descriptor_set,
        "amdgpu.s_mul_hi_i32"
        if register_class == "amdgpu.sgpr"
        else "amdgpu.v_mul_hi_i32",
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=multiply_hi,
        guards=(
            *_signed_magic_guards(
                register_class=register_class,
                kind=kind,
                needs_divisor_value=is_remainder,
            ),
            *_descriptor_available_guards(
                *_signed_magic_descriptors(
                    descriptor_set,
                    register_class=register_class,
                    kind=kind,
                    is_remainder=is_remainder,
                )
            ),
        ),
        emit=tuple(emits),
    )


def _dynamic_division_descriptors(
    descriptor_set: DescriptorSet, *, register_class: str, signed: bool
) -> tuple[Descriptor, ...]:
    keys = [
        "amdgpu.v_cvt_f32_u32",
        "amdgpu.v_rcp_f32",
        "amdgpu.v_mul_f32.lit",
        "amdgpu.v_cvt_u32_f32",
    ]
    if register_class == "amdgpu.sgpr":
        keys.extend(
            (
                "amdgpu.v_readfirstlane_b32",
                "amdgpu.s_mov_b32",
                "amdgpu.s_mul_i32",
                "amdgpu.s_mul_hi_u32",
                "amdgpu.s_add_u32",
                "amdgpu.s_sub_u32",
                "amdgpu.s_sub_co_u32",
                "amdgpu.s_cselect_b32",
            )
        )
        if signed:
            keys.extend(("amdgpu.s_xor_b32", "amdgpu.s_ashr_i32"))
    else:
        keys.extend(
            (
                "amdgpu.v_mov_b32",
                "amdgpu.v_mul_lo_u32",
                "amdgpu.v_mul_hi_u32",
                "amdgpu.v_add_u32",
                "amdgpu.v_sub_u32",
                "amdgpu.v_sub_co_u32",
                "amdgpu.v_cndmask_b32",
            )
        )
        if signed:
            keys.extend(("amdgpu.v_xor_b32", "amdgpu.v_ashrrev_i32.src0_inline"))
    return tuple(descriptor_by_key(descriptor_set, key) for key in keys)


def _dynamic_division_rule(
    descriptor_set: DescriptorSet,
    source_op: Op,
    type_pattern: TypePattern,
    *,
    register_class: str,
    signed: bool,
) -> RecipeRule:
    result_fields = (
        ("quotient", "remainder")
        if source_op
        in (
            scalar_arithmetic.scalar_divremsi,
            scalar_arithmetic.scalar_divremui,
            index.index_divrem,
        )
        else ("result",)
    )
    value_fields = ("lhs", "rhs", *result_fields)
    guards = [Guard.value_type(field, type_pattern) for field in value_fields]
    guards.append(Guard.value_not_exact_i64("rhs"))
    if type_pattern == _INDEX:
        guards.extend(
            (
                Guard.value_unsigned_bit_count(
                    "lhs", 32, diagnostic=_ADDRESS_U32_DIAGNOSTIC
                ),
                Guard.value_i64_range(
                    "rhs",
                    1,
                    _UINT32_MAX,
                    diagnostic=_POSITIVE_U32_DIVISOR_DIAGNOSTIC,
                ),
                *(
                    Guard.value_unsigned_bit_count(
                        field, 32, diagnostic=_ADDRESS_U32_DIAGNOSTIC
                    )
                    for field in result_fields
                ),
            )
        )
    if register_class == "amdgpu.sgpr":
        materializer = (
            ADDRESS_SGPR_MATERIALIZER
            if type_pattern == _INDEX
            else REGISTERS_SGPR_MATERIALIZER
        )
        guards.extend(
            Guard.value_materializable(field, materializer.name)
            for field in ("lhs", "rhs")
        )
        guards.extend(
            Guard.low_value_register_class(field, register_class)
            for field in result_fields
        )
    else:
        materializer = (
            ADDRESS_VGPR_MATERIALIZER
            if type_pattern == _INDEX
            else I32_VGPR_MATERIALIZER
        )
        guards.extend(
            Guard.value_materializable(field, materializer.name)
            for field in ("lhs", "rhs")
        )
        guards.extend(
            Guard.low_value_register_class(field, register_class)
            for field in result_fields
        )
    guards.extend(
        _descriptor_available_guards(
            *_dynamic_division_descriptors(
                descriptor_set, register_class=register_class, signed=signed
            )
        )
    )
    return RecipeRule(source_op=source_op, guards=guards)


def integer_division_rules(
    descriptor_set: DescriptorSet,
) -> tuple[DescriptorRule | RecipeRule, ...]:
    """Builds exact 32-bit constant and runtime division rules."""
    rules = [
        _magic_remainder_vgpr_rule(
            descriptor_set, source_op, type_pattern, kind=kind, product_shift=1
        )
        for source_op, type_pattern in (
            (index.index_rem, _INDEX),
            (scalar_arithmetic.scalar_remui, _I32),
        )
        for kind in (
            UnsignedDivisorMagicKind.MULTIPLY,
            UnsignedDivisorMagicKind.MULTIPLY_SHIFT,
        )
    ]
    for kind in UnsignedDivisorMagicKind:
        for source_op, type_pattern in (
            (index.index_div, _INDEX),
            (scalar_arithmetic.scalar_divui, _I32),
        ):
            rules.extend(
                (
                    _magic_division_sgpr_rule(
                        descriptor_set, source_op, type_pattern, kind=kind
                    ),
                    _magic_division_vgpr_rule(
                        descriptor_set, source_op, type_pattern, kind=kind
                    ),
                )
            )
        for source_op, type_pattern in (
            (index.index_rem, _INDEX),
            (scalar_arithmetic.scalar_remui, _I32),
        ):
            rules.extend(
                (
                    _magic_remainder_sgpr_rule(
                        descriptor_set, source_op, type_pattern, kind=kind
                    ),
                    _magic_remainder_vgpr_rule(
                        descriptor_set, source_op, type_pattern, kind=kind
                    ),
                )
            )
    for register_class in ("amdgpu.sgpr", "amdgpu.vgpr"):
        rules.append(
            _high_bit_division_rule(
                descriptor_set,
                scalar_arithmetic.scalar_divui,
                register_class,
                (-(2**31), -1),
            )
        )
        rules.extend(
            _high_bit_division_rule(
                descriptor_set,
                scalar_arithmetic.scalar_remui,
                register_class,
                divisor_range,
            )
            for divisor_range in (
                (-(2**31), -(2**31)),
                (-(2**31) + 1, -1),
            )
        )
    rules.extend(
        _signed_magic_rule(
            descriptor_set,
            source_op,
            register_class=register_class,
            kind=kind,
        )
        for source_op in (
            scalar_arithmetic.scalar_divsi,
            scalar_arithmetic.scalar_remsi,
        )
        for register_class in ("amdgpu.sgpr", "amdgpu.vgpr")
        for kind in SignedDivisorMagicKind
    )
    rules.extend(
        _dynamic_division_rule(
            descriptor_set,
            source_op,
            type_pattern,
            register_class=register_class,
            signed=signed,
        )
        for source_op, type_pattern, signed in (
            (index.index_div, _INDEX, False),
            (index.index_rem, _INDEX, False),
            (index.index_divrem, _INDEX, False),
            (scalar_arithmetic.scalar_divui, _I32, False),
            (scalar_arithmetic.scalar_remui, _I32, False),
            (scalar_arithmetic.scalar_divremui, _I32, False),
            (scalar_arithmetic.scalar_divsi, _I32, True),
            (scalar_arithmetic.scalar_remsi, _I32, True),
            (scalar_arithmetic.scalar_divremsi, _I32, True),
        )
        for register_class in ("amdgpu.sgpr", "amdgpu.vgpr")
    )
    return tuple(rules)
