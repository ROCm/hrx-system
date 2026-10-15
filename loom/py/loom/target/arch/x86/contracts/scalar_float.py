# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AVX2 scalar floating-point contract rules."""

from __future__ import annotations

from loom.dialect.scalar import arithmetic as scalar_arithmetic
from loom.dialect.scalar import comparison as scalar_comparison
from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.scalar import math as scalar_math
from loom.dialect.scf import defs as scf
from loom.dsl import Op
from loom.target.arch.x86.contracts.constants import (
    f32_scalar_constant_rule,
    f64_scalar_constant_rule,
    floating_scalar_zero_rule,
)
from loom.target.arch.x86.contracts.rule_builders import (
    DescriptorLookup as _DescriptorLookup,
)
from loom.target.arch.x86.contracts.rule_builders import (
    emit_descriptor_op as _op_emit,
)
from loom.target.arch.x86.contracts.rule_builders import (
    value_type_guards as _typed_guards,
)
from loom.target.arch.x86.vector_families import (
    AVX2_FLOAT_COMPARE_MNEMONICS,
    AVX2_SCALAR_FLOAT_BINARY_FAMILIES,
    AVX2_SCALAR_FLOAT_FMA_MNEMONICS,
    FLOAT_ELEMENTS,
    X86_FLOAT_COMPARE_IMMEDIATES,
)
from loom.target.contracts import (
    AttrProject,
    ContractCase,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    Guard,
    Scalar,
    SourceNode,
    TypePattern,
    ValueRef,
    Vector,
)
from loom.target.contracts.templates import (
    DirectDescriptorCase,
    binary_descriptor_rules,
    ternary_descriptor_rules,
)

_I1 = Scalar("i1")
_I32 = Scalar("i32")
_I64 = Scalar("i64")
_BF16 = Scalar("bf16")
_F32 = Scalar("f32")
_F64 = Scalar("f64")
_V2I64 = Vector("i64", lanes=2)


def scalar_register_conversion_rule(
    source_op: Op,
    source_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
    *,
    priority: int = 0,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"input": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
        priority=priority,
    )


def _bf16_to_f32_rule(descriptor_lookup: _DescriptorLookup) -> DescriptorRule:
    shift = descriptor_lookup("x86.scalar.shl.imm.gpr32")
    move = descriptor_lookup("x86.avx2.vmovd.xmm.gpr32")
    # BF16 is the high half of the FP32 encoding. The shift discards unused
    # carrier bits and preserves subnormals without floating-point arithmetic.
    return DescriptorRule(
        source_op=scalar_conversion.scalar_extf,
        descriptor=move,
        guards=(
            Guard.value_type("input", _BF16),
            Guard.value_type("result", _F32),
        ),
        emit=(
            _op_emit(
                descriptor=shift,
                operands={"lhs": ValueRef.operand("input")},
                results={"dst": ValueRef.temporary("bits")},
                result_types={"dst": _I32},
                immediates={"shift": 16},
            ),
            _op_emit(
                descriptor=move,
                operands={"input": ValueRef.temporary("bits")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def xmm_scalar_select_rule(
    type_pattern: TypePattern,
    descriptor_lookup: _DescriptorLookup,
    *,
    priority: int = 0,
) -> DescriptorRule:
    move = descriptor_lookup("x86.avx2.vmovd.xmm.gpr32")
    shift = descriptor_lookup("x86.avx2.vpsllq.xmm")
    blend = descriptor_lookup("x86.avx2.vblendvpd.xmm")
    # Both scalar float widths occupy the low qword. Selecting that qword
    # preserves every payload bit; the remaining XMM bits have no scalar meaning.
    return DescriptorRule(
        source_op=scf.scf_select,
        descriptor=blend,
        guards=(
            Guard.value_type("condition", _I1),
            *_typed_guards(("true_value", "false_value", "result"), type_pattern),
        ),
        emit=(
            _op_emit(
                descriptor=move,
                operands={"input": ValueRef.operand("condition")},
                results={"dst": ValueRef.temporary("condition_bits")},
                result_types={"dst": _V2I64},
            ),
            _op_emit(
                descriptor=shift,
                operands={"source": ValueRef.temporary("condition_bits")},
                results={"dst": ValueRef.temporary("mask")},
                result_types={"dst": _V2I64},
                immediates={"shift": 63},
            ),
            _op_emit(
                descriptor=blend,
                operands={
                    "false_value": ValueRef.operand("false_value"),
                    "true_value": ValueRef.operand("true_value"),
                    "mask": ValueRef.temporary("mask"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
        priority=priority,
    )


def _scalar_float_compare_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    move_bits = descriptor_lookup("x86.avx2.vmovd.gpr32.xmm")
    mask_bit = descriptor_lookup("x86.scalar.and.imm.gpr32")
    blend = descriptor_lookup("x86.avx2.vpblendvb.xmm")
    for element in FLOAT_ELEMENTS:
        value_type = Scalar(element.name)
        # Scalar payloads occupy the low XMM lane. The packed compare's low
        # all-bits mask can be canonicalized or consumed directly by a blend.
        compare = descriptor_lookup(
            f"x86.avx2.{AVX2_FLOAT_COMPARE_MNEMONICS[element.name]}.xmm"
        )
        compare_emit = _op_emit(
            descriptor=compare,
            operands={
                "lhs": ValueRef.operand("lhs"),
                "rhs": ValueRef.operand("rhs"),
            },
            results={"dst": ValueRef.temporary("comparison_mask")},
            result_types={"dst": DescriptorResultType()},
            immediates={
                "predicate": AttrProject.enum_remap(
                    "predicate", X86_FLOAT_COMPARE_IMMEDIATES
                )
            },
        )
        rules.append(
            DescriptorRule(
                source_op=scalar_comparison.scalar_cmpf,
                descriptor=blend,
                source_nodes=(
                    SourceNode.exclusive_user(
                        "select",
                        source_op=scf.scf_select,
                        parent_result=ValueRef.result("result"),
                        node_operand=ValueRef.operand("condition"),
                        guards=(
                            Guard.value_type("condition", _I1),
                            *_typed_guards(
                                ("true_value", "false_value", "result"),
                                value_type,
                            ),
                        ),
                    ),
                ),
                guards=(
                    *_typed_guards(("lhs", "rhs"), value_type),
                    Guard.value_type("result", _I1),
                    Guard.descriptor_available(compare),
                ),
                emit=(
                    compare_emit,
                    _op_emit(
                        descriptor=blend,
                        operands={
                            "false_value": ValueRef.operand(
                                "false_value", source_node="select"
                            ),
                            "true_value": ValueRef.operand(
                                "true_value", source_node="select"
                            ),
                            "mask": ValueRef.temporary("comparison_mask"),
                        },
                        results={
                            "dst": ValueRef.result("result", source_node="select")
                        },
                    ),
                ),
                target_query=False,
            )
        )
        rules.append(
            DescriptorRule(
                source_op=scalar_comparison.scalar_cmpf,
                descriptor=compare,
                guards=(
                    *_typed_guards(("lhs", "rhs"), value_type),
                    Guard.value_type("result", _I1),
                    Guard.descriptor_available(move_bits),
                    Guard.descriptor_available(mask_bit),
                ),
                emit=(
                    compare_emit,
                    _op_emit(
                        descriptor=move_bits,
                        operands={"input": ValueRef.temporary("comparison_mask")},
                        results={"dst": ValueRef.temporary("comparison_bits")},
                        result_types={"dst": _I32},
                    ),
                    _op_emit(
                        descriptor=mask_bit,
                        operands={"lhs": ValueRef.temporary("comparison_bits")},
                        results={"dst": ValueRef.result("result")},
                        immediates={"imm32": 1},
                    ),
                ),
            )
        )
    return tuple(rules)


def avx2_scalar_float_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[ContractCase, ...]:
    return (
        _bf16_to_f32_rule(descriptor_lookup),
        floating_scalar_zero_rule(
            _F32,
            descriptor_lookup("x86.avx2.vxorps.zero.xmm"),
        ),
        floating_scalar_zero_rule(
            _F64,
            descriptor_lookup("x86.avx2.vxorps.zero.xmm"),
        ),
        f32_scalar_constant_rule(
            descriptor_lookup("x86.scalar.movimm.gpr32"),
            descriptor_lookup("x86.avx2.vmovd.xmm.gpr32"),
        ),
        f64_scalar_constant_rule(
            descriptor_lookup("x86.scalar.movimm.gpr64"),
            descriptor_lookup("x86.avx2.vmovq.xmm.gpr64"),
        ),
        *_scalar_float_compare_rules(descriptor_lookup),
        xmm_scalar_select_rule(_F32, descriptor_lookup),
        xmm_scalar_select_rule(_F64, descriptor_lookup),
        scalar_register_conversion_rule(
            scalar_conversion.scalar_bitcast,
            _F32,
            _I32,
            "x86.avx2.vmovd.gpr32.xmm",
            descriptor_lookup,
        ),
        scalar_register_conversion_rule(
            scalar_conversion.scalar_bitcast,
            _I32,
            _F32,
            "x86.avx2.vmovd.xmm.gpr32",
            descriptor_lookup,
        ),
        scalar_register_conversion_rule(
            scalar_conversion.scalar_bitcast,
            _F64,
            _I64,
            "x86.avx2.vmovq.gpr64.xmm",
            descriptor_lookup,
        ),
        scalar_register_conversion_rule(
            scalar_conversion.scalar_bitcast,
            _I64,
            _F64,
            "x86.avx2.vmovq.xmm.gpr64",
            descriptor_lookup,
        ),
        *binary_descriptor_rules(
            tuple(
                DirectDescriptorCase(
                    getattr(scalar_arithmetic, f"scalar_{row.source_operation}"),
                    descriptor_lookup(f"x86.avx2.{row.mnemonic}.xmm"),
                    Scalar(row.element.name),
                )
                for row in AVX2_SCALAR_FLOAT_BINARY_FAMILIES
            ),
            form=DescriptorEmitForm.OP,
        ),
        *ternary_descriptor_rules(
            tuple(
                DirectDescriptorCase(
                    scalar_math.scalar_fmaf,
                    descriptor_lookup(
                        f"x86.avx2.{AVX2_SCALAR_FLOAT_FMA_MNEMONICS[element.name]}.xmm"
                    ),
                    Scalar(element.name),
                )
                for element in FLOAT_ELEMENTS
            ),
            form=DescriptorEmitForm.OP,
            descriptor_a="lhs",
            descriptor_b="rhs",
            descriptor_c="acc",
        ),
    )
