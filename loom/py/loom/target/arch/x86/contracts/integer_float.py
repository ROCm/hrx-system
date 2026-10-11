# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Scalar integer-to-F32/F64 rules, including AVX2 unsigned composition."""

from loom.dialect.scalar import conversion
from loom.dsl import Op
from loom.target.arch.x86.contracts.rule_builders import (
    DescriptorLookup,
    emit_descriptor_op,
)
from loom.target.arch.x86.vector_families import FLOAT_ELEMENTS
from loom.target.contracts import (
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    ValueRef,
    Vector,
)

_I32 = Scalar("i32")
_I64 = Scalar("i64")


def _normalize_integer(
    width: int, *, unsigned: bool, descriptor_lookup: DescriptorLookup
) -> tuple[tuple[EmitDescriptorOp, ...], ValueRef]:
    source = ValueRef.operand("input")
    normalized = ValueRef.temporary("normalized")
    if width == 1:
        if unsigned:
            return (
                emit_descriptor_op(
                    descriptor=descriptor_lookup("x86.scalar.and.imm.gpr32"),
                    operands={"lhs": source},
                    results={"dst": normalized},
                    result_types={"dst": _I32},
                    immediates={"imm32": 1},
                ),
            ), normalized
        return (
            emit_descriptor_op(
                descriptor=descriptor_lookup("x86.scalar.shl.imm.gpr32"),
                operands={"lhs": source},
                results={"dst": ValueRef.temporary("sign")},
                result_types={"dst": _I32},
                immediates={"shift": 31},
            ),
            emit_descriptor_op(
                descriptor=descriptor_lookup("x86.scalar.sar.imm.gpr32"),
                operands={"lhs": ValueRef.temporary("sign")},
                results={"dst": normalized},
                result_types={"dst": _I32},
                immediates={"shift": 31},
            ),
        ), normalized
    if width in (8, 16):
        extension = f"movzx.u{width}" if unsigned else f"movsx.i{width}"
        descriptor = descriptor_lookup(f"x86.scalar.{extension}.gpr32")
        result_type = _I32
    elif unsigned and width == 32:
        descriptor = descriptor_lookup("x86.scalar.movzx.gpr64.gpr32")
        result_type = _I64
    else:
        return (), source
    return (
        emit_descriptor_op(
            descriptor=descriptor,
            operands={"src": source},
            results={"dst": normalized},
            result_types={"dst": result_type},
        ),
    ), normalized


def _conversion_rule(
    source_op: Op,
    width: int,
    float_width: int,
    *,
    native_unsigned: bool,
    descriptor_lookup: DescriptorLookup,
) -> DescriptorRule:
    unsigned = source_op is conversion.scalar_uitofp
    float_type = Scalar(f"f{float_width}")
    suffix = "ss" if float_width == 32 else "sd"
    if native_unsigned:
        prefix, normalized = (), ValueRef.operand("input")
        descriptor_key = f"x86.avx512.vcvtusi2{suffix}.xmm.gpr{width}"
    else:
        native_width = 64 if width == 64 or (unsigned and width == 32) else 32
        prefix, normalized = _normalize_integer(
            width, unsigned=unsigned, descriptor_lookup=descriptor_lookup
        )
        descriptor_key = f"x86.avx2.vcvtsi2{suffix}.xmm.gpr{native_width}"
    convert = descriptor_lookup(descriptor_key)
    passthrough = ValueRef.temporary("passthrough")
    emits = [
        *prefix,
        # Native scalar conversions copy the upper XMM bits from this operand.
        # Give it a defined, dependency-breaking value instead of reading dst.
        emit_descriptor_op(
            descriptor=descriptor_lookup("x86.avx2.vxorps.zero.xmm"),
            results={"dst": passthrough},
            result_types={"dst": float_type},
        ),
    ]
    composed_unsigned64 = unsigned and width == 64 and not native_unsigned
    direct = (
        ValueRef.temporary("direct")
        if composed_unsigned64
        else ValueRef.result("result")
    )
    emits.append(
        emit_descriptor_op(
            descriptor=convert,
            operands={"passthrough": passthrough, "input": normalized},
            results={"dst": direct},
            result_types={"dst": float_type},
        )
    )
    if composed_unsigned64:
        # Above INT64_MAX, jam the lost bit before conversion. This preserves
        # the original value's side of an RNE midpoint; doubling is exact.
        half = ValueRef.temporary("half")
        sticky = ValueRef.temporary("sticky")
        jammed = ValueRef.temporary("jammed")
        converted_half = ValueRef.temporary("converted_half")
        doubled = ValueRef.temporary("doubled")
        mask = ValueRef.temporary("mask")
        emits.extend(
            (
                emit_descriptor_op(
                    descriptor=descriptor_lookup("x86.scalar.shr.imm.gpr64"),
                    operands={"lhs": normalized},
                    results={"dst": half},
                    result_types={"dst": _I64},
                    immediates={"shift": 1},
                ),
                emit_descriptor_op(
                    descriptor=descriptor_lookup("x86.scalar.and.imm.gpr64"),
                    operands={"lhs": normalized},
                    results={"dst": sticky},
                    result_types={"dst": _I64},
                    immediates={"imm32": 1},
                ),
                emit_descriptor_op(
                    descriptor=descriptor_lookup("x86.scalar.or.gpr64"),
                    operands={"lhs": half, "rhs": sticky},
                    results={"dst": jammed},
                    result_types={"dst": _I64},
                ),
                emit_descriptor_op(
                    descriptor=convert,
                    operands={"passthrough": passthrough, "input": jammed},
                    results={"dst": converted_half},
                    result_types={"dst": float_type},
                ),
                emit_descriptor_op(
                    descriptor=descriptor_lookup(f"x86.avx2.vadd{suffix}.xmm"),
                    operands={"lhs": converted_half, "rhs": converted_half},
                    results={"dst": doubled},
                    result_types={"dst": float_type},
                ),
                emit_descriptor_op(
                    descriptor=descriptor_lookup("x86.avx2.vmovq.xmm.gpr64"),
                    operands={"input": normalized},
                    results={"dst": mask},
                    result_types={"dst": Vector("i64", lanes=2)},
                ),
                emit_descriptor_op(
                    descriptor=descriptor_lookup("x86.avx2.vblendvpd.xmm"),
                    operands={
                        "false_value": direct,
                        "true_value": doubled,
                        "mask": mask,
                    },
                    results={"dst": ValueRef.result("result")},
                ),
            )
        )
    return DescriptorRule(
        source_op=source_op,
        descriptor=convert,
        guards=(
            Guard.value_type("input", Scalar(f"i{width}")),
            Guard.value_type("result", float_type),
        ),
        emit=tuple(emits),
        priority=1 if native_unsigned else 0,
    )


def avx2_integer_float_rules(
    descriptor_lookup: DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    return tuple(
        _conversion_rule(
            source_op,
            width,
            element.bit_width,
            native_unsigned=False,
            descriptor_lookup=descriptor_lookup,
        )
        for width in (1, 8, 16, 32, 64)
        for source_op in (conversion.scalar_sitofp, conversion.scalar_uitofp)
        for element in FLOAT_ELEMENTS
    )


def avx512_integer_float_rules(
    descriptor_lookup: DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    return tuple(
        _conversion_rule(
            conversion.scalar_uitofp,
            width,
            element.bit_width,
            native_unsigned=True,
            descriptor_lookup=descriptor_lookup,
        )
        for width in (32, 64)
        for element in FLOAT_ELEMENTS
    )
