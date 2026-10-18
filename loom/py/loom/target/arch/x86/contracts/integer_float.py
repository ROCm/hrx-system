# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete x86 integer-to-F32/F64 conversion contracts."""

from loom.dialect.scalar import conversion
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.x86.contracts.rule_builders import (
    DescriptorLookup,
    emit_descriptor_op,
)
from loom.target.arch.x86.native_conversion import vector_integer_float_mnemonic
from loom.target.arch.x86.vector_families import (
    AVX2_INTEGER_FLOAT_I32_FORMS,
    AVX512_INTEGER_FLOAT_I32_FORMS,
    AVX512_INTEGER_FLOAT_I64_FORMS,
    FLOAT_ELEMENTS,
    IntegerFloatVectorForm,
)
from loom.target.contracts import (
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    ReadOnlyDataLiteral,
    Scalar,
    ValueRef,
    Vector,
)

_I32 = Scalar("i32")
_I64 = Scalar("i64")
_REGISTER_SUFFIXES = {128: "xmm", 256: "ymm", 512: "zmm"}


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


def _vector_descriptor_key(
    form: IntegerFloatVectorForm,
    *,
    unsigned: bool,
    evex: bool,
) -> str:
    feature = "avx512" if evex else "avx2"
    mnemonic = vector_integer_float_mnemonic(
        form.source.bit_width,
        form.result.bit_width,
        unsigned=unsigned,
    )
    result_suffix = _REGISTER_SUFFIXES[max(128, form.result_vector_bit_width)]
    source_suffix = _REGISTER_SUFFIXES[max(128, form.source_vector_bit_width)]
    return f"x86.{feature}.{mnemonic}.{result_suffix}.{source_suffix}"


def _vector_guards(form: IntegerFloatVectorForm) -> tuple[Guard, ...]:
    return (
        Guard.value_type("input", Vector(form.source.name, lanes=form.lane_count)),
        Guard.value_type("result", Vector(form.result.name, lanes=form.lane_count)),
        Guard.value_static_element_count_eq("input", "result"),
    )


def _vector_direct_rule(
    source_op: Op,
    form: IntegerFloatVectorForm,
    *,
    unsigned: bool,
    evex: bool,
    priority: int,
    descriptor_lookup: DescriptorLookup,
) -> DescriptorRule:
    descriptor = descriptor_lookup(
        _vector_descriptor_key(form, unsigned=unsigned, evex=evex)
    )
    return DescriptorRule(
        source_op=source_op,
        descriptor=descriptor,
        guards=_vector_guards(form),
        emit=(
            emit_descriptor_op(
                descriptor=descriptor,
                operands={"input": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
        priority=priority,
    )


def _avx2_unsigned_i32_f32_rule(
    form: IntegerFloatVectorForm,
    descriptor_lookup: DescriptorLookup,
) -> DescriptorRule:
    register_suffix = _REGISTER_SUFFIXES[max(128, form.source_vector_bit_width)]
    integer_type = Vector("i32", lanes=form.lane_count)
    float_type = Vector("f32", lanes=form.lane_count)
    convert = descriptor_lookup(
        _vector_descriptor_key(form, unsigned=False, evex=False)
    )
    shift_right = descriptor_lookup(f"x86.avx2.vpsrld.{register_suffix}")
    shift_left = descriptor_lookup(f"x86.avx2.vpslld.{register_suffix}")
    bitwise_or = descriptor_lookup(f"x86.avx2.vpor.{register_suffix}")
    add = descriptor_lookup(f"x86.avx2.vaddps.{register_suffix}")
    select = descriptor_lookup(f"x86.avx2.vblendvps.{register_suffix}")
    return DescriptorRule(
        source_op=vector.vector_uitofp,
        descriptor=convert,
        guards=_vector_guards(form),
        emit=(
            emit_descriptor_op(
                descriptor=shift_right,
                operands={"source": ValueRef.operand("input")},
                results={"dst": ValueRef.temporary("half")},
                result_types={"dst": integer_type},
                immediates={"shift": 1},
            ),
            emit_descriptor_op(
                descriptor=shift_left,
                operands={"source": ValueRef.operand("input")},
                results={"dst": ValueRef.temporary("sticky_shifted")},
                result_types={"dst": integer_type},
                immediates={"shift": 31},
            ),
            emit_descriptor_op(
                descriptor=shift_right,
                operands={"source": ValueRef.temporary("sticky_shifted")},
                results={"dst": ValueRef.temporary("sticky")},
                result_types={"dst": integer_type},
                immediates={"shift": 31},
            ),
            emit_descriptor_op(
                descriptor=bitwise_or,
                operands={
                    "lhs": ValueRef.temporary("half"),
                    "rhs": ValueRef.temporary("sticky"),
                },
                results={"dst": ValueRef.temporary("jammed")},
                result_types={"dst": integer_type},
            ),
            emit_descriptor_op(
                descriptor=convert,
                operands={"input": ValueRef.operand("input")},
                results={"dst": ValueRef.temporary("direct")},
                result_types={"dst": float_type},
            ),
            emit_descriptor_op(
                descriptor=convert,
                operands={"input": ValueRef.temporary("jammed")},
                results={"dst": ValueRef.temporary("converted_half")},
                result_types={"dst": float_type},
            ),
            emit_descriptor_op(
                descriptor=add,
                operands={
                    "lhs": ValueRef.temporary("converted_half"),
                    "rhs": ValueRef.temporary("converted_half"),
                },
                results={"dst": ValueRef.temporary("doubled")},
                result_types={"dst": float_type},
            ),
            emit_descriptor_op(
                descriptor=select,
                operands={
                    "false_value": ValueRef.temporary("direct"),
                    "true_value": ValueRef.temporary("doubled"),
                    "mask": ValueRef.operand("input"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def _avx2_unsigned_i32_f64_rule(
    form: IntegerFloatVectorForm,
    descriptor_lookup: DescriptorLookup,
) -> DescriptorRule:
    result_bit_width = max(128, form.result_vector_bit_width)
    result_suffix = _REGISTER_SUFFIXES[result_bit_width]
    integer_type = Vector("i64", lanes=form.lane_count)
    float_type = Vector("f64", lanes=form.lane_count)
    widen = descriptor_lookup(f"x86.avx2.vpmovzxdq.{result_suffix}.xmm")
    load = descriptor_lookup(f"x86.avx2.vmovdqu.rodata.{result_suffix}")
    bitwise_or = descriptor_lookup(f"x86.avx2.vpor.{result_suffix}")
    subtract = descriptor_lookup(f"x86.avx2.vsubpd.{result_suffix}")
    exponent_bits = 0x4330000000000000
    exponent = exponent_bits.to_bytes(8, "little") * (result_bit_width // 64)
    return DescriptorRule(
        source_op=vector.vector_uitofp,
        descriptor=widen,
        guards=_vector_guards(form),
        emit=(
            emit_descriptor_op(
                descriptor=widen,
                operands={"source": ValueRef.operand("input")},
                results={"dst": ValueRef.temporary("wide")},
                result_types={"dst": integer_type},
            ),
            emit_descriptor_op(
                descriptor=load,
                results={"dst": ValueRef.temporary("exponent")},
                result_types={"dst": integer_type},
                immediates={"data": ReadOnlyDataLiteral(exponent)},
            ),
            emit_descriptor_op(
                descriptor=bitwise_or,
                operands={
                    "lhs": ValueRef.temporary("wide"),
                    "rhs": ValueRef.temporary("exponent"),
                },
                results={"dst": ValueRef.temporary("biased")},
                result_types={"dst": float_type},
            ),
            emit_descriptor_op(
                descriptor=subtract,
                operands={
                    "lhs": ValueRef.temporary("biased"),
                    "rhs": ValueRef.temporary("exponent"),
                },
                results={"dst": ValueRef.result("result")},
            ),
        ),
    )


def avx2_vector_integer_float_rules(
    descriptor_lookup: DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for form in AVX2_INTEGER_FLOAT_I32_FORMS:
        rules.append(
            _vector_direct_rule(
                vector.vector_sitofp,
                form,
                unsigned=False,
                evex=False,
                priority=0,
                descriptor_lookup=descriptor_lookup,
            )
        )
        rules.append(
            _avx2_unsigned_i32_f32_rule(form, descriptor_lookup)
            if form.result.bit_width == 32
            else _avx2_unsigned_i32_f64_rule(form, descriptor_lookup)
        )
    return tuple(rules)


def avx512_vector_integer_float_rules(
    descriptor_lookup: DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules = [
        _vector_direct_rule(
            vector.vector_sitofp,
            form,
            unsigned=False,
            evex=True,
            priority=1,
            descriptor_lookup=descriptor_lookup,
        )
        for form in AVX512_INTEGER_FLOAT_I32_FORMS[-2:]
    ]
    rules.extend(
        _vector_direct_rule(
            vector.vector_uitofp,
            form,
            unsigned=True,
            evex=True,
            priority=1,
            descriptor_lookup=descriptor_lookup,
        )
        for form in AVX512_INTEGER_FLOAT_I32_FORMS
    )
    rules.extend(
        _vector_direct_rule(
            source_op,
            form,
            unsigned=unsigned,
            evex=True,
            priority=1,
            descriptor_lookup=descriptor_lookup,
        )
        for form in AVX512_INTEGER_FLOAT_I64_FORMS
        for source_op, unsigned in (
            (vector.vector_sitofp, False),
            (vector.vector_uitofp, True),
        )
    )
    return tuple(rules)
