# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete x86 F32/F64-to-integer conversion contracts."""

from __future__ import annotations

from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import defs as vector
from loom.dsl import Op
from loom.target.arch.x86.contracts.rule_builders import (
    DescriptorLookup,
    emit_descriptor_op,
)
from loom.target.arch.x86.contracts.vector_integer_conversion import (
    avx2_integer_narrowing_emits,
    avx512_integer_narrowing_emits,
)
from loom.target.arch.x86.native_conversion import (
    scalar_float_integer_mnemonic,
    vector_float_integer_mnemonic,
)
from loom.target.arch.x86.vector_families import (
    AVX2_FLOAT_INTEGER_I32_FORMS,
    AVX512_FLOAT_INTEGER_I32_FORMS,
    AVX512_FLOAT_INTEGER_I64_FORMS,
    FLOAT_ELEMENTS,
    X86_VECTOR_INTEGER_NARROWING_FAMILIES,
    FloatIntegerVectorForm,
)
from loom.target.contracts import (
    ContractEmit,
    DescriptorEmitForm,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    ReadOnlyDataLiteral,
    Scalar,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import Descriptor

_I1 = Scalar("i1")
_I32 = Scalar("i32")
_I64 = Scalar("i64")
_INTEGER_TYPES = {
    1: _I1,
    8: Scalar("i8"),
    16: Scalar("i16"),
    32: _I32,
    64: _I64,
}
_REGISTER_SUFFIXES = {128: "xmm", 256: "ymm", 512: "zmm"}
_FLOAT_THRESHOLD_BITS = {
    32: {31: 0x4F000000, 63: 0x5F000000},
    64: {31: 0x41E0000000000000, 63: 0x43E0000000000000},
}


def _scalar_descriptor_key(
    source_bit_width: int,
    result_bit_width: int,
    *,
    unsigned: bool,
) -> str:
    feature = "avx512" if unsigned else "avx2"
    mnemonic = scalar_float_integer_mnemonic(source_bit_width, unsigned=unsigned)
    return f"x86.{feature}.{mnemonic}.gpr{result_bit_width}.xmm"


def _vector_descriptor_key(
    form: FloatIntegerVectorForm,
    *,
    unsigned: bool,
    evex: bool,
) -> str:
    feature = "avx512" if evex else "avx2"
    mnemonic = vector_float_integer_mnemonic(
        form.source.bit_width,
        form.result_bit_width,
        unsigned=unsigned,
    )
    result_suffix = _REGISTER_SUFFIXES[max(128, form.result_vector_bit_width)]
    source_suffix = _REGISTER_SUFFIXES[max(128, form.source_vector_bit_width)]
    return f"x86.{feature}.{mnemonic}.{result_suffix}.{source_suffix}"


def _scalar_unsigned64_emits(
    source_bit_width: int,
    convert: Descriptor,
    descriptor_lookup: DescriptorLookup,
) -> tuple[ContractEmit, ...]:
    float_type = Scalar(f"f{source_bit_width}")
    move_bits = descriptor_lookup(f"x86.scalar.movimm.gpr{source_bit_width}")
    move_float = descriptor_lookup(
        "x86.avx2.vmovd.xmm.gpr32"
        if source_bit_width == 32
        else "x86.avx2.vmovq.xmm.gpr64"
    )
    subtract = descriptor_lookup(
        f"x86.avx2.vsub{'ss' if source_bit_width == 32 else 'sd'}.xmm"
    )
    shift = descriptor_lookup("x86.scalar.sar.imm.gpr64")
    bitwise_and = descriptor_lookup("x86.scalar.and.gpr64")
    bitwise_or = descriptor_lookup("x86.scalar.or.gpr64")
    integer_type = _I32 if source_bit_width == 32 else _I64
    threshold_bits = _FLOAT_THRESHOLD_BITS[source_bit_width][63]
    return (
        EmitDescriptorOp(
            descriptor=move_bits,
            results={"dst": ValueRef.temporary("threshold_bits")},
            result_types={"dst": integer_type},
            immediates={"imm32" if source_bit_width == 32 else "imm64": threshold_bits},
            form=DescriptorEmitForm.CONST,
        ),
        emit_descriptor_op(
            descriptor=move_float,
            operands={"input": ValueRef.temporary("threshold_bits")},
            results={"dst": ValueRef.temporary("threshold")},
            result_types={"dst": float_type},
        ),
        emit_descriptor_op(
            descriptor=subtract,
            operands={
                "lhs": ValueRef.operand("input"),
                "rhs": ValueRef.temporary("threshold"),
            },
            results={"dst": ValueRef.temporary("shifted_input")},
            result_types={"dst": float_type},
        ),
        emit_descriptor_op(
            descriptor=convert,
            operands={"input": ValueRef.operand("input")},
            results={"dst": ValueRef.temporary("direct")},
            result_types={"dst": _I64},
        ),
        emit_descriptor_op(
            descriptor=convert,
            operands={"input": ValueRef.temporary("shifted_input")},
            results={"dst": ValueRef.temporary("shifted")},
            result_types={"dst": _I64},
        ),
        emit_descriptor_op(
            descriptor=shift,
            operands={"lhs": ValueRef.temporary("direct")},
            results={"dst": ValueRef.temporary("high_mask")},
            result_types={"dst": _I64},
            immediates={"shift": 63},
        ),
        emit_descriptor_op(
            descriptor=bitwise_and,
            operands={
                "lhs": ValueRef.temporary("shifted"),
                "rhs": ValueRef.temporary("high_mask"),
            },
            results={"dst": ValueRef.temporary("high_bits")},
            result_types={"dst": _I64},
        ),
        emit_descriptor_op(
            descriptor=bitwise_or,
            operands={
                "lhs": ValueRef.temporary("direct"),
                "rhs": ValueRef.temporary("high_bits"),
            },
            results={"dst": ValueRef.result("result")},
        ),
    )


def _scalar_rule(
    source_op: Op,
    source_bit_width: int,
    result_bit_width: int,
    *,
    native_unsigned: bool,
    descriptor_lookup: DescriptorLookup,
) -> DescriptorRule:
    unsigned = source_op is scalar_conversion.scalar_fptoui
    source_type = Scalar(f"f{source_bit_width}")
    result_type = _INTEGER_TYPES[result_bit_width]
    if native_unsigned:
        carrier_bit_width = result_bit_width
        convert = descriptor_lookup(
            _scalar_descriptor_key(source_bit_width, carrier_bit_width, unsigned=True)
        )
        emits = (
            emit_descriptor_op(
                descriptor=convert,
                operands={"input": ValueRef.operand("input")},
                results={"dst": ValueRef.result("result")},
            ),
        )
    else:
        carrier_bit_width = (
            64
            if result_bit_width == 64 or (unsigned and result_bit_width == 32)
            else 32
        )
        convert = descriptor_lookup(
            _scalar_descriptor_key(source_bit_width, carrier_bit_width, unsigned=False)
        )
        if unsigned and result_bit_width == 64:
            emits = _scalar_unsigned64_emits(
                source_bit_width, convert, descriptor_lookup
            )
        elif unsigned and result_bit_width == 32:
            emits = (
                emit_descriptor_op(
                    descriptor=convert,
                    operands={"input": ValueRef.operand("input")},
                    results={"dst": ValueRef.temporary("wide")},
                    result_types={"dst": _I64},
                ),
                emit_descriptor_op(
                    descriptor=descriptor_lookup("x86.scalar.mov.trunc.gpr32.gpr64"),
                    operands={"src": ValueRef.temporary("wide")},
                    results={"dst": ValueRef.result("result")},
                ),
            )
        elif result_bit_width == 1:
            emits = (
                emit_descriptor_op(
                    descriptor=convert,
                    operands={"input": ValueRef.operand("input")},
                    results={"dst": ValueRef.temporary("integer")},
                    result_types={"dst": _I32},
                ),
                emit_descriptor_op(
                    descriptor=descriptor_lookup("x86.scalar.and.imm.gpr32"),
                    operands={"lhs": ValueRef.temporary("integer")},
                    results={"dst": ValueRef.result("result")},
                    immediates={"imm32": 1},
                ),
            )
        else:
            emits = (
                emit_descriptor_op(
                    descriptor=convert,
                    operands={"input": ValueRef.operand("input")},
                    results={"dst": ValueRef.result("result")},
                ),
            )
    return DescriptorRule(
        source_op=source_op,
        descriptor=convert,
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
        ),
        emit=emits,
        priority=int(native_unsigned),
    )


def avx2_scalar_float_integer_rules(
    descriptor_lookup: DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    return tuple(
        _scalar_rule(
            source_op,
            source.bit_width,
            result_bit_width,
            native_unsigned=False,
            descriptor_lookup=descriptor_lookup,
        )
        for source_op in (
            scalar_conversion.scalar_fptosi,
            scalar_conversion.scalar_fptoui,
        )
        for source in FLOAT_ELEMENTS
        for result_bit_width in _INTEGER_TYPES
    )


def avx512_scalar_float_integer_rules(
    descriptor_lookup: DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    return tuple(
        _scalar_rule(
            scalar_conversion.scalar_fptoui,
            source.bit_width,
            result_bit_width,
            native_unsigned=True,
            descriptor_lookup=descriptor_lookup,
        )
        for source in FLOAT_ELEMENTS
        for result_bit_width in (32, 64)
    )


def _conversion_emit(
    descriptor: Descriptor,
    *,
    result: ValueRef,
    result_type: Vector | None,
) -> ContractEmit:
    return emit_descriptor_op(
        descriptor=descriptor,
        operands={"input": ValueRef.operand("input")},
        results={"dst": result},
        result_types=None if result_type is None else {"dst": result_type},
    )


def _vector_guards(
    form: FloatIntegerVectorForm, result_bit_width: int
) -> tuple[Guard, ...]:
    return (
        Guard.value_type("input", Vector(form.source.name, lanes=form.lane_count)),
        Guard.value_type(
            "result", Vector(f"i{result_bit_width}", lanes=form.lane_count)
        ),
        Guard.value_static_element_count_eq("input", "result"),
    )


def _avx2_unsigned_i32_rule(
    form: FloatIntegerVectorForm,
    descriptor_lookup: DescriptorLookup,
) -> DescriptorRule:
    source_type = Vector(form.source.name, lanes=form.lane_count)
    result_type = Vector("i32", lanes=form.lane_count)
    source_bit_width = max(128, form.source_vector_bit_width)
    result_bit_width = max(128, form.result_vector_bit_width)
    source_suffix = _REGISTER_SUFFIXES[source_bit_width]
    result_suffix = _REGISTER_SUFFIXES[result_bit_width]
    convert = descriptor_lookup(
        _vector_descriptor_key(form, unsigned=False, evex=False)
    )
    load = descriptor_lookup(f"x86.avx2.vmovdqu.rodata.{source_suffix}")
    subtract = descriptor_lookup(
        f"x86.avx2.vsub{'ps' if form.source.bit_width == 32 else 'pd'}.{source_suffix}"
    )
    shift = descriptor_lookup(f"x86.avx2.vpsrad.{result_suffix}")
    bitwise_and = descriptor_lookup(f"x86.avx2.vpand.{result_suffix}")
    bitwise_or = descriptor_lookup(f"x86.avx2.vpor.{result_suffix}")
    threshold_element = _FLOAT_THRESHOLD_BITS[form.source.bit_width][31]
    threshold = threshold_element.to_bytes(form.source.bit_width // 8, "little") * (
        source_bit_width // form.source.bit_width
    )
    emits = (
        emit_descriptor_op(
            descriptor=load,
            results={"dst": ValueRef.temporary("threshold")},
            result_types={"dst": source_type},
            immediates={"data": ReadOnlyDataLiteral(threshold)},
        ),
        emit_descriptor_op(
            descriptor=subtract,
            operands={
                "lhs": ValueRef.operand("input"),
                "rhs": ValueRef.temporary("threshold"),
            },
            results={"dst": ValueRef.temporary("shifted_input")},
            result_types={"dst": source_type},
        ),
        _conversion_emit(
            convert,
            result=ValueRef.temporary("direct"),
            result_type=result_type,
        ),
        emit_descriptor_op(
            descriptor=convert,
            operands={"input": ValueRef.temporary("shifted_input")},
            results={"dst": ValueRef.temporary("shifted")},
            result_types={"dst": result_type},
        ),
        emit_descriptor_op(
            descriptor=shift,
            operands={"source": ValueRef.temporary("direct")},
            results={"dst": ValueRef.temporary("high_mask")},
            result_types={"dst": result_type},
            immediates={"shift": 31},
        ),
        emit_descriptor_op(
            descriptor=bitwise_and,
            operands={
                "lhs": ValueRef.temporary("shifted"),
                "rhs": ValueRef.temporary("high_mask"),
            },
            results={"dst": ValueRef.temporary("high_bits")},
            result_types={"dst": result_type},
        ),
        emit_descriptor_op(
            descriptor=bitwise_or,
            operands={
                "lhs": ValueRef.temporary("direct"),
                "rhs": ValueRef.temporary("high_bits"),
            },
            results={"dst": ValueRef.result("result")},
        ),
    )
    return DescriptorRule(
        source_op=vector.vector_fptoui,
        descriptor=convert,
        guards=_vector_guards(form, 32),
        emit=emits,
    )


def _vector_direct_rule(
    source_op: Op,
    form: FloatIntegerVectorForm,
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
        guards=_vector_guards(form, form.result_bit_width),
        emit=(
            _conversion_emit(
                descriptor,
                result=ValueRef.result("result"),
                result_type=None,
            ),
        ),
        priority=priority,
    )


def _narrowing_family(result_bit_width: int):
    return next(
        family
        for family in X86_VECTOR_INTEGER_NARROWING_FAMILIES
        if family.source.bit_width == 32 and family.result.bit_width == result_bit_width
    )


def _vector_narrow_rule(
    source_op: Op,
    form: FloatIntegerVectorForm,
    result_bit_width: int,
    *,
    avx512_narrow: bool,
    priority: int,
    descriptor_lookup: DescriptorLookup,
) -> DescriptorRule:
    evex_conversion = form not in AVX2_FLOAT_INTEGER_I32_FORMS
    convert = descriptor_lookup(
        _vector_descriptor_key(form, unsigned=False, evex=evex_conversion)
    )
    converted = ValueRef.temporary("converted")
    converted_type = Vector("i32", lanes=form.lane_count)
    family = _narrowing_family(result_bit_width)
    if avx512_narrow:
        narrowing_emits, narrowing_descriptors = avx512_integer_narrowing_emits(
            family,
            form.lane_count,
            descriptor_lookup,
            source=converted,
            result=ValueRef.result("result"),
            result_type=None,
        )
    else:
        narrowing_emits, narrowing_descriptors = avx2_integer_narrowing_emits(
            family,
            form.lane_count,
            descriptor_lookup,
            source=converted,
            result=ValueRef.result("result"),
            result_type=None,
            temporary_prefix="float_integer",
        )
    return DescriptorRule(
        source_op=source_op,
        descriptor=convert,
        guards=(
            *_vector_guards(form, result_bit_width),
            *(
                Guard.descriptor_available(descriptor)
                for descriptor in narrowing_descriptors
            ),
        ),
        emit=(
            _conversion_emit(convert, result=converted, result_type=converted_type),
            *narrowing_emits,
        ),
        priority=priority,
    )


def avx2_vector_float_integer_rules(
    descriptor_lookup: DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for form in AVX2_FLOAT_INTEGER_I32_FORMS:
        rules.append(
            _vector_direct_rule(
                vector.vector_fptosi,
                form,
                unsigned=False,
                evex=False,
                priority=0,
                descriptor_lookup=descriptor_lookup,
            )
        )
        rules.append(_avx2_unsigned_i32_rule(form, descriptor_lookup))
        rules.extend(
            _vector_narrow_rule(
                source_op,
                form,
                result_bit_width,
                avx512_narrow=False,
                priority=0,
                descriptor_lookup=descriptor_lookup,
            )
            for source_op in (vector.vector_fptosi, vector.vector_fptoui)
            for result_bit_width in (8, 16)
        )
    return tuple(rules)


def avx512_vector_float_integer_rules(
    descriptor_lookup: DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    rules = [
        _vector_direct_rule(
            vector.vector_fptosi,
            form,
            unsigned=False,
            evex=True,
            priority=1,
            descriptor_lookup=descriptor_lookup,
        )
        for form in AVX512_FLOAT_INTEGER_I32_FORMS[-2:]
    ]
    for form in AVX512_FLOAT_INTEGER_I32_FORMS:
        rules.append(
            _vector_direct_rule(
                vector.vector_fptoui,
                form,
                unsigned=True,
                evex=True,
                priority=1,
                descriptor_lookup=descriptor_lookup,
            )
        )
        rules.extend(
            _vector_narrow_rule(
                source_op,
                form,
                result_bit_width,
                avx512_narrow=True,
                priority=1,
                descriptor_lookup=descriptor_lookup,
            )
            for source_op in (vector.vector_fptosi, vector.vector_fptoui)
            for result_bit_width in (8, 16)
        )
    f64x8 = AVX512_FLOAT_INTEGER_I32_FORMS[-1]
    rules.extend(
        _vector_narrow_rule(
            source_op,
            f64x8,
            result_bit_width,
            avx512_narrow=False,
            priority=0,
            descriptor_lookup=descriptor_lookup,
        )
        for source_op in (vector.vector_fptosi, vector.vector_fptoui)
        for result_bit_width in (8, 16)
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
        for form in AVX512_FLOAT_INTEGER_I64_FORMS
        for source_op, unsigned in (
            (vector.vector_fptosi, False),
            (vector.vector_fptoui, True),
        )
    )
    return tuple(rules)
