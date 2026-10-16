# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exact scalar and packed float-narrowing rules for x86 SIMD profiles."""

from collections.abc import Callable
from dataclasses import replace

from loom.dialect.scalar import conversion as scalar_conversion
from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.rule_builders import (
    VectorCarrier,
    emit_descriptor_op,
    vector_carriers,
)
from loom.target.arch.x86.contracts.vector_integer_conversion import (
    avx2_integer_narrowing_emits,
)
from loom.target.arch.x86.vector_families import (
    AVX2_CARRIER_VECTOR_BIT_WIDTHS,
    AVX512_CARRIER_VECTOR_BIT_WIDTHS,
    FLOAT_ELEMENTS,
    X86_VECTOR_INTEGER_NARROWING_FAMILIES,
)
from loom.target.contracts import (
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    ReadOnlyDataLiteral,
    Scalar,
    TypePattern,
    ValueRef,
)
from loom.target.emit.float_narrowing import (
    BF16_SOURCE_FORMAT,
    F8E4M3_FORMAT,
    F8E5M2_FORMAT,
    F16_FORMAT,
    F16_SOURCE_FORMAT,
    F32_FORMAT,
    BinaryFloatFormat,
    IntegerNarrowingDescriptors,
    IntegerNarrowingImmediateForms,
    NarrowFloatFormat,
    NarrowFloatSubnormalRounding,
    build_f32_bits_to_bf16_emits,
    build_integer_bits_to_narrow_float_emits,
    build_representable_f32_bits_to_bf16_emits,
)
from loom.target.low_descriptors import Descriptor

_DescriptorLookup = Callable[[str], Descriptor]

_F8E4M3 = Scalar("f8E4M3")
_F8E5M2 = Scalar("f8E5M2")
_F16 = Scalar("f16")
_BF16 = Scalar("bf16")
_F32 = Scalar("f32")


def _packed_f32_to_bf16_rule(
    descriptor_lookup: _DescriptorLookup,
    carrier: VectorCarrier,
    *,
    evex: bool,
) -> DescriptorRule:
    suffix = carrier.descriptor_suffix
    feature = "avx512" if evex else "avx2"
    prefix = f"x86.{feature}"
    # XMM/YMM arithmetic uses compact VEX forms even when the final narrowing
    # and NaN selection use AVX-512 registers and predicates.
    arithmetic_prefix = "x86.avx512" if suffix == "zmm" else "x86.avx2"
    bitwise_and = "vpandd" if suffix == "zmm" else "vpand"
    bitwise_or = "vpord" if suffix == "zmm" else "vpor"
    load_key = (
        "x86.avx512.vmovdqu64.rodata.zmm"
        if suffix == "zmm"
        else f"x86.avx2.vmovdqu.rodata.{suffix}"
    )
    emits = []
    for name, value in (
        ("one", 1),
        ("rounding_bias", 0x7FFF),
        ("nonsign_mask", 0x7FFFFFFF),
        ("infinity_bits", 0x7F800000),
        ("quiet_nan_bit", 0x0040),
    ):
        emits.append(
            emit_descriptor_op(
                descriptor=descriptor_lookup(load_key),
                results={"dst": ValueRef.temporary(name)},
                result_types={"dst": DescriptorResultType()},
                immediates={
                    "data": ReadOnlyDataLiteral(
                        value.to_bytes(4, "little") * (carrier.register_bit_width // 32)
                    )
                },
            )
        )

    def operation(
        name: str,
        key: str,
        operands: dict[str, ValueRef],
        immediates: dict[str, int] | None = None,
    ) -> ValueRef:
        result = ValueRef.temporary(name)
        emits.append(
            emit_descriptor_op(
                descriptor=descriptor_lookup(key),
                operands=operands,
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
                immediates=immediates,
            )
        )
        return result

    input_bits = ValueRef.operand("input")
    shift_key = f"{arithmetic_prefix}.vpsrld.{suffix}"
    upper = operation("upper", shift_key, {"source": input_bits}, {"shift": 16})
    retained_lsb = operation(
        "retained_lsb",
        f"{arithmetic_prefix}.{bitwise_and}.{suffix}",
        {"lhs": upper, "rhs": ValueRef.temporary("one")},
    )
    bias = operation(
        "bias",
        f"{arithmetic_prefix}.vpaddd.{suffix}",
        {"lhs": retained_lsb, "rhs": ValueRef.temporary("rounding_bias")},
    )
    rounded = operation(
        "rounded",
        f"{arithmetic_prefix}.vpaddd.{suffix}",
        {"lhs": input_bits, "rhs": bias},
    )
    finite = operation("finite", shift_key, {"source": rounded}, {"shift": 16})
    magnitude = operation(
        "magnitude",
        f"{arithmetic_prefix}.{bitwise_and}.{suffix}",
        {"lhs": input_bits, "rhs": ValueRef.temporary("nonsign_mask")},
    )
    # Clearing the sign makes a signed comparison valid. NaNs retain their
    # sign and high payload bits, with the destination quiet bit set.
    is_nan = operation(
        "is_nan",
        f"{prefix}.{'vpcmpd' if evex else 'vpcmpgtd'}.{suffix}",
        {"lhs": magnitude, "rhs": ValueRef.temporary("infinity_bits")},
        {"predicate": 6} if evex else None,
    )
    nan = operation(
        "nan",
        f"{arithmetic_prefix}.{bitwise_or}.{suffix}",
        {"lhs": upper, "rhs": ValueRef.temporary("quiet_nan_bit")},
    )
    selected = operation(
        "selected",
        f"{prefix}.{'vpblendmd' if evex else 'vpblendvb'}.{suffix}",
        {"mask": is_nan, "true_value": nan, "false_value": finite},
    )
    if evex:
        result_suffix = "ymm" if suffix == "zmm" else "xmm"
        narrowing = descriptor_lookup(f"{prefix}.vpmovdw.{result_suffix}.{suffix}")
        emits.append(
            emit_descriptor_op(
                descriptor=narrowing,
                operands={"source": selected},
                results={"dst": ValueRef.result("result")},
            )
        )
    else:
        family = next(
            family
            for family in X86_VECTOR_INTEGER_NARROWING_FAMILIES
            if family.source.bit_width == 32 and family.result.bit_width == 16
        )
        packing, _ = avx2_integer_narrowing_emits(
            family,
            carrier.register_bit_width // 32,
            descriptor_lookup,
            source=selected,
            result=ValueRef.result("result"),
            result_type=None,
            temporary_prefix="pack",
        )
        emits.extend(packing)
        narrowing = packing[-1].descriptor
    return DescriptorRule(
        source_op=vector.vector_fptrunc,
        descriptor=narrowing,
        guards=(
            Guard.value_type("input", carrier.value_type),
            Guard.value_type(
                "result", replace(carrier.value_type, element=None, elements=("bf16",))
            ),
            Guard.value_static_element_count_eq("input", "result"),
        ),
        emit=tuple(emits),
        priority=int(evex),
        report_key="exact_packed_binary32_to_bfloat16",
    )


def packed_bfloat_narrowing_rules(
    descriptor_lookup: _DescriptorLookup, *, evex: bool
) -> tuple[DescriptorRule, ...]:
    """Rounds F32 lanes directly into BF16 storage without scalar extraction."""

    widths = (
        AVX512_CARRIER_VECTOR_BIT_WIDTHS if evex else AVX2_CARRIER_VECTOR_BIT_WIDTHS
    )
    return tuple(
        _packed_f32_to_bf16_rule(descriptor_lookup, carrier, evex=evex)
        for carrier in vector_carriers(FLOAT_ELEMENTS[0], widths)
    )


def _integer_narrowing_descriptors(
    descriptor_lookup: _DescriptorLookup,
) -> IntegerNarrowingDescriptors:
    return IntegerNarrowingDescriptors(
        integer_bit_width=32,
        integer_constant=descriptor_lookup("x86.scalar.movimm.gpr32"),
        integer_add=descriptor_lookup("x86.scalar.add.gpr32"),
        integer_subtract=descriptor_lookup("x86.scalar.sub.gpr32"),
        integer_shift_left=descriptor_lookup("x86.scalar.shl.cl.gpr32"),
        integer_shift_right_logical=descriptor_lookup("x86.scalar.shr.cl.gpr32"),
        integer_bitwise_and=descriptor_lookup("x86.scalar.and.gpr32"),
        integer_bitwise_or=descriptor_lookup("x86.scalar.or.gpr32"),
        integer_less_than_nonnegative=descriptor_lookup("x86.scalar.cmp.slt.gpr32"),
        integer_greater_than_equal_nonnegative=descriptor_lookup(
            "x86.scalar.cmp.sge.gpr32"
        ),
        integer_greater_than_nonnegative=descriptor_lookup("x86.scalar.cmp.sgt.gpr32"),
        integer_select=descriptor_lookup("x86.scalar.select.gpr32"),
        immediate_forms=IntegerNarrowingImmediateForms(
            add=descriptor_lookup("x86.scalar.add.imm.gpr32"),
            subtract=descriptor_lookup("x86.scalar.sub.imm.gpr32"),
            shift_left=descriptor_lookup("x86.scalar.shl.imm.gpr32"),
            shift_right_logical=descriptor_lookup("x86.scalar.shr.imm.gpr32"),
            bitwise_and=descriptor_lookup("x86.scalar.and.imm.gpr32"),
            bitwise_or=descriptor_lookup("x86.scalar.or.imm.gpr32"),
            less_than_nonnegative=descriptor_lookup("x86.scalar.cmp.slt.imm.gpr32"),
            greater_than_equal_nonnegative=descriptor_lookup(
                "x86.scalar.cmp.sge.imm.gpr32"
            ),
            greater_than_nonnegative=descriptor_lookup("x86.scalar.cmp.sgt.imm.gpr32"),
        ),
    )


def _f32_bits_emit(
    descriptor_lookup: _DescriptorLookup,
) -> EmitDescriptorOp:
    return EmitDescriptorOp(
        descriptor=descriptor_lookup("x86.avx2.vmovd.gpr32.xmm"),
        operands={"input": ValueRef.operand("input")},
        results={"dst": ValueRef.temporary("input_bits")},
        result_types={"dst": DescriptorResultType()},
    )


def _rule(
    source_type: TypePattern,
    result_type: TypePattern,
    emits: tuple[EmitDescriptorOp, ...],
    report_key: str,
    *,
    fact_guards: tuple[Guard, ...] = (),
) -> DescriptorRule:
    # Variable x86 shifts consume their count through ECX. Bind that physical
    # register only at the shift use so the computed count remains an ordinary
    # GPR value for its other uses.
    emits = tuple(
        replace(emit, copy_operands=("rhs",))
        if emit.descriptor.key in ("x86.scalar.shl.cl.gpr32", "x86.scalar.shr.cl.gpr32")
        else emit
        for emit in emits
    )
    return DescriptorRule(
        source_op=scalar_conversion.scalar_fptrunc,
        descriptor=emits[-1].descriptor,
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
            *fact_guards,
        ),
        emit=emits,
        report_key=report_key,
    )


def _f32_to_bf16_rule(
    descriptor_lookup: _DescriptorLookup,
    descriptors: IntegerNarrowingDescriptors,
) -> DescriptorRule:
    emits = (
        _f32_bits_emit(descriptor_lookup),
        *build_f32_bits_to_bf16_emits(
            descriptors,
            ValueRef.temporary("input_bits"),
            ValueRef.result("result"),
        ),
    )
    return _rule(_F32, _BF16, emits, "exact_binary32_to_bfloat16")


def _representable_f32_to_bf16_rule(
    descriptor_lookup: _DescriptorLookup,
    descriptors: IntegerNarrowingDescriptors,
) -> DescriptorRule:
    emits = (
        _f32_bits_emit(descriptor_lookup),
        *build_representable_f32_bits_to_bf16_emits(
            descriptors,
            ValueRef.temporary("input_bits"),
            ValueRef.result("result"),
        ),
    )
    return _rule(
        _F32,
        _BF16,
        emits,
        "exactly_representable_binary32_to_bfloat16",
        fact_guards=(Guard.value_float_exactly_representable_as("input", "result"),),
    )


def _f32_to_narrow_rule(
    descriptor_lookup: _DescriptorLookup,
    descriptors: IntegerNarrowingDescriptors,
    result_type: TypePattern,
    result_format: NarrowFloatFormat,
    result_name: str,
) -> DescriptorRule:
    emits = (
        _f32_bits_emit(descriptor_lookup),
        *build_integer_bits_to_narrow_float_emits(
            descriptors,
            F32_FORMAT,
            result_format,
            ValueRef.temporary("input_bits"),
            ValueRef.result("result"),
            subnormal_rounding=NarrowFloatSubnormalRounding.INTEGER,
        ),
    )
    return _rule(_F32, result_type, emits, f"exact_binary32_to_{result_name}")


def _integer_carrier_to_float8_rule(
    descriptors: IntegerNarrowingDescriptors,
    source_type: TypePattern,
    source_format: BinaryFloatFormat,
    source_name: str,
    result_type: TypePattern,
    result_format: NarrowFloatFormat,
    result_name: str,
) -> DescriptorRule:
    if (
        source_format.exponent_bias
        >= result_format.exponent_bias + result_format.mantissa_bits + 1
    ):
        subnormal_rounding = NarrowFloatSubnormalRounding.INTEGER
    else:
        subnormal_rounding = NarrowFloatSubnormalRounding.INTEGER_SOURCE_SUBNORMALS
    emits = build_integer_bits_to_narrow_float_emits(
        descriptors,
        source_format,
        result_format,
        ValueRef.operand("input"),
        ValueRef.result("result"),
        subnormal_rounding=subnormal_rounding,
    )
    return _rule(
        source_type,
        result_type,
        emits,
        f"exact_{source_name}_to_{result_name}",
    )


def x86_float_narrowing_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    """Returns the exact narrow-float matrix supported by x86 carriers."""

    descriptors = _integer_narrowing_descriptors(descriptor_lookup)
    return (
        _representable_f32_to_bf16_rule(descriptor_lookup, descriptors),
        _f32_to_bf16_rule(descriptor_lookup, descriptors),
        _f32_to_narrow_rule(
            descriptor_lookup,
            descriptors,
            _F16,
            F16_FORMAT,
            "binary16",
        ),
        _f32_to_narrow_rule(
            descriptor_lookup,
            descriptors,
            _F8E4M3,
            F8E4M3_FORMAT,
            "float8_e4m3",
        ),
        _f32_to_narrow_rule(
            descriptor_lookup,
            descriptors,
            _F8E5M2,
            F8E5M2_FORMAT,
            "float8_e5m2",
        ),
        *(
            _integer_carrier_to_float8_rule(
                descriptors,
                source_type,
                source_format,
                source_name,
                result_type,
                result_format,
                result_name,
            )
            for source_type, source_format, source_name in (
                (_F16, F16_SOURCE_FORMAT, "binary16"),
                (_BF16, BF16_SOURCE_FORMAT, "bfloat16"),
            )
            for result_type, result_format, result_name in (
                (_F8E4M3, F8E4M3_FORMAT, "float8_e4m3"),
                (_F8E5M2, F8E5M2_FORMAT, "float8_e5m2"),
            )
        ),
    )
