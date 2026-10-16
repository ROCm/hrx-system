# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 packed-integer vector conversion contract rules."""

from __future__ import annotations

from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.rule_builders import (
    DescriptorLookup as _DescriptorLookup,
)
from loom.target.arch.x86.contracts.rule_builders import emit_descriptor_op as _op_emit
from loom.target.arch.x86.vector_families import (
    AVX2_CARRIER_VECTOR_BIT_WIDTHS,
    AVX512_CARRIER_VECTOR_BIT_WIDTHS,
    X86_VECTOR_INTEGER_NARROWING_FAMILIES,
    X86_VECTOR_INTEGER_WIDENING_FAMILIES,
    VectorIntegerNarrowingFamily,
)
from loom.target.contracts import (
    ContractEmit,
    DescriptorResultType,
    DescriptorRule,
    EmitRegisterCopy,
    Guard,
    ReadOnlyDataLiteral,
    ResultTypeBinding,
    TypePattern,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import Descriptor

_REGISTER_SUFFIXES = {128: "xmm", 256: "ymm", 512: "zmm"}
_AVX2_NARROW_JOIN_MNEMONICS = {
    16: "vpunpcklwd",
    32: "vpunpckldq",
    64: "vpunpcklqdq",
}
_INTEGER_SHIFT_RIGHT_MNEMONICS = {
    16: "vpsrlw",
    32: "vpsrld",
    64: "vpsrlq",
}


def _physical_register_suffix(vector_bit_width: int) -> str:
    return _REGISTER_SUFFIXES[max(128, vector_bit_width)]


def vector_integer_widening_rules(
    descriptor_lookup: _DescriptorLookup,
    *,
    descriptor_key_prefix: str,
    source_vector_bit_widths: tuple[int, ...],
    result_vector_bit_widths: tuple[int, ...],
) -> tuple[DescriptorRule, ...]:
    """Returns native rules between supported one-register vector carriers."""

    rules: list[DescriptorRule] = []
    for family in X86_VECTOR_INTEGER_WIDENING_FAMILIES:
        for source_op, mnemonic in (
            (vector.vector_extsi, family.signed_mnemonic),
            (vector.vector_extui, family.unsigned_mnemonic),
        ):
            for result_vector_bit_width in result_vector_bit_widths:
                if result_vector_bit_width < family.result.bit_width:
                    continue
                lane_count = result_vector_bit_width // family.result.bit_width
                source_vector_bit_width = lane_count * family.source.bit_width
                if source_vector_bit_width not in source_vector_bit_widths:
                    continue
                source_type = Vector(family.source.name, lanes=lane_count)
                result_type = Vector(family.result.name, lanes=lane_count)
                source_suffix = _physical_register_suffix(source_vector_bit_width)
                descriptor = descriptor_lookup(
                    f"{descriptor_key_prefix}.{mnemonic}."
                    f"{_physical_register_suffix(result_vector_bit_width)}."
                    f"{source_suffix}"
                )
                rules.append(
                    DescriptorRule(
                        source_op=source_op,
                        descriptor=descriptor,
                        guards=(
                            Guard.value_type("input", source_type),
                            Guard.value_type("result", result_type),
                            Guard.value_static_element_count_eq("input", "result"),
                        ),
                        emit=(
                            _op_emit(
                                descriptor=descriptor,
                                operands={"source": ValueRef.operand("input")},
                                results={"dst": ValueRef.result("result")},
                            ),
                        ),
                        report_key="x86.vector_integer_widen.native",
                    )
                )
    return tuple(rules)


def _avx2_narrowing_mask(
    family: VectorIntegerNarrowingFamily,
    source_vector_bit_width: int,
) -> bytes:
    """Builds per-128-bit-lane PSHUFB controls for one logical source."""

    physical_vector_bit_width = max(128, source_vector_bit_width)
    source_bytes = family.source.bit_width // 8
    result_bytes = family.result.bit_width // 8
    mask = bytearray()
    for lane_bit_offset in range(0, physical_vector_bit_width, 128):
        logical_lane_bit_width = min(
            128, max(0, source_vector_bit_width - lane_bit_offset)
        )
        lane_count = logical_lane_bit_width // family.source.bit_width
        lane_mask = [
            lane * source_bytes + byte
            for lane in range(lane_count)
            for byte in range(result_bytes)
        ]
        mask.extend(lane_mask)
        mask.extend([0x80] * (16 - len(lane_mask)))
    return bytes(mask)


def _temporary(prefix: str, name: str) -> ValueRef:
    return ValueRef.temporary(f"{prefix}_{name}" if prefix else name)


def avx2_integer_narrowing_emits(
    family: VectorIntegerNarrowingFamily,
    lane_count: int,
    descriptor_lookup: _DescriptorLookup,
    *,
    source: ValueRef,
    result: ValueRef,
    result_type: ResultTypeBinding | None,
    temporary_prefix: str,
) -> tuple[tuple[ContractEmit, ...], tuple[Descriptor, ...]]:
    """Packs the low bits of each integer lane into one narrower carrier."""

    source_vector_bit_width = lane_count * family.source.bit_width
    physical_source_bit_width = max(128, source_vector_bit_width)
    source_suffix = _physical_register_suffix(source_vector_bit_width)
    load = descriptor_lookup(f"x86.avx2.vmovdqu.rodata.{source_suffix}")
    shuffle = descriptor_lookup(f"x86.avx2.vpshufb.{source_suffix}")
    control = _temporary(temporary_prefix, "control")
    emits: list[ContractEmit] = [
        _op_emit(
            descriptor=load,
            results={"dst": control},
            result_types={"dst": DescriptorResultType()},
            immediates={
                "data": ReadOnlyDataLiteral(
                    _avx2_narrowing_mask(family, source_vector_bit_width)
                )
            },
        )
    ]
    descriptors = [shuffle, load]
    shuffled_result = result
    shuffled_result_type = result_type
    if physical_source_bit_width == 256:
        shuffled_result = _temporary(temporary_prefix, "packed")
        shuffled_result_type = DescriptorResultType()
    emits.append(
        _op_emit(
            descriptor=shuffle,
            operands={"lhs": source, "rhs": control},
            results={"dst": shuffled_result},
            result_types=(
                None if shuffled_result_type is None else {"dst": shuffled_result_type}
            ),
        )
    )
    if physical_source_bit_width == 128:
        return tuple(emits), tuple(descriptors)

    extract = descriptor_lookup("x86.avx2.vextractf128.xmm.ymm")
    halves = tuple(_temporary(temporary_prefix, f"half{lane}") for lane in range(2))
    for lane, half in enumerate(halves):
        emits.append(
            _op_emit(
                descriptor=extract,
                operands={"source": shuffled_result},
                results={"dst": half},
                result_types={"dst": DescriptorResultType()},
                immediates={"lane": lane},
            )
        )
    join_mnemonic = _AVX2_NARROW_JOIN_MNEMONICS[family.avx2_half_result_bit_width]
    join = descriptor_lookup(f"x86.avx2.{join_mnemonic}.xmm")
    emits.append(
        _op_emit(
            descriptor=join,
            operands={"lhs": halves[0], "rhs": halves[1]},
            results={"dst": result},
            result_types=None if result_type is None else {"dst": result_type},
        )
    )
    descriptors.extend((extract, join))
    return tuple(emits), tuple(descriptors)


def native_signed_integer_widening_emits(
    descriptor_lookup: _DescriptorLookup,
    *,
    descriptor_key_prefix: str,
    source_element_bit_width: int,
    result_element_bit_width: int,
    lane_count: int,
    source: ValueRef,
    result: ValueRef,
    result_type: ResultTypeBinding | None,
) -> tuple[tuple[ContractEmit, ...], tuple[Descriptor, ...]]:
    """Copies or natively widens one register of signed integer lanes."""

    if source_element_bit_width == result_element_bit_width:
        return (
            (
                EmitRegisterCopy(
                    source=source,
                    result=result,
                    result_type=result_type,
                ),
            ),
            (),
        )
    if source_element_bit_width > result_element_bit_width:
        raise ValueError("native signed integer conversion cannot narrow lanes")

    family = next(
        family
        for family in X86_VECTOR_INTEGER_WIDENING_FAMILIES
        if family.source.bit_width == source_element_bit_width
        and family.result.bit_width == result_element_bit_width
    )
    result_vector_bit_width = lane_count * result_element_bit_width
    source_vector_bit_width = lane_count * source_element_bit_width
    descriptor = descriptor_lookup(
        f"{descriptor_key_prefix}.{family.signed_mnemonic}."
        f"{_physical_register_suffix(result_vector_bit_width)}."
        f"{_physical_register_suffix(source_vector_bit_width)}"
    )
    return (
        (
            _op_emit(
                descriptor=descriptor,
                operands={"source": source},
                results={"dst": result},
                result_types=None if result_type is None else {"dst": result_type},
            ),
        ),
        (descriptor,),
    )


def avx2_signed_integer_conversion_emits(
    descriptor_lookup: _DescriptorLookup,
    *,
    source_element_bit_width: int,
    result_element_bit_width: int,
    lane_count: int,
    source: ValueRef,
    result: ValueRef,
    result_type: ResultTypeBinding | None,
    temporary_prefix: str,
) -> tuple[tuple[ContractEmit, ...], tuple[Descriptor, ...]]:
    """Converts one AVX2 register of signed integer lanes without scalarizing."""

    if source_element_bit_width <= result_element_bit_width:
        return native_signed_integer_widening_emits(
            descriptor_lookup,
            descriptor_key_prefix="x86.avx2",
            source_element_bit_width=source_element_bit_width,
            result_element_bit_width=result_element_bit_width,
            lane_count=lane_count,
            source=source,
            result=result,
            result_type=result_type,
        )

    family = next(
        family
        for family in X86_VECTOR_INTEGER_NARROWING_FAMILIES
        if family.source.bit_width == source_element_bit_width
        and family.result.bit_width == result_element_bit_width
    )
    return avx2_integer_narrowing_emits(
        family,
        lane_count,
        descriptor_lookup,
        source=source,
        result=result,
        result_type=result_type,
        temporary_prefix=temporary_prefix,
    )


def unsigned_integer_mask_normalization_emits(
    descriptor_lookup: _DescriptorLookup,
    *,
    descriptor_key_prefix: str,
    element_bit_width: int,
    vector_bit_width: int,
    source: ValueRef,
    result: ValueRef,
    result_type: ResultTypeBinding | None,
    temporary_type: TypePattern,
    temporary_prefix: str,
) -> tuple[tuple[ContractEmit, ...], tuple[Descriptor, ...]]:
    """Normalizes packed zero/all-one lanes to zero/one lanes."""

    register_suffix = _physical_register_suffix(vector_bit_width)
    if element_bit_width != 8:
        shift = descriptor_lookup(
            f"{descriptor_key_prefix}."
            f"{_INTEGER_SHIFT_RIGHT_MNEMONICS[element_bit_width]}."
            f"{register_suffix}"
        )
        return (
            (
                _op_emit(
                    descriptor=shift,
                    operands={"source": source},
                    results={"dst": result},
                    result_types=(
                        None if result_type is None else {"dst": result_type}
                    ),
                    immediates={"shift": element_bit_width - 1},
                ),
            ),
            (shift,),
        )

    zero = descriptor_lookup(f"{descriptor_key_prefix}.vxorps.zero.{register_suffix}")
    subtract = descriptor_lookup(f"{descriptor_key_prefix}.vpsubb.{register_suffix}")
    zero_value = ValueRef.temporary(f"{temporary_prefix}_zero")
    return (
        (
            _op_emit(
                descriptor=zero,
                results={"dst": zero_value},
                result_types={"dst": temporary_type},
            ),
            _op_emit(
                descriptor=subtract,
                operands={"lhs": zero_value, "rhs": source},
                results={"dst": result},
                result_types=None if result_type is None else {"dst": result_type},
            ),
        ),
        (zero, subtract),
    )


def _avx2_vector_integer_narrowing_rule(
    family: VectorIntegerNarrowingFamily,
    source_vector_bit_width: int,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    lane_count = source_vector_bit_width // family.source.bit_width
    source_type = Vector(family.source.name, lanes=lane_count)
    result_type = Vector(family.result.name, lanes=lane_count)
    emits, descriptors = avx2_integer_narrowing_emits(
        family,
        lane_count,
        descriptor_lookup,
        source=ValueRef.operand("input"),
        result=ValueRef.result("result"),
        result_type=None,
        temporary_prefix="",
    )
    primary_descriptor = descriptors[0]

    return DescriptorRule(
        source_op=vector.vector_trunci,
        descriptor=primary_descriptor,
        guards=(
            Guard.value_type("input", source_type),
            Guard.value_type("result", result_type),
            Guard.value_static_element_count_eq("input", "result"),
            *(Guard.descriptor_available(descriptor) for descriptor in descriptors[1:]),
        ),
        emit=emits,
        report_key="x86.vector_integer_narrow.avx2_shuffle",
    )


def avx2_vector_integer_narrowing_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    """Returns vectorized AVX2 rules for every integer narrowing pair."""

    return tuple(
        _avx2_vector_integer_narrowing_rule(
            family, source_vector_bit_width, descriptor_lookup
        )
        for family in X86_VECTOR_INTEGER_NARROWING_FAMILIES
        for source_vector_bit_width in AVX2_CARRIER_VECTOR_BIT_WIDTHS
        if source_vector_bit_width >= family.source.bit_width
        if (
            source_vector_bit_width * family.result.bit_width // family.source.bit_width
            in AVX2_CARRIER_VECTOR_BIT_WIDTHS
        )
    )


def avx512_vector_integer_narrowing_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    """Returns native AVX-512 rules for every integer narrowing pair."""

    rules: list[DescriptorRule] = []
    for family in X86_VECTOR_INTEGER_NARROWING_FAMILIES:
        for source_vector_bit_width in AVX512_CARRIER_VECTOR_BIT_WIDTHS:
            if source_vector_bit_width < family.source.bit_width:
                continue
            lane_count = source_vector_bit_width // family.source.bit_width
            result_vector_bit_width = lane_count * family.result.bit_width
            if result_vector_bit_width not in AVX512_CARRIER_VECTOR_BIT_WIDTHS:
                continue
            physical_source_bit_width = max(128, source_vector_bit_width)
            physical_result_bit_width = max(
                128,
                physical_source_bit_width
                * family.result.bit_width
                // family.source.bit_width,
            )
            descriptor = descriptor_lookup(
                f"x86.avx512.{family.mnemonic}."
                f"{_physical_register_suffix(physical_result_bit_width)}."
                f"{_physical_register_suffix(physical_source_bit_width)}"
            )
            rules.append(
                DescriptorRule(
                    source_op=vector.vector_trunci,
                    descriptor=descriptor,
                    guards=(
                        Guard.value_type(
                            "input", Vector(family.source.name, lanes=lane_count)
                        ),
                        Guard.value_type(
                            "result", Vector(family.result.name, lanes=lane_count)
                        ),
                        Guard.value_static_element_count_eq("input", "result"),
                    ),
                    emit=(
                        _op_emit(
                            descriptor=descriptor,
                            operands={"source": ValueRef.operand("input")},
                            results={"dst": ValueRef.result("result")},
                        ),
                    ),
                    priority=1,
                    report_key="x86.vector_integer_narrow.native",
                )
            )
    return tuple(rules)
