# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""x86 vector construction and iota contract rules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from enum import Enum

from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.constants import (
    floating_vector_splat_zero_rule,
    integer_vector_splat_rule,
    integer_vector_splat_zero_rule,
)
from loom.target.arch.x86.contracts.rule_builders import (
    DescriptorLookup as _DescriptorLookup,
)
from loom.target.arch.x86.contracts.rule_builders import (
    emit_descriptor_op as _op_emit,
)
from loom.target.arch.x86.contracts.rule_builders import (
    full_vector_type as _full_vector_type,
)
from loom.target.arch.x86.contracts.rule_builders import vector_carriers
from loom.target.arch.x86.vector_families import (
    AVX2_CARRIER_VECTOR_BIT_WIDTHS,
    AVX2_VECTOR_BIT_WIDTHS,
    AVX512_DIRECT_BROADCAST_VECTOR_BIT_WIDTHS,
    AVX512_FP16_VECTOR_BIT_WIDTHS,
    AVX512_VECTOR_BIT_WIDTHS,
    FLOAT_ELEMENTS,
    FP16_ELEMENT,
    INTEGER_ELEMENTS,
    STORAGE_ELEMENTS,
    X86_VECTOR_STORAGE_FAMILIES,
    VectorElement,
)
from loom.target.contracts import (
    ContractCase,
    DescriptorEmitForm,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    OperandSegmentProject,
    Scalar,
    TypePattern,
    ValueAliasRule,
    ValueRef,
    Vector,
)
from loom.target.low_descriptors import Descriptor

_I1 = Scalar("i1")
_I32 = Scalar("i32")
_I64 = Scalar("i64")
_V2I64 = Vector("i64", lanes=2)
_REGISTER_SUFFIXES = {
    8: "xmm",
    16: "xmm",
    32: "xmm",
    64: "xmm",
    128: "xmm",
    256: "ymm",
    512: "zmm",
}
_REGISTER_CLASSES = {
    64: "x86.xmm",
    128: "x86.xmm",
    256: "x86.ymm",
    512: "x86.zmm",
}
_INTEGER_BROADCAST_MNEMONICS = {
    8: "vpbroadcastb",
    16: "vpbroadcastw",
    32: "vpbroadcastd",
    64: "vpbroadcastq",
}
_FLOAT_BROADCAST_MNEMONICS = {
    32: {128: "vbroadcastss", 256: "vbroadcastss", 512: "vbroadcastss"},
    64: {128: "vmovddup", 256: "vbroadcastsd", 512: "vbroadcastsd"},
}


class _IntegerBroadcastSource(Enum):
    GPR = "gpr"
    XMM_LOW_LANE = "xmm-low-lane"


def _integer_lane_move_descriptor(
    element_bit_width: int,
    source: _IntegerBroadcastSource,
    descriptor_lookup: _DescriptorLookup,
) -> Descriptor | None:
    if source == _IntegerBroadcastSource.GPR:
        return None
    return descriptor_lookup(
        "x86.avx2.vmovq.xmm.gpr64"
        if element_bit_width == 64
        else "x86.avx2.vmovd.xmm.gpr32"
    )


@dataclass(frozen=True, slots=True)
class _IotaProfile:
    descriptor_prefix: str
    broadcast_source: _IntegerBroadcastSource


_AVX2_IOTA_PROFILE = _IotaProfile("x86.avx2", _IntegerBroadcastSource.XMM_LOW_LANE)
_AVX512_IOTA_PROFILE = _IotaProfile("x86.avx512", _IntegerBroadcastSource.GPR)


def _integer_broadcast_descriptor_key(
    descriptor_prefix: str,
    element_bit_width: int,
    vector_bit_width: int,
    *,
    broadcast_source: _IntegerBroadcastSource,
) -> str:
    key = (
        f"{descriptor_prefix}.{_INTEGER_BROADCAST_MNEMONICS[element_bit_width]}."
        f"{_REGISTER_SUFFIXES[vector_bit_width]}"
    )
    if vector_bit_width == 512 and broadcast_source != _IntegerBroadcastSource.GPR:
        return f"{key}.xmm"
    return key


def _vector_zero_rules(
    descriptor_lookup: _DescriptorLookup,
    *,
    descriptor_key_prefix: str,
    vector_bit_widths: tuple[int, ...],
) -> tuple[DescriptorRule, ...]:
    maximum_lanes = max(vector_bit_widths) // 8
    integer_element_names = tuple(element.name for element in INTEGER_ELEMENTS)
    integer_scalar_types = Scalar(integer_element_names)
    integer_vector_types = Vector(
        integer_element_names,
        minimum_lanes=2,
        maximum_lanes=maximum_lanes,
    )
    float_vector_types = Vector(
        tuple(element.name for element in (*STORAGE_ELEMENTS, *FLOAT_ELEMENTS)),
        minimum_lanes=2,
        maximum_lanes=maximum_lanes,
    )
    rules: list[DescriptorRule] = []
    for vector_bit_width in vector_bit_widths:
        descriptor = descriptor_lookup(
            f"{descriptor_key_prefix}.vxorps.zero."
            f"{_REGISTER_SUFFIXES[vector_bit_width]}"
        )
        register_class = _REGISTER_CLASSES[vector_bit_width]
        rules.extend(
            (
                integer_vector_splat_zero_rule(
                    integer_scalar_types,
                    integer_vector_types,
                    descriptor,
                    result_register_class=register_class,
                ),
                floating_vector_splat_zero_rule(
                    float_vector_types,
                    descriptor,
                    result_register_class=register_class,
                ),
            )
        )
    return tuple(rules)


def _vector_splat_rules(
    descriptor_lookup: _DescriptorLookup,
    *,
    descriptor_key_prefix: str,
    integer_vector_bit_widths: tuple[int, ...],
    float_vector_bit_widths: tuple[int, ...],
    broadcast_source: _IntegerBroadcastSource,
    priority: int = 0,
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for element in (*INTEGER_ELEMENTS, *STORAGE_ELEMENTS):
        scalar_type = Scalar(element.name)
        lane_type = _full_vector_type(element, 128)
        rules.extend(
            (
                integer_vector_splat_rule(
                    scalar_type,
                    carrier.value_type,
                    _integer_lane_move_descriptor(
                        element.bit_width,
                        broadcast_source,
                        descriptor_lookup,
                    ),
                    descriptor_lookup(
                        f"{descriptor_key_prefix}."
                        f"{_INTEGER_BROADCAST_MNEMONICS[element.bit_width]}."
                        f"{carrier.descriptor_suffix}"
                    ),
                    broadcast_operand="value",
                    lane_type=lane_type,
                    priority=priority,
                    result_register_class=carrier.register_class,
                )
            )
            for carrier in vector_carriers(element, integer_vector_bit_widths)
        )
    for element in FLOAT_ELEMENTS:
        for carrier in vector_carriers(element, float_vector_bit_widths):
            mnemonic = _FLOAT_BROADCAST_MNEMONICS[element.bit_width][
                carrier.register_bit_width
            ]
            rules.append(
                _splat_rule(
                    Scalar(element.name),
                    carrier.value_type,
                    f"{descriptor_key_prefix}.{mnemonic}.{carrier.descriptor_suffix}",
                    descriptor_lookup,
                    result_register_class=carrier.register_class,
                )
            )
    return tuple(rules)


def _exact_lane_vector_rules(
    descriptor_lookup: _DescriptorLookup,
    *,
    load_descriptor_prefix: str,
    vector_bit_widths: tuple[int, ...],
) -> tuple[DescriptorRule, ...]:
    rules: list[DescriptorRule] = []
    for family in X86_VECTOR_STORAGE_FAMILIES:
        for vector_bit_width in vector_bit_widths:
            lane_count = vector_bit_width // family.element_bit_width
            descriptor = descriptor_lookup(
                f"{load_descriptor_prefix}.{_REGISTER_SUFFIXES[vector_bit_width]}"
            )
            rules.append(
                DescriptorRule(
                    source_op=vector.vector_from_elements,
                    descriptor=descriptor,
                    guards=(
                        Guard.operand_segment_count("elements", lane_count),
                        Guard.operand_segment_exact_bits(
                            "elements", bit_width=family.element_bit_width
                        ),
                        Guard.value_type(
                            "result", Vector(family.element_names, lanes=lane_count)
                        ),
                    ),
                    emit=(
                        EmitDescriptorOp(
                            descriptor=descriptor,
                            results={"dst": ValueRef.result("result")},
                            immediates={
                                "data": OperandSegmentProject.exact_bits_read_only(
                                    "elements", bit_width=family.element_bit_width
                                )
                            },
                            form=DescriptorEmitForm.OP,
                        ),
                    ),
                )
            )
    return tuple(rules)


def _vector_bitcast_alias_rules(
    vector_bit_widths: tuple[int, ...],
) -> tuple[ValueAliasRule, ...]:
    """Aliases bit-compatible vector payloads carried by one register."""
    rules: list[ValueAliasRule] = []
    for float_elements, integer_element, element_bit_width in (
        (("f8E4M3", "f8E5M2"), "i8", 8),
        (("f16", "bf16"), "i16", 16),
        (("f32",), "i32", 32),
        (("f64",), "i64", 64),
    ):
        for vector_bit_width in vector_bit_widths:
            if vector_bit_width < element_bit_width:
                continue
            lane_count = vector_bit_width // element_bit_width
            integer_type = Vector(integer_element, lanes=lane_count)
            float_type = Vector(float_elements, lanes=lane_count)
            for source_type, result_type in (
                (float_type, integer_type),
                (integer_type, float_type),
            ):
                rules.append(
                    ValueAliasRule(
                        source_op=vector.vector_bitcast,
                        source=ValueRef.operand("input"),
                        result=ValueRef.result("result"),
                        guards=(
                            Guard.value_type("input", source_type),
                            Guard.value_type("result", result_type),
                        ),
                    )
                )
    for vector_bit_width in vector_bit_widths:
        for source_element in INTEGER_ELEMENTS:
            if vector_bit_width < source_element.bit_width:
                continue
            source_type = _full_vector_type(source_element, vector_bit_width)
            for result_element in INTEGER_ELEMENTS:
                if (
                    source_element == result_element
                    or vector_bit_width < result_element.bit_width
                ):
                    continue
                rules.append(
                    ValueAliasRule(
                        source_op=vector.vector_bitcast,
                        source=ValueRef.operand("input"),
                        result=ValueRef.result("result"),
                        guards=(
                            Guard.value_type("input", source_type),
                            Guard.value_type(
                                "result",
                                _full_vector_type(result_element, vector_bit_width),
                            ),
                        ),
                    )
                )
    return tuple(rules)


def _splat_rule(
    scalar_type: TypePattern,
    result_type: TypePattern,
    descriptor_key: str,
    descriptor_lookup: _DescriptorLookup,
    *,
    priority: int = 0,
    result_register_class: str | None = None,
) -> DescriptorRule:
    descriptor = descriptor_lookup(descriptor_key)
    return DescriptorRule(
        source_op=vector.vector_splat,
        descriptor=descriptor,
        guards=(
            Guard.value_type("scalar", scalar_type),
            Guard.value_type("result", result_type),
            *(
                (Guard.low_value_register_class("result", result_register_class),)
                if result_register_class is not None
                else ()
            ),
        ),
        emit=(
            _op_emit(
                descriptor=descriptor,
                operands={"value": ValueRef.operand("scalar")},
                results={"dst": ValueRef.result("result")},
            ),
        ),
        priority=priority,
    )


def _assemble_chunks(
    chunks: Sequence[ValueRef],
    result_bit_width: int,
    name: str,
    descriptor_lookup: _DescriptorLookup,
) -> tuple[list[EmitDescriptorOp], ValueRef]:
    if len(chunks) == 1:
        return [], chunks[0]
    zero = descriptor_lookup(
        "x86.avx2.vxorps.zero.ymm"
        if result_bit_width == 256
        else "x86.avx512.vxorps.zero.zmm"
    )
    insert = descriptor_lookup(
        "x86.avx2.vinsertf128.ymm.xmm"
        if result_bit_width == 256
        else "x86.avx512.vinserti64x4.zmm.ymm"
    )
    empty = ValueRef.temporary(f"empty_{name}")
    emits = [
        _op_emit(
            descriptor=zero,
            results={"dst": empty},
            result_types={"dst": DescriptorResultType()},
        )
    ]
    dest = empty
    for lane, chunk in enumerate(chunks):
        next_dest = (
            ValueRef.temporary(name)
            if lane + 1 == len(chunks)
            else ValueRef.temporary(f"{name}_with_chunk{lane}")
        )
        emits.append(
            _op_emit(
                descriptor=insert,
                operands={"dest": dest, "value": chunk},
                results={"dst": next_dest},
                result_types={"dst": DescriptorResultType()},
                immediates={"lane": lane},
            )
        )
        dest = next_dest
    return emits, dest


def _concat_rules(
    descriptor_lookup: _DescriptorLookup,
    input_bit_widths: tuple[int, ...],
) -> tuple[DescriptorRule, ...]:
    rules = []
    for family in X86_VECTOR_STORAGE_FAMILIES:
        for input_bit_width in input_bit_widths:
            if input_bit_width < family.element_bit_width:
                continue
            lane_count = input_bit_width // family.element_bit_width
            inputs = tuple(ValueRef.operand("inputs", element=i) for i in range(2))
            if input_bit_width < 128:
                mnemonic = {
                    8: "vpunpcklbw",
                    16: "vpunpcklwd",
                    32: "vpunpckldq",
                    64: "vpunpcklqdq",
                }[input_bit_width]
                descriptor = descriptor_lookup(f"x86.avx2.{mnemonic}.xmm")
                emits = [
                    _op_emit(
                        descriptor=descriptor,
                        operands={"lhs": inputs[0], "rhs": inputs[1]},
                        results={"dst": ValueRef.result("result")},
                    )
                ]
            else:
                emits, _ = _assemble_chunks(
                    inputs, input_bit_width * 2, "joined", descriptor_lookup
                )
                emits[-1] = replace(
                    emits[-1], results={"dst": ValueRef.result("result")}
                )
                descriptor = emits[-1].descriptor
            rules.append(
                DescriptorRule(
                    source_op=vector.vector_concat,
                    descriptor=descriptor,
                    guards=(
                        Guard.i64_range("axis", 0, 0),
                        Guard.operand_segment_count("inputs", 2),
                        Guard.value_type(
                            "inputs", Vector(family.element_names, lanes=lane_count)
                        ),
                        Guard.value_type(
                            "inputs",
                            Vector(family.element_names, lanes=lane_count),
                            element=1,
                        ),
                        Guard.value_type(
                            "result", Vector(family.element_names, lanes=lane_count * 2)
                        ),
                    ),
                    emit=tuple(emits),
                )
            )
    return tuple(rules)


def _packed_byte_ordinals(start: int, count: int) -> int:
    return sum((start + index) << (index * 8) for index in range(count))


def _iota_seed_emits(
    element_bit_width: int,
    vector_bit_width: int,
    profile: _IotaProfile,
    descriptor_lookup: _DescriptorLookup,
) -> tuple[list[EmitDescriptorOp], ValueRef]:
    lane_count = vector_bit_width // element_bit_width
    emits: list[EmitDescriptorOp] = []

    def build_qword(start: int, name: str) -> ValueRef:
        move_imm = descriptor_lookup("x86.scalar.movimm.gpr64")
        move = descriptor_lookup("x86.avx2.vmovq.xmm.gpr64")
        bits = ValueRef.temporary(f"{name}_bits")
        value = ValueRef.temporary(name)
        emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=move_imm,
                    results={"dst": bits},
                    result_types={"dst": _I64},
                    immediates={"imm64": _packed_byte_ordinals(start, 8)},
                    form=DescriptorEmitForm.CONST,
                ),
                _op_emit(
                    descriptor=move,
                    operands={"input": bits},
                    results={"dst": value},
                    result_types={"dst": DescriptorResultType()},
                ),
            )
        )
        return value

    def add_byte_offset(
        source: ValueRef,
        source_bit_width: int,
        offset: int,
        name: str,
    ) -> ValueRef:
        move_imm = descriptor_lookup("x86.scalar.movimm.gpr32")
        broadcast = descriptor_lookup(
            f"x86.avx512.vpbroadcastb.{_REGISTER_SUFFIXES[source_bit_width]}"
        )
        add = descriptor_lookup(
            f"x86.avx2.vpaddb.{_REGISTER_SUFFIXES[source_bit_width]}"
        )
        offset_bits = ValueRef.temporary(f"{name}_offset_bits")
        offsets = ValueRef.temporary(f"{name}_offsets")
        result = ValueRef.temporary(name)
        emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=move_imm,
                    results={"dst": offset_bits},
                    result_types={"dst": _I32},
                    immediates={"imm32": offset},
                    form=DescriptorEmitForm.CONST,
                ),
                _op_emit(
                    descriptor=broadcast,
                    operands={"value": offset_bits},
                    results={"dst": offsets},
                    result_types={"dst": DescriptorResultType()},
                ),
                _op_emit(
                    descriptor=add,
                    operands={"lhs": source, "rhs": offsets},
                    results={"dst": result},
                    result_types={"dst": DescriptorResultType()},
                ),
            )
        )
        return result

    if lane_count <= 4:
        move_imm = descriptor_lookup("x86.scalar.movimm.gpr32")
        move = descriptor_lookup("x86.avx2.vmovd.xmm.gpr32")
        bits = ValueRef.temporary("seed_bits")
        byte_seed = ValueRef.temporary("byte_seed")
        emits.extend(
            (
                EmitDescriptorOp(
                    descriptor=move_imm,
                    results={"dst": bits},
                    result_types={"dst": _I32},
                    immediates={"imm32": _packed_byte_ordinals(0, lane_count)},
                    form=DescriptorEmitForm.CONST,
                ),
                _op_emit(
                    descriptor=move,
                    operands={"input": bits},
                    results={"dst": byte_seed},
                    result_types={"dst": DescriptorResultType()},
                ),
            )
        )
    elif lane_count == 8:
        byte_seed = build_qword(0, "byte_seed")
    else:
        interleave = descriptor_lookup("x86.avx2.vpunpcklqdq.xmm")
        chunks: list[ValueRef] = []
        materialized_chunk_count = (
            1
            if profile.broadcast_source == _IntegerBroadcastSource.GPR
            and lane_count in (32, 64)
            else lane_count // 16
        )
        for chunk_index in range(materialized_chunk_count):
            start = chunk_index * 16
            low_qword = build_qword(start, f"seed_qword{chunk_index * 2}")
            high_qword = build_qword(start + 8, f"seed_qword{chunk_index * 2 + 1}")
            chunk = ValueRef.temporary(f"byte_seed_chunk{chunk_index}")
            emits.append(
                _op_emit(
                    descriptor=interleave,
                    operands={"lhs": low_qword, "rhs": high_qword},
                    results={"dst": chunk},
                    result_types={"dst": DescriptorResultType()},
                )
            )
            chunks.append(chunk)
        if materialized_chunk_count == 1 and lane_count in (32, 64):
            chunks.append(add_byte_offset(chunks[0], 128, 16, "byte_seed_chunk1"))
            assemble_emits, low_half = _assemble_chunks(
                chunks,
                256,
                "byte_seed_low",
                descriptor_lookup,
            )
            emits.extend(assemble_emits)
            if lane_count == 32:
                byte_seed = low_half
            else:
                high_half = add_byte_offset(low_half, 256, 32, "byte_seed_high")
                assemble_emits, byte_seed = _assemble_chunks(
                    (low_half, high_half),
                    512,
                    "byte_seed",
                    descriptor_lookup,
                )
                emits.extend(assemble_emits)
        else:
            assemble_emits, byte_seed = _assemble_chunks(
                chunks,
                lane_count * 8,
                "byte_seed",
                descriptor_lookup,
            )
            emits.extend(assemble_emits)

    if element_bit_width == 8:
        return emits, byte_seed
    suffix = {16: "w", 32: "d", 64: "q"}[element_bit_width]
    widen_prefix = profile.descriptor_prefix if vector_bit_width == 512 else "x86.avx2"
    source_suffix = (
        "ymm" if element_bit_width == 16 and vector_bit_width == 512 else "xmm"
    )
    widen = descriptor_lookup(
        f"{widen_prefix}.vpmovzxb{suffix}."
        f"{_REGISTER_SUFFIXES[vector_bit_width]}.{source_suffix}"
    )
    seed = ValueRef.temporary("lane_seed")
    emits.append(
        _op_emit(
            descriptor=widen,
            operands={"source": byte_seed},
            results={"dst": seed},
            result_types={"dst": DescriptorResultType()},
        )
    )
    return emits, seed


def _iota_splat_emits(
    source: ValueRef,
    *,
    element_bit_width: int,
    vector_bit_width: int,
    name: str,
    profile: _IotaProfile,
    descriptor_lookup: _DescriptorLookup,
    result: ValueRef | None = None,
) -> tuple[list[EmitDescriptorOp], ValueRef]:
    broadcast = descriptor_lookup(
        _integer_broadcast_descriptor_key(
            profile.descriptor_prefix,
            element_bit_width,
            vector_bit_width,
            broadcast_source=profile.broadcast_source,
        )
    )
    result = result or ValueRef.temporary(name)
    if profile.broadcast_source == _IntegerBroadcastSource.GPR:
        return [
            _op_emit(
                descriptor=broadcast,
                operands={"value": source},
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
            )
        ], result
    if element_bit_width == 64:
        move = descriptor_lookup("x86.avx2.vmovq.xmm.gpr64")
    else:
        move = descriptor_lookup("x86.avx2.vmovd.xmm.gpr32")
    lane = ValueRef.temporary(f"{name}_lane")
    return [
        _op_emit(
            descriptor=move,
            operands={"input": source},
            results={"dst": lane},
            result_types={"dst": DescriptorResultType()},
        ),
        _op_emit(
            descriptor=broadcast,
            operands={"value": lane},
            results={"dst": result},
            result_types={"dst": DescriptorResultType()},
        ),
    ], result


def _avx2_iota_scale_i8_emits(
    seed: ValueRef,
    *,
    vector_bit_width: int,
    descriptor_lookup: _DescriptorLookup,
) -> tuple[list[EmitDescriptorOp], ValueRef]:
    emits, step_words = _iota_splat_emits(
        ValueRef.operand("step"),
        element_bit_width=16,
        vector_bit_width=128,
        name="step_words",
        profile=_AVX2_IOTA_PROFILE,
        descriptor_lookup=descriptor_lookup,
    )
    move_imm = descriptor_lookup("x86.scalar.movimm.gpr32")
    mask_bits = ValueRef.temporary("byte_mask_bits")
    emits.append(
        EmitDescriptorOp(
            descriptor=move_imm,
            results={"dst": mask_bits},
            result_types={"dst": _I32},
            immediates={"imm32": 255},
            form=DescriptorEmitForm.CONST,
        )
    )
    mask_emits, byte_mask = _iota_splat_emits(
        mask_bits,
        element_bit_width=16,
        vector_bit_width=128,
        name="byte_mask",
        profile=_AVX2_IOTA_PROFILE,
        descriptor_lookup=descriptor_lookup,
    )
    emits.extend(mask_emits)
    extract_chunk = (
        descriptor_lookup("x86.avx2.vextractf128.xmm.ymm")
        if vector_bit_width == 256
        else None
    )
    shift = descriptor_lookup("x86.avx2.vpsrldq.xmm")
    widen = descriptor_lookup("x86.avx2.vpmovzxbw.xmm.xmm")
    multiply = descriptor_lookup("x86.avx2.vpmullw.xmm")
    and_ = descriptor_lookup("x86.avx2.vpand.xmm")
    pack = descriptor_lookup("x86.avx2.vpackuswb.xmm")
    packed_chunks: list[ValueRef] = []
    # A 64-bit logical packet occupies the low half of one XMM carrier. Run the
    # same physical 128-bit chunk program and retain its low eight byte lanes.
    chunk_count = max(1, vector_bit_width // 128)
    for chunk_index in range(chunk_count):
        chunk = seed
        if extract_chunk is not None:
            chunk = ValueRef.temporary(f"seed_chunk{chunk_index}")
            emits.append(
                _op_emit(
                    descriptor=extract_chunk,
                    operands={"source": seed},
                    results={"dst": chunk},
                    result_types={"dst": DescriptorResultType()},
                    immediates={"lane": chunk_index},
                )
            )
        high_bytes = ValueRef.temporary(f"seed_chunk{chunk_index}_high_bytes")
        low_words = ValueRef.temporary(f"seed_chunk{chunk_index}_low_words")
        high_words = ValueRef.temporary(f"seed_chunk{chunk_index}_high_words")
        low_products = ValueRef.temporary(f"seed_chunk{chunk_index}_low_products")
        high_products = ValueRef.temporary(f"seed_chunk{chunk_index}_high_products")
        low_bytes = ValueRef.temporary(f"seed_chunk{chunk_index}_low_bytes")
        high_bytes_masked = ValueRef.temporary(
            f"seed_chunk{chunk_index}_high_bytes_masked"
        )
        packed = ValueRef.temporary(f"scaled_chunk{chunk_index}")
        emits.extend(
            (
                _op_emit(
                    descriptor=shift,
                    operands={"source": chunk},
                    results={"dst": high_bytes},
                    result_types={"dst": DescriptorResultType()},
                    immediates={"bytes": 8},
                ),
                _op_emit(
                    descriptor=widen,
                    operands={"source": chunk},
                    results={"dst": low_words},
                    result_types={"dst": DescriptorResultType()},
                ),
                _op_emit(
                    descriptor=widen,
                    operands={"source": high_bytes},
                    results={"dst": high_words},
                    result_types={"dst": DescriptorResultType()},
                ),
                _op_emit(
                    descriptor=multiply,
                    operands={"lhs": low_words, "rhs": step_words},
                    results={"dst": low_products},
                    result_types={"dst": DescriptorResultType()},
                ),
                _op_emit(
                    descriptor=multiply,
                    operands={"lhs": high_words, "rhs": step_words},
                    results={"dst": high_products},
                    result_types={"dst": DescriptorResultType()},
                ),
                _op_emit(
                    descriptor=and_,
                    operands={"lhs": low_products, "rhs": byte_mask},
                    results={"dst": low_bytes},
                    result_types={"dst": DescriptorResultType()},
                ),
                _op_emit(
                    descriptor=and_,
                    operands={"lhs": high_products, "rhs": byte_mask},
                    results={"dst": high_bytes_masked},
                    result_types={"dst": DescriptorResultType()},
                ),
                _op_emit(
                    descriptor=pack,
                    operands={"lhs": low_bytes, "rhs": high_bytes_masked},
                    results={"dst": packed},
                    result_types={"dst": DescriptorResultType()},
                ),
            )
        )
        packed_chunks.append(packed)
    assemble_emits, result = _assemble_chunks(
        packed_chunks,
        vector_bit_width,
        "scaled_offsets",
        descriptor_lookup,
    )
    emits.extend(assemble_emits)
    return emits, result


def _avx512_iota_scale_i8_emits(
    seed: ValueRef,
    descriptor_lookup: _DescriptorLookup,
) -> tuple[list[EmitDescriptorOp], ValueRef]:
    emits, steps = _iota_splat_emits(
        ValueRef.operand("step"),
        element_bit_width=16,
        vector_bit_width=512,
        name="step_words",
        profile=_AVX512_IOTA_PROFILE,
        descriptor_lookup=descriptor_lookup,
    )
    extract = descriptor_lookup("x86.avx512.vextracti64x4.ymm.zmm")
    widen = descriptor_lookup("x86.avx512.vpmovzxbw.zmm.ymm")
    multiply = descriptor_lookup("x86.avx512.vpmullw.zmm")
    narrow = descriptor_lookup("x86.avx512.vpmovwb.ymm.zmm")
    scaled_halves: list[ValueRef] = []
    for half_index in range(2):
        source_half = ValueRef.temporary(f"seed_half{half_index}")
        words = ValueRef.temporary(f"seed_half{half_index}_words")
        products = ValueRef.temporary(f"seed_half{half_index}_products")
        scaled_half = ValueRef.temporary(f"scaled_half{half_index}")
        emits.extend(
            (
                _op_emit(
                    descriptor=extract,
                    operands={"source": seed},
                    results={"dst": source_half},
                    result_types={"dst": DescriptorResultType()},
                    immediates={"lane": half_index},
                ),
                _op_emit(
                    descriptor=widen,
                    operands={"source": source_half},
                    results={"dst": words},
                    result_types={"dst": DescriptorResultType()},
                ),
                _op_emit(
                    descriptor=multiply,
                    operands={"lhs": words, "rhs": steps},
                    results={"dst": products},
                    result_types={"dst": DescriptorResultType()},
                ),
                _op_emit(
                    descriptor=narrow,
                    operands={"source": products},
                    results={"dst": scaled_half},
                    result_types={"dst": DescriptorResultType()},
                ),
            )
        )
        scaled_halves.append(scaled_half)
    assemble_emits, result = _assemble_chunks(
        (scaled_halves[0], scaled_halves[1]),
        512,
        "scaled_offsets",
        descriptor_lookup,
    )
    emits.extend(assemble_emits)
    return emits, result


def _avx2_iota_scale_i64_emits(
    seed: ValueRef,
    *,
    vector_bit_width: int,
    descriptor_lookup: _DescriptorLookup,
) -> tuple[list[EmitDescriptorOp], ValueRef]:
    emits, steps = _iota_splat_emits(
        ValueRef.operand("step"),
        element_bit_width=64,
        vector_bit_width=vector_bit_width,
        name="steps",
        profile=_AVX2_IOTA_PROFILE,
        descriptor_lookup=descriptor_lookup,
    )
    zero_descriptor = descriptor_lookup(
        f"x86.avx2.vxorps.zero.{_REGISTER_SUFFIXES[vector_bit_width]}"
    )
    subtract = descriptor_lookup(
        f"x86.avx2.vpsubq.{_REGISTER_SUFFIXES[vector_bit_width]}"
    )
    and_ = descriptor_lookup(f"x86.avx2.vpand.{_REGISTER_SUFFIXES[vector_bit_width]}")
    add = descriptor_lookup(f"x86.avx2.vpaddq.{_REGISTER_SUFFIXES[vector_bit_width]}")
    zero = ValueRef.temporary("zero_offsets")
    low_mask = ValueRef.temporary("low_lane_mask")
    if vector_bit_width == 128:
        emits.extend(
            (
                _op_emit(
                    descriptor=zero_descriptor,
                    results={"dst": zero},
                    result_types={"dst": DescriptorResultType()},
                ),
                _op_emit(
                    descriptor=subtract,
                    operands={"lhs": zero, "rhs": seed},
                    results={"dst": low_mask},
                    result_types={"dst": DescriptorResultType()},
                ),
            )
        )
        result = ValueRef.temporary("scaled_offsets")
        emits.append(
            _op_emit(
                descriptor=and_,
                operands={"lhs": steps, "rhs": low_mask},
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
            )
        )
        return emits, result

    move_imm = descriptor_lookup("x86.scalar.movimm.gpr64")
    one = ValueRef.temporary("one")
    emits.append(
        EmitDescriptorOp(
            descriptor=move_imm,
            results={"dst": one},
            result_types={"dst": _I64},
            immediates={"imm64": 1},
            form=DescriptorEmitForm.CONST,
        )
    )
    one_emits, ones = _iota_splat_emits(
        one,
        element_bit_width=64,
        vector_bit_width=256,
        name="ones",
        profile=_AVX2_IOTA_PROFILE,
        descriptor_lookup=descriptor_lookup,
    )
    emits.extend(one_emits)
    shift = descriptor_lookup("x86.avx2.vpsrlq.ymm")
    low_bits = ValueRef.temporary("low_lane_bits")
    high_bits = ValueRef.temporary("high_lane_bits")
    high_mask = ValueRef.temporary("high_lane_mask")
    low_offsets = ValueRef.temporary("low_offsets")
    doubled_steps = ValueRef.temporary("doubled_steps")
    high_offsets = ValueRef.temporary("high_offsets")
    result = ValueRef.temporary("scaled_offsets")
    emits.extend(
        (
            _op_emit(
                descriptor=zero_descriptor,
                results={"dst": zero},
                result_types={"dst": DescriptorResultType()},
            ),
            _op_emit(
                descriptor=and_,
                operands={"lhs": seed, "rhs": ones},
                results={"dst": low_bits},
                result_types={"dst": DescriptorResultType()},
            ),
            _op_emit(
                descriptor=subtract,
                operands={"lhs": zero, "rhs": low_bits},
                results={"dst": low_mask},
                result_types={"dst": DescriptorResultType()},
            ),
            _op_emit(
                descriptor=shift,
                operands={"source": seed},
                results={"dst": high_bits},
                result_types={"dst": DescriptorResultType()},
                immediates={"shift": 1},
            ),
            _op_emit(
                descriptor=subtract,
                operands={"lhs": zero, "rhs": high_bits},
                results={"dst": high_mask},
                result_types={"dst": DescriptorResultType()},
            ),
            _op_emit(
                descriptor=and_,
                operands={"lhs": steps, "rhs": low_mask},
                results={"dst": low_offsets},
                result_types={"dst": DescriptorResultType()},
            ),
            _op_emit(
                descriptor=add,
                operands={"lhs": steps, "rhs": steps},
                results={"dst": doubled_steps},
                result_types={"dst": DescriptorResultType()},
            ),
            _op_emit(
                descriptor=and_,
                operands={"lhs": doubled_steps, "rhs": high_mask},
                results={"dst": high_offsets},
                result_types={"dst": DescriptorResultType()},
            ),
            _op_emit(
                descriptor=add,
                operands={"lhs": low_offsets, "rhs": high_offsets},
                results={"dst": result},
                result_types={"dst": DescriptorResultType()},
            ),
        )
    )
    return emits, result


def _unique_emit_descriptors(
    emits: Iterable[EmitDescriptorOp],
) -> tuple[Descriptor, ...]:
    descriptors: dict[str, Descriptor] = {}
    for emit in emits:
        descriptors.setdefault(emit.descriptor.key, emit.descriptor)
    return tuple(descriptors.values())


def _integer_iota_rule(
    element: VectorElement,
    vector_bit_width: int,
    *,
    unit_step: bool,
    profile: _IotaProfile,
    descriptor_lookup: _DescriptorLookup,
) -> DescriptorRule:
    if vector_bit_width == element.bit_width:
        emits, _ = _iota_splat_emits(
            ValueRef.operand("base"),
            element_bit_width=element.bit_width,
            vector_bit_width=vector_bit_width,
            name="result",
            profile=profile,
            descriptor_lookup=descriptor_lookup,
            result=ValueRef.result("result"),
        )
        descriptor = emits[-1].descriptor
        return DescriptorRule(
            source_op=vector.vector_iota,
            descriptor=descriptor,
            guards=(
                Guard.value_type("base", Scalar(element.name)),
                Guard.value_type("step", Scalar(element.name)),
                Guard.value_type(
                    "result", _full_vector_type(element, vector_bit_width)
                ),
                *(
                    Guard.descriptor_available(required_descriptor)
                    for required_descriptor in _unique_emit_descriptors(emits)
                    if required_descriptor != descriptor
                ),
            ),
            emit=tuple(emits),
            priority=0,
        )

    emits, offsets = _iota_seed_emits(
        element.bit_width, vector_bit_width, profile, descriptor_lookup
    )
    direct_gpr_broadcast = profile.broadcast_source == _IntegerBroadcastSource.GPR
    if not unit_step:
        if element.bit_width == 8:
            if direct_gpr_broadcast:
                scale_emits, offsets = _avx512_iota_scale_i8_emits(
                    offsets, descriptor_lookup
                )
            else:
                scale_emits, offsets = _avx2_iota_scale_i8_emits(
                    offsets,
                    vector_bit_width=vector_bit_width,
                    descriptor_lookup=descriptor_lookup,
                )
            emits.extend(scale_emits)
        elif element.bit_width in (16, 32) or direct_gpr_broadcast:
            step_emits, steps = _iota_splat_emits(
                ValueRef.operand("step"),
                element_bit_width=element.bit_width,
                vector_bit_width=vector_bit_width,
                name="steps",
                profile=profile,
                descriptor_lookup=descriptor_lookup,
            )
            emits.extend(step_emits)
            multiply_suffix = {16: "w", 32: "d", 64: "q"}[element.bit_width]
            multiply = descriptor_lookup(
                f"{profile.descriptor_prefix}.vpmull{multiply_suffix}."
                f"{_REGISTER_SUFFIXES[vector_bit_width]}"
            )
            scaled_offsets = ValueRef.temporary("scaled_offsets")
            emits.append(
                _op_emit(
                    descriptor=multiply,
                    operands={"lhs": offsets, "rhs": steps},
                    results={"dst": scaled_offsets},
                    result_types={"dst": DescriptorResultType()},
                )
            )
            offsets = scaled_offsets
        else:
            scale_emits, offsets = _avx2_iota_scale_i64_emits(
                offsets,
                vector_bit_width=vector_bit_width,
                descriptor_lookup=descriptor_lookup,
            )
            emits.extend(scale_emits)
    base_emits, bases = _iota_splat_emits(
        ValueRef.operand("base"),
        element_bit_width=element.bit_width,
        vector_bit_width=vector_bit_width,
        name="bases",
        profile=profile,
        descriptor_lookup=descriptor_lookup,
    )
    emits.extend(base_emits)
    suffix = {8: "b", 16: "w", 32: "d", 64: "q"}[element.bit_width]
    add = descriptor_lookup(
        f"{profile.descriptor_prefix}.vpadd{suffix}."
        f"{_REGISTER_SUFFIXES[vector_bit_width]}"
    )
    emits.append(
        _op_emit(
            descriptor=add,
            operands={"lhs": bases, "rhs": offsets},
            results={"dst": ValueRef.result("result")},
        )
    )
    descriptors = _unique_emit_descriptors(emits)
    return DescriptorRule(
        source_op=vector.vector_iota,
        descriptor=add,
        guards=(
            Guard.value_type("base", Scalar(element.name)),
            Guard.value_type("step", Scalar(element.name)),
            Guard.value_type("result", _full_vector_type(element, vector_bit_width)),
            *((Guard.value_i64_range("step", 1, 1),) if unit_step else ()),
            *(
                Guard.descriptor_available(descriptor)
                for descriptor in descriptors
                if descriptor != add
            ),
        ),
        emit=tuple(emits),
        priority=1 if unit_step else 0,
    )


def avx2_vector_construction_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[ContractCase, ...]:
    return (
        *_concat_rules(descriptor_lookup, (8, 16, 32, 64, 128)),
        *_vector_zero_rules(
            descriptor_lookup,
            descriptor_key_prefix="x86.avx2",
            vector_bit_widths=AVX2_VECTOR_BIT_WIDTHS,
        ),
        *_vector_splat_rules(
            descriptor_lookup,
            descriptor_key_prefix="x86.avx2",
            integer_vector_bit_widths=AVX2_CARRIER_VECTOR_BIT_WIDTHS,
            float_vector_bit_widths=AVX2_CARRIER_VECTOR_BIT_WIDTHS,
            broadcast_source=_IntegerBroadcastSource.XMM_LOW_LANE,
        ),
        *_exact_lane_vector_rules(
            descriptor_lookup,
            load_descriptor_prefix="x86.avx2.vmovdqu.rodata",
            vector_bit_widths=AVX2_VECTOR_BIT_WIDTHS,
        ),
        *_vector_bitcast_alias_rules(AVX2_CARRIER_VECTOR_BIT_WIDTHS),
        *(
            _integer_iota_rule(
                element,
                vector_bit_width,
                unit_step=unit_step,
                profile=_AVX2_IOTA_PROFILE,
                descriptor_lookup=descriptor_lookup,
            )
            for element in INTEGER_ELEMENTS
            for vector_bit_width in AVX2_CARRIER_VECTOR_BIT_WIDTHS
            for unit_step in (True, False)
            if vector_bit_width >= element.bit_width
            if vector_bit_width != element.bit_width or not unit_step
        ),
    )


def avx512_fp16_vector_construction_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[ContractCase, ...]:
    """Constructs FP16 vectors including the 64-bit low-XMM payload."""
    low_xmm_type = Vector(FP16_ELEMENT.name, lanes=4)
    low_xmm_bits_type = Vector("i16", lanes=4)
    splat_rules = tuple(
        _splat_rule(
            Scalar(FP16_ELEMENT.name),
            _full_vector_type(FP16_ELEMENT, vector_bit_width),
            (
                "x86.avx512_fp16.vpbroadcastw.zmm.xmm"
                if vector_bit_width == 512
                else f"x86.avx2.vpbroadcastw.{_REGISTER_SUFFIXES[vector_bit_width]}"
            ),
            descriptor_lookup,
            # The AVX-512 base fragment also has a priority-1 storage splat
            # for the ordinary GPR scalar carrier. The FP16 feature overlay
            # must select its native XMM carrier instead.
            priority=2,
        )
        for vector_bit_width in AVX512_FP16_VECTOR_BIT_WIDTHS
    )
    return (
        *splat_rules,
        ValueAliasRule(
            source_op=vector.vector_bitcast,
            source=ValueRef.operand("input"),
            result=ValueRef.result("result"),
            guards=(
                Guard.value_type("input", low_xmm_type),
                Guard.value_type("result", low_xmm_bits_type),
            ),
        ),
        ValueAliasRule(
            source_op=vector.vector_bitcast,
            source=ValueRef.operand("input"),
            result=ValueRef.result("result"),
            guards=(
                Guard.value_type("input", low_xmm_bits_type),
                Guard.value_type("result", low_xmm_type),
            ),
        ),
    )


def avx512_vector_construction_rules(
    descriptor_lookup: _DescriptorLookup,
) -> tuple[ContractCase, ...]:
    return (
        *_concat_rules(descriptor_lookup, (256,)),
        *_vector_zero_rules(
            descriptor_lookup,
            descriptor_key_prefix="x86.avx512",
            vector_bit_widths=AVX512_VECTOR_BIT_WIDTHS,
        ),
        *_vector_splat_rules(
            descriptor_lookup,
            descriptor_key_prefix="x86.avx512",
            integer_vector_bit_widths=AVX512_DIRECT_BROADCAST_VECTOR_BIT_WIDTHS,
            float_vector_bit_widths=AVX512_VECTOR_BIT_WIDTHS,
            broadcast_source=_IntegerBroadcastSource.GPR,
            priority=1,
        ),
        *_exact_lane_vector_rules(
            descriptor_lookup,
            load_descriptor_prefix="x86.avx512.vmovdqu64.rodata",
            vector_bit_widths=AVX512_VECTOR_BIT_WIDTHS,
        ),
        *_vector_bitcast_alias_rules(AVX512_VECTOR_BIT_WIDTHS),
        *(
            _integer_iota_rule(
                element,
                512,
                unit_step=unit_step,
                profile=_AVX512_IOTA_PROFILE,
                descriptor_lookup=descriptor_lookup,
            )
            for element in INTEGER_ELEMENTS
            for unit_step in (True, False)
        ),
    )
