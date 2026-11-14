# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Bit-preserving interleaving and deinterleaving of native numeric packets."""

from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.rule_builders import (
    DescriptorLookup,
)
from loom.target.arch.x86.contracts.rule_builders import (
    emit_descriptor_op as _op_emit,
)
from loom.target.arch.x86.vector_families import X86_VECTOR_STORAGE_FAMILIES
from loom.target.contracts import (
    ContractEmit,
    DescriptorResultType,
    DescriptorRule,
    EmitDescriptorOp,
    EmitRegisterCopy,
    Guard,
    ReadOnlyDataLiteral,
    ValueRef,
    Vector,
)

_SUFFIXES = {128: "xmm", 256: "ymm", 512: "zmm"}
_ELEMENT_SUFFIXES = {8: "bw", 16: "wd", 32: "dq", 64: "qdq"}
_PERMUTE_SUFFIXES = {8: "b", 16: "w", 32: "d", 64: "q"}


def _literal(lookup: DescriptorLookup, prefix: str, width: int, data: bytes):
    load = "vmovdqu64" if prefix == "avx512" else "vmovdqu"
    return _op_emit(
        descriptor=lookup(f"x86.{prefix}.{load}.rodata.{_SUFFIXES[width]}"),
        results={"dst": ValueRef.temporary("control")},
        result_types={"dst": DescriptorResultType()},
        immediates={"data": ReadOnlyDataLiteral(data)},
    )


def _unpack_emits(
    lookup: DescriptorLookup,
    element_bit_width: int,
    result_bit_width: int,
    *,
    evex: bool,
) -> tuple[ContractEmit, ...]:
    even, odd = ValueRef.operand("even"), ValueRef.operand("odd")
    result = ValueRef.result("result")
    input_width = max(128, result_bit_width // 2)
    suffix = _SUFFIXES[input_width]
    # The base 512-bit composition must remain available without AVX512VL.
    prefix = "avx512" if evex and result_bit_width <= 128 else "avx2"
    low = lookup(
        f"x86.{prefix}.vpunpckl{_ELEMENT_SUFFIXES[element_bit_width]}.{suffix}"
    )
    if result_bit_width <= 128:
        return (
            _op_emit(
                descriptor=low,
                operands={"lhs": even, "rhs": odd},
                results={"dst": result},
            ),
        )
    high = lookup(
        f"x86.{prefix}.vpunpckh{_ELEMENT_SUFFIXES[element_bit_width]}.{suffix}"
    )
    chunks = [ValueRef.temporary(name) for name in ("low", "high")]
    emits = [
        _op_emit(
            descriptor=descriptor,
            operands={"lhs": even, "rhs": odd},
            results={"dst": chunk},
            result_types={"dst": DescriptorResultType()},
        )
        for descriptor, chunk in zip((low, high), chunks, strict=True)
    ]
    concat = lookup(
        "x86.avx2.vinsertf128.concat.ymm"
        if result_bit_width == 256
        else "x86.avx512.vinserti64x4.concat.zmm"
    )
    joined = result if result_bit_width == 256 else ValueRef.temporary("joined")
    emits.append(
        _op_emit(
            descriptor=concat,
            operands={"low": chunks[0], "high": chunks[1]},
            results={"dst": joined},
            result_types={"dst": DescriptorResultType()},
        )
    )
    if result_bit_width == 512:
        # Unpacks act independently in each 128-bit lane. Restore lane order
        # after joining their low/high results: [0, 2, 1, 3] -> [0, 1, 2, 3].
        emits.append(
            _op_emit(
                descriptor=lookup("x86.avx512.vshufi64x2.zmm"),
                operands={"lhs": joined, "rhs": joined},
                results={"dst": result},
                immediates={"control": 0xD8},
            )
        )
    return tuple(emits)


def _permute_emits(
    lookup: DescriptorLookup,
    element_bit_width: int,
    result_bit_width: int,
) -> tuple[ContractEmit, ...]:
    suffix = _SUFFIXES[result_bit_width]
    count = result_bit_width // element_bit_width
    control_data = b"".join(
        index.to_bytes(element_bit_width // 8, "little")
        for lane in range(count // 2)
        for index in (lane, lane + count)
    )
    control = ValueRef.temporary("control")
    even, odd = ValueRef.temporary("wide_even"), ValueRef.temporary("wide_odd")
    return (
        _literal(lookup, "avx512", result_bit_width, control_data),
        # The control selects only the original low half of each table. The
        # existing register copy maps physical carriers; source types are exact.
        EmitRegisterCopy(
            source=ValueRef.operand("even"),
            result=even,
            result_type=Vector("i8", lanes=result_bit_width // 8),
        ),
        EmitRegisterCopy(
            source=ValueRef.operand("odd"),
            result=odd,
            result_type=Vector("i8", lanes=result_bit_width // 8),
        ),
        _op_emit(
            descriptor=lookup(
                f"x86.avx512.vpermt2{_PERMUTE_SUFFIXES[element_bit_width]}.{suffix}"
            ),
            operands={"lhs": even, "control": control, "rhs": odd},
            results={"dst": ValueRef.result("result")},
        ),
    )


def _deinterleave_narrow_emits(
    lookup: DescriptorLookup,
    element_bit_width: int,
    source_bit_width: int,
) -> tuple[ContractEmit, ...]:
    suffix = _SUFFIXES[max(128, source_bit_width)]
    result_suffix = _SUFFIXES[max(128, source_bit_width // 2)]
    pair_suffix = _PERMUTE_SUFFIXES[element_bit_width * 2]
    narrow = lookup(
        f"x86.avx512.vpmov{pair_suffix}{_PERMUTE_SUFFIXES[element_bit_width]}."
        f"{result_suffix}.{suffix}"
    )
    shifted = ValueRef.temporary("shifted")
    return (
        _op_emit(
            descriptor=narrow,
            operands={"source": ValueRef.operand("source")},
            results={"dst": ValueRef.result("even")},
        ),
        _op_emit(
            descriptor=lookup(f"x86.avx512.vpsrl{pair_suffix}.{suffix}"),
            operands={"source": ValueRef.operand("source")},
            results={"dst": shifted},
            result_types={"dst": DescriptorResultType()},
            immediates={"shift": element_bit_width},
        ),
        _op_emit(
            descriptor=narrow,
            operands={"source": shifted},
            results={"dst": ValueRef.result("odd")},
        ),
    )


def _deinterleave_shuffle_emits(
    lookup: DescriptorLookup,
    element_bit_width: int,
    source_bit_width: int,
) -> tuple[ContractEmit, ...]:
    width = max(128, source_bit_width)
    suffix = _SUFFIXES[width]
    source = ValueRef.operand("source")
    grouped = ValueRef.temporary("grouped")
    emits: list[ContractEmit] = []
    if width == 512:
        # Eight qwords, grouped by parity. Smaller storage elements use native
        # narrowing, which extracts the low half of each pair without rounding.
        data = b"".join(i.to_bytes(8, "little") for i in (0, 2, 4, 6, 1, 3, 5, 7))
        emits.extend(
            (
                _literal(lookup, "avx512", width, data),
                _op_emit(
                    descriptor=lookup("x86.avx512.vpermq.zmm"),
                    operands={
                        "source": source,
                        "control": ValueRef.temporary("control"),
                    },
                    results={"dst": grouped},
                    result_types={"dst": DescriptorResultType()},
                ),
            )
        )
        source = grouped
    elif element_bit_width <= 32 and source_bit_width > 2 * element_bit_width:
        shuffled = ValueRef.temporary("shuffled")
        if element_bit_width == 32 and width == 128:
            emits.append(
                _op_emit(
                    descriptor=lookup("x86.avx2.vpshufd.xmm"),
                    operands={"source": source},
                    results={"dst": shuffled},
                    result_types={"dst": DescriptorResultType()},
                    immediates={"control": 0xD8},
                )
            )
        else:
            lane_bytes = min(16, source_bit_width // 8)
            element_bytes = element_bit_width // 8
            lane_count = lane_bytes // element_bytes
            lane_control = bytes(
                lane * element_bytes + byte
                for parity in (0, 1)
                for lane in range(parity, lane_count, 2)
                for byte in range(element_bytes)
            ) + bytes([0x80] * (16 - lane_bytes))
            emits.extend(
                (
                    _literal(lookup, "avx2", width, lane_control * (width // 128)),
                    _op_emit(
                        descriptor=lookup(f"x86.avx2.vpshufb.{suffix}"),
                        operands={"lhs": source, "rhs": ValueRef.temporary("control")},
                        results={"dst": shuffled},
                        result_types={"dst": DescriptorResultType()},
                    ),
                )
            )
        source = shuffled
    if width == 256:
        emits.append(
            _op_emit(
                descriptor=lookup("x86.avx2.vpermq.ymm"),
                operands={"source": source},
                results={"dst": grouped},
                result_types={"dst": DescriptorResultType()},
                immediates={"control": 0xD8},
            )
        )
        source = grouped
    emits.append(EmitRegisterCopy(source=source, result=ValueRef.result("even")))
    if width == 128:
        descriptor = lookup("x86.avx2.vpsrldq.xmm")
        immediate = {"bytes": source_bit_width // 16}
    else:
        descriptor = lookup(
            "x86.avx2.vextractf128.xmm.ymm"
            if width == 256
            else "x86.avx512.vextracti64x4.ymm.zmm"
        )
        immediate = {"lane": 1}
    emits.append(
        _op_emit(
            descriptor=descriptor,
            operands={"source": source},
            results={"dst": ValueRef.result("odd")},
            immediates=immediate,
        )
    )
    return tuple(emits)


def deinterleave_rules(
    lookup: DescriptorLookup,
    *,
    evex: bool,
) -> tuple[DescriptorRule, ...]:
    rules = []
    for family in X86_VECTOR_STORAGE_FAMILIES:
        for width in (16, 32, 64, 128, 256, 512) if evex else (16, 32, 64, 128, 256):
            lanes = width // family.element_bit_width
            if lanes < 2 or (evex and family.element_bit_width == 64 and width < 512):
                continue
            emits = (
                _deinterleave_narrow_emits(lookup, family.element_bit_width, width)
                if evex and family.element_bit_width < 64
                else _deinterleave_shuffle_emits(
                    lookup, family.element_bit_width, width
                )
            )
            descriptors = tuple(
                emit.descriptor for emit in emits if isinstance(emit, EmitDescriptorOp)
            )
            rules.append(
                DescriptorRule(
                    source_op=vector.vector_deinterleave,
                    descriptor=descriptors[-1],
                    guards=(
                        Guard.i64_range("axis", 0, 0),
                        Guard.value_type(
                            "source", Vector(family.element_names, lanes=lanes)
                        ),
                        Guard.value_type(
                            "even", Vector(family.element_names, lanes=lanes // 2)
                        ),
                        Guard.value_type(
                            "odd", Vector(family.element_names, lanes=lanes // 2)
                        ),
                        *(Guard.descriptor_available(row) for row in descriptors),
                    ),
                    emit=emits,
                    priority=int(evex),
                )
            )
    return tuple(rules)


def interleave_rules(
    lookup: DescriptorLookup,
    *,
    evex: bool,
) -> tuple[DescriptorRule, ...]:
    rules = []
    for family in X86_VECTOR_STORAGE_FAMILIES:
        for width in (16, 32, 64, 128, 256, 512) if evex else (16, 32, 64, 128, 256):
            lanes = width // family.element_bit_width
            if lanes < 2:
                continue
            guards = (
                Guard.i64_range("axis", 0, 0),
                Guard.value_type(
                    "even", Vector(family.element_names, lanes=lanes // 2)
                ),
                Guard.value_type("odd", Vector(family.element_names, lanes=lanes // 2)),
                Guard.value_type("result", Vector(family.element_names, lanes=lanes)),
            )
            candidates = [
                (
                    _unpack_emits(lookup, family.element_bit_width, width, evex=evex),
                    int(evex),
                )
            ]
            if evex and width >= 256:
                candidates.append(
                    (_permute_emits(lookup, family.element_bit_width, width), 2)
                )
            for emits, priority in candidates:
                descriptors = tuple(
                    emit.descriptor
                    for emit in emits
                    if isinstance(emit, EmitDescriptorOp)
                )
                rules.append(
                    DescriptorRule(
                        source_op=vector.vector_interleave,
                        descriptor=descriptors[-1],
                        guards=(
                            *guards,
                            *(Guard.descriptor_available(row) for row in descriptors),
                        ),
                        emit=emits,
                        priority=priority,
                    )
                )
    return tuple(rules)
