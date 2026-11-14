# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Native predicate transport to integer bitstreams in logical lane order."""

from __future__ import annotations

from loom.dialect.vector import defs as vector
from loom.target.arch.x86.contracts.avx512_predicate import (
    avx512_predicate_representation_bit_widths,
    avx512_predicate_simd_to_mask_emits,
)
from loom.target.arch.x86.contracts.predicate import (
    avx2_predicate_representation_bit_widths,
    avx2_predicate_simd_conversion_emits,
)
from loom.target.arch.x86.contracts.rule_builders import (
    DescriptorLookup,
    emit_descriptor_op,
)
from loom.target.contracts import (
    DescriptorRule,
    EmitDescriptorOp,
    Guard,
    Scalar,
    ValueRef,
    Vector,
)


def _bitpack_rules(
    lane_count: int,
    representation: int,
    emits: tuple[EmitDescriptorOp, ...],
    *,
    priority: int,
) -> tuple[DescriptorRule, ...]:
    descriptors = tuple(
        {emit.descriptor.key: emit.descriptor for emit in emits}.values()
    )
    return tuple(
        DescriptorRule(
            source_op=vector.vector_bitpack,
            descriptor=descriptors[0],
            guards=(
                Guard.value_type("source", Vector("i1", lanes=lane_count)),
                Guard.value_type(
                    "result",
                    Vector(f"i{storage_width}", lanes=lane_count // storage_width),
                ),
                Guard.i64_range("width", 1, 1),
                Guard.low_value_representation("source", representation),
                *(
                    Guard.descriptor_available(descriptor)
                    for descriptor in descriptors[1:]
                ),
            ),
            emit=emits,
            priority=priority,
        )
        for storage_width in (8, 16, 32, 64)
        if storage_width <= lane_count
    )


def avx2_predicate_bitpack_rules(
    lookup: DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    """Packs every byte-addressable predicate carried by AVX2 SIMD registers."""
    transfer = lookup("x86.avx2.vmovd.xmm.gpr32")
    rules = []
    for lane_count in (8, 16, 32):
        for vector_width in avx2_predicate_representation_bit_widths(lane_count):
            element_width = vector_width // lane_count
            source = ValueRef.operand("source")
            emits = ()
            # Sign bits are truth bits only for the existing normalized i1
            # representations. Arbitrary integer low fields use bitstream
            # legalization instead of this mask extraction.
            if element_width == 16:
                compacted = ValueRef.temporary("bytes")
                emits, _ = avx2_predicate_simd_conversion_emits(
                    lookup,
                    source_element_bit_width=16,
                    result_element_bit_width=8,
                    lane_count=lane_count,
                    source=source,
                    result=compacted,
                    result_type=Vector("i8", lanes=lane_count),
                    temporary_prefix="compact",
                )
                source = compacted
            suffix = "ymm" if lane_count == 32 or element_width == 32 else "xmm"
            mnemonic = "vmovmskps" if element_width == 32 else "vpmovmskb"
            extract = lookup(f"x86.avx2.{mnemonic}.gpr32.{suffix}")
            emits = (
                *emits,
                emit_descriptor_op(
                    descriptor=extract,
                    operands={"source": source},
                    results={"dst": ValueRef.temporary("bits")},
                    result_types={"dst": Scalar("i32")},
                ),
                emit_descriptor_op(
                    descriptor=transfer,
                    operands={"input": ValueRef.temporary("bits")},
                    results={"dst": ValueRef.result("result")},
                ),
            )
            rules.extend(_bitpack_rules(lane_count, element_width, emits, priority=0))
    return tuple(rules)


def avx512_predicate_bitpack_rules(
    lookup: DescriptorLookup,
) -> tuple[DescriptorRule, ...]:
    """Packs native masks and every admitted SIMD predicate representation."""
    move = lookup("x86.avx512.kmovq.gpr64.k")
    transfer = lookup("x86.avx2.vmovq.xmm.gpr64")
    rules = []
    for lane_count in (8, 16, 32, 64):
        for vector_width in avx512_predicate_representation_bit_widths(lane_count):
            source = ValueRef.operand("source")
            emits = ()
            representation = 1 if vector_width is None else vector_width // lane_count
            if vector_width is not None:
                mask = ValueRef.temporary("mask")
                emits, _ = avx512_predicate_simd_to_mask_emits(
                    lookup,
                    lane_count=lane_count,
                    source_element_bit_width=representation,
                    source=source,
                    result=mask,
                )
                source = mask
            emits = (
                *emits,
                emit_descriptor_op(
                    descriptor=move,
                    operands={"source": source},
                    results={"dst": ValueRef.temporary("bits")},
                    result_types={"dst": Scalar("i64")},
                ),
                emit_descriptor_op(
                    descriptor=transfer,
                    operands={"input": ValueRef.temporary("bits")},
                    results={"dst": ValueRef.result("result")},
                ),
            )
            rules.extend(_bitpack_rules(lane_count, representation, emits, priority=1))
    return tuple(rules)
