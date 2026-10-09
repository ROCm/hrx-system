# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from collections import Counter

from loom.target.arch.x86.descriptors import X86_AVX512_BF16_DESCRIPTOR_SET
from loom.target.arch.x86.feature_bits import (
    FEATURE_AVX512_BF16,
    FEATURE_AVX512_VL,
)
from loom.target.arch.x86.vector_encoding import VectorEncodingBehavior
from loom.target.low_descriptors import (
    Constraint,
    ConstraintKind,
    Descriptor,
    DescriptorFlag,
    OperandAddressMapKind,
)


def _conversion_descriptors() -> tuple[Descriptor, ...]:
    return tuple(
        descriptor
        for descriptor in X86_AVX512_BF16_DESCRIPTOR_SET.descriptors
        if descriptor.mnemonic in ("vcvtneps2bf16", "vcvtne2ps2bf16")
    )


def _vector_bit_width(descriptor: Descriptor) -> int:
    return {0: 128, 1: 256, 2: 512}[descriptor.encoding_id >> 14]


def _masking(descriptor: Descriptor) -> str:
    if ".merge." in descriptor.key:
        return "merge"
    if ".zero." in descriptor.key:
        return "zero"
    return "none"


def _memory_source(descriptor: Descriptor) -> str:
    if ".broadcast." in descriptor.key:
        return "broadcast32"
    if ".load." in descriptor.key:
        return "full"
    return "none"


def test_conversion_family_covers_every_evex_form() -> None:
    actual = Counter(
        (
            1 if descriptor.mnemonic == "vcvtneps2bf16" else 2,
            _vector_bit_width(descriptor),
            _masking(descriptor),
            _memory_source(descriptor),
            ".indexed." in descriptor.key,
        )
        for descriptor in _conversion_descriptors()
    )
    expected = Counter(
        (
            source_count,
            vector_bit_width,
            masking,
            memory_source,
            indexed,
        )
        for source_count in (1, 2)
        for vector_bit_width in (128, 256, 512)
        for masking in ("none", "merge", "zero")
        for memory_source in ("none", "full", "broadcast32")
        for indexed in ((False, True) if memory_source != "none" else (False,))
    )
    assert actual == expected
    assert sum(actual.values()) == 90


def test_conversion_encodings_match_the_avx512_bf16_map() -> None:
    for descriptor in _conversion_descriptors():
        vector_bit_width = _vector_bit_width(descriptor)
        opcode_prefix = 0x2E00 if descriptor.mnemonic == "vcvtne2ps2bf16" else 0x2A00
        vector_length = {128: 0, 256: 1, 512: 2}[vector_bit_width]
        assert descriptor.encoding_id == opcode_prefix | (vector_length << 14) | 0x72

        memory_source = _memory_source(descriptor)
        masking = _masking(descriptor)
        behavior = VectorEncodingBehavior((descriptor.encoding_format_id >> 12) & 7)
        if memory_source != "none":
            expected_behavior = (
                VectorEncodingBehavior.EVEX_MASK_LOAD
                if masking != "none"
                else VectorEncodingBehavior.LOAD
            )
        else:
            expected_behavior = (
                VectorEncodingBehavior.EVEX_MASK
                if masking != "none"
                else VectorEncodingBehavior.REGISTERS
            )
        assert behavior is expected_behavior
        assert bool(descriptor.encoding_format_id & 8) == (masking == "zero")
        assert bool(descriptor.encoding_format_id & (8 << 4)) == (
            memory_source == "full"
        )
        assert bool(descriptor.encoding_format_id & (8 << 8)) == (
            memory_source == "broadcast32"
        )


def test_conversion_semantic_tags_distinguish_single_and_packed_results() -> None:
    for descriptor in _conversion_descriptors():
        vector_bit_width = _vector_bit_width(descriptor)
        if descriptor.mnemonic == "vcvtne2ps2bf16":
            assert descriptor.semantic_tag == (
                f"float.truncate.concat.f32.bf16x{vector_bit_width // 16}"
            )
        else:
            assert descriptor.semantic_tag == (
                f"float.truncate.f32.bf16x{vector_bit_width // 32}"
            )


def test_conversion_operands_preserve_mask_and_address_contracts() -> None:
    for descriptor in _conversion_descriptors():
        operands = descriptor.operands
        input_operands = operands[1:]
        masking = _masking(descriptor)
        memory_source = _memory_source(descriptor)
        indexed = ".indexed." in descriptor.key

        mask_operands = tuple(
            operand for operand in input_operands if operand.field_name == "mask"
        )
        if masking == "none":
            assert not mask_operands
        else:
            assert len(mask_operands) == 1
            assert mask_operands[0].address_map_kind is OperandAddressMapKind.LOW_SUBSET
            assert mask_operands[0].addressable_unit_count == 7

        passthrough_indices = tuple(
            index
            for index, operand in enumerate(operands)
            if operand.field_name == "passthrough"
        )
        if masking == "merge":
            assert passthrough_indices == (len(operands) - 1,)
            assert descriptor.constraints == (
                Constraint(ConstraintKind.TIED, 0, passthrough_indices[0]),
                Constraint(ConstraintKind.DESTRUCTIVE, 0, passthrough_indices[0]),
            )
        else:
            assert not passthrough_indices
            assert not descriptor.constraints

        if memory_source == "none":
            assert not descriptor.effects
            assert descriptor.flags == (DescriptorFlag.DEAD_REMOVABLE,)
            assert not descriptor.immediates
            continue

        fields = tuple(operand.field_name for operand in input_operands)
        base_index = fields.index("base")
        if indexed:
            assert fields[base_index + 1] == "index"
        else:
            assert "index" not in fields
        assert tuple(immediate.field_name for immediate in descriptor.immediates) == (
            ("disp32", "scale") if indexed else ("disp32",)
        )
        assert len(descriptor.effects) == 1
        assert descriptor.effects[0].width_bits == (
            32 if memory_source == "broadcast32" else _vector_bit_width(descriptor)
        )
        assert descriptor.flags == (DescriptorFlag.SIDE_EFFECTING,)


def test_conversion_features_and_asm_surface_are_width_exact() -> None:
    asm_mnemonics = set()
    for descriptor in _conversion_descriptors():
        vector_bit_width = _vector_bit_width(descriptor)
        assert descriptor.feature_mask_words == (
            FEATURE_AVX512_BF16 | (FEATURE_AVX512_VL if vector_bit_width < 512 else 0),
        )
        assert len(descriptor.asm_forms) == 1
        asm_mnemonic = descriptor.asm_forms[0].mnemonic
        assert asm_mnemonic is not None
        assert asm_mnemonic.startswith("avx512_bf16.vcvtne")
        assert asm_mnemonic not in asm_mnemonics
        asm_mnemonics.add(asm_mnemonic)


def test_conversion_family_extends_the_existing_bf16_dot_surface() -> None:
    dot_descriptors = tuple(
        descriptor
        for descriptor in X86_AVX512_BF16_DESCRIPTOR_SET.descriptors
        if descriptor.mnemonic == "vdpbf16ps"
    )
    assert tuple(descriptor.key for descriptor in dot_descriptors) == (
        "x86.avx512_bf16.vdpbf16ps.xmm",
        "x86.avx512_bf16.vdpbf16ps.ymm",
        "x86.avx512_bf16.vdpbf16ps.zmm",
    )
    assert all(
        descriptor.asm_forms[0].mnemonic is not None
        and descriptor.asm_forms[0].mnemonic.startswith("avx512_bf16.")
        for descriptor in dot_descriptors
    )
