# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import replace

import pytest

from loom.gen.target.arch.amdgpu.lower.candidates import (
    amdgpu_arithmetic_candidates,
    amdgpu_async_gather_candidates,
    amdgpu_atomic_candidates,
    amdgpu_compare_candidates,
    amdgpu_memory_candidates,
)
from loom.gen.target.arch.amdgpu.lower.candidates.validation import (
    dense_candidate_ranges,
)

_ATOMIC_HEADER = "loom/target/arch/amdgpu/lower/candidates/atomic_candidates.h"
_ARITHMETIC_HEADER = "loom/target/arch/amdgpu/lower/candidates/arithmetic_candidates.h"
_ASYNC_GATHER_HEADER = "loom/target/arch/amdgpu/lower/candidates/async_gather_candidates.h"
_COMPARE_HEADER = "loom/target/arch/amdgpu/lower/candidates/compare_candidates.h"


def test_arithmetic_generator_emits_fma_mix_data_source_only() -> None:
    source = amdgpu_arithmetic_candidates._emit_source(public_header=_ARITHMETIC_HEADER)

    assert f'#include "{_ARITHMETIC_HEADER}"' in source
    assert "typedef " not in source
    assert "#ifndef " not in source
    assert "\nif " not in source
    assert "\nreturn " not in source
    assert "kLoomAmdgpuFmaMixF32DescriptorRefs" in source
    assert "kLoomAmdgpuFmaMixF32Src2LiteralDescriptorRefs" in source
    assert "kLoomAmdgpuFmaMixloF16Src2LiteralDescriptorRefs" in source
    assert "kLoomAmdgpuFmaMixhiF16Src2LiteralDescriptorRefs" in source
    assert "kLoomAmdgpuMadMixhiF16DescriptorRefs" in source
    assert "kLoomAmdgpuPackedFmafF16DescriptorCandidates" in source
    assert "kLoomAmdgpuPackedFmaiUnsignedPreferenceDescriptorCandidates" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_FMA_MIX_F32_F32_F16LO_F16HI" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_MAD_MIXHI_F16_F32_F32_F32" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_FMAC_F16" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_PK_MAD_U16" in source
    assert ".source_permutation = {2, 0, 1}" in source
    assert ".flags = LOOM_AMDGPU_PACKED_TERNARY_FLAG_TIED_ACCUMULATOR" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_V_FMA_MIX_F32_F32_F32_F32" not in source


def test_arithmetic_generator_covers_fma_mix_descriptor_lattice() -> None:
    assert len(amdgpu_arithmetic_candidates._SOURCE_KINDS) == 3
    assert len(amdgpu_arithmetic_candidates._FMA_MIX_DESCRIPTOR_CUBES) == 6
    assert len(amdgpu_arithmetic_candidates._FMA_MIX_SRC2_LITERAL_TABLES) == 3
    assert len(amdgpu_arithmetic_candidates._fma_mix_source_combinations(include_all_f32=True)) == 27
    assert len(amdgpu_arithmetic_candidates._fma_mix_source_combinations(include_all_f32=False)) == 26
    f32_table = amdgpu_arithmetic_candidates._FMA_MIX_SRC2_LITERAL_TABLES[0]
    mixlo_table = amdgpu_arithmetic_candidates._FMA_MIX_SRC2_LITERAL_TABLES[1]
    assert amdgpu_arithmetic_candidates._fma_mix_src2_literal_descriptor_key(f32_table, "f16lo", "f32") == "amdgpu.v_fma_mix_f32.f16lo_f32_f32.src2_lit"
    assert amdgpu_arithmetic_candidates._fma_mix_src2_literal_descriptor_key(f32_table, "f32", "f32") is None
    assert amdgpu_arithmetic_candidates._fma_mix_src2_literal_descriptor_key(mixlo_table, "f32", "f32") == "amdgpu.v_fma_mixlo_f16.f32_f32_f32.src2_lit"


def test_arithmetic_generator_covers_packed_ternary_descriptor_candidates() -> None:
    arrays = amdgpu_arithmetic_candidates._PACKED_TERNARY_DESCRIPTOR_CANDIDATE_ARRAYS

    assert [len(array.candidates) for array in arrays] == [2, 1, 2, 2]
    assert arrays[0].candidates[0].descriptor_key == "amdgpu.v_pk_fmac_f16"
    assert arrays[0].candidates[0].source_permutation == (2, 0, 1)
    assert arrays[0].candidates[0].flags == ("LOOM_AMDGPU_PACKED_TERNARY_FLAG_TIED_ACCUMULATOR",)
    assert arrays[1].candidates[0].descriptor_key == "amdgpu.v_pk_fma_f32"
    assert arrays[1].candidates[0].packet_unit_count == 2
    assert [candidate.descriptor_key for candidate in arrays[2].candidates] == [
        "amdgpu.v_pk_mad_i16",
        "amdgpu.v_pk_mad_u16",
    ]
    assert [candidate.descriptor_key for candidate in arrays[3].candidates] == [
        "amdgpu.v_pk_mad_u16",
        "amdgpu.v_pk_mad_i16",
    ]


def test_arithmetic_generator_rejects_missing_fma_mix_descriptor_ref() -> None:
    descriptor_ref_keys = tuple(key for key in amdgpu_arithmetic_candidates.amdgpu_descriptor_ref_keys() if key != "amdgpu.v_fma_mix_f32.f32_f32_f16lo")
    original_descriptor_ref_keys = amdgpu_arithmetic_candidates.amdgpu_descriptor_ref_keys
    try:
        amdgpu_arithmetic_candidates.amdgpu_descriptor_ref_keys = lambda: descriptor_ref_keys
        with pytest.raises(
            ValueError,
            match=(
                r"AMDGPU FMA-mix descriptor candidate "
                r"kLoomAmdgpuFmaMixF32DescriptorRefs f32 f32 f16lo "
                r"requires missing descriptor refs: "
                r"amdgpu\.v_fma_mix_f32\.f32_f32_f16lo"
            ),
        ):
            amdgpu_arithmetic_candidates._emit_source(public_header=_ARITHMETIC_HEADER)
    finally:
        amdgpu_arithmetic_candidates.amdgpu_descriptor_ref_keys = original_descriptor_ref_keys


def test_arithmetic_generator_rejects_missing_packed_ternary_descriptor_ref() -> None:
    descriptor_ref_keys = tuple(key for key in amdgpu_arithmetic_candidates.amdgpu_descriptor_ref_keys() if key != "amdgpu.v_pk_fmac_f16")
    original_descriptor_ref_keys = amdgpu_arithmetic_candidates.amdgpu_descriptor_ref_keys
    try:
        amdgpu_arithmetic_candidates.amdgpu_descriptor_ref_keys = lambda: descriptor_ref_keys
        with pytest.raises(
            ValueError,
            match=(
                r"AMDGPU packed ternary descriptor candidate "
                r"kLoomAmdgpuPackedFmafF16DescriptorCandidates "
                r"requires missing descriptor refs: amdgpu\.v_pk_fmac_f16"
            ),
        ):
            amdgpu_arithmetic_candidates._emit_source(public_header=_ARITHMETIC_HEADER)
    finally:
        amdgpu_arithmetic_candidates.amdgpu_descriptor_ref_keys = original_descriptor_ref_keys


def test_async_gather_generator_emits_data_source_only() -> None:
    source = amdgpu_async_gather_candidates._emit_source(public_header=_ASYNC_GATHER_HEADER)

    assert f'#include "{_ASYNC_GATHER_HEADER}"' in source
    assert "typedef " not in source
    assert "#ifndef " not in source
    assert "\nif " not in source
    assert "\nreturn " not in source
    assert "kLoomAmdgpuAsyncGatherDescriptorCandidates[]" in source
    assert "kLoomAmdgpuAsyncGatherDescriptorCandidateCount" in source
    assert ".packet_byte_count" in source
    assert ".descriptor_ref" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_LDS_DWORD_SADDR" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_LDS_DWORDX3_SADDR" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_LDS_DWORDX4_SADDR" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_LDS_UBYTE_SADDR" not in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_LDS_USHORT_SADDR" not in source


def test_async_gather_generator_covers_packet_widths() -> None:
    candidates = amdgpu_async_gather_candidates.amdgpu_async_gather_descriptor_candidates()

    assert [(candidate.packet_byte_count, candidate.descriptor_key) for candidate in candidates] == [
        (4, "amdgpu.global_load_lds_dword_saddr"),
        (12, "amdgpu.global_load_lds_dwordx3_saddr"),
        (16, "amdgpu.global_load_lds_dwordx4_saddr"),
    ]


def test_async_gather_generator_rejects_missing_descriptor_ref() -> None:
    candidates = amdgpu_async_gather_candidates.amdgpu_async_gather_descriptor_candidates()
    bad_candidate = replace(candidates[0], descriptor_key="amdgpu.missing")

    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU async gather descriptor candidate table requires missing "
            r"descriptor refs: amdgpu\.missing"
        ),
    ):
        amdgpu_async_gather_candidates._ordered_candidates((bad_candidate,))


def test_async_gather_generator_rejects_conflicting_packet_widths() -> None:
    candidates = amdgpu_async_gather_candidates.amdgpu_async_gather_descriptor_candidates()
    conflicting_candidate = replace(candidates[1], packet_byte_count=candidates[0].packet_byte_count)

    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU async gather descriptor candidate table has conflicting "
            r"rows for packet byte count 4"
        ),
    ):
        amdgpu_async_gather_candidates._ordered_candidates((candidates[0], conflicting_candidate))


def test_atomic_generator_emits_data_source_only() -> None:
    source = amdgpu_atomic_candidates._emit_source(public_header=_ATOMIC_HEADER)

    assert f'#include "{_ATOMIC_HEADER}"' in source
    assert "typedef " not in source
    assert "#ifndef " not in source
    assert "\nif " not in source
    assert "\nreturn " not in source
    assert "kLoomAmdgpuAtomicDescriptorCandidates[]" in source
    assert "kLoomAmdgpuAtomicDescriptorCandidateCount" in source
    assert "kLoomAmdgpuAtomicDescriptorCandidateRanges" in source
    assert ".memory_space" not in source
    assert ".address_form" not in source
    assert ".operation_kind" not in source
    assert ".atomic_kind" not in source
    assert ".value_kind" in source
    assert ".descriptor_ref" in source


def test_atomic_generator_builds_contiguous_candidate_ranges() -> None:
    candidates = amdgpu_atomic_candidates.amdgpu_atomic_descriptor_candidates()
    ranges = amdgpu_atomic_candidates._candidate_ranges(candidates)

    assert ranges
    assert len(ranges) < len(candidates)
    covered_candidate_count = sum(candidate_count for _, _, candidate_count in ranges)
    assert covered_candidate_count == len(candidates)


def test_atomic_generator_rejects_missing_descriptor_ref() -> None:
    candidates = amdgpu_atomic_candidates.amdgpu_atomic_descriptor_candidates()
    bad_candidate = replace(candidates[0], descriptor_key="amdgpu.missing")

    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU atomic descriptor candidate table requires missing "
            r"descriptor refs: amdgpu\.missing"
        ),
    ):
        amdgpu_atomic_candidates._candidate_ranges((bad_candidate,))


def test_atomic_generator_rejects_noncontiguous_candidate_ranges() -> None:
    candidates = amdgpu_atomic_candidates.amdgpu_atomic_descriptor_candidates()

    with pytest.raises(
        ValueError,
        match=("AMDGPU atomic descriptor candidate table must be contiguous by range key"),
    ):
        amdgpu_atomic_candidates._candidate_ranges((candidates[0], candidates[-1], candidates[0]))


def test_candidate_range_validation_rejects_uint16_overflow() -> None:
    with pytest.raises(ValueError, match="candidate index 65536"):
        dense_candidate_ranges(
            range(65537),
            lambda value: value,
            owner="AMDGPU test candidates",
        )


def test_memory_generator_emits_data_fragments_only() -> None:
    candidate_rows = amdgpu_memory_candidates._emit_candidate_rows()
    candidate_ranges = amdgpu_memory_candidates._emit_candidate_ranges()

    for source in (candidate_rows, candidate_ranges):
        assert "typedef " not in source
        assert "#ifndef " not in source
        assert "#include " not in source
        assert "\nif " not in source
        assert "\nreturn " not in source
    assert "LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE(" in candidate_rows
    assert "LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_RANGE(" in candidate_ranges
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_LOAD_B64" in candidate_rows
    assert "LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE" in candidate_ranges


def test_memory_generator_builds_dense_candidate_ranges() -> None:
    memory_candidates = amdgpu_memory_candidates.amdgpu_memory_descriptor_candidates()
    candidates = amdgpu_memory_candidates._ordered_candidates(memory_candidates)
    ranges = amdgpu_memory_candidates._candidate_ranges(candidates)

    assert ranges
    assert len(ranges) < len(candidates)
    covered_candidate_count = sum(candidate_count for _, _, candidate_count in ranges)
    assert covered_candidate_count == len(candidates)
    packed_key = amdgpu_memory_candidates._range_key_packed_value
    packed_keys = [packed_key(key) for key, _, _ in ranges]
    assert packed_keys == sorted(packed_keys)


def test_memory_generator_preserves_same_key_fallback_order() -> None:
    memory_candidates = amdgpu_memory_candidates.amdgpu_memory_descriptor_candidates()
    candidates = amdgpu_memory_candidates._ordered_candidates(memory_candidates)
    domain = amdgpu_memory_candidates.AmdgpuMemoryDescriptorDomain.BUFFER_RESOURCE
    address_form = amdgpu_memory_candidates.AmdgpuMemoryAddressForm.DEFAULT
    operation_kind = amdgpu_memory_candidates.AmdgpuMemoryOperationKind.LOAD
    register_class = amdgpu_memory_candidates.AmdgpuMemoryPayloadRegisterClass.VGPR
    payload_format = amdgpu_memory_candidates.AmdgpuMemoryPayloadFormat.GENERIC

    buffer_load8_candidates = [
        candidate.descriptor_key
        for candidate in candidates
        if candidate.domain == domain
        and candidate.address_form == address_form
        and candidate.operation_kind == operation_kind
        and candidate.packet_byte_count == 8
        and candidate.payload_register_class == register_class
        and candidate.payload_format == payload_format
        and candidate.payload_register_count == 2
    ]

    assert buffer_load8_candidates == [
        "amdgpu.buffer_load_b64",
        "amdgpu.buffer_load_dwordx2",
    ]


def test_memory_generator_covers_96_bit_source_memory_packets() -> None:
    memory_candidates = amdgpu_memory_candidates.amdgpu_memory_descriptor_candidates()
    candidates = amdgpu_memory_candidates._ordered_candidates(memory_candidates)
    register_class = amdgpu_memory_candidates.AmdgpuMemoryPayloadRegisterClass.VGPR
    payload_format = amdgpu_memory_candidates.AmdgpuMemoryPayloadFormat.GENERIC
    operations = (
        amdgpu_memory_candidates.AmdgpuMemoryOperationKind.LOAD,
        amdgpu_memory_candidates.AmdgpuMemoryOperationKind.STORE,
    )
    domains = (
        amdgpu_memory_candidates.AmdgpuMemoryDescriptorDomain.BUFFER_RESOURCE,
        amdgpu_memory_candidates.AmdgpuMemoryDescriptorDomain.GLOBAL_SADDR,
        amdgpu_memory_candidates.AmdgpuMemoryDescriptorDomain.GLOBAL_FLAT,
        amdgpu_memory_candidates.AmdgpuMemoryDescriptorDomain.LDS,
        amdgpu_memory_candidates.AmdgpuMemoryDescriptorDomain.SCRATCH,
    )

    b96_candidates = {
        (candidate.domain, candidate.operation_kind): candidate.descriptor_key
        for candidate in candidates
        if (
            candidate.packet_byte_count,
            candidate.payload_register_class,
            candidate.payload_format,
            candidate.payload_register_count,
        )
        == (12, register_class, payload_format, 3)
    }

    for domain in domains:
        for operation_kind in operations:
            assert (domain, operation_kind) in b96_candidates


def test_memory_generator_rejects_missing_descriptor_ref() -> None:
    candidates = amdgpu_memory_candidates.amdgpu_memory_descriptor_candidates()
    bad_candidate = replace(candidates[0], descriptor_key="amdgpu.missing")

    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU memory descriptor candidate table requires missing "
            r"descriptor refs: amdgpu\.missing"
        ),
    ):
        amdgpu_memory_candidates._ordered_candidates((bad_candidate,))


def test_memory_generator_rejects_out_of_range_packet_shape() -> None:
    candidates = amdgpu_memory_candidates.amdgpu_memory_descriptor_candidates()
    bad_candidate = replace(candidates[0], packet_byte_count=32)

    with pytest.raises(
        ValueError,
        match=(
            r"AMDGPU memory descriptor candidate amdgpu\.buffer_load_i8 "
            r"has packet_byte_count 32"
        ),
    ):
        amdgpu_memory_candidates._ordered_candidates((bad_candidate,))


def test_compare_generator_emits_data_source_only() -> None:
    source = amdgpu_compare_candidates._emit_source(public_header=_COMPARE_HEADER)

    assert f'#include "{_COMPARE_HEADER}"' in source
    assert "typedef " not in source
    assert "#ifndef " not in source
    assert "\nif " not in source
    assert "\nreturn " not in source
    assert "kLoomAmdgpuVectorCmpiCompareDescriptorCandidates" in source
    assert "kLoomAmdgpuScalarCmpfCompareDescriptorCandidates" in source
    assert "kLoomAmdgpuVectorCmpfCompareDescriptorCandidates" in source
    assert ".op_kind" not in source
    assert ".predicate" not in source
    assert "[LOOM_VECTOR_CMPI_PREDICATE_EQ]" in source
    assert "[LOOM_SCALAR_CMPF_PREDICATE_OLT]" in source
    assert "[LOOM_VECTOR_CMPF_PREDICATE_OLT]" in source


def test_compare_generator_covers_predicate_rows() -> None:
    candidates = amdgpu_compare_candidates._compare_candidates()

    assert candidates
    assert any(family.source_op_name == "vector.cmpi" and predicate == "eq" for family, predicate, _ in candidates)
    assert any(family.source_op_name == "vector.cmpf" and predicate == "olt" for family, predicate, _ in candidates)


def test_compare_generator_rejects_missing_descriptor_ref() -> None:
    descriptor_ref_keys = tuple(key for key in amdgpu_compare_candidates.amdgpu_descriptor_ref_keys() if key != "amdgpu.v_cmp_eq_i32")
    original_descriptor_ref_keys = amdgpu_compare_candidates.amdgpu_descriptor_ref_keys
    try:
        amdgpu_compare_candidates.amdgpu_descriptor_ref_keys = lambda: descriptor_ref_keys
        with pytest.raises(
            ValueError,
            match=(
                r"AMDGPU compare candidate vector\.cmpi eq requires missing "
                r"descriptor refs: amdgpu\.v_cmp_eq_i32"
            ),
        ):
            amdgpu_compare_candidates._compare_candidates()
    finally:
        amdgpu_compare_candidates.amdgpu_descriptor_ref_keys = original_descriptor_ref_keys
