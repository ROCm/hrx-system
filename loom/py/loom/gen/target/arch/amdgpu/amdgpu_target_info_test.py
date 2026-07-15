# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager

from loom.gen.target.arch.amdgpu import amdgpu_target_info
from loom.target.arch.amdgpu import target_info as amdgpu_target_info_data
from loom.target.arch.amdgpu.target_info import (
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION,
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH,
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID,
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11,
    AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
    AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU,
    AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER,
    AMDGPU_WAVEFRONT_SIZE_FLAG_32,
    AmdgpuDescriptorSetInfo,
    AmdgpuDescriptorSetVectorMemoryInfo,
    AmdgpuKernelDescriptorVgprGranules,
    AmdgpuProcessorKernelDescriptorInfo,
    processor_info,
)


@contextmanager
def _raises_value_error(match: str) -> Iterator[None]:
    try:
        yield
    except ValueError as exc:
        if re.search(match, str(exc)) is None:
            raise AssertionError(f"ValueError message {exc!s} did not match {match}") from exc
    else:
        raise AssertionError("expected ValueError")


def _descriptor_set_info() -> AmdgpuDescriptorSetInfo:
    return AmdgpuDescriptorSetInfo(
        generator_target="test",
        key="amdgpu.test.core",
        isa_xml_key="test",
        isa_architecture_name="AMDGPU Test",
        isa_architecture_id=1,
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
    )


def test_memory_cache_policy_fragments_are_data_only() -> None:
    policy_rows = amdgpu_target_info._emit_memory_cache_policy_encoding_rows()
    temporal_th = amdgpu_target_info._emit_memory_cache_policy_temporal_th()

    for source in (policy_rows, temporal_th):
        assert "typedef " not in source
        assert "#ifndef " not in source
        assert "#include " not in source
        assert "\nif " not in source
        assert "\nreturn " not in source

    assert "LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH" in policy_rows
    assert 'IREE_SVL("memory_cache_policy.gfx12_nv_scope_th")' in policy_rows
    assert "UINT32_C(0x000003ff)" in policy_rows
    assert "[LOOM_CACHE_TEMPORAL_BYPASS] = 3" in temporal_th


def test_target_info_table_source_is_data_only() -> None:
    descriptor_set_info = amdgpu_target_info.sorted_descriptor_set_infos()[0]
    descriptor_row = amdgpu_target_info._AmdgpuDescriptorSetRow(
        info=descriptor_set_info,
        sopp=amdgpu_target_info._AmdgpuSoppOpcodeRow(
            nop=0,
            delay_alu=0,
            endpgm=1,
            branch=2,
            conditional_branch_scc0=4,
            conditional_branch_scc1=5,
        ),
    )
    processor = processor_info(
        "gfx-test",
        0x001,
        descriptor_set_key=descriptor_set_info.key,
    )

    source = amdgpu_target_info._emit_tables_source(
        processors=(processor,),
        descriptor_set_rows=(descriptor_row,),
    )

    assert "typedef " not in source
    assert "#ifndef " not in source
    assert "#define LOOM_AMDGPU_DESCRIPTOR_SET_INFO" not in source
    assert "#define LOOM_AMDGPU_PROCESSOR_INFO" not in source
    assert "\nif " not in source
    assert "\nreturn " not in source
    assert ".descriptor_set = {" in source
    assert ".kernel_descriptor = {" in source


def test_memory_cache_policy_rejects_missing_encoding_row() -> None:
    rows = amdgpu_target_info_data.AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_INFOS[:-1]

    with _raises_value_error("memory cache-policy encoding table must cover every non-none encoding"):
        amdgpu_target_info._ordered_memory_cache_policy_encoding_infos(rows=rows)


def test_memory_cache_policy_rejects_unknown_descriptor_encoding() -> None:
    descriptor_set_info = AmdgpuDescriptorSetInfo(
        generator_target="test",
        key="amdgpu.test.core",
        isa_xml_key="test",
        isa_architecture_name="AMDGPU Test",
        isa_architecture_id=1,
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(cache_policy_encoding="future_encoding"),
    )

    with _raises_value_error("descriptor sets reference unknown memory cache-policy encodings"):
        amdgpu_target_info._ordered_memory_cache_policy_encoding_infos(descriptor_sets=(descriptor_set_info,))


def test_descriptor_sets_reject_none_memory_cache_policy() -> None:
    with _raises_value_error("non-none vector-memory cache-policy encoding"):
        amdgpu_target_info._validate_descriptor_sets((_descriptor_set_info(),))


def test_memory_cache_policy_rejects_incomplete_temporal_th_table() -> None:
    temporal_th = amdgpu_target_info_data.AMDGPU_VECTOR_MEMORY_CACHE_POLICY_TEMPORAL_TH
    rows = tuple(row for row in temporal_th if row[0] != "bypass")

    with _raises_value_error("temporal TH table must cover every cache temporal"):
        amdgpu_target_info._ordered_memory_cache_policy_temporal_th(rows=rows)


def test_target_info_flag_expressions_validate_known_bits() -> None:
    assert amdgpu_target_info._descriptor_set_info_flags_expr(AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING) == "LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING"
    assert amdgpu_target_info._descriptor_set_info_flags_expr(AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION) == "LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION"
    assert amdgpu_target_info._processor_info_flags_expr(AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION) == "LOOM_AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION"
    assert amdgpu_target_info._wavefront_size_flags_expr(AMDGPU_WAVEFRONT_SIZE_FLAG_32) == "LOOM_AMDGPU_WAVEFRONT_SIZE_FLAG_32"
    assert amdgpu_target_info._kernel_descriptor_abi_flags_expr(AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID) == "LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID"
    assert amdgpu_target_info._processor_scheduling_bits_expr(AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU) == "LOOM_AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU"
    assert amdgpu_target_info._processor_scheduling_bits_expr(AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER) == "LOOM_AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER"


def test_vmem_result_write_ordering_matches_processor_families() -> None:
    processors = {info.processor: info for info in amdgpu_target_info_data.AMDGPU_PROCESSOR_INFOS}
    ordered_processors = (
        "gfx900",
        "gfx942",
        "gfx950",
        "gfx1010",
        "gfx1036",
        "gfx1100",
        "gfx1172",
        "gfx9-generic",
        "gfx10-1-generic",
        "gfx10-3-generic",
        "gfx11-generic",
        "gfx9-4-generic",
    )
    unordered_processors = (
        "gfx1200",
        "gfx1201",
        "gfx1250",
        "gfx1251",
        "gfx1310",
        "gfx12-generic",
        "gfx12-5-generic",
    )

    for processor in ordered_processors:
        assert processors[processor].features.scheduling & AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER
    for processor in unordered_processors:
        assert not processors[processor].features.scheduling & AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER


def test_target_info_flag_expressions_reject_unknown_bits() -> None:
    with _raises_value_error("unknown AMDGPU descriptor-set info flags"):
        amdgpu_target_info._descriptor_set_info_flags_expr(1 << 63)
    with _raises_value_error("unknown AMDGPU processor info flags"):
        amdgpu_target_info._processor_info_flags_expr(1 << 31)
    with _raises_value_error("unknown AMDGPU wavefront-size flags"):
        amdgpu_target_info._wavefront_size_flags_expr(1 << 31)
    with _raises_value_error("unknown AMDGPU kernel descriptor ABI flags"):
        amdgpu_target_info._kernel_descriptor_abi_flags_expr(1 << 63)
    with _raises_value_error("unknown AMDGPU processor scheduling flags"):
        amdgpu_target_info._processor_scheduling_bits_expr(1 << 31)


def test_profileless_kernel_descriptor_accepts_packed_workitem_id_fact() -> None:
    processor = processor_info(
        "gfx-test",
        0x001,
        descriptor_set_key="amdgpu.test.core",
        kernel_descriptor=AmdgpuProcessorKernelDescriptorInfo(
            flags=AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID,
        ),
    )

    amdgpu_target_info._validate_processors((processor,), (_descriptor_set_info(),))


def test_profileless_kernel_descriptor_rejects_vgpr_granules() -> None:
    processor = processor_info(
        "gfx-test",
        0x001,
        descriptor_set_key="amdgpu.test.core",
        kernel_descriptor=AmdgpuProcessorKernelDescriptorInfo(
            vgpr_granules=AmdgpuKernelDescriptorVgprGranules(wave32=8, wave64=4),
        ),
    )

    with _raises_value_error("no kernel descriptor profile but has VGPR"):
        amdgpu_target_info._validate_processors((processor,), (_descriptor_set_info(),))


def test_profileless_kernel_descriptor_rejects_profile_owned_flags() -> None:
    processor = processor_info(
        "gfx-test",
        0x001,
        descriptor_set_key="amdgpu.test.core",
        kernel_descriptor=AmdgpuProcessorKernelDescriptorInfo(
            flags=AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH,
        ),
    )

    with _raises_value_error("profile-owned ABI flags"):
        amdgpu_target_info._validate_processors((processor,), (_descriptor_set_info(),))


def test_profiled_kernel_descriptor_requires_vgpr_granules() -> None:
    processor = processor_info(
        "gfx-test",
        0x001,
        descriptor_set_key="amdgpu.test.core",
        kernel_descriptor=AmdgpuProcessorKernelDescriptorInfo(
            profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11,
        ),
    )

    with _raises_value_error("descriptor profile but no VGPR encoding granules"):
        amdgpu_target_info._validate_processors((processor,), (_descriptor_set_info(),))


def test_hsaco_emission_support_requires_descriptor_set() -> None:
    processor = processor_info(
        "gfx-test",
        0x001,
        flags=AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
        kernel_descriptor=AmdgpuProcessorKernelDescriptorInfo(
            profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11,
            vgpr_granules=AmdgpuKernelDescriptorVgprGranules(wave32=8, wave64=4),
        ),
    )

    with _raises_value_error("HSACO emission support but no descriptor set"):
        amdgpu_target_info._validate_processors((processor,), (_descriptor_set_info(),))


def test_hsaco_emission_support_requires_kernel_descriptor_profile() -> None:
    processor = processor_info(
        "gfx-test",
        0x001,
        flags=AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
        descriptor_set_key="amdgpu.test.core",
    )

    with _raises_value_error("HSACO emission support but no kernel descriptor profile"):
        amdgpu_target_info._validate_processors((processor,), (_descriptor_set_info(),))
