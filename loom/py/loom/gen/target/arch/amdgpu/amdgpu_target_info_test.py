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
from loom.target.arch.amdgpu.lds_bank_service import (
    AMDGPU_LDS_BANK_SERVICE_MODEL_INFOS,
    validate_amdgpu_lds_bank_service_model_coverage,
)
from loom.target.arch.amdgpu.target_info import (
    AMDGPU_BUFFER_RESOURCE_INFO_BASE48_UNIFIED,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
    AMDGPU_INSTRUCTION_CONSTRAINT_KNOWN_BITS,
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH,
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID,
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11,
    AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY,
    AMDGPU_PROCESSOR_INFO_FLAG_CLUSTER_LAUNCH_STATE,
    AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
    AMDGPU_SOPP_OPCODE_INFO_CDNA,
    AMDGPU_SOPP_OPCODE_INFO_RDNA,
    AMDGPU_TARGET_ID_FEATURE_SUPPORT_KNOWN_FLAGS,
    AmdgpuDescriptorSetInfo,
    AmdgpuDescriptorSetIsaInfo,
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
        isa_infos=(
            AmdgpuDescriptorSetIsaInfo(
                isa_xml_key="test",
                isa_architecture_name="AMDGPU Test",
                isa_architecture_id=1,
                sopp_opcodes=AMDGPU_SOPP_OPCODE_INFO_RDNA,
            ),
        ),
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
        buffer_resource=AMDGPU_BUFFER_RESOURCE_INFO_BASE48_UNIFIED,
    )


def test_memory_cache_policy_fragments_are_data_only() -> None:
    policy_rows = amdgpu_target_info._emit_memory_cache_policy_encoding_rows(
        amdgpu_target_info_data.sorted_descriptor_set_infos(),
    )
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


def test_lds_bank_service_fragment_is_data_only() -> None:
    processors = amdgpu_target_info_data.sorted_processor_infos()
    targets = amdgpu_target_info_data.sorted_target_infos()
    source = amdgpu_target_info._emit_lds_bank_service_model_rows(
        amdgpu_target_info_data.amdgpu_lds_bank_service_model_sets(
            processors,
            targets,
        ),
    )

    assert "typedef " not in source
    assert "#ifndef " not in source
    assert "#include " not in source
    assert "\nif " not in source
    assert "\nreturn " not in source
    assert "kAmdgpuLdsBankServiceModelSet0Bindings[]" in source
    assert "kAmdgpuLdsBankServiceModelSets[]" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B128" in source
    assert "LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B128" in source
    assert 'IREE_SVL("amdgpu.lds.wave32.b128.quad-phases.read.count-each")' in source


def test_lds_bank_service_rejects_unselected_model_data() -> None:
    with _raises_value_error("models are not selected by a target row"):
        validate_amdgpu_lds_bank_service_model_coverage(
            (),
            AMDGPU_LDS_BANK_SERVICE_MODEL_INFOS,
        )


def test_target_info_table_source_is_data_only() -> None:
    descriptor_set_info = amdgpu_target_info.sorted_descriptor_set_infos()[0]
    descriptor_row = amdgpu_target_info._AmdgpuDescriptorSetRow(
        info=descriptor_set_info,
        sopp=AMDGPU_SOPP_OPCODE_INFO_CDNA,
    )
    source = amdgpu_target_info._emit_tables_source(
        processors=amdgpu_target_info.sorted_processor_infos(),
        targets=amdgpu_target_info.sorted_target_infos(),
        descriptor_set_rows=(descriptor_row,),
    )

    assert "typedef " not in source
    assert "#ifndef " not in source
    assert "#define LOOM_AMDGPU_DESCRIPTOR_SET_INFO" not in source
    assert "#define LOOM_AMDGPU_PROCESSOR_INFO" not in source
    assert "\nif " not in source
    assert "\nreturn " not in source
    assert ".descriptor_set = {" in source
    assert ".generic_version = UINT32_C(0)," in source
    assert ".target_id = {" in source
    assert (".supported_features = LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_NONE,") in source
    assert ".asic_revisions = {" not in source
    assert ".generic_code_object = {" in source
    assert ".processor_ordinal = LOOM_AMDGPU_PROCESSOR_ORDINAL_NONE," in source
    assert ".introduction_version = UINT16_C(0)," in source
    assert ".kernel_descriptor = {" in source
    assert ".kernel_entry = {" in source
    assert ".instructions = {" in source
    assert (".lds_bank_service_model_set_ordinal = LOOM_AMDGPU_LDS_BANK_SERVICE_MODEL_SET_ORDINAL_NONE,") in source
    assert "loom_amdgpu_target_info_target_infos[]" in source
    assert "loom_amdgpu_target_info_physical_target_infos[]" in source


def test_target_info_headers_keep_address_stable_target_rows_public() -> None:
    public_header = amdgpu_target_info._emit_header(amdgpu_target_info.sorted_descriptor_set_infos())
    private_header = amdgpu_target_info._emit_tables_header()

    assert "loom_amdgpu_target_info_target_infos[]" in public_header
    assert "loom_amdgpu_target_info_target_infos[]" not in private_header
    assert "loom_amdgpu_target_info_target_info_count" in private_header


def test_overlay_and_physical_targets_generate_data_rows_only() -> None:
    descriptor_set_info = amdgpu_target_info.sorted_descriptor_set_infos()[0]
    descriptor_row = amdgpu_target_info._AmdgpuDescriptorSetRow(
        info=descriptor_set_info,
        sopp=AMDGPU_SOPP_OPCODE_INFO_CDNA,
    )

    source = amdgpu_target_info._emit_tables_source(
        processors=amdgpu_target_info.sorted_processor_infos(),
        targets=amdgpu_target_info.sorted_target_infos(),
        descriptor_set_rows=(descriptor_row,),
    )

    assert "loom_amdgpu_target_info_gfx1250_a0_kernel_metadata_extensions[]" in source
    assert '.name = IREE_SVL("gfx1250-a0"),' in source
    assert '.key = IREE_SVL(".gfx1250_revision"),' in source
    assert ".asic_revision = UINT32_C(0)," in source
    assert ".target_kind = UINT32_C(24)," in source
    assert "\nif " not in source
    assert "\nreturn " not in source


def test_physical_targets_are_limited_to_supported_compiler_targets() -> None:
    gfx1151 = next(target for target in amdgpu_target_info.sorted_target_infos() if target.target == "gfx1151")

    source = "\n".join(amdgpu_target_info._emit_physical_target_rows((gfx1151,)))

    assert ".asic_revision" not in source
    assert ".target_kind" not in source


def test_generic_code_object_fields_cover_canonical_map() -> None:
    processors = amdgpu_target_info.sorted_processor_infos()
    processor_ordinals = {info.processor: ordinal for ordinal, info in enumerate(processors)}
    processors_by_name = {info.processor: info for info in processors}

    for compatibility in amdgpu_target_info_data.AMDGPU_EXACT_TARGET_INFOS:
        exact_processor = processors_by_name[compatibility.exact_processor]
        generic_processor_ordinal, introduction_version = amdgpu_target_info._processor_generic_code_object_fields(exact_processor, processor_ordinals)
        if compatibility.generic_introduction_version != 0:
            assert generic_processor_ordinal == processor_ordinals[compatibility.code_object_processor]
            assert introduction_version == compatibility.generic_introduction_version
        else:
            assert generic_processor_ordinal is None
            assert introduction_version == 0


def test_memory_cache_policy_rejects_missing_encoding_row() -> None:
    rows = amdgpu_target_info_data.AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_INFOS[:-1]

    with _raises_value_error("memory cache-policy encoding table must cover every non-none encoding"):
        amdgpu_target_info._ordered_memory_cache_policy_encoding_infos(
            amdgpu_target_info_data.sorted_descriptor_set_infos(),
            rows=rows,
        )


def test_memory_cache_policy_rejects_unknown_descriptor_encoding() -> None:
    descriptor_set_info = AmdgpuDescriptorSetInfo(
        generator_target="test",
        key="amdgpu.test.core",
        isa_infos=(
            AmdgpuDescriptorSetIsaInfo(
                isa_xml_key="test",
                isa_architecture_name="AMDGPU Test",
                isa_architecture_id=1,
                sopp_opcodes=AMDGPU_SOPP_OPCODE_INFO_RDNA,
            ),
        ),
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(cache_policy_encoding="future_encoding"),
    )

    with _raises_value_error("descriptor sets reference unknown memory cache-policy encodings"):
        amdgpu_target_info._ordered_memory_cache_policy_encoding_infos(descriptor_sets=(descriptor_set_info,))


def test_descriptor_sets_reject_none_memory_cache_policy() -> None:
    with _raises_value_error("non-none vector-memory cache-policy encoding"):
        amdgpu_target_info._validate_descriptor_sets((_descriptor_set_info(),))


def test_generic_descriptor_members_may_share_an_isa_contract() -> None:
    descriptor_sets_by_key = {info.key: info for info in amdgpu_target_info_data.AMDGPU_DESCRIPTOR_SET_INFOS}
    descriptor_sets = tuple(
        sorted(
            (
                descriptor_sets_by_key["amdgpu.gfx12_5.generic.core"],
                descriptor_sets_by_key["amdgpu.rdna4.gfx1251.core"],
                descriptor_sets_by_key["amdgpu.rdna4.gfx125x.core"],
            ),
            key=lambda info: info.key,
        )
    )

    amdgpu_target_info._validate_descriptor_sets(descriptor_sets)


def test_memory_cache_policy_rejects_incomplete_temporal_th_table() -> None:
    temporal_th = amdgpu_target_info_data.AMDGPU_VECTOR_MEMORY_CACHE_POLICY_TEMPORAL_TH
    rows = tuple(row for row in temporal_th if row[0] != "bypass")

    with _raises_value_error("temporal TH table must cover every cache temporal"):
        amdgpu_target_info._ordered_memory_cache_policy_temporal_th(rows=rows)


def test_target_info_flag_expressions_cover_every_known_bit() -> None:
    cases = (
        (
            amdgpu_target_info._descriptor_set_info_flags_expr,
            amdgpu_target_info_data.AMDGPU_DESCRIPTOR_SET_INFO_KNOWN_FLAGS,
            amdgpu_target_info._DESCRIPTOR_SET_INFO_FLAG_EXPRS,
        ),
        (
            amdgpu_target_info._processor_info_flags_expr,
            amdgpu_target_info_data.AMDGPU_PROCESSOR_INFO_KNOWN_FLAGS,
            amdgpu_target_info._PROCESSOR_INFO_FLAG_EXPRS,
        ),
        (
            amdgpu_target_info._instruction_constraint_bits_expr,
            AMDGPU_INSTRUCTION_CONSTRAINT_KNOWN_BITS,
            amdgpu_target_info._INSTRUCTION_CONSTRAINT_BIT_EXPRS,
        ),
        (
            amdgpu_target_info._target_id_feature_support_flags_expr,
            AMDGPU_TARGET_ID_FEATURE_SUPPORT_KNOWN_FLAGS,
            amdgpu_target_info._TARGET_ID_FEATURE_SUPPORT_FLAG_EXPRS,
        ),
        (
            amdgpu_target_info._wavefront_size_flags_expr,
            amdgpu_target_info_data.AMDGPU_WAVEFRONT_SIZE_KNOWN_FLAGS,
            amdgpu_target_info._WAVEFRONT_SIZE_FLAG_EXPRS,
        ),
        (
            amdgpu_target_info._kernel_descriptor_abi_flags_expr,
            amdgpu_target_info_data.AMDGPU_KERNEL_DESCRIPTOR_ABI_KNOWN_FLAGS,
            amdgpu_target_info._KERNEL_DESCRIPTOR_ABI_FLAG_EXPRS,
        ),
        (
            amdgpu_target_info._processor_scheduling_bits_expr,
            amdgpu_target_info_data.AMDGPU_PROCESSOR_SCHEDULING_KNOWN_BITS,
            amdgpu_target_info._PROCESSOR_SCHEDULING_BIT_EXPRS,
        ),
    )
    for expression_function, known_bits, expression_rows in cases:
        mapped_bits = 0
        for flag, _ in expression_rows:
            mapped_bits |= flag
        assert mapped_bits == known_bits
        assert expression_function(known_bits) == " | ".join(expression for _, expression in expression_rows)


def test_cluster_launch_state_is_scoped_to_gfx1250() -> None:
    assert {info.processor for info in amdgpu_target_info_data.AMDGPU_PROCESSOR_INFOS if info.flags & AMDGPU_PROCESSOR_INFO_FLAG_CLUSTER_LAUNCH_STATE} == {"gfx1250"}


def test_initial_vmem_replay_entry_profile_covers_required_processors() -> None:
    processors = {info.processor: info for info in amdgpu_target_info_data.AMDGPU_PROCESSOR_INFOS}

    for processor_name in ("gfx1250", "gfx1251", "gfx12-5-generic"):
        assert processors[processor_name].kernel_entry.profile == AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY
    assert processors["gfx1200"].kernel_entry.profile != AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY


def test_target_info_flag_expressions_reject_unknown_bits() -> None:
    with _raises_value_error("unknown AMDGPU descriptor-set info flags"):
        amdgpu_target_info._descriptor_set_info_flags_expr(1 << 63)
    with _raises_value_error("unknown AMDGPU processor info flags"):
        amdgpu_target_info._processor_info_flags_expr(1 << 31)
    with _raises_value_error("unknown AMDGPU instruction constraint flags"):
        amdgpu_target_info._instruction_constraint_bits_expr(1 << 31)
    with _raises_value_error("unknown AMDGPU target-ID feature support"):
        amdgpu_target_info._target_id_feature_support_flags_expr(1 << 31)
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


def test_exact_processor_rejects_generic_elf_version() -> None:
    processor = processor_info(
        "gfx-test",
        0x001,
        elf_generic_version=1,
        descriptor_set_key="amdgpu.test.core",
    )

    with _raises_value_error("AMDGPU processor gfx-test generic identity and ELF generic version disagree"):
        amdgpu_target_info._validate_processors((processor,), (_descriptor_set_info(),))


def test_elf_feature_flags_reject_packed_generic_version() -> None:
    processor = processor_info(
        "gfx-test",
        0x001,
        elf_feature_flags=0x01000000,
        descriptor_set_key="amdgpu.test.core",
    )

    with _raises_value_error("AMDGPU ELF feature flags for gfx-test must not overlap the generic version"):
        amdgpu_target_info._validate_processors((processor,), (_descriptor_set_info(),))


def test_generic_processor_requires_elf_version() -> None:
    processor = processor_info(
        "gfx-test-generic",
        0x001,
        descriptor_set_key="amdgpu.test.core",
    )

    with _raises_value_error("AMDGPU processor gfx-test-generic generic identity and ELF generic version disagree"):
        amdgpu_target_info._validate_processors((processor,), (_descriptor_set_info(),))


def test_generic_processor_elf_version_must_fit_u8() -> None:
    processor = processor_info(
        "gfx-test-generic",
        0x001,
        elf_generic_version=0x100,
        descriptor_set_key="amdgpu.test.core",
    )

    with _raises_value_error("AMDGPU ELF generic version for gfx-test-generic must fit u8"):
        amdgpu_target_info._validate_processors((processor,), (_descriptor_set_info(),))
