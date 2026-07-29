# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU processor and descriptor-set row data for native target emission.

The public C representation of these facts lives in
`loom/src/loom/target/arch/amdgpu/target_info_defs.h`. This module owns the
Python input rows consumed by the table generator, not emitted C ABI shapes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from build_tools.amdgpu.target_map_data import (
    AMDGPU_EXACT_TARGET_INFOS,
    AMDGPU_GENERIC_CODE_OBJECT_INFOS,
    AMDGPU_TARGET_ID_FEATURE_ORDER,
    TARGET_ID_FEATURE_SRAMECC,
    TARGET_ID_FEATURE_XNACK,
    AmdgpuAsicRevisionInfo,
    AmdgpuExactTargetInfo,
    exact_target_info,
    generic_code_object_current_version,
    target_id_features_for_processor,
)

from loom.dialect.cache import CacheScope, CacheTemporal
from loom.target.arch.amdgpu.lds_bank_service import (
    AMDGPU_LDS_BANK_SERVICE_MODELS_WAVE32_B128_QUAD_PHASES,
    amdgpu_lds_bank_service_model_info_by_key,
    validate_amdgpu_lds_bank_service_model_selection,
)

AMDGPU_AMDHSA_TARGET_TRIPLE = "amdgcn-amd-amdhsa"
AMDGPU_PROCESSOR_ORDINAL_NONE = (2**16) - 1
AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE = (2**16) - 1
AMDGPU_LDS_BANK_SERVICE_MODEL_SET_ORDINAL_NONE = (2**16) - 1

AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE = "none"
AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9 = "gfx9"
AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11 = "gfx11"
AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12 = "gfx12"
AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125 = "gfx125"

AMDGPU_KERNEL_ENTRY_PROFILE_NONE = "none"
AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY = "initial_vmem_replay"

AMDGPU_MATRIX_FEATURE_PROFILE_NONE = "none"
AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908 = "mfma_gfx908"
AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A = "mfma_gfx90a"
AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940 = "mfma_gfx940"
AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950 = "mfma_gfx950"
AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC = "mfma_gfx9_4_generic"
AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11 = "wmma_gfx11"
AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12 = "wmma_gfx12"
AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250 = "wmma_gfx1250"
AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12_5_GENERIC = "wmma_gfx12_5_generic"
# Exact source-contract feature inventories in C enum order. These are not
# cumulative ISA generations because later processors can replace operand and
# fragment layouts while retaining the same semantic operation.
AMDGPU_MATRIX_FEATURE_PROFILES = (
    AMDGPU_MATRIX_FEATURE_PROFILE_NONE,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12_5_GENERIC,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC,
)
AMDGPU_MATRIX_FEATURES_BY_PROFILE = {
    AMDGPU_MATRIX_FEATURE_PROFILE_NONE: (),
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908: (
        "mfma_gfx908",
        "mfma_gfx908_gfx90a",
    ),
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A: (
        "mfma_gfx908",
        "mfma_gfx908_gfx90a",
        "mfma_gfx90a_bf16_1k",
        "mfma_gfx90a_f64",
    ),
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940: (
        "mfma_gfx908",
        "mfma_gfx90a_bf16_1k",
        "mfma_gfx90a_f64",
        "mfma_gfx940_fp8",
        "smfmac_gfx940",
        "mfma_gfx940_xf32",
        "smfmac_gfx940_fp8",
        "mfma_gfx940_i8",
    ),
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950: (
        "mfma_gfx908",
        "mfma_gfx90a_bf16_1k",
        "mfma_gfx90a_f64",
        "mfma_gfx940_fp8",
        "mfma_gfx950",
        "mfma_gfx950_scale_f8f6f4",
        "smfmac_gfx940",
        "smfmac_gfx950",
        "smfmac_gfx940_fp8",
        "mfma_gfx940_i8",
    ),
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11: ("wmma_gfx11",),
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12: (
        "wmma_gfx12",
        "swmmac_gfx12",
    ),
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250: (
        "wmma_gfx1250",
        "wmma_gfx1250_scale_f8f6f4",
        "swmmac_gfx1250",
    ),
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12_5_GENERIC: (
        "wmma_gfx1250",
        "wmma_gfx1250_scale_f8f6f4",
    ),
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC: (
        "mfma_gfx908",
        "mfma_gfx90a_bf16_1k",
        "mfma_gfx90a_f64",
        "smfmac_gfx940",
        "mfma_gfx940_i8",
    ),
}

# Features present on every exact member but intentionally absent from the
# corresponding ROCm generic processor contract.
AMDGPU_GENERIC_MATRIX_FEATURE_EXCLUSIONS = {
    "gfx9-4-generic": (
        "mfma_gfx940_fp8",
        "smfmac_gfx940_fp8",
    ),
    "gfx12-5-generic": ("swmmac_gfx1250",),
}

AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION = 1 << 0
AMDGPU_PROCESSOR_INFO_FLAG_CLUSTER_LAUNCH_STATE = 1 << 1
AMDGPU_PROCESSOR_INFO_KNOWN_FLAGS = (
    AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION
    | AMDGPU_PROCESSOR_INFO_FLAG_CLUSTER_LAUNCH_STATE
)

AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT = 1 << 0
AMDGPU_INSTRUCTION_CONSTRAINT_DS_ADDTID_ADDRESS_MATERIALIZATION = 1 << 1
AMDGPU_INSTRUCTION_CONSTRAINT_CLUSTER_MULTICAST_MASK_PRESERVATION = 1 << 2
AMDGPU_INSTRUCTION_CONSTRAINT_TENSOR_MULTICAST_MASK_PRESERVATION = 1 << 3
AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K64_SCALE_PREFIX = 1 << 4
AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K128_SPLIT = 1 << 5
AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_F4_32X16_SPLIT = 1 << 6
AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_SCALE_ENCODING = 1 << 7
AMDGPU_INSTRUCTION_CONSTRAINT_SWMMAC_LOW_PRECISION_LOWERING = 1 << 8
AMDGPU_INSTRUCTION_CONSTRAINT_INTEGER_MATRIX_COEXECUTION_SPACING = 1 << 9
AMDGPU_INSTRUCTION_CONSTRAINT_KNOWN_BITS = (
    AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT
    | AMDGPU_INSTRUCTION_CONSTRAINT_DS_ADDTID_ADDRESS_MATERIALIZATION
    | AMDGPU_INSTRUCTION_CONSTRAINT_CLUSTER_MULTICAST_MASK_PRESERVATION
    | AMDGPU_INSTRUCTION_CONSTRAINT_TENSOR_MULTICAST_MASK_PRESERVATION
    | AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K64_SCALE_PREFIX
    | AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K128_SPLIT
    | AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_F4_32X16_SPLIT
    | AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_SCALE_ENCODING
    | AMDGPU_INSTRUCTION_CONSTRAINT_SWMMAC_LOW_PRECISION_LOWERING
    | AMDGPU_INSTRUCTION_CONSTRAINT_INTEGER_MATRIX_COEXECUTION_SPACING
)

AMDGPU_TARGET_ID_FEATURE_SUPPORT_NONE = 0
AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC = 1 << 0
AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK = 1 << 1
AMDGPU_TARGET_ID_FEATURE_SUPPORT_KNOWN_FLAGS = (
    AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC | AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK
)
AMDGPU_TARGET_ID_FEATURE_SUPPORT_FLAGS_BY_NAME = {
    TARGET_ID_FEATURE_SRAMECC: AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC,
    TARGET_ID_FEATURE_XNACK: AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK,
}

AMDGPU_DEFAULT_MAX_WORKGROUP_STORAGE_BYTES = 64 * 1024
AMDGPU_CDNA4_MAX_WORKGROUP_STORAGE_BYTES = 160 * 1024
AMDGPU_GFX125X_MAX_WORKGROUP_STORAGE_BYTES = 320 * 1024

AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE = "none"
AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT = "stride14_enable_bit"

AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE = "none"
AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC = "gfx9_11_glc_slc_dlc"
AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH = "gfx12_nv_scope_th"
AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1 = "gfx950_nt_sc0_sc1"
AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODINGS = (
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1,
)

AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ATTR_SCOPE = "scope"
AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ATTR_TH = "th"
AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ATTR_NT = "nt"
AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ATTRS = (
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ATTR_SCOPE,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ATTR_TH,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ATTR_NT,
)

AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR = 1 << 0
AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_WAIT_STATES = 1 << 1
AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES = 1 << 2
AMDGPU_PROCESSOR_SCHEDULING_SDWA_DST_SEL_WAIT_STATES = 1 << 3
AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR = 1 << 4
AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU = 1 << 5
AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER = 1 << 6
AMDGPU_PROCESSOR_SCHEDULING_KNOWN_BITS = (
    AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR
    | AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_WAIT_STATES
    | AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES
    | AMDGPU_PROCESSOR_SCHEDULING_SDWA_DST_SEL_WAIT_STATES
    | AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR
    | AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU
    | AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER
)
AMDGPU_PROCESSOR_SCHEDULING_CDNA_FIXED_WAIT_STATES = (
    AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_WAIT_STATES
    | AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES
    | AMDGPU_PROCESSOR_SCHEDULING_SDWA_DST_SEL_WAIT_STATES
    | AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER
)

AMDGPU_WAVEFRONT_SIZE_FLAG_32 = 1 << 0
AMDGPU_WAVEFRONT_SIZE_FLAG_64 = 1 << 1
AMDGPU_WAVEFRONT_SIZE_KNOWN_FLAGS = (
    AMDGPU_WAVEFRONT_SIZE_FLAG_32 | AMDGPU_WAVEFRONT_SIZE_FLAG_64
)

AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING = 1 << 0
AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION = 1 << 1
AMDGPU_DESCRIPTOR_SET_INFO_KNOWN_FLAGS = (
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING
    | AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION
)
AMDGPU_DESCRIPTOR_SET_INFO_FLAGS_RDNA_VOPD = (
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING
    | AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION
)

AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH = 1 << 0
AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING = 1 << 1
AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ACCUM_OFFSET = 1 << 2
AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_DX10_CLAMP_AND_IEEE_MODE = 1 << 3
AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID = 1 << 4
AMDGPU_KERNEL_DESCRIPTOR_ABI_KNOWN_FLAGS = (
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH
    | AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING
    | AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ACCUM_OFFSET
    | AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_DX10_CLAMP_AND_IEEE_MODE
    | AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID
)
AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAGS_CDNA_GFX9 = (
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH
    | AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ACCUM_OFFSET
    | AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_DX10_CLAMP_AND_IEEE_MODE
    | AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID
)
AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAGS_RDNA3 = (
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH
    | AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING
    | AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_DX10_CLAMP_AND_IEEE_MODE
    | AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID
)
AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAGS_RDNA4 = (
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH
    | AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING
    | AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID
)
AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAGS_PROFILELESS = (
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID
)

AMDGPU_ELF_FEATURE_XNACK_ANY_V4 = 0x100
AMDGPU_ELF_FEATURE_SRAMECC_ANY_V4 = 0x400
AMDGPU_ELF_FEATURE_XNACK_SRAMECC_ANY_V4 = (
    AMDGPU_ELF_FEATURE_XNACK_ANY_V4 | AMDGPU_ELF_FEATURE_SRAMECC_ANY_V4
)
AMDGPU_ELF_GENERIC_VERSION_MASK_V6 = 0xFF000000


def kernel_descriptor_profile_supports_wavefront_size(
    kernel_descriptor_profile: str, wavefront_size: int
) -> bool:
    if wavefront_size == 32:
        return kernel_descriptor_profile not in (
            AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
            AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9,
        )
    if wavefront_size == 64:
        return kernel_descriptor_profile not in (
            AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
            AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125,
        )
    return False


@dataclass(frozen=True, slots=True)
class AmdgpuDescriptorSetBufferResourceInfo:
    cache_swizzle: str = AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE


@dataclass(frozen=True, slots=True)
class AmdgpuDescriptorSetVectorMemoryInfo:
    cache_policy_encoding: str = AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE


@dataclass(frozen=True, slots=True)
class AmdgpuVectorMemoryCachePolicyEncodingInfo:
    encoding: str
    selected_key: str
    cache_scopes: tuple[str, ...]
    cache_temporals: tuple[str, ...]
    attrs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AmdgpuDescriptorSetIsaInfo:
    isa_xml_key: str
    isa_architecture_name: str
    isa_architecture_id: int


@dataclass(frozen=True, slots=True)
class AmdgpuDescriptorSetInfo:
    generator_target: str
    key: str
    isa_infos: tuple[AmdgpuDescriptorSetIsaInfo, ...]
    flags: int
    storage_generator_target: str | None = None
    member_generator_targets: tuple[str, ...] = ()
    buffer_resource: AmdgpuDescriptorSetBufferResourceInfo = (
        AmdgpuDescriptorSetBufferResourceInfo()
    )
    vector_memory: AmdgpuDescriptorSetVectorMemoryInfo = (
        AmdgpuDescriptorSetVectorMemoryInfo()
    )


AMDGPU_DESCRIPTOR_SET_ISA_CDNA3 = AmdgpuDescriptorSetIsaInfo(
    isa_xml_key="cdna3",
    isa_architecture_name="AMD CDNA 3",
    isa_architecture_id=2,
)

AMDGPU_DESCRIPTOR_SET_ISA_CDNA4 = AmdgpuDescriptorSetIsaInfo(
    isa_xml_key="cdna4",
    isa_architecture_name="AMD CDNA 4",
    isa_architecture_id=3,
)

AMDGPU_DESCRIPTOR_SET_ISA_RDNA3 = AmdgpuDescriptorSetIsaInfo(
    isa_xml_key="rdna3",
    isa_architecture_name="AMD RDNA 3",
    isa_architecture_id=8,
)

AMDGPU_DESCRIPTOR_SET_ISA_RDNA3_5 = AmdgpuDescriptorSetIsaInfo(
    isa_xml_key="rdna3_5",
    isa_architecture_name="AMD RDNA 3.5",
    isa_architecture_id=9,
)

AMDGPU_DESCRIPTOR_SET_ISA_RDNA4 = AmdgpuDescriptorSetIsaInfo(
    isa_xml_key="rdna4",
    isa_architecture_name="AMD RDNA 4",
    isa_architecture_id=10,
)

# Kernel metadata fields written directly by the native AMDGPU emitter.
# Revision rows may add external extension fields but may not replace these
# standard fields.
AMDGPU_STANDARD_KERNEL_METADATA_KEYS = frozenset(
    (
        ".args",
        ".cluster_dims",
        ".group_segment_fixed_size",
        ".kernarg_segment_align",
        ".kernarg_segment_size",
        ".max_flat_workgroup_size",
        ".name",
        ".private_segment_fixed_size",
        ".reqd_workgroup_size",
        ".sgpr_count",
        ".symbol",
        ".vgpr_count",
        ".wavefront_size",
    )
)


@dataclass(frozen=True, slots=True)
class AmdgpuProcessorDescriptorSetInfo:
    key: str


@dataclass(frozen=True, slots=True)
class AmdgpuProcessorElfInfo:
    machine_flags: int
    feature_flags: int = 0
    generic_version: int = 0


@dataclass(frozen=True, slots=True)
class AmdgpuProcessorTargetIdInfo:
    supported_features: int = AMDGPU_TARGET_ID_FEATURE_SUPPORT_NONE


@dataclass(frozen=True, slots=True)
class AmdgpuProcessorAsicRevisionInfo:
    value: int
    name: str
    instruction_constraints: int = 0
    lds_bank_service_models: tuple[str, ...] = ()
    kernel_metadata_extensions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class AmdgpuProcessorAsicRevisionSemantics:
    value: int
    instruction_constraints: int = 0
    lds_bank_service_models: tuple[str, ...] | None = None
    kernel_metadata_extensions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class AmdgpuProcessorWavefrontInfo:
    default_size: int = 64


@dataclass(frozen=True, slots=True)
class AmdgpuKernelDescriptorVgprGranules:
    wave32: int = 0
    wave64: int = 0


@dataclass(frozen=True, slots=True)
class AmdgpuProcessorKernelDescriptorInfo:
    profile: str = AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE
    flags: int = 0
    vgpr_granules: AmdgpuKernelDescriptorVgprGranules = (
        AmdgpuKernelDescriptorVgprGranules()
    )


@dataclass(frozen=True, slots=True)
class AmdgpuProcessorKernelEntryInfo:
    profile: str = AMDGPU_KERNEL_ENTRY_PROFILE_NONE


@dataclass(frozen=True, slots=True)
class AmdgpuProcessorInstructionInfo:
    base_constraints: int = 0


@dataclass(frozen=True, slots=True)
class AmdgpuProcessorFeatureInfo:
    matrix: str = AMDGPU_MATRIX_FEATURE_PROFILE_NONE
    scheduling: int = 0
    lds_bank_service_models: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AmdgpuProcessorLimitInfo:
    max_workgroup_storage_bytes: int = 0


@dataclass(frozen=True, slots=True)
class AmdgpuOccupancyRegisterClassInfo:
    register_class: str
    pool_units: int
    allocation_granularity: int
    limits_occupancy: bool = True


@dataclass(frozen=True, slots=True)
class AmdgpuOccupancyResourceMemberInfo:
    register_class: str
    contribution_granularity: int = 1


@dataclass(frozen=True, slots=True)
class AmdgpuOccupancyResourceInfo:
    resource: str
    pool_units: int
    allocation_granularity: int
    members: tuple[AmdgpuOccupancyResourceMemberInfo, ...]


@dataclass(frozen=True, slots=True)
class AmdgpuOccupancyDomainInfo:
    simd_count: int
    local_memory_bytes: int
    local_memory_allocation_granularity: int
    max_barrier_workgroup_count: int


@dataclass(frozen=True, slots=True)
class AmdgpuOccupancyModelInfo:
    max_waves_per_simd: int
    domain: AmdgpuOccupancyDomainInfo
    register_classes: tuple[AmdgpuOccupancyRegisterClassInfo, ...]
    resources: tuple[AmdgpuOccupancyResourceInfo, ...] = ()


@dataclass(frozen=True, slots=True)
class AmdgpuProcessorOccupancyInfo:
    wave32: AmdgpuOccupancyModelInfo | None = None
    wave64: AmdgpuOccupancyModelInfo | None = None


@dataclass(frozen=True, slots=True)
class AmdgpuProcessorInfo:
    processor: str
    flags: int
    descriptor_set: AmdgpuProcessorDescriptorSetInfo
    elf: AmdgpuProcessorElfInfo
    target_id: AmdgpuProcessorTargetIdInfo
    asic_revisions: tuple[AmdgpuProcessorAsicRevisionInfo, ...]
    default_asic_revision: int | None
    wavefront: AmdgpuProcessorWavefrontInfo
    kernel_descriptor: AmdgpuProcessorKernelDescriptorInfo
    kernel_entry: AmdgpuProcessorKernelEntryInfo = AmdgpuProcessorKernelEntryInfo()
    instructions: AmdgpuProcessorInstructionInfo = AmdgpuProcessorInstructionInfo()
    features: AmdgpuProcessorFeatureInfo = AmdgpuProcessorFeatureInfo()
    limits: AmdgpuProcessorLimitInfo = AmdgpuProcessorLimitInfo()
    occupancy: AmdgpuProcessorOccupancyInfo = AmdgpuProcessorOccupancyInfo()


AMDGPU_OCCUPANCY_NONE = AmdgpuProcessorOccupancyInfo()


AMDGPU_KERNEL_DESCRIPTOR_INFO_NONE = AmdgpuProcessorKernelDescriptorInfo()
AMDGPU_KERNEL_ENTRY_INFO_NONE = AmdgpuProcessorKernelEntryInfo()
AMDGPU_PROCESSOR_INSTRUCTION_INFO_NONE = AmdgpuProcessorInstructionInfo()
AMDGPU_KERNEL_DESCRIPTOR_INFO_PACKED_WORKITEM_ID = AmdgpuProcessorKernelDescriptorInfo(
    flags=AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID,
)
AMDGPU_KERNEL_DESCRIPTOR_INFO_CDNA_GFX9 = AmdgpuProcessorKernelDescriptorInfo(
    profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9,
    flags=AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAGS_CDNA_GFX9,
    vgpr_granules=AmdgpuKernelDescriptorVgprGranules(wave32=8, wave64=8),
)
AMDGPU_KERNEL_DESCRIPTOR_INFO_RDNA3_GFX11 = AmdgpuProcessorKernelDescriptorInfo(
    profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11,
    flags=AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAGS_RDNA3,
    vgpr_granules=AmdgpuKernelDescriptorVgprGranules(wave32=8, wave64=4),
)
AMDGPU_KERNEL_DESCRIPTOR_INFO_RDNA4_GFX12 = AmdgpuProcessorKernelDescriptorInfo(
    profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12,
    flags=AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAGS_RDNA4,
    vgpr_granules=AmdgpuKernelDescriptorVgprGranules(wave32=8, wave64=4),
)
AMDGPU_KERNEL_DESCRIPTOR_INFO_RDNA4_GFX125 = AmdgpuProcessorKernelDescriptorInfo(
    profile=AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125,
    flags=AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAGS_RDNA4,
    vgpr_granules=AmdgpuKernelDescriptorVgprGranules(wave32=16, wave64=8),
)

AMDGPU_KERNEL_ENTRY_INFO_INITIAL_VMEM_REPLAY = AmdgpuProcessorKernelEntryInfo(
    profile=AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY
)

# These facts mirror LLVM's AMDGPU::IsaInfo occupancy model:
# - GFX10+ SGPR allocation never limits occupancy.
# - Feature1536VGPRs selects 1536/24 VGPRs for wave32 and 768/12 for
#   wave64; other GFX10+ targets use 1024/16 and 512/8.
# - GFX10.3+ targets expose at most 16 resident waves per SIMD.
AMDGPU_OCCUPANCY_REGISTER_CLASS_SGPR_CDNA = AmdgpuOccupancyRegisterClassInfo(
    "amdgpu.sgpr", 800, 16
)
AMDGPU_OCCUPANCY_REGISTER_CLASS_SGPR_RDNA = AmdgpuOccupancyRegisterClassInfo(
    "amdgpu.sgpr", 800, 106, limits_occupancy=False
)
AMDGPU_OCCUPANCY_REGISTER_CLASS_AGPR_CDNA = AmdgpuOccupancyRegisterClassInfo(
    "amdgpu.agpr", 256, 4, limits_occupancy=False
)

AMDGPU_OCCUPANCY_DOMAIN_CDNA3 = AmdgpuOccupancyDomainInfo(
    simd_count=4,
    local_memory_bytes=64 * 1024,
    local_memory_allocation_granularity=512,
    max_barrier_workgroup_count=16,
)
AMDGPU_OCCUPANCY_DOMAIN_CDNA4 = AmdgpuOccupancyDomainInfo(
    simd_count=4,
    local_memory_bytes=160 * 1024,
    local_memory_allocation_granularity=1280,
    max_barrier_workgroup_count=16,
)
AMDGPU_OCCUPANCY_DOMAIN_RDNA = AmdgpuOccupancyDomainInfo(
    simd_count=4,
    local_memory_bytes=128 * 1024,
    local_memory_allocation_granularity=512,
    max_barrier_workgroup_count=32,
)
AMDGPU_OCCUPANCY_DOMAIN_GFX125X = AmdgpuOccupancyDomainInfo(
    simd_count=4,
    local_memory_bytes=320 * 1024,
    local_memory_allocation_granularity=2048,
    max_barrier_workgroup_count=16,
)

AMDGPU_OCCUPANCY_CDNA3 = AmdgpuProcessorOccupancyInfo(
    wave64=AmdgpuOccupancyModelInfo(
        max_waves_per_simd=8,
        domain=AMDGPU_OCCUPANCY_DOMAIN_CDNA3,
        register_classes=(
            AMDGPU_OCCUPANCY_REGISTER_CLASS_SGPR_CDNA,
            AmdgpuOccupancyRegisterClassInfo("amdgpu.vgpr", 512, 8),
            AMDGPU_OCCUPANCY_REGISTER_CLASS_AGPR_CDNA,
        ),
        resources=(
            AmdgpuOccupancyResourceInfo(
                "amdgpu.vgpr_agpr",
                512,
                8,
                (
                    AmdgpuOccupancyResourceMemberInfo("amdgpu.vgpr", 4),
                    AmdgpuOccupancyResourceMemberInfo("amdgpu.agpr"),
                ),
            ),
        ),
    ),
)
AMDGPU_OCCUPANCY_CDNA4 = AmdgpuProcessorOccupancyInfo(
    wave64=AmdgpuOccupancyModelInfo(
        max_waves_per_simd=8,
        domain=AMDGPU_OCCUPANCY_DOMAIN_CDNA4,
        register_classes=(
            AMDGPU_OCCUPANCY_REGISTER_CLASS_SGPR_CDNA,
            AmdgpuOccupancyRegisterClassInfo("amdgpu.vgpr", 512, 8),
            AMDGPU_OCCUPANCY_REGISTER_CLASS_AGPR_CDNA,
        ),
        resources=(
            AmdgpuOccupancyResourceInfo(
                "amdgpu.vgpr_agpr",
                512,
                8,
                (
                    AmdgpuOccupancyResourceMemberInfo("amdgpu.vgpr", 4),
                    AmdgpuOccupancyResourceMemberInfo("amdgpu.agpr"),
                ),
            ),
        ),
    ),
)
AMDGPU_OCCUPANCY_GFX9_4_GENERIC = AMDGPU_OCCUPANCY_CDNA3
AMDGPU_OCCUPANCY_RDNA_1024 = AmdgpuProcessorOccupancyInfo(
    wave32=AmdgpuOccupancyModelInfo(
        max_waves_per_simd=16,
        domain=AMDGPU_OCCUPANCY_DOMAIN_RDNA,
        register_classes=(
            AMDGPU_OCCUPANCY_REGISTER_CLASS_SGPR_RDNA,
            AmdgpuOccupancyRegisterClassInfo("amdgpu.vgpr", 1024, 16),
        ),
    ),
    wave64=AmdgpuOccupancyModelInfo(
        max_waves_per_simd=16,
        domain=AMDGPU_OCCUPANCY_DOMAIN_RDNA,
        register_classes=(
            AMDGPU_OCCUPANCY_REGISTER_CLASS_SGPR_RDNA,
            AmdgpuOccupancyRegisterClassInfo("amdgpu.vgpr", 512, 8),
        ),
    ),
)
AMDGPU_OCCUPANCY_RDNA_1536 = AmdgpuProcessorOccupancyInfo(
    wave32=AmdgpuOccupancyModelInfo(
        max_waves_per_simd=16,
        domain=AMDGPU_OCCUPANCY_DOMAIN_RDNA,
        register_classes=(
            AMDGPU_OCCUPANCY_REGISTER_CLASS_SGPR_RDNA,
            AmdgpuOccupancyRegisterClassInfo("amdgpu.vgpr", 1536, 24),
        ),
    ),
    wave64=AmdgpuOccupancyModelInfo(
        max_waves_per_simd=16,
        domain=AMDGPU_OCCUPANCY_DOMAIN_RDNA,
        register_classes=(
            AMDGPU_OCCUPANCY_REGISTER_CLASS_SGPR_RDNA,
            AmdgpuOccupancyRegisterClassInfo("amdgpu.vgpr", 768, 12),
        ),
    ),
)
AMDGPU_OCCUPANCY_GFX125X = AmdgpuProcessorOccupancyInfo(
    wave32=AmdgpuOccupancyModelInfo(
        max_waves_per_simd=16,
        domain=AMDGPU_OCCUPANCY_DOMAIN_GFX125X,
        register_classes=(
            AMDGPU_OCCUPANCY_REGISTER_CLASS_SGPR_RDNA,
            AmdgpuOccupancyRegisterClassInfo("amdgpu.vgpr", 1024, 16),
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class AmdgpuTargetRecordInfo:
    processor: str
    enum_value: int
    doc: str
    default_for_descriptor_set: bool = False


class AmdgpuIsaArchitectureInfo(Protocol):
    @property
    def source_name(self) -> str: ...

    @property
    def architecture_name(self) -> str: ...

    @property
    def architecture_id(self) -> int: ...


AMDGPU_CACHE_SCOPE_KEYWORDS = tuple(case.keyword for case in CacheScope.cases)
AMDGPU_CACHE_TEMPORAL_KEYWORDS = tuple(case.keyword for case in CacheTemporal.cases)

AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_INFOS: tuple[
    AmdgpuVectorMemoryCachePolicyEncodingInfo, ...
] = (
    AmdgpuVectorMemoryCachePolicyEncodingInfo(
        encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC,
        selected_key="memory_cache_policy.gfx9_11_glc_slc_dlc",
        cache_scopes=("device",),
        cache_temporals=("regular",),
    ),
    AmdgpuVectorMemoryCachePolicyEncodingInfo(
        encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH,
        selected_key="memory_cache_policy.gfx12_nv_scope_th",
        cache_scopes=AMDGPU_CACHE_SCOPE_KEYWORDS,
        cache_temporals=AMDGPU_CACHE_TEMPORAL_KEYWORDS,
        attrs=(
            AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ATTR_SCOPE,
            AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ATTR_TH,
        ),
    ),
    AmdgpuVectorMemoryCachePolicyEncodingInfo(
        encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1,
        selected_key="memory_cache_policy.gfx950_nt_sc0_sc1",
        cache_scopes=("device",),
        cache_temporals=("regular", "non_temporal"),
        attrs=(AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ATTR_NT,),
    ),
)

AMDGPU_VECTOR_MEMORY_CACHE_POLICY_TEMPORAL_TH: tuple[tuple[str, int], ...] = (
    ("regular", 0),
    ("non_temporal", 1),
    ("high_temporal", 2),
    ("last_use", 3),
    ("writeback", 3),
    ("non_temporal_regular", 4),
    ("regular_non_temporal", 5),
    ("non_temporal_high_temporal", 6),
    ("non_temporal_writeback", 7),
    ("bypass", 3),
)


def amdgpu_target_id_feature_support_flags(
    target_id_features: Sequence[str],
) -> int:
    flags = AMDGPU_TARGET_ID_FEATURE_SUPPORT_NONE
    for feature in target_id_features:
        try:
            flags |= AMDGPU_TARGET_ID_FEATURE_SUPPORT_FLAGS_BY_NAME[feature]
        except KeyError as exc:
            raise ValueError(f"unknown AMDGPU target-ID feature '{feature}'") from exc
    return flags


def amdgpu_processor_target_id_info(
    processor: str,
) -> AmdgpuProcessorTargetIdInfo:
    return AmdgpuProcessorTargetIdInfo(
        supported_features=amdgpu_target_id_feature_support_flags(
            target_id_features_for_processor(processor)
        )
    )


def processor_info(
    processor: str,
    elf_machine_flags: int,
    *,
    elf_feature_flags: int = 0,
    elf_generic_version: int = 0,
    default_wavefront_size: int = 64,
    descriptor_set_key: str = "",
    kernel_descriptor: AmdgpuProcessorKernelDescriptorInfo = (
        AMDGPU_KERNEL_DESCRIPTOR_INFO_NONE
    ),
    kernel_entry: AmdgpuProcessorKernelEntryInfo = AMDGPU_KERNEL_ENTRY_INFO_NONE,
    instructions: AmdgpuProcessorInstructionInfo = (
        AMDGPU_PROCESSOR_INSTRUCTION_INFO_NONE
    ),
    asic_revision_semantics: tuple[AmdgpuProcessorAsicRevisionSemantics, ...] = (),
    matrix_feature_profile: str = AMDGPU_MATRIX_FEATURE_PROFILE_NONE,
    scheduling_bits: int = 0,
    lds_bank_service_models: tuple[str, ...] = (),
    max_workgroup_storage_bytes: int = 0,
    flags: int = 0,
    occupancy: AmdgpuProcessorOccupancyInfo = AMDGPU_OCCUPANCY_NONE,
) -> AmdgpuProcessorInfo:
    if flags & AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION:
        max_workgroup_storage_bytes = (
            max_workgroup_storage_bytes
            if max_workgroup_storage_bytes != 0
            else AMDGPU_DEFAULT_MAX_WORKGROUP_STORAGE_BYTES
        )
    canonical_exact_target = exact_target_info(processor)
    semantic_values = tuple(semantics.value for semantics in asic_revision_semantics)
    if len(set(semantic_values)) != len(semantic_values):
        raise ValueError(
            f"AMDGPU processor {processor} repeats an ASIC revision semantic overlay"
        )
    if semantic_values != tuple(sorted(semantic_values)):
        raise ValueError(
            f"AMDGPU processor {processor} ASIC revision semantic overlays "
            "are not in ascending value order"
        )
    canonical_revision_values = (
        {revision.value for revision in canonical_exact_target.asic_revisions}
        if canonical_exact_target is not None
        else set()
    )
    unknown_semantic_values = sorted(set(semantic_values) - canonical_revision_values)
    if unknown_semantic_values:
        raise ValueError(
            f"AMDGPU processor {processor} has semantic overlays for unknown "
            "ASIC revisions: "
            + ", ".join(str(value) for value in unknown_semantic_values)
        )
    semantics_by_value = {
        semantics.value: semantics for semantics in asic_revision_semantics
    }

    def materialize_revision(
        revision: AmdgpuAsicRevisionInfo,
    ) -> AmdgpuProcessorAsicRevisionInfo:
        semantics = semantics_by_value.get(revision.value)
        return AmdgpuProcessorAsicRevisionInfo(
            value=revision.value,
            name=revision.name,
            instruction_constraints=(
                semantics.instruction_constraints if semantics is not None else 0
            ),
            lds_bank_service_models=(
                semantics.lds_bank_service_models
                if semantics is not None
                and semantics.lds_bank_service_models is not None
                else lds_bank_service_models
            ),
            kernel_metadata_extensions=(
                semantics.kernel_metadata_extensions if semantics is not None else ()
            ),
        )

    asic_revisions = tuple(
        materialize_revision(revision)
        for revision in (
            canonical_exact_target.asic_revisions
            if canonical_exact_target is not None
            else ()
        )
    )
    default_asic_revision = (
        canonical_exact_target.default_asic_revision
        if canonical_exact_target is not None
        else None
    )
    return AmdgpuProcessorInfo(
        processor=processor,
        flags=flags,
        descriptor_set=AmdgpuProcessorDescriptorSetInfo(key=descriptor_set_key),
        elf=AmdgpuProcessorElfInfo(
            machine_flags=elf_machine_flags,
            feature_flags=elf_feature_flags,
            generic_version=elf_generic_version,
        ),
        target_id=amdgpu_processor_target_id_info(processor),
        asic_revisions=asic_revisions,
        default_asic_revision=default_asic_revision,
        wavefront=AmdgpuProcessorWavefrontInfo(default_size=default_wavefront_size),
        kernel_descriptor=kernel_descriptor,
        kernel_entry=kernel_entry,
        instructions=instructions,
        features=AmdgpuProcessorFeatureInfo(
            matrix=matrix_feature_profile,
            scheduling=scheduling_bits,
            lds_bank_service_models=lds_bank_service_models,
        ),
        limits=AmdgpuProcessorLimitInfo(
            max_workgroup_storage_bytes=max_workgroup_storage_bytes,
        ),
        occupancy=occupancy,
    )


def gfx9_10_processor_info(
    processor: str,
    elf_machine_flags: int,
    *,
    elf_feature_flags: int = 0,
    elf_generic_version: int = 0,
    default_wavefront_size: int = 64,
    matrix_feature_profile: str = AMDGPU_MATRIX_FEATURE_PROFILE_NONE,
    kernel_descriptor: AmdgpuProcessorKernelDescriptorInfo = (
        AMDGPU_KERNEL_DESCRIPTOR_INFO_NONE
    ),
) -> AmdgpuProcessorInfo:
    return processor_info(
        processor,
        elf_machine_flags,
        elf_feature_flags=elf_feature_flags,
        elf_generic_version=elf_generic_version,
        default_wavefront_size=default_wavefront_size,
        matrix_feature_profile=matrix_feature_profile,
        kernel_descriptor=kernel_descriptor,
        scheduling_bits=AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER,
    )


def rdna3_processor_info(
    processor: str,
    elf_machine_flags: int,
    *,
    descriptor_set_key: str = "amdgpu.rdna3.core",
    elf_feature_flags: int = 0,
    elf_generic_version: int = 0,
    scheduling_bits: int = 0,
    occupancy: AmdgpuProcessorOccupancyInfo = AMDGPU_OCCUPANCY_RDNA_1024,
) -> AmdgpuProcessorInfo:
    return processor_info(
        processor=processor,
        elf_machine_flags=elf_machine_flags,
        flags=AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
        descriptor_set_key=descriptor_set_key,
        elf_feature_flags=elf_feature_flags,
        elf_generic_version=elf_generic_version,
        default_wavefront_size=32,
        kernel_descriptor=AMDGPU_KERNEL_DESCRIPTOR_INFO_RDNA3_GFX11,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11,
        scheduling_bits=(
            scheduling_bits
            | AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU
            | AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER
        ),
        occupancy=occupancy,
    )


def cdna3_processor_info(
    processor: str,
    elf_machine_flags: int,
    *,
    flags: int = AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
    matrix_feature_profile: str = AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940,
) -> AmdgpuProcessorInfo:
    return processor_info(
        processor,
        elf_machine_flags,
        flags=flags,
        descriptor_set_key="amdgpu.cdna3.core",
        elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_SRAMECC_ANY_V4,
        kernel_descriptor=AMDGPU_KERNEL_DESCRIPTOR_INFO_CDNA_GFX9,
        matrix_feature_profile=matrix_feature_profile,
        scheduling_bits=AMDGPU_PROCESSOR_SCHEDULING_CDNA_FIXED_WAIT_STATES,
        max_workgroup_storage_bytes=AMDGPU_DEFAULT_MAX_WORKGROUP_STORAGE_BYTES,
        occupancy=AMDGPU_OCCUPANCY_CDNA3,
    )


def gfx117x_processor_info(
    processor: str,
    elf_machine_flags: int,
    *,
    occupancy: AmdgpuProcessorOccupancyInfo = AMDGPU_OCCUPANCY_RDNA_1024,
) -> AmdgpuProcessorInfo:
    return processor_info(
        processor=processor,
        flags=AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
        descriptor_set_key="amdgpu.rdna3_5.core",
        elf_machine_flags=elf_machine_flags,
        default_wavefront_size=32,
        kernel_descriptor=AMDGPU_KERNEL_DESCRIPTOR_INFO_RDNA3_GFX11,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11,
        scheduling_bits=(
            AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR
            | AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU
            | AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER
        ),
        occupancy=occupancy,
    )


def rdna4_processor_info(
    processor: str,
    elf_machine_flags: int,
    *,
    descriptor_set_key: str = "amdgpu.rdna4.core",
    elf_feature_flags: int = 0,
    elf_generic_version: int = 0,
) -> AmdgpuProcessorInfo:
    return processor_info(
        processor=processor,
        flags=AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
        descriptor_set_key=descriptor_set_key,
        elf_machine_flags=elf_machine_flags,
        elf_feature_flags=elf_feature_flags,
        elf_generic_version=elf_generic_version,
        default_wavefront_size=32,
        kernel_descriptor=AMDGPU_KERNEL_DESCRIPTOR_INFO_RDNA4_GFX12,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12,
        scheduling_bits=(
            AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR
            | AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU
        ),
        occupancy=AMDGPU_OCCUPANCY_RDNA_1536,
    )


def gfx125x_processor_info(
    processor: str,
    elf_machine_flags: int,
    *,
    descriptor_set_key: str = "amdgpu.rdna4.gfx125x.core",
    elf_feature_flags: int = 0,
    elf_generic_version: int = 0,
    processor_flags: int = 0,
    instructions: AmdgpuProcessorInstructionInfo = (
        AMDGPU_PROCESSOR_INSTRUCTION_INFO_NONE
    ),
    asic_revision_semantics: tuple[AmdgpuProcessorAsicRevisionSemantics, ...] = (),
    matrix_feature_profile: str = AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250,
    lds_bank_service_models: tuple[str, ...] = (),
) -> AmdgpuProcessorInfo:
    return processor_info(
        processor=processor,
        flags=AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION | processor_flags,
        descriptor_set_key=descriptor_set_key,
        elf_machine_flags=elf_machine_flags,
        elf_feature_flags=elf_feature_flags,
        elf_generic_version=elf_generic_version,
        default_wavefront_size=32,
        kernel_descriptor=AMDGPU_KERNEL_DESCRIPTOR_INFO_RDNA4_GFX125,
        kernel_entry=AMDGPU_KERNEL_ENTRY_INFO_INITIAL_VMEM_REPLAY,
        instructions=instructions,
        asic_revision_semantics=asic_revision_semantics,
        matrix_feature_profile=matrix_feature_profile,
        lds_bank_service_models=lds_bank_service_models,
        scheduling_bits=AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU,
        max_workgroup_storage_bytes=AMDGPU_GFX125X_MAX_WORKGROUP_STORAGE_BYTES,
        occupancy=AMDGPU_OCCUPANCY_GFX125X,
    )


AMDGPU_DESCRIPTOR_SET_INFOS: tuple[AmdgpuDescriptorSetInfo, ...] = (
    AmdgpuDescriptorSetInfo(
        generator_target="cdna3",
        key="amdgpu.cdna3.core",
        isa_infos=(AMDGPU_DESCRIPTOR_SET_ISA_CDNA3,),
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1,
        ),
    ),
    AmdgpuDescriptorSetInfo(
        generator_target="rdna4_gfx125x",
        key="amdgpu.rdna4.gfx125x.core",
        isa_infos=(AMDGPU_DESCRIPTOR_SET_ISA_RDNA4,),
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAGS_RDNA_VOPD,
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH,
        ),
    ),
    AmdgpuDescriptorSetInfo(
        generator_target="rdna4_gfx1251",
        key="amdgpu.rdna4.gfx1251.core",
        isa_infos=(AMDGPU_DESCRIPTOR_SET_ISA_RDNA4,),
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAGS_RDNA_VOPD,
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH,
        ),
    ),
    AmdgpuDescriptorSetInfo(
        generator_target="rdna3",
        key="amdgpu.rdna3.core",
        isa_infos=(AMDGPU_DESCRIPTOR_SET_ISA_RDNA3,),
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAGS_RDNA_VOPD,
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC,
        ),
    ),
    AmdgpuDescriptorSetInfo(
        generator_target="rdna3_5",
        key="amdgpu.rdna3_5.core",
        isa_infos=(AMDGPU_DESCRIPTOR_SET_ISA_RDNA3_5,),
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAGS_RDNA_VOPD,
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC,
        ),
    ),
    AmdgpuDescriptorSetInfo(
        generator_target="rdna4",
        key="amdgpu.rdna4.core",
        isa_infos=(AMDGPU_DESCRIPTOR_SET_ISA_RDNA4,),
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAGS_RDNA_VOPD,
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH,
        ),
    ),
    AmdgpuDescriptorSetInfo(
        generator_target="cdna4",
        key="amdgpu.cdna4.core",
        isa_infos=(AMDGPU_DESCRIPTOR_SET_ISA_CDNA4,),
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
        buffer_resource=AmdgpuDescriptorSetBufferResourceInfo(
            cache_swizzle=AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT,
        ),
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1,
        ),
    ),
    AmdgpuDescriptorSetInfo(
        generator_target="gfx9_4_generic",
        key="amdgpu.gfx9_4.generic.core",
        isa_infos=(
            AMDGPU_DESCRIPTOR_SET_ISA_CDNA3,
            AMDGPU_DESCRIPTOR_SET_ISA_CDNA4,
        ),
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
        storage_generator_target="cdna3",
        member_generator_targets=("cdna3", "cdna4"),
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1,
        ),
    ),
    AmdgpuDescriptorSetInfo(
        generator_target="gfx11_generic",
        key="amdgpu.gfx11.generic.core",
        isa_infos=(
            AMDGPU_DESCRIPTOR_SET_ISA_RDNA3,
            AMDGPU_DESCRIPTOR_SET_ISA_RDNA3_5,
        ),
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAGS_RDNA_VOPD,
        storage_generator_target="rdna3",
        member_generator_targets=("rdna3", "rdna3_5"),
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC,
        ),
    ),
    AmdgpuDescriptorSetInfo(
        generator_target="gfx12_generic",
        key="amdgpu.gfx12.generic.core",
        isa_infos=(AMDGPU_DESCRIPTOR_SET_ISA_RDNA4,),
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAGS_RDNA_VOPD,
        storage_generator_target="rdna4",
        member_generator_targets=("rdna4",),
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH,
        ),
    ),
    AmdgpuDescriptorSetInfo(
        generator_target="gfx12_5_generic",
        key="amdgpu.gfx12_5.generic.core",
        isa_infos=(AMDGPU_DESCRIPTOR_SET_ISA_RDNA4,),
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAGS_RDNA_VOPD,
        member_generator_targets=("rdna4_gfx1251", "rdna4_gfx125x"),
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH,
        ),
    ),
)


AMDGPU_PROCESSOR_INFOS: tuple[AmdgpuProcessorInfo, ...] = (
    gfx9_10_processor_info(
        "gfx900", 0x02C, elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_ANY_V4
    ),
    gfx9_10_processor_info(
        "gfx902", 0x02D, elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_ANY_V4
    ),
    gfx9_10_processor_info(
        "gfx904", 0x02E, elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_ANY_V4
    ),
    gfx9_10_processor_info(
        "gfx906", 0x02F, elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_SRAMECC_ANY_V4
    ),
    gfx9_10_processor_info(
        "gfx908",
        0x030,
        elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_SRAMECC_ANY_V4,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908,
    ),
    gfx9_10_processor_info(
        "gfx909", 0x031, elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_ANY_V4
    ),
    gfx9_10_processor_info(
        "gfx90a",
        0x03F,
        elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_SRAMECC_ANY_V4,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A,
        kernel_descriptor=AMDGPU_KERNEL_DESCRIPTOR_INFO_PACKED_WORKITEM_ID,
    ),
    gfx9_10_processor_info(
        "gfx90c", 0x032, elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_ANY_V4
    ),
    cdna3_processor_info("gfx940", 0x040),
    cdna3_processor_info("gfx941", 0x04B),
    cdna3_processor_info("gfx942", 0x04C),
    processor_info(
        "gfx950",
        0x04F,
        flags=AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
        descriptor_set_key="amdgpu.cdna4.core",
        elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_SRAMECC_ANY_V4,
        kernel_descriptor=AMDGPU_KERNEL_DESCRIPTOR_INFO_CDNA_GFX9,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950,
        scheduling_bits=AMDGPU_PROCESSOR_SCHEDULING_CDNA_FIXED_WAIT_STATES,
        max_workgroup_storage_bytes=AMDGPU_CDNA4_MAX_WORKGROUP_STORAGE_BYTES,
        occupancy=AMDGPU_OCCUPANCY_CDNA4,
    ),
    gfx9_10_processor_info(
        "gfx1010",
        0x033,
        elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_ANY_V4,
        default_wavefront_size=32,
    ),
    gfx9_10_processor_info(
        "gfx1011",
        0x034,
        elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_ANY_V4,
        default_wavefront_size=32,
    ),
    gfx9_10_processor_info(
        "gfx1012",
        0x035,
        elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_ANY_V4,
        default_wavefront_size=32,
    ),
    gfx9_10_processor_info(
        "gfx1013",
        0x042,
        elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_ANY_V4,
        default_wavefront_size=32,
    ),
    gfx9_10_processor_info("gfx1030", 0x036, default_wavefront_size=32),
    gfx9_10_processor_info("gfx1031", 0x037, default_wavefront_size=32),
    gfx9_10_processor_info("gfx1032", 0x038, default_wavefront_size=32),
    gfx9_10_processor_info("gfx1033", 0x039, default_wavefront_size=32),
    gfx9_10_processor_info("gfx1034", 0x03E, default_wavefront_size=32),
    gfx9_10_processor_info("gfx1035", 0x03D, default_wavefront_size=32),
    gfx9_10_processor_info("gfx1036", 0x045, default_wavefront_size=32),
    rdna3_processor_info(
        processor="gfx1100",
        elf_machine_flags=0x041,
        scheduling_bits=AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR,
        occupancy=AMDGPU_OCCUPANCY_RDNA_1536,
    ),
    rdna3_processor_info(
        processor="gfx1101",
        elf_machine_flags=0x046,
        scheduling_bits=AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR,
        occupancy=AMDGPU_OCCUPANCY_RDNA_1536,
    ),
    rdna3_processor_info(
        processor="gfx1102",
        elf_machine_flags=0x047,
        scheduling_bits=AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR,
    ),
    rdna3_processor_info(
        processor="gfx1103",
        elf_machine_flags=0x044,
        scheduling_bits=AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR,
    ),
    gfx117x_processor_info("gfx1150", 0x043),
    gfx117x_processor_info("gfx1151", 0x04A, occupancy=AMDGPU_OCCUPANCY_RDNA_1536),
    gfx117x_processor_info("gfx1152", 0x055),
    gfx117x_processor_info("gfx1153", 0x058),
    gfx117x_processor_info("gfx1170", 0x05D),
    gfx117x_processor_info("gfx1171", 0x05E),
    gfx117x_processor_info("gfx1172", 0x05C),
    rdna4_processor_info("gfx1200", 0x048),
    rdna4_processor_info("gfx1201", 0x04E),
    gfx125x_processor_info(
        "gfx1250",
        0x049,
        elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_SRAMECC_ANY_V4,
        processor_flags=AMDGPU_PROCESSOR_INFO_FLAG_CLUSTER_LAUNCH_STATE,
        lds_bank_service_models=(
            AMDGPU_LDS_BANK_SERVICE_MODELS_WAVE32_B128_QUAD_PHASES
        ),
        asic_revision_semantics=(
            AmdgpuProcessorAsicRevisionSemantics(
                value=0,
                instruction_constraints=(
                    AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT
                    | AMDGPU_INSTRUCTION_CONSTRAINT_DS_ADDTID_ADDRESS_MATERIALIZATION
                    | AMDGPU_INSTRUCTION_CONSTRAINT_CLUSTER_MULTICAST_MASK_PRESERVATION
                    | AMDGPU_INSTRUCTION_CONSTRAINT_TENSOR_MULTICAST_MASK_PRESERVATION
                    | AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K64_SCALE_PREFIX
                    | AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K128_SPLIT
                    | AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_F4_32X16_SPLIT
                    | AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_SCALE_ENCODING
                    | AMDGPU_INSTRUCTION_CONSTRAINT_SWMMAC_LOW_PRECISION_LOWERING
                    | AMDGPU_INSTRUCTION_CONSTRAINT_INTEGER_MATRIX_COEXECUTION_SPACING
                ),
                kernel_metadata_extensions=((".gfx1250_revision", "A0"),),
            ),
            AmdgpuProcessorAsicRevisionSemantics(
                value=1,
                kernel_metadata_extensions=((".gfx1250_revision", "B0"),),
            ),
        ),
    ),
    gfx125x_processor_info(
        "gfx1251",
        0x05A,
        descriptor_set_key="amdgpu.rdna4.gfx1251.core",
        elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_SRAMECC_ANY_V4,
    ),
    processor_info(
        "gfx1310",
        0x050,
        default_wavefront_size=32,
        kernel_descriptor=AMDGPU_KERNEL_DESCRIPTOR_INFO_PACKED_WORKITEM_ID,
    ),
    gfx9_10_processor_info(
        "gfx9-generic",
        0x051,
        elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_ANY_V4,
        elf_generic_version=generic_code_object_current_version("gfx9-generic"),
    ),
    gfx9_10_processor_info(
        "gfx10-1-generic",
        0x052,
        elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_ANY_V4,
        elf_generic_version=generic_code_object_current_version("gfx10-1-generic"),
        default_wavefront_size=32,
    ),
    gfx9_10_processor_info(
        "gfx10-3-generic",
        0x053,
        elf_generic_version=generic_code_object_current_version("gfx10-3-generic"),
        default_wavefront_size=32,
    ),
    rdna3_processor_info(
        "gfx11-generic",
        0x054,
        descriptor_set_key="amdgpu.gfx11.generic.core",
        elf_generic_version=generic_code_object_current_version("gfx11-generic"),
        scheduling_bits=AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR,
    ),
    rdna4_processor_info(
        "gfx12-generic",
        0x059,
        descriptor_set_key="amdgpu.gfx12.generic.core",
        elf_generic_version=generic_code_object_current_version("gfx12-generic"),
    ),
    processor_info(
        "gfx9-4-generic",
        0x05F,
        flags=AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
        descriptor_set_key="amdgpu.gfx9_4.generic.core",
        elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_SRAMECC_ANY_V4,
        elf_generic_version=generic_code_object_current_version("gfx9-4-generic"),
        kernel_descriptor=AMDGPU_KERNEL_DESCRIPTOR_INFO_CDNA_GFX9,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC,
        scheduling_bits=AMDGPU_PROCESSOR_SCHEDULING_CDNA_FIXED_WAIT_STATES,
        max_workgroup_storage_bytes=AMDGPU_DEFAULT_MAX_WORKGROUP_STORAGE_BYTES,
        occupancy=AMDGPU_OCCUPANCY_GFX9_4_GENERIC,
    ),
    gfx125x_processor_info(
        "gfx12-5-generic",
        0x05B,
        descriptor_set_key="amdgpu.gfx12_5.generic.core",
        elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_SRAMECC_ANY_V4,
        elf_generic_version=generic_code_object_current_version("gfx12-5-generic"),
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12_5_GENERIC,
    ),
)


AMDGPU_TARGET_RECORD_INFOS: tuple[AmdgpuTargetRecordInfo, ...] = (
    AmdgpuTargetRecordInfo(
        processor="gfx942",
        enum_value=1,
        doc="CDNA 3 gfx942 target row.",
        default_for_descriptor_set=True,
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx950",
        enum_value=2,
        doc="CDNA 4 gfx950 target row.",
        default_for_descriptor_set=True,
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx1100",
        enum_value=3,
        doc="RDNA 3 gfx1100 target row.",
        default_for_descriptor_set=True,
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx1200",
        enum_value=4,
        doc="RDNA 4 gfx1200 target row.",
        default_for_descriptor_set=True,
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx1250",
        enum_value=5,
        doc="RDNA 4 gfx1250 target row.",
        default_for_descriptor_set=True,
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx1150",
        enum_value=6,
        doc="RDNA 3.5 gfx1150 target row.",
        default_for_descriptor_set=True,
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx11-generic",
        enum_value=7,
        doc="GFX11 generic code-object target row.",
        default_for_descriptor_set=True,
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx12-generic",
        enum_value=8,
        doc="GFX12 generic code-object target row.",
        default_for_descriptor_set=True,
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx12-5-generic",
        enum_value=9,
        doc="GFX12.5 generic code-object target row.",
        default_for_descriptor_set=True,
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx940",
        enum_value=10,
        doc="CDNA 3 gfx940 target row.",
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx941",
        enum_value=11,
        doc="CDNA 3 gfx941 target row.",
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx1101",
        enum_value=12,
        doc="RDNA 3 gfx1101 target row.",
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx1102",
        enum_value=13,
        doc="RDNA 3 gfx1102 target row.",
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx1103",
        enum_value=14,
        doc="RDNA 3 gfx1103 target row.",
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx1151",
        enum_value=15,
        doc="RDNA 3.5 gfx1151 target row.",
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx1152",
        enum_value=16,
        doc="RDNA 3.5 gfx1152 target row.",
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx1153",
        enum_value=17,
        doc="RDNA 3.5 gfx1153 target row.",
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx1170",
        enum_value=18,
        doc="RDNA 3.5 gfx1170 target row.",
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx1171",
        enum_value=19,
        doc="RDNA 3.5 gfx1171 target row.",
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx1172",
        enum_value=20,
        doc="RDNA 3.5 gfx1172 target row.",
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx1201",
        enum_value=21,
        doc="RDNA 4 gfx1201 target row.",
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx1251",
        enum_value=22,
        doc="RDNA 4 gfx1251 target row.",
        default_for_descriptor_set=True,
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx9-4-generic",
        enum_value=23,
        doc="GFX9.4 generic code-object target row.",
        default_for_descriptor_set=True,
    ),
)


def sorted_descriptor_set_infos() -> tuple[AmdgpuDescriptorSetInfo, ...]:
    return tuple(sorted(AMDGPU_DESCRIPTOR_SET_INFOS, key=lambda info: info.key))


def amdgpu_descriptor_set_ordinal(key: str) -> int:
    for ordinal, info in enumerate(sorted_descriptor_set_infos()):
        if info.key == key:
            return ordinal
    raise ValueError(f"unknown AMDGPU descriptor set '{key}'")


def sorted_processor_infos() -> tuple[AmdgpuProcessorInfo, ...]:
    return tuple(sorted(AMDGPU_PROCESSOR_INFOS, key=lambda info: info.processor))


def amdgpu_processor_ordinal(processor: str) -> int:
    for ordinal, info in enumerate(sorted_processor_infos()):
        if info.processor == processor:
            return ordinal
    raise ValueError(f"unknown AMDGPU processor '{processor}'")


def amdgpu_processor_occupancy_model(
    info: AmdgpuProcessorInfo, wave_size: int
) -> AmdgpuOccupancyModelInfo | None:
    if wave_size == 32:
        return info.occupancy.wave32
    if wave_size == 64:
        return info.occupancy.wave64
    raise ValueError(f"unsupported AMDGPU wave size {wave_size}")


def sorted_target_record_infos() -> tuple[AmdgpuTargetRecordInfo, ...]:
    return tuple(sorted(AMDGPU_TARGET_RECORD_INFOS, key=lambda info: info.enum_value))


def amdgpu_processor_info_by_name(processor: str) -> AmdgpuProcessorInfo | None:
    for info in AMDGPU_PROCESSOR_INFOS:
        if info.processor == processor:
            return info
    return None


def amdgpu_processor_default_instruction_constraints(
    info: AmdgpuProcessorInfo,
) -> int:
    constraints = info.instructions.base_constraints
    if info.default_asic_revision is not None:
        revision = next(
            revision
            for revision in info.asic_revisions
            if revision.value == info.default_asic_revision
        )
        constraints |= revision.instruction_constraints
    return constraints


def amdgpu_lds_bank_service_model_sets(
    processors: Sequence[AmdgpuProcessorInfo],
) -> tuple[tuple[str, ...], ...]:
    """Returns the interned non-empty model sets selected by target rows."""

    model_sets = {
        model_keys
        for processor in processors
        for model_keys in (
            processor.features.lds_bank_service_models,
            *(
                revision.lds_bank_service_models
                for revision in processor.asic_revisions
            ),
        )
        if model_keys
    }
    return tuple(sorted(model_sets))


def amdgpu_generic_code_object_compatibility_info(
    exact_processor: str,
) -> AmdgpuExactTargetInfo | None:
    for info in AMDGPU_EXACT_TARGET_INFOS:
        if info.exact_processor != exact_processor:
            continue
        if info.generic_introduction_version != 0:
            return info
        return None
    return None


def validate_amdgpu_code_object_processor_rows(
    processors: Sequence[AmdgpuProcessorInfo],
) -> None:
    processors_by_name = {info.processor: info for info in processors}
    required_processor_names = {
        info.exact_processor for info in AMDGPU_EXACT_TARGET_INFOS
    } | {info.processor for info in AMDGPU_GENERIC_CODE_OBJECT_INFOS}
    missing_processor_names = sorted(
        required_processor_names - processors_by_name.keys()
    )
    if missing_processor_names:
        raise ValueError(
            "AMDGPU target-info table is missing canonical code-object "
            f"processors: {', '.join(missing_processor_names)}"
        )

    expected_generic_processor_names = {
        info.processor for info in AMDGPU_GENERIC_CODE_OBJECT_INFOS
    }
    actual_generic_processor_names = {
        info.processor for info in processors if info.elf.generic_version != 0
    }
    if actual_generic_processor_names != expected_generic_processor_names:
        missing_generic_processor_names = sorted(
            expected_generic_processor_names - actual_generic_processor_names
        )
        unexpected_processor_names = sorted(
            actual_generic_processor_names - expected_generic_processor_names
        )
        raise ValueError(
            "AMDGPU target-info generic code-object processors disagree with "
            f"the canonical map; missing: "
            f"{', '.join(missing_generic_processor_names) or 'none'}; "
            f"unexpected: {', '.join(unexpected_processor_names) or 'none'}"
        )

    for generic_info in AMDGPU_GENERIC_CODE_OBJECT_INFOS:
        processor = processors_by_name[generic_info.processor]
        if processor.elf.generic_version != generic_info.current_version:
            raise ValueError(
                f"AMDGPU generic processor {generic_info.processor} has "
                f"target-info version {processor.elf.generic_version}, "
                f"expected canonical version {generic_info.current_version}"
            )


def validate_amdgpu_target_id_processor_rows(
    processors: Sequence[AmdgpuProcessorInfo],
) -> None:
    expected_feature_names = set(AMDGPU_TARGET_ID_FEATURE_ORDER)
    mapped_feature_names = set(AMDGPU_TARGET_ID_FEATURE_SUPPORT_FLAGS_BY_NAME)
    if mapped_feature_names != expected_feature_names:
        raise ValueError(
            "Loom AMDGPU target-ID feature projection disagrees with the "
            "canonical target map"
        )

    processors_by_name = {info.processor: info for info in processors}
    required_processor_names = {
        info.exact_processor for info in AMDGPU_EXACT_TARGET_INFOS
    } | {info.processor for info in AMDGPU_GENERIC_CODE_OBJECT_INFOS}
    missing_processor_names = sorted(
        required_processor_names - processors_by_name.keys()
    )
    if missing_processor_names:
        raise ValueError(
            "AMDGPU target-info table is missing canonical target-ID "
            f"processors: {', '.join(missing_processor_names)}"
        )

    for processor in required_processor_names:
        actual = processors_by_name[processor].target_id
        expected = amdgpu_processor_target_id_info(processor)
        if actual != expected:
            raise ValueError(
                f"AMDGPU processor {processor} target-ID qualification "
                "disagrees with the canonical target map"
            )


def validate_amdgpu_processor_revision_rows(
    processors: Sequence[AmdgpuProcessorInfo],
) -> None:
    """Validates Loom semantics attached to canonical physical revisions."""

    lds_model_infos_by_key = amdgpu_lds_bank_service_model_info_by_key()
    for processor in processors:
        constraints = processor.instructions.base_constraints
        validate_amdgpu_lds_bank_service_model_selection(
            f"AMDGPU processor {processor.processor}",
            processor.features.lds_bank_service_models,
            lds_model_infos_by_key,
        )
        for revision in processor.asic_revisions:
            constraints |= revision.instruction_constraints
            validate_amdgpu_lds_bank_service_model_selection(
                f"AMDGPU processor {processor.processor} ASIC revision {revision.name}",
                revision.lds_bank_service_models,
                lds_model_infos_by_key,
            )
            previous_metadata_key: str | None = None
            for key, value in revision.kernel_metadata_extensions:
                if (
                    not key
                    or key[0] != "."
                    or any(
                        ord(character) <= ord(" ") or character in ("'", "\\")
                        for character in key
                    )
                ):
                    raise ValueError(
                        f"AMDGPU processor {processor.processor} ASIC revision "
                        f"{revision.name} has invalid metadata key {key!r}"
                    )
                if not value or any(
                    ord(character) <= ord(" ") or character in ("'", "\\")
                    for character in value
                ):
                    raise ValueError(
                        f"AMDGPU processor {processor.processor} ASIC revision "
                        f"{revision.name} has invalid metadata value {value!r}"
                    )
                if previous_metadata_key is not None and previous_metadata_key >= key:
                    raise ValueError(
                        f"AMDGPU processor {processor.processor} ASIC revision "
                        f"{revision.name} metadata keys are not unique and sorted"
                    )
                if key in AMDGPU_STANDARD_KERNEL_METADATA_KEYS:
                    raise ValueError(
                        f"AMDGPU processor {processor.processor} ASIC revision "
                        f"{revision.name} metadata key {key!r} replaces a "
                        "standard kernel metadata field"
                    )
                previous_metadata_key = key
        unknown_constraints = constraints & ~AMDGPU_INSTRUCTION_CONSTRAINT_KNOWN_BITS
        if unknown_constraints:
            raise ValueError(
                f"AMDGPU processor {processor.processor} uses unknown "
                f"instruction constraints 0x{unknown_constraints:x}"
            )
        if processor.asic_revisions:
            portable_model_keys = set(
                processor.asic_revisions[0].lds_bank_service_models
            )
            for revision in processor.asic_revisions[1:]:
                portable_model_keys.intersection_update(
                    revision.lds_bank_service_models
                )
            portable_models = tuple(
                model_key
                for model_key in processor.asic_revisions[0].lds_bank_service_models
                if model_key in portable_model_keys
            )
            if processor.features.lds_bank_service_models != portable_models:
                raise ValueError(
                    f"AMDGPU processor {processor.processor} LDS bank-service "
                    "models do not match the ASIC-revision intersection"
                )
        canonical_exact_target = exact_target_info(processor.processor)
        canonical_revisions = (
            tuple(
                (revision.value, revision.name)
                for revision in canonical_exact_target.asic_revisions
            )
            if canonical_exact_target is not None
            else ()
        )
        actual_revisions = tuple(
            (revision.value, revision.name) for revision in processor.asic_revisions
        )
        if actual_revisions != canonical_revisions:
            raise ValueError(
                f"AMDGPU processor {processor.processor} ASIC revisions "
                "disagree with the canonical target map"
            )
        canonical_default_revision = (
            canonical_exact_target.default_asic_revision
            if canonical_exact_target is not None
            else None
        )
        if processor.default_asic_revision != canonical_default_revision:
            raise ValueError(
                f"AMDGPU processor {processor.processor} default ASIC revision "
                "disagrees with the canonical target map"
            )


def _occupancy_rounded_units(units: int, granularity: int) -> int:
    return ((units + granularity - 1) // granularity) * granularity


def _occupancy_capacity(pool_units: int, granularity: int, units: int) -> int:
    if units == 0:
        return pool_units
    return pool_units // _occupancy_rounded_units(units, granularity)


def _validate_portable_occupancy_model(
    generic_processor: str,
    wave_size: int,
    generic_model: AmdgpuOccupancyModelInfo,
    member_models: tuple[AmdgpuOccupancyModelInfo, ...],
) -> None:
    if generic_model.max_waves_per_simd > min(
        model.max_waves_per_simd for model in member_models
    ):
        raise ValueError(
            f"AMDGPU generic processor {generic_processor} wave{wave_size} "
            "occupancy overstates resident waves"
        )
    if generic_model.domain.simd_count != member_models[0].domain.simd_count or any(
        model.domain.simd_count != generic_model.domain.simd_count
        for model in member_models[1:]
    ):
        raise ValueError(
            f"AMDGPU generic processor {generic_processor} wave{wave_size} "
            "occupancy has divergent SIMD topology"
        )
    if any(model.resources != generic_model.resources for model in member_models):
        raise ValueError(
            f"AMDGPU generic processor {generic_processor} wave{wave_size} "
            "occupancy has divergent coupled resources"
        )
    if generic_model.domain.max_barrier_workgroup_count > min(
        model.domain.max_barrier_workgroup_count for model in member_models
    ):
        raise ValueError(
            f"AMDGPU generic processor {generic_processor} wave{wave_size} "
            "occupancy overstates barrier workgroups"
        )

    maximum_local_memory_bytes = max(
        model.domain.local_memory_bytes for model in member_models
    )
    for local_memory_bytes in range(1, maximum_local_memory_bytes + 1):
        generic_capacity = _occupancy_capacity(
            generic_model.domain.local_memory_bytes,
            generic_model.domain.local_memory_allocation_granularity,
            local_memory_bytes,
        )
        for member_model in member_models:
            member_capacity = _occupancy_capacity(
                member_model.domain.local_memory_bytes,
                member_model.domain.local_memory_allocation_granularity,
                local_memory_bytes,
            )
            if generic_capacity > member_capacity:
                raise ValueError(
                    f"AMDGPU generic processor {generic_processor} wave{wave_size} "
                    f"occupancy overstates local-memory capacity at "
                    f"{local_memory_bytes} bytes"
                )

    member_register_classes = tuple(
        {
            register_class.register_class: register_class
            for register_class in model.register_classes
        }
        for model in member_models
    )
    generic_register_classes = {
        register_class.register_class: register_class
        for register_class in generic_model.register_classes
    }
    limiting_register_classes = {
        register_class.register_class
        for model in member_models
        for register_class in model.register_classes
        if register_class.limits_occupancy
    }
    for register_class in sorted(limiting_register_classes):
        generic_register_class = generic_register_classes.get(register_class)
        if (
            generic_register_class is None
            or not generic_register_class.limits_occupancy
        ):
            raise ValueError(
                f"AMDGPU generic processor {generic_processor} wave{wave_size} "
                f"occupancy does not model limiting register class "
                f"{register_class}"
            )
        member_rows = tuple(
            member_map.get(register_class) for member_map in member_register_classes
        )
        if any(member_row is None for member_row in member_rows):
            raise ValueError(
                f"AMDGPU generic processor {generic_processor} wave{wave_size} "
                f"occupancy register class {register_class} is not common"
            )
        maximum_units = max(
            member_row.pool_units
            for member_row in member_rows
            if member_row is not None
        )
        for units in range(1, maximum_units + 1):
            generic_capacity = min(
                generic_model.max_waves_per_simd,
                _occupancy_capacity(
                    generic_register_class.pool_units,
                    generic_register_class.allocation_granularity,
                    units,
                ),
            )
            for member_model, member_row in zip(
                member_models, member_rows, strict=True
            ):
                if member_row is None or not member_row.limits_occupancy:
                    continue
                member_capacity = min(
                    member_model.max_waves_per_simd,
                    _occupancy_capacity(
                        member_row.pool_units,
                        member_row.allocation_granularity,
                        units,
                    ),
                )
                if generic_capacity > member_capacity:
                    raise ValueError(
                        f"AMDGPU generic processor {generic_processor} "
                        f"wave{wave_size} occupancy overstates "
                        f"{register_class} capacity at {units} units"
                    )


def validate_amdgpu_generic_contracts(
    processors: Sequence[AmdgpuProcessorInfo],
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
) -> None:
    processors_by_name = {info.processor: info for info in processors}
    descriptor_sets_by_key = {info.key: info for info in descriptor_sets}
    for descriptor_set in descriptor_sets:
        if not descriptor_set.member_generator_targets:
            continue
        generic_processors = tuple(
            info
            for info in processors
            if info.descriptor_set.key == descriptor_set.key
            and info.processor.endswith("-generic")
        )
        if len(generic_processors) != 1:
            raise ValueError(
                f"AMDGPU generic descriptor set {descriptor_set.key} must have "
                "one processor"
            )
        generic_processor = generic_processors[0]
        exact_members = tuple(
            processors_by_name[compatibility.exact_processor]
            for compatibility in AMDGPU_EXACT_TARGET_INFOS
            if (
                compatibility.code_object_processor == generic_processor.processor
                and compatibility.generic_introduction_version
                <= generic_processor.elf.generic_version
            )
        )
        if not exact_members:
            raise ValueError(
                f"AMDGPU generic processor {generic_processor.processor} has "
                "no exact members"
            )

        exact_member_generator_targets = tuple(
            sorted(
                {
                    descriptor_sets_by_key[member.descriptor_set.key].generator_target
                    for member in exact_members
                }
            )
        )
        if exact_member_generator_targets != descriptor_set.member_generator_targets:
            raise ValueError(
                f"AMDGPU generic processor {generic_processor.processor} "
                "descriptor membership does not match the target map"
            )
        if any(
            member.descriptor_set.key == generic_processor.descriptor_set.key
            for member in exact_members
        ):
            raise ValueError(
                f"AMDGPU generic processor {generic_processor.processor} "
                "aliases an exact descriptor contract"
            )

        exact_member_descriptor_sets = tuple(
            descriptor_sets_by_key[member.descriptor_set.key]
            for member in exact_members
        )
        portable_descriptor_flags = exact_member_descriptor_sets[0].flags
        for member_descriptor_set in exact_member_descriptor_sets[1:]:
            portable_descriptor_flags &= member_descriptor_set.flags
        if descriptor_set.flags != portable_descriptor_flags:
            raise ValueError(
                f"AMDGPU generic descriptor set {descriptor_set.key} flags do "
                "not match the member intersection"
            )
        vector_memory_encoding = exact_member_descriptor_sets[
            0
        ].vector_memory.cache_policy_encoding
        if any(
            member_descriptor_set.vector_memory.cache_policy_encoding
            != vector_memory_encoding
            for member_descriptor_set in exact_member_descriptor_sets[1:]
        ):
            raise ValueError(
                f"AMDGPU generic descriptor set {descriptor_set.key} members "
                "have divergent vector-memory cache-policy encodings"
            )
        if descriptor_set.vector_memory.cache_policy_encoding != vector_memory_encoding:
            raise ValueError(
                f"AMDGPU generic descriptor set {descriptor_set.key} "
                "vector-memory cache-policy encoding does not match every member"
            )
        member_cache_swizzles = {
            member_descriptor_set.buffer_resource.cache_swizzle
            for member_descriptor_set in exact_member_descriptor_sets
        }
        portable_cache_swizzle = (
            next(iter(member_cache_swizzles))
            if len(member_cache_swizzles) == 1
            else AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE
        )
        if descriptor_set.buffer_resource.cache_swizzle != portable_cache_swizzle:
            raise ValueError(
                f"AMDGPU generic descriptor set {descriptor_set.key} buffer "
                "resource cache swizzle is not portable across every member"
            )

        portable_flags = exact_members[0].flags
        for member in exact_members[1:]:
            portable_flags &= member.flags
        if generic_processor.flags != portable_flags:
            raise ValueError(
                f"AMDGPU generic processor {generic_processor.processor} flags "
                "do not match the member intersection"
            )
        portable_elf_feature_flags = exact_members[0].elf.feature_flags
        for member in exact_members[1:]:
            portable_elf_feature_flags &= member.elf.feature_flags
        if generic_processor.elf.feature_flags != portable_elf_feature_flags:
            raise ValueError(
                f"AMDGPU generic processor {generic_processor.processor} ELF "
                "feature flags do not match the member intersection"
            )
        if any(
            member.wavefront != generic_processor.wavefront for member in exact_members
        ):
            raise ValueError(
                f"AMDGPU generic processor {generic_processor.processor} "
                "wavefront facts do not match every member"
            )
        if any(
            member.kernel_descriptor != generic_processor.kernel_descriptor
            for member in exact_members
        ):
            raise ValueError(
                f"AMDGPU generic processor {generic_processor.processor} "
                "kernel descriptor facts do not match every member"
            )
        if any(
            member.kernel_entry != generic_processor.kernel_entry
            for member in exact_members
        ):
            raise ValueError(
                f"AMDGPU generic processor {generic_processor.processor} "
                "kernel entry facts do not match every member"
            )

        portable_instruction_constraints = 0
        for member in exact_members:
            portable_instruction_constraints |= (
                amdgpu_processor_default_instruction_constraints(member)
            )
        if (
            amdgpu_processor_default_instruction_constraints(generic_processor)
            != portable_instruction_constraints
        ):
            raise ValueError(
                f"AMDGPU generic processor {generic_processor.processor} "
                "instruction constraints do not match the union of member "
                "restrictions"
            )

        portable_lds_model_keys = set(exact_members[0].features.lds_bank_service_models)
        for member in exact_members[1:]:
            portable_lds_model_keys.intersection_update(
                member.features.lds_bank_service_models
            )
        portable_lds_bank_service_models = tuple(
            model_key
            for model_key in exact_members[0].features.lds_bank_service_models
            if model_key in portable_lds_model_keys
        )
        if (
            generic_processor.features.lds_bank_service_models
            != portable_lds_bank_service_models
        ):
            raise ValueError(
                f"AMDGPU generic processor {generic_processor.processor} LDS "
                "bank-service models do not match the member intersection"
            )

        scheduling_bits = 0
        for member in exact_members:
            scheduling_bits |= member.features.scheduling
        if generic_processor.features.scheduling != scheduling_bits:
            raise ValueError(
                f"AMDGPU generic processor {generic_processor.processor} "
                "scheduling facts do not cover every member"
            )
        portable_matrix_features = set(
            AMDGPU_MATRIX_FEATURES_BY_PROFILE[exact_members[0].features.matrix]
        )
        for member in exact_members[1:]:
            portable_matrix_features.intersection_update(
                AMDGPU_MATRIX_FEATURES_BY_PROFILE[member.features.matrix]
            )
        excluded_matrix_features = set(
            AMDGPU_GENERIC_MATRIX_FEATURE_EXCLUSIONS.get(
                generic_processor.processor, ()
            )
        )
        if not excluded_matrix_features.issubset(portable_matrix_features):
            raise ValueError(
                f"AMDGPU generic processor {generic_processor.processor} "
                "excludes matrix features absent from its members"
            )
        expected_matrix_features = portable_matrix_features - excluded_matrix_features
        if (
            set(AMDGPU_MATRIX_FEATURES_BY_PROFILE[generic_processor.features.matrix])
            != expected_matrix_features
        ):
            raise ValueError(
                f"AMDGPU generic processor {generic_processor.processor} "
                "matrix features do not match its portable contract"
            )
        if generic_processor.limits.max_workgroup_storage_bytes != min(
            member.limits.max_workgroup_storage_bytes for member in exact_members
        ):
            raise ValueError(
                f"AMDGPU generic processor {generic_processor.processor} "
                "workgroup storage does not match the portable minimum"
            )

        for wave_size in (32, 64):
            generic_model = amdgpu_processor_occupancy_model(
                generic_processor, wave_size
            )
            member_models = tuple(
                amdgpu_processor_occupancy_model(member, wave_size)
                for member in exact_members
            )
            if generic_model is None:
                if any(member_model is not None for member_model in member_models):
                    raise ValueError(
                        f"AMDGPU generic processor "
                        f"{generic_processor.processor} lacks wave{wave_size} "
                        "occupancy shared by its members"
                    )
                continue
            if any(member_model is None for member_model in member_models):
                raise ValueError(
                    f"AMDGPU generic processor {generic_processor.processor} "
                    f"wave{wave_size} occupancy is absent from a member"
                )
            _validate_portable_occupancy_model(
                generic_processor.processor,
                wave_size,
                generic_model,
                tuple(
                    member_model
                    for member_model in member_models
                    if member_model is not None
                ),
            )


def amdgpu_target_record_info_for_processor(
    processor: str,
) -> AmdgpuTargetRecordInfo | None:
    for info in AMDGPU_TARGET_RECORD_INFOS:
        if info.processor == processor:
            return info
    return None


def amdgpu_default_target_record_info_for_descriptor_set(
    descriptor_set_key: str,
) -> AmdgpuTargetRecordInfo | None:
    for info in AMDGPU_TARGET_RECORD_INFOS:
        processor_info = amdgpu_processor_info_by_name(info.processor)
        if (
            info.default_for_descriptor_set
            and processor_info is not None
            and processor_info.descriptor_set.key == descriptor_set_key
        ):
            return info
    return None


def amdgpu_descriptor_set_info_by_generator_target(
    generator_target: str,
) -> AmdgpuDescriptorSetInfo:
    for info in AMDGPU_DESCRIPTOR_SET_INFOS:
        if info.generator_target == generator_target:
            return info
    raise ValueError(f"unknown AMDGPU descriptor generator target '{generator_target}'")


def amdgpu_descriptor_set_storage_info_by_generator_target(
    generator_target: str,
) -> AmdgpuDescriptorSetInfo:
    info = amdgpu_descriptor_set_info_by_generator_target(generator_target)
    if info.storage_generator_target is None:
        return info
    storage_info = amdgpu_descriptor_set_info_by_generator_target(
        info.storage_generator_target
    )
    if storage_info.storage_generator_target is not None:
        raise ValueError(
            f"AMDGPU descriptor generator target '{generator_target}' uses "
            f"view-only target '{storage_info.generator_target}' as storage"
        )
    return storage_info


def amdgpu_descriptor_set_view_infos_by_storage_generator_target(
    storage_generator_target: str,
) -> tuple[AmdgpuDescriptorSetInfo, ...]:
    storage_info = amdgpu_descriptor_set_info_by_generator_target(
        storage_generator_target
    )
    if storage_info.storage_generator_target is not None:
        raise ValueError(
            f"AMDGPU descriptor generator target '{storage_generator_target}' "
            "is a view target, not a storage target"
        )
    return tuple(
        sorted(
            (
                info
                for info in AMDGPU_DESCRIPTOR_SET_INFOS
                if info.storage_generator_target == storage_generator_target
            ),
            key=lambda info: info.key,
        )
    )


def validate_amdgpu_descriptor_set_isa_xml(
    info: AmdgpuDescriptorSetInfo,
    spec: AmdgpuIsaArchitectureInfo,
) -> None:
    for isa_info in info.isa_infos:
        if (
            spec.architecture_name == isa_info.isa_architecture_name
            and spec.architecture_id == isa_info.isa_architecture_id
        ):
            return
    expected_members = tuple(
        f"{isa_info.isa_architecture_name} architecture id "
        f"{isa_info.isa_architecture_id}"
        for isa_info in info.isa_infos
    )
    expected = expected_members[0]
    if len(expected_members) > 1:
        expected = f"one of [{', '.join(expected_members)}]"
    raise ValueError(
        f"{spec.source_name}: AMDGPU descriptor set {info.key} expects "
        f"{expected}, "
        f"found {spec.architecture_name} architecture id {spec.architecture_id}"
    )
