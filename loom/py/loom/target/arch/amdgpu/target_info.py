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

from dataclasses import dataclass
from typing import Protocol

from build_tools.amdgpu.target_map_data import generic_code_object_current_version

from loom.dialect.cache import CacheScope, CacheTemporal

AMDGPU_AMDHSA_TARGET_TRIPLE = "amdgcn-amd-amdhsa"
AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE = (2**16) - 1

AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE = "none"
AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9 = "gfx9"
AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11 = "gfx11"
AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12 = "gfx12"
AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125 = "gfx125"

AMDGPU_MATRIX_FEATURE_PROFILE_NONE = "none"
AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908 = "mfma_gfx908"
AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A = "mfma_gfx90a"
AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940 = "mfma_gfx940"
AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950 = "mfma_gfx950"
AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11 = "wmma_gfx11"
AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12 = "wmma_gfx12"
AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250 = "wmma_gfx1250"
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
        "mfma_gfx940_xf32",
        "smfmac_gfx940",
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
}

AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION = 1 << 0
AMDGPU_PROCESSOR_INFO_FLAG_CLUSTER_LAUNCH_STATE = 1 << 1
AMDGPU_PROCESSOR_INFO_KNOWN_FLAGS = (
    AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION
    | AMDGPU_PROCESSOR_INFO_FLAG_CLUSTER_LAUNCH_STATE
)
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
class AmdgpuDescriptorSetInfo:
    generator_target: str
    key: str
    isa_xml_key: str
    isa_architecture_name: str
    isa_architecture_id: int
    flags: int
    storage_generator_target: str | None = None
    buffer_resource: AmdgpuDescriptorSetBufferResourceInfo = (
        AmdgpuDescriptorSetBufferResourceInfo()
    )
    vector_memory: AmdgpuDescriptorSetVectorMemoryInfo = (
        AmdgpuDescriptorSetVectorMemoryInfo()
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
class AmdgpuProcessorFeatureInfo:
    matrix: str = AMDGPU_MATRIX_FEATURE_PROFILE_NONE
    scheduling: int = 0


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
    wavefront: AmdgpuProcessorWavefrontInfo
    kernel_descriptor: AmdgpuProcessorKernelDescriptorInfo
    features: AmdgpuProcessorFeatureInfo = AmdgpuProcessorFeatureInfo()
    limits: AmdgpuProcessorLimitInfo = AmdgpuProcessorLimitInfo()
    occupancy: AmdgpuProcessorOccupancyInfo = AmdgpuProcessorOccupancyInfo()


AMDGPU_OCCUPANCY_NONE = AmdgpuProcessorOccupancyInfo()


AMDGPU_KERNEL_DESCRIPTOR_INFO_NONE = AmdgpuProcessorKernelDescriptorInfo()
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
    matrix_feature_profile: str = AMDGPU_MATRIX_FEATURE_PROFILE_NONE,
    scheduling_bits: int = 0,
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
    return AmdgpuProcessorInfo(
        processor=processor,
        flags=flags,
        descriptor_set=AmdgpuProcessorDescriptorSetInfo(key=descriptor_set_key),
        elf=AmdgpuProcessorElfInfo(
            machine_flags=elf_machine_flags,
            feature_flags=elf_feature_flags,
            generic_version=elf_generic_version,
        ),
        wavefront=AmdgpuProcessorWavefrontInfo(default_size=default_wavefront_size),
        kernel_descriptor=kernel_descriptor,
        features=AmdgpuProcessorFeatureInfo(
            matrix=matrix_feature_profile,
            scheduling=scheduling_bits,
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
    elf_feature_flags: int = 0,
    elf_generic_version: int = 0,
    scheduling_bits: int = 0,
    occupancy: AmdgpuProcessorOccupancyInfo = AMDGPU_OCCUPANCY_RDNA_1024,
) -> AmdgpuProcessorInfo:
    return processor_info(
        processor=processor,
        elf_machine_flags=elf_machine_flags,
        flags=AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
        descriptor_set_key="amdgpu.rdna3.core",
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
    flags: int = 0,
    matrix_feature_profile: str = AMDGPU_MATRIX_FEATURE_PROFILE_NONE,
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
    elf_feature_flags: int = 0,
    elf_generic_version: int = 0,
) -> AmdgpuProcessorInfo:
    return processor_info(
        processor=processor,
        flags=AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
        descriptor_set_key="amdgpu.rdna4.core",
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
    elf_feature_flags: int = 0,
    elf_generic_version: int = 0,
    processor_flags: int = 0,
) -> AmdgpuProcessorInfo:
    return processor_info(
        processor=processor,
        flags=AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION | processor_flags,
        descriptor_set_key="amdgpu.rdna4.gfx125x.core",
        elf_machine_flags=elf_machine_flags,
        elf_feature_flags=elf_feature_flags,
        elf_generic_version=elf_generic_version,
        default_wavefront_size=32,
        kernel_descriptor=AMDGPU_KERNEL_DESCRIPTOR_INFO_RDNA4_GFX125,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250,
        scheduling_bits=AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU,
        max_workgroup_storage_bytes=AMDGPU_GFX125X_MAX_WORKGROUP_STORAGE_BYTES,
        occupancy=AMDGPU_OCCUPANCY_GFX125X,
    )


AMDGPU_DESCRIPTOR_SET_INFOS: tuple[AmdgpuDescriptorSetInfo, ...] = (
    AmdgpuDescriptorSetInfo(
        generator_target="cdna3",
        key="amdgpu.cdna3.core",
        isa_xml_key="cdna3",
        isa_architecture_name="AMD CDNA 3",
        isa_architecture_id=2,
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1,
        ),
    ),
    AmdgpuDescriptorSetInfo(
        generator_target="rdna4_gfx125x",
        key="amdgpu.rdna4.gfx125x.core",
        isa_xml_key="rdna4",
        isa_architecture_name="AMD RDNA 4",
        isa_architecture_id=10,
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAGS_RDNA_VOPD,
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH,
        ),
    ),
    AmdgpuDescriptorSetInfo(
        generator_target="rdna3",
        key="amdgpu.rdna3.core",
        isa_xml_key="rdna3",
        isa_architecture_name="AMD RDNA 3",
        isa_architecture_id=8,
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAGS_RDNA_VOPD,
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC,
        ),
    ),
    AmdgpuDescriptorSetInfo(
        generator_target="rdna3_5",
        key="amdgpu.rdna3_5.core",
        isa_xml_key="rdna3_5",
        isa_architecture_name="AMD RDNA 3.5",
        isa_architecture_id=9,
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC,
        ),
    ),
    AmdgpuDescriptorSetInfo(
        generator_target="rdna4",
        key="amdgpu.rdna4.core",
        isa_xml_key="rdna4",
        isa_architecture_name="AMD RDNA 4",
        isa_architecture_id=10,
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAGS_RDNA_VOPD,
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH,
        ),
    ),
    AmdgpuDescriptorSetInfo(
        generator_target="cdna4",
        key="amdgpu.cdna4.core",
        isa_xml_key="cdna4",
        isa_architecture_name="AMD CDNA 4",
        isa_architecture_id=3,
        flags=AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
        buffer_resource=AmdgpuDescriptorSetBufferResourceInfo(
            cache_swizzle=AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT,
        ),
        vector_memory=AmdgpuDescriptorSetVectorMemoryInfo(
            cache_policy_encoding=AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1,
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
    cdna3_processor_info(
        "gfx942",
        0x04C,
        matrix_feature_profile=AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940,
        flags=AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
    ),
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
    ),
    gfx125x_processor_info(
        "gfx1251",
        0x05A,
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
        elf_generic_version=generic_code_object_current_version("gfx11-generic"),
        scheduling_bits=AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR,
    ),
    rdna4_processor_info(
        "gfx12-generic",
        0x059,
        elf_generic_version=generic_code_object_current_version("gfx12-generic"),
    ),
    processor_info(
        "gfx9-4-generic",
        0x05F,
        elf_feature_flags=AMDGPU_ELF_FEATURE_XNACK_SRAMECC_ANY_V4,
        elf_generic_version=generic_code_object_current_version("gfx9-4-generic"),
        scheduling_bits=AMDGPU_PROCESSOR_SCHEDULING_CDNA_FIXED_WAIT_STATES,
    ),
    gfx125x_processor_info(
        "gfx12-5-generic",
        0x05B,
        elf_generic_version=generic_code_object_current_version("gfx12-5-generic"),
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
        doc="RDNA 3 generic code-object target row.",
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx12-generic",
        enum_value=8,
        doc="RDNA 4 generic code-object target row.",
    ),
    AmdgpuTargetRecordInfo(
        processor="gfx12-5-generic",
        enum_value=9,
        doc="RDNA 4 gfx125x generic code-object target row.",
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
    if (
        spec.architecture_name == info.isa_architecture_name
        and spec.architecture_id == info.isa_architecture_id
    ):
        return
    raise ValueError(
        f"{spec.source_name}: AMDGPU descriptor set {info.key} expects "
        f"{info.isa_architecture_name} architecture id {info.isa_architecture_id}, "
        f"found {spec.architecture_name} architecture id {spec.architecture_id}"
    )
