# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU target-info overlay -> compact C tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[5]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from build_tools.amdgpu.target_map_data import (  # noqa: E402
    AMDGPU_PHYSICAL_TARGET_INFOS,
)

from loom.gen.support.c import c_string_arg as _c_string_arg  # noqa: E402
from loom.gen.support.c import c_string_literal as _c_string_literal  # noqa: E402
from loom.gen.support.files import write_text_file  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.gen.target.arch.amdgpu.amdgpu_config_tables import (  # noqa: E402
    write_config_tables_to_paths,
)
from loom.gen.target.arch.amdgpu.amdgpu_low_aliases import (  # noqa: E402
    write_low_aliases_to_path,
)
from loom.gen.target.arch.amdgpu.records.amdgpu_target_records import (  # noqa: E402
    write_target_record_tables_to_path,
)
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    amdgpu_descriptor_ref_keys,
)
from loom.target.arch.amdgpu.lds_bank_service import (  # noqa: E402
    AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ,
    AMDGPU_LDS_BANK_SERVICE_DIRECTION_WRITE,
    AMDGPU_LDS_BANK_SERVICE_EVIDENCE_PUBLIC_VENDOR_DOCUMENTATION,
    AMDGPU_LDS_BANK_SERVICE_EVIDENCE_SILICON_CALIBRATED_VENDOR_MODEL,
    AMDGPU_LDS_BANK_SERVICE_EVIDENCE_VENDOR_SOFTWARE_MODEL_UNVALIDATED,
    AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COALESCE_IDENTICAL_READS,
    AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COUNT_EACH,
    AmdgpuLdsBankServiceModelInfo,
    amdgpu_lds_bank_service_model_info_by_key,
    validate_amdgpu_lds_bank_service_model_coverage,
    validate_amdgpu_lds_bank_service_model_infos,
)
from loom.target.arch.amdgpu.names import (  # noqa: E402
    amdgpu_descriptor_set_ordinal_constant_name,
)
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AMDGPU_AMDHSA_TARGET_TRIPLE,
    AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE,
    AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT,
    AMDGPU_BUFFER_RESOURCE_RECORD_ENCODING_NONE,
    AMDGPU_BUFFER_RESOURCE_RECORD_ENCODINGS,
    AMDGPU_CACHE_SCOPE_KEYWORDS,
    AMDGPU_CACHE_TEMPORAL_KEYWORDS,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_OCP_FP8_NONCANONICAL_NAN,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_PACKED_BF16_ARITHMETIC,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_ARITHMETIC,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_COMPARE,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_CONVERSION,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOP3_TWO_SCALAR_SOURCES,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_DUAL_MOV_SRC2_CACHE,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_NUMERIC_MINMAX_MNEMONICS,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION,
    AMDGPU_DESCRIPTOR_SET_INFO_KNOWN_FLAGS,
    AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE,
    AMDGPU_ELF_GENERIC_VERSION_MASK_V6,
    AMDGPU_INSTRUCTION_CONSTRAINT_CLUSTER_MULTICAST_MASK_PRESERVATION,
    AMDGPU_INSTRUCTION_CONSTRAINT_DS_ADDTID_ADDRESS_MATERIALIZATION,
    AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT,
    AMDGPU_INSTRUCTION_CONSTRAINT_INTEGER_MATRIX_COEXECUTION_SPACING,
    AMDGPU_INSTRUCTION_CONSTRAINT_KNOWN_BITS,
    AMDGPU_INSTRUCTION_CONSTRAINT_SWMMAC_LOW_PRECISION_LOWERING,
    AMDGPU_INSTRUCTION_CONSTRAINT_TENSOR_MULTICAST_MASK_PRESERVATION,
    AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_F4_32X16_SPLIT,
    AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K64_SCALE_PREFIX,
    AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K128_SPLIT,
    AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_SCALE_ENCODING,
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ACCUM_OFFSET,
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH,
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_DX10_CLAMP_AND_IEEE_MODE,
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING,
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID,
    AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAGS_PROFILELESS,
    AMDGPU_KERNEL_DESCRIPTOR_ABI_KNOWN_FLAGS,
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9,
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11,
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12,
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125,
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE,
    AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY,
    AMDGPU_KERNEL_ENTRY_PROFILE_NONE,
    AMDGPU_LDS_BANK_SERVICE_MODEL_SET_ORDINAL_NONE,
    AMDGPU_MATRIX_COEXECUTION_PROFILE_NONE,
    AMDGPU_MATRIX_COEXECUTION_PROFILE_XDL_LATENCY_4_8_16,
    AMDGPU_MATRIX_COEXECUTION_PROFILE_XDL_LATENCY_16_32,
    AMDGPU_MATRIX_COEXECUTION_PROFILES,
    AMDGPU_MATRIX_COEXECUTION_RULES_BY_PROFILE,
    AMDGPU_MATRIX_COEXECUTION_SOURCE_INFOS,
    AMDGPU_MATRIX_COEXECUTION_SOURCE_SWMMAC,
    AMDGPU_MATRIX_COEXECUTION_SOURCE_WMMA,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950,
    AMDGPU_MATRIX_FEATURE_PROFILE_NONE,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12_5_GENERIC,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250,
    AMDGPU_PROCESSOR_INFO_FLAG_ARCHITECTED_WORKGROUP_IDS,
    AMDGPU_PROCESSOR_INFO_FLAG_CLUSTER_LAUNCH_STATE,
    AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
    AMDGPU_PROCESSOR_INFO_KNOWN_FLAGS,
    AMDGPU_PROCESSOR_ORDINAL_NONE,
    AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU,
    AMDGPU_PROCESSOR_SCHEDULING_DESTINATION_SELECTION_WAIT_STATES,
    AMDGPU_PROCESSOR_SCHEDULING_KNOWN_BITS,
    AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR,
    AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES,
    AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR,
    AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_WAIT_STATES,
    AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER,
    AMDGPU_TARGET_ID_FEATURE_SUPPORT_KNOWN_FLAGS,
    AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC,
    AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ATTRS,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_INFOS,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODINGS,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_TEMPORAL_TH,
    AMDGPU_WAVEFRONT_SIZE_FLAG_32,
    AMDGPU_WAVEFRONT_SIZE_FLAG_64,
    AMDGPU_WAVEFRONT_SIZE_KNOWN_FLAGS,
    AmdgpuDescriptorSetInfo,
    AmdgpuProcessorInfo,
    AmdgpuSoppOpcodeInfo,
    AmdgpuTargetInfo,
    AmdgpuVectorMemoryCachePolicyEncodingInfo,
    amdgpu_descriptor_set_ordinal,
    amdgpu_generic_code_object_compatibility_info,
    amdgpu_lds_bank_service_model_sets,
    amdgpu_processor_ordinal,
    amdgpu_target_descriptor_set_key,
    amdgpu_target_instruction_constraints,
    kernel_descriptor_profile_supports_wavefront_size,
    sorted_descriptor_set_infos,
    sorted_processor_infos,
    sorted_target_infos,
    validate_amdgpu_code_object_processor_rows,
    validate_amdgpu_generic_contracts,
    validate_amdgpu_target_id_processor_rows,
    validate_amdgpu_target_rows,
)


@dataclass(frozen=True, slots=True)
class _AmdgpuDescriptorSetRow:
    info: AmdgpuDescriptorSetInfo
    sopp: AmdgpuSoppOpcodeInfo


def _u16_expr(value: int) -> str:
    return f"UINT16_C({value})"


def _u32_expr(value: int) -> str:
    return f"UINT32_C({value})"


def _c_ident(value: str) -> str:
    return value.upper().replace(".", "_").replace("-", "_")


def _cache_temporal_c_name(keyword: str) -> str:
    return f"LOOM_CACHE_TEMPORAL_{_c_ident(keyword)}"


def _cache_policy_attr_flag_c_name(attr: str) -> str:
    return f"LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_{_c_ident(attr)}"


_KERNEL_DESCRIPTOR_PROFILE_EXPRS = {
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE: "LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE",
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9: "LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX9",
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11: "LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX11",
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12: "LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX12",
    AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125: "LOOM_AMDGPU_KERNEL_DESCRIPTOR_PROFILE_GFX125",
}

_KERNEL_ENTRY_PROFILE_EXPRS = {
    AMDGPU_KERNEL_ENTRY_PROFILE_NONE: "LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_NONE",
    AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY: ("LOOM_AMDGPU_KERNEL_ENTRY_PROFILE_INITIAL_VMEM_REPLAY"),
}

_MATRIX_FEATURE_PROFILE_EXPRS = {
    AMDGPU_MATRIX_FEATURE_PROFILE_NONE: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_NONE",
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908",
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A",
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940",
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950",
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX9_4_GENERIC",
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11",
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12",
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250",
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12_5_GENERIC: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12_5_GENERIC",
}

_MATRIX_COEXECUTION_PROFILE_EXPRS = {
    AMDGPU_MATRIX_COEXECUTION_PROFILE_NONE: "LOOM_AMDGPU_MATRIX_COEXECUTION_PROFILE_NONE",
    AMDGPU_MATRIX_COEXECUTION_PROFILE_XDL_LATENCY_4_8_16: "LOOM_AMDGPU_MATRIX_COEXECUTION_PROFILE_XDL_LATENCY_4_8_16",
    AMDGPU_MATRIX_COEXECUTION_PROFILE_XDL_LATENCY_16_32: "LOOM_AMDGPU_MATRIX_COEXECUTION_PROFILE_XDL_LATENCY_16_32",
}

_MATRIX_COEXECUTION_SOURCE_EXPRS = {
    AMDGPU_MATRIX_COEXECUTION_SOURCE_WMMA: "LOOM_AMDGPU_MATRIX_COEXECUTION_SOURCE_WMMA",
    AMDGPU_MATRIX_COEXECUTION_SOURCE_SWMMAC: "LOOM_AMDGPU_MATRIX_COEXECUTION_SOURCE_SWMMAC",
}

_BUFFER_RESOURCE_CACHE_SWIZZLE_EXPRS = {
    AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE: "LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE",
    AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT: "LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT",
}

_BUFFER_RESOURCE_RECORD_ENCODING_EXPRS = {encoding: f"LOOM_AMDGPU_BUFFER_RESOURCE_RECORD_ENCODING_{_c_ident(encoding)}" for encoding in AMDGPU_BUFFER_RESOURCE_RECORD_ENCODINGS}

_VECTOR_MEMORY_CACHE_POLICY_ENCODING_EXPRS = {encoding: f"LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_{_c_ident(encoding)}" for encoding in AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODINGS}

_LDS_BANK_SERVICE_EVIDENCE_CLASS_EXPRS = {
    AMDGPU_LDS_BANK_SERVICE_EVIDENCE_PUBLIC_VENDOR_DOCUMENTATION: ("LOOM_AMDGPU_LDS_BANK_SERVICE_EVIDENCE_PUBLIC_VENDOR_DOCUMENTATION"),
    AMDGPU_LDS_BANK_SERVICE_EVIDENCE_VENDOR_SOFTWARE_MODEL_UNVALIDATED: ("LOOM_AMDGPU_LDS_BANK_SERVICE_EVIDENCE_VENDOR_SOFTWARE_MODEL_UNVALIDATED"),
    AMDGPU_LDS_BANK_SERVICE_EVIDENCE_SILICON_CALIBRATED_VENDOR_MODEL: ("LOOM_AMDGPU_LDS_BANK_SERVICE_EVIDENCE_SILICON_CALIBRATED_VENDOR_MODEL"),
}

_LDS_BANK_SERVICE_DIRECTION_EXPRS = {
    AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ: ("LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_READ"),
    AMDGPU_LDS_BANK_SERVICE_DIRECTION_WRITE: ("LOOM_AMDGPU_LDS_BANK_SERVICE_DIRECTION_WRITE"),
}

_LDS_BANK_SERVICE_REQUEST_POLICY_EXPRS = {
    AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COUNT_EACH: ("LOOM_AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COUNT_EACH"),
    AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COALESCE_IDENTICAL_READS: ("LOOM_AMDGPU_LDS_BANK_SERVICE_REQUEST_POLICY_COALESCE_IDENTICAL_READS"),
}

_DESCRIPTOR_SET_INFO_FLAG_EXPRS = (
    (
        AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
        "LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING",
    ),
    (
        AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION,
        "LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION",
    ),
    (
        AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_NUMERIC_MINMAX_MNEMONICS,
        "LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_NUMERIC_MINMAX_MNEMONICS",
    ),
    (
        AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_PACKED_BF16_ARITHMETIC,
        "LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_PACKED_BF16_ARITHMETIC",
    ),
    (
        AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOP3_TWO_SCALAR_SOURCES,
        "LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOP3_TWO_SCALAR_SOURCES",
    ),
    (
        AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_OCP_FP8_NONCANONICAL_NAN,
        "LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_OCP_FP8_NONCANONICAL_NAN",
    ),
    (
        AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_ARITHMETIC,
        "LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_ARITHMETIC",
    ),
    (
        AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_CONVERSION,
        "LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_CONVERSION",
    ),
    (
        AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_COMPARE,
        "LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_NATIVE_SCALAR_FLOAT_COMPARE",
    ),
    (
        AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_DUAL_MOV_SRC2_CACHE,
        "LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_DUAL_MOV_SRC2_CACHE",
    ),
)

_PROCESSOR_INFO_FLAG_EXPRS = (
    (
        AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
        "LOOM_AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION",
    ),
    (
        AMDGPU_PROCESSOR_INFO_FLAG_CLUSTER_LAUNCH_STATE,
        "LOOM_AMDGPU_PROCESSOR_INFO_FLAG_CLUSTER_LAUNCH_STATE",
    ),
    (
        AMDGPU_PROCESSOR_INFO_FLAG_ARCHITECTED_WORKGROUP_IDS,
        "LOOM_AMDGPU_PROCESSOR_INFO_FLAG_ARCHITECTED_WORKGROUP_IDS",
    ),
)

_INSTRUCTION_CONSTRAINT_BIT_EXPRS = (
    (
        AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT,
        "LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_DS_PAIRED_ADDRESS_ALIGNMENT",
    ),
    (
        AMDGPU_INSTRUCTION_CONSTRAINT_DS_ADDTID_ADDRESS_MATERIALIZATION,
        "LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_DS_ADDTID_ADDRESS_MATERIALIZATION",
    ),
    (
        AMDGPU_INSTRUCTION_CONSTRAINT_CLUSTER_MULTICAST_MASK_PRESERVATION,
        "LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_CLUSTER_MULTICAST_MASK_PRESERVATION",
    ),
    (
        AMDGPU_INSTRUCTION_CONSTRAINT_TENSOR_MULTICAST_MASK_PRESERVATION,
        "LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_TENSOR_MULTICAST_MASK_PRESERVATION",
    ),
    (
        AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K64_SCALE_PREFIX,
        "LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K64_SCALE_PREFIX",
    ),
    (
        AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K128_SPLIT,
        "LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_FP8_BF8_K128_SPLIT",
    ),
    (
        AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_F4_32X16_SPLIT,
        "LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_F4_32X16_SPLIT",
    ),
    (
        AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_SCALE_ENCODING,
        "LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_WMMA_SCALE_ENCODING",
    ),
    (
        AMDGPU_INSTRUCTION_CONSTRAINT_SWMMAC_LOW_PRECISION_LOWERING,
        "LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_SWMMAC_LOW_PRECISION_LOWERING",
    ),
    (
        AMDGPU_INSTRUCTION_CONSTRAINT_INTEGER_MATRIX_COEXECUTION_SPACING,
        "LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_INTEGER_MATRIX_COEXECUTION_SPACING",
    ),
)

_TARGET_ID_FEATURE_SUPPORT_FLAG_EXPRS = (
    (
        AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC,
        "LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC",
    ),
    (
        AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK,
        "LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK",
    ),
)

_WAVEFRONT_SIZE_FLAG_EXPRS = (
    (
        AMDGPU_WAVEFRONT_SIZE_FLAG_32,
        "LOOM_AMDGPU_WAVEFRONT_SIZE_FLAG_32",
    ),
    (
        AMDGPU_WAVEFRONT_SIZE_FLAG_64,
        "LOOM_AMDGPU_WAVEFRONT_SIZE_FLAG_64",
    ),
)

_KERNEL_DESCRIPTOR_ABI_FLAG_EXPRS = (
    (
        AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH,
        "LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ARCHITECTED_FLAT_SCRATCH",
    ),
    (
        AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING,
        "LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_GFX10_SGPR_ENCODING",
    ),
    (
        AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ACCUM_OFFSET,
        "LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_ACCUM_OFFSET",
    ),
    (
        AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_DX10_CLAMP_AND_IEEE_MODE,
        "LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_DX10_CLAMP_AND_IEEE_MODE",
    ),
    (
        AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID,
        "LOOM_AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAG_PACKED_WORKITEM_ID",
    ),
)

_PROCESSOR_SCHEDULING_BIT_EXPRS = (
    (
        AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR,
        "LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR",
    ),
    (
        AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_WAIT_STATES,
        "LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_WAIT_STATES",
    ),
    (
        AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES,
        "LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES",
    ),
    (
        AMDGPU_PROCESSOR_SCHEDULING_DESTINATION_SELECTION_WAIT_STATES,
        "LOOM_AMDGPU_PROCESSOR_SCHEDULING_DESTINATION_SELECTION_WAIT_STATES",
    ),
    (
        AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR,
        "LOOM_AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR",
    ),
    (
        AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU,
        "LOOM_AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU",
    ),
    (
        AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER,
        "LOOM_AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER",
    ),
)


def _enum_expr(value: str, table: Mapping[str, str], description: str) -> str:
    try:
        return table[value]
    except KeyError as exc:
        raise ValueError(f"unknown AMDGPU {description} '{value}'") from exc


def _kernel_descriptor_profile_expr(profile: str) -> str:
    return _enum_expr(profile, _KERNEL_DESCRIPTOR_PROFILE_EXPRS, "kernel descriptor profile")


def _kernel_entry_profile_expr(profile: str) -> str:
    return _enum_expr(profile, _KERNEL_ENTRY_PROFILE_EXPRS, "kernel entry profile")


def _matrix_feature_profile_expr(profile: str) -> str:
    return _enum_expr(profile, _MATRIX_FEATURE_PROFILE_EXPRS, "matrix feature profile")


def _matrix_coexecution_profile_expr(profile: str) -> str:
    return _enum_expr(
        profile,
        _MATRIX_COEXECUTION_PROFILE_EXPRS,
        "matrix coexecution profile",
    )


def _matrix_coexecution_source_expr(source: str) -> str:
    return _enum_expr(
        source,
        _MATRIX_COEXECUTION_SOURCE_EXPRS,
        "matrix coexecution source",
    )


def _buffer_resource_cache_swizzle_expr(kind: str) -> str:
    return _enum_expr(
        kind,
        _BUFFER_RESOURCE_CACHE_SWIZZLE_EXPRS,
        "buffer-resource cache swizzle kind",
    )


def _buffer_resource_record_encoding_expr(kind: str) -> str:
    return _enum_expr(
        kind,
        _BUFFER_RESOURCE_RECORD_ENCODING_EXPRS,
        "buffer-resource record encoding",
    )


def _vector_memory_cache_policy_encoding_expr(kind: str) -> str:
    return _enum_expr(
        kind,
        _VECTOR_MEMORY_CACHE_POLICY_ENCODING_EXPRS,
        "vector-memory cache-policy encoding",
    )


def _lds_bank_service_evidence_class_expr(evidence_class: str) -> str:
    return _enum_expr(
        evidence_class,
        _LDS_BANK_SERVICE_EVIDENCE_CLASS_EXPRS,
        "LDS bank-service evidence class",
    )


def _lds_bank_service_direction_expr(direction: str) -> str:
    return _enum_expr(
        direction,
        _LDS_BANK_SERVICE_DIRECTION_EXPRS,
        "LDS bank-service direction",
    )


def _lds_bank_service_request_policy_expr(request_policy: str) -> str:
    return _enum_expr(
        request_policy,
        _LDS_BANK_SERVICE_REQUEST_POLICY_EXPRS,
        "LDS bank-service request policy",
    )


def _descriptor_ref_expr(descriptor_key: str) -> str:
    prefix = "amdgpu."
    if not descriptor_key.startswith(prefix):
        raise ValueError(f"AMDGPU descriptor key '{descriptor_key}' must start with '{prefix}'")
    return "LOOM_AMDGPU_DESCRIPTOR_REF_" + _c_ident(descriptor_key.removeprefix(prefix))


def _ordinal_bit_expr(owner: str, values: Sequence[str], vocabulary: Sequence[str]) -> str:
    if not values:
        raise ValueError(f"{owner} must have at least one accepted value")
    missing = tuple(value for value in values if value not in vocabulary)
    if missing:
        raise ValueError(f"{owner} references unknown values: {', '.join(missing)}")
    bits = 0
    for value in values:
        ordinal = vocabulary.index(value)
        if ordinal >= 32:
            raise ValueError(f"{owner} value '{value}' has ordinal {ordinal}, expected 0..31")
        bits |= 1 << ordinal
    return f"UINT32_C(0x{bits:08x})"


def _memory_cache_policy_attr_flags_expr(owner: str, attrs: Sequence[str]) -> str:
    missing = tuple(attr for attr in attrs if attr not in AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ATTRS)
    if missing:
        raise ValueError(f"{owner} references unknown attrs: {', '.join(missing)}")
    if not attrs:
        return "0"
    return " | ".join(_cache_policy_attr_flag_c_name(attr) for attr in attrs)


def _validate_memory_cache_policy_encoding_infos(
    rows: Sequence[AmdgpuVectorMemoryCachePolicyEncodingInfo],
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
) -> None:
    expected_encodings = set(AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODINGS)
    row_encodings = {row.encoding for row in rows}
    expected_row_encodings = expected_encodings - {"none"}
    if row_encodings != expected_row_encodings:
        missing = sorted(expected_row_encodings - row_encodings)
        extra = sorted(row_encodings - expected_row_encodings)
        details: list[str] = []
        if missing:
            details.append(f"missing rows for {', '.join(missing)}")
        if extra:
            details.append(f"unexpected rows for {', '.join(extra)}")
        raise ValueError(f"AMDGPU memory cache-policy encoding table must cover every non-none encoding: {'; '.join(details)}")

    selected_keys: set[str] = set()
    for row in rows:
        owner = f"AMDGPU memory cache-policy encoding '{row.encoding}'"
        _vector_memory_cache_policy_encoding_expr(row.encoding)
        _ordinal_bit_expr(owner, row.cache_scopes, AMDGPU_CACHE_SCOPE_KEYWORDS)
        _ordinal_bit_expr(owner, row.cache_temporals, AMDGPU_CACHE_TEMPORAL_KEYWORDS)
        _memory_cache_policy_attr_flags_expr(owner, row.attrs)
        if row.selected_key in selected_keys:
            raise ValueError(f"{owner} has duplicate selected key '{row.selected_key}'")
        selected_keys.add(row.selected_key)

    descriptor_set_encodings = {descriptor_set.vector_memory.cache_policy_encoding for descriptor_set in descriptor_sets}
    unknown_descriptor_set_encodings = sorted(descriptor_set_encodings - expected_encodings)
    if unknown_descriptor_set_encodings:
        raise ValueError(f"AMDGPU descriptor sets reference unknown memory cache-policy encodings: {', '.join(unknown_descriptor_set_encodings)}")


def _ordered_memory_cache_policy_encoding_infos(
    rows: Sequence[AmdgpuVectorMemoryCachePolicyEncodingInfo] = (AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_INFOS),
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo] | None = None,
) -> tuple[AmdgpuVectorMemoryCachePolicyEncodingInfo, ...]:
    descriptor_sets = sorted_descriptor_set_infos() if descriptor_sets is None else descriptor_sets
    _validate_memory_cache_policy_encoding_infos(rows, descriptor_sets)
    return tuple(
        sorted(
            rows,
            key=lambda row: AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODINGS.index(row.encoding),
        )
    )


def _ordered_memory_cache_policy_temporal_th(
    rows: Sequence[tuple[str, int]] = AMDGPU_VECTOR_MEMORY_CACHE_POLICY_TEMPORAL_TH,
) -> tuple[tuple[str, int], ...]:
    expected_temporals = set(AMDGPU_CACHE_TEMPORAL_KEYWORDS)
    row_temporals = {keyword for keyword, _ in rows}
    if row_temporals != expected_temporals:
        missing = sorted(expected_temporals - row_temporals)
        extra = sorted(row_temporals - expected_temporals)
        details: list[str] = []
        if missing:
            details.append(f"missing rows for {', '.join(missing)}")
        if extra:
            details.append(f"unexpected rows for {', '.join(extra)}")
        raise ValueError(f"AMDGPU memory cache-policy temporal TH table must cover every cache temporal: {'; '.join(details)}")
    for keyword, th_value in rows:
        if th_value < 0 or th_value > 7:
            raise ValueError(f"AMDGPU memory cache-policy temporal '{keyword}' has TH value {th_value}, expected 0..7")
    return tuple(sorted(rows, key=lambda row: AMDGPU_CACHE_TEMPORAL_KEYWORDS.index(row[0])))


def _memory_cache_policy_fragment_header() -> list[str]:
    return [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.amdgpu_target_info"),
        "",
    ]


def _memory_cache_policy_encoding_info_initializer(
    row: AmdgpuVectorMemoryCachePolicyEncodingInfo,
) -> str:
    owner = f"AMDGPU memory cache-policy encoding '{row.encoding}'"
    scope_bits = _ordinal_bit_expr(owner, row.cache_scopes, AMDGPU_CACHE_SCOPE_KEYWORDS)
    temporal_bits = _ordinal_bit_expr(owner, row.cache_temporals, AMDGPU_CACHE_TEMPORAL_KEYWORDS)
    attr_flags = _memory_cache_policy_attr_flags_expr(owner, row.attrs)
    return "\n".join(
        [
            "{",
            f"    .encoding = {_vector_memory_cache_policy_encoding_expr(row.encoding)},",
            f'    .encoding_key = IREE_SVL("{row.encoding}"),',
            f'    .selected_key = IREE_SVL("{row.selected_key}"),',
            f"    .scope_bits = {scope_bits},",
            f"    .temporal_bits = {temporal_bits},",
            f"    .attr_flags = {attr_flags},",
            "},",
        ]
    )


def _memory_cache_policy_temporal_th_initializer(row: tuple[str, int]) -> str:
    keyword, th_value = row
    return f"[{_cache_temporal_c_name(keyword)}] = {th_value},"


def _emit_memory_cache_policy_encoding_rows() -> str:
    return (
        "\n".join(
            [
                *_memory_cache_policy_fragment_header(),
                *(_memory_cache_policy_encoding_info_initializer(row) for row in _ordered_memory_cache_policy_encoding_infos()),
            ]
        )
        + "\n"
    )


def _emit_memory_cache_policy_temporal_th() -> str:
    return (
        "\n".join(
            [
                *_memory_cache_policy_fragment_header(),
                *(_memory_cache_policy_temporal_th_initializer(row) for row in _ordered_memory_cache_policy_temporal_th()),
            ]
        )
        + "\n"
    )


def _lds_bank_service_fragment_header() -> list[str]:
    return [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.amdgpu_target_info"),
        "",
    ]


def _lds_bank_service_model_initializer(
    info: AmdgpuLdsBankServiceModelInfo,
) -> list[str]:
    lines = [
        "    {",
        f"      .descriptor_ref = {_descriptor_ref_expr(info.descriptor_key)},",
        "      .model = {",
        f"        .key = IREE_SVL({_c_string_arg(info.key)}),",
        f"        .revision = IREE_SVL({_c_string_arg(info.revision)}),",
        f"        .evidence_class = {_lds_bank_service_evidence_class_expr(info.evidence_class)},",
        f"        .direction = {_lds_bank_service_direction_expr(info.direction)},",
        f"        .request_policy = {_lds_bank_service_request_policy_expr(info.request_policy)},",
        f"        .wave_size = {info.wave_size},",
        f"        .bank_count = {info.bank_count},",
        f"        .bank_word_byte_count = {info.bank_word_byte_count},",
        f"        .packet_word_count = {info.packet_word_count},",
        f"        .phase_count = {len(info.phase_lane_masks)},",
        "        .phase_lane_masks = {",
    ]
    lines.extend(f"          UINT64_C(0x{phase_lane_mask:016x})," for phase_lane_mask in info.phase_lane_masks)
    lines.extend(
        [
            "        },",
            "      },",
            "    },",
        ]
    )
    return lines


def _emit_lds_bank_service_model_rows() -> str:
    processors = sorted_processor_infos()
    targets = sorted_target_infos()
    validate_amdgpu_lds_bank_service_model_infos(amdgpu_descriptor_ref_keys())
    validate_amdgpu_target_rows(processors, targets)
    model_infos_by_key = amdgpu_lds_bank_service_model_info_by_key()
    model_sets = amdgpu_lds_bank_service_model_sets(processors, targets)
    validate_amdgpu_lds_bank_service_model_coverage(model_sets)

    lines = _lds_bank_service_fragment_header()
    for ordinal, model_keys in enumerate(model_sets):
        lines.append(f"static const loom_amdgpu_lds_bank_service_model_binding_t kAmdgpuLdsBankServiceModelSet{ordinal}Bindings[] = {{")
        for model_key in model_keys:
            lines.extend(_lds_bank_service_model_initializer(model_infos_by_key[model_key]))
        lines.extend(["};", ""])
    lines.append("static const loom_amdgpu_lds_bank_service_model_set_t kAmdgpuLdsBankServiceModelSets[] = {")
    for ordinal, _ in enumerate(model_sets):
        array_name = f"kAmdgpuLdsBankServiceModelSet{ordinal}Bindings"
        lines.extend(
            [
                "  {",
                f"    .bindings = {array_name},",
                f"    .count = IREE_ARRAYSIZE({array_name}),",
                "  },",
            ]
        )
    lines.extend(["};", ""])
    return "\n".join(lines)


def _flag_bits_expr(
    flags: int,
    *,
    known_bits: int,
    rows: Sequence[tuple[int, str]],
    zero_expr: str,
    description: str,
) -> str:
    if flags == 0:
        return zero_expr
    remaining_flags = flags
    exprs: list[str] = []
    for flag, expr in rows:
        if flags & flag:
            exprs.append(expr)
            remaining_flags &= ~flag
    unknown_flags = flags & ~known_bits
    if unknown_flags != 0:
        raise ValueError(f"unknown AMDGPU {description} flags 0x{unknown_flags:x}")
    if remaining_flags != 0:
        raise ValueError(f"AMDGPU {description} flags 0x{remaining_flags:x} have no C expression")
    return " | ".join(exprs)


def _descriptor_set_info_flags_expr(flags: int) -> str:
    return _flag_bits_expr(
        flags,
        known_bits=AMDGPU_DESCRIPTOR_SET_INFO_KNOWN_FLAGS,
        rows=_DESCRIPTOR_SET_INFO_FLAG_EXPRS,
        zero_expr="UINT64_C(0)",
        description="descriptor-set info",
    )


def _processor_info_flags_expr(flags: int) -> str:
    return _flag_bits_expr(
        flags,
        known_bits=AMDGPU_PROCESSOR_INFO_KNOWN_FLAGS,
        rows=_PROCESSOR_INFO_FLAG_EXPRS,
        zero_expr="UINT32_C(0)",
        description="processor info",
    )


def _instruction_constraint_bits_expr(flags: int) -> str:
    return _flag_bits_expr(
        flags,
        known_bits=AMDGPU_INSTRUCTION_CONSTRAINT_KNOWN_BITS,
        rows=_INSTRUCTION_CONSTRAINT_BIT_EXPRS,
        zero_expr="UINT32_C(0)",
        description="instruction constraint",
    )


def _target_id_feature_support_flags_expr(flags: int) -> str:
    return _flag_bits_expr(
        flags=flags,
        known_bits=AMDGPU_TARGET_ID_FEATURE_SUPPORT_KNOWN_FLAGS,
        rows=_TARGET_ID_FEATURE_SUPPORT_FLAG_EXPRS,
        zero_expr="LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_NONE",
        description="target-ID feature support",
    )


def _wavefront_size_flags_expr(flags: int) -> str:
    return _flag_bits_expr(
        flags,
        known_bits=AMDGPU_WAVEFRONT_SIZE_KNOWN_FLAGS,
        rows=_WAVEFRONT_SIZE_FLAG_EXPRS,
        zero_expr="UINT32_C(0)",
        description="wavefront-size",
    )


def _kernel_descriptor_abi_flags_expr(flags: int) -> str:
    return _flag_bits_expr(
        flags,
        known_bits=AMDGPU_KERNEL_DESCRIPTOR_ABI_KNOWN_FLAGS,
        rows=_KERNEL_DESCRIPTOR_ABI_FLAG_EXPRS,
        zero_expr="UINT64_C(0)",
        description="kernel descriptor ABI",
    )


def _processor_scheduling_bits_expr(flags: int) -> str:
    return _flag_bits_expr(
        flags,
        known_bits=AMDGPU_PROCESSOR_SCHEDULING_KNOWN_BITS,
        rows=_PROCESSOR_SCHEDULING_BIT_EXPRS,
        zero_expr="UINT32_C(0)",
        description="processor scheduling",
    )


def _supported_wavefront_sizes(info: AmdgpuProcessorInfo) -> int:
    profile = info.kernel_descriptor.profile
    flags = 0
    if kernel_descriptor_profile_supports_wavefront_size(profile, 32):
        flags |= AMDGPU_WAVEFRONT_SIZE_FLAG_32
    if kernel_descriptor_profile_supports_wavefront_size(profile, 64):
        flags |= AMDGPU_WAVEFRONT_SIZE_FLAG_64
    return flags


def _materialize_descriptor_set_rows(
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
) -> tuple[_AmdgpuDescriptorSetRow, ...]:
    rows: list[_AmdgpuDescriptorSetRow] = []
    for info in descriptor_sets:
        sopp = info.isa_infos[0].sopp_opcodes
        for isa_info in info.isa_infos[1:]:
            if isa_info.sopp_opcodes != sopp:
                raise ValueError(f"AMDGPU descriptor set {info.key} has incompatible S_OPP opcodes across ISA XML keys '{info.isa_infos[0].isa_xml_key}' and '{isa_info.isa_xml_key}'")
        rows.append(
            _AmdgpuDescriptorSetRow(
                info=info,
                sopp=sopp,
            )
        )
    return tuple(rows)


def _validate_descriptor_sets(descriptor_sets: Sequence[AmdgpuDescriptorSetInfo]) -> None:
    keys = [info.key for info in descriptor_sets]
    if keys != sorted(keys):
        raise ValueError("AMDGPU descriptor-set target-info keys must be sorted")
    if len(keys) != len(set(keys)):
        raise ValueError("AMDGPU descriptor-set target-info keys must be unique")
    if len(keys) >= AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE:
        raise ValueError("AMDGPU descriptor-set ordinals must fit uint16_t")
    generator_targets = [info.generator_target for info in descriptor_sets]
    if len(generator_targets) != len(set(generator_targets)):
        raise ValueError("AMDGPU descriptor generator targets must be unique")
    infos_by_generator_target = {info.generator_target: info for info in descriptor_sets}
    for info in descriptor_sets:
        if not info.generator_target:
            raise ValueError("AMDGPU descriptor generator target is required")
        if not info.key:
            raise ValueError("AMDGPU descriptor-set key is required")
        if not info.isa_infos:
            raise ValueError(f"AMDGPU ISA membership is required for {info.key}")
        isa_xml_keys = [isa_info.isa_xml_key for isa_info in info.isa_infos]
        if len(isa_xml_keys) != len(set(isa_xml_keys)):
            raise ValueError(f"AMDGPU ISA XML keys must be unique for {info.key}")
        for isa_info in info.isa_infos:
            if not isa_info.isa_xml_key:
                raise ValueError(f"AMDGPU ISA XML key is required for {info.key}")
            if not isa_info.isa_architecture_name:
                raise ValueError(f"AMDGPU ISA XML architecture name is required for {info.key}")
            if isa_info.isa_architecture_id <= 0:
                raise ValueError(f"AMDGPU ISA XML architecture id is required for {info.key}")
        if info.flags < 0 or info.flags > 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"AMDGPU descriptor-set info flags for {info.key} must fit u64")
        _descriptor_set_info_flags_expr(info.flags)
        if info.storage_generator_target is not None:
            if not info.storage_generator_target:
                raise ValueError(f"AMDGPU storage generator target is required for {info.key}")
            if info.storage_generator_target == info.generator_target:
                raise ValueError(f"AMDGPU descriptor set {info.key} cannot store itself as a view")
            storage_info = infos_by_generator_target.get(info.storage_generator_target)
            if storage_info is None:
                raise ValueError(f"AMDGPU descriptor set {info.key} references unknown storage generator target '{info.storage_generator_target}'")
            if storage_info.storage_generator_target is not None:
                raise ValueError(f"AMDGPU descriptor set {info.key} uses view-only target '{storage_info.generator_target}' as storage")
            if not set(storage_info.isa_infos).issubset(info.isa_infos):
                raise ValueError(f"AMDGPU descriptor set {info.key} storage target '{storage_info.generator_target}' has ISA membership outside the view contract")
        if info.member_generator_targets:
            if tuple(sorted(info.member_generator_targets)) != (info.member_generator_targets):
                raise ValueError(f"AMDGPU descriptor set {info.key} member generator targets must be sorted")
            if len(info.member_generator_targets) != len(set(info.member_generator_targets)):
                raise ValueError(f"AMDGPU descriptor set {info.key} member generator targets must be unique")
            member_isa_infos = []
            for member_generator_target in info.member_generator_targets:
                member_info = infos_by_generator_target.get(member_generator_target)
                if member_info is None:
                    raise ValueError(f"AMDGPU descriptor set {info.key} references unknown member generator target '{member_generator_target}'")
                if member_info.member_generator_targets:
                    raise ValueError(f"AMDGPU descriptor set {info.key} uses generic member target '{member_generator_target}'")
                for isa_info in member_info.isa_infos:
                    if isa_info not in member_isa_infos:
                        member_isa_infos.append(isa_info)
            if tuple(member_isa_infos) != info.isa_infos:
                raise ValueError(f"AMDGPU descriptor set {info.key} ISA membership does not match its member generator targets")
        _buffer_resource_record_encoding_expr(info.buffer_resource.record_encoding)
        if info.buffer_resource.record_encoding == AMDGPU_BUFFER_RESOURCE_RECORD_ENCODING_NONE:
            raise ValueError(f"AMDGPU descriptor set {info.key} must declare a buffer-resource record encoding")
        _buffer_resource_cache_swizzle_expr(info.buffer_resource.cache_swizzle)
        _vector_memory_cache_policy_encoding_expr(info.vector_memory.cache_policy_encoding)
        if info.vector_memory.cache_policy_encoding == AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE:
            raise ValueError(f"AMDGPU descriptor set {info.key} must declare a non-none vector-memory cache-policy encoding")


def _validate_descriptor_set_rows(rows: Sequence[_AmdgpuDescriptorSetRow]) -> None:
    for row in rows:
        opcode_rows = (
            ("s_nop", row.sopp.nop),
            ("s_delay_alu", row.sopp.delay_alu),
            ("s_endpgm", row.sopp.endpgm),
            ("s_branch", row.sopp.branch),
            ("s_cbranch_scc0", row.sopp.conditional_branch_scc0),
            ("s_cbranch_scc1", row.sopp.conditional_branch_scc1),
        )
        for name, opcode in opcode_rows:
            if opcode < 0 or opcode > 0xFFFF:
                raise ValueError(f"AMDGPU {name} opcode for {row.info.key} must fit u16")


def _validate_matrix_coexecution_sources() -> None:
    if tuple(_MATRIX_COEXECUTION_SOURCE_EXPRS) != tuple(info.source for info in AMDGPU_MATRIX_COEXECUTION_SOURCE_INFOS):
        raise ValueError("AMDGPU matrix coexecution source expressions must cover sources in enum order")
    for source_info in AMDGPU_MATRIX_COEXECUTION_SOURCE_INFOS:
        _matrix_coexecution_source_expr(source_info.source)
        if source_info.result_operand_index < 0 or source_info.result_operand_index > 0xFF:
            raise ValueError(f"AMDGPU matrix coexecution result operand for {source_info.source} must fit u8")
        if source_info.source_operand_start < 0 or source_info.source_operand_start > 0xFF:
            raise ValueError(f"AMDGPU matrix coexecution source operand start for {source_info.source} must fit u8")
        if source_info.source_operand_count <= 0 or source_info.source_operand_count > 0xFF:
            raise ValueError(f"AMDGPU matrix coexecution source operand count for {source_info.source} must fit nonzero u8")
        source_operand_end = source_info.source_operand_start + source_info.source_operand_count
        if source_operand_end > 0x100:
            raise ValueError(f"AMDGPU matrix coexecution source operand range for {source_info.source} must fit u8 indexes")
        if source_info.source_operand_start <= source_info.result_operand_index < source_operand_end:
            raise ValueError(f"AMDGPU matrix coexecution result operand for {source_info.source} overlaps its sources")


def _validate_matrix_coexecution_profiles() -> None:
    _validate_matrix_coexecution_sources()
    if tuple(AMDGPU_MATRIX_COEXECUTION_RULES_BY_PROFILE) != (AMDGPU_MATRIX_COEXECUTION_PROFILES):
        raise ValueError("AMDGPU matrix coexecution rule tables must cover profiles in enum order")
    for profile, rules in AMDGPU_MATRIX_COEXECUTION_RULES_BY_PROFILE.items():
        if profile == AMDGPU_MATRIX_COEXECUTION_PROFILE_NONE:
            if rules:
                raise ValueError("AMDGPU none matrix coexecution profile cannot contain rules")
            continue
        if not rules:
            raise ValueError(f"AMDGPU matrix coexecution profile {profile} must contain rules")
        if len(rules) > 0xFF:
            raise ValueError(f"AMDGPU matrix coexecution profile {profile} must fit u8 rule count")
        keys: set[tuple[str, int]] = set()
        for rule in rules:
            _matrix_coexecution_source_expr(rule.source)
            if rule.latency_cycles <= 0 or rule.latency_cycles > 0xFF:
                raise ValueError(f"AMDGPU matrix coexecution latency for {profile} must fit nonzero u8")
            if rule.vector_issue_distance <= 0 or rule.vector_issue_distance > 0xFF or rule.matrix_issue_distance <= 0 or rule.matrix_issue_distance > 0xFF:
                raise ValueError(f"AMDGPU matrix coexecution distances for {profile} must fit nonzero u8")
            if rule.matrix_issue_distance < rule.vector_issue_distance:
                raise ValueError(f"AMDGPU matrix coexecution matrix distance for {profile} cannot be shorter than its vector distance")
            key = (rule.source, rule.latency_cycles)
            if key in keys:
                raise ValueError(f"AMDGPU matrix coexecution profile {profile} has duplicate rule {key}")
            keys.add(key)


def _validate_processors(
    processors: Sequence[AmdgpuProcessorInfo],
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
) -> None:
    processor_names = [info.processor for info in processors]
    if processor_names != sorted(processor_names):
        raise ValueError("AMDGPU processor target-info keys must be sorted")
    if len(processor_names) != len(set(processor_names)):
        raise ValueError("AMDGPU processor target-info keys must be unique")
    if len(processors) >= AMDGPU_PROCESSOR_ORDINAL_NONE:
        raise ValueError("AMDGPU processor target-info ordinals must fit uint16_t")
    descriptor_set_keys = {info.key for info in descriptor_sets}
    for info in processors:
        kernel_descriptor = info.kernel_descriptor
        profile = kernel_descriptor.profile
        vgpr_granules = kernel_descriptor.vgpr_granules
        if not info.processor:
            raise ValueError("AMDGPU processor is required")
        if info.descriptor_set.key and info.descriptor_set.key not in descriptor_set_keys:
            raise ValueError(f"AMDGPU processor {info.processor} references unknown descriptor set {info.descriptor_set.key}")
        if info.elf.machine_flags < 0 or info.elf.machine_flags > 0x0FF:
            raise ValueError(f"AMDGPU ELF machine flags for {info.processor} must fit EF_AMDGPU_MACH")
        if info.elf.feature_flags < 0 or info.elf.feature_flags > 0xFFFFFFFF:
            raise ValueError(f"AMDGPU ELF feature flags for {info.processor} must fit u32")
        if info.elf.feature_flags & 0x0FF:
            raise ValueError(f"AMDGPU ELF feature flags for {info.processor} must not overlap EF_AMDGPU_MACH")
        if info.elf.feature_flags & AMDGPU_ELF_GENERIC_VERSION_MASK_V6:
            raise ValueError(f"AMDGPU ELF feature flags for {info.processor} must not overlap the generic version")
        if info.elf.generic_version < 0 or info.elf.generic_version > 0xFF:
            raise ValueError(f"AMDGPU ELF generic version for {info.processor} must fit u8")
        is_generic_processor = info.processor.endswith("-generic")
        if is_generic_processor != (info.elf.generic_version != 0):
            raise ValueError(f"AMDGPU processor {info.processor} generic identity and ELF generic version disagree")
        if info.flags < 0 or info.flags > 0xFFFFFFFF:
            raise ValueError(f"AMDGPU processor info flags for {info.processor} must fit u32")
        _processor_info_flags_expr(info.flags)
        if info.wavefront.default_size not in (32, 64):
            raise ValueError(f"AMDGPU default wavefront size for {info.processor} must be 32 or 64")
        supported_wavefront_sizes = _supported_wavefront_sizes(info)
        _matrix_feature_profile_expr(info.features.matrix)
        _matrix_coexecution_profile_expr(info.features.matrix_coexecution)
        if kernel_descriptor.flags < 0 or kernel_descriptor.flags > 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"AMDGPU kernel descriptor ABI flags for {info.processor} must fit u64")
        _kernel_descriptor_abi_flags_expr(kernel_descriptor.flags)
        if profile == AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE:
            if vgpr_granules.wave32 != 0 or vgpr_granules.wave64 != 0:
                raise ValueError(f"AMDGPU processor {info.processor} has no kernel descriptor profile but has VGPR encoding granules")
            profileless_flags = kernel_descriptor.flags & ~AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAGS_PROFILELESS
            if profileless_flags != 0:
                raise ValueError(f"AMDGPU processor {info.processor} has no kernel descriptor profile but has profile-owned ABI flags 0x{profileless_flags:x}")
        else:
            default_wavefront_size = AMDGPU_WAVEFRONT_SIZE_FLAG_32 if info.wavefront.default_size == 32 else AMDGPU_WAVEFRONT_SIZE_FLAG_64
            if (supported_wavefront_sizes & default_wavefront_size) == 0:
                raise ValueError(f"AMDGPU default wavefront size for {info.processor} is not supported by its kernel descriptor profile")
            if vgpr_granules.wave32 == 0 or vgpr_granules.wave64 == 0:
                raise ValueError(f"AMDGPU processor {info.processor} has descriptor profile but no VGPR encoding granules")
            if info.elf.machine_flags == 0:
                raise ValueError(f"AMDGPU processor {info.processor} has a kernel descriptor profile but no ELF machine flags")
        if info.flags & AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION:
            if not info.descriptor_set.key:
                raise ValueError(f"AMDGPU processor {info.processor} has HSACO emission support but no descriptor set")
            if profile == AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE:
                raise ValueError(f"AMDGPU processor {info.processor} has HSACO emission support but no kernel descriptor profile")
            if info.elf.machine_flags == 0:
                raise ValueError(f"AMDGPU processor {info.processor} has HSACO emission support but no ELF machine flags")
        if info.features.scheduling < 0 or info.features.scheduling > 0xFFFFFFFF:
            raise ValueError(f"AMDGPU scheduling bits for {info.processor} must fit u32")
        _processor_scheduling_bits_expr(info.features.scheduling)
    validate_amdgpu_generic_contracts(processors, descriptor_sets)


def _emit_header(descriptor_sets: Sequence[AmdgpuDescriptorSetInfo]) -> str:
    guard = "LOOM_TARGET_ARCH_AMDGPU_TARGET_INFO_H_"
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.amdgpu_target_info"),
        "",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        '#include "loom/target/arch/amdgpu/target_info_defs.h"',
        "",
        "// Canonical AMDHSA target-ID prefix.",
        "extern const iree_string_view_t",
        "    loom_amdgpu_target_info_amdhsa_target_id_prefix;",
        "",
        "// Generated dense descriptor-set ordinals.",
    ]
    lines.extend(f"#define {amdgpu_descriptor_set_ordinal_constant_name(info.key)} {_u16_expr(amdgpu_descriptor_set_ordinal(info.key))}" for info in descriptor_sets)
    lines.extend(
        [
            f"#define LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_COUNT {_u16_expr(len(descriptor_sets))}",
            "",
        ]
    )
    lines.extend(
        [
            f"#endif  // {guard}",
        ]
    )
    return "\n".join(lines) + "\n"


def _emit_tables_header() -> str:
    guard = "LOOM_TARGET_ARCH_AMDGPU_TARGET_INFO_TABLES_H_"
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.amdgpu_target_info"),
        "",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        '#include "loom/target/arch/amdgpu/target_info_defs.h"',
        "",
        "extern const loom_amdgpu_descriptor_set_info_t",
        "    loom_amdgpu_target_info_descriptor_set_infos[];",
        "extern const iree_host_size_t",
        "    loom_amdgpu_target_info_descriptor_set_info_count;",
        "",
        "extern const loom_amdgpu_matrix_coexecution_profile_info_t",
        "    loom_amdgpu_target_info_matrix_coexecution_profile_infos[];",
        "",
        "extern const loom_amdgpu_processor_info_t",
        "    loom_amdgpu_target_info_processor_infos[];",
        "extern const iree_host_size_t",
        "    loom_amdgpu_target_info_processor_info_count;",
        "",
        "extern const loom_amdgpu_target_info_t",
        "    loom_amdgpu_target_info_target_infos[];",
        "extern const iree_host_size_t",
        "    loom_amdgpu_target_info_target_info_count;",
        "",
        "extern const loom_amdgpu_physical_target_info_t",
        "    loom_amdgpu_target_info_physical_target_infos[];",
        "extern const iree_host_size_t",
        "    loom_amdgpu_target_info_physical_target_info_count;",
        "",
        f"#endif  // {guard}",
    ]
    return "\n".join(lines) + "\n"


def _emit_descriptor_set_rows(rows: Sequence[_AmdgpuDescriptorSetRow]) -> list[str]:
    lines = [
        "const loom_amdgpu_descriptor_set_info_t loom_amdgpu_target_info_descriptor_set_infos[] = {",
    ]
    for row in rows:
        info = row.info
        lines.extend(
            [
                "  {",
                f"    .key = IREE_SVL({_c_string_arg(info.key)}),",
                f"    .ordinal = {_u16_expr(amdgpu_descriptor_set_ordinal(info.key))},",
                "    .sopp = {",
                f"      .nop = UINT16_C(0x{row.sopp.nop:03x}),",
                f"      .delay_alu = UINT16_C(0x{row.sopp.delay_alu:03x}),",
                f"      .endpgm = UINT16_C(0x{row.sopp.endpgm:03x}),",
                f"      .branch = UINT16_C(0x{row.sopp.branch:03x}),",
                f"      .conditional_branch_scc0 = UINT16_C(0x{row.sopp.conditional_branch_scc0:03x}),",
                f"      .conditional_branch_scc1 = UINT16_C(0x{row.sopp.conditional_branch_scc1:03x}),",
                "    },",
                f"    .flags = {_descriptor_set_info_flags_expr(info.flags)},",
                "    .buffer_resource = {",
                f"      .record_encoding = {_buffer_resource_record_encoding_expr(info.buffer_resource.record_encoding)},",
                f"      .cache_swizzle = {_buffer_resource_cache_swizzle_expr(info.buffer_resource.cache_swizzle)},",
                "    },",
                "    .vector_memory = {",
                f"      .cache_policy_encoding = {_vector_memory_cache_policy_encoding_expr(info.vector_memory.cache_policy_encoding)},",
                "    },",
                "  },",
            ]
        )
    lines.extend(["};", ""])
    return lines


def _processor_descriptor_set_ordinal_expr(info: AmdgpuProcessorInfo) -> str:
    if not info.descriptor_set.key:
        return "LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE"
    return _u16_expr(amdgpu_descriptor_set_ordinal(info.descriptor_set.key))


def _lds_bank_service_model_set_ordinal_expr(
    model_keys: tuple[str, ...],
    model_set_ordinals: Mapping[tuple[str, ...], int],
) -> str:
    if not model_keys:
        return "LOOM_AMDGPU_LDS_BANK_SERVICE_MODEL_SET_ORDINAL_NONE"
    return _u16_expr(model_set_ordinals[model_keys])


def _processor_generic_code_object_fields(
    info: AmdgpuProcessorInfo,
    processor_ordinals: Mapping[str, int],
) -> tuple[int | None, int]:
    compatibility = amdgpu_generic_code_object_compatibility_info(info.processor)
    if compatibility is None:
        return None, 0
    return (
        processor_ordinals[compatibility.code_object_processor],
        compatibility.generic_introduction_version,
    )


def _target_kernel_metadata_extension_array_name(info: AmdgpuTargetInfo) -> str:
    return "loom_amdgpu_target_info_" + info.target.replace("-", "_") + "_kernel_metadata_extensions"


def _emit_target_rows(
    targets: Sequence[AmdgpuTargetInfo],
    processors: Sequence[AmdgpuProcessorInfo],
    model_set_ordinals: Mapping[tuple[str, ...], int],
) -> list[str]:
    lines: list[str] = []
    processors_by_name = {info.processor: info for info in processors}
    for info in targets:
        metadata_extensions = info.semantics.kernel_metadata_extensions
        if not metadata_extensions:
            continue
        metadata_array_name = _target_kernel_metadata_extension_array_name(info)
        lines.append(f"static const loom_amdgpu_metadata_string_property_t {metadata_array_name}[] = {{")
        for key, value in metadata_extensions:
            lines.append(f"  {{.key = IREE_SVL({_c_string_arg(key)}), .value = IREE_SVL({_c_string_arg(value)})}},")
        lines.extend(["};", ""])

    lines.append("const loom_amdgpu_target_info_t loom_amdgpu_target_info_target_infos[] = {")
    for info in targets:
        processor = processors_by_name[info.processor]
        descriptor_set_key = amdgpu_target_descriptor_set_key(info, processor)
        model_keys = info.semantics.lds_bank_service_models if info.semantics.lds_bank_service_models is not None else processor.features.lds_bank_service_models
        metadata_extensions = info.semantics.kernel_metadata_extensions
        metadata_array_name = _target_kernel_metadata_extension_array_name(info)
        metadata_entries = metadata_array_name if metadata_extensions else "NULL"
        metadata_count = f"IREE_ARRAYSIZE({metadata_array_name})" if metadata_extensions else "0"
        lines.extend(
            [
                "  {",
                f"    .name = IREE_SVL({_c_string_arg(info.target)}),",
                f"    .target_kind = {_u32_expr(info.enum_value)},",
                f"    .processor_ordinal = {_u16_expr(amdgpu_processor_ordinal(info.processor))},",
                f"    .descriptor_set_key = IREE_SVL({_c_string_arg(descriptor_set_key)}),",
                f"    .descriptor_set_ordinal = {_u16_expr(amdgpu_descriptor_set_ordinal(descriptor_set_key))},",
                f"    .instruction_constraints = {_instruction_constraint_bits_expr(amdgpu_target_instruction_constraints(info, processor))},",
                f"    .lds_bank_service_model_set_ordinal = {_lds_bank_service_model_set_ordinal_expr(model_keys, model_set_ordinals)},",
                "    .kernel_metadata_extensions = {",
                f"      .entries = {metadata_entries},",
                f"      .count = {metadata_count},",
                "    },",
                "  },",
            ]
        )
    lines.extend(["};", ""])
    return lines


def _emit_physical_target_rows(
    targets: Sequence[AmdgpuTargetInfo],
) -> list[str]:
    targets_by_name = {info.target: info for info in targets}
    supported_processors = {info.processor for info in targets}
    lines = ["const loom_amdgpu_physical_target_info_t loom_amdgpu_target_info_physical_target_infos[] = {"]
    for info in AMDGPU_PHYSICAL_TARGET_INFOS:
        if info.processor not in supported_processors:
            continue
        target = targets_by_name[info.target]
        lines.extend(
            [
                "  {",
                f"    .processor_ordinal = {_u16_expr(amdgpu_processor_ordinal(info.processor))},",
                f"    .asic_revision = {_u32_expr(info.asic_revision)},",
                f"    .target_kind = {_u32_expr(target.enum_value)},",
                "  },",
            ]
        )
    lines.extend(["};", ""])
    return lines


def _matrix_coexecution_release_table_symbol(profile: str) -> str:
    return f"kAmdgpuMatrixCoexecutionReleases{_c_ident(profile)}"


def _emit_matrix_coexecution_source_layouts() -> str:
    _validate_matrix_coexecution_sources()
    lines: list[str] = []
    for source_info in AMDGPU_MATRIX_COEXECUTION_SOURCE_INFOS:
        lines.extend(
            [
                f"  [{_matrix_coexecution_source_expr(source_info.source)}] = {{",
                f"    .result_operand_index = {source_info.result_operand_index},",
                f"    .source_operand_start = {source_info.source_operand_start},",
                f"    .source_operand_count = {source_info.source_operand_count},",
                "  },",
            ]
        )
    return "\n".join(lines) + "\n"


def _emit_matrix_coexecution_rows() -> list[str]:
    lines: list[str] = []
    for profile, rules in AMDGPU_MATRIX_COEXECUTION_RULES_BY_PROFILE.items():
        if not rules:
            continue
        lines.append(
            "static const loom_amdgpu_matrix_coexecution_release_t "
            f"{_matrix_coexecution_release_table_symbol(profile)}"
            "[LOOM_AMDGPU_MATRIX_COEXECUTION_SOURCE_COUNT]"
            "[LOOM_AMDGPU_MATRIX_COEXECUTION_LATENCY_COUNT] = {"
        )
        for source_info in AMDGPU_MATRIX_COEXECUTION_SOURCE_INFOS:
            source_rules = tuple(rule for rule in rules if rule.source == source_info.source)
            if not source_rules:
                continue
            lines.append(f"  [{_matrix_coexecution_source_expr(source_info.source)}] = {{")
            for rule in source_rules:
                lines.extend(
                    [
                        f"    [{rule.latency_cycles}] = {{",
                        f"      .matrix_issue_distance = {rule.matrix_issue_distance},",
                        f"      .vector_issue_distance = {rule.vector_issue_distance},",
                        "    },",
                    ]
                )
            lines.append("  },")
        lines.extend(["};", ""])

    lines.append("const loom_amdgpu_matrix_coexecution_profile_info_t loom_amdgpu_target_info_matrix_coexecution_profile_infos[] = {")
    for profile, rules in AMDGPU_MATRIX_COEXECUTION_RULES_BY_PROFILE.items():
        profile_expr = _matrix_coexecution_profile_expr(profile)
        if not rules:
            lines.append(f"  [{profile_expr}] = {{0}},")
            continue
        maximum_issue_distance = max(max(rule.matrix_issue_distance, rule.vector_issue_distance) for rule in rules)
        lines.extend(
            [
                f"  [{profile_expr}] = {{",
                f"    .releases = {_matrix_coexecution_release_table_symbol(profile)},",
                f"    .maximum_issue_distance = {maximum_issue_distance},",
                "  },",
            ]
        )
    lines.extend(["};", ""])
    return lines


def _emit_processor_rows(
    processors: Sequence[AmdgpuProcessorInfo],
    model_set_ordinals: Mapping[tuple[str, ...], int],
) -> list[str]:
    lines = [
        "const loom_amdgpu_processor_info_t loom_amdgpu_target_info_processor_infos[] = {",
    ]
    processor_ordinals = {info.processor: ordinal for ordinal, info in enumerate(processors)}
    for processor_ordinal, info in enumerate(processors):
        kernel_descriptor = info.kernel_descriptor
        (
            generic_processor_ordinal,
            generic_introduction_version,
        ) = _processor_generic_code_object_fields(info, processor_ordinals)
        generic_processor_ordinal_expr = "LOOM_AMDGPU_PROCESSOR_ORDINAL_NONE" if generic_processor_ordinal is None else _u16_expr(generic_processor_ordinal)
        lines.extend(
            [
                "  {",
                f"    .name = IREE_SVL({_c_string_arg(info.processor)}),",
                f"    .ordinal = {_u16_expr(processor_ordinal)},",
                "    .target_id = {",
                f"      .supported_features = {_target_id_feature_support_flags_expr(info.target_id.supported_features)},",
                "    },",
                "    .generic_code_object = {",
                f"      .processor_ordinal = {generic_processor_ordinal_expr},",
                f"      .introduction_version = {_u16_expr(generic_introduction_version)},",
                "    },",
                "    .properties = {",
                f"      .occupancy_model_ordinal = {_u16_expr(processor_ordinal)},",
                f"      .flags = {_processor_info_flags_expr(info.flags)},",
                "      .descriptor_set = {",
                f"        .key = IREE_SVL({_c_string_arg(info.descriptor_set.key)}),",
                f"        .ordinal = {_processor_descriptor_set_ordinal_expr(info)},",
                "      },",
                "      .elf = {",
                f"        .machine_flags = UINT32_C(0x{info.elf.machine_flags:03x}),",
                f"        .feature_flags = UINT32_C(0x{info.elf.feature_flags:x}),",
                f"        .generic_version = UINT32_C({info.elf.generic_version}),",
                "      },",
                "      .wavefront = {",
                f"        .default_size = {info.wavefront.default_size},",
                f"        .supported_sizes = {_wavefront_size_flags_expr(_supported_wavefront_sizes(info))},",
                "      },",
                "      .kernel_descriptor = {",
                f"        .profile = {_kernel_descriptor_profile_expr(kernel_descriptor.profile)},",
                f"        .flags = {_kernel_descriptor_abi_flags_expr(kernel_descriptor.flags)},",
                "        .vgpr_granules = {",
                f"          .wave32 = {kernel_descriptor.vgpr_granules.wave32},",
                f"          .wave64 = {kernel_descriptor.vgpr_granules.wave64},",
                "        },",
                "      },",
                "      .kernel_entry = {",
                f"        .profile = {_kernel_entry_profile_expr(info.kernel_entry.profile)},",
                "      },",
                "      .instructions = {",
                f"        .base_constraints = {_instruction_constraint_bits_expr(info.instructions.base_constraints)},",
                "      },",
                "      .features = {",
                f"        .matrix = {_matrix_feature_profile_expr(info.features.matrix)},",
                f"        .matrix_coexecution = {_matrix_coexecution_profile_expr(info.features.matrix_coexecution)},",
                f"        .scheduling = {_processor_scheduling_bits_expr(info.features.scheduling)},",
                f"        .lds_bank_service_model_set_ordinal = {_lds_bank_service_model_set_ordinal_expr(info.features.lds_bank_service_models, model_set_ordinals)},",
                "      },",
                "    },",
                "  },",
            ]
        )
    lines.extend(["};", ""])
    return lines


def _emit_tables_source(
    processors: Sequence[AmdgpuProcessorInfo],
    targets: Sequence[AmdgpuTargetInfo],
    descriptor_set_rows: Sequence[_AmdgpuDescriptorSetRow],
) -> str:
    model_sets = amdgpu_lds_bank_service_model_sets(processors, targets)
    model_set_ordinals = {model_keys: ordinal for ordinal, model_keys in enumerate(model_sets)}
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.amdgpu_target_info"),
        "",
        '#include "loom/target/arch/amdgpu/target_info_tables.h"',
        "",
        "#include <stdint.h>",
        "",
        f'const iree_string_view_t loom_amdgpu_target_info_amdhsa_target_id_prefix = IREE_SVL("{_c_string_literal(AMDGPU_AMDHSA_TARGET_TRIPLE)}--");',
        "",
        "// clang-format off",
    ]
    lines.extend(_emit_descriptor_set_rows(descriptor_set_rows))
    lines.extend(_emit_matrix_coexecution_rows())
    lines.extend(_emit_processor_rows(processors, model_set_ordinals))
    lines.extend(_emit_target_rows(targets, processors, model_set_ordinals))
    lines.extend(_emit_physical_target_rows(targets))
    lines.append("// clang-format on")
    lines.append("")
    lines.extend(
        [
            "const iree_host_size_t",
            "    loom_amdgpu_target_info_descriptor_set_info_count =",
            "        IREE_ARRAYSIZE(loom_amdgpu_target_info_descriptor_set_infos);",
            "",
            "const iree_host_size_t",
            "    loom_amdgpu_target_info_processor_info_count =",
            "        IREE_ARRAYSIZE(loom_amdgpu_target_info_processor_infos);",
            "",
            "const iree_host_size_t",
            "    loom_amdgpu_target_info_target_info_count =",
            "        IREE_ARRAYSIZE(loom_amdgpu_target_info_target_infos);",
            "",
            "const iree_host_size_t",
            "    loom_amdgpu_target_info_physical_target_info_count =",
            "        IREE_ARRAYSIZE(",
            "            loom_amdgpu_target_info_physical_target_infos);",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_target_info_to_paths(
    header_path: Path,
    source_path: Path,
    tables_header_path: Path,
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
    processors: Sequence[AmdgpuProcessorInfo],
    targets: Sequence[AmdgpuTargetInfo],
) -> None:
    _validate_descriptor_sets(descriptor_sets)
    descriptor_set_rows = _materialize_descriptor_set_rows(descriptor_sets)
    _validate_descriptor_set_rows(descriptor_set_rows)
    _validate_matrix_coexecution_profiles()
    _validate_processors(processors, descriptor_sets)
    validate_amdgpu_lds_bank_service_model_infos(amdgpu_descriptor_ref_keys())
    validate_amdgpu_code_object_processor_rows(processors)
    validate_amdgpu_target_id_processor_rows(processors)
    validate_amdgpu_target_rows(processors, targets)
    model_sets = amdgpu_lds_bank_service_model_sets(processors, targets)
    validate_amdgpu_lds_bank_service_model_coverage(model_sets)
    if len(model_sets) >= AMDGPU_LDS_BANK_SERVICE_MODEL_SET_ORDINAL_NONE:
        raise ValueError("AMDGPU LDS bank-service model-set ordinals must fit uint16_t")
    header = _emit_header(descriptor_sets)
    tables_header = _emit_tables_header()
    source = _emit_tables_source(processors, targets, descriptor_set_rows)
    write_text_file(header_path, header)
    write_text_file(source_path, source)
    write_text_file(tables_header_path, tables_header)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU target-info C tables from Loom overlay data.")
    parser.add_argument(
        "--header",
        type=Path,
        required=True,
        help="Generated target-info header path.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Generated target-info source path.",
    )
    parser.add_argument(
        "--tables-header",
        type=Path,
        required=True,
        help="Generated target-info private table header path.",
    )
    parser.add_argument(
        "--cache-policy-encoding-rows",
        type=Path,
        required=True,
        help="Generated memory cache-policy encoding row fragment path.",
    )
    parser.add_argument(
        "--cache-policy-temporal-th",
        type=Path,
        required=True,
        help="Generated memory cache-policy temporal TH fragment path.",
    )
    parser.add_argument(
        "--lds-bank-service-model-rows",
        type=Path,
        required=True,
        help="Generated LDS bank-service model-set fragment path.",
    )
    parser.add_argument(
        "--matrix-coexecution-source-layouts",
        type=Path,
        required=True,
        help="Generated matrix coexecution source-layout fragment path.",
    )
    parser.add_argument(
        "--low-registry-tables",
        type=Path,
        required=True,
        help="Generated low descriptor registry X-macro table path.",
    )
    parser.add_argument(
        "--encoding-tables",
        type=Path,
        required=True,
        help="Generated encoding table X-macro table path.",
    )
    parser.add_argument(
        "--encoding-field-ids",
        type=Path,
        required=True,
        help="Generated encoding field ID X-macro row fragment path.",
    )
    parser.add_argument(
        "--low-alias-source",
        type=Path,
        required=True,
        help="Generated blocked low-alias C source path.",
    )
    parser.add_argument(
        "--target-record-tables",
        type=Path,
        required=True,
        help="Generated target-record X-macro table path.",
    )
    args = parser.parse_args(argv)

    descriptor_sets = sorted_descriptor_set_infos()
    processors = sorted_processor_infos()
    targets = sorted_target_infos()
    _write_target_info_to_paths(
        header_path=args.header,
        source_path=args.source,
        tables_header_path=args.tables_header,
        descriptor_sets=descriptor_sets,
        processors=processors,
        targets=targets,
    )
    write_text_file(
        args.cache_policy_encoding_rows,
        _emit_memory_cache_policy_encoding_rows(),
    )
    write_text_file(
        args.cache_policy_temporal_th,
        _emit_memory_cache_policy_temporal_th(),
    )
    write_text_file(
        args.lds_bank_service_model_rows,
        _emit_lds_bank_service_model_rows(),
    )
    write_text_file(
        args.matrix_coexecution_source_layouts,
        _emit_matrix_coexecution_source_layouts(),
    )
    write_config_tables_to_paths(
        descriptor_sets=descriptor_sets,
        processors=processors,
        targets=targets,
        low_registry_tables_path=args.low_registry_tables,
        encoding_tables_path=args.encoding_tables,
        encoding_field_ids_path=args.encoding_field_ids,
    )
    write_low_aliases_to_path(args.low_alias_source)
    write_target_record_tables_to_path(
        args.target_record_tables,
        descriptor_sets=descriptor_sets,
        processors=processors,
        targets=targets,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
