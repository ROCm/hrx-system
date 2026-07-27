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

from loom.gen.support.c import c_string_arg as _c_string_arg  # noqa: E402
from loom.gen.support.c import c_string_literal as _c_string_literal  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.amdgpu.isa_xml import (  # noqa: E402
    AmdgpuIsaFactSource,
    AmdgpuIsaXmlError,
    parse_amdgpu_isa_xml_path,
)
from loom.target.arch.amdgpu.names import (  # noqa: E402
    amdgpu_descriptor_set_ordinal_constant_name,
)
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AMDGPU_AMDHSA_TARGET_TRIPLE,
    AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE,
    AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT,
    AMDGPU_BUFFER_RESOURCE_FLAGS_GFX9,
    AMDGPU_BUFFER_RESOURCE_FLAGS_GFX10_12,
    AMDGPU_BUFFER_RESOURCE_FLAGS_GFX125X,
    AMDGPU_BUFFER_RESOURCE_LAYOUT_LEGACY_32,
    AMDGPU_BUFFER_RESOURCE_LAYOUT_PACKED_45,
    AMDGPU_CACHE_SCOPE_KEYWORDS,
    AMDGPU_CACHE_TEMPORAL_KEYWORDS,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION,
    AMDGPU_DESCRIPTOR_SET_INFO_KNOWN_FLAGS,
    AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE,
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
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940,
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950,
    AMDGPU_MATRIX_FEATURE_PROFILE_NONE,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12,
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250,
    AMDGPU_PROCESSOR_INFO_FLAG_CLUSTER_LAUNCH_STATE,
    AMDGPU_PROCESSOR_INFO_FLAG_GFX125X_ENTRY_ENVELOPE,
    AMDGPU_PROCESSOR_INFO_FLAG_HSACO_EMISSION,
    AMDGPU_PROCESSOR_INFO_KNOWN_FLAGS,
    AMDGPU_PROCESSOR_SCHEDULING_DELAY_ALU,
    AMDGPU_PROCESSOR_SCHEDULING_KNOWN_BITS,
    AMDGPU_PROCESSOR_SCHEDULING_SDWA_DST_SEL_WAIT_STATES,
    AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_DEPCTR,
    AMDGPU_PROCESSOR_SCHEDULING_VALU_SGPR_READ_WAIT_STATES,
    AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_DEPCTR,
    AMDGPU_PROCESSOR_SCHEDULING_VALU_TRANS_USE_WAIT_STATES,
    AMDGPU_PROCESSOR_SCHEDULING_VMEM_RESULT_WRITES_IN_ORDER,
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
    AmdgpuVectorMemoryCachePolicyEncodingInfo,
    amdgpu_descriptor_set_ordinal,
    kernel_descriptor_profile_supports_wavefront_size,
    sorted_descriptor_set_infos,
    sorted_processor_infos,
    validate_amdgpu_descriptor_set_isa_xml,
)


@dataclass(frozen=True, slots=True)
class _AmdgpuSoppOpcodeRow:
    nop: int
    delay_alu: int
    endpgm: int
    branch: int
    conditional_branch_scc0: int
    conditional_branch_scc1: int


@dataclass(frozen=True, slots=True)
class _AmdgpuDescriptorSetRow:
    info: AmdgpuDescriptorSetInfo
    sopp: _AmdgpuSoppOpcodeRow


def _u16_expr(value: int) -> str:
    return f"UINT16_C({value})"


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

_MATRIX_FEATURE_PROFILE_EXPRS = {
    AMDGPU_MATRIX_FEATURE_PROFILE_NONE: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_NONE",
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX908",
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX90A",
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX940",
    AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_MFMA_GFX950",
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX11",
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX12",
    AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250: "LOOM_AMDGPU_MATRIX_FEATURE_PROFILE_WMMA_GFX1250",
}

_BUFFER_RESOURCE_CACHE_SWIZZLE_EXPRS = {
    AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE: "LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE",
    AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT: "LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT",
}

_BUFFER_RESOURCE_LAYOUT_EXPRS = {
    AMDGPU_BUFFER_RESOURCE_LAYOUT_LEGACY_32: "LOOM_AMDGPU_BUFFER_RESOURCE_LAYOUT_LEGACY_32",
    AMDGPU_BUFFER_RESOURCE_LAYOUT_PACKED_45: "LOOM_AMDGPU_BUFFER_RESOURCE_LAYOUT_PACKED_45",
}

_VECTOR_MEMORY_CACHE_POLICY_ENCODING_EXPRS = {encoding: f"LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_{_c_ident(encoding)}" for encoding in AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODINGS}

_DESCRIPTOR_SET_INFO_FLAG_EXPRS = (
    (
        AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
        "LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING",
    ),
    (
        AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION,
        "LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_VOPD_PACKETIZATION",
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
        AMDGPU_PROCESSOR_INFO_FLAG_GFX125X_ENTRY_ENVELOPE,
        "LOOM_AMDGPU_PROCESSOR_INFO_FLAG_GFX125X_ENTRY_ENVELOPE",
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
        AMDGPU_PROCESSOR_SCHEDULING_SDWA_DST_SEL_WAIT_STATES,
        "LOOM_AMDGPU_PROCESSOR_SCHEDULING_SDWA_DST_SEL_WAIT_STATES",
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


def _matrix_feature_profile_expr(profile: str) -> str:
    return _enum_expr(profile, _MATRIX_FEATURE_PROFILE_EXPRS, "matrix feature profile")


def _buffer_resource_cache_swizzle_expr(kind: str) -> str:
    return _enum_expr(
        kind,
        _BUFFER_RESOURCE_CACHE_SWIZZLE_EXPRS,
        "buffer-resource cache swizzle kind",
    )


def _buffer_resource_layout_expr(kind: str) -> str:
    return _enum_expr(
        kind,
        _BUFFER_RESOURCE_LAYOUT_EXPRS,
        "buffer-resource layout",
    )


def _vector_memory_cache_policy_encoding_expr(kind: str) -> str:
    return _enum_expr(
        kind,
        _VECTOR_MEMORY_CACHE_POLICY_ENCODING_EXPRS,
        "vector-memory cache-policy encoding",
    )


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


def _parse_isa_xml_argument(value: str) -> tuple[str, Path]:
    key, separator, path = value.partition(":")
    if not separator or not key or not path:
        raise ValueError("AMDGPU target-info --isa-xml entries must be key:path pairs")
    return key, Path(path)


def _parse_isa_xml_arguments(
    values: Sequence[str],
) -> dict[str, AmdgpuIsaFactSource]:
    specs: dict[str, AmdgpuIsaFactSource] = {}
    for value in values:
        key, path = _parse_isa_xml_argument(value)
        if key in specs:
            raise ValueError(f"AMDGPU target-info ISA XML key '{key}' is duplicate")
        specs[key] = parse_amdgpu_isa_xml_path(path)
    return specs


def _sopp_opcode(spec: AmdgpuIsaFactSource, instruction_name: str) -> int:
    summaries = tuple(
        summary for summary in spec.instruction_encoding_summaries((instruction_name,), include_aliases=False) if summary.encoding_name == "ENC_SOPP" and summary.condition_name == "default"
    )
    if len(summaries) != 1:
        raise ValueError(f"{spec.source_name}: expected one default ENC_SOPP encoding for {instruction_name}, found {len(summaries)}")
    return summaries[0].opcode


def _sopp_opcode_or_zero(spec: AmdgpuIsaFactSource, instruction_name: str) -> int:
    try:
        summaries = tuple(
            summary for summary in spec.instruction_encoding_summaries((instruction_name,), include_aliases=False) if summary.encoding_name == "ENC_SOPP" and summary.condition_name == "default"
        )
    except AmdgpuIsaXmlError as exc:
        message = str(exc)
        if "unknown AMDGPU ISA instruction(s)" not in message or instruction_name not in message:
            raise
        return 0
    if not summaries:
        return 0
    if len(summaries) != 1:
        raise ValueError(f"{spec.source_name}: expected at most one default ENC_SOPP encoding for {instruction_name}, found {len(summaries)}")
    return summaries[0].opcode


def _materialize_descriptor_set_rows(
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
    isa_specs: Mapping[str, AmdgpuIsaFactSource],
) -> tuple[_AmdgpuDescriptorSetRow, ...]:
    rows: list[_AmdgpuDescriptorSetRow] = []
    for info in descriptor_sets:
        spec = isa_specs.get(info.isa_xml_key)
        if spec is None:
            raise ValueError(f"AMDGPU descriptor set {info.key} references missing ISA XML key '{info.isa_xml_key}'")
        validate_amdgpu_descriptor_set_isa_xml(info, spec)
        rows.append(
            _AmdgpuDescriptorSetRow(
                info=info,
                sopp=_AmdgpuSoppOpcodeRow(
                    nop=_sopp_opcode(spec, "S_NOP"),
                    delay_alu=_sopp_opcode_or_zero(spec, "S_DELAY_ALU"),
                    endpgm=_sopp_opcode(spec, "S_ENDPGM"),
                    branch=_sopp_opcode(spec, "S_BRANCH"),
                    conditional_branch_scc0=_sopp_opcode(spec, "S_CBRANCH_SCC0"),
                    conditional_branch_scc1=_sopp_opcode(spec, "S_CBRANCH_SCC1"),
                ),
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
        if not info.isa_xml_key:
            raise ValueError(f"AMDGPU ISA XML key is required for {info.key}")
        if not info.isa_architecture_name:
            raise ValueError(f"AMDGPU ISA XML architecture name is required for {info.key}")
        if info.isa_architecture_id <= 0:
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
            if storage_info.isa_xml_key != info.isa_xml_key:
                raise ValueError(f"AMDGPU descriptor set {info.key} storage target '{storage_info.generator_target}' uses ISA XML key '{storage_info.isa_xml_key}', expected '{info.isa_xml_key}'")
        buffer_resource = info.buffer_resource
        _buffer_resource_layout_expr(buffer_resource.layout)
        if buffer_resource.intrinsic_flags < 0 or buffer_resource.intrinsic_flags > 0xFFFFFFFF:
            raise ValueError(f"AMDGPU buffer-resource intrinsic flags for {info.key} must fit u32")
        if buffer_resource.layout == AMDGPU_BUFFER_RESOURCE_LAYOUT_LEGACY_32:
            if buffer_resource.intrinsic_flags not in (
                AMDGPU_BUFFER_RESOURCE_FLAGS_GFX9,
                AMDGPU_BUFFER_RESOURCE_FLAGS_GFX10_12,
            ):
                raise ValueError(f"AMDGPU legacy buffer-resource flags for {info.key} must name a supported raw-buffer policy")
        elif buffer_resource.layout == AMDGPU_BUFFER_RESOURCE_LAYOUT_PACKED_45:
            if buffer_resource.intrinsic_flags != AMDGPU_BUFFER_RESOURCE_FLAGS_GFX125X:
                raise ValueError(f"AMDGPU packed-45 buffer-resource flags for {info.key} must preserve reserved type bits")
            if buffer_resource.cache_swizzle != AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE:
                raise ValueError(f"AMDGPU packed-45 buffer resources for {info.key} do not support cache swizzle")
        _buffer_resource_cache_swizzle_expr(info.buffer_resource.cache_swizzle)
        _vector_memory_cache_policy_encoding_expr(info.vector_memory.cache_policy_encoding)
        if info.vector_memory.cache_policy_encoding == AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE:
            raise ValueError(f"AMDGPU descriptor set {info.key} must declare a non-none vector-memory cache-policy encoding")


def _validate_descriptor_set_rows(rows: Sequence[_AmdgpuDescriptorSetRow]) -> None:
    for row in rows:
        opcode_rows = (
            ("s_nop", row.sopp.nop),
            ("s_endpgm", row.sopp.endpgm),
            ("s_branch", row.sopp.branch),
            ("s_cbranch_scc0", row.sopp.conditional_branch_scc0),
            ("s_cbranch_scc1", row.sopp.conditional_branch_scc1),
        )
        for name, opcode in opcode_rows:
            if opcode < 0 or opcode > 0xFFFF:
                raise ValueError(f"AMDGPU {name} opcode for {row.info.key} must fit u16")


def _validate_processors(
    processors: Sequence[AmdgpuProcessorInfo],
    descriptor_sets: Sequence[AmdgpuDescriptorSetInfo],
) -> None:
    processor_names = [info.processor for info in processors]
    if processor_names != sorted(processor_names):
        raise ValueError("AMDGPU processor target-info keys must be sorted")
    if len(processor_names) != len(set(processor_names)):
        raise ValueError("AMDGPU processor target-info keys must be unique")
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
        if info.flags < 0 or info.flags > 0xFFFFFFFF:
            raise ValueError(f"AMDGPU processor info flags for {info.processor} must fit u32")
        _processor_info_flags_expr(info.flags)
        if info.wavefront.default_size not in (32, 64):
            raise ValueError(f"AMDGPU default wavefront size for {info.processor} must be 32 or 64")
        supported_wavefront_sizes = _supported_wavefront_sizes(info)
        _matrix_feature_profile_expr(info.features.matrix)
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
        "extern const iree_string_view_t",
        "    loom_amdgpu_target_info_amdhsa_target_id_prefix;",
        "",
        "extern const loom_amdgpu_descriptor_set_info_t",
        "    loom_amdgpu_target_info_descriptor_set_infos[];",
        "extern const iree_host_size_t",
        "    loom_amdgpu_target_info_descriptor_set_info_count;",
        "",
        "extern const loom_amdgpu_processor_info_t",
        "    loom_amdgpu_target_info_processor_infos[];",
        "extern const iree_host_size_t",
        "    loom_amdgpu_target_info_processor_info_count;",
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
                f"      .layout = {_buffer_resource_layout_expr(info.buffer_resource.layout)},",
                f"      .cache_swizzle = {_buffer_resource_cache_swizzle_expr(info.buffer_resource.cache_swizzle)},",
                f"      .intrinsic_flags = UINT32_C(0x{info.buffer_resource.intrinsic_flags:08x}),",
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


def _emit_processor_rows(processors: Sequence[AmdgpuProcessorInfo]) -> list[str]:
    lines = [
        "const loom_amdgpu_processor_info_t loom_amdgpu_target_info_processor_infos[] = {",
    ]
    for info in processors:
        kernel_descriptor = info.kernel_descriptor
        lines.extend(
            [
                "  {",
                f"    .name = IREE_SVL({_c_string_arg(info.processor)}),",
                f"    .flags = {_processor_info_flags_expr(info.flags)},",
                "    .descriptor_set = {",
                f"      .key = IREE_SVL({_c_string_arg(info.descriptor_set.key)}),",
                f"      .ordinal = {_processor_descriptor_set_ordinal_expr(info)},",
                "    },",
                "    .elf = {",
                f"      .machine_flags = UINT32_C(0x{info.elf.machine_flags:03x}),",
                f"      .feature_flags = UINT32_C(0x{info.elf.feature_flags:x}),",
                "    },",
                "    .wavefront = {",
                f"      .default_size = {info.wavefront.default_size},",
                f"      .supported_sizes = {_wavefront_size_flags_expr(_supported_wavefront_sizes(info))},",
                "    },",
                "    .kernel_descriptor = {",
                f"      .profile = {_kernel_descriptor_profile_expr(kernel_descriptor.profile)},",
                f"      .flags = {_kernel_descriptor_abi_flags_expr(kernel_descriptor.flags)},",
                "      .vgpr_granules = {",
                f"        .wave32 = {kernel_descriptor.vgpr_granules.wave32},",
                f"        .wave64 = {kernel_descriptor.vgpr_granules.wave64},",
                "      },",
                "    },",
                "    .features = {",
                f"      .matrix = {_matrix_feature_profile_expr(info.features.matrix)},",
                f"      .scheduling = {_processor_scheduling_bits_expr(info.features.scheduling)},",
                "    },",
                "  },",
            ]
        )
    lines.extend(["};", ""])
    return lines


def _emit_tables_source(
    processors: Sequence[AmdgpuProcessorInfo],
    descriptor_set_rows: Sequence[_AmdgpuDescriptorSetRow],
) -> str:
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
    lines.extend(_emit_processor_rows(processors))
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
        ]
    )
    return "\n".join(lines) + "\n"


def write_target_info_to_paths(
    header_path: Path,
    source_path: Path,
    tables_header_path: Path,
    isa_xml_arguments: Sequence[str],
) -> None:
    descriptor_sets = sorted_descriptor_set_infos()
    processors = sorted_processor_infos()
    isa_specs = _parse_isa_xml_arguments(isa_xml_arguments)
    descriptor_set_rows = _materialize_descriptor_set_rows(descriptor_sets, isa_specs)
    _validate_descriptor_sets(descriptor_sets)
    _validate_descriptor_set_rows(descriptor_set_rows)
    _validate_processors(processors, descriptor_sets)
    header = _emit_header(descriptor_sets)
    tables_header = _emit_tables_header()
    source = _emit_tables_source(processors, descriptor_set_rows)
    header_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.parent.mkdir(parents=True, exist_ok=True)
    tables_header_path.parent.mkdir(parents=True, exist_ok=True)
    header_path.write_text(header, encoding="utf-8")
    source_path.write_text(source, encoding="utf-8")
    tables_header_path.write_text(tables_header, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU target-info C tables from Loom overlay data.")
    parser.add_argument(
        "--header",
        type=Path,
        help="Generated target-info header path.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Generated target-info source path.",
    )
    parser.add_argument(
        "--tables-header",
        type=Path,
        help="Generated target-info private table header path.",
    )
    parser.add_argument(
        "--cache-policy-encoding-rows",
        type=Path,
        help="Generated memory cache-policy encoding row fragment path.",
    )
    parser.add_argument(
        "--cache-policy-temporal-th",
        type=Path,
        help="Generated memory cache-policy temporal TH fragment path.",
    )
    parser.add_argument(
        "--isa-xml",
        action="append",
        default=[],
        help="ISA XML fact source as key:path.",
    )
    args = parser.parse_args(argv)

    wrote_output = False
    target_info_paths = (args.header, args.source, args.tables_header)
    if any(path is not None for path in target_info_paths):
        if not all(path is not None for path in target_info_paths):
            parser.error("--header, --source, and --tables-header must be provided together")
        write_target_info_to_paths(
            header_path=args.header,
            source_path=args.source,
            tables_header_path=args.tables_header,
            isa_xml_arguments=args.isa_xml,
        )
        wrote_output = True
    if args.cache_policy_encoding_rows is not None:
        args.cache_policy_encoding_rows.parent.mkdir(parents=True, exist_ok=True)
        args.cache_policy_encoding_rows.write_text(_emit_memory_cache_policy_encoding_rows(), encoding="utf-8")
        wrote_output = True
    if args.cache_policy_temporal_th is not None:
        args.cache_policy_temporal_th.parent.mkdir(parents=True, exist_ok=True)
        args.cache_policy_temporal_th.write_text(_emit_memory_cache_policy_temporal_th(), encoding="utf-8")
        wrote_output = True
    if not wrote_output:
        parser.error("at least one output path is required")
    return 0


if __name__ == "__main__":
    sys.exit(main())
