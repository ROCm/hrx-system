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
    parse_amdgpu_isa_xml_path,
)
from loom.target.arch.amdgpu.names import (  # noqa: E402
    amdgpu_descriptor_set_ordinal_constant_name,
)
from loom.target.arch.amdgpu.target_info import (  # noqa: E402
    AMDGPU_AMDHSA_TARGET_TRIPLE,
    AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_NONE,
    AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT,
    AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
    AMDGPU_DESCRIPTOR_SET_INFO_KNOWN_FLAGS,
    AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE,
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
    AMDGPU_PROCESSOR_SCHEDULING_KNOWN_BITS,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1,
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE,
    AmdgpuDescriptorSetInfo,
    AmdgpuProcessorInfo,
    amdgpu_descriptor_set_ordinal,
    kernel_descriptor_profile_supports_wavefront_size,
    sorted_descriptor_set_infos,
    sorted_processor_infos,
    validate_amdgpu_descriptor_set_isa_xml,
)


@dataclass(frozen=True, slots=True)
class _AmdgpuSoppOpcodeRow:
    nop: int
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


def _padded_arg(value: str, width: int) -> str:
    return f"{value},{' ' * (width - len(value) + 1)}"


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

_VECTOR_MEMORY_CACHE_POLICY_ENCODING_EXPRS = {
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE: "LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE",
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC: "LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC",
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH: "LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH",
    AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1: "LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1",
}

_DESCRIPTOR_SET_INFO_FLAG_EXPRS = (
    (
        AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING,
        "LOOM_AMDGPU_DESCRIPTOR_SET_INFO_FLAG_DESCRIPTOR_PACKET_ENCODING",
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


def _vector_memory_cache_policy_encoding_expr(kind: str) -> str:
    return _enum_expr(
        kind,
        _VECTOR_MEMORY_CACHE_POLICY_ENCODING_EXPRS,
        "vector-memory cache-policy encoding",
    )


def _descriptor_set_info_flags_expr(flags: int) -> str:
    if flags == 0:
        return "UINT64_C(0)"
    remaining_flags = flags
    exprs: list[str] = []
    for flag, expr in _DESCRIPTOR_SET_INFO_FLAG_EXPRS:
        if flags & flag:
            exprs.append(expr)
            remaining_flags &= ~flag
    if remaining_flags != 0:
        raise ValueError(f"unknown AMDGPU descriptor-set info flags 0x{remaining_flags:x}")
    return " | ".join(exprs)


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
        unknown_descriptor_set_flags = info.flags & ~AMDGPU_DESCRIPTOR_SET_INFO_KNOWN_FLAGS
        if unknown_descriptor_set_flags != 0:
            raise ValueError(f"AMDGPU descriptor set {info.key} has unknown info flags 0x{unknown_descriptor_set_flags:x}")
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
        _buffer_resource_cache_swizzle_expr(info.buffer_resource.cache_swizzle)
        _vector_memory_cache_policy_encoding_expr(info.vector_memory.cache_policy_encoding)


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
        if info.wavefront.default_size not in (32, 64):
            raise ValueError(f"AMDGPU default wavefront size for {info.processor} must be 32 or 64")
        _matrix_feature_profile_expr(info.features.matrix)
        if kernel_descriptor.flags < 0 or kernel_descriptor.flags > 0xFFFFFFFFFFFFFFFF:
            raise ValueError(f"AMDGPU kernel descriptor ABI flags for {info.processor} must fit u64")
        unknown_descriptor_flags = kernel_descriptor.flags & ~AMDGPU_KERNEL_DESCRIPTOR_ABI_KNOWN_FLAGS
        if unknown_descriptor_flags != 0:
            raise ValueError(f"AMDGPU processor {info.processor} has unknown kernel descriptor ABI flags 0x{unknown_descriptor_flags:x}")
        if profile == AMDGPU_KERNEL_DESCRIPTOR_PROFILE_NONE:
            if vgpr_granules.wave32 != 0 or vgpr_granules.wave64 != 0:
                raise ValueError(f"AMDGPU processor {info.processor} has no kernel descriptor profile but has VGPR encoding granules")
            profileless_flags = kernel_descriptor.flags & ~AMDGPU_KERNEL_DESCRIPTOR_ABI_FLAGS_PROFILELESS
            if profileless_flags != 0:
                raise ValueError(f"AMDGPU processor {info.processor} has no kernel descriptor profile but has profile-owned ABI flags 0x{profileless_flags:x}")
        else:
            if not kernel_descriptor_profile_supports_wavefront_size(profile, info.wavefront.default_size):
                raise ValueError(f"AMDGPU default wavefront size for {info.processor} is not supported by its kernel descriptor profile")
            if vgpr_granules.wave32 == 0 or vgpr_granules.wave64 == 0:
                raise ValueError(f"AMDGPU processor {info.processor} has descriptor profile but no VGPR encoding granules")
            if info.elf.machine_flags == 0:
                raise ValueError(f"AMDGPU processor {info.processor} has a kernel descriptor profile but no ELF machine flags")
        if info.features.scheduling < 0 or info.features.scheduling > 0xFFFFFFFF:
            raise ValueError(f"AMDGPU scheduling bits for {info.processor} must fit u32")
        unknown_scheduling_bits = info.features.scheduling & ~AMDGPU_PROCESSOR_SCHEDULING_KNOWN_BITS
        if unknown_scheduling_bits != 0:
            raise ValueError(f"AMDGPU processor {info.processor} has unknown scheduling bits 0x{unknown_scheduling_bits:x}")


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
    key_width = max(len(_c_string_arg(row.info.key)) for row in rows)
    ordinal_width = len("UINT16_C(65535)")
    opcode_width = len("0x000")
    flags_width = max(len(_descriptor_set_info_flags_expr(row.info.flags)) for row in rows)
    cache_swizzle_width = len("LOOM_AMDGPU_BUFFER_RESOURCE_CACHE_SWIZZLE_STRIDE14_ENABLE_BIT")
    lines = [
        "#define LOOM_AMDGPU_DESCRIPTOR_SET_INFO(ordinal_, key_, sopp_nop_, sopp_endpgm_, sopp_branch_, sopp_cbranch_scc0_, sopp_cbranch_scc1_, flags_, buffer_resource_cache_swizzle_, vector_memory_cache_policy_encoding_) \\",
        "  { \\",
        "    .key = IREE_SVL(key_), \\",
        "    .ordinal = ordinal_, \\",
        "    .sopp = { \\",
        "      .nop = UINT16_C(sopp_nop_), \\",
        "      .endpgm = UINT16_C(sopp_endpgm_), \\",
        "      .branch = UINT16_C(sopp_branch_), \\",
        "      .conditional_branch_scc0 = UINT16_C(sopp_cbranch_scc0_), \\",
        "      .conditional_branch_scc1 = UINT16_C(sopp_cbranch_scc1_), \\",
        "    }, \\",
        "    .flags = flags_, \\",
        "    .buffer_resource = { \\",
        "      .cache_swizzle = buffer_resource_cache_swizzle_, \\",
        "    }, \\",
        "    .vector_memory = { \\",
        "      .cache_policy_encoding = vector_memory_cache_policy_encoding_, \\",
        "    }, \\",
        "  }",
        "",
        "const loom_amdgpu_descriptor_set_info_t loom_amdgpu_target_info_descriptor_set_infos[] = {",
        "  // ordinal         key                    s_nop s_endpgm s_branch s_cbranch_scc0 s_cbranch_scc1 flags cache_swizzle cache_policy",
    ]
    lines.extend(
        (
            "  LOOM_AMDGPU_DESCRIPTOR_SET_INFO("
            f"{_padded_arg(_u16_expr(amdgpu_descriptor_set_ordinal(info.key)), ordinal_width)}"
            f"{_padded_arg(_c_string_arg(info.key), key_width)}"
            f"{_padded_arg(f'0x{row.sopp.nop:03x}', opcode_width)}"
            f"{_padded_arg(f'0x{row.sopp.endpgm:03x}', opcode_width)}"
            f"{_padded_arg(f'0x{row.sopp.branch:03x}', opcode_width)}"
            f"{_padded_arg(f'0x{row.sopp.conditional_branch_scc0:03x}', opcode_width)}"
            f"{_padded_arg(f'0x{row.sopp.conditional_branch_scc1:03x}', opcode_width)}"
            f"{_padded_arg(_descriptor_set_info_flags_expr(info.flags), flags_width)}"
            f"{_padded_arg(_buffer_resource_cache_swizzle_expr(info.buffer_resource.cache_swizzle), cache_swizzle_width)}"
            f"{_vector_memory_cache_policy_encoding_expr(info.vector_memory.cache_policy_encoding)}"
            "),"
        )
        for row in rows
        for info in (row.info,)
    )
    lines.extend(["};", "", "#undef LOOM_AMDGPU_DESCRIPTOR_SET_INFO", ""])
    return lines


def _emit_processor_rows(processors: Sequence[AmdgpuProcessorInfo]) -> list[str]:
    processor_width = max(len(_c_string_arg(info.processor)) for info in processors)
    descriptor_set_width = max(len(_c_string_arg(info.descriptor_set.key)) for info in processors)
    ordinal_width = len("UINT16_C(65535)")
    machine_flags_width = len("0x000")
    feature_flags_width = len("0x0")
    wavefront_width = 2
    kernel_profile_width = max(len(_kernel_descriptor_profile_expr(info.kernel_descriptor.profile)) for info in processors)
    kernel_flags_width = max(len(f"0x{info.kernel_descriptor.flags:x}") for info in processors)
    matrix_profile_width = max(len(_matrix_feature_profile_expr(info.features.matrix)) for info in processors)
    scheduling_width = max(len(f"0x{info.features.scheduling:03x}") for info in processors)
    register_granule_width = 1
    lines = [
        "#define LOOM_AMDGPU_PROCESSOR_INFO(processor_, descriptor_set_key_, descriptor_set_ordinal_, elf_machine_flags_, elf_feature_flags_, default_wavefront_size_, kd_profile_, kd_flags_, matrix_profile_, scheduling_bits_, vgpr_granule_wave32_, vgpr_granule_wave64_) \\",
        "  { \\",
        "    .name = IREE_SVL(processor_), \\",
        "    .descriptor_set = { \\",
        "      .key = IREE_SVL(descriptor_set_key_), \\",
        "      .ordinal = descriptor_set_ordinal_, \\",
        "    }, \\",
        "    .elf = { \\",
        "      .machine_flags = UINT32_C(elf_machine_flags_), \\",
        "      .feature_flags = UINT32_C(elf_feature_flags_), \\",
        "    }, \\",
        "    .wavefront = { \\",
        "      .default_size = default_wavefront_size_, \\",
        "    }, \\",
        "    .kernel_descriptor = { \\",
        "      .profile = kd_profile_, \\",
        "      .flags = UINT64_C(kd_flags_), \\",
        "      .vgpr_granules = { \\",
        "        .wave32 = vgpr_granule_wave32_, \\",
        "        .wave64 = vgpr_granule_wave64_, \\",
        "      }, \\",
        "    }, \\",
        "    .features = { \\",
        "      .matrix = matrix_profile_, \\",
        "      .scheduling = UINT32_C(scheduling_bits_), \\",
        "    }, \\",
        "  }",
        "",
        "const loom_amdgpu_processor_info_t loom_amdgpu_target_info_processor_infos[] = {",
        "  // processor descriptor_set_key    ordinal         mach  feat wave kernel_profile                              kd_flags matrix_profile                             sched vgpr32 vgpr64",
    ]
    lines.extend(
        (
            "  LOOM_AMDGPU_PROCESSOR_INFO("
            f"{_padded_arg(_c_string_arg(info.processor), processor_width)}"
            f"{_padded_arg(_c_string_arg(info.descriptor_set.key), descriptor_set_width)}"
            f"{_padded_arg(_u16_expr(amdgpu_descriptor_set_ordinal(info.descriptor_set.key)) if info.descriptor_set.key else 'LOOM_AMDGPU_DESCRIPTOR_SET_ORDINAL_NONE', ordinal_width)}"
            f"{_padded_arg(f'0x{info.elf.machine_flags:03x}', machine_flags_width)}"
            f"{_padded_arg(f'0x{info.elf.feature_flags:x}', feature_flags_width)}"
            f"{_padded_arg(str(info.wavefront.default_size), wavefront_width)}"
            f"{_padded_arg(_kernel_descriptor_profile_expr(info.kernel_descriptor.profile), kernel_profile_width)}"
            f"{_padded_arg(f'0x{info.kernel_descriptor.flags:x}', kernel_flags_width)}"
            f"{_padded_arg(_matrix_feature_profile_expr(info.features.matrix), matrix_profile_width)}"
            f"{_padded_arg(f'0x{info.features.scheduling:03x}', scheduling_width)}"
            f"{_padded_arg(str(info.kernel_descriptor.vgpr_granules.wave32), register_granule_width)}"
            f"{info.kernel_descriptor.vgpr_granules.wave64!s}),"
        )
        for info in processors
    )
    lines.extend(["};", "", "#undef LOOM_AMDGPU_PROCESSOR_INFO", ""])
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
        required=True,
        type=Path,
        help="Generated target-info header path.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Generated target-info source path.",
    )
    parser.add_argument(
        "--tables-header",
        required=True,
        type=Path,
        help="Generated target-info private table header path.",
    )
    parser.add_argument(
        "--isa-xml",
        action="append",
        default=[],
        help="ISA XML fact source as key:path.",
    )
    args = parser.parse_args(argv)

    write_target_info_to_paths(
        header_path=args.header,
        source_path=args.source,
        tables_header_path=args.tables_header,
        isa_xml_arguments=args.isa_xml,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
