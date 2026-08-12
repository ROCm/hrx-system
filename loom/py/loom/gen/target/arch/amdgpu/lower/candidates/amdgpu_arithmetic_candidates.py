# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU source-to-low arithmetic descriptor candidates."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[7]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.files import write_text_file  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.gen.target.arch.amdgpu.lower.candidates.validation import (  # noqa: E402
    required_descriptor_ref_constant_name,
)
from loom.target.arch.amdgpu.descriptors import amdgpu_descriptor_ref_keys  # noqa: E402


@dataclass(frozen=True, slots=True)
class _FmaMixSourceKind:
    keyword: str
    c_enum: str


@dataclass(frozen=True, slots=True)
class _FmaMixDescriptorCube:
    array_name: str
    descriptor_key_prefix: str
    include_all_f32: bool


@dataclass(frozen=True, slots=True)
class _FmaMixSrc2LiteralTable:
    array_name: str
    descriptor_key_prefix: str
    include_all_f32: bool


@dataclass(frozen=True, slots=True)
class _PackedTernaryDescriptorCandidate:
    descriptor_key: str
    source_permutation: tuple[int, int, int] = (0, 1, 2)
    flags: tuple[str, ...] = ()
    packet_unit_count: int = 1


@dataclass(frozen=True, slots=True)
class _PackedTernaryDescriptorCandidateArray:
    array_name: str
    count_name: str
    candidates: tuple[_PackedTernaryDescriptorCandidate, ...]


_SOURCE_KINDS = (
    _FmaMixSourceKind(
        keyword="f32",
        c_enum="LOOM_AMDGPU_FMA_MIX_SOURCE_F32",
    ),
    _FmaMixSourceKind(
        keyword="f16lo",
        c_enum="LOOM_AMDGPU_FMA_MIX_SOURCE_F16LO",
    ),
    _FmaMixSourceKind(
        keyword="f16hi",
        c_enum="LOOM_AMDGPU_FMA_MIX_SOURCE_F16HI",
    ),
)

_F32_SOURCE = ("f32", "f32", "f32")
_PACKED_TERNARY_KNOWN_FLAGS = frozenset(
    {
        "LOOM_AMDGPU_PACKED_TERNARY_FLAG_TIED_ACCUMULATOR",
    }
)

_FMA_MIX_DESCRIPTOR_CUBES = (
    _FmaMixDescriptorCube(
        array_name="kLoomAmdgpuFmaMixF32DescriptorRefs",
        descriptor_key_prefix="amdgpu.v_fma_mix_f32",
        include_all_f32=False,
    ),
    _FmaMixDescriptorCube(
        array_name="kLoomAmdgpuMadMixF32DescriptorRefs",
        descriptor_key_prefix="amdgpu.v_mad_mix_f32",
        include_all_f32=False,
    ),
    _FmaMixDescriptorCube(
        array_name="kLoomAmdgpuFmaMixloF16DescriptorRefs",
        descriptor_key_prefix="amdgpu.v_fma_mixlo_f16",
        include_all_f32=True,
    ),
    _FmaMixDescriptorCube(
        array_name="kLoomAmdgpuFmaMixhiF16DescriptorRefs",
        descriptor_key_prefix="amdgpu.v_fma_mixhi_f16",
        include_all_f32=True,
    ),
    _FmaMixDescriptorCube(
        array_name="kLoomAmdgpuMadMixloF16DescriptorRefs",
        descriptor_key_prefix="amdgpu.v_mad_mixlo_f16",
        include_all_f32=True,
    ),
    _FmaMixDescriptorCube(
        array_name="kLoomAmdgpuMadMixhiF16DescriptorRefs",
        descriptor_key_prefix="amdgpu.v_mad_mixhi_f16",
        include_all_f32=True,
    ),
)

_FMA_MIX_SRC2_LITERAL_TABLES = (
    _FmaMixSrc2LiteralTable(
        array_name="kLoomAmdgpuFmaMixF32Src2LiteralDescriptorRefs",
        descriptor_key_prefix="amdgpu.v_fma_mix_f32",
        include_all_f32=False,
    ),
    _FmaMixSrc2LiteralTable(
        array_name="kLoomAmdgpuFmaMixloF16Src2LiteralDescriptorRefs",
        descriptor_key_prefix="amdgpu.v_fma_mixlo_f16",
        include_all_f32=True,
    ),
    _FmaMixSrc2LiteralTable(
        array_name="kLoomAmdgpuFmaMixhiF16Src2LiteralDescriptorRefs",
        descriptor_key_prefix="amdgpu.v_fma_mixhi_f16",
        include_all_f32=True,
    ),
)

_PACKED_TERNARY_DESCRIPTOR_CANDIDATE_ARRAYS = (
    _PackedTernaryDescriptorCandidateArray(
        array_name="kLoomAmdgpuPackedFmafF16DescriptorCandidates",
        count_name="kLoomAmdgpuPackedFmafF16DescriptorCandidateCount",
        candidates=(
            _PackedTernaryDescriptorCandidate(
                descriptor_key="amdgpu.v_pk_fmac_f16",
                source_permutation=(2, 0, 1),
                flags=("LOOM_AMDGPU_PACKED_TERNARY_FLAG_TIED_ACCUMULATOR",),
            ),
            _PackedTernaryDescriptorCandidate(
                descriptor_key="amdgpu.v_pk_fma_f16",
            ),
        ),
    ),
    _PackedTernaryDescriptorCandidateArray(
        array_name="kLoomAmdgpuPackedFmafF32DescriptorCandidates",
        count_name="kLoomAmdgpuPackedFmafF32DescriptorCandidateCount",
        candidates=(
            _PackedTernaryDescriptorCandidate(
                descriptor_key="amdgpu.v_pk_fma_f32",
                packet_unit_count=2,
            ),
        ),
    ),
    _PackedTernaryDescriptorCandidateArray(
        array_name="kLoomAmdgpuPackedFmaiSignedPreferenceDescriptorCandidates",
        count_name="kLoomAmdgpuPackedFmaiSignedPreferenceDescriptorCandidateCount",
        candidates=(
            _PackedTernaryDescriptorCandidate(
                descriptor_key="amdgpu.v_pk_mad_i16",
            ),
            _PackedTernaryDescriptorCandidate(
                descriptor_key="amdgpu.v_pk_mad_u16",
            ),
        ),
    ),
    _PackedTernaryDescriptorCandidateArray(
        array_name="kLoomAmdgpuPackedFmaiUnsignedPreferenceDescriptorCandidates",
        count_name="kLoomAmdgpuPackedFmaiUnsignedPreferenceDescriptorCandidateCount",
        candidates=(
            _PackedTernaryDescriptorCandidate(
                descriptor_key="amdgpu.v_pk_mad_u16",
            ),
            _PackedTernaryDescriptorCandidate(
                descriptor_key="amdgpu.v_pk_mad_i16",
            ),
        ),
    ),
)


def _fma_mix_source_combinations(*, include_all_f32: bool) -> tuple[tuple[str, str, str], ...]:
    combinations = tuple((source0.keyword, source1.keyword, source2.keyword) for source0 in _SOURCE_KINDS for source1 in _SOURCE_KINDS for source2 in _SOURCE_KINDS)
    if include_all_f32:
        return combinations
    return tuple(source_parts for source_parts in combinations if source_parts != _F32_SOURCE)


def _fma_mix_descriptor_key(
    cube: _FmaMixDescriptorCube,
    source_parts: tuple[str, str, str],
) -> str | None:
    if source_parts == _F32_SOURCE and not cube.include_all_f32:
        return None
    return f"{cube.descriptor_key_prefix}.{'_'.join(source_parts)}"


def _fma_mix_src2_literal_descriptor_key(
    table: _FmaMixSrc2LiteralTable,
    source0: str,
    source1: str,
) -> str | None:
    if (source0, source1) == ("f32", "f32") and not table.include_all_f32:
        return None
    return f"{table.descriptor_key_prefix}.{source0}_{source1}_f32.src2_lit"


def _descriptor_ref_constant(
    owner: str,
    descriptor_key: str | None,
    descriptor_ref_key_set: set[str],
) -> str:
    if descriptor_key is None:
        return "LOOM_AMDGPU_DESCRIPTOR_REF_NONE"
    return required_descriptor_ref_constant_name(owner, descriptor_key, descriptor_ref_key_set)


def _emit_fma_mix_cube(
    cube: _FmaMixDescriptorCube,
    descriptor_ref_key_set: set[str],
) -> Iterable[str]:
    yield "const loom_amdgpu_fma_mix_descriptor_ref_cube_t"
    yield f"    {cube.array_name} = {{"
    for source0 in _SOURCE_KINDS:
        yield f"        [{source0.c_enum}] = {{"
        for source1 in _SOURCE_KINDS:
            yield f"            [{source1.c_enum}] = {{"
            for source2 in _SOURCE_KINDS:
                source_parts = (source0.keyword, source1.keyword, source2.keyword)
                descriptor_key = _fma_mix_descriptor_key(cube, source_parts)
                owner = f"AMDGPU FMA-mix descriptor candidate {cube.array_name} {' '.join(source_parts)}"
                descriptor_ref = _descriptor_ref_constant(owner, descriptor_key, descriptor_ref_key_set)
                yield f"                [{source2.c_enum}] = {descriptor_ref},"
            yield "            },"
        yield "        },"
    yield "};"


def _emit_fma_mix_src2_literal_table(
    table: _FmaMixSrc2LiteralTable,
    descriptor_ref_key_set: set[str],
) -> Iterable[str]:
    yield "const loom_amdgpu_fma_mix_src2_literal_descriptor_ref_table_t"
    yield f"    {table.array_name} = {{"
    for source0 in _SOURCE_KINDS:
        yield f"        [{source0.c_enum}] = {{"
        for source1 in _SOURCE_KINDS:
            descriptor_key = _fma_mix_src2_literal_descriptor_key(table, source0.keyword, source1.keyword)
            owner = f"AMDGPU FMA-mix source-2 literal descriptor candidate {table.array_name} {source0.keyword} {source1.keyword}"
            descriptor_ref = _descriptor_ref_constant(owner, descriptor_key, descriptor_ref_key_set)
            yield f"            [{source1.c_enum}] = {descriptor_ref},"
        yield "        },"
    yield "};"


def _validate_packed_ternary_candidate(
    array: _PackedTernaryDescriptorCandidateArray,
    candidate: _PackedTernaryDescriptorCandidate,
    descriptor_ref_key_set: set[str],
) -> None:
    owner = f"AMDGPU packed ternary descriptor candidate {array.array_name}"
    required_descriptor_ref_constant_name(owner, candidate.descriptor_key, descriptor_ref_key_set)
    if sorted(candidate.source_permutation) != [0, 1, 2]:
        raise ValueError(f"{owner} {candidate.descriptor_key} has invalid source permutation {candidate.source_permutation}")
    unknown_flags = sorted(set(candidate.flags) - _PACKED_TERNARY_KNOWN_FLAGS)
    if unknown_flags:
        raise ValueError(f"{owner} {candidate.descriptor_key} has unknown flags {', '.join(unknown_flags)}")
    if candidate.packet_unit_count not in (1, 2):
        raise ValueError(f"{owner} {candidate.descriptor_key} has packet_unit_count {candidate.packet_unit_count}")


def _packed_ternary_flags_expr(candidate: _PackedTernaryDescriptorCandidate) -> str:
    if not candidate.flags:
        return "0"
    return " | ".join(candidate.flags)


def _emit_packed_ternary_candidate_array(
    array: _PackedTernaryDescriptorCandidateArray,
    descriptor_ref_key_set: set[str],
) -> Iterable[str]:
    if not array.candidates:
        raise ValueError(f"AMDGPU packed ternary descriptor array {array.array_name} is empty")
    yield "const loom_amdgpu_packed_ternary_descriptor_candidate_t"
    yield f"    {array.array_name}[] = {{"
    for candidate in array.candidates:
        _validate_packed_ternary_candidate(array, candidate, descriptor_ref_key_set)
        descriptor_ref = required_descriptor_ref_constant_name(
            f"AMDGPU packed ternary descriptor candidate {array.array_name}",
            candidate.descriptor_key,
            descriptor_ref_key_set,
        )
        yield "        {"
        yield f"            .descriptor_ref = {descriptor_ref},"
        yield f"            .source_permutation = {{{', '.join(str(source) for source in candidate.source_permutation)}}},"
        yield f"            .flags = {_packed_ternary_flags_expr(candidate)},"
        yield f"            .packet_unit_count = {candidate.packet_unit_count},"
        yield "        },"
    yield "};"
    yield (f'static_assert({array.count_name} == {len(array.candidates)}u, "{array.count_name} drifted from generated candidate rows");')


def _emit_source(*, public_header: str) -> str:
    descriptor_ref_key_set = set(amdgpu_descriptor_ref_keys())
    data_lines: list[str] = []
    for cube in _FMA_MIX_DESCRIPTOR_CUBES:
        data_lines.extend(_emit_fma_mix_cube(cube, descriptor_ref_key_set))
        data_lines.append("")
        expected_rows = len(_fma_mix_source_combinations(include_all_f32=cube.include_all_f32))
        if expected_rows != (27 if cube.include_all_f32 else 26):
            raise ValueError(f"AMDGPU FMA-mix cube {cube.array_name} generated {expected_rows} source rows")
    for table in _FMA_MIX_SRC2_LITERAL_TABLES:
        data_lines.extend(_emit_fma_mix_src2_literal_table(table, descriptor_ref_key_set))
        data_lines.append("")
    for array in _PACKED_TERNARY_DESCRIPTOR_CANDIDATE_ARRAYS:
        data_lines.extend(_emit_packed_ternary_candidate_array(array, descriptor_ref_key_set))
        data_lines.append("")

    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header(
            "//",
            generator="loom.gen.target.arch.amdgpu.lower.candidates.amdgpu_arithmetic_candidates",
        ),
        "",
        f'#include "{public_header}"',
        "",
        '#include "loom/target/arch/amdgpu/lower/kinds.h"',
        '#include "loom/target/arch/amdgpu/refs/target_refs.h"',
        "",
        *data_lines,
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU source-to-low arithmetic descriptor candidates.")
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Generated arithmetic candidate source path.",
    )
    parser.add_argument(
        "--public-header",
        default="loom/target/arch/amdgpu/lower/candidates/arithmetic_candidates.h",
        help="Public include path for the generated header.",
    )
    args = parser.parse_args(argv)

    write_text_file(
        args.source,
        _emit_source(public_header=args.public_header),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
