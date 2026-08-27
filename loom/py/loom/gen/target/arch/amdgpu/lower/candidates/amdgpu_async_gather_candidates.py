# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Emits AMDGPU source-to-low async gather descriptor candidates."""

from __future__ import annotations

from collections.abc import Sequence

from loom.gen.support.generated_file import line_comment_header
from loom.gen.target.arch.amdgpu.lower.candidates.validation import (
    descriptor_ref_constant_name,
    require_descriptor_refs,
)
from loom.target.arch.amdgpu.descriptors import (
    AmdgpuAsyncGatherDescriptorCandidate,
    amdgpu_async_gather_descriptor_candidates,
)


def _validate_candidate(candidate: AmdgpuAsyncGatherDescriptorCandidate) -> None:
    if candidate.packet_byte_count < 0 or candidate.packet_byte_count > 0xFFFFFFFF:
        raise ValueError(f"AMDGPU async gather descriptor candidate {candidate.descriptor_key} has packet_byte_count {candidate.packet_byte_count}, which does not fit uint32_t")


def _ordered_candidates(
    candidates: Sequence[AmdgpuAsyncGatherDescriptorCandidate],
) -> tuple[AmdgpuAsyncGatherDescriptorCandidate, ...]:
    require_descriptor_refs(
        "AMDGPU async gather descriptor candidate table",
        (candidate.descriptor_key for candidate in candidates),
    )
    candidates_by_packet_byte_count: dict[int, AmdgpuAsyncGatherDescriptorCandidate] = {}
    for candidate in candidates:
        _validate_candidate(candidate)
        existing = candidates_by_packet_byte_count.get(candidate.packet_byte_count)
        if existing is not None and existing.descriptor_key != candidate.descriptor_key:
            raise ValueError(
                f"AMDGPU async gather descriptor candidate table has conflicting rows for packet byte count {candidate.packet_byte_count}: {existing.descriptor_key}, {candidate.descriptor_key}"
            )
        candidates_by_packet_byte_count[candidate.packet_byte_count] = candidate
    return tuple(sorted(candidates, key=lambda candidate: candidate.packet_byte_count))


def _candidate_initializer(candidate: AmdgpuAsyncGatherDescriptorCandidate) -> str:
    descriptor_ref = descriptor_ref_constant_name(candidate.descriptor_key)
    return "\n".join(
        [
            "    {",
            f"        .packet_byte_count = {candidate.packet_byte_count},",
            f"        .descriptor_ref = {descriptor_ref},",
            "    },",
        ]
    )


def _emit_source(*, public_header: str) -> str:
    candidates = _ordered_candidates(amdgpu_async_gather_descriptor_candidates())
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header(
            "//",
            generator="loom.gen.target.arch.amdgpu.lower.candidates.amdgpu_async_gather_candidates",
        ),
        "",
        f'#include "{public_header}"',
        "",
        '#include "loom/target/arch/amdgpu/refs/target_refs.h"',
        "",
        "const loom_amdgpu_async_gather_descriptor_candidate_t",
        "    kLoomAmdgpuAsyncGatherDescriptorCandidates[] = {",
        *(_candidate_initializer(candidate) for candidate in candidates),
        "};",
        "",
        "const iree_host_size_t kLoomAmdgpuAsyncGatherDescriptorCandidateCount =",
        "    IREE_ARRAYSIZE(kLoomAmdgpuAsyncGatherDescriptorCandidates);",
    ]
    return "\n".join(lines) + "\n"
