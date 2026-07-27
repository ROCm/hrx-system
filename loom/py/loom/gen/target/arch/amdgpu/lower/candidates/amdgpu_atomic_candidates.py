# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU source-to-low atomic descriptor candidates."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[7]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.gen.target.arch.amdgpu.lower.candidates.validation import (  # noqa: E402
    dense_candidate_ranges,
    descriptor_ref_constant_name,
    require_descriptor_refs,
)
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    AmdgpuAtomicDescriptorCandidate,
    AmdgpuAtomicKind,
    AmdgpuAtomicMemorySpace,
    AmdgpuMemoryAddressForm,
    amdgpu_atomic_descriptor_candidates,
)

_ATOMIC_MEMORY_SPACE_INDEX = {
    AmdgpuAtomicMemorySpace.WORKGROUP: 0,
    AmdgpuAtomicMemorySpace.GLOBAL: 1,
    AmdgpuAtomicMemorySpace.GENERIC: 2,
}

_ATOMIC_ADDRESS_FORM_INDEX = {
    AmdgpuMemoryAddressForm.DEFAULT: 0,
    AmdgpuMemoryAddressForm.GLOBAL_SADDR: 1,
    AmdgpuMemoryAddressForm.FLAT: 2,
}


def _candidate_initializer(candidate: AmdgpuAtomicDescriptorCandidate) -> str:
    descriptor_ref = descriptor_ref_constant_name(candidate.descriptor_key)
    return "\n".join(
        [
            "    {",
            f"        .value_kind = {candidate.value_kind.c_name},",
            f"        .descriptor_ref = {descriptor_ref},",
            "    },",
        ]
    )


def _candidate_atomic_kind_index(candidate: AmdgpuAtomicDescriptorCandidate) -> str:
    if candidate.atomic_kind == AmdgpuAtomicKind.NONE:
        return "LOOM_AMDGPU_ATOMIC_KIND_INDEX_NONE"
    return candidate.atomic_kind.c_name


def _candidate_range_key(
    candidate: AmdgpuAtomicDescriptorCandidate,
) -> tuple[int, int, str, str]:
    return (
        _ATOMIC_MEMORY_SPACE_INDEX[candidate.memory_space],
        _ATOMIC_ADDRESS_FORM_INDEX[candidate.address_form],
        candidate.operation_kind.c_name,
        _candidate_atomic_kind_index(candidate),
    )


def _candidate_range_index_expression(key: tuple[int, int, str, str]) -> str:
    memory_space_index, address_form_index, operation_kind, atomic_kind_index = key
    return f"LOOM_AMDGPU_ATOMIC_DESCRIPTOR_CANDIDATE_RANGE_INDEX({memory_space_index}, {address_form_index}, {operation_kind}, {atomic_kind_index})"


def _candidate_ranges(
    candidates: Sequence[AmdgpuAtomicDescriptorCandidate],
) -> tuple[tuple[tuple[int, int, str, str], int, int], ...]:
    require_descriptor_refs(
        "AMDGPU atomic descriptor candidate table",
        (candidate.descriptor_key for candidate in candidates),
    )
    return dense_candidate_ranges(
        candidates,
        _candidate_range_key,
        owner="AMDGPU atomic descriptor candidate table",
    )


def _range_initializer(
    key: tuple[int, int, str, str],
    first_candidate: int,
    candidate_count: int,
) -> str:
    return "\n".join(
        [
            f"    [{_candidate_range_index_expression(key)}] = {{",
            f"        .first_candidate = {first_candidate},",
            f"        .candidate_count = {candidate_count},",
            "    },",
        ]
    )


def _emit_source(*, public_header: str) -> str:
    candidates = amdgpu_atomic_descriptor_candidates()
    ranges = _candidate_ranges(candidates)
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header(
            "//",
            generator="loom.gen.target.arch.amdgpu.lower.candidates.amdgpu_atomic_candidates",
        ),
        "",
        f'#include "{public_header}"',
        "",
        '#include "loom/ops/atomic.h"',
        '#include "loom/target/arch/amdgpu/refs/target_refs.h"',
        "",
        "const loom_amdgpu_atomic_descriptor_candidate_t",
        "    kLoomAmdgpuAtomicDescriptorCandidates[] = {",
        *(_candidate_initializer(candidate) for candidate in candidates),
        "};",
        "",
        "const iree_host_size_t kLoomAmdgpuAtomicDescriptorCandidateCount =",
        "    IREE_ARRAYSIZE(kLoomAmdgpuAtomicDescriptorCandidates);",
        "",
        "const loom_amdgpu_atomic_descriptor_candidate_range_t",
        "    kLoomAmdgpuAtomicDescriptorCandidateRanges",
        "        [LOOM_AMDGPU_ATOMIC_DESCRIPTOR_CANDIDATE_RANGE_COUNT] = {",
        *(_range_initializer(key, first_candidate, candidate_count) for key, first_candidate, candidate_count in ranges),
        "};",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU source-to-low atomic descriptor candidates.")
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Generated atomic candidate source path.",
    )
    parser.add_argument(
        "--public-header",
        default="loom/target/arch/amdgpu/lower/candidates/atomic_candidates.h",
        help="Public include path for the generated header.",
    )
    args = parser.parse_args(argv)

    args.source.parent.mkdir(parents=True, exist_ok=True)
    args.source.write_text(
        _emit_source(public_header=args.public_header),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
