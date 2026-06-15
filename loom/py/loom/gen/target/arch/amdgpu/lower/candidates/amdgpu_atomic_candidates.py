# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU source-to-low atomic descriptor candidates."""

from __future__ import annotations

import argparse
import re
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
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    AmdgpuAtomicDescriptorCandidate,
    amdgpu_atomic_descriptor_candidates,
)
from loom.target.low_descriptors import target_relative_name  # noqa: E402


def _c_identifier(value: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]", "_", value).strip("_")
    if not identifier:
        return "EMPTY"
    if identifier[0].isdigit():
        identifier = "_" + identifier
    return identifier.upper()


def _descriptor_ref_constant_name(key: str) -> str:
    return f"LOOM_AMDGPU_DESCRIPTOR_REF_{_c_identifier(target_relative_name('amdgpu', key))}"


def _emit_header() -> str:
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.lower.candidates.amdgpu_atomic_candidates"),
        "",
        "#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_ATOMIC_CANDIDATES_H_",
        "#define LOOM_TARGET_ARCH_AMDGPU_LOWER_ATOMIC_CANDIDATES_H_",
        "",
        "#include <stdint.h>",
        "",
        '#include "iree/base/api.h"',
        '#include "loom/target/arch/amdgpu/refs/target_refs.h"',
        '#include "loom/target/arch/amdgpu/lower/kinds.h"',
        '#include "loom/util/fact_table.h"',
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        "#define LOOM_AMDGPU_ATOMIC_KIND_NONE UINT8_MAX",
        "",
        "typedef enum loom_amdgpu_atomic_value_kind_e {",
        "  LOOM_AMDGPU_ATOMIC_VALUE_KIND_I32 = 0,",
        "  LOOM_AMDGPU_ATOMIC_VALUE_KIND_F32 = 1,",
        "} loom_amdgpu_atomic_value_kind_t;",
        "",
        "typedef struct loom_amdgpu_atomic_descriptor_candidate_t {",
        "  // Source memory space matched by this row.",
        "  loom_value_fact_memory_space_t memory_space;",
        "  // Target addressing form emitted by this row.",
        "  loom_amdgpu_memory_address_form_t address_form;",
        "  // Source atomic operation form matched by this row.",
        "  loom_amdgpu_atomic_operation_kind_t operation_kind;",
        "  // Source atomic arithmetic kind matched by this row.",
        "  uint8_t atomic_kind;",
        "  // Source scalar value type required by this row.",
        "  loom_amdgpu_atomic_value_kind_t value_kind;",
        "  // Dense AMDGPU descriptor ref selected when present in the descriptor set.",
        "  loom_amdgpu_descriptor_ref_t descriptor_ref;",
        "} loom_amdgpu_atomic_descriptor_candidate_t;",
        "",
        "extern const loom_amdgpu_atomic_descriptor_candidate_t",
        "    kLoomAmdgpuAtomicDescriptorCandidates[];",
        "extern const iree_host_size_t kLoomAmdgpuAtomicDescriptorCandidateCount;",
        "",
        "#ifdef __cplusplus",
        '}  // extern "C"',
        "#endif",
        "",
        "#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_ATOMIC_CANDIDATES_H_",
    ]
    return "\n".join(lines) + "\n"


def _candidate_initializer(candidate: AmdgpuAtomicDescriptorCandidate) -> str:
    return "\n".join(
        [
            "    {",
            f"        .memory_space = {candidate.memory_space.c_name},",
            f"        .address_form = {candidate.address_form.c_name},",
            f"        .operation_kind = {candidate.operation_kind.c_name},",
            f"        .atomic_kind = {candidate.atomic_kind.c_name},",
            f"        .value_kind = {candidate.value_kind.c_name},",
            f"        .descriptor_ref = {_descriptor_ref_constant_name(candidate.descriptor_key)},",
            "    },",
        ]
    )


def _emit_source(*, public_header: str) -> str:
    candidates = amdgpu_atomic_descriptor_candidates()
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.lower.candidates.amdgpu_atomic_candidates"),
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
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU source-to-low atomic descriptor candidates.")
    parser.add_argument(
        "--header",
        required=True,
        type=Path,
        help="Generated atomic candidate header path.",
    )
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

    args.header.parent.mkdir(parents=True, exist_ok=True)
    args.source.parent.mkdir(parents=True, exist_ok=True)
    args.header.write_text(
        _emit_header(),
        encoding="utf-8",
    )
    args.source.write_text(
        _emit_source(public_header=args.public_header),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
