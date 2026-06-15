# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU source-to-low compare descriptor candidates."""

from __future__ import annotations

import argparse
import re
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

from loom.dialect.scalar.comparison import CmpFPredicate, CmpIPredicate  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.amdgpu.descriptors import amdgpu_descriptor_ref_keys  # noqa: E402
from loom.target.low_descriptors import target_relative_name  # noqa: E402


@dataclass(frozen=True, slots=True)
class _CompareFamily:
    source_op_name: str
    predicate_c_prefix: str
    descriptor_element_type: str
    predicates: tuple[str, ...]


_I32_PREDICATES = tuple(case.keyword for case in CmpIPredicate.cases)
_F32_PREDICATES = tuple(case.keyword for case in CmpFPredicate.cases)

_COMPARE_FAMILIES = (
    _CompareFamily(
        source_op_name="vector.cmpi",
        predicate_c_prefix="LOOM_VECTOR_CMPI_PREDICATE",
        descriptor_element_type="i32",
        predicates=_I32_PREDICATES,
    ),
    _CompareFamily(
        source_op_name="scalar.cmpf",
        predicate_c_prefix="LOOM_SCALAR_CMPF_PREDICATE",
        descriptor_element_type="f32",
        predicates=_F32_PREDICATES,
    ),
    _CompareFamily(
        source_op_name="vector.cmpf",
        predicate_c_prefix="LOOM_VECTOR_CMPF_PREDICATE",
        descriptor_element_type="f32",
        predicates=_F32_PREDICATES,
    ),
)


def _c_identifier(value: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]", "_", value).strip("_")
    if not identifier:
        return "EMPTY"
    if identifier[0].isdigit():
        identifier = "_" + identifier
    return identifier.upper()


def _op_kind_constant_name(op_name: str) -> str:
    return f"LOOM_OP_{_c_identifier(op_name)}"


def _descriptor_ref_constant_name(key: str) -> str:
    return f"LOOM_AMDGPU_DESCRIPTOR_REF_{_c_identifier(target_relative_name('amdgpu', key))}"


def _descriptor_element_type(predicate: str, element_type: str) -> str:
    if element_type == "i32" and predicate[0] == "u":
        return "u32"
    return element_type


def _descriptor_key(predicate: str, element_type: str) -> str:
    descriptor_element_type = _descriptor_element_type(predicate, element_type)
    return f"amdgpu.v_cmp_{predicate}_{descriptor_element_type}"


def _optional_descriptor_ref_constant_name(key: str, key_set: set[str]) -> str:
    if key not in key_set:
        return "LOOM_AMDGPU_DESCRIPTOR_REF_NONE"
    return _descriptor_ref_constant_name(key)


def _predicate_constant(family: _CompareFamily, predicate: str) -> str:
    return f"{family.predicate_c_prefix}_{_c_identifier(predicate)}"


def _compare_candidates() -> tuple[tuple[_CompareFamily, str, str], ...]:
    descriptor_ref_key_set = set(amdgpu_descriptor_ref_keys())
    candidates: list[tuple[_CompareFamily, str, str]] = []
    for family in _COMPARE_FAMILIES:
        for predicate in family.predicates:
            descriptor_key = _descriptor_key(predicate, family.descriptor_element_type)
            if descriptor_key not in descriptor_ref_key_set:
                raise ValueError(f"compare candidate for {family.source_op_name} {predicate} requires missing descriptor '{descriptor_key}'")
            candidates.append((family, predicate, descriptor_key))
    return tuple(candidates)


def _emit_header() -> str:
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.lower.candidates.amdgpu_compare_candidates"),
        "",
        "#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_COMPARE_CANDIDATES_H_",
        "#define LOOM_TARGET_ARCH_AMDGPU_LOWER_COMPARE_CANDIDATES_H_",
        "",
        "#include <stdint.h>",
        "",
        '#include "iree/base/api.h"',
        '#include "loom/ir/ir.h"',
        '#include "loom/target/arch/amdgpu/refs/target_refs.h"',
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        "typedef struct loom_amdgpu_compare_descriptor_candidate_t {",
        "  // Source compare op kind handled by this row.",
        "  loom_op_kind_t op_kind;",
        "  // Source compare predicate value matched by this row.",
        "  uint8_t predicate;",
        "  // Dense AMDGPU descriptor ref selected when present in the descriptor set.",
        "  loom_amdgpu_descriptor_ref_t descriptor_ref;",
        "  // Dense AMDGPU descriptor ref selected when the left-hand source is inline.",
        "  loom_amdgpu_descriptor_ref_t src0_inline_descriptor_ref;",
        "  // Dense AMDGPU descriptor ref selected when the right-hand source is inline.",
        "  loom_amdgpu_descriptor_ref_t src1_inline_descriptor_ref;",
        "} loom_amdgpu_compare_descriptor_candidate_t;",
        "",
        "extern const loom_amdgpu_compare_descriptor_candidate_t",
        "    kLoomAmdgpuCompareDescriptorCandidates[];",
        "extern const iree_host_size_t kLoomAmdgpuCompareDescriptorCandidateCount;",
        "",
        "#ifdef __cplusplus",
        '}  // extern "C"',
        "#endif",
        "",
        "#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_COMPARE_CANDIDATES_H_",
    ]
    return "\n".join(lines) + "\n"


def _candidate_initializers(
    candidates: Iterable[tuple[_CompareFamily, str, str]],
) -> Iterable[str]:
    descriptor_ref_key_set = set(amdgpu_descriptor_ref_keys())
    for family, predicate, descriptor_key in candidates:
        yield "\n".join(
            [
                "    {",
                f"        .op_kind = {_op_kind_constant_name(family.source_op_name)},",
                f"        .predicate = {_predicate_constant(family, predicate)},",
                f"        .descriptor_ref = {_descriptor_ref_constant_name(descriptor_key)},",
                f"        .src0_inline_descriptor_ref = {_optional_descriptor_ref_constant_name(f'{descriptor_key}.src0_inline', descriptor_ref_key_set)},",
                f"        .src1_inline_descriptor_ref = {_optional_descriptor_ref_constant_name(f'{descriptor_key}.src1_inline', descriptor_ref_key_set)},",
                "    },",
            ]
        )


def _emit_source(*, public_header: str) -> str:
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.amdgpu.lower.candidates.amdgpu_compare_candidates"),
        "",
        f'#include "{public_header}"',
        "",
        '#include "loom/ops/scalar/ops.h"',
        '#include "loom/ops/vector/ops.h"',
        '#include "loom/target/arch/amdgpu/refs/target_refs.h"',
        "",
        "const loom_amdgpu_compare_descriptor_candidate_t",
        "    kLoomAmdgpuCompareDescriptorCandidates[] = {",
        *_candidate_initializers(_compare_candidates()),
        "};",
        "",
        "const iree_host_size_t kLoomAmdgpuCompareDescriptorCandidateCount =",
        "    IREE_ARRAYSIZE(kLoomAmdgpuCompareDescriptorCandidates);",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU source-to-low compare descriptor candidates.")
    parser.add_argument(
        "--header",
        required=True,
        type=Path,
        help="Generated compare candidate header path.",
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Generated compare candidate source path.",
    )
    parser.add_argument(
        "--public-header",
        default="loom/target/arch/amdgpu/lower/candidates/compare_candidates.h",
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
