# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU source-to-low compare descriptor candidates."""

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

from loom.dialect.scalar.comparison import CmpFPredicate, CmpIPredicate  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.gen.target.arch.amdgpu.lower.candidates.validation import (  # noqa: E402
    c_identifier,
    optional_descriptor_ref_constant_name,
    required_descriptor_ref_constant_name,
)
from loom.target.arch.amdgpu.descriptors import amdgpu_descriptor_ref_keys  # noqa: E402


@dataclass(frozen=True, slots=True)
class _CompareFamily:
    source_op_name: str
    predicate_c_prefix: str
    predicate_count_constant: str
    candidate_array_name: str
    descriptor_element_type: str
    predicates: tuple[str, ...]


_I32_PREDICATES = tuple(case.keyword for case in CmpIPredicate.cases)
_F32_PREDICATES = tuple(case.keyword for case in CmpFPredicate.cases)

_COMPARE_FAMILIES = (
    _CompareFamily(
        source_op_name="vector.cmpi",
        predicate_c_prefix="LOOM_VECTOR_CMPI_PREDICATE",
        predicate_count_constant="LOOM_VECTOR_CMPI_PREDICATE_COUNT_",
        candidate_array_name="kLoomAmdgpuVectorCmpiCompareDescriptorCandidates",
        descriptor_element_type="i32",
        predicates=_I32_PREDICATES,
    ),
    _CompareFamily(
        source_op_name="scalar.cmpf",
        predicate_c_prefix="LOOM_SCALAR_CMPF_PREDICATE",
        predicate_count_constant="LOOM_SCALAR_CMPF_PREDICATE_COUNT_",
        candidate_array_name="kLoomAmdgpuScalarCmpfCompareDescriptorCandidates",
        descriptor_element_type="f32",
        predicates=_F32_PREDICATES,
    ),
    _CompareFamily(
        source_op_name="vector.cmpf",
        predicate_c_prefix="LOOM_VECTOR_CMPF_PREDICATE",
        predicate_count_constant="LOOM_VECTOR_CMPF_PREDICATE_COUNT_",
        candidate_array_name="kLoomAmdgpuVectorCmpfCompareDescriptorCandidates",
        descriptor_element_type="f32",
        predicates=_F32_PREDICATES,
    ),
)


def _descriptor_element_type(predicate: str, element_type: str) -> str:
    if element_type == "i32" and predicate[0] == "u":
        return "u32"
    return element_type


def _descriptor_key(predicate: str, element_type: str) -> str:
    descriptor_element_type = _descriptor_element_type(predicate, element_type)
    return f"amdgpu.v_cmp_{predicate}_{descriptor_element_type}"


def _predicate_constant(family: _CompareFamily, predicate: str) -> str:
    return f"{family.predicate_c_prefix}_{c_identifier(predicate)}"


def _compare_candidates() -> tuple[tuple[_CompareFamily, str, str], ...]:
    descriptor_ref_key_set = set(amdgpu_descriptor_ref_keys())
    candidates: list[tuple[_CompareFamily, str, str]] = []
    for family in _COMPARE_FAMILIES:
        for predicate in family.predicates:
            descriptor_key = _descriptor_key(predicate, family.descriptor_element_type)
            required_descriptor_ref_constant_name(
                f"AMDGPU compare candidate {family.source_op_name} {predicate}",
                descriptor_key,
                descriptor_ref_key_set,
            )
            candidates.append((family, predicate, descriptor_key))
    return tuple(candidates)


def _candidate_initializer(
    family: _CompareFamily,
    predicate: str,
    descriptor_key: str,
    descriptor_ref_key_set: set[str],
) -> str:
    owner = f"AMDGPU compare candidate {family.source_op_name} {predicate}"
    return "\n".join(
        [
            f"    [{_predicate_constant(family, predicate)}] = {{",
            f"        .descriptor_ref = {required_descriptor_ref_constant_name(owner, descriptor_key, descriptor_ref_key_set)},",
            f"        .src0_inline_descriptor_ref = {optional_descriptor_ref_constant_name(f'{descriptor_key}.src0_inline', descriptor_ref_key_set)},",
            f"        .src1_inline_descriptor_ref = {optional_descriptor_ref_constant_name(f'{descriptor_key}.src1_inline', descriptor_ref_key_set)},",
            "    },",
        ]
    )


def _family_array(
    family: _CompareFamily,
    candidates: Iterable[tuple[_CompareFamily, str, str]],
    descriptor_ref_key_set: set[str],
) -> Iterable[str]:
    candidates = [candidate for candidate in candidates if candidate[0] == family]
    if len(candidates) != len(family.predicates):
        raise ValueError(f"compare family {family.source_op_name} generated {len(candidates)} rows for {len(family.predicates)} predicates")
    yield "const loom_amdgpu_compare_descriptor_candidate_t"
    yield f"    {family.candidate_array_name}[{family.predicate_count_constant}] = {{"
    for family, predicate, descriptor_key in candidates:
        yield _candidate_initializer(family, predicate, descriptor_key, descriptor_ref_key_set)
    yield "};"


def _emit_source(*, public_header: str) -> str:
    candidates = _compare_candidates()
    descriptor_ref_key_set = set(amdgpu_descriptor_ref_keys())
    array_lines: list[str] = []
    for family in _COMPARE_FAMILIES:
        array_lines.extend(_family_array(family, candidates, descriptor_ref_key_set))
        array_lines.append("")

    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header(
            "//",
            generator=("loom.gen.target.arch.amdgpu.lower.candidates.amdgpu_compare_candidates"),
        ),
        "",
        f'#include "{public_header}"',
        "",
        '#include "loom/ops/scalar/ops.h"',
        '#include "loom/ops/vector/ops.h"',
        '#include "loom/target/arch/amdgpu/refs/target_refs.h"',
        "",
        *array_lines,
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU source-to-low compare descriptor candidates.")
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

    args.source.parent.mkdir(parents=True, exist_ok=True)
    args.source.write_text(
        _emit_source(public_header=args.public_header),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
