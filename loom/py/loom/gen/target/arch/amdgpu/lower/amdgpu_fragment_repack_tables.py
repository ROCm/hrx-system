# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU matrix fragment repack projection tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Hashable, Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[6]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.support.files import write_text_file  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.amdgpu.matrix_fragment_layouts import (  # noqa: E402
    AMDGPU_MATRIX_FRAGMENT_LAYOUTS,
    AmdgpuMatrixFragmentLayout,
    MatrixFragmentResultToLhsBf16Projection,
    MatrixFragmentResultToRhsPackedB16Projection,
    matrix_fragment_result_to_lhs_bf16_projection,
    matrix_fragment_result_to_rhs_packed_b16_projection,
)


def _deduplicate_projections[ProjectionT: Hashable](
    project: Callable[[AmdgpuMatrixFragmentLayout], ProjectionT | None],
) -> tuple[tuple[int, ...], tuple[ProjectionT, ...]]:
    rows: list[ProjectionT] = []
    row_ordinals: dict[ProjectionT, int] = {}
    layout_ordinals: list[int] = []
    for layout in AMDGPU_MATRIX_FRAGMENT_LAYOUTS:
        projection = project(layout)
        if projection is None:
            layout_ordinals.append(0)
            continue
        ordinal = row_ordinals.get(projection)
        if ordinal is None:
            rows.append(projection)
            ordinal = len(rows)
            if ordinal > 0xFF:
                raise ValueError("fragment repack projection ordinal exceeds uint8_t")
            row_ordinals[projection] = ordinal
        layout_ordinals.append(ordinal)
    return tuple(layout_ordinals), tuple(rows)


def _emit_ordinal_table(name: str, layout_ordinals: tuple[int, ...]) -> list[str]:
    lines = [
        f"static const uint8_t {name}[",
        "    LOOM_AMDGPU_MATRIX_FRAGMENT_LAYOUT_COUNT] = {",
    ]
    for layout, ordinal in zip(AMDGPU_MATRIX_FRAGMENT_LAYOUTS, layout_ordinals, strict=True):
        if ordinal:
            lines.append(f"    [{layout.c_kind}] = UINT8_C({ordinal}),")
    lines.extend(["};", ""])
    return lines


def _header() -> list[str]:
    return [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header(
            "//",
            generator=("loom.gen.target.arch.amdgpu.lower.amdgpu_fragment_repack_tables"),
        ),
        "",
    ]


def _emit_result_to_lhs_projections() -> str:
    layout_ordinals, projections = _deduplicate_projections(matrix_fragment_result_to_lhs_bf16_projection)
    lines = _header()
    lines.extend(_emit_ordinal_table("kResultToLhsBf16ProjectionOrdinals", layout_ordinals))
    lines.extend(
        [
            "static const loom_amdgpu_result_to_lhs_bf16_projection_t",
            "    kResultToLhsBf16Projections[] = {",
            "    [0] = {0},",
        ]
    )
    for ordinal, projection in enumerate(projections, start=1):
        if not isinstance(projection, MatrixFragmentResultToLhsBf16Projection):
            raise TypeError("unexpected result-to-LHS projection type")
        lines.extend(
            [
                f"    [{ordinal}] = {{",
                f"        .source_register_selector_and_mask = UINT16_C(0x{projection.source_register_selector.and_mask:x}),",
                f"        .source_lane_group_and_mask = UINT16_C(0x{projection.source_lane_group.and_mask:x}),",
                f"        .source_lane_group_byte_shift = UINT8_C({projection.source_lane_group_byte_shift}),",
                f"        .result_lane_div_byte_shift = UINT8_C({projection.result_lane_div_byte_shift}),",
                f"        .source_register_selector_right_shift = UINT8_C({projection.source_register_selector.right_shift}),",
                f"        .source_lane_group_right_shift = UINT8_C({projection.source_lane_group.right_shift}),",
                f"        .transpose_bit_count = UINT8_C({projection.transpose_bit_count}),",
                "    },",
            ]
        )
    lines.extend(["};", ""])
    return "\n".join(lines)


def _emit_result_to_rhs_projections() -> str:
    layout_ordinals, projections = _deduplicate_projections(matrix_fragment_result_to_rhs_packed_b16_projection)
    lines = _header()
    lines.extend(_emit_ordinal_table("kResultToRhsPackedB16ProjectionOrdinals", layout_ordinals))
    lines.extend(
        [
            "static const loom_amdgpu_result_to_rhs_packed_b16_projection_t",
            "    kResultToRhsPackedB16Projections[] = {",
            "    [0] = {0},",
        ]
    )
    for ordinal, projection in enumerate(projections, start=1):
        if not isinstance(projection, MatrixFragmentResultToRhsPackedB16Projection):
            raise TypeError("unexpected result-to-RHS projection type")
        lines.extend(
            [
                f"    [{ordinal}] = {{",
                f"        .exchange_lane_mask = UINT16_C(0x{projection.exchange_participant_xor_mask:x}),",
                f"        .reverse_lane_mask = UINT32_C(0x{projection.reverse_participant_mask:x}),",
                "    },",
            ]
        )
    lines.extend(["};", ""])
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate AMDGPU matrix fragment repack projections.")
    parser.add_argument(
        "--result-to-lhs",
        required=True,
        type=Path,
        help="Generated result-to-LHS C projection table path.",
    )
    parser.add_argument(
        "--result-to-rhs",
        required=True,
        type=Path,
        help="Generated result-to-RHS C projection table path.",
    )
    args = parser.parse_args(argv)
    write_text_file(args.result_to_lhs, _emit_result_to_lhs_projections())
    write_text_file(args.result_to_rhs, _emit_result_to_rhs_projections())
    return 0


if __name__ == "__main__":
    sys.exit(main())
