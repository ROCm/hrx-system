# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator helpers for x86 packed-dot contract data tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[7]
    runtime_py = repo_root / "loom/py"
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)
    return repo_root


REPO_ROOT = _ensure_runtime_py_on_path()

from loom.gen.support.c import c_string_literal as _c_string_literal  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.target.arch.x86.packed_dot_data import X86_PACKED_DOT_DESCRIPTORS  # noqa: E402
from loom.target.low_descriptors import descriptor_stable_id  # noqa: E402


def _join_source(lines: Sequence[str]) -> str:
    return "\n".join(lines)


def _hex_u64_literal(value: int) -> str:
    return f"UINT64_C(0x{value:x})"


def _emit_header() -> str:
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header(
            "//",
            generator="loom.gen.target.arch.x86.x86_packed_dot_contract",
        ),
        "// Regenerate:",
        "//   python3 loom/py/loom/gen/run.py x86_packed_dot_contract --in-place",
        "",
        "#ifndef LOOM_TARGET_ARCH_X86_PACKED_DOT_CONTRACT_DATA_H_",
        "#define LOOM_TARGET_ARCH_X86_PACKED_DOT_CONTRACT_DATA_H_",
        "",
        '#include "loom/target/arch/x86/packed_dot_contract.h"',
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif",
        "",
        "extern const loom_x86_packed_dot_descriptor_t",
        "    loom_x86_packed_dot_builtin_descriptors[];",
        "extern const iree_host_size_t loom_x86_packed_dot_builtin_descriptor_count;",
        "",
        "#ifdef __cplusplus",
        '}  // extern "C"',
        "#endif",
        "",
        "#endif  // LOOM_TARGET_ARCH_X86_PACKED_DOT_CONTRACT_DATA_H_",
        "",
    ]
    return _join_source(lines)


def _emit_source() -> str:
    lines = [
        "// Copyright 2026 The IREE Authors",
        "//",
        "// Licensed under the Apache License v2.0 with LLVM Exceptions.",
        "// See https://llvm.org/LICENSE.txt for license information.",
        "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
        "",
        *line_comment_header("//", generator="loom.gen.target.arch.x86.x86_packed_dot_contract"),
        "",
        '#include "loom/target/arch/x86/packed_dot_contract_data.h"',
        "",
        "const loom_x86_packed_dot_descriptor_t",
        "    loom_x86_packed_dot_builtin_descriptors[] = {",
    ]
    for descriptor in X86_PACKED_DOT_DESCRIPTORS:
        lines.extend(
            [
                "    {",
                f'        .name = IREE_SVL("{_c_string_literal(descriptor.key)}"),',
                f"        .stable_id = {_hex_u64_literal(descriptor_stable_id(descriptor.key))},",
                f'        .llvm_intrinsic_name = IREE_SVL("{_c_string_literal(descriptor.llvm_intrinsic_name)}"),',
                f"        .llvm_source_abi = {descriptor.llvm_source_abi},",
                f'        .instruction_mnemonic = IREE_SVL("{_c_string_literal(descriptor.mnemonic)}"),',
                f"        .family = {descriptor.family},",
                f"        .required_feature_bits = UINT64_C(0x{descriptor.required_feature_bits:x}),",
                f"        .flags = UINT32_C(0x{descriptor.flags:x}),",
                "        .shape = {",
                f"            .vector_bit_width = {descriptor.vector_bit_width},",
                f"            .input_lane_count = {descriptor.input_lane_count},",
                f"            .result_lane_count = {descriptor.result_lane_count},",
                f"            .reduction_group_size = {descriptor.reduction_group_size},",
                "        },",
                f"        .lhs_numeric_type = {descriptor.lhs_numeric_type},",
                f"        .rhs_numeric_type = {descriptor.rhs_numeric_type},",
                f"        .accumulator_numeric_type = {descriptor.accumulator_numeric_type},",
                f"        .result_numeric_type = {descriptor.result_numeric_type},",
                "    },",
            ]
        )
    lines.extend(
        [
            "};",
            "",
            "const iree_host_size_t loom_x86_packed_dot_builtin_descriptor_count =",
            f"    {len(X86_PACKED_DOT_DESCRIPTORS)};",
            "",
        ]
    )
    return _join_source(lines)


def write_x86_packed_dot_contract_header() -> None:
    base_path = REPO_ROOT / "loom/src/loom/target/arch/x86"
    (base_path / "packed_dot_contract_data.h").write_text(
        _emit_header(),
        encoding="utf-8",
    )


def write_x86_packed_dot_contract_outputs(
    *,
    header_path: Path | None = None,
    source_path: Path | None = None,
) -> None:
    if header_path is not None:
        header_path.parent.mkdir(parents=True, exist_ok=True)
        header_path.write_text(_emit_header(), encoding="utf-8")
    if source_path is not None:
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(_emit_source(), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate x86 packed-dot contract descriptor data.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Regenerate checked-in generated headers.",
    )
    parser.add_argument(
        "--header",
        type=Path,
        help="Generated descriptor header path.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Generated descriptor source path.",
    )
    args = parser.parse_args(argv)

    if args.in_place:
        if args.header is not None or args.source is not None:
            parser.error("--in-place cannot be combined with explicit outputs")
        write_x86_packed_dot_contract_header()
        return 0
    if args.header is None and args.source is None:
        parser.error("expected --in-place, --header, or --source")
    write_x86_packed_dot_contract_outputs(
        header_path=args.header,
        source_path=args.source,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
