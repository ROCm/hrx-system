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


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[5]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen import bootstrap as _bootstrap  # noqa: E402
from loom.gen.support.c import c_string_literal as _c_string_literal  # noqa: E402
from loom.gen.support.files import write_text_file  # noqa: E402
from loom.gen.support.generated_file import (  # noqa: E402
    GeneratedFileMaintenanceMode,
    GeneratedFileMaintenanceResult,
    GeneratedFileSet,
    line_comment_header,
    maintain_generated_file_set,
)
from loom.gen.support.native_layout import (  # noqa: E402
    NativeContractionFactTable,
)
from loom.target.arch.x86.packed_dot_data import (  # noqa: E402
    X86_PACKED_DOT_DESCRIPTORS,
    packed_dot_native_contraction_facts,
    packed_dot_native_layout,
)
from loom.target.arch.x86.target_info import sorted_descriptor_set_infos  # noqa: E402
from loom.target.descriptor_sets import resolve_descriptor_set  # noqa: E402
from loom.target.low_descriptors import descriptor_stable_id  # noqa: E402

DESCRIPTION = "x86 packed-dot contract header"
REGENERATE_COMMAND = "python3 loom/py/loom/gen/run.py x86_packed_dot_contract --in-place"
_CHECKED_IN_HEADER_PATH = "loom/src/loom/target/arch/x86/packed_dot_contract_data.h"


def _join_source(lines: Sequence[str]) -> str:
    return "\n".join(lines)


def _hex_u64_literal(value: int) -> str:
    return f"UINT64_C(0x{value:x})"


def _low_descriptor_ordinal_tables() -> tuple[tuple[int | None, ...] | None, ...]:
    tables: list[tuple[int | None, ...] | None] = []
    for descriptor_set_info in sorted_descriptor_set_infos():
        descriptor_set = resolve_descriptor_set(descriptor_set_info.key)
        descriptor_ordinals = {descriptor.key: ordinal for ordinal, descriptor in enumerate(descriptor_set.descriptors)}
        table = tuple(descriptor_ordinals.get(descriptor.key) for descriptor in X86_PACKED_DOT_DESCRIPTORS)
        tables.append(table if any(value is not None for value in table) else None)
    return tuple(tables)


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
        f"//   {REGENERATE_COMMAND}",
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
        "extern const uint32_t* const",
        "    loom_x86_packed_dot_low_descriptor_ordinal_tables[];",
        "extern const iree_host_size_t",
        "    loom_x86_packed_dot_low_descriptor_ordinal_table_count;",
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
    native_facts = NativeContractionFactTable("kLoomX86PackedDotNativeContractionFacts")
    descriptor_lines: list[str] = []
    for descriptor_ref, descriptor in enumerate(X86_PACKED_DOT_DESCRIPTORS):
        native_shape = packed_dot_native_layout(descriptor).shape
        native_facts_reference = native_facts.reference(packed_dot_native_contraction_facts(descriptor))
        descriptor_lines.extend(
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
                f"            .input_lane_count = {native_shape.block_count * native_shape.k},",
                f"            .result_lane_count = {native_shape.block_count},",
                f"            .reduction_group_size = {native_shape.k},",
                "        },",
                f"        .lhs_numeric_type = {descriptor.lhs_numeric_type},",
                f"        .rhs_numeric_type = {descriptor.rhs_numeric_type},",
                f"        .accumulator_numeric_type = {descriptor.accumulator_numeric_type},",
                f"        .result_numeric_type = {descriptor.result_numeric_type},",
                f"        .native_contraction_facts = {native_facts_reference},",
                f"        .low_descriptor_ref = UINT16_C({descriptor_ref}),",
                "    },",
            ]
        )

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
        *native_facts.definition_lines(),
        "const loom_x86_packed_dot_descriptor_t",
        "    loom_x86_packed_dot_builtin_descriptors[] = {",
        *descriptor_lines,
    ]
    lines.extend(
        [
            "};",
            "",
            "const iree_host_size_t loom_x86_packed_dot_builtin_descriptor_count =",
            f"    {len(X86_PACKED_DOT_DESCRIPTORS)};",
            "",
        ]
    )
    ordinal_tables = _low_descriptor_ordinal_tables()
    for descriptor_set_ordinal, table in enumerate(ordinal_tables):
        if table is None:
            continue
        lines.extend(
            [
                "static const uint32_t",
                f"    kLoomX86PackedDotLowDescriptorOrdinals{descriptor_set_ordinal}[] = {{",
            ]
        )
        lines.extend("    LOOM_LOW_DESCRIPTOR_ORDINAL_NONE," if descriptor_ordinal is None else f"    UINT32_C({descriptor_ordinal})," for descriptor_ordinal in table)
        lines.extend(["};", ""])
    lines.extend(
        [
            "const uint32_t* const",
            "    loom_x86_packed_dot_low_descriptor_ordinal_tables[] = {",
            *("        NULL," if table is None else f"        kLoomX86PackedDotLowDescriptorOrdinals{ordinal}," for ordinal, table in enumerate(ordinal_tables)),
            "};",
            "",
            "const iree_host_size_t",
            "    loom_x86_packed_dot_low_descriptor_ordinal_table_count =",
            f"        {len(ordinal_tables)};",
            "",
        ]
    )
    return _join_source(lines)


def checked_in_file_set() -> GeneratedFileSet:
    """Returns the checked-in packed-dot contract header ownership set."""
    return GeneratedFileSet.from_mapping({_CHECKED_IN_HEADER_PATH: _emit_header()})


def maintain_checked_in_files(
    mode: GeneratedFileMaintenanceMode,
) -> GeneratedFileMaintenanceResult:
    """Checks or updates the checked-in packed-dot contract header."""
    return maintain_generated_file_set(
        _bootstrap.find_repo_root(),
        checked_in_file_set(),
        mode=mode,
        description=DESCRIPTION,
        regenerate_command=REGENERATE_COMMAND,
    )


def write_x86_packed_dot_contract_outputs(
    *,
    header_path: Path | None = None,
    source_path: Path | None = None,
) -> None:
    if header_path is not None:
        write_text_file(header_path, _emit_header())
    if source_path is not None:
        write_text_file(source_path, _emit_source())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate x86 packed-dot contract descriptor data.",
    )
    maintenance_mode = parser.add_mutually_exclusive_group()
    maintenance_mode.add_argument(
        "--check",
        action="store_true",
        help="Check the checked-in generated header.",
    )
    maintenance_mode.add_argument(
        "--in-place",
        action="store_true",
        help="Regenerate the checked-in generated header.",
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

    if args.check or args.in_place:
        if args.header is not None or args.source is not None:
            parser.error("checked-in maintenance modes cannot be combined with explicit outputs")
        result = maintain_checked_in_files("update" if args.in_place else "check")
        return 0 if result.ok else 1
    if args.header is None and args.source is None:
        parser.error("expected --check, --in-place, --header, or --source")
    write_x86_packed_dot_contract_outputs(
        header_path=args.header,
        source_path=args.source,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
