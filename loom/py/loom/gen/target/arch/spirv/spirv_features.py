# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: SPIR-V feature catalog -> C rodata tables."""

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

from build_tools.spirv.registry.spirv_core_grammar import (  # noqa: E402
    load_spirv_core_grammar,
)
from build_tools.spirv.registry.vulkan_spirv_availability import (  # noqa: E402
    load_vulkan_spirv_availability,
)

from loom.gen.support.c import c_string_view as _c_string_view  # noqa: E402
from loom.gen.support.files import write_text_file as _write_text  # noqa: E402
from loom.gen.support.generated_file import line_comment_header  # noqa: E402
from loom.gen.target.arch.spirv.feature_catalog_validation import (  # noqa: E402
    validate_feature_catalog_sources,
)
from loom.target.arch.spirv.features import (  # noqa: E402
    FEATURE_ATOMS,
    FEATURE_PROFILES,
    feature_atom_enum,
    feature_bits_expression,
    validate_feature_catalog,
)


def _c_identifier_suffix(key: str) -> str:
    return "".join(part.capitalize() for part in key.split("_"))


def _emit_row_array(
    lines: list[str],
    *,
    c_type: str,
    symbol_name: str,
    values: tuple[str, ...],
    string_values: bool = False,
) -> None:
    if not values:
        return
    lines.append(f"static const {c_type} {symbol_name}[] = {{")
    for value in values:
        row = _c_string_view(value) if string_values else value
        lines.append(f"    {row},")
    lines.append("};")
    lines.append("")


def _emit_descriptor_field(
    lines: list[str],
    *,
    field_name: str,
    count_name: str,
    array_symbol: str,
    values: tuple[str, ...],
) -> None:
    if not values:
        return
    lines.append(f"            .{field_name} = {array_symbol},")
    lines.append(f"            .{count_name} = IREE_ARRAYSIZE({array_symbol}),")


def generate_tables() -> str:
    validate_feature_catalog(
        atoms=FEATURE_ATOMS,
        profiles=FEATURE_PROFILES,
    )
    return _emit_tables()


def _emit_tables() -> str:
    lines = [
        *line_comment_header("//", generator="loom.gen.target.arch.spirv.spirv_features"),
        "// clang-format off",
        "",
    ]
    for atom in FEATURE_ATOMS:
        suffix = _c_identifier_suffix(atom.key)
        _emit_row_array(
            lines,
            c_type="iree_string_view_t",
            symbol_name=f"kSpirv{suffix}Extensions",
            values=atom.extensions,
            string_values=True,
        )
        _emit_row_array(
            lines,
            c_type="uint32_t",
            symbol_name=f"kSpirv{suffix}Capabilities",
            values=atom.capabilities,
        )
        _emit_row_array(
            lines,
            c_type="uint32_t",
            symbol_name=f"kSpirv{suffix}Opcodes",
            values=atom.opcodes,
        )
        _emit_row_array(
            lines,
            c_type="uint32_t",
            symbol_name=f"kSpirv{suffix}StorageClasses",
            values=atom.storage_classes,
        )
        _emit_row_array(
            lines,
            c_type="uint32_t",
            symbol_name=f"kSpirv{suffix}Decorations",
            values=atom.decorations,
        )

    lines.extend(
        [
            "static const loom_spirv_feature_atom_descriptor_t kSpirvFeatureAtoms[] = {",
            "    [LOOM_SPIRV_FEATURE_ATOM_UNKNOWN] = {0},",
        ]
    )
    for atom in FEATURE_ATOMS:
        suffix = _c_identifier_suffix(atom.key)
        lines.extend(
            [
                f"    [{feature_atom_enum(atom)}] =",
                "        {",
                f"            .atom = {feature_atom_enum(atom)},",
                f"            .name = {_c_string_view(atom.name)},",
                f"            .required_atom_bits = {feature_bits_expression(atom.required)},",
                f"            .minimum_spirv_version = UINT32_C(0x{atom.minimum_spirv_version:08x}),",
                f"            .addressing_model = {atom.addressing_model},",
                f"            .memory_model = {atom.memory_model},",
            ]
        )
        _emit_descriptor_field(
            lines,
            field_name="extension_names",
            count_name="extension_count",
            array_symbol=f"kSpirv{suffix}Extensions",
            values=atom.extensions,
        )
        _emit_descriptor_field(
            lines,
            field_name="capabilities",
            count_name="capability_count",
            array_symbol=f"kSpirv{suffix}Capabilities",
            values=atom.capabilities,
        )
        _emit_descriptor_field(
            lines,
            field_name="opcodes",
            count_name="opcode_count",
            array_symbol=f"kSpirv{suffix}Opcodes",
            values=atom.opcodes,
        )
        _emit_descriptor_field(
            lines,
            field_name="storage_classes",
            count_name="storage_class_count",
            array_symbol=f"kSpirv{suffix}StorageClasses",
            values=atom.storage_classes,
        )
        _emit_descriptor_field(
            lines,
            field_name="decorations",
            count_name="decoration_count",
            array_symbol=f"kSpirv{suffix}Decorations",
            values=atom.decorations,
        )
        lines.extend(["        },"])
    lines.extend(["};", "", "// clang-format on", ""])
    return "\n".join(lines)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generates SPIR-V feature table snippets.")
    parser.add_argument(
        "--tables",
        type=Path,
        help="Path to write the generated private features_tables.inl include.",
    )
    parser.add_argument(
        "--isa-header",
        type=Path,
        required=True,
        help="Path to isa.h for validating referenced SPIR-V wire symbols.",
    )
    parser.add_argument(
        "--spirv-grammar",
        type=Path,
        required=True,
        help="Path to the pinned SPIR-V core grammar JSON file.",
    )
    parser.add_argument(
        "--vulkan-registry",
        type=Path,
        required=True,
        help="Path to the pinned Vulkan registry XML file.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate generation inputs without writing output files.",
    )
    args = parser.parse_args(argv)
    if args.check and args.tables is not None:
        parser.error("--check cannot be combined with --tables")
    if not args.check and args.tables is None:
        parser.error("--tables is required unless --check is used")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_arguments(argv)
    validate_feature_catalog(atoms=FEATURE_ATOMS, profiles=FEATURE_PROFILES)
    isa_header = args.isa_header.read_text(encoding="utf-8")
    grammar = load_spirv_core_grammar(args.spirv_grammar)
    vulkan_registry = load_vulkan_spirv_availability(args.vulkan_registry)
    validate_feature_catalog_sources(
        atoms=FEATURE_ATOMS,
        isa_header=isa_header,
        isa_source=str(args.isa_header),
        grammar=grammar,
        vulkan_registry=vulkan_registry,
    )
    if args.check:
        return 0
    _write_text(args.tables, _emit_tables())
    return 0


if __name__ == "__main__":
    sys.exit(main())
