# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: Op declarations -> C op tables, accessors, and builders.

Reads Op declarations from the Python DSL and emits C op metadata per dialect:

  ops.h      — enum + ISA macros + accessor macros + builder declarations
  builders.c — builder implementations (macros for common, explicit for complex)
  tables.c   — .rodata: B-string names, format arrays, descriptors, vtables

Public generated headers and compact IR lookup fragments are checked into the
repository for code archaeology and editor/search ergonomics. Bulky generated
C table sources and their private headers are build outputs.

Usage:
    python3 loom/py/loom/gen/run.py c_tables --check
    python3 loom/py/loom/gen/run.py c_tables --in-place
    bazel run //loom/py/loom/gen/ops:c_tables_generator -- --dialect=check --builders=/tmp/builders.c --tables=/tmp/tables.c
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loom.dsl import Op
from loom.gen import bootstrap as _bootstrap
from loom.gen.ops import c_dialect, c_registry
from loom.gen.ops.c_dialect import (
    generate_ops_h,
    generate_sharded_tables_c,
    generate_tables_aggregator_c,
    generate_tables_c,
    generate_tables_h,
)
from loom.gen.ops.c_location_tags import generate_location_tag_table_inc
from loom.gen.ops.c_names import (
    c_dialect_path as _c_dialect_path,
)
from loom.gen.ops.c_scalar_types import generate_scalar_type_table_inc
from loom.gen.ops.keywords import generate_keyword_enum_inc, generate_keyword_table_inc
from loom.gen.ops.model import (
    GenerationModel,
    load_dialect_generation,
    load_generation_model,
)
from loom.gen.ops.type_registry import (
    generate_dialect_type_registry,
    generate_type_registry,
)
from loom.gen.support.files import write_text_file as _write_file
from loom.gen.support.generated_file import (
    GeneratedFileMaintenanceMode,
    GeneratedFileMaintenanceResult,
    GeneratedFileSet,
    maintain_generated_file_set,
)

__all__ = [
    "checked_in_file_set",
    "generate_location_tag_table_inc",
    "generate_ops_h",
    "generate_sharded_tables_c",
    "generate_scalar_type_table_inc",
    "generate_tables_aggregator_c",
    "generate_tables_c",
    "generate_tables_h",
    "generate_dialect_type_registry",
    "generate_type_registry",
    "main",
    "maintain_checked_in_files",
]

DESCRIPTION = "C op table artifacts"
REGENERATE_COMMAND = "python3 loom/py/loom/gen/run.py c_tables --in-place"


# ============================================================================
# CLI
# ============================================================================


@dataclass(frozen=True)
class NamedOutput:
    """One named generated output path from the CLI."""

    name: str
    path: Path


def _parse_named_output(value: str) -> NamedOutput:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return NamedOutput(name=name, path=Path(path))


def _production_dialects(
    model: GenerationModel,
) -> list[tuple[Any, list[Op], Sequence[Any]]]:
    return [
        (
            generation.dialect,
            generation.ops,
            generation.parameterized_attrs,
        )
        for generation in model.dialects
        if generation.dialect.register_by_default
    ]


def _generate_registry_contents(model: GenerationModel) -> dict[str, str]:
    op_reg_h, op_reg_tables_h, op_reg_tables_c = c_registry.generate_op_registry(_production_dialects(model))
    type_reg_h, type_reg_tables_h, type_reg_tables_c = generate_type_registry(model.types)
    return {
        "op_registry.h": op_reg_h,
        "op_registry_tables.h": op_reg_tables_h,
        "op_registry_tables.c": op_reg_tables_c,
        "type_registry.h": type_reg_h,
        "type_registry_tables.h": type_reg_tables_h,
        "type_registry_tables.c": type_reg_tables_c,
        "keyword_enum.inc": generate_keyword_enum_inc(),
        "keyword_table.inc": generate_keyword_table_inc(),
    }


def checked_in_file_set(model: GenerationModel | None = None) -> GeneratedFileSet:
    """Returns the complete checked-in C table artifact ownership set."""
    if model is None:
        model = load_generation_model()

    output_root = Path("loom/src/loom")
    checked_in_outputs: dict[str, str] = {}
    build_output_paths: list[str] = []
    for generation in model.dialects:
        dialect_dir = output_root / _c_dialect_path(generation.dialect)
        for filename, contents in c_dialect.generate_dialect_contents(generation).items():
            path = (dialect_dir / filename).as_posix()
            if generation.dialect.checked_in_headers and filename in (
                "ops.h",
                "types.h",
                "tables.h",
            ):
                checked_in_outputs[path] = contents
            else:
                build_output_paths.append(path)

    registry_dir = output_root / "ops"
    for filename, contents in _generate_registry_contents(model).items():
        path = (registry_dir / filename).as_posix()
        if filename in (
            "op_registry.h",
            "type_registry.h",
            "keyword_enum.inc",
            "keyword_table.inc",
        ):
            checked_in_outputs[path] = contents
        else:
            build_output_paths.append(path)

    checked_in_outputs[(output_root / "ir" / "scalar_type_table.inc").as_posix()] = generate_scalar_type_table_inc()
    checked_in_outputs[(output_root / "ir" / "location_tag_table.inc").as_posix()] = generate_location_tag_table_inc()
    return GeneratedFileSet.from_mapping(
        checked_in_outputs,
        obsolete_paths=build_output_paths,
    )


def maintain_checked_in_files(
    mode: GeneratedFileMaintenanceMode,
) -> GeneratedFileMaintenanceResult:
    """Checks or updates all checked-in C table artifacts."""
    return maintain_generated_file_set(
        _bootstrap.find_repo_root(),
        checked_in_file_set(),
        mode=mode,
        description=DESCRIPTION,
        regenerate_command=REGENERATE_COMMAND,
    )


def _set_output(parser: argparse.ArgumentParser, outputs: dict[str, Path], name: str, path: Path | None) -> None:
    if path is None:
        return
    if name in outputs:
        parser.error(f"duplicate output for {name}")
    outputs[name] = path


def _generate_selected_outputs(
    parser: argparse.ArgumentParser,
    contents: Mapping[str, str],
    outputs: Mapping[str, Path],
) -> int:
    if not outputs:
        parser.error("at least one output path is required")
    unknown_outputs = sorted(name for name in outputs if name not in contents)
    if unknown_outputs:
        parser.error(f"unknown generated output(s): {', '.join(unknown_outputs)}")
    for name, path in outputs.items():
        _write_file(path, contents[name])
    return 0


def _main_build_output_mode(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if args.dialect:
        try:
            generation = load_dialect_generation(args.dialect)
        except ValueError as exc:
            parser.error(str(exc))

        outputs: dict[str, Path] = {}
        _set_output(parser, outputs, "ops.h", args.ops_header)
        _set_output(parser, outputs, "builders.c", args.builders)
        _set_output(parser, outputs, "tables.c", args.tables)
        _set_output(parser, outputs, "tables.h", args.table_header)
        _set_output(parser, outputs, "types.c", args.types)
        for output in args.table_shard:
            _set_output(parser, outputs, f"tables/{output.name}.c", output.path)
        return _generate_selected_outputs(parser, c_dialect.generate_dialect_contents(generation), outputs)

    outputs = {}
    _set_output(parser, outputs, "op_registry.h", args.op_registry_header)
    _set_output(parser, outputs, "op_registry_tables.h", args.op_registry_tables_header)
    _set_output(parser, outputs, "op_registry_tables.c", args.op_registry_tables)
    _set_output(parser, outputs, "type_registry.h", args.type_registry_header)
    _set_output(parser, outputs, "type_registry_tables.h", args.type_registry_tables_header)
    _set_output(parser, outputs, "type_registry_tables.c", args.type_registry_tables)
    _set_output(parser, outputs, "keyword_enum.inc", args.keyword_enum)
    _set_output(parser, outputs, "keyword_table.inc", args.keyword_table)
    model = load_generation_model()
    return _generate_selected_outputs(parser, _generate_registry_contents(model), outputs)


def main(argv: Sequence[str] | None = None) -> int:
    """Generate C tables for Loom dialects and registries."""
    parser = argparse.ArgumentParser(description="Generate Loom C op tables from Python definitions.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify checked-in generated artifacts are current and build outputs are absent.",
    )
    mode.add_argument(
        "--in-place",
        action="store_true",
        help="Regenerate checked-in generated artifacts.",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--dialect", help="Generate selected outputs for one dialect.")
    target.add_argument(
        "--registry",
        action="store_true",
        help="Generate selected cross-dialect registry outputs.",
    )
    parser.add_argument("--ops-header", type=Path, help="Generated dialect ops.h path.")
    parser.add_argument("--builders", type=Path, help="Generated dialect builders.c path.")
    parser.add_argument("--tables", type=Path, help="Generated dialect tables.c path.")
    parser.add_argument("--types", type=Path, help="Generated dialect types.c path.")
    parser.add_argument("--table-header", type=Path, help="Generated sharded-dialect tables.h path.")
    parser.add_argument(
        "--table-shard",
        action="append",
        default=[],
        metavar="NAME=PATH",
        type=_parse_named_output,
        help="Generated sharded-dialect tables/NAME.c path.",
    )
    parser.add_argument("--op-registry-header", type=Path, help="Generated op_registry.h path.")
    parser.add_argument("--op-registry-tables", type=Path, help="Generated op_registry_tables.c path.")
    parser.add_argument("--op-registry-tables-header", type=Path, help="Generated op_registry_tables.h path.")
    parser.add_argument("--type-registry-header", type=Path, help="Generated type_registry.h path.")
    parser.add_argument("--type-registry-tables", type=Path, help="Generated type_registry_tables.c path.")
    parser.add_argument("--type-registry-tables-header", type=Path, help="Generated type_registry_tables.h path.")
    parser.add_argument("--keyword-enum", type=Path, help="Generated keyword_enum.inc path.")
    parser.add_argument("--keyword-table", type=Path, help="Generated keyword_table.inc path.")
    args = parser.parse_args(argv)

    build_output_selected = args.dialect is not None or args.registry
    maintenance_mode_selected = args.check or args.in_place
    if build_output_selected and maintenance_mode_selected:
        parser.error("build-output generation cannot be combined with --check or --in-place")
    if not build_output_selected and not maintenance_mode_selected:
        parser.error("select --check, --in-place, --dialect, or --registry")

    if maintenance_mode_selected:
        result = maintain_checked_in_files("update" if args.in_place else "check")
        return 0 if result.ok else 1
    return _main_build_output_mode(parser, args)


if __name__ == "__main__":
    sys.exit(main())
