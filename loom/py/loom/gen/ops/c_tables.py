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

Public generated headers are checked into the repository for code archaeology
and editor/search ergonomics. Bulky generated C table sources are build outputs.

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
from loom.gen.ops.c_names import (
    c_dialect_path as _c_dialect_path,
)
from loom.gen.ops.keywords import generate_keyword_enum_inc, generate_keyword_table_inc
from loom.gen.ops.model import (
    GenerationModel,
    load_dialect_generation,
    load_generation_model,
)
from loom.gen.ops.type_registry import generate_type_registry
from loom.gen.support.files import write_text_file as _write_file

__all__ = [
    "generate_ops_h",
    "generate_sharded_tables_c",
    "generate_tables_aggregator_c",
    "generate_tables_c",
    "generate_tables_h",
    "generate_type_registry",
    "main",
]


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


def _production_dialects(model: GenerationModel) -> list[tuple[Any, list[Op]]]:
    return [(generation.dialect, generation.ops) for generation in model.dialects if generation.dialect.register_by_default]


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


def _checked_in_output_contents(model: GenerationModel) -> dict[Path, str]:
    output_root = _bootstrap.REPO_ROOT / "loom" / "src" / "loom"
    outputs: dict[Path, str] = {}
    for generation in model.dialects:
        dialect_dir = output_root / _c_dialect_path(generation.dialect)
        contents = c_dialect.generate_dialect_contents(generation)
        outputs[dialect_dir / "ops.h"] = contents["ops.h"]
        if "tables.h" in contents:
            outputs[dialect_dir / "tables.h"] = contents["tables.h"]

    registry_contents = _generate_registry_contents(model)
    registry_dir = output_root / "ops"
    for filename in (
        "op_registry.h",
        "type_registry.h",
        "keyword_enum.inc",
        "keyword_table.inc",
    ):
        outputs[registry_dir / filename] = registry_contents[filename]
    return outputs


def _build_generated_output_paths(model: GenerationModel) -> list[Path]:
    output_root = _bootstrap.REPO_ROOT / "loom" / "src" / "loom"
    paths: list[Path] = []
    for generation in model.dialects:
        dialect_dir = output_root / _c_dialect_path(generation.dialect)
        contents = c_dialect.generate_dialect_contents(generation)
        paths.extend(dialect_dir / filename for filename in contents if filename.endswith(".c"))
    registry_dir = output_root / "ops"
    paths.extend(
        [
            registry_dir / "op_registry_tables.c",
            registry_dir / "op_registry_tables.h",
            registry_dir / "type_registry_tables.c",
            registry_dir / "type_registry_tables.h",
        ]
    )
    return sorted(paths)


def _check_checked_in_outputs(model: GenerationModel) -> int:
    failures: list[str] = []
    for path, expected in sorted(_checked_in_output_contents(model).items()):
        if not path.exists():
            failures.append(f"{path.relative_to(_bootstrap.REPO_ROOT)}: missing generated file")
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            failures.append(f"{path.relative_to(_bootstrap.REPO_ROOT)}: stale generated file")

    failures.extend(f"{path.relative_to(_bootstrap.REPO_ROOT)}: generated file must be a build output" for path in _build_generated_output_paths(model) if path.exists())

    if failures:
        print("c table generation check failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        print("regenerate with python3 loom/py/loom/gen/run.py c_tables --in-place", file=sys.stderr)
        return 1

    print(f"checked {len(_checked_in_output_contents(model))} generated C table headers")
    return 0


def _update_checked_in_outputs(model: GenerationModel) -> int:
    for path, content in _checked_in_output_contents(model).items():
        _write_file(path, content)
    print(f"updated {len(_checked_in_output_contents(model))} generated C table headers")
    return 0


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
        help="Verify checked-in generated headers are current and generated C sources are absent.",
    )
    mode.add_argument(
        "--in-place",
        action="store_true",
        help="Regenerate checked-in generated headers.",
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
    header_mode_selected = args.check or args.in_place
    if build_output_selected and header_mode_selected:
        parser.error("build-output generation cannot be combined with --check or --in-place")
    if not build_output_selected and not header_mode_selected:
        parser.error("select --check, --in-place, --dialect, or --registry")

    if args.check:
        return _check_checked_in_outputs(load_generation_model())
    if args.in_place:
        return _update_checked_in_outputs(load_generation_model())
    return _main_build_output_mode(parser, args)


if __name__ == "__main__":
    sys.exit(main())
