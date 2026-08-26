# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates build-only wire, table, and documentation projections."""

from __future__ import annotations

import argparse
import pathlib
import posixpath
import re
import sys

from execution import EXECUTABLE_INSTRUCTIONS
from model.isa import InstructionFamily
from model.isa.specification import ISA_SPECIFICATION
from model.module.v0 import MODULE_SPECIFICATION
from model.specification import Projection, Specification
from render import (
    isa_family_path,
    render_execution_tables,
    render_isa_assertions,
    render_isa_family_header,
    render_isa_family_markdown,
    render_isa_index_markdown,
    render_isa_opcodes_header,
    render_isa_shared_selectors_header,
    render_module_assertions,
    render_module_header,
    render_module_markdown,
    render_specification_index_markdown,
    render_tooling_isa_tables,
    render_tooling_module_tables,
    shared_selector_table_ids,
)

_ANCHOR_PATTERN = re.compile(r'<a id="([a-z0-9-]+)"></a>')
_LINK_PATTERN = re.compile(r"\]\(([^)#]*)(?:#([a-z0-9-]+))\)")


def _latest_projection(specification: Specification) -> Projection:
    entity_ids = tuple(entity.entity_id for entity in specification.entities)
    return specification.project(specification.derive_projection_versions(entity_ids))


def _family_header_path(family: InstructionFamily) -> str:
    markdown_path = pathlib.PurePosixPath(isa_family_path(family))
    return str(
        pathlib.PurePosixPath("wire")
        / family.since.domain
        / markdown_path.with_suffix(".h").name
    )


def _validate_links(outputs: dict[str, str]) -> None:
    anchors_by_path = {
        path: set(_ANCHOR_PATTERN.findall(contents))
        for path, contents in outputs.items()
        if path.endswith(".md")
    }
    for source_path, contents in outputs.items():
        if not source_path.endswith(".md"):
            continue
        source_parent = pathlib.PurePosixPath(source_path).parent
        for relative_path, anchor in _LINK_PATTERN.findall(contents):
            if not anchor:
                continue
            target_path = source_path
            if relative_path:
                target_path = posixpath.normpath(str(source_parent / relative_path))
            if target_path not in anchors_by_path:
                raise ValueError(
                    f"{source_path}: link targets missing document {target_path}"
                )
            if anchor not in anchors_by_path[target_path]:
                raise ValueError(
                    f"{source_path}: link targets missing anchor {target_path}#{anchor}"
                )


def _projections() -> tuple[Projection, dict[str, Projection]]:
    module_projection = _latest_projection(MODULE_SPECIFICATION)
    isa_projections = {
        "core": ISA_SPECIFICATION.project(
            (ISA_SPECIFICATION.domain_map()["core"].version(),)
        ),
        "hal": _latest_projection(ISA_SPECIFICATION),
    }
    return module_projection, isa_projections


def _instruction_families(
    isa_projection: Projection,
) -> tuple[InstructionFamily, ...]:
    return tuple(
        sorted(
            (
                entity
                for entity in isa_projection.entities
                if isinstance(entity, InstructionFamily)
            ),
            key=lambda value: value.document_order,
        )
    )


def generated_wire_outputs() -> dict[str, str]:
    """Returns the complete deterministic build-only C wire projection."""

    module_projection, isa_projections = _projections()
    outputs = {
        "wire/assertions.c": render_module_assertions(
            module_projection,
            "wire/module_format.h",
        ),
        "wire/module_format.h": render_module_header(
            module_projection,
            "wire/module_format.h",
        ),
    }
    for domain_name in ("core", "hal"):
        isa_projection = isa_projections[domain_name]
        outputs[f"wire/{domain_name}/opcodes.h"] = render_isa_opcodes_header(
            isa_projection,
            domain_name,
            f"wire/{domain_name}/opcodes.h",
        )
        if shared_selector_table_ids(isa_projection, domain_name):
            outputs[f"wire/{domain_name}/selectors.h"] = (
                render_isa_shared_selectors_header(
                    isa_projection,
                    domain_name,
                    f"wire/{domain_name}/selectors.h",
                )
            )

    family_header_paths: dict[str, list[str]] = {"core": [], "hal": []}
    for family in _instruction_families(isa_projections["hal"]):
        isa_projection = isa_projections[family.since.domain]
        header_path = _family_header_path(family)
        family_header_paths[family.since.domain].append(header_path)
        outputs[header_path] = render_isa_family_header(
            isa_projection,
            family.entity_id,
            header_path,
        )
    for domain_name in ("core", "hal"):
        outputs[f"wire/{domain_name}/assertions.c"] = render_isa_assertions(
            isa_projections[domain_name],
            domain_name,
            tuple(family_header_paths[domain_name]),
        )

    outputs = dict(sorted(outputs.items()))
    return outputs


def generated_documentation_outputs() -> dict[str, str]:
    """Returns the complete deterministic build-only Markdown projection."""

    module_projection, isa_projections = _projections()
    outputs = {
        "index.md": render_specification_index_markdown(
            module_projection,
            isa_projections["core"],
            isa_projections["hal"],
        ),
        "module-format.md": render_module_markdown(module_projection),
    }
    for domain_name in ("core", "hal"):
        outputs[f"isa/{domain_name}/index.md"] = render_isa_index_markdown(
            isa_projections[domain_name],
            domain_name,
        )
    for family in _instruction_families(isa_projections["hal"]):
        outputs[isa_family_path(family)] = render_isa_family_markdown(
            isa_projections[family.since.domain],
            family.entity_id,
        )

    outputs = dict(sorted(outputs.items()))
    _validate_links(outputs)
    return outputs


def generated_table_outputs() -> dict[str, str]:
    """Returns all deterministic build-only C table projections."""

    return {
        "execution_tables.inl": render_execution_tables(
            _latest_projection(ISA_SPECIFICATION), EXECUTABLE_INSTRUCTIONS
        ),
        "isa_tables.c.inc": render_tooling_isa_tables(
            _latest_projection(ISA_SPECIFICATION)
        ),
        "module_tables.c.inc": render_tooling_module_tables(
            _latest_projection(MODULE_SPECIFICATION)
        ),
    }


def _existing_documentation_paths(output_directory: pathlib.Path) -> set[str]:
    if not output_directory.is_dir():
        return set()
    return {
        path.relative_to(output_directory).as_posix()
        for path in output_directory.rglob("*")
        if path.is_file()
    }


def _write_or_check(
    output_directory: pathlib.Path,
    outputs: dict[str, str],
    actual_paths: set[str],
    *,
    check_only: bool,
) -> bool:
    """Writes projections or proves the generated-owned paths exactly current."""

    expected_paths = set(outputs)
    if check_only:
        current = True
        for missing in sorted(expected_paths - actual_paths):
            print(
                f"missing generated file: {output_directory / missing}",
                file=sys.stderr,
            )
            current = False
        for extra in sorted(actual_paths - expected_paths):
            print(
                f"unexpected generated file: {output_directory / extra}",
                file=sys.stderr,
            )
            current = False
        for path in sorted(expected_paths & actual_paths):
            if (output_directory / path).read_text(encoding="utf-8") != outputs[path]:
                print(
                    f"stale generated file: {output_directory / path}",
                    file=sys.stderr,
                )
                current = False
        return current

    extra_paths = actual_paths - expected_paths
    if extra_paths:
        raise ValueError(
            "refusing to overwrite generated-owned paths containing unexpected "
            f"files: {sorted(extra_paths)}"
        )
    for path, contents in outputs.items():
        output_path = output_directory / path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(contents, encoding="utf-8")
    return True


def write_or_check_documentation(
    output_directory: pathlib.Path,
    *,
    check_only: bool,
) -> bool:
    """Writes or checks a closed build-only Markdown projection tree."""

    return _write_or_check(
        output_directory,
        generated_documentation_outputs(),
        _existing_documentation_paths(output_directory),
        check_only=check_only,
    )


def _parse_named_output_files(
    parser: argparse.ArgumentParser,
    values: list[str],
    expected_names: set[str],
) -> dict[str, pathlib.Path]:
    output_files: dict[str, pathlib.Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            parser.error("--output-file must have the form NAME=PATH")
        if name in output_files:
            parser.error(f"duplicate --output-file name: {name}")
        output_files[name] = pathlib.Path(path)
    missing_names = expected_names - output_files.keys()
    unknown_names = output_files.keys() - expected_names
    if missing_names:
        parser.error(
            "missing --output-file entries: " + ", ".join(sorted(missing_names))
        )
    if unknown_names:
        parser.error(
            "unknown --output-file entries: " + ", ".join(sorted(unknown_names))
        )
    return output_files


def _write_named_output_files(
    outputs: dict[str, str], output_files: dict[str, pathlib.Path]
) -> None:
    for name, contents in outputs.items():
        output_file = output_files[name]
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if (
            not output_file.is_file()
            or output_file.read_text(encoding="utf-8") != contents
        ):
            output_file.write_text(contents, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate VM bytecode specification projections."
    )
    parser.add_argument(
        "--output-kind",
        choices=("wire", "tables", "documentation"),
        default="wire",
        help="Projection product to generate. Defaults to wire headers.",
    )
    parser.add_argument(
        "--output-directory",
        type=pathlib.Path,
        help=(
            "Projection root for documentation output. Wire and table "
            "outputs are declared individually by the build system."
        ),
    )
    parser.add_argument(
        "--output-file",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help=(
            "Named build-tree output. Wire and table generation require one "
            "entry for every declared projection."
        ),
    )
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.output_kind in ("wire", "tables"):
        if arguments.output_directory is not None:
            parser.error("--output-directory cannot be used for wire or table output")
        if arguments.check:
            parser.error("--check cannot be used for named build outputs")
        outputs = (
            generated_wire_outputs()
            if arguments.output_kind == "wire"
            else generated_table_outputs()
        )
        output_files = _parse_named_output_files(
            parser, arguments.output_file, set(outputs)
        )
        _write_named_output_files(outputs, output_files)
        current = True
    else:
        if arguments.output_file:
            parser.error("--output-file cannot be used for documentation output")
        output_directory = arguments.output_directory
        if output_directory is None:
            parser.error("--output-directory is required for documentation output")
        outputs = generated_documentation_outputs()
        current = _write_or_check(
            output_directory,
            outputs,
            _existing_documentation_paths(output_directory),
            check_only=arguments.check,
        )
    if not current:
        return 1
    action = "checked" if arguments.check else "generated"
    print(f"{action} {len(outputs)} {arguments.output_kind} specification projections")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
