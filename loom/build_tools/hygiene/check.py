# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs a read-only source check against a declared source manifest."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

from build_tools.devtools.command_line import batch_path_commands

_SCENARIO_CORPUS_PREFIXES = tuple(
    f"loom/src/loom/test/corpus/{family}/"
    for family in ("control", "function", "kernel", "memory", "numeric", "sanitizer")
)
_SEMANTIC_CORPUS_PREFIX = "loom/src/loom/test/corpus/"
_TOOLING_TARGET_PREFIX = "loom/src/loom/tooling/target/"
_CHECK_RECORD_PATTERN = re.compile(r"^\s*check\.(case|scenario)(?:\s|$)")
_TARGET_DECLARATION_PATTERN = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_.]*\.target<")
_TARGET_SPECIALIZATION_PATTERN = re.compile(r"\btarget\(\s*@")
_TARGET_PATH_COMPONENT_PATTERN = re.compile(
    r"^(amd|amdgpu|arm|cdna[0-9]*|cuda|gfx[0-9][a-z0-9]*|"
    r"aie[0-9][a-z0-9]*|metal|nvidia|rdna[0-9]*|riscv|spirv|vm|"
    r"vulkan|wasm|x86|xdna)(?:[_-]|$)",
    re.IGNORECASE,
)


def check_module_headers(paths: list[str], root: Path) -> bool:
    """Rejects source-style license headers on authored Loom modules."""
    ok = True
    for path in paths:
        if Path(path).suffix != ".loom":
            continue
        source = root / path
        if not source.is_file():
            continue
        with source.open(encoding="utf-8") as source_file:
            for line_number, line in enumerate(source_file, start=1):
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                if not stripped_line.startswith("//"):
                    break
                if stripped_line.startswith(
                    ("// Copyright ", "// SPDX-License-Identifier:")
                ):
                    print(
                        f"{path}:{line_number}: .loom modules must not carry "
                        "copyright or license headers"
                    )
                    ok = False
                    break
    return ok


def check_corpus_scenarios(paths: list[str], root: Path) -> bool:
    """Requires every manifest-owned corpus module to use scenarios."""
    ok = True
    for path in paths:
        if not path.startswith(_SCENARIO_CORPUS_PREFIXES) or not path.endswith(".loom"):
            continue
        source = root / path
        if not source.is_file():
            continue
        has_scenario = False
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            match = _CHECK_RECORD_PATTERN.match(line)
            if not match:
                continue
            if match.group(1) == "scenario":
                has_scenario = True
            else:
                print(
                    f"{path}:{line_number}: manifest-owned corpus modules must "
                    "use check.scenario instead of check.case"
                )
                ok = False
        if not has_scenario:
            print(
                f"{path}:1: manifest-owned corpus modules must define at least "
                "one check.scenario"
            )
            ok = False
    return ok


def check_corpus_target_neutrality(paths: list[str], root: Path) -> bool:
    """Rejects physical target ownership in semantic corpus sources."""
    ok = True
    for path in paths:
        if not path.startswith(_SEMANTIC_CORPUS_PREFIX) or not path.endswith(".loom"):
            continue
        relative_path = Path(path.removeprefix(_SEMANTIC_CORPUS_PREFIX))
        for component in relative_path.parts:
            component_stem = Path(component).stem
            match = _TARGET_PATH_COMPONENT_PATTERN.match(component_stem)
            if match:
                print(
                    f"{path}:1: semantic corpus paths must describe semantics, "
                    f"not target architecture {match.group(1)!r}"
                )
                ok = False
                break
        source = root / path
        if not source.is_file():
            continue
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            code = line.split("//", maxsplit=1)[0]
            if _TARGET_DECLARATION_PATTERN.match(code):
                print(
                    f"{path}:{line_number}: semantic corpus modules cannot "
                    "declare concrete targets; target profiles belong to "
                    "target-owned tests"
                )
                ok = False
            if _TARGET_SPECIALIZATION_PATTERN.search(code):
                print(
                    f"{path}:{line_number}: semantic corpus modules cannot "
                    "specialize operations to concrete target symbols"
                )
                ok = False
    return ok


def check_tooling_target_testdata_ownership(paths: list[str], root: Path) -> bool:
    """Requires target tooling inputs to live in explicit testdata packages."""
    ok = True
    for path in paths:
        if not path.startswith(_TOOLING_TARGET_PREFIX):
            continue
        source_path = PurePosixPath(path)
        if source_path.suffix not in (".loom", ".loom-test"):
            continue
        if not (root / path).is_file():
            continue
        relative_path = PurePosixPath(path.removeprefix(_TOOLING_TARGET_PREFIX))
        if "testdata" in relative_path.parts:
            continue
        print(
            f"{path}:1: target tooling Loom inputs must live in an explicit "
            "testdata package; semantic qualification belongs to target or "
            "corpus packages"
        )
        ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check",
        choices=("format", "authoring", "repository"),
        required=True,
    )
    arguments = parser.parse_args()
    tool = arguments.tool.absolute()
    source_manifest = arguments.sources.absolute()
    output = arguments.output.absolute()
    if arguments.check == "repository":
        commands = [[str(tool), "--manifest", str(source_manifest)]]
    else:
        paths = json.loads(source_manifest.read_text(encoding="utf-8"))
        if arguments.check == "authoring":
            valid_headers = check_module_headers(paths, source_manifest.parent)
            valid_scenarios = check_corpus_scenarios(paths, source_manifest.parent)
            target_neutral = check_corpus_target_neutrality(
                paths, source_manifest.parent
            )
            tooling_inputs_owned = check_tooling_target_testdata_ownership(
                paths, source_manifest.parent
            )
            if (
                not valid_headers
                or not valid_scenarios
                or not target_neutral
                or not tooling_inputs_owned
            ):
                return 1
        flags = {
            "format": ["--check"],
            "authoring": [],
        }[arguments.check]
        commands = batch_path_commands([str(tool), *flags], paths)
    # Keep the portable command bound while passing every source as its own
    # argument. Native flag files cannot carry positional input paths.
    failed = False
    for command in commands:
        failed = (
            subprocess.run(command, cwd=source_manifest.parent).returncode != 0
            or failed
        )
    if failed:
        return 1
    output.write_text("PASS\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
