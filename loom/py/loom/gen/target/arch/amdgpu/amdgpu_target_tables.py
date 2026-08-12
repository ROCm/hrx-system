# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU ISA XML -> per-target runtime table sources."""

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

from loom.gen.target.arch.amdgpu.amdgpu_target_table_family import (  # noqa: E402
    amdgpu_target_table_family,
    amdgpu_target_table_instruction_names_by_isa_key,
)
from loom.gen.target.arch.amdgpu.amdgpu_target_table_outputs import (  # noqa: E402
    generate_amdgpu_target_table_outputs,
)
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS,
    build_amdgpu_core_descriptor_sets_from_specs,
)
from loom.target.arch.amdgpu.isa_xml import (  # noqa: E402
    parse_amdgpu_isa_xml_paths_for_instructions,
)


def _parse_isa_xml_paths(values: Sequence[str]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        key, separator, path = value.partition(":")
        if not separator or not key or not path:
            raise ValueError("AMDGPU target table --isa-xml entries must be key:path pairs")
        if key in paths:
            raise ValueError(f"AMDGPU target table ISA XML key '{key}' is duplicate")
        paths[key] = Path(path)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=("Generate descriptor and native encoding sources from one AMDGPU target corpus."))
    parser.add_argument(
        "--target",
        action="append",
        required=True,
        choices=AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS,
        help="AMDGPU storage target family to generate; may be repeated.",
    )
    parser.add_argument(
        "--isa-xml",
        action="append",
        default=[],
        help="ISA XML fact source as <key>:<path>.",
    )
    parser.add_argument(
        "--descriptor-source",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--encoding-source",
        action="append",
        type=Path,
        required=True,
    )
    args = parser.parse_args(argv)

    if not (len(args.target) == len(args.descriptor_source) == len(args.encoding_source)):
        raise ValueError("AMDGPU target table generation requires one descriptor and encoding output per target")

    families = tuple(amdgpu_target_table_family(target) for target in args.target)
    instruction_names_by_isa_key: dict[str, set[str]] = {}
    for family in families:
        for isa_key, instruction_names in amdgpu_target_table_instruction_names_by_isa_key(family).items():
            instruction_names_by_isa_key.setdefault(isa_key, set()).update(instruction_names)
    isa_specs = parse_amdgpu_isa_xml_paths_for_instructions(
        _parse_isa_xml_paths(args.isa_xml),
        instruction_names_by_isa_key,
    )
    descriptor_sets_by_generator_target = build_amdgpu_core_descriptor_sets_from_specs(
        tuple(target for family in families for target in family.generator_targets),
        isa_specs,
    )

    for family, descriptor_source, encoding_source in zip(
        families,
        args.descriptor_source,
        args.encoding_source,
        strict=True,
    ):
        generate_amdgpu_target_table_outputs(
            family,
            isa_specs,
            descriptor_sets_by_generator_target,
            descriptor_source_path=descriptor_source,
            encoding_source_path=encoding_source,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
