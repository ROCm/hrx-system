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

from loom.gen.support.files import write_text_file  # noqa: E402
from loom.gen.target.arch.amdgpu.amdgpu_target_table_family import (  # noqa: E402
    amdgpu_target_table_family,
    amdgpu_target_table_instruction_names_by_isa_key,
)
from loom.gen.target.arch.amdgpu.descriptors.amdgpu_descriptors import (  # noqa: E402
    generate_amdgpu_descriptor_table_family,
)
from loom.gen.target.arch.amdgpu.encoding.amdgpu_encoding_tables import (  # noqa: E402
    generate_amdgpu_encoding_table_source,
)
from loom.target.arch.amdgpu.descriptors import (  # noqa: E402
    AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS,
    build_amdgpu_core_descriptor_sets_from_specs,
)
from loom.target.arch.amdgpu.isa_xml import (  # noqa: E402
    parse_amdgpu_isa_xml_paths_for_instructions,
)

_DESCRIPTOR_SOURCE_HEADER = "loom/codegen/low/descriptors.h"
_ENCODING_SOURCE_HEADER = "loom/target/arch/amdgpu/encoding/encoding.h"


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
        required=True,
        choices=AMDGPU_DESCRIPTOR_SET_GENERATOR_TARGETS,
        help="AMDGPU storage target family to generate.",
    )
    parser.add_argument(
        "--isa-xml",
        action="append",
        default=[],
        help="ISA XML fact source as <key>:<path>.",
    )
    parser.add_argument("--descriptor-source", type=Path, required=True)
    parser.add_argument("--encoding-source", type=Path, required=True)
    args = parser.parse_args(argv)

    family = amdgpu_target_table_family(args.target)
    isa_specs = parse_amdgpu_isa_xml_paths_for_instructions(
        _parse_isa_xml_paths(args.isa_xml),
        amdgpu_target_table_instruction_names_by_isa_key(family),
    )
    descriptor_sets = build_amdgpu_core_descriptor_sets_from_specs(
        family.generator_targets,
        isa_specs,
    )

    descriptor_family = generate_amdgpu_descriptor_table_family(
        family,
        descriptor_sets,
        source_public_header=_DESCRIPTOR_SOURCE_HEADER,
    )
    encoding_source = generate_amdgpu_encoding_table_source(
        family,
        isa_specs,
        descriptor_sets,
        public_header=_ENCODING_SOURCE_HEADER,
    )

    write_text_file(args.descriptor_source, descriptor_family.source)
    write_text_file(args.encoding_source, encoding_source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
