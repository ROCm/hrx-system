# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU ISA XML -> descriptor-derived planning table family."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path


def _ensure_runtime_py_on_path() -> None:
    runtime_py = Path(__file__).resolve().parents[6]
    runtime_py_string = str(runtime_py)
    if runtime_py_string not in sys.path:
        sys.path.insert(0, runtime_py_string)


_ensure_runtime_py_on_path()

from loom.gen.target.arch.amdgpu.descriptors.amdgpu_planning_table_inputs import (  # noqa: E402
    load_amdgpu_planning_table_inputs,
)
from loom.gen.target.arch.amdgpu.descriptors.amdgpu_vopd_component_tables import (  # noqa: E402
    amdgpu_vopd_instruction_names_by_isa_key,
    generate_vopd_component_table_outputs,
)
from loom.gen.target.arch.amdgpu.descriptors.amdgpu_wait_packet_tables import (  # noqa: E402
    generate_wait_packet_table_outputs,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=("Generate AMDGPU descriptor-derived planning tables from one ISA corpus."))
    parser.add_argument(
        "--isa-xml",
        action="append",
        default=[],
        help="ISA XML fact source as <key>:<path>.",
    )
    parser.add_argument("--wait-descriptor-rows", type=Path, required=True)
    parser.add_argument("--wait-immediate-rows", type=Path, required=True)
    parser.add_argument("--wait-descriptor-ranges", type=Path, required=True)
    parser.add_argument("--wait-descriptor-lookups", type=Path, required=True)
    parser.add_argument("--wait-selection-rows", type=Path, required=True)
    parser.add_argument("--vopd-component-rules", type=Path, required=True)
    parser.add_argument(
        "--vopd-descriptor-lookup-ranges",
        type=Path,
        required=True,
    )
    parser.add_argument("--vopd-descriptor-lookups", type=Path, required=True)
    parser.add_argument("--vopd-pair-affinity-ranges", type=Path, required=True)
    parser.add_argument("--vopd-pair-affinities", type=Path, required=True)
    parser.add_argument(
        "--vopd-pair-placement-recipes",
        type=Path,
        required=True,
    )
    args = parser.parse_args(argv)

    inputs = load_amdgpu_planning_table_inputs(
        args.isa_xml,
        amdgpu_vopd_instruction_names_by_isa_key(),
    )
    generate_wait_packet_table_outputs(
        inputs,
        descriptor_rows_path=args.wait_descriptor_rows,
        immediate_rows_path=args.wait_immediate_rows,
        descriptor_ranges_path=args.wait_descriptor_ranges,
        descriptor_lookups_path=args.wait_descriptor_lookups,
        selection_rows_path=args.wait_selection_rows,
    )
    generate_vopd_component_table_outputs(
        inputs,
        component_rules_path=args.vopd_component_rules,
        descriptor_lookup_ranges_path=args.vopd_descriptor_lookup_ranges,
        descriptor_lookups_path=args.vopd_descriptor_lookups,
        pair_affinity_ranges_path=args.vopd_pair_affinity_ranges,
        pair_affinities_path=args.vopd_pair_affinities,
        pair_placement_recipes_path=args.vopd_pair_placement_recipes,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
