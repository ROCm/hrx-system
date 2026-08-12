# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generator: AMDGPU ISA XML -> global descriptor table family."""

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
from loom.gen.target.arch.amdgpu.refs.amdgpu_target_refs import (  # noqa: E402
    generate_target_ref_outputs,
    select_target_ref_descriptor_set_infos,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=("Generate global AMDGPU descriptor-derived tables from one ISA corpus."))
    parser.add_argument(
        "--isa-xml",
        action="append",
        default=[],
        help="ISA XML fact source as <key>:<path>.",
    )
    parser.add_argument(
        "--descriptor-set",
        action="append",
        default=[],
        help="Descriptor-set key to include in target-reference tables.",
    )
    parser.add_argument("--target-ref-header", type=Path, required=True)
    parser.add_argument("--target-ref-source", type=Path, required=True)
    parser.add_argument(
        "--target-ref-public-header",
        default="loom/target/arch/amdgpu/refs/target_refs.h",
    )
    parser.add_argument("--wait-packet-source", type=Path, required=True)
    parser.add_argument("--vopd-source", type=Path, required=True)
    args = parser.parse_args(argv)

    inputs = load_amdgpu_planning_table_inputs(
        args.isa_xml,
        amdgpu_vopd_instruction_names_by_isa_key(),
    )
    generate_target_ref_outputs(
        public_header=args.target_ref_public_header,
        descriptor_set_infos=select_target_ref_descriptor_set_infos(args.descriptor_set),
        descriptor_sets_by_key=inputs.descriptor_sets_by_key,
        header_path=args.target_ref_header,
        source_path=args.target_ref_source,
    )
    generate_wait_packet_table_outputs(
        inputs,
        source_path=args.wait_packet_source,
    )
    generate_vopd_component_table_outputs(
        inputs,
        source_path=args.vopd_source,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
