# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates all runtime-side VM specification projections in one action."""

from __future__ import annotations

import argparse
from pathlib import Path

from iree.vm.bytecode.spec.conversion_test_vectors import (
    render_conversion_test_vectors,
)
from iree.vm.bytecode.spec.render.c import (
    render_core_header,
    render_disassembler_data,
    render_instruction_verifier_cases,
    render_interpreter_data,
    render_module_header,
    render_module_verifier_cases,
    render_verifier_data,
    render_wire_assertions,
)
from iree.vm.bytecode.spec.render.fixture import (
    render_core_execution_module_fixture,
    render_launch_config_module_fixture,
    render_structural_module_fixture,
)
from iree.vm.bytecode.spec.render.markdown import render_specification
from iree.vm.bytecode.spec.specification import SPECIFICATION


def _write_output(path: Path, contents: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(contents, bytes):
        if path.is_file() and path.read_bytes() == contents:
            return
        path.write_bytes(contents)
    else:
        if path.is_file() and path.read_text(encoding="utf-8") == contents:
            return
        path.write_text(contents, encoding="utf-8")


def generate_outputs() -> dict[str, str | bytes]:
    """Returns the complete deterministic runtime projection family."""

    return {
        "wire_module_header": render_module_header(SPECIFICATION),
        "wire_core_header": render_core_header(SPECIFICATION),
        "wire_assertions_source": render_wire_assertions(SPECIFICATION),
        "instruction_verifier_cases": render_instruction_verifier_cases(SPECIFICATION),
        "module_verifier_cases": render_module_verifier_cases(SPECIFICATION),
        "verifier_source": render_verifier_data(SPECIFICATION),
        "disassembler_data": render_disassembler_data(SPECIFICATION),
        "interpreter_data": render_interpreter_data(SPECIFICATION),
        "conversion_test_vectors": render_conversion_test_vectors(SPECIFICATION),
        "documentation": render_specification(SPECIFICATION),
        "core_execution_module_fixture": render_core_execution_module_fixture(
            SPECIFICATION
        ),
        "launch_config_module_fixture": render_launch_config_module_fixture(
            SPECIFICATION
        ),
        "structural_module_fixture": render_structural_module_fixture(SPECIFICATION),
    }


def main() -> int:
    outputs = generate_outputs()
    parser = argparse.ArgumentParser(
        description="Generate the runtime VM specification projection family."
    )
    for output_name in outputs:
        parser.add_argument(
            f"--{output_name.replace('_', '-')}", type=Path, required=True
        )
    arguments = parser.parse_args()
    for output_name, contents in outputs.items():
        path = getattr(arguments, output_name)
        _write_output(path, contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
