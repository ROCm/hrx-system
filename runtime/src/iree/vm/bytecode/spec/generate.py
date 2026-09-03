# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generates all runtime-side VM specification projections in one action."""

from __future__ import annotations

import argparse
from pathlib import Path

from iree.vm.bytecode.spec.render.c import (
    render_core_header,
    render_module_header,
    render_tooling_data,
    render_verification_data,
    render_wire_assertions,
)
from iree.vm.bytecode.spec.render.markdown import render_specification
from iree.vm.bytecode.spec.specification import SPECIFICATION


def _write_output(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_text(encoding="utf-8") == contents:
        return
    path.write_text(contents, encoding="utf-8")


def generate_outputs() -> dict[str, str]:
    """Returns the complete deterministic runtime projection family."""

    return {
        "wire_module_header": render_module_header(SPECIFICATION),
        "wire_core_header": render_core_header(SPECIFICATION),
        "wire_assertions_source": render_wire_assertions(SPECIFICATION),
        "verification_source": render_verification_data(SPECIFICATION),
        "tooling_data": render_tooling_data(SPECIFICATION),
        "documentation": render_specification(SPECIFICATION),
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
