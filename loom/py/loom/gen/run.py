# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Root-invocable runner for Loom generators.

Usage:
    python3 loom/py/loom/gen/run.py builders_pyi --in-place
    python3 loom/py/loom/gen/run.py builders_pyi --check
    python3 loom/py/loom/gen/run.py c_errors --check
    python3 loom/py/loom/gen/run.py c_tables --check
    python3 loom/py/loom/gen/run.py c_tables --in-place
    python3 loom/py/loom/gen/run.py low_descriptors
    python3 loom/py/loom/gen/run.py textmate
    python3 loom/py/loom/gen/run.py x86_packed_dot_contract --in-place
    python3 loom/py/loom/gen/run.py x86_target_profiles --check
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import bootstrap  # type: ignore[import-not-found]

GENERATORS = {
    "builders_pyi": "loom.gen.python.builders_pyi",
    "c_errors": "loom.gen.error.c_errors",
    "c_tables": "loom.gen.ops.c_tables",
    "low_descriptors": "loom.gen.target.low.low_descriptors",
    "package_inits": "loom.gen.python.package_inits",
    "textmate": "loom.gen.editor.textmate",
    "x86_packed_dot_contract": "loom.gen.target.arch.x86.x86_packed_dot_contract",
    "x86_target_profiles": "loom.gen.target.arch.x86.x86_target_profiles",
}

ARGUMENT_GENERATORS: set[str] = {
    "builders_pyi",
    "c_errors",
    "c_tables",
    "package_inits",
    "x86_packed_dot_contract",
    "x86_target_profiles",
}


def _usage() -> str:
    names = "|".join(GENERATORS)
    return f"usage: python3 loom/py/loom/gen/run.py <{names}> [generator args...]"


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] not in GENERATORS:
        print(_usage(), file=sys.stderr)
        return 2
    if argv[1] not in ARGUMENT_GENERATORS and len(argv) != 2:
        print(_usage(), file=sys.stderr)
        return 2

    repo_root = bootstrap.ensure_runtime_py_on_path(Path(__file__))
    if Path.cwd().resolve() != repo_root:
        print(f"error: run from repository root: {repo_root}", file=sys.stderr)
        return 2

    module = importlib.import_module(GENERATORS[argv[1]])
    if argv[1] in ARGUMENT_GENERATORS:
        return int(module.main(argv[2:]))
    module.main()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
