# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command line entry point for the derived dependency inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from build_tools.bazel_to_cmake.deps import ModuleParser
from build_tools.dependencies.inventory import build_inventory


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Combine Bazel's resolved direct-module graph with canonical "
            "source dependency declarations."
        )
    )
    parser.add_argument(
        "--bazel-deps",
        required=True,
        help="Bazel `mod deps --depth=1 --output=json` output, or - for stdin.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing MODULE.bazel.",
    )
    args = parser.parse_args()

    if args.bazel_deps == "-":
        bazel_graph = json.load(sys.stdin)
    else:
        with Path(args.bazel_deps).open(encoding="utf-8") as file:
            bazel_graph = json.load(file)
    repo_root = args.repo_root.resolve()
    dependencies = ModuleParser(repo_root).parse(repo_root / "MODULE.bazel")
    json.dump(build_inventory(bazel_graph, dependencies), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
