# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command line entry point for GitHub dependency graph verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from build_tools.dependencies.github_sbom import verify_inventory


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify an HRX dependency inventory in a GitHub SPDX export."
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--sbom", type=Path, required=True)
    args = parser.parse_args()

    with args.inventory.open(encoding="utf-8") as file:
        inventory = json.load(file)
    with args.sbom.open(encoding="utf-8") as file:
        response = json.load(file)
    verification = verify_inventory(inventory, response)
    print(
        f"verified {verification.dependency_count} submitted dependencies "
        f"across {verification.package_count} exported packages"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
