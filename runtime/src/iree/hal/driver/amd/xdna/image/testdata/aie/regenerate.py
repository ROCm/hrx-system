#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Rebuilds the standalone AIE conformance ELFs with a maintainer's aiecc."""

import shutil
import subprocess
import tempfile
from pathlib import Path


def main():
    fixture_dir = Path(__file__).resolve().parent
    variants = {
        "full_elf_42.elf": ("full_elf_42.mlir", []),
        "full_elf_42_expanded.elf": ("full_elf_42.mlir", ["--expand-load-pdis"]),
        "vector_scalar_add.elf": ("vector_scalar_add.mlir", ["--expand-load-pdis"]),
    }
    with tempfile.TemporaryDirectory(prefix="aie-full-elf-") as temporary:
        root = Path(temporary)
        for filename, (source_name, flags) in variants.items():
            build_dir = root / Path(filename).stem
            build_dir.mkdir()
            shutil.copyfile(fixture_dir / source_name, build_dir / "aie.mlir")
            subprocess.run(
                ["aiecc", "--get-full-elf", "--dynamic-objFifos", *flags, "aie.mlir"],
                cwd=build_dir,
                check=True,
            )

        # Publish only after every compiler invocation has succeeded.
        for filename in variants:
            output = root / Path(filename).stem / "aie.elf"
            shutil.copyfile(output, fixture_dir / filename)
            print(f"{filename}: {output.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
