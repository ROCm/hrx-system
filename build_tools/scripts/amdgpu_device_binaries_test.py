#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import amdgpu_device_binaries


def write_executable(path: Path, contents: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    path.chmod(0o755)


class AmdgpuDeviceBinariesTest(unittest.TestCase):
    def make_rocm_tree(self, root: Path) -> argparse.Namespace:
        bin_dir = root / "llvm" / "bin"
        resource_dir = root / "llvm" / "lib" / "clang" / "23"
        include_dir = resource_dir / "include"
        include_dir.mkdir(parents=True)
        (include_dir / "stddef.h").write_text("/* fake resource header marker */\n")

        clang_23 = bin_dir / "clang-23"
        write_executable(
            clang_23,
            f"""#!/bin/sh
if [ "$1" = "-print-resource-dir" ]; then
  echo "{resource_dir}"
  exit 0
fi
if [ "$1" = "--print-prog-name=llvm-link" ]; then
  echo "{bin_dir / "llvm-link"}"
  exit 0
fi
if [ "$1" = "--print-prog-name=lld" ]; then
  echo "{bin_dir / "lld"}"
  exit 0
fi
if [ "$1" = "--print-prog-name=ld.lld" ]; then
  echo "{bin_dir / "lld"}"
  exit 0
fi
if [ "$1" = "--print-prog-name=llvm-objcopy" ]; then
  echo "{bin_dir / "llvm-objcopy"}"
  exit 0
fi
exit 0
""",
        )
        write_executable(
            bin_dir / "clang",
            """#!/bin/sh
exec "$(dirname "$0")/clang-23" "$@"
""",
        )
        write_executable(bin_dir / "llvm-link", "#!/bin/sh\nexit 0\n")
        write_executable(bin_dir / "lld", "#!/bin/sh\nexit 0\n")
        write_executable(bin_dir / "llvm-objcopy", "#!/bin/sh\nexit 0\n")

        return argparse.Namespace(
            clang=None,
            clang_resource_include=None,
            lld=None,
            llvm_link=None,
            llvm_objcopy=None,
            rocm_path=[str(root)],
            tool_dir=[],
        )

    def test_select_invocable_clang_prefers_versioned_rocm_driver(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self.make_rocm_tree(Path(temp_dir))
            clang = Path(args.rocm_path[0]) / "llvm" / "bin" / "clang"

            selected = amdgpu_device_binaries.select_invocable_clang(clang)

            self.assertEqual(selected.name, "clang-23")

    def test_select_invocable_clang_keeps_versioned_driver(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self.make_rocm_tree(Path(temp_dir))
            clang = Path(args.rocm_path[0]) / "llvm" / "bin" / "clang-23"

            selected = amdgpu_device_binaries.select_invocable_clang(clang)

            self.assertEqual(selected, clang.resolve())

    def test_detect_clang_resource_include_requires_marker_header(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            include_dir = Path(temp_dir) / "include"
            include_dir.mkdir()

            with self.assertRaisesRegex(RuntimeError, "stddef.h"):
                amdgpu_device_binaries.detect_clang_resource_include(
                    Path("/does/not/matter"),
                    str(include_dir),
                )

    def test_detect_toolchain_selects_versioned_rocm_driver(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self.make_rocm_tree(Path(temp_dir))

            toolchain = amdgpu_device_binaries.detect_toolchain(args)

            self.assertEqual(toolchain.clang.name, "clang-23")
            self.assertEqual(toolchain.llvm_link.name, "llvm-link")
            self.assertEqual(toolchain.lld.name, "lld")
            self.assertEqual(toolchain.llvm_objcopy.name, "llvm-objcopy")
            self.assertTrue((toolchain.clang_resource_include / "stddef.h").is_file())


if __name__ == "__main__":
    unittest.main()
