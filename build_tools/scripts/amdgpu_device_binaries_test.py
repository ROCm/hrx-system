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
from typing import Sequence
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import amdgpu_device_binaries


def write_fake_tool(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    path.chmod(0o755)


def fake_clang_query(args: Sequence[str]) -> str:
    if len(args) != 2:
        raise AssertionError(f"unexpected fake clang invocation: {args}")
    clang = Path(args[0])
    query = args[1]
    bin_dir = clang.parent
    rocm_root = bin_dir.parent.parent
    if query == "-print-resource-dir":
        return str(rocm_root / "llvm" / "lib" / "clang" / "23")
    if query.startswith("--print-prog-name="):
        return str(bin_dir / query.removeprefix("--print-prog-name="))
    raise AssertionError(f"unexpected fake clang query: {query}")


class AmdgpuDeviceBinariesTest(unittest.TestCase):
    def make_rocm_tree(self, root: Path) -> argparse.Namespace:
        bin_dir = root / "llvm" / "bin"
        resource_dir = root / "llvm" / "lib" / "clang" / "23"
        include_dir = resource_dir / "include"
        include_dir.mkdir(parents=True)
        (include_dir / "stddef.h").write_text("/* fake resource header marker */\n")

        clang_23 = bin_dir / "clang-23"
        for tool_path in (
            clang_23,
            bin_dir / "clang",
            bin_dir / "llvm-link",
            bin_dir / "lld",
            bin_dir / "llvm-objcopy",
        ):
            write_fake_tool(tool_path)

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

            with mock.patch.object(
                amdgpu_device_binaries,
                "run_capture",
                side_effect=fake_clang_query,
            ):
                selected = amdgpu_device_binaries.select_invocable_clang(clang)

            self.assertEqual(selected.name, "clang-23")

    def test_select_invocable_clang_keeps_versioned_driver(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = self.make_rocm_tree(Path(temp_dir))
            clang = Path(args.rocm_path[0]) / "llvm" / "bin" / "clang-23"

            with mock.patch.object(
                amdgpu_device_binaries,
                "run_capture",
                side_effect=fake_clang_query,
            ):
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

            with mock.patch.object(
                amdgpu_device_binaries,
                "run_capture",
                side_effect=fake_clang_query,
            ):
                toolchain = amdgpu_device_binaries.detect_toolchain(args)

            self.assertEqual(toolchain.clang.name, "clang-23")
            self.assertEqual(toolchain.llvm_link.name, "llvm-link")
            self.assertEqual(toolchain.lld.name, "lld")
            self.assertEqual(toolchain.llvm_objcopy.name, "llvm-objcopy")
            self.assertTrue((toolchain.clang_resource_include / "stddef.h").is_file())

    def test_run_capture_invokes_process_without_a_shell(self):
        self.assertEqual(
            "fake tool output",
            amdgpu_device_binaries.run_capture(
                [sys.executable, "-c", "print('fake tool output')"]
            ),
        )


if __name__ == "__main__":
    unittest.main()
