# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from build_tools.cmake.test_environment import (
    CONFIGURATION,
    REPO_ROOT,
    build_project,
    configure_project,
)

FIXTURE = REPO_ROOT / "build_tools/cmake/testdata/loom_binary"


class CMakeLoomBinaryTest(unittest.TestCase):
    def test_kernel_stages_keep_transitive_inputs_and_host_tool_ownership(self):
        with tempfile.TemporaryDirectory(prefix="loom products ") as temporary:
            root = Path(temporary)
            source, build = root / "source with spaces", root / "native build"
            shutil.copytree(FIXTURE, source)
            for name in ("direct", "library", "leaf"):
                (source / f"{name}.loom").write_text(f"{name} original\n")
            (source / "direct.h").write_text("header original\n")
            configure_project(source, build)
            build_project(build, "kernel_consumer", "fixture_exported")
            output = build / "explicit.xdna"
            text = output.read_text()
            for argument in (
                "--mode=merge",
                "--strict-deps",
                "--mode=link",
                "--strip-check",
                "--require-resolved-config",
                "--input-format=fixture",
                "--input-options=fixture:include=direct.h",
                "--input-options=fixture:root=direct",
                "--config=a.value=3",
                "--config=z.limit=16",
                "--target=amd.xdna.aie2p:amd.xdna.strix.17f0_10",
                f"--transitive-library={build.as_posix()}/leaf.loombc",
            ):
                self.assertIn(argument, text)
            self.assertNotIn("--product=", text)
            self.assertLess(text.index("--root=@second"), text.index("--root=@first"))
            exported = (build / "exported.xdna").read_text()
            self.assertIn(f"--root-library={build.as_posix()}/library.loombc", exported)
            self.assertNotIn("--root=@", exported)
            timestamp = output.stat().st_mtime_ns
            build_project(build, "kernel_consumer")
            self.assertEqual(output.stat().st_mtime_ns, timestamp)
            (source / "leaf.loom").write_text("leaf changed\n")
            build_project(build, "kernel_consumer")
            self.assertIn("leaf changed", output.read_text())
            self.assertNotIn("leaf original", output.read_text())
            timestamp = output.stat().st_mtime_ns
            (source / "direct.h").write_text("header changed\n")
            build_project(build, "kernel_consumer")
            self.assertGreater(output.stat().st_mtime_ns, timestamp)

            host_tools = root / "host tools"
            host_tools.mkdir()
            suffix = ".exe" if os.name == "nt" else ""
            tools = (build / f"tools-{CONFIGURATION}.txt").read_text().splitlines()
            for name, tool in zip(("loom-link", "loom-compile"), tools, strict=True):
                shutil.copy2(tool, host_tools / (name + suffix))
            cross = root / "cross build"
            configure_project(
                source,
                cross,
                "-DLOOM_TEST_CROSS=ON",
                f"-DIREE_HOST_BIN_DIR={host_tools}",
            )
            build_project(cross, "kernel_consumer")
            self.assertIn("leaf changed", (cross / "explicit.xdna").read_text())

    def test_cross_generation_requires_available_host_tools(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, arguments, expected in (
                ("absent", [], "requires IREE_HOST_BIN_DIR"),
                (
                    "missing",
                    [f"-DIREE_HOST_BIN_DIR={root / 'missing tools'}"],
                    "provide loom-link in IREE_HOST_BIN_DIR",
                ),
            ):
                with self.subTest(name=name):
                    output = configure_project(
                        FIXTURE,
                        root / name,
                        "-DLOOM_TEST_CROSS=ON",
                        *arguments,
                        expect_failure=True,
                    )
                    self.assertIn(expected, " ".join(output.split()))


if __name__ == "__main__":
    unittest.main()
