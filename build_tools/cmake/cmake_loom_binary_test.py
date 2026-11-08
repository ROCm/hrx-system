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
            (source / "support.cxx").write_text("support original\n")
            (source / "support.h").write_text("support header original\n")
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
                "--config=a.value=3",
                "--config=z.limit=16",
                "--target=amd.xdna.aie2p:amd.xdna.strix.17f0_10",
                f"--transitive-library={build.as_posix()}/leaf.loombc",
            ):
                self.assertIn(argument, text)
            self.assertNotIn("--product=", text)
            cxx = (build / "cxx_library.loombc").read_text()
            imported_path = (
                build / "cxx_library.import-0-support.cxx.loombc"
            ).as_posix()
            source_path = (source / "support.cxx").as_posix()
            self.assertIn(imported_path, cxx)
            self.assertIn("--options=fp-model=fast", cxx)
            self.assertIn("support original", cxx)
            self.assertNotIn("--input-format=cxx", cxx)
            self.assertLess(cxx.index(imported_path), cxx.index(source_path))
            cxx_timestamp = (build / "cxx_library.loombc").stat().st_mtime_ns
            (source / "support.h").write_text("support header changed\n")
            build_project(build, "kernel_consumer")
            self.assertGreater(
                (build / "cxx_library.loombc").stat().st_mtime_ns,
                cxx_timestamp,
            )
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

            tool_source = source / "tool.c"
            tool_source.write_text(
                tool_source.read_text().replace(
                    "int failed = 0;",
                    'int failed = fputs("changed tool\\n", output) < 0;',
                )
            )
            build_project(build, "kernel_consumer")
            self.assertIn("changed tool\n", output.read_text())
            (build / "leaf.loombc").unlink()
            (build / "explicit.compile.json").unlink()
            build_project(build, "kernel_consumer")
            self.assertIn("leaf changed", output.read_text())
            self.assertTrue((build / "explicit.compile.json").is_file())

            host_tools = root / "host tools"
            host_tools.mkdir()
            suffix = ".exe" if os.name == "nt" else ""
            tools = (build / f"tools-{CONFIGURATION}.txt").read_text().splitlines()
            for name, tool in zip(
                ("loom-import-cxx", "loom-link", "loom-compile"),
                tools,
                strict=True,
            ):
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
