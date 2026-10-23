# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from build_tools.cmake.test_environment import (
    CONFIGURATION,
    REPO_ROOT,
    build_project,
    configure_project,
    run_command,
)

FIXTURE = REPO_ROOT / "build_tools/cmake/testdata/hal_cts"


class CMakeHalCtsTest(unittest.TestCase):
    def test_compiled_corpus_filters_targets_and_preserves_source_reuse(self):
        for extra_target, target_count in (("OFF", 2), ("ON", 3)):
            with (
                self.subTest(extra_target=extra_target),
                tempfile.TemporaryDirectory(prefix="hal compiled cts ") as temporary,
            ):
                root = Path(temporary)
                source, build = root / "sources", root / "build"
                shutil.copytree(FIXTURE, source)
                for program in ("one.kernel", "two"):
                    (source / f"{program}.loom").write_text(f"{program} content")
                configure_project(
                    source,
                    build,
                    "-DCTS_COMPILED_CORPUS=ON",
                    f"-DCTS_EXTRA_TARGET={extra_target}",
                )
                build_project(build, "iree_hal_drivers_amd_gpu_cts_probe")
                probe = (build / f"probe-{CONFIGURATION}.txt").read_text().strip()
                run_command(probe, str(target_count))
                self.assertEqual(len(list(build.glob("*_source_*.loombc"))), 3)
                self.assertEqual(
                    len(list(build.glob("*_corpus_*/*.bin"))), 3 * target_count
                )
                timestamp = (
                    (build / "primary_corpus_first-exact/one.kernel.bin")
                    .stat()
                    .st_mtime_ns
                )
                build_project(build, "iree_hal_drivers_amd_gpu_cts_probe")
                self.assertEqual(
                    (build / "primary_corpus_first-exact/one.kernel.bin")
                    .stat()
                    .st_mtime_ns,
                    timestamp,
                )

    def test_embedded_corpus_preserves_alias_registrations(self):
        with tempfile.TemporaryDirectory(prefix="hal cts ") as temporary:
            # Match the package build-directory depth in repository CI. The
            # fixture also uses the real qualified target and corpus names, so
            # linking exercises MSVC's relative object-path limit.
            root = Path(temporary)
            build = root / ("b" * max(1, 120 - len(str(root)) - 1))
            configure_project(FIXTURE, build)
            build_project(build, "iree_hal_drivers_amd_gpu_cts_probe")
            probe = (build / f"probe-{CONFIGURATION}.txt").read_text().strip()
            run_command(probe)


if __name__ == "__main__":
    unittest.main()
