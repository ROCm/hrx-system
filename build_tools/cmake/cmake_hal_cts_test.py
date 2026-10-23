# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

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
    def test_embedded_corpus_preserves_alias_registrations(self):
        with tempfile.TemporaryDirectory(prefix="hal cts ") as temporary:
            build = Path(temporary)
            configure_project(FIXTURE, build)
            build_project(build, "iree_fixture_probe")
            probe = (build / f"probe-{CONFIGURATION}.txt").read_text().strip()
            run_command(probe)


if __name__ == "__main__":
    unittest.main()
