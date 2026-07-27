# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from build_tools.devtools.source_lock import NonEmptyTrackedFileSnapshot


class NonEmptyTrackedFileSnapshotTest(unittest.TestCase):
    def test_non_empty_file_must_remain_non_empty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package_init = root / "pkg" / "__init__.py"
            package_init.parent.mkdir()
            package_init.write_text("# package\n", encoding="utf-8")
            snapshot = NonEmptyTrackedFileSnapshot.capture_paths(
                root,
                ("pkg/__init__.py",),
            )

            package_init.write_text("", encoding="utf-8")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertFalse(snapshot.verify(root))
            self.assertIn("pkg/__init__.py", output.getvalue())
            self.assertIn("became empty", output.getvalue())

    def test_initially_empty_file_is_not_guarded(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package_init = root / "pkg" / "__init__.py"
            package_init.parent.mkdir()
            package_init.write_text("", encoding="utf-8")

            snapshot = NonEmptyTrackedFileSnapshot.capture_paths(
                root,
                ("pkg/__init__.py",),
            )

            self.assertTrue(snapshot.verify(root))


if __name__ == "__main__":
    unittest.main()
