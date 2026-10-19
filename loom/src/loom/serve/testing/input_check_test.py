# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Authored source assertions run against the actual tokenizer capability."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKER, SOURCE, PROVIDER = [Path(sys.argv.pop(1)).resolve() for _ in range(3)]


class InputCheckTest(unittest.TestCase):
    def execute(self, entry="check", vocabulary=None):
        with tempfile.TemporaryDirectory() as temporary:
            tokenizer = Path(temporary) / "tokenizer.json"
            tokenizer.write_text(
                json.dumps(
                    {
                        "model": {
                            "type": "BPE",
                            "vocab": vocabulary or {"a": 0, "b": 1},
                            "merges": [],
                        }
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.run(
                [
                    str(CHECKER),
                    f"--source={SOURCE}",
                    f"--provider={PROVIDER}",
                    f"--tokenizer={tokenizer}",
                    f"--entry={entry}",
                ],
                capture_output=True,
                encoding="utf-8",
                check=False,
            )

    def test_native_tokens_match_authored_expectation(self):
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_changed_native_tokens_fail_the_source_assertion(self):
        result = self.execute(vocabulary={"b": 0, "a": 1})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("token mismatch", result.stderr)

    def test_source_rejection_preserves_the_diagnostic(self):
        result = self.execute(entry="reject")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("INVALID_ARGUMENT", result.stderr)
        self.assertIn("source rejected input", result.stderr)


if __name__ == "__main__":
    unittest.main()
