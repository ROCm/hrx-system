# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Pocket packet assertions with a tiny native Unigram tokenizer."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CHECKER = Path(sys.argv.pop(1)).resolve()
SOURCE = Path(sys.argv.pop(1)).resolve()
PROVIDERS = [Path(value).resolve() for value in sys.argv[1:]]
del sys.argv[1:]


class TextPacketsTest(unittest.TestCase):
    def execute(self, entry):
        # Retain the checkpoint's Prepend/Metaspace boundary shape. The tiny
        # vocabulary has independent ID expectations in text_packets.loom.
        pieces = [
            "<unk>",
            "▁",
            ".",
            "!",
            "?",
            ",",
            ";",
            ":",
            "O",
            "n",
            "e",
            "T",
            "w",
            "o",
            "1",
            "2",
        ]
        metaspace = {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always",
            "split": True,
        }
        with tempfile.TemporaryDirectory() as temporary:
            tokenizer = Path(temporary) / "tokenizer.json"
            tokenizer.write_text(
                json.dumps(
                    {
                        "normalizer": {"type": "Prepend", "prepend": "▁"},
                        "pre_tokenizer": metaspace,
                        "decoder": metaspace,
                        "model": {
                            "type": "Unigram",
                            "unk_id": 0,
                            "vocab": [[piece, -1.0] for piece in pieces],
                            "byte_fallback": False,
                        },
                    }
                ),
                encoding="utf-8",
            )
            return subprocess.run(
                [
                    str(CHECKER),
                    f"--source={SOURCE}",
                    f"--tokenizer={tokenizer}",
                    f"--entry={entry}",
                    *[f"--provider={provider}" for provider in PROVIDERS],
                ],
                capture_output=True,
                encoding="utf-8",
                check=False,
            )

    def test_packets_are_exact_across_input_windows(self):
        result = self.execute("check")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_empty_input_is_rejected(self):
        result = self.execute("reject_empty")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Pocket text must contain speech", result.stderr)

    def test_indivisible_sentence_is_rejected(self):
        result = self.execute("reject_capacity")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Pocket sentence exceeds chunk capacity", result.stderr)


if __name__ == "__main__":
    unittest.main()
