# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executes the public compiler's Wasm artifact with Node.js."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class WasmArtifactTest(unittest.TestCase):
    def _execute_source(self, source, oracle):
        node = os.environ.get("IREE_WASM_NODE") or shutil.which("node")
        self.assertIsNotNone(node, "Install Node.js or set IREE_WASM_NODE")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "module.wasm"
            subprocess.run(
                [
                    sys.argv[1],
                    source,
                    "--format=wasm-binary",
                    f"--output={output}",
                ],
                check=True,
            )
            subprocess.run([node, str(oracle), str(output)], check=True)

    def test_default_pipeline_executes_structured_source(self):
        for source in sys.argv[2:-2]:
            with self.subTest(source=source):
                self._execute_source(source, Path(source).with_suffix(".mjs"))

    def test_boolean_source_corpus(self):
        source = Path(sys.argv[-2]).read_text()
        source = "wasm.target<simd128> @target\n\n" + source.replace(
            "func.def public @", "func.def public target(@target) @"
        )
        with tempfile.TemporaryDirectory() as directory:
            bound_source = Path(directory) / "boolean.loom"
            bound_source.write_text(source)
            self._execute_source(str(bound_source), sys.argv[-1])


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
