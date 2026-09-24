# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executes the public compiler's Wasm artifact with Node.js."""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class WasmArtifactTest(unittest.TestCase):
    def _execute_source(self, source, oracle, compiler_options=()):
        node = os.environ.get("IREE_WASM_NODE") or shutil.which("node")
        self.assertIsNotNone(node, "Install Node.js or set IREE_WASM_NODE")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "module.wasm"
            subprocess.run(
                [
                    _ARGS.compiler,
                    source,
                    *compiler_options,
                    "--format=wasm-binary",
                    f"--output={output}",
                ],
                check=True,
            )
            subprocess.run([node, str(oracle), str(output)], check=True)

    def test_default_pipeline_executes_structured_source(self):
        for source in _ARGS.sources:
            with self.subTest(source=source):
                self._execute_source(source, Path(source).with_suffix(".mjs"))

    def test_source_corpora(self):
        for *source_paths, oracle in _ARGS.corpus:
            with self.subTest(sources=source_paths):
                source = "\n".join(Path(path).read_text() for path in source_paths)
                source = source.replace("func.def @", "func.def public @")
                source = "wasm.target<simd128> @target\n\n" + source.replace(
                    "func.def public @", "func.def public target(@target) @"
                )
                with tempfile.TemporaryDirectory() as directory:
                    bound_source = Path(directory) / "corpus.loom"
                    bound_source.write_text(source)
                    self._execute_source(str(bound_source), oracle)

    def test_cxx_sources(self):
        for source, oracle in _ARGS.cxx:
            with self.subTest(source=source):
                with tempfile.TemporaryDirectory() as directory:
                    imported = Path(directory) / "module.loom"
                    subprocess.run(
                        [
                            _ARGS.cxx_importer,
                            source,
                            "--data-model=ilp32",
                            f"--output={imported}",
                        ],
                        check=True,
                    )
                    self._execute_source(
                        str(imported), oracle, ["--target=wasm:simd128"]
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("compiler")
    parser.add_argument("sources", nargs="*")
    parser.add_argument("--cxx-importer")
    parser.add_argument(
        "--cxx",
        action="append",
        nargs=2,
        default=[],
        metavar=("SOURCE", "ORACLE"),
        help="An ILP32 C/C++ source file and its JavaScript oracle.",
    )
    parser.add_argument(
        "--corpus",
        action="append",
        nargs="+",
        default=[],
        metavar="SOURCE_OR_ORACLE",
        help="One or more source files followed by the JavaScript oracle.",
    )
    _ARGS = parser.parse_args()
    if any(len(corpus) < 2 for corpus in _ARGS.corpus):
        parser.error("--corpus requires source files followed by an oracle")
    if _ARGS.cxx and not _ARGS.cxx_importer:
        parser.error("--cxx requires --cxx-importer")
    unittest.main(argv=[sys.argv[0]])
