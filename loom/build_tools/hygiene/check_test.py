# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs action adapters against the production format and policy tools."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from python.runfiles import runfiles


class CheckTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        resolver = runfiles.Create()
        cls.runner, cls.checker, cls.formatter, cls.authoring, cls.repository = (
            resolver.Rlocation(path) for path in sys.argv[1:]
        )

    def test_native_diagnostics_and_success_stamp(self):
        with tempfile.TemporaryDirectory(prefix="loom hygiene ") as temporary:
            root = Path(temporary)
            source_root = root / "loom/source"
            source_root.mkdir(parents=True)
            template = source_root / "template.loom-test"
            consumers = [
                source_root / "first_consumer.loom-test",
                source_root / "second_consumer.loom-test",
            ]
            contents = "func.def @example() {\n  func.return\n}\n"
            template.write_text(contents, encoding="utf-8")
            for consumer in consumers:
                consumer.write_text(
                    "// TEMPLATE: loom/source/template.loom-test\n\n" + contents,
                    encoding="utf-8",
                )
            manifest = root / "sources.json"
            manifest.write_text(
                json.dumps(
                    [str(path.relative_to(root)) for path in (*consumers, template)]
                ),
                encoding="utf-8",
            )
            output = root / "passed"
            command = [
                self.runner,
                "--check=templates",
                "--tool",
                self.checker,
                "--sources",
                str(manifest),
                "--output",
                str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "PASS\n")
            output.unlink()

            # The consumer is unchanged: its source dependency changed.
            template.write_text(
                contents.replace(
                    "  func.return", "  // changed template\n  func.return"
                ),
                encoding="utf-8",
            )
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("is stale relative to", result.stdout + result.stderr)
            self.assertIn("first_consumer.loom-test", result.stdout + result.stderr)
            self.assertFalse(output.exists())

            template.write_text(contents, encoding="utf-8")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output.unlink()
            template.unlink()
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("NOT_FOUND", result.stdout + result.stderr)
            self.assertFalse(output.exists())

    def test_unshared_template_container_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="loom hygiene ") as temporary:
            root = Path(temporary)
            source_root = root / "loom/source"
            source_root.mkdir(parents=True)
            template = source_root / "template.loom-test"
            consumer = source_root / "consumer.loom-test"
            contents = "func.def @example() {\n  func.return\n}\n"
            template.write_text(contents, encoding="utf-8")
            consumer.write_text(
                "// TEMPLATE: loom/source/template.loom-test\n\n" + contents,
                encoding="utf-8",
            )
            manifest = root / "sources.json"
            manifest.write_text(
                json.dumps(
                    [str(path.relative_to(root)) for path in (consumer, template)]
                ),
                encoding="utf-8",
            )
            output = root / "passed"
            command = [
                self.runner,
                "--check=templates",
                "--tool",
                self.checker,
                "--sources",
                str(manifest),
                "--output",
                str(output),
            ]

            result = subprocess.run(command, capture_output=True, text=True)

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("has only one consumer", result.stdout + result.stderr)
            self.assertFalse(output.exists())

    def test_formatter_and_authoring_use_real_tools_and_preserve_inputs(self):
        for check, tool in (("format", self.formatter), ("authoring", self.authoring)):
            with (
                self.subTest(check=check),
                tempfile.TemporaryDirectory(prefix="hygiene ") as temporary,
            ):
                root = Path(temporary)
                source = root / "unregistered source.loom"
                source.write_text(
                    "func.def @example() {\n  func.return\n}\n", encoding="utf-8"
                )
                if check == "format":
                    result = subprocess.run(
                        [tool, "--in-place", str(source)],
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                contents = source.read_text(encoding="utf-8")
                manifest = root / "sources.json"
                manifest.write_text(json.dumps([source.name]), encoding="utf-8")
                output = root / "passed"
                command = [
                    self.runner,
                    "--check=" + check,
                    "--tool",
                    tool,
                    "--sources",
                    str(manifest),
                    "--output",
                    str(output),
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(output.read_text(encoding="utf-8"), "PASS\n")
                output.unlink()
                invalid = (
                    contents.replace("  func.return", "\tfunc.return")
                    if check == "format"
                    else "%one = scalar.constant 1 : i32\n"
                )
                self.assertNotEqual(invalid, contents)
                source.write_text(invalid, encoding="utf-8")
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(source.name, result.stdout + result.stderr)
                self.assertFalse(output.exists())
                self.assertEqual(source.read_text(encoding="utf-8"), invalid)
                source.write_text(contents, encoding="utf-8")
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_repository_policy_preserves_logical_paths_for_encoded_inputs(self):
        cases = (
            (
                "loom/src/loom/ir/unregistered.c",
                "iree_arena_allocate(arena, module->values.count);\n",
                "module-value-cardinality",
            ),
            (
                "loom/src/loom/ir/flags.c",
                '#include "iree/base/tooling/flags.h"\n',
                "flag",
            ),
            (
                "loom/src/loom/target/example/BUILD.bazel",
                'deps = ["//loom/src/loom/tooling/execution/hal:core"]\n',
                "execution mechanism",
            ),
            (
                "loom/src/loom/test/corpus/authoring/example.loom",
                "%nb0 : index\n",
                "byte strides",
            ),
        )
        with tempfile.TemporaryDirectory(prefix="hygiene ") as temporary:
            root = Path(temporary)
            manifest = root / "policy_sources.json"
            output = root / "passed"
            command = [
                self.runner,
                "--check=repository",
                "--tool",
                self.repository,
                "--sources",
                str(manifest),
                "--output",
                str(output),
            ]
            for source, contents, diagnostic in cases:
                with self.subTest(source=source):
                    payload = root / "encoded-input.source"
                    manifest.write_text(
                        json.dumps({source: payload.name}), encoding="utf-8"
                    )
                    payload.write_text(contents, encoding="utf-8")
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(source + ":1:", result.stdout + result.stderr)
                    self.assertIn(diagnostic, result.stdout + result.stderr)
                    self.assertFalse(output.exists())
                    payload.write_text("// policy repair\n", encoding="utf-8")
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertEqual(
                        result.returncode, 0, result.stdout + result.stderr
                    )
                    self.assertEqual(output.read_text(encoding="utf-8"), "PASS\n")
                    output.unlink()
            payload.unlink()
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("FileNotFoundError", result.stdout + result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
