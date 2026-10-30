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
        cls.runner, cls.formatter, cls.authoring, cls.repository = (
            resolver.Rlocation(path) for path in sys.argv[1:]
        )

    def test_authoring_rejects_headers_only_on_loom_modules(self):
        with tempfile.TemporaryDirectory(prefix="loom hygiene ") as temporary:
            root = Path(temporary)
            module = root / "source.loom"
            container = root / "source.loom-test"
            container.write_text(
                "// Copyright 2026 The IREE Authors\nfunc.decl @container()\n",
                encoding="utf-8",
            )
            manifest = root / "sources.json"
            manifest.write_text(
                json.dumps([module.name, container.name]), encoding="utf-8"
            )
            output = root / "passed"
            command = [
                self.runner,
                "--check=authoring",
                "--tool",
                self.authoring,
                "--sources",
                str(manifest),
                "--output",
                str(output),
            ]

            for header in (
                "// Copyright 2026 The IREE Authors\n",
                "// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n",
            ):
                with self.subTest(header=header):
                    module.write_text(
                        header + "func.decl @module()\n", encoding="utf-8"
                    )
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("source.loom:1:", result.stdout + result.stderr)
                    self.assertNotIn("source.loom-test", result.stdout + result.stderr)
                    self.assertFalse(output.exists())

            module.write_text("func.decl @module()\n", encoding="utf-8")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "PASS\n")

    def test_authoring_requires_scenarios_in_manifest_owned_corpus(self):
        with tempfile.TemporaryDirectory(prefix="loom hygiene ") as temporary:
            root = Path(temporary)
            sources = [
                root / "loom/src/loom/test/corpus/numeric/example.loom",
                root / "loom/src/loom/test/corpus/sanitizer/example.loom",
            ]
            for source in sources:
                source.parent.mkdir(parents=True)
            manifest = root / "sources.json"
            manifest.write_text(
                json.dumps([source.relative_to(root).as_posix() for source in sources]),
                encoding="utf-8",
            )
            output = root / "passed"
            command = [
                self.runner,
                "--check=authoring",
                "--tool",
                self.authoring,
                "--sources",
                str(manifest),
                "--output",
                str(output),
            ]

            for source in sources:
                source.write_text(
                    "check.case @example {\n  check.return\n}\n", encoding="utf-8"
                )
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("use check.scenario instead of check.case", result.stdout)
            self.assertFalse(output.exists())

            for source in sources:
                source.write_text("func.decl @example()\n", encoding="utf-8")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must define at least one check.scenario", result.stdout)
            self.assertFalse(output.exists())

            for source in sources:
                source.write_text(
                    "check.scenario @example {\n  check.return\n}\n", encoding="utf-8"
                )
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "PASS\n")

    def test_authoring_keeps_target_ownership_out_of_semantic_corpus(self):
        with tempfile.TemporaryDirectory(prefix="loom hygiene ") as temporary:
            root = Path(temporary)
            source = root / "loom/src/loom/test/corpus/authoring/example.loom"
            source.parent.mkdir(parents=True)
            manifest = root / "sources.json"
            output = root / "passed"
            command = [
                self.runner,
                "--check=authoring",
                "--tool",
                self.authoring,
                "--sources",
                str(manifest),
                "--output",
                str(output),
            ]

            cases = (
                (
                    "loom/src/loom/test/corpus/authoring/gfx1151_example.loom",
                    "func.decl @example()\n",
                    "paths must describe semantics",
                ),
                (
                    source.relative_to(root).as_posix(),
                    "amdgpu.target<gfx1151> @device\n",
                    "cannot declare concrete targets",
                ),
                (
                    source.relative_to(root).as_posix(),
                    "template.def<@example> target(@device) @provider() {\n"
                    "  template.return\n}\n",
                    "cannot specialize operations to concrete target symbols",
                ),
            )
            for path, contents, diagnostic in cases:
                with self.subTest(path=path, diagnostic=diagnostic):
                    case_source = root / path
                    case_source.parent.mkdir(parents=True, exist_ok=True)
                    case_source.write_text(contents, encoding="utf-8")
                    manifest.write_text(json.dumps([path]), encoding="utf-8")
                    result = subprocess.run(command, capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(diagnostic, result.stdout + result.stderr)
                    self.assertFalse(output.exists())
                    case_source.unlink()

            target_owned = (
                root / "loom/src/loom/target/arch/amdgpu/test/gfx1151_example.loom"
            )
            target_owned.parent.mkdir(parents=True)
            target_owned.write_text("func.decl @example()\n", encoding="utf-8")
            manifest.write_text(
                json.dumps([target_owned.relative_to(root).as_posix()]),
                encoding="utf-8",
            )
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "PASS\n")

    def test_authoring_requires_explicit_tooling_target_testdata(self):
        with tempfile.TemporaryDirectory(prefix="loom hygiene ") as temporary:
            root = Path(temporary)
            source = root / "loom/src/loom/tooling/target/amdgpu/test/example.loom"
            source.parent.mkdir(parents=True)
            source.write_text("func.decl @example()\n", encoding="utf-8")
            manifest = root / "sources.json"
            manifest.write_text(
                json.dumps([source.relative_to(root).as_posix()]), encoding="utf-8"
            )
            output = root / "passed"
            command = [
                self.runner,
                "--check=authoring",
                "--tool",
                self.authoring,
                "--sources",
                str(manifest),
                "--output",
                str(output),
            ]

            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must live in an explicit testdata package", result.stdout)
            self.assertFalse(output.exists())

            testdata_source = source.parent.parent / "testdata/example.loom"
            testdata_source.parent.mkdir()
            testdata_source.write_text(source.read_text(encoding="utf-8"))
            manifest.write_text(
                json.dumps([testdata_source.relative_to(root).as_posix()]),
                encoding="utf-8",
            )
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(output.read_text(encoding="utf-8"), "PASS\n")

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
