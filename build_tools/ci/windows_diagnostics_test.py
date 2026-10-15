# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.ci import windows_diagnostics as diagnostics


class WindowsDiagnosticsTest(unittest.TestCase):
    def test_process_output_and_status_survive_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for code in (0, 7):
                with contextlib.redirect_stdout(io.StringIO()):
                    result = diagnostics.run(
                        [
                            sys.executable,
                            "-c",
                            "import sys; "
                            "sys.stdout.buffer.write(b'out\\xff\\n'); "
                            "sys.stdout.flush(); sys.stderr.buffer.write(b'err\\n'); "
                            f"sys.exit({code})",
                        ],
                        cwd=root,
                        env=None,
                        artifact_dir=root / "artifacts",
                        label="Build repeated phase",
                    )
                self.assertEqual(result, code)
            phases = list((root / "artifacts").iterdir())
            self.assertEqual(len(phases), 2)
            for phase in phases:
                self.assertEqual((phase / "output.log").read_bytes(), b"out\xff\nerr\n")
                command = json.loads((phase / "command.json").read_text())
                self.assertEqual(
                    (phase / "manifest.json").exists(), command["returncode"] != 0
                )

    def test_ninja_inputs_and_response_files_are_copied_before_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build with spaces"
            build.mkdir()
            artifacts = root / "artifacts"
            artifacts.mkdir()
            tool_dir = root / "tools"
            tool_dir.mkdir()
            (tool_dir / "llvm-nm.exe").touch()
            (build / "CMakeCache.txt").write_text(
                "CMAKE_MAKE_PROGRAM:FILEPATH=ninja\n"
                f"CMAKE_C_COMPILER:FILEPATH={tool_dir}/clang-cl.exe\n"
            )
            objects = {
                "backend.obj": b"backend bytes",
                "common.lib": b"archive bytes",
                "common source.obj": b"archive member bytes",
            }
            for name, contents in objects.items():
                (build / name).write_bytes(contents)
            (build / "link.rsp").write_bytes(
                '"common source.obj" common.lib'.encode("utf-16")
            )
            # The graph fits alone but competes with the actual failed-link
            # inputs for the capture budget, as in large generated CMake builds.
            (build / "build.ninja").write_bytes(b"graph" * 20)
            capture_budget = sum(len(contents) for contents in objects.values())
            capture_budget += (build / "CMakeCache.txt").stat().st_size
            capture_budget += (build / "link.rsp").stat().st_size
            (artifacts / "output.log").write_text(
                "FAILED: [code=4294967295] test with spaces.exe test.pdb\n"
            )
            (root / "sdk.lib").write_bytes(b"installed library")

            def native_tool(argv, **kwargs):
                if argv[1:4] == ["-t", "inputs", "--no-shell-escape"]:
                    self.assertEqual(argv[4], "test with spaces.exe")
                    kwargs["stdout"].write(
                        ("\n".join(objects) + "\n../sdk.lib\n").encode()
                    )
                elif argv[1:3] == ["-t", "commands"]:
                    kwargs["stdout"].write(b"lld-link @link.rsp")
                else:
                    self.assertEqual(argv[1], "--print-armap")
                    snapshot = Path(argv[2])
                    self.assertEqual(snapshot.parent, artifacts)
                    self.assertIn(snapshot.read_bytes(), objects.values())
                    kwargs["stdout"].write(b"T defined_symbol\n")
                return subprocess.CompletedProcess(argv, 0)

            # The native graph query and symbol reader are dependencies. The
            # collector, filesystem snapshots, and manifest writer all run.
            with (
                mock.patch.object(subprocess, "run", side_effect=native_tool),
                mock.patch.object(diagnostics, "MAX_CAPTURE_BYTES", capture_budget),
            ):
                diagnostics.capture_failure(artifacts, root, {}, build)
            manifest = json.loads((artifacts / "manifest.json").read_text())
            records = {
                Path(record["path"]).name: record for record in manifest["files"]
            }
            for name, contents in objects.items():
                (build / name).write_bytes(b"rebuild replaced the original")
                self.assertIn("copy", records[name], records[name])
                self.assertEqual(
                    (artifacts / records[name]["copy"]).read_bytes(), contents
                )
                self.assertEqual(
                    records[name]["sha256"], hashlib.sha256(contents).hexdigest()
                )
            self.assertEqual(
                (artifacts / records["link.rsp"]["copy"]).read_bytes(),
                (build / "link.rsp").read_bytes(),
            )
            self.assertNotIn("sdk.lib", records)
            self.assertEqual(
                records["build.ninja"]["error"], "capture byte limit exceeded"
            )
            self.assertEqual(
                sum("--print-armap" in tool["argv"] for tool in manifest["tools"]), 3
            )

    def test_header_access_and_bazel_parameters_are_observed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk = root / "Windows SDK" / "ucrt"
            sdk.mkdir(parents=True)
            (sdk / "wchar.h").write_text("#include <missing.h>\n")
            params = root / "bazel-out/bin/cache.obj.params"
            params.parent.mkdir(parents=True)
            params.write_bytes(b"/I SDK\r\n/c cache.cc\r\n")
            action_error = root / "actions" / "stderr-15695"
            action_error.parent.mkdir()
            action_error.write_bytes(b"compiler diagnostic\r\n")
            artifacts = root / "artifacts"
            artifacts.mkdir()
            (artifacts / "output.log").write_text(
                f"{sdk}/wchar.h(17): fatal error C1083: Cannot open include file: 'missing.h': No such file or directory\n"
                "cl.exe @bazel-out/bin/cache.obj.params\n"
                f"ERROR: Couldn't delete action output directory: {action_error} (Permission denied)\n"
            )
            diagnostics.capture_failure(
                artifacts,
                root,
                {"Include": str(sdk), "A_SECRET": "not captured", "PATH": ""},
                None,
            )
            manifest_text = (artifacts / "manifest.json").read_text()
            manifest = json.loads(manifest_text)
            records = {
                Path(record["path"]).name: record for record in manifest["files"]
            }
            self.assertIn("sha256", records["wchar.h"])
            self.assertNotIn("copy", records["wchar.h"])
            self.assertIn("error", records["missing.h"])
            self.assertEqual(records["missing.h"]["parent_entries"], ["wchar.h"])
            self.assertEqual(
                (artifacts / records[params.name]["copy"]).read_bytes(),
                params.read_bytes(),
            )
            action_error.write_bytes(b"a later action replaced the diagnostic")
            self.assertEqual(
                (artifacts / records[action_error.name]["copy"]).read_bytes(),
                b"compiler diagnostic\r\n",
            )
            self.assertNotIn("A_SECRET", manifest_text)
            self.assertNotIn("not captured", manifest_text)

    def test_unreadable_files_and_limits_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = diagnostics.Capture(root, {})
            unreadable = root / "directory.obj"
            unreadable.mkdir()
            self.assertIn("error", capture.file(unreadable, copy=True))
            self.assertIn("error", capture.file(root / "missing.obj", copy=True))
            oversized = root / "large.obj"
            oversized.write_bytes(b"1234")
            capture.remaining_bytes = 3
            self.assertEqual(
                capture.file(oversized, copy=True)["error"],
                "capture byte limit exceeded",
            )

    def test_native_diagnostic_failure_keeps_output_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = diagnostics.Capture(root, os.environ)
            output = capture.tool(
                [sys.executable, "-c", "import sys; print('tool error'); sys.exit(23)"],
                root,
                "tool",
            )
            self.assertIsNone(output)
            self.assertEqual(capture.tools[0]["returncode"], 23)
            self.assertIn("tool error", (root / capture.tools[0]["output"]).read_text())

    def test_metadata_write_failure_preserves_build_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_write_text = Path.write_text

            def write_text(path, *args, **kwargs):
                if path.name == "command.json":
                    raise PermissionError("worker cannot write command metadata")
                return real_write_text(path, *args, **kwargs)

            with (
                mock.patch.object(Path, "write_text", write_text),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()) as errors,
            ):
                result = diagnostics.run(
                    [sys.executable, "-c", "import sys; sys.exit(29)"],
                    cwd=root,
                    env=None,
                    artifact_dir=root / "artifacts",
                    label="failed",
                )
            self.assertEqual(result, 29)
            self.assertIn("worker cannot write", errors.getvalue())

    def test_log_write_failure_still_drains_the_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_open = Path.open

            def open_file(path, *args, **kwargs):
                stream = real_open(path, *args, **kwargs)
                if path.name == "output.log" and args and args[0] == "wb":
                    stream = mock.Mock(wraps=stream)
                    stream.write.side_effect = OSError("disk full")
                return stream

            with (
                mock.patch.object(Path, "open", open_file),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()) as errors,
            ):
                result = diagnostics.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys; sys.stdout.buffer.write(b'x' * 262144); sys.exit(37)",
                    ],
                    cwd=root,
                    env=None,
                    artifact_dir=root / "artifacts",
                    label="failed",
                )
            self.assertEqual(result, 37)
            command = json.loads(
                next((root / "artifacts").glob("*/command.json")).read_text()
            )
            self.assertEqual(command["log_error"], "disk full")
            self.assertIn("Log write failed", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
