# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.devtools import bazel_launcher


class BazelLauncherTest(unittest.TestCase):
    def create_executable_fixture(self, root: Path) -> Path:
        executable_path = root / "example"
        executable_path.write_text("fixture", encoding="utf-8")
        return executable_path

    def test_configured_environment_replaces_windows_controls_case_insensitively(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            script_path = temporary_path / "launch.bat"
            with mock.patch.object(bazel_launcher.os, "name", "nt"):
                environment = bazel_launcher.configured_environment(
                    {
                        bazel_launcher.CALLER_CWD_ENV.lower(): "stale",
                        "MixedCase": "preserved",
                    },
                    caller_cwd=temporary_path,
                    argument_separator="separator",
                    runfiles_arguments=[],
                    marked_runfiles_arguments=[],
                    runfiles_environment_names=[],
                    script_path=script_path,
                )

        self.assertNotIn(bazel_launcher.CALLER_CWD_ENV.lower(), environment)
        self.assertEqual(
            environment[bazel_launcher.CALLER_CWD_ENV], str(temporary_path)
        )
        self.assertEqual(environment["MixedCase"], "preserved")

    def test_prepare_launch_absolutizes_declared_runfiles_before_chdir(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            runfiles_cwd = temporary_path / "runfiles" / "_main"
            runfiles_cwd.mkdir(parents=True)
            library_path = runfiles_cwd / "runtime" / "libexample.so"
            library_path.parent.mkdir()
            library_path.write_text("fixture", encoding="utf-8")
            executable_path = runfiles_cwd / "runtime" / "example"
            executable_path.write_text("fixture", encoding="utf-8")
            caller_cwd = temporary_path / "caller"
            caller_cwd.mkdir()
            script_path = temporary_path / "launch.sh"
            environment = {
                bazel_launcher.ARGUMENT_SEPARATOR_ENV: "separator",
                bazel_launcher.CALLER_CWD_ENV: str(caller_cwd),
                bazel_launcher.RUNFILES_ARGUMENTS_ENV: json.dumps(
                    {
                        "arguments": ["--library=runtime/libexample.so"],
                        "marked_arguments": [
                            "--library="
                            + bazel_launcher.RUNFILES_PATH_BEGIN
                            + "runtime/libexample.so"
                            + bazel_launcher.RUNFILES_PATH_END
                        ],
                    }
                ),
                bazel_launcher.RUNFILES_ENVIRONMENT_NAMES_ENV: json.dumps(
                    ["IREE_EXAMPLE_LIBRARY"]
                ),
                bazel_launcher.SCRIPT_PATH_ENV: str(script_path),
                "IREE_EXAMPLE_LIBRARY": "runtime/libexample.so",
                "IREE_EXPLICIT_ENV": "preserved",
            }

            launch = bazel_launcher.prepare_launch(
                [
                    "runtime/example",
                    "--library=runtime/libexample.so",
                    "separator",
                    "runtime/libexample.so",
                ],
                environ=environment,
                initial_cwd=runfiles_cwd,
            )

            self.assertEqual(
                launch.argv,
                [
                    str(executable_path),
                    "--library=" + str(library_path),
                    "runtime/libexample.so",
                ],
            )
            self.assertEqual(launch.cwd, caller_cwd)
            self.assertFalse(launch.materialize)
            self.assertEqual(launch.env["IREE_EXAMPLE_LIBRARY"], str(library_path))
            self.assertEqual(launch.env["IREE_EXPLICIT_ENV"], "preserved")
            for name in bazel_launcher.CONTROL_ENVIRONMENT_NAMES:
                self.assertNotIn(name, launch.env)

    def test_resolve_marked_runfiles_arguments_handles_plural_expansion(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            first_path = temporary_path / "one.txt"
            second_path = temporary_path / "two.txt"
            first_path.write_text("one", encoding="utf-8")
            second_path.write_text("two", encoding="utf-8")

            arguments = bazel_launcher.resolve_marked_runfiles_arguments(
                [
                    "--inputs=" + bazel_launcher.RUNFILES_PATH_BEGIN + "one.txt",
                    "two.txt" + bazel_launcher.RUNFILES_PATH_END,
                ],
                bazel_cwd=temporary_path,
            )

            self.assertEqual(
                arguments,
                ["--inputs=" + str(first_path), str(second_path)],
            )

    def test_prepare_launch_rejects_missing_declared_runfile(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            executable_path = self.create_executable_fixture(temporary_path)
            environment = bazel_launcher.configured_environment(
                {"IREE_EXAMPLE_LIBRARY": "missing.so"},
                caller_cwd=temporary_path,
                argument_separator="separator",
                runfiles_arguments=[],
                marked_runfiles_arguments=[],
                runfiles_environment_names=["IREE_EXAMPLE_LIBRARY"],
                script_path=temporary_path / "launch.sh",
            )

            with self.assertRaisesRegex(FileNotFoundError, "IREE_EXAMPLE_LIBRARY"):
                bazel_launcher.prepare_launch(
                    [str(executable_path), "separator"],
                    environ=environment,
                    initial_cwd=temporary_path,
                )

    def test_prepare_launch_uses_host_environment_name_semantics(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            library_path = temporary_path / "libexample.dll"
            library_path.write_text("fixture", encoding="utf-8")
            environment = bazel_launcher.configured_environment(
                {"iree_example_library": str(library_path)},
                caller_cwd=temporary_path,
                argument_separator="separator",
                runfiles_arguments=[],
                marked_runfiles_arguments=[],
                runfiles_environment_names=["IREE_EXAMPLE_LIBRARY"],
                script_path=temporary_path / "launch.bat",
            )

            with mock.patch.object(
                bazel_launcher,
                "environment_name_key",
                side_effect=str.upper,
            ):
                launch = bazel_launcher.prepare_launch(
                    [str(library_path), "separator"],
                    environ=environment,
                    initial_cwd=temporary_path,
                )

            self.assertEqual(launch.env["IREE_EXAMPLE_LIBRARY"], str(library_path))
            self.assertNotIn("iree_example_library", launch.env)

    def test_materialized_process_round_trips_without_private_environment(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            executable_path = self.create_executable_fixture(temporary_path)
            library_path = temporary_path / "libexample.so"
            library_path.write_text("fixture", encoding="utf-8")
            script_path = temporary_path / "launch.bat"
            environment = bazel_launcher.configured_environment(
                {
                    "IREE_EXAMPLE_LIBRARY": str(library_path),
                    bazel_launcher.MATERIALIZE_ENV: "stale",
                },
                caller_cwd=temporary_path,
                argument_separator="separator",
                runfiles_arguments=[],
                marked_runfiles_arguments=[],
                runfiles_environment_names=["IREE_EXAMPLE_LIBRARY"],
                script_path=script_path,
                materialize=True,
            )

            prepared_launch = bazel_launcher.prepare_launch(
                [str(executable_path), "separator", "two words"],
                environ=environment,
                initial_cwd=temporary_path,
            )
            process_launch = bazel_launcher.decode_process_launch(
                bazel_launcher.encode_process_launch(prepared_launch)
            )

            self.assertTrue(prepared_launch.materialize)
            self.assertEqual(process_launch.argv, prepared_launch.argv)
            self.assertEqual(process_launch.cwd, temporary_path)
            self.assertEqual(process_launch.env, prepared_launch.env)
            for name in bazel_launcher.CONTROL_ENVIRONMENT_NAMES:
                self.assertNotIn(name, process_launch.env)

    def test_main_materializes_without_executing_or_deleting_the_script(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            executable_path = self.create_executable_fixture(temporary_path)
            library_path = temporary_path / "libexample.so"
            library_path.write_text("fixture", encoding="utf-8")
            script_path = temporary_path / "launch.bat"
            script_path.write_text("generated", encoding="utf-8")
            environment = bazel_launcher.configured_environment(
                {"IREE_EXAMPLE_LIBRARY": str(library_path)},
                caller_cwd=temporary_path,
                argument_separator="separator",
                runfiles_arguments=[],
                marked_runfiles_arguments=[],
                runfiles_environment_names=["IREE_EXAMPLE_LIBRARY"],
                script_path=script_path,
                materialize=True,
            )
            output = io.StringIO()

            with (
                mock.patch.dict(bazel_launcher.os.environ, environment, clear=True),
                mock.patch.object(bazel_launcher.os, "execvpe") as execvpe,
                contextlib.redirect_stdout(output),
            ):
                result = bazel_launcher.main(
                    [str(executable_path), "separator", "two words"]
                )

            self.assertEqual(result, 0)
            execvpe.assert_not_called()
            process_launch = bazel_launcher.decode_process_launch(output.getvalue())
            self.assertEqual(process_launch.argv, [str(executable_path), "two words"])
            self.assertEqual(process_launch.cwd, temporary_path)
            self.assertEqual(
                process_launch.env["IREE_EXAMPLE_LIBRARY"], str(library_path)
            )
            self.assertTrue(script_path.exists())

    def test_main_removes_the_script_when_handoff_validation_fails(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            executable_path = self.create_executable_fixture(temporary_path)
            script_path = temporary_path / "launch.sh"
            script_path.write_text("generated", encoding="utf-8")
            environment = bazel_launcher.configured_environment(
                {"IREE_EXAMPLE_LIBRARY": "missing.so"},
                caller_cwd=temporary_path,
                argument_separator="separator",
                runfiles_arguments=[],
                marked_runfiles_arguments=[],
                runfiles_environment_names=["IREE_EXAMPLE_LIBRARY"],
                script_path=script_path,
            )

            with (
                mock.patch.dict(bazel_launcher.os.environ, environment, clear=True),
                contextlib.redirect_stderr(io.StringIO()) as error_output,
            ):
                result = bazel_launcher.main([str(executable_path), "separator"])

            self.assertEqual(result, 127)
            self.assertIn("points to missing file", error_output.getvalue())
            self.assertFalse(script_path.exists())

    def test_exec_process_quotes_windows_arguments_for_the_crt(self):
        argv = [r"C:\Program Files\Python\python.exe", "two words", 'a"b']
        environment = {"PATH": r"C:\Windows"}

        with (
            mock.patch.object(bazel_launcher.os, "name", "nt"),
            mock.patch.object(bazel_launcher.os, "execvpe") as execvpe,
        ):
            bazel_launcher.exec_process(argv, environment)

        execvpe.assert_called_once_with(
            argv[0],
            [subprocess.list2cmdline([argument]) for argument in argv],
            environment,
        )

    def test_main_preserves_the_full_process_handoff_contract(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            runfiles_cwd = temporary_path / "runfiles" / "_main"
            runfiles_cwd.mkdir(parents=True)
            library_path = runfiles_cwd / "runtime" / "libexample.so"
            library_path.parent.mkdir()
            library_path.write_text("fixture", encoding="utf-8")
            caller_cwd = temporary_path / "caller"
            caller_cwd.mkdir()
            output_path = caller_cwd / "launch.json"
            script_path = temporary_path / "launch.sh"
            script_path.write_text("generated launch script", encoding="utf-8")
            environment = dict(os.environ)
            environment.update(
                {
                    "IREE_EXAMPLE_LIBRARY": "runtime/libexample.so",
                    "IREE_EXPLICIT_ENV": "preserved",
                }
            )
            environment = bazel_launcher.configured_environment(
                environment,
                caller_cwd=caller_cwd,
                argument_separator="separator",
                runfiles_arguments=[],
                marked_runfiles_arguments=[],
                runfiles_environment_names=["IREE_EXAMPLE_LIBRARY"],
                script_path=script_path,
            )
            child_source = """
import json
import os
import pathlib
import sys

pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "argv": sys.argv[2:],
    "cwd": os.getcwd(),
    "explicit": os.environ["IREE_EXPLICIT_ENV"],
    "library": os.environ["IREE_EXAMPLE_LIBRARY"],
    "pid": os.getpid(),
    "private_environment": sorted(
        name for name in os.environ if name.startswith("IREE_BAZEL_LAUNCH_")
    ),
}))
"""

            process = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(bazel_launcher.__file__).resolve()),
                    sys.executable,
                    "-c",
                    child_source,
                    str(output_path),
                    "separator",
                    "two words",
                ],
                cwd=runfiles_cwd,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = process.communicate()

            self.assertEqual(process.returncode, 0, f"{stdout}\n{stderr}")
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["argv"], ["two words"])
            self.assertEqual(payload["cwd"], str(caller_cwd))
            self.assertEqual(payload["explicit"], "preserved")
            self.assertEqual(payload["library"], str(library_path))
            self.assertEqual(payload["private_environment"], [])
            self.assertFalse(script_path.exists())
            if os.name == "posix":
                self.assertEqual(payload["pid"], process.pid)


if __name__ == "__main__":
    unittest.main()
