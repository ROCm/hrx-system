# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.devtools import bazel as bazel_dev


class BazelTest(unittest.TestCase):
    def test_bazel_output_path_uses_configured_executable(self):
        executable_path = Path("C:/b/execroot/bazel-out/bin/pkg/tool.exe")
        metadata = bazel_dev.BazelLaunchMetadata(executable_path=executable_path)
        with mock.patch.object(
            bazel_dev,
            "resolve_bazel_launch_metadata",
            return_value=(0, metadata),
        ) as resolve_metadata:
            result = bazel_dev.resolve_bazel_output_path(
                bazel="bazel",
                target="//pkg:tool",
                bazel_args=["--config=asan"],
                cwd=bazel_dev.REPO_ROOT,
                env=None,
            )

        self.assertEqual(result, executable_path)
        resolve_metadata.assert_called_once_with(
            bazel="bazel",
            target="//pkg:tool",
            bazel_args=["--config=asan"],
            cwd=bazel_dev.REPO_ROOT,
            env=None,
        )

    def test_bazel_execution_root_uses_configured_arguments(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="C:/b/execroot\n",
            stderr="",
        )
        with mock.patch.object(
            bazel_dev,
            "run_captured",
            return_value=completed,
        ) as run_captured:
            result = bazel_dev.bazel_execution_root(
                bazel="bazel",
                bazel_args=["--config=asan", "--//example:mode=full"],
                cwd=bazel_dev.REPO_ROOT,
                env=None,
            )

        self.assertEqual(result, Path("C:/b/execroot"))
        self.assertEqual(
            run_captured.call_args.args[0],
            [
                "bazel",
                "info",
                "--config=asan",
                "--//example:mode=full",
                "execution_root",
            ],
        )

    def test_bazel_launch_metadata_queries_graph_providers(self):
        execution_root = bazel_dev.REPO_ROOT / ".tmp/test-execution-root"
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "label": "@@//pkg:configured_tool",
                    "executable": "bazel-out/bin/pkg/configured_tool.exe",
                    "argument_provider_count": 1,
                    "arguments": ["", "--input=path/to/input"],
                    "environment_names": ["IREE_EXAMPLE_LIBRARY"],
                    "inherited_environment_names": ["HOME"],
                    "marked_arguments": [
                        "",
                        "--input="
                        + bazel_dev.bazel_launcher.RUNFILES_PATH_BEGIN
                        + "path/to/input"
                        + bazel_dev.bazel_launcher.RUNFILES_PATH_END,
                    ],
                    "run_environment_names": ["IREE_EXPLICIT_ENV"],
                }
            )
            + "\n",
            stderr="",
        )
        with (
            mock.patch.object(
                bazel_dev,
                "run_captured",
                return_value=completed,
            ) as run_captured,
            mock.patch.object(
                bazel_dev,
                "bazel_execution_root",
                return_value=execution_root,
            ) as query_execution_root,
        ):
            result, metadata = bazel_dev.resolve_bazel_launch_metadata(
                bazel="bazel",
                target="//pkg:tool",
                bazel_args=["--config=asan"],
                cwd=bazel_dev.REPO_ROOT,
                env=None,
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            metadata,
            bazel_dev.BazelLaunchMetadata(
                executable_path=(
                    execution_root / "bazel-out/bin/pkg/configured_tool.exe"
                ),
                runfiles_arguments=["", "--input=path/to/input"],
                marked_runfiles_arguments=[
                    "",
                    "--input="
                    + bazel_dev.bazel_launcher.RUNFILES_PATH_BEGIN
                    + "path/to/input"
                    + bazel_dev.bazel_launcher.RUNFILES_PATH_END,
                ],
                runfiles_environment_names=["IREE_EXAMPLE_LIBRARY"],
            ),
        )
        query_argv = run_captured.call_args.args[0]
        self.assertEqual(query_argv[0:2], ["bazel", "cquery"])
        self.assertIn("--output=starlark", query_argv)
        self.assertIn("%IreeRunfilesArgumentsInfo", " ".join(query_argv))
        self.assertIn("%IreeRunfilesEnvironmentInfo", " ".join(query_argv))
        self.assertIn("DefaultInfo", " ".join(query_argv))
        self.assertEqual(query_argv[-2:], ["--config=asan", "//pkg:tool"])
        query_execution_root.assert_called_once_with(
            bazel="bazel",
            bazel_args=["--config=asan"],
            cwd=bazel_dev.REPO_ROOT,
            env=None,
        )

    def test_bazel_launch_metadata_batches_configured_targets(self):
        execution_root = bazel_dev.REPO_ROOT / ".tmp/test-execution-root"
        payloads = [
            {
                "label": label,
                "executable": f"bazel-out/bin/{label.rsplit(':', 1)[1]}.exe",
                "argument_provider_count": 0,
                "arguments": [],
                "environment_names": [],
                "inherited_environment_names": [],
                "marked_arguments": [],
                "run_environment_names": [],
            }
            for label in ("@@//z:z_fuzz", "@@//a:a_fuzz")
        ]
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="\n".join(json.dumps(payload) for payload in payloads) + "\n",
            stderr="",
        )
        targets = ["//a:a_fuzz", "//z:z_fuzz"]
        with (
            mock.patch.object(
                bazel_dev,
                "run_captured",
                return_value=completed,
            ) as run_captured,
            mock.patch.object(
                bazel_dev,
                "bazel_execution_root",
                return_value=execution_root,
            ) as query_execution_root,
        ):
            result, metadata_by_target = (
                bazel_dev.resolve_bazel_launch_metadata_for_targets(
                    bazel="bazel",
                    targets=targets,
                    bazel_args=["--config=fuzzer"],
                    cwd=bazel_dev.REPO_ROOT,
                    env=None,
                )
            )

        self.assertEqual(result, 0)
        self.assertEqual(
            metadata_by_target,
            {
                target: bazel_dev.BazelLaunchMetadata(
                    executable_path=(
                        execution_root / f"bazel-out/bin/{target.rsplit(':', 1)[1]}.exe"
                    )
                )
                for target in targets
            },
        )
        self.assertEqual(
            run_captured.call_args.args[0][-2:],
            ["--config=fuzzer", "set(//a:a_fuzz //z:z_fuzz)"],
        )
        query_execution_root.assert_called_once_with(
            bazel="bazel",
            bazel_args=["--config=fuzzer"],
            cwd=bazel_dev.REPO_ROOT,
            env=None,
        )

    def test_bazel_launch_metadata_rejects_control_environment_collision(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(
                {
                    "label": "@@//pkg:tool",
                    "executable": "bazel-out/bin/pkg/tool.exe",
                    "argument_provider_count": 0,
                    "arguments": [],
                    "environment_names": [],
                    "inherited_environment_names": [],
                    "marked_arguments": [],
                    "run_environment_names": [bazel_dev.bazel_launcher.CALLER_CWD_ENV],
                }
            )
            + "\n",
            stderr="",
        )
        with (
            mock.patch.object(
                bazel_dev,
                "run_captured",
                return_value=completed,
            ),
            mock.patch.object(
                bazel_dev,
                "bazel_execution_root",
                return_value=bazel_dev.REPO_ROOT / ".tmp/test-execution-root",
            ),
            contextlib.redirect_stderr(io.StringIO()) as error_output,
        ):
            result, metadata = bazel_dev.resolve_bazel_launch_metadata(
                bazel="bazel",
                target="//pkg:tool",
                bazel_args=[],
                cwd=bazel_dev.REPO_ROOT,
                env=None,
            )

        self.assertEqual(result, 1)
        self.assertIsNone(metadata)
        self.assertIn("reserved Bazel launcher environment", error_output.getvalue())

    def test_generate_bazel_launch_uses_script_path_without_running_target(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            script_path = temporary_path / "launch.sh"
            executable_path = temporary_path / "tool"

            def write_launch_script(argv, **_kwargs):
                script_path.write_text("#!/bin/bash\n", encoding="utf-8")
                self.assertIn("--script_path=" + str(script_path), argv)
                self.assertIn("--norun_in_cwd", argv)
                self.assertIn("--run_under=python launcher.py", argv)
                self.assertEqual(
                    argv[-5:],
                    ["//pkg:tool", "--", "separator", "--flag", "two words"],
                )
                return 0

            with (
                mock.patch.object(
                    bazel_dev,
                    "resolve_bazel_launch_metadata",
                    return_value=(
                        0,
                        bazel_dev.BazelLaunchMetadata(
                            executable_path=executable_path,
                            runfiles_arguments=["--input=path/to/input"],
                            marked_runfiles_arguments=["--input=path/to/input"],
                            runfiles_environment_names=["IREE_EXAMPLE_LIBRARY"],
                        ),
                    ),
                ),
                mock.patch.object(
                    bazel_dev,
                    "create_bazel_launch_script_path",
                    return_value=script_path,
                ),
                mock.patch.object(
                    bazel_dev,
                    "create_bazel_argument_separator",
                    return_value="separator",
                ),
                mock.patch.object(
                    bazel_dev,
                    "bazel_run_under_command",
                    return_value="python launcher.py",
                ) as run_under_command,
                mock.patch.object(
                    bazel_dev,
                    "run_quietly",
                    side_effect=write_launch_script,
                ),
            ):
                result, launch = bazel_dev.generate_bazel_launch(
                    bazel="bazel",
                    target="//pkg:tool",
                    bazel_args=["--config=asan"],
                    program_args=["--flag", "two words"],
                    run_cwd=temporary_path,
                    env=None,
                    verbose=False,
                )

            self.assertEqual(result, 0)
            self.assertIsNotNone(launch)
            self.assertEqual(launch.target, "//pkg:tool")
            self.assertEqual(launch.run_cwd, temporary_path)
            self.assertEqual(launch.argument_separator, "separator")
            self.assertEqual(launch.program_args, ["--flag", "two words"])
            self.assertEqual(launch.runfiles_arguments, ["--input=path/to/input"])
            self.assertEqual(
                launch.runfiles_environment_names,
                ["IREE_EXAMPLE_LIBRARY"],
            )
            run_under_command.assert_called_once_with(executable_path)

    def test_bazel_launch_rejects_caller_owned_run_under(self):
        with self.assertRaisesRegex(ValueError, "managed by the IREE Bazel launcher"):
            bazel_dev.validate_bazel_launch_args(["--run_under=valgrind"])

    def test_bazel_launch_rejects_unquoted_windows_shell_path(self):
        with (
            mock.patch.object(bazel_dev.os, "name", "nt"),
            self.assertRaisesRegex(ValueError, "BAZEL_SH contains whitespace"),
        ):
            bazel_dev.validate_bazel_launch_environment(
                {bazel_dev.BAZEL_SH_ENV: "C:/Program Files/Git/bin/bash.exe"}
            )

    def test_bazel_launch_uses_the_native_windows_command_interpreter(self):
        launch = bazel_dev.BazelLaunch(
            target="//pkg:tool",
            script_path=Path("C:/work/launch.bat"),
            run_cwd=Path("C:/work"),
            argument_separator="separator",
        )
        environment = {"ComSpec": "C:/Windows/System32/cmd.exe"}

        with mock.patch.object(bazel_dev.os, "name", "nt"):
            argv = launch.argv(environment)

        self.assertEqual(
            argv,
            [
                "C:/Windows/System32/cmd.exe",
                "/d",
                "/c",
                str(Path("C:/work/launch.bat")),
            ],
        )

    def test_bazel_launch_requires_the_native_windows_command_interpreter(self):
        launch = bazel_dev.BazelLaunch(
            target="//pkg:tool",
            script_path=Path("C:/work/launch.bat"),
            run_cwd=Path("C:/work"),
            argument_separator="separator",
        )

        with (
            mock.patch.object(bazel_dev.os, "name", "nt"),
            self.assertRaisesRegex(ValueError, "COMSPEC"),
        ):
            launch.argv({})

    def test_bazel_launch_materializes_before_direct_windows_execution(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            caller_cwd = temporary_path / "caller"
            caller_cwd.mkdir()
            script_path = temporary_path / "launch.bat"
            script_path.write_text("generated", encoding="utf-8")
            launch = bazel_dev.BazelLaunch(
                target="//pkg:tool",
                script_path=script_path,
                run_cwd=caller_cwd,
                argument_separator="separator",
                runfiles_environment_names=["IREE_EXAMPLE_LIBRARY"],
            )
            process_launch = bazel_dev.bazel_launcher.ProcessLaunch(
                argv=["C:/work/tool.exe", "two words"],
                cwd=caller_cwd,
                env={"IREE_EXAMPLE_LIBRARY": "C:/work/libexample.dll"},
            )
            completed = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="encoded launch",
                stderr="",
            )
            environment = {"COMSPEC": "C:/Windows/System32/cmd.exe"}
            with (
                mock.patch.object(bazel_dev.os, "name", "nt"),
                mock.patch.object(
                    bazel_dev.subprocess,
                    "run",
                    return_value=completed,
                ) as run_process,
                mock.patch.object(
                    bazel_dev.bazel_launcher,
                    "decode_process_launch",
                    return_value=process_launch,
                ) as decode_launch,
            ):
                result, prepared_process = bazel_dev.prepare_bazel_process(
                    launch,
                    env=environment,
                )

            self.assertEqual(result, 0)
            self.assertEqual(
                prepared_process,
                bazel_dev.BazelProcess(
                    target="//pkg:tool",
                    argv=process_launch.argv,
                    cwd=process_launch.cwd,
                    env=process_launch.env,
                ),
            )
            decode_launch.assert_called_once_with("encoded launch")
            run_process.assert_called_once()
            run_call = run_process.call_args
            self.assertEqual(
                run_call.args[0],
                [
                    "C:/Windows/System32/cmd.exe",
                    "/d",
                    "/c",
                    str(script_path),
                ],
            )
            self.assertEqual(run_call.kwargs["cwd"], bazel_dev.REPO_ROOT)
            self.assertEqual(run_call.kwargs["stdout"], subprocess.PIPE)
            self.assertTrue(run_call.kwargs["text"])
            self.assertEqual(
                run_call.kwargs["env"][bazel_dev.bazel_launcher.MATERIALIZE_ENV],
                "1",
            )
            self.assertFalse(script_path.exists())

    def test_bazel_run_execs_the_launch_script_with_the_caller_contract(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            caller_cwd = temporary_path / "caller"
            caller_cwd.mkdir()
            script_path = temporary_path / "launch.sh"
            script_path.write_text("generated", encoding="utf-8")
            command = bazel_dev.BazelRunCommand(
                target="//pkg:tool",
                bazel_args=["--config=asan"],
                program_args=["--flag", "two words"],
                run_cwd=caller_cwd,
            )
            environment = {
                "COMSPEC": "cmd.exe",
                "IREE_EXPLICIT_ENV": "preserved",
            }
            launch = bazel_dev.BazelLaunch(
                target=command.target,
                script_path=script_path,
                run_cwd=caller_cwd,
                argument_separator="separator",
                runfiles_environment_names=["IREE_EXAMPLE_LIBRARY"],
            )
            process_launch = bazel_dev.BazelProcess(
                target=command.target,
                argv=["generated-launch"],
                cwd=bazel_dev.REPO_ROOT,
                env={"IREE_EXPLICIT_ENV": "preserved"},
                script_path=script_path,
            )
            step = bazel_dev.BazelRunStep("bazel", command, env=environment)
            with (
                mock.patch.object(
                    bazel_dev,
                    "generate_bazel_launch",
                    return_value=(0, launch),
                ) as generate_launch,
                mock.patch.object(
                    bazel_dev,
                    "prepare_bazel_process",
                    return_value=(0, process_launch),
                ) as prepare_process,
                mock.patch.object(
                    bazel_dev,
                    "exec_path",
                    return_value=17,
                ) as exec_path,
            ):
                result = step.run(verbose=True)

            self.assertEqual(result, 17)
            generate_launch.assert_called_once_with(
                bazel="bazel",
                target="//pkg:tool",
                bazel_args=["--config=asan"],
                program_args=["--flag", "two words"],
                run_cwd=caller_cwd,
                env=environment,
                verbose=True,
            )
            prepare_process.assert_called_once_with(launch, env=environment)
            exec_path.assert_called_once_with(
                process_launch.argv,
                cwd=process_launch.cwd,
                env=process_launch.env,
            )
            self.assertFalse(script_path.exists())

    def test_bazel_run_print_path_keeps_the_direct_build_contract(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            binary_path = Path(temporary_dir) / "tool"
            binary_path.write_text("fixture", encoding="utf-8")
            command = bazel_dev.BazelRunCommand(
                target="//pkg:tool",
                bazel_args=["--config=asan"],
                print_path=True,
            )
            step = bazel_dev.BazelRunStep("bazel", command)
            with (
                mock.patch.object(
                    bazel_dev,
                    "run_quietly",
                    return_value=0,
                ) as run_quietly,
                mock.patch.object(
                    bazel_dev,
                    "resolve_bazel_output_path",
                    return_value=binary_path,
                ) as resolve_output,
                mock.patch.object(
                    bazel_dev,
                    "generate_bazel_launch",
                ) as generate_launch,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                result = step.run()

            self.assertEqual(result, 0)
            self.assertEqual(stdout.getvalue().strip(), str(binary_path))
            run_quietly.assert_called_once_with(
                ["bazel", "build", "--config=asan", "//pkg:tool"],
                cwd=bazel_dev.REPO_ROOT,
                env=None,
                verbose=False,
            )
            resolve_output.assert_called_once_with(
                bazel="bazel",
                target="//pkg:tool",
                bazel_args=["--config=asan"],
                cwd=bazel_dev.REPO_ROOT,
                env=None,
            )
            generate_launch.assert_not_called()

    def test_bazel_fuzz_builds_every_discovered_target_before_launching(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            targets = ["//a:a_fuzz", "//z:z_fuzz"]
            launches = {
                target: bazel_dev.BazelLaunch(
                    target=target,
                    script_path=temporary_path / f"{index}.sh",
                    run_cwd=bazel_dev.REPO_ROOT,
                    argument_separator="separator",
                )
                for index, target in enumerate(targets)
            }
            processes = {
                target: bazel_dev.BazelProcess(
                    target=target,
                    argv=[f"fuzzer-{index}"],
                    cwd=bazel_dev.REPO_ROOT,
                    env={"FUZZER": target},
                )
                for index, target in enumerate(targets)
            }
            metadata_by_target = {
                target: bazel_dev.BazelLaunchMetadata(
                    executable_path=temporary_path / f"fuzzer-{index}",
                    runfiles_environment_names=[f"FUZZER_{index}"],
                )
                for index, target in enumerate(targets)
            }
            command = bazel_dev.BazelTargetCommand(
                target="//...",
                bazel_args=["--config=asan"],
                program_args=["-runs=10"],
            )
            step = bazel_dev.BazelFuzzStep("bazel", command)
            events = []

            def build(argv, **_kwargs):
                events.append(("build", argv))
                return 0

            def resolve_metadata(**kwargs):
                events.append(("metadata", kwargs["targets"], kwargs["bazel_args"]))
                return 0, metadata_by_target

            def generate(**kwargs):
                target = kwargs["target"]
                events.append(
                    (
                        "generate",
                        target,
                        kwargs["program_args"],
                        kwargs["metadata"],
                    )
                )
                return 0, launches[target]

            def prepare(launch, *, env):
                events.append(("prepare", launch.target, env))
                return 0, processes[launch.target]

            def run_many(generated_processes):
                events.append(
                    ("run", [process.target for process in generated_processes])
                )
                return 23

            discovered = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="//z:z_fuzz\n//ignored:library\n//a:a_fuzz\n",
                stderr="",
            )
            with (
                mock.patch.object(
                    bazel_dev,
                    "run_captured",
                    return_value=discovered,
                ),
                mock.patch.object(bazel_dev, "run_quietly", side_effect=build),
                mock.patch.object(
                    bazel_dev,
                    "resolve_bazel_launch_metadata_for_targets",
                    side_effect=resolve_metadata,
                ),
                mock.patch.object(
                    bazel_dev,
                    "generate_bazel_launch",
                    side_effect=generate,
                ),
                mock.patch.object(
                    bazel_dev,
                    "prepare_bazel_process",
                    side_effect=prepare,
                ),
                mock.patch.object(bazel_dev, "run_fuzzers", side_effect=run_many),
                mock.patch.object(
                    bazel_dev.fuzz,
                    "bazel_fuzz_target_dir",
                    side_effect=lambda target: (
                        temporary_path / target.rsplit(":", 1)[1]
                    ),
                ),
            ):
                result = step.run(verbose=True)

            self.assertEqual(result, 23)
            self.assertEqual(events[0][0], "build")
            self.assertEqual(
                events[0][1],
                [
                    "bazel",
                    "build",
                    "--config=fuzzer",
                    "--config=asan",
                    *targets,
                ],
            )
            self.assertEqual(
                events[1],
                ("metadata", targets, ["--config=fuzzer", "--config=asan"]),
            )
            self.assertEqual([event[1] for event in events[2:4]], targets)
            for event, target in zip(events[2:4], targets, strict=True):
                target_dir = temporary_path / target.rsplit(":", 1)[1]
                self.assertEqual(
                    event[2],
                    [
                        str(target_dir / "corpus"),
                        f"-artifact_prefix={target_dir / 'artifacts'}/",
                        "-runs=10",
                    ],
                )
                self.assertEqual(event[3], metadata_by_target[target])
                self.assertTrue((target_dir / "corpus").is_dir())
                self.assertTrue((target_dir / "artifacts").is_dir())
            self.assertEqual(
                events[4:6], [("prepare", target, None) for target in targets]
            )
            self.assertEqual(events[6], ("run", targets))

    def test_bazel_fuzz_stops_remaining_launches_after_a_failure(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            process_launches = [
                bazel_dev.BazelProcess(
                    target=f"//pkg:fuzzer_{index}",
                    argv=[f"fuzzer-{index}"],
                    cwd=bazel_dev.REPO_ROOT,
                    env={"FUZZER": str(index)},
                    script_path=temporary_path / f"launch-{index}.sh",
                )
                for index in range(2)
            ]
            for process_launch in process_launches:
                process_launch.script_path.write_text("generated", encoding="utf-8")
            events = []

            class FuzzerProcess:
                def __init__(self, returncode):
                    self.returncode = returncode
                    self.signal = None

                def poll(self):
                    events.append(("poll", self.returncode, self.signal))
                    return self.returncode

                def send_signal(self, sent_signal):
                    self.signal = sent_signal
                    self.returncode = -sent_signal
                    events.append(("signal", sent_signal))

                def wait(self):
                    events.append(("wait", self.returncode))
                    return self.returncode

            processes = [FuzzerProcess(7), FuzzerProcess(None)]

            def start(*_args, **_kwargs):
                process = processes[
                    len([event for event in events if event == "start"])
                ]
                events.append("start")
                return process

            with mock.patch.object(
                bazel_dev.subprocess,
                "Popen",
                side_effect=start,
            ) as popen:
                result = bazel_dev.run_fuzzers(process_launches)

            self.assertEqual(result, 7)
            self.assertEqual(events[:2], ["start", "start"])
            self.assertEqual(processes[1].signal, bazel_dev.signal.SIGINT)
            self.assertEqual(popen.call_count, 2)
            for call, process_launch in zip(
                popen.call_args_list, process_launches, strict=True
            ):
                self.assertEqual(call.args[0], process_launch.argv)
                self.assertEqual(call.kwargs["cwd"], process_launch.cwd)
                self.assertEqual(call.kwargs["env"], process_launch.env)
                self.assertFalse(process_launch.script_path.exists())


if __name__ == "__main__":
    unittest.main()
