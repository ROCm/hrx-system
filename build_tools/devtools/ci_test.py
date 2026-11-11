# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# =============================================================================
# TEST TOOL BEHAVIOR. DO NOT ADD CHANGE-DETECTOR TESTS.
# =============================================================================
# CI definitions are input data, not a specification to mirror in assertions.
# Adding or changing a target, tag filter, platform, workflow, or other policy
# data must not require adding or changing tests in this file. Changes to Python
# behavior warrant tests; edits to the data consumed by that behavior do not.
#
# Do not assert that a configured list appears in the generated command, pin a
# workflow's current text or matrix, or enumerate today's command definitions.
# Reading expected values from ci_config instead of copying their literals is
# still a change detector. Computing expectations with the same production
# helper being exercised is not an independent behavioral check either.
#
# Exercise the tool's mechanisms with controlled inputs: argument validation,
# quoting and escaping, user overrides, execution ordering, dry-run side effects,
# failure propagation, and filesystem or subprocess behavior. Synthetic fixtures
# describe those behaviors without reproducing the repository's policy tables.
# Actual configured builds and test runs validate the selected product coverage.
#
# A useful test explains what the Python code does incorrectly when it fails,
# not which declarative definition needs to be copied into an expected result.
# =============================================================================

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.bazel import configure as bazel_configure
from build_tools.devtools import ci, ci_config, run_requirements
from runtime.build_tools.bazel import configure as runtime_configure


class CiTest(unittest.TestCase):
    def test_repository_compilation_mode_overrides_build_defaults_only(self):
        for mode in (None, "fastbuild", "dbg", "opt"):
            with self.subTest(mode=mode):
                command = ["iree-bazel-repository-build"]
                if mode is not None:
                    command += ["--compilation-mode", mode]
                args = ci.parse_arguments(command)
                with mock.patch.object(
                    ci_config, "BAZEL_DEFAULT_OPTIONS", ("--compilation_mode=dbg",)
                ):
                    configure, build = ci.steps_from_args(args)
                prefix = "--compilation_mode="
                self.assertFalse(any(arg.startswith(prefix) for arg in configure.argv))
                modes = [arg for arg in build.argv if arg.startswith(prefix)]
                self.assertEqual(
                    modes, [prefix + "dbg"] + ([prefix + mode] if mode else [])
                )

    def test_compilation_mode_rejects_commands_that_cannot_apply_it(self):
        for command in ("iree-cmake-repository-build", "iree-bazel-cpu-asan"):
            with self.subTest(command=command):
                args = ci.parse_arguments([command, "--compilation-mode", "opt"])
                with self.assertRaisesRegex(ValueError, "--compilation-mode"):
                    ci.steps_from_args(args)

    def test_cts_family_filter_composes_with_requested_cases(self):
        with (
            mock.patch.object(
                ci_config,
                "AMD_GPU_CTS_TARGET_PREFIXES",
                {"alpha": ("chip-a",), "beta": ("chip-b",)},
            ),
            mock.patch.dict(os.environ, {"GTEST_FILTER": "Selected.*-Ignored.*"}),
        ):
            self.assertEqual(
                ci.amd_gpu_cts_test_env("chip-a0,chip-a1"),
                (("GTEST_FILTER", "Selected.*-Ignored.*:CTS*/*/beta:CTS*/*/beta_*"),),
            )
            self.assertEqual(
                ci.amd_gpu_cts_test_env("chip-b0"),
                (("GTEST_FILTER", "Selected.*-Ignored.*:CTS*/*/alpha:CTS*/*/alpha_*"),),
            )
            for selector in ("chip-a0,chip-b0", "all", "unknown", ""):
                with self.subTest(selector=selector):
                    self.assertEqual(ci.amd_gpu_cts_test_env(selector), ())

    def test_cts_family_filter_defaults_to_all_other_cases(self):
        with (
            mock.patch.object(
                ci_config,
                "AMD_GPU_CTS_TARGET_PREFIXES",
                {"alpha": ("chip-a",), "beta": ("chip-b",)},
            ),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            self.assertEqual(
                ci.amd_gpu_cts_test_env("chip-a0"),
                (("GTEST_FILTER", "*-CTS*/*/beta:CTS*/*/beta_*"),),
            )

    def test_sanitizer_uses_explicit_symbolizer_before_toolchain_discovery(self):
        with (
            mock.patch.dict(
                os.environ,
                {"UBSAN_SYMBOLIZER_PATH": "/chosen/llvm-symbolizer"},
                clear=True,
            ),
            mock.patch.object(ci.shutil, "which") as which,
        ):
            step = ci.bazel_test_step("Test", ("//suite:test",), config="ubsan")
        self.assertIn(
            "--test_env=UBSAN_SYMBOLIZER_PATH=/chosen/llvm-symbolizer", step.argv
        )
        which.assert_not_called()

    def test_sanitizer_discovers_existing_toolchain_symbolizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binaries = root / "lib" / "llvm" / "bin"
            binaries.mkdir(parents=True)
            for platform, executable in (
                ("linux", "llvm-symbolizer"),
                ("win32", "llvm-symbolizer.exe"),
            ):
                symbolizer = binaries / executable
                symbolizer.touch()
                with (
                    self.subTest(platform=platform),
                    mock.patch.object(ci.sys, "platform", platform),
                    mock.patch.dict(
                        os.environ, {"HRX_ROCM_ROOT": str(root)}, clear=True
                    ),
                    mock.patch.object(ci.shutil, "which") as which,
                ):
                    environment = dict(ci.sanitizer_env("tsan"))
                    self.assertEqual(
                        environment["TSAN_SYMBOLIZER_PATH"], str(symbolizer)
                    )
                    self.assertIn("suppressions=", environment["TSAN_OPTIONS"])
                    which.assert_not_called()

    def test_sanitizer_symbolizer_discovery_does_not_invent_missing_paths(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(ci.shutil, "which", return_value=None) as which,
        ):
            self.assertEqual(ci.sanitizer_env(None), ())
            which.assert_not_called()
            self.assertEqual(ci.sanitizer_env("asan"), ())
            which.return_value = "/tools/llvm-symbolizer"
            self.assertEqual(
                dict(ci.sanitizer_env("msan")),
                {"MSAN_SYMBOLIZER_PATH": "/tools/llvm-symbolizer"},
            )

    def test_requirement_audit_precedes_execution_and_rejects_unknown_tags(self):
        for requirement, audit_code, expected_code in (
            ("device", 0, 0),
            ("typo", 0, 1),
            ("device", 7, 7),
        ):
            with self.subTest(requirement=requirement):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    output = root / "executed"
                    payload = f'<query><rule name="//suite:test"><list name="tags"><string value="iree-run-requirement={requirement}"/></list></rule></query>'
                    step = ci.CiStep(
                        "Test",
                        (
                            sys.executable,
                            "-c",
                            "from pathlib import Path; Path('executed').touch()",
                        ),
                        requirement_audit=ci.RequirementAudit(
                            (
                                sys.executable,
                                "-c",
                                f"print({payload!r}); raise SystemExit({audit_code})",
                            ),
                            run_requirements.audit_bazel_xml,
                        ),
                    )
                    with (
                        mock.patch.object(ci, "REPO_ROOT", root),
                        mock.patch.object(
                            run_requirements, "declared_ids", return_value={"device"}
                        ),
                        mock.patch.dict(os.environ, {}, clear=True),
                        contextlib.redirect_stdout(io.StringIO()),
                        contextlib.redirect_stderr(io.StringIO()),
                    ):
                        result = ci.run_step(step, verbose=False)
                    self.assertEqual(result.returncode, expected_code)
                    self.assertEqual(output.exists(), expected_code == 0)

    def test_resource_admission_composes_with_focus_filters(self):
        with mock.patch.object(
            run_requirements, "declared_ids", return_value={"device.a", "device.b"}
        ):
            bazel_step = ci.bazel_test_step(
                "Test",
                ("//suite/...",),
                test_tag_filters=("focus",),
                test_language_filters=("-py",),
                available_resources=("device.a",),
            )
            cmake_step = ci.cmake_test_step(
                "example",
                "Test",
                label_regex="focus",
                label_exclude_regex="manual",
                available_resources=("device.a",),
            )
        bazel_filter = next(
            arg for arg in bazel_step.argv if arg.startswith("--test_tag_filters=")
        )
        self.assertEqual(
            bazel_filter, "--test_tag_filters=focus,-iree-run-requirement=device.b"
        )
        self.assertIn("--test_lang_filters=-py", bazel_step.argv)
        ctest_filter = cmake_step.argv[cmake_step.argv.index("-LE") + 1]
        self.assertIsNotNone(re.search(ctest_filter, "iree-run-requirement=device.b"))
        self.assertIsNotNone(re.search(ctest_filter, "manual"))
        self.assertIsNone(re.search(ctest_filter, "iree-run-requirement=device.a"))

    def test_job_options_override_defaults_and_filter_overrides_fail_loud(self):
        with (
            mock.patch.object(ci_config, "BAZEL_DEFAULT_OPTIONS", ("--choice=off",)),
            mock.patch.object(
                run_requirements, "declared_ids", return_value={"device"}
            ),
        ):
            options = ("--choice=on",)
            step = ci.bazel_test_step("Test", ("//suite:test",), bazel_options=options)
            self.assertLess(
                step.argv.index("--choice=off"), step.argv.index("--choice=on")
            )
            filters = [
                arg for arg in step.argv if arg.startswith("--test_tag_filters=")
            ]
            self.assertEqual(
                filters[-1], "--test_tag_filters=-iree-run-requirement=device"
            )
            for option in ("--test_tag_filters", "--test_tag_filters=focus"):
                with self.assertRaisesRegex(ValueError, "Use test_tag_filters"):
                    ci.bazel_test_step(
                        "Test", ("//suite:test",), bazel_options=(option,)
                    )

            for option in ("--test_lang_filters", "--test_lang_filters=-py"):
                with self.assertRaisesRegex(ValueError, "Use test_language_filters"):
                    ci.bazel_test_step(
                        "Test", ("//suite:test",), bazel_options=(option,)
                    )

    def test_bazel_test_builds_only_selected_test_rules(self):
        test_step = ci.bazel_test_step("Test", ("//suite/...",))
        build_step = ci.bazel_build_step("Build", ("//suite/...",))

        self.assertIn("--build_tests_only", test_step.argv)
        self.assertNotIn("--build_tests_only", build_step.argv)

    def test_platform_selection_applies_to_each_build_phase(self):
        for selection, bazel_list, cmake_list in (
            (None, "", ""),
            ("0", "", ""),
            ("1", "first,second", "first;second"),
        ):
            with self.subTest(selection=selection):
                environment = (
                    {}
                    if selection is None
                    else {"IREE_CI_PLATFORM_COVERAGE": selection}
                )
                with (
                    mock.patch.dict(os.environ, environment, clear=True),
                    mock.patch.object(
                        ci_config, "OPTIONAL_IMAGE_CODECS", ("first", "second")
                    ),
                    mock.patch.object(ci_config, "OPTIONAL_AUDIO_CODECS", ("voice",)),
                ):
                    steps = (
                        ci.bazel_configure_step(),
                        ci.bazel_build_step("Build", ("//suite:target",)),
                        ci.bazel_test_step("Test", ("//suite:target",)),
                        ci.bazel_run_step("Run", "//suite:target"),
                    )
                    cmake_step = ci.cmake_configure_step("example")
                for step in steps:
                    for kind, codecs in (
                        ("image", bazel_list),
                        ("audio", "voice" if selection == "1" else ""),
                    ):
                        prefix = "--//loom/config/serve:" + kind + "_codecs="
                        self.assertEqual(
                            [arg for arg in step.argv if arg.startswith(prefix)],
                            [prefix + codecs],
                        )
                self.assertIn(
                    "-DLOOM_SERVE_IMAGE_CODECS=" + cmake_list, cmake_step.argv
                )
                self.assertIn(
                    "-DLOOM_SERVE_AUDIO_CODECS="
                    + ("voice" if selection == "1" else ""),
                    cmake_step.argv,
                )
                self.assertEqual(
                    shlex.split(cmake_step.command_line()), list(cmake_step.argv)
                )

    def test_platform_selection_owns_compile_coverage_build_steps(self):
        step = ci.CiStep("Compile coverage", ("tool", "build"))
        for selection, expected_steps in (("0", []), ("1", [step])):
            with self.subTest(selection=selection):
                with mock.patch.dict(
                    os.environ,
                    {"IREE_CI_PLATFORM_COVERAGE": selection},
                    clear=True,
                ):
                    self.assertEqual(
                        ci.platform_coverage_build_steps(step), expected_steps
                    )

    def test_native_amd_engine_composition_matches_between_build_systems(self):
        for drivers, gpu, xdna in (
            ((), False, False),
            (("amd",), True, True),
            (("amd-gpu",), True, False),
            (("amd-xdna",), False, True),
            (("amd-gpu", "amd-xdna"), True, True),
        ):
            with self.subTest(drivers=drivers):
                step = ci.bazel_configure_step(enabled_drivers=drivers)
                request = bazel_configure.request_from_args(
                    bazel_configure.parse_arguments(step.argv[4:])
                )
                cmake = ci.cmake_configure_step("example", enabled_drivers=drivers).argv
                for engine, enabled in (("GPU", gpu), ("XDNA", xdna)):
                    option = f"-DIREE_HAL_DRIVER_AMD_{engine}="
                    self.assertEqual(
                        "amd-" + engine.lower()
                        in runtime_configure.configuration(request).drivers,
                        enabled,
                    )
                    self.assertIn(option + ("ON" if enabled else "OFF"), cmake)

    def test_bazel_complete_driver_list_overrides_ci_selection(self):
        for drivers, selections, expected in (
            (("amd",), ("task",), {"task"}),
            ((), ("task,amd-gpu",), {"task", "amd-gpu"}),
            ((), ("task,amd-xdna",), {"task", "amd-xdna"}),
            (
                (),
                ("task,amd", "task,vulkan,amd"),
                {"task", "vulkan", "amd-gpu", "amd-xdna"},
            ),
            (("amd-gpu",), ("",), set()),
        ):
            with self.subTest(drivers=drivers, selections=selections):
                step = ci.bazel_configure_step(
                    enabled_drivers=drivers,
                    extra_options=tuple(
                        "--//runtime/config/hal:drivers=" + selection
                        for selection in selections
                    ),
                )
                request = bazel_configure.request_from_args(
                    bazel_configure.parse_arguments(step.argv[4:])
                )
                self.assertEqual(
                    runtime_configure.configuration(request).drivers, expected
                )

    def test_codec_policy_precedes_explicit_job_overrides(self):
        with mock.patch.object(ci, "codec_options", return_value=("--choice=default",)):
            override = ("--choice=job",)
            steps = (
                ci.bazel_configure_step(extra_options=override),
                ci.bazel_build_step(
                    "Build", ("//suite:target",), bazel_options=override
                ),
                ci.bazel_test_step("Test", ("//suite:target",), bazel_options=override),
                ci.bazel_run_step("Run", "//suite:target", bazel_options=override),
                ci.cmake_configure_step("example", extra_options=override),
            )
        for step in steps:
            self.assertLess(
                step.argv.index("--choice=default"), step.argv.index("--choice=job")
            )

    def test_invalid_platform_selection_fails_loud(self):
        with mock.patch.dict(os.environ, {"IREE_CI_PLATFORM_COVERAGE": "treu"}):
            with self.assertRaisesRegex(
                ValueError, "IREE_CI_PLATFORM_COVERAGE must be 0 or 1"
            ):
                ci.bazel_configure_step()

    def test_audit_survives_step_options_and_dry_run_has_no_subprocesses(self):
        step = ci.bazel_test_step("Test", ("//suite:test",))
        args = ci.parse_arguments(
            [
                "iree-bazel-cpu",
                "--keep-going",
                "--bazel-profile-dir",
                "/tmp/profiles",
            ]
        )
        with mock.patch.object(ci, "_steps_from_args", return_value=[step]):
            planned = ci.steps_from_args(args)
        self.assertIs(planned[0].requirement_audit, step.requirement_audit)
        with (
            mock.patch.object(ci.subprocess, "run") as run,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(
                ci.run_steps(planned, dry_run=True, keep_going=True, verbose=False), 0
            )
        run.assert_not_called()

    def test_keep_going_reaches_bazel_before_target_and_program_separators(self):
        steps = [
            ci.bazel_build_step("Build", ("//example:one", "-//example:excluded")),
            ci.bazel_test_step("Test", ("//example:one",)),
            ci.bazel_run_step("Run", "//example:tool", ("--nokeep_going",)),
            ci.CiStep("Configure", ("python", "dev.py", "bazel", "configure")),
            ci.CiStep("Other tool", ("python", "tool.py", "--", "--nokeep_going")),
        ]
        for keep_going in (False, True):
            with self.subTest(keep_going=keep_going):
                args = ci.parse_arguments(
                    ["iree-bazel-cpu"] + (["--keep-going"] if keep_going else [])
                )
                with mock.patch.object(
                    ci, "_steps_from_args", return_value=list(steps)
                ):
                    planned = ci.steps_from_args(args)
                self.assertEqual(planned[3:], steps[3:])
                for original, phase in zip(steps[:3], planned[:3]):
                    if keep_going:
                        self.assertEqual(phase.argv[4], "--keep_going")
                        self.assertEqual(phase.argv[5:], original.argv[4:])
                    elif original.argv[1:4] == ("dev.py", "bazel", "test"):
                        self.assertEqual(phase.argv[4], "--notest_keep_going")
                        self.assertEqual(phase.argv[5:], original.argv[4:])
                    else:
                        self.assertEqual(phase, original)

    def test_keep_going_captures_failure_before_next_phase_changes_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sdk = root / "sdk"
            sdk.mkdir()
            includer = sdk / "wchar.h"
            includer.write_text("#include <missing.h>\n")
            missing = sdk / "missing.h"
            message = (
                f"{includer}(17): fatal error C1083: Cannot open include file: "
                "'missing.h': No such file or directory"
            )
            steps = [
                ci.CiStep(
                    "First build",
                    (sys.executable, "-c", f"print({message!r}); raise SystemExit(7)"),
                ),
                ci.CiStep(
                    "Later build",
                    (
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(missing)!r}).write_text('now present')",
                    ),
                ),
            ]
            with (
                mock.patch.object(ci, "REPO_ROOT", root),
                mock.patch.dict(
                    os.environ,
                    {
                        "IREE_CI_FAILURE_ARTIFACT_DIR": str(root / "artifacts"),
                        "INCLUDE": str(sdk),
                    },
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = ci.run_steps(
                    steps, dry_run=False, keep_going=True, verbose=False
                )
            self.assertEqual(result, 7)
            self.assertTrue(missing.exists())
            manifests = list((root / "artifacts").glob("*/manifest.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text())
            record = next(
                file for file in manifest["files"] if file["path"] == str(missing)
            )
            self.assertIn("error", record)
            self.assertNotIn("sha256", record)

    def uses_cmake_build_dir(self, step: ci.CiStep, command_name: str) -> bool:
        expected_args = (
            "--cmake-build-dir",
            str(ci.CMAKE_CI_BUILD_ROOT / command_name),
        )
        return any(
            step.argv[index : index + 2] == expected_args
            for index in range(len(step.argv) - 1)
        )

    def ctest_exclude_regexes(self, step: ci.CiStep) -> list[str]:
        return [
            step.argv[index + 1]
            for index, arg in enumerate(step.argv[:-1])
            if arg == "-E"
        ]

    def workflow_job_block(self, path: str, job_name: str) -> str:
        """Returns the body of ``job_name`` in the workflow at ``path``.

        A job body is blank lines and lines indented at least four spaces, so
        the first line that is neither ends it: the next job's key, a comment
        introducing that job, or anything after ``jobs:``. Letting a comment
        run on into the preceding body would make assertions over that body
        report the wrong thing in both directions -- ``assertIn`` could pass on
        text belonging to the next job, and ``assertNotIn`` could fail on it.
        """
        text = Path(path).read_text()
        match = re.search(
            rf"^  {re.escape(job_name)}:\n"
            r"(?P<body>.*?)(?=^(?![ \t]*\n)(?! {4})|\Z)",
            text,
            re.MULTILINE | re.DOTALL,
        )
        self.assertIsNotNone(match)
        return match.group("body")

    def test_cpu_dry_run_exposes_copyable_test_closure(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-cpu",
                "--dry-run",
                "--target",
                "//runtime/...",
            ]
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                ci.run_steps(
                    ci.steps_from_args(args),
                    dry_run=True,
                    keep_going=False,
                    verbose=False,
                ),
                0,
            )

        text = output.getvalue()
        self.assertIn("dev.py bazel configure", text)
        self.assertNotIn("dev.py bazel build ", text)
        self.assertIn("dev.py bazel test ", text)
        self.assertIn(" -- //runtime/...", text)
        self.assertIn("-//runtime/src/iree/hal/driver/amdgpu/...", text)
        self.assertIn("-//runtime/src/iree/hal/driver/vulkan/...", text)
        self.assertIn(
            "--test_tag_filters=" + ",".join(ci_config.CPU_RESOURCE_TAG_EXCLUDES),
            text,
        )

    def test_bazel_profiles_are_named_for_each_selected_phase(self):
        profile_dir = Path("/tmp/iree-bazel-profiles")
        resolved_profile_dir = profile_dir.resolve()
        args = ci.parse_arguments(
            [
                "iree-bazel-cpu",
                "--target",
                "//runtime/...",
                "--bazel-profile-dir",
                str(profile_dir),
            ]
        )

        for selection, expected_names in (
            ("0", ("test-iree.profile.gz",)),
            ("1", ("build-iree.profile.gz", "test-iree.profile.gz")),
        ):
            with self.subTest(selection=selection):
                with mock.patch.dict(
                    os.environ,
                    {"IREE_CI_PLATFORM_COVERAGE": selection},
                    clear=True,
                ):
                    steps = ci.steps_from_args(args)
                self.assertEqual(
                    [
                        arg
                        for step in steps
                        for arg in step.argv
                        if arg.startswith("--profile=")
                    ],
                    [
                        f"--profile={resolved_profile_dir / name}"
                        for name in expected_names
                    ],
                )
                self.assertEqual(
                    [
                        step.bazel_profile
                        for step in steps
                        if step.bazel_profile is not None
                    ],
                    [resolved_profile_dir / name for name in expected_names],
                )

    def test_profile_summary_is_published_after_success_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile_path = root / "test-example.profile.gz"
            output_path = root / "test-example.summary.json"
            step = ci.CiStep(
                "Test example",
                ("example",),
                bazel_profile=profile_path,
            )
            summary = object()

            for command_code in (0, 7):
                with self.subTest(command_code=command_code):
                    job_summary_path = root / f"job-{command_code}.md"
                    with (
                        mock.patch.object(
                            ci.subprocess,
                            "run",
                            return_value=subprocess.CompletedProcess(
                                step.argv, command_code
                            ),
                        ),
                        mock.patch.object(
                            ci.bazel_profile_summary,
                            "summarize",
                            return_value=summary,
                        ) as summarize,
                        mock.patch.object(
                            ci.bazel_profile_summary,
                            "summary_path",
                            return_value=output_path,
                        ),
                        mock.patch.object(
                            ci.bazel_profile_summary, "write"
                        ) as write_summary,
                        mock.patch.object(
                            ci.bazel_profile_summary,
                            "render_console",
                            return_value="console summary",
                        ),
                        mock.patch.object(
                            ci.bazel_profile_summary,
                            "render_markdown",
                            return_value="markdown summary",
                        ),
                        mock.patch.dict(
                            os.environ,
                            {"GITHUB_STEP_SUMMARY": str(job_summary_path)},
                            clear=True,
                        ),
                        contextlib.redirect_stdout(io.StringIO()),
                        contextlib.redirect_stderr(io.StringIO()),
                    ):
                        result = ci.run_step(step, verbose=False)

                    self.assertEqual(result.returncode, command_code)
                    summarize.assert_called_once_with(profile_path)
                    write_summary.assert_called_once_with(summary, output_path)
                    self.assertEqual(job_summary_path.read_text(), "markdown summary\n")

    def test_profile_summary_failure_preserves_command_failure(self):
        profile_error = ci.bazel_profile_summary.BazelProfileSummaryError(
            "truncated profile"
        )
        step = ci.CiStep(
            "Test example",
            ("example",),
            bazel_profile=Path("/tmp/test-example.profile.gz"),
        )

        for command_code, expected_code in ((0, 1), (7, 7)):
            with self.subTest(command_code=command_code):
                stderr = io.StringIO()
                with (
                    mock.patch.object(
                        ci.subprocess,
                        "run",
                        return_value=subprocess.CompletedProcess(
                            step.argv, command_code
                        ),
                    ),
                    mock.patch.object(
                        ci,
                        "publish_bazel_profile",
                        side_effect=profile_error,
                    ),
                    mock.patch.dict(os.environ, {}, clear=True),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(stderr),
                ):
                    result = ci.run_step(step, verbose=False)

                self.assertEqual(result.returncode, expected_code)
                self.assertIn("truncated profile", stderr.getvalue())

    def test_bazel_test_results_capture_only_test_phases(self):
        result_dir = Path("/tmp/iree-bazel-test-results").resolve()
        steps = [
            ci.bazel_build_step("Build example", ("//example:all",)),
            ci.bazel_test_step("Test example", ("//example:all",)),
        ]
        args = ci.parse_arguments(
            ["iree-bazel-cpu", "--bazel-test-results-dir", str(result_dir)]
        )

        with mock.patch.object(ci, "_steps_from_args", return_value=steps):
            planned = ci.steps_from_args(args)

        self.assertEqual(planned[0], steps[0])
        event_path = result_dir / ".events" / "test-example.json"
        self.assertIn(f"--build_event_json_file={event_path}", planned[1].argv)
        self.assertEqual(
            planned[1].test_results,
            ci.bazel_test_results.BazelTestResultsCapture(
                build_event_path=event_path,
                output_dir=result_dir / "test-example",
            ),
        )

    def test_repeated_test_phases_preserve_each_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "results"
            ci.bazel_test_results.prepare_output_directory(result_dir)
            source = root / "testlogs" / "example" / "test.xml"
            source.parent.mkdir(parents=True)
            steps = ci.add_bazel_test_results(
                [
                    ci.bazel_test_step("First", ("//example:test",)),
                    ci.bazel_test_step("Second", ("//example:test",)),
                ],
                result_dir,
            )
            event = {
                "testResult": {
                    "testActionOutput": [{"name": "test.xml", "uri": source.as_uri()}]
                }
            }
            for step, payload in zip(steps, ("first result", "second result")):
                source.write_text(payload)
                capture = step.test_results
                capture.build_event_path.write_text(json.dumps(event) + "\n")
                self.assertEqual(
                    ci.bazel_test_results.stage(
                        capture.build_event_path, capture.output_dir
                    ),
                    1,
                )
            self.assertEqual(
                sorted(path.read_text() for path in result_dir.rglob("test.xml")),
                ["first result", "second result"],
            )

    def test_failed_test_stages_results_before_stopping_later_phases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "testlogs" / "example" / "test.log"
            source.parent.mkdir(parents=True)
            events = root / "events.json"
            output = root / "artifacts"
            later = root / "later"
            event = (
                json.dumps(
                    {
                        "testResult": {
                            "testActionOutput": [
                                {"name": "test.log", "uri": source.as_uri()}
                            ]
                        }
                    }
                )
                + "\n"
            )
            script = (
                "from pathlib import Path; "
                f"Path({str(source)!r}).write_text('first native failure'); "
                f"Path({str(events)!r}).write_text({event!r}); "
                "raise SystemExit(17)"
            )
            steps = [
                ci.CiStep(
                    "Failing test",
                    (sys.executable, "-c", script),
                    test_results=ci.bazel_test_results.BazelTestResultsCapture(
                        events, output
                    ),
                ),
                ci.CiStep(
                    "Later phase",
                    (
                        sys.executable,
                        "-c",
                        f"from pathlib import Path; Path({str(later)!r}).touch()",
                    ),
                ),
            ]
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                returncode = ci.run_steps(
                    steps, dry_run=False, keep_going=False, verbose=False
                )
            self.assertEqual(returncode, 17)
            self.assertFalse(later.exists())
            self.assertEqual(
                (output / "bazel-testlogs/example/test.log").read_text(),
                "first native failure",
            )

    def test_native_gpu_ci_requires_explicit_windows_target(self):
        with mock.patch.object(ci.sys, "platform", "win32"):
            args = ci.parse_arguments(["iree-bazel-amdgpu-native"])
            with self.assertRaisesRegex(ValueError, "explicit --amdgpu-target"):
                ci.steps_from_args(args)
        with mock.patch.object(ci.sys, "platform", "linux"):
            args = ci.parse_arguments(
                ["iree-bazel-amdgpu-native", "--amdgpu-target", "gfx1201"]
            )
            with self.assertRaisesRegex(ValueError, "requires Windows"):
                ci.steps_from_args(args)

    def test_native_gpu_ci_forwards_target_override_and_sanitizer(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu-native-asan",
                "--amdgpu-target",
                "gfx1201",
                "--target",
                "//example:custom",
            ]
        )
        with (
            mock.patch.object(ci.sys, "platform", "win32"),
            mock.patch.object(ci, "amdgpu_windows_steps", return_value=[]) as plan,
        ):
            ci.steps_from_args(args)
        plan.assert_called_once_with(("//example:custom",), "gfx1201", "asan")

    def test_native_gpu_ci_expands_family_and_exact_target_selections(self):
        with (
            mock.patch.object(
                ci.target_map,
                "exact_targets_for_selection",
                side_effect=[["TARGET_A", "TARGET_B"], ["TARGET_C"]],
            ) as expand,
            mock.patch.object(ci.shutil, "which", return_value=None),
        ):
            steps = ci.amdgpu_windows_steps(
                ("//example:custom",), "FAMILY_A,TARGET_C", "asan"
            )
        self.assertEqual(
            expand.call_args_list, [mock.call("FAMILY_A"), mock.call("TARGET_C")]
        )
        prefix = "--test_arg=--amdf_gpu_target="
        qualification = []
        for step in steps:
            targets = [
                arg.removeprefix(prefix) for arg in step.argv if arg.startswith(prefix)
            ]
            if targets:
                qualification.append(targets)
        self.assertTrue(qualification)
        for targets in qualification:
            self.assertEqual(targets, ["TARGET_A", "TARGET_B", "TARGET_C"])

    def test_native_gpu_ci_rejects_unknown_target_before_configuring(self):
        with (
            mock.patch.object(
                ci.target_map,
                "exact_targets_for_selection",
                side_effect=ValueError("unknown target selector"),
            ),
            mock.patch.object(ci, "bazel_configure_step") as configure,
        ):
            with self.assertRaisesRegex(ValueError, "unknown target selector"):
                ci.amdgpu_windows_steps((), "UNKNOWN_FAMILY", "asan")
        configure.assert_not_called()

    def test_bazel_profile_names_must_be_unique(self):
        duplicate_steps = [
            ci.bazel_build_step("Compile IREE", ("//runtime/...",)),
            ci.bazel_test_step("Compile IREE", ("//runtime/...",)),
        ]

        with self.assertRaisesRegex(ValueError, "Bazel profile path collision"):
            ci.add_bazel_profiles(duplicate_steps, Path("/tmp/profiles"))

    def test_non_bazel_commands_reject_bazel_profile_directory(self):
        for option, path in (
            ("--bazel-profile-dir", "/tmp/profiles"),
            ("--bazel-test-results-dir", "/tmp/results"),
        ):
            with self.subTest(option=option):
                args = ci.parse_arguments(["iree-cmake-cpu", option, path])
                with self.assertRaisesRegex(
                    ValueError,
                    f"{option} is only supported for Bazel CI commands",
                ):
                    ci.steps_from_args(args)

    def test_main_creates_bazel_profile_directory_only_when_executing(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            profile_dir = Path(temporary_dir) / "profiles"
            argv = [
                "iree-bazel-cpu",
                "--target",
                "//runtime/...",
                "--bazel-profile-dir",
                str(profile_dir),
            ]
            with mock.patch.object(ci, "run_steps", return_value=0):
                self.assertEqual(ci.main([*argv, "--dry-run"]), 0)
                self.assertFalse(profile_dir.exists())

                self.assertEqual(ci.main(argv), 0)
                self.assertTrue(profile_dir.is_dir())

    def test_main_creates_fresh_bazel_test_result_directory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            result_dir = Path(temporary_dir) / "results"
            argv = [
                "iree-bazel-cpu",
                "--bazel-test-results-dir",
                str(result_dir),
            ]
            with mock.patch.object(ci, "run_steps", return_value=0):
                self.assertEqual(ci.main([*argv, "--dry-run"]), 0)
                self.assertFalse(result_dir.exists())
                self.assertEqual(ci.main(argv), 0)
                self.assertTrue((result_dir / ".events").is_dir())

    def test_bazel_default_test_targets_include_loom(self):
        args = ci.parse_arguments(["iree-bazel-cpu"])

        steps = ci.steps_from_args(args)
        test_step = next(step for step in steps if step.name == "Test IREE")

        self.assertFalse(any("bazel build" in step.command_line() for step in steps))
        self.assertIn("//runtime/...", test_step.argv)
        self.assertIn("//loom/...", test_step.argv)

    def test_bazel_repository_build_covers_supported_linux_graph(self):
        args = ci.parse_arguments(["iree-bazel-repository-build"])

        with (
            tempfile.TemporaryDirectory() as rocm_root,
            mock.patch.dict(
                ci.os.environ,
                {"HRX_ROCM_ROOT": rocm_root},
                clear=True,
            ),
        ):
            (Path(rocm_root) / "include").mkdir()
            steps = ci.steps_from_args(args)
            configure_step = steps[0]
            request = bazel_configure.request_from_args(
                bazel_configure.parse_arguments(configure_step.argv[4:])
            )

        self.assertEqual(
            [step.name for step in steps], ["Configure Bazel", "Build repository"]
        )
        self.assertEqual(
            runtime_configure.configuration(request).drivers,
            {"task", "amdgpu", "vulkan", "webgpu", "amd-gpu", "amd-xdna"},
        )
        self.assertIn(
            "--//loom/config/arch:enable="
            + ",".join(ci.REPOSITORY_BUILD_LOOM_ARCHITECTURES),
            configure_step.argv,
        )
        self.assertIn(
            "--//loom/config/emit:enable="
            + ",".join(ci.REPOSITORY_BUILD_LOOM_ARTIFACT_FORMATS),
            configure_step.argv,
        )
        self.assertIn(
            "--//loom/config/execute:enable="
            + ",".join(ci.REPOSITORY_BUILD_LOOM_EXECUTION_SUBSTRATES),
            configure_step.argv,
        )
        self.assertIn(
            "--//loom/config/import:enable="
            + ",".join(ci.REPOSITORY_BUILD_LOOM_IMPORTERS),
            configure_step.argv,
        )
        self.assertIn(f"-DIREE_ROCM_PATH={rocm_root}", configure_step.argv)
        build_step = steps[-1]
        self.assertEqual(build_step.argv[-1], "//...")
        self.assertIn("--nobuild_runfile_links", build_step.argv)
        self.assertNotIn("--nobuild_runfile_links", configure_step.argv)
        self.assertFalse(any(arg.startswith("-//") for arg in build_step.argv))
        self.assertFalse(any(step.name.startswith("Test") for step in steps))

    def test_bazel_repository_build_rejects_partial_target_scope(self):
        args = ci.parse_arguments(
            ["iree-bazel-repository-build", "--target", "//runtime/..."]
        )

        with self.assertRaisesRegex(ValueError, "repository-wide"):
            ci.steps_from_args(args)

    def test_bazel_repository_tests_build_only_their_dependencies(self):
        args = ci.parse_arguments(["iree-bazel-repository-integration"])
        for platform_coverage in ("0", "1"):
            with (
                self.subTest(platform_coverage=platform_coverage),
                mock.patch.dict(
                    ci.os.environ,
                    {"IREE_CI_PLATFORM_COVERAGE": platform_coverage},
                    clear=True,
                ),
            ):
                configure, test = ci.steps_from_args(args)
                self.assertEqual(configure.argv[2:4], ("bazel", "configure"))
                self.assertEqual(test.argv[2:4], ("bazel", "test"))
                self.assertIn("--build_tests_only", test.argv)
                self.assertNotIn("--//loom/config:shared_tools", test.argv)
                self.assertEqual(
                    "--test_lang_filters=-py" in test.argv, platform_coverage == "0"
                )

    def test_bazel_repository_device_compilation_shares_the_build_invocation(self):
        args = ci.parse_arguments(
            ["iree-bazel-repository-build", "--amdgpu-target", "gfx11-generic"]
        )
        with mock.patch.dict(ci.os.environ, {"HRX_ROCM_ROOT": "/tmp/rocm-root"}):
            configure, build = ci.steps_from_args(args)
        self.assertIn("-DIREE_ROCM_PATH=/tmp/rocm-root", configure.argv)
        self.assertEqual(build.argv[2:4], ("bazel", "build"))
        self.assertIn("//...", build.argv)
        self.assertIn(
            "--//runtime/src/iree/hal/driver/amdgpu:targets=gfx11-generic",
            build.argv,
        )

    def test_bazel_repository_device_compilation_requires_rocm(self):
        args = ci.parse_arguments(
            ["iree-bazel-repository-build", "--amdgpu-target", "gfx11-generic"]
        )
        with mock.patch.dict(ci.os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "requires HRX_ROCM_ROOT"):
                ci.steps_from_args(args)

    def test_shared_tool_commands_do_not_run_a_static_configuration(self):
        for command in (
            "iree-bazel-repository-shared-tools",
            "iree-bazel-cpu-shared-tools-asan",
        ):
            with self.subTest(command=command):
                configure, test = ci.steps_from_args(ci.parse_arguments([command]))
                self.assertEqual(configure.argv[2:4], ("bazel", "configure"))
                self.assertEqual(test.argv[2:4], ("bazel", "test"))
                self.assertIn("--//loom/config:shared_tools", test.argv)
                with self.assertRaisesRegex(ValueError, "--target"):
                    ci.steps_from_args(
                        ci.parse_arguments([command, "--target", "//example:test"])
                    )

    def test_bazel_repository_integration_rejects_partial_target_scope(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-repository-integration",
                "--target",
                "//runtime/...",
            ]
        )

        with self.assertRaisesRegex(ValueError, "repository-wide"):
            ci.steps_from_args(args)

    def test_cmake_repository_build_covers_supported_graph_with_prebuilts(self):
        args = ci.parse_arguments(["iree-cmake-repository-build"])

        steps = ci.steps_from_args(args)

        self.assertEqual(
            [step.name for step in steps],
            ["Configure CMake", "Build repository", "Test repository smoke"],
        )
        configure_step = steps[0]
        self.assertIn("-DLOOM_TARGET_DEFAULT=OFF", configure_step.argv)
        for _driver, define in ci.CMAKE_HAL_DRIVER_DEFINES:
            self.assertIn(f"-D{define}=ON", configure_step.argv)
        for _architecture, define in ci.CMAKE_LOOM_ARCH_DEFINES:
            self.assertIn(f"-D{define}=ON", configure_step.argv)
        for _artifact_format, define in ci.CMAKE_LOOM_EMIT_DEFINES:
            self.assertIn(f"-D{define}=ON", configure_step.argv)
        for _substrate, define in ci.CMAKE_LOOM_EXECUTE_DEFINES:
            self.assertIn(f"-D{define}=ON", configure_step.argv)
        for _importer, define in ci.CMAKE_LOOM_IMPORTER_DEFINES:
            self.assertIn(f"-D{define}=ON", configure_step.argv)
        for option in ci.AMDGPU_DEVICE_BINARY_PREBUILT_OPTIONS:
            self.assertIn(option, configure_step.argv)
        for option in ci.AMDGPU_DEVICE_BINARY_SOURCE_OPTIONS:
            self.assertNotIn(option, configure_step.argv)
        self.assertFalse(
            any(
                option.startswith("-DIREE_HAL_AMDGPU_TARGETS=")
                or option.startswith("-DLOOM_ARCH_AMDGPU_TARGETS=")
                for option in configure_step.argv
            )
        )
        build_step = steps[1]
        self.assertEqual(build_step.argv[-2:], ("build", "--parallel"))
        smoke_step = steps[2]
        self.assertTrue(
            any(
                ci_config.CTEST_RESOURCE_LABEL_EXCLUDE_REGEX in arg
                for arg in smoke_step.argv
            )
        )
        self.assertTrue(
            any(
                ci_config.CTEST_MANUAL_LABEL_EXCLUDE_REGEX in arg
                for arg in smoke_step.argv
            )
        )

    def test_cmake_configure_rejects_partial_loom_capability_selection(self):
        with self.assertRaisesRegex(ValueError, "requires architecture"):
            ci.cmake_configure_step(
                "iree-cmake-test",
                enabled_loom_architectures=("x86",),
            )

    def test_cmake_build_honors_standard_parallelism_environment(self):
        with mock.patch.dict(os.environ, {"CMAKE_BUILD_PARALLEL_LEVEL": "40"}):
            step = ci.cmake_build_step("iree-cmake-repository-build", "Build")

        self.assertEqual(step.argv[-3:], ("build", "--parallel", "40"))

    def test_amdgpu_dry_run_does_not_embed_machine_paths(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu",
                "--dry-run",
                "--target",
                "//runtime/...",
            ]
        )

        output = io.StringIO()
        with (
            mock.patch.dict(ci.os.environ, {}, clear=True),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(
                ci.run_steps(
                    ci.steps_from_args(args),
                    dry_run=True,
                    keep_going=False,
                    verbose=False,
                ),
                0,
            )

        text = output.getvalue()
        self.assertIn("dev.py bazel configure", text)
        self.assertNotIn("IREE_ROCM_PATH", text)
        self.assertNotIn("/opt/rocm", text)

    def test_amdgpu_target_rejected_by_non_amdgpu_command(self):
        args = ci.parse_arguments(["iree-bazel-cpu", "--amdgpu-target", "gfx1150"])

        with self.assertRaisesRegex(ValueError, "only supported for AMDGPU"):
            ci.steps_from_args(args)

    def test_amdgpu_hardware_selector_scopes_loom_targets(self):
        for platform, command in (
            ("linux", "iree-bazel-amdgpu"),
            ("win32", "iree-bazel-amdgpu-native"),
        ):
            with (
                self.subTest(platform=platform),
                mock.patch.object(ci.sys, "platform", platform),
                mock.patch.object(
                    ci.target_map,
                    "exact_targets_for_selection",
                    return_value=["TARGET_A", "TARGET_B"],
                ),
            ):
                steps = ci.steps_from_args(
                    ci.parse_arguments(
                        [command, "--amdgpu-target", "selected-hardware"]
                    )
                )
                test_step = next(step for step in steps if "test" in step.argv)
                self.assertIn(
                    "--//runtime/src/iree/hal/driver/amd/gpu:targets=selected-hardware",
                    test_step.argv,
                )
                self.assertIn(
                    "--//loom/config/arch/amdgpu:targets=selected-hardware",
                    test_step.argv,
                )

    def test_amdgpu_loom_importer_is_forwarded_to_configuration(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu",
                "--loom-importer",
                "cxx",
            ]
        )

        steps = ci.steps_from_args(args)

        configure_step = next(step for step in steps if step.name == "Configure Bazel")
        self.assertIn(
            "--//loom/config/import:enable=cxx",
            configure_step.argv,
        )

    def test_loom_importer_rejected_by_non_amdgpu_command(self):
        args = ci.parse_arguments(["iree-bazel-cpu", "--loom-importer", "cxx"])

        with self.assertRaisesRegex(ValueError, "only supported for AMDGPU Bazel"):
            ci.steps_from_args(args)

    def test_amdgpu_commands_preserve_device_visibility(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu",
                "--target",
                "//runtime/...",
            ]
        )

        device_visibility = {
            "ROCR_VISIBLE_DEVICES": "GPU-00112233",
            "HIP_VISIBLE_DEVICES": "0",
            "CUDA_VISIBLE_DEVICES": "1",
            "GPU_DEVICE_ORDINAL": "2",
            "HAL_VISIBLE_XDNA_DEVICES": "",
            "HAL_VISIBLE_RDNA_DEVICES": "1,0",
            "HAL_VISIBLE_CDNA_DEVICES": "0",
            "HAL_VISIBLE_AMDGPU_DEVICES": "2,1",
            "AMDF_CTS_GPU_NATIVE_IDENTITY": "linux_device:226:144",
        }
        with mock.patch.dict(ci.os.environ, device_visibility, clear=True):
            steps = ci.steps_from_args(args)
            cmake_steps = ci.steps_from_args(ci.parse_arguments(["iree-cmake-amdgpu"]))

        amdgpu_test = next(step for step in steps if step.name == "Test IREE / AMDGPU")
        for name in ci.AMD_DEVICE_VISIBILITY_ENV_VARS:
            self.assertIn(
                f"--test_env={name}={device_visibility[name]}",
                amdgpu_test.argv,
            )
        cmake_tests = [
            step for step in cmake_steps if step.name.startswith("Test IREE CMake")
        ]
        self.assertEqual(len(cmake_tests), 2)
        for step in cmake_tests:
            for name in ci.AMD_DEVICE_VISIBILITY_ENV_VARS:
                self.assertIn((name, device_visibility[name]), step.env)

    def test_native_amd_profiles_preserve_independent_and_empty_visibility(self):
        selection = {
            "HAL_VISIBLE_XDNA_DEVICES": "",
            "HAL_VISIBLE_RDNA_DEVICES": "1,0",
            "HAL_VISIBLE_CDNA_DEVICES": "0",
            "HAL_VISIBLE_AMDGPU_DEVICES": "2,1",
            "AMDF_CTS_GPU_NATIVE_IDENTITY": "windows_adapter:123456:0",
        }
        profiles = (
            ("linux", ["iree-bazel-xdna-asan"]),
            ("linux", ["iree-bazel-amd-client-asan"]),
            ("win32", ["iree-bazel-amd-client-asan"]),
            (
                "win32",
                ["iree-bazel-amdgpu-native-asan", "--amdgpu-target", "gfx1201"],
            ),
        )
        for platform, arguments in profiles:
            with self.subTest(platform=platform, arguments=arguments):
                with (
                    mock.patch.object(ci.sys, "platform", platform),
                    mock.patch.dict(ci.os.environ, selection, clear=True),
                    mock.patch.object(ci.shutil, "which", return_value=None),
                ):
                    steps = ci.steps_from_args(ci.parse_arguments(arguments))
                tests = [step for step in steps if "test" in step.argv]
                self.assertTrue(tests)
                for step in tests:
                    for name, value in selection.items():
                        self.assertIn(f"--test_env={name}={value}", step.argv)
                    self.assertFalse(
                        any(
                            arg.startswith("--test_env=IREE_HAL_AMDGPU_LIBHSA_PATH=")
                            for arg in step.argv
                        )
                    )

    def test_amdgpu_bazel_device_toolchain_uses_fetched_rocm_root(self):
        args = ci.parse_arguments(["iree-bazel-amdgpu"])

        with mock.patch.dict(
            ci.os.environ,
            {"HRX_ROCM_ROOT": "/tmp/rocm-root"},
            clear=True,
        ):
            steps = ci.steps_from_args(args)

        test_step = next(step for step in steps if step.name == "Test IREE / AMDGPU")
        self.assertIn("--repo_env=IREE_ROCM_PATH=/tmp/rocm-root", test_step.argv)

    def test_amdgpu_loom_target_scope_tests_loom(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu",
                "--target",
                "//loom/...",
            ]
        )

        steps = ci.steps_from_args(args)
        test_step = next(step for step in steps if step.name == "Test IREE / AMDGPU")

        self.assertFalse(any("bazel build" in step.command_line() for step in steps))
        self.assertIn("//loom/...", test_step.argv)
        self.assertNotIn("//runtime/...", test_step.argv)

    def test_bazel_loom_amdgpu_command_runs_compile_coverage_without_driver(self):
        args = ci.parse_arguments(["iree-bazel-loom-amdgpu"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertEqual(len(steps), 2)
        test_step = steps[1]
        self.assertEqual(test_step.name, "Test Loom AMDGPU compile coverage")
        self.assertIn(
            "--test_tag_filters=" + ",".join(ci_config.CPU_RESOURCE_TAG_EXCLUDES),
            test_step.argv,
        )
        for target in ci_config.LOOM_AMDGPU_BAZEL_COMPILE_TEST_TARGETS:
            self.assertIn(target, test_step.argv)
        self.assertNotIn(
            "//loom/src/loom/target/emit/native/amdgpu:hsaco_hsa_test",
            test_step.argv,
        )
        self.assertFalse(
            any("-DIREE_HAL_DRIVER_AMDGPU=ON" in line for line in command_lines)
        )

    def test_cpu_sanitizer_aggregate_configures_once_and_preserves_scope(self):
        with mock.patch.object(
            ci_config,
            "BAZEL_CPU_SANITIZER_TEST_TAG_FILTERS",
            {"first": ("first-subject",), "second": ("second-subject",)},
        ):
            steps = ci.cpu_sanitizer_steps(("//selected/...",))

        self.assertEqual(len(steps), 3)
        self.assertEqual(steps[0].argv[:4], ("python3", "dev.py", "bazel", "configure"))
        for step, config in zip(steps[1:], ("first", "second")):
            self.assertEqual(step.argv[:4], ("python3", "dev.py", "bazel", "test"))
            self.assertIn(f"--config={config}", step.argv)
            self.assertIn("//selected/...", step.argv)
            self.assertLess(
                step.argv.index("--config=" + config), step.argv.index("--")
            )
            self.assertGreater(step.argv.index("//selected/..."), step.argv.index("--"))

    def test_bazel_cpu_single_sanitizer_command_runs_one_configuration(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-cpu-asan",
                "--target",
                "//runtime/...",
            ]
        )

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertEqual(steps[0].argv[:4], ("python3", "dev.py", "bazel", "configure"))
        self.assertTrue(
            any(
                step.argv[:4] == ("python3", "dev.py", "bazel", "test")
                and "--config=asan" in step.argv
                and step.argv.index("--config=asan") < step.argv.index("--")
                for step in steps
            )
        )
        self.assertFalse(any("--config=ubsan" in line for line in command_lines))
        self.assertFalse(any("--config=tsan" in line for line in command_lines))
        self.assertFalse(any("--config=msan" in line for line in command_lines))
        self.assertFalse(
            any("--//loom/config:shared_tools" in step.argv for step in steps)
        )
        self.assertFalse(
            any(step.name == "Build repository fuzz targets" for step in steps)
        )

    def test_bazel_cpu_asan_platform_lane_builds_fuzz_targets(self):
        args = ci.parse_arguments(["iree-bazel-cpu-asan"])

        with mock.patch.dict(
            ci.os.environ, {"IREE_CI_PLATFORM_COVERAGE": "1"}, clear=True
        ):
            steps = ci.steps_from_args(args)

        fuzz_step = next(
            step for step in steps if step.name == "Build repository fuzz targets"
        )
        self.assertEqual(
            fuzz_step.argv[:5],
            ("python3", "dev.py", "bazel", "fuzz", "--compile-only"),
        )
        self.assertEqual(fuzz_step.argv[-1], "//...")
        self.assertIn("--//loom/config/import:enable=cxx", fuzz_step.argv)
        self.assertIn(
            "--//loom/config/serve:image_codecs=png,jpeg,gif,webp", fuzz_step.argv
        )
        self.assertIn("--//loom/config/serve:audio_codecs=wav,mp3,flac", fuzz_step.argv)

    def test_focused_cpu_sanitizer_builds_only_its_selected_test_closures(self):
        with (
            mock.patch.object(
                ci_config,
                "BAZEL_CPU_SANITIZER_TEST_TAG_FILTERS",
                {"focused": ("instrumented-subject",)},
            ),
            mock.patch.object(
                ci_config, "CPU_SANITIZERS_XFAIL_TARGETS", ("-//other:failure",)
            ),
            mock.patch.dict(os.environ, {"IREE_CI_PLATFORM_COVERAGE": "1"}, clear=True),
        ):
            steps = ci.cpu_config_steps(("//selected/...",), "focused")

        # A broad platform build would analyze unqualified native dependencies
        # even though the test invocation filters them out.
        self.assertEqual(len(steps), 1)
        step = steps[0]
        self.assertEqual(step.argv[:4], ("python3", "dev.py", "bazel", "test"))
        self.assertIn("--build_tests_only", step.argv)
        self.assertIn("--config=focused", step.argv)
        self.assertIn("//selected/...", step.argv)
        self.assertNotIn("-//other:failure", step.argv)
        self.assertFalse(
            any(arg.startswith("--test_lang_filters=") for arg in step.argv)
        )
        filters = next(
            arg for arg in step.argv if arg.startswith("--test_tag_filters=")
        )
        self.assertIn("instrumented-subject", filters.split("=", 1)[1].split(","))
        self.assertIsNotNone(step.requirement_audit)

    def test_cpu_sanitizer_rejects_unknown_configuration(self):
        with mock.patch.object(ci_config, "BAZEL_CPU_SANITIZER_TEST_TAG_FILTERS", {}):
            with self.assertRaisesRegex(ValueError, "unknown Bazel sanitizer config"):
                ci.cpu_config_steps(("//selected/...",), "unknown")

    def test_amdgpu_command_tests_one_semantic_graph(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu",
                "--target",
                "//runtime/...",
                "--target",
                "//loom/...",
            ]
        )

        steps = ci.steps_from_args(args)
        self.assertEqual(
            [step.name for step in steps],
            ["Configure Bazel", "Test IREE / AMDGPU"],
        )
        test_step = steps[1]
        for target in ("//runtime/...", "//loom/..."):
            self.assertIn(target, test_step.argv)
        for target in ci_config.AMDGPU_BAZEL_TARGET_EXCLUDES:
            self.assertIn(target, test_step.argv)

    def test_bazel_amdgpu_single_sanitizer_command_runs_one_configuration(self):
        args = ci.parse_arguments(
            [
                "iree-bazel-amdgpu-tsan",
                "--target",
                "//runtime/...",
                "--target",
                "//loom/...",
            ]
        )

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertEqual(
            [step.name for step in steps],
            [
                "Configure Bazel",
                "Test IREE / AMDGPU / TSAN",
            ],
        )
        tsan_test = steps[1]
        self.assertIn("--config=tsan", tsan_test.argv)
        for target in ("//runtime/...", "//loom/..."):
            self.assertIn(target, tsan_test.argv)
        self.assertTrue(
            any(
                arg.startswith("--test_env=TSAN_OPTIONS=suppressions=")
                for arg in tsan_test.argv
            )
        )
        self.assertFalse(any("--config=asan" in line for line in command_lines))
        self.assertFalse(any("--config=ubsan" in line for line in command_lines))
        self.assertFalse(any("--config=msan" in line for line in command_lines))

    def test_bazel_amdgpu_asan_and_ubsan_keep_all_resource_tests(self):
        for sanitizer in ("asan", "ubsan"):
            with self.subTest(sanitizer=sanitizer):
                args = ci.parse_arguments(
                    [
                        f"iree-bazel-amdgpu-{sanitizer}",
                        "--target",
                        "//loom/...",
                    ]
                )

                steps = ci.steps_from_args(args)
                test_step = next(
                    step
                    for step in steps
                    if step.name == f"Test IREE / AMDGPU / {sanitizer.upper()}"
                )
                self.assertFalse(
                    any(
                        ci_config.HOST_TSAN_INCOMPATIBLE_TEST_LABEL in arg
                        for arg in test_step.argv
                    )
                )

    def test_vulkan_command_tests_one_semantic_graph(self):
        args = ci.parse_arguments(["iree-bazel-vulkan"])

        steps = ci.steps_from_args(args)

        self.assertEqual(
            [step.name for step in steps],
            ["Configure Bazel", "Test IREE / Vulkan"],
        )
        test_step = steps[1]
        for target in ("//runtime/...", "//loom/..."):
            self.assertIn(target, test_step.argv)
        for target in ci_config.VULKAN_BAZEL_TARGET_EXCLUDES:
            self.assertIn(target, test_step.argv)
        self.assertFalse(
            any(
                arg.startswith("-//runtime/src/iree/hal/driver/vulkan")
                for arg in test_step.argv
            )
        )
        for xfail_target in ci_config.VULKAN_XFAIL_TARGETS:
            self.assertIn(xfail_target, test_step.argv)

    def test_vulkan_linux_commands_require_namespace_sandboxing(self):
        for host_platform in ("linux", "win32", "darwin"):
            with self.subTest(platform=host_platform):
                with mock.patch.object(ci.sys, "platform", host_platform):
                    steps = ci.steps_from_args(
                        ci.parse_arguments(["iree-bazel-vulkan"])
                    )
                for step in steps[1:]:
                    self.assertEqual(
                        "--spawn_strategy=linux-sandbox" in step.argv,
                        host_platform == "linux",
                    )

    def test_vulkan_commands_preserve_explicit_mesa_device_selection(self):
        selection = {
            "DRI_PRIME": "1!",
            "AMDF_CTS_GPU_NATIVE_IDENTITY": "linux_device:226:144",
        }
        with mock.patch.dict(os.environ, selection):
            bazel_steps = ci.steps_from_args(ci.parse_arguments(["iree-bazel-vulkan"]))
            cmake_steps = ci.steps_from_args(ci.parse_arguments(["iree-cmake-vulkan"]))

        bazel_test_step = next(
            step for step in bazel_steps if step.name == "Test IREE / Vulkan"
        )
        for name, value in selection.items():
            self.assertIn(f"--test_env={name}={value}", bazel_test_step.argv)

        cmake_test_steps = [
            step for step in cmake_steps if step.name.startswith("Test IREE CMake")
        ]
        self.assertGreater(len(cmake_test_steps), 0)
        for step in cmake_test_steps:
            for item in selection.items():
                self.assertIn(item, step.env)

    def test_bazel_gpu_command_surface_omits_nonexecuting_lanes(self):
        for command in (
            "iree-bazel-amdgpu-msan",
            "iree-bazel-amdgpu-sanitizers",
            "iree-bazel-vulkan-asan",
            "iree-bazel-vulkan-msan",
            "iree-bazel-vulkan-tsan",
            "iree-bazel-vulkan-ubsan",
            "iree-bazel-vulkan-sanitizers",
        ):
            with self.subTest(command=command):
                self.assertNotIn(command, ci.BAZEL_COMMANDS)

        for command in (
            "iree-bazel-amdgpu-asan",
            "iree-bazel-amdgpu-tsan",
            "iree-bazel-amdgpu-ubsan",
            "iree-bazel-vulkan",
        ):
            with self.subTest(command=command):
                self.assertIn(command, ci.BAZEL_COMMANDS)

    def test_amdgpu_container_jobs_initialize_git_home(self):
        for path, job_name in (
            (".github/workflows/ci_iree.yml", "bazel_linux"),
            (".github/workflows/ci_iree.yml", "cmake_linux"),
            (
                ".github/workflows/test_core_linux_gpu_source.yml",
                "test_core_linux_gpu_source",
            ),
        ):
            with self.subTest(path=path):
                block = self.workflow_job_block(path, job_name)
                mkdir_command = 'mkdir -p "$HOME"'
                git_config_command = 'git config --global --add safe.directory "$PWD"'
                self.assertIn(mkdir_command, block)
                self.assertIn(git_config_command, block)
                self.assertLess(
                    block.index(mkdir_command), block.index(git_config_command)
                )

    def test_amdgpu_container_jobs_preserve_runner_device_selection(self):
        for path, job_name in (
            (".github/workflows/ci_iree.yml", "bazel_linux"),
            (".github/workflows/ci_libhrx.yml", "cmake_linux_gfx942"),
            (
                ".github/workflows/test_core_linux_gpu_source.yml",
                "test_core_linux_gpu_source",
            ),
        ):
            with self.subTest(path=path, job_name=job_name):
                block = self.workflow_job_block(path, job_name)
                self.assertIn(
                    "--env-file /etc/podinfo/gha-gpu-isolation-settings", block
                )
                for variable_name in (
                    "ROCR_VISIBLE_DEVICES",
                    "HIP_VISIBLE_DEVICES",
                    "CUDA_VISIBLE_DEVICES",
                    "GPU_DEVICE_ORDINAL",
                ):
                    self.assertNotIn(f"{variable_name}:", block)

    def test_fetch_toolchain_uses_owned_ci_entry_point(self):
        script = Path(".github/scripts/fetch_rocm_toolchain.sh").read_text()
        self.assertIn(
            '"${HRX_PYTHON}" build_tools/ci/ci_core_linux.py fetch-rocm', script
        )
        self.assertNotIn("build_tools/ci_core_", script)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux CI toolchain setup")
    def test_fetch_toolchain_versions_root_and_rebases_rocm_paths(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output_dir = root / "output"
            rocm_root = output_dir / "rocm-root"
            github_env = root / "github-env"
            github_path = root / "github-path"
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                """#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "-m" && "$2" == "venv" ]]; then
  mkdir -p "$3/bin"
  cp "$0" "$3/bin/python3"
elif [[ "$1" == "-m" && "$2" == "pip" ]]; then
  exit 0
elif [[ "$1" == "build_tools/ci/ci_core_linux.py" && "$2" == "fetch-rocm" ]]; then
  mkdir -p "${HRX_ROCM_ROOT}/lib/llvm/bin"
  printf '%s\n' \
    '{"artifact_identity":"nightly-123456-linux-release-core"}' \
    >"${HRX_ROCM_ROOT}/.hrx-rocm-artifacts.json"
elif [[ "$1" == "-c" ]]; then
  exec __REAL_PYTHON__ "$@"
else
  exit 1
fi
""".replace("__REAL_PYTHON__", shlex.quote(sys.executable))
            )
            fake_python.chmod(0o755)

            env = os.environ.copy()
            env.update(
                {
                    "AR": "ar",
                    "CC": os.fspath(rocm_root / "lib/llvm/bin/clang"),
                    "CXX": os.fspath(rocm_root / "lib/llvm/bin/clang++"),
                    "GITHUB_ENV": os.fspath(github_env),
                    "GITHUB_PATH": os.fspath(github_path),
                    "HRX_OUTPUT_DIR": os.fspath(output_dir),
                    "HRX_PYTHON": os.fspath(output_dir / "python/bin/python3"),
                    "HRX_ROCM_ROOT": os.fspath(rocm_root),
                    "IREE_CLANG_TIDY_LLVM_ROOT": os.fspath(rocm_root / "lib/llvm"),
                    "IREE_ROCM_PATH": os.fspath(rocm_root),
                    "PATH": f"{fake_bin}:{env['PATH']}",
                }
            )
            subprocess.run(
                ["bash", ".github/scripts/fetch_rocm_toolchain.sh"],
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            versioned_rocm_root = Path(f"{rocm_root}-nightly-123456-linux-release-core")
            exported_env = dict(
                line.split("=", 1) for line in github_env.read_text().splitlines()
            )
            self.assertFalse(rocm_root.exists())
            self.assertTrue(versioned_rocm_root.is_dir())
            self.assertEqual(exported_env["HRX_ROCM_ROOT"], str(versioned_rocm_root))
            self.assertEqual(
                exported_env["CC"], str(versioned_rocm_root / "lib/llvm/bin/clang")
            )
            self.assertEqual(
                exported_env["CXX"],
                str(versioned_rocm_root / "lib/llvm/bin/clang++"),
            )
            self.assertNotIn("AR", exported_env)
            self.assertEqual(
                exported_env["IREE_CLANG_TIDY_LLVM_ROOT"],
                str(versioned_rocm_root / "lib/llvm"),
            )
            self.assertEqual(exported_env["IREE_ROCM_PATH"], str(versioned_rocm_root))
            self.assertEqual(
                exported_env["IREE_HAL_AMDGPU_LIBHSA_PATH"],
                str(versioned_rocm_root / "lib/libhsa-runtime64.so.1"),
            )

    def test_loom_docs_use_the_qualified_sdk_and_deploy_only_main(self):
        build_text = Path(".github/workflows/build_loom_sdk_artifacts.yml").read_text()
        sdk_text = Path(".github/workflows/build_loom_sdk.yml").read_text()
        ci_text = Path(".github/workflows/ci_iree.yml").read_text()
        build_block = self.workflow_job_block(
            ".github/workflows/build_loom_sdk_artifacts.yml", "build"
        )
        deploy_block = self.workflow_job_block(
            ".github/workflows/ci_iree.yml", "docs_deploy"
        )

        for target in (
            "//loom/binding/c/example:jit_task_pool",
            "//loom/binding/c/example:link_modules",
            "//loom/binding/c/example:product_frontier",
        ):
            self.assertIn(target, build_block)
        self.assertIn("build_tools/ci/change_scope.py", build_block)
        self.assertIn("python3 dev.py setup --docs", build_block)
        self.assertIn("pip install --no-deps wheel-artifact/*.whl", build_block)
        self.assertIn("LOOM_DOCS_C_EXAMPLE_DIR", build_block)
        self.assertIn("LOOM_DOCS_TOOL_DIR", build_block)
        self.assertIn("loom_docs.highlight_test", build_block)
        self.assertIn(
            "--site-dir build/loom-pages/loom",
            build_block,
        )
        self.assertIn(
            "test -f build/loom-pages/loom/reference/dialects/index.html",
            build_block,
        )
        self.assertIn(
            "test -f build/loom-pages/loom/reference/c-api/generated/index.html",
            build_block,
        )
        self.assertRegex(
            build_block,
            r"uses: actions/upload-pages-artifact@[0-9a-f]{40} # v\d+\.\d+\.\d+",
        )
        self.assertIn("path: build/loom-pages", build_block)
        self.assertNotIn("pages: write", build_block)
        self.assertNotIn("id-token: write", build_block)
        self.assertNotIn("actions/deploy-pages", build_block)
        self.assertIn("value: ${{ jobs.build.outputs.documentation }}", build_text)
        self.assertIn("value: ${{ jobs.build.outputs.documentation }}", sdk_text)

        self.assertIn("github.ref == 'refs/heads/main'", deploy_block)
        self.assertIn("needs.sdk.outputs.documentation == 'true'", deploy_block)
        self.assertIn("needs: sdk", deploy_block)
        self.assertIn("pages: write", deploy_block)
        self.assertIn("id-token: write", deploy_block)
        self.assertIn("name: github-pages", deploy_block)
        self.assertIn("url: ${{ steps.deployment.outputs.page_url }}", deploy_block)
        self.assertIn("group: github-pages", deploy_block)
        self.assertIn("cancel-in-progress: false", deploy_block)
        self.assertIn("id: deployment", deploy_block)
        self.assertRegex(
            deploy_block,
            r"uses: actions/deploy-pages@[0-9a-f]{40} # v\d+\.\d+\.\d+",
        )
        self.assertEqual(ci_text.count("pages: write"), 1)
        self.assertEqual(ci_text.count("id-token: write"), 1)
        self.assertEqual(ci_text.count("actions/deploy-pages@"), 1)

    def test_cmake_cpu_sanitizer_command_uses_cmake_build_dir_and_xfails(self):
        args = ci.parse_arguments(["iree-cmake-cpu-ubsan"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(
            any(
                self.uses_cmake_build_dir(step, "iree-cmake-cpu-ubsan")
                for step in steps
            )
        )
        self.assertTrue(any("-DIREE_ENABLE_UBSAN=ON" in line for line in command_lines))
        self.assertFalse(any(" cmake build " in line for line in command_lines))
        test_steps = [step for step in steps if step.name.startswith("Test IREE")]
        self.assertTrue(
            any(
                any(
                    ci_config.CPU_SANITIZERS_CTEST_EXCLUDE_REGEX in arg
                    for arg in step.argv
                )
                for step in test_steps
            )
        )
        self.assertTrue(
            any(
                any(
                    ci_config.NON_CPU_HAL_DRIVER_CTEST_REGEX in arg for arg in step.argv
                )
                for step in test_steps
            )
        )
        self.assertTrue(
            any(
                any(
                    ci_config.CTEST_RESOURCE_LABEL_EXCLUDE_REGEX in arg
                    for arg in step.argv
                )
                for step in test_steps
            )
        )
        self.assertTrue(
            any(
                any(
                    ci_config.CTEST_MANUAL_LABEL_EXCLUDE_REGEX in arg
                    for arg in step.argv
                )
                for step in test_steps
            )
        )

    def test_cmake_link_modes_have_independent_configuration_trees(self):
        for command, shared in (
            ("iree-cmake-cpu", False),
            ("iree-cmake-shared-tools", True),
        ):
            with self.subTest(command=command):
                steps = ci.steps_from_args(ci.parse_arguments([command]))
                configure_steps = [step for step in steps if "configure" in step.argv]
                self.assertEqual(len(configure_steps), 1)
                self.assertIn("--fresh", configure_steps[0].argv)
                self.assertEqual(
                    "-DLOOM_LINK_SHARED_TOOLS=ON" in configure_steps[0].argv, shared
                )
                for step in steps:
                    self.assertTrue(self.uses_cmake_build_dir(step, command))

    def test_cmake_cpu_sanitizers_command_runs_each_configuration(self):
        args = ci.parse_arguments(["iree-cmake-cpu-sanitizers"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(any("iree-cmake-cpu-asan" in line for line in command_lines))
        self.assertTrue(any("iree-cmake-cpu-ubsan" in line for line in command_lines))
        self.assertTrue(any("iree-cmake-cpu-tsan" in line for line in command_lines))
        self.assertFalse(any("iree-cmake-cpu-msan" in line for line in command_lines))
        tsan_options = [
            value for step in steps for key, value in step.env if key == "TSAN_OPTIONS"
        ]
        self.assertNotEqual(tsan_options, [])
        for value in tsan_options:
            path = Path(value.removeprefix("suppressions="))
            self.assertTrue(path.is_absolute())
            self.assertTrue(path.is_file())

    def test_cmake_sanitizer_smoke_builds_selected_test_closures(self):
        args = ci.parse_arguments(["iree-cmake-sanitizer-smoke"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        for sanitizer in ("asan", "ubsan", "tsan"):
            self.assertTrue(
                any(
                    self.uses_cmake_build_dir(
                        step, f"iree-cmake-sanitizer-smoke-{sanitizer}"
                    )
                    for step in steps
                )
            )
            self.assertTrue(
                any(
                    f"-DIREE_ENABLE_{sanitizer.upper()}=ON" in line
                    for line in command_lines
                )
            )
            self.assertTrue(
                any(
                    f"iree-cmake-sanitizer-smoke-{sanitizer}" in line
                    and "-DIREE_BUILD_TESTS=ON" in line
                    and "-DIREE_BUILD_BENCHMARKS=ON" in line
                    for line in command_lines
                )
            )
        self.assertFalse(any("-DIREE_ENABLE_MSAN=ON" in line for line in command_lines))

        test_steps = [step for step in steps if step.name.startswith("Test IREE")]
        self.assertEqual(len(test_steps), 3)
        for regex in ci_config.CMAKE_SANITIZER_SMOKE_CTEST_REGEXES:
            self.assertTrue(
                any(regex in arg for step in test_steps for arg in step.argv)
            )
        self.assertFalse(
            any(
                ci_config.CPU_SANITIZERS_CTEST_EXCLUDE_REGEX in arg
                for step in test_steps
                for arg in step.argv
            )
        )

    def test_cmake_amdgpu_command_scopes_tests_to_amdgpu(self):
        args = ci.parse_arguments(["iree-cmake-amdgpu", "--amdgpu-target", "gfx1150"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(
            any(
                "-DIREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=source" in line
                for line in command_lines
            )
        )
        self.assertTrue(
            any(
                "-DIREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm" in line
                for line in command_lines
            )
        )
        build_steps = [step for step in steps if step.name.startswith("Build IREE")]
        self.assertEqual(build_steps, [])
        self.assertTrue(
            any(
                ci_config.AMDGPU_CTEST_RESOURCE_LABEL_REGEX in step.argv
                for step in steps
            )
        )
        package_test = next(
            step
            for step in steps
            if step.name == "Test IREE CMake AMDGPU package tests"
        )
        self.assertEqual(
            self.ctest_exclude_regexes(package_test),
            (
                [ci_config.AMDGPU_CTEST_EXCLUDE_REGEX]
                if ci_config.AMDGPU_CTEST_EXCLUDE_REGEX
                else []
            ),
        )

    def test_cmake_amdgpu_device_binary_source_build_uses_fetched_rocm_root(self):
        args = ci.parse_arguments(["iree-cmake-amdgpu"])

        with mock.patch.dict(
            ci.os.environ,
            {"HRX_ROCM_ROOT": "/tmp/rocm-root"},
            clear=True,
        ):
            steps = ci.steps_from_args(args)

        configure_step = next(step for step in steps if step.name == "Configure CMake")
        self.assertIn(
            "-DIREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH=/tmp/rocm-root",
            configure_step.argv,
        )

    def test_cmake_amdgpu_scopes_loom_to_hardware_selector(self):
        steps = ci.steps_from_args(
            ci.parse_arguments(
                ["iree-cmake-amdgpu", "--amdgpu-target", "selected-hardware"]
            )
        )
        configure_step = steps[0]
        self.assertIn(
            "-DIREE_HAL_AMD_GPU_TARGETS=selected-hardware", configure_step.argv
        )
        self.assertIn(
            "-DLOOM_ARCH_AMDGPU_TARGETS=selected-hardware", configure_step.argv
        )
        self.assertIn(
            "-DIREE_HAL_AMD_GPU_TARGETS=selected-hardware", configure_step.argv
        )

    def test_cmake_amdgpu_msan_builds_driver_targets_without_test_deps(self):
        args = ci.parse_arguments(["iree-cmake-amdgpu-msan"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(any("-DIREE_ENABLE_MSAN=ON" in line for line in command_lines))
        self.assertTrue(any("-DIREE_BUILD_TESTS=OFF" in line for line in command_lines))
        self.assertTrue(
            any("-DIREE_BUILD_BENCHMARKS=OFF" in line for line in command_lines)
        )
        self.assertTrue(any(" cmake build " in line for line in command_lines))
        self.assertFalse(any("Test IREE CMake AMDGPU" in step.name for step in steps))

    def test_cmake_loom_amdgpu_command_runs_compile_coverage_without_driver(self):
        args = ci.parse_arguments(["iree-cmake-loom-amdgpu"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(
            any(
                self.uses_cmake_build_dir(step, "iree-cmake-loom-amdgpu")
                for step in steps
            )
        )
        self.assertTrue(
            any("-DIREE_HAL_DRIVER_AMDGPU=OFF" in line for line in command_lines)
        )
        self.assertFalse(
            any("-DIREE_HAL_AMDGPU_TARGETS=" in line for line in command_lines)
        )
        test_step = next(
            step
            for step in steps
            if step.name == "Test Loom CMake AMDGPU compile coverage"
        )
        for regex in ci_config.LOOM_AMDGPU_CMAKE_COMPILE_CTEST_REGEXES:
            self.assertTrue(any(regex in arg for arg in test_step.argv))

    def test_cmake_vulkan_command_scopes_tests_to_vulkan(self):
        args = ci.parse_arguments(["iree-cmake-vulkan"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(
            any("-DIREE_HAL_DRIVER_VULKAN=ON" in line for line in command_lines)
        )
        self.assertTrue(
            any("-DIREE_HAL_DRIVER_AMDGPU=OFF" in line for line in command_lines)
        )
        build_steps = [step for step in steps if step.name.startswith("Build IREE")]
        self.assertEqual(build_steps, [])
        package_test = next(
            step
            for step in steps
            if step.name == "Test IREE CMake Vulkan package tests"
        )
        self.assertTrue(
            any(ci_config.VULKAN_CTEST_REGEX in arg for arg in package_test.argv)
        )
        resource_test = next(
            step
            for step in steps
            if step.name == "Test IREE CMake Vulkan resource tests"
        )
        self.assertIn(ci_config.VULKAN_CTEST_RESOURCE_LABEL_REGEX, resource_test.argv)
        self.assertIn(ci_config.VULKAN_CTEST_REGEX, resource_test.argv)
        self.assertFalse(
            any("emit_spirv_vulkan_test" in arg for arg in resource_test.argv)
        )
        self.assertFalse(any("-fuse-ld=lld" in line for line in command_lines))

    def test_cmake_vulkan_sanitizers_use_generic_clang_configs(self):
        args = ci.parse_arguments(["iree-cmake-vulkan-sanitizers"])

        steps = ci.steps_from_args(args)
        command_lines = [step.command_line() for step in steps]

        self.assertTrue(any("iree-cmake-vulkan-asan" in line for line in command_lines))
        self.assertTrue(any("iree-cmake-vulkan-tsan" in line for line in command_lines))
        self.assertTrue(
            any("iree-cmake-vulkan-ubsan" in line for line in command_lines)
        )
        self.assertFalse(
            any("iree-cmake-vulkan-msan" in line for line in command_lines)
        )

    def test_cmake_command_rejects_bazel_targets(self):
        args = ci.parse_arguments(["iree-cmake-cpu", "--target", "//runtime/..."])

        with self.assertRaisesRegex(ValueError, "--target"):
            ci.steps_from_args(args)


if __name__ == "__main__":
    unittest.main()
