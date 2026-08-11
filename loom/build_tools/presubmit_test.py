# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import importlib.util
import io
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


def load_presubmit_module():
    presubmit_path = Path(__file__).with_name("presubmit.py")
    spec = importlib.util.spec_from_file_location("loom_presubmit", presubmit_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {presubmit_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LoomPresubmitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.presubmit = load_presubmit_module()

    def test_bazel_tests_exclude_runtime_resource_requirements(self):
        command = self.presubmit.bazel_test_command()

        self.assertEqual(command[:3], ["bazel", "test", "--config=presubmit"])
        self.assertEqual(command[-1], "//loom/...")
        self.assertIn(
            "--//loom/config/target:enable=amdgpu,llvmir,spirv,x86",
            command,
        )

        tag_filter = next(
            arg for arg in command if arg.startswith("--test_tag_filters=")
        )
        self.assertIn("-iree-run-requirement=runtime.resource.amd_gpu", tag_filter)
        self.assertIn(
            "-iree-run-requirement=runtime.resource.vulkan_device", tag_filter
        )
        self.assertNotIn("loom.resource", tag_filter)

    def test_cmake_tests_exclude_runtime_resource_labels(self):
        self.assertEqual(
            self.presubmit.CTEST_RESOURCE_LABEL_EXCLUDE_REGEX,
            "runtime-resource=",
        )

    def test_generated_artifact_check_is_read_only(self):
        result = types.SimpleNamespace(ok=True, changed_paths=())
        with (
            mock.patch.object(
                self.presubmit.checked_in_artifacts,
                "maintain_checked_in_artifacts",
                return_value=result,
            ) as maintain_checked_in_artifacts,
            mock.patch.object(
                self.presubmit.project_presubmit, "stage_changed_paths"
            ) as stage_changed_paths,
        ):
            self.assertTrue(
                self.presubmit.run_generated_artifact_maintenance(fix=False)
            )

        maintain_checked_in_artifacts.assert_called_once_with("check")
        stage_changed_paths.assert_not_called()

    def test_generated_artifact_fix_stages_exact_changed_paths(self):
        changed_paths = (
            "loom/py/loom/dialect/__init__.py",
            "loom/src/loom/target/arch/amdgpu/target_config.inl",
        )
        result = types.SimpleNamespace(ok=True, changed_paths=changed_paths)
        with (
            mock.patch.object(
                self.presubmit.checked_in_artifacts,
                "maintain_checked_in_artifacts",
                return_value=result,
            ) as maintain_checked_in_artifacts,
            mock.patch.object(
                self.presubmit.project_presubmit,
                "stage_changed_paths",
                return_value=True,
            ) as stage_changed_paths,
        ):
            self.assertTrue(self.presubmit.run_generated_artifact_maintenance(fix=True))

        maintain_checked_in_artifacts.assert_called_once_with("update")
        stage_changed_paths.assert_called_once_with(
            self.presubmit.PROJECT_NAME,
            self.presubmit.REPO_ROOT,
            changed_paths,
        )

    def test_generated_artifact_fix_passes_selected_paths(self):
        selected_paths = ["loom/py/loom/dialect/__init__.py"]
        result = types.SimpleNamespace(ok=True, changed_paths=())
        with (
            mock.patch.object(
                self.presubmit, "selected_files", return_value=selected_paths
            ),
            mock.patch.object(
                self.presubmit.checked_in_artifacts,
                "maintain_checked_in_artifacts",
                return_value=result,
            ) as maintain_checked_in_artifacts,
            mock.patch.object(
                self.presubmit.project_presubmit, "stage_changed_paths"
            ) as stage_changed_paths,
        ):
            self.assertTrue(
                self.presubmit.run_generated_artifact_maintenance(
                    fix=True, files_from="paths.txt"
                )
            )

        maintain_checked_in_artifacts.assert_called_once_with(
            "update", selected_paths=selected_paths
        )
        stage_changed_paths.assert_called_once_with(
            self.presubmit.PROJECT_NAME,
            self.presubmit.REPO_ROOT,
            (),
        )

    def test_generated_artifact_failure_does_not_stage_partial_updates(self):
        result = types.SimpleNamespace(
            ok=False,
            changed_paths=("loom/py/loom/dialect/__init__.py",),
        )
        with (
            mock.patch.object(
                self.presubmit.checked_in_artifacts,
                "maintain_checked_in_artifacts",
                return_value=result,
            ),
            mock.patch.object(
                self.presubmit.project_presubmit, "stage_changed_paths"
            ) as stage_changed_paths,
        ):
            self.assertFalse(
                self.presubmit.run_generated_artifact_maintenance(fix=True)
            )

        stage_changed_paths.assert_not_called()

    def test_generated_artifact_staging_failure_fails_maintenance(self):
        result = types.SimpleNamespace(
            ok=True,
            changed_paths=("loom/py/loom/dialect/__init__.py",),
        )
        with (
            mock.patch.object(
                self.presubmit.checked_in_artifacts,
                "maintain_checked_in_artifacts",
                return_value=result,
            ),
            mock.patch.object(
                self.presubmit.project_presubmit,
                "stage_changed_paths",
                return_value=False,
            ),
        ):
            self.assertFalse(
                self.presubmit.run_generated_artifact_maintenance(fix=True)
            )

    def test_format_source_classification_is_strict(self):
        self.assertTrue(
            self.presubmit.is_format_source_path(
                "loom/src/loom/test/corpus/authoring/linking/checks.loom"
            )
        )
        self.assertTrue(
            self.presubmit.is_format_source_path(
                "loom/src/loom/test/corpus/new.invalid.loom"
            )
        )
        self.assertFalse(
            self.presubmit.is_format_source_path(
                "loom/src/loom/test/corpus/text/operations.loom"
            )
        )
        self.assertFalse(
            self.presubmit.is_format_source_path(
                "loom/src/loom/tooling/target/amdgpu/test/amdgpu_bad_return.loom"
            )
        )
        self.assertFalse(
            self.presubmit.is_format_source_path(
                "loom/src/loom/tools/iree-benchmark-loom/testdata/duplicate_symbol.loom"
            )
        )

    def test_loom_test_containers_are_outside_formatter_contract(self):
        self.assertFalse(
            self.presubmit.is_format_source_path(
                "loom/src/loom/test/corpus/source_low/numeric_i32_memory.loom-test"
            )
        )

    def test_format_source_classification_rejects_noncanonical_paths(self):
        self.assertFalse(self.presubmit.is_format_source_path("loom/../outside.loom"))
        self.assertFalse(self.presubmit.is_format_source_path("other/source.loom"))

    def test_tracked_format_sources_use_exact_classification(self):
        canonical_path = "loom/src/loom/test/corpus/authoring/linking/checks.loom"
        result = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                canonical_path
                + "\0loom/src/loom/test/corpus/text/operations.loom"
                + "\0loom/src/loom/test/corpus/source_low/numeric.loom-test\0"
            ),
            stderr="",
        )
        with mock.patch.object(
            self.presubmit.subprocess, "run", return_value=result
        ) as run:
            self.assertEqual(
                self.presubmit.tracked_format_source_paths(), [canonical_path]
            )

        run.assert_called_once_with(
            [
                "git",
                "--literal-pathspecs",
                "ls-files",
                "-z",
                "--",
                "loom/",
            ],
            cwd=self.presubmit.REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

    def test_path_commands_are_batched_below_portable_command_line_limit(self):
        command_prefix = ["loom-format", "--check"]
        paths = ["loom/a.loom", "loom/b.loom", "loom/c.loom"]
        single_path_limit = self.presubmit.command_line_utf16_units(
            [*command_prefix, paths[0]]
        )

        commands = self.presubmit.batch_path_commands(
            command_prefix,
            paths,
            max_command_line_utf16_units=single_path_limit,
        )

        self.assertEqual(
            commands,
            [[*command_prefix, path] for path in paths],
        )
        for command in commands:
            self.assertLessEqual(
                self.presubmit.command_line_utf16_units(command),
                single_path_limit,
            )

    def test_path_command_batching_rejects_one_oversized_path(self):
        command_prefix = ["loom-format", "--check"]
        prefix_limit = self.presubmit.command_line_utf16_units(command_prefix)

        with self.assertRaisesRegex(
            ValueError, "path exceeds the portable command-line limit"
        ):
            self.presubmit.batch_path_commands(
                command_prefix,
                ["loom/a.loom"],
                max_command_line_utf16_units=prefix_limit,
            )

    def test_batched_path_command_runs_every_batch_after_failure(self):
        commands = [
            ["loom-format", "--check", "loom/a.loom"],
            ["loom-format", "--check", "loom/b.loom"],
        ]
        with (
            mock.patch.object(
                self.presubmit, "batch_path_commands", return_value=commands
            ),
            mock.patch.object(
                self.presubmit, "run_command", side_effect=[False, True]
            ) as run_command,
        ):
            self.assertFalse(
                self.presubmit.run_batched_path_command(
                    ["loom-format", "--check"],
                    ["loom/a.loom", "loom/b.loom"],
                    "Canonical Loom source",
                )
            )

        self.assertEqual(
            run_command.call_args_list,
            [
                mock.call(commands[0], "Canonical Loom source (1/2)"),
                mock.call(commands[1], "Canonical Loom source (2/2)"),
            ],
        )

    def test_cmake_source_format_reports_missing_target_providers(self):
        cache_values = {
            "LOOM_TARGET_AMDGPU": "OFF",
            "LOOM_TARGET_LLVMIR": "ON",
            "LOOM_TARGET_SPIRV": "OFF",
            "LOOM_TARGET_X86": "ON",
        }
        output = io.StringIO()
        with (
            mock.patch.object(
                self.presubmit.project_presubmit,
                "cmake_build_dir",
                return_value=Path("/build"),
            ),
            mock.patch.object(
                self.presubmit.project_presubmit,
                "validate_cmake_build_tree",
                return_value=True,
            ),
            mock.patch.object(
                self.presubmit.project_presubmit,
                "cmake_cache_value",
                side_effect=lambda _build_dir, key: cache_values.get(key),
            ),
            contextlib.redirect_stdout(output),
        ):
            self.assertFalse(
                self.presubmit.validate_cmake_source_format_configuration()
            )

        diagnostic = output.getvalue()
        self.assertIn("missing: amdgpu, spirv", diagnostic)
        self.assertIn("-DLOOM_TARGET_AMDGPU=ON", diagnostic)
        self.assertIn("-DLOOM_TARGET_SPIRV=ON", diagnostic)
        self.assertIn("IREE_CMAKE_BUILD_DIR", diagnostic)

    def test_cmake_source_format_accepts_full_target_provider_set(self):
        with (
            mock.patch.object(
                self.presubmit.project_presubmit,
                "cmake_build_dir",
                return_value=Path("/build"),
            ),
            mock.patch.object(
                self.presubmit.project_presubmit,
                "validate_cmake_build_tree",
                return_value=True,
            ),
            mock.patch.object(
                self.presubmit.project_presubmit,
                "cmake_cache_value",
                return_value="ON",
            ),
        ):
            self.assertTrue(self.presubmit.validate_cmake_source_format_configuration())

    def test_source_format_check_covers_tracked_and_selected_sources(self):
        formatter_path = Path("/tools/loom-format")
        with (
            mock.patch.object(
                self.presubmit,
                "tracked_format_source_paths",
                return_value=["loom/a.loom"],
            ),
            mock.patch.object(
                self.presubmit,
                "selected_files",
                return_value=["loom/b.loom", "loom/deferred.loom-test"],
            ),
            mock.patch.object(
                self.presubmit,
                "existing_format_source_paths",
                return_value=["loom/b.loom"],
            ) as existing_format_source_paths,
            mock.patch.object(
                self.presubmit.project_presubmit,
                "build_and_resolve_executable",
                return_value=formatter_path,
            ) as build_and_resolve_executable,
            mock.patch.object(
                self.presubmit, "run_command", return_value=True
            ) as run_command,
            mock.patch.object(
                self.presubmit.project_presubmit, "stage_changed_paths"
            ) as stage_changed_paths,
        ):
            self.assertTrue(
                self.presubmit.run_source_format_maintenance(
                    lane="bazel",
                    files_from="paths.txt",
                    fix=False,
                )
            )

        existing_format_source_paths.assert_called_once_with(
            ["loom/b.loom", "loom/deferred.loom-test"]
        )
        build_and_resolve_executable.assert_called_once_with(
            "loom",
            self.presubmit.REPO_ROOT,
            lane="bazel",
            bazel_target="//loom/src/loom/tools/loom-format:loom-format",
            cmake_target="loom::tools::loom-format",
            bazel_args=(
                "--config=locked",
                "--//loom/config/target:enable=amdgpu,llvmir,spirv,x86",
            ),
        )
        run_command.assert_called_once_with(
            [
                str(formatter_path),
                "--check",
                "loom/a.loom",
                "loom/b.loom",
            ],
            "Canonical Loom source",
        )
        stage_changed_paths.assert_not_called()

    def test_source_format_fix_does_not_rewrite_loom_test_containers(self):
        formatter_path = Path("/tools/loom-format")
        loom_test_path = (
            "loom/src/loom/test/corpus/source_low/numeric_i32_memory.loom-test"
        )
        with (
            mock.patch.object(
                self.presubmit,
                "tracked_format_source_paths",
                return_value=["loom/a.loom"],
            ),
            mock.patch.object(
                self.presubmit,
                "selected_files",
                return_value=[loom_test_path],
            ),
            mock.patch.object(
                self.presubmit.project_presubmit,
                "build_and_resolve_executable",
                return_value=formatter_path,
            ),
            mock.patch.object(
                self.presubmit, "run_command", return_value=True
            ) as run_command,
            mock.patch.object(
                self.presubmit.project_presubmit, "stage_changed_paths"
            ) as stage_changed_paths,
        ):
            self.assertTrue(
                self.presubmit.run_source_format_maintenance(
                    lane="bazel",
                    files_from="paths.txt",
                    fix=True,
                )
            )

        run_command.assert_called_once_with(
            [str(formatter_path), "--check", "loom/a.loom"],
            "Canonical Loom source",
        )
        stage_changed_paths.assert_not_called()

    def test_source_format_fix_stages_selection_then_checks_tree(self):
        formatter_path = Path("/tools/loom-format")
        with (
            mock.patch.object(
                self.presubmit,
                "tracked_format_source_paths",
                return_value=["loom/a.loom"],
            ),
            mock.patch.object(
                self.presubmit,
                "selected_files",
                return_value=["loom/b.loom"],
            ),
            mock.patch.object(
                self.presubmit,
                "existing_format_source_paths",
                return_value=["loom/b.loom"],
            ),
            mock.patch.object(
                self.presubmit,
                "validate_cmake_source_format_configuration",
                return_value=True,
            ) as validate_cmake_source_format_configuration,
            mock.patch.object(
                self.presubmit.project_presubmit,
                "build_and_resolve_executable",
                return_value=formatter_path,
            ),
            mock.patch.object(
                self.presubmit, "run_command", return_value=True
            ) as run_command,
            mock.patch.object(
                self.presubmit.project_presubmit,
                "stage_changed_paths",
                return_value=True,
            ) as stage_changed_paths,
        ):
            self.assertTrue(
                self.presubmit.run_source_format_maintenance(
                    lane="cmake",
                    files_from="paths.txt",
                    fix=True,
                )
            )

        self.assertEqual(
            run_command.call_args_list,
            [
                mock.call(
                    [str(formatter_path), "--in-place", "loom/b.loom"],
                    "Canonicalize selected Loom source",
                ),
                mock.call(
                    [
                        str(formatter_path),
                        "--check",
                        "loom/a.loom",
                        "loom/b.loom",
                    ],
                    "Canonical Loom source",
                ),
            ],
        )
        stage_changed_paths.assert_called_once_with(
            "loom",
            self.presubmit.REPO_ROOT,
            ["loom/b.loom"],
        )
        validate_cmake_source_format_configuration.assert_called_once_with()

    def test_source_format_fix_failure_does_not_stage_partial_updates(self):
        with (
            mock.patch.object(
                self.presubmit,
                "tracked_format_source_paths",
                return_value=["loom/a.loom"],
            ),
            mock.patch.object(
                self.presubmit,
                "selected_files",
                return_value=["loom/a.loom"],
            ),
            mock.patch.object(
                self.presubmit,
                "existing_format_source_paths",
                return_value=["loom/a.loom"],
            ),
            mock.patch.object(
                self.presubmit.project_presubmit,
                "build_and_resolve_executable",
                return_value=Path("/tools/loom-format"),
            ),
            mock.patch.object(
                self.presubmit, "run_command", return_value=False
            ) as run_command,
            mock.patch.object(
                self.presubmit.project_presubmit, "stage_changed_paths"
            ) as stage_changed_paths,
        ):
            self.assertFalse(
                self.presubmit.run_source_format_maintenance(
                    lane="bazel",
                    files_from="paths.txt",
                    fix=True,
                )
            )

        run_command.assert_called_once_with(
            [str(Path("/tools/loom-format")), "--in-place", "loom/a.loom"],
            "Canonicalize selected Loom source",
        )
        stage_changed_paths.assert_not_called()

    def test_source_lint_classification_is_strict(self):
        self.assertTrue(self.presubmit.is_lint_source_path("loom/a.loom"))
        self.assertTrue(self.presubmit.is_lint_source_path("loom/a.loom-test"))
        self.assertFalse(self.presubmit.is_lint_source_path("loom/a.txt"))
        self.assertFalse(self.presubmit.is_lint_source_path("loom/../a.loom"))
        self.assertFalse(self.presubmit.is_lint_source_path("other/a.loom"))

    def test_source_lint_uses_public_tool_then_private_guardrails(self):
        linter_path = Path("/tools/loom-lint")
        with (
            mock.patch.object(
                self.presubmit,
                "tracked_lint_source_paths",
                return_value=["loom/a.loom", "loom/b.loom-test"],
            ),
            mock.patch.object(
                self.presubmit.project_presubmit,
                "build_and_resolve_executable",
                return_value=linter_path,
            ) as build_and_resolve_executable,
            mock.patch.object(
                self.presubmit, "run_command", return_value=True
            ) as run_command,
        ):
            self.assertTrue(
                self.presubmit.run_source_lint(lane="bazel", files_from=None)
            )

        build_and_resolve_executable.assert_called_once_with(
            "loom",
            self.presubmit.REPO_ROOT,
            lane="bazel",
            bazel_target="//loom/py/loom/tools:loom-lint",
            cmake_target="loom::py::loom::tools::loom-lint",
            bazel_args=("--config=locked",),
        )
        self.assertEqual(
            run_command.call_args_list,
            [
                mock.call(
                    [
                        str(linter_path),
                        "loom/a.loom",
                        "loom/b.loom-test",
                    ],
                    "Loom authoring policy",
                ),
                mock.call(
                    [
                        sys.executable,
                        "loom/build_tools/linters/loom_source_lint.py",
                    ],
                    "Loom repository invariants",
                ),
            ],
        )

    def test_source_lint_cmake_lane_invokes_public_python_entrypoint(self):
        with (
            mock.patch.object(
                self.presubmit,
                "tracked_lint_source_paths",
                return_value=["loom/a.loom"],
            ),
            mock.patch.object(
                self.presubmit.project_presubmit, "build_and_resolve_executable"
            ) as build_and_resolve_executable,
            mock.patch.object(
                self.presubmit, "run_command", return_value=True
            ) as run_command,
        ):
            self.assertTrue(
                self.presubmit.run_source_lint(lane="cmake", files_from=None)
            )

        build_and_resolve_executable.assert_not_called()
        self.assertEqual(
            run_command.call_args_list,
            [
                mock.call(
                    [
                        sys.executable,
                        "loom/py/loom/tools/source_lint.py",
                        "loom/a.loom",
                    ],
                    "Loom authoring policy",
                ),
                mock.call(
                    [
                        sys.executable,
                        "loom/build_tools/linters/loom_source_lint.py",
                    ],
                    "Loom repository invariants",
                ),
            ],
        )

    def test_generated_artifact_drift_fails_presubmit(self):
        args = types.SimpleNamespace(
            check=True,
            files_from=None,
            fix=False,
            hygiene=True,
            lane="bazel",
            tests=True,
        )
        with (
            mock.patch.object(
                self.presubmit,
                "run_generated_artifact_maintenance",
                return_value=False,
            ) as generated_artifact_maintenance,
            mock.patch.object(
                self.presubmit, "run_source_lint", return_value=True
            ) as source_lint,
            mock.patch.object(
                self.presubmit,
                "run_source_format_maintenance",
                return_value=True,
            ) as source_format_maintenance,
            mock.patch.object(
                self.presubmit, "run_bazel_tests", return_value=True
            ) as bazel_tests,
        ):
            self.assertEqual(self.presubmit.run_presubmit(args), 1)

        generated_artifact_maintenance.assert_called_once_with(False, None)
        source_format_maintenance.assert_called_once_with(
            lane="bazel", files_from=None, fix=False
        )
        source_lint.assert_called_once_with(lane="bazel", files_from=None)
        bazel_tests.assert_called_once_with()

    def test_source_lint_failure_fails_project_hygiene(self):
        args = types.SimpleNamespace(
            check=True,
            files_from=None,
            fix=False,
            hygiene=True,
            lane="bazel",
            tests=False,
        )
        with (
            mock.patch.object(
                self.presubmit,
                "run_generated_artifact_maintenance",
                return_value=True,
            ) as generated_artifact_maintenance,
            mock.patch.object(
                self.presubmit, "run_source_lint", return_value=False
            ) as source_lint,
            mock.patch.object(
                self.presubmit,
                "run_source_format_maintenance",
                return_value=True,
            ) as source_format_maintenance,
            mock.patch.object(self.presubmit, "run_bazel_tests") as bazel_tests,
        ):
            self.assertEqual(self.presubmit.run_presubmit(args), 1)

        generated_artifact_maintenance.assert_called_once_with(False, None)
        source_format_maintenance.assert_called_once_with(
            lane="bazel", files_from=None, fix=False
        )
        source_lint.assert_called_once_with(lane="bazel", files_from=None)
        bazel_tests.assert_not_called()

    def test_test_phase_does_not_repeat_hygiene_checks(self):
        args = types.SimpleNamespace(
            check=True,
            files_from=None,
            fix=False,
            hygiene=False,
            lane="bazel",
            tests=True,
        )
        with (
            mock.patch.object(
                self.presubmit, "run_generated_artifact_maintenance"
            ) as generated_artifact_maintenance,
            mock.patch.object(
                self.presubmit, "run_source_format_maintenance"
            ) as source_format_maintenance,
            mock.patch.object(self.presubmit, "run_source_lint") as source_lint,
            mock.patch.object(
                self.presubmit, "run_bazel_tests", return_value=True
            ) as bazel_tests,
        ):
            self.assertEqual(self.presubmit.run_presubmit(args), 0)

        generated_artifact_maintenance.assert_not_called()
        source_format_maintenance.assert_not_called()
        source_lint.assert_not_called()
        bazel_tests.assert_called_once_with()

    def test_main_rechecks_package_initializers_after_bazel_tests(self):
        args = types.SimpleNamespace(
            check=True,
            files_from=None,
            fix=False,
            hygiene=False,
            lane="bazel",
            tests=True,
        )
        snapshot = mock.Mock()
        snapshot.verify.return_value = False
        with (
            mock.patch.object(self.presubmit, "parse_arguments", return_value=args),
            mock.patch.object(
                self.presubmit.NonEmptyTrackedFileSnapshot,
                "capture_tracked_package_initializers",
                return_value=snapshot,
            ),
            mock.patch.object(self.presubmit, "run_bazel_tests", return_value=True),
        ):
            self.assertEqual(self.presubmit.main(), 1)
            snapshot.verify.assert_called_once_with(self.presubmit.REPO_ROOT)


if __name__ == "__main__":
    unittest.main()
