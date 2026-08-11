# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib.util
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
            "--//loom/config/target:enable=amdgpu,iree_vm,llvmir,spirv,x86",
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

    def test_source_lint_is_project_owned_and_read_only(self):
        with mock.patch.object(
            self.presubmit, "run_command", return_value=True
        ) as run_command:
            self.assertTrue(self.presubmit.run_source_lint())

        run_command.assert_called_once_with(
            [
                sys.executable,
                "loom/build_tools/linters/loom_source_lint.py",
            ],
            "Loom source invariants",
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
                self.presubmit, "run_bazel_tests", return_value=True
            ) as bazel_tests,
        ):
            self.assertEqual(self.presubmit.run_presubmit(args), 1)

        generated_artifact_maintenance.assert_called_once_with(False)
        source_lint.assert_called_once_with()
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
            mock.patch.object(self.presubmit, "run_bazel_tests") as bazel_tests,
        ):
            self.assertEqual(self.presubmit.run_presubmit(args), 1)

        generated_artifact_maintenance.assert_called_once_with(False)
        source_lint.assert_called_once_with()
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
            mock.patch.object(self.presubmit, "run_source_lint") as source_lint,
            mock.patch.object(
                self.presubmit, "run_bazel_tests", return_value=True
            ) as bazel_tests,
        ):
            self.assertEqual(self.presubmit.run_presubmit(args), 0)

        generated_artifact_maintenance.assert_not_called()
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
