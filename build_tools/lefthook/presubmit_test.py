# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

PRESUBMIT_PATH = Path(__file__).with_name("presubmit.py")
PRESUBMIT_SPEC = importlib.util.spec_from_file_location("presubmit", PRESUBMIT_PATH)
if PRESUBMIT_SPEC is None or PRESUBMIT_SPEC.loader is None:
    raise RuntimeError(f"unable to load {PRESUBMIT_PATH}")
presubmit = importlib.util.module_from_spec(PRESUBMIT_SPEC)
sys.modules[PRESUBMIT_SPEC.name] = presubmit
PRESUBMIT_SPEC.loader.exec_module(presubmit)


def input_scope(
    paths: list[str],
    *,
    mode: str = "explicit",
    changed_paths: list[str] | None = None,
    tracked_paths: list[str] | None = None,
) -> presubmit.PresubmitInputs:
    return presubmit.PresubmitInputs(
        mode=mode,
        selected_paths=paths,
        changed_paths=changed_paths if changed_paths is not None else paths,
        tracked_paths=tracked_paths,
    )


class PresubmitTest(unittest.TestCase):
    def test_commit_scope_excludes_head_only_paths(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)

            def git(*args: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["git", *args],
                    cwd=repo_root,
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            git("init", "--quiet")
            git("config", "user.name", "Presubmit Test")
            git("config", "user.email", "presubmit@example.com")
            (repo_root / "runtime/deleted").mkdir(parents=True)
            (repo_root / "loom").mkdir()
            (repo_root / "head_only.c").write_text("int value = 0;\n")
            (repo_root / "staged.txt").write_text("base\n")
            (repo_root / "runtime/deleted/BUILD.bazel").write_text("# build\n")
            (repo_root / "runtime/old.c").write_text("int old_value;\n")
            git("add", ".")
            git("commit", "--quiet", "-m", "base")

            (repo_root / "head_only.c").write_text("int value = 1;\n")
            git("add", "head_only.c")
            git("commit", "--quiet", "-m", "head")
            with mock.patch.object(presubmit, "REPO_ROOT", repo_root):
                candidate_base_tree = presubmit.index_tree()
            (repo_root / "staged.txt").write_text("candidate\n")
            git("add", "staged.txt")
            (repo_root / "runtime/deleted/BUILD.bazel").unlink()
            git("add", "runtime/deleted/BUILD.bazel")
            git("mv", "runtime/old.c", "loom/new.c")

            with mock.patch.object(presubmit, "REPO_ROOT", repo_root):
                commit_paths = presubmit.commit_files()
                self.assertEqual(
                    set(commit_paths),
                    {
                        "loom/new.c",
                        "runtime/deleted/BUILD.bazel",
                        "runtime/old.c",
                        "staged.txt",
                    },
                )
                self.assertEqual(
                    set(presubmit.amend_files()),
                    {"head_only.c", *commit_paths},
                )
                self.assertEqual(
                    set(presubmit.changed_index_paths(candidate_base_tree)),
                    set(commit_paths),
                )

            self.assertEqual(git("status", "--short", "head_only.c").stdout, "")

            with (repo_root / "staged.txt").open("a") as staged_file:
                staged_file.write("unstaged\n")
            with mock.patch.object(presubmit, "REPO_ROOT", repo_root):
                self.assertEqual(
                    presubmit.index_worktree_conflicts(
                        input_scope(commit_paths, mode="staged")
                    ),
                    ["staged.txt"],
                )

    def test_deleted_paths_route_affected_projects_without_entering_fixers(self):
        projects = presubmit.projects_for_paths(
            ["runtime/deleted/BUILD.bazel", "loom/deleted.loom"]
        )

        self.assertEqual({project.name for project in projects}, {"runtime", "loom"})
        with mock.patch.object(presubmit, "existing_files", return_value=[]):
            self.assertTrue(
                presubmit.run_build_filename_check(["runtime/deleted/BUILD"])
            )

    def test_semgrep_candidates_require_configured_prefix_and_extension(self):
        with (
            mock.patch.object(presubmit, "SEMGREP_PATH_PREFIXES", ("project/src/",)),
            mock.patch.object(presubmit, "SEMGREP_EXTENSIONS", {".c"}),
        ):
            self.assertTrue(presubmit.is_semgrep_candidate_file("project/src/file.c"))
            self.assertFalse(presubmit.is_semgrep_candidate_file("project/src/file.h"))
            self.assertFalse(presubmit.is_semgrep_candidate_file("other/src/file.c"))

    def test_semgrep_scan_command_uses_local_error_rules(self):
        with mock.patch.dict(os.environ, {"IREE_SEMGREP_JOBS": "7"}):
            command = presubmit.semgrep_scan_command(["runtime/src/iree/base/status.c"])

        self.assertEqual(command[0:2], ["semgrep", "scan"])
        self.assertIn("--metrics=off", command)
        self.assertIn("--disable-version-check", command)
        self.assertIn("--strict", command)
        self.assertIn("--error", command)
        self.assertIn("ERROR", command)
        self.assertIn(presubmit.SEMGREP_CONFIG, command)
        self.assertIn("7", command)
        self.assertEqual(command[-1], "runtime/src/iree/base/status.c")

    def test_semgrep_default_jobs_are_capped_on_large_machines(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(presubmit.os, "cpu_count", return_value=192),
        ):
            self.assertEqual(presubmit.semgrep_jobs(), 14)

    def test_clang_tidy_default_jobs_are_capped_on_large_machines(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(presubmit.os, "cpu_count", return_value=192),
        ):
            self.assertEqual(
                presubmit.clang_tidy_jobs(),
                presubmit.CLANG_TIDY_DEFAULT_MAX_JOBS,
            )

    def test_clang_tidy_default_jobs_use_half_machine(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(presubmit.os, "cpu_count", return_value=64),
        ):
            self.assertEqual(presubmit.clang_tidy_jobs(), 32)

    def test_clang_tidy_jobs_can_be_configured(self):
        with mock.patch.dict(os.environ, {"IREE_CLANG_TIDY_JOBS": "48"}):
            self.assertEqual(presubmit.clang_tidy_jobs(), 48)

    def test_clang_format_default_jobs_are_capped_on_large_machines(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(presubmit.os, "cpu_count", return_value=192),
        ):
            self.assertEqual(
                presubmit.clang_format_jobs(),
                presubmit.C_FORMAT_DEFAULT_MAX_JOBS,
            )

    def test_clang_format_jobs_can_be_configured(self):
        with mock.patch.dict(os.environ, {"IREE_CLANG_FORMAT_JOBS": "3"}):
            self.assertEqual(presubmit.clang_format_jobs(), 3)

    def test_clang_format_batches_large_file_sets(self):
        files = [f"runtime/src/iree/base/file_{index}.c" for index in range(130)]
        with (
            mock.patch.object(
                presubmit, "existing_files", side_effect=lambda paths: paths
            ),
            mock.patch.object(presubmit, "require_tool", return_value=True),
            mock.patch.object(presubmit, "clang_format_jobs", return_value=8),
            mock.patch.object(
                presubmit, "run_parallel_commands", return_value=True
            ) as run_parallel_commands,
        ):
            self.assertTrue(presubmit.run_clang_format(files, fix=False, verbose=False))

        commands = run_parallel_commands.call_args.args[0]
        self.assertEqual(len(commands), 3)
        self.assertEqual(commands[0][:3], ["clang-format", "--dry-run", "--Werror"])
        self.assertEqual(len(commands[0][3:]), presubmit.C_FORMAT_BATCH_SIZE)
        self.assertEqual(run_parallel_commands.call_args.kwargs["jobs"], 3)

    def test_stage_files_skips_git_add_when_worktree_is_clean(self):
        clean_diff = presubmit.subprocess.CompletedProcess(
            args=["git", "diff"], returncode=0
        )
        with (
            mock.patch.object(presubmit.subprocess, "run", return_value=clean_diff),
            mock.patch.object(
                presubmit, "run_command", return_value=True
            ) as run_command,
        ):
            self.assertTrue(presubmit.stage_files(["build_tools/file.py"], False))

        run_command.assert_not_called()

    def test_stage_files_adds_formatter_changes(self):
        dirty_diff = presubmit.subprocess.CompletedProcess(
            args=["git", "diff"], returncode=1
        )
        with (
            mock.patch.object(presubmit.subprocess, "run", return_value=dirty_diff),
            mock.patch.object(
                presubmit, "run_command", return_value=True
            ) as run_command,
        ):
            self.assertTrue(presubmit.stage_files(["build_tools/file.py"], False))

        self.assertEqual(run_command.call_args.args[0][0:3], ["git", "add", "--"])

    def test_fixing_the_index_stops_before_validation_and_names_paths(self):
        args = types.SimpleNamespace(fail_on_fix=True, fix=True)
        snapshot = mock.Mock()
        snapshot.verify.return_value = True
        output = io.StringIO()
        with (
            mock.patch.object(presubmit, "index_tree", return_value="before"),
            mock.patch.object(
                presubmit.NonEmptyTrackedFileSnapshot,
                "capture_tracked_package_initializers",
                return_value=snapshot,
            ),
            mock.patch.object(presubmit, "run_presubmit", return_value=0),
            mock.patch.object(
                presubmit,
                "changed_index_paths",
                return_value=["generated/output.cmake", "staged.c"],
            ),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(
                presubmit.run_presubmit_with_source_guard(
                    args, input_scope(["staged.c"], mode="staged"), []
                ),
                1,
            )

        summary = output.getvalue()
        self.assertIn("generated/output.cmake", summary)
        self.assertIn("staged.c", summary)
        self.assertIn("stops before tests", summary)
        self.assertIn("git diff --cached", summary)
        snapshot.verify.assert_called_once_with(presubmit.REPO_ROOT)

    def test_fix_retry_contract_uses_the_real_index(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            repo_root = Path(temporary_directory)

            def git(*args: str) -> None:
                subprocess.run(
                    ["git", *args],
                    cwd=repo_root,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            git("init", "--quiet")
            git("config", "user.name", "Presubmit Test")
            git("config", "user.email", "presubmit@example.com")
            source_path = repo_root / "source.py"
            source_path.write_text("value = 'base'\n")
            git("add", "source.py")
            git("commit", "--quiet", "-m", "base")
            source_path.write_text("value =  'candidate'\n")
            git("add", "source.py")

            run_count = 0

            def run_presubmit(*_args) -> int:
                nonlocal run_count
                run_count += 1
                if run_count == 1:
                    source_path.write_text("value = 'candidate'\n")
                    git("add", "source.py")
                return 0

            args = types.SimpleNamespace(fail_on_fix=True, fix=True)
            snapshot = mock.Mock()
            snapshot.verify.return_value = True
            output = io.StringIO()
            with (
                mock.patch.object(presubmit, "REPO_ROOT", repo_root),
                mock.patch.object(
                    presubmit.NonEmptyTrackedFileSnapshot,
                    "capture_tracked_package_initializers",
                    return_value=snapshot,
                ),
                mock.patch.object(
                    presubmit, "run_presubmit", side_effect=run_presubmit
                ),
                contextlib.redirect_stdout(output),
            ):
                inputs = input_scope(["source.py"], mode="staged")
                self.assertEqual(
                    presubmit.run_presubmit_with_source_guard(args, inputs, []), 1
                )
                self.assertEqual(
                    presubmit.run_presubmit_with_source_guard(args, inputs, []), 0
                )

            self.assertEqual(run_count, 2)
            self.assertIn("source.py", output.getvalue())
            self.assertIn("stops before tests", output.getvalue())

    def test_bazel_to_cmake_fix_receives_only_selected_build_files(self):
        selected_paths = ["runtime/src/iree/base/BUILD.bazel"]
        with (
            mock.patch.object(presubmit, "existing_files", return_value=selected_paths),
            mock.patch.object(
                presubmit, "run_command", return_value=True
            ) as run_command,
        ):
            self.assertTrue(
                presubmit.run_bazel_to_cmake(
                    [*selected_paths, "runtime/src/iree/base/status.c"],
                    fix=True,
                    verbose=False,
                )
            )

        command = run_command.call_args.args[0]
        self.assertIn("--stage-updates", command)
        self.assertEqual(command[-1], selected_paths[0])
        self.assertNotIn("runtime/src/iree/base/status.c", command)

    def test_bazel_to_cmake_global_fix_is_read_only(self):
        with mock.patch.object(
            presubmit, "run_command", return_value=True
        ) as run_command:
            self.assertTrue(
                presubmit.run_bazel_to_cmake(
                    ["build_tools/bazel_to_cmake/bazel_to_cmake.py"],
                    fix=True,
                    verbose=False,
                )
            )

        command = run_command.call_args.args[0]
        self.assertIn("--check", command)
        self.assertNotIn("--stage-updates", command)

        self.assertTrue(
            presubmit.is_bazel_to_cmake_global_trigger(
                "runtime/requirements/package_policy.bzl"
            )
        )
        self.assertTrue(
            presubmit.is_bazel_to_cmake_global_trigger(
                "loom/build_tools/amdgpu/target_config.bzl"
            )
        )

    def test_bazel_to_cmake_skips_unrelated_paths(self):
        with (
            mock.patch.object(presubmit, "existing_files", return_value=[]),
            mock.patch.object(presubmit, "skip_step", return_value=True) as skip_step,
            mock.patch.object(presubmit, "run_command") as run_command,
        ):
            self.assertTrue(
                presubmit.run_bazel_to_cmake(
                    ["build_tools/lefthook/presubmit.py"],
                    fix=True,
                    verbose=False,
                )
            )

        skip_step.assert_called_once_with(
            "Bazel-to-CMake", "no selected Bazel/CMake files"
        )
        run_command.assert_not_called()

    def test_clang_tidy_candidates_require_configured_prefix_and_extension(self):
        with (
            mock.patch.object(presubmit, "CLANG_TIDY_PATH_PREFIXES", ("project/src/",)),
            mock.patch.object(presubmit, "CLANG_TIDY_EXTENSIONS", {".c", ".h", ".inl"}),
        ):
            self.assertTrue(
                presubmit.is_clang_tidy_candidate_file("project/src/file.c")
            )
            self.assertTrue(
                presubmit.is_clang_tidy_candidate_file("project/src/file.h")
            )
            self.assertTrue(
                presubmit.is_clang_tidy_candidate_file("project/src/file.inl")
            )
            self.assertFalse(
                presubmit.is_clang_tidy_candidate_file("project/src/file.py")
            )
            self.assertFalse(presubmit.is_clang_tidy_candidate_file("other/src/file.c"))

    def test_clang_tidy_bazel_command_uses_aspect_output_group(self):
        command = presubmit.clang_tidy_bazel_command(["//runtime/src/iree/base:all"])

        self.assertEqual(command[0:2], ["bazel", "build"])
        self.assertNotIn("--keep_going", command)
        self.assertIn(presubmit.CLANG_TIDY_REPO_ENV, command)
        self.assertIn(f"--aspects={presubmit.CLANG_TIDY_ASPECT}", command)
        self.assertIn(f"--output_groups={presubmit.CLANG_TIDY_OUTPUT_GROUP}", command)
        self.assertEqual(command[-1], "//runtime/src/iree/base:all")

    def test_clang_tidy_bazel_command_obeys_configured_jobs(self):
        with mock.patch.dict(os.environ, {"IREE_CLANG_TIDY_JOBS": "7"}):
            command = presubmit.clang_tidy_bazel_command(
                ["//runtime/src/iree/base:all"]
            )

        self.assertEqual(command[2:4], ["--jobs", "7"])

    def test_clang_tidy_bazel_command_can_keep_going(self):
        command = presubmit.clang_tidy_bazel_command(
            ["//runtime/src/iree/base:all"],
            keep_going=True,
        )

        self.assertEqual(command[0:3], ["bazel", "build", "--keep_going"])

    def test_clang_tidy_bazel_command_can_export_fixes(self):
        command = presubmit.clang_tidy_bazel_command(
            ["//runtime/src/iree/base:all"],
            build_events_path=Path(".tmp/fixes/build_events.json"),
            emit_fixes=True,
        )

        self.assertIn("--aspects_parameters=emit_fixes=true", command)
        self.assertIn(
            "--output_groups=iree_clang_tidy_reports,iree_clang_tidy_fixes",
            command,
        )
        self.assertIn(
            f"--build_event_json_file={Path('.tmp/fixes/build_events.json')}",
            command,
        )

    def test_clang_tidy_bazel_command_can_use_local_output_groups(self):
        command = presubmit.clang_tidy_bazel_command(
            ["//runtime/src/iree/base:all"],
            emit_fixes=True,
            local_outputs=True,
        )

        self.assertIn("--aspects_parameters=emit_fixes=true", command)
        self.assertIn(
            "--output_groups=iree_clang_tidy_local_reports,iree_clang_tidy_local_fixes",
            command,
        )

    def test_clang_tidy_replacements_are_rebased_to_worktree_paths(self):
        sandbox_path = (
            "/tmp/bazel/123/sandbox/linux-sandbox/1/"
            "execroot/_main/runtime/src/iree/base/status.c"
        )
        text = (
            "---\n"
            f"MainSourceFile: '{sandbox_path}'\n"
            f"FilePath: '{sandbox_path}'\n"
            "Replacements:\n"
            f"  - FilePath: '{sandbox_path}'\n"
            "    Offset: 1\n"
            "BuildDirectory: '/tmp/bazel/123/sandbox/linux-sandbox/1/"
            "execroot/_main'\n"
            "...\n"
        )

        normalized = presubmit.normalize_clang_tidy_replacements_yaml(text)

        expected_path = str(presubmit.REPO_ROOT / "runtime/src/iree/base/status.c")
        self.assertIn(f"MainSourceFile: '{expected_path}'", normalized)
        self.assertIn(f"FilePath: '{expected_path}'", normalized)
        self.assertIn(f"  - FilePath: '{expected_path}'", normalized)
        self.assertIn(f"BuildDirectory: '{presubmit.REPO_ROOT}'", normalized)
        self.assertNotIn("/execroot/_main/", normalized)

    def test_clang_tidy_fix_paths_filter_to_selected_translation_units(self):
        fix_paths = [
            Path(
                "label.runtime_src_iree_base_status_c.clang_tidy_fixes.yaml",
            ),
            Path(
                "label.runtime_src_iree_base_allocator_c.clang_tidy_fixes.yaml",
            ),
        ]

        self.assertEqual(
            presubmit.clang_tidy_fix_paths_for_files(
                fix_paths,
                ["runtime/src/iree/base/status.c"],
            ),
            [fix_paths[0]],
        )

    def test_cmake_clang_tidy_candidates_are_translation_units(self):
        with mock.patch.object(
            presubmit, "CLANG_TIDY_PATH_PREFIXES", ("runtime/src/iree/",)
        ):
            self.assertEqual(
                presubmit.cmake_clang_tidy_candidate_files(
                    [
                        "runtime/src/iree/base/status.c",
                        "runtime/src/iree/base/status.h",
                        "runtime/src/iree/base/status.py",
                    ]
                ),
                ["runtime/src/iree/base/status.c"],
            )

    def test_cmake_clang_tidy_command_uses_parallel_driver(self):
        with mock.patch.object(presubmit, "clang_tidy_jobs", return_value=17):
            command = presubmit.cmake_clang_tidy_command(
                run_clang_tidy="run-clang-tidy",
                clang_tidy="clang-tidy",
                plugin=Path(".tmp/plugin/libIREEClangTidyPlugin.so"),
                compile_commands_dir=Path("build/cmake-debug"),
                files=["runtime/src/iree/base/status.c"],
            )

        self.assertEqual(command[0], "run-clang-tidy")
        self.assertIn("-clang-tidy-binary", command)
        self.assertIn("clang-tidy", command)
        self.assertIn(f"-load={Path('.tmp/plugin/libIREEClangTidyPlugin.so')}", command)
        self.assertIn(f"-config-file={presubmit.CLANG_TIDY_CONFIG}", command)
        self.assertIn("-p", command)
        self.assertIn(str(Path("build/cmake-debug")), command)
        self.assertIn("-j", command)
        self.assertIn("17", command)
        self.assertIn("-warnings-as-errors=*", command)
        self.assertEqual(command[-1], "runtime/src/iree/base/status.c")

    def test_cmake_clang_tidy_plugin_configure_pins_matching_packages(self):
        llvm_package_dir = Path("/opt/llvm/lib/cmake/llvm")

        command = presubmit.cmake_clang_tidy_plugin_configure_command(
            llvm_package_dir=llvm_package_dir
        )

        self.assertEqual(command[0], "cmake")
        self.assertIn("-S", command)
        self.assertIn("build_tools/clang_tidy", command)
        self.assertIn(f"-DLLVM_DIR={llvm_package_dir}", command)
        self.assertIn("-DClang_DIR=/opt/llvm/lib/cmake/clang", command)

    def test_cmake_generated_compile_inputs_command_uses_cmake_build(self):
        with mock.patch.object(presubmit, "clang_tidy_jobs", return_value=23):
            command = presubmit.cmake_generated_compile_inputs_command(
                compile_commands_dir=Path("build/cmake-debug"),
            )

        self.assertEqual(command[0], "cmake")
        self.assertIn("--build", command)
        self.assertIn(str(Path("build/cmake-debug")), command)
        self.assertIn("--target", command)
        self.assertIn(presubmit.CLANG_TIDY_CMAKE_GENERATED_INPUTS_TARGET, command)
        self.assertIn("--parallel", command)
        self.assertIn("23", command)

    def test_cmake_clang_tidy_fix_command_uses_parallel_driver(self):
        with mock.patch.object(presubmit, "clang_tidy_jobs", return_value=19):
            command = presubmit.cmake_run_clang_tidy_fix_command(
                run_clang_tidy="run-clang-tidy",
                clang_tidy="clang-tidy",
                clang_apply_replacements="clang-apply-replacements",
                plugin=Path(".tmp/plugin/libIREEClangTidyPlugin.so"),
                compile_commands_dir=Path("build/cmake-debug"),
                files=["runtime/src/iree/base/status.c"],
            )

        self.assertEqual(command[0], "run-clang-tidy")
        self.assertIn("-clang-tidy-binary", command)
        self.assertIn("clang-tidy", command)
        self.assertIn("-clang-apply-replacements-binary", command)
        self.assertIn("clang-apply-replacements", command)
        self.assertIn(f"-config-file={presubmit.CLANG_TIDY_CONFIG}", command)
        self.assertIn("-j", command)
        self.assertIn("19", command)
        self.assertIn("-fix", command)
        self.assertIn("-format", command)
        self.assertEqual(command[-1], "runtime/src/iree/base/status.c")

    def test_cmake_build_dir_uses_recorded_devtools_state(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            state_file = Path(temporary_dir) / "iree" / "cmake_build_dir"
            state_file.parent.mkdir()
            configured_build_dir = Path(temporary_dir) / "configured-build"
            state_file.write_text(f"{configured_build_dir}\n", encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {presubmit.DEVTOOLS_TMP_ENV: temporary_dir},
                clear=True,
            ):
                self.assertEqual(
                    presubmit.cmake_build_dir_from_env(),
                    configured_build_dir,
                )

    def test_bazel_package_target_for_path_finds_nearest_package(self):
        self.assertEqual(
            presubmit.bazel_package_target_for_path(
                "build_tools/bazel/test/cc_benchmark_smoke_test_fixture.c"
            ),
            "//build_tools/bazel/test:all",
        )

    def test_clang_tidy_skips_locally_when_llvm_is_missing(self):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(
                presubmit, "CLANG_TIDY_PATH_PREFIXES", ("runtime/src/iree/",)
            ),
            mock.patch.object(
                presubmit, "clang_tidy_llvm_available", return_value=False
            ),
        ):
            ok = presubmit.run_clang_tidy(
                input_scope(["runtime/src/iree/base/status.c"]),
                profile="paranoid",
                lane="bazel",
                verbose=False,
            )

        self.assertTrue(ok)
        self.assertIn("[skip] clang-tidy", output.getvalue())

    def test_clang_tidy_required_ci_fails_when_llvm_is_missing(self):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            mock.patch.dict(os.environ, {"IREE_CLANG_TIDY_REQUIRED": "1"}),
            mock.patch.object(
                presubmit, "CLANG_TIDY_PATH_PREFIXES", ("runtime/src/iree/",)
            ),
            mock.patch.object(
                presubmit, "clang_tidy_llvm_available", return_value=False
            ),
        ):
            ok = presubmit.run_clang_tidy(
                input_scope(["runtime/src/iree/base/status.c"]),
                profile="ci",
                lane="bazel",
                verbose=False,
            )

        self.assertFalse(ok)
        self.assertIn("[fail] clang-tidy", output.getvalue())

    def test_clang_tidy_runs_bazel_package_for_candidate_file(self):
        with (
            mock.patch.object(
                presubmit, "CLANG_TIDY_PATH_PREFIXES", ("build_tools/bazel/test/",)
            ),
            mock.patch.object(
                presubmit, "clang_tidy_llvm_available", return_value=True
            ),
            mock.patch.object(
                presubmit, "run_command", return_value=True
            ) as run_command,
        ):
            ok = presubmit.run_clang_tidy(
                input_scope(
                    ["build_tools/bazel/test/cc_benchmark_smoke_test_fixture.c"]
                ),
                profile="paranoid",
                lane="bazel",
                verbose=False,
            )

        self.assertTrue(ok)
        command = run_command.call_args.args[0]
        self.assertNotIn("--keep_going", command)
        self.assertIn(
            f"--output_groups={presubmit.CLANG_TIDY_LOCAL_OUTPUT_GROUP}", command
        )
        self.assertIn("//build_tools/bazel/test:all", command)

    def test_clang_tidy_ci_runs_bazel_packages_keep_going(self):
        with (
            mock.patch.object(
                presubmit, "CLANG_TIDY_PATH_PREFIXES", ("build_tools/bazel/test/",)
            ),
            mock.patch.object(
                presubmit, "clang_tidy_llvm_available", return_value=True
            ),
            mock.patch.object(
                presubmit, "run_command", return_value=True
            ) as run_command,
        ):
            ok = presubmit.run_clang_tidy(
                input_scope(
                    ["build_tools/bazel/test/cc_benchmark_smoke_test_fixture.c"]
                ),
                profile="ci",
                lane="bazel",
                verbose=False,
            )

        self.assertTrue(ok)
        command = run_command.call_args.args[0]
        self.assertIn("--keep_going", command)
        self.assertIn(
            f"--output_groups={presubmit.CLANG_TIDY_LOCAL_OUTPUT_GROUP}", command
        )
        self.assertIn("//build_tools/bazel/test:all", command)

    def test_clang_tidy_base_implementation_scope_stays_bounded(self):
        inputs = input_scope(
            ["runtime/src/iree/base/status.c"],
            mode="base",
            tracked_paths=[
                "runtime/src/iree/base/status.c",
                "runtime/src/iree/base/allocator.c",
            ],
        )

        self.assertEqual(
            presubmit.clang_tidy_analysis_paths(inputs),
            ["runtime/src/iree/base/status.c"],
        )

    def test_clang_tidy_base_header_scope_escalates_to_full(self):
        inputs = input_scope(
            ["runtime/src/iree/base/status.h"],
            mode="base",
            tracked_paths=[
                "runtime/src/iree/base/status.c",
                "runtime/src/iree/base/status.h",
                "runtime/src/iree/base/allocator.c",
            ],
        )

        self.assertEqual(
            presubmit.clang_tidy_analysis_paths(inputs),
            [
                "runtime/src/iree/base/status.c",
                "runtime/src/iree/base/status.h",
                "runtime/src/iree/base/allocator.c",
            ],
        )

    def test_clang_tidy_base_inl_scope_escalates_to_full(self):
        inputs = input_scope(
            ["runtime/src/iree/base/table.inl"],
            mode="base",
            tracked_paths=[
                "runtime/src/iree/base/status.c",
                "runtime/src/iree/base/table.inl",
            ],
        )

        self.assertEqual(
            presubmit.clang_tidy_analysis_paths(inputs),
            [
                "runtime/src/iree/base/status.c",
                "runtime/src/iree/base/table.inl",
            ],
        )

    def test_clang_tidy_base_infra_scope_escalates_and_runs_plugin_tests(self):
        inputs = input_scope(
            ["build_tools/clang_tidy/iree/IreeTidyModule.cc"],
            mode="base",
            tracked_paths=[
                "runtime/src/iree/base/status.c",
                "build_tools/clang_tidy/iree/IreeTidyModule.cc",
            ],
        )
        with (
            mock.patch.object(
                presubmit, "clang_tidy_llvm_available", return_value=True
            ),
            mock.patch.object(
                presubmit, "run_command", return_value=True
            ) as run_command,
        ):
            ok = presubmit.run_clang_tidy(
                inputs,
                profile="paranoid",
                lane="bazel",
                verbose=False,
            )

        self.assertTrue(ok)
        plugin_test_command = run_command.call_args_list[0].args[0]
        self.assertIn("//build_tools/clang_tidy:plugin_tests", plugin_test_command)
        clang_tidy_command = run_command.call_args_list[-1].args[0]
        self.assertIn("//runtime/src/iree/base:all", clang_tidy_command)

    def test_clang_tidy_infra_runs_plugin_tests(self):
        with (
            mock.patch.object(
                presubmit, "clang_tidy_llvm_available", return_value=True
            ),
            mock.patch.object(
                presubmit, "run_command", return_value=True
            ) as run_command,
        ):
            ok = presubmit.run_clang_tidy(
                input_scope(["build_tools/clang_tidy/iree/IreeTidyModule.cc"]),
                profile="paranoid",
                lane="bazel",
                verbose=False,
            )

        self.assertTrue(ok)
        plugin_test_command = run_command.call_args_list[0].args[0]
        self.assertIn("//build_tools/clang_tidy:plugin_tests", plugin_test_command)

    def test_default_profile_has_no_static_analysis_provider(self):
        ok = presubmit.run_static_analysis(
            input_scope(["runtime/src/iree/base/status.c"]),
            profile="default",
            lane="bazel",
            verbose=False,
        )

        self.assertTrue(ok)

    def test_missing_static_tool_is_only_fatal_in_ci(self):
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            mock.patch.object(presubmit.shutil, "which", return_value=None),
        ):
            self.assertTrue(
                presubmit.require_static_tool(
                    "missing-tool", "Missing tool", "paranoid"
                )
            )
            self.assertFalse(
                presubmit.require_static_tool("missing-tool", "Missing tool", "ci")
            )

        self.assertIn("[skip]", output.getvalue())
        self.assertIn("[fail]", output.getvalue())

    def test_missing_semgrep_does_not_attempt_scan(self):
        output = io.StringIO()
        inputs = input_scope(["runtime/src/iree/base/status.c"])
        with (
            contextlib.redirect_stdout(output),
            mock.patch.object(presubmit.shutil, "which", return_value=None),
            mock.patch.object(presubmit, "run_command") as run_command,
        ):
            self.assertTrue(
                presubmit.run_semgrep(inputs, profile="paranoid", verbose=False)
            )
            self.assertFalse(presubmit.run_semgrep(inputs, profile="ci", verbose=False))

        run_command.assert_not_called()
        self.assertIn("[skip]", output.getvalue())
        self.assertIn("[fail]", output.getvalue())

    def test_semgrep_is_explicitly_delegated_to_linux_on_windows(self):
        output = io.StringIO()
        inputs = input_scope(["runtime/src/iree/base/status.c"])
        with (
            contextlib.redirect_stdout(output),
            mock.patch.object(presubmit.sys, "platform", "win32"),
            mock.patch.object(presubmit, "require_static_tool") as require_tool,
            mock.patch.object(presubmit, "run_command") as run_command,
        ):
            self.assertTrue(
                presubmit.run_semgrep(inputs, profile="paranoid", verbose=False)
            )

        require_tool.assert_not_called()
        run_command.assert_not_called()
        self.assertIn("enforced by the Linux paranoid presubmit", output.getvalue())

    def test_requirements_and_docs_workflow_trigger_root_devtools_tests(self):
        self.assertTrue(
            presubmit.is_root_devtools_trigger(".github/workflows/docs.yml")
        )
        self.assertTrue(presubmit.is_root_devtools_trigger("requirements-analysis.in"))
        self.assertTrue(
            presubmit.is_root_devtools_trigger("requirements-analysis.lock.txt")
        )
        self.assertTrue(
            presubmit.is_root_devtools_trigger("loom/docs/requirements.lock.txt")
        )
        self.assertFalse(presubmit.is_root_devtools_trigger("runtime/requirements.txt"))

    def test_existing_project_scripts_include_loom(self):
        self.assertIn(
            "loom",
            {project.name for project in presubmit.existing_project_scripts()},
        )

    def test_project_hygiene_dispatch_uses_project_presubmit_interface(self):
        project = presubmit.Project(
            name="loom",
            root="loom/",
            script="loom/build_tools/presubmit.py",
        )
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            mock.patch.object(
                presubmit,
                "git_worktree_dir",
                return_value=Path(temporary_directory),
            ),
            mock.patch.object(
                presubmit, "run_command", return_value=True
            ) as run_command,
        ):
            self.assertTrue(
                presubmit.run_project_presubmits(
                    [project],
                    ["loom/test.loom"],
                    fix=False,
                    verbose=False,
                    lane="bazel",
                    phase="hygiene",
                )
            )

        command, description, verbose = run_command.call_args.args
        self.assertEqual(command[:2], ["python", "loom/build_tools/presubmit.py"])
        self.assertIn("--hygiene", command)
        self.assertNotIn("--tests", command)
        self.assertIn("--check", command)
        self.assertEqual(description, "loom hygiene")
        self.assertFalse(verbose)

    def test_disabling_project_tests_preserves_project_hygiene(self):
        project = presubmit.Project(
            name="loom",
            root="loom/",
            script="loom/build_tools/presubmit.py",
        )
        args = types.SimpleNamespace(
            check=True,
            clang_tidy=False,
            fix=False,
            hygiene=True,
            lane="bazel",
            print_plan=False,
            profile="ci",
            project_tests=False,
            static_analysis=False,
            tests=True,
            verbose=False,
        )
        with (
            mock.patch.object(presubmit, "run_hygiene", return_value=True),
            mock.patch.object(
                presubmit, "run_project_presubmits", return_value=True
            ) as run_project_presubmits,
            mock.patch.object(
                presubmit, "run_root_devtools_tests_for_lane", return_value=True
            ),
            mock.patch.object(presubmit, "skip_step", return_value=True),
        ):
            self.assertEqual(
                presubmit.run_presubmit(
                    args,
                    input_scope(["loom/test.loom"]),
                    [project],
                ),
                0,
            )

        run_project_presubmits.assert_called_once()
        self.assertEqual(
            run_project_presubmits.call_args.kwargs["phase"],
            "hygiene",
        )


if __name__ == "__main__":
    unittest.main()
