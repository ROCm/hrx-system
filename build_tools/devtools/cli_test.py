# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from build_tools.devtools import aliases, cli
from build_tools.devtools import bazel as bazel_dev
from build_tools.devtools.command_plan import WriteFileStep


class CliTest(unittest.TestCase):
    def planned_argv(self, arguments: list[str]) -> list[str]:
        args = cli.parse_arguments(arguments)
        plan = args.handler(args)
        return plan.steps[0].argv

    def parse_help(self, arguments: list[str]) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as cm:
            cli.parse_arguments(arguments)

        self.assertEqual(cm.exception.code, 0)
        return output.getvalue()

    def parse_agent_md(self, arguments: list[str]) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as cm:
            cli.parse_arguments(arguments)

        self.assertEqual(cm.exception.code, 0)
        return output.getvalue()

    def test_cmake_configure_forwards_options_without_separator(self):
        args = cli.parse_arguments(["cmake", "configure", "--fresh"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--fresh", description)
        self.assertNotIn("-- --fresh", description)

    def test_cmake_build_dir_can_be_selected(self):
        args = cli.parse_arguments(
            [
                "--cmake-build-dir",
                "build/cmake-ci",
                "cmake",
                "configure",
                "-DIREE_HAL_DRIVER_AMDGPU=OFF",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("build/cmake-ci", description)
        self.assertNotIn("../builds", description)

    def test_bazel_configure_accepts_portable_project_options(self):
        args = cli.parse_arguments(
            [
                "bazel",
                "configure",
                "-DIREE_HAL_DRIVER_AMDGPU=ON",
                "-DIREE_ROCM_PATH=/opt/rocm",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("-DIREE_HAL_DRIVER_AMDGPU=ON", description)
        self.assertIn("-DIREE_ROCM_PATH=/opt/rocm", description)

    def test_bazel_configure_accepts_portable_loom_target_options(self):
        args = cli.parse_arguments(
            [
                "bazel",
                "configure",
                "-DLOOM_TARGET_AMDGPU=ON",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("-DLOOM_TARGET_AMDGPU=ON", description)

    def test_bazel_configure_accepts_native_driver_options(self):
        args = cli.parse_arguments(
            [
                "bazel",
                "configure",
                "--//runtime/config/hal:drivers=amdgpu,local-sync,local-task,null",
                "--repo_env=IREE_ROCM_PATH=/opt/rocm",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--//runtime/config/hal:drivers=amdgpu", description)
        self.assertIn("--repo_env=IREE_ROCM_PATH=/opt/rocm", description)

    def test_bazel_build_defaults_to_project_targets(self):
        args = cli.parse_arguments(["bazel", "build"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("//runtime/...", description)
        self.assertIn("//libhrx/...", description)

    def test_bazel_build_with_only_options_still_uses_default_targets(self):
        args = cli.parse_arguments(["bazel", "build", "--config=asan"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--config=asan", description)
        self.assertIn("//runtime/...", description)
        self.assertIn("//libhrx/...", description)

    def test_bazel_build_preserves_negative_target_separator(self):
        argv = self.planned_argv(
            [
                "bazel",
                "build",
                "--",
                "//runtime/...",
                "-//runtime/src/iree/hal/drivers/amdgpu/...",
            ]
        )

        self.assertEqual(
            argv[1:],
            [
                "build",
                "--",
                "//runtime/...",
                "-//runtime/src/iree/hal/drivers/amdgpu/...",
            ],
        )

    def test_bazel_test_preserves_negative_target_separator_after_options(self):
        argv = self.planned_argv(
            [
                "bazel",
                "test",
                "--test_tag_filters=-requires-gpu",
                "//runtime/...",
                "-//runtime/src/iree/hal/drivers/amdgpu/...",
            ]
        )

        self.assertEqual(
            argv[1:],
            [
                "test",
                "--config=presubmit",
                "--test_tag_filters=-requires-gpu",
                "--",
                "//runtime/...",
                "-//runtime/src/iree/hal/drivers/amdgpu/...",
            ],
        )

    def test_bazel_query_preserves_negative_target_separator(self):
        argv = self.planned_argv(
            [
                "bazel",
                "query",
                "//runtime/...",
                "-//runtime/src/iree/hal/drivers/amdgpu/...",
            ]
        )

        self.assertEqual(
            argv[1:],
            [
                "query",
                "--",
                "//runtime/...",
                "-//runtime/src/iree/hal/drivers/amdgpu/...",
            ],
        )

    def test_bazel_direct_query_commands_forward_to_bazel(self):
        args = cli.parse_arguments(
            ["bazel", "query", "kind(cc_library, //runtime/...)", "--output=label"]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("bazel query", description)
        self.assertIn("kind(cc_library, //runtime/...)", description)
        self.assertIn("--output=label", description)

        args = cli.parse_arguments(
            ["bazel", "cquery", "--output=files", "//runtime/..."]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("bazel cquery --output=files //runtime/...", description)

        args = cli.parse_arguments(["bazel", "info", "execution_root"])

        plan = args.handler(args)
        self.assertIn("bazel info execution_root", plan.describe())

    def test_bazel_run_builds_and_resolves_binary_before_exec(self):
        args = cli.parse_arguments(
            [
                "bazel",
                "run",
                "--config=asan",
                "//runtime/src/iree/base:allocator_benchmark",
                "--",
                "--benchmark_filter=Alloc",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("bazel build --config=asan", description)
        self.assertIn("bazel cquery --output=files --config=asan", description)
        self.assertIn("exec '<built executable>' --benchmark_filter=Alloc", description)

    def test_bazel_run_can_print_resolved_executable_path(self):
        args = cli.parse_arguments(
            [
                "bazel",
                "run",
                "-p",
                "//runtime/src/iree/base:allocator_benchmark",
            ]
        )

        plan = args.handler(args)
        self.assertIn("# print built executable path", plan.describe())

    def test_bazel_try_generates_scratch_package(self):
        args = cli.parse_arguments(
            [
                "bazel",
                "try",
                "-c",
                "--dep",
                "//runtime/src/iree/base",
                "-e",
                "int main() { return 0; }",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn(".tmp/iree-bazel-try/run-<pid>/BUILD.bazel", description)
        self.assertIn("bazel build", description)
        self.assertIn("//.tmp/iree-bazel-try/run-<pid>:snippet", description)
        self.assertIn("# compile only", description)

    def test_cmake_try_generates_scratch_build(self):
        args = cli.parse_arguments(
            [
                "cmake",
                "try",
                "-c",
                "--dep",
                "iree::base",
                "-e",
                "int main() { return 0; }",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn(".tmp/iree-cmake-try/run-<pid>/try.cmake", description)
        self.assertIn("cmake -S", description)
        self.assertIn("--target iree_cmake_try_snippet", description)
        self.assertIn("# compile only", description)

    def test_bazel_try_preserves_local_input_paths(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_path = Path(temporary_dir)
            source_dir = temporary_path / "probe"
            source_dir.mkdir()
            (source_dir / "helper.h").write_text(
                "static int helper(void) { return 0; }\n", encoding="utf-8"
            )
            (source_dir / "main.c").write_text(
                '#include "helper.h"\nint main(void) { return helper(); }\n',
                encoding="utf-8",
            )
            scratch_dir = temporary_path / "scratch"
            scratch_dir.mkdir()
            command = bazel_dev.BazelTryCommand(
                files=[
                    Path("probe/main.c"),
                    Path("probe/helper.h"),
                ],
                run_cwd=temporary_path,
            )
            step = bazel_dev.BazelTryStep("bazel", command)

            source_names, _ = step.materialize_sources(scratch_dir)

            self.assertEqual(source_names, ["probe/main.c", "probe/helper.h"])
            self.assertTrue((scratch_dir / "probe/main.c").is_file())
            self.assertTrue((scratch_dir / "probe/helper.h").is_file())

    def test_bazel_try_knows_project_include_roots(self):
        self.assertEqual(
            bazel_dev.header_path_for_include("iree/base/api.h"),
            bazel_dev.REPO_ROOT / "runtime/src/iree/base/api.h",
        )
        self.assertEqual(
            bazel_dev.header_path_for_include("loom/ir/module.h"),
            bazel_dev.REPO_ROOT / "loom/src/loom/ir/module.h",
        )
        self.assertEqual(
            bazel_dev.header_path_for_include("loomc/context.h"),
            bazel_dev.REPO_ROOT / "loom/binding/c/include/loomc/context.h",
        )
        self.assertIsNone(bazel_dev.header_path_for_include("stdio.h"))

    def test_bazel_fuzz_builds_with_fuzzer_config_before_running_binary(self):
        args = cli.parse_arguments(
            [
                "bazel",
                "fuzz",
                "//runtime/src/iree/tokenizer:special_tokens_fuzz",
                "--",
                "-max_total_time=1",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("bazel build --config=fuzzer", description)
        self.assertIn("exec '<built fuzzer>' '<corpus>'", description)

    def test_bazel_compile_commands_defaults_to_repo_roots(self):
        args = cli.parse_arguments(["bazel", "compile-commands"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("# bazel compile-commands", description)
        self.assertIn("bazel build", description)
        self.assertIn(
            "--aspects=//build_tools/bazel:compile_commands.bzl%collect_compile_commands_aspect",
            description,
        )
        self.assertIn("--output_groups=iree_compile_commands_fragments", description)
        self.assertIn("--build_event_json_file=", description)
        self.assertIn("merge compile command fragments", description)
        self.assertIn("//runtime/...", description)
        self.assertIn("//libhrx/...", description)
        self.assertIn("//loom/...", description)
        self.assertIn("compile_commands.json", description)
        self.assertNotIn("--check_visibility=false", description)

    def test_bazel_compile_commands_accepts_options_and_target_patterns(self):
        args = cli.parse_arguments(
            [
                "bazel",
                "compile-commands",
                "--config=asan",
                "-o",
                "out/compile_commands.json",
                "//runtime/...",
                "-//runtime/src/iree/hal/drivers/cuda/...",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--config=asan", description)
        self.assertIn("out/compile_commands.json", description)
        self.assertIn("//runtime/...", description)
        self.assertIn("-//runtime/src/iree/hal/drivers/cuda/...", description)

    def test_bazel_clang_tidy_target_runs_aspect_directly(self):
        args = cli.parse_arguments(
            [
                "bazel",
                "clang-tidy",
                "//runtime/src/iree/vm:all",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("# bazel clang-tidy", description)
        self.assertIn("bazel build", description)
        self.assertIn(
            "--aspects=//build_tools/clang_tidy:clang_tidy.bzl%collect_clang_tidy_aspect",
            description,
        )
        self.assertIn("--output_groups=iree_clang_tidy_reports", description)
        self.assertIn("//runtime/src/iree/vm:all", description)
        self.assertNotIn("build_tools/lefthook/presubmit.py", description)
        self.assertNotIn("--keep_going", description)

    def test_bazel_clang_tidy_ci_target_keeps_going(self):
        args = cli.parse_arguments(
            [
                "bazel",
                "clang-tidy",
                "--profile",
                "ci",
                "//runtime/src/iree/vm:all",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--keep_going", description)
        self.assertIn("//runtime/src/iree/vm:all", description)

    def test_bazel_clang_tidy_git_scope_uses_presubmit_provider(self):
        args = cli.parse_arguments(["bazel", "clang-tidy", "--base", "origin/main"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("build_tools/lefthook/presubmit.py", description)
        self.assertIn("--clang-tidy", description)
        self.assertIn("--base origin/main", description)
        self.assertIn("--profile paranoid", description)
        self.assertNotIn("--static-analysis", description)

    def test_bazel_clang_tidy_explicit_paths_use_presubmit_provider(self):
        args = cli.parse_arguments(
            ["bazel", "clang-tidy", "runtime/src/iree/vm/native_module.c"]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("build_tools/lefthook/presubmit.py", description)
        self.assertIn("--clang-tidy", description)
        self.assertIn("runtime/src/iree/vm/native_module.c", description)

    def test_bazel_clang_tidy_fix_uses_presubmit_provider(self):
        args = cli.parse_arguments(
            [
                "bazel",
                "clang-tidy",
                "--fix",
                "runtime/src/iree/vm/native_module.c",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("build_tools/lefthook/presubmit.py", description)
        self.assertIn("--fix", description)
        self.assertIn("--clang-tidy", description)
        self.assertIn("runtime/src/iree/vm/native_module.c", description)

    def test_bazel_clang_tidy_fix_rejects_target_patterns(self):
        with self.assertRaises(SystemExit):
            cli.parse_arguments(
                ["bazel", "clang-tidy", "--fix", "//runtime/src/iree/vm:all"]
            )

    def test_bazel_clang_tidy_fix_rejects_all_files(self):
        with self.assertRaises(SystemExit):
            cli.parse_arguments(["bazel", "clang-tidy", "--fix", "--all"])

    def test_cmake_compile_commands_prints_configured_database_path(self):
        args = cli.parse_arguments(
            [
                "--cmake-build-dir",
                "build/cmake-debug",
                "cmake",
                "compile-commands",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("# cmake compile-commands", description)
        self.assertIn("build/cmake-debug/compile_commands.json", description)
        self.assertIn("print", description)

    def test_cmake_compile_commands_accepts_output_path(self):
        args = cli.parse_arguments(
            [
                "--cmake-build-dir",
                "build/cmake-debug",
                "cmake",
                "compile-commands",
                "-o",
                "out/cmake_compile_commands.json",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("build/cmake-debug/compile_commands.json", description)
        self.assertIn("out/cmake_compile_commands.json", description)
        self.assertIn("cp", description)

    def test_cmake_clang_tidy_git_scope_uses_presubmit_provider(self):
        args = cli.parse_arguments(
            [
                "--cmake-build-dir",
                "build/cmake-debug",
                "cmake",
                "clang-tidy",
                "--base",
                "origin/main",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("IREE_CMAKE_BUILD_DIR=", description)
        self.assertIn("build/cmake-debug", description)
        self.assertIn("build_tools/lefthook/presubmit.py", description)
        self.assertIn("--lane cmake", description)
        self.assertIn("--clang-tidy", description)
        self.assertIn("--base origin/main", description)
        self.assertNotIn("--static-analysis", description)

    def test_cmake_clang_tidy_explicit_paths_use_presubmit_provider(self):
        args = cli.parse_arguments(
            ["cmake", "clang-tidy", "runtime/src/iree/vm/native_module.c"]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("build_tools/lefthook/presubmit.py", description)
        self.assertIn("--lane cmake", description)
        self.assertIn("--clang-tidy", description)
        self.assertIn("runtime/src/iree/vm/native_module.c", description)

    def test_cmake_clang_tidy_fix_uses_presubmit_provider(self):
        args = cli.parse_arguments(
            [
                "cmake",
                "clang-tidy",
                "--fix",
                "runtime/src/iree/vm/native_module.c",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("build_tools/lefthook/presubmit.py", description)
        self.assertIn("--lane cmake", description)
        self.assertIn("--fix", description)
        self.assertIn("--clang-tidy", description)

    def test_cmake_clang_tidy_rejects_bazel_targets(self):
        with self.assertRaises(SystemExit):
            cli.parse_arguments(["cmake", "clang-tidy", "//runtime/src/iree/vm:all"])

    def test_bazel_fuzz_normalizes_signal_exit_codes(self):
        self.assertEqual(bazel_dev.process_exit_code(-2), 130)
        self.assertEqual(bazel_dev.process_exit_code(7), 7)

    def test_bazel_aliases_include_query_cquery_info(self):
        self.assertEqual(aliases.BAZEL_ALIASES["iree-bazel-query"], ["bazel", "query"])
        self.assertEqual(
            aliases.BAZEL_ALIASES["iree-bazel-cquery"], ["bazel", "cquery"]
        )
        self.assertEqual(aliases.BAZEL_ALIASES["iree-bazel-info"], ["bazel", "info"])
        self.assertNotIn("iree-bazel-compile-commands", aliases.BAZEL_ALIASES)

    def test_cmake_aliases_include_fuzz(self):
        self.assertEqual(aliases.CMAKE_ALIASES["iree-cmake-fuzz"], ["cmake", "fuzz"])

    def test_root_verbose_survives_nested_command_parser(self):
        args = cli.parse_arguments(["--verbose", "bazel", "build"])

        self.assertTrue(args.verbose)

    def test_root_dry_run_survives_nested_command_parser(self):
        args = cli.parse_arguments(["--dry-run", "bazel", "build"])

        self.assertTrue(args.dry_run)

    def test_tool_environment_options_can_precede_passthrough_command(self):
        args = cli.parse_arguments(["--system", "bazel", "build"])

        self.assertTrue(args.system)

        args = cli.parse_arguments(["bazel", "--system", "build"])

        self.assertTrue(args.system)

    def test_passthrough_dry_run_and_verbose_are_wrapper_flags(self):
        args = cli.parse_arguments(["bazel", "build", "-nv", "--config=asan"])

        self.assertTrue(args.dry_run)
        self.assertTrue(args.verbose)
        plan = args.handler(args)
        description = plan.describe()
        self.assertIn("bazel build --config=asan", description)
        self.assertNotIn("-n", description)
        self.assertNotIn("-v", description)

        args = cli.parse_arguments(["bazel", "build", "--verbose"])

        self.assertTrue(args.verbose)
        plan = args.handler(args)
        self.assertNotIn("--verbose", plan.describe())

        args = cli.parse_arguments(["bazel", "build", "--dry-run"])

        self.assertTrue(args.dry_run)
        plan = args.handler(args)
        self.assertNotIn("--dry-run", plan.describe())

    def test_explicit_separator_forwards_conflicting_native_options(self):
        args = cli.parse_arguments(["bazel", "build", "-n", "--", "--dry-run"])

        self.assertTrue(args.dry_run)
        plan = args.handler(args)
        self.assertIn("bazel build --dry-run", plan.describe())

    def test_passthrough_tool_environment_options_are_wrapper_flags(self):
        args = cli.parse_arguments(["bazel", "build", "--system", "//runtime/..."])

        self.assertTrue(args.system)
        plan = args.handler(args)
        self.assertNotIn("--system", plan.describe())

        args = cli.parse_arguments(
            [
                "cmake",
                "build",
                "--cmake-build-dir",
                "build/cmake-debug",
                "hrx",
            ]
        )

        self.assertEqual(args.cmake_build_dir, Path("build/cmake-debug"))
        plan = args.handler(args)
        description = plan.describe()
        self.assertIn("build/cmake-debug", description)
        self.assertNotIn("--cmake-build-dir", description)

    def test_hyphenated_flags_accept_underscore_aliases(self):
        args = cli.parse_arguments(["--dry_run", "bazel", "build"])

        self.assertTrue(args.dry_run)

        args = cli.parse_arguments(
            ["bazel", "setup", "--tool_root", ".tools", "--alias_dir", "aliases"]
        )

        self.assertEqual(args.tool_root, Path(".tools"))
        self.assertEqual(args.alias_dir, Path("aliases"))

    def test_agents_md_survives_hoisted_wrapper_flags(self):
        output = self.parse_agent_md(["bazel", "build", "-n", "--agents_md"])

        self.assertIn("## iree-bazel-build", output)

    def test_bazel_try_keep_has_short_form(self):
        command = bazel_dev.parse_bazel_try_args(
            ["-k", "-e", "int main(void) { return 0; }"]
        )

        self.assertTrue(command.keep)

    def test_root_agents_md_includes_bazel_and_cmake_sections(self):
        output = self.parse_agent_md(["--agents_md"])

        self.assertIn("### Bazel", output)
        self.assertIn("### CMake", output)

    def test_bazel_agents_md_includes_only_bazel_section(self):
        output = self.parse_agent_md(["bazel", "--agents_md"])

        self.assertIn("### Bazel", output)
        self.assertNotIn("### CMake", output)

    def test_cmake_agents_md_includes_only_cmake_section(self):
        output = self.parse_agent_md(["cmake", "--agent_md"])

        self.assertIn("### CMake", output)
        self.assertIn("iree-cmake-build", output)
        self.assertIn("build_tools/bin/iree-*-*", output)
        self.assertNotIn("### Bazel", output)

    def test_bazel_build_agents_md_uses_public_wrapper_names(self):
        output = self.parse_agent_md(["bazel", "build", "--agents_md"])

        self.assertIn("## iree-bazel-build", output)
        self.assertIn("iree-bazel-build //runtime/src/iree/base/...", output)
        self.assertIn("iree-bazel-run", output)
        self.assertIn("iree-bazel-fuzz", output)
        self.assertIn("Wrapper flags", output)
        self.assertNotIn("Pass `--agents-md`", output)
        self.assertNotIn("python dev.py", output)
        self.assertNotIn("### CMake", output)

    def test_bazel_run_agents_md_explains_process_contract(self):
        output = self.parse_agent_md(["bazel", "run", "--agents-md"])

        self.assertIn("## iree-bazel-run", output)
        self.assertIn(
            "iree-bazel-run //runtime/src/tools:iree-run-module -- --help", output
        )
        self.assertIn("execs the binary from the current directory", output)
        self.assertIn("Bazel server lock is not", output)
        self.assertNotIn("python dev.py", output)

    def test_bazel_try_and_fuzz_agents_md_are_focused(self):
        try_output = self.parse_agent_md(["bazel", "try", "--agents-md"])
        fuzz_output = self.parse_agent_md(["bazel", "fuzz", "--agents_md"])

        self.assertIn("## iree-bazel-try", try_output)
        self.assertIn("one-shot C/C++ probes", try_output)
        self.assertIn(".tmp/iree-bazel-try/", try_output)
        self.assertNotIn("## iree-bazel-fuzz", try_output)

        self.assertIn("## iree-bazel-fuzz", fuzz_output)
        self.assertIn("--config=fuzzer", fuzz_output)
        self.assertIn("IREE_FUZZ_CACHE", fuzz_output)
        self.assertNotIn("## iree-bazel-try", fuzz_output)

    def test_cmake_try_agents_md_is_focused(self):
        output = self.parse_agent_md(["cmake", "try", "--agents-md"])

        self.assertIn("## iree-cmake-try", output)
        self.assertIn("one-shot C/C++ probes", output)
        self.assertIn(".tmp/iree-cmake-try/", output)
        self.assertNotIn("## iree-cmake-run", output)

    def test_cmake_fuzz_agents_md_is_focused(self):
        output = self.parse_agent_md(["cmake", "fuzz", "--agents-md"])

        self.assertIn("## iree-cmake-fuzz", output)
        self.assertIn("IREE_ENABLE_FUZZING=ON", output)
        self.assertIn("IREE_FUZZ_CACHE", output)
        self.assertNotIn("## iree-cmake-try", output)

    def test_cmake_build_target_shorthand(self):
        args = cli.parse_arguments(["cmake", "build", "hrx"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--target hrx", description)

    def test_cmake_build_forwards_raw_backend_args_without_separator(self):
        args = cli.parse_arguments(["cmake", "build", "--parallel", "8"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--parallel 8", description)
        self.assertNotIn("--target --parallel", description)

    def test_cmake_build_accepts_targets_before_raw_backend_args(self):
        args = cli.parse_arguments(["cmake", "build", "hrx", "--parallel", "8"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--target hrx", description)

    def test_cmake_fuzz_builds_target_before_execing_binary(self):
        args = cli.parse_arguments(
            [
                "cmake",
                "fuzz",
                "iree::tokenizer::special_tokens_fuzz",
                "--parallel",
                "8",
                "--",
                "-max_total_time=1",
            ]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("cmake fuzz iree::tokenizer::special_tokens_fuzz", description)
        self.assertIn("--target iree::tokenizer::special_tokens_fuzz", description)
        self.assertIn("exec '<built fuzzer>' '<corpus>'", description)
        self.assertIn("--parallel 8", description)

    def test_cmake_test_forwards_options_without_separator(self):
        args = cli.parse_arguments(["cmake", "test", "-R", "hrx"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("-R hrx", description)
        self.assertNotIn("-- -R", description)

    def test_cmake_run_resolves_existing_executable_target(self):
        args = cli.parse_arguments(["cmake", "run", "iree-run-module", "--", "--help"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("# cmake run iree-run-module", description)
        self.assertIn("CMake File API", description)
        self.assertIn("exec '<built executable>' --help", description)

    def test_cmake_run_agents_md_explains_no_implicit_build(self):
        output = self.parse_agent_md(["cmake", "run", "--agents-md"])

        self.assertIn("## iree-cmake-run", output)
        self.assertIn("iree-cmake-build iree-run-module", output)
        self.assertIn("does not build", output)
        self.assertNotIn("python dev.py", output)

    def test_bazel_precommit_defaults_to_changed_paranoid_profile(self):
        args = cli.parse_arguments(["bazel", "precommit"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertEqual(len(plan.steps), 1)
        self.assertIn("--check", plan.steps[0].argv)
        self.assertIn("--changed", description)
        self.assertIn("--lane bazel", description)
        self.assertIn("--profile paranoid", description)

    def test_precommit_profile_can_be_selected(self):
        args = cli.parse_arguments(["bazel", "precommit", "--profile", "default"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertEqual(len(plan.steps), 1)
        self.assertIn("--check", plan.steps[0].argv)
        self.assertIn("--lane bazel", description)
        self.assertIn("--profile default", description)
        self.assertNotIn("--profile paranoid", description)
        self.assertNotIn("--fix", description)

        args = cli.parse_arguments(["cmake", "precommit", "--profile", "paranoid"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertEqual(len(plan.steps), 1)
        self.assertIn("--check", plan.steps[0].argv)
        self.assertIn("--lane cmake", description)
        self.assertIn("--profile paranoid", description)
        self.assertNotIn("--hygiene", description)

    def test_bazel_precommit_can_use_base_ref(self):
        args = cli.parse_arguments(["bazel", "precommit", "--base", "origin/main"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--base origin/main", description)
        self.assertNotIn("--changed", description)

    def test_bazel_precommit_can_use_commit_scope(self):
        args = cli.parse_arguments(["bazel", "precommit", "--commit"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertEqual(len(plan.steps), 2)
        self.assertIn("--fix", plan.steps[0].argv)
        self.assertIn("--hygiene", plan.steps[0].argv)
        self.assertIn("--check", plan.steps[1].argv)
        self.assertNotIn("--hygiene", plan.steps[1].argv)
        self.assertIn("--tests", plan.steps[1].argv)
        self.assertIn("--static-analysis", plan.steps[1].argv)
        self.assertIn("--commit", description)
        self.assertNotIn("--changed", description)

    def test_bazel_precommit_can_use_staged_files(self):
        args = cli.parse_arguments(["bazel", "precommit", "--staged"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertEqual(len(plan.steps), 2)
        self.assertIn("--fix", plan.steps[0].argv)
        self.assertIn("--hygiene", plan.steps[0].argv)
        self.assertIn("--check", plan.steps[1].argv)
        self.assertNotIn("--hygiene", plan.steps[1].argv)
        self.assertIn("--tests", plan.steps[1].argv)
        self.assertIn("--static-analysis", plan.steps[1].argv)
        self.assertIn("--staged", description)
        self.assertNotIn("--changed", description)

    def test_bazel_precommit_can_use_explicit_paths(self):
        args = cli.parse_arguments(["bazel", "precommit", "README.md", "dev.py"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertEqual(len(plan.steps), 2)
        self.assertIn("--fix", plan.steps[0].argv)
        self.assertIn("--hygiene", plan.steps[0].argv)
        self.assertIn("--check", plan.steps[1].argv)
        self.assertNotIn("--hygiene", plan.steps[1].argv)
        self.assertIn("--tests", plan.steps[1].argv)
        self.assertIn("--static-analysis", plan.steps[1].argv)
        self.assertIn("README.md dev.py", description)
        self.assertNotIn("--changed", description)
        self.assertNotIn("--staged", description)

    def test_precommit_rejects_explicit_paths_with_input_mode(self):
        with self.assertRaises(SystemExit):
            cli.parse_arguments(
                ["bazel", "precommit", "--base", "origin/main", "README.md"]
            )

    def test_cmake_precommit_uses_selected_lane(self):
        args = cli.parse_arguments(["cmake", "precommit"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--changed", description)
        self.assertIn("--lane cmake", description)
        self.assertNotIn("--hygiene", description)
        self.assertIn("--profile default", description)

    def test_cmake_precommit_exports_selected_build_dir(self):
        args = cli.parse_arguments(
            ["--cmake-build-dir", "build/cmake-ci", "cmake", "precommit"]
        )

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("IREE_CMAKE_BUILD_DIR=", description)
        self.assertIn("build/cmake-ci", description)

    def test_cmake_presubmit_profile_can_be_selected(self):
        args = cli.parse_arguments(["cmake", "presubmit", "--profile", "paranoid"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--lane cmake", description)
        self.assertIn("--profile paranoid", description)
        self.assertIn("--all", description)
        self.assertNotIn("--hygiene", description)

    def test_bazel_presubmit_uses_full_tree_input(self):
        args = cli.parse_arguments(["bazel", "presubmit", "--profile", "default"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--lane bazel", description)
        self.assertIn("--profile default", description)
        self.assertIn("--all", description)

    def test_bazel_presubmit_can_use_base_ref(self):
        args = cli.parse_arguments(["bazel", "presubmit", "--base", "origin/main"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--lane bazel", description)
        self.assertIn("--base origin/main", description)
        self.assertNotIn("--all", description)

    def test_bazel_presubmit_can_skip_project_tests(self):
        args = cli.parse_arguments(["bazel", "presubmit", "--no_project_tests"])

        plan = args.handler(args)
        description = plan.describe()

        self.assertIn("--no-project-tests", description)

    def test_hook_writes_selected_profile(self):
        args = cli.parse_arguments(["bazel", "hook", "--profile", "ci"])

        plan = args.handler(args)
        step = plan.steps[0]

        self.assertIsInstance(step, WriteFileStep)
        self.assertIn("python dev.py bazel hook --profile ci", step.content)
        self.assertIn(
            "run: python dev.py bazel precommit --profile ci --commit",
            step.content,
        )

    def test_cmake_hook_defaults_to_default_profile(self):
        args = cli.parse_arguments(["cmake", "hook"])

        plan = args.handler(args)
        step = plan.steps[0]

        self.assertIsInstance(step, WriteFileStep)
        self.assertIn(
            "run: python dev.py cmake precommit --profile default --commit",
            step.content,
        )

    def test_cmake_configure_help_explains_user_visible_build_tree(self):
        output = self.parse_help(["cmake", "configure", "--help"])

        self.assertIn("Configure the CMake package/install-test build tree", output)
        self.assertIn("build/cmake", output)
        self.assertIn("IREE_CMAKE_BUILD_DIR", output)
        self.assertNotIn("../builds", output)
        self.assertNotIn("backend configure tool", output)

    def test_root_help_lists_nested_build_system_commands(self):
        output = self.parse_help(["--help"])

        self.assertIn("Common build-system commands", output)
        self.assertIn("python dev.py bazel precommit", output)
        self.assertIn("python dev.py bazel run", output)
        self.assertIn("python dev.py bazel compile-commands", output)
        self.assertIn("python dev.py bazel clang-tidy", output)
        self.assertIn("python dev.py cmake precommit", output)
        self.assertIn("python dev.py cmake run", output)
        self.assertIn("python dev.py cmake compile-commands", output)
        self.assertIn("python dev.py cmake clang-tidy", output)

    def test_bazel_build_help_explains_default_targets(self):
        output = self.parse_help(["bazel", "build", "--help"])

        self.assertIn("Build Bazel targets", output)
        self.assertIn("//runtime/...", output)
        self.assertIn("//libhrx/...", output)
        self.assertNotIn("backend", output.lower())

    def test_precommit_help_explains_changed_file_set(self):
        output = self.parse_help(["bazel", "precommit", "--help"])

        self.assertIn("staged, unstaged, and untracked files", output)
        self.assertIn("--commit", output)
        self.assertIn("--base", output)
        self.assertIn("--staged", output)
        self.assertIn("Explicit paths", output)
        self.assertNotIn("generated Git hook", output)
        self.assertIn("mechanical fixups", output)
        self.assertIn("non-mutating check mode", output)

    def test_presubmit_help_calls_out_full_tree_default(self):
        output = self.parse_help(["bazel", "presubmit", "--help"])

        self.assertIn("The default profile is ci", output)
        self.assertIn("full-tree", output)
        self.assertIn("--base", output)
        self.assertIn("precommit", output)

    def test_bazel_clang_tidy_help_explains_modes(self):
        output = self.parse_help(["bazel", "clang-tidy", "--help"])

        self.assertIn("Bazel target patterns", output)
        self.assertIn("aspect directly", output)
        self.assertIn("--keep_going", output)
        self.assertIn("--base", output)

    def test_cmake_clang_tidy_help_explains_modes(self):
        output = self.parse_help(["cmake", "clang-tidy", "--help"])

        self.assertIn("CMake compile database", output)
        self.assertIn("--cmake-build-dir", output)
        self.assertIn("--base", output)


if __name__ == "__main__":
    unittest.main()
