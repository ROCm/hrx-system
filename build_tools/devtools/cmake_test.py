# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from build_tools.devtools import cmake as cmake_dev
from build_tools.devtools import (
    cmake_file_api,
    cmake_fuzz,
    cmake_try,
    environment,
    fuzz,
)
from build_tools.devtools.environment import REPO_ROOT, ToolEnvironment, ToolMode


class CMakeTest(unittest.TestCase):
    def test_local_tmp_root_defaults_to_repo_tmp(self):
        self.assertEqual(
            environment.local_tmp_root(environ={}),
            REPO_ROOT / ".tmp",
        )

    def test_local_tmp_root_uses_environment_override(self):
        self.assertEqual(
            environment.local_tmp_root(
                environ={environment.DEVTOOLS_TMP_ENV: "/tmp/iree-x"}
            ),
            Path("/tmp/iree-x"),
        )
        self.assertEqual(
            environment.local_tmp_root(
                environ={environment.DEVTOOLS_TMP_ENV: "scratch/dev"}
            ),
            REPO_ROOT / "scratch/dev",
        )

    def test_build_dir_defaults_to_repo_local_cmake_tree(self):
        self.assertEqual(
            cmake_dev.build_dir(
                environ={},
                state_file=Path("/nonexistent/cmake_build_dir"),
            ),
            REPO_ROOT / "build/cmake",
        )

    def test_build_dir_resolves_relative_paths_from_repo_root(self):
        self.assertEqual(
            cmake_dev.build_dir(Path("build/cmake-debug")),
            REPO_ROOT / "build/cmake-debug",
        )

    def test_build_dir_preserves_absolute_paths(self):
        self.assertEqual(cmake_dev.build_dir(Path("/tmp/cmake")), Path("/tmp/cmake"))

    def test_build_dir_prefers_environment_over_recorded_state(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            state_file = Path(temporary_dir) / "cmake_build_dir"
            state_file.write_text(
                str(Path(temporary_dir) / "recorded"),
                encoding="utf-8",
            )

            self.assertEqual(
                cmake_dev.build_dir(
                    environ={"IREE_CMAKE_BUILD_DIR": "build/from-env"},
                    state_file=state_file,
                ),
                REPO_ROOT / "build/from-env",
            )

    def test_build_dir_uses_recorded_state(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            recorded_build_dir = Path(temporary_dir) / "configured"
            state_file = Path(temporary_dir) / "cmake_build_dir"
            state_file.write_text(str(recorded_build_dir), encoding="utf-8")

            self.assertEqual(
                cmake_dev.build_dir(environ={}, state_file=state_file),
                recorded_build_dir,
            )

    def test_build_args_translate_target_shorthand(self):
        self.assertEqual(
            cmake_dev.build_args(["hrx", "iree-run-module", "--parallel", "8"]),
            ["--target", "hrx", "--target", "iree-run-module", "--parallel", "8"],
        )

    def test_default_fuzz_cache_dir_uses_platform_cache_roots(self):
        self.assertEqual(
            fuzz.default_fuzz_cache_dir(
                environ={"XDG_CACHE_HOME": "/cache"},
                platform="linux",
                home=Path("/home/user"),
            ),
            Path("/cache/iree-fuzz-cache"),
        )
        self.assertEqual(
            fuzz.default_fuzz_cache_dir(
                environ={"LOCALAPPDATA": "C:/Users/user/AppData/Local"},
                platform="win32",
                home=Path("C:/Users/user"),
            ),
            Path("C:/Users/user/AppData/Local/iree-fuzz-cache"),
        )
        self.assertEqual(
            fuzz.default_fuzz_cache_dir(
                environ={},
                platform="darwin",
                home=Path("/Users/user"),
            ),
            Path("/Users/user/Library/Caches/iree-fuzz-cache"),
        )

    def test_default_fuzz_cache_dir_respects_environment_override(self):
        self.assertEqual(
            fuzz.default_fuzz_cache_dir(
                environ={"IREE_FUZZ_CACHE": "custom-corpus"},
                platform="linux",
                home=Path("/home/user"),
            ),
            Path("custom-corpus"),
        )

    def test_build_args_translate_configured_aliases(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir)
            alias_path = build_dir / ".iree/target_aliases.json"
            alias_path.parent.mkdir()
            alias_path.write_text(
                json.dumps(
                    {
                        "iree::tools::iree-run-module": (
                            "runtime_src_tools_iree-run-module"
                        )
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                cmake_dev.build_args(
                    ["iree::tools::iree-run-module", "--parallel", "8"],
                    configured_build_dir=build_dir,
                ),
                ["--target", "runtime_src_tools_iree-run-module", "--parallel", "8"],
            )

    def test_configure_build_and_test_plans_use_selected_build_dir(self):
        tool_env = ToolEnvironment(ToolMode.SYSTEM, None)

        configure_plan = cmake_dev.configure_plan(
            tool_env,
            configured_build_dir=Path("build/cmake-debug"),
            backend_args=["-DIREE_HAL_DRIVER_AMDGPU=OFF"],
        )
        configure_description = configure_plan.describe()
        self.assertIn("build/cmake-debug", configure_description)
        self.assertIn("codemodel-v2", configure_description)
        self.assertIn("record CMake build directory", configure_description)
        self.assertIn("-DIREE_HAL_DRIVER_AMDGPU=OFF", configure_description)

        build_plan = cmake_dev.build_plan(
            tool_env,
            configured_build_dir=Path("build/cmake-debug"),
            backend_args=["hrx"],
        )
        self.assertIn("--target hrx", build_plan.describe())

        test_plan = cmake_dev.test_plan(
            tool_env,
            configured_build_dir=Path("build/cmake-debug"),
            backend_args=["-R", "hrx"],
        )
        self.assertIn("ctest", test_plan.describe())
        self.assertIn("-R hrx", test_plan.describe())

    def test_run_plan_resolves_target_with_cmake_file_api(self):
        tool_env = ToolEnvironment(ToolMode.SYSTEM, None)

        plan = cmake_dev.run_plan(
            tool_env,
            configured_build_dir=Path("build/cmake-debug"),
            backend_args=["iree-run-module", "--", "--help"],
        )
        description = plan.describe()

        self.assertIn("# cmake run iree-run-module", description)
        self.assertIn("CMake File API", description)
        self.assertIn("exec '<built executable>' --help", description)

    def test_run_parse_supports_print_path(self):
        command = cmake_dev.parse_run_args(["-p", "iree-run-module"])

        self.assertTrue(command.print_path)
        self.assertEqual(command.target, "iree-run-module")

    def test_compile_commands_plan_prints_configured_database_path(self):
        tool_env = ToolEnvironment(ToolMode.SYSTEM, None)

        plan = cmake_dev.compile_commands_plan(
            tool_env,
            configured_build_dir=Path("build/cmake-debug"),
            backend_args=[],
        )

        description = plan.describe()
        self.assertIn("# cmake compile-commands", description)
        self.assertIn("build/cmake-debug/compile_commands.json", description)
        self.assertIn("print", description)

    def test_compile_commands_step_prints_existing_database_path(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir) / "build"
            compile_commands = build_dir / "compile_commands.json"
            compile_commands.parent.mkdir()
            compile_commands.write_text("[]\n", encoding="utf-8")
            step = cmake_dev.CMakeCompileCommandsStep(
                cmake_dev.CMakeCompileCommandsCommand(),
                build_dir,
            )
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                result = step.run()

            self.assertEqual(result, 0)
            self.assertEqual(str(compile_commands), output.getvalue().strip())

    def test_compile_commands_step_copies_to_requested_output_path(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            build_dir = root / "build"
            compile_commands = build_dir / "compile_commands.json"
            destination = root / "out" / "compile_commands.json"
            compile_commands.parent.mkdir()
            compile_commands.write_text("[{}]\n", encoding="utf-8")
            step = cmake_dev.CMakeCompileCommandsStep(
                cmake_dev.CMakeCompileCommandsCommand(output=destination),
                build_dir,
            )
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                result = step.run()

            self.assertEqual(result, 0)
            self.assertEqual(str(destination), output.getvalue().strip())
            self.assertEqual("[{}]\n", destination.read_text(encoding="utf-8"))

    def test_fuzz_parse_splits_build_and_fuzzer_args(self):
        command = cmake_fuzz.parse_fuzz_args(
            [
                "iree::tokenizer::special_tokens_fuzz",
                "--parallel",
                "8",
                "--",
                "-max_total_time=1",
            ]
        )

        self.assertEqual(command.target, "iree::tokenizer::special_tokens_fuzz")
        self.assertEqual(command.build_args, ["--parallel", "8"])
        self.assertEqual(command.fuzzer_args, ["-max_total_time=1"])

    def test_fuzz_plan_builds_then_execs_fuzzer(self):
        tool_env = ToolEnvironment(ToolMode.SYSTEM, None)

        plan = cmake_dev.fuzz_plan(
            tool_env,
            configured_build_dir=Path("build/cmake-fuzz"),
            backend_args=[
                "iree::tokenizer::special_tokens_fuzz",
                "--parallel",
                "8",
                "--",
                "-max_total_time=1",
            ],
        )
        description = plan.describe()

        self.assertIn("# cmake fuzz iree::tokenizer::special_tokens_fuzz", description)
        self.assertIn("--target iree::tokenizer::special_tokens_fuzz", description)
        self.assertIn("--parallel 8", description)
        self.assertIn("exec '<built fuzzer>' '<corpus>'", description)

    def test_fuzz_requires_fuzz_enabled_cache(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir)
            (build_dir / "CMakeCache.txt").write_text(
                "IREE_ENABLE_FUZZING:BOOL=OFF\n",
                encoding="utf-8",
            )

            self.assertFalse(
                cmake_fuzz.cache_bool_enabled(build_dir, "IREE_ENABLE_FUZZING")
            )

            (build_dir / "CMakeCache.txt").write_text(
                "IREE_ENABLE_FUZZING:BOOL=ON\n",
                encoding="utf-8",
            )

            self.assertTrue(
                cmake_fuzz.cache_bool_enabled(build_dir, "IREE_ENABLE_FUZZING")
            )

    def test_try_parse_supports_snippet_options(self):
        command = cmake_try.parse_try_args(
            [
                "-c",
                "--dep",
                "iree::base",
                "--no-infer",
                "-x",
                "c++",
                "-e",
                "int main() { return 0; }",
                "--",
                "--flag",
            ]
        )

        self.assertTrue(command.compile_only)
        self.assertEqual(command.explicit_deps, ["iree::base"])
        self.assertFalse(command.infer_deps)
        self.assertEqual(command.language, "c++")
        self.assertEqual(command.inline_sources, ["int main() { return 0; }"])
        self.assertEqual(command.program_args, ["--flag"])

    def test_try_plan_configures_and_builds_scratch_tree(self):
        tool_env = ToolEnvironment(ToolMode.SYSTEM, None)

        plan = cmake_dev.try_plan(
            tool_env,
            configured_build_dir=Path("build/cmake-debug"),
            backend_args=[
                "-c",
                "--dep",
                "iree::base",
                "-e",
                "int main() { return 0; }",
            ],
        )
        description = plan.describe()

        self.assertIn(".tmp/iree-cmake-try/run-<pid>/try.cmake", description)
        self.assertIn("-DIREE_CMAKE_TRY_FILE=", description)
        self.assertIn("--target iree_cmake_try_snippet", description)
        self.assertIn("# compile only", description)

    def test_try_preserves_local_input_paths(self):
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
            command = cmake_try.CMakeTryCommand(
                files=[
                    Path("probe/main.c"),
                    Path("probe/helper.h"),
                ],
                run_cwd=temporary_path,
            )
            step = cmake_try.CMakeTryStep("cmake", command, temporary_path)

            source_names, _ = step.materialize_sources(scratch_dir)

            self.assertEqual(source_names, ["probe/main.c", "probe/helper.h"])
            self.assertTrue((scratch_dir / "probe/main.c").is_file())
            self.assertTrue((scratch_dir / "probe/helper.h").is_file())

    def test_try_infers_deps_from_configured_aliases(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir)
            alias_path = cmake_file_api.target_aliases_path(build_dir)
            alias_path.parent.mkdir(parents=True)
            alias_path.write_text(
                json.dumps(
                    {
                        "iree::base": "iree_base_base",
                        "hrx::hrx": "libhrx_src_libhrx_hrx",
                    }
                ),
                encoding="utf-8",
            )
            command = cmake_try.CMakeTryCommand(
                inline_sources=[
                    '#include "iree/base/api.h"\n'
                    '#include "hrx_runtime.h"\n'
                    "int main(void) { return 0; }"
                ],
            )
            step = cmake_try.CMakeTryStep("cmake", command, build_dir)

            self.assertEqual(
                step.deps_for_sources(command.inline_sources),
                ["hrx::hrx", "iree::base"],
            )

    def test_try_cache_copy_preserves_project_configuration(self):
        entries = [
            cmake_try.CMakeCacheEntry("IREE_HAL_DRIVER_AMDGPU", "BOOL", "OFF"),
            cmake_try.CMakeCacheEntry("LIBHRX_BUILD", "BOOL", "OFF"),
            cmake_try.CMakeCacheEntry("CMAKE_C_COMPILER", "FILEPATH", "/bin/cc"),
            cmake_try.CMakeCacheEntry("BENCHMARK_ENABLE_TESTING", "BOOL", "OFF"),
            cmake_try.CMakeCacheEntry("CMAKE_GENERATOR", "INTERNAL", "Ninja"),
        ]

        with tempfile.TemporaryDirectory() as temporary_dir:
            cache_path = Path(temporary_dir) / "cache.cmake"
            cmake_try.write_initial_cache_file(cache_path, entries)
            cache_text = cache_path.read_text(encoding="utf-8")

        self.assertIn("IREE_HAL_DRIVER_AMDGPU", cache_text)
        self.assertIn("LIBHRX_BUILD", cache_text)
        self.assertIn("CMAKE_C_COMPILER", cache_text)
        self.assertNotIn("BENCHMARK_ENABLE_TESTING", cache_text)
        self.assertNotIn("CMAKE_GENERATOR", cache_text)

    def test_try_reads_generator_args_from_configured_cache(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            build_dir = Path(temporary_dir)
            (build_dir / "CMakeCache.txt").write_text(
                "CMAKE_GENERATOR:INTERNAL=Ninja\n"
                "CMAKE_GENERATOR_PLATFORM:INTERNAL=x64\n"
                "CMAKE_GENERATOR_TOOLSET:INTERNAL=host=x64\n",
                encoding="utf-8",
            )

            self.assertEqual(
                cmake_try.cmake_generator_args(build_dir),
                ["-G", "Ninja", "-A", "x64", "-T", "host=x64"],
            )

    def test_try_cmake_file_links_deps(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            cmake_path = Path(temporary_dir) / "try.cmake"
            cmake_try.write_try_cmake_file(
                cmake_path,
                source_paths=[Path("/tmp/snippet.c")],
                deps=["iree::base"],
            )
            cmake_text = cmake_path.read_text(encoding="utf-8")

        self.assertIn("add_executable(iree_cmake_try_snippet", cmake_text)
        self.assertIn(
            "target_link_libraries(iree_cmake_try_snippet PRIVATE", cmake_text
        )
        self.assertIn("iree::base", cmake_text)


if __name__ == "__main__":
    unittest.main()
