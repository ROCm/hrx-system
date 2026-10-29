# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from argparse import Namespace
from pathlib import Path
from unittest import mock

from build_tools.ci import ci_core_common, ci_core_windows

REPO_ROOT = Path(__file__).resolve().parents[2]


class CiCoreWindowsTest(unittest.TestCase):
    def test_diagnostic_command_preserves_failure_and_success_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = dict(os.environ)
            environment["IREE_CI_FAILURE_ARTIFACT_DIR"] = str(root / "artifacts")
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError) as failure:
                    ci_core_windows.run(
                        [sys.executable, "-c", "raise SystemExit(17)"],
                        cwd=root,
                        env=environment,
                    )
                self.assertEqual(failure.exception.returncode, 17)
                ci_core_windows.run(
                    [sys.executable, "-c", "print('success')"],
                    cwd=root,
                    env=environment,
                )
            commands = [
                json.loads(path.read_text())
                for path in (root / "artifacts").glob("*/command.json")
            ]
            self.assertEqual(
                sorted(command["returncode"] for command in commands), [0, 17]
            )

    def _symlink_or_skip(self, source: Path, link: Path) -> None:
        try:
            link.symlink_to(source, target_is_directory=source.is_dir())
        except OSError as exc:
            self.skipTest(f"symlink creation unavailable: {exc}")

    def test_script_runs_by_path_without_pythonpath(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, "build_tools/ci/ci_core_windows.py", "--help"],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        self.assertIn("fetch-rocm", result.stdout)

    def test_copy_tree_contents_materializes_file_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            real_file = source / "real.py"
            real_file.write_text("value = 1\n")
            self._symlink_or_skip(real_file, source / "link.py")

            ci_core_common.copy_tree_contents(source, target, preserve_symlinks=False)

            copied_link = target / "link.py"
            self.assertFalse(copied_link.is_symlink())
            self.assertEqual(copied_link.read_text(), "value = 1\n")

    def test_copy_tree_contents_materializes_file_symlink_when_resolve_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            target = root / "target"
            source.mkdir()
            real_file = source / "real.py"
            real_file.write_text("value = 1\n")
            link_file = source / "link.py"
            self._symlink_or_skip(real_file, link_file)
            path_type = type(link_file)
            original_resolve = path_type.resolve

            def resolve(self, *args, **kwargs):
                if self == link_file:
                    raise OSError("realpath failed")
                return original_resolve(self, *args, **kwargs)

            with mock.patch.object(path_type, "resolve", resolve):
                ci_core_common.copy_tree_contents(
                    source, target, preserve_symlinks=False
                )

            copied_link = target / "link.py"
            self.assertFalse(copied_link.is_symlink())
            self.assertEqual(copied_link.read_text(), "value = 1\n")

    def test_copy_tree_contents_materializes_directory_symlinks_into_overlay(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            real_package = root / "real_package"
            real_package.mkdir()
            (real_package / "module.py").write_text("value = 1\n")
            source = root / "source"
            source.mkdir()
            self._symlink_or_skip(real_package, source / "package")
            target = root / "target"
            (target / "package").mkdir(parents=True)
            (target / "package" / "existing.py").write_text("value = 0\n")

            ci_core_common.copy_tree_contents(source, target, preserve_symlinks=False)

            copied_package = target / "package"
            self.assertFalse(copied_package.is_symlink())
            self.assertEqual(
                (copied_package / "existing.py").read_text(), "value = 0\n"
            )
            self.assertEqual((copied_package / "module.py").read_text(), "value = 1\n")

    def test_create_zip_materializes_file_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source"
            source.mkdir()
            real_file = source / "real.py"
            fixture_contents = b"value = 1\n"
            real_file.write_bytes(fixture_contents)
            self._symlink_or_skip(real_file, source / "link.py")
            zip_path = root / "archive.zip"

            ci_core_windows.create_zip(source, zip_path)

            with zipfile.ZipFile(zip_path) as zf:
                self.assertEqual(zf.read("link.py"), fixture_contents)

    def test_s3_cache_path_preserves_artifact_identity(self):
        cache_root = Path("C:/cache")

        self.assertEqual(
            ci_core_windows.s3_cache_path(
                cache_root,
                "therock-nightly-artifacts",
                "123-windows/core-runtime_lib_generic.tar.zst",
            ),
            cache_root
            / "therock-nightly-artifacts"
            / "123-windows"
            / "core-runtime_lib_generic.tar.zst",
        )
        self.assertNotEqual(
            ci_core_windows.s3_cache_path(
                cache_root,
                "therock-nightly-artifacts",
                "123-windows/core-runtime_lib_generic.tar.zst",
            ),
            ci_core_windows.s3_cache_path(
                cache_root,
                "therock-nightly-artifacts",
                "456-windows/core-runtime_lib_generic.tar.zst",
            ),
        )

    def test_s3_cache_path_rejects_unsafe_keys(self):
        with self.assertRaisesRegex(RuntimeError, "Unsafe S3 key"):
            ci_core_windows.s3_cache_path(Path("C:/cache"), "bucket", "../evil")

    def test_windows_core_artifact_set_matches_windows_packaging(self):
        self.assertEqual(
            ci_core_common.wanted_artifacts(
                "core", artifact_sets=ci_core_windows.ARTIFACT_SETS
            ),
            [
                "sysdeps_lib_generic",
                "sysdeps_dev_generic",
                "base_lib_generic",
                "base_run_generic",
                "base_dev_generic",
                "amd-llvm_lib_generic",
                "amd-llvm_run_generic",
            ],
        )

    def test_windows_upstream_hip_artifact_set_is_explicit(self):
        self.assertEqual(
            ci_core_common.wanted_artifacts(
                "core-with-upstream-hip", artifact_sets=ci_core_windows.ARTIFACT_SETS
            ),
            [
                "sysdeps_lib_generic",
                "sysdeps_dev_generic",
                "base_lib_generic",
                "base_run_generic",
                "base_dev_generic",
                "amd-llvm_lib_generic",
                "amd-llvm_run_generic",
                "amd-llvm_dev_generic",
                "core-hip_lib_generic",
                "core-hip_run_generic",
                "core-hip_dev_generic",
                "core-hipinfo_run_generic",
                "core-kpack_lib_generic",
                "core-kpack_dev_generic",
            ],
        )

    def test_fetch_rocm_uses_windows_artifact_set(self):
        called_kwargs = {}

        def fake_fetch_rocm(_args, **kwargs):
            called_kwargs.update(kwargs)

        old_fetch_rocm = ci_core_windows.common.fetch_rocm
        try:
            ci_core_windows.common.fetch_rocm = fake_fetch_rocm
            ci_core_windows.fetch_rocm(Namespace())
        finally:
            ci_core_windows.common.fetch_rocm = old_fetch_rocm

        self.assertIs(called_kwargs["artifact_sets"], ci_core_windows.ARTIFACT_SETS)

    def test_windows_package_names(self):
        self.assertEqual(
            ci_core_windows.PACKAGE_NAMES,
            [
                "hrx-public-windows-x86_64",
                "hrx-public-deps-windows-x86_64",
                "hrx-rocm-buildenv-windows-x86_64",
            ],
        )

    def test_windows_public_deps_do_not_ship_rocm_runtime_artifacts(self):
        self.assertEqual(ci_core_windows.PUBLIC_DEPS_REQUIRED_GLOBS, [])
        self.assertEqual(ci_core_windows.PUBLIC_DEPS_OPTIONAL_GLOBS, [])

    def test_windows_rocm_dependency_mode_allows_pinned_header_fallback(self):
        self.assertEqual(ci_core_windows.WINDOWS_ROCM_DEPENDENCY_MODE, "auto")

    def test_windows_public_component_defaults_to_hrx_public_dist(self):
        parser = argparse.ArgumentParser()
        with mock.patch.dict(os.environ, {}, clear=True):
            ci_core_windows.add_shared_args(parser)

            args = parser.parse_args([])

        self.assertEqual(args.public_component, "HrxPublicDist")

    def test_windows_executable_name(self):
        self.assertEqual(
            ci_core_windows.windows_executable_name("hrx-info"), "hrx-info.exe"
        )
        self.assertEqual(
            ci_core_windows.windows_executable_name("tool.exe"), "tool.exe"
        )

    def test_cmake_path_normalizes_windows_backslashes(self):
        self.assertEqual(
            ci_core_windows.cmake_path(
                r"C:\Program Files (x86)\Windows Kits\10\bin\x64\rc.exe"
            ),
            "C:/Program Files (x86)/Windows Kits/10/bin/x64/rc.exe",
        )

    def test_cmake_path_list_uses_cmake_separators(self):
        self.assertEqual(
            ci_core_windows.cmake_path_list([r"C:\a\b", r"D:\c\d"]),
            "C:/a/b;D:/c/d",
        )

    def test_windows_sdk_tool_uses_latest_sdk_env_path(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            old_rc = (
                root / "Windows Kits" / "10" / "bin" / "10.0.22000.0" / "x64" / "rc.exe"
            )
            new_rc = (
                root / "Windows Kits" / "10" / "bin" / "10.0.26100.0" / "x64" / "rc.exe"
            )
            old_rc.parent.mkdir(parents=True)
            new_rc.parent.mkdir(parents=True)
            old_rc.touch()
            new_rc.touch()
            env = {
                "PATH": "",
                "PROCESSOR_ARCHITECTURE": "AMD64",
                "ProgramFiles(x86)": str(root),
            }

            self.assertEqual(ci_core_windows.windows_sdk_tool("rc.exe", env), new_rc)

    def test_find_tool_in_path_uses_case_insensitive_path_env(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rc = root / "rc.exe"
            rc.touch()
            rc.chmod(0o755)

            self.assertEqual(
                ci_core_windows.find_tool_in_path(["rc.exe"], {"Path": str(root)}),
                rc,
            )

    def test_msvc_build_env_tries_next_vcvarsall_candidate(self):
        tool_dir = Path("C:/sdk/bin")
        calls = []

        def fake_vcvarsall_candidates(_env):
            return [Path("bad.bat"), Path("good.bat")]

        def fake_run_vcvarsall(path, _env):
            calls.append(path)
            if path.name == "bad.bat":
                raise RuntimeError("bad setup failed")
            return {"Path": os.fspath(tool_dir), "VSCMD_ARG_TGT_ARCH": "x64"}

        def fake_find_tool_in_path(names, env=None):
            if ci_core_windows.env_get(env or {}, "PATH") != os.fspath(tool_dir):
                return None
            return tool_dir / next(iter(names))

        old_os_name = ci_core_windows.os.name
        old_cache = ci_core_windows._MSVC_BUILD_ENV_CACHE
        old_candidates = ci_core_windows.vcvarsall_candidates
        old_run_vcvarsall = ci_core_windows.run_vcvarsall
        old_find_tool_in_path = ci_core_windows.find_tool_in_path
        try:
            ci_core_windows.os.name = "nt"
            ci_core_windows._MSVC_BUILD_ENV_CACHE = None
            ci_core_windows.vcvarsall_candidates = fake_vcvarsall_candidates
            ci_core_windows.run_vcvarsall = fake_run_vcvarsall
            ci_core_windows.find_tool_in_path = fake_find_tool_in_path

            env = ci_core_windows.msvc_build_env({})
        finally:
            ci_core_windows.os.name = old_os_name
            ci_core_windows._MSVC_BUILD_ENV_CACHE = old_cache
            ci_core_windows.vcvarsall_candidates = old_candidates
            ci_core_windows.run_vcvarsall = old_run_vcvarsall
            ci_core_windows.find_tool_in_path = old_find_tool_in_path

        self.assertEqual([os.fspath(path) for path in calls], ["bad.bat", "good.bat"])
        self.assertEqual(env["VSCMD_ARG_TGT_ARCH"], "x64")

    def test_vcvarsall_wrapper_quotes_batch_path_without_backslash_escapes(self):
        content = ci_core_windows.make_vcvarsall_wrapper_content(
            Path("C:/Program Files/Microsoft Visual Studio/vcvarsall.bat")
        )

        self.assertIn(
            'call "C:/Program Files/Microsoft Visual Studio/vcvarsall.bat" x64',
            content,
        )
        self.assertNotIn(r"\"", content)

    def test_windows_toolchain_resolves_clang_cl_tuple(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            llvm_bin = root / "lib" / "llvm" / "bin"
            llvm_bin.mkdir(parents=True)
            for tool in ["clang-cl.exe", "llvm-lib.exe", "lld-link.exe"]:
                (llvm_bin / tool).touch()
            rc = root / "sdk" / "rc.exe"
            mt = root / "sdk" / "mt.exe"
            rc.parent.mkdir()
            rc.touch()
            mt.touch()
            args = Namespace(windows_toolchain="clang-cl")

            old_find_tool_in_path = ci_core_windows.find_tool_in_path
            try:

                def fake_find_tool_in_path(names, env=None):
                    if "rc.exe" in names:
                        return rc
                    if "mt.exe" in names:
                        return mt
                    return None

                ci_core_windows.find_tool_in_path = fake_find_tool_in_path
                toolchain = ci_core_windows.windows_toolchain(args, root)
            finally:
                ci_core_windows.find_tool_in_path = old_find_tool_in_path

        self.assertEqual(toolchain.name, "clang-cl")
        self.assertEqual(toolchain.c_compiler, llvm_bin / "clang-cl.exe")
        self.assertEqual(toolchain.cxx_compiler, llvm_bin / "clang-cl.exe")
        self.assertEqual(toolchain.asm_compiler, llvm_bin / "clang-cl.exe")
        self.assertEqual(toolchain.ar, llvm_bin / "llvm-lib.exe")
        self.assertEqual(toolchain.linker, llvm_bin / "lld-link.exe")
        self.assertEqual(toolchain.rc, rc)
        self.assertEqual(toolchain.mt, mt)

    def test_windows_toolchain_resolves_msvc_tuple(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cl = root / "msvc" / "cl.exe"
            lib = root / "msvc" / "lib.exe"
            link = root / "msvc" / "link.exe"
            rc = root / "sdk" / "rc.exe"
            mt = root / "sdk" / "mt.exe"
            cl.parent.mkdir()
            rc.parent.mkdir()
            for path in [cl, lib, link, rc, mt]:
                path.touch()
            args = Namespace(windows_toolchain="msvc")

            old_find_tool_in_path = ci_core_windows.find_tool_in_path
            try:

                def fake_find_tool_in_path(names, env=None):
                    mapping = {
                        "cl.exe": cl,
                        "lib.exe": lib,
                        "link.exe": link,
                        "rc.exe": rc,
                        "mt.exe": mt,
                    }
                    for name in names:
                        if name in mapping:
                            return mapping[name]
                    return None

                ci_core_windows.find_tool_in_path = fake_find_tool_in_path
                toolchain = ci_core_windows.windows_toolchain(args, root)
            finally:
                ci_core_windows.find_tool_in_path = old_find_tool_in_path

        self.assertEqual(toolchain.name, "msvc")
        self.assertEqual(toolchain.c_compiler, cl)
        self.assertEqual(toolchain.cxx_compiler, cl)
        self.assertEqual(toolchain.asm_compiler, cl)
        self.assertEqual(toolchain.ar, lib)
        self.assertEqual(toolchain.linker, link)
        self.assertEqual(toolchain.rc, rc)
        self.assertEqual(toolchain.mt, mt)

    def test_windows_toolchain_rejects_unknown_name(self):
        with self.assertRaisesRegex(ValueError, "Unsupported Windows toolchain"):
            ci_core_windows.windows_toolchain(
                Namespace(windows_toolchain="icc"), Path("C:/rocm")
            )

    def test_build_core_configures_reduced_windows_profile(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rocm_root = root / "rocm"
            rocm_root.mkdir()
            args = Namespace(
                rocm_root=rocm_root,
                build_dir=root / "build",
                public_install_dir=root / "public",
                build_type="RelWithDebInfo",
                passthrough=False,
                assertions=False,
                sanitizer="",
                cmake_option=[],
                target="all",
                windows_toolchain="msvc",
                public_component="IREEPackage",
                package=True,
                package_smoke=True,
            )
            toolchain = ci_core_windows.WindowsToolchain(
                name="msvc",
                c_compiler=root / "msvc" / "cl.exe",
                cxx_compiler=root / "msvc" / "cl.exe",
                asm_compiler=root / "msvc" / "cl.exe",
                ar=root / "msvc" / "lib.exe",
                linker=root / "msvc" / "link.exe",
                rc=root / "sdk" / "rc.exe",
                mt=root / "sdk" / "mt.exe",
            )
            commands = []

            old_rocm_build_env = ci_core_windows.rocm_build_env
            old_windows_toolchain = ci_core_windows.windows_toolchain
            old_run = ci_core_windows.run
            try:
                ci_core_windows.rocm_build_env = lambda _rocm_root: {}
                ci_core_windows.windows_toolchain = lambda _args, _rocm_root, _env: (
                    toolchain
                )
                ci_core_windows.run = lambda cmd, **_kwargs: commands.append(
                    [os.fspath(arg) for arg in cmd]
                )

                ci_core_windows.build_core(args)
            finally:
                ci_core_windows.rocm_build_env = old_rocm_build_env
                ci_core_windows.windows_toolchain = old_windows_toolchain
                ci_core_windows.run = old_run

        configure_cmd = commands[0]
        self.assertIn(
            "-DCMAKE_C_COMPILER=" + ci_core_windows.cmake_path(toolchain.c_compiler),
            configure_cmd,
        )
        self.assertIn(
            "-DCMAKE_ASM_COMPILER="
            + ci_core_windows.cmake_path(toolchain.asm_compiler),
            configure_cmd,
        )
        self.assertFalse(
            any(option.startswith("-DIREE_CLANG_BINARY=") for option in configure_cmd)
        )
        self.assertIn("-DIREE_BUILD_BENCHMARKS=OFF", configure_cmd)
        self.assertIn("-DLOOM_BUILD=OFF", configure_cmd)
        self.assertIn("-DIREE_HAL_DRIVER_AMD=ON", configure_cmd)
        self.assertIn("-DAMDF_BUILD=ON", configure_cmd)
        self.assertIn("-DLIBHRX_BUILD=ON", configure_cmd)
        self.assertIn("-DLIBHRX_BUILD_CTS=OFF", configure_cmd)
        self.assertIn("-DLIBHRX_BUILD_HIP_BINDING=OFF", configure_cmd)
        self.assertIn("-DLIBHRX_BUILD_PASSTHROUGH=OFF", configure_cmd)
        self.assertIn("-DIREE_HAL_DRIVER_AMDGPU=OFF", configure_cmd)

    def test_test_core_builds_the_selected_source_ctest_closure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rocm_root = root / "rocm"
            rocm_root.mkdir()
            build_dir = root / "build"
            build_dir.mkdir()
            (build_dir / "CTestTestfile.cmake").touch()
            parser = argparse.ArgumentParser()
            with mock.patch.dict(os.environ, {}, clear=True):
                ci_core_windows.add_shared_args(parser)
                args = parser.parse_args(["--no-package-smoke"])
            args.rocm_root = rocm_root
            args.build_dir = build_dir
            args.ctest_regex = "^hrx/"
            args.ctest_parallelism = 2
            args.ctest_label_exclude_regex = "manual"
            selected_test_step = mock.Mock()
            selected_test_step.run.return_value = 0
            with (
                mock.patch.object(
                    ci_core_windows,
                    "rocm_build_env",
                    return_value={"PATH": "toolchain"},
                ),
                mock.patch.object(
                    ci_core_common.ctest_dev,
                    "CTestBuildAndRunStep",
                    return_value=selected_test_step,
                ) as step_type,
            ):
                ci_core_windows.test_core(args)
            self.assertEqual(step_type.call_args.kwargs["build_dir"], build_dir)
            self.assertEqual(step_type.call_args.kwargs["env"]["PATH"], "toolchain")
            self.assertEqual(
                step_type.call_args.kwargs["arguments"],
                [
                    "--parallel",
                    "2",
                    "-R",
                    "^hrx/",
                    "-E",
                    "(^build_tools/cmake/)",
                    "-LE",
                    "manual",
                    "--no-tests=error",
                ],
            )
            selected_test_step.run.assert_called_once_with(verbose=True)
            selected_test_step.run.return_value = 17
            with (
                mock.patch.object(ci_core_windows, "rocm_build_env", return_value={}),
                mock.patch.object(
                    ci_core_common.ctest_dev,
                    "CTestBuildAndRunStep",
                    return_value=selected_test_step,
                ),
                self.assertRaises(subprocess.CalledProcessError) as failure,
            ):
                ci_core_windows.test_core(args)
            self.assertEqual(failure.exception.returncode, 17)

    def test_generated_package_scan_rejects_original_roots(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            build_root = root / "build"
            package_root = root / "public"
            config_file = package_root / "lib/cmake/hrx/hrx-config.cmake"
            config_file.parent.mkdir(parents=True)
            config_file.write_text(
                f'set(IMPORT_PATH "{ci_core_windows.cmake_path(build_root)}")\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "captured"):
                ci_core_windows.scan_generated_package_files(
                    [package_root],
                    forbidden_paths=[build_root],
                )

    def test_package_core_accepts_libhrx_public_install_without_loomc(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rocm_root = root / "rocm"
            public_install_dir = root / "public"
            public_deps_dir = root / "public-deps"
            (public_install_dir / "bin").mkdir(parents=True)
            (public_install_dir / "bin" / "hrx.dll").touch()
            (public_install_dir / "bin" / "hrx-info.exe").touch()
            (public_install_dir / "lib").mkdir()
            (public_install_dir / "lib" / "hrx.lib").touch()
            (public_install_dir / "include" / "hrx").mkdir(parents=True)
            (public_install_dir / "include" / "hrx" / "hrx_runtime.h").write_text(
                "// c api\n"
            )
            (public_install_dir / "include" / "hrx" / "hrx_runtime_cxx.h").write_text(
                "// cxx api\n"
            )
            (public_install_dir / "lib" / "cmake" / "hrx").mkdir(parents=True)
            (
                public_install_dir / "lib" / "cmake" / "hrx" / "hrx-config.cmake"
            ).write_text("# config\n")
            rocm_root.mkdir()
            args = Namespace(
                rocm_root=rocm_root,
                package_output_dir=root / "dist",
                public_install_dir=public_install_dir,
                public_deps_dir=public_deps_dir,
                package_suffix="test",
                build_dir=root / "build",
            )

            ci_core_windows.package_core(args)

            self.assertTrue(
                (root / "dist" / "hrx-public-windows-x86_64-test.zip").exists()
            )


if __name__ == "__main__":
    unittest.main()
