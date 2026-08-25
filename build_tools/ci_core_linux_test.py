# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools import ci_core_linux

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeS3:
    def __init__(self, objects: dict[tuple[str, str], str]):
        self.objects = objects

    def get_object(self, Bucket: str, Key: str):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)].encode())}


class CiCoreLinuxTest(unittest.TestCase):
    def build_core_commands(self, env: dict[str, str] | None = None) -> list[list[str]]:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rocm_root = root / "rocm"
            llvm_bin = rocm_root / "lib" / "llvm" / "bin"
            llvm_bin.mkdir(parents=True)
            for tool in ["clang", "clang++", "llvm-ar", "llvm-ranlib"]:
                (llvm_bin / tool).touch()

            parser = argparse.ArgumentParser()
            with mock.patch.dict(os.environ, env or {}, clear=True):
                ci_core_linux.add_shared_args(parser)
                args = parser.parse_args([])
                args.rocm_root = rocm_root
                args.build_dir = root / "build"
                args.public_install_dir = root / "public"
                args.tests_install_dir = root / "tests"

                commands = []
                with mock.patch.object(
                    ci_core_linux,
                    "run",
                    side_effect=lambda cmd, **_kwargs: commands.append(
                        [os.fspath(arg) for arg in cmd]
                    ),
                ):
                    ci_core_linux.build_core(args)

        return commands

    def test_script_runs_by_path_without_pythonpath(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        result = subprocess.run(
            [sys.executable, "build_tools/ci_core_linux.py", "--help"],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        self.assertIn("fetch-rocm", result.stdout)

    def test_rocm_artifact_variant_from_configure_log(self):
        self.assertEqual(
            ci_core_linux.rocm_artifact_variant_from_configure_log(""),
            "release",
        )
        self.assertEqual(
            ci_core_linux.rocm_artifact_variant_from_configure_log(
                "Override ASAN GPU_TARGETS = gfx942:xnack+"
            ),
            "asan",
        )
        self.assertEqual(
            ci_core_linux.rocm_artifact_variant_from_configure_log(
                "Override TSAN GPU_TARGETS = gfx942:xnack+"
            ),
            "tsan",
        )
        self.assertEqual(
            ci_core_linux.rocm_artifact_variant_from_configure_log(
                "SANITIZER = HOST_ASAN"
            ),
            "host-asan",
        )

    def test_s3_cache_path_preserves_artifact_identity(self):
        cache_root = Path("/tmp/cache")

        self.assertEqual(
            ci_core_linux.s3_cache_path(
                cache_root,
                "therock-nightly-artifacts",
                "123-linux/core-runtime_lib_generic.tar.zst",
            ),
            cache_root
            / "therock-nightly-artifacts"
            / "123-linux"
            / "core-runtime_lib_generic.tar.zst",
        )
        self.assertNotEqual(
            ci_core_linux.s3_cache_path(
                cache_root,
                "therock-nightly-artifacts",
                "123-linux/core-runtime_lib_generic.tar.zst",
            ),
            ci_core_linux.s3_cache_path(
                cache_root,
                "therock-nightly-artifacts",
                "456-linux/core-runtime_lib_generic.tar.zst",
            ),
        )

    def test_s3_cache_path_rejects_unsafe_keys(self):
        with self.assertRaisesRegex(RuntimeError, "Unsafe S3 key"):
            ci_core_linux.s3_cache_path(Path("/tmp/cache"), "bucket", "../evil")

    def test_rocm_manifest_carries_immutable_artifact_identity(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            manifest_path = Path(temporary_dir) / "manifest.json"
            ci_core_linux.common.write_rocm_manifest(
                manifest_path,
                release_type="nightly",
                run_id="123456",
                platform_name="linux",
                bucket="therock-nightly-artifacts",
                artifact_variant="release",
                artifact_set="core-with-llvm-dev",
                artifacts=[],
            )

            manifest = json.loads(manifest_path.read_text())

        self.assertEqual(
            manifest["artifact_identity"],
            "nightly-123456-linux-release-core-with-llvm-dev",
        )

    def test_rocm_artifact_identity_rejects_path_components(self):
        with self.assertRaisesRegex(RuntimeError, "Unsafe ROCm artifact identity"):
            ci_core_linux.common.rocm_artifact_identity(
                release_type="nightly",
                run_id="../123456",
                platform_name="linux",
                artifact_variant="release",
                artifact_set="core",
            )

    def test_upstream_hip_artifact_set_includes_llvm_development_files(self):
        self.assertIn(
            "amd-llvm_dev_generic",
            ci_core_linux.common.wanted_artifacts(
                "core-with-upstream-hip",
                artifact_sets=ci_core_linux.ARTIFACT_SETS,
            ),
        )

    def test_validate_rocm_artifact_variant_rejects_mismatch(self):
        bucket = "therock-nightly-artifacts"
        prefix = "123-linux/"
        log_key = prefix + ci_core_linux.ROCM_ARTIFACT_VARIANT_LOG_KEY
        available = [
            ci_core_linux.S3Object(key=log_key, size=1, last_modified=""),
        ]
        s3 = FakeS3({(bucket, log_key): "SANITIZER = ASAN\n"})

        with self.assertRaisesRegex(RuntimeError, "variant 'asan'.*'release'"):
            ci_core_linux.validate_rocm_artifact_variant(
                s3,
                bucket,
                prefix,
                available,
                "release",
            )

    def test_amdgpu_device_binary_source_options_pin_rocm_root(self):
        self.assertEqual(
            ci_core_linux.amdgpu_device_binary_source_options(Path("/tmp/rocm-root")),
            [
                "IREE_HAL_AMDGPU_DEVICE_BINARY_BUILD_MODE=source",
                "IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN=rocm",
                "IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH=/tmp/rocm-root",
            ],
        )

    def test_build_core_preserves_linux_ci_feature_defaults(self):
        commands = self.build_core_commands()
        configure_cmd = commands[0]
        self.assertIn("-DIREE_BUILD_TESTS=ON", configure_cmd)
        self.assertIn("-DIREE_BUILD_BENCHMARKS=ON", configure_cmd)
        self.assertIn("-DLOOM_BUILD=ON", configure_cmd)
        self.assertIn("-DLIBHRX_BUILD_CTS=ON", configure_cmd)
        self.assertIn("-DHRX_INSTALL_TESTS=ON", configure_cmd)
        self.assertIn("-DLIBHRX_BUILD_PASSTHROUGH=ON", configure_cmd)
        self.assertIn("-DIREE_HAL_DRIVER_AMDGPU=ON", configure_cmd)
        self.assertNotIn("-DCMAKE_C_COMPILER_LAUNCHER=ccache", configure_cmd)
        self.assertNotIn("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache", configure_cmd)
        self.assertEqual(commands[1][0:2], ["cmake", "--build"])
        self.assertIn("all", commands[1])
        self.assertEqual(
            [command[0:2] for command in commands[2:]],
            [["cmake", "--install"], ["cmake", "--install"]],
        )

    def test_sanitizer_build_only_configures_the_source_test_tree(self):
        commands = self.build_core_commands({"HRX_SANITIZER": "asan"})

        self.assertEqual(len(commands), 1)
        self.assertIn("-DIREE_ENABLE_ASAN=ON", commands[0])
        self.assertIn("-DIREE_BUILD_TESTS=ON", commands[0])

    def test_sanitizer_tests_build_the_selected_source_ctest_closure(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            rocm_root = root / "rocm"
            rocm_root.mkdir()
            build_dir = root / "build"
            build_dir.mkdir()
            (build_dir / "CTestTestfile.cmake").touch()

            parser = argparse.ArgumentParser()
            with mock.patch.dict(
                os.environ,
                {
                    "HRX_SANITIZER": "asan",
                    "HRX_CTEST_REGEX": "^iree/",
                    "HRX_CTEST_EXCLUDE_REGEX": "excluded",
                    "HRX_CTEST_LABEL_EXCLUDE_REGEX": "runtime-resource=|manual",
                    "HRX_CTEST_PARALLELISM": "3",
                },
                clear=True,
            ):
                ci_core_linux.add_shared_args(parser)
                args = parser.parse_args([])
                args.rocm_root = rocm_root
                args.build_dir = build_dir

                selected_test_step = mock.Mock()
                selected_test_step.run.return_value = 0
                with mock.patch.object(
                    ci_core_linux.ctest_dev,
                    "CTestBuildAndRunStep",
                    return_value=selected_test_step,
                ) as step_type:
                    ci_core_linux.test_core(args)

        step_type.assert_called_once()
        step_args = step_type.call_args.kwargs
        self.assertEqual(step_args["build_dir"], build_dir)
        self.assertEqual(
            step_args["arguments"],
            [
                "--parallel",
                "3",
                "-R",
                "^iree/",
                "-E",
                "(excluded)",
                "-LE",
                "runtime-resource=|manual",
                "--no-tests=error",
            ],
        )
        self.assertIn("ASAN_OPTIONS", step_args["env"])
        selected_test_step.run.assert_called_once_with(verbose=True)

    def test_build_core_preserves_standard_compiler_launcher_options(self):
        commands = self.build_core_commands(
            {
                "HRX_CMAKE_OPTIONS": "\n".join(
                    [
                        "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                        "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                    ]
                )
            }
        )

        configure_cmd = commands[0]
        self.assertIn("-DCMAKE_C_COMPILER_LAUNCHER=ccache", configure_cmd)
        self.assertIn("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache", configure_cmd)

    def test_core_linux_workflow_has_no_unbacked_ccache_directory(self):
        workflow = (REPO_ROOT / ".github/workflows/build_core_linux.yml").read_text()
        self.assertNotIn("CCACHE_DIR", workflow)
        self.assertIn("- name: Prepare CMake build", workflow)
        self.assertIn("- name: Run tests", workflow)


if __name__ == "__main__":
    unittest.main()
