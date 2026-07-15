# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import io
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
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rocm_root = root / "rocm"
            llvm_bin = rocm_root / "lib" / "llvm" / "bin"
            llvm_bin.mkdir(parents=True)
            for tool in ["clang", "clang++", "llvm-ar", "llvm-ranlib"]:
                (llvm_bin / tool).touch()

            parser = argparse.ArgumentParser()
            with mock.patch.dict(os.environ, {}, clear=True):
                ci_core_linux.add_shared_args(parser)
                args = parser.parse_args([])

            args.rocm_root = rocm_root
            args.build_dir = root / "build"
            args.public_install_dir = root / "public"
            args.tests_install_dir = root / "tests"

            commands = []
            old_run = ci_core_linux.run
            try:
                ci_core_linux.run = lambda cmd, **_kwargs: commands.append(
                    [os.fspath(arg) for arg in cmd]
                )

                ci_core_linux.build_core(args)
            finally:
                ci_core_linux.run = old_run

        configure_cmd = commands[0]
        self.assertIn("-DIREE_BUILD_TESTS=ON", configure_cmd)
        self.assertIn("-DIREE_BUILD_BENCHMARKS=ON", configure_cmd)
        self.assertIn("-DLOOM_BUILD=ON", configure_cmd)
        self.assertIn("-DLIBHRX_BUILD_CTS=ON", configure_cmd)
        self.assertIn("-DHRX_INSTALL_TESTS=ON", configure_cmd)
        self.assertIn("-DLIBHRX_BUILD_PASSTHROUGH=ON", configure_cmd)
        self.assertIn("-DIREE_HAL_DRIVER_AMDGPU=ON", configure_cmd)


if __name__ == "__main__":
    unittest.main()
