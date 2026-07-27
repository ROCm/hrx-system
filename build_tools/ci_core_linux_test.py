# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import io
import unittest
from pathlib import Path

from build_tools import ci_core_linux


class FakeS3:
    def __init__(self, objects: dict[tuple[str, str], str]):
        self.objects = objects

    def get_object(self, Bucket: str, Key: str):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)].encode())}


class CiCoreLinuxTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
