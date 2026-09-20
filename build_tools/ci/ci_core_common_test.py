# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import datetime as dt
import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from build_tools.ci import ci_core_common as common
from build_tools.ci import ci_core_linux, ci_core_windows

BUCKET = "therock-nightly-artifacts"
CONFIGURE_LOG_KEY = "logs/compiler-runtime/amd-llvm_configure.log"


def configure_log(sanitizer: str, platform_name: str = "linux") -> str:
    if platform_name == "windows":
        command = "'C:/Program Files/CMake/bin/cmake.exe'"
        directory = "B:\\build\\compiler\\amd-llvm\\build"
    else:
        command = "/usr/local/therock-tools/bin/cmake"
        directory = "/build/compiler/amd-llvm/build"
    return (
        "BEGIN\t0\n"
        f"EXEC\t{directory}\t{command} -GNinja "
        f"-DTHEROCK_SANITIZER={sanitizer} -DCOMPILER_RT_DEBUG=OFF\n"
        "END\t1\t1\t0\n"
    )


class FakeS3:
    def __init__(self):
        self.objects: dict[str, str] = {}

    def get_object(self, Bucket: str, Key: str):
        return {"Body": io.BytesIO(self.objects[Key].encode())}

    def get_paginator(self, operation: str):
        if operation != "list_objects_v2":
            raise ValueError(operation)
        return self

    def paginate(self, Bucket: str, Prefix: str = "", Delimiter: str = ""):
        if Delimiter:
            prefixes = sorted({key.split("/")[0] + "/" for key in self.objects})
            for prefix in prefixes:
                yield {"CommonPrefixes": [{"Prefix": prefix}]}
        else:
            yield {
                "Contents": [
                    {
                        "Key": key,
                        "Size": len(value.encode()),
                        "LastModified": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
                    }
                    for key, value in sorted(self.objects.items())
                    if key.startswith(Prefix)
                ]
            }

    def publish_archives(self, run_id: str, platform) -> str:
        prefix = f"{run_id}-{platform.PLATFORM}/"
        for name in common.wanted_artifacts(
            "core", artifact_sets=platform.ARTIFACT_SETS
        ):
            self.objects[f"{prefix}{name}.tar.zst"] = "archive"
        return prefix


class CiCoreCommonTest(unittest.TestCase):
    def discover(self, s3: FakeS3, platform, variant: str = "release") -> str:
        return common.discover_latest_run_id(
            s3,
            "nightly",
            "core",
            variant,
            platform_name=platform.PLATFORM,
            platform_display=platform.PLATFORM,
            artifact_sets=platform.ARTIFACT_SETS,
        )

    def test_variant_requires_explicit_supported_compiler_setting(self):
        for sanitizer, expected in (
            ("", "release"),
            ("ASAN", "asan"),
            ("HOST_ASAN", "host-asan"),
            ("TSAN", "tsan"),
        ):
            with self.subTest(sanitizer=sanitizer):
                self.assertEqual(
                    common.rocm_artifact_variant_from_configure_log(
                        configure_log(sanitizer)
                    ),
                    expected,
                )
        self.assertEqual(
            common.rocm_artifact_variant_from_configure_log(
                configure_log("", "windows").replace("\n", "\r\n")
            ),
            "release",
        )
        for text in (
            "",
            "-- Configuring done\n",
            "EXEC\t/build\tcmake -GNinja\n",
            "diagnostic mentions -DTHEROCK_SANITIZER=\n",
            configure_log("unsupported"),
            configure_log("ASAN_UNKNOWN"),
        ):
            with self.subTest(text=text):
                self.assertIsNone(common.rocm_artifact_variant_from_configure_log(text))

    def test_latest_discovery_waits_for_classification_metadata(self):
        for platform, sanitizer, variant in (
            (ci_core_linux, "ASAN", "asan"),
            (ci_core_windows, "", "release"),
        ):
            with self.subTest(platform=platform.PLATFORM):
                s3 = FakeS3()
                older = s3.publish_archives("100", platform)
                s3.objects[older + CONFIGURE_LOG_KEY] = configure_log(
                    "", platform.PLATFORM
                )
                newer = s3.publish_archives("200", platform)
                for text in (
                    None,
                    "",
                    "-- Configuring done\n",
                    configure_log("unknown"),
                ):
                    if text is not None:
                        s3.objects[newer + CONFIGURE_LOG_KEY] = text
                    self.assertEqual(self.discover(s3, platform), "100")

                s3.objects[newer + CONFIGURE_LOG_KEY] = configure_log(
                    sanitizer, platform.PLATFORM
                )
                self.assertEqual(self.discover(s3, platform, variant), "200")
                self.assertEqual(
                    self.discover(s3, platform),
                    "200" if variant == "release" else "100",
                )

                # A classified upload still needs the complete requested archive set.
                artifact = common.wanted_artifacts(
                    "core", artifact_sets=platform.ARTIFACT_SETS
                )[0]
                del s3.objects[f"{newer}{artifact}.tar.zst"]
                if variant == "release":
                    self.assertEqual(self.discover(s3, platform, variant), "100")
                else:
                    with self.assertRaisesRegex(RuntimeError, "Could not discover"):
                        self.discover(s3, platform, variant)

    def test_metadata_read_errors_are_not_an_incomplete_upload(self):
        s3 = FakeS3()
        prefix = s3.publish_archives("200", ci_core_linux)
        s3.objects[prefix + CONFIGURE_LOG_KEY] = configure_log("")
        with mock.patch.object(s3, "get_object", side_effect=OSError("S3 read failed")):
            with self.assertRaisesRegex(OSError, "S3 read failed"):
                self.discover(s3, ci_core_linux)

    def test_no_classified_candidate_is_not_a_release(self):
        for platform in (ci_core_linux, ci_core_windows):
            with self.subTest(platform=platform.PLATFORM):
                s3 = FakeS3()
                s3.publish_archives("200", platform)
                with self.assertRaisesRegex(RuntimeError, "Could not discover"):
                    self.discover(s3, platform)

    def test_pinned_fetch_rejects_unclassified_upload_before_creating_outputs(self):
        for platform in (ci_core_linux, ci_core_windows):
            for text in (None, "", configure_log("unsupported")):
                with self.subTest(platform=platform.PLATFORM, text=text):
                    s3 = FakeS3()
                    prefix = s3.publish_archives("200", platform)
                    if text is not None:
                        s3.objects[prefix + CONFIGURE_LOG_KEY] = text
                    with tempfile.TemporaryDirectory() as temporary_dir:
                        root = Path(temporary_dir)
                        args = argparse.Namespace(
                            run_id="200",
                            latest=False,
                            release_type="nightly",
                            artifact_set="core",
                            artifact_variant="release",
                            rocm_root=root / "rocm",
                            download_cache_dir=root / "downloads",
                            download_concurrency=1,
                        )
                        with mock.patch.object(
                            common, "create_s3_client", return_value=s3
                        ):
                            with self.assertRaisesRegex(
                                RuntimeError,
                                "Cannot determine.*200-.*amd-llvm_configure.log",
                            ):
                                platform.fetch_rocm(args)
                        self.assertEqual(list(root.iterdir()), [])

    def test_validation_preserves_exact_variant_matching(self):
        for sanitizer, variant in (
            ("", "release"),
            ("ASAN", "asan"),
            ("HOST_ASAN", "host-asan"),
            ("TSAN", "tsan"),
        ):
            with self.subTest(variant=variant):
                s3 = FakeS3()
                prefix = s3.publish_archives("200", ci_core_linux)
                s3.objects[prefix + CONFIGURE_LOG_KEY] = configure_log(sanitizer)
                available = common.list_prefix(s3, BUCKET, prefix)
                common.validate_rocm_artifact_variant(
                    s3, BUCKET, prefix, available, variant
                )
                other = "asan" if variant == "release" else "release"
                with self.assertRaisesRegex(
                    RuntimeError, f"variant '{variant}'.*'{other}'"
                ):
                    common.validate_rocm_artifact_variant(
                        s3, BUCKET, prefix, available, other
                    )


if __name__ == "__main__":
    unittest.main()
