# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import tarfile
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
        self.objects: dict[str, str | bytes] = {}

    def read(self, key: str) -> bytes:
        value = self.objects[key]
        return value.encode() if isinstance(value, str) else value

    def get_object(self, Bucket: str, Key: str):
        return {"Body": io.BytesIO(self.read(Key))}

    def download_file(self, Bucket: str, Key: str, Filename: str):
        Path(Filename).write_bytes(self.read(Key))

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
                        "Size": len(self.read(key)),
                        "LastModified": dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
                    }
                    for key in sorted(self.objects)
                    if key.startswith(Prefix)
                ]
            }

    def publish_archives(
        self,
        run_id: str,
        platform,
        *,
        content: bytes = b"archive",
        extension: str = ".tar.zst",
    ) -> str:
        prefix = f"{run_id}-{platform.PLATFORM}/"
        for name in common.wanted_artifacts(
            "core", artifact_sets=platform.ARTIFACT_SETS
        ):
            key = f"{prefix}{name}{extension}"
            self.objects[key] = content
            self.objects[key + ".sha256sum"] = (
                hashlib.sha256(content).hexdigest() + "\n"
            )
        return prefix


class CiCoreCommonTest(unittest.TestCase):
    def fetch_args(self, root: Path) -> argparse.Namespace:
        return argparse.Namespace(
            run_id="200",
            latest=False,
            release_type="nightly",
            artifact_set="core",
            artifact_variant="release",
            rocm_root=root / "rocm",
            download_cache_dir=root / "downloads",
            download_concurrency=1,
        )

    def make_fetch_source(self, platform) -> FakeS3:
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w:xz", preset=0) as tar:
            for name, content in (
                ("artifact_manifest.txt", b"component/stage\n"),
                ("component/stage/include/runtime.h", b"verified runtime\n"),
            ):
                member = tarfile.TarInfo(name)
                member.size = len(content)
                tar.addfile(member, io.BytesIO(content))
        s3 = FakeS3()
        prefix = s3.publish_archives(
            "200", platform, content=archive.getvalue(), extension=".tar.xz"
        )
        s3.objects[prefix + CONFIGURE_LOG_KEY] = configure_log("", platform.PLATFORM)
        return s3

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

    def test_latest_discovery_waits_for_checksum_publication(self):
        for platform in (ci_core_linux, ci_core_windows):
            with self.subTest(platform=platform.PLATFORM):
                s3 = FakeS3()
                for run_id in ("100", "200"):
                    prefix = s3.publish_archives(run_id, platform)
                    s3.objects[prefix + CONFIGURE_LOG_KEY] = configure_log("")
                key = next(
                    key
                    for key in s3.objects
                    if key.startswith(prefix) and key.endswith(".sha256sum")
                )
                checksum = s3.objects.pop(key)
                self.assertEqual(self.discover(s3, platform), "100")
                s3.objects[key] = checksum
                self.assertEqual(self.discover(s3, platform), "200")

    def test_selection_requires_checksum_of_preferred_archive(self):
        prefix = "200-linux/"
        s3 = FakeS3()
        s3.objects[prefix + "runtime.tar.xz"] = "xz"
        s3.objects[prefix + "runtime.tar.xz.sha256sum"] = "checksum"
        s3.objects[prefix + "runtime.tar.zst"] = "zst"
        selected, missing = common.select_available(
            common.list_prefix(s3, BUCKET, prefix), prefix, ["runtime"]
        )
        self.assertEqual([obj.key for obj in selected], [prefix + "runtime.tar.zst"])
        self.assertEqual(missing, ["runtime.tar.zst.sha256sum"])

    def test_pinned_fetch_rejects_missing_checksum_before_creating_outputs(self):
        for platform in (ci_core_linux, ci_core_windows):
            with self.subTest(platform=platform.PLATFORM):
                s3 = self.make_fetch_source(platform)
                key = next(key for key in s3.objects if key.endswith(".sha256sum"))
                del s3.objects[key]
                with tempfile.TemporaryDirectory() as temporary_dir:
                    root = Path(temporary_dir)
                    with mock.patch.object(common, "create_s3_client", return_value=s3):
                        with self.assertRaisesRegex(
                            RuntimeError, r"Missing required.*"
                        ) as raised:
                            platform.fetch_rocm(self.fetch_args(root))
                    self.assertIn(Path(key).name, str(raised.exception))
                    self.assertEqual(list(root.iterdir()), [])

    def test_checksum_download_preserves_transfer_failures(self):
        for error in (
            ConnectionError("S3 connection failed"),
            PermissionError("Access denied"),
            FileNotFoundError("Checksum disappeared"),
        ):
            with self.subTest(error=error):
                s3 = mock.Mock()
                s3.download_file.side_effect = error
                with self.assertRaises(type(error)) as raised:
                    common.download_checksum(
                        s3, BUCKET, "runtime.tar.xz", Path("runtime.tar.xz")
                    )
                self.assertIs(raised.exception, error)

    def test_checksum_download_preserves_local_write_failure(self):
        s3 = FakeS3()
        s3.objects["runtime.tar.xz.sha256sum"] = "checksum"
        with tempfile.TemporaryDirectory() as temporary_dir:
            archive = Path(temporary_dir) / "runtime.tar.xz"
            archive.with_name(archive.name + ".sha256sum").mkdir()
            with self.assertRaises(OSError):
                common.download_checksum(s3, BUCKET, archive.name, archive)

    def test_checksum_accepts_only_one_matching_sha256_record(self):
        content = b"archive contents"
        digest = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as temporary_dir:
            archive = Path(temporary_dir) / "runtime.tar.xz"
            archive.write_bytes(content)
            checksum = archive.with_name(archive.name + ".sha256sum")
            for text in (
                digest,
                digest.upper() + "\r\n",
                f"{digest}  {archive.name}\n",
                f"{digest} *{archive.name}\n",
            ):
                with self.subTest(text=text):
                    checksum.write_text(text)
                    common.verify_checksum(archive, checksum)
            for text in (
                "",
                " \n",
                "bad checksum",
                "0" * 63,
                "g" * 64,
                f"{digest}\n{digest}\n",
                f"{digest}\n{archive.name}\n",
                f"{digest}  different.tar.xz\n",
                f"{digest}  {archive.name} extra\n",
            ):
                with self.subTest(text=text):
                    checksum.write_text(text)
                    with self.assertRaisesRegex(
                        RuntimeError, "Invalid SHA-256 checksum"
                    ):
                        common.verify_checksum(archive, checksum)
            checksum.write_text("0" * 64 + "\n")
            with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
                common.verify_checksum(archive, checksum)
            with self.assertRaises(FileNotFoundError):
                common.verify_checksum(
                    archive, Path(temporary_dir) / "missing.sha256sum"
                )
            checksum.write_bytes(b"\xff")
            with self.assertRaises(UnicodeDecodeError):
                common.verify_checksum(archive, checksum)
            with self.assertRaises(OSError):
                common.verify_checksum(archive, Path(temporary_dir))

    def test_fetch_extracts_verified_archives_and_records_manifest(self):
        for platform in (ci_core_linux, ci_core_windows):
            with self.subTest(platform=platform.PLATFORM):
                s3 = self.make_fetch_source(platform)
                with tempfile.TemporaryDirectory() as temporary_dir:
                    args = self.fetch_args(Path(temporary_dir))
                    with mock.patch.object(common, "create_s3_client", return_value=s3):
                        platform.fetch_rocm(args)
                    self.assertEqual(
                        (args.rocm_root / "include/runtime.h").read_bytes(),
                        b"verified runtime\n",
                    )
                    manifest = json.loads(
                        (args.rocm_root / ".hrx-rocm-artifacts.json").read_text()
                    )
                    self.assertEqual(manifest["platform"], platform.PLATFORM)
                    self.assertEqual(
                        len(manifest["artifacts"]),
                        len(
                            common.wanted_artifacts(
                                "core", artifact_sets=platform.ARTIFACT_SETS
                            )
                        ),
                    )

    def test_fetch_verifies_entire_set_before_extracting(self):
        for platform in (ci_core_linux, ci_core_windows):
            for failure in ("mismatch", "disappeared", "corrupt_cache"):
                with self.subTest(platform=platform.PLATFORM, failure=failure):
                    s3 = self.make_fetch_source(platform)
                    key = max(key for key in s3.objects if key.endswith(".tar.xz"))
                    if failure == "mismatch":
                        s3.objects[key + ".sha256sum"] = "0" * 64 + "\n"
                    download = s3.download_file

                    def download_file(bucket, object_key, filename):
                        if (
                            failure == "disappeared"
                            and object_key == key + ".sha256sum"
                        ):
                            raise FileNotFoundError("Checksum disappeared")
                        download(bucket, object_key, filename)

                    with tempfile.TemporaryDirectory() as temporary_dir:
                        args = self.fetch_args(Path(temporary_dir))
                        if failure == "corrupt_cache":
                            cached = common.s3_cache_path(
                                args.download_cache_dir, BUCKET, key
                            )
                            cached.parent.mkdir(parents=True)
                            cached.write_bytes(b"x" * len(s3.read(key)))
                        with (
                            mock.patch.object(
                                common, "create_s3_client", return_value=s3
                            ),
                            mock.patch.object(
                                s3, "download_file", side_effect=download_file
                            ),
                        ):
                            error_type = (
                                FileNotFoundError
                                if failure == "disappeared"
                                else RuntimeError
                            )
                            diagnostic = (
                                "Checksum disappeared"
                                if failure == "disappeared"
                                else "Checksum mismatch"
                            )
                            with self.assertRaisesRegex(error_type, diagnostic):
                                platform.fetch_rocm(args)
                        self.assertEqual(list(args.rocm_root.iterdir()), [])

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
                        args = self.fetch_args(root)
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
