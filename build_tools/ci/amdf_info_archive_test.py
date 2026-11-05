# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Archive export behavior; only the detached native process is substituted."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from zipfile import ZipFile

from build_tools.ci import amdf_info_archive
from loom.packaging.python import release


class AmdfInfoArchiveTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "published"
        self.identity = release.Identity("1.2.3rc1", None)
        self.description = "SDK\n"
        self.payload = b"native amdf-info executable"
        self.calls = []

    def wheel(
        self, platform=release.LINUX_X86_64_PLATFORM, profile="release", payload=True
    ):
        version = self.identity.version + (
            "" if profile == "release" else "+" + profile
        )
        path = self.root / f"hrx_loom-{version}-py3-none-{platform}.whl"
        prefix = f"hrx_loom-{version}.dist-info/"
        with ZipFile(path, "w") as archive:
            archive.writestr(
                prefix + "METADATA",
                f"Name: hrx-loom\nVersion: {version}\n"
                "Description-Content-Type: text/markdown\nRequires-Python: >=3.12\n"
                "\n" + self.description + "\n",
            )
            archive.writestr(
                prefix + "WHEEL",
                f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-{platform}\n",
            )
            archive.writestr(prefix + "loom.json", '{"catalog":null}')
            if payload:
                binary = release.standalone_binary_name("amdf-info", platform)
                archive.writestr(
                    f"hrx_loom-{version}.data/scripts/{binary}", self.payload
                )
            archive.writestr(
                f"hrx_loom-{version}.data/scripts/companion.dll", b"companion"
            )
        return path

    def native_process(self, arguments, **options):
        binary = Path(arguments[0])
        self.assertEqual(binary.read_bytes(), self.payload)
        self.assertEqual(list(binary.parent.iterdir()), [binary])
        self.assertEqual(options["cwd"], binary.parent)
        self.assertEqual(options["env"]["PATH"], "")
        self.assertNotIn("LD_LIBRARY_PATH", options["env"])
        self.assertNotIn("LD_PRELOAD", options["env"])
        self.calls.append(arguments[1:])
        output = (
            '{"schema_version":1,"endpoints":[],"result":{"exit_code":0}}\n'
            if arguments[1:] == ["--format=json"]
            else "Native inventory or help\n"
        )
        return subprocess.CompletedProcess(arguments, 0, output)

    def package(self, wheel, output=None):
        return amdf_info_archive.package_tool(
            wheel, output or self.output, self.identity, self.description
        )

    def test_exports_exact_bytes_only_after_detached_native_invocations(self):
        for platform, native in (
            (release.LINUX_X86_64_PLATFORM, "linux"),
            (release.WINDOWS_X86_64_PLATFORM, "win32"),
        ):
            with self.subTest(platform=platform):
                wheel = self.wheel(platform)
                original = wheel.read_bytes()
                self.calls.clear()
                with (
                    mock.patch.object(amdf_info_archive.sys, "platform", native),
                    mock.patch.dict(
                        amdf_info_archive.os.environ,
                        {"LD_LIBRARY_PATH": "sdk", "LD_PRELOAD": "runtime"},
                    ),
                    mock.patch.object(
                        amdf_info_archive.subprocess,
                        "run",
                        side_effect=self.native_process,
                    ),
                ):
                    first = self.package(wheel, self.output / "first")
                    second = self.package(wheel, self.output / "second")
                self.assertEqual(self.calls, [["--help"], [], ["--format=json"]] * 2)
                self.assertEqual(wheel.read_bytes(), original)
                self.assertEqual(first.path.read_bytes(), second.path.read_bytes())
                self.assertEqual(
                    first.digest, hashlib.sha256(first.path.read_bytes()).hexdigest()
                )
                with ZipFile(first.path) as archive:
                    self.assertEqual(len(archive.infolist()), 1)
                    self.assertEqual(archive.read(archive.infolist()[0]), self.payload)
                self.assertEqual((first.tool, first.platform), ("amdf-info", platform))

    def test_any_failed_invocation_prevents_export(self):
        wheel = self.wheel()
        for failed in (["--help"], [], ["--format=json"]):

            def run(arguments, **options):
                if arguments[1:] == failed:
                    raise subprocess.CalledProcessError(1, arguments)
                return self.native_process(arguments, **options)

            with (
                self.subTest(failed=failed),
                mock.patch.object(amdf_info_archive.sys, "platform", "linux"),
                mock.patch.object(amdf_info_archive.subprocess, "run", side_effect=run),
                self.assertRaises(subprocess.CalledProcessError),
            ):
                self.package(wheel)
            self.assertFalse(self.output.exists())

    def test_successful_process_must_supply_valid_output(self):
        wheel = self.wheel()
        for output in (
            "",
            "{",
            "[]",
            '{"schema_version":2,"result":{"exit_code":0}}',
            '{"schema_version":1,"result":{"exit_code":1}}',
            '{"schema_version":1,"result":{}}',
        ):

            def run(arguments, **options):
                result = self.native_process(arguments, **options)
                if arguments[1:] == ["--format=json"]:
                    result.stdout = output
                return result

            with (
                self.subTest(output=output),
                mock.patch.object(amdf_info_archive.sys, "platform", "linux"),
                mock.patch.object(amdf_info_archive.subprocess, "run", side_effect=run),
                self.assertRaises(ValueError),
            ):
                self.package(wheel)
            self.assertFalse(self.output.exists())

    def test_wrong_profile_platform_or_missing_tool_never_executes(self):
        cases = (
            (self.wheel(profile="asan"), "Release wheel"),
            (self.wheel(platform=release.WINDOWS_X86_64_PLATFORM), "native OS"),
            (self.wheel(payload=False), "missing"),
        )
        for wheel, message in cases:
            with (
                self.subTest(wheel=wheel),
                mock.patch.object(amdf_info_archive.sys, "platform", "linux"),
                mock.patch.object(amdf_info_archive.subprocess, "run") as run,
                self.assertRaisesRegex(ValueError, message),
            ):
                self.package(wheel)
            run.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_export_cannot_replace_an_existing_archive(self):
        wheel = self.wheel()
        with (
            mock.patch.object(amdf_info_archive.sys, "platform", "linux"),
            mock.patch.object(
                amdf_info_archive.subprocess, "run", side_effect=self.native_process
            ),
        ):
            artifact = self.package(wheel)
            artifact.path.write_bytes(b"prior artifact")
            with self.assertRaises(FileExistsError):
                self.package(wheel)
        self.assertEqual(artifact.path.read_bytes(), b"prior artifact")


if __name__ == "__main__":
    unittest.main()
