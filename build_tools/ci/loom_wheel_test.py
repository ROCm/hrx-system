# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from build_tools.ci import loom_wheel
from loom.packaging.python import release


class LoomWheelTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "export"
        self.bazel_log = self.root / "bazel_calls.jsonl"
        package = self.root / "loom" / "packaging" / "python"
        package.mkdir(parents=True)
        (package / "version.json").write_text(
            json.dumps({"package-version": "1.2.3", "model-catalog": None})
        )
        (package / "PYPI_README.md").write_text("SDK\n")

    def wheel(self, profile="release", platform="win_amd64"):
        version = "1.2.3" + ("" if profile == "release" else "+" + profile)
        path = self.root / f"hrx_loom-{version}-py3-none-{platform}.whl"
        prefix = f"hrx_loom-{version}.dist-info/"
        with ZipFile(path, "w") as archive:
            archive.writestr(
                prefix + "METADATA",
                f"Name: hrx-loom\nVersion: {version}\n"
                "Description-Content-Type: text/markdown\nRequires-Python: >=3.12\n"
                "\nSDK\n\n",
            )
            archive.writestr(
                prefix + "WHEEL",
                f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-{platform}\n",
            )
            archive.writestr(prefix + "loom.json", '{"catalog": null}')
        return path

    def query(self, path):
        model = self.root / "fixture"
        model.mkdir(exist_ok=True)
        (model / "serve.json").write_text("{}")
        (model / "weights.data").write_bytes(b"weights")
        # An old, undeclared file in the same output directory must not travel.
        (model / "stale.data").write_bytes(b"stale")
        return "\n".join(
            json.dumps(record)
            for record in (
                {
                    "package": "loom/packaging/python",
                    "files": [str(path)],
                    "runfiles": [],
                },
                {
                    "package": "loom/packaging/python/test",
                    "files": [str(path)],
                    "runfiles": [],
                },
                {
                    "package": "loom/src/loom/serve/fixture/text",
                    "files": [str(model / "serve.json")],
                    "runfiles": [
                        str(model / "serve.json"),
                        str(model / "weights.data"),
                    ],
                },
            )
        )

    def bazel(self, query="", build_status=0):
        """Creates a process-compatible Bazel substitute for orchestration tests."""
        self.bazel_log.unlink(missing_ok=True)
        script = self.root / "fake_bazel.py"
        script.write_text(
            "import json\n"
            "import sys\n"
            f"query = {query!r}\n"
            f"build_status = {build_status}\n"
            f"log = {os.fspath(self.bazel_log)!r}\n"
            "with open(log, 'a', encoding='utf-8') as stream:\n"
            "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "command = sys.argv[1] if len(sys.argv) > 1 else None\n"
            "if command == 'build':\n"
            "    raise SystemExit(build_status)\n"
            "if command == 'cquery':\n"
            "    print(query)\n"
            "    raise SystemExit(0)\n"
            "raise SystemExit(2)\n",
            encoding="utf-8",
        )
        if os.name == "nt":
            launcher = self.root / "test-bazel.cmd"
            launcher.write_text(
                f'@"{sys.executable}" "{script}" %*\n', encoding="utf-8"
            )
        else:
            launcher = self.root / "test-bazel"
            launcher.write_text(
                "#!/bin/sh\n"
                f'exec {shlex.quote(sys.executable)} {shlex.quote(str(script))} "$@"\n',
                encoding="utf-8",
            )
            launcher.chmod(0o755)
        return launcher

    def bazel_calls(self):
        return [
            json.loads(line)
            for line in self.bazel_log.read_text(encoding="utf-8").splitlines()
        ]

    def qualifier(self, status=0):
        script = self.root / loom_wheel.TEST_DIRECTORY / "wheel_model_card_test.py"
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(f"raise SystemExit({status})\n", encoding="utf-8")

    def test_build_exports_exact_declared_inputs_without_running_them(self):
        for profile in loom_wheel.PROFILE_FLAGS:
            with self.subTest(profile=profile):
                path = self.wheel(profile)
                output = self.output / profile
                wheel = loom_wheel.build_wheel(
                    self.root,
                    output,
                    profile,
                    release.WINDOWS_X86_64_PLATFORM,
                    os.fspath(self.bazel(self.query(path))),
                    [],
                )
                self.assertEqual(wheel.path.read_bytes(), path.read_bytes())
                self.assertEqual(
                    wheel.digest, hashlib.sha256(path.read_bytes()).hexdigest()
                )
                self.assertEqual(
                    (output / "associated" / path.name).read_bytes(), path.read_bytes()
                )
                self.assertEqual(
                    (output / "model/weights.data").read_bytes(), b"weights"
                )
                self.assertFalse((output / "model/stale.data").exists())

    def test_fortify2_is_scoped_to_optimized_linux_artifacts(self):
        cases = (
            ("release", release.LINUX_X86_64_PLATFORM, True),
            ("tracy", release.LINUX_X86_64_PLATFORM, True),
            ("debug", release.LINUX_X86_64_PLATFORM, False),
            ("asan", release.LINUX_X86_64_PLATFORM, False),
            ("release", release.WINDOWS_X86_64_PLATFORM, False),
        )
        override = "--copt=-D_FORTIFY_SOURCE=0"
        for profile, platform, fortified in cases:
            with self.subTest(profile=profile, platform=platform):
                path = self.wheel(profile, platform)
                loom_wheel.build_wheel(
                    self.root,
                    self.output / f"{platform}-{profile}",
                    profile,
                    platform,
                    os.fspath(self.bazel(self.query(path))),
                    [override],
                )
                calls = self.bazel_calls()
                self.assertEqual([call[0] for call in calls], ["build", "cquery"])
                for call in calls:
                    for option in loom_wheel.LINUX_OPTIMIZED_COPTS:
                        self.assertEqual(option in call, fortified)
                    if fortified:
                        self.assertLess(
                            call.index(override),
                            call.index(loom_wheel.LINUX_OPTIMIZED_COPTS[0]),
                        )

    def test_failed_build_cannot_export_existing_output(self):
        path = self.wheel()
        bazel = self.bazel(self.query(path), build_status=1)
        with self.assertRaises(subprocess.CalledProcessError):
            loom_wheel.build_wheel(
                self.root,
                self.output,
                "release",
                release.WINDOWS_X86_64_PLATFORM,
                os.fspath(bazel),
                [],
            )
        self.assertFalse(self.output.exists())

    def test_missing_or_duplicate_declared_inputs_cannot_export(self):
        path = self.wheel()
        query = self.query(path)
        for document in ("", query + "\n" + query):
            with self.subTest(query=document):
                with self.assertRaises(ValueError):
                    loom_wheel.build_wheel(
                        self.root,
                        self.output,
                        "release",
                        release.WINDOWS_X86_64_PLATFORM,
                        os.fspath(self.bazel(document)),
                        [],
                    )
                self.assertFalse(self.output.exists())

    def test_profile_or_platform_mismatch_cannot_export(self):
        path = self.wheel("debug")
        cases = (
            (path, "release", release.WINDOWS_X86_64_PLATFORM, "Requested release"),
            (
                self.wheel(),
                "release",
                release.LINUX_X86_64_PLATFORM,
                "Requested manylinux_2_39_x86_64",
            ),
        )
        for path, profile, platform, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                loom_wheel.build_wheel(
                    self.root,
                    self.output,
                    profile,
                    platform,
                    os.fspath(self.bazel(self.query(path))),
                    [],
                )
            self.assertFalse(self.output.exists())

    def test_build_does_not_overlay_old_inputs(self):
        path = self.wheel()
        self.output.mkdir()
        destination = self.output / path.name
        destination.write_bytes(b"previous output")
        with self.assertRaises(FileExistsError):
            loom_wheel.build_wheel(
                self.root,
                self.output,
                "release",
                release.WINDOWS_X86_64_PLATFORM,
                os.fspath(self.bazel(self.query(path))),
                [],
            )
        self.assertEqual(destination.read_bytes(), b"previous output")

    def inputs(self, platform=None):
        platform = platform or (
            "win_amd64" if os.name == "nt" else "manylinux_2_39_x86_64"
        )
        path = self.wheel(platform=platform)
        loom_wheel.build_wheel(
            self.root,
            self.output,
            "release",
            platform,
            os.fspath(self.bazel(self.query(path))),
            [],
        )
        return path

    def test_successful_native_qualification_exports_publication_artifact(self):
        path = self.inputs()
        published = self.root / "published"
        self.qualifier()
        loom_wheel.test_wheel(self.root, self.output, published, "patchelf")
        self.assertEqual((published / path.name).read_bytes(), path.read_bytes())

    def test_failed_qualification_cannot_export_existing_wheel(self):
        self.inputs()
        published = self.root / "published"
        self.qualifier(status=1)
        with self.assertRaises(subprocess.CalledProcessError):
            loom_wheel.test_wheel(self.root, self.output, published, "patchelf")
        self.assertFalse(published.exists())

    def test_non_native_wheel_cannot_be_qualified(self):
        platform = "manylinux_2_39_x86_64" if os.name == "nt" else "win_amd64"
        self.inputs(platform)
        published = self.root / "published"
        self.qualifier()
        with self.assertRaises(ValueError):
            loom_wheel.test_wheel(self.root, self.output, published, "patchelf")
        self.assertFalse(published.exists())

    def test_cli_resolves_the_platform_launcher(self):
        path = self.wheel()
        self.bazel(self.query(path))
        with (
            chdir(self.root),
            patch.dict(
                os.environ, {"PATH": str(self.root) + os.pathsep + os.environ["PATH"]}
            ),
        ):
            self.assertEqual(
                loom_wheel.main(
                    [
                        "build",
                        "--profile=release",
                        "--platform=win_amd64",
                        "--output=" + str(self.output),
                        "--bazel=test-bazel",
                    ]
                ),
                0,
            )
        self.assertEqual((self.output / path.name).read_bytes(), path.read_bytes())


if __name__ == "__main__":
    unittest.main()
