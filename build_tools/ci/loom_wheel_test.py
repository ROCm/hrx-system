# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from build_tools.ci import loom_wheel


class LoomWheelTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "export"
        package = self.root / "loom" / "packaging" / "python"
        package.mkdir(parents=True)
        (package / "version.json").write_text(
            json.dumps({"package-version": "1.2.3", "model-catalog": None})
        )
        (package / "PYPI_README.md").write_text("SDK\n")

    def wheel(self, profile="release"):
        version = "1.2.3" + ("" if profile == "release" else "+" + profile)
        path = self.root / f"hrx_loom-{version}-py3-none-win_amd64.whl"
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
                "Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-win_amd64\n",
            )
            archive.writestr(prefix + "loom.json", '{"catalog": null}')
        return path

    def test_exports_only_the_tested_configuration(self):
        for profile in loom_wheel.PROFILE_FLAGS:
            with self.subTest(profile=profile):
                path = self.wheel(profile)
                calls = []

                def bazel(arguments, **kwargs):
                    self.assertEqual(kwargs["cwd"], self.root)
                    calls.append(arguments)
                    return subprocess.CompletedProcess(arguments, 0, path.name + "\n")

                with patch("subprocess.run", side_effect=bazel):
                    wheel = loom_wheel.build_wheel(
                        self.root, self.output, profile, "bazel", ["--jobs=2"]
                    )
                self.assertEqual(calls[0][1], "test")
                self.assertEqual(calls[0][-1], loom_wheel.INSTALLED_TEST)
                self.assertEqual(calls[1][1], "cquery")
                self.assertEqual(calls[0][2:-1], calls[1][2:-2])
                self.assertEqual(wheel.path.read_bytes(), path.read_bytes())
                self.assertEqual(
                    wheel.digest, hashlib.sha256(path.read_bytes()).hexdigest()
                )

    def test_failed_test_cannot_export_existing_output(self):
        self.wheel()
        with patch(
            "subprocess.run", side_effect=subprocess.CalledProcessError(1, "bazel test")
        ) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                loom_wheel.build_wheel(self.root, self.output, "release", "bazel", [])
        self.assertEqual(run.call_count, 1)
        self.assertFalse(self.output.exists())

    def test_query_or_profile_mismatch_cannot_export(self):
        path = self.wheel("debug")
        for query in ("", f"{path}\n{path}\n", str(path) + "\n"):
            with (
                self.subTest(query=query),
                patch(
                    "subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, query),
                ),
            ):
                with self.assertRaises(ValueError):
                    loom_wheel.build_wheel(
                        self.root, self.output, "release", "bazel", []
                    )
                self.assertFalse(self.output.exists())

    def test_export_does_not_overwrite_an_existing_artifact(self):
        path = self.wheel()
        self.output.mkdir()
        destination = self.output / path.name
        destination.write_bytes(b"previous output")
        with patch(
            "subprocess.run", return_value=subprocess.CompletedProcess([], 0, str(path))
        ):
            with self.assertRaises(FileExistsError):
                loom_wheel.build_wheel(self.root, self.output, "release", "bazel", [])
        self.assertEqual(destination.read_bytes(), b"previous output")


if __name__ == "__main__":
    unittest.main()
