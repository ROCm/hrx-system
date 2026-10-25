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
import tempfile
import unittest
from contextlib import chdir
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

    def test_build_exports_exact_declared_inputs_without_running_them(self):
        for profile in loom_wheel.PROFILE_FLAGS:
            with self.subTest(profile=profile):
                path = self.wheel(profile)
                query = self.query(path)
                calls = []

                def bazel(arguments, **kwargs):
                    self.assertEqual(kwargs["cwd"], self.root)
                    calls.append(arguments)
                    return subprocess.CompletedProcess(arguments, 0, query)

                output = self.output / profile
                with patch("subprocess.run", side_effect=bazel):
                    wheel = loom_wheel.build_wheel(
                        self.root, output, profile, "bazel", ["--config=windows-x86_64"]
                    )
                self.assertEqual([call[1] for call in calls], ["build", "cquery"])
                self.assertEqual(calls[0][2:-3], calls[1][2:-3])
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

    def test_failed_build_cannot_export_existing_output(self):
        self.wheel()
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "bazel build"),
        ) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                loom_wheel.build_wheel(self.root, self.output, "release", "bazel", [])
        self.assertEqual(run.call_count, 1)
        self.assertFalse(self.output.exists())

    def test_query_or_profile_mismatch_cannot_export(self):
        path = self.wheel("debug")
        query = self.query(path)
        for document in ("", query + "\n" + query, query):
            with (
                self.subTest(query=document),
                patch(
                    "subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, document),
                ),
            ):
                with self.assertRaises(ValueError):
                    loom_wheel.build_wheel(
                        self.root, self.output, "release", "bazel", []
                    )
                self.assertFalse(self.output.exists())

    def test_build_does_not_overlay_old_inputs(self):
        path = self.wheel()
        self.output.mkdir()
        destination = self.output / path.name
        destination.write_bytes(b"previous output")
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, self.query(path)),
        ):
            with self.assertRaises(FileExistsError):
                loom_wheel.build_wheel(self.root, self.output, "release", "bazel", [])
        self.assertEqual(destination.read_bytes(), b"previous output")

    def inputs(self):
        platform = "win_amd64" if os.name == "nt" else "manylinux_2_39_x86_64"
        path = self.wheel(platform=platform)
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, self.query(path)),
        ):
            loom_wheel.build_wheel(self.root, self.output, "release", "bazel", [])
        return path

    def test_only_native_qualification_exports_publication_artifact(self):
        path = self.inputs()
        published = self.root / "published"
        with patch("subprocess.run") as run:
            loom_wheel.test_wheel(self.root, self.output, published, "patchelf")
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(Path(command[1]).name, "wheel_model_card_test.py")
        self.assertEqual(Path(command[2]), self.output / path.name)
        self.assertEqual(command[-1], "task,amdgpu,rdna,cdna,xdna")
        self.assertEqual((published / path.name).read_bytes(), path.read_bytes())

    def test_failed_qualification_cannot_export_existing_wheel(self):
        self.inputs()
        published = self.root / "published"
        with patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "installed test"),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                loom_wheel.test_wheel(self.root, self.output, published, "patchelf")
        self.assertFalse(published.exists())

    def test_cli_resolves_the_platform_launcher(self):
        path = self.wheel()
        query = self.query(path)
        if os.name == "nt":
            launcher = self.root / "test-bazel.cmd"
            launcher.write_text(
                "@echo off\n"
                + "".join(
                    f'@if "%1"=="cquery" echo {line}\n' for line in query.splitlines()
                )
                + "@exit /b 0\n"
            )
        else:
            launcher = self.root / "test-bazel"
            launcher.write_text(
                '#!/bin/sh\nif [ "$1" = cquery ]; then\n'
                f"  printf '%s\\n' {shlex.quote(query)}\nfi\n"
            )
            launcher.chmod(0o755)
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
                        "--output=" + str(self.output),
                        "--bazel=test-bazel",
                    ]
                ),
                0,
            )
        self.assertEqual((self.output / path.name).read_bytes(), path.read_bytes())


if __name__ == "__main__":
    unittest.main()
