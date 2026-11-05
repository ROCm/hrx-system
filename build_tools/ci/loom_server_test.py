# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import contextlib
import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from zipfile import ZipFile

from build_tools.ci import loom_server, loom_server_qualify
from loom.packaging.python import release


class LoomServerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        package = self.root / "loom/packaging/python"
        package.mkdir(parents=True)
        self.identity = release.Identity("1.2.3", None)
        (package / "version.json").write_text(json.dumps(self.identity.document()))
        self.commit = "a" * 40
        self.llvm_bin = self.root / "llvm/bin"
        self.llvm_bin.mkdir(parents=True)

    def build_calls(self, source: Path, dependencies: list[str]):
        calls = []

        def run(arguments, **kwargs):
            calls.append((arguments, kwargs))
            if arguments[:3] == ["git", "rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(arguments, 0, self.commit + "\n")
            if len(arguments) > 1 and arguments[1] == "cquery":
                return subprocess.CompletedProcess(
                    arguments, 0, json.dumps([os.fspath(source.relative_to(self.root))])
                )
            if arguments[0] == os.fspath(self.llvm_bin / "llvm-readobj"):
                return subprocess.CompletedProcess(
                    arguments,
                    0,
                    json.dumps([{"NeededLibraries": dependencies}]),
                )
            return subprocess.CompletedProcess(arguments, 0, "")

        return calls, run

    def test_build_exports_the_declared_static_release_executable(self):
        source = self.root / "bazel-out/bin/loom-serve"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"executable")
        calls, run = self.build_calls(source, ["libc.so.6", "libm.so.6"])
        output = self.root / "output"
        with (
            mock.patch("subprocess.run", side_effect=run),
            mock.patch.object(
                loom_server.native_artifact_security, "inspect_file"
            ) as inspect_file,
        ):
            path = loom_server.build_server(
                self.root,
                output,
                release.LINUX_X86_64_PLATFORM,
                self.commit,
                "bazel",
                self.llvm_bin,
                ["--cxxopt=fixture"],
            )
        inspect_file.assert_called_once_with(
            self.llvm_bin / "llvm-readobj",
            mock.ANY,
            executable=True,
            runpath_policy=(
                loom_server.native_artifact_security.ElfRunpathPolicy.ABSENT
            ),
        )
        self.assertEqual(path.read_bytes(), b"executable")
        build = next(arguments for arguments, _ in calls if arguments[1] == "build")
        query = next(arguments for arguments, _ in calls if arguments[1] == "cquery")
        for required in (
            "--config=loom-sdk",
            "--config=opt",
            "--copt=--start-no-unused-arguments",
            "--copt=-U_FORTIFY_SOURCE",
            "--copt=--end-no-unused-arguments",
            "--copt=-D_FORTIFY_SOURCE=2",
            "--stamp",
            f"--embed_label=1.2.3@{self.commit}",
            "--//loom/config:shared_tools=false",
        ):
            self.assertIn(required, build)
            self.assertIn(required, query)

    def test_build_rejects_checkout_or_dynamic_dependency_mismatch(self):
        source = self.root / "bazel-out/bin/loom-serve"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"executable")
        output = self.root / "output"
        with mock.patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, "b" * 40 + "\n"),
        ):
            with self.assertRaisesRegex(ValueError, "does not match checkout"):
                loom_server.build_server(
                    self.root,
                    output,
                    release.LINUX_X86_64_PLATFORM,
                    self.commit,
                    "bazel",
                    self.llvm_bin,
                    [],
                )
        self.assertFalse(output.exists())

        _, run = self.build_calls(source, ["libc.so.6", "libloom.so"])
        with mock.patch("subprocess.run", side_effect=run):
            with self.assertRaisesRegex(ValueError, "unapproved dependencies"):
                loom_server.build_server(
                    self.root,
                    output,
                    release.LINUX_X86_64_PLATFORM,
                    self.commit,
                    "bazel",
                    self.llvm_bin,
                    [],
                )
        self.assertFalse(output.exists())

    def test_windows_import_policy_accepts_only_core_operating_system_apis(self):
        output = "\n".join(
            [
                "  Name: api-ms-win-core-memory-l1-1-0.dll",
                "  Name: bcrypt.dll",
                "  Name: KERNEL32.dll",
                "  Name: WS2_32.dll",
            ]
        )
        with mock.patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, output),
        ):
            dependencies = loom_server._windows_dependencies(
                self.llvm_bin / "llvm-readobj", self.root / "loom-serve.exe"
            )
        self.assertIn("KERNEL32.dll", dependencies)
        with mock.patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                [], 0, output + "\n  Name: VCRUNTIME140.dll\n"
            ),
        ):
            with self.assertRaisesRegex(ValueError, "VCRUNTIME140"):
                loom_server._windows_dependencies(
                    self.llvm_bin / "llvm-readobj", self.root / "loom-serve.exe"
                )

    def server_inputs(self, platform: str) -> Path:
        inputs = self.root / f"inputs-{platform}"
        inputs.mkdir()
        binary = inputs / release.SERVER_BINARIES[platform]
        binary.write_bytes(b"standalone server")
        return inputs

    def test_native_qualification_is_the_only_archive_export_boundary(self):
        for platform in release.PLATFORMS:
            with self.subTest(platform=platform):
                inputs = self.server_inputs(platform)
                first = self.root / f"first-{platform}"
                second = self.root / f"second-{platform}"
                with (
                    mock.patch.object(
                        loom_server, "_native_platform", return_value=platform
                    ),
                    mock.patch(
                        "subprocess.run",
                        return_value=subprocess.CompletedProcess([], 0),
                    ) as run,
                ):
                    one = loom_server.qualify_server(
                        self.root, inputs, first, platform, self.commit
                    )
                    two = loom_server.qualify_server(
                        self.root, inputs, second, platform, self.commit
                    )
                self.assertEqual(one.path.read_bytes(), two.path.read_bytes())
                self.assertEqual(one.digest, two.digest)
                command = run.call_args_list[0].args[0]
                self.assertEqual(Path(command[1]).name, "loom_server_qualify.py")
                self.assertEqual(command[-2:], ["1.2.3", self.commit])
                binary_name = release.SERVER_BINARIES[platform]
                self.assertEqual(
                    one.path.name, release.server_archive_name(self.identity, platform)
                )
                with ZipFile(one.path) as archive:
                    self.assertEqual(archive.namelist(), [binary_name])
                    information = archive.getinfo(binary_name)
                    self.assertEqual(information.date_time, (1980, 1, 1, 0, 0, 0))
                    self.assertEqual(
                        stat.S_IMODE(information.external_attr >> 16), 0o755
                    )
                    self.assertEqual(archive.read(binary_name), b"standalone server")

    def test_failed_native_qualification_exports_nothing(self):
        platform = release.LINUX_X86_64_PLATFORM
        inputs = self.server_inputs(platform)
        output = self.root / "published"
        with (
            mock.patch.object(loom_server, "_native_platform", return_value=platform),
            mock.patch(
                "subprocess.run",
                side_effect=subprocess.CalledProcessError(1, "qualification"),
            ),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                loom_server.qualify_server(
                    self.root, inputs, output, platform, self.commit
                )
        self.assertFalse(output.exists())

    def test_non_native_qualification_exports_nothing(self):
        platform = release.LINUX_X86_64_PLATFORM
        inputs = self.server_inputs(platform)
        output = self.root / "published"
        with mock.patch.object(
            loom_server,
            "_native_platform",
            return_value=release.WINDOWS_X86_64_PLATFORM,
        ):
            with self.assertRaisesRegex(ValueError, "native OS"):
                loom_server.qualify_server(
                    self.root, inputs, output, platform, self.commit
                )
        self.assertFalse(output.exists())

    def test_qualification_rejects_companion_files(self):
        platform = release.LINUX_X86_64_PLATFORM
        inputs = self.server_inputs(platform)
        (inputs / "libloom.so").write_bytes(b"companion")
        with (
            mock.patch.object(loom_server, "_native_platform", return_value=platform),
            self.assertRaisesRegex(ValueError, "require only loom-serve"),
        ):
            loom_server.qualify_server(
                self.root, inputs, self.root / "published", platform, self.commit
            )


class LoomServerQualifierTest(unittest.TestCase):
    def test_identity_help_and_empty_http_server_are_exercised(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / "loom-serve"
            binary.write_bytes(b"executable")
            identity = "loom-serve 1.2.3 (commit " + "a" * 40 + ")"
            process = mock.Mock(returncode=0)

            @contextlib.contextmanager
            def server(*args, **kwargs):
                yield process, "127.0.0.1:1234"

            responses = [
                {"status": "ready"},
                {"data": [], "server": {"resources": {}}},
            ]
            commands = []

            def run(arguments, **kwargs):
                commands.append((arguments, kwargs))
                output = (
                    identity + "\n"
                    if arguments[-1] == "--version"
                    else identity + "\n\n--device --listen --model\n"
                )
                return subprocess.CompletedProcess(arguments, 0, output, "")

            with (
                mock.patch.object(
                    loom_server_qualify.subprocess, "run", side_effect=run
                ),
                mock.patch.object(loom_server_qualify, "running_server", server),
                mock.patch.object(
                    loom_server_qualify, "control", side_effect=responses
                ) as control,
                mock.patch.dict(
                    os.environ,
                    {"LD_LIBRARY_PATH": "/untrusted", "PYTHONPATH": "/source"},
                    clear=False,
                ),
            ):
                loom_server_qualify.qualify(binary, "1.2.3", "a" * 40)
            self.assertEqual(
                [call[0][-1] for call in commands], ["--version", "--help"]
            )
            self.assertNotIn("LD_LIBRARY_PATH", commands[0][1]["env"])
            self.assertNotIn("PYTHONPATH", commands[0][1]["env"])
            self.assertEqual(
                [call.args[1:] for call in control.call_args_list],
                [("GET", "/readyz", 200), ("GET", "/v1/status", 200)],
            )


if __name__ == "__main__":
    unittest.main()
