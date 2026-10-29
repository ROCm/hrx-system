# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Publication boundaries; remote HTTP and gh are the only replaced dependencies."""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile

from build_tools.ci import github_api, loom_release
from loom.packaging.python import release


class LoomReleaseTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.output = self.root / "pypi"
        self.description = "# SDK\n"
        self.bundle("1.2.3")

    def bundle(self, version):
        self.identity = release.Identity(version, f"github:fixture/library@v{version}")
        self.tag = "v" + version
        self.directory = self.root / version
        self.directory.mkdir()
        for profile in release.PROFILES:
            for platform in release.PLATFORMS:
                variant = version + ("" if profile == "release" else "+" + profile)
                path = self.directory / f"hrx_loom-{variant}-py3-none-{platform}.whl"
                prefix = f"hrx_loom-{variant}.dist-info/"
                with ZipFile(path, "w") as archive:
                    archive.writestr(
                        prefix + "METADATA",
                        f"Name: hrx-loom\nVersion: {variant}\n"
                        "Description-Content-Type: text/markdown\nRequires-Python: >=3.12\n"
                        "\n" + self.description + "\n",
                    )
                    archive.writestr(
                        prefix + "WHEEL",
                        f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-{platform}\n",
                    )
                    archive.writestr(
                        prefix + "loom.json",
                        json.dumps({"catalog": self.identity.catalog}),
                    )
        self.wheels = release.release_wheels(
            self.directory, self.identity, self.description
        )
        release.write_checksums(self.directory, self.wheels)
        self.assets = {wheel.path.name: wheel.path for wheel in self.wheels}
        self.assets["SHA256SUMS"] = self.directory / "SHA256SUMS"

    def publish(self):
        loom_release.publish_github(
            "fixture/sdk",
            self.tag,
            self.identity,
            self.wheels,
            self.assets["SHA256SUMS"],
        )

    def existing_release(self):
        return dict(id=123, tag_name=self.tag, draft=False, prerelease=False)

    def records(self, names):
        return [{"name": name, "state": "uploaded"} for name in names]

    def test_verification_requires_existing_matching_complete_checksums(self):
        actual = loom_release.verified_release(
            self.directory, self.identity, self.description, self.tag
        )
        self.assertEqual(actual, self.wheels)
        path = self.wheels[0].path
        path.write_bytes(path.read_bytes() + b"changed archive bytes")
        with self.assertRaisesRegex(ValueError, "SHA256SUMS differs"):
            loom_release.verified_release(
                self.directory, self.identity, self.description, self.tag
            )
        self.assets["SHA256SUMS"].unlink()
        with self.assertRaises(FileNotFoundError):
            loom_release.verified_release(
                self.directory, self.identity, self.description, self.tag
            )

    def test_new_github_release_uses_existing_tag_and_complete_assets(self):
        for version in ("1.2.3", "1.2.3rc1"):
            if version != self.identity.version:
                self.bundle(version)
            with (
                self.subTest(version=version),
                patch.object(
                    github_api,
                    "get_json",
                    side_effect=urllib.error.HTTPError(
                        "https://api.github.com", 404, "Not Found", {}, None
                    ),
                ),
                patch("subprocess.run") as run,
            ):
                self.publish()
                command = run.call_args.args[0]
                self.assertEqual(command[:4], ["gh", "release", "create", self.tag])
                self.assertIn("--verify-tag", command)
                self.assertNotIn("--clobber", command)
                self.assertNotIn("--generate-notes", command)
                self.assertEqual("--prerelease" in command, version.endswith("rc1"))
                body = command[command.index("--notes") + 1]
                self.assertIn(f"hrx-loom=={version}", body)
                self.assertIn("Release, Tracy, Debug, and ASAN", body)
                self.assertIn("SHA256SUMS", body)
                self.assertLessEqual(
                    len(body), loom_release.GITHUB_RELEASE_BODY_MAXIMUM
                )
                for path in self.assets.values():
                    self.assertIn(str(path), command)

    def test_github_release_body_rejects_service_limit_overflow(self):
        identity = release.Identity("1" * 125_000, self.identity.catalog)
        with self.assertRaisesRegex(ValueError, "125000-character service limit"):
            loom_release.github_release_body(identity)

    def test_partial_retry_checks_all_overlaps_before_upload(self):
        present = list(self.assets)[:2] + ["SHA256SUMS"]
        events = []

        def gh(command, **kwargs):
            events.append(command)
            if command[2] == "download":
                name = command[command.index("--pattern") + 1]
                directory = Path(command[command.index("--dir") + 1])
                (directory / name).write_bytes(self.assets[name].read_bytes())
            return subprocess.CompletedProcess(command, 0)

        with (
            patch.object(
                github_api,
                "get_json",
                side_effect=[self.existing_release(), self.records(present)],
            ),
            patch("subprocess.run", side_effect=gh),
        ):
            self.publish()
        self.assertEqual([event[2] for event in events], ["download"] * 3 + ["upload"])
        for name, path in self.assets.items():
            self.assertEqual(str(path) in events[-1], name not in present)
        self.assertNotIn("--clobber", events[-1])

    def test_conflicting_overlap_prevents_all_uploads(self):
        present = list(self.assets)[:2]
        events = []

        def gh(command, **kwargs):
            events.append(command[2])
            self.assertEqual(command[2], "download")
            name = command[command.index("--pattern") + 1]
            directory = Path(command[command.index("--dir") + 1])
            content = (
                self.assets[name].read_bytes() if len(events) == 1 else b"conflict"
            )
            (directory / name).write_bytes(content)

        with (
            patch.object(
                github_api,
                "get_json",
                side_effect=[self.existing_release(), self.records(present)],
            ),
            patch("subprocess.run", side_effect=gh),
        ):
            with self.assertRaisesRegex(ValueError, "Existing GitHub asset differs"):
                self.publish()
        self.assertEqual(events, ["download", "download"])

    def test_complete_retry_does_not_write(self):
        def gh(command, **kwargs):
            self.assertEqual(command[2], "download")
            name = command[command.index("--pattern") + 1]
            directory = Path(command[command.index("--dir") + 1])
            (directory / name).write_bytes(self.assets[name].read_bytes())

        with (
            patch.object(
                github_api,
                "get_json",
                side_effect=[self.existing_release(), self.records(self.assets)],
            ),
            patch("subprocess.run", side_effect=gh) as run,
        ):
            self.publish()
        self.assertEqual(run.call_count, 9)

    def test_conflicting_release_metadata_stops_before_commands(self):
        for change in ({"draft": True}, {"prerelease": True}, {"tag_name": "v2.0.0"}):
            document = self.existing_release() | change
            with (
                self.subTest(change=change),
                patch.object(github_api, "get_json", return_value=document),
                patch("subprocess.run") as run,
            ):
                with self.assertRaisesRegex(ValueError, "conflicting tag"):
                    self.publish()
                run.assert_not_called()

    def test_asset_inventory_is_complete_and_rejects_invalid_records(self):
        name = self.wheels[0].path.name
        for records in (
            self.records([name, name]),
            self.records(["unexpected.whl"]),
            [{"name": name, "state": "starter"}],
        ):
            with (
                self.subTest(records=records),
                patch.object(
                    github_api,
                    "get_json",
                    side_effect=[self.existing_release(), records],
                ),
                patch("subprocess.run") as run,
            ):
                with self.assertRaises(ValueError):
                    self.publish()
                run.assert_not_called()
        with (
            patch.object(
                github_api,
                "get_json",
                side_effect=[
                    self.existing_release(),
                    self.records([f"notes-{i}" for i in range(100)]),
                    self.records(["unexpected.whl"]),
                ],
            ) as get,
            patch("subprocess.run") as run,
        ):
            with self.assertRaisesRegex(ValueError, "Unexpected GitHub wheel"):
                self.publish()
            self.assertIn("page=2", get.call_args.args[0])
            run.assert_not_called()

    def test_github_authorization_and_service_errors_are_not_absence(self):
        for code in (401, 403, 429, 500):
            with (
                self.subTest(code=code),
                patch.object(
                    github_api,
                    "get_json",
                    side_effect=urllib.error.HTTPError(
                        "https://api.github.com", code, "Failure", {}, None
                    ),
                ),
                patch("subprocess.run") as run,
            ):
                with self.assertRaises(urllib.error.HTTPError):
                    self.publish()
                run.assert_not_called()

    def pypi_records(self):
        return [
            {
                "filename": wheel.path.name,
                "digests": {"sha256": wheel.digest},
                "yanked": False,
            }
            for wheel in self.wheels
            if wheel.profile == "release"
        ]

    def test_pypi_only_stages_absent_normal_wheels(self):
        normal = self.pypi_records()
        for count in (0, 1, 2):
            output = self.output / str(count)
            with (
                self.subTest(count=count),
                patch(
                    "urllib.request.urlopen",
                    return_value=io.BytesIO(
                        json.dumps({"urls": normal[:count]}).encode()
                    ),
                ),
            ):
                pending = loom_release.stage_pypi(self.wheels, self.identity, output)
            self.assertEqual(pending, count < 2)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {record["filename"] for record in normal[count:]},
            )
            for path in output.iterdir():
                self.assertEqual(path.read_bytes(), self.assets[path.name].read_bytes())

    def test_pypi_conflicts_are_not_silently_skipped(self):
        record = self.pypi_records()[0]
        for records in (
            [record | {"yanked": True}],
            [record | {"digests": {"sha256": "0" * 64}}],
            [record | {"filename": "hrx_loom-1.2.3+tracy-py3-none-win_amd64.whl"}],
            [record, record],
        ):
            with (
                self.subTest(records=records),
                patch(
                    "urllib.request.urlopen",
                    return_value=io.BytesIO(json.dumps({"urls": records}).encode()),
                ),
            ):
                with self.assertRaises(ValueError):
                    loom_release.stage_pypi(self.wheels, self.identity, self.output)
                self.assertFalse(self.output.exists())

    def test_only_pypi_404_means_not_published(self):
        for code in (401, 403, 429, 500, 404):
            with (
                self.subTest(code=code),
                patch(
                    "urllib.request.urlopen",
                    side_effect=urllib.error.HTTPError(
                        "https://pypi.org", code, "Failure", {}, None
                    ),
                ),
            ):
                if code == 404:
                    self.assertTrue(
                        loom_release.stage_pypi(self.wheels, self.identity, self.output)
                    )
                    self.assertEqual(len(list(self.output.iterdir())), 2)
                else:
                    with self.assertRaises(urllib.error.HTTPError):
                        loom_release.stage_pypi(self.wheels, self.identity, self.output)
                    self.assertFalse(self.output.exists())

    def command(self, operation, *arguments):
        identity = self.root / "version.json"
        identity.write_text(json.dumps(self.identity.document()))
        readme = self.root / "README.md"
        readme.write_text(self.description)
        return loom_release.main(
            [
                operation,
                "--tag",
                self.tag,
                "--directory",
                str(self.directory),
                "--identity",
                str(identity),
                "--description",
                str(readme),
                *arguments,
            ]
        )

    def test_cli_verifies_bundle_before_any_remote_operation(self):
        self.assets["SHA256SUMS"].write_text("conflict")
        with (
            patch.object(github_api, "get_json") as github,
            patch("urllib.request.urlopen") as pypi,
            patch("subprocess.run") as commands,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            for operation, arguments in (
                ("github", ["--repository=fixture/sdk"]),
                ("pypi-stage", ["--output", str(self.output)]),
            ):
                with self.subTest(operation=operation), self.assertRaises(SystemExit):
                    self.command(operation, *arguments)
            github.assert_not_called()
            pypi.assert_not_called()
            commands.assert_not_called()

    def test_cli_reports_complete_pypi_retry_without_upload(self):
        github_output = self.root / "github-output"
        with (
            patch.dict(os.environ, {"GITHUB_OUTPUT": str(github_output)}),
            patch(
                "urllib.request.urlopen",
                return_value=io.BytesIO(
                    json.dumps({"urls": self.pypi_records()}).encode()
                ),
            ),
            patch("subprocess.run") as commands,
        ):
            self.assertEqual(
                self.command("pypi-stage", "--output", str(self.output)), 0
            )
            commands.assert_not_called()
        self.assertEqual(github_output.read_text(), "pending=false\n")
        self.assertEqual(list(self.output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
