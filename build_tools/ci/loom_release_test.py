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
import stat
import subprocess
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

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
        for platform in release.PLATFORMS:
            path = self.directory / release.server_archive_name(self.identity, platform)
            binary_name = release.SERVER_BINARIES[platform]
            information = ZipInfo(binary_name, (1980, 1, 1, 0, 0, 0))
            information.create_system = 3
            information.external_attr = (stat.S_IFREG | 0o755) << 16
            information.compress_type = ZIP_DEFLATED
            with ZipFile(
                path, "w", compression=ZIP_DEFLATED, compresslevel=9
            ) as archive:
                archive.writestr(information, b"native executable")
        self.wheels = release.release_wheels(
            self.directory, self.identity, self.description
        )
        self.artifacts = release.release_artifacts(
            self.directory, self.identity, self.description
        )
        self.servers = [
            artifact
            for artifact in self.artifacts
            if isinstance(artifact, release.ServerArchive)
        ]
        release.write_checksums(self.directory, self.artifacts)
        self.assets = {artifact.path.name: artifact.path for artifact in self.artifacts}
        self.assets["SHA256SUMS"] = self.directory / "SHA256SUMS"

    def publish(self):
        loom_release.publish_github(
            "fixture/sdk",
            self.tag,
            self.identity,
            self.artifacts,
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
        self.assertEqual(actual, self.artifacts)
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
        for version in ("1.2.3", "1.2.3a1", "1.2.3b2", "1.2.3rc1"):
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
                prerelease = (
                    self.identity.release_channel is release.ReleaseChannel.PRERELEASE
                )
                self.assertEqual("--prerelease" in command, prerelease)
                body = command[command.index("--notes") + 1]
                if prerelease:
                    self.assertIn("not published to PyPI", body)
                    self.assertNotIn(f"hrx-loom=={version}", body)
                    for platform in release.PLATFORMS:
                        wheel = f"hrx_loom-{version}-py3-none-{platform}.whl"
                        self.assertIn(
                            f"https://github.com/fixture/sdk/releases/download/"
                            f"{self.identity.tag}/{wheel}",
                            body,
                        )
                else:
                    self.assertIn(f"hrx-loom=={version}", body)
                for server in self.servers:
                    self.assertIn(server.path.name, body)
                self.assertIn("./loom-serve --version", body)
                self.assertIn("Release, Tracy, Debug, and ASAN", body)
                self.assertIn("SHA256SUMS", body)
                self.assertLessEqual(
                    len(body), loom_release.GITHUB_RELEASE_BODY_MAXIMUM
                )
                asset_arguments = command[4 : command.index("--repo")]
                self.assertCountEqual(
                    asset_arguments, [str(path) for path in self.assets.values()]
                )
                self.assertEqual(len(asset_arguments), len(self.assets))

    def test_github_release_body_rejects_service_limit_overflow(self):
        limit = loom_release.GITHUB_RELEASE_BODY_MAXIMUM
        identity = release.Identity("1" * limit, self.identity.catalog)
        with self.assertRaisesRegex(ValueError, f"{limit}-character service limit"):
            loom_release.github_release_body(identity, "fixture/sdk")

    def test_partial_retry_checks_all_overlaps_before_upload(self):
        present = {
            self.wheels[0].path.name,
            self.servers[0].path.name,
            "SHA256SUMS",
        }
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
                side_effect=[self.existing_release(), self.records(sorted(present))],
            ),
            patch("subprocess.run", side_effect=gh),
        ):
            self.publish()
        downloads = [
            event[event.index("--pattern") + 1]
            for event in events
            if event[2] == "download"
        ]
        self.assertEqual(downloads, sorted(present))
        upload = events[-1]
        self.assertEqual(upload[2], "upload")
        self.assertCountEqual(
            upload[4 : upload.index("--repo")],
            [str(path) for name, path in self.assets.items() if name not in present],
        )
        self.assertNotIn("--clobber", upload)

    def test_conflicting_overlap_prevents_all_uploads(self):
        present = {self.wheels[0].path.name, self.servers[0].path.name}
        conflicting_name = max(present)
        downloads = []

        def gh(command, **kwargs):
            self.assertEqual(command[2], "download")
            name = command[command.index("--pattern") + 1]
            downloads.append(name)
            directory = Path(command[command.index("--dir") + 1])
            content = b"conflict"
            if name != conflicting_name:
                content = self.assets[name].read_bytes()
            (directory / name).write_bytes(content)

        with (
            patch.object(
                github_api,
                "get_json",
                side_effect=[self.existing_release(), self.records(sorted(present))],
            ),
            patch("subprocess.run", side_effect=gh),
        ):
            with self.assertRaisesRegex(ValueError, "Existing GitHub asset differs"):
                self.publish()
        self.assertEqual(downloads, sorted(present))

    def test_complete_retry_downloads_every_asset_once_and_does_not_write(self):
        downloads = []

        def gh(command, **kwargs):
            self.assertEqual(command[2], "download")
            name = command[command.index("--pattern") + 1]
            downloads.append(name)
            directory = Path(command[command.index("--dir") + 1])
            (directory / name).write_bytes(self.assets[name].read_bytes())

        with (
            patch.object(
                github_api,
                "get_json",
                side_effect=[self.existing_release(), self.records(self.assets)],
            ),
            patch("subprocess.run", side_effect=gh),
        ):
            self.publish()
        self.assertEqual(downloads, sorted(self.assets))

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
        server = self.servers[0]
        unexpected_server = server.path.name.replace(self.identity.version, "0.0.1", 1)
        for records in (
            self.records([name, name]),
            self.records(["unexpected.whl"]),
            self.records([unexpected_server]),
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
                    self.records(
                        [
                            f"notes-{i}"
                            for i in range(loom_release.GITHUB_ASSET_PAGE_SIZE)
                        ]
                    ),
                    self.records(["unexpected.whl"]),
                ],
            ) as get,
            patch("subprocess.run") as run,
        ):
            with self.assertRaisesRegex(ValueError, "Unexpected GitHub release"):
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
        for count in range(len(normal) + 1):
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
            self.assertEqual(pending, count < len(normal))
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {record["filename"] for record in normal[count:]},
            )
            for path in output.iterdir():
                self.assertEqual(path.read_bytes(), self.assets[path.name].read_bytes())

    def test_pypi_rejects_prerelease_before_network_or_staging(self):
        self.bundle("1.2.3rc1")
        with (
            patch("urllib.request.urlopen") as request,
            self.assertRaisesRegex(ValueError, "stable X.Y.Z"),
        ):
            loom_release.stage_pypi(self.wheels, self.identity, self.output)
        request.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_pypi_conflicts_are_not_silently_skipped(self):
        record = self.pypi_records()[0]
        instrumented = next(
            wheel.path.name for wheel in self.wheels if wheel.profile != "release"
        )
        for records in (
            [record | {"yanked": True}],
            [record | {"digests": {"sha256": "0" * 64}}],
            [record | {"filename": instrumented}],
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
                    self.assertEqual(
                        len(list(self.output.iterdir())), len(self.pypi_records())
                    )
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
