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
        self.revision = "a" * 40
        self.catalog_revision = "b" * 40
        self.bundle("1.2.3")

    def bundle(self, version):
        self.identity = release.Identity(version, f"github:fixture/library@v{version}")
        self.editorial = loom_release.ReleaseEditorial(
            321,
            self.revision,
            f"Fixture release notes for {version}.\n\n"
            "## Highlights\n\n"
            "- Exercises the reviewed editorial boundary.",
        )
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
        for tool in release.STANDALONE_TOOLS:
            for platform in release.PLATFORMS:
                path = self.directory / release.standalone_archive_name(
                    self.identity, tool, platform
                )
                binary_name = release.standalone_binary_name(tool, platform)
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
        self.standalones = [
            artifact
            for artifact in self.artifacts
            if isinstance(artifact, release.StandaloneArchive)
        ]
        release.write_checksums(self.directory, self.artifacts)
        self.assets = {artifact.path.name: artifact.path for artifact in self.artifacts}
        self.assets["SHA256SUMS"] = self.directory / "SHA256SUMS"
        self.asset_digests = {
            name: loom_release._digest(path) for name, path in self.assets.items()
        }

    def release_pull_request(self, **changes):
        version = changes.pop("version", self.identity.version)
        editorial = changes.pop("editorial", self.editorial.text)
        body = (
            f"Prepares Loom SDK {version}.\n\n"
            "## Release notes\n\n"
            f"<!-- loom-release-notes:begin version={version} -->\n"
            f"{editorial}\n"
            "<!-- loom-release-notes:end -->\n\n"
            "## Reviewer Notes\n\nReview the release boundary."
        )
        return {
            "number": self.editorial.pull_request,
            "state": "closed",
            "merged_at": "2026-11-06T00:00:00Z",
            "merge_commit_sha": self.revision,
            "body": body,
        } | changes

    def publish(self):
        with patch.object(loom_release, "_require_catalog_release"):
            loom_release.publish_github(
                "fixture/sdk",
                self.revision,
                self.tag,
                self.identity,
                self.artifacts,
                self.assets["SHA256SUMS"],
            )

    def existing_release(self, *, draft=False, immutable=None):
        if immutable is None:
            immutable = not draft
        return dict(
            id=123,
            tag_name=self.tag,
            name=f"Loom {self.tag}",
            body=loom_release.github_release_body(
                self.identity, "fixture/sdk", self.editorial
            ),
            draft=draft,
            prerelease=(
                self.identity.release_channel is release.ReleaseChannel.PRERELEASE
            ),
            immutable=immutable,
        )

    def records(self, names):
        return [
            {
                "name": name,
                "state": "uploaded",
                "digest": "sha256:" + self.asset_digests.get(name, "0" * 64),
            }
            for name in names
        ]

    def catalog_release(self, **changes):
        return {
            "tag_name": self.tag,
            "draft": False,
            "prerelease": (
                self.identity.release_channel is release.ReleaseChannel.PRERELEASE
            ),
            "immutable": True,
        } | changes

    def catalog_pin(self, **changes):
        wheel = next(
            wheel
            for wheel in self.wheels
            if wheel.profile == "release"
            and wheel.platform == release.LINUX_X86_64_PLATFORM
        )
        return {
            "repository": "fixture/sdk",
            "revision": self.revision,
            "run_id": 123,
            "artifact_id": 456,
            "release_tag": self.tag,
            "wheel": wheel.path.name,
            "sha256": wheel.digest,
            "catalog": self.identity.catalog,
        } | changes

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

    def test_release_notes_come_from_the_exact_merged_pull_request(self):
        with patch.object(
            github_api, "get_json", return_value=[self.release_pull_request()]
        ) as get:
            actual = loom_release._release_editorial_from_pull_request(
                "fixture/sdk", self.revision, self.identity, "token"
            )
        self.assertEqual(actual, self.editorial)
        self.assertIn(f"commits/{self.revision}/pulls", get.call_args.args[0])

        invalid = (
            ([], "merge commit of one"),
            (
                [
                    self.release_pull_request(),
                    self.release_pull_request(number=322),
                ],
                "merge commit of one",
            ),
            ([self.release_pull_request(version="2.0.0")], "does not match"),
            ([self.release_pull_request(editorial="")], "notes are empty"),
            (
                [self.release_pull_request(editorial="Replace {{OUTCOME}}")],
                "contain placeholder",
            ),
            ([self.release_pull_request(body="No marked notes")], "one marked"),
            (
                [self.release_pull_request(merge_commit_sha="b" * 40)],
                "merge commit of one",
            ),
        )
        for records, message in invalid:
            with (
                self.subTest(message=message),
                patch.object(github_api, "get_json", return_value=records),
                self.assertRaisesRegex(ValueError, message),
            ):
                loom_release._release_editorial_from_pull_request(
                    "fixture/sdk", self.revision, self.identity, "token"
                )

    def test_catalog_tag_requires_a_matching_immutable_public_release(self):
        with (
            patch.object(
                github_api,
                "get_json",
                side_effect=[
                    self.catalog_release(),
                    {"object": {"type": "commit", "sha": self.catalog_revision}},
                ],
            ) as get,
            patch.object(
                github_api,
                "get_bytes",
                return_value=json.dumps(self.catalog_pin()).encode(),
            ),
        ):
            loom_release._require_catalog_release(
                "fixture/sdk",
                self.revision,
                self.identity,
                self.artifacts,
                "token",
            )
        self.assertIn(
            f"repos/fixture/library/releases/tags/{self.tag}",
            get.call_args_list[0].args[0],
        )

        for change in (
            {"tag_name": "different"},
            {"draft": True},
            {"prerelease": True},
            {"immutable": False},
        ):
            with (
                self.subTest(change=change),
                patch.object(
                    github_api,
                    "get_json",
                    return_value=self.catalog_release(**change),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "immutable public"):
                    loom_release._require_catalog_release(
                        "fixture/sdk",
                        self.revision,
                        self.identity,
                        self.artifacts,
                        "token",
                    )

        mismatched = release.Identity(
            self.identity.version, "github:fixture/library@different"
        )
        with self.assertRaisesRegex(ValueError, "matching versioned"):
            loom_release._require_catalog_release(
                "fixture/sdk",
                self.revision,
                mismatched,
                self.artifacts,
                "token",
            )

    def test_catalog_pin_must_name_the_exact_sdk_bytes(self):
        changes = (
            {"revision": "c" * 40},
            {"release_tag": "different"},
            {"wheel": "different.whl"},
            {"sha256": "d" * 64},
            {"catalog": "github:fixture/library@different"},
        )
        for change in changes:
            with (
                self.subTest(change=change),
                patch.object(
                    github_api,
                    "get_json",
                    side_effect=[
                        self.catalog_release(),
                        {
                            "object": {
                                "type": "commit",
                                "sha": self.catalog_revision,
                            }
                        },
                    ],
                ),
                patch.object(
                    github_api,
                    "get_bytes",
                    return_value=json.dumps(self.catalog_pin(**change)).encode(),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "exact SDK bytes"):
                    loom_release._require_catalog_release(
                        "fixture/sdk",
                        self.revision,
                        self.identity,
                        self.artifacts,
                        "token",
                    )

    def test_catalog_failure_prevents_sdk_release_mutation(self):
        with (
            patch.object(
                loom_release,
                "_require_catalog_release",
                side_effect=ValueError("catalog unavailable"),
            ),
            patch.object(github_api, "get_json") as get,
            patch("subprocess.run") as run,
        ):
            with self.assertRaisesRegex(ValueError, "catalog unavailable"):
                loom_release.publish_github(
                    "fixture/sdk",
                    self.revision,
                    self.tag,
                    self.identity,
                    self.artifacts,
                    self.assets["SHA256SUMS"],
                )
        self.assertEqual(get.call_count, 1)
        run.assert_not_called()

    def test_new_github_release_uses_existing_tag_and_complete_assets(self):
        for version in ("1.2.3", "1.2.3a1", "1.2.3b2", "1.2.3rc1"):
            if version != self.identity.version:
                self.bundle(version)
            with (
                self.subTest(version=version),
                patch.object(
                    github_api,
                    "get_json",
                    side_effect=[
                        urllib.error.HTTPError(
                            "https://api.github.com", 404, "Not Found", {}, None
                        ),
                        [self.release_pull_request()],
                        self.existing_release(draft=True),
                        self.records(self.assets),
                        self.existing_release(),
                    ],
                ),
                patch("subprocess.run") as run,
            ):
                self.publish()
                self.assertEqual(run.call_count, 2)
                command = run.call_args_list[0].args[0]
                self.assertEqual(command[:4], ["gh", "release", "create", self.tag])
                self.assertIn("--verify-tag", command)
                self.assertIn("--draft", command)
                self.assertNotIn("--clobber", command)
                self.assertNotIn("--generate-notes", command)
                prerelease = (
                    self.identity.release_channel is release.ReleaseChannel.PRERELEASE
                )
                self.assertEqual("--prerelease" in command, prerelease)
                self.assertEqual("--latest=false" in command, prerelease)
                body = command[command.index("--notes") + 1]
                self.assertIn(
                    f"loom-release-source: pr={self.editorial.pull_request}", body
                )
                self.assertLess(
                    body.index(self.editorial.text),
                    body.index("## Install and artifacts"),
                )
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
                for standalone in self.standalones:
                    self.assertIn(standalone.path.name, body)
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
                self.assertEqual(
                    run.call_args_list[1].args[0],
                    [
                        "gh",
                        "release",
                        "edit",
                        self.tag,
                        "--repo",
                        "fixture/sdk",
                        "--draft=false",
                    ],
                )

    def test_github_release_body_rejects_service_limit_overflow(self):
        limit = loom_release.GITHUB_RELEASE_BODY_MAXIMUM
        with self.assertRaisesRegex(ValueError, f"{limit}-character service limit"):
            loom_release.github_release_body(
                self.identity,
                "fixture/sdk",
                loom_release.ReleaseEditorial(321, self.revision, "x" * limit),
            )

    def test_partial_retry_checks_all_overlaps_before_upload(self):
        present = {
            self.wheels[0].path.name,
            self.standalones[0].path.name,
            "SHA256SUMS",
        }
        with (
            patch.object(
                github_api,
                "get_json",
                side_effect=[
                    self.existing_release(draft=True),
                    [self.release_pull_request()],
                    self.records(sorted(present)),
                    self.records(self.assets),
                    self.existing_release(),
                ],
            ),
            patch("subprocess.run") as run,
        ):
            self.publish()
        self.assertEqual(run.call_count, 2)
        upload = run.call_args_list[0].args[0]
        self.assertEqual(upload[2], "upload")
        self.assertCountEqual(
            upload[4 : upload.index("--repo")],
            [str(path) for name, path in self.assets.items() if name not in present],
        )
        self.assertNotIn("--clobber", upload)
        self.assertEqual(run.call_args_list[1].args[0][2], "edit")

    def test_conflicting_digest_prevents_all_uploads(self):
        present = {self.wheels[0].path.name, self.standalones[0].path.name}
        conflicting_name = max(present)
        records = self.records(sorted(present))
        next(record for record in records if record["name"] == conflicting_name)[
            "digest"
        ] = "sha256:" + "0" * 64

        with (
            patch.object(
                github_api,
                "get_json",
                side_effect=[
                    self.existing_release(draft=True),
                    [self.release_pull_request()],
                    records,
                ],
            ),
            patch("subprocess.run") as run,
        ):
            with self.assertRaisesRegex(ValueError, "Existing GitHub asset differs"):
                self.publish()
        run.assert_not_called()

    def test_complete_published_retry_is_read_only(self):
        with (
            patch.object(
                github_api,
                "get_json",
                side_effect=[self.existing_release(), self.records(self.assets)],
            ) as get_json,
            patch("subprocess.run") as run,
        ):
            self.publish()
        self.assertNotIn(
            "/pulls", " ".join(call.args[0] for call in get_json.call_args_list)
        )
        run.assert_not_called()

    def test_complete_published_retry_reads_every_asset_page(self):
        records = self.records(self.assets)
        pages = [records[index : index + 2] for index in range(0, len(records), 2)]
        with (
            patch.object(loom_release, "GITHUB_ASSET_PAGE_SIZE", 2),
            patch.object(
                github_api,
                "get_json",
                side_effect=[self.existing_release(), *pages],
            ) as get_json,
            patch("subprocess.run") as run,
        ):
            self.publish()
        self.assertIn("page=2", get_json.call_args_list[2].args[0])
        run.assert_not_called()

    def test_complete_draft_retry_publishes_without_uploading(self):
        with (
            patch.object(
                github_api,
                "get_json",
                side_effect=[
                    self.existing_release(draft=True),
                    [self.release_pull_request()],
                    self.records(self.assets),
                    self.existing_release(),
                ],
            ),
            patch("subprocess.run") as run,
        ):
            self.publish()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][2], "edit")

    def test_draft_retry_rejects_release_notes_edited_after_snapshot(self):
        with (
            patch.object(
                github_api,
                "get_json",
                side_effect=[
                    self.existing_release(draft=True),
                    [self.release_pull_request(editorial="Edited after snapshot")],
                ],
            ),
            patch("subprocess.run") as run,
            self.assertRaisesRegex(ValueError, "conflicting metadata"),
        ):
            self.publish()
        run.assert_not_called()

    def test_incomplete_published_release_cannot_be_modified(self):
        records = self.records(list(self.assets)[:-1])
        with (
            patch.object(
                github_api,
                "get_json",
                side_effect=[self.existing_release(), records],
            ),
            patch("subprocess.run") as run,
        ):
            with self.assertRaisesRegex(ValueError, "incomplete asset set"):
                self.publish()
        run.assert_not_called()

    def test_conflicting_release_metadata_stops_before_commands(self):
        for change in (
            {"prerelease": True},
            {"tag_name": "v2.0.0"},
            {"name": "Different"},
            {"body": "Different"},
            {"immutable": False},
        ):
            document = self.existing_release() | change
            with (
                self.subTest(change=change),
                patch.object(github_api, "get_json", return_value=document),
                patch("subprocess.run") as run,
            ):
                with self.assertRaises(ValueError):
                    self.publish()
                run.assert_not_called()

    def test_publication_requires_the_immutable_terminal_state(self):
        with (
            patch.object(
                github_api,
                "get_json",
                side_effect=[
                    self.existing_release(draft=True),
                    [self.release_pull_request()],
                    self.records(self.assets),
                    self.existing_release(immutable=False),
                ],
            ),
            patch("subprocess.run") as run,
        ):
            with self.assertRaisesRegex(ValueError, "mutable"):
                self.publish()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][2], "edit")

    def test_asset_inventory_is_complete_and_rejects_invalid_records(self):
        name = self.wheels[0].path.name
        unexpected_archives = [
            archive.path.name.replace(self.identity.version, "0.0.1", 1)
            for archive in self.standalones
        ]
        for records in (
            self.records([name, name]),
            self.records(["unexpected.whl"]),
            *(self.records([name]) for name in unexpected_archives),
            [self.records([name])[0] | {"state": "starter"}],
            [self.records([name])[0] | {"digest": None}],
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

    def test_cli_rejects_a_release_pull_request_without_notes_before_commands(self):
        with (
            patch.object(loom_release, "_require_catalog_release"),
            patch.object(
                github_api,
                "get_json",
                side_effect=[
                    urllib.error.HTTPError(
                        "https://api.github.com", 404, "Not Found", {}, None
                    ),
                    [self.release_pull_request(body="No marked release notes")],
                ],
            ) as github,
            patch("subprocess.run") as commands,
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            self.command(
                "github",
                "--repository=fixture/sdk",
                "--revision=" + self.revision,
            )
        self.assertTrue(any("/pulls" in call.args[0] for call in github.call_args_list))
        commands.assert_not_called()

    def test_cli_verifies_bundle_before_any_remote_operation(self):
        self.assets["SHA256SUMS"].write_text("conflict")
        with (
            patch.object(github_api, "get_json") as github,
            patch("urllib.request.urlopen") as pypi,
            patch("subprocess.run") as commands,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            for operation, arguments in (
                (
                    "github",
                    ["--repository=fixture/sdk", "--revision=" + self.revision],
                ),
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
