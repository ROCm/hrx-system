# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Release metadata boundaries; installed SDK tests exercise the native payload."""

import contextlib
import hashlib
import io
import json
import stat
import tempfile
import unittest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

from release import (
    LINUX_X86_64_PLATFORM,
    PLATFORMS,
    PROFILES,
    STANDALONE_TOOLS,
    ReleaseChannel,
    checksum_document,
    inspect_standalone_archive,
    inspect_wheel,
    main,
    parse_identity,
    read_identity,
    release_artifacts,
    release_standalones,
    standalone_archive_name,
    standalone_binary_name,
    write_checksums,
    write_standalone_archive,
)


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.identity = parse_identity(
            {
                "package-version": "1.2.3rc1",
                "model-catalog": "github:fixture/library@v1.2.3rc1",
            }
        )
        self.description = "# Loom SDK\n\nA standalone description.\n"
        self.version_file = self.root / "version.json"
        self.version_file.write_text(json.dumps(self.identity.document()))
        self.readme = self.root / "PYPI_README.md"
        self.readme.write_text(self.description)

    def wheel(
        self, profile="release", platform=LINUX_X86_64_PLATFORM, replacements=None
    ):
        version = self.identity.version + (
            "" if profile == "release" else "+" + profile
        )
        path = self.root / f"hrx_loom-{version}-py3-none-{platform}.whl"
        prefix = f"hrx_loom-{version}.dist-info/"
        files = {
            "METADATA": (
                "Metadata-Version: 2.1\nName: hrx-loom\n"
                f"Version: {version}\nDescription-Content-Type: text/markdown\n"
                "Requires-Python: >=3.12\n\n" + self.description + "\n"
            ),
            "WHEEL": f"Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: py3-none-{platform}\n",
            "loom.json": json.dumps({"catalog": self.identity.catalog}),
        }
        for name, transform in (replacements or {}).items():
            files[name] = transform(files[name])
        with ZipFile(path, "w") as archive:
            for name, text in files.items():
                archive.writestr(prefix + name, text)
        return path

    def standalone(
        self,
        platform=LINUX_X86_64_PLATFORM,
        *,
        tool="loom-serve",
        member_name=None,
        timestamp=(1980, 1, 1, 0, 0, 0),
        create_system=3,
        mode=stat.S_IFREG | 0o755,
        compression=ZIP_DEFLATED,
        extra=b"",
        member_comment=b"",
        archive_comment=b"",
        payload=b"native executable",
    ):
        path = self.root / standalone_archive_name(self.identity, tool, platform)
        expected_name = standalone_binary_name(tool, platform)
        information = ZipInfo(member_name or expected_name, timestamp)
        information.create_system = create_system
        information.external_attr = mode << 16
        information.compress_type = compression
        information.extra = extra
        information.comment = member_comment
        with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
            archive.comment = archive_comment
            archive.writestr(information, payload)
        return path

    def command(self, *arguments):
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            return main(
                [
                    "--identity",
                    str(self.version_file),
                    "--description",
                    str(self.readme),
                    *arguments,
                ]
            )

    def test_canonical_versions_and_tag_boundary(self):
        versions = {
            "1.2.3": ReleaseChannel.STABLE,
            "1.2.3a1": ReleaseChannel.PRERELEASE,
            "1.2.3b2": ReleaseChannel.PRERELEASE,
            "1.2.3rc3": ReleaseChannel.PRERELEASE,
        }
        for version, channel in versions.items():
            with self.subTest(version=version):
                identity = parse_identity(
                    dict(self.identity.document(), **{"package-version": version})
                )
                self.assertEqual(identity.tag, "v" + version)
                self.assertIs(identity.check_tag(identity.tag), channel)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    identity.check_tag(version)
        for version in ("1.2.3.dev0", "1.2.3rc2.dev1"):
            identity = parse_identity(
                dict(self.identity.document(), **{"package-version": version})
            )
            with self.assertRaisesRegex(ValueError, "non-development"):
                identity.check_tag("v" + version)
        for version in (
            "v1.2.3",
            "01.2.3",
            "1.2.3-rc1",
            "1.2.3+tracy",
            "1.2.3.post1",
            123,
        ):
            with self.subTest(invalid=version), self.assertRaises(ValueError):
                parse_identity(
                    dict(self.identity.document(), **{"package-version": version})
                )
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.identity.check_tag("v1.2.4")

    def test_check_tag_reports_release_channel(self):
        for version, channel in (
            ("1.2.3", ReleaseChannel.STABLE),
            ("1.2.3rc1", ReleaseChannel.PRERELEASE),
        ):
            with self.subTest(version=version):
                identity = parse_identity(
                    dict(self.identity.document(), **{"package-version": version})
                )
                self.version_file.write_text(json.dumps(identity.document()))
                output = io.StringIO()
                with (
                    contextlib.redirect_stdout(output),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    self.assertEqual(
                        main(
                            [
                                "--identity",
                                str(self.version_file),
                                "check-tag",
                                identity.tag,
                            ]
                        ),
                        0,
                    )
                self.assertEqual(output.getvalue(), channel.value + "\n")

    def test_release_requires_public_catalog(self):
        for catalog in (
            "",
            "./catalog.json",
            "https://example.org/catalog.json",
            "github:../repo@v1",
            "github:owner/..@v1",
            "github:owner/repo@two words",
            True,
        ):
            with self.subTest(catalog=catalog), self.assertRaises(ValueError):
                parse_identity(
                    dict(self.identity.document(), **{"model-catalog": catalog})
                )
        identity = parse_identity(
            dict(self.identity.document(), **{"model-catalog": None})
        )
        with self.assertRaisesRegex(ValueError, "model catalog"):
            identity.check_tag("v" + identity.version)
        with self.assertRaises(ValueError):
            parse_identity(dict(self.identity.document(), override="ignored"))
        for document in (None, [], "1.2.3", {"package-version": "1.2.3"}):
            with self.subTest(document=document), self.assertRaises(ValueError):
                parse_identity(document)

    def test_every_platform_and_profile_keeps_its_identity(self):
        for profile in PROFILES:
            for platform in PLATFORMS:
                with self.subTest(profile=profile, platform=platform):
                    path = self.wheel(profile, platform)
                    wheel = inspect_wheel(path, self.identity, self.description)
                    self.assertEqual(
                        (wheel.profile, wheel.platform), (profile, platform)
                    )
                    self.assertEqual(
                        wheel.digest, hashlib.sha256(path.read_bytes()).hexdigest()
                    )
                    self.assertEqual(self.command("verify", str(path)), 0)

    def test_metadata_disagreement_is_rejected(self):
        changes = (
            ("METADATA", "hrx-loom", "hrx-solutions"),
            ("METADATA", "1.2.3rc1", "1.2.4"),
            ("METADATA", "text/markdown", "text/plain"),
            ("METADATA", ">=3.12", ">=3.9"),
            ("METADATA", "A standalone description.", "A stale description."),
            ("WHEEL", LINUX_X86_64_PLATFORM, "linux_x86_64"),
            ("WHEEL", "false", "true"),
            ("loom.json", "fixture/library", "other/library"),
        )
        for filename, before, after in changes:
            with self.subTest(filename=filename, before=before):
                path = self.wheel(
                    replacements={filename: lambda text: text.replace(before, after)}
                )
                with self.assertRaises(ValueError):
                    inspect_wheel(path, self.identity, self.description)

    def test_description_preserves_utf8(self):
        self.description = "# Loom SDK\n\nCPU → GPU; modèles et 日本語.\n"
        inspect_wheel(self.wheel(), self.identity, self.description)

    def test_duplicate_metadata_is_rejected(self):
        path = self.wheel()
        prefix = f"hrx_loom-{self.identity.version}.dist-info/"
        with ZipFile(path, "a") as archive, self.assertWarns(UserWarning):
            archive.writestr(prefix + "loom.json", '{"catalog":null}')
        with self.assertRaisesRegex(ValueError, "Duplicate ZIP"):
            inspect_wheel(path, self.identity, self.description)
        path = self.wheel(
            replacements={
                "METADATA": lambda text: text.replace(
                    "Name: hrx-loom", "Name: hrx-loom\nName: other"
                )
            }
        )
        with self.assertRaisesRegex(ValueError, "Invalid Name"):
            inspect_wheel(path, self.identity, self.description)

    def test_filename_cannot_relabel_an_artifact(self):
        for old, new in (
            ("1.2.3rc1", "1.2.4"),
            ("1.2.3rc1", "1.2.3rc1+tracy"),
            (LINUX_X86_64_PLATFORM, "linux_x86_64"),
            ("py3-none", "cp312-cp312"),
        ):
            with self.subTest(new=new):
                path = self.wheel()
                renamed = path.with_name(path.name.replace(old, new))
                path.rename(renamed)
                with self.assertRaises(ValueError):
                    inspect_wheel(renamed, self.identity, self.description)

    def test_standalone_archives_preserve_platform_identity(self):
        for tool in STANDALONE_TOOLS:
            for platform in PLATFORMS:
                with self.subTest(tool=tool, platform=platform):
                    path = self.standalone(platform, tool=tool)
                    artifact = inspect_standalone_archive(path, self.identity)
                    self.assertEqual(
                        (artifact.tool, artifact.platform), (tool, platform)
                    )
                    self.assertEqual(
                        artifact.digest, hashlib.sha256(path.read_bytes()).hexdigest()
                    )

    def test_archive_writer_preserves_payload_with_reproducible_metadata(self):
        binary = self.root / "amdf-info"
        binary.write_bytes(b"qualified executable")
        name = standalone_archive_name(
            self.identity, "amdf-info", LINUX_X86_64_PLATFORM
        )
        first = self.root / name
        write_standalone_archive(binary, first)
        artifact = inspect_standalone_archive(first, self.identity)
        with ZipFile(first) as archive:
            self.assertEqual(archive.read(binary.name), binary.read_bytes())
        other = self.root / "other"
        other.mkdir()
        second = other / name
        write_standalone_archive(binary, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(
            artifact.digest, inspect_standalone_archive(second, self.identity).digest
        )
        with self.assertRaises(FileExistsError):
            write_standalone_archive(binary, first)

    def test_standalone_archive_shape_is_exact(self):
        invalid = (
            {"member_name": "nested/loom-serve"},
            {"timestamp": (2026, 1, 1, 0, 0, 0)},
            {"create_system": 0},
            {"mode": stat.S_IFREG | 0o644},
            {"compression": ZIP_STORED},
            {"extra": b"\xfe\xca\x00\x00"},
            {"member_comment": b"comment"},
            {"archive_comment": b"comment"},
            {"payload": b""},
        )
        for options in invalid:
            with self.subTest(options=options):
                path = self.standalone(**options)
                with self.assertRaises(ValueError):
                    inspect_standalone_archive(path, self.identity)

        path = self.standalone()
        with ZipFile(path, "a") as archive:
            archive.writestr("companion.txt", "not part of the distribution")
        with self.assertRaisesRegex(ValueError, "must contain only"):
            inspect_standalone_archive(path, self.identity)

    def test_standalone_filename_cannot_relabel_an_archive(self):
        path = self.standalone()
        renamed = path.with_name(path.name.replace(self.identity.version, "1.2.4"))
        path.rename(renamed)
        with self.assertRaisesRegex(ValueError, "Unexpected standalone"):
            inspect_standalone_archive(renamed, self.identity)
        with self.assertRaisesRegex(ValueError, "Unexpected standalone"):
            release_standalones(self.root, self.identity)

        renamed.unlink()
        path = self.standalone()
        renamed = path.with_name(path.name.replace("loom-serve", "amdf-info"))
        path.rename(renamed)
        with self.assertRaisesRegex(ValueError, "must contain only amdf-info"):
            inspect_standalone_archive(renamed, self.identity)

    def test_bundle_is_complete_and_retries_preserve_exact_bytes(self):
        self.wheel()
        with self.assertRaises(SystemExit):
            self.command("bundle", str(self.root), "--tag", "v" + self.identity.version)
        self.assertFalse((self.root / "SHA256SUMS").exists())
        for profile in PROFILES:
            for platform in PLATFORMS:
                self.wheel(profile, platform)
        with self.assertRaises(SystemExit):
            self.command("bundle", str(self.root), "--tag", "v" + self.identity.version)
        for tool in STANDALONE_TOOLS:
            for platform in PLATFORMS:
                self.standalone(platform, tool=tool)
        self.command("bundle", str(self.root), "--tag", "v" + self.identity.version)
        artifacts = release_artifacts(self.root, self.identity, self.description)
        checksum = self.root / "SHA256SUMS"
        original = checksum.read_bytes()
        self.assertEqual(
            {artifact.path for artifact in artifacts},
            set(self.root.glob("*.whl")) | set(self.root.glob("*.zip")),
        )
        self.assertEqual(original, checksum_document(artifacts).encode())
        write_checksums(self.root, artifacts)
        self.assertEqual(checksum.read_bytes(), original)
        checksum.write_bytes(original.replace(b"\n", b"\r\n"))
        with self.assertRaisesRegex(ValueError, "SHA256SUMS differs"):
            write_checksums(self.root, artifacts)
        self.assertEqual(checksum.read_bytes(), original.replace(b"\n", b"\r\n"))
        checksum.write_bytes(original)
        with ZipFile(artifacts[0].path, "a") as archive:
            archive.writestr("changed-payload", "different bytes, identical metadata")
        changed = release_artifacts(self.root, self.identity, self.description)
        with self.assertRaisesRegex(ValueError, "SHA256SUMS differs"):
            write_checksums(self.root, changed)
        self.assertEqual(checksum.read_bytes(), original)

    def test_prepare_dry_run_and_edit_touch_only_identity(self):
        original = self.version_file.read_bytes()
        arguments = (
            "prepare",
            "--version",
            "1.2.4.dev0",
            "--catalog",
            "github:fixture/library@v1.2.4",
        )
        self.command(*arguments, "--dry-run")
        self.assertEqual(self.version_file.read_bytes(), original)
        self.command(*arguments)
        identity = read_identity(self.version_file)
        self.assertEqual(identity.version, "1.2.4.dev0")
        with self.assertRaisesRegex(ValueError, "non-development"):
            identity.check_tag("v1.2.4.dev0")
        self.assertEqual(identity.catalog, "github:fixture/library@v1.2.4")
        self.assertEqual(self.readme.read_text(), self.description)
        self.assertEqual(
            {path.name for path in self.root.iterdir()},
            {"version.json", "PYPI_README.md"},
        )

    def test_prepare_cannot_edit_implicit_bazel_runfiles(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["prepare", "--version", "1.2.4", "--catalog", self.identity.catalog])


if __name__ == "__main__":
    unittest.main()
