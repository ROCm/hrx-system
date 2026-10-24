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
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from release import (
    PLATFORMS,
    PROFILES,
    checksum_document,
    inspect_wheel,
    main,
    parse_identity,
    read_identity,
    release_wheels,
    write_checksums,
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

    def wheel(self, profile="release", platform=PLATFORMS[0], replacements=None):
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
        for version in ("1.2.3", "1.2.3a1", "1.2.3b2", "1.2.3rc3"):
            with self.subTest(version=version):
                identity = parse_identity(
                    dict(self.identity.document(), **{"package-version": version})
                )
                identity.check_tag("v" + version)
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
            ("WHEEL", "manylinux_2_39_x86_64", "linux_x86_64"),
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
            (PLATFORMS[0], "linux_x86_64"),
            ("py3-none", "cp312-cp312"),
        ):
            with self.subTest(new=new):
                path = self.wheel()
                renamed = path.with_name(path.name.replace(old, new))
                path.rename(renamed)
                with self.assertRaises(ValueError):
                    inspect_wheel(renamed, self.identity, self.description)

    def test_bundle_is_complete_and_retries_preserve_exact_bytes(self):
        self.wheel()
        with self.assertRaises(SystemExit):
            self.command("bundle", str(self.root), "--tag", "v" + self.identity.version)
        self.assertFalse((self.root / "SHA256SUMS").exists())
        for profile in PROFILES:
            for platform in PLATFORMS:
                self.wheel(profile, platform)
        self.command("bundle", str(self.root), "--tag", "v" + self.identity.version)
        wheels = release_wheels(self.root, self.identity, self.description)
        checksum = self.root / "SHA256SUMS"
        original = checksum.read_bytes()
        self.assertEqual(original, checksum_document(wheels).encode())
        write_checksums(self.root, wheels)
        self.assertEqual(checksum.read_bytes(), original)
        checksum.write_bytes(original.replace(b"\n", b"\r\n"))
        with self.assertRaisesRegex(ValueError, "SHA256SUMS differs"):
            write_checksums(self.root, wheels)
        self.assertEqual(checksum.read_bytes(), original.replace(b"\n", b"\r\n"))
        checksum.write_bytes(original)
        with ZipFile(wheels[0].path, "a") as archive:
            archive.writestr("changed-payload", "different bytes, identical metadata")
        changed = release_wheels(self.root, self.identity, self.description)
        with self.assertRaisesRegex(ValueError, "SHA256SUMS differs"):
            write_checksums(self.root, changed)
        self.assertEqual(checksum.read_bytes(), original)

    def test_prepare_dry_run_and_edit_touch_only_identity(self):
        original = self.version_file.read_bytes()
        arguments = (
            "prepare",
            "--version",
            "1.2.4",
            "--catalog",
            "github:fixture/library@v1.2.4",
        )
        self.command(*arguments, "--dry-run")
        self.assertEqual(self.version_file.read_bytes(), original)
        self.command(*arguments)
        identity = read_identity(self.version_file)
        identity.check_tag("v1.2.4")
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
