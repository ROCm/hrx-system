# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Prepare SDK identities and verify exact release artifacts without publishing."""

import argparse
import email.parser
import enum
import hashlib
import json
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

_NUMBER = r"(?:0|[1-9][0-9]*)"
_RELEASE = rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}(?:(?:a|b|rc){_NUMBER})?"
_VERSION = re.compile(rf"{_RELEASE}(?:\.dev{_NUMBER})?")
_CATALOG = re.compile(r"github:([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)@([^\s]+)")
PROFILES = ("release", "tracy", "debug", "asan")
LINUX_X86_64_PLATFORM = "manylinux_2_39_x86_64"
WINDOWS_X86_64_PLATFORM = "win_amd64"
PLATFORMS = (LINUX_X86_64_PLATFORM, WINDOWS_X86_64_PLATFORM)
SERVER_BINARIES = {
    LINUX_X86_64_PLATFORM: "loom-serve",
    WINDOWS_X86_64_PLATFORM: "loom-serve.exe",
}


class ReleaseChannel(enum.StrEnum):
    STABLE = "stable"
    PRERELEASE = "prerelease"


@dataclass(frozen=True)
class Identity:
    # Canonical PEP 440 base version; native configuration owns local suffixes.
    version: str
    # Published discovery reference, or None for an unassociated build.
    catalog: str | None

    @property
    def tag(self) -> str:
        return "v" + self.version

    @property
    def release_channel(self) -> ReleaseChannel:
        if ".dev" in self.version:
            raise ValueError("A release needs a non-development version")
        if re.fullmatch(rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}", self.version):
            return ReleaseChannel.STABLE
        return ReleaseChannel.PRERELEASE

    def check_tag(self, tag: str) -> ReleaseChannel:
        if self.catalog is None:
            raise ValueError("A release needs a model catalog")
        channel = self.release_channel
        if tag != self.tag:
            raise ValueError(f"Tag {tag!r} does not match version.json ({self.tag})")
        return channel

    def document(self) -> dict:
        return {"package-version": self.version, "model-catalog": self.catalog}


def parse_identity(document: object) -> Identity:
    if not isinstance(document, dict) or set(document) != {
        "package-version",
        "model-catalog",
    }:
        raise ValueError("version.json requires package-version and model-catalog")
    version = document["package-version"]
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise ValueError(
            "package-version must use X.Y.Z, X.Y.ZaN, X.Y.ZbN, or X.Y.ZrcN, "
            "optionally followed by .devN"
        )
    catalog = document["model-catalog"]
    if catalog is not None:
        match = _CATALOG.fullmatch(catalog) if isinstance(catalog, str) else None
        if match is None or any(part in (".", "..") for part in match.groups()[:2]):
            raise ValueError(
                "model-catalog must be github:OWNER/REPOSITORY@TAG_OR_COMMIT"
            )
    return Identity(version, catalog)


def read_identity(path: Path) -> Identity:
    return parse_identity(json.loads(path.read_text(encoding="utf-8")))


def server_archive_name(identity: Identity, platform: str) -> str:
    if platform not in SERVER_BINARIES:
        raise ValueError(f"Unsupported standalone server platform: {platform}")
    return f"loom-serve-{identity.version}-{platform}.zip"


@dataclass(frozen=True)
class Wheel:
    # Exact artifact inspected; publishers retain these bytes instead of rebuilding.
    path: Path
    # Native instrumentation profile represented by the installed version.
    profile: str
    # Audited Linux or native Windows platform tag.
    platform: str
    # SHA-256 of the complete wheel archive, including its metadata.
    digest: str


@dataclass(frozen=True)
class ServerArchive:
    # Exact standalone archive inspected; publishers retain these bytes.
    path: Path
    # Audited Linux or native Windows platform tag.
    platform: str
    # SHA-256 of the complete ZIP archive.
    digest: str


ReleaseArtifact = Wheel | ServerArchive


def inspect_wheel(path: Path, identity: Identity, description: str) -> Wheel:
    versions = {
        identity.version + ("" if profile == "release" else "+" + profile): profile
        for profile in PROFILES
    }
    parts = path.name.removesuffix(".whl").split("-")
    if (
        path.suffix != ".whl"
        or len(parts) != 5
        or parts[0] != "hrx_loom"
        or parts[1] not in versions
        or parts[2:4] != ["py3", "none"]
        or parts[4] not in PLATFORMS
    ):
        raise ValueError(f"Unexpected SDK release wheel: {path.name}")
    _, version, _, _, platform = parts
    dist_info = f"hrx_loom-{version}.dist-info/"
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate ZIP entries in {path.name}")
        expected = {
            "METADATA": dist_info + "METADATA",
            "WHEEL": dist_info + "WHEEL",
            "loom.json": dist_info + "loom.json",
        }
        for filename, member in expected.items():
            actual = [name for name in names if name.endswith(".dist-info/" + filename)]
            if actual != [member]:
                raise ValueError(f"Invalid {filename} identity in {path.name}")
        metadata = email.parser.Parser().parsestr(
            archive.read(expected["METADATA"]).decode("utf-8")
        )
        for field, value in (
            ("Name", "hrx-loom"),
            ("Version", version),
            ("Description-Content-Type", "text/markdown"),
            ("Requires-Python", ">=3.12"),
        ):
            if metadata.get_all(field) != [value]:
                raise ValueError(f"Invalid {field} in {path.name}; expected {value!r}")
        # rules_python appends a newline after the authored description.
        if metadata.get_payload() != description + "\n":
            raise ValueError(f"Description differs from PYPI_README.md in {path.name}")
        wheel = email.parser.Parser().parsestr(
            archive.read(expected["WHEEL"]).decode("utf-8")
        )
        for field, value in (
            ("Wheel-Version", "1.0"),
            ("Root-Is-Purelib", "false"),
            ("Tag", "py3-none-" + platform),
        ):
            if wheel.get_all(field) != [value]:
                raise ValueError(f"Invalid {field} in {path.name}; expected {value!r}")
        if json.loads(archive.read(expected["loom.json"])) != {
            "catalog": identity.catalog
        }:
            raise ValueError(f"Model catalog differs from version.json in {path.name}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return Wheel(path, versions[version], platform, digest)


def release_wheels(
    directory: Path, identity: Identity, description: str
) -> list[Wheel]:
    wheels = [
        inspect_wheel(path, identity, description)
        for path in sorted(directory.glob("*.whl"))
    ]
    expected = {(profile, platform) for profile in PROFILES for platform in PLATFORMS}
    actual = {(wheel.profile, wheel.platform) for wheel in wheels}
    missing = expected - actual
    if missing:
        raise ValueError(
            "Release requires one wheel per platform and profile; "
            f"missing={sorted(missing)}"
        )
    return wheels


def inspect_server_archive(path: Path, identity: Identity) -> ServerArchive:
    filenames = {
        server_archive_name(identity, platform): platform for platform in PLATFORMS
    }
    if path.name not in filenames:
        raise ValueError(f"Unexpected standalone server release: {path.name}")
    platform = filenames[path.name]
    binary_name = SERVER_BINARIES[platform]
    with ZipFile(path) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != binary_name:
            raise ValueError(
                f"Standalone server archive {path.name} must contain only {binary_name}"
            )
        member = members[0]
        if (
            member.date_time != (1980, 1, 1, 0, 0, 0)
            or member.create_system != 3
            or member.external_attr >> 16 != (stat.S_IFREG | 0o755)
            or member.compress_type != ZIP_DEFLATED
            or member.extra
            or member.comment
            or archive.comment
        ):
            raise ValueError(
                f"Standalone server archive {path.name} has noncanonical ZIP metadata"
            )
        if not archive.read(member):
            raise ValueError(f"Standalone server archive {path.name} is empty")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return ServerArchive(path, platform, digest)


def release_servers(directory: Path, identity: Identity) -> list[ServerArchive]:
    servers = [
        inspect_server_archive(path, identity)
        for path in sorted(directory.glob("loom-serve-*.zip"))
    ]
    actual = {server.platform for server in servers}
    missing = set(PLATFORMS) - actual
    if missing:
        raise ValueError(
            "Release requires one standalone server per platform; "
            f"missing={sorted(missing)}"
        )
    return servers


def release_artifacts(
    directory: Path, identity: Identity, description: str
) -> list[ReleaseArtifact]:
    return [
        *release_wheels(directory, identity, description),
        *release_servers(directory, identity),
    ]


def checksum_document(artifacts: list[ReleaseArtifact]) -> str:
    return "".join(
        f"{artifact.digest}  {artifact.path.name}\n"
        for artifact in sorted(artifacts, key=lambda artifact: artifact.path.name)
    )


def write_checksums(directory: Path, artifacts: list[ReleaseArtifact]) -> None:
    path = directory / "SHA256SUMS"
    document = checksum_document(artifacts)
    if path.exists():
        if path.read_bytes() != document.encode("utf-8"):
            raise ValueError("SHA256SUMS differs from the verified release artifacts")
    else:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(document)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--identity", type=Path, help="canonical version.json; required when editing"
    )
    parser.add_argument(
        "--description", type=Path, default=Path(__file__).with_name("PYPI_README.md")
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="edit the release identity for branch review"
    )
    prepare.add_argument("--version", required=True)
    prepare.add_argument("--catalog", required=True)
    prepare.add_argument("--dry-run", action="store_true")
    tag = commands.add_parser(
        "check-tag", help="check a release tag against version.json"
    )
    tag.add_argument("tag")
    verify = commands.add_parser(
        "verify", help="inspect one built release-platform wheel"
    )
    verify.add_argument("wheel", type=Path)
    bundle = commands.add_parser(
        "bundle", help="verify a complete release and write SHA256SUMS"
    )
    bundle.add_argument("directory", type=Path)
    bundle.add_argument("--tag", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare" and args.identity is None:
        parser.error("prepare requires --identity naming the source JSON to edit")
    if args.identity is None:
        args.identity = Path(__file__).with_name("version.json")
    try:
        identity = read_identity(args.identity)
        if args.command == "prepare":
            identity = parse_identity(
                {"package-version": args.version, "model-catalog": args.catalog}
            )
            identity.check_tag(identity.tag)
            document = json.dumps(identity.document(), indent=2) + "\n"
            print(document, end="")
            if not args.dry_run:
                args.identity.write_text(document, encoding="utf-8", newline="\n")
        elif args.command == "check-tag":
            print(identity.check_tag(args.tag).value)
        else:
            description = args.description.read_text(encoding="utf-8")
            if args.command == "verify":
                wheel = inspect_wheel(args.wheel, identity, description)
                print(f"{wheel.digest}  {wheel.path.name}")
            else:
                identity.check_tag(args.tag)
                artifacts = release_artifacts(args.directory, identity, description)
                write_checksums(args.directory, artifacts)
                print(checksum_document(artifacts), end="")
    except (ValueError, OSError, BadZipFile) as error:
        parser.exit(1, f"SDK release check failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
