#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Publishes verified SDK assets and prepares immutable PyPI upload retries."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

from build_tools.ci import github_api
from loom.packaging.python import release

GITHUB_RELEASE_BODY_MAXIMUM = 125_000
GITHUB_ASSET_PAGE_SIZE = 100
GITHUB_PULL_REQUEST_PAGE_SIZE = 100
_CATALOG_TAG = re.compile(r"github:([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@([^\s]+)")
_PULL_REQUEST_RELEASE_NOTES = re.compile(
    r"^<!-- loom-release-notes:begin version=([^\s]+) -->\n"
    r"(.*?)\n"
    r"<!-- loom-release-notes:end -->$",
    re.MULTILINE | re.DOTALL,
)
_RELEASE_NOTES_PLACEHOLDER = re.compile(r"{{[A-Z][A-Z0-9_]*}}")
_RELEASE_SOURCE = re.compile(
    r"\A<!-- loom-release-source: pr=([1-9][0-9]*) "
    r"merge=([0-9a-f]{40}) version=([^\s]+) "
    r"editorial-sha256=([0-9a-f]{64}) -->\n\n"
)
_RELEASE_APPENDIX_SEPARATOR = "\n\n---\n\n"


@dataclass(frozen=True)
class ReleaseEditorial:
    pull_request: int
    revision: str
    text: str


def _release_editorial_from_pull_request(
    repository: str,
    revision: str,
    identity: release.Identity,
    token: str,
) -> ReleaseEditorial:
    """Reads reviewed notes from the PR merged as the release revision."""
    api = f"https://api.github.com/repos/{repository}"
    records = github_api.get_json(
        f"{api}/commits/{revision}/pulls?per_page={GITHUB_PULL_REQUEST_PAGE_SIZE}",
        token,
    )
    if not isinstance(records, list):
        raise ValueError("GitHub returned an invalid associated pull request list")
    if len(records) == GITHUB_PULL_REQUEST_PAGE_SIZE:
        raise ValueError("SDK release revision has too many associated pull requests")
    merged = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("state") == "closed"
        and isinstance(record.get("merged_at"), str)
        and record.get("merge_commit_sha") == revision
    ]
    if len(merged) != 1:
        raise ValueError(
            "SDK release revision must be the merge commit of one pull request"
        )
    record = merged[0]
    pull_request = record.get("number")
    body = record.get("body")
    if type(pull_request) is not int or pull_request <= 0 or not isinstance(body, str):
        raise ValueError("GitHub returned an invalid release pull request")
    matches = list(_PULL_REQUEST_RELEASE_NOTES.finditer(body))
    if len(matches) != 1:
        raise ValueError(
            "Release pull request must contain one marked release notes section"
        )
    version, editorial = matches[0].groups()
    if version != identity.version:
        raise ValueError(
            f"Release notes version {version!r} does not match {identity.version!r}"
        )
    editorial = editorial.strip()
    if not editorial:
        raise ValueError("Release pull request notes are empty")
    placeholder = _RELEASE_NOTES_PLACEHOLDER.search(editorial)
    if placeholder is not None:
        raise ValueError(
            f"Release pull request notes contain placeholder {placeholder.group()}"
        )
    return ReleaseEditorial(pull_request, revision, editorial)


def _github_release_appendix(identity: release.Identity, repository: str) -> str:
    """Builds mechanically derived installation and qualification guidance."""
    server_archives = "".join(
        f"- `{release.standalone_archive_name(identity, 'loom-serve', platform)}`\n"
        for platform in release.PLATFORMS
    )
    info_archives = "".join(
        f"- `{release.standalone_archive_name(identity, 'amdf-info', platform)}`\n"
        for platform in release.PLATFORMS
    )
    linux_archive = release.standalone_archive_name(
        identity, "loom-serve", release.LINUX_X86_64_PLATFORM
    )
    if identity.release_channel is release.ReleaseChannel.STABLE:
        installation = (
            "Install the optimized profile from PyPI:\n\n"
            "```console\n"
            f"python -m pip install hrx-loom=={identity.version}\n"
            "```\n"
        )
    else:
        wheels = {
            platform: f"hrx_loom-{identity.version}-py3-none-{platform}.whl"
            for platform in release.PLATFORMS
        }
        installation = (
            "This public release candidate is not published to PyPI. Install the "
            "attached Release wheel directly:\n\n"
            "```console\n"
            "# Linux x86-64\n"
            f"python -m pip install https://github.com/{repository}/releases/download/"
            f"{identity.tag}/{wheels[release.LINUX_X86_64_PLATFORM]}\n"
            "\n"
            "# Windows x86-64\n"
            f"python -m pip install https://github.com/{repository}/releases/download/"
            f"{identity.tag}/{wheels[release.WINDOWS_X86_64_PLATFORM]}\n"
            "```\n"
        )
    return (
        "## Install and artifacts\n\n"
        f"Loom SDK {identity.version} contains the compiler, runtime, command-line "
        "tools, and Python APIs for Linux x86-64 and Windows x86-64.\n\n"
        f"{installation}\n"
        "For use outside Python, download the standalone archive for your platform:\n\n"
        f"{server_archives}\n"
        "Each archive contains one `loom-serve` executable with the SDK target "
        "providers built in:\n\n"
        "```console\n"
        f"unzip {linux_archive}\n"
        "./loom-serve --version\n"
        "```\n\n"
        "For native AMD GPU/NPU discovery and activation checks without installing "
        "the SDK, use the standalone `amdf-info` archive:\n\n"
        f"{info_archives}\n"
        "Each contains one executable with libamdf embedded and requires neither "
        "ROCr nor XRT. `amdf-info --format=json` reports the native inventory; "
        "`amdf-info --device=rdna --activate` checks device acquisition and release."
        "\n\n## Qualification\n\n"
        "The attached wheels are the exact Release, Tracy, Debug, and ASAN "
        "profile artifacts qualified on Linux x86-64 and Windows x86-64. "
        "`SHA256SUMS` records the qualified wheels and standalone archives."
    )


def github_release_body(
    identity: release.Identity, repository: str, editorial: ReleaseEditorial
) -> str:
    """Composes reviewed notes with the generated release artifact appendix."""
    text = editorial.text.strip()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    body = (
        f"<!-- loom-release-source: pr={editorial.pull_request} "
        f"merge={editorial.revision} version={identity.version} "
        f"editorial-sha256={digest} -->\n\n"
        f"{text}{_RELEASE_APPENDIX_SEPARATOR}"
        f"{_github_release_appendix(identity, repository)}"
    )
    if len(body) > GITHUB_RELEASE_BODY_MAXIMUM:
        raise ValueError(
            "GitHub release body exceeds the 125000-character service limit"
        )
    return body


def _validate_published_release_body(
    identity: release.Identity, repository: str, revision: str, body: str
) -> None:
    """Validates a self-contained immutable release body without its source PR."""
    match = _RELEASE_SOURCE.match(body)
    if match is None:
        raise ValueError("Published GitHub release has invalid source provenance")
    _, source_revision, version, digest = match.groups()
    if source_revision != revision or version != identity.version:
        raise ValueError("Published GitHub release has conflicting source provenance")
    appendix = _RELEASE_APPENDIX_SEPARATOR + _github_release_appendix(
        identity, repository
    )
    if not body.endswith(appendix):
        raise ValueError("Published GitHub release has a conflicting artifact appendix")
    editorial = body[match.end() : -len(appendix)]
    if not editorial or hashlib.sha256(editorial.encode("utf-8")).hexdigest() != digest:
        raise ValueError("Published GitHub release has conflicting editorial notes")


def verified_release(
    directory: Path, identity: release.Identity, description: str, tag: str
) -> list[release.ReleaseArtifact]:
    identity.check_tag(tag)
    artifacts = release.release_artifacts(directory, identity, description)
    expected = release.checksum_document(artifacts).encode("utf-8")
    if (directory / "SHA256SUMS").read_bytes() != expected:
        raise ValueError("SHA256SUMS differs from the verified release artifacts")
    return artifacts


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _gh(*arguments: str) -> None:
    subprocess.run(["gh", *arguments], check=True)


def _require_catalog_release(
    repository: str,
    revision: str,
    identity: release.Identity,
    artifacts: list[release.ReleaseArtifact],
    token: str,
) -> None:
    """Requires an immutable catalog qualified with these exact SDK bytes."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("SDK release revision must be a full Git commit")
    match = (
        _CATALOG_TAG.fullmatch(identity.catalog)
        if isinstance(identity.catalog, str)
        else None
    )
    if match is None or match.group(2) != identity.tag:
        raise ValueError(
            "SDK release catalog must use its matching versioned GitHub tag"
        )
    catalog_repository, tag = match.groups()
    api = f"https://api.github.com/repos/{catalog_repository}"
    document = github_api.get_json(
        f"{api}/releases/tags/" + urllib.parse.quote(tag, safe=""),
        token,
    )
    prerelease = identity.release_channel is release.ReleaseChannel.PRERELEASE
    if (
        not isinstance(document, dict)
        or document.get("tag_name") != tag
        or document.get("draft") is not False
        or document.get("prerelease") is not prerelease
        or document.get("immutable") is not True
    ):
        raise ValueError(
            "SDK release catalog tag is not a matching immutable public release"
        )

    reference = github_api.get_json(
        f"{api}/git/ref/tags/" + urllib.parse.quote(tag, safe=""), token
    )
    if (
        not isinstance(reference, dict)
        or not isinstance(reference.get("object"), dict)
        or reference["object"].get("type") != "commit"
        or not isinstance(reference["object"].get("sha"), str)
        or re.fullmatch(r"[0-9a-f]{40}", reference["object"]["sha"]) is None
    ):
        raise ValueError("SDK release catalog tag has an invalid Git reference")
    catalog_revision = reference["object"]["sha"]
    pin = json.loads(
        github_api.get_bytes(
            f"{api}/contents/build_tools/ci/sdk.json?ref={catalog_revision}",
            token,
            accept="application/vnd.github.raw+json",
        )
    )
    expected_pin_keys = {
        "repository",
        "revision",
        "run_id",
        "artifact_id",
        "release_tag",
        "wheel",
        "sha256",
        "catalog",
    }
    wheels = [
        artifact
        for artifact in artifacts
        if isinstance(artifact, release.Wheel)
        and artifact.profile == "release"
        and artifact.platform == release.LINUX_X86_64_PLATFORM
    ]
    if len(wheels) != 1:
        raise ValueError("SDK release has no unique Linux Release wheel")
    wheel = wheels[0]
    if (
        not isinstance(pin, dict)
        or set(pin) != expected_pin_keys
        or pin.get("repository") != repository
        or pin.get("revision") != revision
        or type(pin.get("run_id")) is not int
        or pin["run_id"] <= 0
        or type(pin.get("artifact_id")) is not int
        or pin["artifact_id"] <= 0
        or pin.get("release_tag") != identity.tag
        or pin.get("wheel") != wheel.path.name
        or pin.get("sha256") != wheel.digest
        or pin.get("catalog") != identity.catalog
    ):
        raise ValueError(
            "immutable catalog was not qualified with these exact SDK bytes"
        )


def _release_state(
    document: object,
    *,
    tag: str,
    title: str,
    body: str | None,
    prerelease: bool,
) -> tuple[int, bool, str]:
    expected = {
        "tag_name": tag,
        "name": title,
        "prerelease": prerelease,
    }
    if not isinstance(document, dict) or any(
        document.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Existing GitHub release has conflicting metadata")
    release_id = document.get("id")
    draft = document.get("draft")
    actual_body = document.get("body")
    if (
        type(release_id) is not int
        or release_id <= 0
        or type(draft) is not bool
        or not isinstance(actual_body, str)
    ):
        raise ValueError("GitHub returned an invalid release state")
    if body is not None and actual_body != body:
        raise ValueError("Existing GitHub release has conflicting metadata")
    if not draft and document.get("immutable") is not True:
        raise ValueError(
            "Published GitHub release is mutable; use a new version after enabling "
            "repository release immutability"
        )
    return release_id, draft, actual_body


def _missing_release_assets(
    api: str,
    release_id: int,
    token: str,
    assets: dict[str, tuple[Path, str]],
) -> list[str]:
    names = set()
    page = 1
    while True:
        records = github_api.get_json(
            f"{api}/releases/{release_id}/assets?"
            f"per_page={GITHUB_ASSET_PAGE_SIZE}&page={page}",
            token,
        )
        if not isinstance(records, list):
            raise ValueError("GitHub returned an invalid release asset inventory")
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("name"), str):
                raise ValueError("GitHub returned an invalid release asset record")
            name = record["name"]
            if name in names:
                raise ValueError(f"Duplicate GitHub release asset: {name}")
            names.add(name)
            if name not in assets:
                raise ValueError(
                    f"Unexpected GitHub release artifact: {name}; use a new version"
                )
            if record.get("state") != "uploaded":
                raise ValueError(
                    f"Incomplete GitHub asset: {name}; review the failed upload"
                )
            if record.get("digest") != "sha256:" + assets[name][1]:
                raise ValueError(
                    f"Existing GitHub asset differs: {name}; use a new version"
                )
        if len(records) < GITHUB_ASSET_PAGE_SIZE:
            break
        page += 1
    return [str(path) for name, (path, _) in assets.items() if name not in names]


def publish_github(
    repository: str,
    revision: str,
    tag: str,
    identity: release.Identity,
    artifacts: list[release.ReleaseArtifact],
    checksums: Path,
) -> None:
    """Completes a verified draft and publishes its immutable asset set."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) or any(
        part in (".", "..") for part in repository.split("/")
    ):
        raise ValueError("GitHub repository must be OWNER/REPOSITORY")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("SDK release revision must be a full Git commit")
    prerelease = identity.release_channel is release.ReleaseChannel.PRERELEASE
    title = f"Loom {tag}"
    api = f"https://api.github.com/repos/{repository}"
    token = os.environ.get("GH_TOKEN", "")
    try:
        existing = github_api.get_json(
            f"{api}/releases/tags/{urllib.parse.quote(tag, safe='')}", token
        )
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        existing = None

    assets = {
        artifact.path.name: (artifact.path, artifact.digest) for artifact in artifacts
    }
    assets[checksums.name] = (checksums, _digest(checksums))
    _require_catalog_release(repository, revision, identity, artifacts, token)

    if existing is not None:
        release_id, draft, existing_body = _release_state(
            existing, tag=tag, title=title, body=None, prerelease=prerelease
        )
        if not draft:
            _validate_published_release_body(
                identity, repository, revision, existing_body
            )
            if _missing_release_assets(api, release_id, token, assets):
                raise ValueError(
                    "Published GitHub release has an incomplete asset set; "
                    "use a new version"
                )
            print("GitHub release already contains identical immutable assets.")
            return

    editorial = _release_editorial_from_pull_request(
        repository, revision, identity, token
    )
    body = github_release_body(identity, repository, editorial)
    created = existing is None
    if existing is None:
        options = ["--prerelease", "--latest=false"] if prerelease else []
        _gh(
            "release",
            "create",
            tag,
            *[str(path) for path, _ in assets.values()],
            "--repo",
            repository,
            "--verify-tag",
            "--draft",
            "--title",
            title,
            "--notes",
            body,
            *options,
        )
        existing = github_api.get_json(
            f"{api}/releases/tags/{urllib.parse.quote(tag, safe='')}", token
        )
    release_id, draft, _ = _release_state(
        existing, tag=tag, title=title, body=body, prerelease=prerelease
    )
    if created and not draft:
        raise ValueError("New GitHub release did not remain a draft")
    missing = _missing_release_assets(api, release_id, token, assets)
    if missing:
        _gh("release", "upload", tag, *missing, "--repo", repository)
        if _missing_release_assets(api, release_id, token, assets):
            raise ValueError("GitHub release asset upload did not complete")
    _gh("release", "edit", tag, "--repo", repository, "--draft=false")
    published = github_api.get_json(
        f"{api}/releases/tags/{urllib.parse.quote(tag, safe='')}", token
    )
    published_id, published_draft, _ = _release_state(
        published, tag=tag, title=title, body=body, prerelease=prerelease
    )
    if published_id != release_id or published_draft:
        raise ValueError("GitHub release did not reach its immutable public state")


def stage_pypi(
    wheels: list[release.Wheel], identity: release.Identity, output: Path
) -> bool:
    """Copies only missing normal wheels into a fresh upload directory."""
    if identity.release_channel is not release.ReleaseChannel.STABLE:
        raise ValueError("PyPI publication requires a stable X.Y.Z version")
    normal = {wheel.path.name: wheel for wheel in wheels if wheel.profile == "release"}
    request = urllib.request.Request(
        f"https://pypi.org/pypi/hrx-loom/{identity.version}/json",
        headers={"Accept": "application/json", "User-Agent": "hrx-loom-release"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            records = json.load(response)["urls"]
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        records = []
    existing = set()
    for record in records:
        name = record["filename"]
        if name not in normal or name in existing:
            raise ValueError(
                f"Unexpected or duplicate PyPI asset: {name}; use a new version"
            )
        if record["yanked"] or record["digests"]["sha256"] != normal[name].digest:
            raise ValueError(
                f"Existing PyPI asset differs or is yanked: {name}; use a new version"
            )
        existing.add(name)

    missing = [wheel for name, wheel in normal.items() if name not in existing]
    output.mkdir(parents=True, exist_ok=False)
    for wheel in missing:
        shutil.copyfile(wheel.path, output / wheel.path.name)
    return bool(missing)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("verify", "github", "pypi-stage"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--directory", type=Path, default=Path("release-assets"))
    parser.add_argument(
        "--identity", type=Path, default=Path("loom/packaging/python/version.json")
    )
    parser.add_argument(
        "--description", type=Path, default=Path("loom/packaging/python/PYPI_README.md")
    )
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY"))
    parser.add_argument("--revision", default=os.environ.get("GITHUB_SHA"))
    parser.add_argument("--output", type=Path, help="fresh PyPI upload directory")
    args = parser.parse_args(argv)
    if args.operation == "github" and (not args.repository or not args.revision):
        parser.error(
            "github requires repository and revision from GitHub Actions or options"
        )
    if args.operation == "pypi-stage" and args.output is None:
        parser.error("pypi-stage requires --output")
    try:
        identity = release.read_identity(args.identity)
        artifacts = verified_release(
            args.directory,
            identity,
            args.description.read_text(encoding="utf-8"),
            args.tag,
        )
        wheels = [
            artifact for artifact in artifacts if isinstance(artifact, release.Wheel)
        ]
        if args.operation == "github":
            publish_github(
                args.repository,
                args.revision,
                args.tag,
                identity,
                artifacts,
                args.directory / "SHA256SUMS",
            )
        elif args.operation == "pypi-stage":
            pending = stage_pypi(wheels, identity, args.output)
            result = f"pending={str(pending).lower()}\n"
            print(result, end="")
            if os.environ.get("GITHUB_OUTPUT"):
                with Path(os.environ["GITHUB_OUTPUT"]).open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(result)
        else:
            print(release.checksum_document(artifacts), end="")
    except (ValueError, OSError, BadZipFile, subprocess.CalledProcessError) as error:
        parser.exit(1, f"SDK publication failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
