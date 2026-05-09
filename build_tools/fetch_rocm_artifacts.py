#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch TheRock release artifacts into an HRX ROCm build root.

This is intentionally standalone instead of importing TheRock build_tools. It
uses public S3 buckets directly and flattens TheRock artifact archives into a
single ROCm-style prefix.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from botocore import UNSIGNED
from botocore.config import Config
import boto3

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from hrx_build_tools import REPO_ROOT, flatten_therock_artifact, log, sha256_file


PLATFORM = "linux"

# The default HRX core root is deliberately smaller than a full ROCm SDK. It
# contains ROCr/HSA, AMD SMI, bundled sysdeps, ROCm base files, and LLVM runtime
# tools/libs needed by the build. HRX provides libamdhip64.so in the final tree.
ARTIFACT_SETS = {
    "core": {
        "sysdeps": ["lib", "run", "dev"],
        "base": ["lib", "run", "dev"],
        "amd-llvm": ["lib", "run"],
        "core-runtime": ["lib", "run", "dev"],
        "core-amdsmi": ["lib", "run", "dev"],
        "aqlprofile": ["lib", "run", "dev"],
    },
    "core-with-llvm-dev": {
        "sysdeps": ["lib", "run", "dev"],
        "base": ["lib", "run", "dev"],
        "amd-llvm": ["lib", "run", "dev"],
        "core-runtime": ["lib", "run", "dev"],
        "core-amdsmi": ["lib", "run", "dev"],
        "aqlprofile": ["lib", "run", "dev"],
    },
    "core-with-upstream-hip": {
        "sysdeps": ["lib", "run", "dev"],
        "base": ["lib", "run", "dev"],
        "amd-llvm": ["lib", "run"],
        "core-runtime": ["lib", "run", "dev"],
        "core-amdsmi": ["lib", "run", "dev"],
        "aqlprofile": ["lib", "run", "dev"],
        "core-kpack": ["lib", "dev"],
        "core-hip": ["lib", "run", "dev"],
    },
}


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int
    last_modified: str


def create_s3_client():
    return boto3.client(
        "s3",
        region_name="us-east-2",
        config=Config(signature_version=UNSIGNED, max_pool_connections=64),
    )


def release_bucket(release_type: str, kind: str) -> str:
    if release_type not in {"dev", "nightly", "prerelease"}:
        raise ValueError("--release-type must be dev, nightly, or prerelease")
    if kind not in {"artifacts", "packages"}:
        raise ValueError(kind)
    return f"therock-{release_type}-{kind}"


def list_prefix(s3, bucket: str, prefix: str) -> list[S3Object]:
    paginator = s3.get_paginator("list_objects_v2")
    objects: list[S3Object] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append(
                S3Object(
                    key=obj["Key"],
                    size=obj["Size"],
                    last_modified=obj["LastModified"].isoformat(),
                )
            )
    return objects


def discover_latest_run_id(s3, release_type: str, artifact_set: str) -> str:
    """Discover the newest Linux artifacts prefix containing the requested set."""
    bucket = release_bucket(release_type, "artifacts")
    paginator = s3.get_paginator("list_objects_v2")
    candidates: list[int] = []
    for page in paginator.paginate(Bucket=bucket, Delimiter="/"):
        for common_prefix in page.get("CommonPrefixes", []):
            prefix = common_prefix["Prefix"]
            match = re.match(r"^(\d+)-linux/$", prefix)
            if match:
                candidates.append(int(match.group(1)))
    for run_id in sorted(candidates, reverse=True):
        prefix = f"{run_id}-{PLATFORM}/"
        available = list_prefix(s3, bucket, prefix)
        _, missing = select_available(available, prefix, wanted_artifacts(artifact_set))
        if not missing:
            return str(run_id)
    raise RuntimeError(
        f"Could not discover a {release_type} Linux run with artifact set "
        f"{artifact_set!r}. Pass --run-id explicitly."
    )


def wanted_artifacts(artifact_set: str) -> list[str]:
    try:
        mapping = ARTIFACT_SETS[artifact_set]
    except KeyError:
        raise ValueError(
            f"Unknown artifact set {artifact_set!r}. "
            f"Expected one of: {', '.join(sorted(ARTIFACT_SETS))}"
        )
    return [
        f"{name}_{component}_generic"
        for name, components in mapping.items()
        for component in components
    ]


def select_available(
    available: list[S3Object], prefix: str, wanted: list[str]
) -> tuple[list[S3Object], list[str]]:
    by_name: dict[str, S3Object] = {}
    for obj in available:
        filename = obj.key.removeprefix(prefix)
        if filename.endswith(".sha256sum"):
            continue
        if filename.endswith(".tar.zst"):
            base = filename.removesuffix(".tar.zst")
            by_name[base] = obj
        elif filename.endswith(".tar.xz"):
            base = filename.removesuffix(".tar.xz")
            by_name.setdefault(base, obj)
    selected = [by_name[name] for name in wanted if name in by_name]
    missing = [name for name in wanted if name not in by_name]
    return selected, missing


def download_one(s3, bucket: str, obj: S3Object, cache_dir: Path) -> Path:
    dest = cache_dir / Path(obj.key).name
    if dest.exists() and dest.stat().st_size == obj.size:
        log(f"  == Cached {dest.name}")
        return dest
    log(f"  ++ Downloading {obj.key}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    s3.download_file(bucket, obj.key, str(tmp))
    tmp.replace(dest)
    return dest


def download_checksum(s3, bucket: str, key: str, dest: Path) -> Path | None:
    checksum_key = f"{key}.sha256sum"
    checksum_dest = dest.with_name(dest.name + ".sha256sum")
    try:
        s3.download_file(bucket, checksum_key, str(checksum_dest))
    except Exception:
        return None
    return checksum_dest


def verify_checksum(archive_path: Path, checksum_path: Path | None) -> None:
    if checksum_path is None or not checksum_path.exists():
        log(f"  ?? No checksum for {archive_path.name}")
        return
    text = checksum_path.read_text().strip()
    if not text:
        log(f"  ?? Empty checksum for {archive_path.name}")
        return
    expected = text.split()[0]
    actual = sha256_file(archive_path)
    if actual != expected:
        raise RuntimeError(
            f"Checksum mismatch for {archive_path.name}: expected {expected}, got {actual}"
        )


def write_manifest(
    path: Path,
    *,
    release_type: str,
    run_id: str,
    bucket: str,
    artifact_set: str,
    artifacts: list[S3Object],
) -> None:
    data = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "release_type": release_type,
        "run_id": run_id,
        "platform": PLATFORM,
        "bucket": bucket,
        "artifact_set": artifact_set,
        "artifacts": [obj.__dict__ for obj in artifacts],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def run_fetch(args: argparse.Namespace) -> None:
    s3 = create_s3_client()
    run_id = args.run_id
    if args.latest:
        run_id = discover_latest_run_id(s3, args.release_type, args.set)
        log(f"Resolved latest {args.release_type} Linux run id: {run_id}")
    if not run_id:
        raise RuntimeError("Pass --run-id or --latest")

    bucket = release_bucket(args.release_type, "artifacts")
    prefix = f"{run_id}-{PLATFORM}/"
    available = list_prefix(s3, bucket, prefix)
    if not available:
        raise RuntimeError(f"No artifacts found at s3://{bucket}/{prefix}")

    wanted = wanted_artifacts(args.set)
    selected, missing = select_available(available, prefix, wanted)
    if missing:
        missing_text = "\n  ".join(missing)
        raise RuntimeError(f"Missing required artifacts:\n  {missing_text}")

    log("Artifacts selected:")
    for obj in selected:
        log(f"  {obj.key} ({obj.size / 1024 / 1024:.1f} MiB)")

    if args.dry_run:
        log("Dry run requested; not downloading.")
        return

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir: Path = args.download_cache_dir or (
        output_dir.parent / "rocm-artifact-downloads"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[tuple[S3Object, Path]] = []
    with ThreadPoolExecutor(max_workers=args.download_concurrency) as executor:
        futures = {
            executor.submit(download_one, s3, bucket, obj, cache_dir): obj
            for obj in selected
        }
        for future in as_completed(futures):
            obj = futures[future]
            downloaded.append((obj, future.result()))

    for obj, archive_path in sorted(downloaded, key=lambda item: item[1].name):
        checksum = download_checksum(s3, bucket, obj.key, archive_path)
        verify_checksum(archive_path, checksum)
        log(f"  ++ Flattening {archive_path.name}")
        flatten_therock_artifact(archive_path, output_dir)

    write_manifest(
        args.manifest or (output_dir / ".hrx-rocm-artifacts.json"),
        release_type=args.release_type,
        run_id=run_id,
        bucket=bucket,
        artifact_set=args.set,
        artifacts=selected,
    )
    log(f"ROCm build root ready: {output_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-type", default="nightly", choices=["dev", "nightly", "prerelease"]
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id")
    source.add_argument("--latest", action="store_true")
    parser.add_argument("--set", default="core", choices=sorted(ARTIFACT_SETS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT.parent / "build" / "rocm-root",
    )
    parser.add_argument("--download-cache-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--download-concurrency", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    run_fetch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
