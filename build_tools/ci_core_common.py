#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared CI helpers for core HRX CMake build/package flows."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent

# Minimal TheRock artifact closure used by HRX CI. Platform selection is handled
# by the TheRock run prefix, not by changing component names here.
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
        "amd-llvm": ["lib", "run", "dev"],
        "core-runtime": ["lib", "run", "dev"],
        "core-amdsmi": ["lib", "run", "dev"],
        "aqlprofile": ["lib", "run", "dev"],
        "core-kpack": ["lib", "dev"],
        "core-hip": ["lib", "run", "dev"],
    },
}

ROCM_ARTIFACT_VARIANTS = ("release", "asan", "host-asan", "tsan")
ROCM_ARTIFACT_VARIANT_LOG_KEY = "logs/compiler-runtime/ROCR-Runtime_configure.log"
CORE_CTEST_EXCLUDE_REGEXES = ()


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int
    last_modified: str


def log(message: str = "") -> None:
    print(message, flush=True)


def env_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in ("1", "true", "yes", "on")


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, os.fspath(default)))


def default_ctest_parallelism() -> int:
    cpu_count = os.cpu_count() or 2
    return max(1, (cpu_count + 1) // 2)


def run(
    args: Iterable[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stderr_to_stdout: bool = False,
    pretty_command: bool = False,
    line_continuation: str = "\\",
) -> None:
    cmd = [os.fspath(arg) for arg in args]
    if pretty_command:
        formatted_cmd = f" {line_continuation}\n  ".join(
            shlex.join([arg]) for arg in cmd
        )
        log(f"++ Exec [{cwd or Path.cwd()}]$")
        log(f"  {formatted_cmd}")
    else:
        log(f"++ Exec [{cwd or Path.cwd()}]$ {shlex.join(cmd)}")
    stderr = subprocess.STDOUT if stderr_to_stdout else None
    subprocess.run(cmd, cwd=cwd, env=env, check=True, stderr=stderr)


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")


def remove_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _repo_source_for_installed_build_testdata(path: Path) -> Path | None:
    # Installed tests copy CMake build-tree mirrors under testdata/build. Those
    # mirrors use symlinks back to source files, so their repo-relative suffix
    # is a valid source fallback when Windows cannot resolve the link directly.
    parts = path.parts
    lower_parts = [part.lower() for part in parts]
    for i in range(len(parts) - 1):
        if lower_parts[i] == "testdata" and lower_parts[i + 1] == "build":
            rel_parts = parts[i + 2 :]
            if rel_parts:
                return REPO_ROOT.joinpath(*rel_parts)
    return None


def _existing_materialized_path(path: Path) -> Path | None:
    try:
        if path.is_symlink():
            return path.resolve(strict=True)
        if path.exists():
            return path
    except OSError:
        return None
    return None


def materialized_source_path(path: Path) -> Path:
    """Returns the file or directory to copy when symlinks should be flattened."""
    if not path.is_symlink():
        return path
    try:
        return path.resolve(strict=True)
    except OSError as resolve_error:
        link_target = os.readlink(path)
        target_path = Path(link_target)
        candidates = [
            target_path if target_path.is_absolute() else path.parent / target_path
        ]
        if repo_source := _repo_source_for_installed_build_testdata(path):
            candidates.append(repo_source)

        for candidate in candidates:
            if materialized := _existing_materialized_path(candidate):
                return materialized

        raise OSError(
            f"Could not materialize symlink {path} -> {link_target}"
        ) from resolve_error


def copy_tree_contents(
    src: Path,
    dst: Path,
    *,
    skip_names: set[str] | None = None,
    preserve_symlinks: bool = True,
) -> None:
    skip_names = skip_names or set()
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if child.name in skip_names:
            continue
        target = dst / child.name
        if child.is_symlink() and preserve_symlinks:
            if target.exists() or target.is_symlink():
                remove_tree(target)
            target.symlink_to(os.readlink(child))
        else:
            source = materialized_source_path(child)
            if source.is_dir():
                if target.exists() or target.is_symlink():
                    if target.is_dir() and not target.is_symlink():
                        copy_tree_contents(
                            source,
                            target,
                            preserve_symlinks=preserve_symlinks,
                        )
                    else:
                        remove_tree(target)
                        target.mkdir(parents=True)
                        copy_tree_contents(
                            source,
                            target,
                            preserve_symlinks=preserve_symlinks,
                        )
                else:
                    target.mkdir(parents=True)
                    copy_tree_contents(
                        source,
                        target,
                        preserve_symlinks=preserve_symlinks,
                    )
            else:
                if target.exists() or target.is_symlink():
                    remove_tree(target)
                shutil.copy2(source, target)


def copy_path(src: Path, dst: Path, *, preserve_symlinks: bool = True) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        remove_tree(dst)
    if src.is_symlink() and preserve_symlinks:
        dst.symlink_to(os.readlink(src))
    else:
        source = materialized_source_path(src)
        if source.is_dir():
            dst.mkdir(parents=True)
            copy_tree_contents(source, dst, preserve_symlinks=preserve_symlinks)
        else:
            shutil.copy2(source, dst)


def copy_relative_path(
    src_root: Path,
    dst_root: Path,
    relpath: Path,
    *,
    preserve_symlinks: bool = True,
) -> None:
    copy_path(
        src_root / relpath,
        dst_root / relpath,
        preserve_symlinks=preserve_symlinks,
    )


def path_relative_to(path: Path, root: Path) -> Path | None:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return None


def rocm_llvm_bin(rocm_root: Path) -> Path:
    return rocm_root / "lib" / "llvm" / "bin"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_tar_archive(path: Path) -> tarfile.TarFile:
    if path.name.endswith(".tar.zst"):
        try:
            import zstandard
        except ModuleNotFoundError as e:
            raise RuntimeError("Install the zstandard Python package") from e
        backing_file = path.open("rb")
        stream = zstandard.ZstdDecompressor().stream_reader(backing_file)
        try:
            tf = tarfile.open(fileobj=stream, mode="r|")
        except Exception:
            stream.close()
            backing_file.close()
            raise
        tf._hrx_owned_streams = (stream, backing_file)  # type: ignore[attr-defined]
        return tf
    if path.name.endswith(".tar.xz"):
        return tarfile.open(path, mode="r:xz")
    if path.name.endswith(".tar.gz") or path.name.endswith(".tgz"):
        return tarfile.open(path, mode="r:gz")
    raise ValueError(f"Unsupported archive extension: {path}")


def close_tar_archive(tf: tarfile.TarFile) -> None:
    owned = getattr(tf, "_hrx_owned_streams", ())
    tf.close()
    for stream in owned:
        stream.close()


def checked_dest(base: Path, relpath: str) -> Path:
    rel = PurePosixPath(relpath.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError(f"Unsafe archive path: {relpath}")
    dest = base.joinpath(*rel.parts)
    base_resolved = base.resolve()
    parent_resolved = dest.parent.resolve()
    if (
        base_resolved != parent_resolved
        and base_resolved not in parent_resolved.parents
    ):
        raise RuntimeError(f"Archive path escapes output directory: {relpath}")
    return dest


def strip_manifest_root(member_name: str, relroots: list[str]) -> str:
    for root in relroots:
        prefix = root.rstrip("/") + "/"
        if member_name.startswith(prefix):
            scoped = member_name[len(prefix) :]
            if scoped:
                return scoped
    raise RuntimeError(f"Archive member is outside manifest roots: {member_name}")


def flatten_therock_artifact(
    archive_path: Path,
    output_dir: Path,
    *,
    materialize_links: bool = False,
    preserve_symlinks: bool = True,
) -> set[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tf = open_tar_archive(archive_path)
    hardlinks: list[tuple[Path, str, list[str]]] = []
    try:
        manifest = tf.next()
        if manifest is None or manifest.name != "artifact_manifest.txt":
            raise RuntimeError(
                f"{archive_path.name} is not a TheRock artifact archive "
                "(artifact_manifest.txt was not the first member)"
            )
        manifest_file = tf.extractfile(manifest)
        if manifest_file is None:
            raise RuntimeError(f"Could not read manifest in {archive_path.name}")
        relroots = [line for line in manifest_file.read().decode().splitlines() if line]

        while member := tf.next():
            scoped_name = strip_manifest_root(member.name, relroots)
            dest_path = checked_dest(output_dir, scoped_name)
            if member.isdir():
                dest_path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if dest_path.exists() or dest_path.is_symlink():
                    dest_path.unlink()
                source = tf.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Could not read {member.name}")
                with source, dest_path.open("wb") as out:
                    shutil.copyfileobj(source, out)
                mode = 0o666 | (member.mode & 0o111)
                os.chmod(dest_path, mode)
            elif member.issym():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if dest_path.exists() or dest_path.is_symlink():
                    dest_path.unlink()
                if materialize_links:
                    target_name = strip_manifest_root(member.linkname, relroots)
                    target_path = checked_dest(output_dir, target_name)
                    if not target_path.exists():
                        raise RuntimeError(
                            "Cannot materialize symlink before target: "
                            f"{member.name} -> {member.linkname}"
                        )
                    copy_path(
                        target_path,
                        dest_path,
                        preserve_symlinks=preserve_symlinks,
                    )
                else:
                    dest_path.symlink_to(member.linkname)
            elif member.islnk():
                hardlinks.append((dest_path, member.linkname, relroots))
            else:
                raise RuntimeError(f"Unhandled tar member type: {member.name}")

        for dest_path, linkname, link_relroots in hardlinks:
            target_name = strip_manifest_root(linkname, link_relroots)
            target_path = checked_dest(output_dir, target_name)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if dest_path.exists() or dest_path.is_symlink():
                dest_path.unlink()
            if materialize_links:
                copy_path(
                    target_path,
                    dest_path,
                    preserve_symlinks=preserve_symlinks,
                )
            else:
                os.link(target_path, dest_path)
        return set(relroots)
    finally:
        close_tar_archive(tf)


def extract_tar_package_archive(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tf = open_tar_archive(archive_path)
    hardlinks: list[tuple[Path, str]] = []
    try:
        while member := tf.next():
            dest_path = checked_dest(output_dir, member.name)
            if member.isdir():
                dest_path.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if dest_path.exists() or dest_path.is_symlink():
                    dest_path.unlink()
                source = tf.extractfile(member)
                if source is None:
                    raise RuntimeError(f"Could not read {member.name}")
                with source, dest_path.open("wb") as out:
                    shutil.copyfileobj(source, out)
                mode = 0o666 | (member.mode & 0o111)
                os.chmod(dest_path, mode)
            elif member.issym():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                if dest_path.exists() or dest_path.is_symlink():
                    dest_path.unlink()
                dest_path.symlink_to(member.linkname)
            elif member.islnk():
                hardlinks.append((dest_path, member.linkname))
            else:
                raise RuntimeError(f"Unhandled tar member type: {member.name}")

        for dest_path, linkname in hardlinks:
            target_path = checked_dest(output_dir, linkname)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if dest_path.exists() or dest_path.is_symlink():
                dest_path.unlink()
            os.link(target_path, dest_path)
    finally:
        close_tar_archive(tf)


def find_downloaded_package(
    artifact_dir: Path,
    package_name: str,
    *,
    extension: str,
) -> Path:
    candidates = sorted(artifact_dir.glob(f"**/{package_name}-*{extension}"))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find {package_name}-*{extension} under {artifact_dir}"
        )
    if len(candidates) > 1:
        log(f"  ?? Multiple {package_name} archives found; using {candidates[-1]}")
    return candidates[-1]


def extract_packages(
    args: argparse.Namespace,
    *,
    package_roots: dict[str, Path],
    extension: str,
    extract_archive: Callable[[Path, Path], None],
) -> None:
    artifact_dir = args.artifact_download_dir.resolve()
    require_path(artifact_dir, "downloaded artifact directory")
    for package_name, output_dir in package_roots.items():
        archive_path = find_downloaded_package(
            artifact_dir, package_name, extension=extension
        )
        if output_dir.exists():
            remove_tree(output_dir)
        output_dir.mkdir(parents=True)
        log(f"  ++ Extracting {archive_path} -> {output_dir}")
        extract_archive(archive_path, output_dir)


def create_s3_client():
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ModuleNotFoundError as e:
        raise RuntimeError("Install boto3 and botocore to fetch ROCm artifacts") from e
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


def wanted_artifacts(
    artifact_set: str,
    *,
    artifact_sets: dict[str, dict[str, list[str]]] = ARTIFACT_SETS,
) -> list[str]:
    try:
        mapping = artifact_sets[artifact_set]
    except KeyError as e:
        raise ValueError(
            f"Unknown artifact set {artifact_set!r}; expected one of "
            f"{', '.join(sorted(artifact_sets))}"
        ) from e
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
            by_name[filename.removesuffix(".tar.zst")] = obj
        elif filename.endswith(".tar.xz"):
            by_name.setdefault(filename.removesuffix(".tar.xz"), obj)
    selected = [by_name[name] for name in wanted if name in by_name]
    missing = [name for name in wanted if name not in by_name]
    return selected, missing


def read_s3_text(s3, bucket: str, key: str) -> str:
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response.get("Body")
    if body is None:
        raise RuntimeError(f"S3 object has no body: s3://{bucket}/{key}")
    return body.read().decode(errors="replace")


def rocm_artifact_variant_from_configure_log(text: str) -> str:
    if "Override ASAN GPU_TARGETS" in text or "SANITIZER = ASAN" in text:
        return "asan"
    if "Override TSAN GPU_TARGETS" in text or "SANITIZER = TSAN" in text:
        return "tsan"
    if "HOST_ASAN enabled" in text or "SANITIZER = HOST_ASAN" in text:
        return "host-asan"
    return "release"


def rocm_artifact_variant(
    s3, bucket: str, prefix: str, available: list[S3Object]
) -> str:
    key = prefix + ROCM_ARTIFACT_VARIANT_LOG_KEY
    if not any(obj.key == key for obj in available):
        return "release"
    return rocm_artifact_variant_from_configure_log(read_s3_text(s3, bucket, key))


def validate_rocm_artifact_variant(
    s3,
    bucket: str,
    prefix: str,
    available: list[S3Object],
    expected_variant: str,
) -> None:
    actual_variant = rocm_artifact_variant(s3, bucket, prefix, available)
    if actual_variant == expected_variant:
        return
    raise RuntimeError(
        f"ROCm artifact prefix s3://{bucket}/{prefix} has variant "
        f"{actual_variant!r}, but {expected_variant!r} was requested. Set "
        "HRX_ROCM_ARTIFACT_VARIANT or choose a matching --run-id."
    )


def discover_latest_run_id(
    s3,
    release_type: str,
    artifact_set: str,
    artifact_variant: str,
    *,
    platform_name: str,
    platform_display: str,
    artifact_sets: dict[str, dict[str, list[str]]] = ARTIFACT_SETS,
) -> str:
    bucket = release_bucket(release_type, "artifacts")
    paginator = s3.get_paginator("list_objects_v2")
    candidates: list[int] = []
    platform_prefix = re.escape(platform_name)
    for page in paginator.paginate(Bucket=bucket, Delimiter="/"):
        for common_prefix in page.get("CommonPrefixes", []):
            match = re.match(rf"^(\d+)-{platform_prefix}/$", common_prefix["Prefix"])
            if match:
                candidates.append(int(match.group(1)))
    for run_id in sorted(candidates, reverse=True):
        prefix = f"{run_id}-{platform_name}/"
        available = list_prefix(s3, bucket, prefix)
        _, missing = select_available(
            available,
            prefix,
            wanted_artifacts(artifact_set, artifact_sets=artifact_sets),
        )
        if (
            not missing
            and rocm_artifact_variant(s3, bucket, prefix, available) == artifact_variant
        ):
            return str(run_id)
    raise RuntimeError(
        f"Could not discover a {release_type} {platform_display} "
        f"{artifact_variant} run with artifact set {artifact_set!r}. Pass --run-id "
        "explicitly."
    )


def download_one(s3, bucket: str, obj: S3Object, cache_dir: Path) -> Path:
    dest = s3_cache_path(cache_dir, bucket, obj.key)
    if dest.exists() and dest.stat().st_size == obj.size:
        log(f"  == Cached {obj.key}")
        return dest
    log(f"  ++ Downloading {obj.key}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.tmp")
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


def s3_cache_path(cache_dir: Path, bucket: str, key: str) -> Path:
    relpath = PurePosixPath(key)
    if relpath.is_absolute() or ".." in relpath.parts:
        raise RuntimeError(f"Unsafe S3 key: {key}")
    return cache_dir / bucket / relpath


def rocm_artifact_identity(
    *,
    release_type: str,
    run_id: str,
    platform_name: str,
    artifact_variant: str,
    artifact_set: str,
) -> str:
    """Returns the immutable filesystem identity of a fetched ROCm root."""
    components = (
        release_type,
        run_id,
        platform_name,
        artifact_variant,
        artifact_set,
    )
    for component in components:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", component):
            raise RuntimeError(
                f"Unsafe ROCm artifact identity component: {component!r}"
            )
    return "-".join(components)


def write_rocm_manifest(
    path: Path,
    *,
    release_type: str,
    run_id: str,
    platform_name: str,
    bucket: str,
    artifact_variant: str,
    artifact_set: str,
    artifacts: list[S3Object],
) -> None:
    artifact_identity = rocm_artifact_identity(
        release_type=release_type,
        run_id=run_id,
        platform_name=platform_name,
        artifact_variant=artifact_variant,
        artifact_set=artifact_set,
    )
    data = {
        "artifact_identity": artifact_identity,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "release_type": release_type,
        "run_id": run_id,
        "platform": platform_name,
        "bucket": bucket,
        "artifact_variant": artifact_variant,
        "artifact_set": artifact_set,
        "artifacts": [obj.__dict__ for obj in artifacts],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def fetch_rocm(
    args: argparse.Namespace,
    *,
    platform_name: str,
    platform_display: str,
    artifact_sets: dict[str, dict[str, list[str]]] = ARTIFACT_SETS,
    materialize_links: bool = False,
    preserve_symlinks: bool = True,
) -> None:
    s3 = create_s3_client()
    run_id = args.run_id
    if args.latest:
        run_id = discover_latest_run_id(
            s3,
            args.release_type,
            args.artifact_set,
            args.artifact_variant,
            platform_name=platform_name,
            platform_display=platform_display,
            artifact_sets=artifact_sets,
        )
        log(
            f"Resolved latest {args.release_type} {platform_display} "
            f"{args.artifact_variant} run id: {run_id}"
        )
    if not run_id:
        raise RuntimeError("Pass --run-id or --latest")

    bucket = release_bucket(args.release_type, "artifacts")
    prefix = f"{run_id}-{platform_name}/"
    available = list_prefix(s3, bucket, prefix)
    if not available:
        raise RuntimeError(f"No artifacts found at s3://{bucket}/{prefix}")
    validate_rocm_artifact_variant(s3, bucket, prefix, available, args.artifact_variant)

    selected, missing = select_available(
        available,
        prefix,
        wanted_artifacts(args.artifact_set, artifact_sets=artifact_sets),
    )
    if missing:
        raise RuntimeError("Missing required artifacts:\n  " + "\n  ".join(missing))

    log("Artifacts selected:")
    for obj in selected:
        log(f"  {obj.key} ({obj.size / 1024 / 1024:.1f} MiB)")

    output_dir = args.rocm_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.download_cache_dir.resolve()
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
        flatten_therock_artifact(
            archive_path,
            output_dir,
            materialize_links=materialize_links,
            preserve_symlinks=preserve_symlinks,
        )

    write_rocm_manifest(
        output_dir / ".hrx-rocm-artifacts.json",
        release_type=args.release_type,
        run_id=run_id,
        platform_name=platform_name,
        bucket=bucket,
        artifact_variant=args.artifact_variant,
        artifact_set=args.artifact_set,
        artifacts=selected,
    )
    log(f"ROCm build root ready: {output_dir}")


def cmake_options_from_env() -> list[str]:
    raw = os.environ.get("HRX_CMAKE_OPTIONS", "")
    options: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        options.extend(shlex.split(line))
    return options


def sanitizer_options(sanitizer: str) -> list[str]:
    if sanitizer == "none":
        return []
    return [f"IREE_ENABLE_{sanitizer.upper()}=ON"]


def combine_ctest_exclude_regex(*regexes: str) -> str:
    return "|".join(f"({regex})" for regex in regexes if regex)


def copy_matching_rocm_paths(
    rocm_root: Path,
    dst_root: Path,
    patterns: list[str],
    *,
    preserve_symlinks: bool = True,
) -> list[Path]:
    copied: list[Path] = []
    for pattern in patterns:
        for src in sorted(rocm_root.glob(pattern)):
            relpath = src.relative_to(rocm_root)
            copy_relative_path(
                rocm_root,
                dst_root,
                relpath,
                preserve_symlinks=preserve_symlinks,
            )
            copied.append(relpath)
    return copied


def prepare_composed_install_root(
    args: argparse.Namespace, *, preserve_symlinks: bool = True
) -> Path:
    composed_root = args.composed_install_dir.resolve()
    if composed_root.exists():
        remove_tree(composed_root)
    composed_root.mkdir(parents=True)
    for source_root in [
        args.public_deps_dir.resolve(),
        args.public_install_dir.resolve(),
        args.tests_install_dir.resolve(),
    ]:
        require_path(source_root, "install overlay root")
        copy_tree_contents(
            source_root,
            composed_root,
            preserve_symlinks=preserve_symlinks,
        )
    return composed_root


def write_package_manifest(
    manifest_path: Path,
    *,
    package_name: str,
    source_root: Path,
    archive_path: Path,
    rocm_root: Path | None = None,
    archive_name_key: str = "archive",
    archive_sha256_key: str = "archive_sha256",
) -> None:
    manifest = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "package": package_name,
        "platform": platform.platform(),
        "source_root": str(source_root),
        archive_name_key: archive_path.name,
        archive_sha256_key: sha256_file(archive_path),
    }
    if rocm_root:
        rocm_manifest = rocm_root / ".hrx-rocm-artifacts.json"
        if rocm_manifest.exists():
            manifest["rocm_artifacts"] = json.loads(rocm_manifest.read_text())
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def run_all(
    args: argparse.Namespace,
    *,
    fetch_rocm_fn: Callable[[argparse.Namespace], None],
    build_core_fn: Callable[[argparse.Namespace], None],
    test_core_fn: Callable[[argparse.Namespace], None],
    package_core_fn: Callable[[argparse.Namespace], None],
) -> None:
    fetch_args = argparse.Namespace(
        release_type=args.release_type,
        run_id=args.run_id,
        latest=not bool(args.run_id),
        artifact_set=args.artifact_set,
        artifact_variant=args.artifact_variant,
        rocm_root=args.rocm_root,
        download_cache_dir=args.download_cache_dir,
        download_concurrency=args.download_concurrency,
    )
    fetch_rocm_fn(fetch_args)
    build_core_fn(args)
    test_core_fn(args)
    if args.package:
        package_core_fn(args)


def add_shared_args(
    parser: argparse.ArgumentParser,
    *,
    default_output: Path,
    sanitizer_choices: list[str],
    passthrough_default: bool,
    public_component_default: str = "HrxPublicDist",
    tests_component_default: str = "HrxTestsDist",
    include_gpu: bool = False,
    include_amdgpu: bool = False,
    add_toolchain_args: bool = False,
    artifact_sets: dict[str, dict[str, list[str]]] = ARTIFACT_SETS,
) -> None:
    parser.add_argument(
        "--release-type",
        default=env_default("HRX_RELEASE_TYPE", "nightly"),
        choices=["dev", "nightly", "prerelease"],
    )
    parser.add_argument("--run-id", default=env_default("HRX_RUN_ID", ""))
    parser.add_argument(
        "--artifact-set",
        default=env_default("HRX_ARTIFACT_SET", "core"),
        choices=sorted(artifact_sets),
    )
    parser.add_argument(
        "--artifact-variant",
        default=env_default("HRX_ROCM_ARTIFACT_VARIANT", "release"),
        choices=ROCM_ARTIFACT_VARIANTS,
    )
    parser.add_argument(
        "--rocm-root",
        type=Path,
        default=env_path("HRX_ROCM_ROOT", default_output / "rocm-root"),
    )
    parser.add_argument(
        "--download-cache-dir",
        type=Path,
        default=env_path("HRX_DOWNLOAD_CACHE_DIR", default_output / "downloads"),
    )
    parser.add_argument(
        "--download-concurrency",
        type=int,
        default=env_int("HRX_DOWNLOAD_CONCURRENCY", 8),
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=env_path("HRX_BUILD_DIR", default_output / "build" / "hrx-core"),
    )
    parser.add_argument(
        "--public-install-dir",
        type=Path,
        default=env_path(
            "HRX_PUBLIC_INSTALL_DIR", default_output / "install" / "public"
        ),
    )
    parser.add_argument(
        "--tests-install-dir",
        type=Path,
        default=env_path("HRX_TESTS_INSTALL_DIR", default_output / "install" / "tests"),
    )
    parser.add_argument(
        "--public-deps-dir",
        type=Path,
        default=env_path(
            "HRX_PUBLIC_DEPS_DIR", default_output / "install" / "public-deps"
        ),
    )
    parser.add_argument(
        "--composed-install-dir",
        type=Path,
        default=env_path(
            "HRX_COMPOSED_INSTALL_DIR", default_output / "install" / "composed"
        ),
    )
    parser.add_argument(
        "--artifact-download-dir",
        type=Path,
        default=env_path(
            "HRX_ARTIFACT_DOWNLOAD_DIR", default_output / "downloaded-artifacts"
        ),
    )
    parser.add_argument(
        "--package-smoke-build-dir",
        type=Path,
        default=env_path(
            "HRX_PACKAGE_SMOKE_BUILD_DIR", default_output / "build" / "package-smoke"
        ),
    )
    parser.add_argument(
        "--package-output-dir",
        type=Path,
        default=env_path("HRX_PACKAGE_OUTPUT_DIR", default_output / "dist"),
    )
    parser.add_argument(
        "--public-component",
        default=env_default("HRX_PUBLIC_COMPONENT", public_component_default),
    )
    parser.add_argument(
        "--tests-component",
        default=env_default("HRX_TESTS_COMPONENT", tests_component_default),
    )
    parser.add_argument(
        "--build-type", default=env_default("HRX_BUILD_TYPE", "RelWithDebInfo")
    )
    parser.add_argument("--target", default=env_default("HRX_BUILD_TARGET", "all"))
    parser.add_argument(
        "--sanitizer",
        default=env_default("HRX_SANITIZER", "none"),
        choices=sanitizer_choices,
    )
    parser.add_argument(
        "--assertions",
        action=argparse.BooleanOptionalAction,
        default=env_bool("HRX_ASSERTIONS", False),
    )
    parser.add_argument("--ctest-regex", default=env_default("HRX_CTEST_REGEX", ""))
    parser.add_argument(
        "--ctest-exclude-regex", default=env_default("HRX_CTEST_EXCLUDE_REGEX", "")
    )
    parser.add_argument(
        "--ctest-label-regex", default=env_default("HRX_CTEST_LABEL_REGEX", "")
    )
    parser.add_argument(
        "--ctest-label-exclude-regex",
        default=env_default("HRX_CTEST_LABEL_EXCLUDE_REGEX", ""),
    )
    parser.add_argument(
        "--ctest-parallelism", type=int, default=env_int("HRX_CTEST_PARALLELISM", 0)
    )
    parser.add_argument("--cts-device", default=env_default("HRX_CTS_DEVICE", ""))
    parser.add_argument(
        "--test-tmpdir",
        type=Path,
        default=Path(os.environ["HRX_TEST_TMPDIR"])
        if os.environ.get("HRX_TEST_TMPDIR")
        else None,
    )
    if include_gpu:
        parser.add_argument(
            "--gpu", action="store_true", default=env_bool("HRX_TEST_GPU", False)
        )
    parser.add_argument(
        "--prepare-public-deps",
        action=argparse.BooleanOptionalAction,
        default=env_bool("HRX_PREPARE_PUBLIC_DEPS", True),
    )
    parser.add_argument(
        "--package-smoke",
        action=argparse.BooleanOptionalAction,
        default=env_bool("HRX_PACKAGE_SMOKE", True),
    )
    parser.add_argument(
        "--package",
        action=argparse.BooleanOptionalAction,
        default=env_bool("HRX_PACKAGE", True),
    )
    parser.add_argument(
        "--package-suffix", default=env_default("HRX_PACKAGE_SUFFIX", "")
    )
    parser.add_argument(
        "--passthrough",
        action=argparse.BooleanOptionalAction,
        default=env_bool("HRX_PASSTHROUGH", passthrough_default),
    )
    if include_amdgpu:
        parser.add_argument(
            "--amdgpu",
            action=argparse.BooleanOptionalAction,
            default=env_bool("HRX_AMDGPU", True),
        )
    if add_toolchain_args:
        parser.add_argument("--c-compiler", default=env_default("HRX_C_COMPILER", ""))
        parser.add_argument(
            "--cxx-compiler", default=env_default("HRX_CXX_COMPILER", "")
        )
        parser.add_argument("--ar", default=env_default("HRX_AR", ""))
        parser.add_argument("--linker", default=env_default("HRX_LINKER", ""))
    parser.add_argument("-D", dest="cmake_option", action="append", default=[])


def main(
    argv: list[str] | None,
    *,
    description: str | None,
    add_shared_args_fn: Callable[[argparse.ArgumentParser], None],
    run_all_fn: Callable[[argparse.Namespace], None],
    fetch_rocm_fn: Callable[[argparse.Namespace], None],
    build_core_fn: Callable[[argparse.Namespace], None],
    test_core_fn: Callable[[argparse.Namespace], None],
    extract_packages_fn: Callable[[argparse.Namespace], None],
    package_core_fn: Callable[[argparse.Namespace], None],
) -> int:
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run", help="Fetch ROCm, build, test, and optionally package"
    )
    add_shared_args_fn(run_parser)
    fetch_parser = subparsers.add_parser(
        "fetch-rocm", help="Fetch and flatten TheRock ROCm artifacts"
    )
    add_shared_args_fn(fetch_parser)
    build_parser = subparsers.add_parser("build", help="Configure, build, and install")
    add_shared_args_fn(build_parser)
    test_parser = subparsers.add_parser("test", help="Validate build and install trees")
    add_shared_args_fn(test_parser)
    extract_parser = subparsers.add_parser(
        "extract-packages", help="Extract downloaded HRX package artifacts"
    )
    add_shared_args_fn(extract_parser)
    package_parser = subparsers.add_parser(
        "package", help="Package an installed HRX/ROCm tree"
    )
    add_shared_args_fn(package_parser)
    args = parser.parse_args(argv)

    if args.command == "run":
        run_all_fn(args)
    elif args.command == "fetch-rocm":
        fetch_rocm_fn(
            argparse.Namespace(
                release_type=args.release_type,
                run_id=args.run_id,
                latest=not bool(args.run_id),
                artifact_set=args.artifact_set,
                artifact_variant=args.artifact_variant,
                rocm_root=args.rocm_root,
                download_cache_dir=args.download_cache_dir,
                download_concurrency=args.download_concurrency,
            )
        )
    elif args.command == "build":
        build_core_fn(args)
    elif args.command == "test":
        test_core_fn(args)
    elif args.command == "extract-packages":
        extract_packages_fn(args)
    elif args.command == "package":
        package_core_fn(args)
    return 0
