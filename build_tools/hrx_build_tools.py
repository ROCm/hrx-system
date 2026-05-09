# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for HRX local build and packaging scripts."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BUILD_IMAGE = (
    "ghcr.io/rocm/therock_build_manylinux_x86_64"
    "@sha256:702a5133851e6d1daf1207d2c9fbb01c2667914a5b6dc5a01faeb3ce66ea6421"
)


def log(message: str = "") -> None:
    print(message, flush=True)


def run(
    args: Iterable[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    cmd = [os.fspath(arg) for arg in args]
    log(f"++ Exec [{cwd or Path.cwd()}]$ {' '.join(_quote(a) for a in cmd)}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _quote(value: str) -> str:
    if not value or any(c.isspace() or c in "\"'\\$" for c in value):
        return repr(value)
    return value


def remove_tree(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def copy_tree_contents(
    src: Path, dst: Path, *, overwrite: bool = True, skip_names: set[str] | None = None
) -> None:
    """Copy one prefix into another, preserving symlinks."""
    if not src.exists():
        raise FileNotFoundError(src)
    skip_names = skip_names or set()
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if child.name in skip_names:
            continue
        target = dst / child.name
        if overwrite and (target.exists() or target.is_symlink()):
            remove_tree(target)
        if child.is_symlink():
            target.symlink_to(os.readlink(child))
        elif child.is_dir():
            shutil.copytree(child, target, symlinks=True, dirs_exist_ok=overwrite)
        else:
            shutil.copy2(child, target, follow_symlinks=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_tar_archive(path: Path) -> tarfile.TarFile:
    """Open .tar.xz, .tar.gz, or .tar.zst archives."""
    name = path.name
    if name.endswith(".tar.zst"):
        try:
            import zstandard
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "The zstandard package is required to read .tar.zst artifacts"
            ) from e
        f = path.open("rb")
        dctx = zstandard.ZstdDecompressor()
        stream = dctx.stream_reader(f)
        try:
            tf = tarfile.open(fileobj=stream, mode="r|")
        except Exception:
            stream.close()
            f.close()
            raise
        tf._hrx_owned_streams = (stream, f)  # type: ignore[attr-defined]
        return tf
    if name.endswith(".tar.xz"):
        return tarfile.open(path, mode="r:xz")
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        return tarfile.open(path, mode="r:gz")
    raise ValueError(f"Unsupported archive extension: {path}")


def close_tar_archive(tf: tarfile.TarFile) -> None:
    owned = getattr(tf, "_hrx_owned_streams", ())
    tf.close()
    for stream in owned:
        stream.close()


def _checked_dest(base: Path, relpath: str) -> Path:
    rel = PurePosixPath(relpath)
    if rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError(f"Unsafe archive path: {relpath}")
    dest = base / rel
    base_resolved = base.resolve()
    parent_resolved = dest.parent.resolve()
    if (
        base_resolved != parent_resolved
        and base_resolved not in parent_resolved.parents
    ):
        raise RuntimeError(f"Archive path escapes output directory: {relpath}")
    return dest


def flatten_therock_artifact(archive_path: Path, output_dir: Path) -> set[str]:
    """Flatten a TheRock artifact archive into a normal install prefix.

    TheRock artifact archives begin with artifact_manifest.txt. All later
    members live below one of the manifest root paths; flattening strips that
    root path and copies the staged files into output_dir.
    """
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
            scoped_name = _strip_manifest_root(member.name, relroots)
            dest_path = _checked_dest(output_dir, scoped_name)
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
                hardlinks.append((dest_path, member.linkname, relroots))
            else:
                raise RuntimeError(f"Unhandled tar member type: {member.name}")

        for dest_path, linkname, relroots in hardlinks:
            target_name = _strip_manifest_root(linkname, relroots)
            target_path = _checked_dest(output_dir, target_name)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if dest_path.exists() or dest_path.is_symlink():
                dest_path.unlink()
            os.link(target_path, dest_path)
        return set(relroots)
    finally:
        close_tar_archive(tf)


def _strip_manifest_root(member_name: str, relroots: list[str]) -> str:
    for root in relroots:
        prefix = root.rstrip("/") + "/"
        if member_name.startswith(prefix):
            scoped = member_name[len(prefix) :]
            if scoped:
                return scoped
    raise RuntimeError(f"Archive member is outside manifest roots: {member_name}")


def write_env_script(prefix: Path, path: Path) -> None:
    content = """#!/usr/bin/env bash
# Source this file to use this HRX core installation.
_hrx_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HRX_HOME="${_hrx_env_dir}"
export ROCM_HOME="${_hrx_env_dir}"
export PATH="${_hrx_env_dir}/bin:${PATH}"
export LD_LIBRARY_PATH="${_hrx_env_dir}/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${_hrx_env_dir}:${CMAKE_PREFIX_PATH:-}"
unset _hrx_env_dir
"""
    path.write_text(content)
    path.chmod(0o755)


def require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
