#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
# SPDX-License-Identifier: Apache-2.0

"""Package HRX core plus its ROCm build root as one install tree."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
from pathlib import Path
import sys
import tarfile

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from hrx_build_tools import (
    REPO_ROOT,
    copy_tree_contents,
    log,
    remove_tree,
    require_file,
    write_env_script,
)


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _add_archive_contents(tf: tarfile.TarFile, source_dir: Path) -> None:
    for child in sorted(source_dir.iterdir(), key=lambda p: p.name):
        tf.add(child, arcname=child.name, recursive=True, filter=_tar_filter)


def _create_tar_zst(source_dir: Path, tarball_path: Path) -> None:
    try:
        import zstandard
    except ImportError as e:
        raise RuntimeError(
            "The zstandard package is required to create .tar.zst packages"
        ) from e

    cctx = zstandard.ZstdCompressor(level=3, threads=-1)
    with tarball_path.open("wb") as f:
        with cctx.stream_writer(f, closefd=False) as zstd_stream:
            with tarfile.open(fileobj=zstd_stream, mode="w|") as tf:
                _add_archive_contents(tf, source_dir)


def _create_tar_gz(source_dir: Path, tarball_path: Path) -> None:
    with tarfile.open(tarball_path, "w:gz") as tf:
        _add_archive_contents(tf, source_dir)


def create_tarball(source_dir: Path, tarball_path: Path) -> None:
    tarball_path.parent.mkdir(parents=True, exist_ok=True)
    if tarball_path.name.endswith(".tar.zst"):
        _create_tar_zst(source_dir, tarball_path)
    elif tarball_path.name.endswith(".tar.gz") or tarball_path.name.endswith(".tgz"):
        _create_tar_gz(source_dir, tarball_path)
    else:
        raise ValueError(
            f"Unsupported package archive extension: {tarball_path.name} "
            "(expected .tar.zst or .tar.gz)"
        )


def package(args: argparse.Namespace) -> Path:
    rocm_root = args.rocm_root.resolve()
    hrx_install = args.hrx_install.resolve()
    output_dir = args.output_dir.resolve()
    package_root = (
        args.package_root.resolve() if args.package_root else output_dir / "hrx-core"
    )

    require_file(rocm_root, "ROCm build root")
    require_file(hrx_install / "lib" / "libhrx.so", "installed libhrx.so")
    require_file(hrx_install / "lib" / "libamdhip64.so", "installed HRX libamdhip64.so")

    if package_root.exists():
        remove_tree(package_root)
    package_root.mkdir(parents=True)

    log(f"Copying ROCm root: {rocm_root} -> {package_root}")
    copy_tree_contents(rocm_root, package_root, skip_names={".download_cache"})

    if hrx_install != rocm_root:
        upstream_hip = package_root / "lib" / "libamdhip64.so"
        if upstream_hip.exists() or upstream_hip.is_symlink():
            if not args.allow_hip_overwrite:
                raise RuntimeError(
                    f"{upstream_hip} already exists. The default HRX core package "
                    "expects HRX to own libamdhip64.so; remove upstream HIP artifacts "
                    "or pass --allow-hip-overwrite."
                )
            upstream_hip.unlink()
        log(f"Overlaying HRX install: {hrx_install} -> {package_root}")
        copy_tree_contents(hrx_install, package_root)
    write_env_script(package_root, package_root / "env.sh")

    manifest = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "package": "hrx-core",
        "platform": platform.platform(),
        "rocm_root": str(rocm_root),
        "hrx_install": str(hrx_install),
        "hip_runtime_owner": "hrx",
    }
    rocm_manifest = rocm_root / ".hrx-rocm-artifacts.json"
    if rocm_manifest.exists():
        manifest["rocm_artifacts"] = json.loads(rocm_manifest.read_text())
    (package_root / ".hrx-core-package.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    tarball_name = (
        args.tarball_name
        or f"hrx-core-linux-x86_64-{manifest['generated_at'][:10]}.tar.zst"
    )
    tarball_path = output_dir / tarball_name
    create_tarball(package_root, tarball_path)
    log(f"Created {tarball_path}")
    return tarball_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    shared_root = REPO_ROOT.parent / "build" / "rocm-root"
    core_root = REPO_ROOT.parent / "build" / "hrx-core"
    parser.add_argument("--rocm-root", type=Path, default=shared_root)
    parser.add_argument("--hrx-install", type=Path, default=shared_root)
    parser.add_argument("--output-dir", type=Path, default=core_root / "dist")
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--tarball-name")
    parser.add_argument("--allow-hip-overwrite", action="store_true")
    args = parser.parse_args(argv)
    package(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
