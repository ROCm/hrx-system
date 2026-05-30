#!/usr/bin/env python3
# Copyright 2026 The HRX Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Linux CI driver for configuring, building, testing, and packaging HRX.

The GitHub workflow is intentionally thin: it selects a configuration and this
script owns the build details. That keeps sanitizer, packaging, and future
release variants reusable without growing large stringly shell snippets in YAML.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent
PLATFORM = "linux"

# Minimal TheRock artifact closures used by HRX CI. The default set provides
# ROCr/HSA, AQL profile, AMD SMI, base ROCm metadata, sysdeps, and the ROCm LLVM
# toolchain used to build host code and amdgcn device binaries.
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

PACKAGE_NAMES = [
    "hrx-public-linux-x86_64",
    "hrx-public-deps-linux-x86_64",
    "hrx-tests-linux-x86_64",
    "hrx-rocm-buildenv-linux-x86_64",
]

PUBLIC_DEPS_REQUIRED_GLOBS = [
    "lib/libhsa-runtime64.so*",
    "lib/libhsa-amd-aqlprofile64.so*",
]

PUBLIC_DEPS_OPTIONAL_GLOBS = [
    "lib/libhsakmt.*",
    "lib/librocprofiler-register.so*",
    "lib/rocm_sysdeps/lib/*.so*",
]


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
) -> None:
    cmd = [os.fspath(arg) for arg in args]
    log(f"++ Exec [{cwd or Path.cwd()}]$ {' '.join(shlex.quote(a) for a in cmd)}")
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


def copy_tree_contents(src: Path, dst: Path, *, skip_names: set[str] | None = None) -> None:
    skip_names = skip_names or set()
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if child.name in skip_names:
            continue
        target = dst / child.name
        if child.is_symlink():
            if target.exists() or target.is_symlink():
                remove_tree(target)
            target.symlink_to(os.readlink(child))
        elif child.is_dir():
            if target.exists() or target.is_symlink():
                if not target.is_dir() or target.is_symlink():
                    remove_tree(target)
                    shutil.copytree(child, target, symlinks=True)
                else:
                    copy_tree_contents(child, target)
            else:
                shutil.copytree(child, target, symlinks=True)
        else:
            if target.exists() or target.is_symlink():
                remove_tree(target)
            shutil.copy2(child, target, follow_symlinks=False)


def copy_path(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        remove_tree(dst)
    if src.is_symlink():
        dst.symlink_to(os.readlink(src))
    elif src.is_dir():
        shutil.copytree(src, dst, symlinks=True)
    else:
        shutil.copy2(src, dst, follow_symlinks=False)


def copy_relative_path(src_root: Path, dst_root: Path, relpath: Path) -> None:
    copy_path(src_root / relpath, dst_root / relpath)


def path_relative_to(path: Path, root: Path) -> Path | None:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return None


def rocm_llvm_bin(rocm_root: Path) -> Path:
    return rocm_root / "lib" / "llvm" / "bin"


def rocm_tool(rocm_root: Path, name: str) -> Path:
    tool_path = rocm_llvm_bin(rocm_root) / name
    if not tool_path.exists():
        raise FileNotFoundError(f"Missing ROCm LLVM tool {name}: {tool_path}")
    return tool_path


def rocm_build_env(rocm_root: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    llvm_bin = rocm_llvm_bin(rocm_root)
    env["PATH"] = f"{llvm_bin}:{rocm_root / 'bin'}:{env.get('PATH', '')}"
    env["CMAKE_PREFIX_PATH"] = f"{rocm_root}:{env.get('CMAKE_PREFIX_PATH', '')}"
    return env


def install_runtime_env(root: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base_env or os.environ)
    lib_paths = [
        root / "lib",
        root / "lib" / "rocm_sysdeps" / "lib",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(path) for path in lib_paths] + [env.get("LD_LIBRARY_PATH", "")]
    )
    env["PATH"] = f"{root / 'bin'}:{env.get('PATH', '')}"
    env["CMAKE_PREFIX_PATH"] = f"{root}:{env.get('CMAKE_PREFIX_PATH', '')}"
    return env


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
    rel = PurePosixPath(relpath)
    if rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError(f"Unsafe archive path: {relpath}")
    dest = base / rel
    base_resolved = base.resolve()
    parent_resolved = dest.parent.resolve()
    if base_resolved != parent_resolved and base_resolved not in parent_resolved.parents:
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


def flatten_therock_artifact(archive_path: Path, output_dir: Path) -> set[str]:
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
            os.link(target_path, dest_path)
        return set(relroots)
    finally:
        close_tar_archive(tf)


def create_s3_client():
    try:
        from botocore import UNSIGNED
        from botocore.config import Config
        import boto3
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


def wanted_artifacts(artifact_set: str) -> list[str]:
    try:
        mapping = ARTIFACT_SETS[artifact_set]
    except KeyError as e:
        raise ValueError(
            f"Unknown artifact set {artifact_set!r}; expected one of "
            f"{', '.join(sorted(ARTIFACT_SETS))}"
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


def discover_latest_run_id(s3, release_type: str, artifact_set: str) -> str:
    bucket = release_bucket(release_type, "artifacts")
    paginator = s3.get_paginator("list_objects_v2")
    candidates: list[int] = []
    for page in paginator.paginate(Bucket=bucket, Delimiter="/"):
        for common_prefix in page.get("CommonPrefixes", []):
            match = re.match(r"^(\d+)-linux/$", common_prefix["Prefix"])
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


def write_rocm_manifest(
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


def fetch_rocm(args: argparse.Namespace) -> None:
    s3 = create_s3_client()
    run_id = args.run_id
    if args.latest:
        run_id = discover_latest_run_id(s3, args.release_type, args.artifact_set)
        log(f"Resolved latest {args.release_type} Linux run id: {run_id}")
    if not run_id:
        raise RuntimeError("Pass --run-id or --latest")

    bucket = release_bucket(args.release_type, "artifacts")
    prefix = f"{run_id}-{PLATFORM}/"
    available = list_prefix(s3, bucket, prefix)
    if not available:
        raise RuntimeError(f"No artifacts found at s3://{bucket}/{prefix}")

    selected, missing = select_available(
        available, prefix, wanted_artifacts(args.artifact_set)
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
        flatten_therock_artifact(archive_path, output_dir)

    write_rocm_manifest(
        output_dir / ".hrx-rocm-artifacts.json",
        release_type=args.release_type,
        run_id=run_id,
        bucket=bucket,
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


def build_core(args: argparse.Namespace) -> None:
    rocm_root = args.rocm_root.resolve()
    build_dir = args.build_dir.resolve()
    public_install_dir = args.public_install_dir.resolve()
    tests_install_dir = args.tests_install_dir.resolve()
    require_path(rocm_root, "ROCm build root")

    c_compiler = rocm_tool(rocm_root, "clang")
    cxx_compiler = rocm_tool(rocm_root, "clang++")
    ar = rocm_tool(rocm_root, "llvm-ar")
    ranlib = rocm_tool(rocm_root, "llvm-ranlib")

    cmake_prefix_path = ";".join([str(rocm_root)])
    ctest_enabled = True
    cmake_defines = [
        f"CMAKE_PREFIX_PATH={cmake_prefix_path}",
        f"CMAKE_INSTALL_PREFIX={public_install_dir}",
        "CMAKE_INSTALL_LIBDIR=lib",
        f"CMAKE_C_COMPILER={c_compiler}",
        f"CMAKE_CXX_COMPILER={cxx_compiler}",
        f"CMAKE_ASM_COMPILER={c_compiler}",
        f"CMAKE_AR={ar}",
        f"CMAKE_RANLIB={ranlib}",
        "CMAKE_C_COMPILER_LAUNCHER=ccache",
        "CMAKE_CXX_COMPILER_LAUNCHER=ccache",
        "CMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
        "CMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
        "CMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
        f"CMAKE_BUILD_TYPE={args.build_type}",
        f"IREE_BUILD_TESTS={'ON' if ctest_enabled else 'OFF'}",
        "IREE_BUILD_BENCHMARKS=ON",
        f"LIBHRX_BUILD_CTS={'ON' if ctest_enabled else 'OFF'}",
        f"HRX_INSTALL_TESTS={'ON' if ctest_enabled else 'OFF'}",
        f"LIBHRX_BUILD_PASSTHROUGH={'ON' if args.passthrough else 'OFF'}",
        f"IREE_HAL_DRIVER_AMDGPU={'ON' if args.amdgpu else 'OFF'}",
        "IREE_ENABLE_LIBBACKTRACE=OFF",
    ]
    cmake_defines.extend(sanitizer_options(args.sanitizer))
    cmake_defines.extend(cmake_options_from_env())
    cmake_defines.extend(args.cmake_option)

    env = rocm_build_env(rocm_root)
    run(
        [
            "cmake",
            "-S",
            REPO_ROOT,
            "-B",
            build_dir,
            "-GNinja",
            *[f"-D{option.removeprefix('-D')}" for option in cmake_defines],
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    run(["cmake", "--build", build_dir, "--target", args.target], cwd=REPO_ROOT, env=env)

    for install_dir, component in [
        (public_install_dir, args.public_component),
        (tests_install_dir, args.tests_component),
    ]:
        if install_dir.exists():
            remove_tree(install_dir)
        run(
            [
                "cmake",
                "--install",
                build_dir,
                "--prefix",
                install_dir,
                "--component",
                component,
            ],
            cwd=REPO_ROOT,
            env=env,
        )


def test_core(args: argparse.Namespace) -> None:
    rocm_root = args.rocm_root.resolve()
    smoke_build_dir = args.package_smoke_build_dir.resolve()
    prepare_public_deps_root(args)
    install_root = prepare_composed_install_root(args)
    installed_tests_dir = install_root / "share" / "hrx-system" / "tests"

    require_path(rocm_root, "ROCm build root")
    require_path(installed_tests_dir / "CTestTestfile.cmake", "installed CTest file")
    require_path(install_root / "lib" / "libhrx.so", "installed libhrx.so")
    require_path(install_root / "bin" / "hrx-info", "installed hrx-info")

    env = install_runtime_env(install_root, rocm_build_env(rocm_root))

    ctest_parallelism = args.ctest_parallelism or default_ctest_parallelism()
    ctest_cmd = [
        "ctest",
        "--test-dir",
        installed_tests_dir,
        "--output-on-failure",
        "--parallel",
        str(ctest_parallelism),
    ]
    if args.ctest_regex:
        ctest_cmd.extend(["-R", args.ctest_regex])
    run(ctest_cmd, cwd=REPO_ROOT, env=env, stderr_to_stdout=True)
    run([install_root / "bin" / "hrx-info"], cwd=REPO_ROOT, env=env)
    run([install_root / "bin" / "hrx-info", "--device=cpu:0"], cwd=REPO_ROOT, env=env)
    if args.gpu:
        run([install_root / "bin" / "hrx-info", "--device=gpu:0"], cwd=REPO_ROOT, env=env)

    run(
        [
            "cmake",
            "-S",
            REPO_ROOT / "libhrx" / "cts" / "package_smoke",
            "-B",
            smoke_build_dir,
            "-GNinja",
            f"-DCMAKE_PREFIX_PATH={install_root};{rocm_root}",
            f"-DCMAKE_C_COMPILER={rocm_tool(rocm_root, 'clang')}",
            f"-DCMAKE_CXX_COMPILER={rocm_tool(rocm_root, 'clang++')}",
            f"-DCMAKE_AR={rocm_tool(rocm_root, 'llvm-ar')}",
            f"-DCMAKE_RANLIB={rocm_tool(rocm_root, 'llvm-ranlib')}",
            "-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld",
            "-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld",
            "-DCMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld",
        ],
        cwd=REPO_ROOT,
        env=env,
    )
    run(["cmake", "--build", smoke_build_dir], cwd=REPO_ROOT, env=env)
    run([smoke_build_dir / "hrx_package_smoke"], cwd=REPO_ROOT, env=env)
    run([smoke_build_dir / "hrx_package_smoke_cxx"], cwd=REPO_ROOT, env=env)


ROCM_BUILDENV_EXCLUDE_PATHS = {
    "bin/hrx-info",
    "env.sh",
    "hrx-public-linux-x86_64-env.sh",
    "hrx-public-deps-linux-x86_64-env.sh",
    "hrx-tests-linux-x86_64-env.sh",
    "hrx-rocm-buildenv-linux-x86_64-env.sh",
    "include/hrx",
    "include/passthrough",
    "lib/cmake/hrx",
    "lib/libhrx.so",
    "share/hrx-cts",
    "share/hrx-system",
}

ROCM_BUILDENV_EXCLUDE_PREFIXES = (
    "include/hrx/",
    "include/passthrough/",
    "lib/cmake/hrx/",
    "lib/libhrx.so.",
    "share/hrx-cts/",
    "share/hrx-system/",
)


def tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def make_env_script_content() -> str:
    return """#!/usr/bin/env bash
# Source this file to use this HRX installation overlay.
_hrx_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HRX_HOME="${_hrx_env_dir}"
export ROCM_HOME="${_hrx_env_dir}"
export PATH="${_hrx_env_dir}/bin:${PATH}"
export LD_LIBRARY_PATH="${_hrx_env_dir}/lib:${_hrx_env_dir}/lib/rocm_sysdeps/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="${_hrx_env_dir}:${CMAKE_PREFIX_PATH:-}"
unset _hrx_env_dir
"""


def should_exclude_tar_path(
    arcname: str,
    *,
    exclude_paths: set[str],
    exclude_prefixes: tuple[str, ...],
) -> bool:
    return arcname in exclude_paths or any(
        arcname.startswith(prefix) for prefix in exclude_prefixes
    )


def create_tar_zst(
    source_dir: Path,
    tarball_path: Path,
    *,
    exclude_paths: set[str] | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> None:
    try:
        import zstandard
    except ModuleNotFoundError as e:
        raise RuntimeError("Install the zstandard Python package") from e
    tarball_path.parent.mkdir(parents=True, exist_ok=True)
    exclude_paths = exclude_paths or set()

    def filter_member(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if should_exclude_tar_path(
            info.name, exclude_paths=exclude_paths, exclude_prefixes=exclude_prefixes
        ):
            return None
        return tar_filter(info)

    cctx = zstandard.ZstdCompressor(level=3, threads=-1)
    with tarball_path.open("wb") as f:
        with cctx.stream_writer(f, closefd=False) as zstd_stream:
            with tarfile.open(fileobj=zstd_stream, mode="w|") as tf:
                for child in sorted(source_dir.iterdir(), key=lambda p: p.name):
                    tf.add(child, arcname=child.name, recursive=True, filter=filter_member)


def write_package_env_script(output_dir: Path, script_name: str) -> Path:
    script_path = output_dir / script_name
    script_path.write_text(make_env_script_content())
    script_path.chmod(0o755)
    return script_path


def ldd_rocm_dependencies(rocm_root: Path, binary: Path) -> list[Path]:
    if not binary.exists():
        return []
    result = subprocess.run(
        ["ldd", binary],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    deps: list[Path] = []
    for line in result.stdout.splitlines():
        match = re.search(r"=>\s+(\S+)\s+\(", line)
        if not match:
            match = re.match(r"\s*(/\S+)\s+\(", line)
        if not match:
            continue
        dep_path = Path(match.group(1))
        if path_relative_to(dep_path, rocm_root) is not None:
            deps.append(dep_path)
    return deps


def copy_matching_rocm_paths(rocm_root: Path, dst_root: Path, patterns: list[str]) -> list[Path]:
    copied: list[Path] = []
    for pattern in patterns:
        for src in sorted(rocm_root.glob(pattern)):
            relpath = src.relative_to(rocm_root)
            copy_relative_path(rocm_root, dst_root, relpath)
            copied.append(relpath)
    return copied


def prepare_public_deps_root(args: argparse.Namespace) -> None:
    rocm_root = args.rocm_root.resolve()
    deps_root = args.public_deps_dir.resolve()
    require_path(rocm_root, "ROCm build root")

    if deps_root.exists():
        remove_tree(deps_root)
    deps_root.mkdir(parents=True)

    missing_required = []
    for pattern in PUBLIC_DEPS_REQUIRED_GLOBS:
        if not list(rocm_root.glob(pattern)):
            missing_required.append(pattern)
    if missing_required:
        raise RuntimeError(
            "Missing required public dependency files in ROCm root:\n  "
            + "\n  ".join(missing_required)
        )

    copied = copy_matching_rocm_paths(
        rocm_root, deps_root, PUBLIC_DEPS_REQUIRED_GLOBS + PUBLIC_DEPS_OPTIONAL_GLOBS
    )
    for seed in [
        rocm_root / "lib" / "libhsa-runtime64.so",
        rocm_root / "lib" / "libhsa-amd-aqlprofile64.so",
    ]:
        for dep_path in ldd_rocm_dependencies(rocm_root, seed):
            relpath = dep_path.resolve().relative_to(rocm_root.resolve())
            copy_relative_path(rocm_root, deps_root, relpath)
            copied.append(relpath)

    manifest = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "package": "hrx-public-deps-linux-x86_64",
        "platform": platform.platform(),
        "rocm_root": str(rocm_root),
        "copied_paths": sorted({os.fspath(path) for path in copied}),
    }
    rocm_manifest = rocm_root / ".hrx-rocm-artifacts.json"
    if rocm_manifest.exists():
        manifest["rocm_artifacts"] = json.loads(rocm_manifest.read_text())
    (deps_root / ".hrx-package.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    log(f"Public dependency root ready: {deps_root}")


def prepare_composed_install_root(args: argparse.Namespace) -> Path:
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
        copy_tree_contents(source_root, composed_root)
    return composed_root


def write_package_manifest(
    manifest_path: Path,
    *,
    package_name: str,
    source_root: Path,
    tarball_path: Path,
    rocm_root: Path | None = None,
) -> None:
    manifest = {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "package": package_name,
        "platform": platform.platform(),
        "source_root": str(source_root),
        "tarball": tarball_path.name,
        "tarball_sha256": sha256_file(tarball_path),
    }
    if rocm_root:
        rocm_manifest = rocm_root / ".hrx-rocm-artifacts.json"
        if rocm_manifest.exists():
            manifest["rocm_artifacts"] = json.loads(rocm_manifest.read_text())
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def package_tree(
    *,
    package_name: str,
    source_root: Path,
    output_dir: Path,
    suffix: str,
    rocm_root: Path | None = None,
    env_script_name: str | None = None,
    exclude_paths: set[str] | None = None,
    exclude_prefixes: tuple[str, ...] = (),
) -> None:
    require_path(source_root, f"{package_name} source root")
    output_dir.mkdir(parents=True, exist_ok=True)
    package_exclude_paths = set(exclude_paths or set())
    package_exclude_paths.add("env.sh")
    if env_script_name:
        package_exclude_paths.add(env_script_name)

    package_stem = f"{package_name}-{suffix}" if suffix else package_name
    tarball_path = output_dir / f"{package_stem}.tar.zst"
    if tarball_path.exists():
        tarball_path.unlink()
    create_tar_zst(
        source_root,
        tarball_path,
        exclude_paths=package_exclude_paths,
        exclude_prefixes=exclude_prefixes,
    )
    env_script_path = None
    if env_script_name:
        env_script_path = write_package_env_script(output_dir, env_script_name)
    manifest_path = output_dir / f"{package_stem}.manifest.json"
    write_package_manifest(
        manifest_path,
        package_name=package_name,
        source_root=source_root,
        tarball_path=tarball_path,
        rocm_root=rocm_root,
    )
    log(f"Created {tarball_path}")
    if env_script_path:
        log(f"Created {env_script_path}")
    log(f"Created {manifest_path}")


def package_core(args: argparse.Namespace) -> None:
    rocm_root = args.rocm_root.resolve()
    output_dir = args.package_output_dir.resolve()
    suffix = args.package_suffix or dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")

    require_path(args.public_install_dir.resolve() / "lib" / "libhrx.so", "public libhrx.so")
    require_path(args.public_install_dir.resolve() / "bin" / "hrx-info", "public hrx-info")
    require_path(
        args.public_install_dir.resolve() / "lib" / "cmake" / "hrx" / "hrx-config.cmake",
        "public hrx CMake package",
    )
    require_path(
        args.tests_install_dir.resolve() / "share" / "hrx-system" / "tests" / "CTestTestfile.cmake",
        "installed CTest file",
    )
    prepare_public_deps_root(args)

    package_tree(
        package_name="hrx-public-linux-x86_64",
        source_root=args.public_install_dir.resolve(),
        output_dir=output_dir,
        suffix=suffix,
        rocm_root=rocm_root,
        env_script_name="hrx-public-linux-x86_64-env.sh",
    )
    package_tree(
        package_name="hrx-public-deps-linux-x86_64",
        source_root=args.public_deps_dir.resolve(),
        output_dir=output_dir,
        suffix=suffix,
        rocm_root=rocm_root,
        env_script_name="hrx-public-deps-linux-x86_64-env.sh",
    )
    package_tree(
        package_name="hrx-tests-linux-x86_64",
        source_root=args.tests_install_dir.resolve(),
        output_dir=output_dir,
        suffix=suffix,
        rocm_root=rocm_root,
    )
    package_tree(
        package_name="hrx-rocm-buildenv-linux-x86_64",
        source_root=rocm_root,
        output_dir=output_dir,
        suffix=suffix,
        rocm_root=rocm_root,
        env_script_name="hrx-rocm-buildenv-linux-x86_64-env.sh",
        exclude_paths=ROCM_BUILDENV_EXCLUDE_PATHS,
        exclude_prefixes=ROCM_BUILDENV_EXCLUDE_PREFIXES,
    )


def run_all(args: argparse.Namespace) -> None:
    fetch_args = argparse.Namespace(
        release_type=args.release_type,
        run_id=args.run_id,
        latest=not bool(args.run_id),
        artifact_set=args.artifact_set,
        rocm_root=args.rocm_root,
        download_cache_dir=args.download_cache_dir,
        download_concurrency=args.download_concurrency,
    )
    fetch_rocm(fetch_args)
    build_core(args)
    test_core(args)
    if args.package:
        package_core(args)


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    default_output = REPO_ROOT / "build" / "linux"
    parser.add_argument("--release-type", default=env_default("HRX_RELEASE_TYPE", "nightly"), choices=["dev", "nightly", "prerelease"])
    parser.add_argument("--run-id", default=env_default("HRX_RUN_ID", ""))
    parser.add_argument("--artifact-set", default=env_default("HRX_ARTIFACT_SET", "core"), choices=sorted(ARTIFACT_SETS))
    parser.add_argument("--rocm-root", type=Path, default=env_path("HRX_ROCM_ROOT", default_output / "rocm-root"))
    parser.add_argument("--download-cache-dir", type=Path, default=env_path("HRX_DOWNLOAD_CACHE_DIR", default_output / "downloads"))
    parser.add_argument("--download-concurrency", type=int, default=env_int("HRX_DOWNLOAD_CONCURRENCY", 8))
    parser.add_argument("--build-dir", type=Path, default=env_path("HRX_BUILD_DIR", default_output / "build" / "hrx-core"))
    parser.add_argument("--public-install-dir", type=Path, default=env_path("HRX_PUBLIC_INSTALL_DIR", default_output / "install" / "public"))
    parser.add_argument("--tests-install-dir", type=Path, default=env_path("HRX_TESTS_INSTALL_DIR", default_output / "install" / "tests"))
    parser.add_argument("--public-deps-dir", type=Path, default=env_path("HRX_PUBLIC_DEPS_DIR", default_output / "install" / "public-deps"))
    parser.add_argument("--composed-install-dir", type=Path, default=env_path("HRX_COMPOSED_INSTALL_DIR", default_output / "install" / "composed"))
    parser.add_argument("--package-smoke-build-dir", type=Path, default=env_path("HRX_PACKAGE_SMOKE_BUILD_DIR", default_output / "build" / "package-smoke"))
    parser.add_argument("--package-output-dir", type=Path, default=env_path("HRX_PACKAGE_OUTPUT_DIR", default_output / "dist"))
    parser.add_argument("--public-component", default=env_default("HRX_PUBLIC_COMPONENT", "HrxPublicDist"))
    parser.add_argument("--tests-component", default=env_default("HRX_TESTS_COMPONENT", "HrxTestsDist"))
    parser.add_argument("--build-type", default=env_default("HRX_BUILD_TYPE", "RelWithDebInfo"))
    parser.add_argument("--target", default=env_default("HRX_BUILD_TARGET", "all"))
    parser.add_argument("--sanitizer", default=env_default("HRX_SANITIZER", "none"), choices=["none", "asan", "tsan", "ubsan"])
    parser.add_argument("--ctest-regex", default=env_default("HRX_CTEST_REGEX", ""))
    parser.add_argument("--ctest-parallelism", type=int, default=env_int("HRX_CTEST_PARALLELISM", 0))
    parser.add_argument("--gpu", action="store_true", default=env_bool("HRX_TEST_GPU", False))
    parser.add_argument("--package", action=argparse.BooleanOptionalAction, default=env_bool("HRX_PACKAGE", True))
    parser.add_argument("--package-suffix", default=env_default("HRX_PACKAGE_SUFFIX", ""))
    parser.add_argument("--passthrough", action=argparse.BooleanOptionalAction, default=env_bool("HRX_PASSTHROUGH", True))
    parser.add_argument("--amdgpu", action=argparse.BooleanOptionalAction, default=env_bool("HRX_AMDGPU", True))
    parser.add_argument("-D", dest="cmake_option", action="append", default=[])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Fetch ROCm, build, test, and optionally package")
    add_shared_args(run_parser)
    fetch_parser = subparsers.add_parser("fetch-rocm", help="Fetch and flatten TheRock ROCm artifacts")
    add_shared_args(fetch_parser)
    build_parser = subparsers.add_parser("build", help="Configure, build, and install")
    add_shared_args(build_parser)
    test_parser = subparsers.add_parser("test", help="Validate build and install trees")
    add_shared_args(test_parser)
    package_parser = subparsers.add_parser("package", help="Package an installed HRX/ROCm tree")
    add_shared_args(package_parser)
    args = parser.parse_args(argv)

    if args.command == "run":
        run_all(args)
    elif args.command == "fetch-rocm":
        fetch_rocm(
            argparse.Namespace(
                release_type=args.release_type,
                run_id=args.run_id,
                latest=not bool(args.run_id),
                artifact_set=args.artifact_set,
                rocm_root=args.rocm_root,
                download_cache_dir=args.download_cache_dir,
                download_concurrency=args.download_concurrency,
            )
        )
    elif args.command == "build":
        build_core(args)
    elif args.command == "test":
        test_core(args)
    elif args.command == "package":
        package_core(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
