# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Snapshots failed Windows CI builds before a later phase can rebuild them.

The caller owns the build command and its exit status. Collection is read-only:
Ninja's input query enumerates the failed link's objects and archives without
running actions, and symbol tools inspect copies of those inputs. Missing files,
access errors, and capture limits are evidence and appear in the manifest.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Mapping, Sequence

ARTIFACT_DIR_ENV = "IREE_CI_FAILURE_ARTIFACT_DIR"
MAX_CAPTURE_BYTES = 512 * 1024 * 1024
BINARY_SUFFIXES = {".obj", ".lib", ".o", ".a"}
ENVIRONMENT_KEYS = {
    "CC",
    "CXX",
    "AR",
    "INCLUDE",
    "LIB",
    "LIBPATH",
    "WINDOWSSDKDIR",
    "WINDOWSSDKVERSION",
    "UCRTVERSION",
    "VCTOOLSINSTALLDIR",
    "VCTOOLSVERSION",
    "BAZEL_LLVM",
    "HRX_ROCM_ROOT",
    "GITHUB_SHA",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "RUNNER_NAME",
    "RUNNER_OS",
}


def environment_value(environment: Mapping[str, str], name: str) -> str | None:
    return next(
        (value for key, value in environment.items() if key.upper() == name.upper()),
        None,
    )


class Capture:
    def __init__(self, directory: Path, environment: Mapping[str, str]):
        # Each phase has its own directory, so later builds cannot replace it.
        self.directory = directory
        # Only toolchain and job provenance fields enter the manifest.
        self.environment = environment
        # Original absolute paths identify copied bytes and unsuccessful reads.
        self.files: dict[str, dict] = {}
        # Tool invocations retain their exit code and output, including failures.
        self.tools: list[dict] = []
        # The byte budget bounds binary snapshots from large failed links.
        self.remaining_bytes = MAX_CAPTURE_BYTES

    def file(self, path: Path, *, copy: bool) -> dict:
        path = path.absolute()
        key = str(path)
        if key in self.files:
            return self.files[key]
        record = {"path": key, "observed_at": dt.datetime.now(dt.UTC).isoformat()}
        self.files[key] = record
        try:
            metadata = path.stat()
            record.update(
                size=metadata.st_size,
                mtime_ns=metadata.st_mtime_ns,
                mode=metadata.st_mode,
                file_attributes=getattr(metadata, "st_file_attributes", None),
            )
            if metadata.st_size > self.remaining_bytes:
                record["error"] = "capture byte limit exceeded"
                return record
            destination = self.directory / f"{len(self.files):04d}-{path.name}"
            digest = hashlib.sha256()
            with path.open("rb") as source:
                with destination.open("wb") if copy else open(os.devnull, "wb") as sink:
                    length = 0
                    while chunk := source.read(1024 * 1024):
                        if len(chunk) > self.remaining_bytes:
                            raise OSError("capture byte limit exceeded while reading")
                        digest.update(chunk)
                        sink.write(chunk)
                        length += len(chunk)
                        self.remaining_bytes -= len(chunk)
                after = os.fstat(source.fileno())
            record.update(
                sha256=digest.hexdigest(),
                bytes_read=length,
                mtime_ns_after_read=after.st_mtime_ns,
                size_after_read=after.st_size,
            )
            if copy:
                record["copy"] = destination.name
        except OSError as error:
            record.update(
                error=str(error),
                errno=error.errno,
                winerror=getattr(error, "winerror", None),
            )
        return record

    def tool(self, argv: Sequence[str], cwd: Path, name: str) -> Path | None:
        output_path = self.directory / f"{len(self.tools):04d}-{name}.txt"
        record = {"argv": list(argv), "cwd": str(cwd), "output": output_path.name}
        self.tools.append(record)
        try:
            with output_path.open("wb") as output:
                result = subprocess.run(
                    argv,
                    cwd=cwd,
                    env=dict(self.environment),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=30,
                )
            record["returncode"] = result.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            record["error"] = str(error)
            return None
        return output_path if result.returncode == 0 else None


def cmake_cache(build_dir: Path) -> dict[str, str]:
    values = {}
    for line in (build_dir / "CMakeCache.txt").read_text(encoding="utf-8").splitlines():
        match = re.match(r"([A-Z_]+):[^=]+=(.*)", line)
        if match:
            values[match[1]] = match[2]
    return values


def capture_cmake(capture: Capture, build_dir: Path, output: str) -> dict[str, str]:
    capture.file(build_dir / "CMakeCache.txt", copy=True)
    # Ninja retains response files for failed actions until the next invocation.
    for path in sorted(build_dir.rglob("*.rsp")):
        capture.file(path, copy=True)
    cache = cmake_cache(build_dir)
    ninja = cache.get("CMAKE_MAKE_PROGRAM")
    if not ninja:
        raise ValueError("CMAKE_MAKE_PROGRAM is absent from the failed build cache")
    targets = re.findall(
        r"^FAILED:[ \t]+(?:\[code=[^\]]+\][ \t]+)?"
        r"([^\r\n]+?\.(?:exe|dll|lib|obj))(?=\s|$)",
        output,
        re.MULTILINE,
    )
    for target in dict.fromkeys(targets):
        inputs = capture.tool(
            [ninja, "-t", "inputs", "--no-shell-escape", target],
            build_dir,
            "ninja-inputs",
        )
        capture.tool(
            [ninja, "-t", "commands", "-s", target], build_dir, "ninja-command"
        )
        if inputs is None:
            continue
        for line in inputs.read_text(encoding="utf-8", errors="replace").splitlines():
            path = build_dir / line
            if path.suffix.lower() in BINARY_SUFFIXES:
                # Snapshot products owned by this build, not installed SDK libs.
                if path.resolve().is_relative_to(build_dir.resolve()):
                    capture.file(path, copy=True)
    # The full graph can be larger than the failed link's inputs. Preserve those
    # inputs before auxiliary build metadata consumes the bounded snapshot budget.
    for relative_path in (
        "build.ninja",
        "CMakeFiles/rules.ninja",
        ".ninja_log",
        ".ninja_deps",
    ):
        capture.file(build_dir / relative_path, copy=True)
    return cache


def capture_access_failures(capture: Capture, cwd: Path, output: str) -> None:
    read_paths = []
    delete_paths = []
    include_paths = [
        Path(value)
        for value in (environment_value(capture.environment, "INCLUDE") or "").split(
            ";"
        )
        if value
    ]
    for match in re.finditer(
        r"^(.+?)\(\d+(?:,\d+)?\): fatal error C1083: "
        r"Cannot open include file: ['\"]([^'\"]+)['\"]",
        output,
        re.MULTILINE,
    ):
        includer = Path(match[1])
        if not includer.is_absolute():
            includer = cwd / includer
        read_paths.append(capture.file(includer, copy=False)["path"])
        for parent in dict.fromkeys([includer.parent, *include_paths]):
            record = capture.file(parent / match[2], copy=False)
            read_paths.append(record["path"])
            try:
                record["parent_entries"] = sorted(
                    entry.name for entry in parent.iterdir()
                )
            except OSError as error:
                record["parent_error"] = str(error)
    # Parameter files are addressed relative to the execution root's bazel-out
    # link. Reading them does not start another Bazel invocation or action.
    for relative_path in dict.fromkeys(
        re.findall(r"@(?:\")?(bazel-out[/\\][^\s\"]+\.params)", output)
    ):
        capture.file(cwd / relative_path, copy=True)
    for name in dict.fromkeys(
        re.findall(
            r"Couldn't delete action output directory: (.+) \(Permission denied\)",
            output,
        )
    ):
        delete_paths.append(capture.file(Path(name), copy=True)["path"])
    if os.name == "nt" and (read_paths or delete_paths):
        arguments = [
            sys.executable,
            str(Path(__file__).with_name("windows_file_access.py")),
        ]
        for path in dict.fromkeys(read_paths):
            arguments.extend(["--read", path])
        for path in dict.fromkeys(delete_paths):
            arguments.extend(["--delete", path])
        capture.tool(arguments, cwd, "windows-file-access")


def symbol_tool(environment: Mapping[str, str], cache: Mapping[str, str]) -> str | None:
    directories = []
    if compiler := cache.get("CMAKE_C_COMPILER"):
        directories.append(Path(compiler).parent)
    if llvm := environment_value(environment, "BAZEL_LLVM"):
        directories.append(Path(llvm) / "bin")
    if rocm := environment_value(environment, "HRX_ROCM_ROOT"):
        directories.append(Path(rocm) / "lib/llvm/bin")
    for directory in directories:
        for name in ("llvm-nm.exe", "llvm-nm"):
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    search_path = environment_value(environment, "PATH")
    return shutil.which("llvm-nm", path=search_path) or shutil.which(
        "dumpbin", path=search_path
    )


def capture_failure(
    directory: Path, cwd: Path, environment: Mapping[str, str], build_dir: Path | None
) -> None:
    capture = Capture(directory, environment)
    metadata = {
        "environment": {
            key: value
            for key, value in environment.items()
            if key.upper() in ENVIRONMENT_KEYS
        }
    }
    output = (directory / "output.log").read_text(encoding="utf-8", errors="replace")
    output = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output)
    try:
        capture_access_failures(capture, cwd, output)
        cache = capture_cmake(capture, build_dir, output) if build_dir else {}
        tool = symbol_tool(environment, cache)
        metadata["symbol_tool"] = tool
        if tool:
            for record in list(capture.files.values()):
                if (
                    "copy" in record
                    and Path(record["path"]).suffix.lower() in BINARY_SUFFIXES
                ):
                    snapshot = directory / record["copy"]
                    if Path(tool).stem.lower() == "dumpbin":
                        options = ["/symbols"]
                        if snapshot.suffix.lower() == ".lib":
                            options.append("/linkermember:1")
                    else:
                        options = ["--print-armap"]
                    capture.tool([tool, *options, str(snapshot)], cwd, "symbols")
        else:
            metadata["symbol_error"] = "Neither llvm-nm nor dumpbin was available"
    except Exception:
        metadata["capture_error"] = traceback.format_exc()
        print(metadata["capture_error"], file=sys.stderr)
    finally:
        metadata.update(files=list(capture.files.values()), tools=capture.tools)
        (directory / "manifest.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )


def run(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None,
    artifact_dir: Path,
    label: str,
    build_dir: Path | None = None,
) -> int:
    """Runs once, preserving failed inputs before returning the original status."""
    environment = dict(os.environ if env is None else env)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prefix = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-") or "phase"
    directory = Path(tempfile.mkdtemp(prefix=prefix + "-", dir=artifact_dir)).resolve()
    command = {
        "argv": list(argv),
        "cwd": str(cwd),
        "started_at": dt.datetime.now(dt.UTC).isoformat(),
    }
    log_error = None
    output = (directory / "output.log").open("wb", buffering=0)
    try:
        with subprocess.Popen(
            argv,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as process:
            while chunk := process.stdout.read1(65536):
                if log_error is None:
                    try:
                        output.write(chunk)
                        output.flush()
                    except OSError as error:
                        # Keep draining the pipe even if the worker filesystem
                        # fails; waiting with a full child pipe would deadlock.
                        log_error = str(error)
                        print(
                            f"[diagnostics] Log write failed: {error}", file=sys.stderr
                        )
                if hasattr(sys.stdout, "buffer"):
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                else:
                    sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
            returncode = process.wait()
    finally:
        try:
            output.close()
        except OSError as error:
            log_error = str(error)
            print(f"[diagnostics] Log close failed: {error}", file=sys.stderr)
    command.update(
        returncode=returncode, completed_at=dt.datetime.now(dt.UTC).isoformat()
    )
    if log_error is not None:
        command["log_error"] = log_error
    try:
        (directory / "command.json").write_text(
            json.dumps(command, indent=2) + "\n", encoding="utf-8"
        )
        if returncode:
            print(f"[diagnostics] Capturing failed inputs in {directory}", flush=True)
            capture_failure(directory, cwd, environment, build_dir)
    except Exception:
        # Diagnostic IO can fail on the same worker as the build. Report it
        # without replacing a failure's exit code or accidentally passing it.
        traceback.print_exc()
        if returncode == 0:
            return 1
    return returncode or int(log_error is not None)
