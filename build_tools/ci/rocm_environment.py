#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs the native ROCm probe and records a stalled process before termination."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TextIO


def select_native_gpu(
    output: str, topology_path: Path, drm_path: Path, device_path: Path
) -> str:
    """Correlates the visible ROCr agent with KFD and its actual render node."""
    agents = []
    for block in re.split(r"^Agent \d+\s*$", output, flags=re.MULTILINE)[1:]:
        fields = dict(
            re.findall(r"^  ([^:\n]+):[ \t]+([^\n]*?)\s*$", block, re.MULTILINE)
        )
        if fields.get("Device Type") == "GPU":
            agents.append(fields)
    if len(agents) != 1:
        raise RuntimeError(
            f"Native CTS allocation requires one visible ROCr GPU, found {len(agents)}"
        )
    agent = agents[0]
    location = int(agent["BDFID"])
    uuid = re.fullmatch(r"GPU-([0-9a-fA-F]{16})", agent["Uuid"])
    if uuid is None and agent["Uuid"] != "GPU-XX":
        raise RuntimeError(f"Unrecognized ROCr GPU UUID {agent['Uuid']!r}")
    candidates = []
    for node in topology_path.iterdir():
        if int((node / "gpu_id").read_text()) == 0:
            continue
        properties = dict(
            line.split(maxsplit=1)
            for line in (node / "properties").read_text().splitlines()
        )
        if int(properties["location_id"]) != location:
            continue
        if uuid and int(properties["unique_id"]) != int(uuid[1], 16):
            continue
        candidates.append(properties)
    if len(candidates) != 1:
        raise RuntimeError(
            f"ROCr GPU {agent['Uuid']} at BDFID {location} matches "
            f"{len(candidates)} KFD nodes; refusing an ambiguous allocation"
        )
    minor = int(candidates[0]["drm_render_minor"])
    name = f"renderD{minor}"
    major, reported_minor = map(int, (drm_path / name / "dev").read_text().split(":"))
    node = device_path / name
    metadata = node.stat()
    if (
        reported_minor != minor
        or not stat.S_ISCHR(metadata.st_mode)
        or (os.major(metadata.st_rdev), os.minor(metadata.st_rdev)) != (major, minor)
    ):
        raise RuntimeError(f"{node} does not match its KFD/DRM device identity")
    identity = f"linux_device:{major}:{minor}"
    print(
        f"Native CTS GPU: {agent['Uuid']}, BDFID {location}, {node}, {identity}",
        flush=True,
    )
    return identity


def print_proc_file(path: Path) -> str:
    try:
        with path.open() as source:
            text = source.read(8193)
        print(f"{path}:\n{text[:8192].rstrip()}", flush=True)
        if len(text) > 8192:
            print("File snapshot truncated after 8192 characters.", flush=True)
        return text
    except OSError as error:
        # Containers may deny stack access, and a thread may exit mid-snapshot.
        print(f"{path}: unavailable: {error}", flush=True)
        return ""


def diagnose_process(process_id: int) -> None:
    """Reads procfs without creating another client of the stalled GPU driver."""
    print(f"Kernel: {os.uname().release}", flush=True)
    pending = [process_id]
    visited = set()
    while pending and len(visited) < 64:
        current_id = pending.pop()
        if current_id in visited:
            continue
        visited.add(current_id)
        process_path = Path(f"/proc/{current_id}")
        print_proc_file(process_path / "status")
        try:
            tasks = sorted((process_path / "task").iterdir())
        except OSError as error:
            print(f"{process_path}/task: unavailable: {error}", flush=True)
            continue
        for task in tasks[:256]:
            print_proc_file(task / "comm")
            print_proc_file(task / "wchan")
            print_proc_file(task / "stack")
            children = print_proc_file(task / "children")
            pending.extend(int(child) for child in children.split())
        if len(tasks) > 256:
            print("Thread snapshot truncated after 256 threads.", flush=True)
    if pending:
        print("Process snapshot truncated after 64 processes.", flush=True)

    print("Recent kernel warnings and errors (last 200 lines):", flush=True)
    try:
        result = subprocess.run(
            ["dmesg", "--level=err,warn", "--since=-2min"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        print("\n".join(result.stdout.splitlines()[-200:]), flush=True)
        if result.returncode:
            print(
                f"Kernel log unavailable (dmesg exit {result.returncode}).", flush=True
            )
    except OSError as error:
        print(f"Kernel log unavailable: {error}", flush=True)


def require_gnu_timeout() -> str:
    """Finds the supervisor whose process-group and kill-grace semantics we use."""
    observed = []
    for name in ("timeout", "gnutimeout"):
        path = shutil.which(name)
        if path is None:
            continue
        result = subprocess.run(
            [path, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.startswith(
            "timeout (GNU coreutils)"
        ):
            return path
        observed.append(f"{path} (exit {result.returncode}): {result.stdout.strip()}")
    detail = "; ".join(observed) or "neither command was found on PATH"
    raise RuntimeError(
        "GNU coreutils timeout is required as timeout or gnutimeout; " + detail
    )


def run_probe(
    command: list[str],
    *,
    timeout_tool: str,
    timeout_seconds: float = 30,
    kill_after_seconds: float = 5,
    diagnostic_delay_seconds: float = 25,
    stdout: TextIO | None = None,
) -> int:
    # GNU timeout remains the independent supervisor. Diagnostics cannot delay
    # its TERM/KILL deadlines, even when reading kernel state stalls as well.
    process = subprocess.Popen(
        [
            timeout_tool,
            f"--kill-after={kill_after_seconds}s",
            f"{timeout_seconds}s",
            *command,
        ],
        start_new_session=True,
        stdout=stdout,
    )
    try:
        try:
            process.wait(timeout=diagnostic_delay_seconds)
        except subprocess.TimeoutExpired:
            print(
                f"ROCm probe still running after {diagnostic_delay_seconds:g}s; "
                "capturing process state before termination.",
                flush=True,
            )
            result = subprocess.run(
                [
                    timeout_tool,
                    "--kill-after=1s",
                    "3s",
                    sys.executable,
                    "-u",
                    __file__,
                    "--diagnose",
                    str(process.pid),
                ],
                start_new_session=True,
                check=False,
            )
            if result.returncode:
                print(
                    f"Process diagnostics incomplete (exit {result.returncode}).",
                    flush=True,
                )
        returncode = process.wait()
        return returncode if returncode >= 0 else 128 - returncode
    finally:
        if process.poll() is None:
            # Cancellation must reach both the supervisor and its native child.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                # The process group exited between poll and killpg.
                process.wait()
            try:
                process.wait(timeout=kill_after_seconds)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    # The supervisor's own kill deadline may have fired first.
                    pass
                process.wait()


def handle_termination(signum: int, frame: object) -> None:
    raise SystemExit(128 + signum)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose", type=int, metavar="PID")
    parser.add_argument("--export-libamdf-gpu-selection", action="store_true")
    args = parser.parse_args()
    if args.diagnose is not None:
        diagnose_process(args.diagnose)
        return 0

    if not shutil.which("rocminfo"):
        print("::error::Required ROCm preflight tool rocminfo was not found.")
        return 1
    try:
        timeout_tool = require_gnu_timeout()
    except (OSError, RuntimeError) as error:
        print(f"::error::{error}")
        return 1
    runner_name = os.environ.get("RUNNER_NAME", "unknown")
    print(f"rocminfo path: {shutil.which('rocminfo')}", flush=True)
    print(f"GNU timeout path: {timeout_tool}", flush=True)
    print(f"Checking ROCm hardware on runner {runner_name} (timeout: 30s).", flush=True)
    signal.signal(signal.SIGTERM, handle_termination)
    try:
        if args.export_libamdf_gpu_selection:
            with tempfile.TemporaryFile(mode="w+") as output_file:
                status = run_probe(
                    ["rocminfo"], timeout_tool=timeout_tool, stdout=output_file
                )
                output_file.seek(0)
                output = output_file.read()
            print(output, end="", flush=True)
            if status == 0:
                identity = select_native_gpu(
                    output,
                    Path("/sys/class/kfd/kfd/topology/nodes"),
                    Path("/sys/class/drm"),
                    Path("/dev/dri"),
                )
                previous = os.environ.get("AMDF_CTS_GPU_NATIVE_IDENTITY")
                if previous is not None and previous != identity:
                    raise RuntimeError(
                        f"AMDF_CTS_GPU_NATIVE_IDENTITY={previous!r} conflicts with "
                        f"the ROCr allocation {identity}"
                    )
                if github_env := os.environ.get("GITHUB_ENV"):
                    with Path(github_env).open("a") as environment_file:
                        environment_file.write(
                            f"AMDF_CTS_GPU_NATIVE_IDENTITY={identity}\n"
                        )
        else:
            status = run_probe(["rocminfo"], timeout_tool=timeout_tool)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"::error::ROCm native GPU selection failed: {error}", flush=True)
        return 1
    if status == 124:
        print(f"::error::ROCm hardware probe exceeded 30s on runner {runner_name}.")
    elif status == 137:
        print(f"::error::ROCm hardware probe required SIGKILL on runner {runner_name}.")
    elif status:
        print(
            f"::error::ROCm hardware probe failed with exit code {status} on "
            f"runner {runner_name} before GPU tests."
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
