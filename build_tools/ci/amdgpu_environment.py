#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Checks GPU admission and captures stalled native processes before termination."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import TextIO

from build_tools.amdgpu import target_map
from build_tools.ci import gpu_assignment

AMDGPU_FENCE_FALLBACK_PATTERN = re.compile(
    r"\bamdgpu\b.*Fence fallback timer expired on ring (?P<ring>\S+)"
)


def parse_amdgpu_fence_fallbacks(output: str) -> list[tuple[str, str]]:
    """Returns ring names and complete kernel lines for fence fallbacks."""
    fallbacks = []
    for line in output.splitlines():
        if match := AMDGPU_FENCE_FALLBACK_PATTERN.search(line):
            fallbacks.append((match.group("ring"), line))
    return fallbacks


def find_amdgpu_interrupt_failures(
    baseline: str, current: str | None = None
) -> list[str]:
    """Finds systemic boot or newly observed AMDGPU interrupt failures."""
    baseline_fallbacks = parse_amdgpu_fence_fallbacks(baseline)
    engine_kinds = {
        match.group(0)
        for ring, _ in baseline_fallbacks
        if (match := re.match(r"[a-z]+", ring))
    }
    # One fallback means the timer recovered a fence whose interrupt was lost.
    # A boot sweep spanning command, compute, and transfer engines instead
    # establishes that interrupt delivery is broken across the GPU.
    if {"gfx", "comp", "sdma"}.issubset(engine_kinds):
        return [line for _, line in baseline_fallbacks]
    if current is None:
        return []

    # During the admission probe even one new fallback is an active failure.
    # Compare as a multiset because the kernel may repeat identical messages.
    previous_lines = collections.Counter(line for _, line in baseline_fallbacks)
    failures = []
    for _, line in parse_amdgpu_fence_fallbacks(current):
        if previous_lines[line]:
            previous_lines[line] -= 1
        else:
            failures.append(line)
    return failures


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


def read_kernel_warnings(timeout_tool: str) -> str:
    """Reads the kernel warning/error log under an independent deadline."""
    # Keep the diagnostic delay beyond dmesg's full deadline: collecting
    # diagnostics for a stalled kernel log reader would read the log again.
    return probe_output(
        ["dmesg", "--level=err,warn"],
        timeout_tool=timeout_tool,
        timeout_seconds=5,
        kill_after_seconds=1,
        diagnostic_delay_seconds=7,
    )


def require_healthy_amdgpu_interrupts(
    baseline: str, current: str | None = None
) -> None:
    """Rejects a runner with systemic or active fence-interrupt failures."""
    failures = find_amdgpu_interrupt_failures(baseline, current)
    if not failures:
        return
    detail = "\n".join(failures[-20:])
    raise RuntimeError(
        "AMDGPU fence interrupts are not healthy; refusing native GPU "
        f"admission. Kernel evidence (last {min(len(failures), 20)}):\n{detail}"
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
                f"{Path(command[0]).name} still running after {diagnostic_delay_seconds:g}s; "
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
                    "-m",
                    "build_tools.ci.amdgpu_environment",
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


def probe_output(
    command: list[str],
    *,
    timeout_tool: str,
    accepted_statuses: tuple[int, ...] = (0,),
    **timing: float,
) -> str:
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stream:
        status = run_probe(command, timeout_tool=timeout_tool, stdout=stream, **timing)
        stream.seek(0)
        output = stream.read()
    if status not in accepted_statuses:
        print(output, end="", flush=True)
        print(f"::error::{command} exited {status}.", flush=True)
        raise subprocess.CalledProcessError(status, command)
    return output


def admit_native_gpu(executable: str, target: str, timeout_tool: str) -> str:
    probe = partial(probe_output, timeout_tool=timeout_tool)
    processors = {
        processor
        for selector in re.split("[,;]", target)
        for processor in target_map.exact_targets_for_selection(selector.strip())
    }
    nodes = gpu_assignment.assigned_render_nodes()
    for path, number in nodes.items():
        report = probe(
            [executable, "select", f"--device={path}", "--engine=gpu", "--format=json"],
            # amdf-info exits 3 with an empty selection for an unsupported
            # render node. Other members of a mixed grant may be supported.
            accepted_statuses=(0, 3),
        )
        endpoints = json.loads(report)["endpoints"]
        if not endpoints:
            print(f"libamdf reports no GPU endpoint at {path}", flush=True)
            continue
        (endpoint,) = endpoints
        native = endpoint["native_identity"]
        if (native["major"], native["minor"]) != (os.major(number), os.minor(number)):
            raise ValueError(
                f"libamdf selected a different render node for {path}: {native}"
            )
        if endpoint["gpu"]["target"] in processors:
            break
    else:
        raise ValueError(f"No assigned GPU in {list(nodes)} matches {target}")
    identity = gpu_assignment.native_identity(number)
    uri = endpoint["uri"]
    processor = endpoint["gpu"]["target"]
    probe(
        [executable, "select", f"--device={uri}", f"--target={processor}", "--activate"]
    )
    print(f"Activated and released {uri}: {processor}, {identity}", flush=True)
    return identity


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--diagnose", type=int, metavar="PID")
    action.add_argument(
        "--native",
        nargs=2,
        metavar=("AMDF_INFO", "TARGET"),
        help="Check the assigned GPU with the specified libamdf tool and target.",
    )
    args = parser.parse_args(argv)
    if args.diagnose is not None:
        diagnose_process(args.diagnose)
        return 0

    runner_name = os.environ.get("RUNNER_NAME", "unknown")
    signal.signal(signal.SIGTERM, handle_termination)
    try:
        timeout_tool = require_gnu_timeout()
        print(f"Checking GPU on {runner_name} (30s per native probe).", flush=True)
        baseline = read_kernel_warnings(timeout_tool)
        require_healthy_amdgpu_interrupts(baseline)
        try:
            identity = admit_native_gpu(*args.native, timeout_tool)
        finally:
            require_healthy_amdgpu_interrupts(
                baseline, read_kernel_warnings(timeout_tool)
            )
        if github_env := os.environ.get("GITHUB_ENV"):
            with Path(github_env).open("a") as stream:
                stream.write(
                    f"HAL_AMD_GPU_ASSIGNMENT={identity}\nAMDF_CTS_GPU_NATIVE_IDENTITY={identity}\n"
                )
    except subprocess.CalledProcessError as error:
        return error.returncode
    except KeyboardInterrupt:
        return 130
    except (OSError, TypeError, ValueError, KeyError, RuntimeError) as error:
        print(f"::error::GPU preflight failed: {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
