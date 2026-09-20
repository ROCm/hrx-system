#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exercises the lock-free Bazel launcher against a real Bazel server."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_PY = REPO_ROOT / "dev.py"
FIXTURE_TARGET = "//build_tools/devtools:bazel_launcher_integration_fixture"

# This integration harness also runs directly from a source checkout in CI.
sys.path.insert(0, str(REPO_ROOT))
from build_tools.devtools import bazel as bazel_dev  # noqa: E402


def dev_command(*args: str) -> list[str]:
    executable = Path(sys.executable)
    if os.name == "nt" and sys.prefix != sys.base_prefix:
        # A Windows venv executable is a redirector that waits on the real
        # interpreter. Bypass it so process.pid below identifies the dev.py
        # process whose direct-child contract the fixture verifies.
        executable = Path(sys.base_prefix) / executable.name
        if not executable.is_file():
            raise RuntimeError(f"base Python executable does not exist: {executable}")
    return [str(executable), str(DEV_PY), *args]


def process_output(process: subprocess.Popen[bytes], output_path: Path) -> str:
    process.wait()
    return output_path.read_text(encoding="utf-8", errors="replace")


def wait_until_ready(
    process: subprocess.Popen[bytes], ready_file: Path, output_path: Path
) -> None:
    while not ready_file.is_file():
        returncode = process.poll()
        if returncode is not None:
            output = output_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(
                f"launcher fixture exited {returncode} before becoming ready:\n{output}"
            )
        time.sleep(0.05)


def verify_lock_free_launch(temporary_root: Path) -> None:
    caller_cwd = temporary_root / "caller directory"
    caller_cwd.mkdir()
    ready_file = caller_cwd / "ready"
    result_file = caller_cwd / "result.txt"
    absolute_argument = temporary_root / "absolute argument"
    output_path = temporary_root / "launch.log"
    with output_path.open("wb") as output_file:
        process = subprocess.Popen(
            dev_command(
                "bazel",
                "run",
                "--compilation_mode=dbg",
                FIXTURE_TARGET,
                "--",
                "--ready-file",
                ready_file.name,
                "--result-file",
                result_file.name,
                "two words",
                "--literal=three words",
                str(absolute_argument),
            ),
            cwd=caller_cwd,
            stdin=subprocess.PIPE,
            stdout=output_file,
            stderr=subprocess.STDOUT,
        )
        wait_until_ready(process, ready_file, output_path)

        workspace = subprocess.run(
            dev_command("bazel", "info", "workspace"),
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if workspace.returncode != 0:
            if process.stdin is not None:
                process.stdin.write(b"\n")
                process.stdin.close()
            output = process_output(process, output_path)
            raise RuntimeError(
                "second Bazel command failed while launched target was live:\n"
                + workspace.stdout
                + "\nlauncher output:\n"
                + output
            )

        if process.stdin is None:
            raise RuntimeError("launcher fixture has no release channel")
        process.stdin.write(b"\n")
        process.stdin.close()
        returncode = process.wait()

    if returncode != 0:
        raise RuntimeError(
            f"launcher fixture exited {returncode}:\n"
            + output_path.read_text(encoding="utf-8", errors="replace")
        )
    result_lines = result_file.read_text(encoding="utf-8").splitlines()
    if len(result_lines) < 4:
        raise RuntimeError(f"invalid launcher fixture result: {result_lines!r}")
    process_id = int(result_lines[0])
    parent_process_id = int(result_lines[1])
    process_cwd = result_lines[2]
    argument_count = int(result_lines[3])
    arguments = result_lines[4:]
    if len(arguments) != argument_count:
        raise RuntimeError(f"launcher fixture argument count mismatch: {arguments!r}")
    if Path(process_cwd).resolve() != caller_cwd.resolve():
        raise RuntimeError(
            f"launcher cwd mismatch: {process_cwd!r} != {str(caller_cwd)!r}"
        )
    expected_arguments = [
        "two words",
        "--literal=three words",
        str(absolute_argument),
    ]
    if arguments != expected_arguments:
        raise RuntimeError(
            f"launcher arguments mismatch: {arguments!r} != {expected_arguments!r}"
        )
    if os.name == "nt":
        if parent_process_id != process.pid:
            raise RuntimeError(
                "Windows launcher left an intermediary process between dev.py "
                "and the target"
            )
    elif process_id != process.pid:
        raise RuntimeError("POSIX launcher did not overlay dev.py with the target")


def verify_exit_code(temporary_root: Path) -> None:
    caller_cwd = temporary_root / "exit caller"
    caller_cwd.mkdir()
    result = subprocess.run(
        dev_command(
            "bazel",
            "run",
            "--compilation_mode=dbg",
            FIXTURE_TARGET,
            "--",
            "--exit-code",
            "37",
        ),
        cwd=caller_cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 37:
        raise RuntimeError(
            f"launcher returned {result.returncode} instead of 37:\n{result.stdout}"
        )


def verify_try_dependency_aliases(temporary_root: Path) -> None:
    source = temporary_root / "dependency_aliases.c"
    source.write_text(
        '#include "iree/base/api.h"\n'
        "int main(void) {\n"
        "  return iree_status_is_ok(iree_ok_status()) ? 0 : 1;\n"
        "}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        dev_command(
            "bazel",
            "try",
            "--config=asan",
            "--dep",
            "//runtime/src/iree/base",
            "--dep",
            "//runtime/src/iree/base:base",
            "--dep",
            "//runtime/src/iree/base",
            str(source),
        ),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Bazel try failed with explicit and inferred dependency aliases:\n"
            + result.stdout
        )


def try_packages(process_id: int) -> list[Path]:
    pattern = f"run-{process_id}-*"
    paths = list((REPO_ROOT / ".iree/bazel-try").glob(pattern))
    output_root = (REPO_ROOT / "bazel-out").resolve()
    for kind in ("bin", "genfiles", "testlogs"):
        paths.extend(output_root.glob(f"*/{kind}/.iree/bazel-try/{pattern}"))
    return paths


def verify_try_lifecycle(temporary_root: Path) -> None:
    def run_probe(*arguments: str, expected: int = 0, keep: bool = False) -> None:
        process = subprocess.Popen(
            dev_command("bazel", "try", "--config=asan", *arguments),
            cwd=temporary_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output, _ = process.communicate()
        if process.returncode != expected:
            raise RuntimeError(
                f"try exited {process.returncode}, expected {expected}:\n{output}"
            )
        packages = try_packages(process.pid)
        if keep:
            try:
                if not any((package / "BUILD.bazel").exists() for package in packages):
                    raise RuntimeError("--keep lost the source package")
                if not any(
                    (package / "snippet").exists() or (package / "snippet.exe").exists()
                    for package in packages
                ):
                    raise RuntimeError("--keep lost the executable")
            finally:
                for package in packages:
                    shutil.rmtree(package, onexc=bazel_dev.remove_readonly_try_file)
        elif packages:
            raise RuntimeError(f"try leaked source or output packages: {packages}")

    run_probe("-e", "int main(void) { return 37; }", expected=37)
    copied = temporary_root / ("copied.exe" if os.name == "nt" else "copied")
    run_probe(
        "--compile-only", "--output", str(copied), "-e", "int main(void) { return 0; }"
    )
    subprocess.run([str(copied)], check=True)
    run_probe("--compile-only", "-e", "#error deliberate compile failure", expected=1)
    run_probe("--keep", "-e", "int main(void) { return 0; }", keep=True)

    # Keep one probe live while another builds and cleans its own package.
    # The pipe is the readiness/release contract, without wall-clock deadlines.
    source = '#include <stdio.h>\nint main(void) { puts("ready"); fflush(stdout); return getchar() == EOF; }'
    process = subprocess.Popen(
        dev_command("bazel", "try", "--config=asan", "-e", source),
        cwd=temporary_root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        if process.stdout.readline().strip() != "ready":
            _, error = process.communicate()
            raise RuntimeError(f"try did not reach readiness: {error}")
        packages = try_packages(process.pid)
        if not any(
            (package / "snippet").is_file() or (package / "snippet.exe").is_file()
            for package in packages
        ):
            raise RuntimeError("live try lost its executable package")
        run_probe("-e", "int main(void) { return 0; }")
        if not all(package.exists() for package in packages):
            raise RuntimeError("concurrent try removed a live probe's package")
        if os.name != "nt":
            process.send_signal(signal.SIGTERM)
            expected = 128 + signal.SIGTERM
            output, error = process.communicate()
        else:
            expected = 0
            output, error = process.communicate("\n")
        if process.returncode != expected:
            raise RuntimeError(
                f"try termination returned {process.returncode}: {output} {error}"
            )
        if try_packages(process.pid):
            raise RuntimeError("try leaked packages after termination")
    finally:
        if process.poll() is None:
            process.communicate("\n")


def main() -> int:
    # The debug fixture consumes an artifact made by its own host-tool version.
    # Both versions are in its dependency closure when launch metadata is read.
    result = subprocess.run(
        dev_command(
            "bazel",
            "build",
            "--compilation_mode=dbg",
            FIXTURE_TARGET,
        ),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"dual-configuration fixture build failed:\n{result.stdout}")
    with tempfile.TemporaryDirectory(prefix="iree-bazel-launch-") as temporary_name:
        temporary_root = Path(temporary_name)
        verify_lock_free_launch(temporary_root)
        verify_exit_code(temporary_root)
        verify_try_dependency_aliases(temporary_root)
        verify_try_lifecycle(temporary_root)
    print("Bazel launcher integration passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
