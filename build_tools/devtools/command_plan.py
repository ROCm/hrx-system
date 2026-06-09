# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command planning and execution helpers for dev.py."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


class Step(Protocol):
    def describe(self) -> str: ...

    def run(self, verbose: bool = False) -> int: ...


def quote_command(argv: list[str]) -> str:
    return shlex.join(argv)


@dataclass(frozen=True)
class CommandStep:
    argv: list[str]
    cwd: Path
    env: dict[str, str] | None = None
    label: str | None = None

    def describe(self) -> str:
        pieces = []
        if self.label:
            pieces.append(f"# {self.label}")
        if self.env:
            env_prefix = []
            for key, value in sorted(self.env.items()):
                old_value = os.environ.get(key)
                if old_value == value:
                    continue
                if (
                    key == "PATH"
                    and old_value
                    and value.endswith(os.pathsep + old_value)
                ):
                    prefix = value[: -(len(old_value) + 1)]
                    env_prefix.append(f"PATH={shlex.quote(prefix)}:$PATH")
                else:
                    env_prefix.append(f"{key}={shlex.quote(value)}")
            if env_prefix:
                pieces.append(" ".join(env_prefix))
        command = quote_command(self.argv)
        if self.cwd != Path.cwd():
            pieces.append(f"(cd {shlex.quote(str(self.cwd))} && {command})")
        else:
            pieces.append(command)
        return "\n".join(pieces)

    def run(self, verbose: bool = False) -> int:
        if verbose:
            print(f"dev.py: {self.label or quote_command(self.argv)}")
            print("  " + quote_command(self.argv))
            sys.stdout.flush()
        return subprocess.run(self.argv, cwd=self.cwd, env=self.env).returncode


@dataclass(frozen=True)
class ExecCommandStep:
    argv: list[str]
    cwd: Path
    env: dict[str, str] | None = None
    label: str | None = None

    def describe(self) -> str:
        return CommandStep(
            self.argv,
            cwd=self.cwd,
            env=self.env,
            label=self.label,
        ).describe()

    def run(self, verbose: bool = False) -> int:
        if verbose:
            print(f"dev.py: {self.label or quote_command(self.argv)}")
            print("  " + quote_command(self.argv))
            sys.stdout.flush()
        try:
            os.chdir(self.cwd)
            os.execvpe(self.argv[0], self.argv, self.env or os.environ)
        except OSError as exc:
            print(
                f"dev.py: failed to exec {quote_command(self.argv)}: {exc}",
                file=sys.stderr,
            )
            return 127
        raise AssertionError("os.execvpe returned unexpectedly")


@dataclass(frozen=True)
class CheckCommandStep:
    argv: list[str]
    cwd: Path
    expected_pattern: str | None = None
    env: dict[str, str] | None = None
    label: str | None = None

    def describe(self) -> str:
        command = quote_command(self.argv)
        if self.expected_pattern:
            command += f"  # expect /{self.expected_pattern}/"
        if self.cwd != Path.cwd():
            return f"(cd {shlex.quote(str(self.cwd))} && {command})"
        return command

    def run(self, verbose: bool = False) -> int:
        if verbose:
            print(f"dev.py: {self.label or quote_command(self.argv)}")
            print("  " + quote_command(self.argv))
            sys.stdout.flush()
        result = subprocess.run(
            self.argv,
            cwd=self.cwd,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = result.stdout.rstrip()
        if output:
            print(output)
        if result.returncode != 0:
            return result.returncode
        if self.expected_pattern and not re.search(self.expected_pattern, output):
            print(
                f"dev.py: expected {quote_command(self.argv)} output to match "
                f"/{self.expected_pattern}/",
                file=sys.stderr,
            )
            return 1
        return 0


@dataclass(frozen=True)
class OptionalCheckCommandStep:
    argv: list[str]
    cwd: Path
    expected_pattern: str | None = None
    env: dict[str, str] | None = None
    label: str | None = None

    def describe(self) -> str:
        command = quote_command(self.argv)
        if self.expected_pattern:
            command += f"  # optional, expect /{self.expected_pattern}/"
        else:
            command += "  # optional"
        if self.cwd != Path.cwd():
            return f"(cd {shlex.quote(str(self.cwd))} && {command})"
        return command

    def run(self, verbose: bool = False) -> int:
        if verbose:
            print(f"dev.py: {self.label or quote_command(self.argv)}")
            print("  " + quote_command(self.argv))
            sys.stdout.flush()
        try:
            result = subprocess.run(
                self.argv,
                cwd=self.cwd,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except FileNotFoundError:
            print(
                f"dev.py: warning: optional tool {quote_command(self.argv)} "
                "is not available"
            )
            return 0
        output = result.stdout.rstrip()
        if output:
            print(output)
        if result.returncode != 0:
            print(
                f"dev.py: warning: optional tool {quote_command(self.argv)} "
                f"exited {result.returncode}"
            )
            return 0
        if self.expected_pattern and not re.search(self.expected_pattern, output):
            print(
                f"dev.py: warning: expected {quote_command(self.argv)} output "
                f"to match /{self.expected_pattern}/"
            )
        return 0


@dataclass(frozen=True)
class EnsureDirectoryStep:
    path: Path

    def describe(self) -> str:
        return f"mkdir -p {shlex.quote(str(self.path))}"

    def run(self, verbose: bool = False) -> int:
        self.path.mkdir(parents=True, exist_ok=True)
        return 0


@dataclass(frozen=True)
class WriteFileStep:
    path: Path
    content: str
    executable: bool = False
    label: str | None = None

    def describe(self) -> str:
        suffix = " and mark executable" if self.executable else ""
        if self.label and not self.label.startswith("write "):
            return f"# {self.label}: write {self.path}{suffix}"
        return f"# write {self.path}{suffix}"

    def run(self, verbose: bool = False) -> int:
        if verbose and self.label:
            print(f"dev.py: {self.label}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        previous_content = None
        if self.path.is_file():
            previous_content = self.path.read_text(encoding="utf-8")
        if previous_content != self.content:
            self.path.write_text(self.content, encoding="utf-8")
        if self.executable:
            mode = self.path.stat().st_mode
            self.path.chmod(mode | 0o755)
        return 0


@dataclass(frozen=True)
class CopyFileStep:
    source: Path
    destination: Path
    label: str | None = None

    def describe(self) -> str:
        return (
            f"cp {shlex.quote(str(self.source))} {shlex.quote(str(self.destination))}"
        )

    def run(self, verbose: bool = False) -> int:
        if verbose and self.label:
            print(f"dev.py: {self.label}")
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.source, self.destination)
        return 0


@dataclass
class CommandPlan:
    steps: list[Step] = field(default_factory=list)

    def add(self, step: Step) -> None:
        self.steps.append(step)

    def extend(self, steps: list[Step]) -> None:
        self.steps.extend(steps)

    def describe(self) -> str:
        return "\n".join(step.describe() for step in self.steps)

    def run(self, dry_run: bool = False, verbose: bool = False) -> int:
        if dry_run:
            description = self.describe()
            if description:
                print(description)
            return 0
        for step in self.steps:
            result = step.run(verbose=verbose)
            if result != 0:
                return result
        return 0
