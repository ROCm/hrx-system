# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Developer tool environment helpers."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVTOOLS_TMP_ENV = "IREE_DEVTOOLS_TMP"
DEFAULT_LOCAL_TMP_ROOT = REPO_ROOT / ".tmp"


class ToolMode(Enum):
    VENV = "venv"
    SYSTEM = "system"
    TOOL_ROOT = "tool-root"


@dataclass(frozen=True)
class ToolEnvironment:
    mode: ToolMode
    root: Path | None

    @property
    def bin_dir(self) -> Path | None:
        if self.root is None:
            return None
        return venv_bin_dir(self.root)

    @property
    def python(self) -> str:
        if self.bin_dir is None:
            return sys.executable
        if os.name == "nt":
            return str(self.bin_dir / "python.exe")
        return str(self.bin_dir / "python")

    def tool(self, name: str) -> str:
        if self.bin_dir is None:
            return name
        candidate = self.bin_dir / executable_name(name)
        if candidate.is_file():
            return str(candidate)
        script_candidate = self.bin_dir / script_name(name)
        if script_candidate.is_file():
            return str(script_candidate)
        return name

    def path_env(self, base_env: dict[str, str] | None = None) -> dict[str, str]:
        env = dict(os.environ if base_env is None else base_env)
        if self.bin_dir is None:
            return env
        current_path = env.get("PATH", "")
        env["PATH"] = os.pathsep.join([str(self.bin_dir), current_path])
        return env


def local_tmp_root(environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    value = environ.get(DEVTOOLS_TMP_ENV)
    if not value:
        return DEFAULT_LOCAL_TMP_ROOT
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


LOCAL_TMP_ROOT = local_tmp_root()


def executable_name(name: str) -> str:
    if os.name == "nt" and not name.lower().endswith((".exe", ".cmd", ".bat")):
        return name + ".exe"
    return name


def script_name(name: str) -> str:
    if os.name == "nt" and not name.lower().endswith((".cmd", ".ps1")):
        return name + ".cmd"
    return name


def default_venv_root() -> Path:
    return REPO_ROOT / ".venv"


def venv_bin_dir(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts"
    return venv_root / "bin"


def tool_environment_from_args(args) -> ToolEnvironment:
    if getattr(args, "system", False):
        return ToolEnvironment(ToolMode.SYSTEM, None)
    tool_root = getattr(args, "tool_root", None)
    if tool_root:
        return ToolEnvironment(ToolMode.TOOL_ROOT, Path(tool_root))
    return ToolEnvironment(ToolMode.VENV, default_venv_root())


def existing_or_system_environment(args) -> ToolEnvironment:
    if getattr(args, "system", False):
        return ToolEnvironment(ToolMode.SYSTEM, None)
    tool_root = getattr(args, "tool_root", None)
    if tool_root:
        return ToolEnvironment(ToolMode.TOOL_ROOT, Path(tool_root))
    default_root = default_venv_root()
    if default_root.is_dir():
        return ToolEnvironment(ToolMode.VENV, default_root)
    return ToolEnvironment(ToolMode.SYSTEM, None)
