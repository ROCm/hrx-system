# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Developer tool environment helpers."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVTOOLS_TMP_ENV = "IREE_DEVTOOLS_TMP"
DEFAULT_LOCAL_TMP_ROOT = REPO_ROOT / ".tmp"
BAZEL_SH_ENV = "BAZEL_SH"


def _read_windows_short_path(path: str) -> str | None:
    """Returns the Win32 short spelling of an existing path when available."""
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_short_path_name = kernel32.GetShortPathNameW
    get_short_path_name.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
    ]
    get_short_path_name.restype = ctypes.c_uint32
    required_length = get_short_path_name(path, None, 0)
    if required_length == 0:
        return None
    buffer = ctypes.create_unicode_buffer(required_length)
    written_length = get_short_path_name(path, buffer, required_length)
    if written_length == 0 or written_length >= required_length:
        return None
    return buffer.value


def _create_windows_directory_junction(link: Path, target: Path) -> None:
    """Creates a Windows directory junction without requiring symlink rights."""
    command_interpreter = os.environ.get("COMSPEC", "cmd.exe")
    completed = subprocess.run(
        [command_interpreter, "/d", "/c", "mklink", "/J", str(link), str(target)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stdout.strip()
        if detail:
            detail = f": {detail}"
        raise OSError(
            f"unable to create Windows directory junction {link} -> {target}{detail}"
        )


def _windows_shell_install_root(shell_path: Path) -> Path:
    """Returns the installation root whose relative layout the shell needs."""
    shell_directory = shell_path.parent
    if shell_directory.name.casefold() != "bin":
        return shell_directory
    installation_root = shell_directory.parent
    if installation_root.name.casefold() == "usr":
        installation_root = installation_root.parent
    return installation_root


def _same_file(first: Path, second: Path) -> bool:
    try:
        return first.samefile(second)
    except OSError:
        return False


def bazel_compatible_windows_shell_path(
    path: str,
    *,
    platform_name: str | None = None,
    short_path_reader: Callable[[str], str | None] | None = None,
    alias_root: Path | None = None,
    junction_creator: Callable[[Path, Path], None] | None = None,
) -> str:
    """Returns a shell path safe for Bazel's generated Windows batch files."""
    platform_name = os.name if platform_name is None else platform_name
    if platform_name != "nt" or not any(character.isspace() for character in path):
        return path
    if short_path_reader is None:
        short_path_reader = _read_windows_short_path
    short_path = short_path_reader(path)
    if short_path and not any(character.isspace() for character in short_path):
        return short_path

    shell_path = Path(path).resolve()
    install_root = _windows_shell_install_root(shell_path)
    relative_shell_path = shell_path.relative_to(install_root)
    if alias_root is None:
        alias_root = local_tmp_root() / "iree-windows-shell"
    if any(character.isspace() for character in str(alias_root)):
        raise RuntimeError(
            f"{DEVTOOLS_TMP_ENV} must name a whitespace-free path when the "
            "Windows shell installation path contains whitespace"
        )
    alias_root.mkdir(parents=True, exist_ok=True)
    install_key = str(install_root.resolve()).casefold().encode("utf-8")
    alias_path = alias_root / hashlib.sha256(install_key).hexdigest()[:16]
    alias_shell_path = alias_path / relative_shell_path
    if alias_shell_path.is_file():
        if _same_file(alias_shell_path, shell_path):
            return str(alias_shell_path)
        raise RuntimeError(
            f"Windows shell alias {alias_shell_path} resolves to a different file"
        )
    if alias_path.exists():
        raise RuntimeError(
            f"Windows shell alias path exists without the expected shell: {alias_path}"
        )
    if junction_creator is None:
        junction_creator = _create_windows_directory_junction
    try:
        junction_creator(alias_path, install_root)
    except OSError:
        # Another concurrent dev.py invocation may have created the same
        # installation-keyed alias after the checks above.
        if not alias_shell_path.is_file() or not _same_file(
            alias_shell_path, shell_path
        ):
            raise
    if not alias_shell_path.is_file() or not _same_file(alias_shell_path, shell_path):
        raise RuntimeError(
            f"Windows shell alias did not resolve to {shell_path}: {alias_shell_path}"
        )
    return str(alias_shell_path)


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
        if self.bin_dir is not None:
            current_path = env.get("PATH", "")
            env["PATH"] = os.pathsep.join([str(self.bin_dir), current_path])
        bazel_sh = find_windows_bazel_sh(env)
        if bazel_sh:
            env[BAZEL_SH_ENV] = bazel_sh
        return env


def find_windows_bazel_sh(
    environ: Mapping[str, str] | None = None,
    *,
    platform_name: str | None = None,
    git_executable: str | None = None,
) -> str | None:
    environ = os.environ if environ is None else environ
    platform_name = os.name if platform_name is None else platform_name
    configured_path = environ.get(BAZEL_SH_ENV)
    if configured_path:
        return bazel_compatible_windows_shell_path(
            configured_path,
            platform_name=platform_name,
            alias_root=local_tmp_root(environ) / "iree-windows-shell",
        )
    if platform_name != "nt":
        return None

    if git_executable is None:
        git_executable = shutil.which("git", path=environ.get("PATH"))

    candidates = []
    if git_executable:
        git_path = Path(git_executable)
        for parent in list(git_path.parents)[:3]:
            candidates.extend(
                (parent / "bin" / "bash.exe", parent / "usr/bin/bash.exe")
            )

    for key in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData"):
        root = environ.get(key)
        if not root:
            continue
        root_path = Path(root)
        if key == "LocalAppData":
            root_path /= "Programs"
        candidates.extend(
            (
                root_path / "Git/bin/bash.exe",
                root_path / "Git/usr/bin/bash.exe",
            )
        )

    seen = set()
    for candidate in candidates:
        candidate_key = str(candidate).casefold()
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        if candidate.is_file():
            return bazel_compatible_windows_shell_path(
                str(candidate),
                platform_name=platform_name,
                alias_root=local_tmp_root(environ) / "iree-windows-shell",
            )
    return None


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
