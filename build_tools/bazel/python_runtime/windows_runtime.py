# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Initializes paths owned by the compact Windows Python runtime."""

from __future__ import annotations

import ntpath
import os

_EXTENDED_PATH_PREFIX = "\\\\?\\"
_EXTENDED_UNC_PATH_PREFIX = "\\\\?\\UNC\\"


def _ordinary_windows_path(path: str) -> str:
    """Removes a Windows extended-length prefix without changing the target."""

    path = ntpath.normpath(path)
    folded_path = path.casefold()
    if folded_path.startswith(_EXTENDED_UNC_PATH_PREFIX.casefold()):
        return "\\\\" + path[len(_EXTENDED_UNC_PATH_PREFIX) :]
    if folded_path.startswith(_EXTENDED_PATH_PREFIX.casefold()):
        return path[len(_EXTENDED_PATH_PREFIX) :]
    return path


def _configure_runtime(module_file: str, system, environment) -> None:
    """Binds interpreter and Tcl paths to the archive containing this module."""

    archive_path = ntpath.dirname(module_file)
    runtime_root = _ordinary_windows_path(
        ntpath.dirname(os.path.realpath(archive_path))
    )
    system.base_prefix = runtime_root
    system.base_exec_prefix = runtime_root
    environment["TCL_LIBRARY"] = ntpath.join(runtime_root, "tcl", "tcl8.6")
    environment["TK_LIBRARY"] = ntpath.join(runtime_root, "tcl", "tk8.6")
