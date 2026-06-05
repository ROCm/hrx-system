# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared libFuzzer developer-tool helpers."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Mapping

FUZZ_CACHE_DIR_ENV = "IREE_FUZZ_CACHE"
FUZZ_CACHE_DIR_NAME = "iree-fuzz-cache"
WINDOWS_LOCAL_APPDATA_ENV = "LOCALAPPDATA"
WINDOWS_APPDATA_ENV = "APPDATA"
XDG_CACHE_HOME_ENV = "XDG_CACHE_HOME"


def default_fuzz_cache_dir(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    home: Path | None = None,
) -> Path:
    environ = os.environ if environ is None else environ
    configured_cache_dir = environ.get(FUZZ_CACHE_DIR_ENV)
    if configured_cache_dir:
        return Path(configured_cache_dir).expanduser()

    platform = sys.platform if platform is None else platform
    home = Path.home() if home is None else home
    if platform.startswith("win"):
        cache_root = environ.get(WINDOWS_LOCAL_APPDATA_ENV) or environ.get(
            WINDOWS_APPDATA_ENV
        )
        if cache_root:
            return Path(cache_root) / FUZZ_CACHE_DIR_NAME
        return home / "AppData" / "Local" / FUZZ_CACHE_DIR_NAME
    if platform == "darwin":
        return home / "Library" / "Caches" / FUZZ_CACHE_DIR_NAME

    cache_root = environ.get(XDG_CACHE_HOME_ENV)
    if cache_root:
        return Path(cache_root).expanduser() / FUZZ_CACHE_DIR_NAME
    return home / ".cache" / FUZZ_CACHE_DIR_NAME


FUZZ_CACHE_DIR = default_fuzz_cache_dir()


def bazel_fuzz_target_dir(target: str) -> Path:
    package, target_name = split_bazel_target_label(target)
    return FUZZ_CACHE_DIR.expanduser() / package / target_name


def cmake_fuzz_target_dir(target: str) -> Path:
    return FUZZ_CACHE_DIR.expanduser() / "cmake" / sanitized_identifier_path(target)


def split_bazel_target_label(target: str) -> tuple[str, str]:
    label = target
    if label.startswith("@"):
        label = label.split("//", 1)[-1]
    elif label.startswith("//"):
        label = label[2:]
    if ":" in label:
        package, target_name = label.split(":", 1)
    else:
        package = label
        target_name = Path(package).name
    return package, target_name


def sanitized_identifier_path(identifier: str) -> Path:
    parts = [part for part in identifier.split("::") if part]
    if not parts:
        parts = [identifier]
    return Path(*(sanitize_path_part(part) for part in parts))


def sanitize_path_part(part: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._+-]+", "_", part).strip("._")
    return sanitized or "target"
