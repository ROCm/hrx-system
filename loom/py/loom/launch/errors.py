# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Launcher failures that carry an operator-actionable message."""

from __future__ import annotations


class LauncherError(RuntimeError):
    """A launcher operation cannot continue."""


class ProfileError(LauncherError):
    """A published profile is missing or malformed."""


class InstallError(LauncherError):
    """A profile could not be installed locally."""


class LaunchError(LauncherError):
    """A server could not be started or reached."""


class BenchmarkError(LauncherError):
    """A benchmark run or its record could not be completed."""
