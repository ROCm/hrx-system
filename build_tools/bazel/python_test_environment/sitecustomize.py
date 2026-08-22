# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 WITH LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Normalizes the Python home directory inside Bazel tests on Windows."""

from __future__ import annotations

import os


def _configure_windows_test_home() -> None:
    if os.name != "nt" or "USERPROFILE" in os.environ:
        return
    test_tmpdir = os.environ.get("TEST_TMPDIR")
    if test_tmpdir:
        os.environ["USERPROFILE"] = test_tmpdir


_configure_windows_test_home()
