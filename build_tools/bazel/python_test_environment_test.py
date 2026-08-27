# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 WITH LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
from pathlib import Path


def test_windows_home_uses_bazel_test_tmpdir() -> None:
    if os.name != "nt":
        return
    assert Path.home() == Path(os.environ["TEST_TMPDIR"])
