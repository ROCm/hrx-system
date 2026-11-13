# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Runs the launcher as `python -m loom.launch`."""

from __future__ import annotations

import sys

from loom.launch.cli import main

if __name__ == "__main__":
    sys.exit(main())
