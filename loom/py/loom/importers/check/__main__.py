# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Executable entry point for importer checks."""

from __future__ import annotations

from loom.importers.check.main import main

if __name__ == "__main__":
    raise SystemExit(main())
