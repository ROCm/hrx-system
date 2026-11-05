# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Exhaustive AMDGPU model catalog, lifecycle, connection, and request churn."""

from loom.src.loom.serve.server.catalog_test import run_from_command_line

if __name__ == "__main__":
    run_from_command_line(stress=True)
