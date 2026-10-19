# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Complete physical-pressure matrix for source-JIT HTTP model ownership."""

import itertools

from loom.src.loom.serve.server.pressure_test import run_from_command_line

_CASES = tuple(
    itertools.product(
        (1, 4),
        (2, 1),
        ("none", "checkpoint", "parameters"),
    )
)


if __name__ == "__main__":
    run_from_command_line(_CASES)
