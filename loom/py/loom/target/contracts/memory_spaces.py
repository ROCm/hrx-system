# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-independent memory-space vocabulary for target contracts."""

MEMORY_SPACE_NAMES = frozenset(
    (
        "unknown",
        "global",
        "workgroup",
        "private",
        "constant",
        "host",
        "descriptor",
        "generic",
    )
)
