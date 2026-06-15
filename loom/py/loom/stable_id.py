# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Stable integer identities derived from textual keys."""

from __future__ import annotations

STABLE_ID_NONE = 0


def stable_id_from_string(key: str) -> int:
    """Returns a stable non-zero 63-bit identity for ``key``."""

    value = 0xCBF29CE484222325
    for byte in key.encode("utf-8"):
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    value &= 0x7FFFFFFFFFFFFFFF
    return 1 if value == STABLE_ID_NONE else value
