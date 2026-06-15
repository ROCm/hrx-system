# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Default TileLang converter registration."""

from __future__ import annotations

from loom.importers.tilelang.converter import TileLangConverterRegistry
from loom.importers.tilelang.ops import (
    calls,
    control,
    memory,
    scalar,
    structure,
    vector,
)


def build_default_registry() -> TileLangConverterRegistry:
    """Builds the production TileLang/TIR converter registry."""

    registry = TileLangConverterRegistry()
    structure.register(registry)
    control.register(registry)
    memory.register(registry)
    scalar.register(registry)
    vector.register(registry)
    calls.register(registry)
    return registry
