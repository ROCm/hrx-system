# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Target-owned source value materializer callbacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValueMaterializer:
    """Callback pair referenced by materialized source value refs."""

    name: str
    can_materialize: str
    materialize: str
    header: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("value materializer name must be non-empty")
        if not self.can_materialize:
            raise ValueError(f"value materializer '{self.name}' needs a predicate")
        if not self.materialize:
            raise ValueError(f"value materializer '{self.name}' needs an emitter")
        if not self.header:
            raise ValueError(f"value materializer '{self.name}' needs a C header")
