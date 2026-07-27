# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structured MLIR attribute decoding helpers for importers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MlirAttributeDecoder:
    """Small adapter around binding gaps in MLIR Python attributes."""

    def dense_i64_array(self, attr: object | None) -> tuple[int, ...]:
        return self.dense_integer_array(attr, element_type="i64")

    def dense_i32_array(self, attr: object | None) -> tuple[int, ...]:
        return self.dense_integer_array(attr, element_type="i32")

    def dense_integer_array(
        self,
        attr: object | None,
        *,
        element_type: str,
    ) -> tuple[int, ...]:
        if attr is None:
            return ()
        values = getattr(attr, "value", None)
        if isinstance(values, tuple | list):
            return tuple(int(value) for value in values)
        return _parse_dense_integer_array_text(str(attr), element_type=element_type)

    def integer(self, attr: object | None) -> int | None:
        if attr is None:
            return None
        value = getattr(attr, "value", None)
        return int(value) if value is not None else None

    def string(self, attr: object | None) -> str | None:
        if attr is None:
            return None
        value = getattr(attr, "value", None)
        return str(value) if value is not None else None


def _parse_dense_integer_array_text(text: str, *, element_type: str) -> tuple[int, ...]:
    stripped = text.strip()
    prefix = f"array<{element_type}:"
    if not stripped.startswith(prefix) or not stripped.endswith(">"):
        raise ValueError(f"unsupported DenseIntegerArrayAttr spelling `{text}`")
    body = stripped[len(prefix) : -1].strip()
    if not body:
        return ()
    return tuple(int(piece.strip()) for piece in body.split(","))
