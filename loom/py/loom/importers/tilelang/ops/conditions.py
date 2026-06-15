# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared TileLang/TIR condition conversion helpers."""

from __future__ import annotations

from loom.builder import ValueRef
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.nodes import node_text
from loom.ir import I1


def coerce_condition(
    condition: ValueRef,
    context: TileLangConversionContext,
    owner: object,
) -> ValueRef | None:
    """Normalize TileLang truthy scalar conditions to Loom i1."""

    source_type = str(condition.type)
    if source_type == "i1":
        return condition
    if source_type in ("index", "offset"):
        zero = context.ensure_constant("0", source_type, "c0")
        return context.builder.index.cmp(
            predicate="ne",
            lhs=condition,
            rhs=zero,
            results=[I1],
            name=context.fresh_name("cmp"),
        )
    if _is_integer_type(source_type):
        zero = ensure_scalar_zero(context, source_type)
        return context.builder.scalar.cmpi(
            predicate="ne",
            lhs=condition,
            rhs=zero,
            results=[I1],
            name=context.fresh_name("cmp"),
        )
    context.record_blocked(
        node_text(owner),
        f"condition must be i1 or integer-like, got {condition.type}",
    )
    return None


def ensure_scalar_zero(
    context: TileLangConversionContext,
    value_type: str,
) -> ValueRef:
    """Build or reuse a scalar integer zero constant."""

    existing = context.constants.get((value_type, "0"))
    if existing is not None:
        return existing
    result = context.build_constant(0, value_type, context.reserve_name("const"))
    context.remember_constant("0", value_type, result)
    return result


def _is_integer_type(value_type: str) -> bool:
    return value_type in {"i8", "i16", "i32", "i64"}
