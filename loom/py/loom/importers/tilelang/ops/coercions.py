# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared TileLang/TIR operand coercion helpers."""

from __future__ import annotations

from loom.builder import ValueRef
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.nodes import node_text
from loom.ir import Type


def coerce_integer_operand(
    value: ValueRef,
    target_type: Type,
    context: TileLangConversionContext,
    owner: object,
    *,
    name: str,
    operand_name: str,
) -> ValueRef | None:
    """Normalize an operand for a Loom op that requires a scalar integer."""

    source_type = str(value.type)
    target_type_text = str(target_type)
    if source_type == target_type_text:
        return value
    if _is_index_type(source_type) and _is_integer_type(target_type_text):
        result = context.builder.index.cast(
            input=value,
            results=[target_type],
            name=context.fresh_name(name),
        )
        context.record_converted(
            node_text(owner),
            f"{context.ssa(result)} = index.cast",
        )
        return result
    context.record_blocked(
        node_text(owner),
        (
            f"operand `{operand_name}` must be {target_type_text} or index-like, "
            f"got {source_type}"
        ),
    )
    return None


def _is_index_type(value_type: str) -> bool:
    return value_type in ("index", "offset")


def _is_integer_type(value_type: str) -> bool:
    return value_type != "i1" and value_type.startswith("i")
