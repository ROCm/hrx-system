# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Registry and dispatcher for TileLang/TVM TIR node converters."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from inspect import getdoc
from typing import cast, overload

from loom.builder import ValueRef
from loom.importers.core import ImportBodyReport
from loom.importers.tilelang.context import TileLangConversionContext
from loom.importers.tilelang.coverage import CoverageState, coverage_by_name
from loom.importers.tilelang.nodes import node_kind, node_text
from loom.importers.tilelang.types import TileLangTypeConversionError
from loom.ir import INDEX, OFFSET


@dataclass(frozen=True, slots=True)
class ExpressionOptions:
    """Options for converting one expression node."""

    index_like: bool = False
    effect: bool = False


StatementHandler = Callable[
    [object, TileLangConversionContext, "TileLangConverter"], None
]
ExpressionHandler = Callable[
    [object, TileLangConversionContext, "TileLangConverter", ExpressionOptions],
    ValueRef | None,
]


@dataclass(frozen=True, slots=True)
class ConverterInfo:
    """Registered TileLang/TIR node converter metadata."""

    node_name: str
    handler: StatementHandler | ExpressionHandler
    summary: str | None


class TileLangConverterRegistry:
    """Maps TileLang/TIR node names to conversion handlers."""

    def __init__(self) -> None:
        self._statements: dict[str, ConverterInfo] = {}
        self._expressions: dict[str, ConverterInfo] = {}

    @overload
    def register_statement(
        self,
        node_name: str,
        handler: StatementHandler,
        *,
        summary: str | None = None,
    ) -> StatementHandler: ...

    @overload
    def register_statement(
        self,
        node_name: str,
        handler: None = None,
        *,
        summary: str | None = None,
    ) -> Callable[[StatementHandler], StatementHandler]: ...

    def register_statement(
        self,
        node_name: str,
        handler: StatementHandler | None = None,
        *,
        summary: str | None = None,
    ) -> StatementHandler | Callable[[StatementHandler], StatementHandler]:
        def bind(bound_handler: StatementHandler) -> StatementHandler:
            _register(
                self._statements,
                node_name,
                bound_handler,
                summary=summary,
                kind="statement",
            )
            return bound_handler

        if handler is None:
            return bind
        return bind(handler)

    @overload
    def register_expression(
        self,
        node_name: str,
        handler: ExpressionHandler,
        *,
        summary: str | None = None,
    ) -> ExpressionHandler: ...

    @overload
    def register_expression(
        self,
        node_name: str,
        handler: None = None,
        *,
        summary: str | None = None,
    ) -> Callable[[ExpressionHandler], ExpressionHandler]: ...

    def register_expression(
        self,
        node_name: str,
        handler: ExpressionHandler | None = None,
        *,
        summary: str | None = None,
    ) -> ExpressionHandler | Callable[[ExpressionHandler], ExpressionHandler]:
        def bind(bound_handler: ExpressionHandler) -> ExpressionHandler:
            _register(
                self._expressions,
                node_name,
                bound_handler,
                summary=summary,
                kind="expression",
            )
            return bound_handler

        if handler is None:
            return bind
        return bind(handler)

    def statement(self, node_name: str) -> ConverterInfo | None:
        return self._statements.get(node_name)

    def expression(self, node_name: str) -> ConverterInfo | None:
        return self._expressions.get(node_name)


class TileLangConverter:
    """Converts TileLang/TIR nodes using a registered handler set."""

    def __init__(self, registry: TileLangConverterRegistry) -> None:
        self.registry = registry
        self.operation_counts: Counter[str] = Counter()

    def convert_body(
        self,
        stmt: object,
        context: TileLangConversionContext,
    ) -> ImportBodyReport:
        self.convert_stmt(stmt, context)
        return context.finish()

    def convert_stmt(
        self,
        stmt: object,
        context: TileLangConversionContext,
    ) -> None:
        if stmt is None:
            return
        kind = node_kind(stmt)
        self.operation_counts[kind] += 1
        converter = self.registry.statement(kind)
        if converter is None:
            self.record_unsupported(stmt, context)
            return
        handler = cast(StatementHandler, converter.handler)
        try:
            handler(stmt, context, self)
        except TileLangTypeConversionError as exc:
            context.record_blocked(node_text(stmt), str(exc))

    def convert_expr(
        self,
        expr: object,
        context: TileLangConversionContext,
        *,
        index_like: bool = False,
        effect: bool = False,
    ) -> ValueRef | None:
        if expr is None:
            return None
        if index_like:
            mapped_index = context.mapped_index_value(expr)
            if mapped_index is not None:
                return mapped_index
        mapped = context.mapped(expr)
        if mapped is not None and (not index_like or mapped.type in (INDEX, OFFSET)):
            return mapped
        kind = node_kind(expr)
        self.operation_counts[kind] += 1
        converter = self.registry.expression(kind)
        if converter is None:
            self.record_unsupported(expr, context)
            return None
        handler = cast(ExpressionHandler, converter.handler)
        try:
            return handler(
                expr,
                context,
                self,
                ExpressionOptions(index_like=index_like, effect=effect),
            )
        except TileLangTypeConversionError as exc:
            context.record_blocked(node_text(expr), str(exc))
            return None

    def record_unsupported(
        self,
        node: object,
        context: TileLangConversionContext,
    ) -> None:
        kind = node_kind(node)
        row = coverage_by_name().get(f"tir.{kind}") or coverage_by_name().get(kind)
        if row is None:
            context.record_unsupported(kind, node_text(node))
            return
        if row.state == CoverageState.SUPPORTED:
            reason = "coverage manifest marks this node supported but no converter ran"
        else:
            reason = f"coverage state is {row.state.value}: {row.note}"
        context.record_blocked(node_text(node), reason)


def converter_summary(
    handler: StatementHandler | ExpressionHandler,
) -> str | None:
    doc = getdoc(handler)
    if doc is None:
        return None
    return doc.splitlines()[0]


def _register(
    converters: dict[str, ConverterInfo],
    node_name: str,
    handler: StatementHandler | ExpressionHandler,
    *,
    summary: str | None,
    kind: str,
) -> None:
    if node_name in converters:
        raise ValueError(f"{kind} converter already registered for {node_name}")
    converters[node_name] = ConverterInfo(
        node_name=node_name,
        handler=handler,
        summary=summary or converter_summary(handler),
    )
