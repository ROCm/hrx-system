# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source op field references used by target contracts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

from loom.dsl import AttrDef, Op, Operand, Result, TiedResult
from loom.target.contracts.kinds import SourceValueKind


@dataclass(frozen=True, slots=True)
class ValueRef:
    """Named source value consumed or produced by a contract row."""

    kind: SourceValueKind
    field: str
    materializer: str | None = None
    element: int = 0

    @classmethod
    def operand(
        cls,
        field: str,
        *,
        materializer: str | None = None,
        element: int = 0,
    ) -> Self:
        return cls(
            kind=SourceValueKind.OPERAND,
            field=field,
            materializer=materializer,
            element=element,
        )

    @classmethod
    def result(cls, field: str) -> Self:
        return cls(kind=SourceValueKind.RESULT, field=field)

    @classmethod
    def temporary(cls, field: str) -> Self:
        return cls(kind=SourceValueKind.TEMPORARY, field=field)

    @classmethod
    def source_memory_dynamic_term(cls, *, term: int = 0) -> Self:
        return cls(
            kind=SourceValueKind.SOURCE_MEMORY_DYNAMIC_TERM,
            field="",
            element=term,
        )

    @classmethod
    def source_memory_dynamic_byte_offset(cls) -> Self:
        return cls(
            kind=SourceValueKind.SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET,
            field="",
        )

    def validate(
        self,
        source_op: Op,
        subject: str,
        *,
        defined_temporaries: Iterable[str] = (),
    ) -> None:
        if self.materializer is not None:
            if not self.materializer:
                raise ValueError(
                    f"{source_op.name}: {subject} materializer must be non-empty"
                )
            if self.kind != SourceValueKind.OPERAND:
                raise ValueError(
                    f"{source_op.name}: {subject} materializer requires an operand"
                )
        if self.kind == SourceValueKind.OPERAND:
            if not self.field:
                raise ValueError(f"{source_op.name}: {subject} field must be non-empty")
            operand = _require_operand(source_op, self.field, subject)
            if self.element < 0:
                raise ValueError(
                    f"{source_op.name}: {subject} operand element must be non-negative"
                )
            if self.element != 0 and not operand.variadic:
                raise ValueError(
                    f"{source_op.name}: {subject} operand field '{self.field}' "
                    "is not variadic"
                )
            return
        if self.kind == SourceValueKind.RESULT:
            if not self.field:
                raise ValueError(f"{source_op.name}: {subject} field must be non-empty")
            if self.element != 0:
                raise ValueError(
                    f"{source_op.name}: {subject} element selection requires an operand"
                )
            _require_result(source_op, self.field, subject)
            return
        if self.kind == SourceValueKind.SOURCE_MEMORY_DYNAMIC_TERM:
            if self.field:
                raise ValueError(
                    f"{source_op.name}: {subject} source-memory term must not name "
                    "a source field"
                )
            if self.element < 0:
                raise ValueError(
                    f"{source_op.name}: {subject} source-memory term must be "
                    "non-negative"
                )
            return
        if self.kind == SourceValueKind.SOURCE_MEMORY_DYNAMIC_BYTE_OFFSET:
            if self.field:
                raise ValueError(
                    f"{source_op.name}: {subject} source-memory byte offset must "
                    "not name a source field"
                )
            if self.element != 0:
                raise ValueError(
                    f"{source_op.name}: {subject} source-memory byte offset must "
                    "not select an element"
                )
            return
        if not self.field:
            raise ValueError(f"{source_op.name}: {subject} field must be non-empty")
        if self.element != 0:
            raise ValueError(
                f"{source_op.name}: {subject} element selection requires an operand"
            )
        if self.field not in set(defined_temporaries):
            raise ValueError(
                f"{source_op.name}: {subject} temporary '{self.field}' is not defined"
            )


def _require_operand(source_op: Op, field: str, subject: str) -> Operand:
    operand = source_op.operand(field)
    if operand is None:
        raise ValueError(
            f"{source_op.name}: {subject} field '{field}' is not an operand"
        )
    return operand


def _require_result(source_op: Op, field: str, subject: str) -> Result | TiedResult:
    result = source_op.result(field)
    if result is None:
        raise ValueError(f"{source_op.name}: {subject} field '{field}' is not a result")
    return result


def _require_attr(source_op: Op, field: str, subject: str) -> AttrDef:
    attr = source_op.attr(field)
    if attr is None:
        raise ValueError(f"{source_op.name}: {subject} field '{field}' is not an attr")
    return attr


def _require_value(source_op: Op, field: str, subject: str) -> None:
    if source_op.operand(field) is None and source_op.result(field) is None:
        raise ValueError(
            f"{source_op.name}: {subject} field '{field}' is not an operand or result"
        )
