# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structured diagnostics authored by target contract fragments."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique

from loom.errors import ErrorDef, ParamKind

MAX_TARGET_DIAGNOSTIC_PARAMS = 16


@unique
class DiagnosticParamKind(Enum):
    """Projection kind for one generated diagnostic parameter."""

    TARGET_KEY = "target_key"
    EXPORT_NAME = "export_name"
    CONFIG_KEY = "config_key"
    FUNCTION_NAME = "function_name"
    SOURCE_OP_NAME = "source_op_name"
    STRING_LITERAL = "string_literal"
    VALUE_TYPE = "value_type"
    I64_LITERAL = "i64_literal"
    U32_LITERAL = "u32_literal"
    U64_LITERAL = "u64_literal"
    BOOL_LITERAL = "bool_literal"
    SOURCE_MEMORY_MINIMUM_ALIGNMENT = "source_memory_minimum_alignment"


_PARAM_KIND_BY_PROJECTION = {
    DiagnosticParamKind.TARGET_KEY: ParamKind.STRING,
    DiagnosticParamKind.EXPORT_NAME: ParamKind.STRING,
    DiagnosticParamKind.CONFIG_KEY: ParamKind.STRING,
    DiagnosticParamKind.FUNCTION_NAME: ParamKind.STRING,
    DiagnosticParamKind.SOURCE_OP_NAME: ParamKind.STRING,
    DiagnosticParamKind.STRING_LITERAL: ParamKind.STRING,
    DiagnosticParamKind.VALUE_TYPE: ParamKind.TYPE,
    DiagnosticParamKind.I64_LITERAL: ParamKind.I64,
    DiagnosticParamKind.U32_LITERAL: ParamKind.U32,
    DiagnosticParamKind.U64_LITERAL: ParamKind.U64,
    DiagnosticParamKind.BOOL_LITERAL: ParamKind.BOOL,
    DiagnosticParamKind.SOURCE_MEMORY_MINIMUM_ALIGNMENT: ParamKind.U32,
}


@dataclass(frozen=True, slots=True)
class DiagnosticParam:
    """One typed parameter projection in an authored diagnostic."""

    name: str
    kind: DiagnosticParamKind
    string_value: str = ""
    field: str = ""
    i64_value: int = 0
    u32_value: int = 0
    u64_value: int = 0
    bool_value: bool = False

    @property
    def param_kind(self) -> ParamKind:
        return _PARAM_KIND_BY_PROJECTION[self.kind]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("diagnostic parameter name must be non-empty")
        if self.kind == DiagnosticParamKind.STRING_LITERAL and not self.string_value:
            raise ValueError("string diagnostic parameter must be non-empty")
        if self.kind == DiagnosticParamKind.VALUE_TYPE and not self.field:
            raise ValueError("value-type diagnostic parameter needs a source field")


@dataclass(frozen=True, slots=True)
class DiagnosticRef:
    """Stable diagnostic identity plus generated parameter projections."""

    error: ErrorDef
    params: tuple[DiagnosticParam, ...]

    def __post_init__(self) -> None:
        expected_params = self.error.params
        if len(self.params) > MAX_TARGET_DIAGNOSTIC_PARAMS:
            raise ValueError(
                f"{self.error.error_id} uses {len(self.params)} params, "
                f"but generated target diagnostics support at most "
                f"{MAX_TARGET_DIAGNOSTIC_PARAMS}"
            )
        if len(self.params) != len(expected_params):
            raise ValueError(
                f"{self.error.error_id} expects {len(expected_params)} params, "
                f"got {len(self.params)}"
            )
        for actual, expected in zip(self.params, expected_params, strict=True):
            if actual.name != expected.name:
                raise ValueError(
                    f"{self.error.error_id} expected param '{expected.name}', "
                    f"got '{actual.name}'"
                )
            if actual.param_kind != expected.kind:
                raise ValueError(
                    f"{self.error.error_id} param '{expected.name}' expects "
                    f"{expected.kind.name}, got {actual.param_kind.name}"
                )


def target_key() -> DiagnosticParam:
    return DiagnosticParam("target_key", DiagnosticParamKind.TARGET_KEY)


def export_name() -> DiagnosticParam:
    return DiagnosticParam("export_name", DiagnosticParamKind.EXPORT_NAME)


def config_key() -> DiagnosticParam:
    return DiagnosticParam("config_key", DiagnosticParamKind.CONFIG_KEY)


def function_name() -> DiagnosticParam:
    return DiagnosticParam("function_name", DiagnosticParamKind.FUNCTION_NAME)


def source_op_name() -> DiagnosticParam:
    return DiagnosticParam("op_name", DiagnosticParamKind.SOURCE_OP_NAME)


def string_param(name: str, value: str) -> DiagnosticParam:
    return DiagnosticParam(
        name,
        DiagnosticParamKind.STRING_LITERAL,
        string_value=value,
    )


def value_type_param(name: str, field: str) -> DiagnosticParam:
    return DiagnosticParam(name, DiagnosticParamKind.VALUE_TYPE, field=field)


def i64_param(name: str, value: int) -> DiagnosticParam:
    return DiagnosticParam(name, DiagnosticParamKind.I64_LITERAL, i64_value=value)


def u32_param(name: str, value: int) -> DiagnosticParam:
    if value < 0 or value > 0xFFFF_FFFF:
        raise ValueError(f"u32 diagnostic parameter '{name}' out of range: {value}")
    return DiagnosticParam(name, DiagnosticParamKind.U32_LITERAL, u32_value=value)


def source_memory_minimum_alignment_param(name: str) -> DiagnosticParam:
    return DiagnosticParam(name, DiagnosticParamKind.SOURCE_MEMORY_MINIMUM_ALIGNMENT)


def u64_param(name: str, value: int) -> DiagnosticParam:
    if value < 0 or value > 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError(f"u64 diagnostic parameter '{name}' out of range: {value}")
    return DiagnosticParam(name, DiagnosticParamKind.U64_LITERAL, u64_value=value)


def bool_param(name: str, value: bool) -> DiagnosticParam:
    return DiagnosticParam(name, DiagnosticParamKind.BOOL_LITERAL, bool_value=value)


def diagnostic_ref(error: ErrorDef, *params: DiagnosticParam) -> DiagnosticRef:
    return DiagnosticRef(error, params)


def target_diagnostic(error: ErrorDef, *params: DiagnosticParam) -> DiagnosticRef:
    return DiagnosticRef(
        error,
        (
            target_key(),
            export_name(),
            config_key(),
            function_name(),
            source_op_name(),
            *params,
        ),
    )
