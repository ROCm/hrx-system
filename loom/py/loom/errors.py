# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Loom structured error DSL.

This module defines the types used to declare structured errors in Python.
These declarations are the single source of truth: generators consume them
to produce C error tables, JSON error catalogs, and documentation.

Error definitions are frozen dataclass instances organized by semantic
domain (TYPE, SHAPE, PARSE, etc.). Each error has a stable numeric code
within its domain, typed parameters, and a message template with {param}
placeholders.

Quick reference — declaring an error:

    ERR_TYPE_001 = ErrorDef(
        domain=ErrorDomain.TYPE,
        code=1,
        severity=Severity.ERROR,
        summary="SameType constraint violated.",
        message="'{field_a}' type {type_a} does not match "
                "'{field_b}' type {type_b}",
        params=(
            ErrorParam("field_a", ParamKind.STRING),
            ErrorParam("type_a", ParamKind.TYPE),
            ErrorParam("field_b", ParamKind.STRING),
            ErrorParam("type_b", ParamKind.TYPE),
        ),
        fix_hint="Ensure '{field_a}' and '{field_b}' have the same type",
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum, unique

__all__ = [
    # Enums.
    "ParamKind",
    "ErrorDomain",
    "Emitter",
    "Severity",
    # Dataclasses.
    "ErrorParam",
    "ErrorDef",
]

# Matches {param_name} placeholders in message templates.
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


@unique
class ParamKind(IntEnum):
    """Parameter kind for diagnostic parameters.

    Maps 1:1 to loom_param_kind_t in C. The kind determines how the
    parameter is stored, rendered, and serialized to JSON.
    """

    STRING = 0  # iree_string_view_t / str
    I64 = 1  # int64_t / int
    U32 = 2  # uint32_t / int
    BOOL = 3  # bool
    TYPE = 4  # loom_type_t (rendered by type printer)
    U64 = 5  # uint64_t / int
    STRING_LIST = 6  # Sequence of iree_string_view_t / list[str]


@unique
class ErrorDomain(IntEnum):
    """Semantic error domain.

    Describes *what* the error is about, not which subsystem emitted it.
    An agent filtering for type errors doesn't care whether the verifier
    or the parser caught it.

    Maps 1:1 to loom_error_domain_t in C.
    """

    TYPE = 0  # Type mismatches, constraint violations.
    SHAPE = 1  # Rank mismatches, shape inconsistencies.
    SUBRANGE = 2  # Offset/size counts, bounds violations.
    ENCODING = 3  # Encoding mismatches.
    STRUCTURE = 4  # Count errors, terminators, regions.
    DOMINANCE = 5  # Undefined values, use-after-consume.
    SYMBOL = 6  # Unresolved references.
    PARSE = 7  # Syntax errors, tokenization.
    BYTECODE = 8  # Format errors, version mismatches.
    FOLD = 9  # Poison folding, canonicalization.
    LOWERING = 10  # Pass lowering legality and unsupported mappings.
    BACKEND = 11  # Target codegen feedback, allocation, scheduling, emission.
    TARGET = 12  # Target contract and target legality failures.
    AMDGPU = 13  # AMDGPU-owned legality and lowering failures.
    X86 = 14  # X86-owned legality and lowering failures.
    WASM = 15  # WebAssembly-owned legality and lowering failures.
    SPIRV = 16  # SPIR-V-owned legality and lowering failures.


@unique
class Emitter(IntEnum):
    """Which subsystem emitted the diagnostic.

    Context for debugging and filtering, not part of the error identity.
    The same error definition can be emitted by the verifier, parser,
    or a compiler pass.

    Maps 1:1 to loom_emitter_t in C.
    """

    VERIFIER = 0
    PARSER = 1
    BYTECODE_READER = 2
    PASS = 3
    BUILDER = 4


@unique
class Severity(IntEnum):
    """Diagnostic severity level.

    Maps 1:1 to loom_diagnostic_severity_t in C.
    """

    ERROR = 0
    WARNING = 1
    REMARK = 2


@dataclass(frozen=True, slots=True)
class ErrorParam:
    """A typed parameter in an error's message template.

    The name must match a {placeholder} in the ErrorDef's message
    template. The kind determines how the parameter is stored and
    rendered.
    """

    name: str
    kind: ParamKind

    def __repr__(self) -> str:
        return f"{self.name}: {self.kind.name}"


@dataclass(frozen=True, slots=True)
class ErrorDef:
    """A structured error with stable code and typed parameters.

    The (domain, code) pair is the unique identity. Codes are never
    reused within a domain. The message template uses {param_name}
    placeholders that reference the params tuple.

    Display format is zero-padded: ERR_TYPE_001, ERR_SHAPE_012.
    The underlying code is a plain int; padding is a display convention.
    """

    domain: ErrorDomain
    code: int
    severity: Severity
    summary: str
    message: str  # Template with {param_name} placeholders.
    params: tuple[ErrorParam, ...]
    description: str = ""
    fix_hint: str = ""  # Template with {param_name} placeholders.
    example: str = ""

    def __post_init__(self) -> None:
        # Code must be positive.
        if self.code < 1:
            raise ValueError(
                f"error code must be >= 1, got {self.code} "
                f"in {self.domain.name}/{self.code}"
            )

        # No duplicate param names.
        param_names = [p.name for p in self.params]
        seen: set[str] = set()
        for name in param_names:
            if name in seen:
                raise ValueError(
                    f"duplicate param name '{name}' in "
                    f"ERR_{self.domain.name}_{self.code:03d}"
                )
            seen.add(name)

        # All message placeholders must reference declared params.
        param_name_set = set(param_names)
        for template_name, template in [
            ("message", self.message),
            ("fix_hint", self.fix_hint),
        ]:
            placeholders = set(_PLACEHOLDER_RE.findall(template))
            undefined = placeholders - param_name_set
            if undefined:
                raise ValueError(
                    f"{template_name} references undefined params "
                    f"{sorted(undefined)} in "
                    f"ERR_{self.domain.name}_{self.code:03d}; "
                    f"declared params: {sorted(param_name_set)}"
                )

    @property
    def error_id(self) -> str:
        """The canonical display string: ERR_TYPE_001, etc."""
        return f"ERR_{self.domain.name}_{self.code:03d}"

    def __repr__(self) -> str:
        return self.error_id
