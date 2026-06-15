# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Diagnostics shared by Python Loom APIs.

This mirrors the C diagnostic vocabulary in `loom/src/loom/error/`: a
diagnostic has a severity, structured error identity, origin/source ranges,
typed parameters, highlights, related locations, and emitter identity.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from loom.errors import Emitter, ErrorDef

__all__ = [
    "Diagnostic",
    "DiagnosticEngine",
    "DiagnosticFieldRef",
    "DiagnosticHighlightRange",
    "DiagnosticParam",
    "DiagnosticRelatedLocation",
    "DiagnosticSeverity",
    "SourceProvenance",
    "SourceRange",
    "LoomDiagnosticError",
]


class DiagnosticSeverity(StrEnum):
    """Severity for Python diagnostics."""

    ERROR = "error"
    WARNING = "warning"
    REMARK = "remark"


class SourceProvenance(StrEnum):
    """Identifies which bytes back a diagnostic source range."""

    EXACT_SOURCE = "exact_source"
    PRINTED_IR_FALLBACK = "printed_ir_fallback"
    UNAVAILABLE_SOURCE = "unavailable_source"


@dataclass(frozen=True, slots=True)
class SourceRange:
    """A byte and line range in a source buffer.

    Field names intentionally follow `loom_source_range_t` closely. Offsets are
    byte offsets into `source`; the range is half-open: [start, end).
    """

    provenance: SourceProvenance = SourceProvenance.UNAVAILABLE_SOURCE
    filename: Path | None = None
    source: str = ""
    start: int = 0
    end: int = 0
    start_line: int = 0
    start_column: int = 0
    end_line: int = 0
    end_column: int = 0

    @classmethod
    def unavailable(cls) -> SourceRange:
        return cls()

    @classmethod
    def line(
        cls,
        filename: Path,
        line: int,
        *,
        column: int = 0,
        source: str = "",
        provenance: SourceProvenance = SourceProvenance.EXACT_SOURCE,
    ) -> SourceRange:
        return cls(
            provenance=provenance,
            filename=filename,
            source=source,
            start_line=line,
            start_column=column,
            end_line=line,
            end_column=column,
        )

    @property
    def has_location(self) -> bool:
        return self.filename is not None and self.start_line > 0

    def display(self) -> str:
        if self.filename is None:
            return "<unknown>"
        if self.start_column > 0:
            return f"{self.filename}:{self.start_line}:{self.start_column}"
        return f"{self.filename}:{self.start_line}"


@dataclass(frozen=True, slots=True)
class DiagnosticFieldRef:
    """Structured reference to an op field named by a diagnostic parameter."""

    kind: str
    index: int
    occurrence: int = 0


@dataclass(frozen=True, slots=True)
class DiagnosticHighlightRange:
    """A byte range within a diagnostic source buffer for highlighting."""

    start: int
    end: int
    field_ref: DiagnosticFieldRef | None = None
    param_index: int | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticParam:
    """One rendered runtime parameter for a structured diagnostic."""

    name: str
    value: Any
    kind: str | None = None
    field_ref: DiagnosticFieldRef | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticRelatedLocation:
    """A labeled secondary diagnostic source location."""

    label: str
    source_location: SourceRange
    highlights: tuple[DiagnosticHighlightRange, ...] = ()
    highlight_omitted_count: int = 0


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """A diagnostic attached to a Python API step."""

    severity: DiagnosticSeverity
    message: str
    origin: SourceRange | None = None
    source_location: SourceRange | None = None
    params: tuple[DiagnosticParam, ...] = ()
    emitter: Emitter | None = None
    highlights: tuple[DiagnosticHighlightRange, ...] = ()
    highlight_omitted_count: int = 0
    related_locations: tuple[DiagnosticRelatedLocation, ...] = ()
    related_location_omitted_count: int = 0
    source: str | None = None
    details: tuple[str, ...] = ()
    error_def: ErrorDef | None = None

    @property
    def domain(self) -> str | None:
        if self.error_def is None:
            return None
        return self.error_def.domain.name

    @property
    def code(self) -> str | None:
        if self.error_def is None:
            return None
        return str(self.error_def.code)

    @property
    def error_id(self) -> str | None:
        if self.error_def is None:
            return None
        return self.error_def.error_id

    @property
    def primary_location(self) -> SourceRange | None:
        return self.source_location or self.origin

    def rendered_message(self) -> str:
        pieces = [self.message]
        if self.source:
            pieces.append(f"source: {self.source}")
        pieces.extend(self.details)
        return "\n".join(piece for piece in pieces if piece)

    def rendered_params(self) -> dict[str, str]:
        """Returns diagnostic parameters rendered for structured matching."""

        return {
            param.name: _render_diagnostic_param_value(param.value)
            for param in self.params
        }

    def __str__(self) -> str:
        prefix = self.severity.value
        if self.error_def is not None:
            prefix = f"{prefix} {self.error_def}"
        location = self.primary_location
        if location is not None and location.has_location:
            pieces = [f"{location.display()}: {prefix}: {self.message}"]
        else:
            pieces = [f"{prefix}: {self.message}"]
        if self.source:
            pieces.append(f"  source: {self.source}")
        pieces.extend(f"  {detail}" for detail in self.details)
        for related in self.related_locations:
            pieces.append(
                f"  {related.source_location.display()}: note: {related.label}"
            )
        return "\n".join(pieces)


class LoomDiagnosticError(RuntimeError):
    """Raised when diagnostics contain errors at a fail-loud API boundary."""

    def __init__(self, diagnostics: Iterable[Diagnostic]) -> None:
        self.diagnostics = tuple(diagnostics)
        message = "\n".join(str(diagnostic) for diagnostic in self.diagnostics)
        super().__init__(message or "Loom operation failed")


@dataclass(slots=True)
class DiagnosticEngine:
    """Collects diagnostics and owns fail-loud API boundaries."""

    _diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def diagnostics(self) -> tuple[Diagnostic, ...]:
        return tuple(self._diagnostics)

    @property
    def has_errors(self) -> bool:
        return any(
            diagnostic.severity == DiagnosticSeverity.ERROR
            for diagnostic in self._diagnostics
        )

    def remark(
        self,
        message: str,
        *,
        origin: SourceRange | None = None,
        source_location: SourceRange | None = None,
        params: Iterable[DiagnosticParam] = (),
        emitter: Emitter | None = None,
        highlights: Iterable[DiagnosticHighlightRange] = (),
        highlight_omitted_count: int = 0,
        related_locations: Iterable[DiagnosticRelatedLocation] = (),
        related_location_omitted_count: int = 0,
        source: str | None = None,
        details: Iterable[str] = (),
        error_def: ErrorDef | None = None,
    ) -> Diagnostic:
        return self.emit(
            DiagnosticSeverity.REMARK,
            message,
            origin=origin,
            source_location=source_location,
            params=params,
            emitter=emitter,
            highlights=highlights,
            highlight_omitted_count=highlight_omitted_count,
            related_locations=related_locations,
            related_location_omitted_count=related_location_omitted_count,
            source=source,
            details=details,
            error_def=error_def,
        )

    def warning(
        self,
        message: str,
        *,
        origin: SourceRange | None = None,
        source_location: SourceRange | None = None,
        params: Iterable[DiagnosticParam] = (),
        emitter: Emitter | None = None,
        highlights: Iterable[DiagnosticHighlightRange] = (),
        highlight_omitted_count: int = 0,
        related_locations: Iterable[DiagnosticRelatedLocation] = (),
        related_location_omitted_count: int = 0,
        source: str | None = None,
        details: Iterable[str] = (),
        error_def: ErrorDef | None = None,
    ) -> Diagnostic:
        return self.emit(
            DiagnosticSeverity.WARNING,
            message,
            origin=origin,
            source_location=source_location,
            params=params,
            emitter=emitter,
            highlights=highlights,
            highlight_omitted_count=highlight_omitted_count,
            related_locations=related_locations,
            related_location_omitted_count=related_location_omitted_count,
            source=source,
            details=details,
            error_def=error_def,
        )

    def error(
        self,
        message: str,
        *,
        origin: SourceRange | None = None,
        source_location: SourceRange | None = None,
        params: Iterable[DiagnosticParam] = (),
        emitter: Emitter | None = None,
        highlights: Iterable[DiagnosticHighlightRange] = (),
        highlight_omitted_count: int = 0,
        related_locations: Iterable[DiagnosticRelatedLocation] = (),
        related_location_omitted_count: int = 0,
        source: str | None = None,
        details: Iterable[str] = (),
        error_def: ErrorDef | None = None,
    ) -> Diagnostic:
        return self.emit(
            DiagnosticSeverity.ERROR,
            message,
            origin=origin,
            source_location=source_location,
            params=params,
            emitter=emitter,
            highlights=highlights,
            highlight_omitted_count=highlight_omitted_count,
            related_locations=related_locations,
            related_location_omitted_count=related_location_omitted_count,
            source=source,
            details=details,
            error_def=error_def,
        )

    def unsupported(
        self,
        operation: str,
        reason: str,
        *,
        origin: SourceRange | None = None,
        source_location: SourceRange | None = None,
        params: Iterable[DiagnosticParam] = (),
        emitter: Emitter | None = None,
        highlights: Iterable[DiagnosticHighlightRange] = (),
        highlight_omitted_count: int = 0,
        related_locations: Iterable[DiagnosticRelatedLocation] = (),
        related_location_omitted_count: int = 0,
        details: Iterable[str] = (),
        error_def: ErrorDef | None = None,
    ) -> Diagnostic:
        return self.error(
            f"unsupported source operation: {reason}",
            origin=origin,
            source_location=source_location,
            params=params,
            emitter=emitter,
            highlights=highlights,
            highlight_omitted_count=highlight_omitted_count,
            related_locations=related_locations,
            related_location_omitted_count=related_location_omitted_count,
            source=operation,
            details=details,
            error_def=error_def,
        )

    def emit(
        self,
        severity: DiagnosticSeverity,
        message: str,
        *,
        origin: SourceRange | None = None,
        source_location: SourceRange | None = None,
        params: Iterable[DiagnosticParam] = (),
        emitter: Emitter | None = None,
        highlights: Iterable[DiagnosticHighlightRange] = (),
        highlight_omitted_count: int = 0,
        related_locations: Iterable[DiagnosticRelatedLocation] = (),
        related_location_omitted_count: int = 0,
        source: str | None = None,
        details: Iterable[str] = (),
        error_def: ErrorDef | None = None,
    ) -> Diagnostic:
        diagnostic = Diagnostic(
            severity=severity,
            message=message,
            origin=origin,
            source_location=source_location,
            params=tuple(params),
            emitter=emitter,
            highlights=tuple(highlights),
            highlight_omitted_count=highlight_omitted_count,
            related_locations=tuple(related_locations),
            related_location_omitted_count=related_location_omitted_count,
            source=source,
            details=tuple(details),
            error_def=error_def,
        )
        self._diagnostics.append(diagnostic)
        return diagnostic

    def extend(self, diagnostics: Iterable[Diagnostic]) -> None:
        self._diagnostics.extend(diagnostics)

    def raise_if_errors(self) -> None:
        if self.has_errors:
            raise LoomDiagnosticError(self._diagnostics)


def _render_diagnostic_param_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, tuple | list):
        return ", ".join(_render_diagnostic_param_value(item) for item in value)
    return str(value)
