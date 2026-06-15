# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python subprocess API for the production `loom-opt` tool."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from loom.diagnostics import (
    Diagnostic,
    DiagnosticFieldRef,
    DiagnosticHighlightRange,
    DiagnosticParam,
    DiagnosticRelatedLocation,
    DiagnosticSeverity,
    SourceProvenance,
    SourceRange,
)
from loom.error import ALL_ERRORS
from loom.errors import Emitter, ErrorDef

_ERROR_BY_ID = {error.error_id: error for error in ALL_ERRORS}
LOOM_BIN_DIR_ENV_VAR = "LOOM_BIN_DIR"
_LOOM_OPT_TOOL_NAME = "loom-opt"


@dataclass(frozen=True, slots=True)
class LoomOptResult:
    """Result of one `loom-opt` invocation."""

    returncode: int
    stdout: str
    stderr: str
    diagnostics: tuple[Diagnostic, ...] = ()
    residual_stderr: str = ""
    invocation_error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0 and self.invocation_error is None

    def diagnostic_text(self) -> str:
        """Returns human-readable diagnostics for failure reporting."""

        if self.diagnostics:
            return "".join(f"{diagnostic}\n" for diagnostic in self.diagnostics)
        if self.residual_stderr:
            return self.residual_stderr
        if self.stderr:
            return self.stderr
        if self.invocation_error:
            return f"{self.invocation_error}\n"
        return ""


@dataclass(frozen=True, slots=True)
class LoomOpt:
    """Resolved `loom-opt` executable with structured invocation helpers."""

    executable: Path

    @classmethod
    def resolve(cls, explicit_path: Path | None = None) -> LoomOpt | None:
        """Resolves `loom-opt` from an explicit path, bin-root env, or PATH."""

        if explicit_path is not None:
            return cls(explicit_path)
        bin_dir = os.environ.get(LOOM_BIN_DIR_ENV_VAR)
        if bin_dir:
            return cls(_tool_path_in_bin_dir(Path(bin_dir), _LOOM_OPT_TOOL_NAME))
        executable = shutil.which(_LOOM_OPT_TOOL_NAME)
        if executable is None:
            return None
        return cls(Path(executable))

    def verify_module_text(self, module_text: str) -> LoomOptResult:
        """Runs production parse and verification on a printed Loom module."""

        return self.run(module_text, diagnostic_format="json")

    def run(
        self,
        module_text: str,
        *,
        passes: Sequence[str] = (),
        pipeline: str | None = None,
        verify: bool = True,
        diagnostic_format: str = "json",
    ) -> LoomOptResult:
        """Runs `loom-opt` with stdin input and structured diagnostic capture."""

        args = [
            str(self.executable),
            f"--verify={'true' if verify else 'false'}",
            f"--diagnostic-format={diagnostic_format}",
        ]
        if pipeline is not None:
            args.append(f"--pipeline={pipeline}")
        args.extend(f"--pass={pass_name}" for pass_name in passes)
        try:
            process = subprocess.run(
                args,
                input=module_text,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            return LoomOptResult(
                returncode=1,
                stdout="",
                stderr="",
                invocation_error=f"{type(exc).__name__}: {exc}",
            )

        diagnostics, residual_stderr = parse_loom_diagnostic_stream(process.stderr)
        return LoomOptResult(
            returncode=process.returncode,
            stdout=process.stdout,
            stderr=process.stderr,
            diagnostics=diagnostics,
            residual_stderr=residual_stderr,
        )


def parse_loom_diagnostic_stream(text: str) -> tuple[tuple[Diagnostic, ...], str]:
    """Parses JSONL diagnostics from a tool stderr stream.

    Non-diagnostic stderr lines are preserved so tool-level status text is not
    lost when a command mixes structured diagnostics with ordinary process
    errors.
    """

    diagnostics: list[Diagnostic] = []
    residual_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if not stripped.startswith("{"):
            residual_lines.append(line)
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            residual_lines.append(line)
            continue
        if not _looks_like_diagnostic_object(payload):
            residual_lines.append(line)
            continue
        diagnostics.append(_diagnostic_from_json_object(payload))
    return tuple(diagnostics), "".join(residual_lines)


def _looks_like_diagnostic_object(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    return all(key in payload for key in ("severity", "error_id", "message"))


def _diagnostic_from_json_object(payload: Mapping[str, object]) -> Diagnostic:
    return Diagnostic(
        severity=_diagnostic_severity(payload.get("severity")),
        message=_json_string(payload.get("message"))
        or _json_string(payload.get("summary"))
        or "",
        origin=_source_range(payload.get("origin")),
        source_location=_source_range(payload.get("source_location")),
        params=_diagnostic_params(payload),
        emitter=_emitter(payload.get("emitter")),
        highlights=_highlight_ranges(payload.get("highlights")),
        highlight_omitted_count=_json_int(payload.get("highlight_omitted_count")),
        related_locations=_related_locations(payload.get("related_locations")),
        related_location_omitted_count=_json_int(
            payload.get("related_location_omitted_count")
        ),
        details=_diagnostic_details(payload),
        error_def=_error_def(payload.get("error_id")),
    )


def _diagnostic_severity(value: object) -> DiagnosticSeverity:
    if isinstance(value, str):
        try:
            return DiagnosticSeverity(value)
        except ValueError:
            pass
    return DiagnosticSeverity.ERROR


def _emitter(value: object) -> Emitter | None:
    if not isinstance(value, str):
        return None
    try:
        return Emitter[value.upper()]
    except KeyError:
        return None


def _error_def(value: object) -> ErrorDef | None:
    if not isinstance(value, str):
        return None
    return _ERROR_BY_ID.get(value)


def _diagnostic_params(payload: Mapping[str, object]) -> tuple[DiagnosticParam, ...]:
    raw_params = payload.get("params")
    if not isinstance(raw_params, Mapping):
        return ()
    raw_param_fields = payload.get("param_fields")
    param_fields = raw_param_fields if isinstance(raw_param_fields, Mapping) else {}
    return tuple(
        DiagnosticParam(
            name=str(name),
            value=value,
            field_ref=_field_ref(param_fields.get(name)),
        )
        for name, value in raw_params.items()
    )


def _diagnostic_details(payload: Mapping[str, object]) -> tuple[str, ...]:
    fix_hint = _json_string(payload.get("fix_hint"))
    return (f"fix: {fix_hint}",) if fix_hint else ()


def _source_range(value: object) -> SourceRange | None:
    if not isinstance(value, Mapping):
        return None
    provenance = _source_provenance(value.get("provenance"))
    filename = _json_string(value.get("filename"))
    excerpt = value.get("excerpt")
    source = ""
    if isinstance(excerpt, Mapping):
        source = _json_string(excerpt.get("text")) or ""
    return SourceRange(
        provenance=provenance,
        filename=Path(filename) if filename else None,
        source=source,
        start=_json_int(value.get("start_byte")),
        end=_json_int(value.get("end_byte")),
        start_line=_json_int(value.get("start_line")),
        start_column=_json_int(value.get("start_column")),
        end_line=_json_int(value.get("end_line")),
        end_column=_json_int(value.get("end_column")),
    )


def _source_provenance(value: object) -> SourceProvenance:
    if isinstance(value, str):
        try:
            return SourceProvenance(value)
        except ValueError:
            pass
    return SourceProvenance.UNAVAILABLE_SOURCE


def _highlight_ranges(value: object) -> tuple[DiagnosticHighlightRange, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    return tuple(
        highlight
        for item in value
        if isinstance(item, Mapping)
        if (highlight := _highlight_range(item)) is not None
    )


def _highlight_range(value: Mapping[str, object]) -> DiagnosticHighlightRange | None:
    start = _json_optional_int(value.get("start_byte"))
    end = _json_optional_int(value.get("end_byte"))
    if start is None or end is None:
        return None
    return DiagnosticHighlightRange(
        start=start,
        end=end,
        field_ref=_field_ref(value.get("field")),
    )


def _related_locations(value: object) -> tuple[DiagnosticRelatedLocation, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    locations: list[DiagnosticRelatedLocation] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        source_location = _source_range(item.get("source_location"))
        if source_location is None:
            continue
        locations.append(
            DiagnosticRelatedLocation(
                label=_json_string(item.get("label")) or "",
                source_location=source_location,
                highlights=_highlight_ranges(item.get("highlights")),
            )
        )
    return tuple(locations)


def _field_ref(value: object) -> DiagnosticFieldRef | None:
    if not isinstance(value, Mapping):
        return None
    kind = _json_string(value.get("kind"))
    index = _json_optional_int(value.get("index"))
    if kind is None or index is None:
        return None
    return DiagnosticFieldRef(
        kind=kind,
        index=index,
        occurrence=_json_int(value.get("occurrence")),
    )


def _json_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _json_optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _json_int(value: object) -> int:
    return _json_optional_int(value) or 0


def _tool_path_in_bin_dir(bin_dir: Path, tool_name: str) -> Path:
    if os.name == "nt":
        return bin_dir / f"{tool_name}.exe"
    return bin_dir / tool_name
