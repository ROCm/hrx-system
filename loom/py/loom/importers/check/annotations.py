# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source diagnostic annotations for importer check fixtures."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from loom.diagnostics import Diagnostic, DiagnosticSeverity, SourceRange
from loom.errors import ErrorDomain
from loom.importers.check.cases import CheckCase
from loom.importers.check.results import CheckResult, unified_diff

_ANNOTATION_RE = re.compile(
    r"^(?P<severity>ERROR|WARNING|REMARK)(?:@(?P<offset>[+-]?\d+))?:"
    r"(?P<body>.*)$"
)
_DOMAIN_CODE_RE = re.compile(r"^(?P<domain>[A-Z][A-Z0-9_]*)(?:/(?P<code>[^/]+))?$")
_PARAM_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")
_QUOTED_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_MAX_SUBSTRINGS = 4
_MAX_PARAM_MATCHES = 8


class AnnotationSyntaxError(ValueError):
    """Raised when an importer check diagnostic annotation is malformed."""


@dataclass(frozen=True, slots=True)
class ExpectedDiagnosticParamMatch:
    """One structured diagnostic parameter matcher from an annotation."""

    name: str
    value: str


@dataclass(frozen=True, slots=True)
class ExpectedDiagnostic:
    """One expected diagnostic annotation from a source fixture."""

    severity: DiagnosticSeverity
    target: SourceRange
    annotation: SourceRange
    domain: str | None = None
    code: int | None = None
    param_matches: tuple[ExpectedDiagnosticParamMatch, ...] = ()
    message_substrings: tuple[str, ...] = ()

    def display(self) -> str:
        identity = ""
        if self.domain is not None:
            identity = f" {self.domain}"
            if self.code is not None and self.code != 0:
                identity = f"{identity}/{self.code:03d}"
            elif self.code == 0:
                identity = f"{identity}/000"
        params = "".join(
            f" {match.name}={match.value!r}" for match in self.param_matches
        )
        substrings = "".join(f" {substring!r}" for substring in self.message_substrings)
        return (
            f"{self.annotation.display()}: expected {self.severity.value}"
            f"{identity} at {self.target.display()}{params}{substrings}"
        )


@dataclass(frozen=True, slots=True)
class DiagnosticMatchResult:
    """Result of matching source diagnostics against expected annotations."""

    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures

    def message(self) -> str:
        return "\n".join(self.failures)


def parse_expected_diagnostics(
    source: str,
    *,
    path: Path,
    line_start: int,
    comment_prefix: str,
) -> tuple[ExpectedDiagnostic, ...]:
    """Parses source diagnostic annotations from one check case input."""

    expected: list[ExpectedDiagnostic] = []
    for local_index, line in enumerate(source.splitlines(), start=0):
        comment = _extract_comment_text(line, comment_prefix=comment_prefix)
        if comment is None or not _looks_like_annotation(comment):
            continue
        line_number = line_start + local_index
        expected.append(
            _parse_annotation(
                comment,
                path=path,
                line_number=line_number,
                first_case_line=line_start,
            )
        )
    return tuple(expected)


def match_source_diagnostics(
    expected: tuple[ExpectedDiagnostic, ...],
    actual: tuple[Diagnostic, ...],
) -> DiagnosticMatchResult:
    """Matches actual diagnostics against source annotations."""

    failures: list[str] = []
    used_actual: set[int] = set()
    for annotation in expected:
        match_index = _find_matching_diagnostic(annotation, actual, used_actual)
        if match_index is None:
            failures.append(_unmatched_annotation_message(annotation, actual))
        else:
            used_actual.add(match_index)
    for index, diagnostic in enumerate(actual):
        if index in used_actual:
            continue
        failures.append(f"unexpected diagnostic:\n{diagnostic}")
    return DiagnosticMatchResult(tuple(failures))


def source_diagnostic_check_result(
    case: CheckCase,
    *,
    expected_diagnostics: tuple[ExpectedDiagnostic, ...],
    actual_diagnostics: tuple[Diagnostic, ...],
    stdout: str = "",
) -> CheckResult:
    """Builds a check result for a case with source diagnostic annotations."""

    match = match_source_diagnostics(expected_diagnostics, actual_diagnostics)
    stderr = "".join(f"{diagnostic}\n" for diagnostic in actual_diagnostics)
    if not match.passed:
        return CheckResult(
            path=case.path,
            case_index=case.index,
            returncode=0,
            stdout=stdout,
            stderr=stderr,
            input=case.input,
            expected=case.expected if case.has_expected else "",
            mismatch="diagnostics differ from expected annotations",
            diff=match.message(),
        )
    if case.has_expected and stdout != case.expected:
        return CheckResult(
            path=case.path,
            case_index=case.index,
            returncode=0,
            stdout=stdout,
            stderr=stderr,
            input=case.input,
            expected=case.expected,
            mismatch="output differs from expected output",
            diff=unified_diff(
                case.expected,
                stdout,
                fromfile=f"{case.path}:case{case.index}:expected",
                tofile=f"{case.path}:case{case.index}:actual",
            ),
        )
    return CheckResult(
        path=case.path,
        case_index=case.index,
        returncode=0,
        stdout=stdout,
        stderr=stderr,
        input=case.input,
        expected=case.expected if case.has_expected else "",
    )


def has_expected_diagnostics(source: str, *, comment_prefix: str) -> bool:
    """Returns whether a source fragment contains any diagnostic annotation."""

    return any(
        (comment := _extract_comment_text(line, comment_prefix=comment_prefix))
        is not None
        and _looks_like_annotation(comment)
        for line in source.splitlines()
    )


def _parse_annotation(
    text: str,
    *,
    path: Path,
    line_number: int,
    first_case_line: int,
) -> ExpectedDiagnostic:
    match = _ANNOTATION_RE.match(text)
    if match is None:
        raise AnnotationSyntaxError(
            f"{path}:{line_number}: malformed diagnostic annotation"
        )
    severity = DiagnosticSeverity(match.group("severity").lower())
    offset_text = match.group("offset")
    offset = int(offset_text) if offset_text is not None else 0
    target_line = line_number + offset
    if target_line < first_case_line:
        raise AnnotationSyntaxError(
            f"{path}:{line_number}: annotation target is before the case input"
        )
    body = match.group("body").strip()
    domain, code, param_matches, substrings = _parse_annotation_body(
        body,
        path=path,
        line_number=line_number,
    )
    return ExpectedDiagnostic(
        severity=severity,
        domain=domain,
        code=code,
        param_matches=param_matches,
        message_substrings=substrings,
        annotation=SourceRange.line(path, line_number),
        target=SourceRange.line(path, target_line),
    )


def _parse_annotation_body(
    body: str,
    *,
    path: Path,
    line_number: int,
) -> tuple[
    str | None,
    int | None,
    tuple[ExpectedDiagnosticParamMatch, ...],
    tuple[str, ...],
]:
    text = body.strip()
    if not text:
        return None, None, (), ()
    if text.startswith('"'):
        return (
            None,
            None,
            (),
            _parse_substring_list(
                text,
                path=path,
                line_number=line_number,
            ),
        )

    domain_code, remainder = _split_first_token(text)
    domain, code = _parse_domain_code(
        domain_code,
        path=path,
        line_number=line_number,
    )
    param_matches, remainder = _parse_param_matchers(
        remainder,
        path=path,
        line_number=line_number,
    )
    substrings = _parse_substring_list(
        remainder,
        path=path,
        line_number=line_number,
    )
    return domain, code, param_matches, substrings


def _find_matching_diagnostic(
    annotation: ExpectedDiagnostic,
    actual: tuple[Diagnostic, ...],
    used_actual: set[int],
) -> int | None:
    for index, diagnostic in enumerate(actual):
        if index in used_actual:
            continue
        if _diagnostic_matches(annotation, diagnostic):
            return index
    return None


def _diagnostic_matches(
    annotation: ExpectedDiagnostic,
    diagnostic: Diagnostic,
) -> bool:
    location = diagnostic.primary_location
    if location is None or location.filename is None:
        return False
    if location.filename != annotation.target.filename:
        return False
    if location.start_line != annotation.target.start_line:
        return False
    if diagnostic.severity != annotation.severity:
        return False
    if annotation.domain is not None and diagnostic.domain != annotation.domain:
        return False
    if annotation.code is not None and annotation.code != 0:
        if diagnostic.code is None:
            return False
        if _parse_code(diagnostic.code) != annotation.code:
            return False
    rendered_message = diagnostic.rendered_message()
    if not all(
        substring in rendered_message for substring in annotation.message_substrings
    ):
        return False
    params = diagnostic.rendered_params()
    return all(
        params.get(match.name) == match.value for match in annotation.param_matches
    )


def _unmatched_annotation_message(
    annotation: ExpectedDiagnostic,
    actual: tuple[Diagnostic, ...],
) -> str:
    lines = [f"expected diagnostic was not produced:\n{annotation.display()}"]
    nearby = tuple(
        diagnostic
        for diagnostic in actual
        if diagnostic.primary_location is not None
        and diagnostic.primary_location.filename == annotation.target.filename
        and diagnostic.primary_location.start_line == annotation.target.start_line
    )
    if nearby:
        lines.append("diagnostics at the target line:")
        lines.extend(str(diagnostic) for diagnostic in nearby)
    return "\n".join(lines)


def _extract_comment_text(line: str, *, comment_prefix: str) -> str | None:
    stripped = line.strip()
    if not stripped.startswith(comment_prefix):
        return None
    return stripped[len(comment_prefix) :].strip()


def _looks_like_annotation(comment: str) -> bool:
    text = comment.strip()
    for severity in ("ERROR", "WARNING", "REMARK"):
        if text.startswith(severity):
            suffix = text[len(severity) :]
            return suffix.startswith((":", "@"))
    return False


def _split_first_token(text: str) -> tuple[str, str]:
    pieces = text.split(maxsplit=1)
    if len(pieces) == 1:
        return pieces[0], ""
    return pieces[0], pieces[1].strip()


def _parse_domain_code(
    text: str,
    *,
    path: Path,
    line_number: int,
) -> tuple[str, int]:
    match = _DOMAIN_CODE_RE.match(text)
    if match is None:
        raise AnnotationSyntaxError(
            f"{path}:{line_number}: expected DOMAIN or DOMAIN/CODE"
        )
    domain = match.group("domain")
    if domain not in ErrorDomain.__members__:
        raise AnnotationSyntaxError(
            f"{path}:{line_number}: unknown error domain in annotation: `{domain}`"
        )
    code_text = match.group("code")
    if code_text is None:
        return domain, 0
    if not code_text:
        raise AnnotationSyntaxError(
            f"{path}:{line_number}: empty code after `/` in annotation"
        )
    return domain, _parse_code_for_annotation(
        code_text,
        path=path,
        line_number=line_number,
    )


def _parse_code_for_annotation(
    text: str,
    *,
    path: Path,
    line_number: int,
) -> int:
    if not text.isdecimal():
        raise AnnotationSyntaxError(
            f"{path}:{line_number}: non-numeric code in annotation: `{text}`"
        )
    code = int(text)
    if code > 65535:
        raise AnnotationSyntaxError(
            f"{path}:{line_number}: annotation code out of range: {code}"
        )
    return code


def _parse_code(text: str) -> int | None:
    if not text.isdecimal():
        return None
    return int(text)


def _parse_param_matchers(
    text: str,
    *,
    path: Path,
    line_number: int,
) -> tuple[tuple[ExpectedDiagnosticParamMatch, ...], str]:
    remainder = text.strip()
    if not remainder.startswith("{"):
        return (), remainder
    remainder = remainder[1:].strip()
    matches: list[ExpectedDiagnosticParamMatch] = []
    while True:
        if remainder.startswith("}"):
            return tuple(matches), remainder[1:].strip()
        if len(matches) >= _MAX_PARAM_MATCHES:
            raise AnnotationSyntaxError(
                f"{path}:{line_number}: annotation has more than "
                f"{_MAX_PARAM_MATCHES} diagnostic param matchers"
            )
        name_match = _PARAM_NAME_RE.match(remainder)
        if name_match is None:
            raise AnnotationSyntaxError(
                f"{path}:{line_number}: expected diagnostic param name"
            )
        name = name_match.group(0)
        remainder = remainder[name_match.end() :].strip()
        if not remainder.startswith("="):
            raise AnnotationSyntaxError(
                f"{path}:{line_number}: diagnostic param matcher `{name}` missing `=`"
            )
        value, remainder = _parse_quoted_string(
            remainder[1:].strip(),
            path=path,
            line_number=line_number,
        )
        matches.append(ExpectedDiagnosticParamMatch(name=name, value=value))
        if remainder.startswith(","):
            remainder = remainder[1:].strip()
            continue
        if remainder.startswith("}"):
            continue
        raise AnnotationSyntaxError(
            f"{path}:{line_number}: diagnostic param matcher expected `,` or `}}`"
        )


def _parse_substring_list(
    text: str,
    *,
    path: Path,
    line_number: int,
) -> tuple[str, ...]:
    remainder = text.strip()
    substrings: list[str] = []
    while remainder:
        if len(substrings) >= _MAX_SUBSTRINGS:
            raise AnnotationSyntaxError(
                f"{path}:{line_number}: annotation has more than "
                f"{_MAX_SUBSTRINGS} quoted substrings"
            )
        if not remainder.startswith('"'):
            raise AnnotationSyntaxError(
                f"{path}:{line_number}: unexpected text after annotation matcher"
            )
        substring, remainder = _parse_quoted_string(
            remainder,
            path=path,
            line_number=line_number,
        )
        substrings.append(substring)
    return tuple(substrings)


def _parse_quoted_string(
    text: str,
    *,
    path: Path,
    line_number: int,
) -> tuple[str, str]:
    match = _QUOTED_RE.match(text)
    if match is None:
        raise AnnotationSyntaxError(
            f"{path}:{line_number}: malformed quoted diagnostic matcher"
        )
    try:
        value = ast.literal_eval(match.group(0))
    except (SyntaxError, ValueError) as exc:
        raise AnnotationSyntaxError(
            f"{path}:{line_number}: malformed quoted diagnostic matcher"
        ) from exc
    if not isinstance(value, str):
        raise AnnotationSyntaxError(
            f"{path}:{line_number}: diagnostic matcher must be a string"
        )
    return value, text[match.end() :].strip()
