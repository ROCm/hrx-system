# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Inline importer check case parsing."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class InlineCheckSyntax:
    """Comment syntax and separators for one inline check file format."""

    case_separator_prefix: str
    expected_separator: str
    run_prefix: str | None = None
    comment_prefix: str | None = None
    case_separator_spacer: str = "\n"


DEFAULT_CHECK_SYNTAX = InlineCheckSyntax(
    case_separator_prefix="// ====",
    expected_separator="// ----",
    run_prefix="// RUN: ",
    comment_prefix="//",
)


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Half-open character-offset span within an inline check source file."""

    start: int
    end: int

    def text_from(self, source: str) -> str:
        """Returns the source text covered by this span."""
        return source[self.start : self.end]


@dataclass(frozen=True, slots=True)
class CheckCase:
    """One parsed inline importer check case."""

    path: Path
    index: int
    source: str
    input: str
    expected: str
    has_expected: bool = False
    run: str | None = None
    line_start: int = field(default=1, compare=False)
    line_end: int = field(default=1, compare=False)
    raw_source: str = field(default="", compare=False)
    source_span: SourceSpan | None = field(default=None, compare=False)
    input_span: SourceSpan | None = field(default=None, compare=False)
    expected_span: SourceSpan | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True)
class InlineCaseInputRewriteResult:
    """Source text after rewriting only inline check case input spans."""

    source: str
    cases: tuple[CheckCase, ...]
    changed_cases: tuple[CheckCase, ...]

    @property
    def changed(self) -> bool:
        return bool(self.changed_cases)


def parse_inline_cases(
    path: Path,
    source: str,
    *,
    syntax: InlineCheckSyntax = DEFAULT_CHECK_SYNTAX,
    default_run: str | None = None,
    allow_preamble: bool = False,
) -> tuple[CheckCase, ...]:
    """Parses loom-check style inline cases from one source file."""

    raw_cases = tuple(_split_raw_case_spans(source, syntax=syntax))
    if allow_preamble:
        raw_cases = _strip_preamble_case(raw_cases, syntax=syntax)
    if not raw_cases:
        raise ValueError(f"{path} has no check cases")

    runs = [_run_directive(raw.source, syntax=syntax) for raw in raw_cases]
    file_run = runs[0] or default_run
    cases: list[CheckCase] = []
    for index, raw_case in enumerate(raw_cases):
        case_source, expected, has_expected = split_expected(
            raw_case.source,
            syntax=syntax,
        )
        case_source_span, expected_span, input_span = _split_raw_case_spans_for_case(
            raw_case,
            syntax=syntax,
        )
        input_text = strip_case_directives(case_source, syntax=syntax)
        cases.append(
            CheckCase(
                path=path,
                index=index,
                source=case_source,
                input=input_text,
                expected=expected if has_expected else input_text,
                has_expected=has_expected,
                run=runs[index] or file_run,
                line_start=raw_case.line_start,
                line_end=raw_case.line_end,
                raw_source=raw_case.source,
                source_span=case_source_span,
                input_span=input_span,
                expected_span=expected_span if has_expected else None,
            )
        )
    return tuple(cases)


def rewrite_inline_case_inputs(
    path: Path,
    source: str,
    rewrite: Callable[[CheckCase, str], str],
    *,
    syntax: InlineCheckSyntax = DEFAULT_CHECK_SYNTAX,
    default_run: str | None = None,
    allow_preamble: bool = False,
) -> InlineCaseInputRewriteResult:
    """Rewrites input IR sections while leaving expected sections untouched."""

    cases = parse_inline_cases(
        path,
        source,
        syntax=syntax,
        default_run=default_run,
        allow_preamble=allow_preamble,
    )
    replacements: list[tuple[SourceSpan, str]] = []
    changed_cases: list[CheckCase] = []
    for check_case in cases:
        if check_case.input_span is None:
            continue
        input_text = check_case.input_span.text_from(source)
        replacement_text = rewrite(check_case, input_text)
        if replacement_text == input_text:
            continue
        replacements.append((check_case.input_span, replacement_text))
        changed_cases.append(check_case)

    rewritten_source = source
    for span, replacement_text in reversed(replacements):
        rewritten_source = (
            rewritten_source[: span.start]
            + replacement_text
            + rewritten_source[span.end :]
        )
    return InlineCaseInputRewriteResult(
        source=rewritten_source,
        cases=cases,
        changed_cases=tuple(changed_cases),
    )


def split_raw_cases(
    source: str,
    *,
    syntax: InlineCheckSyntax = DEFAULT_CHECK_SYNTAX,
) -> tuple[str, ...]:
    """Splits one inline check source into raw case sources."""

    return tuple(raw.source for raw in _split_raw_case_spans(source, syntax=syntax))


@dataclass(frozen=True, slots=True)
class _RawCheckCase:
    source: str
    line_start: int
    line_end: int
    source_span: SourceSpan
    is_preamble: bool = False


def _strip_preamble_case(
    raw_cases: tuple[_RawCheckCase, ...],
    *,
    syntax: InlineCheckSyntax,
) -> tuple[_RawCheckCase, ...]:
    """Drops a non-case prefix before the first explicit case separator."""

    if len(raw_cases) < 2 or not raw_cases[0].is_preamble:
        return raw_cases
    _case_source, _expected, has_expected = split_expected(
        raw_cases[0].source,
        syntax=syntax,
    )
    if has_expected:
        return raw_cases
    return raw_cases[1:]


def _split_raw_case_spans(
    source: str,
    *,
    syntax: InlineCheckSyntax,
) -> tuple[_RawCheckCase, ...]:
    cases: list[_RawCheckCase] = []
    lines: list[str] = []
    line_start = 1
    line_number = 1
    character_start = 0
    character_offset = 0
    before_first_separator = True
    for line in source.splitlines(keepends=True):
        line_character_start = character_offset
        character_offset += len(line)
        if line.startswith(syntax.case_separator_prefix):
            trimmed_character_count = trim_separator_spacer(lines)
            case_source = "".join(lines)
            if case_source.strip():
                character_end = line_character_start - trimmed_character_count
                cases.append(
                    _RawCheckCase(
                        source=case_source,
                        line_start=line_start,
                        line_end=max(line_start, line_number - 1),
                        source_span=SourceSpan(character_start, character_end),
                        is_preamble=before_first_separator,
                    )
                )
            lines = []
            line_start = line_number + 1
            character_start = character_offset
            line_number += 1
            before_first_separator = False
            continue
        lines.append(line)
        line_number += 1
    trimmed_character_count = trim_separator_spacer(lines)
    case_source = "".join(lines)
    if case_source.strip():
        character_end = len(source) - trimmed_character_count
        cases.append(
            _RawCheckCase(
                source=case_source,
                line_start=line_start,
                line_end=max(line_start, line_number - 1),
                source_span=SourceSpan(character_start, character_end),
                is_preamble=before_first_separator,
            )
        )
    return tuple(cases)


def trim_separator_spacer(lines: list[str]) -> int:
    """Drops formatting-only blank lines before a case separator."""

    trimmed_character_count = 0
    while lines and not lines[-1].strip():
        trimmed_character_count += len(lines.pop())
    return trimmed_character_count


def split_expected(
    source: str,
    *,
    syntax: InlineCheckSyntax = DEFAULT_CHECK_SYNTAX,
) -> tuple[str, str, bool]:
    """Splits a raw case into input and expected-output sections."""

    input_lines: list[str] = []
    expected_lines: list[str] = []
    in_expected = False
    for line in source.splitlines(keepends=True):
        if line.strip() == syntax.expected_separator:
            in_expected = True
            continue
        if in_expected:
            expected_lines.append(line)
        else:
            input_lines.append(line)
    return "".join(input_lines), "".join(expected_lines), in_expected


def strip_case_directives(
    source: str,
    *,
    syntax: InlineCheckSyntax = DEFAULT_CHECK_SYNTAX,
) -> str:
    """Removes check-runner directives from source input passed to an importer."""

    if syntax.run_prefix is None:
        return source
    lines: list[str] = []
    for line in source.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith(syntax.run_prefix):
            continue
        lines.append(line)
    return "".join(lines)


def _split_raw_case_spans_for_case(
    raw_case: _RawCheckCase,
    *,
    syntax: InlineCheckSyntax,
) -> tuple[SourceSpan, SourceSpan | None, SourceSpan]:
    input_end_relative = len(raw_case.source)
    expected_span = None
    scanner_offset = 0
    for line in raw_case.source.splitlines(keepends=True):
        line_end_offset = scanner_offset + len(line)
        if line.strip() == syntax.expected_separator:
            input_end_relative = scanner_offset
            expected_span = SourceSpan(
                raw_case.source_span.start + line_end_offset,
                raw_case.source_span.end,
            )
            break
        scanner_offset = line_end_offset

    input_start_relative = _input_start_relative(
        raw_case.source[:input_end_relative],
        syntax=syntax,
    )
    return (
        SourceSpan(
            raw_case.source_span.start,
            raw_case.source_span.start + input_end_relative,
        ),
        expected_span,
        SourceSpan(
            raw_case.source_span.start + input_start_relative,
            raw_case.source_span.start + input_end_relative,
        ),
    )


def _input_start_relative(source: str, *, syntax: InlineCheckSyntax) -> int:
    scanner_offset = 0
    body_start_offset = 0
    for line in source.splitlines(keepends=True):
        line_end_offset = scanner_offset + len(line)
        stripped = line.strip()
        if _is_header_line(stripped, syntax=syntax):
            body_start_offset = line_end_offset
            scanner_offset = line_end_offset
            continue
        break
    return body_start_offset


def _is_header_line(stripped_line: str, *, syntax: InlineCheckSyntax) -> bool:
    if not stripped_line:
        return True
    if stripped_line == syntax.expected_separator:
        return False
    if _is_diagnostic_annotation_line(stripped_line, syntax=syntax):
        return False
    return syntax.comment_prefix is not None and stripped_line.startswith(
        syntax.comment_prefix
    )


def _is_diagnostic_annotation_line(
    stripped_line: str,
    *,
    syntax: InlineCheckSyntax,
) -> bool:
    if syntax.comment_prefix is None:
        return False
    if not stripped_line.startswith(syntax.comment_prefix):
        return False
    payload = stripped_line[len(syntax.comment_prefix) :].strip()
    annotation_prefixes = (
        "ERROR:",
        "ERROR@",
        "WARNING:",
        "WARNING@",
        "REMARK:",
        "REMARK@",
    )
    return payload in ("ERROR", "WARNING", "REMARK") or payload.startswith(
        annotation_prefixes
    )


def case_matches_filter(
    check_case: CheckCase,
    case_filter: str | None,
    *,
    labels: Sequence[str] = (),
) -> bool:
    """Returns whether a parsed case should run under a substring filter."""

    if not case_filter:
        return True
    candidates = [
        str(check_case.path),
        check_case.path.name,
        f"{check_case.path}:case{check_case.index}",
        f"case{check_case.index}",
    ]
    if check_case.run:
        candidates.append(check_case.run)
    candidates.extend(label for label in labels if label)
    return any(case_filter in candidate for candidate in candidates)


def _run_directive(
    source: str,
    *,
    syntax: InlineCheckSyntax,
) -> str | None:
    if syntax.run_prefix is None:
        return None
    for line in source.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(syntax.run_prefix):
            return stripped[len(syntax.run_prefix) :].strip()
        if syntax.comment_prefix is not None and stripped.startswith(
            syntax.comment_prefix
        ):
            continue
        return None
    return None
