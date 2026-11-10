# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Input-section rewriting for canonical ``.loom-test`` sources."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

CASE_SEPARATOR_PREFIX = "// ===="
EXPECTED_SEPARATOR = "// ----"


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """Half-open character-offset span within a test source."""

    start: int
    end: int

    def text_from(self, source: str) -> str:
        return source[self.start : self.end]


@dataclass(frozen=True, slots=True)
class LoomTestCase:
    """One test case input section and its position in the source."""

    index: int
    input_span: SourceSpan


@dataclass(frozen=True, slots=True)
class LoomTestInputRewriteResult:
    """Source text after rewriting only test case input sections."""

    source: str
    cases: tuple[LoomTestCase, ...]


def rewrite_loom_test_inputs(
    source: str,
    rewrite: Callable[[LoomTestCase, str], str],
) -> LoomTestInputRewriteResult:
    """Rewrites input IR while preserving directives and expected output."""

    cases = _parse_loom_test_cases(source)
    replacements: list[tuple[SourceSpan, str]] = []
    for test_case in cases:
        input_text = test_case.input_span.text_from(source)
        replacement_text = rewrite(test_case, input_text)
        if replacement_text != input_text:
            replacements.append((test_case.input_span, replacement_text))

    rewritten_source = source
    for span, replacement_text in reversed(replacements):
        rewritten_source = (
            rewritten_source[: span.start]
            + replacement_text
            + rewritten_source[span.end :]
        )
    return LoomTestInputRewriteResult(source=rewritten_source, cases=cases)


@dataclass(frozen=True, slots=True)
class _RawCase:
    source: str
    source_span: SourceSpan


def _parse_loom_test_cases(source: str) -> tuple[LoomTestCase, ...]:
    raw_cases = _split_raw_cases(source)
    if not raw_cases:
        raise ValueError("test source has no cases")
    return tuple(
        LoomTestCase(index=index, input_span=_input_span(raw_case))
        for index, raw_case in enumerate(raw_cases)
    )


def _split_raw_cases(source: str) -> tuple[_RawCase, ...]:
    cases: list[_RawCase] = []
    lines: list[str] = []
    character_start = 0
    character_offset = 0
    for line in source.splitlines(keepends=True):
        line_character_start = character_offset
        character_offset += len(line)
        if line.startswith(CASE_SEPARATOR_PREFIX):
            trimmed_character_count = _trim_separator_spacer(lines)
            case_source = "".join(lines)
            if case_source.strip():
                cases.append(
                    _RawCase(
                        source=case_source,
                        source_span=SourceSpan(
                            character_start,
                            line_character_start - trimmed_character_count,
                        ),
                    )
                )
            lines = []
            character_start = character_offset
            continue
        lines.append(line)

    trimmed_character_count = _trim_separator_spacer(lines)
    case_source = "".join(lines)
    if case_source.strip():
        cases.append(
            _RawCase(
                source=case_source,
                source_span=SourceSpan(
                    character_start,
                    len(source) - trimmed_character_count,
                ),
            )
        )
    return tuple(cases)


def _trim_separator_spacer(lines: list[str]) -> int:
    trimmed_character_count = 0
    while lines and not lines[-1].strip():
        trimmed_character_count += len(lines.pop())
    return trimmed_character_count


def _input_span(raw_case: _RawCase) -> SourceSpan:
    input_end = len(raw_case.source)
    scanner_offset = 0
    for line in raw_case.source.splitlines(keepends=True):
        if line.strip() == EXPECTED_SEPARATOR:
            input_end = scanner_offset
            break
        scanner_offset += len(line)

    input_start = 0
    scanner_offset = 0
    for line in raw_case.source[:input_end].splitlines(keepends=True):
        line_end = scanner_offset + len(line)
        if not _is_header_line(line.strip()):
            break
        input_start = line_end
        scanner_offset = line_end

    return SourceSpan(
        raw_case.source_span.start + input_start,
        raw_case.source_span.start + input_end,
    )


def _is_header_line(line: str) -> bool:
    if not line:
        return True
    if not line.startswith("//"):
        return False
    payload = line[2:].strip()
    annotation_prefixes = (
        "ERROR:",
        "ERROR@",
        "WARNING:",
        "WARNING@",
        "REMARK:",
        "REMARK@",
    )
    if payload in ("ERROR", "WARNING", "REMARK") or payload.startswith(
        annotation_prefixes
    ):
        return False
    return True
