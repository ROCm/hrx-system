# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Inline importer check update formatting."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from loom.importers.check.cases import (
    DEFAULT_CHECK_SYNTAX,
    CheckCase,
    InlineCheckSyntax,
    split_expected,
)
from loom.importers.check.results import CheckResult


def format_updated_source(
    original_source: str,
    cases: Sequence[CheckCase],
    results: Sequence[CheckResult],
    *,
    syntax: InlineCheckSyntax = DEFAULT_CHECK_SYNTAX,
    expected_encoder: Callable[[str], str] | None = None,
) -> str:
    encode_expected = expected_encoder or _identity_expected
    updated_cases: list[str] = []
    result_by_index = {result.case_index: result for result in results}
    for case in cases:
        result = result_by_index.get(case.index)
        if result is None:
            updated_cases.append(case.raw_source or case.source)
            continue
        if (
            result.returncode == 0
            and not result.skipped
            and (case.has_expected or result.stdout)
        ):
            updated_cases.append(
                format_updated_case(
                    case.source,
                    result.stdout,
                    syntax=syntax,
                    expected_encoder=encode_expected,
                )
            )
        else:
            updated_cases.append(case.raw_source or case.source)
    body = f"{syntax.case_separator_spacer}{syntax.case_separator_prefix}\n".join(
        updated_cases
    )
    prefix = leading_case_separator_prefix(original_source, cases, syntax)
    if prefix is None:
        return body
    return f"{prefix}{syntax.case_separator_prefix}\n{body}"


def format_updated_case(
    source: str,
    actual: str,
    *,
    syntax: InlineCheckSyntax = DEFAULT_CHECK_SYNTAX,
    expected_encoder: Callable[[str], str] | None = None,
) -> str:
    encode_expected = expected_encoder or _identity_expected
    case_source, _expected, _has_expected = split_expected(source, syntax=syntax)
    return (
        ensure_trailing_newline(case_source)
        + f"{syntax.expected_separator}\n"
        + ensure_trailing_newline(encode_expected(actual))
    )


def ensure_trailing_newline(text: str) -> str:
    return text if not text or text.endswith("\n") else f"{text}\n"


def _identity_expected(text: str) -> str:
    return text


def leading_case_separator_prefix(
    source: str,
    cases: Sequence[CheckCase],
    syntax: InlineCheckSyntax,
) -> str | None:
    """Returns source before a leading separator when one should be preserved."""

    if not cases:
        return None
    lines = source.splitlines(keepends=True)
    separator_index: int | None = None
    for index, line in enumerate(lines[: max(cases[0].line_start - 1, 0)]):
        if line.startswith(syntax.case_separator_prefix):
            separator_index = index
    if separator_index is None:
        return None
    return "".join(lines[:separator_index])
