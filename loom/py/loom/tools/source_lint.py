# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Command-line authoring policy checks for explicit Loom source files."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

SUPPORTED_SUFFIXES = frozenset({".loom", ".loom-test"})
CONSTANT_NAME_RULE = "constant-name"

_CONSTANT_RESULT_PATTERN = re.compile(
    r"(?P<result>%(?P<name>[A-Za-z_][A-Za-z0-9_]*))\s*=\s*"
    r"(?P<operation>[A-Za-z_][A-Za-z0-9_.]*constant)\b"
)

# Type, domain, unit, and duplicate markers can decorate either side of a
# spelled literal without giving it a program role. Semantic continuations
# such as zero_point and fourth_word_ordinal remain intact.
_NUMBER_NAME_DECORATOR_PATTERN = re.compile(
    r"(?:"
    r"(?:[su]?i\d+|(?:bf|fp|f)\d+)(?:x\d+|v)?"
    r"|index|offset|scalars?|vectors?"
    r"|bytes?|bits?|elements?|lanes?|rows?|columns?|words?|values?"
    r"|all|[a-z]"
    r")"
)
_NUMBER_WORDS = tuple(
    sorted(
        (
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
            "hundred",
            "thousand",
            "million",
            "billion",
            "trillion",
        ),
        key=len,
        reverse=True,
    )
)
_NUMBER_CONNECTOR = "and"
_NUMBER_SIGN_PREFIXES = ("negative", "positive", "minus", "plus", "neg")


@dataclass(frozen=True)
class Finding:
    """One source policy violation."""

    path: Path
    line: int
    column: int
    name: str

    def format(self) -> str:
        return (
            f"{self.path}:{self.line}:{self.column}: error: constant SSA name "
            f"%{self.name} spells a numeric literal in English; use a "
            "program-role name or %c<literal> "
            f"[{CONSTANT_NAME_RULE}]"
        )


class SourceLintError(Exception):
    """An invalid invocation input or source read failure."""


def _is_spelled_number_sequence(text: str) -> bool:
    for prefix in _NUMBER_SIGN_PREFIXES:
        if text.startswith(prefix):
            text = text.removeprefix(prefix)
            break
    if not text:
        return False

    # Each state records whether the previous token was numeric. `and` is an
    # interior connector, not a number by itself: this accepts
    # `onehundredandone` while leaving semantic names such as `and_zero` alone.
    reachable = {(0, False)}
    for start in range(len(text)):
        states = tuple(
            previous_was_number
            for position, previous_was_number in reachable
            if position == start
        )
        for previous_was_number in states:
            for word in _NUMBER_WORDS:
                if text.startswith(word, start):
                    reachable.add((start + len(word), True))
            if previous_was_number and text.startswith(_NUMBER_CONNECTOR, start):
                reachable.add((start + len(_NUMBER_CONNECTOR), False))
    return (len(text), True) in reachable


def _is_spelled_number_or_plural(text: str) -> bool:
    if _is_spelled_number_sequence(text):
        return True

    singular_candidates: list[str] = []
    if text.endswith("ies"):
        singular_candidates.append(text[:-3] + "y")
    if text.endswith("es"):
        singular_candidates.append(text[:-2])
    if text.endswith("s"):
        singular_candidates.append(text[:-1])
    return any(
        _is_spelled_number_sequence(candidate) for candidate in singular_candidates
    )


def _is_spelled_number_constant_name(name: str) -> bool:
    tokens = [token for token in name.lower().split("_") if token]
    while len(tokens) > 1:
        stripped_decorator = False
        if _NUMBER_NAME_DECORATOR_PATTERN.fullmatch(tokens[0]):
            tokens.pop(0)
            stripped_decorator = True
        if len(tokens) > 1 and _NUMBER_NAME_DECORATOR_PATTERN.fullmatch(tokens[-1]):
            tokens.pop()
            stripped_decorator = True
        if not stripped_decorator:
            break
    return _is_spelled_number_or_plural("".join(tokens))


def _blank_preserving_newlines(text: str) -> str:
    return "".join(character if character in "\r\n" else " " for character in text)


def _mask_source_line(line: str, in_string: bool) -> tuple[str, bool]:
    """Masks comments and quoted strings without changing source positions."""

    masked = list(line)
    position = 0
    while position < len(line):
        character = line[position]
        if character in "\r\n":
            position += 1
            continue
        if in_string:
            masked[position] = " "
            if character == "\\" and position + 1 < len(line):
                position += 1
                if line[position] not in "\r\n":
                    masked[position] = " "
            elif character == '"':
                in_string = False
            position += 1
            continue
        if character == '"':
            masked[position] = " "
            in_string = True
            position += 1
            continue
        if character == "/" and position + 1 < len(line) and line[position + 1] == "/":
            for comment_position in range(position, len(line)):
                if line[comment_position] not in "\r\n":
                    masked[comment_position] = " "
            break
        position += 1
    return "".join(masked), in_string


def _mask_authored_source(path: Path, text: str) -> str:
    """Returns position-preserving authored code suitable for lexical checks."""

    is_loom_test = path.suffix == ".loom-test"
    in_expected_section = False
    in_string = False
    masked_lines: list[str] = []
    for line in text.splitlines(keepends=True):
        stripped_line = line.strip()
        if is_loom_test and stripped_line.startswith("// ===="):
            in_expected_section = False
            in_string = False
            masked_lines.append(_blank_preserving_newlines(line))
            continue
        if is_loom_test and stripped_line == "// ----":
            in_expected_section = True
            in_string = False
            masked_lines.append(_blank_preserving_newlines(line))
            continue
        if in_expected_section:
            masked_lines.append(_blank_preserving_newlines(line))
            continue
        masked_line, in_string = _mask_source_line(line, in_string)
        masked_lines.append(masked_line)
    return "".join(masked_lines)


def lint_source(path: Path, text: str) -> list[Finding]:
    """Returns authoring-policy findings for one supported source."""

    masked_source = _mask_authored_source(path, text)
    findings: list[Finding] = []
    for match in _CONSTANT_RESULT_PATTERN.finditer(masked_source):
        name = match.group("name")
        if not _is_spelled_number_constant_name(name):
            continue
        source_offset = match.start("result")
        line = masked_source.count("\n", 0, source_offset) + 1
        line_start = masked_source.rfind("\n", 0, source_offset) + 1
        findings.append(
            Finding(
                path=path,
                line=line,
                column=source_offset - line_start + 1,
                name=name,
            )
        )
    return findings


def _read_source(path: Path) -> str:
    if path.suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise SourceLintError(
            f"unsupported source suffix for {path}; expected {supported}"
        )
    if not path.is_file():
        raise SourceLintError(f"source file does not exist: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise SourceLintError(f"cannot read source file {path}: {error}") from error


def run(paths: Sequence[Path], *, stderr: TextIO) -> int:
    """Lints explicit paths and returns the public process exit code."""

    findings: list[Finding] = []
    try:
        for path in paths:
            findings.extend(lint_source(path, _read_source(path)))
    except SourceLintError as error:
        stderr.write(f"loom-lint: error: {error}\n")
        return 2

    for finding in findings:
        stderr.write(finding.format() + "\n")
    return 1 if findings else 0


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loom-lint",
        description="Checks authoring policy in explicit Loom source files.",
    )
    parser.add_argument(
        "sources",
        type=Path,
        nargs="+",
        metavar="SOURCE",
        help="A .loom module or .loom-test source container.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _create_argument_parser().parse_args(argv)
    return run(args.sources, stderr=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
