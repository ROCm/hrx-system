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
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from heapq import merge
from pathlib import Path
from typing import TextIO

SUPPORTED_SUFFIXES = frozenset({".loom", ".loom-test"})
CONSTANT_NAME_RULE = "constant-name"
GENERATED_NAME_RULE = "generated-name"

# Names include SSA values, symbols, block labels, aliases and bare identifiers.
# Literal/comment contents are masked before matching this lexical spelling.
_NAME_PATTERN = re.compile(r"[%@^#]?[A-Za-z0-9_$][A-Za-z0-9_$.-]*")

_CONSTANT_RESULT_PATTERN = re.compile(
    r"(?P<result>%(?P<name>[A-Za-z_][A-Za-z0-9_]*))\s*=\s*"
    r"(?P<operation>(?:[A-Za-z_][A-Za-z0-9_.]*constant|vector\.splat[fi]))\b"
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

    # Authored source file containing the name.
    path: Path
    # One-based source line.
    line: int
    # One-based character column.
    column: int
    # Complete identifier spelling, including its sigil when present.
    name: str
    # Authoring policy violated by this name.
    rule: str

    def format(self) -> str:
        if self.rule == GENERATED_NAME_RULE:
            message = (
                f"authored name {self.name} contains '$', which marks "
                "compiler-generated names; use a name that explains its program role"
            )
        else:
            message = (
                f"constant SSA name {self.name} spells a numeric literal in English; "
                "use a program-role name or %c<literal>"
            )
        return f"{self.path}:{self.line}:{self.column}: error: {message} [{self.rule}]"


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
    # Bit 0 permits a number; bit 1 also permits a following connector.
    # Direct position indexing keeps long names linear in their spelling length.
    reachable = bytearray(len(text) + 1)
    reachable[0] = 1
    for start in range(len(text)):
        if not reachable[start]:
            continue
        for word in _NUMBER_WORDS:
            if text.startswith(word, start):
                reachable[start + len(word)] |= 2
        if reachable[start] & 2 and text.startswith(_NUMBER_CONNECTOR, start):
            reachable[start + len(_NUMBER_CONNECTOR)] |= 1
    return bool(reachable[len(text)] & 2)


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
    first = 0
    end = len(tokens)
    while end - first > 1:
        stripped_decorator = False
        if _NUMBER_NAME_DECORATOR_PATTERN.fullmatch(tokens[first]):
            first += 1
            stripped_decorator = True
        if end - first > 1 and _NUMBER_NAME_DECORATOR_PATTERN.fullmatch(
            tokens[end - 1]
        ):
            end -= 1
            stripped_decorator = True
        if not stripped_decorator:
            break
    return _is_spelled_number_or_plural("".join(tokens[first:end]))


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


def _constant_name_findings(source: str) -> Iterator[tuple[int, str, str]]:
    for match in _CONSTANT_RESULT_PATTERN.finditer(source):
        if _is_spelled_number_constant_name(match.group("name")):
            yield match.start("result"), match.group("result"), CONSTANT_NAME_RULE


def _generated_name_findings(source: str) -> Iterator[tuple[int, str, str]]:
    if "$" not in source:
        return
    for match in _NAME_PATTERN.finditer(source):
        if "$" in match.group():
            yield match.start(), match.group(), GENERATED_NAME_RULE


def lint_source(path: Path, text: str) -> list[Finding]:
    """Returns authoring-policy findings for one supported source."""

    masked_source = _mask_authored_source(path, text)
    findings: list[Finding] = []
    line = 1
    line_start = 0
    next_newline = masked_source.find("\n")
    # Both rule streams are source-ordered. Merge and advance the line cursor
    # once instead of rescanning the file prefix for every diagnostic.
    matches = merge(
        _constant_name_findings(masked_source),
        _generated_name_findings(masked_source),
        key=lambda match: match[0],
    )
    for source_offset, name, rule in matches:
        while next_newline != -1 and next_newline < source_offset:
            line += 1
            line_start = next_newline + 1
            next_newline = masked_source.find("\n", line_start)
        findings.append(
            Finding(
                path=path,
                line=line,
                column=source_offset - line_start + 1,
                name=name,
                rule=rule,
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
