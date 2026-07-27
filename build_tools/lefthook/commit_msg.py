#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validates Git commit message text."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]

COMMENT_PREFIX = "#"
SCISSORS_MARKER = ">8"
LINE_LENGTH_LIMIT = 72

LITERAL_ESCAPE_PATTERN = re.compile(r"\\[nr]")
BEAD_ID_PATTERN = re.compile(
    r"\b(?P<id>(?:bd-[0-9a-z]+|loom-[0-9a-z]*[0-9][0-9a-z]*)(?:\.\d+)?)\b",
    re.IGNORECASE,
)
SUBJECT_TAG_PATTERN = re.compile(
    r"^\[(?P<tag>[A-Za-z][A-Za-z0-9]*(?:/[A-Za-z][A-Za-z0-9]*)?)\](?:\s+|$)"
)
AUTOSQUASH_SUBJECT_PREFIX_PATTERN = re.compile(r"^(?P<prefix>fixup|squash|amend)!\s+")
CODE_FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")
INDENTED_CODE_PATTERN = re.compile(r"^(?: {4,}|\t)")
LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
MARKDOWN_TABLE_PATTERN = re.compile(r"^\s*\|.*\|\s*$")
URL_PATTERN = re.compile(r"\b(?:[a-z][a-z0-9+.-]*://|www\.)", re.IGNORECASE)
TRAILER_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9-]*: .+")
TAG_EXAMPLES = ("[Loom]", "[HRX]", "[HAL]", "[Runtime]", "[Infra]", "[CI]")


@dataclass(frozen=True)
class TagSuggestionRule:
    tag: str
    score: int
    path_prefixes: tuple[str, ...]
    exact_paths: tuple[str, ...] = ()


TAG_SUGGESTION_RULES = (
    TagSuggestionRule(
        tag="[HAL/AMDGPU]",
        score=40,
        path_prefixes=(
            "runtime/src/iree/hal/drivers/amdgpu/",
            "runtime/src/iree/hal/drivers/hip/",
            "build_tools/amdgpu/",
        ),
    ),
    TagSuggestionRule(
        tag="[Loom/AMDGPU]",
        score=38,
        path_prefixes=(
            "loom/src/loom/target/arch/amdgpu/",
            "loom/py/loom/target/arch/amdgpu/",
        ),
    ),
    TagSuggestionRule(
        tag="[Runtime/VM]",
        score=36,
        path_prefixes=("runtime/src/iree/vm/",),
    ),
    TagSuggestionRule(
        tag="[Loom]",
        score=30,
        path_prefixes=("loom/",),
    ),
    TagSuggestionRule(
        tag="[HRX]",
        score=30,
        path_prefixes=("libhrx/",),
    ),
    TagSuggestionRule(
        tag="[HAL]",
        score=28,
        path_prefixes=("runtime/src/iree/hal/",),
    ),
    TagSuggestionRule(
        tag="[Runtime]",
        score=24,
        path_prefixes=("runtime/src/iree/",),
    ),
    TagSuggestionRule(
        tag="[CI]",
        score=22,
        path_prefixes=(".github/workflows/", "build_tools/devtools/ci"),
    ),
    TagSuggestionRule(
        tag="[Infra]",
        score=20,
        path_prefixes=(
            ".github/",
            "build_tools/",
            "requirements",
            "third_party/",
        ),
        exact_paths=(
            ".bazelrc",
            ".bazelversion",
            "AGENTS.override.md",
            "BUILD.bazel",
            "CONTRIBUTING.md",
            "MODULE.bazel",
            "MODULE.bazel.lock",
            "README.md",
            "WORKSPACE",
            "WORKSPACE.bazel",
            "WORKSPACE.bzlmod",
            "dev.py",
            "lefthook.yml",
        ),
    ),
)


@dataclass(frozen=True)
class CommitMessageDiagnostic:
    line_number: int
    message: str
    text: str
    hint: str = ""
    kind: str = "policy"
    autofixable: bool = False


def staged_commit_paths() -> list[str]:
    """Returns paths staged for the commit being validated."""

    output = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMRT"],
        cwd=REPO_ROOT,
        text=True,
    )
    return [line for line in output.splitlines() if line]


def commit_message_text_lines(message_text: str) -> list[tuple[int, str]]:
    """Returns user-authored commit message lines with Git comments removed."""

    lines: list[tuple[int, str]] = []
    for line_number, line in enumerate(message_text.splitlines(), start=1):
        stripped = line.lstrip()
        if stripped.startswith(COMMENT_PREFIX):
            if SCISSORS_MARKER in stripped:
                break
            continue
        lines.append((line_number, line))
    return lines


def commit_message_user_text(message_text: str) -> str:
    """Returns the commit message text that Git would commit."""

    lines = [line for _, line in commit_message_text_lines(message_text)]
    return "\n".join(lines).rstrip() + "\n"


def tag_suggestions_for_paths(paths: Sequence[str]) -> list[str]:
    """Returns ranked subsystem tag suggestions for changed paths."""

    scores: dict[str, int] = {}
    for path in paths:
        for rule in TAG_SUGGESTION_RULES:
            if path in rule.exact_paths or any(
                path.startswith(prefix) for prefix in rule.path_prefixes
            ):
                scores[rule.tag] = scores.get(rule.tag, 0) + rule.score
    if not scores:
        return list(TAG_EXAMPLES)

    order = {rule.tag: index for index, rule in enumerate(TAG_SUGGESTION_RULES)}
    return sorted(scores, key=lambda tag: (-scores[tag], order.get(tag, 999)))[:5]


def tag_hint(paths: Sequence[str]) -> str:
    suggestions = tag_suggestions_for_paths(paths)
    hint = (
        "You need to start the subject with a [Tag], such as "
        f"{', '.join(TAG_EXAMPLES)}. Suggested tag"
        f"{'' if len(suggestions) == 1 else 's'} for the staged paths: "
        f"{', '.join(suggestions)}."
    )
    if paths:
        shown_paths = list(paths[:5])
        path_list = ", ".join(shown_paths)
        if len(paths) > len(shown_paths):
            path_list += f", and {len(paths) - len(shown_paths)} more"
        hint += f" Staged paths considered: {path_list}."
    return hint


def unwrap_autosquash_subject(subject: str) -> tuple[str, bool]:
    """Returns the subject after removing Git autosquash prefixes."""

    unwrapped_subject = subject
    is_autosquash_subject = False
    while True:
        prefix_match = AUTOSQUASH_SUBJECT_PREFIX_PATTERN.match(unwrapped_subject)
        if prefix_match is None:
            return unwrapped_subject, is_autosquash_subject
        is_autosquash_subject = True
        unwrapped_subject = unwrapped_subject[prefix_match.end() :]


def trailer_line_numbers(text_lines: Sequence[tuple[int, str]]) -> set[int]:
    """Returns line numbers that are part of a final Git trailer block."""

    trailer_lines: set[int] = set()
    continuation_lines: list[int] = []
    index = len(text_lines) - 1
    while index >= 0 and text_lines[index][1] == "":
        index -= 1

    while index >= 0:
        line_number, line = text_lines[index]
        if line.startswith((" ", "\t")):
            continuation_lines.append(line_number)
            index -= 1
            continue
        if TRAILER_PATTERN.match(line):
            trailer_lines.add(line_number)
            trailer_lines.update(continuation_lines)
            continuation_lines.clear()
            index -= 1
            continue
        break
    return trailer_lines


def is_markdown_table_line(line: str) -> bool:
    if MARKDOWN_TABLE_PATTERN.match(line):
        return True
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def is_line_length_exempt(line_number: int, line: str, trailer_lines: set[int]) -> bool:
    """Returns whether line length should be preserved instead of checked."""

    return (
        line_number in trailer_lines
        or INDENTED_CODE_PATTERN.match(line) is not None
        or is_markdown_table_line(line)
        or URL_PATTERN.search(line) is not None
    )


def is_reflowable_body_line(
    line_number: int,
    line: str,
    trailer_lines: set[int],
) -> bool:
    """Returns whether a body line is ordinary prose safe to wrap."""

    if not line:
        return False
    if is_line_length_exempt(line_number, line, trailer_lines):
        return False
    if LIST_ITEM_PATTERN.match(line):
        return False
    if line.lstrip().startswith(">"):
        return False
    return True


def reflow_commit_message_text(message_text: str) -> str:
    """Reflows ordinary prose paragraphs while preserving structured lines."""

    text_lines = commit_message_text_lines(message_text)
    if not text_lines:
        return "\n"

    subject_line_number = text_lines[0][0]
    trailer_lines = trailer_line_numbers(text_lines)
    output_lines: list[str] = []
    paragraph_lines: list[str] = []
    in_code_block = False

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph_text = " ".join(line.strip() for line in paragraph_lines)
        output_lines.extend(
            textwrap.wrap(
                paragraph_text,
                width=LINE_LENGTH_LIMIT,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )
        paragraph_lines.clear()

    for line_number, line in text_lines:
        is_code_fence = CODE_FENCE_PATTERN.match(line) is not None
        if is_code_fence:
            flush_paragraph()
            output_lines.append(line)
            in_code_block = not in_code_block
            continue

        if (
            in_code_block
            or line_number == subject_line_number
            or not is_reflowable_body_line(line_number, line, trailer_lines)
        ):
            flush_paragraph()
            output_lines.append(line)
            continue

        paragraph_lines.append(line)

    flush_paragraph()
    return "\n".join(output_lines).rstrip() + "\n"


def validate_subject_line(
    line_number: int,
    subject: str,
    changed_paths: Sequence[str],
) -> list[CommitMessageDiagnostic]:
    diagnostics: list[CommitMessageDiagnostic] = []
    if not subject:
        diagnostics.append(
            CommitMessageDiagnostic(
                line_number=line_number,
                message="commit message must start with a non-empty subject line",
                text=subject,
                hint=(
                    "Use a short subject such as "
                    f"{tag_suggestions_for_paths(changed_paths)[0]} Describe the change."
                ),
            )
        )
        return diagnostics

    policy_subject, is_autosquash_subject = unwrap_autosquash_subject(subject)

    if len(policy_subject) > LINE_LENGTH_LIMIT:
        diagnostics.append(
            CommitMessageDiagnostic(
                line_number=line_number,
                message=(
                    f"subject line is {len(policy_subject)} characters; keep it at or "
                    f"below {LINE_LENGTH_LIMIT}"
                ),
                text=policy_subject,
                hint="Move detail into the body and keep the subject crisp.",
            )
        )

    tag_match = SUBJECT_TAG_PATTERN.match(policy_subject)
    if tag_match is None:
        diagnostics.append(
            CommitMessageDiagnostic(
                line_number=line_number,
                message=(
                    "autosquash target subject must start with a bracketed "
                    "subsystem tag"
                    if is_autosquash_subject
                    else "subject line must start with a bracketed subsystem tag"
                ),
                text=policy_subject,
                hint=tag_hint(changed_paths),
            )
        )
    elif policy_subject[tag_match.end() :].strip() == "":
        diagnostics.append(
            CommitMessageDiagnostic(
                line_number=line_number,
                message="subject line must include a description after the tag",
                text=subject,
                hint=(
                    f"Write the subject as {tag_match.group(0).strip()} "
                    "Describe the change."
                ),
            )
        )
    return diagnostics


def validate_line_lengths(
    text_lines: Sequence[tuple[int, str]],
    subject_line_number: int,
) -> list[CommitMessageDiagnostic]:
    diagnostics: list[CommitMessageDiagnostic] = []
    in_code_block = False
    trailer_lines = trailer_line_numbers(text_lines)
    for line_number, line in text_lines:
        if CODE_FENCE_PATTERN.match(line):
            in_code_block = not in_code_block
            continue
        if (
            in_code_block
            or line_number == subject_line_number
            or not line
            or is_line_length_exempt(line_number, line, trailer_lines)
        ):
            continue
        if len(line) > LINE_LENGTH_LIMIT:
            is_autofixable = is_reflowable_body_line(line_number, line, trailer_lines)
            hint = (
                "Preserve the message content and reflow prose instead of "
                "shortening, fragmenting, or rewriting it."
            )
            if not is_autofixable:
                hint = (
                    "Wrap this line manually. The prose reflow suggestion "
                    "preserves structured lines such as lists, examples, and "
                    "tables."
                )
            diagnostics.append(
                CommitMessageDiagnostic(
                    line_number=line_number,
                    message=(
                        f"commit message body line is {len(line)} characters; "
                        f"wrap prose to {LINE_LENGTH_LIMIT} columns"
                    ),
                    text=line,
                    hint=hint,
                    kind="line-length",
                    autofixable=is_autofixable,
                )
            )
    return diagnostics


def validate_commit_message_text(
    message_text: str,
    changed_paths: Sequence[str] = (),
) -> list[CommitMessageDiagnostic]:
    """Returns diagnostics for commit-message policy violations."""

    diagnostics: list[CommitMessageDiagnostic] = []
    text_lines = commit_message_text_lines(message_text)
    if not text_lines:
        diagnostics.append(
            CommitMessageDiagnostic(
                line_number=1,
                message="commit message must start with a non-empty subject line",
                text="",
                hint=(
                    "Use a short subject such as "
                    f"{tag_suggestions_for_paths(changed_paths)[0]} Describe the change."
                ),
            )
        )
        return diagnostics

    subject_line_number, subject = text_lines[0]
    diagnostics.extend(
        validate_subject_line(subject_line_number, subject, changed_paths)
    )
    diagnostics.extend(validate_line_lengths(text_lines, subject_line_number))

    for line_number, line in text_lines:
        literal_escape = LITERAL_ESCAPE_PATTERN.search(line)
        if literal_escape:
            diagnostics.append(
                CommitMessageDiagnostic(
                    line_number=line_number,
                    message=(
                        "literal newline/carriage-return escape sequences are "
                        "not allowed; use real paragraph breaks"
                    ),
                    text=literal_escape.group(0),
                )
            )
        for match in BEAD_ID_PATTERN.finditer(line):
            diagnostics.append(
                CommitMessageDiagnostic(
                    line_number=line_number,
                    message="bead identifiers must not appear in commit messages",
                    text=match.group("id"),
                )
            )
    return diagnostics


def validate_commit_message_file(
    path: Path,
    changed_paths: Sequence[str] = (),
) -> list[CommitMessageDiagnostic]:
    """Reads and validates a Git commit message file."""

    return validate_commit_message_text(
        path.read_text(encoding="utf-8"),
        changed_paths=changed_paths,
    )


def write_reflow_suggestion(message_file: Path, message_text: str) -> Path | None:
    """Writes a reflowed message suggestion when it changes the user text."""

    reflowed_text = reflow_commit_message_text(message_text)
    if reflowed_text == commit_message_user_text(message_text):
        return None

    suggestion_directory = REPO_ROOT / ".tmp" / "commit-msg"
    suggestion_directory.mkdir(parents=True, exist_ok=True)
    suggestion_path = suggestion_directory / f"{message_file.name}.reflowed"
    suggestion_path.write_text(reflowed_text, encoding="utf-8")
    return suggestion_path


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        action="store_true",
        help=(
            "Print the commit message with ordinary prose paragraphs reflowed "
            "to 72 columns. Structured lines are preserved."
        ),
    )
    parser.add_argument(
        "message_file",
        type=Path,
        help="Git commit message file passed to the commit-msg hook.",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_arguments(argv)
    message_text = args.message_file.read_text(encoding="utf-8")
    if args.format:
        print(reflow_commit_message_text(message_text), end="")
        return 0

    diagnostics = validate_commit_message_file(
        args.message_file,
        changed_paths=staged_commit_paths(),
    )
    if not diagnostics:
        return 0

    print("commit message policy failed:", file=sys.stderr)
    if any(diagnostic.kind == "line-length" for diagnostic in diagnostics):
        print("", file=sys.stderr)
        print(
            f"commit message body lines must be wrapped to {LINE_LENGTH_LIMIT} columns.",
            file=sys.stderr,
        )
        print(
            "Preserve the message content and reflow the prose instead of "
            "shortening, fragmenting, or rewriting it.",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        print("Example:", file=sys.stderr)
        print(
            "  Before: This is a useful commit-message paragraph that is too "
            "long for the Git log body width.",
            file=sys.stderr,
        )
        print(
            "  After:  This is a useful commit-message paragraph that is too",
            file=sys.stderr,
        )
        print("          long for the Git log body width.", file=sys.stderr)
        print("", file=sys.stderr)

    for diagnostic in diagnostics:
        print(
            f"  {args.message_file}:{diagnostic.line_number}: "
            f"{diagnostic.message}: {diagnostic.text!r}",
            file=sys.stderr,
        )
        if diagnostic.hint:
            print(f"    hint: {diagnostic.hint}", file=sys.stderr)

    if any(diagnostic.kind == "line-length" for diagnostic in diagnostics):
        suggestion_path = write_reflow_suggestion(args.message_file, message_text)
        if suggestion_path is not None:
            relative_suggestion_path = suggestion_path.relative_to(REPO_ROOT)
            print("", file=sys.stderr)
            print(
                f"Suggested prose-only reflow written to {relative_suggestion_path}.",
                file=sys.stderr,
            )
            print("Retry a normal commit with:", file=sys.stderr)
            print(
                f"  git commit -F {shlex.quote(str(relative_suggestion_path))}",
                file=sys.stderr,
            )
            if any(
                diagnostic.kind == "line-length" and not diagnostic.autofixable
                for diagnostic in diagnostics
            ):
                print(
                    "Some long lines were preserved because they look "
                    "structured; wrap those diagnostics manually before "
                    "retrying.",
                    file=sys.stderr,
                )
            print(
                "For amend, squash, or fixup commits, rerun the same git "
                "commit command and replace the message source with that -F "
                "argument.",
                file=sys.stderr,
            )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
