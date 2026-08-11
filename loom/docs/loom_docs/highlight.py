# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Highlights Loom source with the checked-in TextMate grammars."""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pygments.lexer import Lexer
from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Text,
)
from pymdownx.highlight import Highlight, HighlightExtension
from textmate_grammar.elements import ContentElement
from textmate_grammar.parsers.base import LanguageParser

REPO_ROOT = Path(__file__).resolve().parents[3]
GRAMMAR_ROOT = REPO_ROOT / "loom" / "src" / "loom" / "editor" / "textmate"

_GRAMMAR_PATHS = {
    "source.loom": GRAMMAR_ROOT / "loom.tmLanguage.json",
    "source.loom-test": GRAMMAR_ROOT / "loom-test.tmLanguage.json",
}


class _TextMateDiagnosticFilter(logging.Filter):
    """Drops diagnostics for source that a lexical grammar leaves unscoped."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return (
            "skipping <" not in message
            and "remainder of line not parsed" not in message
        )


# Unscoped fragments are ordinary in a lexical grammar and are emitted as plain
# text below. Other parser diagnostics remain visible, and grammar-loading or
# tokenization failures propagate out of the documentation build.
logging.getLogger("textmate_grammar").addFilter(_TextMateDiagnosticFilter())


def _load_grammar(path: Path, expected_scope: str) -> dict[str, Any]:
    grammar = json.loads(path.read_text(encoding="utf-8"))
    actual_scope = grammar.get("scopeName")
    if actual_scope != expected_scope:
        raise ValueError(
            f"TextMate grammar {path} has scope {actual_scope!r}; "
            f"expected {expected_scope!r}"
        )
    return grammar


@functools.cache
def _language_parsers() -> dict[str, LanguageParser]:
    # The test grammar includes source.loom by scope name. Register the base
    # grammar first so the TextMate repository resolves that include directly.
    loom_parser = LanguageParser(
        _load_grammar(_GRAMMAR_PATHS["source.loom"], "source.loom")
    )
    loom_test_parser = LanguageParser(
        _load_grammar(_GRAMMAR_PATHS["source.loom-test"], "source.loom-test")
    )
    return {
        "source.loom": loom_parser,
        "source.loom-test": loom_test_parser,
    }


def _pygments_token(scope: str) -> Any:
    """Maps a TextMate scope to the site's existing Pygments color roles."""

    mappings = (
        ("invalid.", Error),
        ("comment.", Comment),
        ("constant.character.escape.", String.Escape),
        ("constant.language.", Keyword.Constant),
        ("constant.numeric.float.", Number.Float),
        ("constant.numeric.integer.hex.", Number.Hex),
        ("constant.numeric.", Number),
        ("constant.other.", Name.Constant),
        ("string.", String),
        ("keyword.operator.", Operator),
        ("keyword.", Keyword),
        ("support.namespace.", Name.Namespace),
        ("support.function.", Name.Builtin),
        ("support.type.", Keyword.Type),
        ("entity.name.function.", Name.Function),
        ("entity.name.label.", Name.Label),
        ("entity.name.section.", Generic.Heading),
        ("entity.other.attribute-name.", Name.Attribute),
        ("variable.other.attribute.", Name.Attribute),
        ("variable.other.symbol.", Name.Function),
        ("variable.other.", Name.Variable),
        ("punctuation.", Punctuation),
        ("meta.operation-name.", Name.Function),
        ("meta.separator.", Punctuation),
    )
    for prefix, token in mappings:
        if scope.startswith(prefix):
            return token
    return Text


def _line_offsets(source: str) -> list[int]:
    offsets = [0]
    offsets.extend(
        index + 1 for index, character in enumerate(source) if character == "\n"
    )
    return offsets


def _paint_element(
    element: ContentElement,
    source_length: int,
    line_offsets: list[int],
    tokens: list[Any],
) -> None:
    token = _pygments_token(element.token)
    for line, column in element.characters:
        if line >= len(line_offsets):
            raise ValueError(
                f"TextMate scope {element.token} produced invalid line {line}"
            )
        offset = line_offsets[line] + column
        if offset >= source_length:
            raise ValueError(
                f"TextMate scope {element.token} produced invalid column {column} "
                f"on line {line}"
            )
        tokens[offset] = token
    for child in element.children:
        _paint_element(child, source_length, line_offsets, tokens)


class _TextMateLoomLexer(Lexer):
    """Pygments presentation adapter over one checked-in TextMate grammar."""

    grammar_scope = ""

    def get_tokens_unprocessed(self, text: str) -> Iterator[tuple[int, Any, str]]:
        if not text:
            return

        root = _language_parsers()[self.grammar_scope].parse_string(text)
        if root is None:
            raise ValueError(
                f"TextMate grammar {self.grammar_scope} did not parse input"
            )

        tokens = [Text] * len(text)
        line_offsets = _line_offsets(text)
        for child in root.children:
            _paint_element(child, len(text), line_offsets, tokens)

        run_start = 0
        run_token = tokens[0]
        for index, token in enumerate(tokens[1:], start=1):
            if token != run_token:
                yield run_start, run_token, text[run_start:index]
                run_start = index
                run_token = token
        yield run_start, run_token, text[run_start:]


class LoomLexer(_TextMateLoomLexer):
    """Highlights ordinary Loom source."""

    name = "Loom"
    aliases = ["loom"]
    filenames = ["*.loom"]
    grammar_scope = "source.loom"


class LoomTestLexer(_TextMateLoomLexer):
    """Highlights Loom source with textual test directives."""

    name = "Loom Test"
    aliases = ["loom-test"]
    filenames = ["*.loom-test"]
    grammar_scope = "source.loom-test"


_LEXERS = {
    "loom": LoomLexer,
    "loom-test": LoomTestLexer,
}


class _LoomHighlight(Highlight):
    """Selects Loom's TextMate-backed lexers before Pygments lookup."""

    def get_lexer(
        self, src: str, language: str, inline: bool, stripnl: bool
    ) -> tuple[Lexer, str]:
        lexer_type = _LEXERS.get(language.lower()) if language else None
        if lexer_type is not None:
            return lexer_type(stripnl=stripnl), language.lower()
        return super().get_lexer(src, language, inline, stripnl)


class _LoomHighlightExtension(HighlightExtension):
    """Installs the Loom-aware highlighter into PyMdown Extensions."""

    def get_pymdownx_highlighter(self) -> type[Highlight]:
        return _LoomHighlight


def makeExtension(*args: Any, **kwargs: Any) -> HighlightExtension:
    """Creates the Python-Markdown extension."""

    return _LoomHighlightExtension(*args, **kwargs)
