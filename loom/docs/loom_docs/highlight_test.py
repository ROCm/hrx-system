# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for TextMate-backed Loom documentation highlighting."""

from __future__ import annotations

import unittest

import markdown
from pygments.token import Keyword, Name, Punctuation

from loom_docs.highlight import LoomLexer, LoomTestLexer


def _token_pairs(
    lexer: LoomLexer | LoomTestLexer, source: str
) -> list[tuple[object, str]]:
    return [(token, text) for _, token, text in lexer.get_tokens_unprocessed(source)]


class HighlightTest(unittest.TestCase):
    def test_loom_grammar_colors_operation_and_ssa_names(self) -> None:
        tokens = _token_pairs(
            LoomLexer(),
            "%result = vector.load %source[%index] : vector<4xf32>\n",
        )

        self.assertIn((Name.Namespace, "vector"), tokens)
        self.assertIn((Name.Function, ".load"), tokens)
        self.assertIn((Punctuation, "%"), tokens)
        self.assertIn((Name.Variable, "result"), tokens)

    def test_loom_test_grammar_colors_directives_and_embedded_source(self) -> None:
        tokens = _token_pairs(
            LoomTestLexer(),
            "// RUN: verify\n%result = vector.load %source[%index] : vector<4xf32>\n",
        )

        self.assertIn((Keyword, "RUN"), tokens)
        self.assertIn((Name.Builtin, "verify"), tokens)
        self.assertIn((Name.Namespace, "vector"), tokens)
        self.assertIn((Name.Variable, "result"), tokens)

    def test_markdown_extension_preserves_material_highlight_markup(self) -> None:
        html = markdown.markdown(
            "```loom\n%result = vector.load %source[%index] : vector<4xf32>\n```",
            extensions=[
                "loom_docs.highlight",
                "pymdownx.superfences",
            ],
            extension_configs={
                "loom_docs.highlight": {
                    "line_spans": "__span",
                    "pygments_lang_class": True,
                }
            },
        )

        self.assertIn('class="language-loom highlight"', html)
        self.assertIn('<span class="nn">vector</span>', html)
        self.assertIn('<span class="nf">.load</span>', html)
        self.assertNotIn('class="language-text highlight"', html)


if __name__ == "__main__":
    unittest.main()
