# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

import pytest

from loom.diagnostics import DiagnosticSeverity
from loom.format.text.tokenizer import Token, Tokenizer, TokenKind
from loom.migration.source import (
    MigrationSourceDiagnostic,
    SourceDocument,
    SourceEdit,
)


def _source_text_for_token(document: SourceDocument, token: Token) -> str:
    source_range = document.token_source_range(token)
    source_bytes = document.text.encode("utf-8")
    return source_bytes[source_range.start : source_range.end].decode("utf-8")


def test_token_source_range_maps_character_offsets_to_utf8_bytes() -> None:
    document = SourceDocument("µµ target", Path("model.loom"))
    tokenizer = Tokenizer(document.text, str(document.filename))

    token = tokenizer.next()
    assert token.text == "µµ"

    token = tokenizer.next()
    source_range = document.token_source_range(token)

    assert token.text == "target"
    assert source_range.filename == Path("model.loom")
    assert source_range.start == 5
    assert source_range.end == 11
    assert source_range.start_line == 1
    assert source_range.start_column == 4
    assert source_range.end_line == 1
    assert source_range.end_column == 10


def test_byte_source_range_maps_utf8_offsets_to_line_columns() -> None:
    document = SourceDocument("µ first\n  second\n", Path("model.loom"))
    byte_start = len("µ first\n  ".encode())
    byte_end = byte_start + len(b"second")

    source_range = document.byte_source_range(byte_start, byte_end)

    assert source_range.filename == Path("model.loom")
    assert source_range.start == byte_start
    assert source_range.end == byte_end
    assert source_range.start_line == 2
    assert source_range.start_column == 3
    assert source_range.end_line == 2
    assert source_range.end_column == 9


def test_token_ranges_cover_operation_syntax_categories() -> None:
    text = "%r = scalar.addi %lhs, %rhs {rounding = fast} : i32"
    document = SourceDocument(text, Path("model.loom"))
    tokenizer = Tokenizer(document.text, str(document.filename))
    tokens: list[Token] = []
    while not tokenizer.at(TokenKind.EOF):
        tokens.append(tokenizer.next())

    interesting_tokens = {
        "result": tokens[0],
        "op": tokens[2],
        "lhs": tokens[3],
        "rhs": tokens[5],
        "attr_name": tokens[7],
        "attr_value": tokens[9],
        "type": tokens[12],
    }

    assert {
        label: _source_text_for_token(document, token)
        for label, token in interesting_tokens.items()
    } == {
        "result": "%r",
        "op": "scalar.addi",
        "lhs": "%lhs",
        "rhs": "%rhs",
        "attr_name": "rounding",
        "attr_value": "fast",
        "type": "i32",
    }


def test_same_line_edits_apply_in_order() -> None:
    document = SourceDocument("alpha beta gamma")
    edits = (
        SourceEdit(byte_start=6, byte_end=10, replacement_text="BETA"),
        SourceEdit(byte_start=11, byte_end=16, replacement_text="GAMMA"),
    )

    assert document.apply_edits(edits) == "alpha BETA GAMMA"


def test_multiline_edit_preserves_adjacent_comments() -> None:
    document = SourceDocument("func.def @f {\n  // keep\n  old.op\n}\n")
    byte_start = document.text.index("old.op")
    byte_end = byte_start + len("old.op")

    result = document.apply_edits(
        (SourceEdit(byte_start, byte_end, "new.op", rule_id="rename-op"),)
    )

    assert result == "func.def @f {\n  // keep\n  new.op\n}\n"


def test_utf8_text_before_edit_does_not_shift_byte_range() -> None:
    document = SourceDocument("µµ target")
    byte_start = document.byte_offset_at(Tokenizer(document.text).next().end_location)

    result = document.apply_edits(
        (
            SourceEdit(
                byte_start=byte_start + 1,
                byte_end=document.byte_length,
                replacement_text="done",
            ),
        )
    )

    assert result == "µµ done"


def test_adjacent_edits_are_allowed() -> None:
    document = SourceDocument("abcdef")
    edits = (
        SourceEdit(1, 3, "BC"),
        SourceEdit(3, 5, "DE"),
    )

    assert document.apply_edits(edits) == "aBCDEf"


def test_overlapping_edits_are_rejected() -> None:
    document = SourceDocument("abcdef")
    edits = (
        SourceEdit(1, 4, "BCD"),
        SourceEdit(3, 5, "DE"),
    )

    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        document.apply_edits(edits)


def test_unordered_edits_are_rejected() -> None:
    document = SourceDocument("abcdef")
    edits = (
        SourceEdit(4, 5, "E"),
        SourceEdit(1, 2, "B"),
    )

    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        document.apply_edits(edits)


def test_edit_must_align_to_utf8_boundary() -> None:
    document = SourceDocument("µx")

    with pytest.raises(ValueError, match="UTF-8 boundary"):
        document.apply_edits((SourceEdit(1, 2, "u"),))


def test_migration_source_diagnostic_carries_rule_and_fixup_hint() -> None:
    document = SourceDocument("old.op", Path("model.loom"))
    token = Tokenizer(document.text).next()
    diagnostic = MigrationSourceDiagnostic(
        severity=DiagnosticSeverity.WARNING,
        message="old syntax",
        source_range=document.token_source_range(token),
        rule_id="rename-old-op",
        fixup_hint="replace old.op with new.op",
    )

    assert diagnostic.rule_id == "rename-old-op"
    assert diagnostic.fixup_hint == "replace old.op with new.op"
    assert diagnostic.source_range.display() == "model.loom:1:1"
