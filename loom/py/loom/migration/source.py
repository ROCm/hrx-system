# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Source ranges and edit application primitives for text migrations."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path

from loom.diagnostics import DiagnosticSeverity, SourceProvenance, SourceRange
from loom.format.text.tokenizer import SourceLocation, Token


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """Source text plus a UTF-8 byte offset index.

    The text tokenizer tracks offsets in Python string characters so that
    line/column math stays natural. Migrations write files as UTF-8 bytes, so
    the source document owns the exact character-offset to byte-offset map used
    for precision edits.
    """

    text: str
    filename: Path | None = None
    _character_byte_offsets: tuple[int, ...] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        byte_offsets = [0]
        byte_offset = 0
        for character in self.text:
            byte_offset += len(character.encode("utf-8"))
            byte_offsets.append(byte_offset)
        object.__setattr__(self, "_character_byte_offsets", tuple(byte_offsets))

    @property
    def byte_length(self) -> int:
        """Returns the total UTF-8 byte length of the source text."""
        return self._character_byte_offsets[-1]

    def byte_offset_at(self, location: SourceLocation) -> int:
        """Maps a tokenizer source location to a UTF-8 byte offset."""
        if location.offset < 0 or location.offset > len(self.text):
            raise ValueError(
                f"source location offset {location.offset} is outside the source"
            )
        return self._character_byte_offsets[location.offset]

    def is_byte_boundary(self, byte_offset: int) -> bool:
        """Returns true if byte_offset is a valid UTF-8 character boundary."""
        index = bisect_left(self._character_byte_offsets, byte_offset)
        return (
            index < len(self._character_byte_offsets)
            and self._character_byte_offsets[index] == byte_offset
        )

    def source_location_at_byte_offset(self, byte_offset: int) -> SourceLocation:
        """Maps a UTF-8 byte offset to a tokenizer source location."""
        if not self.is_byte_boundary(byte_offset):
            raise ValueError("byte offset must align to a UTF-8 boundary")
        character_offset = bisect_left(self._character_byte_offsets, byte_offset)
        prefix = self.text[:character_offset]
        line = prefix.count("\n") + 1
        line_start = prefix.rfind("\n")
        column = (
            character_offset + 1 if line_start < 0 else character_offset - line_start
        )
        return SourceLocation(line=line, column=column, offset=character_offset)

    def source_range(
        self,
        start_location: SourceLocation,
        end_location: SourceLocation,
    ) -> SourceRange:
        """Builds a byte-backed diagnostic source range from token locations."""
        if start_location.offset > end_location.offset:
            raise ValueError("source range start must not be after end")
        return SourceRange(
            provenance=SourceProvenance.EXACT_SOURCE,
            filename=self.filename,
            source=self.text,
            start=self.byte_offset_at(start_location),
            end=self.byte_offset_at(end_location),
            start_line=start_location.line,
            start_column=start_location.column,
            end_line=end_location.line,
            end_column=end_location.column,
        )

    def token_source_range(self, token: Token) -> SourceRange:
        """Returns the byte-backed source range for a tokenizer token."""
        return self.source_range(token.location, token.end_location)

    def byte_source_range(self, byte_start: int, byte_end: int) -> SourceRange:
        """Builds a source range from half-open UTF-8 byte offsets."""
        if byte_start > byte_end:
            raise ValueError("source range start must not be after end")
        return self.source_range(
            self.source_location_at_byte_offset(byte_start),
            self.source_location_at_byte_offset(byte_end),
        )

    def apply_edits(self, edits: tuple[SourceEdit, ...]) -> str:
        """Applies ordered, non-overlapping UTF-8 byte replacements."""
        return apply_source_edits(self, edits)


@dataclass(frozen=True, slots=True)
class SourceEdit:
    """One half-open byte range replacement in a source document."""

    byte_start: int
    byte_end: int
    replacement_text: str
    rule_id: str | None = None


@dataclass(frozen=True, slots=True)
class MigrationSourceDiagnostic:
    """Structured diagnostic produced by a source migration rule."""

    severity: DiagnosticSeverity
    message: str
    source_range: SourceRange
    rule_id: str
    fixup_hint: str = ""


def apply_source_edits(document: SourceDocument, edits: tuple[SourceEdit, ...]) -> str:
    """Applies ordered, non-overlapping edits to a UTF-8 source document."""
    source_bytes = document.text.encode("utf-8")
    result = bytearray()
    previous_byte_end = 0

    for edit in edits:
        _validate_edit(document, edit)
        if edit.byte_start < previous_byte_end:
            raise ValueError("source edits must be ordered and non-overlapping")
        result.extend(source_bytes[previous_byte_end : edit.byte_start])
        result.extend(edit.replacement_text.encode("utf-8"))
        previous_byte_end = edit.byte_end

    result.extend(source_bytes[previous_byte_end:])
    return result.decode("utf-8")


def _validate_edit(document: SourceDocument, edit: SourceEdit) -> None:
    if edit.byte_start < 0:
        raise ValueError("source edit start must be non-negative")
    if edit.byte_end < edit.byte_start:
        raise ValueError("source edit end must not be before start")
    if edit.byte_end > document.byte_length:
        raise ValueError("source edit end is outside the source")
    if not document.is_byte_boundary(edit.byte_start):
        raise ValueError("source edit start must align to a UTF-8 boundary")
    if not document.is_byte_boundary(edit.byte_end):
        raise ValueError("source edit end must align to a UTF-8 boundary")
