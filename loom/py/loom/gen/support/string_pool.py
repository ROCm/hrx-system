# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C string-pool helpers shared by Loom generators."""

from __future__ import annotations

from dataclasses import dataclass, field

from loom.gen.support.c import CIdentifierCase, c_identifier, c_string_literal


@dataclass(frozen=True, slots=True)
class CStringEntry:
    """One unique string payload in a generated C string table."""

    label: str
    value: str


@dataclass(slots=True)
class CStringPool:
    """Interns strings and assigns stable labels independently of byte layout."""

    c_enum_prefix: str
    max_payload_length: int | None = 255
    entries: list[CStringEntry] = field(default_factory=list)
    value_to_label: dict[str, str] = field(default_factory=dict)
    label_to_primary: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def canonical_label(label: str) -> str:
        """Returns the canonical C enum label for an arbitrary string label."""
        return c_identifier(label, case=CIdentifierCase.LOWER, empty="empty")

    def intern(self, label: str, value: str) -> str:
        """Interns |value| with |label| and returns the primary canonical label."""
        if self.max_payload_length is not None and len(value.encode()) > self.max_payload_length:
            raise ValueError(f"string '{value}' exceeds {self.max_payload_length} bytes")
        label = self.canonical_label(label)
        if label in self.label_to_primary:
            primary_label = self.label_to_primary[label]
            if self.value_to_label.get(value) != primary_label:
                raise ValueError(f"string label '{label}' was reused for different values")
            return primary_label
        if value in self.value_to_label:
            primary_label = self.value_to_label[value]
            self.label_to_primary[label] = primary_label
            return primary_label
        self.entries.append(CStringEntry(label, value))
        self.value_to_label[value] = label
        self.label_to_primary[label] = label
        return label

    def enum_name(self, label: str) -> str:
        """Returns the generated C enum name for an interned string label."""
        primary_label = self.label_to_primary[self.canonical_label(label)]
        return f"{self.c_enum_prefix}_STRING_{primary_label}"

    def ref(self, label: str) -> str:
        """Returns a C expression naming the interned string reference enum."""
        return self.enum_name(label)


@dataclass(frozen=True, slots=True)
class CStringLayout:
    """Packed payload and 24-bit-offset/8-bit-length references by primary label."""

    data: bytes
    references: dict[str, int]


def _slice_reference(offset: int, length: int) -> int:
    """Checks the C slice encoding, including the complete payload bound."""
    if not 0 <= length <= 255:
        raise ValueError("string slice exceeds 255 bytes")
    if not 0 <= offset < 1 << 24 or offset + length > 1 << 24:
        raise ValueError("string pool exceeds 24-bit slice offset encoding")
    return (length << 24) | offset


def layout_c_string_pool(pool: CStringPool) -> CStringLayout:
    """Shares contiguous substrings, with deterministic longest-first placement."""
    data = bytearray()
    references: dict[str, int] = {}
    payloads = [(entry.label, entry.value.encode()) for entry in pool.entries]
    for label, payload in sorted(payloads, key=lambda entry: (-len(entry[1]), entry[1])):
        length = len(payload)
        offset = data.find(payload)
        if offset < 0:
            offset = len(data)
            reference = _slice_reference(offset, length)
            data.extend(payload)
        else:
            reference = _slice_reference(offset, length)
        references[label] = reference
    return CStringLayout(bytes(data), references)


def emit_c_string_pool(pool: CStringPool, data_name: str) -> list[str]:
    """Emits shared character bytes and constant compact slice references."""
    if not pool.entries:
        return []
    layout = layout_c_string_pool(pool)
    text = layout.data.decode()
    lines = [
        "// clang-format off",
        f"static const char {data_name}[] =",
    ]
    # Split by codepoints so each source literal remains valid UTF-8. References
    # count encoded bytes, independently of source formatting and escapes.
    for start in range(0, max(1, len(text)), 80):
        escaped = c_string_literal(text[start : start + 80]).replace("\0", "\\000")
        lines.append(f'    "{escaped}"')
    lines[-1] += ";"
    lines.extend(["// clang-format on", "", "enum {"])
    for entry in pool.entries:
        reference = layout.references[entry.label]
        lines.append(f"  {pool.enum_name(entry.label)} = 0x{reference:08X}u,")
    lines.extend(
        [
            "};",
            f'static_assert(sizeof({data_name}) - 1 == {len(layout.data)}, "string pool byte length must match generated slices");',
            "",
        ]
    )
    return lines
