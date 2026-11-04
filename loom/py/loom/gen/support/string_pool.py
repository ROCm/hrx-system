# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C string-pool helpers shared by Loom generators."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from loom.gen.support.c import CIdentifierCase, c_identifier, c_string_literal
from loom.gen.support.generated_file import line_comment_header


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
    """Inline bytes and bank/23-bit-offset/8-bit-length references by label."""

    data: bytes
    references: dict[str, int]
    primary: CStringBank | None = None
    shared: CStringBank | None = None

    def initializer(self, data_name: str) -> str:
        """Returns the initializer for the checked-in C pool representation."""
        primary_data = self.primary.symbol if self.primary is not None else data_name
        primary_length = len(self.primary.data) if self.primary is not None else len(self.data)
        shared_data = self.shared.symbol if self.shared is not None else "NULL"
        shared_length = len(self.shared.data) if self.shared is not None else 0
        return f"{{.data = {{{primary_data}, {shared_data}}}, .data_length = {{{primary_length}, {shared_length}}}}}"


@dataclass(frozen=True, slots=True)
class CStringBank:
    """An independently linked immutable byte resource and its producer layout."""

    data: bytes
    values: frozenset[str]
    symbol: str
    header: str

    @staticmethod
    def build(
        values: frozenset[str],
        *,
        symbol: str,
        header: str,
        shared: CStringBank | None = None,
    ) -> CStringBank:
        pool = CStringPool(symbol.upper())
        for index, value in enumerate(sorted(values)):
            pool.intern(f"value_{index}", value)
        layout = layout_c_string_pool(pool, shared)
        primary_values = frozenset(entry.value for entry in pool.entries if not (layout.references[entry.label] >> 23) & 1)
        return CStringBank(layout.data, primary_values, symbol, header)

    def manifest(self) -> str:
        """Carries the producer's layout across independent generator actions."""
        return json.dumps(dict(data=self.data.decode(), values=sorted(self.values), symbol=self.symbol, header=self.header), ensure_ascii=False) + "\n"

    @staticmethod
    def read(path: Path) -> CStringBank:
        payload = json.loads(path.read_text())
        return CStringBank(payload["data"].encode(), frozenset(payload["values"]), payload["symbol"], payload["header"])

    def emit_header(self) -> str:
        guard = c_identifier(self.header, case=CIdentifierCase.UPPER)
        return "\n".join(
            [
                *line_comment_header("//", generator="loom.gen.support.string_pool"),
                "",
                f"#ifndef {guard}",
                f"#define {guard}",
                "",
                "#ifdef __cplusplus",
                'extern "C" {',
                "#endif",
                "",
                f"extern const char {self.symbol}[{len(self.data) + 1}];",
                "",
                "#ifdef __cplusplus",
                "}",
                "#endif",
                "",
                f"#endif  // {guard}",
                "",
            ]
        )

    def emit_source(self) -> str:
        return "\n".join([*line_comment_header("//", generator="loom.gen.support.string_pool"), "", f'#include "{self.header}"', "", *_emit_data(self.data, self.symbol, storage=""), ""])


def _slice_reference(offset: int, length: int, bank: int = 0) -> int:
    """Checks the C slice encoding, including the complete payload bound."""
    if not 0 <= length <= 255:
        raise ValueError("string slice exceeds 255 bytes")
    if not 0 <= offset < 1 << 23 or offset + length > 1 << 23:
        raise ValueError("string bank exceeds 23-bit slice offset encoding")
    if bank not in (0, 1):
        raise ValueError("string bank must be primary or secondary")
    return (length << 24) | (bank << 23) | offset


def layout_c_string_pool(
    pool: CStringPool,
    shared: CStringBank | None = None,
    *,
    primary: CStringBank | None = None,
) -> CStringLayout:
    """Lays out local or external banks with longest-first substring sharing."""

    if primary is shared and primary is not None:
        raise ValueError("primary and shared string banks must be distinct")
    data = bytearray()
    references: dict[str, int] = {}
    # An incidental substring must not introduce an otherwise unused library.
    if shared is not None and not shared.values.intersection(pool.value_to_label):
        shared = None
    payloads = [(entry.label, entry.value.encode()) for entry in pool.entries]
    for label, payload in sorted(payloads, key=lambda entry: (-len(entry[1]), entry[1])):
        length = len(payload)
        # Keep the empty slice canonical in the private bank.
        if length and shared is not None:
            offset = shared.data.find(payload)
            if offset >= 0:
                references[label] = _slice_reference(offset, length, 1)
                continue
        if primary is not None:
            offset = primary.data.find(payload)
            if offset < 0:
                raise ValueError(f"primary string bank does not contain '{payload.decode()}'")
            references[label] = _slice_reference(offset, length)
            continue
        offset = data.find(payload)
        if offset < 0:
            offset = len(data)
            reference = _slice_reference(offset, length)
            data.extend(payload)
        else:
            reference = _slice_reference(offset, length)
        references[label] = reference
    return CStringLayout(bytes(data), references, primary, shared)


def _emit_data(data: bytes, data_name: str, *, storage: str = "static ") -> list[str]:
    text = data.decode()
    lines = ["// clang-format off", f"{storage}const char {data_name}[] ="]
    # Split by codepoints so each source literal remains valid UTF-8. References
    # count encoded bytes, independently of source formatting and escapes.
    for start in range(0, max(1, len(text)), 80):
        escaped = c_string_literal(text[start : start + 80]).replace("\0", "\\000")
        lines.append(f'    "{escaped}"')
    lines[-1] += ";"
    lines.extend(["// clang-format on", ""])
    return lines


def emit_c_string_pool(pool: CStringPool, data_name: str, *, layout: CStringLayout | None = None) -> list[str]:
    """Emits shared character bytes and constant compact slice references."""
    if not pool.entries:
        return []
    if layout is None:
        layout = layout_c_string_pool(pool)
    lines = [] if layout.primary is not None else _emit_data(layout.data, data_name)
    lines.append("enum {")
    for entry in pool.entries:
        reference = layout.references[entry.label]
        lines.append(f"  {pool.enum_name(entry.label)} = 0x{reference:08X}u,")
    lines.extend(
        [
            "};",
            f'static_assert(sizeof({layout.primary.symbol if layout.primary is not None else data_name}) - 1 == {len(layout.primary.data) if layout.primary is not None else len(layout.data)}, "string pool byte length must match generated slices");',
            "",
        ]
    )
    return lines
