# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""C B-string pool helpers shared by Loom generators."""

from __future__ import annotations

from dataclasses import dataclass, field

from loom.gen.support.c import CIdentifierCase, c_identifier, c_string_literal


@dataclass(frozen=True, slots=True)
class CStringEntry:
    """One unique string payload in a generated C string table."""

    label: str
    value: str
    offset: int


@dataclass(slots=True)
class CStringPool:
    """Interns byte-length-prefixed strings and emits stable enum references."""

    c_enum_prefix: str
    entries: list[CStringEntry] = field(default_factory=list)
    value_to_label: dict[str, str] = field(default_factory=dict)
    label_to_primary: dict[str, str] = field(default_factory=dict)
    next_offset: int = 0

    @staticmethod
    def canonical_label(label: str) -> str:
        """Returns the canonical C enum label for an arbitrary string label."""
        return c_identifier(label, case=CIdentifierCase.LOWER, empty="empty")

    def intern(self, label: str, value: str) -> str:
        """Interns |value| with |label| and returns the primary canonical label."""
        if len(value.encode()) > 255:
            raise ValueError(f"B-string '{value}' exceeds 255 bytes")
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
        self.entries.append(CStringEntry(label, value, self.next_offset))
        self.value_to_label[value] = label
        self.label_to_primary[label] = label
        self.next_offset += len(value.encode()) + 1
        return label

    @property
    def entries_by_label(self) -> dict[str, CStringEntry]:
        """Returns the unique entries indexed by primary canonical label."""
        return {entry.label: entry for entry in self.entries}

    def enum_name(self, label: str) -> str:
        """Returns the generated C enum name for an interned string label."""
        primary_label = self.label_to_primary[self.canonical_label(label)]
        return f"{self.c_enum_prefix}_STRING_{primary_label}"

    def ref(self, label: str) -> str:
        """Returns a C expression naming the interned string offset enum."""
        return self.enum_name(label)


def emit_c_string_table(pool: CStringPool, data_name: str) -> list[str]:
    """Emits one packed B-string byte table and its offset constants."""
    if not pool.entries:
        return []
    lines = [
        "// clang-format off",
        f"static const uint8_t {data_name}[] =",
    ]
    for entry in pool.entries:
        length = len(entry.value.encode())
        escaped = c_string_literal(entry.value)
        lines.append(f'    LOOM_BSTRING_LITERAL({length}, "{escaped}")')
    lines[-1] += ";"
    lines.extend(["// clang-format on", "", "enum {"])
    for index, entry in enumerate(pool.entries):
        enum_name = pool.enum_name(entry.label)
        if index == 0:
            lines.append(f"  {enum_name} = 0,")
        else:
            previous_entry = pool.entries[index - 1]
            previous_enum_name = pool.enum_name(previous_entry.label)
            previous_value = c_string_literal(previous_entry.value)
            lines.append(f'  {enum_name} = {previous_enum_name} + sizeof("{previous_value}"),')
    previous_entry = pool.entries[-1]
    previous_enum_name = pool.enum_name(previous_entry.label)
    previous_value = c_string_literal(previous_entry.value)
    lines.extend(
        [
            f'  {pool.c_enum_prefix}_STRING_END = {previous_enum_name} + sizeof("{previous_value}"),',
            "};",
            "",
        ]
    )
    return lines
