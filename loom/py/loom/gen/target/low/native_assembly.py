# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generation-time sharing of native assembly recipes across descriptor sets."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from loom.gen.support.c import CIdentifierCase, c_identifier
from loom.gen.support.generated_file import line_comment_header
from loom.gen.support.string_pool import CStringBank, CStringPool, layout_c_string_pool
from loom.gen.target.low import c_spelling, validation
from loom.gen.target.low.compiled import CompiledAsmLayout, CompiledAsmTableStorage, CompiledNativeAsmValue, append_interned_sequence
from loom.target.low_descriptors import NativeAsmValueKind


def native_asm_value_row_lines(pool: CStringPool, values: Sequence[CompiledNativeAsmValue]) -> list[list[str]]:
    """Initializes the same C row for private and shared recipe storage."""
    return [
        [
            f".kind = {value.kind.c_name},",
            f".index = {value.index},",
            f".bit_width = {value.bit_width},",
            f".target_format_id = {value.target_format_id},",
            f".literal_string_ref = {c_spelling.optional_string_expr(pool, value.literal_label)},",
        ]
        for value in values
    ]


def _canonical_span(values: Sequence[CompiledNativeAsmValue]) -> tuple[CompiledNativeAsmValue, ...]:
    # Labels belong to the provider's C namespace; literal text is the value.
    return tuple(replace(value, literal_label=None) for value in values)


@dataclass(frozen=True, slots=True)
class NativeAsmValuePool:
    """An immutable recipe bank shared by sets using the same string banks.

    Operand/immediate indices and target formats retain their descriptor-local
    meaning. Existing layout offsets directly address the shared array; the
    span map exists only during generation.
    """

    # C array name referenced by every participating descriptor set.
    symbol: str
    # Public generated header declaring the array and its extent.
    header: str
    # Canonical rows in first-use span order, without provider-local labels.
    values: tuple[CompiledNativeAsmValue, ...]
    # Exact nonempty recipe spans mapped to their first row in values.
    starts: dict[tuple[CompiledNativeAsmValue, ...], int]

    @staticmethod
    def build(spans: Iterable[Sequence[CompiledNativeAsmValue]], *, symbol: str, header: str) -> NativeAsmValuePool:
        values: list[CompiledNativeAsmValue] = []
        starts: dict[tuple[CompiledNativeAsmValue, ...], int] = {}
        for span in spans:
            validation.validate_u8(len(span), "native assembly recipe length")
            append_interned_sequence(_canonical_span(span), values, starts)
        validation.validate_u16_table_count(len(values), "shared native assembly value")
        return NativeAsmValuePool(symbol, header, tuple(values), starts)

    def layouts_for(self, storage: CompiledAsmTableStorage) -> list[CompiledAsmLayout]:
        """Rebases native spans without changing any other layout coordinates."""
        layouts = []
        for layout in storage.layouts:
            start = layout.native_assembly_value_start
            count = layout.native_assembly_value_count
            span = _canonical_span(storage.native_values[start : start + count])
            if span and span not in self.starts:
                raise ValueError("shared native assembly pool does not cover descriptor recipe")
            layouts.append(replace(layout, native_assembly_value_start=self.starts[span] if span else 0))
        return layouts

    def manifest(self) -> str:
        """Carries exact span order across independent generator actions."""
        return (
            json.dumps(
                dict(
                    symbol=self.symbol,
                    header=self.header,
                    spans=[[(value.kind.name, value.index, value.bit_width, value.target_format_id, value.literal) for value in span] for span in self.starts],
                )
            )
            + "\n"
        )

    @staticmethod
    def read(path: Path) -> NativeAsmValuePool:
        payload = json.loads(path.read_text())
        return NativeAsmValuePool.build(
            (
                tuple(CompiledNativeAsmValue(NativeAsmValueKind[kind], index, bit_width, target_format_id, None, literal) for kind, index, bit_width, target_format_id, literal in span)
                for span in payload["spans"]
            ),
            symbol=payload["symbol"],
            header=payload["header"],
        )

    def emit_header(self) -> str:
        guard = c_identifier(self.header, case=CIdentifierCase.UPPER)
        return "\n".join(
            [
                *line_comment_header("//", generator="loom.gen.target.low.native_assembly"),
                "",
                f"#ifndef {guard}",
                f"#define {guard}",
                "",
                '#include "loom/codegen/low/descriptors.h"',
                "",
                f"extern const loom_low_native_asm_value_t {self.symbol}[{max(1, len(self.values))}];",
                "",
                f"#endif  // {guard}",
                "",
            ]
        )

    def emit_source(self, primary_strings: CStringBank, shared_strings: CStringBank | None) -> str:
        # The shared string banks have the same coordinates in every provider.
        pool = CStringPool(self.symbol.upper())
        values = [replace(value, literal_label=pool.intern(f"literal_{index}", value.literal) if value.literal is not None else None) for index, value in enumerate(self.values)]
        layout = layout_c_string_pool(pool, shared_strings, primary=primary_strings)
        lines = [
            *line_comment_header("//", generator="loom.gen.target.low.native_assembly"),
            "",
            f'#include "{self.header}"',
            "",
        ]
        if pool.entries:
            lines.extend(["enum {", *(f"  {pool.enum_name(entry.label)} = 0x{layout.references[entry.label]:08X}u," for entry in pool.entries), "};", ""])
        lines.append(f"const loom_low_native_asm_value_t {self.symbol}[{max(1, len(values))}] = {{")
        for row in native_asm_value_row_lines(pool, values):
            lines.extend(["    {", *(f"        {field}" for field in row), "    },"])
        if not values:
            lines.append("    {0},")
        lines.extend(["};", ""])
        return "\n".join(lines)
