# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""AMDGPU assembly and disassembly summarization helpers."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

_ADDRESS_PREFIX_PATTERN = re.compile(r"^\s*(?:[0-9a-fA-F]+:|[0-9a-fA-F]+\s+)?\s*")
_MNEMONIC_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.$]*")
_SYMBOL_PATTERN = re.compile(r"^\s*[0-9a-fA-F]+\s+<([^>]+)>:\s*$")
_ASSEMBLY_LABEL_PATTERN = re.compile(r"^([A-Za-z_$][A-Za-z0-9_.$]*):\s*$")
_BIT_WIDTH_PATTERN = re.compile(r"_b(8|16|32|64|128|256)(?:$|_)")
_DWORDX_WIDTH_PATTERN = re.compile(r"_dwordx([0-9]+)(?:$|_)")

_FAMILY_PREFIXES = (
    ("v_mfma", "v_mfma"),
    ("v_wmma", "v_wmma"),
    ("v_smfmac", "v_smfmac"),
    ("v_dot", "v_dot"),
    ("global_load", "global_load"),
    ("global_store", "global_store"),
    ("global_atomic", "global_atomic"),
    ("buffer_load", "buffer_load"),
    ("buffer_store", "buffer_store"),
    ("buffer_atomic", "buffer_atomic"),
    ("flat_load", "flat_load"),
    ("flat_store", "flat_store"),
    ("flat_atomic", "flat_atomic"),
    ("ds_load", "ds_read"),
    ("ds_store", "ds_write"),
    ("ds_read", "ds_read"),
    ("ds_write", "ds_write"),
    ("ds_", "ds_other"),
    ("s_waitcnt", "s_waitcnt"),
    ("s_barrier", "s_barrier"),
    ("s_load", "s_load"),
    ("s_buffer_load", "s_load"),
    ("s_branch", "s_branch"),
    ("s_cbranch", "s_branch"),
    ("s_endpgm", "s_endpgm"),
    ("s_", "s_alu"),
    ("v_", "v_alu"),
)


@dataclass(frozen=True, slots=True)
class AmdgpuDisassemblySummary:
    """Stable instruction summary for externally produced AMDGPU disassembly."""

    instruction_count: int
    family_counts: Mapping[str, int]
    mnemonic_counts: Mapping[str, int]
    matrix_mnemonic_counts: Mapping[str, int]
    memory_byte_counts: Mapping[str, int]

    def metadata(self) -> Mapping[str, object]:
        """Returns a JSON-serializable summary object."""

        return {
            "instruction_count": self.instruction_count,
            "family_counts": dict(self.family_counts),
            "matrix_mnemonic_counts": dict(self.matrix_mnemonic_counts),
            "memory_byte_counts": dict(self.memory_byte_counts),
            "mnemonic_counts": dict(self.mnemonic_counts),
        }


@dataclass(frozen=True, slots=True)
class AmdgpuDisassemblyBlock:
    """Instruction summary for one symbol block in AMDGPU disassembly."""

    symbol: str
    address: int | None
    start_line: int
    summary: AmdgpuDisassemblySummary

    def metadata(self) -> Mapping[str, object]:
        """Returns a JSON-serializable summary object."""

        return {
            "symbol": self.symbol,
            "address": self.address,
            "start_line": self.start_line,
            "summary": self.summary.metadata(),
        }


def summarize_amdgpu_disassembly(disassembly: str) -> AmdgpuDisassemblySummary:
    """Summarizes AMDGPU instruction families and exact mnemonics."""

    mnemonics = tuple(iter_amdgpu_mnemonics(disassembly.splitlines()))
    mnemonic_counts = Counter(mnemonics)
    family_counts = Counter(_instruction_family(mnemonic) for mnemonic in mnemonics)
    matrix_mnemonic_counts = {
        mnemonic: count
        for mnemonic, count in sorted(mnemonic_counts.items())
        if _instruction_family(mnemonic) in {"v_mfma", "v_wmma", "v_smfmac"}
    }
    memory_byte_counts = _summarize_memory_bytes(mnemonic_counts)
    return AmdgpuDisassemblySummary(
        instruction_count=len(mnemonics),
        family_counts=MappingProxyType(dict(sorted(family_counts.items()))),
        mnemonic_counts=MappingProxyType(dict(sorted(mnemonic_counts.items()))),
        matrix_mnemonic_counts=MappingProxyType(matrix_mnemonic_counts),
        memory_byte_counts=MappingProxyType(memory_byte_counts),
    )


def summarize_amdgpu_disassembly_blocks(
    disassembly: str,
) -> tuple[AmdgpuDisassemblyBlock, ...]:
    """Summarizes each symbol block in `llvm-objdump -d` AMDGPU output."""

    blocks: list[tuple[str, int | None, int, list[str]]] = []
    current_symbol: str | None = None
    current_address: int | None = None
    current_start_line = 0
    current_lines: list[str] = []
    for line_number, line in enumerate(disassembly.splitlines(), start=1):
        next_symbol = _match_disassembly_symbol(line)
        if next_symbol is not None:
            if current_symbol is not None:
                blocks.append(
                    (
                        current_symbol,
                        current_address,
                        current_start_line,
                        current_lines,
                    )
                )
            current_symbol, current_address = next_symbol
            current_start_line = line_number
            current_lines = []
            continue
        if current_symbol is not None:
            current_lines.append(line)
    if current_symbol is not None:
        blocks.append(
            (current_symbol, current_address, current_start_line, current_lines)
        )
    return tuple(
        AmdgpuDisassemblyBlock(
            symbol=symbol,
            address=address,
            start_line=start_line,
            summary=summarize_amdgpu_disassembly("\n".join(lines)),
        )
        for symbol, address, start_line, lines in blocks
    )


def _match_disassembly_symbol(line: str) -> tuple[str, int | None] | None:
    match = _SYMBOL_PATTERN.match(line)
    if match is not None:
        return match.group(1), int(line.strip().split(maxsplit=1)[0], 16)
    match = _ASSEMBLY_LABEL_PATTERN.match(line)
    if match is not None:
        return match.group(1), None
    return None


def iter_amdgpu_mnemonics(lines: Iterable[str]) -> Iterable[str]:
    """Yields instruction mnemonics from `llvm-objdump -d` style text."""

    for line in lines:
        mnemonic = parse_amdgpu_mnemonic(line)
        if mnemonic is not None:
            yield mnemonic


def parse_amdgpu_mnemonic(line: str) -> str | None:
    """Returns the instruction mnemonic on a disassembly line, if present."""

    stripped = line.strip()
    if not stripped or stripped.startswith(("#", ";")) or stripped.endswith(":"):
        return None
    candidate = _ADDRESS_PREFIX_PATTERN.sub("", line, count=1).lstrip()
    match = _MNEMONIC_PATTERN.match(candidate)
    if match is None:
        return None
    mnemonic = match.group(0)
    if mnemonic in {"Disassembly", "file", "format", "s_code_end"}:
        return None
    if mnemonic.startswith("amdhsa."):
        return None
    return mnemonic


def _instruction_family(mnemonic: str) -> str:
    return classify_amdgpu_instruction_family(mnemonic)


def classify_amdgpu_instruction_family(mnemonic: str) -> str:
    """Classifies an AMDGPU instruction mnemonic into a stable family name."""

    for prefix, family in _FAMILY_PREFIXES:
        if mnemonic.startswith(prefix):
            return family
    return "other"


def _summarize_memory_bytes(mnemonic_counts: Mapping[str, int]) -> dict[str, int]:
    byte_counts: Counter[str] = Counter()
    for mnemonic, count in mnemonic_counts.items():
        family = _instruction_family(mnemonic)
        direction = _memory_direction(family)
        if direction is None:
            continue
        width = _instruction_byte_width(mnemonic)
        if width is None:
            continue
        byte_counts[f"{family}_bytes"] += width * count
        byte_counts[f"{direction}_bytes"] += width * count
    return dict(sorted(byte_counts.items()))


def _memory_direction(family: str) -> str | None:
    if family in {
        "global_load",
        "buffer_load",
        "flat_load",
        "ds_read",
        "s_load",
    }:
        return "read"
    if family in {
        "global_store",
        "buffer_store",
        "flat_store",
        "ds_write",
    }:
        return "write"
    return None


def _instruction_byte_width(mnemonic: str) -> int | None:
    bit_match = _BIT_WIDTH_PATTERN.search(mnemonic)
    if bit_match is not None:
        return int(bit_match.group(1)) // 8
    dwordx_match = _DWORDX_WIDTH_PATTERN.search(mnemonic)
    if dwordx_match is not None:
        return 4 * int(dwordx_match.group(1))
    if "_dword" in mnemonic:
        return 4
    if "_word" in mnemonic or "_short" in mnemonic:
        return 2
    if "_byte" in mnemonic:
        return 1
    return None
