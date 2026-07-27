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

    def metadata(self) -> Mapping[str, object]:
        """Returns a JSON-serializable summary object."""

        return MappingProxyType(
            {
                "instruction_count": self.instruction_count,
                "family_counts": dict(self.family_counts),
                "mnemonic_counts": dict(self.mnemonic_counts),
            }
        )


def summarize_amdgpu_disassembly(disassembly: str) -> AmdgpuDisassemblySummary:
    """Summarizes AMDGPU instruction families and exact mnemonics."""

    mnemonics = tuple(iter_amdgpu_mnemonics(disassembly.splitlines()))
    mnemonic_counts = Counter(mnemonics)
    family_counts = Counter(_instruction_family(mnemonic) for mnemonic in mnemonics)
    return AmdgpuDisassemblySummary(
        instruction_count=len(mnemonics),
        family_counts=MappingProxyType(dict(sorted(family_counts.items()))),
        mnemonic_counts=MappingProxyType(dict(sorted(mnemonic_counts.items()))),
    )


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
    return mnemonic


def _instruction_family(mnemonic: str) -> str:
    for prefix, family in _FAMILY_PREFIXES:
        if mnemonic.startswith(prefix):
            return family
    return "other"
