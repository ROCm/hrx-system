# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.tools.amdgpu_asm import (
    parse_amdgpu_mnemonic,
    summarize_amdgpu_disassembly,
)


def test_parse_amdgpu_mnemonic_accepts_objdump_address_forms() -> None:
    assert parse_amdgpu_mnemonic("      s_load_b128 s[0:3], s[0:1], null") == (
        "s_load_b128"
    )
    assert parse_amdgpu_mnemonic("   14: global_load_b128 v[0:3], v0") == (
        "global_load_b128"
    )
    assert parse_amdgpu_mnemonic("0000000000000000 <kernel>:") is None
    assert parse_amdgpu_mnemonic("Disassembly of section .text:") is None


def test_summarize_amdgpu_disassembly_reports_families_and_mnemonics() -> None:
    summary = summarize_amdgpu_disassembly(
        """
0000000000000000 <kernel>:
      s_load_b128 s[0:3], s[0:1], null
   14: global_load_b128 v[0:3], v0
      global_store_b128 v0, v[0:3]
      s_waitcnt vmcnt(0)
      s_waitcnt lgkmcnt(0)
      v_mfma_f32_16x16x16f16 a[0:7], v[0:1], v[2:3], a[0:7]
      v_add_u32_e32 v0, v1, v2
      s_endpgm
"""
    )

    assert summary.instruction_count == 8
    assert summary.family_counts == {
        "global_load": 1,
        "global_store": 1,
        "s_endpgm": 1,
        "s_load": 1,
        "s_waitcnt": 2,
        "v_alu": 1,
        "v_mfma": 1,
    }
    assert summary.mnemonic_counts["s_waitcnt"] == 2
    assert summary.mnemonic_counts["global_load_b128"] == 1
    assert summary.mnemonic_counts["v_add_u32_e32"] == 1
