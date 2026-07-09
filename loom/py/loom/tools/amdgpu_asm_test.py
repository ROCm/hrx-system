# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from loom.tools.amdgpu_asm import (
    parse_amdgpu_mnemonic,
    summarize_amdgpu_disassembly,
    summarize_amdgpu_disassembly_blocks,
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
    assert (
        parse_amdgpu_mnemonic("  amdhsa.target: 'amdgcn-amd-amdhsa--gfx1100'") is None
    )
    assert parse_amdgpu_mnemonic("      s_code_end") is None


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
      ds_load_b128 v[0:3], v0
      ds_store_b64 v0, v[0:1]
      s_endpgm
      s_code_end
"""
    )

    assert summary.instruction_count == 10
    assert summary.family_counts == {
        "ds_read": 1,
        "ds_write": 1,
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
    assert summary.matrix_mnemonic_counts == {
        "v_mfma_f32_16x16x16f16": 1,
    }
    assert summary.memory_byte_counts == {
        "global_load_bytes": 16,
        "global_store_bytes": 16,
        "ds_read_bytes": 16,
        "ds_write_bytes": 8,
        "read_bytes": 48,
        "s_load_bytes": 16,
        "write_bytes": 24,
    }


def test_summarize_amdgpu_disassembly_blocks_splits_symbols() -> None:
    blocks = summarize_amdgpu_disassembly_blocks(
        """
0000000000000000 <first_kernel>:
      global_load_b128 v[0:3], v0
      v_wmma_f32_16x16x16_bf16 v[0:7], v[8:9], v[10:11], v[0:7]
0000000000000100 <second_kernel>:
      ds_read_b64 v[0:1], v0
      ds_write_b32 v0, v1
"""
    )

    assert [block.symbol for block in blocks] == ["first_kernel", "second_kernel"]
    assert [block.address for block in blocks] == [0, 0x100]
    assert blocks[0].summary.family_counts["v_wmma"] == 1
    assert blocks[0].summary.memory_byte_counts["global_load_bytes"] == 16
    assert blocks[1].summary.memory_byte_counts == {
        "ds_read_bytes": 8,
        "ds_write_bytes": 4,
        "read_bytes": 8,
        "write_bytes": 4,
    }


def test_summarize_amdgpu_disassembly_blocks_accepts_assembly_labels() -> None:
    blocks = summarize_amdgpu_disassembly_blocks(
        """
.globl plain_kernel
plain_kernel:
      global_load_b128 v[0:3], v0
      amdhsa.target: 'amdgcn-amd-amdhsa--gfx1100'
"""
    )

    assert [block.symbol for block in blocks] == ["plain_kernel"]
    assert blocks[0].address is None
    assert blocks[0].summary.instruction_count == 1
    assert blocks[0].summary.family_counts["global_load"] == 1
