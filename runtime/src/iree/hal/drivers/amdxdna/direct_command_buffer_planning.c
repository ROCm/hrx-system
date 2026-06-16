// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/direct_command_buffer_planning.h"

#include <string.h>

// AIE-visible DDR address offset added to every shim-DMA buffer address.
// Validated for npu4 / AIE2P_STRIX_B0 (the only target the chained path is
// enabled for); other AIE generations may use a different offset / BD address
// layout. This is an AIE DMA address ABI, not a native queue command window.
static const uint64_t iree_hal_amdxdna_ddr_aie_addr_offset = 0x80000000ULL;
// AIEC RTP lowering emits amdaie.npu.write32 values tagged with this sentinel;
// the HAL replaces the low bits with the corresponding dispatch constant before
// handing the transaction to firmware.
static const uint32_t iree_hal_amdxdna_write32_constant_sentinel = 0xA1EC0000u;
static const uint32_t iree_hal_amdxdna_write32_constant_mask = 0xFFFF0000u;

// Misalignment-safe little-endian 32-bit read: a malformed op size can leave
// `p` non-word-aligned, so read through memcpy rather than an unaligned cast.
static uint32_t iree_hal_amdxdna_read_u32(const uint8_t* p) {
  uint32_t value;
  memcpy(&value, p, sizeof(value));
  return value;
}

static void iree_hal_amdxdna_write_u32(uint8_t* p, uint32_t value) {
  memcpy(p, &value, sizeof(value));
}

uint32_t iree_hal_amdxdna_txn_op_size(const uint8_t* b, size_t total,
                                      size_t p) {
  if (p >= total) return 0;
  uint8_t op = b[p];
  if (op == 0) {  // WRITE32.
    if (p + 24 > total) return 0;
    return iree_hal_amdxdna_read_u32(b + p + 20);
  }
  if (op == 1) {  // BLOCKWRITE.
    if (p + 16 > total) return 0;
    return iree_hal_amdxdna_read_u32(b + p + 12);
  }
  if (op == 3 || op == 4) {
    if (p + 28 > total) return 0;
    return iree_hal_amdxdna_read_u32(b + p + 24);
  }
  if (op >= 128) {  // Custom op.
    if (p + 8 > total) return 0;
    return iree_hal_amdxdna_read_u32(b + p + 4);
  }
  return 4;
}

iree_status_t iree_hal_amdxdna_patch_write32_constants(
    uint32_t* txn, size_t txn_words, iree_const_byte_span_t constants) {
  if (txn_words < 4) return iree_ok_status();
  uint8_t* b = (uint8_t*)txn;
  size_t total = txn_words * sizeof(uint32_t);
  uint32_t num_ops = txn[2];  // TXN header word 2 = NumOps.
  size_t p = 16;              // Past the 16-byte XAie_TxnHeader.
  for (uint32_t i = 0; i < num_ops; ++i) {
    uint32_t sz = iree_hal_amdxdna_txn_op_size(b, total, p);
    if (sz == 0 || p + sz > total) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna write32 RTP patch saw malformed transaction op %u at byte "
          "offset %zu",
          i, p);
    }
    if (b[p] == 0) {  // WRITE32: patch sentinel values from HAL constants.
      uint32_t value = iree_hal_amdxdna_read_u32(b + p + 16);
      if ((value & iree_hal_amdxdna_write32_constant_mask) ==
          iree_hal_amdxdna_write32_constant_sentinel) {
        uint32_t constant_index =
            value & ~iree_hal_amdxdna_write32_constant_mask;
        iree_host_size_t byte_offset =
            (iree_host_size_t)constant_index * sizeof(uint32_t);
        if (byte_offset + sizeof(uint32_t) > constants.data_length) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "amdxdna write32 RTP constant index %u out of bounds for "
              "%zu-byte constants block",
              constant_index, constants.data_length);
        }
        memcpy(&value, constants.data + byte_offset, sizeof(uint32_t));
        iree_hal_amdxdna_write_u32(b + p + 16, value);
      }
    }
    p += sz;
  }
  return iree_ok_status();
}

bool iree_hal_amdxdna_apply_patch_table(uint32_t* ctrl_code, size_t ctrl_words,
                                        const uint32_t* patches,
                                        size_t patch_count,
                                        const uint64_t* args,
                                        size_t arg_count) {
  if (patch_count % 3 != 0) return false;
  if (patch_count && !patches) return false;
  uint8_t* b = (uint8_t*)ctrl_code;
  size_t total = ctrl_words * sizeof(uint32_t);
  for (size_t i = 0; i < patch_count; i += 3) {
    uint32_t offset = patches[i];        // byte offset of the BD base word
    uint32_t arg_idx = patches[i + 1];   // index into `args`
    uint32_t arg_plus = patches[i + 2];  // byte addend into that buffer
    if (arg_idx >= arg_count) return false;
    // We touch bd[1] at offset+4 and bd[2] at offset+8 (4 bytes each).
    if ((size_t)offset + 12 > total || (offset & 0x3u) != 0) {
      return false;
    }
    uint32_t bd1 = iree_hal_amdxdna_read_u32(b + offset + 4);
    uint32_t bd2 = iree_hal_amdxdna_read_u32(b + offset + 8);
    uint64_t base = ((uint64_t)(bd2 & 0xFFFF) << 32) | bd1;
    base += args[arg_idx] + arg_plus + iree_hal_amdxdna_ddr_aie_addr_offset;
    bd1 = (uint32_t)(base & 0xFFFFFFFC);
    bd2 = (bd2 & 0xFFFF0000) | (uint32_t)(base >> 32);
    iree_hal_amdxdna_write_u32(b + offset + 4, bd1);
    iree_hal_amdxdna_write_u32(b + offset + 8, bd2);
  }
  return true;
}
