// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_PLANNING_H_
#define IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_PLANNING_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"

// Pure, host-side control-code planning helpers extracted from the amdxdna
// direct command buffer. These do not touch device/native state and are unit
// tested in direct_command_buffer_planning_test.cc. All XAie-format knowledge
// lives in the compiler; the only hardware facts encoded here are the XAie
// transaction op-size table and the shim-DMA buffer-descriptor address split
// (a DMA-address ABI).

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Size in bytes of one XAie transaction operation starting at byte offset `p`
// within the `total`-byte buffer `b`. Returns 0 on malformed/truncated input.
uint32_t iree_hal_amdxdna_txn_op_size(const uint8_t* b, size_t total, size_t p);

// Patches amdaie.npu.write32 sentinel values inside a TXN stream `txn`
// (`txn_words` 32-bit words) with the corresponding dispatch constant pulled
// from `constants`. Returns INVALID_ARGUMENT on a malformed transaction op or
// an out-of-range constant index.
iree_status_t iree_hal_amdxdna_patch_write32_constants(
    uint32_t* txn, size_t txn_words, iree_const_byte_span_t constants);

// Applies the compiler-emitted host patch table to `ctrl_code` (`ctrl_words`
// 32-bit words). `patches` (`patch_count` 32-bit words) is a flat list of
// (offset, arg_idx, arg_plus) triples; for each, the 48-bit shim-DMA address
// `args[arg_idx] + arg_plus + AIE_DDR_offset` is written into the
// buffer-descriptor address words at byte `offset` (low 32 into bd[1], high 16
// into bd[2]). Returns false on any malformed / out-of-bounds table entry.
bool iree_hal_amdxdna_apply_patch_table(uint32_t* ctrl_code, size_t ctrl_words,
                                        const uint32_t* patches,
                                        size_t patch_count,
                                        const uint64_t* args, size_t arg_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_PLANNING_H_
