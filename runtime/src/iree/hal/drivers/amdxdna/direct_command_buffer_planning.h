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

typedef struct iree_hal_amdxdna_write32_constant_patch_site_t {
  uint32_t byte_offset;
  uint32_t constant_index;
} iree_hal_amdxdna_write32_constant_patch_site_t;

typedef struct iree_hal_amdxdna_write32_constant_patch_list_t {
  iree_hal_amdxdna_write32_constant_patch_site_t* data;
  iree_host_size_t count;
} iree_hal_amdxdna_write32_constant_patch_list_t;

typedef struct iree_hal_amdxdna_host_patch_table_t {
  uint32_t* data;
  iree_host_size_t count;
} iree_hal_amdxdna_host_patch_table_t;

void iree_hal_amdxdna_host_patch_table_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_host_patch_table_t* table);

// Derives the host-patch optimization table from semantic DDR_PATCH operations
// in an XAie transaction. The returned flat list contains
// (BD byte offset, argument index, argument addend) triples.
iree_status_t iree_hal_amdxdna_build_host_patch_table(
    iree_allocator_t host_allocator, iree_const_byte_span_t transaction,
    iree_hal_amdxdna_host_patch_table_t* out_table);

void iree_hal_amdxdna_write32_constant_patch_list_deinitialize(
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_write32_constant_patch_list_t* list);

iree_status_t iree_hal_amdxdna_build_write32_constant_patch_list(
    iree_allocator_t host_allocator, const uint32_t* txn, size_t txn_words,
    iree_hal_amdxdna_write32_constant_patch_list_t* out_list);

// Patches amdaie.npu.write32 sentinel values inside a TXN stream `txn`
// (`txn_words` 32-bit words) with the corresponding dispatch constant pulled
// from `constants`. Returns INVALID_ARGUMENT on a malformed transaction op or
// an out-of-range constant index.
iree_status_t iree_hal_amdxdna_patch_write32_constants(
    uint32_t* txn, size_t txn_words, iree_const_byte_span_t constants);

// Patches amdaie.npu.write32 values using a precomputed patch-site list from
// iree_hal_amdxdna_build_write32_constant_patch_list. This avoids a full XAie
// transaction walk on every late-bound dispatch materialization.
iree_status_t iree_hal_amdxdna_patch_write32_constants_with_list(
    uint32_t* txn, size_t txn_words,
    const iree_hal_amdxdna_write32_constant_patch_list_t* patch_list,
    iree_const_byte_span_t constants);

// Applies the compiler-emitted host patch table to `ctrl_code` (`ctrl_words`
// 32-bit words). `patches` (`patch_count` 32-bit words) is a flat list of
// (offset, arg_idx, arg_plus) triples; for each, the 48-bit shim-DMA address
// `args[arg_idx] + arg_plus + AIE_DDR_offset` is written into the
// buffer-descriptor address words at byte `offset` (low 32 into bd[1], high 16
// into bd[2]). Returns false on any malformed / out-of-bounds table entry, or
// when the computed address overflows or is not 4-byte aligned.
bool iree_hal_amdxdna_apply_patch_table(uint32_t* ctrl_code, size_t ctrl_words,
                                        const uint32_t* patches,
                                        size_t patch_count,
                                        const uint64_t* args, size_t arg_count);

// Rewrites only dynamic words in `ctrl_code` using immutable `template_code` as
// the source of truth. This is intended for cached command-chain control-code
// BOs: constants and buffer-descriptor addresses may change per dispatch, but
// the rest of the XAie transaction stream remains fixed. Unlike
// iree_hal_amdxdna_apply_patch_table, this is safe to call repeatedly on an
// already patched control-code buffer because constants and BD control bits are
// restored from `template_code`, while BD addresses are recomputed from
// `args[arg_idx] + arg_plus + AIE_DDR_offset`.
iree_status_t iree_hal_amdxdna_patch_dynamic_fields_from_template(
    uint32_t* ctrl_code, const uint32_t* template_code, size_t ctrl_words,
    const iree_hal_amdxdna_write32_constant_patch_list_t* constant_patches,
    iree_const_byte_span_t constants, const uint32_t* patches,
    size_t patch_count, const uint64_t* args, size_t arg_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_DIRECT_COMMAND_BUFFER_PLANNING_H_
