// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_TEXT_STORAGE_H_
#define EXPERIMENTAL_LOOM_SERVE_TEXT_STORAGE_H_

#include "experimental/loom_serve/storage/relocation.h"
#include "experimental/loom_serve/storage/snapshot.h"
#include "iree/vm/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Private source bootstrap records consumed once at model creation.
enum {
  LOOM_SERVE_TEXT_STORAGE_ALLOCATIONS = 0,
  LOOM_SERVE_TEXT_STORAGE_VIEWS = 1,
  LOOM_SERVE_TEXT_STORAGE_TARGET_ORIGINS = 2,
  LOOM_SERVE_TEXT_STORAGE_DRAFT_ORIGINS = 3,
  LOOM_SERVE_TEXT_STORAGE_GEOMETRY = 4,
  LOOM_SERVE_TEXT_STORAGE_RESULT_COUNT = 5,
  LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT = 11,
};

typedef struct loom_serve_text_cache_region_t {
  // Source allocation slot containing these repeated cache planes.
  iree_host_size_t allocation;
  // Source-owned disjoint cache planes inside that allocation.
  loom_serve_block_region_t blocks;
} loom_serve_text_cache_region_t;

// Source storage description, separate from device ownership and model work.
// Validation establishes dimensions once; consumers use these facts directly.
typedef struct loom_serve_text_storage_t {
  // Allocator owning parsed region metadata.
  iree_allocator_t allocator;
  // Source descriptors and upload payloads retained through initialization.
  iree_vm_buffer_t* buffers[LOOM_SERVE_TEXT_STORAGE_RESULT_COUNT];
  // Immutable mappings borrowing buffers until release_payloads.
  iree_const_byte_span_t bytes[LOOM_SERVE_TEXT_STORAGE_RESULT_COUNT];
  // Positions in one page, declared by the model source.
  iree_host_size_t block_size;
  // Byte origin of row-major page maps in each device origin table.
  iree_device_size_t map_origin;
  // Maximum logical blocks per row, independent of physical capacity.
  iree_host_size_t blocks_per_row;
  // Byte extent of one row's retained MTP hidden state.
  iree_device_size_t carry_stride;
  // Split between first and second MTP feedback banks, in bytes.
  iree_device_size_t feedback_split;
  // Recurrent state slots in the state arena, independent of row identity.
  struct {
    // Byte origin of the first slot.
    iree_device_size_t origin;
    // Byte distance between adjacent slot origins.
    iree_device_size_t stride;
    // Bytes initialized and retained for one slot.
    iree_device_size_t length;
    // Number of independently owned slots.
    uint32_t capacity;
  } recurrent;
  // Owned source-declared cache plane groups.
  loom_serve_text_cache_region_t* regions;
  // Number of validated cache plane groups.
  iree_host_size_t region_count;
} loom_serve_text_storage_t;

// Moves bootstrap results into storage and validates the external model
// records. Origin payload contents and record widths are source-owned; native
// code only uploads them before the separate page-map extent. Partial
// initialization remains safe to deinitialize on failure.
iree_status_t loom_serve_text_storage_initialize(
    const iree_vm_ref_types_t* types, iree_vm_variant_t* results,
    iree_host_size_t row_count, iree_host_size_t context_capacity,
    iree_host_size_t pool_capacity, loom_serve_text_storage_t* out_storage,
    iree_allocator_t allocator);

// After initialization transfers retire, discard their source payloads while
// keeping parsed geometry for growth, compaction and retained-state transfers.
void loom_serve_text_storage_release_payloads(
    loom_serve_text_storage_t* storage);
void loom_serve_text_storage_deinitialize(loom_serve_text_storage_t* storage);

// Builds a cold logical-order transfer plan from the source geometry. The five
// private views are row bindings 1-5; null views are omitted. A row supplies
// control, recurrence and input/progress; an independent endpoint supplies only
// recurrence. carry_index selects the source-declared row or checkpoint carry
// entry. Workspace and proposal scratch are not retained. The caller owns the
// returned array. Capture and restore supply their respective physical IDs.
iree_status_t loom_serve_text_storage_plan_snapshot(
    const loom_serve_text_storage_t* storage, iree_host_size_t carry_index,
    iree_hal_buffer_t* const* private_views, bool enable_mtp,
    uint32_t block_count, const uint32_t* blocks, iree_host_size_t* out_count,
    loom_serve_snapshot_range_t** out_ranges, iree_allocator_t allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_TEXT_STORAGE_H_
