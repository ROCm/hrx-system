// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU fixed-segment layout for function-local storage.
//
// This layout is the single contract shared by kernel metadata and native
// encoding. Each low.storage.reserve inside a target-low function is packed in
// function body order into the target segment selected by its storage type,
// with the reservation's requested alignment applied before assigning its byte
// offset.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_STORAGE_LAYOUT_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_STORAGE_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/storage_layout.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_amdgpu_storage_layout_segment_sizes_t {
  // Fixed bytes of workgroup storage required by low.storage.reserve ops.
  uint64_t group_segment_fixed_size;
  // Fixed bytes of invocation-private storage required by low.storage.reserve
  // ops.
  uint64_t private_segment_fixed_size;
} loom_amdgpu_storage_layout_segment_sizes_t;

// AMDGPU target-segment placement for one low.storage.reserve result. The
// storage space remains the original low space, while byte_offset is relative
// to the projected AMDGPU segment.
typedef loom_low_storage_layout_reservation_t
    loom_amdgpu_storage_layout_reservation_t;

// AMDGPU storage layout record in function storage declaration order.
typedef loom_low_storage_layout_record_t loom_amdgpu_storage_layout_record_t;

// AMDGPU resolved storage reference after applying low.storage.view offsets to
// the target-segment reservation.
typedef loom_low_storage_layout_reference_t
    loom_amdgpu_storage_layout_reference_t;

typedef struct loom_amdgpu_storage_layout_t {
  // Fixed segment sizes after laying out all function-local storage.
  loom_amdgpu_storage_layout_segment_sizes_t segment_sizes;
  // Arena-owned records in function storage layout order.
  const loom_amdgpu_storage_layout_record_t* records;
  // Number of records in |records|.
  iree_host_size_t record_count;
} loom_amdgpu_storage_layout_t;

// Computes fixed AMDGPU segment sizes for every low.storage.reserve in
// |function_op|. Scratch and private storage both contribute to the AMDGPU
// private segment; stack storage is rejected until its ABI lowering is
// represented in the native path.
iree_status_t loom_amdgpu_storage_layout_collect_segment_sizes(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_amdgpu_storage_layout_segment_sizes_t* out_sizes);

// Builds the full fixed-segment layout for every low.storage.reserve in
// |function_op|. Records are arena-owned and stay valid for the arena lifetime.
iree_status_t loom_amdgpu_storage_layout_build(
    const loom_module_t* module, const loom_op_t* function_op,
    iree_arena_allocator_t* arena, loom_amdgpu_storage_layout_t* out_layout);

// Looks up |storage_value_id| in a previously built layout.
iree_status_t loom_amdgpu_storage_layout_lookup(
    const loom_amdgpu_storage_layout_t* layout,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reservation_t* out_reservation);

// Resolves a low.storage.reserve or low.storage.view handle against a
// previously built AMDGPU fixed-segment layout.
iree_status_t loom_amdgpu_storage_layout_lookup_reference(
    const loom_amdgpu_storage_layout_t* layout, const loom_module_t* module,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reference_t* out_reference);

// Resolves one storage reservation from |function_op| without materializing the
// complete layout record table.
iree_status_t loom_amdgpu_storage_layout_resolve(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reservation_t* out_reservation);

// Resolves a low.storage.reserve or low.storage.view handle from |function_op|
// without materializing the complete layout record table.
iree_status_t loom_amdgpu_storage_layout_resolve_reference(
    const loom_module_t* module, const loom_op_t* function_op,
    loom_value_id_t storage_value_id,
    loom_amdgpu_storage_layout_reference_t* out_reference);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_STORAGE_LAYOUT_H_
