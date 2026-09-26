// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU source allocation layout analysis.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_SOURCE_ALLOCA_LAYOUT_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_SOURCE_ALLOCA_LAYOUT_H_

#include <stdbool.h>
#include <stdint.h>

#include "loom/codegen/low/lower/lower.h"
#include "loom/ir/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_target_low_legality_context_t
    loom_target_low_legality_context_t;
typedef struct loom_amdgpu_source_alloca_layout_t
    loom_amdgpu_source_alloca_layout_t;

typedef struct loom_amdgpu_source_alloca_storage_requirement_t {
  // Packed high-water extent in bytes.
  uint64_t byte_length;
  // Strongest base alignment required by any packed source allocation.
  uint64_t byte_alignment;
} loom_amdgpu_source_alloca_storage_requirement_t;

// Returns an initialized analysis with no source allocation entries. This is
// used by cold helper paths that intentionally lack a source-to-low or
// low-legality context; it is still a real analysis object, not a nullable
// cache sentinel.
const loom_amdgpu_source_alloca_layout_t*
loom_amdgpu_source_alloca_layout_empty(void);

// Returns the function-local source allocation layout analysis for |context|.
// The returned object is allocated from the lowering arena and remains valid
// until the current source function lowering finishes.
iree_status_t loom_amdgpu_source_alloca_layout_for_lower_context(
    loom_low_lower_context_t* context,
    const loom_amdgpu_source_alloca_layout_t** out_layout);

// Records one selected source buffer allocation in the lowering analysis.
// Source-to-low planning calls this while visiting buffer.alloca ops, so later
// selectors can resolve allocation roots without scanning source IR.
iree_status_t loom_amdgpu_source_alloca_layout_record_lower_alloca(
    loom_low_lower_context_t* context, const loom_op_t* alloca_op,
    uint64_t byte_length);

// Records the verified borrowed workgroup tail root without reserving its
// launch-dependent extent in the fixed source allocation arena.
iree_status_t loom_amdgpu_source_alloca_layout_record_lower_workgroup_tail(
    loom_low_lower_context_t* context, loom_value_id_t root_value_id);

// Emits one physical low-storage arena for each populated memory space. Source
// allocation plans are complete before entry setup, so each arena carries the
// final packed extent and strongest required base alignment.
iree_status_t loom_amdgpu_source_alloca_layout_emit_low_storage_roots(
    loom_low_lower_context_t* context);

// Returns the function-local source allocation layout analysis for low-legality
// verification. The returned object is allocated from the legality context's
// scratch arena and populated by the existing target-low verification walk.
iree_status_t loom_amdgpu_source_alloca_layout_for_low_legality(
    loom_target_low_legality_context_t* context,
    const loom_amdgpu_source_alloca_layout_t** out_layout);

// Records one source buffer allocation in the low-legality analysis. Target-low
// verification calls this while visiting buffer.alloca ops, so later provider
// checks can resolve allocation roots without scanning source IR.
iree_status_t loom_amdgpu_source_alloca_layout_record_low_legality_alloca(
    loom_target_low_legality_context_t* context, const loom_op_t* alloca_op,
    uint64_t byte_length);

// Records the borrowed workgroup tail root during the existing legality walk.
iree_status_t
loom_amdgpu_source_alloca_layout_record_low_legality_workgroup_tail(
    loom_target_low_legality_context_t* context, loom_value_id_t root_value_id);

// Returns whether |root_value_id| is the recorded symbolic workgroup tail. Its
// native byte base is resolved after all fixed Low reservations are complete.
bool loom_amdgpu_source_alloca_layout_is_workgroup_tail(
    const loom_amdgpu_source_alloca_layout_t* layout,
    loom_value_id_t root_value_id);

// Returns the physical storage-root requirement retained for |memory_space|.
// Returns false when no selected source allocation occupies that space.
bool loom_amdgpu_source_alloca_layout_storage_requirement(
    const loom_amdgpu_source_alloca_layout_t* layout,
    loom_value_fact_memory_space_t memory_space,
    loom_amdgpu_source_alloca_storage_requirement_t* out_requirement);

// Resolves the analyzed packed byte offset for a source buffer.alloca root in
// the requested memory space. Returns false when the analysis cannot prove the
// root has a statically encodable offset in that memory space.
bool loom_amdgpu_source_alloca_layout_lookup_byte_offset(
    const loom_amdgpu_source_alloca_layout_t* layout,
    loom_value_fact_memory_space_t memory_space, loom_value_id_t root_value_id,
    uint64_t* out_byte_offset);

// Resolves the emitted low-storage arena and packed byte offset for a planned
// source allocation. Entry setup must have emitted the arena before this is
// called during body lowering.
void loom_amdgpu_source_alloca_layout_lookup_low_storage(
    const loom_amdgpu_source_alloca_layout_t* layout,
    loom_value_fact_memory_space_t memory_space, loom_value_id_t root_value_id,
    loom_value_id_t* out_storage_value_id, int64_t* out_byte_offset);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_SOURCE_ALLOCA_LAYOUT_H_
