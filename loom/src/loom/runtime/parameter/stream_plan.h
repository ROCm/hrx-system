// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_PARAMETER_STREAM_PLAN_H_
#define LOOM_RUNTIME_PARAMETER_STREAM_PLAN_H_

#include "loom/runtime/parameter/inputs.h"
#include "loom/runtime/storage/memory.h"
#include "loom/runtime/command/program.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_runtime_weight_span_t {
  // Final placement with a key borrowed from the plan's owned key storage.
  loom_cmd_program_parameter_t parameter;
  // Unique final storage index, resolved once before any queue work.
  iree_host_size_t storage_index;
  // Owned cached preparation command, or null for unchanged bytes.
  iree_hal_command_buffer_t* prepare;
  // Captured source ranges; a command consumes them through input staging.
  loom_runtime_weight_input_recipe_t inputs;
} loom_runtime_weight_span_t;

typedef struct loom_runtime_weight_fragment_t {
  // Borrowed checkpoint identity from the retained parameter plan.
  iree_string_view_t key;
  // File and target byte extents, possibly only part of a tensor.
  iree_io_parameter_span_t range;
  // Resolved checkpoint or captured-host source owning the identity.
  uint32_t source;
} loom_runtime_weight_fragment_t;

typedef struct loom_runtime_weight_direct_read_t {
  // Borrowed consecutive fragments from one source in a mapped window.
  const loom_runtime_weight_fragment_t* fragments;
  // Number of fragments captured by this gather.
  iree_host_size_t count;
  // Payload bytes contributed by these fragments, excluding storage padding.
  uint64_t byte_length;
  // Previous read on the lane and this window's mapping readiness.
  iree_hal_semaphore_t* waits[2];
  // Values paired with waits, retained through provider submission.
  uint64_t wait_values[2];
  // Completion point on the window's existing read lane.
  uint64_t signal_value;
} loom_runtime_weight_direct_read_t;

typedef struct loom_runtime_weight_group_t {
  // Borrowed fragments in this group's mapped storage range.
  loom_runtime_weight_fragment_t* fragments;
  // Unique storage containing every fragment in this group.
  iree_host_size_t storage_index;
  // Number of fragments exposed to the parameter provider.
  iree_host_size_t count;
  // First source run in the plan's direct read array.
  iree_host_size_t read_begin;
  // Exclusive end of the source runs in this window.
  iree_host_size_t read_end;
  // Read window within one physical slab; empty for fixed backing.
  loom_runtime_memory_range_t backing;
  // First read group sharing this physical slab's mapping readiness.
  iree_host_size_t mapping_group;
  // Owned mapping readiness, released after accepted reads capture it.
  iree_hal_semaphore_t* mapped;
  // Read frontier established by this group.
  uint64_t signal_value;
} loom_runtime_weight_group_t;

typedef struct loom_runtime_weight_tensor_plan_t {
  // First read group contributing bytes to a tensor.
  iree_host_size_t begin;
  // Exclusive last read group contributing bytes to that tensor.
  iree_host_size_t end;
  // Previous preparation on the lane owning this tensor's input staging.
  uint64_t prepare_wait;
  // Completion of this tensor's preparation, including its staging reads.
  uint64_t prepare_signal;
  // Last captured-input read needed by this tensor's preparation.
  uint64_t input_signal;
} loom_runtime_weight_tensor_plan_t;

typedef struct loom_runtime_weight_stream_plan_t {
  // Owned read groups, in physical mapping order for elastic backing.
  loom_runtime_weight_group_t* groups;
  // Number of initialized groups, including padding-only read windows.
  iree_host_size_t group_count;
  // Number of distinct physical slabs backing all read groups.
  iree_host_size_t backing_count;
  // Owned contiguous gather fragments borrowed by groups.
  loom_runtime_weight_fragment_t* fragments;
  // Owned source runs carrying each window's provider captures.
  loom_runtime_weight_direct_read_t* reads;
  // Number of nonempty source runs across all windows.
  iree_host_size_t read_count;
  // Owned tensor-to-group intervals consumed by preparation submission.
  loom_runtime_weight_tensor_plan_t* spans;
  // At least one tensor requires a preparation dispatch.
  bool has_preparation;
} loom_runtime_weight_stream_plan_t;

typedef struct loom_runtime_weight_storage_t {
  // Owned unique root retained independently of the caller's output views.
  iree_hal_buffer_t* buffer;
  // Owned virtual reservation, or NULL for explicitly fixed backing.
  loom_runtime_virtual_buffer_t* reservation;
} loom_runtime_weight_storage_t;

// Plans physical read windows and retains each tensor's contributing interval.
// Inputs are verified placement and source facts owned by weights. The returned
// arrays borrow those facts until deinitialization, including after a partial
// allocation failure. No device operations are submitted by this planner.
iree_status_t loom_runtime_weight_stream_plan_initialize(
    iree_host_size_t span_count, const loom_runtime_weight_span_t* spans,
    iree_host_size_t storage_count, const loom_runtime_weight_storage_t* storages,
    const loom_runtime_weight_inputs_t* inputs, bool elastic,
    iree_allocator_t allocator, loom_runtime_weight_stream_plan_t* out_plan);

// Releases plan arrays and any mapping readiness attached by the stream owner.
void loom_runtime_weight_stream_plan_deinitialize(
    loom_runtime_weight_stream_plan_t* plan, iree_allocator_t allocator);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_PARAMETER_STREAM_PLAN_H_
