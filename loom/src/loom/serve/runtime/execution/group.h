// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_EXECUTION_GROUP_H_
#define LOOM_SERVE_RUNTIME_EXECUTION_GROUP_H_

#include "loom/serve/runtime/execution/execution.h"

#ifdef __cplusplus
extern "C" {
#endif

// An owner-serialized, single-use group on one execution lane. Append copies
// tables and upload/fill payloads and retains every device resource. Submission
// preserves node order, including feedback before a subsequent overwrite.
// No source callback runs under the lane's submission lock.
typedef struct loom_serve_execution_group_t loom_serve_execution_group_t;

typedef enum loom_serve_group_node_kind_e {
  LOOM_SERVE_GROUP_EXECUTE,
  LOOM_SERVE_GROUP_DISPATCH,
  LOOM_SERVE_GROUP_TRANSFER,
} loom_serve_group_node_kind_t;

typedef struct loom_serve_group_node_t {
  loom_serve_group_node_kind_t kind;
  iree_hal_command_buffer_t* command_buffer;
  iree_hal_buffer_binding_table_t bindings;
  iree_hal_executable_t* executable;
  iree_hal_executable_function_t function;
  iree_hal_dispatch_config_t config;
  iree_hal_buffer_ref_list_t refs;
  iree_hal_transfer_operation_t transfer;
} loom_serve_group_node_t;

// Accepted prefix is reported even on error. frontier is always a snapshot of
// actual acceptance, never a promised signal after a rejected enqueue. Failure
// after acceptance is terminal for this execution: retire, never replay.
typedef struct loom_serve_group_submission_t {
  iree_host_size_t accepted_count;
  loom_serve_execution_frontier_t frontier;
} loom_serve_group_submission_t;

// Capacity is an explicit upper bound (1..65536 nodes), allocated before any
// enqueue. Empty groups cannot submit; discarded/submitted groups cannot
// append.
iree_status_t loom_serve_execution_group_create(
    loom_serve_execution_t* execution, iree_host_size_t capacity,
    iree_allocator_t allocator, loom_serve_execution_group_t** out_group);
iree_status_t loom_serve_execution_group_append(
    loom_serve_execution_group_t* group, const loom_serve_group_node_t* node);
iree_status_t loom_serve_execution_group_submit(
    loom_serve_execution_group_t* group,
    loom_serve_group_submission_t* out_submission);

// Drops caller-side device borrows before observing execution retirement. Host
// transfer payloads survive until free. Safe for partial construction/failure.
void loom_serve_execution_group_retire(loom_serve_execution_group_t* group);
// Before submission, frees immediately (discard). After any accepted work the
// owner first retires the group and observes actual execution retirement or a
// successful frontier. Failed semaphore readiness alone never permits free.
void loom_serve_execution_group_free(loom_serve_execution_group_t* group);

// Lane primitive used by the owned group after all nodes have been validated.
// Checks capacity for the whole batch under the lock before its first enqueue.
iree_status_t loom_serve_execution_submit_group(
    loom_serve_execution_t* execution, iree_host_size_t count,
    const loom_serve_group_node_t* nodes,
    loom_serve_group_submission_t* out_submission);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_RUNTIME_EXECUTION_GROUP_H_
