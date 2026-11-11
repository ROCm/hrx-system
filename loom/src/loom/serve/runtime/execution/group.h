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

// An owner-serialized, single-use group. Append copies
// tables and upload/fill payloads and retains every device resource. Submission
// preserves node order, including feedback before a subsequent overwrite.
// No source callback runs under the lane's submission lock.
typedef struct loom_serve_execution_group_t loom_serve_execution_group_t;

enum { LOOM_SERVE_GROUP_MAX_LANES = 8 };

typedef enum loom_serve_group_node_kind_e {
  LOOM_SERVE_GROUP_EXECUTE,
  LOOM_SERVE_GROUP_DISPATCH,
  LOOM_SERVE_GROUP_TRANSFER,
  LOOM_SERVE_GROUP_BARRIER,
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
  // Optional independently published readiness. The transport owner signals
  // this only after payload visibility; queue completion still owns retirement.
  // Append retains the semaphore. Submission does not signal it automatically.
  iree_hal_semaphore_t* payload_semaphore;
  uint64_t payload_value;
  // Optional external prerequisite in addition to graph and FIFO dependencies.
  iree_hal_semaphore_t* wait_semaphore;
  uint64_t wait_value;
} loom_serve_group_node_t;

typedef enum loom_serve_group_edge_kind_e {
  LOOM_SERVE_GROUP_EDGE_TERMINAL = 0,
  LOOM_SERVE_GROUP_EDGE_PAYLOAD,
} loom_serve_group_edge_kind_t;

typedef struct loom_serve_group_edge_t {
  uint32_t predecessor;
  loom_serve_group_edge_kind_t kind;
} loom_serve_group_edge_t;

// Accepted prefix is reported even on error. frontier is always a snapshot of
// actual acceptance, never a promised signal after a rejected enqueue. Failure
// after acceptance is terminal for this execution: retire, never replay.
typedef struct loom_serve_group_submission_t {
  iree_host_size_t accepted_count;
  loom_serve_execution_frontier_t frontier;
  // Per-lane acceptance survives a rejected node or terminal join. Lane zero
  // is the primary execution supplied at creation. frontier covers all lanes
  // only when joined is true. Single-lane groups need no additional join.
  bool joined;
  iree_host_size_t lane_count;
  loom_serve_execution_frontier_t lanes[LOOM_SERVE_GROUP_MAX_LANES];
} loom_serve_group_submission_t;

// Capacity is an explicit upper bound (1..65536 nodes), allocated before any
// enqueue. Empty groups cannot submit; discarded/submitted groups cannot
// append.
iree_status_t loom_serve_execution_group_create(
    loom_serve_execution_t* execution, iree_host_size_t capacity,
    iree_allocator_t allocator, loom_serve_execution_group_t** out_group);
iree_status_t loom_serve_execution_group_append(
    loom_serve_execution_group_t* group, const loom_serve_group_node_t* node);

// Appends to an exact execution lane after the preceding node, retaining the
// lane. NULL selects the primary lane. All lanes must belong to one HAL group.
// Shared buffers must admit every consuming family. Cross-device HAL waits
// may use host deferral. These calls do not implement a resident protocol.
iree_status_t loom_serve_execution_group_append_on(
    loom_serve_execution_group_t* group, loom_serve_execution_t* execution,
    const loom_serve_group_node_t* node);

// Appends a DAG node with explicit earlier-node dependencies. An empty list
// permits overlap with preceding nodes on other lanes. FIFO ordering within
// each lane and all pre-group work/feedback are always preserved. The caller
// declares every shared-state edge and joins disjoint output branches before
// consumption. At most eight predecessors are accepted per node. On success
// out_index receives a stable ordinal. Rejection leaves the group unchanged.
iree_status_t loom_serve_execution_group_append_after(
    loom_serve_execution_group_t* group, loom_serve_execution_t* execution,
    const loom_serve_group_node_t* node, iree_host_size_t predecessor_count,
    const uint32_t* predecessors, uint32_t* out_index);

// Payload edges wait on the producer's independent readiness token, permitting
// a consumer to acknowledge data before the producer retires. They require
// distinct lanes; FIFO still orders every command within a lane. The transport
// owner supplies visibility, admission/abort and ACK, and must resolve
// readiness or fail it on errors. The terminal join and accepted-lane ledger
// always cover producer completion, including when a payload edge allowed an
// earlier read.
iree_status_t loom_serve_execution_group_append_edges(
    loom_serve_execution_group_t* group, loom_serve_execution_t* execution,
    const loom_serve_group_node_t* node, iree_host_size_t edge_count,
    const loom_serve_group_edge_t* edges, uint32_t* out_index);
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

// Applies the checks the lane's HAL queues perform at enqueue (queue family,
// buffer usage/access/memory type, flags, fill alignment, copy overlap), so a
// faulty node is rejected at append before any node of its group is accepted.
iree_status_t loom_serve_execution_validate_group_node(
    loom_serve_execution_t* execution, const loom_serve_group_node_t* node);

// Lane primitive used by the owned group after all nodes have been validated.
// Checks capacity for the whole batch under the lock before its first enqueue.
iree_status_t loom_serve_execution_submit_group(
    loom_serve_execution_t* execution, iree_host_size_t count,
    const loom_serve_group_node_t* nodes,
    loom_serve_group_submission_t* out_submission);

// Owned group implementation data. Dependencies reference earlier nodes in
// append order. The submitter writes accepted values for downstream edges.
typedef struct loom_serve_group_dependency_t {
  uint32_t lane;
  uint32_t count;
  uint32_t predecessors[LOOM_SERVE_GROUP_MAX_LANES];
  uint32_t payload_mask;
  uint64_t value;
} loom_serve_group_dependency_t;

iree_status_t loom_serve_execution_submit_multi_group(
    iree_host_size_t lane_count, loom_serve_execution_t* const* lanes,
    iree_host_size_t count, const loom_serve_group_node_t* nodes,
    loom_serve_group_dependency_t* dependencies,
    loom_serve_group_submission_t* out_submission);

bool loom_serve_execution_shares_group(loom_serve_execution_t* lhs,
                                       loom_serve_execution_t* rhs);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_RUNTIME_EXECUTION_GROUP_H_
