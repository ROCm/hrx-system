// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_CHANNEL_COMPLETION_H_
#define LOOM_ANALYSIS_CHANNEL_COMPLETION_H_

#include "loom/analysis/channel_plan.h"
#include "loom/util/cfg_graph.h"

typedef struct loom_view_region_table_t loom_view_region_table_t;

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_channel_completion_requirement_e {
  LOOM_CHANNEL_COMPLETION_REQUIREMENT_NONE = 0,
  // A channel action needs a realization other than direct FIFO accesses.
  LOOM_CHANNEL_COMPLETION_REQUIREMENT_ACTION,
  // An owned access needs a record identity carried across this boundary.
  LOOM_CHANNEL_COMPLETION_REQUIREMENT_IDENTITY,
  // Incoming paths require different completion state at the same action.
  LOOM_CHANNEL_COMPLETION_REQUIREMENT_CONTROL,
  // A later blocking admission can require an independently reusable slot.
  LOOM_CHANNEL_COMPLETION_REQUIREMENT_REUSE,
} loom_channel_completion_requirement_t;

// Endpoint activity retained at each source action's protocol and direction.
typedef struct loom_channel_completion_action_t {
  // Number of consecutive records made available by each source action.
  // Non-completions have zero.
  uint32_t credits;
  // Conservative bound on admissions by this endpoint during one invocation.
  // Exact block execution counts refine this bound when supplied. Otherwise
  // a cyclic admission has UINT32_MAX; acyclic sites each contribute one.
  // Zero or one proves that this endpoint never advances beyond its first slot.
  uint32_t maximum_admissions;
} loom_channel_completion_action_t;

// Record currently owning one borrowed channel payload view.
typedef struct loom_channel_payload_record_t {
  // Read/write SSA value carrying the record's physical slot.
  loom_value_id_t value_id;
  // Conservative admission bound for the record's channel endpoint.
  uint32_t maximum_admissions;
} loom_channel_payload_record_t;

// Completion of a contiguous FIFO prefix, in owned-record coordinates.
// This is a realization proof, not a restriction on valid channel programs.
typedef struct loom_channel_completion_t {
  // Indexed exactly like channel_plan.actions; valid when requirement is NONE.
  const loom_channel_completion_action_t* actions;
  // Borrowed payload records, including derived views, indexed by value domain;
  // NULL for no-action workers and invalid when requirement is not NONE.
  const loom_channel_payload_record_t* payload_records;
  // Value-domain prefix covered by payload_records.
  loom_value_ordinal_t payload_record_count;
  // Borrowed value domain indexing payload_records.
  const loom_local_value_domain_t* value_domain;
  // Capability required when aggregate prefix credits cannot implement source.
  loom_channel_completion_requirement_t requirement;
  // Action or control edge exposing that requirement, or NULL on success.
  const loom_op_t* op;
} loom_channel_completion_t;

// Computes FIFO publication/reclamation frontiers and admission bounds for one
// immutable worker invocation starting at each endpoint's initial slot.
// The acquired value domain, channel actions and CFG are retained by the same
// owner. Source ownership has already established distinct live obligations.
// Each direction has one cursor owner; selection establishes this separately.
//
// A later completion produces no credit until its predecessors complete. The
// final completion of that prefix exposes all its records. Wait borrows an
// already acquired read without changing the frontier. Obligation names
// rotate simultaneously across CFG arguments, so loop-carried records need no
// absolute sequence numbers. Compatible incoming states share constant credit
// deltas; incompatible states require runtime record identity/frontier state.
//
// Deferred reader reclamation cannot cross another channel admission: delaying
// those credits could introduce a wait cycle that independently reusable slots
// would avoid. Such programs require a different physical realization. Ordinary
// payload operations cannot acquire channels through a hidden side channel.
// This proof establishes neither payload visibility nor DMA completion; the
// selected transport and asynchronous-access analysis own those contracts.
//
// Status carries allocation failure. Requirements describe valid source needing
// another realization. No target diagnostics or source mutations occur here.
// No-action workers allocate nothing. Other state lives in arena through the
// consuming rewrite; emission uses the direct action-indexed result table.
// When borrowed payload correspondence is needed, |view_regions| is the
// complete analyzed canonical view table for the same value domain; CFG
// projection uses it to retain channel ownership through derived views. NULL
// avoids its table and dataflow cost when the worker has no payload consumer.
// Optional block_execution_counts contains exact counts from the shared CFG
// recurrence analysis; NULL retains conservative cyclic admission bounds.
iree_status_t loom_channel_completion_analyze(
    const loom_channel_plan_t* plan, const loom_cfg_graph_t* graph,
    const loom_view_region_table_t* view_regions,
    const uint64_t* block_execution_counts, iree_arena_allocator_t* arena,
    loom_channel_completion_t* out_completion);

// Looks up the current owned record for one borrowed payload view.
//
// Returns false when the value is not a channel payload or when control flow
// does not carry the correlated record needed by a physical realization. The
// completion's retained local value domain must be active during the query.
bool loom_channel_completion_try_lookup_payload_record(
    const loom_channel_completion_t* completion, loom_value_id_t payload,
    loom_channel_payload_record_t* out_record);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_CHANNEL_COMPLETION_H_
