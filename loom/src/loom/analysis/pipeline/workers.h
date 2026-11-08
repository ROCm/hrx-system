// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_WORKERS_H_
#define LOOM_ANALYSIS_PIPELINE_WORKERS_H_

#include "loom/analysis/channel_completion.h"
#include "loom/analysis/kernel_synchronization_legality.h"
#include "loom/analysis/pipeline/arguments.h"
#include "loom/analysis/pipeline/resources.h"
#include "loom/analysis/storage_access.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/fact_cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_pipeline_transport_step_kind_e {
  LOOM_PIPELINE_TRANSPORT_STEP_CHANNEL,
  LOOM_PIPELINE_TRANSPORT_STEP_TRANSFER,
  LOOM_PIPELINE_TRANSPORT_STEP_WAIT,
} loom_pipeline_transport_step_kind_t;

// An observable communication site. Pure address calculations remain
// represented by the movement endpoint facts.
typedef struct loom_pipeline_transport_step_t {
  // Source effect retained for this communication site.
  const loom_op_t* op;
  // Which retained communication plan supplies this step's operands.
  loom_pipeline_transport_step_kind_t kind;
  // Canonical action or movement, borrowed through physical selection.
  union {
    // Channel obligation established by membership analysis.
    const loom_channel_plan_action_t* channel;
    // Borrowed movement established by asynchronous-access analysis.
    const loom_kernel_async_transfer_t* transfer;
  } source;
} loom_pipeline_transport_step_t;

// One destination whose reservation and source obligation are transferred to
// asynchronous channel transport. Both actions belong to the same immutable
// worker invocation as their enclosing cohort.
typedef struct loom_pipeline_transport_destination_t {
  // Destination FIFO reservation consumed by copy.
  const loom_op_t* reservation;
  // Copy consuming one source obligation and the destination reservation.
  const loom_op_t* copy;
  // Destination protocol identity established by channel membership analysis.
  const loom_channel_identity_t* channel;
} loom_pipeline_transport_destination_t;

// One admitted source record distributed to independently completing
// destination records. Fanout preserves one source-record identity across all
// obligations; the source FIFO receives one reclamation credit only after the
// transport has consumed every obligation. Each destination publishes
// independently when its own copy completes.
typedef struct loom_pipeline_transport_cohort_t {
  // Source FIFO admission supplying the record.
  const loom_channel_plan_action_t* admission;
  // Arena-owned copy sites in structural execution order.
  const loom_pipeline_transport_destination_t* destinations;
  // Number of copy sites, including revisits to the same destination channel.
  iree_host_size_t destination_count;
  // Final release of a retained source after replay, or NULL when copy alone
  // consumes all source obligations in the admission's execution scope.
  const loom_op_t* release;
  // Exact number of distinct source records admitted by this source site.
  uint64_t admissions;
  // Common number of copies per destination site over the invocation, or zero
  // when the sites have different counts or loop scopes.
  uint64_t transfers;
  // Natural loop admitting/releasing this record, or LOOP_NEST_NONE.
  uint16_t admission_scope;
  // Natural loop containing its copies, or LOOP_NEST_NONE. Independent replay
  // loops remain separate batches even when their execution counts agree.
  uint16_t copy_scope;
  // Consecutive source admissions before any copy or release. Separate
  // ownership intervals stay distinct even when their loop scopes agree.
  uint32_t admission_sequence;
} loom_pipeline_transport_cohort_t;

// One copy in structural execution order. Source identity is independent of
// visit order: a retained record may be revisited after other records.
typedef struct loom_pipeline_transport_copy_t {
  // Admitted source record supplying this copy's obligation.
  const loom_pipeline_transport_cohort_t* source;
  // Reservation, copy operation, and destination channel owned by the cohort.
  const loom_pipeline_transport_destination_t* destination;
  // Exact execution count of this copy site over the invocation.
  uint64_t executions;
  // Innermost natural loop containing this copy, or LOOP_NEST_NONE.
  uint16_t scope;
} loom_pipeline_transport_copy_t;

// Copy span of one exact natural loop. The worker analysis owns its extent;
// consumers never reconstruct loop membership from source operations.
typedef struct loom_pipeline_transport_loop_t {
  // First copy within this loop, or NULL when the loop has no copies.
  const loom_pipeline_transport_copy_t* begin;
  // One past the last copy, including nested loops.
  const loom_pipeline_transport_copy_t* end;
  // Exact iterations per entry into this loop.
  uint64_t iterations;
  // Immediately enclosing loop index, or LOOP_NEST_NONE.
  uint16_t parent;
} loom_pipeline_transport_loop_t;

// Consecutive source admissions sharing one ownership interval. Copies retain
// their independent order and loop scopes; uniform replay is a compact subset.
typedef struct loom_pipeline_transport_batch_t {
  // Consecutive source-record cohorts in acquisition order.
  const loom_pipeline_transport_cohort_t* records;
  // Number of distinct source records in one sequence.
  iree_host_size_t record_count;
  // Copy sites in structural execution order within this ownership interval.
  const loom_pipeline_transport_copy_t* copies;
  // Number of copy sites, including independently publishing fanout branches.
  iree_host_size_t copy_count;
  // Complete uniform replays, or zero for a nonuniform ordered copy program.
  uint64_t replays;
} loom_pipeline_transport_batch_t;

typedef struct loom_pipeline_transport_t {
  // All observable sites in structural execution order, visiting loop bodies
  // before their continuations; NULL when ordinary execution is required. For
  // borrowed movement, admission proves one straight-line sequence or repeated
  // body. Record-batch lifetime and replay order are retained separately below.
  const loom_pipeline_transport_step_t* steps;
  // Number of steps in the complete invocation.
  iree_host_size_t count;
  // Borrowed movement streams in CFG execution order, independent of block
  // storage order. Each stream completes its groups before the next block.
  const loom_kernel_async_stream_t* const* streams;
  // Number of movement-bearing blocks on the execution chain.
  iree_host_size_t stream_count;
  // Source-record cohorts whose complete ownership is transferred through
  // channel.copy. Empty for ordinary instruction-executing channel use.
  const loom_pipeline_transport_cohort_t* cohorts;
  // Number of channel transport cohorts in one ordered execution.
  iree_host_size_t cohort_count;
  // Ordered record sequences retaining a common lifetime and destination set.
  const loom_pipeline_transport_batch_t* batches;
  // Number of independently owned source sequences.
  iree_host_size_t batch_count;
  // Exact copy spans indexed by natural loop, retained for nonuniform programs.
  const loom_pipeline_transport_loop_t* copy_loops;
  // Number of copy-loop spans; zero when uniform replay or straight-line.
  iree_host_size_t copy_loop_count;
  // Number of executions of the borrowed-movement sequence. A straight-line
  // invocation has one; a regular loop has its proven exact trip count.
  // Channel-copy batches instead retain their own admission and replay counts.
  uint64_t repetitions;
  // Controlling recurrence for repeated execution, or NULL for straight-line
  // execution. Borrowed from the worker's retained CFG snapshot.
  const loom_value_fact_induction_t* induction;
  // External source roots cannot change during this closed execution. This
  // proof includes every strand's effects and scoped alias contracts.
  bool sources_stable;
} loom_pipeline_transport_t;

// Finite interpretation of one admitted batch's copy order. Iteration stops at
// the ownership scope, so repeated admissions remain a separate lifetime axis.
typedef struct loom_pipeline_transport_copy_cursor_t {
  // Next dynamic copy visit, or end when this source lifetime is complete.
  const loom_pipeline_transport_copy_t* next;
  // One past this batch's structural copy span.
  const loom_pipeline_transport_copy_t* end;
  // Borrowed exact loop spans owned by the transport analysis.
  const loom_pipeline_transport_loop_t* loops;
  // Arena-owned remaining iterations, indexed by loop.
  uint64_t* remaining;
  // Admission scope excluded from expansion, or LOOP_NEST_NONE.
  uint16_t scope;
} loom_pipeline_transport_copy_cursor_t;

// Initializes a cursor for a transport containing a nonuniform copy program.
// Only mutable loop counters allocate; straight-line programs need no storage.
iree_status_t loom_pipeline_transport_copy_cursor_initialize(
    const loom_pipeline_transport_t* transport,
    const loom_pipeline_transport_batch_t* batch, iree_arena_allocator_t* arena,
    loom_pipeline_transport_copy_cursor_t* out_cursor);

// Returns the next copy in exact execution order, or NULL after one lifetime.
const loom_pipeline_transport_copy_t* loom_pipeline_transport_copy_cursor_next(
    loom_pipeline_transport_copy_cursor_t* cursor);

typedef struct loom_pipeline_worker_t {
  // Authored strand and its ordinary call, borrowed through consuming rewrite.
  const loom_pipeline_resource_strand_t* source;
  // Visible body bound by the call's actual captures. Shared definitions remain
  // immutable until the materialization owner has assigned distinct instances.
  loom_func_like_t function;
  // Borrowed construction-owned geometry in source axis order.
  const loom_pipeline_worker_axis_t* axes;
  // Number of worker axes.
  iree_host_size_t rank;
  // Arena-owned channel bindings, indexed by ascending formal value ID.
  const loom_pipeline_channel_binding_t* bindings;
  // Number of channel formals in bindings.
  iree_host_size_t binding_count;
  // Facts seeded from the actual captures, including storage projections.
  loom_value_fact_table_t facts;
  // Closed constant producers not representable by scalar facts, indexed by
  // formal argument. NULL when every capture uses facts or runtime bindings.
  // Producers remain immutable through the consuming worker rewrite.
  const loom_op_t* const* literal_captures;
  // Retained ordinal domain. Released after construction; restore before
  // querying ordinal-indexed plans or starting their consuming rewrite.
  loom_local_value_domain_t value_domain;
  // Communication membership and actions established in this occurrence.
  loom_channel_plan_t channels;
  // FIFO credit deltas and borrowed-payload record correspondence, or the
  // record-aware realization requirement.
  loom_channel_completion_t completion;
  // Borrowed asynchronous movement and its admitted group completion edges.
  loom_kernel_synchronization_legality_result_t asynchronous;
  // Complete communication-only execution when no residual effects or control
  // require a processor. Physical owners consume this instead of walking IR.
  loom_pipeline_transport_t transport;
  // CFG topology retained for scheduling and handoff analysis.
  const loom_cfg_graph_t* graph;
  // Exact invocation execution counts per CFG block, or NULL when conditional
  // control or unknown recurrence prevents a complete count proof.
  const uint64_t* block_execution_counts;
  // Exact loop trip counts retained with block counts, or NULL when unproven.
  const uint64_t* loop_trip_counts;
} loom_pipeline_worker_t;

// Binds every outlined strand in one construction occurrence. The construction
// facts and local resources are already computed. incoming is indexed in
// ascending value_id order. Incoming channels retain
// their originating construction instead of being reconstructed from addresses
// or payload types. callables supplies visible channel-aware helper summaries
// when such calls survive source inlining; ordinary calls need no summary.
// storage_access belongs to the enclosing immutable source snapshot and is
// shared by every concurrent child. This function does not prove external
// source immutability from the effects of only one child.
//
// Each worker receives its own facts and communication plan, even when several
// strands call the same function. Target selection consumes these results and
// never rewalks source to recover captures, placement, or protocol membership.
// functions optionally supplies an independent callable instance per strand;
// NULL analyzes the original definitions. Instances preserve the source call
// signature. The realization owner creates them before this analysis, so all
// facts and selected sites already name the bodies that will be consumed.
// workers has resources->strand_count entries, supplied by the enclosing
// inventory owner so worker ordinals span the complete resident invocation.
// No source mutation occurs. All plans and their domains live in arena, while
// the module ordinal scratch is available on return. Rejected authored input
// emits a diagnostic and leaves out_valid false; status carries allocation or
// diagnostic emission failure. The caller admits all resources before mutation.
iree_status_t loom_pipeline_workers_build(
    loom_module_t* module, const loom_pipeline_resources_t* resources,
    const loom_value_fact_table_t* construction_facts,
    const loom_pipeline_channel_binding_t* incoming,
    iree_host_size_t incoming_count, const loom_func_like_t* functions,
    const loom_channel_plan_callable_t* const* callables,
    loom_storage_access_scope_t* storage_access,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_pipeline_worker_t* workers, bool* out_valid);

// Proves repeated transport sources immutable across a complete concurrent
// execution. The invocation owner supplies every worker occurrence in that
// closed execution, including children receiving incoming channels. An open
// child or a construction with unresolved compositions cannot close this proof.
// Call once after construction and before physical selection or source
// mutation. Uses the same canonical access scope as worker construction;
// modified roots and alias partitions are indexed once for all source queries.
// Unproven sources keep sources_stable false. Status carries arena allocation
// failure.
iree_status_t loom_pipeline_workers_prove_sources(
    loom_storage_access_scope_t* storage_access,
    loom_pipeline_worker_t* workers, iree_host_size_t worker_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_WORKERS_H_
