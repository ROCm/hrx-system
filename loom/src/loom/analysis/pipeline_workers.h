// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_WORKERS_H_
#define LOOM_ANALYSIS_PIPELINE_WORKERS_H_

#include "loom/analysis/channel_completion.h"
#include "loom/analysis/kernel_async_legality.h"
#include "loom/analysis/pipeline_resources.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/fact_cfg.h"

#ifdef __cplusplus
extern "C" {
#endif

// Actual channel correspondence at a construction or worker boundary. Storage
// and protocol identities remain owned by the construction that created them.
typedef struct loom_pipeline_channel_binding_t {
  // Value in the receiving source scope.
  loom_value_id_t value_id;
  // Construction occurrence owning the channel and its allocation placements.
  const loom_pipeline_resources_t* resources;
  // Binding in resources, independent of its storage address or source symbol.
  const loom_pipeline_resource_channel_t* channel;
} loom_pipeline_channel_binding_t;

// Specialized authored worker geometry. This is a domain of execution units,
// not a count of records and not a request to repeat the strand body per
// record.
typedef struct loom_pipeline_worker_axis_t {
  // First worker coordinate along this axis.
  uint64_t origin;
  // Number of workers along this axis.
  uint64_t count;
  // Coordinate increment between workers along this axis.
  uint64_t stride;
} loom_pipeline_worker_axis_t;

typedef enum loom_pipeline_transport_step_kind_e {
  LOOM_PIPELINE_TRANSPORT_STEP_CHANNEL,
  LOOM_PIPELINE_TRANSPORT_STEP_TRANSFER,
  LOOM_PIPELINE_TRANSPORT_STEP_WAIT,
} loom_pipeline_transport_step_kind_t;

// An observable step in a straight-line communication-only invocation. Pure
// address calculations remain represented by the movement endpoint facts.
typedef struct loom_pipeline_transport_step_t {
  // Source effect and its position in the invocation's sequential order.
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
  // Arena-owned destinations in fanout result order.
  const loom_pipeline_transport_destination_t* destinations;
  // Number of independently completing destinations.
  iree_host_size_t destination_count;
} loom_pipeline_transport_cohort_t;

typedef struct loom_pipeline_transport_t {
  // All observable effects, in execution order; NULL when ordinary execution
  // is required. This does not select a device engine or relax any ordering.
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
  // Number of executions of the complete ordered sequence. A straight-line
  // invocation has one; a regular loop has its proven exact trip count.
  uint64_t repetitions;
  // Controlling recurrence for repeated execution, or NULL for straight-line
  // execution. Borrowed from the worker's retained CFG snapshot.
  const loom_value_fact_induction_t* induction;
  // External source roots cannot change during this closed execution. This
  // proof includes every strand's effects and scoped alias contracts.
  bool sources_stable;
} loom_pipeline_transport_t;

typedef struct loom_pipeline_worker_t {
  // Authored strand and its ordinary call, borrowed through consuming rewrite.
  const loom_pipeline_resource_strand_t* source;
  // Visible body bound by the call's actual captures. Shared definitions remain
  // immutable until the materialization owner has assigned distinct instances.
  loom_func_like_t function;
  // Arena-owned worker geometry in source axis order.
  const loom_pipeline_worker_axis_t* axes;
  // Number of worker axes.
  iree_host_size_t rank;
  // Arena-owned formal channel bindings, in formal argument order.
  const loom_pipeline_channel_binding_t* bindings;
  // Number of channel formals in bindings.
  iree_host_size_t binding_count;
  // Facts seeded from the actual captures, including storage projections.
  loom_value_fact_table_t facts;
  // Retained ordinal domain. Released after construction; restore before
  // querying ordinal-indexed plans or starting their consuming rewrite.
  loom_local_value_domain_t value_domain;
  // Communication membership and actions established in this occurrence.
  loom_channel_plan_t channels;
  // Aggregate FIFO credit deltas, or the record-aware realization requirement.
  loom_channel_completion_t completion;
  // Borrowed asynchronous movement and its admitted group completion edges.
  loom_kernel_async_legality_result_t asynchronous;
  // Complete communication-only execution when no residual effects or control
  // require a processor. Physical owners consume this instead of walking IR.
  loom_pipeline_transport_t transport;
  // CFG topology retained for scheduling and handoff analysis.
  const loom_cfg_graph_t* graph;
  // Exact invocation execution counts per CFG block, or NULL when conditional
  // control or unknown recurrence prevents a complete count proof.
  const uint64_t* block_execution_counts;
} loom_pipeline_worker_t;

// Binds every outlined strand in one construction occurrence. The construction
// facts and local resources are already computed. incoming is indexed in
// ascending value_id order. Incoming channels retain
// their originating construction instead of being reconstructed from addresses
// or payload types. callables supplies visible channel-aware helper summaries
// when such calls survive source inlining; ordinary calls need no summary.
//
// Each worker receives its own facts and communication plan, even when several
// strands call the same function. Target selection consumes these results and
// never rewalks source to recover captures, placement, or protocol membership.
// No source mutation occurs. All plans and their domains live in arena, while
// the module ordinal scratch is available on return. Rejected authored input
// emits a diagnostic and leaves out_valid false; status carries allocation or
// diagnostic emission failure. The caller admits all resources before mutation.
iree_status_t loom_pipeline_workers_build(
    loom_module_t* module, const loom_pipeline_resources_t* resources,
    const loom_value_fact_table_t* construction_facts,
    const loom_pipeline_channel_binding_t* incoming,
    iree_host_size_t incoming_count,
    const loom_channel_plan_callable_t* const* callables,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_pipeline_worker_t** out_workers, bool* out_valid);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_WORKERS_H_
