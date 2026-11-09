// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_INVOCATIONS_H_
#define LOOM_ANALYSIS_PIPELINE_INVOCATIONS_H_

#include "loom/analysis/pipeline/construction_storage.h"
#include "loom/analysis/pipeline/lifetimes.h"
#include "loom/analysis/pipeline/workers.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_pipeline_invocation_t loom_pipeline_invocation_t;
typedef struct loom_pipeline_invocation_schedule_t
    loom_pipeline_invocation_schedule_t;

// A storage root visible in one construction occurrence. Imported roots retain
// their original placement; the same child definition can allocate different
// storage at several call sites without conflating their source value IDs.
typedef struct loom_pipeline_invocation_allocation_t {
  // Source root in the occurrence's retained facts.
  loom_value_id_t root_value_id;
  // Placement owned by the construction that allocated this root.
  const loom_pipeline_storage_allocation_t* allocation;
} loom_pipeline_invocation_allocation_t;

// Parallel to one retained schedule step. Its kind belongs to the source plan;
// non-composition and non-serial steps have an empty binding.
typedef union loom_pipeline_invocation_step_t {
  // The child occurrence selected by a COMPOSITION step.
  const loom_pipeline_invocation_t* child;
  // Bound control selected by a nested SERIAL step.
  const loom_pipeline_invocation_schedule_t* nested;
  // Canonical protocol consumed by a COLLECTIVE step.
  const loom_pipeline_resource_channel_t* channel;
} loom_pipeline_invocation_step_t;

struct loom_pipeline_invocation_schedule_t {
  // Immutable source control and construction classifications.
  const loom_pipeline_schedule_t* source;
  // Per-block bindings indexed exactly like source->blocks[b].steps.
  const loom_pipeline_invocation_step_t* const* blocks;
};

// A static child call site with its actual resource correspondence. A loop
// revisits this occurrence; its run arguments vary without making another
// construction plan or treating the callee symbol as a resource identity.
struct loom_pipeline_invocation_t {
  // Position in the graph's parent-before-child occurrence table.
  iree_host_size_t ordinal;
  // Source callable whose construction is being bound.
  loom_func_like_t function;
  // Actual composition, or NULL for the root invocation.
  loom_op_t* call;
  // Enclosing construction, or NULL at the root.
  const loom_pipeline_invocation_t* parent;
  // Source serial scope joining this child, or NULL for concurrent composition.
  const loom_pipeline_schedule_t* schedule;
  // Immutable facts seeded from this occurrence's actual arguments.
  const loom_value_fact_table_t* facts;
  // Local resource identities and requirements; parent resources stay separate.
  const loom_pipeline_resources_t* resources;
  // Placed local allocations, indexed like resources->allocations. Imported
  // placements retain their original owner through the visible table below.
  const loom_pipeline_storage_allocation_t* storage;
  // Visible allocation placements, sorted by source root ID.
  const loom_pipeline_invocation_allocation_t* allocations;
  // Number of visible allocation roots.
  iree_host_size_t allocation_count;
  // Captured resource identities in the receiving callable's formal IDs.
  loom_pipeline_arguments_t arguments;
  // First worker ordinal in the complete resident inventory.
  iree_host_size_t first_worker;
  // Concurrent children, indexed like resources->compositions.
  const loom_pipeline_invocation_t* const* compositions;
  // Bound schedules, indexed like resources->schedules.
  const loom_pipeline_invocation_schedule_t* schedules;
};

typedef struct loom_pipeline_invocation_worker_t {
  // Construction binding the worker's resources and execution lifetime.
  const loom_pipeline_invocation_t* invocation;
  // Retained strand declaration and call supplying geometry and captures.
  const loom_pipeline_resource_strand_t* strand;
} loom_pipeline_invocation_worker_t;

typedef struct loom_pipeline_invocation_channel_t {
  // Construction owning this protocol instance.
  const loom_pipeline_invocation_t* invocation;
  // Fresh protocol, independent of source value IDs in other occurrences.
  const loom_pipeline_resource_channel_t* source;
  // Backing placement, or NULL for caller-owned external storage.
  const loom_pipeline_storage_allocation_t* allocation;
} loom_pipeline_invocation_channel_t;

// Complete static inventory in construction order. Runtime loops revisit these
// occurrences. Physical consumers iterate the retained tables rather than
// walking child control or recovering ownership from source IDs.
typedef struct loom_pipeline_invocations_t {
  // Parent-before-child constructions; the first entry is the root.
  const loom_pipeline_invocation_t* const* entries;
  // Number of static construction occurrences.
  iree_host_size_t count;
  // All workers, grouped by construction and then local strand order.
  const loom_pipeline_invocation_worker_t* workers;
  // Number of static worker occurrences, including independent outer strands.
  iree_host_size_t worker_count;
  // All fresh protocols in construction order.
  const loom_pipeline_invocation_channel_t* channels;
  // Number of distinct protocol instances.
  iree_host_size_t channel_count;
  // Channel ordinals sorted by canonical identity address for indexed lookup.
  const iree_host_size_t* channel_order;
  // Complete construction coexistence tree, owned by this invocation graph.
  loom_pipeline_lifetimes_t lifetimes;
} loom_pipeline_invocations_t;

typedef struct loom_pipeline_invocation_environment_t {
  // Returns the already selected function environment. A NULL callback keeps
  // inherited target-neutral inference; it does not select a physical target.
  const loom_target_facts_t* (*target_facts)(void* user_data,
                                             loom_func_like_t function);
  // Borrowed owner of the selected environments.
  void* user_data;
} loom_pipeline_invocation_environment_t;

// Binds the retained child graph of one immutable source construction.
// Resources and facts for the root are borrowed. Child facts,
// argument bindings, and constructions live in arena. This owner visits
// retained control rather than rewalking source regions to rediscover
// composition. It retains the exact step-to-child correspondence for physical
// consumers.
//
// Runtime CFG loops stay in their source schedule. Recursive construction and
// unavailable child bodies require another realization and emit a diagnostic.
// Construction lifetimes determine final storage placement: serialized child
// lifetimes share ranges, while parent-owned and concurrent storage coexist.
// The graph and its immutable placements are published together.
// Source mutation invalidates the graph. A consuming projection translates its
// retained sites and keeps outer protocol identity across child completion.
// Worker calls remain source sites. Their complete inventory lets realization
// instantiate every worker before binding its analysis, including repeated
// child calls that share an authored definition. This analysis assigns no
// execution engines, cursors, or completion mechanics.
iree_status_t loom_pipeline_invocations_build(
    loom_module_t* module, loom_func_like_t root_function,
    const loom_value_fact_table_t* root_facts,
    const loom_pipeline_resources_t* root_resources,
    const loom_pipeline_resource_memory_t* memories,
    iree_host_size_t memory_count,
    loom_pipeline_invocation_environment_t environment,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_pipeline_invocations_t* out_invocations, bool* out_valid);

// Binds geometry, communication, and asynchronous completion for the complete
// worker inventory. functions optionally supplies one independent callable per
// inventory worker; NULL analyzes the original definitions without mutation.
// The returned array uses invocations->workers order. Each worker is analyzed
// once after its construction bindings and final callable identity are known.
// Incoming resources do not establish phase-local source stability: effects of
// concurrent outer siblings still belong to the admitting owner.
iree_status_t loom_pipeline_invocations_analyze_workers(
    loom_module_t* module, const loom_pipeline_invocations_t* invocations,
    const loom_func_like_t* functions,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_pipeline_worker_t** out_workers, bool* out_valid);

// Returns the retained placement for a visible root, or NULL for external
// storage. This lookup does not traverse parents or recompute alias facts.
const loom_pipeline_storage_allocation_t*
loom_pipeline_invocation_lookup_allocation(
    const loom_pipeline_invocation_t* invocation, loom_value_id_t root);

// Returns the protocol's construction-order ordinal, or IREE_HOST_SIZE_MAX
// when it was created outside pipeline construction (for example in a worker).
iree_host_size_t loom_pipeline_invocations_channel_index(
    const loom_pipeline_invocations_t* invocations,
    const loom_channel_identity_t* identity);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_INVOCATIONS_H_
