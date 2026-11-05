// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_ANALYSIS_PIPELINE_RESOURCES_H_
#define LOOM_ANALYSIS_PIPELINE_RESOURCES_H_

#include "loom/analysis/channel_plan.h"
#include "loom/analysis/pipeline/schedule.h"
#include "loom/analysis/source_storage_packing.h"
#include "loom/error/emitter.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// One backing store selected by the admitting compiler plan. The plan supplies
// one row per distinct resource instance, not per target, memory space,
// coordinate, or source pool value. Several pool values may bind to the same
// row. Target placement retains its execution binding and owner alongside this
// table; this analysis neither chooses a device nor interprets physical
// addresses.
typedef struct loom_pipeline_resource_pool_t {
  // Addressing scope requested by allocations using this pool.
  loom_value_fact_memory_space_t memory_space;
  // Available extent relative to this backing store's origin.
  uint64_t byte_capacity;
  // Borrowed canonical packing owned by the admitting plan. Every construction
  // using this backing appends to the same packing, including child
  // invocations. Fixed transport ranges precede source construction; generated
  // services and compiled workers join before final capacity admission.
  loom_source_storage_packing_t* packing;
} loom_pipeline_resource_pool_t;

// One selectable memory in this invocation's worker domain. Several selections
// may name the same backing row. The execution owner supplies this catalog from
// its admitted resources; source analysis never interprets target addresses or
// treats a target profile as a device identity.
typedef struct loom_pipeline_resource_memory_t {
  // Borrowed, nonnegative worker coordinates identifying this selection.
  const uint64_t* coordinates;
  // Number of worker dimensions in coordinates.
  iree_host_size_t rank;
  // Canonical backing row, which also supplies the selected memory space.
  uint32_t pool_index;
} loom_pipeline_resource_memory_t;

typedef struct loom_pipeline_resource_pool_binding_t {
  // Source pool value resolved from an incoming argument or memory selection.
  loom_value_id_t value_id;
  // Canonical backing row in the caller's resource-pool table.
  uint32_t pool_index;
} loom_pipeline_resource_pool_binding_t;

typedef struct loom_pipeline_resource_allocation_t {
  // Fresh source allocation identity, independent of pool and channel identity.
  loom_value_id_t root_value_id;
  // Source pool value selecting the backing store.
  loom_value_id_t pool_value_id;
  // Canonical backing row selected by that value's explicit binding.
  uint32_t pool_index;
  // Immutable byte offset within the backing store.
  uint64_t byte_offset;
  // Complete allocation extent in bytes.
  uint64_t byte_length;
  // Alignment required by this allocation.
  uint64_t byte_alignment;
} loom_pipeline_resource_allocation_t;

typedef struct loom_pipeline_resource_channel_t {
  // Fresh protocol binding, independent of its backing allocation identity.
  loom_op_t* binding;
  // Fresh identity for this construction occurrence, borrowed by strand and
  // child plans. Resources remain at stable addresses after construction.
  loom_channel_identity_t identity;
  // Complete slot-view projection retained from the construction's facts.
  // The footprint is an envelope, not a promise of dense record placement.
  loom_value_fact_view_reference_t storage;
  // Specialized number of records in the authored binding.
  uint64_t capacity;
} loom_pipeline_resource_channel_t;

typedef struct loom_pipeline_resource_strand_t {
  // Source declaration retaining target and worker-domain geometry.
  loom_op_t* declaration;
  // Outlined ordinary call whose operands supply the actual captures.
  // Its operand order is the callee's formal order, not a resource ABI.
  loom_op_t* call;
} loom_pipeline_resource_strand_t;

typedef struct loom_pipeline_resources_t {
  // Borrowed canonical backing rows; the admitting plan outlives this result.
  const loom_pipeline_resource_pool_t* pools;
  // Number of canonical backing rows.
  iree_host_size_t pool_count;
  // Incoming and selected pool identities, sorted by source value for lookup.
  const loom_pipeline_resource_pool_binding_t* pool_bindings;
  // Number of retained pool bindings.
  iree_host_size_t pool_binding_count;
  // Arena-owned source placements, sorted by root identity for indexed lookup.
  const loom_pipeline_resource_allocation_t* allocations;
  // Number of source placements.
  iree_host_size_t allocation_count;
  // Fresh channel bindings, sorted by source identity for indexed lookup.
  const loom_pipeline_resource_channel_t* channels;
  // Number of retained local channel bindings.
  iree_host_size_t channel_count;
  // Nonempty outlined strands in construction order.
  const loom_pipeline_resource_strand_t* strands;
  // Number of retained strand calls.
  iree_host_size_t strand_count;
  // Empty strand declarations with no worker lifetime. Consuming rewrites erase
  // these retained construction sites without rediscovering their bodies.
  loom_op_t* const* empty_strands;
  // Number of empty strand declarations.
  iree_host_size_t empty_strand_count;
  // Child compositions retained at independent execution boundaries. Each call
  // supplies the actual arguments for a distinct child construction; equal
  // callee symbols or targets do not identify one resource instance. The
  // enclosing planner consumes these edges before erasing their source IR.
  loom_op_t* const* compositions;
  // Number of retained child compositions.
  iree_host_size_t composition_count;
  // Independent resident serial scopes. Their control and child lifetimes
  // remain explicit instead of flattening their strands into this occurrence.
  const loom_pipeline_schedule_t* const* schedules;
  // Number of outer serial scopes; nested scopes belong to their parent plan.
  iree_host_size_t schedule_count;
} loom_pipeline_resources_t;

// Resolves invocation-owned allocations in a specialized pipeline construction.
// The caller has inlined child construction sharing one execution boundary and
// supplied canonical backing rows, selectable memories, and incoming pool
// bindings. Memory selections in construction resolve against that invocation's
// catalog. Each (memory space, coordinates) selection has one canonical
// backing; several selections or incoming values may name the same backing.
// This owner visits construction once, consumes retained storage
// facts, and retains channel identities, outlined strand calls, and child
// compositions alongside allocation placements. Child execution boundaries
// remain separate: the enclosing planner binds their formal arguments using
// these retained calls and plans each child's construction with its own
// admitted resources. This result describes this construction's resources,
// not the transitive resources of its children. Every allocation stays live
// throughout the invocation.
// Repeated construction from the same source gets fresh allocation ranges in
// the selected pool's packing. Source IDs identify allocations only within
// this occurrence; they never identify ranges across occurrences. Strand-local
// storage remains with worker compilation and joins the same packings using
// source_storage_packing_reserve before capacity admission.
//
// Allocation construction must be straight-line with finite specialized sizes
// and capacities. Strand bodies must have been outlined into ordinary calls;
// executable control flow inside strands is unrestricted by this analysis.
// Serial scopes retain their CFG and classified construction separately. Their
// dynamic resources and child calls belong to the enclosing realization owner,
// not this occurrence's invocation-long allocation table.
// Other construction operations must be pure. Observable initialization must
// have its own execution placement before this resource-only boundary.
// Unresolved authored bindings, extents, or construction are diagnostics with
// out_valid false. Statuses carry allocation and layout-arithmetic failures.
// No IR mutation occurs. Source mutation invalidates this result; a consuming
// rewrite must retain or translate its source correspondence before erasure.
// The fact table and its extension storage outlive this result.
// A rejected construction ends admission of the enclosing plan; its pool
// packings may already include allocations from the rejected construction.
iree_status_t loom_pipeline_resources_build(
    loom_module_t* module, loom_func_like_t pipeline,
    loom_value_fact_table_t* facts, const loom_pipeline_resource_pool_t* pools,
    iree_host_size_t pool_count,
    const loom_pipeline_resource_memory_t* memories,
    iree_host_size_t memory_count,
    const loom_pipeline_resource_pool_binding_t* bindings,
    iree_host_size_t binding_count,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    loom_pipeline_resources_t* out_resources, bool* out_valid);

// Returns an incoming or selected pool's retained canonical binding, or NULL
// when this construction has no binding for the value. Helpers can receive any
// pool without knowing whether its origin was an argument or memory selection.
const loom_pipeline_resource_pool_binding_t*
loom_pipeline_resources_lookup_pool(const loom_pipeline_resources_t* resources,
                                    loom_value_id_t value_id);

// Returns the retained placement for a source allocation root, or NULL when
// the root is external to this construction. This is an indexed table lookup;
// channel/view consumers obtain root identity from their retained value facts.
const loom_pipeline_resource_allocation_t*
loom_pipeline_resources_lookup_allocation(
    const loom_pipeline_resources_t* resources, loom_value_id_t root_value_id);

// Returns the local channel binding for an actual captured source identity.
// NULL means this construction does not directly define the binding; the
// enclosing admission must resolve it. Equal storage roots never substitute
// for this explicit protocol correspondence.
const loom_pipeline_resource_channel_t* loom_pipeline_resources_lookup_channel(
    const loom_pipeline_resources_t* resources, loom_value_id_t value_id);

// Checks the complete pool budgets after source, transport and worker storage
// have joined the packings. Failure names the canonical pool, required extent,
// and available extent. The caller does not materialize addresses or publish
// an executable until this admission succeeds.
iree_status_t loom_pipeline_resources_check_capacity(
    const loom_pipeline_resources_t* resources, const loom_op_t* entry,
    iree_diagnostic_emitter_t diagnostic_emitter, bool* out_valid);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_PIPELINE_RESOURCES_H_
