// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical reference transport and memory uses in an immutable source scope.

#ifndef LOOM_ANALYSIS_STORAGE_ACCESS_H_
#define LOOM_ANALYSIS_STORAGE_ACCESS_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_storage_access_effect_bits_e {
  // Bytes reached through the reference may be read.
  LOOM_STORAGE_ACCESS_READ = 1u << 0,
  // Bytes reached through the reference may be written.
  LOOM_STORAGE_ACCESS_WRITE = 1u << 1,
  // The reference may escape, or its use has no bounded access contract.
  LOOM_STORAGE_ACCESS_ESCAPE = 1u << 2,
  // This direct use starts an asynchronous access requiring local completion.
  // Callable effects project this bit to ESCAPE across the function boundary.
  LOOM_STORAGE_ACCESS_ASYNC = 1u << 3,
};
typedef uint8_t loom_storage_access_effects_t;

// Envelope of accesses reached through one reference. This describes proven
// confinement, not an exact byte interval or a restriction on derived views.
enum loom_storage_access_footprint_e {
  // Every reached access stays within this reference's current byte span.
  LOOM_STORAGE_ACCESS_FOOTPRINT_REFERENCE = 0,
  // Accesses may reach elsewhere in the reference's backing storage.
  LOOM_STORAGE_ACCESS_FOOTPRINT_STORAGE = 1,
};
typedef uint8_t loom_storage_access_footprint_t;

typedef struct loom_storage_reference_t loom_storage_reference_t;
typedef struct loom_template_provider_catalog_t
    loom_template_provider_catalog_t;

typedef struct loom_storage_access_use_t {
  // Source operation containing the classified reference operand.
  const loom_op_t* operation;
  // Reference operand classified by this use.
  loom_value_id_t value_id;
  // Effects of this direct use, excluding any separately referenced callee.
  loom_storage_access_effects_t effects;
  // Completed formal-argument effects for an exact source call, or NULL.
  const loom_storage_reference_t* callee_argument;
  // Next classified use of the same reference.
  struct loom_storage_access_use_t* next;
  // Next classified use in the function's source walk order.
  struct loom_storage_access_use_t* next_in_function;
} loom_storage_access_use_t;

typedef struct loom_storage_reference_edge_t {
  // Origin receiving effects propagated backward through this edge.
  loom_storage_reference_t* origin;
  // Alias or callee formal supplying effects to the origin.
  loom_storage_reference_t* destination;
  // Next local transport edge from the same origin; call edges are excluded.
  struct loom_storage_reference_edge_t* next_destination;
  // Next origin receiving effects from the same destination.
  struct loom_storage_reference_edge_t* next_origin;
} loom_storage_reference_edge_t;

struct loom_storage_reference_t {
  // Source reference identity.
  loom_value_id_t value_id;
  // Compact reference index used only while constructing its function graph.
  uint32_t index;
  // Completed union of read, write, and escape effects through all uses.
  loom_storage_access_effects_t effects;
  // Completed access envelope relative to this reference's current span.
  loom_storage_access_footprint_t footprint;
  // Envelope required when transporting this reference's accesses to origins.
  // Value-preserving relations retain the span; projections may change it.
  loom_storage_access_footprint_t origin_footprint;
  // True while this node awaits monotone effect propagation.
  bool queued;
  // Next node in the construction worklist.
  loom_storage_reference_t* next_pending;
  // Next reference in this function's canonical graph.
  loom_storage_reference_t* next;
  // Local aliases receiving allocation provenance from this reference.
  loom_storage_reference_edge_t* destinations;
  // Local origins and call actuals receiving this reference's effects.
  loom_storage_reference_edge_t* origins;
  // Direct classified uses; transitive alias uses remain on their own nodes.
  loom_storage_access_use_t* uses;
};

typedef struct loom_storage_access_barrier_t {
  // Workgroup acq_rel rendezvous requiring an execution-uniformity proof.
  const loom_op_t* operation;
  // Next qualifying rendezvous in source traversal order.
  struct loom_storage_access_barrier_t* next;
} loom_storage_access_barrier_t;

typedef struct loom_storage_access_function_t {
  // Formal references in entry-block argument order; scalar entries are NULL.
  loom_storage_reference_t** arguments;
  // Canonical reference nodes, including unused formal references.
  loom_storage_reference_t* references;
  // Direct uses in source walk order across all canonical references.
  loom_storage_access_use_t* uses;
  // Qualifying workgroup barriers in source traversal order.
  loom_storage_access_barrier_t* barriers;
  // A local non-call operation may access storage without a bounded operand.
  bool has_unknown_memory_access;
  // This body or a reached callee may access storage without a bounded operand.
  bool has_unbounded_memory_access;
} loom_storage_access_function_t;

// One source application whose implementation has already been selected by
// its owning specialization plan. Bindings borrow an immutable source module
// and are sorted by the uintptr_t value of operation. The callee is a local
// source function with a body; no provider selection occurs during analysis.
typedef struct loom_storage_access_binding_t {
  // Callable-boundary operation receiving this exact source implementation.
  const loom_op_t* operation;
  // Selected source definition in the analyzed module.
  loom_func_like_t function;
  // First operation operand corresponding to a callee argument.
  uint16_t argument_offset;
  // Number of operation operands passed to the callee.
  uint16_t argument_count;
} loom_storage_access_binding_t;

// Graph-independent argument facts retained across source-scope retirement.
typedef struct loom_storage_access_argument_t {
  // Union of effects through this argument; zero for scalars or unused values.
  loom_storage_access_effects_t effects;
  // Proven confinement relative to this argument's current span.
  loom_storage_access_footprint_t footprint;
} loom_storage_access_argument_t;

typedef struct loom_storage_access_summary_t {
  // Arena-owned argument facts in complete source-signature order.
  const loom_storage_access_argument_t* arguments;
  // Number of entries in arguments.
  uint16_t argument_count;
  // Reached memory effects not confined to any argument.
  bool has_unbounded_memory_access;
} loom_storage_access_summary_t;

typedef struct loom_storage_access_scope_t {
  // Source module, whose represented program remains immutable in this scope.
  loom_module_t* module;
  // Construction owner; all graph and index storage belongs to this arena.
  iree_arena_allocator_t* arena;
  // Demand-allocated state; NULL until the first storage query.
  struct loom_storage_access_state_t* state;
  // Optional, preselected source bindings installed before the first query.
  const loom_storage_access_binding_t* bindings;
  // Number of sorted entries in bindings.
  iree_host_size_t binding_count;
  // Optional existing provider catalog for unresolved source applications.
  // Their effects are the union of available bodies, without selecting one.
  // Exact bindings take precedence; an unavailable body remains opaque.
  const loom_template_provider_catalog_t* providers;
} loom_storage_access_scope_t;

// Memory effects projected through one call boundary. Reference arguments
// retain their own effects and proven footprint through all derived aliases.
// Effect flags alone do not bound accesses to an argument view's current byte
// span: a callee may move that view within its backing storage. Unbounded
// accesses are not confined to arguments at all.
typedef struct loom_storage_access_call_t {
  // Completed formal references in call argument order, with NULL entries for
  // scalar arguments. NULL when no bounded argument summary is needed or known.
  const loom_storage_reference_t* const* arguments;
  // The callee or a reached call may access storage without a bounded operand.
  bool has_unbounded_memory_access;
} loom_storage_access_call_t;

// Initializes a demand scope without traversing source or allocating storage.
// The scope ends before any represented source mutation. All borrowed graphs
// and use records expire with |arena|; no graph is needed by plan execution.
void loom_storage_access_scope_initialize(
    loom_module_t* module, iree_arena_allocator_t* arena,
    loom_storage_access_scope_t* out_scope);

// Builds the canonical graph for |function| and its reached source callees.
// Existing graphs are returned directly. Each reached body is interpreted once;
// a fixed-bit worklist closes recursive effects without revisiting source.
//
// Construction temporarily releases |value_domain|'s scratch borrow and then
// restores its exact numbering. Its value IDs and the caller's fact table are
// untouched. No callee fact scope is acquired. On allocation failure the caller
// domain is restored and this scope must be retired rather than queried again.
iree_status_t loom_storage_access_require_function(
    loom_storage_access_scope_t* scope, loom_func_like_t function,
    loom_local_value_domain_t* value_domain,
    const loom_storage_access_function_t** out_function);

// Retains only the canonical argument effects and confinement in
// |result_arena|. Neither the returned record nor its arguments borrow the
// source or graph. Source interpretation and recursive closure use the ordinary
// demand scope.
iree_status_t loom_storage_access_require_summary(
    loom_storage_access_scope_t* scope, loom_func_like_t function,
    loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* result_arena,
    const loom_storage_access_summary_t** out_summary);

// Projects the canonical reached-body summaries through |call|. Pure calls
// need no graph; opaque memory effects remain unbounded. Visible semantic and
// exact template callees share the demand scope and fixed point of function
// queries; unresolved template applications remain opaque.
// The caller's active value domain is restored before returning.
iree_status_t loom_storage_access_require_call(
    loom_storage_access_scope_t* scope, loom_call_like_t call,
    loom_local_value_domain_t* value_domain,
    loom_storage_access_call_t* out_call);

// Returns the classified effects of a direct use after graph construction.
static inline loom_storage_access_effects_t loom_storage_access_use_effects(
    const loom_storage_access_use_t* use) {
  return use->effects |
         (use->callee_argument ? use->callee_argument->effects : 0);
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_STORAGE_ACCESS_H_
