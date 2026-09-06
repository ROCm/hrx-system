// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiler-owned physical-representation continuity observation.
//
// Common lowering records representation-preserving SSA relations while it
// performs its existing source-plan traversal. Targets declare which value
// pairs participate and contribute exact alternatives only at a compact table
// of operation boundaries. No provider owns traversal or retains a source-IR
// graph.

#ifndef LOOM_CODEGEN_LOW_LOWER_REPRESENTATION_OBSERVER_H_
#define LOOM_CODEGEN_LOW_LOWER_REPRESENTATION_OBSERVER_H_

#include "iree/base/api.h"
#include "loom/analysis/value_relation.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/codegen/low/representation_plan.h"

#ifdef __cplusplus
extern "C" {
#endif

enum loom_low_lower_representation_boundary_flag_bits_e {
  // No source operation ports participate in the target action.
  LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_NONE = 0,
  // The complete source operand slice participates in the target action.
  LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS = 1u << 0,
  // The complete source result slice participates in the target action.
  LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS = 1u << 1,
  LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_ALL =
      LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_OPERANDS |
      LOOM_LOW_LOWER_REPRESENTATION_BOUNDARY_FLAG_RESULTS,
};
typedef uint8_t loom_low_lower_representation_boundary_flags_t;

// One target boundary observed for an exact source operation kind. The source
// function op is observed during begin; body ops are observed during the
// compiler-owned source-plan traversal. Tables must be strictly increasing by
// |op_kind| and are verified by target tests or their generators rather than
// rescanned during compilation.
typedef struct loom_low_lower_representation_boundary_t {
  // Exact source operation kind that invokes the target observer.
  loom_op_kind_t op_kind;
  // Target-defined compact action passed to the target observer.
  uint8_t action;
  // Operation ports offered to the target observer.
  loom_low_lower_representation_boundary_flags_t flags;
} loom_low_lower_representation_boundary_t;
static_assert(sizeof(loom_low_lower_representation_boundary_t) == 4,
              "representation boundaries must stay compact");

typedef struct loom_low_lower_representation_recorder_t
    loom_low_lower_representation_recorder_t;

typedef bool (*loom_low_lower_representation_relation_fn_t)(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, const loom_value_relation_t* relation);

typedef void (*loom_low_lower_representation_boundary_fn_t)(
    void* user_data, uint8_t action,
    loom_low_lower_representation_boundary_flags_t flags,
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_representation_recorder_t* recorder);

// Target policy for one function-local physical-representation plan.
typedef struct loom_low_lower_representation_provider_t {
  // Returns true when a common relation on |source_op| requires the two source
  // values to use one target representation. This callback is infallible and
  // must not walk source IR.
  loom_low_lower_representation_relation_fn_t relation;
  // Observes operation boundaries selected by |boundaries|. |flags| identifies
  // the source operation ports relevant to the target action. Failures and
  // exact alternatives are recorded through |recorder|.
  loom_low_lower_representation_boundary_fn_t observe_boundary;
  // Strictly increasing source operation boundary table.
  const loom_low_lower_representation_boundary_t* boundaries;
  // Number of rows in |boundaries|.
  uint16_t boundary_count;
  // Common relation kinds offered to |relation|. Zero disables structural
  // relation observation and requires |relation| to be NULL.
  loom_value_relation_mask_t relation_mask;
  // Caller-owned immutable payload passed to target callbacks.
  void* user_data;
} loom_low_lower_representation_provider_t;

// Records an exact equality requested by the active target boundary callback.
// Any failure is retained and returned by the observer's end callback.
void loom_low_lower_representation_record_union(
    loom_low_lower_representation_recorder_t* recorder,
    loom_value_id_t left_value_id, loom_value_id_t right_value_id);

// Adds one exact candidate domain for a source value at the current operation
// boundary. Candidate rows are copied into function-local storage.
void loom_low_lower_representation_record_candidates(
    loom_low_lower_representation_recorder_t* recorder,
    loom_value_id_t source_value_id,
    const loom_low_representation_candidate_t* candidates,
    iree_host_size_t candidate_count);

// Returns whether |source_value_id|'s current component has already received
// an exact candidate domain. The query does not make an absent value
// participate.
bool loom_low_lower_representation_component_is_constrained(
    loom_low_lower_representation_recorder_t* recorder,
    loom_value_id_t source_value_id);

// Retains a boundary-observation failure for propagation from observer end.
// Ownership of |status| transfers to the recorder.
void loom_low_lower_representation_record_failure(
    loom_low_lower_representation_recorder_t* recorder, iree_status_t status);

// Source-plan observer callbacks. Targets compose these into a normal
// loom_low_lower_source_plan_observer_t whose user_data points at a static
// loom_low_lower_representation_provider_t.
iree_status_t loom_low_lower_representation_observer_begin(
    void* user_data, loom_low_lower_context_t* context,
    void** out_observer_state);
void loom_low_lower_representation_observer_observe(
    void* observer_state, loom_low_lower_context_t* context,
    const loom_op_t* source_op);
iree_status_t loom_low_lower_representation_observer_end(
    void* observer_state, loom_low_lower_context_t* context);

// Returns the selected representation for |source_value_id|, or NONE when its
// component remained unconstrained. The representation observer must have
// completed successfully before this query.
iree_status_t loom_low_lower_representation_lookup(
    loom_low_lower_context_t* context, loom_value_id_t source_value_id,
    loom_low_representation_id_t* out_representation);

// Returns the selected representation through a source-contract query scope,
// or NONE when the value is unconstrained. Query scopes created from Low
// lowering share the observer's target-state allocator and therefore retrieve
// the same function-local plan without copying it into the query environment.
iree_status_t loom_low_lower_representation_query_lookup(
    const loom_target_contract_query_environment_t* environment,
    loom_value_id_t source_value_id,
    loom_low_representation_id_t* out_representation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_REPRESENTATION_OBSERVER_H_
