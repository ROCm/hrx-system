// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared placement and CFG ownership of target-shaped pure values.

#ifndef LOOM_CODEGEN_LOW_LOWER_REALIZATION_H_
#define LOOM_CODEGEN_LOW_LOWER_REALIZATION_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_realization_t loom_low_lower_realization_t;
typedef struct loom_low_lower_realizations_t loom_low_lower_realizations_t;

typedef enum loom_low_lower_realization_input_kind_e {
  // An authored value whose definition must precede initialization.
  LOOM_LOW_LOWER_REALIZATION_INPUT_SOURCE = 0,
  // Target entry state established by ABI import and entry setup. Its identity
  // is target-owned; it is not a source SSA value or a storage demand.
  LOOM_LOW_LOWER_REALIZATION_INPUT_ENTRY = 1,
  // A previously requested physical value. Construction order makes these
  // dependencies acyclic; the shared owner retains their emission order.
  LOOM_LOW_LOWER_REALIZATION_INPUT_REALIZATION = 2,
} loom_low_lower_realization_input_kind_t;

typedef struct loom_low_lower_realization_input_t {
  // Dependency namespace.
  loom_low_lower_realization_input_kind_t kind;
  // Identity in the selected namespace.
  union {
    // Source value ID or target entry-state key.
    uint32_t identity;
    // Previously interned physical dependency.
    const loom_low_lower_realization_t* realization;
  } value;
} loom_low_lower_realization_input_t;

// Retained uniform loop domain admitting complete, lane-valid edge payloads.
// Entry and backedge are dedicated unconditional edges; the sole exit is a
// uniform header test. This is a realization candidate domain, not a source
// program restriction. Other loops retain their ordinary direct lowering.
typedef struct loom_low_lower_realization_loop_t {
  // Source header defining the supplemental carried values.
  const loom_block_t* header;
  // Dedicated entry edge, executed even when the loop has zero trips.
  const loom_op_t* entry;
  // Dedicated backedge, executed once per body iteration.
  const loom_op_t* backedge;
  // Source value governed by the proved induction equation.
  loom_value_id_t induction_value;
  // Exact source-integer initial value.
  int64_t initial_value;
  // Exact positive source-integer increment.
  int64_t step;
  // Exact source-integer exit value, including zero-trip loops.
  int64_t exit_value;
  // Exact number of body executions.
  uint64_t trip_count;
} loom_low_lower_realization_loop_t;

// Pure target arithmetic only: no memory effects, control changes or source
// value rebinding. Initializers consume only their declared dependencies.
// Updates consume the current carried value. Builders own emitted IR; recipe
// data and its inputs retain function-arena lifetime.
typedef iree_status_t (*loom_low_lower_realization_emit_fn_t)(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const void* data, loom_value_id_t carried_value,
    loom_value_id_t* out_value);

typedef struct loom_low_lower_realization_recipe_t {
  // Physical carrier of both the initialized and carried value.
  loom_type_t type;
  // Target recipe family; distinct families never share physical values.
  loom_low_lower_plan_id_t id;
  // Canonical opaque recipe key, excluding inputs and loop identity. The
  // producer initializes every byte; core compares the complete payload.
  iree_const_byte_span_t key;
  // Explicit dependencies used for dominance, placement and storage demand.
  const loom_low_lower_realization_input_t* inputs;
  // Number of populated input records.
  uint16_t input_count;
  // Pure arithmetic emitted once after all dependencies have become available.
  loom_low_lower_realization_emit_fn_t initialize;
  // Backedge arithmetic, or NULL for an invariant realization.
  loom_low_lower_realization_emit_fn_t update;
  // Function-owned target data consumed by initialize and update.
  const void* data;
} loom_low_lower_realization_recipe_t;

// Deferred plan replacement, applied only after the shared owner selects the
// complete reuse group. The owner passes the requested physical value to the
// callback, which updates retained plans without emitting IR or traversing
// source operations.
typedef iree_status_t (*loom_low_lower_realization_apply_fn_t)(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_realization_t* realization, void* data);

typedef struct loom_low_lower_realization_offer_t {
  // Replacement strategy family defining the cost units and opaque keys.
  loom_low_lower_plan_id_t id;
  // Replacement strategy identity, excluding the loop and source block.
  // Offers with equal strategy IDs, alternative IDs, and keys share group
  // costs and are selected together.
  iree_const_byte_span_t key;
  // Optional family of mutually exclusive replacement strategies. Groups
  // with different IDs but the same nonzero alternative ID and key compete by
  // total savings; only the most profitable group is selected.
  loom_low_lower_plan_id_t alternative_id;
  // Cost paid once to initialize the group's shared physical state.
  uint32_t group_setup_cost;
  // Cost paid once per iteration for the group's shared physical state.
  uint32_t group_iteration_cost;
  // Target identity of the direct calculation removed by this offer. Equal
  // identities receive credit once, even when several packets reuse it.
  uint64_t removed_key;
  // Per-iteration cost of the removed direct calculation.
  uint32_t removed_iteration_cost;
  // Minimum number of distinct direct calculations the strategy group must
  // replace before its shared physical state is profitable.
  uint32_t minimum_removed_count;
  // Conservative setup cost for this offer's additional invariant values.
  uint32_t setup_cost;
  // Per-iteration cost for this offer's independent physical state.
  uint32_t iteration_cost;
  // Physical value requested only when the offer group wins. NULL
  // leaves physical realization entirely to the target callback.
  const loom_low_lower_realization_recipe_t* recipe;
  // Applies the selected replacement to its retained target plan.
  loom_low_lower_realization_apply_fn_t apply;
  // Function-owned callback payload, including the retained packet plan.
  void* data;
} loom_low_lower_realization_offer_t;

// Creates the shared owner and retains loop eligibility from existing facts.
// It does not traverse operations or recover induction expressions. Offers,
// interning tables, and placement work use |construction_arena|, which must
// remain live until finalize returns. The frozen schedule borrows none of it.
iree_status_t loom_low_lower_realizations_create(
    loom_low_lower_context_t* context,
    iree_arena_allocator_t* construction_arena);

// Selects and applies offers made during target source-memory preparation,
// then marks every retained source input before source-graph selection. The
// owner remains open for shared offers derived from exact selected rules.
iree_status_t loom_low_lower_realizations_prepare_selection(
    loom_low_lower_context_t* context);

// Returns the admitted innermost loop, or NULL outside the candidate domain.
const loom_low_lower_realization_loop_t* loom_low_lower_realization_loop(
    const loom_low_lower_context_t* context, const loom_op_t* source_op);

// Offers a replacement whose shared loop cost must be amortized by distinct
// direct calculations. The bounded domain is the dedicated latch block, which
// executes exactly trip_count times. Other blocks keep their direct plans.
// No physical values or source demands are created for rejected offers.
iree_status_t loom_low_lower_realization_offer(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_realization_loop_t* loop,
    const loom_low_lower_realization_offer_t* offer);

// Interns a physical value and retains its source-use boundary. A NULL result
// means the input definitions cannot be placed in the requested scope. Loop
// recipes initialize before entry and produce a supplemental header argument;
// invariant recipes initialize immediately after their last dependency. Calls
// are valid while realization construction remains open.
iree_status_t loom_low_lower_realization_request(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_low_lower_realization_loop_t* loop,
    const loom_low_lower_realization_recipe_t* recipe,
    const loom_low_lower_realization_t** out_realization);

// Binds one source result to a selected carried realization before body
// emission. Multiple source producers may alias the same physical value.
iree_status_t loom_low_lower_realization_bind_source_value(
    loom_low_lower_context_t* context,
    const loom_low_lower_realization_t* realization,
    loom_value_id_t source_value_id);

// Selects any remaining complete reuse groups, freezes placement/use
// boundaries and marks source dependencies. No subsequent request is
// permitted.
iree_status_t loom_low_lower_realizations_finalize(
    loom_low_lower_context_t* context, bool* out_storage_refinement_required);

// Creates supplemental block arguments after all authored arguments exist.
iree_status_t loom_low_lower_realizations_map_blocks(
    loom_low_lower_context_t* context);

// Indexed signature of the supplemental tuple retained for a top-level source
// block. Control expansion consumes the same canonical recipes as emission.
uint16_t loom_low_lower_realization_block_argument_count(
    const loom_low_lower_context_t* context, const loom_block_t* source_block);
loom_type_t loom_low_lower_realization_block_argument_type(
    const loom_low_lower_context_t* context, const loom_block_t* source_block,
    uint16_t argument_index);

// Emits initializers at block entry and initializers/updates after an authored
// operation. Both run inside the normal shared emission scratch lifetime.
iree_status_t loom_low_lower_realizations_emit_entry(
    loom_low_lower_context_t* context, const loom_block_t* source_block);
iree_status_t loom_low_lower_realizations_emit_after(
    loom_low_lower_context_t* context, const loom_op_t* source_op);

// Emits any backedge updates whose use boundary is the edge itself.
iree_status_t loom_low_lower_realizations_emit_edge(
    loom_low_lower_context_t* context, const loom_op_t* source_terminator);

// Returns the physical value selected for ordinary uses. Carried values are
// header arguments; invariant values were emitted at their source anchor.
loom_value_id_t loom_low_lower_realization_value(
    const loom_low_lower_realization_t* realization);

// Number of supplemental values on an admitted entry/backedge.
uint16_t loom_low_lower_realization_edge_count(
    const loom_low_lower_context_t* context,
    const loom_op_t* source_terminator);

// Writes the complete supplemental edge payload in header-argument order.
void loom_low_lower_realization_edge_values(
    const loom_low_lower_context_t* context, const loom_op_t* source_terminator,
    loom_value_id_t* values);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_REALIZATION_H_
