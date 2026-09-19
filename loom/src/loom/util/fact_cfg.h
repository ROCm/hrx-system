// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CFG forwarding components retained with a value-fact scope. Mutually
// forwarded arguments share the same possible incoming values, so their facts
// are joined as one component before widening. Arithmetic producers remain
// external inputs to the component and retain normal iterative inference.

#ifndef LOOM_UTIL_FACT_CFG_H_
#define LOOM_UTIL_FACT_CFG_H_

#include "loom/analysis/loop_domain.h"
#include "loom/analysis/scc.h"
#include "loom/util/cfg_graph.h"
#include "loom/util/cfg_loop_nest.h"
#include "loom/util/fact_control.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_table_t loom_value_fact_table_t;

// A recognized header-tested recurrence. Its inputs are invariant within the
// natural loop. The CFG fact solver refreshes this equation after semantic
// edits; numeric queries consume current table entries, never cached counts.
typedef struct loom_value_fact_cfg_induction_t {
  // Header argument, or INVALID for an unrecognized or literal-false guard.
  loom_value_id_t value;
  // Value entering the header from outside the loop.
  loom_value_id_t initial_value;
  // Invariant upper bound tested by the header guard.
  loom_value_id_t upper_bound;
  // Invariant increment added by the unique backedge, or INVALID. A missing
  // increment can still establish zero trips from a false entry guard.
  loom_value_id_t step;
  // Signedness and inclusivity of the guard.
  loom_loop_bound_flags_t bound_flags;
  // True when the header condition is a literal false. No backedge executes,
  // including after canonicalization removes the original comparison.
  bool exits_at_header;
} loom_value_fact_cfg_induction_t;

// One argument participating in the region's forwarding graph.
typedef struct loom_value_fact_cfg_argument_t {
  // SSA value defined by the argument.
  loom_value_id_t value_id;
  // Owning block's dense CFG index.
  uint16_t block_index;
  // Position in the owning block's argument list.
  uint16_t argument_index;
} loom_value_fact_cfg_argument_t;

// Forwarding graph partition owned by one control-flow component. Forwarding
// cycles cannot cross control-flow components; other arguments are inputs.
typedef struct loom_value_fact_cfg_forwarding_t {
  // First argument ordinal in this partition's contiguous argument span.
  iree_host_size_t argument_offset;
  // Number of arguments and reserved component/member slots in the span.
  iree_host_size_t argument_count;
  // Populated components in dependency order, starting at argument_offset.
  iree_host_size_t component_count;
  // True after a branch payload edit changes the forwarding graph.
  bool dirty;
} loom_value_fact_cfg_forwarding_t;

// Retained control-flow structure and cyclic-summary state for one region.
// CFG edits replace the snapshot. Payload edits invalidate forwarding structure
// within the affected component; numeric input edits invalidate its summary.
typedef struct loom_value_fact_cfg_region_t {
  // CFG edges and reachability owned by this analysis.
  loom_cfg_graph_t graph;
  // Semantic natural loops, independent of textual block order.
  loom_cfg_loop_nest_t loops;
  // Current recurrence equations, indexed by loops.loops.
  loom_value_fact_cfg_induction_t* inductions;
  // Immutable compressed control dependencies for this graph snapshot.
  loom_cfg_control_t control_structure;
  // Selector distributions and live execution facts with snapshot lifetime.
  loom_value_fact_control_t* control;
  // First forwarding node for each block, grouped by control-flow component.
  iree_host_size_t* argument_offsets;
  // Forwarding nodes for reachable non-entry arguments.
  loom_value_fact_cfg_argument_t* arguments;
  // Number of forwarding nodes.
  iree_host_size_t argument_count;
  // Forwarding components in reserved per-partition slots. Each partition's
  // component_count entries begin at its argument_offset in dependency order.
  loom_scc_t* components;
  // Backing member storage for components, with one slot per argument.
  iree_host_size_t* component_nodes;
  // Component index for each forwarding node.
  iree_host_size_t* argument_components;
  // Control-flow components bound the values that must restart together when
  // an edit changes a cyclic dataflow equation.
  struct {
    // Member spans grouped by graph-owned reachable component ordinal, with
    // each span in reverse postorder for initial and restarted propagation.
    loom_scc_list_t components;
    // Retained forwarding structure validity and argument span per component.
    loom_value_fact_cfg_forwarding_t* forwarding;
    // First reverse-postorder member's terminator schedules each cyclic
    // summary, including cycles with observations but no carried block
    // arguments.
    loom_op_t** anchors;
    // True when semantic edits or input changes require a cyclic summary.
    bool* dirty;
  } control_flow;
} loom_value_fact_cfg_region_t;

// Constructs CFG and forwarding structure once. Acyclic regions need no
// forwarding components; ordinary block argument inference handles them.
iree_status_t loom_value_fact_cfg_region_initialize(
    const loom_module_t* module, const loom_region_t* region,
    iree_arena_allocator_t* arena, loom_value_fact_cfg_region_t* out_region);

// Refreshes the recurrence equation if block_index is a natural-loop header.
// Called by the owning CFG solve after predecessor facts become available.
void loom_value_fact_cfg_update_induction(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_region_t* region, uint16_t block_index);

// Evaluates a retained recurrence against current facts and target carriers.
// Unknown or nonconstant inputs do not establish a numeric recurrence proof.
loom_loop_recurrence_facts_t loom_value_fact_cfg_induction_facts(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_induction_t* induction);

// Refreshes one retained selector from its current SSA identity and facts,
// then settles indexed control dependents. Returns whether execution changed.
bool loom_value_fact_cfg_update_control(
    const loom_value_fact_table_t* table,
    const loom_value_fact_cfg_region_t* region, uint16_t block_index);

// Seeds selectors together after snapshot publication or a cyclic value reset.
// NULL selects the whole region; otherwise only the component's blocks are
// revisited. Uncomputed values seed the optimistic distribution, whereas
// defined unknown values contribute unknown control.
void loom_value_fact_cfg_seed_control(
    const loom_value_fact_table_t* table,
    const loom_value_fact_cfg_region_t* region, const loom_scc_t* component);

// Within a region with forwarding components, returns the node for a value,
// or IREE_HOST_SIZE_MAX for an operation result, entry argument, or value
// outside the reachable region.
iree_host_size_t loom_value_fact_cfg_region_argument_index(
    const loom_value_fact_cfg_region_t* region, loom_value_id_t value_id);

// Refreshes an invalidated forwarding partition in place before its numeric
// facts are solved. Scratch storage is temporary; retained storage is bounded
// by the partition's argument count and reused across payload edits.
iree_status_t loom_value_fact_cfg_update_forwarding(
    const loom_value_fact_cfg_region_t* region,
    iree_host_size_t component_index, iree_arena_allocator_t* scratch_arena);

// Returns the retained structural snapshot without constructing one. A caller
// comparing snapshots across an edit uses this before publishing new structure.
const loom_value_fact_cfg_region_t* loom_value_fact_table_lookup_cfg_region(
    const loom_value_fact_table_t* table, const loom_region_t* region);

// Returns the region structure cached in the fact scope, constructing it on
// first use. It remains valid until the scope is cleared or the region's
// structure is replaced.
iree_status_t loom_value_fact_table_get_or_build_cfg_region(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_region_t* region,
    const loom_value_fact_cfg_region_t** out_region);

// Publishes caller-owned structure for a region after a completed CFG edit.
// The caller keeps the structure alive until replacing it again or forgetting
// it. This changes only structural analysis; existing value facts are retained.
iree_status_t loom_value_fact_table_set_cfg_region(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_region_t* region, const loom_value_fact_cfg_region_t* structure);

// Withdraws a region's structure before its storage or CFG becomes invalid.
// A subsequent structural query rebuilds it from the current IR.
void loom_value_fact_table_forget_cfg_region(loom_value_fact_table_t* table,
                                             const loom_region_t* region);

// Receives each value whose facts changed, including block arguments and
// results recomputed in a cyclic component. The callback schedules dependents.
typedef iree_status_t (*loom_value_fact_cfg_changed_fn_t)(
    void* user_data, loom_value_id_t value_id);

// Recomputes the joins defined by an acyclic block using current incoming
// value facts. Cyclic blocks use recompute_cfg_component so obsolete feedback
// cannot prevent narrowing after an edit. The caller propagates changed facts
// through users before querying the updated fixed point.
iree_status_t loom_value_fact_table_update_cfg_block_args(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_region_t* region, uint16_t block_index,
    loom_value_fact_cfg_changed_fn_t on_changed, void* user_data);

// Restarts one cyclic control-flow component from its unchanged external
// inputs, then reports values whose converged facts differ from the old facts.
// Storage for the solve is temporary; extension payloads stay in the table.
iree_status_t loom_value_fact_table_recompute_cfg_component(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_value_fact_cfg_region_t* region, const loom_scc_t* component,
    iree_arena_allocator_t* scratch_arena,
    loom_value_fact_cfg_changed_fn_t on_changed, void* user_data);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_CFG_H_
