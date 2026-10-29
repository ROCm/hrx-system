// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IR mutation with change tracking and optional worklist scheduling.
//
// The rewriter wraps a builder and tracks all IR mutations (creation,
// erasure, operand changes, RAUW). Drivers that revisit affected operations
// enable the worklist before mutating IR. Direct mutation passes retain SSA,
// effect and analysis maintenance without allocating or filling an unused
// queue. The active worklist uses LOOM_OP_FLAG_ON_WORKLIST for O(1) dedup.
//
// Usage (custom pass):
//   loom_rewriter_t rewriter;
//   loom_rewriter_initialize(&rewriter, module, pass->arena);
//   loom_rewriter_seed_function(&rewriter, function);
//   while (loom_op_t* op = loom_rewriter_pop(&rewriter)) {
//     const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
//     if (vtable && vtable->canonicalize) {
//       rewriter.flags = 0;
//       vtable->canonicalize(op, &rewriter);
//     }
//   }
//   loom_rewriter_deinitialize(&rewriter);

#ifndef LOOM_REWRITE_REWRITER_H_
#define LOOM_REWRITE_REWRITER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/action.h"
#include "loom/rewrite/fact_propagation.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Rewriter
//===----------------------------------------------------------------------===//

enum loom_rewriter_flag_bits_e {
  // Set by any mutation (create, erase, RAUW, set_operand).
  // Cleared by the driver before each pattern invocation.
  LOOM_REWRITER_FLAG_CHANGED = 1u << 0,
  // Set when a mutation changes a value type.
  LOOM_REWRITER_FLAG_TYPE_CHANGED = 1u << 1,
  // Set when incremental fact recomputation changes a value's facts.
  LOOM_REWRITER_FLAG_FACTS_CHANGED = 1u << 2,
};
typedef uint8_t loom_rewriter_flags_t;

enum loom_rewriter_name_policy_flag_bits_e {
  // Preserve explicit source/result names across 1:1 rewrites.
  LOOM_REWRITER_NAME_POLICY_PRESERVE_NAMES = 1u << 0,
  // Materialize optional derived names such as %index_bounded from already
  // named source values. Anonymous compiler-generated values stay anonymous.
  LOOM_REWRITER_NAME_POLICY_DERIVE_DEBUG_NAMES = 1u << 1,
};
typedef uint8_t loom_rewriter_name_policy_flags_t;

#define LOOM_REWRITER_NAME_POLICY_DEFAULT     \
  (LOOM_REWRITER_NAME_POLICY_PRESERVE_NAMES | \
   LOOM_REWRITER_NAME_POLICY_DERIVE_DEBUG_NAMES)

// Callback to query whether exact facts can be materialized as a constant of
// the requested type. The query is pure and must agree with the paired
// materialization callback's supported fact/type domain.
typedef bool (*loom_constant_materializer_support_fn_t)(
    loom_value_facts_t facts, loom_type_t result_type);

// Callback to materialize a supported constant from exact facts into IR. The
// implementation builds the appropriate constant op and returns its result.
typedef iree_status_t (*loom_materialize_constant_fn_t)(
    loom_builder_t* builder, loom_value_facts_t facts, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value_id);

// Constant materialization policy selected for one rewrite scope. Both
// callbacks must be set for the materializer to be available. Capability can
// be queried for a complete replacement family before any IR is constructed.
typedef struct loom_constant_materializer_t {
  // Pure applicability query for an exact fact/type pair.
  loom_constant_materializer_support_fn_t supports;
  // Builds a pair accepted by |supports|; allocation/build failures remain
  // possible after a successful query.
  loom_materialize_constant_fn_t materialize;
} loom_constant_materializer_t;

// Callback to materialize a typed replacement value into IR.
typedef iree_status_t (*loom_materialize_value_fn_t)(
    loom_builder_t* builder, loom_type_t result_type,
    loom_location_id_t location, loom_value_id_t* out_value_id);

// Called before an exact fold when the attached fact table has retained
// predicate-bearing identities. The callback consumes those observations
// through ordinary rewriter mutations before the proven fold completes.
typedef struct loom_rewriter_pending_exact_relations_callback_t {
  // Callback state borrowed for the call.
  void* user_data;
  // Optional pending-relation consumer.
  iree_status_t (*fn)(void* user_data, loom_rewriter_t* rewriter);
} loom_rewriter_pending_exact_relations_callback_t;

// Observes operations created while rewriting an explicit source operation.
// The callback is a cold accounting surface; ordinary rewriting leaves it
// unset, so finalization takes only a disabled observer branch.
typedef struct loom_rewriter_operation_creation_callback_t {
  // Callback state borrowed for the active rewrite run.
  void* user_data;
  // Optional creation observer receiving the immediate rewrite source.
  iree_status_t (*fn)(void* user_data, const loom_op_t* source_op,
                      loom_op_t* created_op);
} loom_rewriter_operation_creation_callback_t;

typedef struct loom_rewriter_cfg_region_t loom_rewriter_cfg_region_t;
typedef struct loom_rewriter_worklist_chunk_t loom_rewriter_worklist_chunk_t;
typedef struct loom_rewriter_pending_facts_t loom_rewriter_pending_facts_t;

struct loom_rewriter_t {
  // Builder whose finalize callback records mutations and maintains facts.
  // Newly created ops also enter the worklist when tracking is enabled.
  loom_builder_t builder;

  // The module being transformed.
  loom_module_t* module;

  // Pass arena for scratch allocations (worklist growth, etc.).
  iree_arena_allocator_t* arena;

  // Mutation tracking. Set by any mutation, cleared by the driver
  // before each pattern invocation.
  loom_rewriter_flags_t flags;

  // Number of operations finalized by the builder during this rewrite run.
  uint64_t created_op_count;

  // Number of operations erased through this rewriter during this rewrite run.
  uint64_t erased_op_count;

  // Number of successful vector-to-scalar rewrites.
  uint64_t scalarized_op_count;

  // Creation provenance active during a rewrite callback.
  struct {
    // Optional observer for creations with an explicit rewrite source.
    loom_rewriter_operation_creation_callback_t callback;
    // Immediate operation responsible for creations in the active callback.
    const loom_op_t* source;
  } operation_creation;

  // Central policy for optional SSA display names created during rewrites.
  loom_rewriter_name_policy_flags_t name_policy;

  // Active chunk in the LIFO worklist, or NULL when scheduling is disabled.
  // Entries are deduplicated through LOOM_OP_FLAG_ON_WORKLIST.
  loom_rewriter_worklist_chunk_t* worklist_chunk;

  // Borrowed per-value analysis facts. Populated when a pass calls
  // loom_rewriter_enable_analysis. NULL means analysis is disabled.
  loom_value_fact_table_t* fact_table;

  // Lazy arena-owned equation membership, independent of rewrite scheduling.
  loom_rewriter_pending_facts_t* pending_facts;

  // Borrowed math policy selected once for this rewrite scope. NULL preserves
  // target-neutral optional arithmetic choices until a target is selected.
  const struct loom_target_math_policy_t* math_policy;

  // Resolved target bundle paired with |math_policy|. NULL denotes a
  // target-neutral rewrite scope.
  const struct loom_target_bundle_t* math_target_bundle;

  // Structural snapshots published into the fact table for edited regions.
  struct {
    // Region-address hash buckets for O(1) snapshot storage lookup.
    loom_rewriter_cfg_region_t** buckets;
    // Power-of-two bucket count.
    iree_host_size_t bucket_count;
    // Number of regions with replaceable snapshots.
    iree_host_size_t count;
    // All region records for snapshot withdrawal and arena teardown.
    loom_rewriter_cfg_region_t* entries;
  } cfg_facts;

  // Dialect-provided constant materialization policy. Set by the pass before
  // enabling analysis. A zero policy leaves facts computed and propagated but
  // prevents exact-value folding.
  loom_constant_materializer_t constant_materializer;

  // Optional consumer invoked when an exact fold has retained relations.
  loom_rewriter_pending_exact_relations_callback_t
      pending_exact_relations_callback;
};

// Initializes mutation tracking without allocating storage. Worklist scheduling
// remains disabled until explicitly enabled or seeded by a rewrite driver.
void loom_rewriter_initialize(loom_rewriter_t* rewriter, loom_module_t* module,
                              iree_arena_allocator_t* arena);

void loom_rewriter_deinitialize(loom_rewriter_t* rewriter);

// Installs an optional cold observer for operation creation provenance.
void loom_rewriter_set_operation_creation_callback(
    loom_rewriter_t* rewriter,
    loom_rewriter_operation_creation_callback_t callback);

// Sets the immediate source operation for subsequently finalized operations.
// Passes and batched transforms clear the source after the rewrite callback.
void loom_rewriter_set_operation_creation_source(loom_rewriter_t* rewriter,
                                                 const loom_op_t* source_op);

// Queries the retained target preference after the caller has established that
// multiply and add both permit contraction. Does not resolve target context.
bool loom_rewriter_prefers_fma(const loom_rewriter_t* rewriter,
                               loom_type_t value_type, uint8_t fastmath_flags);

// Records effect summaries and inherited execution facts for a reinserted
// subtree, then schedules its operations when analysis is attached.
iree_status_t loom_rewriter_record_subtree(loom_rewriter_t* rewriter,
                                           loom_op_t* op);

// Enables tracking of operations affected by subsequent mutations. Calling this
// on an enabled rewriter preserves its pending operations and allocated
// storage. A driver enables tracking before edits whose notifications it must
// consume; enabling does not discover edits made while scheduling was disabled.
iree_status_t loom_rewriter_enable_worklist(loom_rewriter_t* rewriter);

// Enables and seeds the worklist with live ops in |region| and nested regions.
iree_status_t loom_rewriter_seed_region(loom_rewriter_t* rewriter,
                                        loom_region_t* region);

// Enables and seeds the worklist with live ops in a function-like body region.
iree_status_t loom_rewriter_seed_function(loom_rewriter_t* rewriter,
                                          loom_func_like_t function);

// Looks up facts from the rewriter's fact table. Returns unknown
// facts if analysis is not enabled or the value is not defined.
static inline loom_value_facts_t loom_rewriter_value_facts(
    const loom_rewriter_t* rewriter, loom_value_id_t value_id) {
  if (!rewriter->fact_table) {
    return loom_value_facts_unknown();
  }
  return loom_value_fact_table_lookup(rewriter->fact_table, value_id);
}

// Returns the retained SSA identity without changing the facts attached to
// |value_id|. This is an indexed lookup through rank-bounded retained handles,
// not an IR walk. Without analysis, the value is its own identity.
static inline loom_value_id_t loom_rewriter_value_identity(
    const loom_rewriter_t* rewriter, loom_value_id_t value_id) {
  return loom_value_fact_table_query_identity(rewriter->fact_table, value_id);
}

// Returns whether same-typed operands carry identical values using the retained
// identity and exact integer facts. Equal values can carry different facts:
// choosing either operand must not discard facts contributed by the other.
// This does not prove sole ownership or ordinary floating-point equality;
// NaN-sensitive patterns retain their predicate rules.
static inline bool loom_rewriter_values_equal(const loom_rewriter_t* rewriter,
                                              loom_value_id_t lhs,
                                              loom_value_id_t rhs) {
  return loom_value_fact_table_values_equal(rewriter->fact_table, lhs, rhs);
}

// Returns the live SSA definition of |value_id|, or NULL for block arguments
// and erased producers. Unlike value_identity, this does not bypass aliases:
// their predicates, refined types, and interpretation facts belong to this
// value, even when another value has identical bits. Rewrites that compose
// producers must retain those facts. Patterns that retire a producer must
// also check its actual SSA uses.
loom_op_t* loom_rewriter_value_defining_op(const loom_rewriter_t* rewriter,
                                           loom_value_id_t value_id);

// Returns whether a complete constant materializer policy is selected.
static inline bool loom_rewriter_has_constant_materializer(
    const loom_rewriter_t* rewriter) {
  return rewriter->constant_materializer.supports != NULL &&
         rewriter->constant_materializer.materialize != NULL;
}

// Returns whether the selected materializer supports an exact fact/type pair.
// False also covers a rewrite scope with no complete materializer policy.
static inline bool loom_rewriter_can_materialize_constant(
    const loom_rewriter_t* rewriter, loom_value_facts_t facts,
    loom_type_t result_type) {
  return loom_rewriter_has_constant_materializer(rewriter) &&
         rewriter->constant_materializer.supports(facts, result_type);
}

// Materializes a constant value using the rewriter's selected policy.
// Convenience for patterns that have already established applicability and
// need to create constants without depending on a specific dialect. Returns
// IREE_STATUS_UNIMPLEMENTED if no complete materializer policy is set.
iree_status_t loom_rewriter_build_constant(loom_rewriter_t* rewriter,
                                           loom_value_facts_t facts,
                                           loom_type_t result_type,
                                           loom_location_id_t location,
                                           loom_value_id_t* out_value_id);

// Publishes pending fact equations, then attempts to fold a side-effect-free op
// with vtable fact inference to constants using the maintained result facts.
// Materializes constants when every result is exact and supported by the
// selected materializer, then replaces the original op via RAUW+erase. The
// complete result family is checked before any relation preservation or IR
// construction. Sets |*out_folded| to true if the op was erased, false
// otherwise. No-op if analysis is not enabled or the op has no fact inference
// or region summary interface.
iree_status_t loom_rewriter_try_fold(loom_rewriter_t* rewriter, loom_op_t* op,
                                     bool* out_folded);

// Prepares the same exact fold for a regionless root from already-published
// facts. Retains only exact result facts in the batch arena; apply consumes
// live result types and never re-queries facts. Predicate-bearing identities
// with pending exact relations remain on the ordinary fold path so their
// relation-preservation callback runs before they disappear.
iree_status_t loom_rewriter_prepare_fold(const loom_rewriter_t* rewriter,
                                         const loom_op_t* op,
                                         iree_arena_allocator_t* arena,
                                         loom_rewrite_action_t** out_action);

// Captures the current high-water mark of the module value table.
//
// Use before materializing replacement IR. Pass the returned checkpoint to
// loom_rewriter_preserve_result_names_on_new_values so named result values
// keep their human/agent-facing spelling without accidentally renaming an
// existing operand or block argument used as a replacement.
loom_value_id_t loom_rewriter_value_checkpoint(const loom_rewriter_t* rewriter);

// Copies result names from |op| to replacement values created after
// |value_checkpoint|. Existing replacement values, already-named replacements,
// unnamed old results, and invalid result slots are left unchanged.
//
// This is intentionally separate from RAUW: many canonicalizations replace a
// named result with a pre-existing operand, and mutating that operand's name
// would make the IR less true, not more readable.
iree_status_t loom_rewriter_preserve_result_names_on_new_values(
    loom_rewriter_t* rewriter, const loom_op_t* op,
    const loom_value_id_t* replacements, uint16_t count,
    loom_value_id_t value_checkpoint);

// Copies |source_value|'s optional display name to |target_value| when enabled
// by the rewriter name policy and the target is unnamed.
iree_status_t loom_rewriter_copy_value_name(loom_rewriter_t* rewriter,
                                            loom_value_id_t source_value,
                                            loom_value_id_t target_value);

// Moves |source_value|'s optional display name to |target_value| when enabled
// by the rewriter name policy and the source is named.
iree_status_t loom_rewriter_move_value_name(loom_rewriter_t* rewriter,
                                            loom_value_id_t source_value,
                                            loom_value_id_t target_value);

// Clears |value|'s optional display name.
iree_status_t loom_rewriter_clear_value_name(loom_rewriter_t* rewriter,
                                             loom_value_id_t value);

// Derives |target_value|'s optional display name from |source_value| when
// enabled by policy. |suffix| is a snake_case word without a leading
// underscore. If |source_value| is anonymous, no name is created.
iree_status_t loom_rewriter_try_set_derived_value_name(
    loom_rewriter_t* rewriter, loom_value_id_t source_value,
    loom_value_id_t target_value, iree_string_view_t suffix);

// Returns true if the op has no uses, no side effects, and is not a
// terminator — safe to erase without affecting program semantics.
bool loom_rewriter_is_trivially_dead(const loom_rewriter_t* rewriter,
                                     const loom_op_t* op);

// If the op is trivially dead, erases it and adds its operand
// providers to the worklist for cascading DCE. Returns true via
// |*out_erased| if the op was erased. No-op if the op is live.
iree_status_t loom_rewriter_erase_if_dead(loom_rewriter_t* rewriter,
                                          loom_op_t* op, bool* out_erased);

//===----------------------------------------------------------------------===//
// Worklist
//===----------------------------------------------------------------------===//

// Adds an op to the enabled worklist. No-op when tracking is disabled or the
// operation is dead or already queued.
iree_status_t loom_rewriter_add_to_worklist(loom_rewriter_t* rewriter,
                                            loom_op_t* op);

// Pops the next op from the worklist. Returns NULL when empty.
// Clears LOOM_OP_FLAG_ON_WORKLIST on the returned op.
loom_op_t* loom_rewriter_pop(loom_rewriter_t* rewriter);

//===----------------------------------------------------------------------===//
// Safe IR mutation
//===----------------------------------------------------------------------===//
//
// These functions perform the mutation AND update the worklist with
// affected ops. Always use these instead of direct module functions
// when inside a pattern or canonicalize callback.

// Replaces all uses of |op|'s results with |replacements| (one per
// result), adds all affected user ops to the worklist, then erases
// |op|. The replacements array must have op->result_count entries.
iree_status_t loom_rewriter_replace_all_uses_and_erase(
    loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_value_id_t* replacements, uint16_t count);

// Replaces all uses and type uses of one value with another value and adds
// affected users to the worklist. Retained SSA identities are retargeted before
// returning, independently of deferred numeric inference. Does not erase the
// defining op; later uses of the old value retain its original meaning.
iree_status_t loom_rewriter_replace_all_uses_with(loom_rewriter_t* rewriter,
                                                  loom_value_id_t old_value,
                                                  loom_value_id_t new_value);

// Replaces operand uses of one value with another value except uses by
// |except_op|, and adds affected users to the worklist. Does not erase the
// defining op and intentionally does not rewrite type or attribute references.
iree_status_t loom_rewriter_replace_all_uses_except(loom_rewriter_t* rewriter,
                                                    loom_value_id_t old_value,
                                                    loom_value_id_t new_value,
                                                    const loom_op_t* except_op);

// Materializes one replacement value per result using |materialize_value|,
// preserves existing result names, replaces all uses, and erases |op|.
iree_status_t loom_rewriter_replace_results_with_materialized_values_and_erase(
    loom_rewriter_t* rewriter, loom_op_t* op,
    loom_materialize_value_fn_t materialize_value);

// Erases an op that has no remaining uses.
iree_status_t loom_rewriter_erase(loom_rewriter_t* rewriter, loom_op_t* op);

// Erases disjoint operation subtrees whose results have no uses outside the
// supplied set. The consuming plan owns that closure proof. Retained provider
// and parent facts are updated once per member before any member is erased.
iree_status_t loom_rewriter_erase_closed_set(loom_rewriter_t* rewriter,
                                             loom_op_t* const* ops,
                                             iree_host_size_t count);

// Removes a verified closed set of blocks, withdrawing retained SSA references
// before their definitions are retired. The mask covers the region's blocks.
// The caller refreshes CFG structure after completing its structural edits.
iree_status_t loom_rewriter_remove_blocks(loom_rewriter_t* rewriter,
                                          loom_region_t* region,
                                          const bool* remove_blocks,
                                          uint16_t remove_block_count,
                                          iree_arena_allocator_t* scratch_arena,
                                          uint16_t* out_removed_count);

// Removes selected unused arguments and their retained SSA references. The
// mask covers the block's arguments. The caller repairs branch payloads and
// refreshes CFG structure after completing the structural edit.
iree_status_t loom_rewriter_remove_block_args(loom_rewriter_t* rewriter,
                                              loom_block_t* block,
                                              const bool* remove_args,
                                              uint16_t remove_arg_count,
                                              uint16_t* out_removed_count);

// Moves |op| before |before_op|, preserving operands/results/regions while
// retargeting ancestry, worklist membership, and region effect summaries.
// Both ops must be owned by the rewrite module, though |op| may be staged in a
// detached same-module region. |before_op| must be rooted in the module body.
// Moving an op before itself or before its current immediate successor is a
// no-op.
iree_status_t loom_rewriter_move_before(loom_rewriter_t* rewriter,
                                        loom_op_t* op, loom_op_t* before_op);

// Moves |op| to the end of |target_block| while preserving its subtree.
//
// The target block must belong to the same module. |target_parent_op| is the
// operation that owns the target block's region and becomes the moved op's
// parent. Moving an op already at the target block end with the same parent is
// a no-op.
iree_status_t loom_rewriter_move_to_block_end(loom_rewriter_t* rewriter,
                                              loom_op_t* op,
                                              loom_block_t* target_block,
                                              loom_op_t* target_parent_op);

// Moves every block payload from |source_region| into |target_region| at
// |target_block_index| without cloning SSA values or operations.
//
// The source and target regions must be owned by |source_parent_op| and
// |target_parent_op| in the rewrite module. The source region is consumed and
// left with one empty embedded entry block so its parent can be erased. The
// moved entry payload is copied into one fresh block because a region's entry
// block has inline storage; every other block and all operations retain their
// identity. Successors targeting the old embedded entry are repaired, block
// argument definitions and operation ancestry are reparented, region-local
// block labels are cleared, and comments and semantic summaries are preserved.
// All fallible allocation and ownership checks complete before payloads move.
iree_status_t loom_rewriter_move_region_blocks(
    loom_rewriter_t* rewriter, loom_region_t* source_region,
    loom_op_t* source_parent_op, loom_region_t* target_region,
    uint16_t target_block_index, loom_op_t* target_parent_op,
    loom_block_t** out_moved_entry_block);

// Replaces one operand of an op. Adds the op and all users of its results to
// the worklist because producer semantics can change without changing the
// direct result facts.
iree_status_t loom_rewriter_set_operand(loom_rewriter_t* rewriter,
                                        loom_op_t* op, uint16_t operand_index,
                                        loom_value_id_t new_value);

// Replaces the complete operand tuple of |op| and publishes the semantic
// mutation once after all changed use-list edges have been retargeted. This is
// equivalent to setting each changed operand individually, without exposing
// analyses or worklist consumers to intermediate tuples.
iree_status_t loom_rewriter_set_operands(loom_rewriter_t* rewriter,
                                         loom_op_t* op,
                                         const loom_value_id_t* new_values);

// Changes the type of a value. Adds all users of the value to the
// worklist since they may be simplifiable with the new type (e.g.,
// encoding materialization resolving SSA encoding refs to static
// encoding table entries).
iree_status_t loom_rewriter_set_value_type(loom_rewriter_t* rewriter,
                                           loom_value_id_t value_id,
                                           loom_type_t new_type);

// Sets an attribute on an op by index. Adds all users of the op's
// results to the worklist since derived properties may change (e.g.,
// setting the effect-contract attr on a func.call changes its effective
// traits).
iree_status_t loom_rewriter_set_attr(loom_rewriter_t* rewriter, loom_op_t* op,
                                     uint16_t attr_index,
                                     loom_attribute_t value);

// Applies key-level updates to a DICT attribute on |op| and stores the fresh
// canonical dict back into the same attribute slot. Adds all users of |op|'s
// results to the worklist, just like loom_rewriter_set_attr.
iree_status_t loom_rewriter_replace_attr_dict(
    loom_rewriter_t* rewriter, loom_op_t* op, uint16_t attr_index,
    loom_named_attr_update_slice_t updates);

// Sets the instance flags on an op. Adds all users of the op's results
// to the worklist since their effective traits may change.
iree_status_t loom_rewriter_set_instance_flags(loom_rewriter_t* rewriter,
                                               loom_op_t* op, uint8_t flags);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_REWRITE_REWRITER_H_
