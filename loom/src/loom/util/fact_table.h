// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Per-value fact table: dense range of loom_value_facts_t keyed by
// loom_value_id_t. Arena-allocated. A zero-initialized table is valid (empty).
//
// Lookup always succeeds: returns unknown facts for undefined entries
// or out-of-range value IDs. Undefined entries are detected by
// known_divisor == 0 (valid facts always have known_divisor >= 1), so newly
// allocated slots need only zero initialization.
//
// Define stores facts for a value ID, growing the dense value-entry range as
// needed. Lazy tables start at their first definition instead of allocating an
// empty prefix for unrelated module values. Explicit capacity reservations
// cover the zero-based domain. Compute runs a forward pass over a region tree,
// calling each op's fact inference function to seed initial facts from
// constants and op semantics.
//
// The table is a reusable component: borrowed by the rewriter for
// canonicalization, owned by pass-scoped storage, and usable standalone for IPO
// or analysis tools.
//
// Typical lifecycle:
//
//   loom_value_fact_table_t table = {0};
//   IREE_RETURN_IF_ERROR(loom_value_fact_table_initialize(
//       &table, arena, value_count));
//   IREE_RETURN_IF_ERROR(loom_value_fact_table_compute(
//       &table, module, function));
//   loom_value_facts_t facts = loom_value_fact_table_lookup(&table, id);

#ifndef LOOM_UTIL_FACT_TABLE_H_
#define LOOM_UTIL_FACT_TABLE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/facts.h"
#include "loom/ir/ir.h"
#include "loom/ir/type_dependencies.h"
#include "loom/util/fact_extensions.h"
#include "loom/util/fact_identity.h"
#include "loom/util/fact_layout.h"
#include "loom/util/fact_origin.h"
#include "loom/util/fact_range.h"
#include "loom/util/fact_select_dependencies.h"

#ifdef __cplusplus
extern "C" {
#endif

static_assert(LOOM_TYPE_COUNT_ <= 16,
              "widen fact-table result type-kind storage");

//===----------------------------------------------------------------------===//
// Fact table
//===----------------------------------------------------------------------===//

typedef struct loom_value_fact_region_entry_t loom_value_fact_region_entry_t;
typedef struct loom_value_fact_region_frame_t loom_value_fact_region_frame_t;
typedef struct loom_value_fact_condition_scratch_t
    loom_value_fact_condition_scratch_t;
typedef struct loom_value_fact_exact_lane_origin_entry_t
    loom_value_fact_exact_lane_origin_entry_t;
typedef struct loom_value_fact_vector_partitions_t
    loom_value_fact_vector_partitions_t;
typedef struct loom_value_fact_integer_relations_t
    loom_value_fact_integer_relations_t;
typedef struct loom_condition_edge_projection_t
    loom_condition_edge_projection_t;
typedef struct loom_cfg_graph_t loom_cfg_graph_t;
typedef struct loom_target_facts_t loom_target_facts_t;
typedef struct loom_value_fact_loop_replay_t loom_value_fact_loop_replay_t;
typedef struct loom_value_fact_loop_reads_t loom_value_fact_loop_reads_t;

// Per-analysis context passed to op fact inference callbacks.
struct loom_fact_context_t {
  // Table that owns the dense facts and any extension payloads allocated by
  // inference helpers.
  loom_value_fact_table_t* table;

  // Function-like op whose projected region is currently being analyzed. Empty
  // when facts are computed for individual detached ops instead of a full
  // function-like projection.
  loom_func_like_t function;

  // Entry origin for the projected region currently being analyzed.
  // Unknown for detached operations and non-function region analyses.
  loom_value_fact_reference_origin_t reference_origin;

  // Optional immutable target facts for target-sensitive inference. Generic
  // analyses leave this NULL and receive source-level facts. Target-family
  // domains access family-owned data through their checked fact cast.
  const loom_target_facts_t* target_facts;

  // Resolves an authored region target through its owner's symbol-fact cache.
  // The returned immutable facts outlive this populated scope. Missing target
  // definitions or an absent resolver leave an explicitly targeted region
  // unbound. Region seeding projects the selected environment to its worker.
  struct {
    // Owner of the symbol-fact projection cache.
    void* user_data;
    // Projects a verified target reference; only allocation may fail.
    iree_status_t (*fn)(void* user_data, const loom_module_t* module,
                        loom_symbol_ref_t target,
                        const loom_target_facts_t** out_facts);
  } resolve_region_target;

  // Optionally resolves direct-call result facts through an analysis-owned
  // callable summary. Generic scopes leave this empty and retain ordinary
  // type-derived facts at call boundaries. The callback fills every result;
  // unsupported calls return unknown facts rather than inspecting callees.
  struct {
    // Owner of the callable summary cache.
    void* user_data;
    // Imports one verified call's result facts into this fact context.
    iree_status_t (*fn)(void* user_data, loom_fact_context_t* context,
                        const loom_module_t* module, const loom_op_t* op,
                        const loom_value_facts_t* operand_facts,
                        loom_value_facts_t* result_facts);
  } resolve_call_results;

  // Optional type-domain resolver installed by layers that own registered type
  // descriptors. The fact table itself intentionally does not depend on the
  // generated type registry; callers that can map |type| to a descriptor can
  // return descriptor->fact_domain here.
  loom_value_fact_type_domain_resolver_callback_t resolve_type_domain;

  // Optional immutable operand scope supplied by the owning analysis during
  // a complete solve. Refinement changes the scratch operands, not the global
  // facts of values defined outside the guarded operation.
  struct {
    // Borrowed state that outlives the solve.
    void* user_data;
    // Applies retained scope constraints to op->operand_count scratch facts.
    void (*fn)(void* user_data, const loom_value_fact_table_t* table,
               const loom_op_t* op, loom_value_facts_t* operand_facts);
  } refine_operands;

  // Active change accumulator for auxiliary result facts published by the
  // current inference callback. NULL when publication needs no change report,
  // and outside an inference callback.
  bool* result_facts_changed;
};

struct loom_value_fact_table_t {
  // Arena for reusable direct-address, touched-value, and oversized extension
  // storage.
  iree_arena_allocator_t* arena;
  // Arena for pooled extension entries, payloads, hash buckets, and scratch.
  iree_arena_allocator_t* transient_arena;

  // Dense fact entries indexed by value ID minus first_value_id.
  loom_value_facts_t* entries;
  // First value ID covered by entries, aligned to a touched-membership word.
  // Lazy scopes omit the empty prefix preceding their first definition.
  loom_value_id_t first_value_id;
  // Highest defined value ID plus one.
  iree_host_size_t count;
  // Allocated entry count.
  iree_host_size_t capacity;
  // Scope membership bits indexed relative to first_value_id, retained while
  // a cyclic solve undefines entries.
  uint64_t* touched_bits;
  // Values with numeric facts or direct-origin storage in the scope,
  // including temporarily undefined entries.
  loom_value_id_t* touched_values;
  // Number of populated entries in touched_values.
  iree_host_size_t touched_count;
  // Allocated touched_values entry count.
  iree_host_size_t touched_capacity;
  // Type kinds of results defined in the current scope. Incremental rewrites
  // retain kinds until the scope is cleared, making this a conservative set.
  uint16_t result_type_kind_bits;
  // Context object passed to op-specific fact inference callbacks.
  loom_fact_context_t context;
  // Stack-owned ordinary-result publication, active only during inference.
  loom_value_fact_origin_publication_t* origin_publication;

  // Optional completion consumer installed by the owner of pending inference.
  // Nested summaries report completion within this table's solver traversal.
  // Preview tables can borrow inference context without completing this work.
  struct {
    // Borrowed consumer state, valid while the callback is installed.
    void* user_data;
    // Consumes completed work without modifying source IR or value facts.
    void (*fn)(void* user_data, const loom_op_t* op);
  } summary_computed;

  // Immutable scope inputs supplied by a surrounding analysis. Independent
  // entry arguments and opaque results consult these on every solve instead
  // of retaining constraints accumulated in their derived rows.
  struct {
    // Borrowed source whose selected facts outlive this populated scope.
    const loom_value_fact_table_t* table;
    // Transient bitmap selecting source entries by module-local value ID.
    uint64_t* selected_bits;
    // Number of allocated words in selected_bits.
    iree_host_size_t word_count;
  } seeds;

  // The current numeric solve is replaying a completed traversal without IR
  // edits. Constant results and structural selector dependencies remain valid.
  // Nested solves inherit this state; each solver restores its caller's state.
  bool replaying;
  // Rewriter-owned full body witness permits only this result-only loop to
  // refresh its carried closure on its first feedback round. Nested solves
  // do not inherit the permission.
  const loom_op_t* sparse_first_owner;

  // Facts derived under path conditions require whole-scope invalidation after
  // edits. The incremental rewriter cannot maintain their guard dependencies.
  bool has_conditioned_results;

  // At least one structured region entry has Boolean branch semantics.
  bool has_boolean_branch_regions;

  // This fact scope has published at least one counted-loop domain.
  bool has_counted_loop_domains;

  // Extension entries bypassed the block pool and persist across scopes.
  bool has_persistent_extension_entries;

  // Structured region entries with visible projected integer relations.
  uint32_t condition_integer_projection_count;

  // Canonical SSA identities retained while computing value facts. Only
  // declared identity operations populate this map; numeric equality does not.
  struct {
    // Publication tokens indexed relative to range.first_value_id. NULL
    // leaves the value its own identity.
    loom_value_fact_identity_t** entries;
    // Value-ID range covered by the direct entries.
    loom_value_fact_range_t range;
    // Scope-owned tokens, released with the transient arena after clearing.
    loom_value_fact_identity_t* nodes;
  } identities;

  // Signed relations carried by scoped SSA aliases, including sparse interior
  // literal exclusions that scalar interval facts cannot represent. All
  // storage is allocated lazily in the transient arena; scopes without
  // retained relations leave this NULL.
  loom_value_fact_integer_relations_t* integer_relations;

  // Region execution context and optional CFG structure retained by the fact
  // owner. Entries and buckets have transient scope lifetime.
  struct {
    // Hash buckets containing collision chains keyed by region address.
    loom_value_fact_region_entry_t** buckets;
    // Power-of-two hash bucket count.
    iree_host_size_t bucket_count;
    // Number of analyzed regions.
    iree_host_size_t count;
    // Number of published CFG snapshots, used to skip CFG-only rewrite work.
    iree_host_size_t cfg_count;
    // This scope has computed structured loop or branch summaries. Retained
    // until scope reset, including after rewrites remove the original owners.
    bool has_structured_summaries;
    // True after encountering an independent execution target. Ordinary
    // function inference bypasses per-operation region target lookup.
    bool has_independent_targets;
  } regions;

  // Interned fact extension payloads. Extension IDs stored in
  // loom_value_facts_t are one-based indexes into entries and are only valid
  // for this table/context.
  struct {
    // Extension entries indexed by one-based extension ID minus one. Only
    // entries below count are valid in the current scope.
    loom_value_fact_extension_entry_t* entries;
    // Allocated extension entry count.
    iree_host_size_t capacity;
    // Defined extension entry count.
    iree_host_size_t count;
    // Hash buckets storing one-based extension IDs, or zero for empty buckets.
    loom_value_fact_extension_id_t* buckets;
    // Allocated hash bucket count.
    iree_host_size_t bucket_count;
  } extensions;

  // Uniform-element materialization origins keyed by aggregate value ID. An
  // entry with scalar_value_id == LOOM_VALUE_ID_INVALID has no known origin.
  struct {
    // Origin entries indexed relative to range.first_value_id.
    loom_value_fact_uniform_element_origin_t* entries;
    // Value-ID range covered by the direct entries.
    loom_value_fact_range_t range;
  } uniform_element_origins;

  // Static logical-lane origins keyed by aggregate value ID. An entry with
  // source_value_id == LOOM_VALUE_ID_INVALID has no known lane origin.
  struct {
    // Origin entries indexed relative to range.first_value_id.
    loom_value_fact_static_lane_origin_t* entries;
    // Value-ID range covered by the direct entries.
    loom_value_fact_range_t range;
  } static_lane_origins;

  // Exact logical-lane value origins keyed by aggregate value ID. This is
  // separate from static_lane_origins because provenance may cross a lossy
  // conversion while the nearest exact numeric origin stops at that
  // conversion. Sparse storage makes the cost proportional to known exact
  // origins instead of the highest module-wide value ID.
  struct {
    // Sparse entries sorted by aggregate value ID.
    loom_value_fact_exact_lane_origin_entry_t* entries;
    // Retained slots in the scope, including withdrawn proofs.
    iree_host_size_t count;
    // Allocated sparse entry count retained across populated scopes.
    iree_host_size_t capacity;
  } exact_lane_origins;

  // Exact scalar numeric origins keyed by result value ID. Direct entries are
  // allocated lazily when a function first publishes a preserving conversion.
  struct {
    // Source value IDs indexed relative to range.first_value_id.
    loom_value_id_t* entries;
    // Value-ID range covered by the direct entries.
    loom_value_fact_range_t range;
  } exact_scalar_origins;

  // Uniform scalar-scale origins keyed by aggregate value ID. An entry with
  // source_value_id == LOOM_VALUE_ID_INVALID has no known scaled origin.
  struct {
    // Origin entries indexed relative to range.first_value_id.
    loom_value_fact_uniform_scale_origin_t* entries;
    // Value-ID range covered by the direct entries.
    loom_value_fact_range_t range;
  } uniform_scale_origins;

  // Contextual query origins keyed by SSA value ID. Dense entries contain
  // one-based indexes into the sparse origin array and are allocated only when
  // a function materializes at least one contextual query.
  struct {
    // One-based descriptor indexes relative to range.first_value_id; zero
    // is absent.
    uint32_t* entries;
    // Value-ID range covered by the direct entries.
    loom_value_fact_range_t range;
    // SSA value IDs with origins defined in the current populated scope.
    loom_value_id_t* touched_values;
    // Number of populated entries in touched_values.
    iree_host_size_t touched_count;
    // Allocated touched_values entry count.
    iree_host_size_t touched_capacity;
    // One descriptor per retained value; ANY marks a withdrawn proof.
    loom_value_fact_contextual_query_origin_t* origins;
    // Number of populated sparse origins in the current scope.
    iree_host_size_t origin_count;
    // Allocated sparse origin entry count.
    iree_host_size_t origin_capacity;
  } contextual_query_origins;

  // Scope-local physical-layout provenance. Allocated only when an exact SSA
  // layout recipe or storage-schema identity must survive local inference;
  // separate from numeric extensions that cross call scopes.
  loom_value_fact_layout_origins_t* layout_origins;

  // Scope-local exact vector partitions indexed by logical vector value ID.
  loom_value_fact_vector_partitions_t* vector_partitions;

  // Canonical transitive select conditions keyed by SSA value ID. Both the
  // set storage and root pages have populated-scope lifetime and are absent
  // for scopes with no value-selecting operations.
  struct {
    // Canonical set storage allocated on the first nonempty dependency.
    loom_value_set_index_t* index;
    // Sparse root pages; NULL entries represent empty ranges of SSA IDs.
    loom_value_fact_select_dependency_page_t** pages;
    // Allocated page pointer count, or zero before the first publication.
    iree_host_size_t page_capacity;
  } select_dependencies;

  // Exact predicate-bearing identities awaiting path-local fact transport.
  // The pointer array is allocated only when such an identity is discovered.
  struct {
    // Operations observed exact during scope population or incremental change.
    loom_op_t** ops;
    // Number of pending observations. Cyclic solves may repeat an operation.
    iree_host_size_t count;
    // Allocated operation pointer count.
    iree_host_size_t capacity;
  } exact_relations;

  // Reusable scratch buffers for fact inference calls. Allocated on first use,
  // grown only when an evaluation needs more slots. Old buffers are
  // abandoned in the arena and freed in bulk with the arena.
  struct {
    // Fact scratch buffer state.
    struct {
      // Scratch fact values used for operand and result fact arrays.
      loom_value_facts_t* values;
      // Allocated fact scratch entry count.
      iree_host_size_t capacity;
    } facts;
    // Continuations shared by lexical traversals and reentrant loop solvers.
    struct {
      // Active nested frames; each traversal keeps its root on the native
      // stack.
      loom_value_fact_region_frame_t* values;
      // Live prefix reserved across all suspended and current traversals.
      iree_host_size_t count;
      // Allocated frame count, reused until the scope is cleared.
      iree_host_size_t capacity;
    } regions;
    // Family publication marks for the current ordinary operation.
    struct {
      // One family bitset per result ordinal, reused between operations.
      uint8_t* values;
      // Allocated result count, grown only for larger operations.
      iree_host_size_t capacity;
    } origins;
    // Value ID scratch buffer state.
    struct {
      // Scratch value IDs used by transforms that materialize replacements.
      loom_value_id_t* values;
      // Allocated value ID scratch entry count.
      iree_host_size_t capacity;
    } value_ids;
    // Exact vector-partition component scratch buffer state.
    struct {
      // Scratch components populated by structural fact producers.
      struct loom_value_fact_vector_partition_component_t* values;
      // Allocated component scratch entry count.
      iree_host_size_t capacity;
    } vector_partition_components;
    // Reusable alias-ordinal index used by predicate and relation publication.
    struct {
      // Hash buckets of one-based alias ordinals; all zero outside a map lease.
      uint16_t* buckets;
      // Allocated bucket count, grown geometrically with alias cardinality.
      iree_host_size_t capacity;
    } alias_ordinals;
    // Lazily allocated query scratch for condition-loop edge projections.
    loom_value_fact_condition_scratch_t* condition;
  } scratch;
  // Numeric dependency tracking, activated lazily by iterative solvers.
  struct {
    // Stack-owned outer/nested solve boundary; NULL outside loop inference.
    loom_value_fact_loop_replay_t* loop;
    // Active dependency collector; NULL for flat loops and snapshot comparison.
    loom_value_fact_loop_reads_t* reads;
    // One transient-arena generation per dense row. Zero is unevaluated; the
    // low bit marks a publication not yet established by its defining op.
    // Completed equations record the scope generation at evaluation, allowing
    // consumers to see changes even after a nested equation has converged.
    uint32_t* generations;
    // Even publication serial, or zero when tracking is inactive. Wraparound
    // expires every evaluation before the serial restarts.
    uint32_t generation;
  } replay;
#ifndef NDEBUG
  // Diagnostics for retained coherent observations; absent in release builds.
  struct {
    // Current semantic mutation batch or populated scope generation.
    uint64_t generation;
    // Queued equations have not completed publication.
    bool pending;
  } observation;
#endif
};

// Initializes the table with the given arena and pre-allocates
// |initial_capacity| value entries. All entries are zero-initialized
// (known_divisor == 0 means undefined). The arena is stored and used for
// subsequent reusable entry growth, extension payloads, and scratch buffers.
iree_status_t loom_value_fact_table_initialize(
    loom_value_fact_table_t* table, iree_arena_allocator_t* arena,
    iree_host_size_t initial_capacity);

// Initializes the table using separate arenas for persistent direct-address
// entries and transient scope-local payloads. Use this when reusing one table
// across multiple populated scopes: clear touched entries, reset the transient
// arena, and keep the direct entry array live.
iree_status_t loom_value_fact_table_initialize_with_arenas(
    loom_value_fact_table_t* table, iree_arena_allocator_t* arena,
    iree_arena_allocator_t* transient_arena, iree_host_size_t initial_capacity);

// Reserves entries for |range|, preserving defined facts and touched
// membership. Empty ranges allocate nothing. Existing storage grows
// geometrically when it cannot cover the range; an empty table allocates only
// the requested range and omits preceding module values. Does not populate
// entries.
iree_status_t loom_value_fact_table_reserve_range(
    loom_value_fact_table_t* table, loom_value_fact_range_t range);

// Reserves entries for [0, |minimum_capacity|), preserving defined facts and
// touched membership. An existing capacity doubles until it covers the
// minimum, but only the final array is allocated. This retains growth headroom
// without leaving intermediate arrays in the arena during batch population.
// An empty table starts at the requested minimum. Does not populate entries.
iree_status_t loom_value_fact_table_reserve(loom_value_fact_table_t* table,
                                            iree_host_size_t minimum_capacity);

// Clears facts populated in the current scope and forgets transient extension
// payload, hash, and scratch state. Callers that provided a separate transient
// arena should reset that arena after this call. Direct-address, touched-value,
// and oversized extension-entry storage remain allocated for reuse.
void loom_value_fact_table_clear_scope(loom_value_fact_table_t* table);

// Retains unique scope membership without defining numeric facts. Direct-origin
// setters use this so withdrawal and republication cannot duplicate cleanup
// work.
iree_status_t loom_value_fact_table_touch(loom_value_fact_table_t* table,
                                          loom_value_id_t value_id);

// Retains a dependency in a preallocated active collector. Reads allocate
// nothing; their owning solve publishes complete snapshots at completion.
// Keep this bookkeeping out of the many inlined numeric query callsites.
IREE_ATTRIBUTE_NOINLINE void loom_value_fact_loop_record_read(
    loom_value_fact_loop_reads_t* reads, loom_value_id_t value_id);

static inline void loom_value_fact_table_record_read(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (table && table->replay.reads) {
    loom_value_fact_loop_record_read(table->replay.reads, value_id);
  }
}

// Returns true when |value_id| has explicitly defined facts in |table|.
static inline bool loom_value_fact_table_has_entry(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  loom_value_fact_table_record_read(table, value_id);
  const iree_host_size_t index =
      (iree_host_size_t)value_id - table->first_value_id;
  return index < table->capacity && table->entries[index].known_divisor != 0;
}

// Looks up defined facts for a value in O(1). Returns false for undefined or
// out-of-range value IDs without writing |out_facts|.
static inline bool loom_value_fact_table_try_lookup(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_facts_t* out_facts) {
  if (!loom_value_fact_table_has_entry(table, value_id)) {
    return false;
  }
  *out_facts = table->entries[value_id - table->first_value_id];
  return true;
}

// Looks up facts for a value. Returns unknown facts if the value ID is out of
// range or the entry is undefined.
static inline loom_value_facts_t loom_value_fact_table_lookup(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (!loom_value_fact_table_has_entry(table, value_id)) {
    return loom_value_facts_unknown();
  }
  return table->entries[value_id - table->first_value_id];
}

// Returns a shaped type's maximum element count from its static dimensions and
// indexed dynamic-extent facts. UINT64_MAX denotes an unavailable or
// overflowing bound. Work is bounded by the type's rank and allocates no
// storage.
uint64_t loom_value_fact_table_maximum_element_count(
    const loom_value_fact_table_t* table, loom_type_t type);

// Returns true for a shared retained identity or equal exact integer values.
// Callers establish compatible types and interpretation; this is value
// identity, not ordered floating-point equality (an identical NaN remains
// unordered). Equal ranges do not prove runtime equality. A NULL table permits
// raw SSA identity alone; invalid value IDs never compare equal. The query is
// indexed and does not change either value's refined facts.
bool loom_value_fact_table_values_equal(const loom_value_fact_table_t* table,
                                        loom_value_id_t lhs,
                                        loom_value_id_t rhs);

// Returns the CFG graph built while computing facts for |region|, or NULL when
// the region was not part of the populated fact scope. The returned graph is
// borrowed from |table| and remains valid until the scope is cleared or the
// owning rewriter replaces or withdraws the region's structural snapshot.
const loom_cfg_graph_t* loom_value_fact_table_lookup_cfg_graph(
    const loom_value_fact_table_t* table, const loom_region_t* region);

// Returns the operation retained when this region's context was established.
loom_op_t* loom_value_fact_table_region_parent(
    const loom_value_fact_table_t* table, const loom_region_t* region);

// Rebuilds independent entry arguments from immutable inputs and contracts.
iree_status_t loom_value_fact_table_refresh_entry_args(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_block_t* block, loom_op_t* parent_op);

// Publishes the temporal distribution and repetition inherited from enclosing
// control flow. |may_repeat| includes structured loops and enclosing CFG
// cycles. The region traversal owns this transitive context; nested regions
// import their parent block's scope before computing any child facts.
iree_status_t loom_value_fact_table_set_region_temporal_scope(
    loom_value_fact_table_t* table, const loom_region_t* region,
    loom_op_t* parent_op, loom_value_facts_t scope, bool may_repeat);

// Returns inherited temporal context met with this block's execution scope
// when it belongs to a CFG cycle. This bounds facts for operations whose
// results can differ between dynamic executions. Missing context is unknown;
// a detached op cannot establish uniformity across unknown enclosing cycles.
loom_value_facts_t loom_value_fact_table_block_temporal_scope(
    const loom_value_fact_table_t* table, const loom_block_t* block);

// Returns whether a block may execute more than once per function invocation.
// Consumes retained enclosing repetition and the block's own CFG component.
// Missing context cannot prove single execution and returns true.
bool loom_value_fact_table_block_may_repeat(
    const loom_value_fact_table_t* table, const loom_block_t* block);

// Publishes the root context before a region solve. The caller supplies the
// enclosing target through context.target_facts; an independent target on
// |parent_op|'s region descriptor takes precedence.
iree_status_t loom_value_fact_table_seed_root_target_scope(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_region_t* region, const loom_op_t* parent_op);

// Resolves direct child region targets once at their owning operation. Ordinary
// structured regions inherit the operation's retained context; independently
// executing regions use an explicit target or inherit the enclosing target
// when their target attribute is absent, then project to its worker contract.
iree_status_t loom_value_fact_table_seed_nested_target_scopes(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op);

// Returns the retained execution target of |block|. This lookup never walks
// ancestors. Unregistered blocks in a mixed-target solve have no target facts.
const loom_target_facts_t* loom_value_fact_table_block_target_facts(
    const loom_value_fact_table_t* table, const loom_block_t* block);

// Publishes condition facts and their SSA mapping onto |region| arguments. The
// projection must use the table's transient arena and remains producer-owned.
// Recomputing a structured summary updates it in place.
iree_status_t loom_value_fact_table_set_region_condition_projection(
    loom_value_fact_table_t* table, const loom_region_t* region,
    loom_condition_edge_projection_t* projection);

// Retains the Boolean selector truth established upon entry to |region|.
// Repeated publication of the same immutable interface role is idempotent.
iree_status_t loom_value_fact_table_set_region_branch_truth(
    loom_value_fact_table_t* table, const loom_region_t* region,
    loom_region_branch_truth_t truth);

// Returns the Boolean selector truth established upon entry to |region|.
loom_region_branch_truth_t loom_value_fact_table_lookup_region_branch_truth(
    const loom_value_fact_table_t* table, const loom_region_t* region);

// Returns retained condition-projection storage for |region|, or NULL when no
// condition facts have been established. Recomputing a condition may leave the
// projection empty so later rewrites can reuse its capacity.
const loom_condition_edge_projection_t*
loom_value_fact_table_lookup_region_condition_projection(
    const loom_value_fact_table_t* table, const loom_region_t* region);

// Returns mutable retained condition-projection storage for an owning fact
// producer. Consumers query the const view above.
loom_condition_edge_projection_t*
loom_value_fact_table_lookup_mutable_region_condition_projection(
    loom_value_fact_table_t* table, const loom_region_t* region);

// Receives a borrowed graph from a populated fact scope. The callback must not
// mutate that scope or its graphs; any retained pointer has the same lifetime
// as lookup_cfg_graph. A failing callback terminates enumeration.
typedef struct loom_value_fact_cfg_graph_callback_t {
  // Callback state borrowed for the duration of enumeration.
  void* user_data;
  // Invoked once for each currently published graph, in unspecified order.
  iree_status_t (*fn)(void* user_data, const loom_cfg_graph_t* graph);
} loom_value_fact_cfg_graph_callback_t;

// Enumerates existing graph snapshots without walking IR or constructing
// graphs. Withdrawn snapshots are omitted. An empty scope invokes no callbacks.
iree_status_t loom_value_fact_table_enumerate_cfg_graphs(
    const loom_value_fact_table_t* table,
    loom_value_fact_cfg_graph_callback_t callback);

// Defines (or updates) facts for a value, growing the table if needed.
iree_status_t loom_value_fact_table_define(loom_value_fact_table_t* table,
                                           loom_value_id_t value_id,
                                           loom_value_facts_t facts);

// Returns a populated value to the not-yet-computed state during a cyclic
// update without removing its scope membership or allocating another record.
void loom_value_fact_table_undefine(loom_value_fact_table_t* table,
                                    loom_value_id_t value_id);

// Returns the sparse value index for the populated fact scope. The borrowed
// values are unique, appear in first-definition order, and remain valid until
// the scope is cleared. Values temporarily undefined during a cyclic solve
// remain in the index so consumers observe the complete analyzed scope.
void loom_value_fact_table_scope_values(const loom_value_fact_table_t* table,
                                        const loom_value_id_t** out_value_ids,
                                        iree_host_size_t* out_value_count);

// Returns true when the populated scope has defined a result of |type_kind|.
static inline bool loom_value_fact_table_has_result_type_kind(
    const loom_value_fact_table_t* table, loom_type_kind_t type_kind) {
  return table != NULL &&
         iree_any_bit_set(table->result_type_kind_bits,
                          (uint16_t)(UINT16_C(1) << type_kind));
}

// Returns the canonical SSA identity retained for |value_id|, or |value_id|
// itself when the table is NULL or no identity has been established. This is
// read-only indexed lookup through O(log publications) representative links,
// not an IR traversal. Unmerged publications have no links to follow.
// Replacement preserves live representatives before deferred numeric inference.
static inline loom_value_id_t loom_value_fact_table_query_identity(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  loom_value_fact_table_record_read(table, value_id);
  if (!table ||
      !loom_value_fact_range_contains(table->identities.range, value_id)) {
    return value_id;
  }
  loom_value_fact_identity_t* identity =
      table->identities.entries[loom_value_fact_range_offset(
          table->identities.range, value_id)];
  if (!identity) {
    return value_id;
  }
  const loom_value_id_t representative =
      loom_value_fact_identity_root(identity)->representative;
  return representative != LOOM_VALUE_ID_INVALID ? representative : value_id;
}

// Returns true when exact predicate-bearing identities await consumption.
static inline bool loom_value_fact_table_has_pending_exact_relations(
    const loom_value_fact_table_t* table) {
  return table && table->exact_relations.count > 0;
}

// Returns exact predicate-bearing identity operations that still relate at
// least one dynamic SSA value. The borrowed observation array may contain the
// same operation more than once after a cyclic solve and remains valid until
// the transient fact arena is reset. No storage is allocated when empty.
void loom_value_fact_table_pending_exact_relations(
    const loom_value_fact_table_t* table, loom_op_t* const** out_ops,
    iree_host_size_t* out_op_count);

// Marks every currently pending exact-relation observation as consumed. The
// pointer storage remains available for later incremental observations.
void loom_value_fact_table_clear_pending_exact_relations(
    loom_value_fact_table_t* table);

// Borrowed subset of a fact table. The table and value IDs must remain valid
// while the view is consumed. Value IDs are unique and may name unset entries.
typedef struct loom_value_fact_table_view_t {
  // Source table owning the facts and their extension payloads.
  const loom_value_fact_table_t* table;
  // Module-local value IDs selected by the producer of this scope.
  const loom_value_id_t* value_ids;
  // Number of entries in value_ids.
  iree_host_size_t value_count;
} loom_value_fact_table_view_t;

// Clones the defined entries selected by |source|, with their identities and
// origins, into |target| using the same value IDs. When |module| is provided,
// extension payloads use the type-owned domain implied by each value ID.
// Undefined entries remain unset in |target| so normal block-argument and op
// fact seeding can fill them.
iree_status_t loom_value_fact_table_clone_values(
    loom_value_fact_table_t* target, loom_value_fact_table_view_t source,
    const loom_module_t* module);

// Imports selected inputs and retains their immutable source for entry
// arguments and opaque result inference. Unlike an ordinary clone, this
// preserves external inputs when subsequent solves replace their derived facts.
// The source and selected facts remain valid and unchanged until target's
// scope is cleared. Changing a seeded value's meaning requires rebuilding the
// scope. Ordinary semantic-preserving rewrites and type refinement retain the
// input contract.
iree_status_t loom_value_fact_table_seed_values(
    loom_value_fact_table_t* target, loom_value_fact_table_view_t source,
    const loom_module_t* module);

// Propagates SSA identities and retained materialization and contextual origins
// through declared value-alias and fact-identity operations. Numeric facts are
// computed separately. Identity changes set |inout_changed| when non-NULL
// without clearing changes already reported by numeric inference.
iree_status_t loom_value_fact_table_propagate_origins(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, bool* inout_changed);

// Computes facts for a single op. Ordinary ops call their vtable fact inference
// function; LoopLike and RegionBranch ops visit their nested regions and
// summarize the values returned to the parent results. Ops without either form
// of fact inference derive result facts only from their types. Scalar integer
// results retain their type's numeric domain even when inference is unknown.
iree_status_t loom_value_fact_table_compute_op(loom_value_fact_table_t* table,
                                               const loom_module_t* module,
                                               const loom_op_t* op);

// Activates numeric change tracking for the current fact scope. Iterative
// solvers call this before their first traversal; nested solves share the
// publication serial. clear_scope releases the transient column. This does not
// enter immutable replay: the solver owns table->replaying separately.
iree_status_t loom_value_fact_table_enable_replay_generations(
    loom_value_fact_table_t* table);

// Computes facts for a single op and reports whether any result facts changed
// relative to the table's previous entries.
iree_status_t loom_value_fact_table_compute_op_and_report(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, bool* out_changed);

// Publishes an op's results using maintained nested-region facts. RegionBranch
// ops join current yields without reevaluating their bodies; LoopLike ops still
// solve feedback. Callers establish direct inputs and nested execution context
// before using this boundary. Ordinary ops compute their complete result facts.
iree_status_t loom_value_fact_table_compute_op_summary(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, bool* out_changed);

// Computes nested regions using the already established function context.
// Acyclic lexical descent retains continuations in transient storage. LoopLike
// and CFG summaries own the iterative solves for their regions.
iree_status_t loom_value_fact_table_compute_region_tree(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_region_t* region, loom_op_t* parent_op);

// Applies each result's indexed type-extent constraints before publishing its
// facts. Missing result facts are unknown. Only final changes are reported to
// the rewrite owner; other definitions are never modified by publication.
iree_status_t loom_value_fact_table_define_results(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* result_facts,
    uint16_t result_count, bool* out_changed);

// Seeds the table by running a forward pass over |region| and its nested
// regions. |function| supplies the logical function context for op fact
// inference, and may be empty for detached regions. |parent_op| is the op that
// owns |region|; it is used to seed entry-block argument facts such as loop IV
// ranges. Visits ops in dominance order so operand facts are available before
// use. Unconstrained index entry parameters have the selected target's signed
// native domain, or the source domain without a target. Declared preconditions,
// imported facts, computed values, and internal block arguments retain their
// producer-established domains. Offset and fixed-width parameters retain their
// source domains. Independent entry arguments are rebuilt from immutable
// imported inputs, declaration projections, and current IR contracts on each
// solve. External argument facts must be supplied with seed_values; define and
// clone_values publish derived facts that inference may replace.
iree_status_t loom_value_fact_table_compute_region(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_func_like_t function, loom_region_t* region, loom_op_t* parent_op);

// Seeds the table by running a forward pass over |function|'s root regions and
// nested regions. Non-body root regions are analyzed before the body so source
// containers can compute launch/configuration facts that body ops query through
// the same fact table.
iree_status_t loom_value_fact_table_compute(loom_value_fact_table_t* table,
                                            const loom_module_t* module,
                                            loom_func_like_t function);

// Returns a facts scratch buffer with at least |count| entries.
// The returned pointer is valid until the next call. Grows the
// allocation if needed; never shrinks.
iree_status_t loom_value_fact_table_facts_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_facts_t** out);

// Returns a value ID scratch buffer with at least |count| entries.
// Same lifetime and growth semantics as facts_scratch.
iree_status_t loom_value_fact_table_value_id_scratch(
    loom_value_fact_table_t* table, iree_host_size_t count,
    loom_value_id_t** out);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_UTIL_FACT_TABLE_H_
