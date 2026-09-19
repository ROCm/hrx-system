// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Per-value fact table: dense array of loom_value_facts_t keyed by
// loom_value_id_t. Arena-allocated. A zero-initialized table is valid (empty).
//
// Lookup always succeeds: returns unknown facts for undefined entries
// or out-of-range value IDs. Undefined entries are detected by
// known_divisor == 0 (valid facts always have known_divisor >= 1),
// allowing O(1) initialization via memset(0).
//
// Define stores facts for a value ID, growing the dense value-entry array as
// needed. Compute runs a forward pass over an explicit region tree, calling
// each op's fact inference function to seed initial facts from constants and op
// semantics.
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

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Fact table
//===----------------------------------------------------------------------===//

typedef struct loom_value_fact_region_entry_t loom_value_fact_region_entry_t;
typedef struct loom_cfg_graph_t loom_cfg_graph_t;
typedef struct loom_target_facts_t loom_target_facts_t;

// Static strided logical-lane origin for one aggregate value. Result lane N is
// derived from source lane source_lane_offset + N * source_lane_stride. The
// source and result may have different element types when an operation
// preserves lane provenance while changing representation.
typedef struct loom_value_fact_static_lane_origin_t {
  // Aggregate source value containing the materialized lanes.
  loom_value_id_t source_value_id;
  // First logical source lane used by result lane zero.
  uint32_t source_lane_offset;
  // Logical source lane stride between adjacent result lanes.
  uint32_t source_lane_stride;
} loom_value_fact_static_lane_origin_t;

// Uniform scalar scale applied lanewise to one aggregate value. Result lane N
// is source lane N multiplied by scale_value_id.
typedef struct loom_value_fact_uniform_scale_origin_t {
  // Aggregate source value containing the unscaled lanes.
  loom_value_id_t source_value_id;
  // Scalar value uniformly multiplied into each result lane.
  loom_value_id_t scale_value_id;
} loom_value_fact_uniform_scale_origin_t;

// Structured contextual query materialized by an SSA value. The family is
// shared with the paired static applicability attribute. The optional key
// distinguishes projections within a family, such as one SPIR-V extension;
// queries with no additional identity use an absent key.
typedef struct loom_value_fact_contextual_query_origin_t {
  // Context-local parameterized attribute family identifying query semantics.
  loom_parameterized_attr_kind_t family_kind;

  // Reserved for alignment and future query-origin flags. Always zero.
  uint16_t reserved;

  // Borrowed structured projection key owned by the current module.
  loom_attribute_t key;
} loom_value_fact_contextual_query_origin_t;

static_assert(sizeof(loom_value_fact_contextual_query_origin_t) == 24,
              "contextual query origin must remain 24 bytes");

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

  // Optional type-domain resolver installed by layers that own registered type
  // descriptors. The fact table itself intentionally does not depend on the
  // generated type registry; callers that can map |type| to a descriptor can
  // return descriptor->fact_domain here.
  loom_value_fact_type_domain_resolver_callback_t resolve_type_domain;
};

struct loom_value_fact_table_t {
  // Arena for direct-address entries and touched-value storage.
  iree_arena_allocator_t* arena;
  // Arena for scope-local extension payloads and inference scratch buffers.
  iree_arena_allocator_t* transient_arena;

  // Dense fact entries indexed by value ID.
  loom_value_facts_t* entries;
  // Highest defined value ID plus one.
  iree_host_size_t count;
  // Allocated entry count.
  iree_host_size_t capacity;
  // Scope membership bits, retained while a cyclic solve undefines entries.
  uint64_t* touched_bits;
  // Value IDs touched in the scope, including temporarily undefined entries.
  loom_value_id_t* touched_values;
  // Number of populated entries in touched_values.
  iree_host_size_t touched_count;
  // Allocated touched_values entry count.
  iree_host_size_t touched_capacity;
  // Context object passed to op-specific fact inference callbacks.
  loom_fact_context_t context;

  // Canonical SSA identities retained while computing value facts. Only
  // declared identity operations populate this map; numeric equality does not.
  struct {
    // Canonical value IDs, or LOOM_VALUE_ID_INVALID for a value's own identity.
    loom_value_id_t* entries;
    // Allocated entry count. Storage grows lazily on the first alias.
    iree_host_size_t capacity;
  } identities;

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
    // Intrusive list of all entries for bucket-table rehashing.
    loom_value_fact_region_entry_t* entries;
  } regions;

  // Interned fact extension payloads. Extension IDs stored in
  // loom_value_facts_t are one-based indexes into entries and are only valid
  // for this table/context.
  struct {
    // Extension entries indexed by one-based extension ID minus one.
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
  // entry is LOOM_VALUE_ID_INVALID when no scalar SSA origin is known.
  struct {
    // Dense origin entries indexed by aggregate value ID.
    loom_value_id_t* entries;
    // Allocated origin entry count.
    iree_host_size_t capacity;
    // Aggregate value IDs with origins defined in the current populated scope.
    loom_value_id_t* touched_values;
    // Number of populated entries in touched_values.
    iree_host_size_t touched_count;
    // Allocated touched_values entry count.
    iree_host_size_t touched_capacity;
  } uniform_element_origins;

  // Static logical-lane origins keyed by aggregate value ID. An entry with
  // source_value_id == LOOM_VALUE_ID_INVALID has no known lane origin.
  struct {
    // Dense origin entries indexed by aggregate value ID.
    loom_value_fact_static_lane_origin_t* entries;
    // Allocated origin entry count.
    iree_host_size_t capacity;
    // Aggregate value IDs with origins defined in the current populated scope.
    loom_value_id_t* touched_values;
    // Number of populated entries in touched_values.
    iree_host_size_t touched_count;
    // Allocated touched_values entry count.
    iree_host_size_t touched_capacity;
  } static_lane_origins;

  // Uniform scalar-scale origins keyed by aggregate value ID. An entry with
  // source_value_id == LOOM_VALUE_ID_INVALID has no known scaled origin.
  struct {
    // Dense origin entries indexed by aggregate value ID.
    loom_value_fact_uniform_scale_origin_t* entries;
    // Allocated origin entry count.
    iree_host_size_t capacity;
    // Aggregate value IDs with origins defined in the current populated scope.
    loom_value_id_t* touched_values;
    // Number of populated entries in touched_values.
    iree_host_size_t touched_count;
    // Allocated touched_values entry count.
    iree_host_size_t touched_capacity;
  } uniform_scale_origins;

  // Contextual query origins keyed by SSA value ID. Dense entries contain
  // one-based indexes into the sparse origin array and are allocated only when
  // a function materializes at least one contextual query.
  struct {
    // One-based sparse origin indexes keyed by SSA value ID; zero is absent.
    uint32_t* entries;
    // Allocated dense entry count.
    iree_host_size_t capacity;
    // SSA value IDs with origins defined in the current populated scope.
    loom_value_id_t* touched_values;
    // Number of populated entries in touched_values.
    iree_host_size_t touched_count;
    // Allocated touched_values entry count.
    iree_host_size_t touched_capacity;
    // Sparse structured origins referenced by entries.
    loom_value_fact_contextual_query_origin_t* origins;
    // Number of populated sparse origins in the current scope.
    iree_host_size_t origin_count;
    // Allocated sparse origin entry count.
    iree_host_size_t origin_capacity;
  } contextual_query_origins;

  // Canonical transitive select conditions keyed by SSA value ID. Both the
  // index and dense roots have populated-scope lifetime and remain absent for
  // scopes with no value-selecting operations.
  struct {
    // Canonical set storage allocated on the first nonempty dependency.
    loom_value_set_index_t* index;
    // Canonical set roots indexed by SSA value ID; zero is empty.
    loom_value_set_id_t* roots;
    // Allocated root entry count.
    iree_host_size_t capacity;
  } select_dependencies;

  // Reusable scratch buffers for fact inference calls. Allocated on first use,
  // grown only when an op needs more slots. Never shrinks. Old buffers are
  // abandoned in the arena and freed in bulk with the arena.
  struct {
    // Fact scratch buffer state.
    struct {
      // Scratch fact values used for operand and result fact arrays.
      loom_value_facts_t* values;
      // Allocated fact scratch entry count.
      iree_host_size_t capacity;
    } facts;
    // Value ID scratch buffer state.
    struct {
      // Scratch value IDs used by transforms that materialize replacements.
      loom_value_id_t* values;
      // Allocated value ID scratch entry count.
      iree_host_size_t capacity;
    } value_ids;
    // Direct alias-ordinal map used while applying predicate facts.
    struct {
      // One-based alias ordinals keyed by SSA value ID; zero means absent.
      uint16_t* values;
      // Allocated ordinal entry count.
      iree_host_size_t capacity;
    } alias_ordinals;
  } scratch;
};

// Initializes the table with the given arena and pre-allocates
// |initial_capacity| value entries. All entries are zero-initialized
// (known_divisor == 0 means undefined). The arena is stored and used for
// subsequent value-entry growth, extension payloads, and scratch buffers.
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

// Clears facts populated in the current scope and forgets transient extension
// and scratch state. Callers that provided a separate transient arena should
// reset that arena after this call. Direct-address entry storage and touched
// storage remain allocated for reuse.
void loom_value_fact_table_clear_scope(loom_value_fact_table_t* table);

// Returns true when |value_id| has explicitly defined facts in |table|.
static inline bool loom_value_fact_table_has_entry(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  return value_id < table->capacity &&
         table->entries[value_id].known_divisor != 0;
}

// Looks up defined facts for a value in O(1). Returns false for undefined or
// out-of-range value IDs without writing |out_facts|.
static inline bool loom_value_fact_table_try_lookup(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_facts_t* out_facts) {
  if (!loom_value_fact_table_has_entry(table, value_id)) {
    return false;
  }
  *out_facts = table->entries[value_id];
  return true;
}

// Looks up facts for a value. Returns unknown facts if the value ID is out of
// range or the entry is undefined.
static inline loom_value_facts_t loom_value_fact_table_lookup(
    const loom_value_fact_table_t* table, loom_value_id_t value_id) {
  if (!loom_value_fact_table_has_entry(table, value_id)) {
    return loom_value_facts_unknown();
  }
  return table->entries[value_id];
}

// Returns true for the same valid SSA value or equal exact integer values.
// Equal ranges do not prove runtime equality. A NULL table permits SSA identity
// alone; invalid value IDs never compare equal.
bool loom_value_fact_table_values_equal(const loom_value_fact_table_t* table,
                                        loom_value_id_t lhs,
                                        loom_value_id_t rhs);

// Returns the CFG graph built while computing facts for |region|, or NULL when
// the region was not part of the populated fact scope. The returned graph is
// borrowed from |table| and remains valid until the scope is cleared or the
// owning rewriter replaces or withdraws the region's structural snapshot.
const loom_cfg_graph_t* loom_value_fact_table_lookup_cfg_graph(
    const loom_value_fact_table_t* table, const loom_region_t* region);

// Publishes the temporal distribution inherited from enclosing CFG cycles.
// The region traversal owns this transitive context; nested regions import
// their parent block's scope before computing any child facts.
iree_status_t loom_value_fact_table_set_region_temporal_scope(
    loom_value_fact_table_t* table, const loom_region_t* region,
    loom_value_facts_t scope);

// Returns inherited temporal context met with this block's execution scope
// when it belongs to a CFG cycle. This bounds facts for operations whose
// results can differ between dynamic executions. Missing context is unknown;
// a detached op cannot establish uniformity across unknown enclosing cycles.
loom_value_facts_t loom_value_fact_table_block_temporal_scope(
    const loom_value_fact_table_t* table, const loom_block_t* block);

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

// Defines |scalar_value_id| as the SSA value that can materialize every element
// of aggregate |value_id|. The relation itself is the materialization proof:
// some values also carry loom_value_fact_uniform_element_t, while others need
// the fact extension slot for a type-owned domain such as fragment metadata.
iree_status_t loom_value_fact_table_define_uniform_element_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_id_t scalar_value_id);

// Returns true when |value_id| has a known scalar SSA origin that materializes
// every element. The query validates that |value_id| is shaped, the origin is
// scalar, and both have matching element types.
bool loom_value_fact_table_query_uniform_element_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id, loom_value_id_t* out_scalar_value_id);

// Defines a static strided source-lane view for aggregate |value_id|. The
// relation is a materialization proof and is validated by the query API against
// the current module value types and static lane counts.
iree_status_t loom_value_fact_table_define_static_lane_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_static_lane_origin_t origin);

// Returns true when |value_id| has a known static source-lane view. The query
// validates that both values are vectors with static lane counts and an
// in-bounds strided source lane mapping. Element types may differ when the
// defining operation preserves lane provenance while changing representation.
bool loom_value_fact_table_query_static_lane_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id, loom_value_fact_static_lane_origin_t* out_origin);

// Defines a value as the lanewise multiplication of |origin.source_value_id|
// and scalar |origin.scale_value_id|. The relation is a materialization proof
// and is validated by the query API against the current module value types.
iree_status_t loom_value_fact_table_define_uniform_scale_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_uniform_scale_origin_t origin);

// Returns true when |value_id| has a known uniform scalar-scale origin. The
// query validates that the result/source are same-typed vectors and the scale
// is a same-element scalar.
bool loom_value_fact_table_query_uniform_scale_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id,
    loom_value_fact_uniform_scale_origin_t* out_origin);

// Defines |value_id| as the SSA materialization of one contextual query. The
// family is the descriptor-backed static applicability family with the same
// public dotted name. The optional structured key distinguishes projections
// within that family without flattening them into strings or feature bits.
iree_status_t loom_value_fact_table_define_contextual_query_origin(
    loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_fact_contextual_query_origin_t origin);

// Returns true when scalar |value_id| materializes a contextual query. The
// returned origin borrows its structured key from the current module and is
// valid until the fact-table scope is cleared.
bool loom_value_fact_table_query_contextual_query_origin(
    const loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_id_t value_id,
    loom_value_fact_contextual_query_origin_t* out_origin);

// Returns the sparse value index for retained contextual query origins. The
// borrowed values are unique, appear in first-definition order and remain valid
// until the fact-table scope is cleared.
void loom_value_fact_table_contextual_query_values(
    const loom_value_fact_table_t* table, const loom_value_id_t** out_value_ids,
    iree_host_size_t* out_value_count);

// Returns the canonical SSA identity retained for |value_id|, or |value_id|
// itself when the table is NULL or no identity has been established. This is
// an O(1) lookup, not an IR traversal. Identities have the same populated-scope
// lifetime and mutation/recomputation contract as numeric facts.
loom_value_id_t loom_value_fact_table_query_identity(
    const loom_value_fact_table_t* table, loom_value_id_t value_id);

// Begins increasing-order iteration over select conditions transitively
// controlling |value_id|. Returns the canonical set identity, or zero when
// empty; identities may be compared within one fact-table scope. The cursor
// and its results remain valid until the scope is cleared.
loom_value_set_id_t loom_value_fact_table_select_dependencies_begin(
    const loom_value_fact_table_t* table, loom_value_id_t value_id,
    loom_value_set_cursor_t* out_cursor);

// Publishes the transitive select-condition summary for each result of |op|.
// Operand summaries, scalar Boolean selector roles, and predicate-list value
// references on fact identities contribute to the canonical result set.
// Summary changes set |inout_changed| when non-NULL without clearing an
// existing change.
iree_status_t loom_value_fact_table_propagate_select_dependencies(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, const loom_op_vtable_t* vtable, bool* inout_changed);

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
// of fact inference define their results with unknown facts.
iree_status_t loom_value_fact_table_compute_op(loom_value_fact_table_t* table,
                                               const loom_module_t* module,
                                               const loom_op_t* op);

// Computes facts for a single op and reports whether any result facts changed
// relative to the table's previous entries.
iree_status_t loom_value_fact_table_compute_op_and_report(
    loom_value_fact_table_t* table, const loom_module_t* module,
    const loom_op_t* op, bool* out_changed);

// Seeds the table by running a forward pass over |region| and its nested
// regions. |function| supplies the logical function context for op fact
// inference, and may be empty for detached regions. |parent_op| is the op that
// owns |region|; it is used to seed entry-block argument facts such as loop IV
// ranges. Visits ops in dominance order so operand facts are available before
// use.
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
