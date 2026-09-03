// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-independent physical source-representation planning.
//
// Targets describe bounded operation-local representation alternatives with
// compact generated tables. The common planner connects those alternatives
// through the retained source program, intersects finite representation
// domains, rejects non-separable choices, compares verified Low descriptor
// recipes, and retains one result for constant-time lowering queries.

#ifndef LOOM_CODEGEN_LOW_SOURCE_REPRESENTATION_H_
#define LOOM_CODEGEN_LOW_SOURCE_REPRESENTATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/source_dataflow.h"
#include "loom/analysis/source_program.h"
#include "loom/codegen/low/descriptor_cost.h"
#include "loom/util/bstring.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_table_t loom_value_fact_table_t;
typedef struct loom_target_facts_t loom_target_facts_t;

// Stable target-owned identity of one physical source representation.
typedef uint64_t loom_low_source_representation_key_t;

// Sentinel for an absent physical source representation.
#define LOOM_LOW_SOURCE_REPRESENTATION_KEY_NONE UINT64_C(0)

// Sentinel for an absent provider-local representation index.
#define LOOM_LOW_SOURCE_REPRESENTATION_INDEX_NONE UINT16_MAX

// Sentinel for an absent provider-local candidate index.
#define LOOM_LOW_SOURCE_REPRESENTATION_CANDIDATE_INDEX_NONE UINT16_MAX

// Sentinel for an absent target-data row.
#define LOOM_LOW_SOURCE_REPRESENTATION_TARGET_DATA_ORDINAL_NONE UINT32_MAX

// Sentinel in dense dialect tables. Populated entries are one-based operation
// indices so zero-initialized generated tables remain empty.
#define LOOM_LOW_SOURCE_REPRESENTATION_OPERATION_INDEX_NONE 0

// Selects every value in an author-facing variadic field. Fixed fields may use
// zero or this sentinel equivalently.
#define LOOM_LOW_SOURCE_REPRESENTATION_PORT_ALL_ELEMENTS UINT16_MAX

// Maximum number of logical component slots in one candidate group.
#define LOOM_LOW_SOURCE_REPRESENTATION_MAX_COMPONENT_COUNT 32

enum loom_low_source_representation_port_kind_e {
  // An author-facing operand field resolved through the op vtable.
  LOOM_LOW_SOURCE_REPRESENTATION_PORT_OPERAND_FIELD = 0,
  // An author-facing result field resolved through the op vtable.
  LOOM_LOW_SOURCE_REPRESENTATION_PORT_RESULT_FIELD = 1,
};
typedef uint8_t loom_low_source_representation_port_kind_t;

// One source value field participating in an operation-local component slot.
typedef struct loom_low_source_representation_port_t {
  // Operand or result field kind.
  loom_low_source_representation_port_kind_t kind;
  // Author-facing field index within the selected kind.
  uint8_t field_index;
  // Group-local logical component slot containing the selected value.
  uint8_t component_index;
  // Reserved for generated port flags; must be zero.
  uint8_t reserved;
  // Field element index, or PORT_ALL_ELEMENTS for the complete field.
  uint16_t element_index;
  // Reserved for generated port extensions; must be zero.
  uint16_t extension;
} loom_low_source_representation_port_t;

static_assert(sizeof(loom_low_source_representation_port_t) == 8,
              "loom_low_source_representation_port_t must be 8 bytes");

// One provider-owned physical representation identity. The identity covers
// every component-wide choice that must remain coherent, including exact
// materialization policy; two costed realizations sharing a carrier layout use
// distinct keys when downstream operations must preserve that choice.
typedef struct loom_low_source_representation_t {
  // Stable nonzero identity generated from the representation key.
  loom_low_source_representation_key_t stable_key;
  // B-string table offset of the human-readable stable key.
  loom_bstring_table_offset_t name_string_offset;
  // Reserved for generated representation metadata; must be zero.
  uint32_t reserved;
} loom_low_source_representation_t;

static_assert(sizeof(loom_low_source_representation_t) == 16,
              "loom_low_source_representation_t must be 16 bytes");

enum loom_low_source_representation_binding_flag_bits_e {
  // This representation is the canonical fallback for the component domain.
  LOOM_LOW_SOURCE_REPRESENTATION_BINDING_CANONICAL = (uint8_t)1u << 0,
};
typedef uint8_t loom_low_source_representation_binding_flags_t;

#define LOOM_LOW_SOURCE_REPRESENTATION_BINDING_FLAG_MASK \
  ((loom_low_source_representation_binding_flags_t)      \
       LOOM_LOW_SOURCE_REPRESENTATION_BINDING_CANONICAL)

// One candidate binding for a group-local component slot.
typedef struct loom_low_source_representation_binding_t {
  // Provider-local physical representation index.
  uint16_t representation_index;
  // Canonical-fallback classification for this domain occurrence.
  loom_low_source_representation_binding_flags_t flags;
  // Reserved for generated binding metadata; must be zero.
  uint8_t reserved;
} loom_low_source_representation_binding_t;

static_assert(sizeof(loom_low_source_representation_binding_t) == 4,
              "loom_low_source_representation_binding_t must be 4 bytes");

enum loom_low_source_representation_proof_e {
  // Unknown proof classification; invalid on selectable candidates.
  LOOM_LOW_SOURCE_REPRESENTATION_PROOF_UNKNOWN = 0,
  // The target table proves the complete logical-to-physical mapping exactly.
  LOOM_LOW_SOURCE_REPRESENTATION_PROOF_EXACT = 1,
  // A target capability contract makes the representation legal while the
  // implementation-defined physical mapping remains opaque to Loom.
  LOOM_LOW_SOURCE_REPRESENTATION_PROOF_OPAQUE = 2,
};
typedef uint8_t loom_low_source_representation_proof_t;

// One operation-local realization candidate.
typedef struct loom_low_source_representation_candidate_t {
  // Stable nonzero identity used for diagnostics and deterministic lookup.
  uint64_t stable_key;
  // B-string table offset of the human-readable candidate key.
  loom_bstring_table_offset_t name_string_offset;
  // Provider target-data row retained for target lowering, or NONE.
  uint32_t target_data_ordinal;
  // First component binding in the provider binding pool.
  uint16_t binding_start;
  // Number of component bindings; equal to the owning group's component count.
  uint16_t binding_count;
  // First Low descriptor recipe entry in the provider entry pool.
  uint16_t recipe_entry_start;
  // Number of Low descriptor recipe entries.
  uint16_t recipe_entry_count;
  // First recipe dependency in the provider dependency pool.
  uint16_t recipe_dependency_start;
  // Number of recipe dependencies.
  uint16_t recipe_dependency_count;
  // First durable-pressure delta in the provider pressure pool.
  uint16_t durable_pressure_delta_start;
  // Number of durable-pressure deltas.
  uint16_t durable_pressure_delta_count;
  // One-based provider predicate index, or zero when unconditional.
  uint8_t predicate_index_plus_one;
  // Correctness evidence for the physical mapping.
  loom_low_source_representation_proof_t proof;
  // Reserved for generated candidate flags; must be zero.
  uint16_t reserved;
} loom_low_source_representation_candidate_t;

static_assert(sizeof(loom_low_source_representation_candidate_t) == 40,
              "loom_low_source_representation_candidate_t must be 40 bytes");

enum loom_low_source_representation_group_flag_bits_e {
  // The group imposes no constraint when target predicates reject every
  // candidate. Without this flag an unavailable group is a planning problem.
  LOOM_LOW_SOURCE_REPRESENTATION_GROUP_OPTIONAL = (uint8_t)1u << 0,
};
typedef uint8_t loom_low_source_representation_group_flags_t;

#define LOOM_LOW_SOURCE_REPRESENTATION_GROUP_FLAG_MASK \
  ((loom_low_source_representation_group_flags_t)      \
       LOOM_LOW_SOURCE_REPRESENTATION_GROUP_OPTIONAL)

// One operation-local set of representation alternatives.
typedef struct loom_low_source_representation_group_t {
  // Stable nonzero group identity.
  uint64_t stable_key;
  // B-string table offset of the human-readable group key.
  loom_bstring_table_offset_t name_string_offset;
  // First source value port in the provider port pool.
  uint16_t port_start;
  // First realization candidate in the provider candidate pool.
  uint16_t candidate_start;
  // Number of source value ports.
  uint8_t port_count;
  // Number of realization candidates.
  uint8_t candidate_count;
  // Number of group-local logical component slots.
  uint8_t component_count;
  // Group behavior flags.
  loom_low_source_representation_group_flags_t flags;
} loom_low_source_representation_group_t;

static_assert(sizeof(loom_low_source_representation_group_t) == 24,
              "loom_low_source_representation_group_t must be 24 bytes");

// Compact span of candidate groups attached to one source operation kind.
typedef struct loom_low_source_representation_operation_t {
  // First candidate group in the provider group pool.
  uint16_t group_start;
  // Number of candidate groups.
  uint8_t group_count;
  // Reserved for generated operation flags; must be zero.
  uint8_t reserved;
} loom_low_source_representation_operation_t;

static_assert(sizeof(loom_low_source_representation_operation_t) == 4,
              "loom_low_source_representation_operation_t must be 4 bytes");

// Dense operation lookup for one dialect.
typedef struct loom_low_source_representation_dialect_table_t {
  // Number of dialect-local operation entries.
  uint16_t operation_count;
  // One-based operation indices keyed by loom_op_dialect_index.
  const uint16_t* operation_indices;
} loom_low_source_representation_dialect_table_t;

// Immutable inputs visible to bounded provider callbacks.
typedef struct loom_low_source_representation_environment_t {
  // Source module containing callback values and operations.
  const loom_module_t* module;
  // Semantic source facts already computed for the program.
  const loom_value_fact_table_t* fact_table;
  // Optional solved physical source-value dataflow for the same program.
  const loom_source_dataflow_result_t* source_dataflow;
  // Active verified Low descriptor set used to cost candidate recipes.
  const loom_low_descriptor_set_t* descriptor_set;
  // Typed target facts selected for this planning attempt.
  const loom_target_facts_t* target_facts;
  // Caller-defined immutable target configuration for this plan.
  const void* configuration;
} loom_low_source_representation_environment_t;

// One finite representation domain supplied for a concrete source value.
typedef struct loom_low_source_representation_domain_t {
  // Borrowed strictly increasing provider-local representation indices.
  const uint16_t* representation_indices;
  // Number of indices in representation_indices.
  uint16_t representation_count;
  // Unique canonical representation index in representation_indices.
  uint16_t canonical_representation_index;
  // True when this callback constrains the value. False leaves it
  // unconstrained.
  bool constrained;
} loom_low_source_representation_domain_t;

typedef iree_status_t (*loom_low_source_representation_seed_value_fn_t)(
    void* user_data,
    const loom_low_source_representation_environment_t* environment,
    loom_value_id_t value_id,
    loom_low_source_representation_domain_t* out_domain);

typedef struct loom_low_source_representation_seed_value_callback_t {
  // Optional callback invoked exactly once per indexed source value.
  loom_low_source_representation_seed_value_fn_t fn;
  // Provider-owned payload passed to fn.
  void* user_data;
} loom_low_source_representation_seed_value_callback_t;

typedef iree_status_t (*loom_low_source_representation_predicate_fn_t)(
    void* user_data,
    const loom_low_source_representation_environment_t* environment,
    const loom_op_t* op, bool* out_matches);

typedef struct loom_low_source_representation_predicate_t {
  // Operation-local target capability predicate.
  loom_low_source_representation_predicate_fn_t fn;
  // Provider-owned payload passed to fn.
  void* user_data;
} loom_low_source_representation_predicate_t;

// Static finite-domain representation provider.
typedef struct loom_low_source_representation_provider_t {
  // Stable provider name used in malformed-table diagnostics.
  iree_string_view_t name;
  // Packed representation, group, and candidate names.
  loom_bstring_table_t string_table;
  // Number of physical representations.
  uint16_t representation_count;
  // Physical representation identity rows.
  const loom_low_source_representation_t* representations;
  // First dialect ID covered by dialects.
  uint8_t dialect_base_id;
  // Number of dense dialect slots.
  uint8_t dialect_count;
  // Dense dialect slots keyed by dialect ID minus dialect_base_id.
  const loom_low_source_representation_dialect_table_t* dialects;
  // Number of sparse operation rows.
  uint16_t operation_count;
  // Sparse operation candidate-group rows.
  const loom_low_source_representation_operation_t* operations;
  // Number of candidate groups.
  uint16_t group_count;
  // Candidate group pool.
  const loom_low_source_representation_group_t* groups;
  // Number of source value ports.
  uint16_t port_count;
  // Source value port pool.
  const loom_low_source_representation_port_t* ports;
  // Number of realization candidates.
  uint16_t candidate_count;
  // Realization candidate pool.
  const loom_low_source_representation_candidate_t* candidates;
  // Number of candidate component bindings.
  uint16_t binding_count;
  // Candidate component binding pool.
  const loom_low_source_representation_binding_t* bindings;
  // Number of Low descriptor recipe entries.
  uint16_t recipe_entry_count;
  // Low descriptor recipe entry pool.
  const loom_low_descriptor_recipe_entry_t* recipe_entries;
  // Number of Low descriptor recipe dependencies.
  uint16_t recipe_dependency_count;
  // Low descriptor recipe dependency pool.
  const loom_low_descriptor_recipe_dependency_t* recipe_dependencies;
  // Number of durable-pressure delta rows.
  uint16_t durable_pressure_delta_count;
  // Durable-pressure delta pool.
  const loom_low_pressure_delta_t* durable_pressure_deltas;
  // Number of registered candidate predicates.
  uint8_t predicate_count;
  // Reserved for provider flags; must be zero.
  uint8_t reserved;
  // Candidate predicates indexed by one-based candidate references.
  const loom_low_source_representation_predicate_t* predicates;
  // Byte stride of one target-data row, or zero when no rows exist.
  uint32_t target_data_stride;
  // Number of target-data rows.
  uint32_t target_data_count;
  // Target-owned static row storage addressed by candidate ordinals.
  const uint8_t* target_data;
  // Optional direct value-domain callback.
  loom_low_source_representation_seed_value_callback_t seed_value;
} loom_low_source_representation_provider_t;

enum loom_low_source_representation_problem_kind_e {
  LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_NONE = 0,
  // A required operation-local candidate group has no legal candidates.
  LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_UNAVAILABLE_GROUP = 1,
  // Intersecting exact component constraints produced an empty domain.
  LOOM_LOW_SOURCE_REPRESENTATION_PROBLEM_EMPTY_DOMAIN = 2,
};
typedef uint8_t loom_low_source_representation_problem_kind_t;

// User-program planning failure retained for common lowering diagnostics.
typedef struct loom_low_source_representation_problem_t {
  // Planning failure classification.
  loom_low_source_representation_problem_kind_t kind;
  // Source operation contributing the failing constraint, when available.
  const loom_op_t* source_op;
  // Representative source value in the failing component, when available.
  loom_value_id_t source_value_id;
  // Provider-local group index, or UINT16_MAX when no group owns the failure.
  uint16_t group_index;
} loom_low_source_representation_problem_t;

// Deterministic planner counters retained for tests and diagnostics.
typedef struct loom_low_source_representation_statistics_t {
  // Number of direct value-domain callback invocations.
  uint64_t value_seed_invocation_count;
  // Number of candidate predicate callback invocations.
  uint64_t predicate_invocation_count;
  // Number of source-program preserving edges unioned.
  uint64_t preserving_flow_count;
  // Number of operation-local candidate groups instantiated.
  uint64_t candidate_group_count;
  // Number of constrained physical components selected.
  uint64_t selected_component_count;
} loom_low_source_representation_statistics_t;

// Dense per-node span into selected_groups. Block and unhandled operation nodes
// have empty spans.
typedef struct loom_low_source_representation_node_selection_t {
  // First selected group row.
  uint32_t group_start;
  // Number of group rows for this source operation.
  uint8_t group_count;
  // Reserved for retained node-selection metadata; must be zero.
  uint8_t reserved[3];
} loom_low_source_representation_node_selection_t;

// One retained operation-local group choice.
typedef struct loom_low_source_representation_group_selection_t {
  // Provider-local candidate group index.
  uint16_t group_index;
  // Selected provider-local candidate index, or NONE for a skipped optional
  // group.
  uint16_t candidate_index;
} loom_low_source_representation_group_selection_t;

// Retained common representation plan for one source value domain.
typedef struct loom_low_source_representation_plan_t {
  // Provider whose representation indices define this plan.
  const loom_low_source_representation_provider_t* provider;
  // Borrowed retained source program used for operation lookup.
  const loom_source_program_t* program;
  // Borrowed value domain used for constant-time value lookup.
  const loom_local_value_domain_t* value_domain;
  // Provider-local selected representation indexed by local value ordinal.
  uint16_t* value_representation_indices;
  // Union-find root ordinal indexed by local value ordinal.
  loom_value_ordinal_t* value_component_roots;
  // Selected aggregate cost indexed by component root ordinal.
  const loom_low_descriptor_cost_t** component_costs;
  // Per-source-program-node selected group spans.
  loom_low_source_representation_node_selection_t* node_selections;
  // Selected candidate rows grouped by source operation node.
  loom_low_source_representation_group_selection_t* selected_groups;
  // Number of rows in selected_groups.
  uint32_t selected_group_count;
  // Open-addressed source-op pointer to source-program-node lookup slots.
  uint32_t* operation_lookup_slots;
  // Power-of-two number of entries in operation_lookup_slots.
  uint32_t operation_lookup_slot_count;
  // Optional user-program planning failure.
  loom_low_source_representation_problem_t problem;
  // Planner counters for this result.
  loom_low_source_representation_statistics_t statistics;
} loom_low_source_representation_plan_t;

// Read-only selected physical representation for one source value.
typedef struct loom_low_source_representation_value_view_t {
  // True when the value belongs to a constrained selected component.
  bool selected;
  // Dense component representative ordinal.
  loom_value_ordinal_t component_ordinal;
  // Provider representation row, or NULL when unselected.
  const loom_low_source_representation_t* representation;
  // Human-readable stable representation key, or an empty view.
  iree_string_view_t representation_name;
  // Aggregate component choice cost, or NULL when unselected.
  const loom_low_descriptor_cost_t* cost;
} loom_low_source_representation_value_view_t;

// Read-only selected realization for one operation-local candidate group.
typedef struct loom_low_source_representation_candidate_view_t {
  // True when a required or matching optional candidate was selected.
  bool selected;
  // Provider candidate group row.
  const loom_low_source_representation_group_t* group;
  // Human-readable stable group key.
  iree_string_view_t group_name;
  // Provider candidate row, or NULL for a skipped optional group.
  const loom_low_source_representation_candidate_t* candidate;
  // Human-readable stable candidate key, or an empty view.
  iree_string_view_t candidate_name;
  // Target-owned retained data row, or NULL when the candidate has none.
  const void* target_data;
} loom_low_source_representation_candidate_view_t;

// Plans physical source representations over |program|.
//
// Providers are trusted static compiler data and must pass the build-time
// verifier before registration. Result and all work storage are arena-owned.
// A seed callback is invoked once per indexed value and each predicate is
// invoked once per relevant operation. Callbacks may inspect only that value
// or operation plus immutable environment data. They must not walk
// definitions, users, blocks, regions, or recursively invoke another analysis.
iree_status_t loom_low_source_representation_plan(
    const loom_low_source_representation_provider_t* provider,
    const loom_source_program_t* program,
    const loom_low_source_representation_environment_t* environment,
    iree_arena_allocator_t* arena,
    loom_low_source_representation_plan_t* out_plan);

// Returns the selected representation for |value_id|. Values outside the plan
// domain or in unconstrained components return an unselected view.
loom_low_source_representation_value_view_t
loom_low_source_representation_plan_lookup_value(
    const loom_low_source_representation_plan_t* plan,
    loom_value_id_t value_id);

// Returns the number of retained candidate groups for |source_op|.
iree_host_size_t loom_low_source_representation_plan_candidate_count(
    const loom_low_source_representation_plan_t* plan,
    const loom_op_t* source_op);

// Returns one retained candidate group choice for |source_op|.
loom_low_source_representation_candidate_view_t
loom_low_source_representation_plan_candidate_view(
    const loom_low_source_representation_plan_t* plan,
    const loom_op_t* source_op, iree_host_size_t group_ordinal);

// Finds one retained group choice by stable group key.
bool loom_low_source_representation_plan_find_candidate(
    const loom_low_source_representation_plan_t* plan,
    const loom_op_t* source_op, uint64_t group_key,
    loom_low_source_representation_candidate_view_t* out_view);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_SOURCE_REPRESENTATION_H_
