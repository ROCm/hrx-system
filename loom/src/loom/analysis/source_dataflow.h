// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Table-driven monotone dataflow over a retained source program.
//
// Providers describe operation-local transfer equations over a finite bitset
// lattice. The common engine resolves source ports, adds target-independent
// structural value relations, schedules dependencies, and retains one dense
// result. Every bit accumulates monotonically. Descending feasibility domains
// are represented by accumulating rejection bits, making cyclic solutions
// deterministic and independent of traversal order.

#ifndef LOOM_ANALYSIS_SOURCE_DATAFLOW_H_
#define LOOM_ANALYSIS_SOURCE_DATAFLOW_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/source_program.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_value_fact_table_t loom_value_fact_table_t;

// Provider-defined monotone evidence attached to one source value.
typedef uint64_t loom_source_dataflow_bits_t;

// Maximum number of monotone strata in one provider.
#define LOOM_SOURCE_DATAFLOW_MAX_PHASE_COUNT 8

// Sentinel used by dense dialect tables for operation kinds with no transfer.
// Populated entries are one-based indices into the provider operation pool so
// zero-initialized generated tables are empty by construction.
#define LOOM_SOURCE_DATAFLOW_OPERATION_INDEX_NONE 0

// Maximum number of author-facing operand/result fields in one transfer row.
#define LOOM_SOURCE_DATAFLOW_MAX_PORT_COUNT 32

enum loom_source_dataflow_port_kind_e {
  // An author-facing operand field resolved through the op vtable.
  LOOM_SOURCE_DATAFLOW_PORT_OPERAND_FIELD = 0,
  // An author-facing result field resolved through the op vtable.
  LOOM_SOURCE_DATAFLOW_PORT_RESULT_FIELD = 1,
};
typedef uint8_t loom_source_dataflow_port_kind_t;

// One operation field referenced by transfer rules.
typedef struct loom_source_dataflow_port_t {
  // Operand or result field kind.
  loom_source_dataflow_port_kind_t kind;
  // Author-facing field index within the selected kind.
  uint8_t field_index;
} loom_source_dataflow_port_t;

static_assert(sizeof(loom_source_dataflow_port_t) == 2,
              "loom_source_dataflow_port_t must be 2 bytes");

enum loom_source_dataflow_rule_kind_e {
  // Adds target_bits to every selected target without a source dependency.
  LOOM_SOURCE_DATAFLOW_RULE_SEED = 0,
  // Zips selected source and target values and copies source_bits unchanged.
  LOOM_SOURCE_DATAFLOW_RULE_COPY = 1,
  // Adds target_bits when any selected source contains all source_bits.
  LOOM_SOURCE_DATAFLOW_RULE_ANY = 2,
  // Adds target_bits when every selected source contains all source_bits.
  LOOM_SOURCE_DATAFLOW_RULE_ALL = 3,
};
typedef uint8_t loom_source_dataflow_rule_kind_t;

// One compact operation-local transfer equation.
typedef struct loom_source_dataflow_rule_t {
  // Evidence required on source values, or zero for a seed.
  loom_source_dataflow_bits_t source_bits;
  // Evidence added to target values when this rule fires.
  loom_source_dataflow_bits_t target_bits;
  // Bitmask selecting source ports from the operation's port span.
  uint32_t source_port_mask;
  // Bitmask selecting target ports from the operation's port span.
  uint32_t target_port_mask;
  // Transfer equation kind.
  loom_source_dataflow_rule_kind_t kind;
  // One-based provider predicate index, or zero when unconditional.
  uint8_t predicate_index_plus_one;
  // Monotone phase that owns target_bits.
  uint8_t phase;
  // Reserved for generated rule flags; must be zero.
  uint8_t reserved;
} loom_source_dataflow_rule_t;

static_assert(sizeof(loom_source_dataflow_rule_t) == 32,
              "loom_source_dataflow_rule_t must be 32 bytes");

// Compact spans describing one source operation kind.
typedef struct loom_source_dataflow_operation_t {
  // First port in the provider port pool.
  uint16_t port_start;
  // First rule in the provider rule pool.
  uint16_t rule_start;
  // Number of operation ports.
  uint8_t port_count;
  // Number of operation rules.
  uint8_t rule_count;
  // Reserved for generated operation flags; must be zero.
  uint16_t reserved;
} loom_source_dataflow_operation_t;

static_assert(sizeof(loom_source_dataflow_operation_t) == 8,
              "loom_source_dataflow_operation_t must be 8 bytes");

// Dense operation lookup for one dialect.
typedef struct loom_source_dataflow_dialect_table_t {
  // Number of dialect-local entries.
  uint16_t operation_count;
  // One-based operation-row indices keyed by loom_op_dialect_index.
  const uint16_t* operation_indices;
} loom_source_dataflow_dialect_table_t;

// Immutable inputs visible to bounded provider callbacks.
typedef struct loom_source_dataflow_environment_t {
  // Retained source program being solved.
  const loom_source_program_t* program;
  // Semantic source facts already computed for the program.
  const loom_value_fact_table_t* fact_table;
  // Caller-defined immutable target configuration for this solve.
  const void* configuration;
} loom_source_dataflow_environment_t;

typedef iree_status_t (*loom_source_dataflow_seed_value_fn_t)(
    void* user_data, const loom_source_dataflow_environment_t* environment,
    loom_value_id_t value_id, loom_source_dataflow_bits_t* out_bits);

typedef struct loom_source_dataflow_seed_value_callback_t {
  // Optional callback invoked exactly once per indexed source value.
  loom_source_dataflow_seed_value_fn_t fn;
  // Provider-owned payload passed to fn.
  void* user_data;
} loom_source_dataflow_seed_value_callback_t;

typedef iree_status_t (*loom_source_dataflow_predicate_fn_t)(
    void* user_data, const loom_source_dataflow_environment_t* environment,
    const loom_op_t* op, bool* out_matches);

typedef struct loom_source_dataflow_predicate_t {
  // Operation-local predicate callback.
  loom_source_dataflow_predicate_fn_t fn;
  // Provider-owned payload passed to fn.
  void* user_data;
} loom_source_dataflow_predicate_t;

enum loom_source_dataflow_flow_direction_bits_e {
  // Transfer from the source-program flow source to its target.
  LOOM_SOURCE_DATAFLOW_FLOW_FORWARD = (uint8_t)1u << 0,
  // Transfer from the source-program flow target to its source.
  LOOM_SOURCE_DATAFLOW_FLOW_REVERSE = (uint8_t)1u << 1,
};
typedef uint8_t loom_source_dataflow_flow_directions_t;

#define LOOM_SOURCE_DATAFLOW_FLOW_DIRECTION_MASK                                \
  ((loom_source_dataflow_flow_directions_t)(LOOM_SOURCE_DATAFLOW_FLOW_FORWARD | \
                                            LOOM_SOURCE_DATAFLOW_FLOW_REVERSE))

// One transfer equation instantiated for matching source-program flows.
typedef struct loom_source_dataflow_flow_rule_t {
  // Evidence required or copied from the selected flow source.
  loom_source_dataflow_bits_t source_bits;
  // Evidence added to the selected flow target.
  loom_source_dataflow_bits_t target_bits;
  // Source-program flow kinds that instantiate this equation.
  loom_source_program_value_flow_kinds_t flow_kinds;
  // Natural flow directions in which to instantiate this equation.
  loom_source_dataflow_flow_directions_t directions;
  // COPY, ANY, or ALL transfer equation kind.
  loom_source_dataflow_rule_kind_t kind;
  // Monotone phase that owns target_bits.
  uint8_t phase;
  // Reserved for generated flow flags; must be zero.
  uint8_t reserved[3];
} loom_source_dataflow_flow_rule_t;

static_assert(sizeof(loom_source_dataflow_flow_rule_t) == 24,
              "loom_source_dataflow_flow_rule_t must be 24 bytes");

// One phase-boundary projection over solved evidence and indexed structure.
typedef struct loom_source_dataflow_projection_t {
  // Evidence that must all be present before the destination phase.
  loom_source_dataflow_bits_t required_bits;
  // Evidence that must all be absent before the destination phase.
  loom_source_dataflow_bits_t forbidden_bits;
  // Evidence added when the projection matches.
  loom_source_dataflow_bits_t target_bits;
  // Source-program value flags that must all be present.
  loom_source_program_value_flags_t required_value_flags;
  // Source-program value flags that must all be absent.
  loom_source_program_value_flags_t forbidden_value_flags;
  // Destination phase owning target_bits. Phase zero is invalid.
  uint8_t phase;
  // Reserved for generated projection flags; must be zero.
  uint8_t reserved[3];
} loom_source_dataflow_projection_t;

static_assert(sizeof(loom_source_dataflow_projection_t) == 32,
              "loom_source_dataflow_projection_t must be 32 bytes");

// Static finite-domain transfer provider.
typedef struct loom_source_dataflow_provider_t {
  // Stable provider name used in malformed-table diagnostics.
  iree_string_view_t name;
  // Complete set of bits the provider may produce or transfer.
  loom_source_dataflow_bits_t valid_bits;
  // Number of monotone phases in phase_bits.
  uint8_t phase_count;
  // First dialect ID covered by dialects.
  uint8_t dialect_base_id;
  // Number of dense dialect slots.
  uint8_t dialect_count;
  // Number of operation rows in operations.
  uint16_t operation_count;
  // Dense dialect slots keyed by dialect ID minus dialect_base_id.
  const loom_source_dataflow_dialect_table_t* dialects;
  // Sparse operation transfer rows.
  const loom_source_dataflow_operation_t* operations;
  // Number of ports in ports.
  uint16_t port_count;
  // Generated operation port pool.
  const loom_source_dataflow_port_t* ports;
  // Number of rules in rules.
  uint16_t rule_count;
  // Generated transfer rule pool.
  const loom_source_dataflow_rule_t* rules;
  // Number of registered operation predicates.
  uint8_t predicate_count;
  // Reserved for provider flags; must be zero.
  uint8_t reserved;
  // Disjoint evidence sets introduced by each monotone phase.
  const loom_source_dataflow_bits_t* phase_bits;
  // Registered operation predicates indexed by one-based rule references.
  const loom_source_dataflow_predicate_t* predicates;
  // Number of source-program flow transfer rows.
  uint16_t flow_rule_count;
  // Source-program flow transfer rows.
  const loom_source_dataflow_flow_rule_t* flow_rules;
  // Number of phase-boundary projection rows.
  uint16_t projection_count;
  // Phase-boundary projection rows.
  const loom_source_dataflow_projection_t* projections;
  // Optional direct value seed callback.
  loom_source_dataflow_seed_value_callback_t seed_value;
} loom_source_dataflow_provider_t;

// Deterministic solver counters retained for validation and diagnostics.
typedef struct loom_source_dataflow_statistics_t {
  // Number of direct value seed callback invocations.
  uint64_t value_seed_invocation_count;
  // Number of operation predicate callback invocations.
  uint64_t predicate_invocation_count;
  // Number of normalized transfer equations evaluated.
  uint64_t rule_evaluation_count;
  // Number of value/projection pairs evaluated at phase boundaries.
  uint64_t projection_evaluation_count;
} loom_source_dataflow_statistics_t;

// Retained dense result for one provider and source value domain.
typedef struct loom_source_dataflow_result_t {
  // Provider whose bit meanings define states.
  const loom_source_dataflow_provider_t* provider;
  // Borrowed value domain used for O(1) value-ID lookup.
  const loom_local_value_domain_t* value_domain;
  // Dense monotone evidence indexed by local value ordinal.
  loom_source_dataflow_bits_t* states;
  // Number of initialized entries in states.
  loom_value_ordinal_t state_count;
  // Solver counters for this result.
  loom_source_dataflow_statistics_t statistics;
} loom_source_dataflow_result_t;

// Solves |provider|'s transfer equations over |environment->program|.
//
// Providers are trusted static compiler data and must pass the build-time
// verifier before being registered. Result and worklist storage is arena-owned.
// Provider callbacks must be O(1) in source-program size and may inspect only
// their current value or operation, the supplied immutable environment, and
// target configuration. They must not walk definitions, users, blocks, or
// regions, recursively invoke analysis, or otherwise rediscover graph
// structure; graph discovery and convergence remain common compiler
// responsibilities.
iree_status_t loom_source_dataflow_solve(
    const loom_source_dataflow_provider_t* provider,
    const loom_source_dataflow_environment_t* environment,
    iree_arena_allocator_t* arena, loom_source_dataflow_result_t* out_result);

// Returns the evidence bits for |value_id|, or zero when it is outside the
// result domain. This is one checked ordinal lookup with no analysis work.
loom_source_dataflow_bits_t loom_source_dataflow_result_lookup(
    const loom_source_dataflow_result_t* result, loom_value_id_t value_id);

// Returns true when |value_id| contains every bit in |required_bits|.
static inline bool loom_source_dataflow_result_has_all(
    const loom_source_dataflow_result_t* result, loom_value_id_t value_id,
    loom_source_dataflow_bits_t required_bits) {
  return (loom_source_dataflow_result_lookup(result, value_id) &
          required_bits) == required_bits;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_SOURCE_DATAFLOW_H_
