// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Static semantic rules composed by the boundary projection engine.
//
// A rule claims an original logical boundary slot and selects its final
// physical component schema. Rules may depend on other original slots, but
// generated components are never recursively dispatched. A pass supplies one
// complete rule list so the engine can plan and apply every projection in one
// traversal without constructing transiently invalid signatures.

#ifndef LOOM_TRANSFORMS_BOUNDARY_PROJECTION_RULE_H_
#define LOOM_TRANSFORMS_BOUNDARY_PROJECTION_RULE_H_

#include "iree/base/api.h"
#include "loom/ir/location.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_block_t loom_block_t;
typedef struct loom_op_t loom_op_t;

typedef enum loom_boundary_projection_slot_role_e {
  LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_ARGUMENT = 0,
  LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_RESULT = 1,
  LOOM_BOUNDARY_PROJECTION_SLOT_BLOCK_ARGUMENT = 2,
  LOOM_BOUNDARY_PROJECTION_SLOT_CALL_RESULT = 3,
} loom_boundary_projection_slot_role_t;

// Cheap candidate domain for a projection rule. Bit i corresponds to
// loom_type_kind_t value i.
typedef uint32_t loom_boundary_projection_type_kind_bits_t;

#define LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(kind) \
  ((loom_boundary_projection_type_kind_bits_t)1u << (kind))

typedef struct loom_boundary_projection_plan_t loom_boundary_projection_plan_t;
typedef struct loom_boundary_projection_function_t
    loom_boundary_projection_function_t;
typedef struct loom_boundary_projection_slot_t loom_boundary_projection_slot_t;
typedef struct loom_boundary_projection_source_t
    loom_boundary_projection_source_t;
typedef struct loom_boundary_projection_rule_t loom_boundary_projection_rule_t;
typedef struct loom_boundary_projection_schema_t
    loom_boundary_projection_schema_t;

// Initializes invocation-wide state retained by a rule in the shared plan.
typedef iree_status_t (*loom_boundary_projection_initialize_fn_t)(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan);

// Prepares function-local semantic analysis after all candidate slots have
// been discovered and indexed. The callback must retain every recipe needed by
// source planning and application; later callbacks must not walk the IR to
// rediscover it.
typedef iree_status_t (*loom_boundary_projection_prepare_function_fn_t)(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function);

// Returns whether |rule| participates in |function|. This is a bounded policy
// query over the supplied function version; it must not walk IR.
typedef bool (*loom_boundary_projection_function_applies_fn_t)(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function);

// Selects the final physical schema for one logical boundary slot. A false
// result leaves the original slot unchanged. Exactly one active rule may claim
// any slot.
typedef iree_status_t (*loom_boundary_projection_plan_slot_fn_t)(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block, loom_boundary_projection_schema_t* out_schema,
    bool* out_claimed);

// Plans the physical components supplied by one outgoing call, return, or CFG
// payload. |destination| is present for CFG slots and absent for callable
// signature slots. A false result rejects the connected projection.
typedef iree_status_t (*loom_boundary_projection_plan_source_fn_t)(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_slot_t* destination,
    const loom_boundary_projection_schema_t* schema,
    loom_value_id_t source_value_id, loom_op_t* boundary_op,
    loom_boundary_projection_source_t* out_source, bool* out_planned);

// Materializes the physical components planned for one outgoing call, return,
// or successor payload.
typedef iree_status_t (*loom_boundary_projection_materialize_source_fn_t)(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_source_t* source,
    loom_value_id_t* out_component_values);

// Reconstructs one semantic definition from already-defined components at the
// active builder insertion point.
typedef iree_status_t (*loom_boundary_projection_reconstruct_fn_t)(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_t* slot, loom_type_t logical_type,
    loom_location_id_t location, loom_value_id_t* out_logical_value);

// Coherent transport behavior for a compiler-owned representation rule.
typedef struct loom_boundary_projection_transport_vtable_t {
  // Plans one outgoing payload for a claimed destination schema.
  loom_boundary_projection_plan_source_fn_t plan_source;
  // Materializes one planned outgoing payload.
  loom_boundary_projection_materialize_source_fn_t materialize_source;
  // Reconstructs the semantic value at its destination definition.
  loom_boundary_projection_reconstruct_fn_t reconstruct;
} loom_boundary_projection_transport_vtable_t;

// Compiler-owned semantic projection rule.
struct loom_boundary_projection_rule_t {
  // Stable rule name used by diagnostics and compile reports.
  iree_string_view_t name;
  // Built-in type kinds that may be claimed by this rule.
  loom_boundary_projection_type_kind_bits_t type_kind_bits;
  // Optional cheap function applicability query. NULL means all functions.
  loom_boundary_projection_function_applies_fn_t function_applies;
  // Optional invocation-wide initialization before any slots are queried.
  loom_boundary_projection_initialize_fn_t initialize;
  // Optional function-local semantic analysis after candidate discovery.
  loom_boundary_projection_prepare_function_fn_t prepare_function;
  // Bounded semantic query and schema planner.
  loom_boundary_projection_plan_slot_fn_t plan_slot;
  // Complete transport behavior. All callbacks are required for a claim.
  loom_boundary_projection_transport_vtable_t transport;
};

// Borrowed compiler-owned projection rules composed by one pass invocation.
typedef struct loom_boundary_projection_rule_list_t {
  // Static descriptors in diagnostic order only; ordering grants no priority.
  const loom_boundary_projection_rule_t* const* values;
  // Number of descriptors in values.
  iree_host_size_t count;
} loom_boundary_projection_rule_list_t;

// Generic work performed by one rule during application.
typedef struct loom_boundary_projection_rule_statistics_t {
  // Semantic values projected to a physical component schema.
  int64_t projections;
  // Physical components materialized for projected semantic values.
  int64_t components;
} loom_boundary_projection_rule_statistics_t;

// Final physical component schema selected by one rule for one logical slot.
// A NULL rule is the identity projection and has one implicit component with
// the original logical type.
struct loom_boundary_projection_schema_t {
  // Sole rule claiming the logical slot, or NULL for identity.
  const loom_boundary_projection_rule_t* rule;
  // Physical component types in boundary order, or NULL when count is zero.
  const loom_type_t* component_types;
  // Optional derived-name suffixes parallel to component_types.
  const iree_string_view_t* component_name_suffixes;
  // Number of physical components when rule is non-NULL. Zero is a valid
  // projection when reconstruct can synthesize the logical value.
  uint16_t component_count;
  // Arena-owned semantic recipe retained by the claiming rule.
  void* rule_plan;
};

// Returns true when a semantic rule replaces the logical slot.
static inline bool loom_boundary_projection_schema_is_projected(
    const loom_boundary_projection_schema_t* schema) {
  return schema->rule != NULL;
}

// One rule-owned outgoing projection recipe. Identity slots leave this zeroed.
struct loom_boundary_projection_source_t {
  // Rule that planned this source; matches the destination slot schema.
  const loom_boundary_projection_rule_t* rule;
  // Arena-owned semantic recipe interpreted by rule.
  void* rule_plan;
  // Outgoing boundary operation anchoring any required materialization.
  loom_op_t* boundary_op;
};

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_BOUNDARY_PROJECTION_RULE_H_
