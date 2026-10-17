// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/boundary/loop_predicate_projection.h"

#include "loom/ir/module.h"
#include "loom/target/function_version.h"
#include "loom/target/provider.h"
#include "loom/transforms/boundary/predicate_word.h"
#include "loom/transforms/boundary/projection_plan.h"

// Derived name suffix for the physical predicate component.
static const iree_string_view_t kLoopPredicateComponentNameSuffix =
    IREE_SVL("word");

typedef struct loom_loop_predicate_source_plan_t {
  // Projected loop slot supplying the word, or IREE_HOST_SIZE_MAX.
  iree_host_size_t slot_index;
  // Logical i1 value materialized when slot_index is IREE_HOST_SIZE_MAX.
  loom_value_id_t logical_value_id;
} loom_loop_predicate_source_plan_t;

static bool loom_loop_predicate_projection_function_applies(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function) {
  (void)rule;
  (void)plan;
  const loom_target_function_version_t* version =
      loom_target_function_version_const_cast(function->version);
  return version != NULL && version->resolved_target.provider != NULL &&
         version->resolved_target.provider->loop_predicate_carrier ==
             LOOM_TARGET_LOOP_PREDICATE_CARRIER_I32;
}

static bool loom_loop_predicate_projection_slot_matches(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block) {
  (void)rule;
  (void)role;
  (void)block;
  const loom_target_function_version_t* version =
      loom_target_function_version_const_cast(function->version);
  return loom_predicate_word_supports_type(
      loom_module_value_type(plan->module, value_id),
      version->resolved_target.provider
          ->loop_predicate_max_vector_element_count);
}

static iree_status_t loom_loop_predicate_projection_plan_slot(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block, loom_boundary_projection_schema_t* out_schema,
    bool* out_claimed) {
  (void)role;
  (void)block;
  *out_schema = (loom_boundary_projection_schema_t){0};
  *out_claimed = false;
  const loom_type_t logical_type =
      loom_module_value_type(plan->module, value_id);
  const loom_value_facts_t facts =
      loom_value_fact_table_lookup(function->facts, value_id);
  // Scalar i1 values with a subgroup-uniform proof lower as ordinary wave
  // booleans. Every other loop-carried scalar may require a native lane mask;
  // unknown distribution is therefore not safe to leave direct. Vector i1
  // values always use mask storage on targets selecting this carrier.
  if (loom_type_is_scalar(logical_type) &&
      loom_value_facts_is_subgroup_uniform(facts)) {
    return iree_ok_status();
  }

  loom_type_t* component_type = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(plan->arena, sizeof(*component_type),
                                           (void**)&component_type));
  *component_type = loom_predicate_word_type(logical_type);
  *out_schema = (loom_boundary_projection_schema_t){
      .rule = rule,
      .component_types = component_type,
      .component_name_suffixes = &kLoopPredicateComponentNameSuffix,
      .component_count = 1,
  };
  *out_claimed = true;
  return iree_ok_status();
}

static iree_status_t loom_loop_predicate_projection_plan_source(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_slot_t* destination,
    const loom_boundary_projection_schema_t* schema,
    loom_value_id_t source_value_id, loom_op_t* boundary_op,
    loom_boundary_projection_source_t* out_source, bool* out_planned) {
  (void)schema;
  *out_source = (loom_boundary_projection_source_t){0};
  *out_planned = false;

  loom_loop_predicate_source_plan_t* source_plan = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(plan->arena, sizeof(*source_plan),
                                           (void**)&source_plan));
  *source_plan = (loom_loop_predicate_source_plan_t){
      .slot_index = IREE_HOST_SIZE_MAX,
      .logical_value_id = source_value_id,
  };
  const iree_host_size_t source_index =
      loom_boundary_projection_slot_index(function, source_value_id);
  if (source_index != IREE_HOST_SIZE_MAX &&
      function->candidates[source_index].selected &&
      function->candidates[source_index].schema.rule == rule) {
    source_plan->slot_index = source_index;
    source_plan->logical_value_id = LOOM_VALUE_ID_INVALID;
    const iree_host_size_t destination_index =
        (iree_host_size_t)(destination - function->candidates);
    if (source_index != destination_index) {
      IREE_RETURN_IF_ERROR(loom_boundary_projection_add_dependency(
          plan, function, source_index, destination_index,
          /*orders_realization=*/false));
    }
  }
  *out_source = (loom_boundary_projection_source_t){
      .rule = rule,
      .rule_plan = source_plan,
      .boundary_op = boundary_op,
  };
  *out_planned = true;
  return iree_ok_status();
}

static iree_status_t loom_loop_predicate_projection_materialize_source(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_source_t* source,
    loom_value_id_t* out_component_values) {
  (void)rule;
  const loom_loop_predicate_source_plan_t* source_plan =
      (const loom_loop_predicate_source_plan_t*)source->rule_plan;
  if (source_plan->slot_index != IREE_HOST_SIZE_MAX) {
    out_component_values[0] =
        function->candidates[source_plan->slot_index].component_value_ids[0];
    return iree_ok_status();
  }

  loom_builder_ip_t saved_ip = loom_builder_save(&plan->rewriter.builder);
  const loom_value_id_t logical_value_id =
      loom_boundary_projection_resolve_value(function,
                                             source_plan->logical_value_id);
  const loom_type_t logical_type =
      loom_module_value_type(plan->module, logical_value_id);
  const loom_type_t component_type = loom_predicate_word_type(logical_type);
  iree_status_t status = loom_predicate_word_materialize(
      &plan->rewriter.builder, logical_value_id, component_type,
      source->boundary_op->location, &out_component_values[0]);
  loom_builder_restore(&plan->rewriter.builder, saved_ip);
  return status;
}

static iree_status_t loom_loop_predicate_projection_reconstruct(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_t* slot, loom_type_t logical_type,
    loom_location_id_t location, loom_value_id_t* out_logical_value) {
  (void)function;
  IREE_RETURN_IF_ERROR(loom_predicate_word_reconstruct(
      &plan->rewriter.builder, slot->component_value_ids[0], logical_type,
      location, out_logical_value));
  loom_boundary_projection_record(plan, rule, 1, 1);
  return iree_ok_status();
}

static const loom_boundary_projection_rule_t kLoopPredicateProjectionRule = {
    .name = IREE_SVL("loop-predicate-i32"),
    .type_kind_bits = LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(LOOM_TYPE_SCALAR) |
                      LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(LOOM_TYPE_VECTOR),
    .slot_role_bits = LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
        LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE),
    .function_applies = loom_loop_predicate_projection_function_applies,
    .slot_matches = loom_loop_predicate_projection_slot_matches,
    .plan_slot = loom_loop_predicate_projection_plan_slot,
    .transport =
        {
            .plan_source = loom_loop_predicate_projection_plan_source,
            .materialize_source =
                loom_loop_predicate_projection_materialize_source,
            .reconstruct = loom_loop_predicate_projection_reconstruct,
        },
};

const loom_boundary_projection_rule_t*
loom_loop_predicate_boundary_projection_rule(void) {
  return &kLoopPredicateProjectionRule;
}
