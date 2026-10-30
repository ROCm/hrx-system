// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/cfg_tuple_projection.h"

#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/capture.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/target/registers.h"
#include "loom/transforms/boundary/projection_plan.h"

typedef struct loom_low_cfg_tuple_slice_use_t {
  // Admitted low.slice operation consuming the logical tuple.
  loom_op_t* op;
  // First physical component observed by the slice.
  uint16_t first_component;
  // Number of consecutive unit components observed by the slice.
  uint16_t component_count;
} loom_low_cfg_tuple_slice_use_t;

typedef struct loom_low_cfg_tuple_slot_plan_t {
  // Exact admitted destination uses indexed during planning.
  loom_low_cfg_tuple_slice_use_t* slice_uses;
  // Number of entries in slice_uses.
  uint32_t slice_use_count;
} loom_low_cfg_tuple_slot_plan_t;

typedef enum loom_low_cfg_tuple_source_kind_e {
  // Slices a stable tuple value retained by the recipe.
  LOOM_LOW_CFG_TUPLE_SOURCE_SLICE_VALUE = 0,
  // Captures a retained concat operand at the concat's original program point.
  LOOM_LOW_CFG_TUPLE_SOURCE_SLICE_CONCAT_OPERAND = 1,
} loom_low_cfg_tuple_source_kind_t;

typedef struct loom_low_cfg_tuple_source_component_t {
  // Materialization strategy for this component.
  loom_low_cfg_tuple_source_kind_t kind;
  // Retained source locator interpreted according to kind.
  union {
    // Stable value sliced by LOOM_LOW_CFG_TUPLE_SOURCE_SLICE_VALUE.
    loom_value_id_t value_id;
    // Retained concat operand read after destination-use elimination.
    struct {
      // Exact concat operation selected during planning.
      loom_op_t* op;
      // Operand containing this component.
      uint16_t operand_index;
    } concat_operand;
  } source;
  // Unit offset within the selected source when a slice is required.
  uint32_t source_offset;
} loom_low_cfg_tuple_source_component_t;

typedef struct loom_low_cfg_tuple_source_plan_t {
  // One materialization recipe per destination component.
  loom_low_cfg_tuple_source_component_t* components;
  // Unit-register type produced by every component recipe.
  loom_type_t component_type;
  // Number of entries in components.
  uint16_t component_count;
} loom_low_cfg_tuple_source_plan_t;

static bool loom_low_cfg_tuple_function_applies(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function) {
  (void)rule;
  (void)plan;
  loom_region_t* body = loom_func_like_body(function->function);
  return loom_low_function_def_isa(function->function.op) && body &&
         body->block_count > 1;
}

static bool loom_low_cfg_tuple_slot_matches(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block) {
  (void)rule;
  if (role != LOOM_BOUNDARY_PROJECTION_SLOT_BLOCK_ARGUMENT || !block) {
    return false;
  }
  loom_region_t* body = loom_func_like_body(function->function);
  if (!body || block == loom_region_entry_block(body)) {
    return false;
  }
  const loom_type_t type = loom_module_value_type(plan->module, value_id);
  if (!loom_type_is_register(type) || loom_type_register_has_value_type(type)) {
    return false;
  }
  const uint32_t component_count = loom_low_register_type_unit_count(type);
  return component_count > 1 && component_count <= UINT16_MAX;
}

static bool loom_low_cfg_tuple_match_slice(
    const loom_boundary_projection_plan_t* plan, loom_type_t component_type,
    uint16_t component_count, loom_op_t* op,
    loom_low_cfg_tuple_slice_use_t* out_slice_use) {
  if (!loom_low_slice_isa(op)) {
    return false;
  }
  const loom_type_t result_type =
      loom_module_value_type(plan->module, loom_low_slice_result(op));
  const uint32_t slice_component_count =
      loom_low_register_type_unit_count(result_type);
  if (loom_type_register_has_value_type(result_type) ||
      !loom_low_register_type_same_class(result_type, component_type) ||
      slice_component_count == component_count) {
    return false;
  }
  *out_slice_use = (loom_low_cfg_tuple_slice_use_t){
      .op = op,
      .first_component = (uint16_t)loom_low_slice_offset(op),
      .component_count = (uint16_t)slice_component_count,
  };
  return true;
}

static iree_status_t loom_low_cfg_tuple_plan_slot(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block, loom_boundary_projection_schema_t* out_schema,
    bool* out_claimed) {
  *out_schema = (loom_boundary_projection_schema_t){0};
  *out_claimed = false;
  IREE_ASSERT(loom_low_cfg_tuple_slot_matches(rule, plan, function, role,
                                              value_id, block));

  const loom_type_t tuple_type = loom_module_value_type(plan->module, value_id);
  const uint16_t component_count =
      (uint16_t)loom_low_register_type_unit_count(tuple_type);
  const loom_type_t component_type =
      loom_low_register_carrier_type_with_unit_count(tuple_type, 1);
  const loom_value_t* value = loom_module_value(plan->module, value_id);
  if (value->use_count == 0 || loom_value_has_attribute_uses(value) ||
      loom_module_value_has_type_uses(plan->module, value_id)) {
    return iree_ok_status();
  }

  loom_low_cfg_tuple_slot_plan_t* slot_plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(plan->arena, sizeof(*slot_plan), (void**)&slot_plan));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, value->use_count, sizeof(*slot_plan->slice_uses),
      (void**)&slot_plan->slice_uses));
  slot_plan->slice_use_count = value->use_count;

  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    if (loom_use_operand_index(uses[i]) != 0) {
      return iree_ok_status();
    }
    loom_op_t* slice_op = loom_use_user_op(uses[i]);
    if (!loom_low_cfg_tuple_match_slice(plan, component_type, component_count,
                                        slice_op, &slot_plan->slice_uses[i])) {
      return iree_ok_status();
    }
  }

  loom_type_t* component_types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, component_count,
                                                 sizeof(*component_types),
                                                 (void**)&component_types));
  for (uint16_t i = 0; i < component_count; ++i) {
    component_types[i] = component_type;
  }
  *out_schema = (loom_boundary_projection_schema_t){
      .rule = rule,
      .component_types = component_types,
      .component_count = component_count,
      .destination_mode = LOOM_BOUNDARY_PROJECTION_DESTINATION_ELIMINATE,
      .rule_plan = slot_plan,
  };
  *out_claimed = true;
  return iree_ok_status();
}

static void loom_low_cfg_tuple_refine_concat_component(
    const loom_boundary_projection_plan_t* plan, loom_value_id_t value_id,
    uint16_t component, loom_low_cfg_tuple_source_component_t* out_source) {
  const loom_value_t* value = loom_module_value(plan->module, value_id);
  if (loom_value_is_block_arg(value)) {
    return;
  }
  loom_op_t* op = loom_value_def_op(value);
  if (!loom_low_concat_isa(op) || loom_low_concat_result(op) != value_id) {
    return;
  }

  uint32_t source_begin = 0;
  const loom_value_slice_t sources = loom_low_concat_sources(op);
  for (uint16_t i = 0; i < sources.count; ++i) {
    const loom_value_id_t source_value_id = sources.values[i];
    const loom_type_t source_type =
        loom_module_value_type(plan->module, source_value_id);
    if (!loom_type_is_register(source_type)) {
      return;
    }
    const uint32_t source_component_count =
        loom_low_register_type_unit_count(source_type);
    if (source_component_count > UINT32_MAX - source_begin) {
      return;
    }
    if (component >= source_begin &&
        component < source_begin + source_component_count) {
      out_source->kind = LOOM_LOW_CFG_TUPLE_SOURCE_SLICE_CONCAT_OPERAND;
      out_source->source.concat_operand.op = op;
      out_source->source.concat_operand.operand_index = i;
      out_source->source_offset = component - source_begin;
      return;
    }
    source_begin += source_component_count;
  }
}

static iree_status_t loom_low_cfg_tuple_plan_source(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_slot_t* destination,
    const loom_boundary_projection_schema_t* schema,
    loom_value_id_t source_value_id, loom_op_t* boundary_op,
    loom_boundary_projection_source_t* out_source, bool* out_planned) {
  (void)function;
  *out_source = (loom_boundary_projection_source_t){0};
  *out_planned = false;
  IREE_ASSERT(destination != NULL);
  IREE_ASSERT(schema->rule == rule);
  if (!loom_type_equal(loom_module_value_type(plan->module, source_value_id),
                       destination->logical_type)) {
    return iree_ok_status();
  }

  loom_low_cfg_tuple_source_plan_t* source_plan = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(plan->arena, sizeof(*source_plan),
                                           (void**)&source_plan));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, schema->component_count, sizeof(*source_plan->components),
      (void**)&source_plan->components));
  source_plan->component_count = schema->component_count;

  const loom_type_t component_type = schema->component_types[0];
  source_plan->component_type = component_type;
  for (uint16_t component = 0; component < schema->component_count;
       ++component) {
    loom_low_cfg_tuple_source_component_t* source =
        &source_plan->components[component];
    *source = (loom_low_cfg_tuple_source_component_t){
        .kind = LOOM_LOW_CFG_TUPLE_SOURCE_SLICE_VALUE,
        .source.value_id = source_value_id,
        .source_offset = component,
    };
    loom_low_cfg_tuple_refine_concat_component(plan, source_value_id, component,
                                               source);
  }

  *out_source = (loom_boundary_projection_source_t){
      .rule = rule,
      .rule_plan = source_plan,
      .boundary_op = boundary_op,
  };
  *out_planned = true;
  return iree_ok_status();
}

static iree_status_t loom_low_cfg_tuple_materialize_source(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_source_t* source,
    loom_value_id_t* out_component_values) {
  (void)function;
  IREE_ASSERT(source->rule == rule);
  IREE_ASSERT(source->boundary_op != NULL);
  const loom_low_cfg_tuple_source_plan_t* source_plan =
      (const loom_low_cfg_tuple_source_plan_t*)source->rule_plan;
  IREE_ASSERT(source_plan != NULL);
  for (uint16_t component = 0; component < source_plan->component_count;
       ++component) {
    const loom_low_cfg_tuple_source_component_t* source_component =
        &source_plan->components[component];
    loom_value_id_t source_value_id = LOOM_VALUE_ID_INVALID;
    if (source_component->kind == LOOM_LOW_CFG_TUPLE_SOURCE_SLICE_VALUE) {
      source_value_id = source_component->source.value_id;
    } else {
      const loom_value_slice_t sources =
          loom_low_concat_sources(source_component->source.concat_operand.op);
      IREE_ASSERT_LT(source_component->source.concat_operand.operand_index,
                     sources.count);
      source_value_id =
          sources.values[source_component->source.concat_operand.operand_index];
    }
    // The destination block argument supplies independent ownership. A stable
    // whole component needs no intermediate capture on the incoming edge.
    if (source_component->source_offset == 0 &&
        loom_type_equal(loom_module_value_type(plan->module, source_value_id),
                        source_plan->component_type) &&
        loom_low_capture_source_is_stable(plan->module, source_value_id)) {
      out_component_values[component] = source_value_id;
      continue;
    }
    loom_builder_ip_t saved_ip = loom_builder_save(&plan->rewriter.builder);
    loom_op_t* capture_op =
        source_component->kind == LOOM_LOW_CFG_TUPLE_SOURCE_SLICE_VALUE
            ? source->boundary_op
            : source_component->source.concat_operand.op;
    loom_builder_set_before(&plan->rewriter.builder, capture_op);
    loom_op_t* slice_op = NULL;
    iree_status_t status = loom_low_slice_build(
        &plan->rewriter.builder, source_value_id,
        source_component->source_offset, source_plan->component_type,
        source->boundary_op->location, &slice_op);
    loom_builder_restore(&plan->rewriter.builder, saved_ip);
    if (iree_status_is_ok(status)) {
      out_component_values[component] = loom_low_slice_result(slice_op);
    }
    IREE_RETURN_IF_ERROR(status);
  }
  return iree_ok_status();
}

static iree_status_t loom_low_cfg_tuple_eliminate(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_t* slot,
    loom_boundary_projection_elimination_t* out_elimination) {
  (void)out_elimination;
  (void)function;
  IREE_ASSERT(slot->schema.rule == rule);
  const loom_low_cfg_tuple_slot_plan_t* slot_plan =
      (const loom_low_cfg_tuple_slot_plan_t*)slot->schema.rule_plan;
  IREE_ASSERT(slot_plan != NULL);
  // Publish every component observation before testing any owner merge. A
  // later slice can consume its capture independently of an earlier observer.
  for (uint32_t i = 0; i < slot_plan->slice_use_count; ++i) {
    const loom_low_cfg_tuple_slice_use_t* slice_use = &slot_plan->slice_uses[i];
    loom_value_id_t replacement =
        slot->component_value_ids[slice_use->first_component];
    if (slice_use->component_count > 1) {
      // Reconstruct only the observed range at its original capture point.
      // The components can retain independent lifetimes across CFG edges.
      loom_builder_ip_t saved_ip = loom_builder_save(&plan->rewriter.builder);
      loom_builder_set_before(&plan->rewriter.builder, slice_use->op);
      loom_op_t* concat_op = NULL;
      iree_status_t status = loom_low_concat_build(
          &plan->rewriter.builder,
          &slot->component_value_ids[slice_use->first_component],
          slice_use->component_count,
          loom_module_value_type(plan->module,
                                 loom_low_slice_result(slice_use->op)),
          slice_use->op->location, &concat_op);
      loom_builder_restore(&plan->rewriter.builder, saved_ip);
      IREE_RETURN_IF_ERROR(status);
      replacement = loom_low_concat_result(concat_op);
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_set_attr(&plan->rewriter, slice_use->op,
                                                0, loom_attr_i64(0)));
    IREE_RETURN_IF_ERROR(loom_rewriter_set_operand(
        &plan->rewriter, slice_use->op, 0, replacement));
  }
  uint32_t removed_count = 0;
  for (uint32_t i = 0; i < slot_plan->slice_use_count; ++i) {
    loom_op_t* slice_op = slot_plan->slice_uses[i].op;
    const loom_value_id_t source = loom_low_slice_source(slice_op);
    if (loom_low_capture_can_forward(plan->module, source,
                                     loom_low_slice_result(slice_op), 1)) {
      IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
          &plan->rewriter, slice_op, &source, 1));
      ++removed_count;
    }
  }
  loom_boundary_projection_record(plan, rule, 1, slot->schema.component_count);
  loom_boundary_projection_record_destination_uses(plan, rule, removed_count);
  return iree_ok_status();
}

static const loom_boundary_projection_rule_t kLowCfgTupleProjectionRule = {
    .name = IREE_SVL("low-register-tuple"),
    .type_kind_bits =
        LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(LOOM_TYPE_REGISTER),
    .slot_role_bits = LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
        LOOM_BOUNDARY_PROJECTION_SLOT_BLOCK_ARGUMENT),
    .function_applies = loom_low_cfg_tuple_function_applies,
    .slot_matches = loom_low_cfg_tuple_slot_matches,
    .plan_slot = loom_low_cfg_tuple_plan_slot,
    .transport =
        {
            .plan_source = loom_low_cfg_tuple_plan_source,
            .materialize_source = loom_low_cfg_tuple_materialize_source,
            .eliminate = loom_low_cfg_tuple_eliminate,
        },
};

const loom_boundary_projection_rule_t*
loom_low_cfg_tuple_boundary_projection_rule(void) {
  return &kLowCfgTupleProjectionRule;
}
