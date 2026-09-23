// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/encoding/cfg_layout_projection.h"

#include <inttypes.h>
#include <string.h>

#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/index/ops.h"
#include "loom/transforms/boundary/projection_plan.h"

typedef struct loom_cfg_layout_projection_slot_plan_t {
  // Static stride or INT64_MIN for each transported dynamic axis.
  int64_t static_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  // Layout axis supplied by each physical component.
  uint8_t component_axes[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  // Layout rank.
  uint8_t rank;
  // Number of scalar stride components.
  uint8_t component_count;
} loom_cfg_layout_projection_slot_plan_t;

typedef enum loom_cfg_layout_projection_component_kind_e {
  LOOM_CFG_LAYOUT_PROJECTION_COMPONENT_VALUE = 0,
  LOOM_CFG_LAYOUT_PROJECTION_COMPONENT_CONSTANT = 1,
  LOOM_CFG_LAYOUT_PROJECTION_COMPONENT_SLOT = 2,
} loom_cfg_layout_projection_component_kind_t;

typedef struct loom_cfg_layout_projection_component_t {
  // How the scalar component is obtained at the outgoing boundary.
  loom_cfg_layout_projection_component_kind_t kind;
  union {
    // Existing index SSA value for VALUE.
    loom_value_id_t value_id;
    // Exact nonnegative element stride for CONSTANT.
    int64_t constant;
    struct {
      // Projected slot supplying the future scalar value.
      iree_host_size_t slot;
      // Physical component within slot.
      uint8_t component;
    } slot;
  } value;
} loom_cfg_layout_projection_component_t;

typedef struct loom_cfg_layout_projection_source_plan_t {
  // One recipe per destination component.
  loom_cfg_layout_projection_component_t* components;
  // Number of entries in components.
  uint8_t component_count;
} loom_cfg_layout_projection_source_plan_t;

static iree_status_t loom_cfg_layout_projection_plan_slot(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block, loom_boundary_projection_schema_t* out_schema,
    bool* out_claimed) {
  *out_schema = (loom_boundary_projection_schema_t){0};
  *out_claimed = false;
  if (role != LOOM_BOUNDARY_PROJECTION_SLOT_BLOCK_ARGUMENT || !block ||
      !function->facts) {
    return iree_ok_status();
  }
  loom_region_t* body = loom_func_like_body(function->function);
  if (!body || block == loom_region_entry_block(body)) {
    return iree_ok_status();
  }

  const loom_type_t type = loom_module_value_type(plan->module, value_id);
  if (!loom_type_is_encoding(type) ||
      loom_type_encoding_role(type) != LOOM_ENCODING_ROLE_ADDRESS_LAYOUT) {
    return iree_ok_status();
  }
  loom_value_fact_address_layout_t layout = {0};
  if (!loom_encoding_query_value_address_layout(&function->facts->context,
                                                value_id, &layout) ||
      layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED ||
      layout.rank == 0 || layout.rank > LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK ||
      !layout.strides) {
    return iree_ok_status();
  }

  loom_cfg_layout_projection_slot_plan_t* slot_plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(plan->arena, sizeof(*slot_plan), (void**)&slot_plan));
  memset(slot_plan, 0, sizeof(*slot_plan));
  slot_plan->rank = layout.rank;
  for (uint8_t axis = 0; axis < layout.rank; ++axis) {
    int64_t exact_stride = 0;
    if (loom_value_facts_as_exact_i64(layout.strides[axis], &exact_stride)) {
      if (exact_stride < 0) {
        return iree_ok_status();
      }
      slot_plan->static_strides[axis] = exact_stride;
    } else {
      slot_plan->static_strides[axis] = INT64_MIN;
      slot_plan->component_axes[slot_plan->component_count++] = axis;
    }
  }
  if (slot_plan->component_count == 0) {
    return iree_ok_status();
  }

  loom_type_t* component_types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, slot_plan->component_count, sizeof(*component_types),
      (void**)&component_types));
  iree_string_view_t* component_name_suffixes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, slot_plan->component_count, sizeof(*component_name_suffixes),
      (void**)&component_name_suffixes));
  char* suffix_storage = NULL;
  const iree_host_size_t suffix_capacity = 16;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      plan->arena, slot_plan->component_count * suffix_capacity,
      (void**)&suffix_storage));
  for (uint8_t component = 0; component < slot_plan->component_count;
       ++component) {
    component_types[component] = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    char* suffix = suffix_storage + component * suffix_capacity;
    const int length = iree_snprintf(suffix, suffix_capacity, "stride_%" PRIu8,
                                     slot_plan->component_axes[component]);
    IREE_ASSERT(length > 0 && (iree_host_size_t)length < suffix_capacity);
    component_name_suffixes[component] =
        iree_make_string_view(suffix, (iree_host_size_t)length);
  }

  *out_schema = (loom_boundary_projection_schema_t){
      .rule = rule,
      .component_types = component_types,
      .component_name_suffixes = component_name_suffixes,
      .component_count = slot_plan->component_count,
      .rule_plan = slot_plan,
  };
  *out_claimed = true;
  return iree_ok_status();
}

static bool loom_cfg_layout_projection_direct_component(
    const loom_boundary_projection_function_t* function,
    loom_value_id_t source_value_id,
    const loom_cfg_layout_projection_slot_plan_t* destination_plan,
    uint8_t axis, loom_cfg_layout_projection_component_t* out_component) {
  loom_value_fact_address_layout_t layout = {0};
  if (!loom_encoding_query_value_address_layout(&function->facts->context,
                                                source_value_id, &layout) ||
      layout.kind != LOOM_VALUE_FACT_ADDRESS_LAYOUT_STRIDED ||
      layout.rank != destination_plan->rank || !layout.strides) {
    return false;
  }
  int64_t exact_stride = 0;
  if (loom_value_facts_as_exact_i64(layout.strides[axis], &exact_stride) &&
      exact_stride >= 0) {
    *out_component = (loom_cfg_layout_projection_component_t){
        .kind = LOOM_CFG_LAYOUT_PROJECTION_COMPONENT_CONSTANT,
        .value.constant = exact_stride,
    };
    return true;
  }
  const loom_value_fact_layout_strides_t bindings =
      loom_encoding_query_value_layout_strides(&function->facts->context,
                                               source_value_id);
  if (bindings.count != destination_plan->rank ||
      bindings.values[axis] == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  *out_component = (loom_cfg_layout_projection_component_t){
      .kind = LOOM_CFG_LAYOUT_PROJECTION_COMPONENT_VALUE,
      .value.value_id = bindings.values[axis],
  };
  return true;
}

static bool loom_cfg_layout_projection_component_for_axis(
    const loom_cfg_layout_projection_slot_plan_t* slot_plan, uint8_t axis,
    uint8_t* out_component) {
  for (uint8_t component = 0; component < slot_plan->component_count;
       ++component) {
    if (slot_plan->component_axes[component] == axis) {
      *out_component = component;
      return true;
    }
  }
  return false;
}

static iree_status_t loom_cfg_layout_projection_plan_source(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_slot_t* destination,
    const loom_boundary_projection_schema_t* schema,
    loom_value_id_t source_value_id, loom_op_t* boundary_op,
    loom_boundary_projection_source_t* out_source, bool* out_planned) {
  *out_source = (loom_boundary_projection_source_t){0};
  *out_planned = false;
  IREE_ASSERT(destination != NULL);
  IREE_ASSERT(schema->rule == rule);
  const loom_cfg_layout_projection_slot_plan_t* destination_plan =
      (const loom_cfg_layout_projection_slot_plan_t*)schema->rule_plan;
  IREE_ASSERT(destination_plan != NULL);
  IREE_ASSERT_EQ(destination_plan->component_count, schema->component_count);

  loom_cfg_layout_projection_source_plan_t* source_plan = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(plan->arena, sizeof(*source_plan),
                                           (void**)&source_plan));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, schema->component_count, sizeof(*source_plan->components),
      (void**)&source_plan->components));
  source_plan->component_count = (uint8_t)schema->component_count;

  const iree_host_size_t destination_index =
      loom_boundary_projection_slot_index(function, destination->value_id);
  IREE_ASSERT_NE(destination_index, IREE_HOST_SIZE_MAX);
  for (uint8_t component = 0; component < source_plan->component_count;
       ++component) {
    const uint8_t axis = destination_plan->component_axes[component];
    if (loom_cfg_layout_projection_direct_component(
            function, source_value_id, destination_plan, axis,
            &source_plan->components[component])) {
      continue;
    }
    const iree_host_size_t source_index =
        loom_boundary_projection_slot_index(function, source_value_id);
    if (source_index == IREE_HOST_SIZE_MAX) {
      return iree_ok_status();
    }
    const loom_boundary_projection_slot_t* source =
        &function->candidates[source_index];
    if (!source->selected || source->schema.rule != rule) {
      return iree_ok_status();
    }
    const loom_cfg_layout_projection_slot_plan_t* source_slot_plan =
        (const loom_cfg_layout_projection_slot_plan_t*)source->schema.rule_plan;
    uint8_t source_component = 0;
    if (source_slot_plan->rank != destination_plan->rank ||
        !loom_cfg_layout_projection_component_for_axis(source_slot_plan, axis,
                                                       &source_component)) {
      return iree_ok_status();
    }
    const loom_cfg_layout_projection_component_t projected_component = {
        .kind = LOOM_CFG_LAYOUT_PROJECTION_COMPONENT_SLOT,
        .value.slot =
            {
                .slot = source_index,
                .component = source_component,
            },
    };
    source_plan->components[component] = projected_component;
    IREE_RETURN_IF_ERROR(loom_boundary_projection_add_dependency(
        plan, function, source_index, destination_index,
        /*orders_reconstruction=*/false));
  }

  *out_source = (loom_boundary_projection_source_t){
      .rule = rule,
      .rule_plan = source_plan,
      .boundary_op = boundary_op,
  };
  *out_planned = true;
  return iree_ok_status();
}

static iree_status_t loom_cfg_layout_projection_materialize_source(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_source_t* source,
    loom_value_id_t* out_component_values) {
  IREE_ASSERT(source->rule == rule);
  IREE_ASSERT(source->boundary_op != NULL);
  const loom_cfg_layout_projection_source_plan_t* source_plan =
      (const loom_cfg_layout_projection_source_plan_t*)source->rule_plan;
  IREE_ASSERT(source_plan != NULL);
  for (uint8_t component = 0; component < source_plan->component_count;
       ++component) {
    const loom_cfg_layout_projection_component_t* projection =
        &source_plan->components[component];
    switch (projection->kind) {
      case LOOM_CFG_LAYOUT_PROJECTION_COMPONENT_VALUE:
        out_component_values[component] = projection->value.value_id;
        break;
      case LOOM_CFG_LAYOUT_PROJECTION_COMPONENT_CONSTANT: {
        loom_builder_ip_t saved_ip = loom_builder_save(&plan->rewriter.builder);
        loom_builder_set_before(&plan->rewriter.builder, source->boundary_op);
        loom_op_t* constant_op = NULL;
        iree_status_t status = loom_index_constant_build(
            &plan->rewriter.builder, loom_attr_i64(projection->value.constant),
            loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
            source->boundary_op->location, &constant_op);
        loom_builder_restore(&plan->rewriter.builder, saved_ip);
        IREE_RETURN_IF_ERROR(status);
        out_component_values[component] =
            loom_index_constant_result(constant_op);
        break;
      }
      case LOOM_CFG_LAYOUT_PROJECTION_COMPONENT_SLOT: {
        const loom_boundary_projection_slot_t* projected_slot =
            &function->candidates[projection->value.slot.slot];
        out_component_values[component] =
            projected_slot
                ->component_value_ids[projection->value.slot.component];
        IREE_ASSERT(out_component_values[component] != LOOM_VALUE_ID_INVALID);
        break;
      }
      default:
        IREE_ASSERT(false);
        break;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_cfg_layout_projection_reconstruct(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_t* slot, loom_type_t logical_type,
    loom_location_id_t location, loom_value_id_t* out_logical_value) {
  (void)function;
  IREE_ASSERT(slot->schema.rule == rule);
  const loom_cfg_layout_projection_slot_plan_t* slot_plan =
      (const loom_cfg_layout_projection_slot_plan_t*)slot->schema.rule_plan;
  IREE_ASSERT(slot_plan != NULL);
  loom_op_t* layout_op = NULL;
  IREE_RETURN_IF_ERROR(loom_encoding_layout_strided_build(
      &plan->rewriter.builder, slot->component_value_ids,
      slot_plan->component_count, slot_plan->static_strides, slot_plan->rank,
      logical_type, location, &layout_op));
  *out_logical_value = loom_encoding_layout_strided_result(layout_op);
  loom_boundary_projection_record(plan, rule, 1, slot_plan->component_count);
  return iree_ok_status();
}

static const loom_boundary_projection_rule_t kCfgLayoutProjectionRule = {
    .name = IREE_SVL("cfg-dynamic-strided-layout"),
    .type_kind_bits =
        LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(LOOM_TYPE_ENCODING),
    .plan_slot = loom_cfg_layout_projection_plan_slot,
    .transport =
        {
            .plan_source = loom_cfg_layout_projection_plan_source,
            .materialize_source = loom_cfg_layout_projection_materialize_source,
            .reconstruct = loom_cfg_layout_projection_reconstruct,
        },
};

const loom_boundary_projection_rule_t* loom_cfg_layout_boundary_projection_rule(
    void) {
  return &kCfgLayoutProjectionRule;
}
