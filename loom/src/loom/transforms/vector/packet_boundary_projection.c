// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/packet_boundary_projection.h"

#include <string.h>

#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/ir/module.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/contract.h"
#include "loom/target/facts.h"
#include "loom/target/function_version.h"
#include "loom/transforms/boundary/projection_plan.h"

// Keeps boundary signature growth bounded independently of authored vector
// width. Larger values retain their logical carrier and receive the ordinary
// target legality diagnostic.
#define LOOM_VECTOR_PACKET_BOUNDARY_COMPONENT_LIMIT 64u

typedef struct loom_vector_packet_boundary_slot_plan_t {
  // Physical component types in boundary order.
  const loom_type_t* component_types;
  // Number of physical components.
  uint16_t component_count;
  // True when every one-lane component is represented as a scalar.
  bool scalar_components;
} loom_vector_packet_boundary_slot_plan_t;

typedef struct loom_vector_packet_boundary_source_plan_t {
  // Destination packet schema used to decompose a logical source.
  const loom_vector_packet_boundary_slot_plan_t* slot_plan;
  // Projected slot supplying components directly, or IREE_HOST_SIZE_MAX.
  iree_host_size_t slot_index;
  // Logical source sliced when slot_index is IREE_HOST_SIZE_MAX.
  loom_value_id_t logical_value_id;
} loom_vector_packet_boundary_source_plan_t;

static const loom_target_facts_t* loom_vector_packet_boundary_target_facts(
    const loom_boundary_projection_function_t* function) {
  const loom_target_function_version_t* version =
      loom_target_function_version_const_cast(function->version);
  return version != NULL ? version->function_target_facts : NULL;
}

static void loom_vector_packet_boundary_mark_function(
    loom_boundary_projection_function_t* function) {
  loom_target_function_version_t* version =
      loom_target_function_version_cast(function->version);
  IREE_ASSERT(version != NULL);
  version->representation_flags |=
      LOOM_TARGET_FUNCTION_REPRESENTATION_FLAG_VECTOR_PACKET_BOUNDARIES;
}

static const loom_low_pass_capability_t*
loom_vector_packet_boundary_low_capability(
    const loom_boundary_projection_plan_t* plan) {
  return loom_low_pass_capability_from_pass(plan->pass);
}

static const loom_target_vector_packet_policy_t*
loom_vector_packet_boundary_policy(
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function) {
  return loom_low_pass_capability_vector_packet_policy(
      loom_vector_packet_boundary_low_capability(plan),
      loom_vector_packet_boundary_target_facts(function));
}

static bool loom_vector_packet_boundary_is_callable_role(
    loom_boundary_projection_slot_role_t role) {
  return role == LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_ARGUMENT ||
         role == LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_RESULT ||
         role == LOOM_BOUNDARY_PROJECTION_SLOT_CALL_RESULT;
}

static bool loom_vector_packet_boundary_is_object_signature_role(
    const loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role) {
  if (role != LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_ARGUMENT &&
      role != LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_RESULT) {
    return false;
  }
  const loom_target_bundle_t* bundle = loom_target_facts_bundle(
      loom_vector_packet_boundary_target_facts(function));
  return bundle != NULL && bundle->export_plan != NULL &&
         bundle->export_plan->abi_kind == LOOM_TARGET_ABI_OBJECT_FUNCTION;
}

static bool loom_vector_packet_boundary_carrier_supported(
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_type_t type,
    loom_value_id_t value_id) {
  const loom_low_pass_capability_t* capability =
      loom_vector_packet_boundary_low_capability(plan);
  const loom_target_facts_t* target_facts =
      loom_vector_packet_boundary_target_facts(function);
  return loom_vector_packet_boundary_is_callable_role(role)
             ? loom_low_pass_capability_source_function_vector_carrier_supported(
                   capability, target_facts, plan->module,
                   value_id != LOOM_VALUE_ID_INVALID ? function->facts : NULL,
                   value_id, type)
             : loom_low_pass_capability_source_vector_carrier_supported(
                   capability, target_facts, plan->module,
                   value_id != LOOM_VALUE_ID_INVALID ? function->facts : NULL,
                   value_id, type);
}

static loom_type_t loom_vector_packet_boundary_type(loom_type_t source_type,
                                                    uint32_t lane_count,
                                                    bool scalar_components) {
  if (scalar_components) {
    IREE_ASSERT_EQ(lane_count, 1u);
    return loom_type_scalar(loom_type_element_type(source_type));
  }
  return loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, loom_type_element_type(source_type),
      loom_dim_pack_static(lane_count), source_type.encoding_id);
}

static bool loom_vector_packet_boundary_scalar_components_enabled(
    const loom_target_vector_packet_policy_t* policy) {
  return policy != NULL &&
         iree_any_bit_set(
             policy->flags,
             LOOM_TARGET_VECTOR_PACKET_POLICY_FLAG_SCALAR_BOUNDARY_COMPONENTS);
}

static bool loom_vector_packet_boundary_select_schema(
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_type_t logical_type,
    loom_value_id_t value_id, uint32_t* out_component_lane_count,
    uint16_t* out_component_count, bool* out_scalar_components) {
  *out_component_lane_count = 0;
  *out_component_count = 0;
  *out_scalar_components = false;
  const loom_target_vector_packet_policy_t* policy =
      loom_vector_packet_boundary_policy(plan, function);
  const bool scalar_components_enabled =
      loom_vector_packet_boundary_scalar_components_enabled(policy);
  if (loom_vector_packet_boundary_is_object_signature_role(function, role) ||
      !loom_type_is_vector(logical_type) || loom_type_rank(logical_type) == 0 ||
      (!scalar_components_enabled && loom_type_rank(logical_type) != 1 &&
       logical_type.encoding_id != 0) ||
      !loom_type_is_all_static(logical_type) ||
      loom_vector_packet_boundary_carrier_supported(plan, function, role,
                                                    logical_type, value_id)) {
    return false;
  }

  uint64_t logical_lane_count = 0;
  if (!loom_type_static_element_count(logical_type, &logical_lane_count) ||
      logical_lane_count == 0 || logical_lane_count > UINT32_MAX ||
      (!scalar_components_enabled && loom_type_rank(logical_type) == 1 &&
       logical_lane_count < 2)) {
    return false;
  }
  if (policy == NULL) {
    return false;
  }

  uint32_t component_lane_count = 0;
  if (loom_type_rank(logical_type) == 1 || logical_type.encoding_id == 0) {
    for (uint8_t i = 0; i < policy->native_lane_count_count; ++i) {
      const uint32_t candidate_lane_count = policy->native_lane_counts[i];
      if (candidate_lane_count == 0 ||
          candidate_lane_count > logical_lane_count ||
          (candidate_lane_count == logical_lane_count &&
           loom_type_rank(logical_type) == 1) ||
          candidate_lane_count <= component_lane_count) {
        continue;
      }
      const loom_type_t packet_type = loom_vector_packet_boundary_type(
          logical_type, candidate_lane_count, /*scalar_components=*/false);
      if (!loom_vector_packet_boundary_carrier_supported(
              plan, function, role, packet_type, LOOM_VALUE_ID_INVALID)) {
        continue;
      }
      const uint32_t tail_lane_count =
          (uint32_t)(logical_lane_count % candidate_lane_count);
      if (tail_lane_count != 0 &&
          !loom_vector_packet_boundary_carrier_supported(
              plan, function, role,
              loom_vector_packet_boundary_type(logical_type, tail_lane_count,
                                               /*scalar_components=*/false),
              LOOM_VALUE_ID_INVALID)) {
        continue;
      }
      component_lane_count = candidate_lane_count;
    }
  }
  if (component_lane_count == 0 && scalar_components_enabled) {
    const loom_type_t scalar_type =
        loom_type_scalar(loom_type_element_type(logical_type));
    if (!loom_low_pass_capability_source_type_supported(
            loom_vector_packet_boundary_low_capability(plan),
            loom_vector_packet_boundary_target_facts(function), plan->module,
            scalar_type)) {
      return false;
    }
    component_lane_count = 1;
    *out_scalar_components = true;
  } else if (component_lane_count == 0) {
    return false;
  }
  const uint64_t component_count =
      (logical_lane_count - 1u) / component_lane_count + 1u;
  if (component_count > LOOM_VECTOR_PACKET_BOUNDARY_COMPONENT_LIMIT ||
      component_count > UINT16_MAX) {
    return false;
  }
  *out_component_lane_count = component_lane_count;
  *out_component_count = (uint16_t)component_count;
  return true;
}

static bool loom_vector_packet_boundary_function_applies(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function) {
  (void)rule;
  const loom_target_vector_packet_policy_t* policy =
      loom_vector_packet_boundary_policy(plan, function);
  return policy != NULL &&
         (policy->native_lane_count_count != 0 ||
          loom_vector_packet_boundary_scalar_components_enabled(policy)) &&
         iree_any_bit_set(
             policy->flags,
             LOOM_TARGET_VECTOR_PACKET_POLICY_FLAG_BOUNDARY_PROJECTION);
}

static bool loom_vector_packet_boundary_slot_matches(
    const loom_boundary_projection_rule_t* rule,
    const loom_boundary_projection_plan_t* plan,
    const loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block) {
  (void)rule;
  (void)block;
  uint32_t component_lane_count = 0;
  uint16_t component_count = 0;
  bool scalar_components = false;
  return loom_vector_packet_boundary_select_schema(
      plan, function, role, loom_module_value_type(plan->module, value_id),
      LOOM_VALUE_ID_INVALID, &component_lane_count, &component_count,
      &scalar_components);
}

static iree_status_t loom_vector_packet_boundary_plan_slot(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_role_t role, loom_value_id_t value_id,
    loom_block_t* block, loom_boundary_projection_schema_t* out_schema,
    bool* out_claimed) {
  (void)block;
  *out_schema = (loom_boundary_projection_schema_t){0};
  *out_claimed = false;
  const loom_type_t logical_type =
      loom_module_value_type(plan->module, value_id);
  uint32_t component_lane_count = 0;
  uint16_t component_count = 0;
  bool scalar_components = false;
  if (!loom_vector_packet_boundary_select_schema(
          plan, function, role, logical_type, value_id, &component_lane_count,
          &component_count, &scalar_components)) {
    return iree_ok_status();
  }

  loom_vector_packet_boundary_slot_plan_t* slot_plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(plan->arena, sizeof(*slot_plan), (void**)&slot_plan));
  loom_type_t* component_types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(plan->arena, component_count,
                                                 sizeof(*component_types),
                                                 (void**)&component_types));
  uint64_t remaining_lane_count = 0;
  const bool has_static_lane_count =
      loom_type_static_element_count(logical_type, &remaining_lane_count);
  IREE_ASSERT(has_static_lane_count);
  (void)has_static_lane_count;
  for (uint16_t i = 0; i < component_count; ++i) {
    const uint32_t lane_count =
        (uint32_t)iree_min(remaining_lane_count, component_lane_count);
    component_types[i] = loom_vector_packet_boundary_type(
        logical_type, lane_count, scalar_components);
    remaining_lane_count -= lane_count;
  }
  *slot_plan = (loom_vector_packet_boundary_slot_plan_t){
      .component_types = component_types,
      .component_count = component_count,
      .scalar_components = scalar_components,
  };
  *out_schema = (loom_boundary_projection_schema_t){
      .rule = rule,
      .component_types = component_types,
      .component_count = component_count,
      .rule_plan = slot_plan,
  };
  *out_claimed = true;
  return iree_ok_status();
}

static bool loom_vector_packet_boundary_schemas_equal(
    const loom_boundary_projection_schema_t* lhs,
    const loom_boundary_projection_schema_t* rhs) {
  if (lhs->component_count != rhs->component_count) {
    return false;
  }
  for (uint16_t i = 0; i < lhs->component_count; ++i) {
    if (!loom_type_equal(lhs->component_types[i], rhs->component_types[i])) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_vector_packet_boundary_plan_source(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_slot_t* destination,
    const loom_boundary_projection_schema_t* schema,
    loom_value_id_t source_value_id, loom_op_t* boundary_op,
    loom_boundary_projection_source_t* out_source, bool* out_planned) {
  *out_source = (loom_boundary_projection_source_t){0};
  *out_planned = false;
  IREE_ASSERT(schema->rule == rule);
  const loom_vector_packet_boundary_slot_plan_t* slot_plan =
      (const loom_vector_packet_boundary_slot_plan_t*)schema->rule_plan;
  IREE_ASSERT(slot_plan != NULL);

  loom_vector_packet_boundary_source_plan_t* source_plan = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(plan->arena, sizeof(*source_plan),
                                           (void**)&source_plan));
  *source_plan = (loom_vector_packet_boundary_source_plan_t){
      .slot_plan = slot_plan,
      .slot_index = IREE_HOST_SIZE_MAX,
      .logical_value_id = source_value_id,
  };
  const iree_host_size_t source_index =
      loom_boundary_projection_slot_index(function, source_value_id);
  if (source_index != IREE_HOST_SIZE_MAX) {
    const loom_boundary_projection_slot_t* source_slot =
        &function->candidates[source_index];
    if (source_slot->selected && source_slot->schema.rule == rule) {
      const iree_host_size_t destination_index =
          destination != NULL
              ? (iree_host_size_t)(destination - function->candidates)
              : IREE_HOST_SIZE_MAX;
      if (loom_vector_packet_boundary_schemas_equal(&source_slot->schema,
                                                    schema)) {
        source_plan->slot_index = source_index;
        source_plan->logical_value_id = LOOM_VALUE_ID_INVALID;
        if (destination_index != IREE_HOST_SIZE_MAX &&
            source_index != destination_index) {
          IREE_RETURN_IF_ERROR(loom_boundary_projection_add_dependency(
              plan, function, source_index, destination_index,
              /*orders_realization=*/false));
        }
      } else if (destination_index != IREE_HOST_SIZE_MAX &&
                 source_index != destination_index) {
        IREE_RETURN_IF_ERROR(loom_boundary_projection_add_dependency(
            plan, function, source_index, destination_index,
            /*orders_realization=*/true));
      }
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

// Ordinary static shapes flatten in row-major lane order. The schema selector
// leaves opaque carriers and encoded multi-rank values with their owning
// target.
static loom_type_t loom_vector_packet_boundary_flat_type(loom_type_t type) {
  if (loom_type_rank(type) == 1) {
    return type;
  }
  uint64_t lane_count = 0;
  const bool is_static = loom_type_static_element_count(type, &lane_count);
  IREE_ASSERT(is_static);
  (void)is_static;
  return loom_vector_packet_boundary_type(type, (uint32_t)lane_count,
                                          /*scalar_components=*/false);
}

static iree_status_t loom_vector_packet_boundary_materialize_source(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    const loom_boundary_projection_source_t* source,
    loom_value_id_t* out_component_values) {
  IREE_ASSERT(source->rule == rule);
  loom_vector_packet_boundary_mark_function(function);
  const loom_vector_packet_boundary_source_plan_t* source_plan =
      (const loom_vector_packet_boundary_source_plan_t*)source->rule_plan;
  IREE_ASSERT(source_plan != NULL);
  if (source_plan->slot_index != IREE_HOST_SIZE_MAX) {
    const loom_boundary_projection_slot_t* source_slot =
        &function->candidates[source_plan->slot_index];
    memcpy(out_component_values, source_slot->component_value_ids,
           source_plan->slot_plan->component_count *
               sizeof(*out_component_values));
    return iree_ok_status();
  }

  const loom_value_id_t logical_value = loom_boundary_projection_resolve_value(
      function, source_plan->logical_value_id);
  if (source_plan->slot_plan->scalar_components) {
    const loom_type_t logical_type =
        loom_module_value_type(plan->module, logical_value);
    const uint8_t rank = loom_type_rank(logical_type);
    int64_t indices[LOOM_TYPE_MAX_RANK];
    for (uint16_t i = 0; i < source_plan->slot_plan->component_count; ++i) {
      uint64_t remaining_ordinal = i;
      for (uint8_t reverse_axis = 0; reverse_axis < rank; ++reverse_axis) {
        const uint8_t axis = rank - reverse_axis - 1u;
        const int64_t dimension =
            loom_type_dim_static_size_at(logical_type, axis);
        indices[axis] = (int64_t)(remaining_ordinal % (uint64_t)dimension);
        remaining_ordinal /= (uint64_t)dimension;
      }
      loom_op_t* extract_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_extract_build(
          &plan->rewriter.builder, logical_value,
          /*indices=*/NULL, /*indices_count=*/0, indices, rank,
          source_plan->slot_plan->component_types[i],
          source->boundary_op->location, &extract_op));
      out_component_values[i] = loom_vector_extract_result(extract_op);
    }
    return iree_ok_status();
  }
  loom_value_id_t flat_value = logical_value;
  const loom_type_t logical_type =
      loom_module_value_type(plan->module, logical_value);
  const loom_type_t flat_type =
      loom_vector_packet_boundary_flat_type(logical_type);
  if (!loom_type_equal(logical_type, flat_type)) {
    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
        &plan->rewriter.builder, logical_value, logical_type, flat_type,
        source->boundary_op->location, &bitcast_op));
    flat_value = loom_vector_bitcast_result(bitcast_op);
  }
  uint32_t lane_offset = 0;
  for (uint16_t i = 0; i < source_plan->slot_plan->component_count; ++i) {
    const int64_t static_offset = lane_offset;
    loom_op_t* slice_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_slice_build(
        &plan->rewriter.builder, flat_value,
        /*offsets=*/NULL, /*offsets_count=*/0, &static_offset,
        /*static_offsets_count=*/1, source_plan->slot_plan->component_types[i],
        source->boundary_op->location, &slice_op));
    out_component_values[i] = loom_vector_slice_result(slice_op);
    uint64_t component_lane_count = 0;
    const bool has_static_lane_count = loom_type_static_element_count(
        source_plan->slot_plan->component_types[i], &component_lane_count);
    IREE_ASSERT(has_static_lane_count);
    (void)has_static_lane_count;
    lane_offset += (uint32_t)component_lane_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_boundary_reconstruct(
    const loom_boundary_projection_rule_t* rule,
    loom_boundary_projection_plan_t* plan,
    loom_boundary_projection_function_t* function,
    loom_boundary_projection_slot_t* slot, loom_type_t logical_type,
    loom_location_id_t location, loom_value_id_t* out_logical_value) {
  IREE_ASSERT(slot->schema.rule == rule);
  loom_vector_packet_boundary_mark_function(function);
  const loom_vector_packet_boundary_slot_plan_t* slot_plan =
      (const loom_vector_packet_boundary_slot_plan_t*)slot->schema.rule_plan;
  IREE_ASSERT(slot_plan != NULL);
  if (slot_plan->scalar_components) {
    loom_op_t* from_elements_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
        &plan->rewriter.builder, slot->component_value_ids,
        slot->schema.component_count, logical_type, location,
        &from_elements_op));
    *out_logical_value = loom_vector_from_elements_result(from_elements_op);
    loom_boundary_projection_record(plan, rule, 1,
                                    slot->schema.component_count);
    return iree_ok_status();
  }
  const loom_type_t flat_type =
      loom_vector_packet_boundary_flat_type(logical_type);
  loom_value_id_t flat_value = slot->component_value_ids[0];
  if (slot->schema.component_count > 1) {
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_concat_build(
        &plan->rewriter.builder, /*axis=*/0, slot->component_value_ids,
        slot->schema.component_count, flat_type, location, &concat_op));
    flat_value = loom_vector_concat_result(concat_op);
  }
  *out_logical_value = flat_value;
  if (!loom_type_equal(logical_type, flat_type)) {
    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
        &plan->rewriter.builder, flat_value, flat_type, logical_type, location,
        &bitcast_op));
    *out_logical_value = loom_vector_bitcast_result(bitcast_op);
  }
  loom_boundary_projection_record(plan, rule, 1, slot->schema.component_count);
  return iree_ok_status();
}

static const loom_boundary_projection_rule_t
    kVectorPacketBoundaryProjectionRule = {
        .name = IREE_SVL("vector-packet-carrier"),
        .type_kind_bits =
            LOOM_BOUNDARY_PROJECTION_TYPE_KIND_BIT(LOOM_TYPE_VECTOR),
        .slot_role_bits = LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
                              LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_ARGUMENT) |
                          LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
                              LOOM_BOUNDARY_PROJECTION_SLOT_FUNCTION_RESULT) |
                          LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
                              LOOM_BOUNDARY_PROJECTION_SLOT_BLOCK_ARGUMENT) |
                          LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
                              LOOM_BOUNDARY_PROJECTION_SLOT_CALL_RESULT) |
                          LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
                              LOOM_BOUNDARY_PROJECTION_SLOT_LOOP_STATE) |
                          LOOM_BOUNDARY_PROJECTION_SLOT_ROLE_BIT(
                              LOOM_BOUNDARY_PROJECTION_SLOT_BRANCH_RESULT),
        .function_applies = loom_vector_packet_boundary_function_applies,
        .slot_matches = loom_vector_packet_boundary_slot_matches,
        .plan_slot = loom_vector_packet_boundary_plan_slot,
        .transport =
            {
                .plan_source = loom_vector_packet_boundary_plan_source,
                .materialize_source =
                    loom_vector_packet_boundary_materialize_source,
                .reconstruct = loom_vector_packet_boundary_reconstruct,
            },
};

const loom_boundary_projection_rule_t*
loom_vector_packet_boundary_projection_rule(void) {
  return &kVectorPacketBoundaryProjectionRule;
}
