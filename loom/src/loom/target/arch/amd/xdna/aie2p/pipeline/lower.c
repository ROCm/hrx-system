// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/lower.h"

#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/analysis/pipeline_plan.h"
#include "loom/codegen/low/builder.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/array/plan.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/array_descriptors.h"
#include "loom/target/arch/amd/xdna/array/facts.h"

enum { LOOM_AIE2P_PIPELINE_DEFAULT_CHANNEL_CAPACITY = 2 };

static_assert((int)LOOM_PIPELINE_BINDING_ACCESS_FLAG_READ ==
                  (int)LOOM_AIE2P_ARRAY_BINDING_ACCESS_READ,
              "pipeline and AIE2P read access must have identical encodings");
static_assert((int)LOOM_PIPELINE_BINDING_ACCESS_FLAG_WRITE ==
                  (int)LOOM_AIE2P_ARRAY_BINDING_ACCESS_WRITE,
              "pipeline and AIE2P write access must have identical encodings");

typedef struct loom_aie2p_pipeline_placement_t {
  // Immutable target-independent concrete pipeline plan.
  const loom_pipeline_plan_t* plan;

  // Immutable NPU2 physical-array facts.
  const loom_xdna_array_family_t* family;

  // Legal compute coordinates in deterministic column-major order.
  loom_xdna_tile_coordinate_t* compute_coordinates;

  // Number of legal compute coordinates.
  uint32_t compute_coordinate_count;

  // Physical coordinate selected for each resident instance.
  loom_xdna_tile_coordinate_t* instance_coordinates;

  // Scratch arena owning placement storage.
  iree_arena_allocator_t* arena;
} loom_aie2p_pipeline_placement_t;

static iree_status_t loom_aie2p_pipeline_allocate_array(
    iree_arena_allocator_t* arena, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) return iree_ok_status();
  return iree_arena_allocate_array(arena, count, element_size, out_ptr);
}

static iree_status_t loom_aie2p_pipeline_placement_initialize(
    const loom_pipeline_plan_t* plan, iree_arena_allocator_t* arena,
    loom_aie2p_pipeline_placement_t* out_placement) {
  const loom_xdna_array_family_t* family = loom_xdna_npu2_array_family();
  iree_host_size_t physical_coordinate_count = 0;
  if (!iree_host_size_checked_mul(family->column_count, family->row_count,
                                  &physical_coordinate_count) ||
      physical_coordinate_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "AIE2P physical coordinate domain is too large");
  }

  *out_placement = (loom_aie2p_pipeline_placement_t){
      .plan = plan,
      .family = family,
      .arena = arena,
  };
  loom_aie2p_pipeline_placement_t* placement = out_placement;
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_allocate_array(
      arena, physical_coordinate_count, sizeof(*placement->compute_coordinates),
      (void**)&placement->compute_coordinates));
  for (uint16_t column = 0; column < family->column_count; ++column) {
    for (uint16_t row = 0; row < family->row_count; ++row) {
      const loom_xdna_tile_coordinate_t coordinate = {
          .column = column,
          .row = row,
      };
      const loom_xdna_tile_facts_t* tile = NULL;
      IREE_RETURN_IF_ERROR(
          loom_xdna_array_tile_facts(family, coordinate, &tile));
      if (tile->kind == LOOM_XDNA_TILE_KIND_COMPUTE) {
        placement->compute_coordinates[placement->compute_coordinate_count++] =
            coordinate;
      }
    }
  }
  if (placement->compute_coordinate_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "AIE2P target has no compute tiles");
  }
  if (plan->instance_count > placement->compute_coordinate_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P pipeline requires %u resident instances but has %u compute "
        "tiles",
        plan->instance_count, placement->compute_coordinate_count);
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_allocate_array(
      arena, plan->instance_count, sizeof(*placement->instance_coordinates),
      (void**)&placement->instance_coordinates));
  for (uint32_t i = 0; i < plan->instance_count; ++i) {
    placement->instance_coordinates[i] = (loom_xdna_tile_coordinate_t){
        .column = UINT16_MAX,
        .row = UINT16_MAX,
    };
  }
  return iree_ok_status();
}

static bool loom_aie2p_pipeline_edge_is_internal(
    const loom_pipeline_plan_edge_t* edge) {
  return edge->source_kind == LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE &&
         edge->target_kind == LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE;
}

static uint32_t loom_aie2p_pipeline_instance_degree(
    const loom_aie2p_pipeline_placement_t* placement, uint32_t instance_index,
    uint32_t* out_assigned_neighbor_count) {
  uint32_t degree = 0;
  uint32_t assigned_neighbor_count = 0;
  for (uint32_t i = 0; i < placement->plan->edge_count; ++i) {
    const loom_pipeline_plan_edge_t* edge = &placement->plan->edges[i];
    if (!loom_aie2p_pipeline_edge_is_internal(edge)) continue;
    uint32_t other_index = UINT32_MAX;
    if (edge->source_index == instance_index) {
      other_index = edge->target_index;
    } else if (edge->target_index == instance_index) {
      other_index = edge->source_index;
    } else {
      continue;
    }
    ++degree;
    if (placement->instance_coordinates[other_index].column != UINT16_MAX) {
      ++assigned_neighbor_count;
    }
  }
  *out_assigned_neighbor_count = assigned_neighbor_count;
  return degree;
}

static bool loom_aie2p_pipeline_coordinate_compatible(
    const loom_aie2p_pipeline_placement_t* placement, uint32_t instance_index,
    loom_xdna_tile_coordinate_t coordinate) {
  for (uint32_t i = 0; i < placement->plan->edge_count; ++i) {
    const loom_pipeline_plan_edge_t* edge = &placement->plan->edges[i];
    if (!loom_aie2p_pipeline_edge_is_internal(edge)) continue;
    uint32_t other_index = UINT32_MAX;
    if (edge->source_index == instance_index) {
      other_index = edge->target_index;
    } else if (edge->target_index == instance_index) {
      other_index = edge->source_index;
    } else {
      continue;
    }
    if (other_index == instance_index) return false;
    const loom_xdna_tile_coordinate_t other =
        placement->instance_coordinates[other_index];
    if (other.column == UINT16_MAX) continue;
    const int row_delta = (int)coordinate.row - (int)other.row;
    if (coordinate.column != other.column ||
        (row_delta != -1 && row_delta != 1)) {
      return false;
    }
  }
  return true;
}

static bool loom_aie2p_pipeline_place_next(
    loom_aie2p_pipeline_placement_t* placement, bool* coordinate_used,
    uint32_t placed_count) {
  if (placed_count == placement->plan->instance_count) return true;

  uint32_t selected_instance = UINT32_MAX;
  uint32_t selected_assigned_neighbors = 0;
  uint32_t selected_degree = 0;
  for (uint32_t instance_index = 0;
       instance_index < placement->plan->instance_count; ++instance_index) {
    if (placement->instance_coordinates[instance_index].column != UINT16_MAX) {
      continue;
    }
    uint32_t assigned_neighbors = 0;
    const uint32_t degree = loom_aie2p_pipeline_instance_degree(
        placement, instance_index, &assigned_neighbors);
    if (selected_instance == UINT32_MAX ||
        assigned_neighbors > selected_assigned_neighbors ||
        (assigned_neighbors == selected_assigned_neighbors &&
         degree > selected_degree)) {
      selected_instance = instance_index;
      selected_assigned_neighbors = assigned_neighbors;
      selected_degree = degree;
    }
  }
  IREE_ASSERT_NE(selected_instance, UINT32_MAX);

  for (uint32_t coordinate_index = 0;
       coordinate_index < placement->compute_coordinate_count;
       ++coordinate_index) {
    if (coordinate_used[coordinate_index]) continue;
    const loom_xdna_tile_coordinate_t coordinate =
        placement->compute_coordinates[coordinate_index];
    if (!loom_aie2p_pipeline_coordinate_compatible(placement, selected_instance,
                                                   coordinate)) {
      continue;
    }
    coordinate_used[coordinate_index] = true;
    placement->instance_coordinates[selected_instance] = coordinate;
    if (loom_aie2p_pipeline_place_next(placement, coordinate_used,
                                       placed_count + 1)) {
      return true;
    }
    placement->instance_coordinates[selected_instance] =
        (loom_xdna_tile_coordinate_t){
            .column = UINT16_MAX,
            .row = UINT16_MAX,
        };
    coordinate_used[coordinate_index] = false;
  }
  return false;
}

static iree_status_t loom_aie2p_pipeline_place_instances(
    loom_aie2p_pipeline_placement_t* placement) {
  bool* coordinate_used = NULL;
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_allocate_array(
      placement->arena, placement->compute_coordinate_count,
      sizeof(*coordinate_used), (void**)&coordinate_used));
  memset(coordinate_used, 0,
         placement->compute_coordinate_count * sizeof(*coordinate_used));
  if (!loom_aie2p_pipeline_place_next(placement, coordinate_used, 0)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "AIE2P pipeline graph cannot be embedded in adjacent compute tiles");
  }
  return iree_ok_status();
}

typedef struct loom_aie2p_pipeline_emitter_t {
  // Mutable module receiving the array Low function.
  loom_module_t* module;

  // Parsed and placed target-independent plan.
  const loom_pipeline_plan_t* plan;

  // AIE2P physical placement for resident instances.
  const loom_aie2p_pipeline_placement_t* placement;

  // Generated AIE2P array descriptor set.
  const loom_low_descriptor_set_t* descriptor_set;

  // Low function body builder.
  loom_builder_t builder;

  // Interned immediate name `value`.
  loom_string_id_t value_attr_name;

  // Interned immediate name `ordinal`.
  loom_string_id_t ordinal_attr_name;

  // Interned immediate name `access`.
  loom_string_id_t access_attr_name;

  // Interned immediate name `entry`.
  loom_string_id_t entry_attr_name;

  // Interned immediate name `port`.
  loom_string_id_t port_attr_name;

  // Scalar register carrying index values.
  loom_type_t scalar_type;

  // Group resource register.
  loom_type_t group_type;

  // Binding resource register.
  loom_type_t binding_type;

  // Worker resource register.
  loom_type_t worker_type;

  // Emitted group registers indexed by plan group.
  loom_value_id_t* group_values;

  // Emitted worker registers indexed by plan instance.
  loom_value_id_t* instance_values;

  // Emitted binding registers indexed by launch binding.
  loom_value_id_t* binding_values;

  // Shared source endpoint for each partitioned plan flow.
  loom_value_id_t* partition_source_values;
} loom_aie2p_pipeline_emitter_t;

static iree_status_t loom_aie2p_pipeline_emitter_initialize(
    loom_module_t* module, const loom_pipeline_plan_t* plan,
    const loom_aie2p_pipeline_placement_t* placement,
    iree_arena_allocator_t* arena, loom_aie2p_pipeline_emitter_t* out_emitter) {
  *out_emitter = (loom_aie2p_pipeline_emitter_t){
      .module = module,
      .plan = plan,
      .placement = placement,
      .descriptor_set = loom_aie2p_array_descriptor_set(),
  };
  loom_aie2p_pipeline_emitter_t* emitter = out_emitter;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("value"),
                                                 &emitter->value_attr_name));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("ordinal"),
                                                 &emitter->ordinal_attr_name));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("access"),
                                                 &emitter->access_attr_name));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("entry"),
                                                 &emitter->entry_attr_name));
  IREE_RETURN_IF_ERROR(loom_module_intern_string(module, IREE_SV("port"),
                                                 &emitter->port_attr_name));
  IREE_RETURN_IF_ERROR(loom_low_build_typed_register_type(
      module, emitter->descriptor_set,
      AIE2P_ARRAY_REG_CLASS_ID_AIE2P_ARRAY_SCALAR, 1,
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &emitter->scalar_type));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      emitter->descriptor_set, AIE2P_ARRAY_REG_CLASS_ID_AIE2P_ARRAY_GROUP, 1,
      &emitter->group_type));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      emitter->descriptor_set, AIE2P_ARRAY_REG_CLASS_ID_AIE2P_ARRAY_BINDING, 1,
      &emitter->binding_type));
  IREE_RETURN_IF_ERROR(loom_low_build_register_type(
      emitter->descriptor_set, AIE2P_ARRAY_REG_CLASS_ID_AIE2P_ARRAY_WORKER, 1,
      &emitter->worker_type));

#define LOOM_AIE2P_PIPELINE_ALLOCATE_VALUES(field, count)                 \
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_allocate_array(                \
      arena, (count), sizeof(*emitter->field), (void**)&emitter->field)); \
  for (uint32_t i = 0; i < (count); ++i) {                                \
    emitter->field[i] = LOOM_VALUE_ID_INVALID;                            \
  }
  LOOM_AIE2P_PIPELINE_ALLOCATE_VALUES(group_values, plan->group_count);
  LOOM_AIE2P_PIPELINE_ALLOCATE_VALUES(instance_values, plan->instance_count);
  LOOM_AIE2P_PIPELINE_ALLOCATE_VALUES(binding_values, plan->binding_count);
  LOOM_AIE2P_PIPELINE_ALLOCATE_VALUES(partition_source_values,
                                      plan->flow_count);
#undef LOOM_AIE2P_PIPELINE_ALLOCATE_VALUES
  return iree_ok_status();
}

static const loom_low_descriptor_t* loom_aie2p_pipeline_descriptor(
    const loom_aie2p_pipeline_emitter_t* emitter, uint32_t ordinal) {
  IREE_ASSERT_LT(ordinal, emitter->descriptor_set->descriptor_count);
  return &emitter->descriptor_set->descriptors[ordinal];
}

static iree_status_t loom_aie2p_pipeline_emit_constant(
    loom_aie2p_pipeline_emitter_t* emitter, uint32_t value,
    loom_location_id_t location, loom_value_id_t* out_value) {
  const loom_named_attr_t attr = {
      .name_id = emitter->value_attr_name,
      .value = loom_attr_i64(value),
  };
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_const(
      &emitter->builder, emitter->descriptor_set,
      loom_aie2p_pipeline_descriptor(
          emitter, AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CONSTANT_U32),
      loom_make_named_attr_slice(&attr, 1), emitter->scalar_type, location,
      &op));
  *out_value = loom_op_results(op)[0];
  return iree_ok_status();
}

static iree_status_t loom_aie2p_pipeline_emit_op(
    loom_aie2p_pipeline_emitter_t* emitter, uint32_t descriptor_ordinal,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, const loom_type_t* result_type,
    loom_location_id_t location, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      &emitter->builder, emitter->descriptor_set,
      loom_aie2p_pipeline_descriptor(emitter, descriptor_ordinal), operands,
      operand_count, attrs, result_type, result_type != NULL ? 1 : 0,
      /*tied_results=*/NULL, /*tied_result_count=*/0, location, &op));
  if (out_value != NULL) {
    IREE_ASSERT(result_type != NULL);
    *out_value = loom_op_results(op)[0];
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_pipeline_emit_typed_resource_type(
    loom_aie2p_pipeline_emitter_t* emitter, uint16_t reg_class_id,
    loom_type_t tile_type, loom_type_t* out_type) {
  return loom_low_build_typed_register_type(
      emitter->module, emitter->descriptor_set, reg_class_id, 1, tile_type,
      out_type);
}

static iree_status_t loom_aie2p_pipeline_emit_groups(
    loom_aie2p_pipeline_emitter_t* emitter, loom_location_id_t location) {
  for (uint32_t i = 0; i < emitter->plan->group_count; ++i) {
    loom_value_id_t lane_count = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_constant(
        emitter, emitter->plan->groups[i].lane_count, location, &lane_count));
    IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_op(
        emitter, AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_GROUP, &lane_count, 1,
        loom_named_attr_slice_empty(), &emitter->group_type, location,
        &emitter->group_values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_pipeline_emit_instances(
    loom_aie2p_pipeline_emitter_t* emitter, loom_location_id_t location) {
  for (uint32_t i = 0; i < emitter->plan->instance_count; ++i) {
    const loom_pipeline_plan_instance_t* instance =
        &emitter->plan->instances[i];
    loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_constant(
        emitter, instance->lane, location, &lane));
    const loom_value_id_t operands[] = {
        emitter->group_values[instance->group_index],
        lane,
    };
    const loom_named_attr_t entry_attr = {
        .name_id = emitter->entry_attr_name,
        .value = loom_attr_symbol(instance->entry),
    };
    IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_op(
        emitter, AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_WORKER, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(&entry_attr, 1),
        &emitter->worker_type, location, &emitter->instance_values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_pipeline_emit_bindings(
    loom_aie2p_pipeline_emitter_t* emitter, loom_location_id_t location) {
  for (uint32_t i = 0; i < emitter->plan->binding_count; ++i) {
    const loom_pipeline_plan_binding_t* binding = &emitter->plan->bindings[i];
    if (binding->access == 0) continue;
    const loom_named_attr_t attrs[] = {
        {
            .name_id = emitter->ordinal_attr_name,
            .value = loom_attr_i64(i),
        },
        {
            .name_id = emitter->access_attr_name,
            .value = loom_attr_i64(binding->access),
        },
    };
    IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_op(
        emitter, AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_BINDING,
        /*operands=*/NULL, /*operand_count=*/0,
        loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)),
        &emitter->binding_type, location, &emitter->binding_values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_pipeline_emit_endpoint(
    loom_aie2p_pipeline_emitter_t* emitter, uint32_t descriptor_ordinal,
    loom_value_id_t owner, uint32_t port, loom_type_t tile_type,
    loom_location_id_t location, loom_value_id_t* out_endpoint) {
  const loom_named_attr_t port_attr = {
      .name_id = emitter->port_attr_name,
      .value = loom_attr_i64(port),
  };
  const uint16_t reg_class_id =
      descriptor_ordinal == AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_SENDER
          ? AIE2P_ARRAY_REG_CLASS_ID_AIE2P_ARRAY_SENDER
          : AIE2P_ARRAY_REG_CLASS_ID_AIE2P_ARRAY_RECEIVER;
  loom_type_t endpoint_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_typed_resource_type(
      emitter, reg_class_id, tile_type, &endpoint_type));
  return loom_aie2p_pipeline_emit_op(emitter, descriptor_ordinal, &owner, 1,
                                     loom_make_named_attr_slice(&port_attr, 1),
                                     &endpoint_type, location, out_endpoint);
}

static loom_value_id_t loom_aie2p_pipeline_endpoint_owner(
    const loom_aie2p_pipeline_emitter_t* emitter,
    loom_pipeline_endpoint_kind_t kind, uint32_t index) {
  return kind == LOOM_PIPELINE_ENDPOINT_KIND_BINDING
             ? emitter->binding_values[index]
             : emitter->instance_values[index];
}

static iree_status_t loom_aie2p_pipeline_emit_edge(
    loom_aie2p_pipeline_emitter_t* emitter, uint32_t edge_index,
    loom_location_id_t location) {
  const loom_pipeline_plan_edge_t* edge = &emitter->plan->edges[edge_index];
  const loom_pipeline_plan_flow_t* flow =
      &emitter->plan->flows[edge->flow_index];
  loom_value_id_t sender = LOOM_VALUE_ID_INVALID;
  if (flow->partitioned) {
    IREE_ASSERT_EQ(edge->source_kind, LOOM_PIPELINE_ENDPOINT_KIND_BINDING);
    loom_value_id_t* partition_source =
        &emitter->partition_source_values[edge->flow_index];
    if (*partition_source == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_endpoint(
          emitter, AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_SENDER,
          emitter->binding_values[edge->source_index], flow->producer_port,
          flow->partition_source_type, location, partition_source));
    }
    loom_value_id_t lane = LOOM_VALUE_ID_INVALID;
    loom_value_id_t lane_count = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_constant(
        emitter, edge->partition_lane, location, &lane));
    IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_constant(
        emitter, emitter->plan->groups[flow->group_index].lane_count, location,
        &lane_count));
    const loom_value_id_t operands[] = {
        *partition_source,
        lane,
        lane_count,
    };
    loom_type_t sender_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_typed_resource_type(
        emitter, AIE2P_ARRAY_REG_CLASS_ID_AIE2P_ARRAY_SENDER, flow->tile_type,
        &sender_type));
    IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_op(
        emitter, AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_PARTITION, operands,
        IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &sender_type,
        location, &sender));
  } else {
    IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_endpoint(
        emitter, AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_SENDER,
        loom_aie2p_pipeline_endpoint_owner(emitter, edge->source_kind,
                                           edge->source_index),
        edge->source_port, flow->tile_type, location, &sender));
  }

  loom_value_id_t receiver = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_endpoint(
      emitter, AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_RECEIVER,
      loom_aie2p_pipeline_endpoint_owner(emitter, edge->target_kind,
                                         edge->target_index),
      edge->target_port, flow->tile_type, location, &receiver));

  const uint32_t capacity =
      flow->minimum_capacity > LOOM_AIE2P_PIPELINE_DEFAULT_CHANNEL_CAPACITY
          ? flow->minimum_capacity
          : LOOM_AIE2P_PIPELINE_DEFAULT_CHANNEL_CAPACITY;
  loom_value_id_t capacity_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_constant(
      emitter, capacity, location, &capacity_value));
  const loom_value_id_t channel_operands[] = {
      sender,
      receiver,
      capacity_value,
  };
  loom_type_t channel_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_typed_resource_type(
      emitter, AIE2P_ARRAY_REG_CLASS_ID_AIE2P_ARRAY_CHANNEL, flow->tile_type,
      &channel_type));
  loom_value_id_t channel = LOOM_VALUE_ID_INVALID;
  return loom_aie2p_pipeline_emit_op(
      emitter, AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CHANNEL, channel_operands,
      IREE_ARRAYSIZE(channel_operands), loom_named_attr_slice_empty(),
      &channel_type, location, &channel);
}

static iree_status_t loom_aie2p_pipeline_emit_edges(
    loom_aie2p_pipeline_emitter_t* emitter, loom_location_id_t location) {
  for (uint32_t i = 0; i < emitter->plan->edge_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_edge(emitter, i, location));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_pipeline_emit_locations(
    loom_aie2p_pipeline_emitter_t* emitter, loom_location_id_t location) {
  for (uint32_t i = 0; i < emitter->plan->instance_count; ++i) {
    const loom_xdna_tile_coordinate_t coordinate =
        emitter->placement->instance_coordinates[i];
    loom_value_id_t column = LOOM_VALUE_ID_INVALID;
    loom_value_id_t row = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_constant(
        emitter, coordinate.column, location, &column));
    IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_constant(
        emitter, coordinate.row, location, &row));
    const loom_value_id_t operands[] = {
        emitter->instance_values[i],
        column,
        row,
    };
    IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_op(
        emitter, AIE2P_ARRAY_DESCRIPTOR_REF_ARRAY_CONSTRAIN_LOCATION, operands,
        IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
        /*result_type=*/NULL, location, /*out_value=*/NULL));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_pipeline_create_low_function(
    loom_aie2p_pipeline_emitter_t* emitter, loom_op_t** out_low_function) {
  const loom_func_like_t pipeline = emitter->plan->pipeline;
  const loom_symbol_ref_t callee = loom_func_like_callee(pipeline);
  const loom_symbol_ref_t target = loom_func_like_target(pipeline);
  IREE_ASSERT(loom_symbol_ref_is_valid(callee));
  IREE_ASSERT(loom_symbol_ref_is_valid(target));

  loom_string_id_t descriptor_set_key = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_module_intern_string(
      emitter->module,
      loom_low_descriptor_set_string(
          emitter->descriptor_set, emitter->descriptor_set->key_string_offset),
      &descriptor_set_key));

  loom_low_func_def_build_flags_t build_flags =
      LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_TARGET |
      LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_ABI;
  uint8_t visibility = 0;
  if (loom_func_like_visibility(pipeline) != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_VISIBILITY;
    visibility = LOOM_LOW_VISIBILITY_PUBLIC;
  }
  uint8_t retain = 0;
  if (iree_any_bit_set(emitter->module->symbols.entries[callee.symbol_id].flags,
                       LOOM_SYMBOL_FLAG_RETAIN)) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_RETAIN;
    retain = LOOM_LOW_RETAIN_RETAIN;
  }
  uint16_t predicate_count = 0;
  const loom_predicate_t* predicates =
      loom_func_like_predicates(pipeline, &predicate_count);
  if (predicate_count != 0) {
    build_flags |= LOOM_LOW_FUNC_DEF_BUILD_FLAG_HAS_PREDICATES;
  }

  loom_builder_initialize(emitter->module, &emitter->module->arena,
                          loom_module_block(emitter->module),
                          &emitter->builder);
  loom_builder_set_before(&emitter->builder, pipeline.op);
  IREE_RETURN_IF_ERROR(loom_low_func_def_build(
      &emitter->builder, build_flags, visibility, retain,
      /*cc=*/0, /*purity=*/0, /*inline_policy=*/0, /*allocation=*/0,
      /*schedule=*/0, descriptor_set_key, target, LOOM_TARGET_ABI_ARRAY_PROGRAM,
      loom_named_attr_slice_empty(), loom_named_attr_slice_empty(),
      LOOM_STRING_ID_INVALID, loom_named_attr_slice_empty(), callee,
      /*arg_types=*/NULL, /*arg_types_count=*/0,
      /*result_types=*/NULL, /*result_count=*/0,
      /*tied_results=*/NULL, /*tied_result_count=*/0, predicates,
      predicate_count, pipeline.op->location, out_low_function));
  loom_builder_enter_region(&emitter->builder, *out_low_function,
                            loom_low_func_def_body(*out_low_function));
  return iree_ok_status();
}

static iree_status_t loom_aie2p_pipeline_emit_low_function(
    loom_module_t* module, const loom_pipeline_plan_t* plan,
    const loom_aie2p_pipeline_placement_t* placement,
    iree_arena_allocator_t* arena, loom_op_t** out_low_function) {
  loom_aie2p_pipeline_emitter_t emitter = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emitter_initialize(
      module, plan, placement, arena, &emitter));
  IREE_RETURN_IF_ERROR(
      loom_aie2p_pipeline_create_low_function(&emitter, out_low_function));
  const loom_location_id_t location = plan->pipeline.op->location;
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_groups(&emitter, location));
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_instances(&emitter, location));
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_bindings(&emitter, location));
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_edges(&emitter, location));
  IREE_RETURN_IF_ERROR(loom_aie2p_pipeline_emit_locations(&emitter, location));
  loom_op_t* return_op = NULL;
  return loom_low_return_build(&emitter.builder, /*values=*/NULL,
                               /*values_count=*/0, location, &return_op);
}

iree_status_t loom_aie2p_pipeline_lower_to_array_low(
    loom_module_t* module, loom_func_like_t pipeline,
    const loom_value_fact_table_t* facts, loom_op_t** out_low_function) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT(loom_func_like_isa(pipeline));
  IREE_ASSERT_ARGUMENT(facts);
  IREE_ASSERT_ARGUMENT(out_low_function);
  *out_low_function = NULL;

  IREE_ASSERT(loom_pipeline_def_isa(pipeline.op));
  if (loom_pipeline_def_scope(pipeline.op) != LOOM_PIPELINE_DEF_SCOPE_KERNEL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AIE2P array pipelines must declare kernel materialization scope");
  }
  if (!loom_symbol_ref_is_valid(loom_func_like_target(pipeline))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AIE2P array pipeline requires an exact target");
  }

  iree_arena_allocator_t scratch_arena;
  iree_arena_initialize(module->arena.block_pool, &scratch_arena);
  loom_aie2p_pipeline_placement_t placement = {0};
  loom_pipeline_plan_t plan = {0};
  const loom_xdna_array_family_t* family = loom_xdna_npu2_array_family();
  const uint32_t maximum_instance_count =
      (uint32_t)family->column_count * family->row_count;
  iree_status_t status =
      loom_pipeline_plan_build(module, pipeline, facts,
                               (loom_pipeline_plan_limits_t){
                                   .instance_count = maximum_instance_count,
                               },
                               &scratch_arena, &plan);
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_pipeline_placement_initialize(&plan, &scratch_arena,
                                                      &placement);
  }
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_pipeline_place_instances(&placement);
  }
  loom_op_t* low_function = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_aie2p_pipeline_emit_low_function(
        module, &plan, &placement, &scratch_arena, &low_function);
  }
  if (!iree_status_is_ok(status) && low_function != NULL) {
    status = iree_status_join(status, loom_op_erase(module, low_function));
    low_function = NULL;
  }
  if (iree_status_is_ok(status)) {
    status = loom_op_erase(module, pipeline.op);
    if (!iree_status_is_ok(status)) {
      status = iree_status_join(status, loom_op_erase(module, low_function));
      low_function = NULL;
    }
  }
  if (iree_status_is_ok(status)) {
    loom_module_link_symbol_defining_op(module, low_function,
                                        loom_op_vtable(module, low_function));
    *out_low_function = low_function;
  }
  iree_arena_deinitialize(&scratch_arena);
  return status;
}
