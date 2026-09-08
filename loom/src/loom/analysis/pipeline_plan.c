// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline_plan.h"

#include <string.h>

#include "loom/analysis/type_refinement.h"
#include "loom/ir/context.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/group/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/type_registry.h"

enum { LOOM_PIPELINE_PLAN_IMPLICIT_MINIMUM_CAPACITY = 1 };

typedef enum loom_pipeline_external_flow_kind_e {
  LOOM_PIPELINE_EXTERNAL_FLOW_KIND_READ = 0,
  LOOM_PIPELINE_EXTERNAL_FLOW_KIND_SCATTER = 1,
} loom_pipeline_external_flow_kind_t;

typedef enum loom_pipeline_plan_flow_usage_e {
  LOOM_PIPELINE_PLAN_FLOW_USAGE_NONE = 0,
  LOOM_PIPELINE_PLAN_FLOW_USAGE_USED = 1u << 0,
  LOOM_PIPELINE_PLAN_FLOW_USAGE_FOLDED = 1u << 1,
} loom_pipeline_plan_flow_usage_t;

typedef struct loom_pipeline_plan_view_binding_t {
  // Source view SSA identity.
  loom_value_id_t source_value;

  // Launch binding ordinal referenced by the view.
  uint32_t binding_index;

  // Exact byte offset from the launch binding base.
  uint64_t byte_offset;
} loom_pipeline_plan_view_binding_t;

typedef struct loom_pipeline_plan_write_t {
  // Logical flow written to the launch binding.
  uint32_t flow_index;

  // Destination launch binding ordinal.
  uint32_t binding_index;

  // Typed destination binding view.
  uint32_t binding_view_index;

  // Destination binding port ordinal.
  uint32_t binding_port;

  // Whether the destination view has one leading lane dimension.
  bool partitioned;
} loom_pipeline_plan_write_t;

typedef struct loom_pipeline_plan_builder_t {
  // Module containing the source pipeline.
  const loom_module_t* module;

  // Source pipeline function.
  loom_func_like_t pipeline;

  // Exact value facts for the source pipeline.
  const loom_value_fact_table_t* facts;

  // Scratch arena owning plan and builder storage.
  iree_arena_allocator_t* arena;

  // Launch binding table.
  loom_pipeline_plan_binding_t* bindings;

  // Number of launch binding slots.
  uint32_t binding_count;

  // Next unclaimed endpoint port for each binding.
  uint32_t* binding_next_ports;

  // Typed launch-binding views used by external flows.
  loom_pipeline_plan_binding_view_t* binding_views;

  // Number of defined binding views.
  uint32_t binding_view_count;

  // Maximum binding views allocated.
  uint32_t binding_view_capacity;

  // Scheduling group table.
  loom_pipeline_plan_group_t* groups;

  // Number of defined scheduling groups.
  uint32_t group_count;

  // Maximum scheduling groups allocated.
  uint32_t group_capacity;

  // Resident callable instance table.
  loom_pipeline_plan_instance_t* instances;

  // Number of defined resident instances.
  uint32_t instance_count;

  // Maximum resident instances accepted by the materializer.
  uint32_t instance_capacity;

  // Logical stage table.
  loom_pipeline_plan_stage_t* stages;

  // Number of defined logical stages.
  uint32_t stage_count;

  // Maximum logical stages allocated.
  uint32_t stage_capacity;

  // Contiguous logical callable-port mappings.
  loom_pipeline_plan_stage_port_t* stage_ports;

  // Number of defined logical callable-port mappings.
  uint32_t stage_port_count;

  // Maximum logical callable-port mappings allocated.
  uint32_t stage_port_capacity;

  // Physical resident-worker boundary ports.
  loom_pipeline_plan_group_port_t* group_ports;

  // Number of defined physical resident-worker boundary ports.
  uint32_t group_port_count;

  // Maximum physical resident-worker boundary ports allocated.
  uint32_t group_port_capacity;

  // Typed logical flow table.
  loom_pipeline_plan_flow_t* flows;

  // Number of defined logical flows.
  uint32_t flow_count;

  // Maximum logical flows allocated.
  uint32_t flow_capacity;

  // Physical consumer and fold ownership flags for each flow version.
  uint8_t* flow_usage;

  // Concrete point-to-point edge table.
  loom_pipeline_plan_edge_t* edges;

  // Number of defined edges.
  uint32_t edge_count;

  // Maximum concrete edges allocated.
  uint32_t edge_capacity;

  // Deferred launch-binding writes.
  loom_pipeline_plan_write_t* writes;

  // Number of deferred launch-binding writes.
  uint32_t write_count;

  // Maximum deferred launch-binding writes allocated.
  uint32_t write_capacity;

  // Source launch views and their binding ordinals.
  loom_pipeline_plan_view_binding_t* view_bindings;

  // Number of source launch views recorded.
  uint32_t view_binding_count;

  // Maximum source launch views allocated.
  uint32_t view_binding_capacity;
} loom_pipeline_plan_builder_t;

static iree_status_t loom_pipeline_plan_allocate_array(
    iree_arena_allocator_t* arena, iree_host_size_t count,
    iree_host_size_t element_size, void** out_ptr) {
  *out_ptr = NULL;
  if (count == 0) return iree_ok_status();
  return iree_arena_allocate_array(arena, count, element_size, out_ptr);
}

static iree_status_t loom_pipeline_plan_exact_u32(
    const loom_pipeline_plan_builder_t* builder, loom_value_id_t value_id,
    const char* purpose, uint32_t* out_value) {
  loom_value_facts_t element_facts = loom_value_facts_unknown();
  int64_t value = 0;
  if (!loom_value_facts_query_all_equal_element(
          &builder->facts->context,
          loom_value_fact_table_lookup(builder->facts, value_id),
          &element_facts) ||
      !loom_value_facts_as_exact_i64(element_facts, &value) || value < 0 ||
      value > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline %s must resolve to one exact non-negative u32 fact", purpose);
  }
  *out_value = (uint32_t)value;
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_exact_u64(
    const loom_pipeline_plan_builder_t* builder, loom_value_id_t value_id,
    const char* purpose, uint64_t* out_value) {
  loom_value_facts_t element_facts = loom_value_facts_unknown();
  int64_t value = 0;
  if (!loom_value_facts_query_all_equal_element(
          &builder->facts->context,
          loom_value_fact_table_lookup(builder->facts, value_id),
          &element_facts) ||
      !loom_value_facts_as_exact_i64(element_facts, &value) || value < 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline %s must resolve to one exact non-negative i64 fact", purpose);
  }
  *out_value = (uint64_t)value;
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_exact_dimension(
    const loom_pipeline_plan_builder_t* builder, loom_type_t type,
    uint8_t dimension, uint32_t* out_value) {
  if (!loom_type_dim_is_dynamic_at(type, dimension)) {
    const int64_t value = loom_type_dim_static_size_at(type, dimension);
    if (value < 0 || value > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "pipeline tile dimension is out of range");
    }
    *out_value = (uint32_t)value;
    return iree_ok_status();
  }
  return loom_pipeline_plan_exact_u32(
      builder, loom_type_dim_value_id_at(type, dimension), "tile dimension",
      out_value);
}

static iree_status_t loom_pipeline_plan_exact_record_shape(
    const loom_pipeline_plan_builder_t* builder, loom_type_t view_type,
    loom_type_t tile_type, bool partitioned,
    loom_pipeline_plan_record_shape_t* out_record_shape,
    uint32_t* out_record_count) {
  const uint8_t view_rank = loom_type_rank(view_type);
  const uint8_t tile_rank = loom_type_rank(tile_type);
  const uint8_t sequence_start = partitioned ? 1 : 0;
  IREE_ASSERT_GE(view_rank, tile_rank + sequence_start);
  const uint8_t sequence_end = view_rank - tile_rank;
  const uint8_t sequence_rank = sequence_end - sequence_start;
  uint32_t* sequence_dimensions = NULL;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_allocate_array(
      builder->arena, sequence_rank, sizeof(*sequence_dimensions),
      (void**)&sequence_dimensions));
  uint64_t record_count = 1;
  for (uint8_t axis = sequence_start; axis < sequence_end; ++axis) {
    uint32_t dimension = 0;
    IREE_RETURN_IF_ERROR(loom_pipeline_plan_exact_dimension(builder, view_type,
                                                            axis, &dimension));
    if (dimension == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pipeline record sequence cannot be empty");
    }
    sequence_dimensions[axis - sequence_start] = dimension;
    record_count *= dimension;
    if (record_count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "pipeline record count exceeds u32 range");
    }
  }
  *out_record_shape = (loom_pipeline_plan_record_shape_t){
      .dimensions = sequence_dimensions,
      .rank = sequence_rank,
  };
  if (out_record_count != NULL) {
    *out_record_count = (uint32_t)record_count;
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_validate_fixed_record_tile(
    const loom_pipeline_plan_builder_t* builder, loom_type_t view_type,
    loom_type_t tile_type) {
  const loom_encoding_record_layout_t* record_layout = NULL;
  if (!loom_encoding_query_type_record_layout(&builder->facts->context,
                                              builder->module, view_type,
                                              &record_layout)) {
    return iree_ok_status();
  }

  uint64_t element_count = 0;
  const int32_t element_bit_width =
      loom_scalar_type_bitwidth(loom_type_element_type(tile_type));
  uint64_t bit_length = 0;
  if (!loom_type_static_element_count(tile_type, &element_count) ||
      element_bit_width <= 0 ||
      !iree_checked_mul_u64(element_count, (uint64_t)element_bit_width,
                            &bit_length) ||
      (bit_length & 7u) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline encoded flow tile must have a whole-byte physical shape");
  }
  const uint64_t tile_byte_length = bit_length / 8u;
  const uint16_t record_byte_length =
      record_layout->geometry.storage_byte_count;
  if (record_byte_length == 0 || tile_byte_length == 0 ||
      tile_byte_length % record_byte_length != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline encoded flow tile must contain whole fixed storage records");
  }
  return iree_ok_status();
}

static bool loom_pipeline_plan_record_shapes_equal(
    loom_pipeline_plan_record_shape_t lhs,
    loom_pipeline_plan_record_shape_t rhs) {
  return lhs.rank == rhs.rank &&
         (lhs.rank == 0 || memcmp(lhs.dimensions, rhs.dimensions,
                                  lhs.rank * sizeof(*lhs.dimensions)) == 0);
}

static iree_status_t loom_pipeline_plan_refine_record_type(
    const loom_pipeline_plan_builder_t* builder, loom_type_t source_type,
    const char* purpose, loom_type_t* out_type) {
  loom_type_refinement_result_t result = LOOM_TYPE_REFINEMENT_UNCHANGED;
  IREE_RETURN_IF_ERROR(loom_type_refine_with_value_facts(
      source_type, builder->facts, builder->arena, out_type, &result));
  if (result == LOOM_TYPE_REFINEMENT_CONFLICT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline %s type conflicts with its exact specialization facts",
        purpose);
  }
  if (!loom_type_is_all_static(*out_type) ||
      loom_type_has_ssa_encoding(*out_type)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline %s type must resolve to static dimensions and encoding",
        purpose);
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_flow_tile_type(
    const loom_pipeline_plan_builder_t* builder, loom_value_id_t flow_value,
    loom_type_t* out_type) {
  const loom_type_t flow_type =
      loom_module_value_type(builder->module, flow_value);
  IREE_ASSERT(loom_pipeline_flow_type_isa(flow_type));
  const loom_type_id_t tile_type_id =
      loom_pipeline_flow_type_element_type(flow_type);
  IREE_ASSERT_LT(tile_type_id, builder->module->types.count);
  const loom_type_t tile_type = builder->module->types.entries[tile_type_id];
  IREE_ASSERT(loom_type_is_tile(tile_type));
  return loom_pipeline_plan_refine_record_type(builder, tile_type, "flow",
                                               out_type);
}

static iree_status_t loom_pipeline_plan_view_tile_type(
    const loom_pipeline_plan_builder_t* builder, loom_type_t view_type,
    loom_type_t* out_type) {
  IREE_ASSERT(loom_type_is_view(view_type));
  view_type.header = loom_type_make_header(
      LOOM_TYPE_TILE, loom_type_element_type(view_type),
      loom_type_rank(view_type), loom_type_flags(view_type));
  return loom_pipeline_plan_refine_record_type(builder, view_type,
                                               "binding view", out_type);
}

static iree_status_t loom_pipeline_plan_lookup_group(
    const loom_pipeline_plan_builder_t* builder, loom_value_id_t value_id,
    uint32_t* out_index) {
  for (uint32_t i = 0; i < builder->group_count; ++i) {
    if (builder->groups[i].source_value == value_id) {
      *out_index = i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "pipeline references an unknown group");
}

static iree_status_t loom_pipeline_plan_lookup_flow(
    const loom_pipeline_plan_builder_t* builder, loom_value_id_t value_id,
    uint32_t* out_index) {
  for (uint32_t i = 0; i < builder->flow_count; ++i) {
    if (builder->flows[i].source_value == value_id) {
      *out_index = i;
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "pipeline references an unknown flow");
}

static void loom_pipeline_plan_define_flow(
    loom_pipeline_plan_builder_t* builder, loom_value_id_t source_value,
    loom_pipeline_plan_flow_t flow, uint32_t* out_index) {
  IREE_ASSERT_LT(builder->flow_count, builder->flow_capacity);
  const uint32_t index = builder->flow_count++;
  flow.source_value = source_value;
  if (flow.storage_flow_index == UINT32_MAX) {
    flow.storage_flow_index = index;
  }
  builder->flows[index] = flow;
  if (out_index != NULL) *out_index = index;
}

static uint32_t loom_pipeline_plan_define_binding_view(
    loom_pipeline_plan_builder_t* builder, loom_type_t binding_type,
    uint64_t byte_offset) {
  IREE_ASSERT_LT(builder->binding_view_count, builder->binding_view_capacity);
  const uint32_t index = builder->binding_view_count++;
  builder->binding_views[index] = (loom_pipeline_plan_binding_view_t){
      .binding_type = binding_type,
      .byte_offset = byte_offset,
  };
  return index;
}

static bool loom_pipeline_plan_flow_binding_is_partitioned(
    const loom_pipeline_plan_builder_t* builder,
    const loom_pipeline_plan_flow_t* flow) {
  if (flow->producer_kind != LOOM_PIPELINE_ENDPOINT_KIND_BINDING) return false;
  IREE_ASSERT_LT(flow->binding_view_index, builder->binding_view_count);
  const loom_type_t binding_type =
      builder->binding_views[flow->binding_view_index].binding_type;
  return loom_type_rank(binding_type) ==
         loom_type_rank(flow->tile_type) + flow->record_shape.rank + 1u;
}

static iree_status_t loom_pipeline_plan_use_flow(
    loom_pipeline_plan_builder_t* builder, loom_value_id_t value_id,
    uint32_t* out_index) {
  IREE_RETURN_IF_ERROR(
      loom_pipeline_plan_lookup_flow(builder, value_id, out_index));
  if (iree_any_bit_set(builder->flow_usage[*out_index],
                       LOOM_PIPELINE_PLAN_FLOW_USAGE_FOLDED)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline fold source cannot also have a recordwise consumer");
  }
  builder->flow_usage[*out_index] |= LOOM_PIPELINE_PLAN_FLOW_USAGE_USED;
  return iree_ok_status();
}

static void loom_pipeline_plan_append_edge(
    loom_pipeline_plan_builder_t* builder, loom_pipeline_plan_edge_t edge) {
  IREE_ASSERT_LT(builder->edge_count, builder->edge_capacity);
  builder->edges[builder->edge_count++] = edge;
}

static iree_status_t loom_pipeline_plan_binding_for_view(
    const loom_pipeline_plan_builder_t* builder, loom_value_id_t view_value,
    uint32_t* out_binding_index, uint64_t* out_byte_offset) {
  for (uint32_t i = 0; i < builder->view_binding_count; ++i) {
    if (builder->view_bindings[i].source_value == view_value) {
      *out_binding_index = builder->view_bindings[i].binding_index;
      *out_byte_offset = builder->view_bindings[i].byte_offset;
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "pipeline external flow must use a direct launch-binding view");
}

static iree_status_t loom_pipeline_plan_parse_view(
    loom_pipeline_plan_builder_t* builder, const loom_op_t* op,
    const loom_block_t* entry_block, uint32_t specialization_count) {
  const loom_value_id_t buffer_value = loom_buffer_view_buffer(op);
  const loom_value_t* buffer = loom_module_value(builder->module, buffer_value);
  if (!loom_value_is_block_arg(buffer) ||
      loom_value_def_block(buffer) != entry_block) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline views must be formed directly from launch bindings");
  }
  const uint32_t argument_index = loom_value_def_index(buffer);
  if (argument_index < specialization_count ||
      argument_index - specialization_count >= builder->binding_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline view does not reference a launch binding");
  }
  uint64_t byte_offset = 0;
  IREE_RETURN_IF_ERROR(
      loom_pipeline_plan_exact_u64(builder, loom_buffer_view_byte_offset(op),
                                   "view byte offset", &byte_offset));
  IREE_ASSERT_LT(builder->view_binding_count, builder->view_binding_capacity);
  builder->view_bindings[builder->view_binding_count++] =
      (loom_pipeline_plan_view_binding_t){
          .source_value = loom_buffer_view_result(op),
          .binding_index = argument_index - specialization_count,
          .byte_offset = byte_offset,
      };
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_define_group(
    loom_pipeline_plan_builder_t* builder, const loom_op_t* op) {
  IREE_ASSERT_LT(builder->group_count, builder->group_capacity);
  uint32_t lane_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_pipeline_plan_exact_u32(builder, loom_group_create_cardinality(op),
                                   "group cardinality", &lane_count));
  if (lane_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline groups cannot be empty");
  }
  const loom_value_id_t result = loom_group_create_result(op);
  const uint32_t index = builder->group_count++;
  builder->groups[index] = (loom_pipeline_plan_group_t){
      .source_value = result,
      .lane_count = lane_count,
      .instance_start = UINT32_MAX,
  };
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_define_external_flow(
    loom_pipeline_plan_builder_t* builder, const loom_op_t* op,
    loom_pipeline_external_flow_kind_t kind) {
  const bool partitioned = kind == LOOM_PIPELINE_EXTERNAL_FLOW_KIND_SCATTER;
  const loom_value_id_t source_view = partitioned
                                          ? loom_pipeline_scatter_source(op)
                                          : loom_pipeline_read_source(op);
  const loom_value_id_t group_value = partitioned
                                          ? loom_pipeline_scatter_group(op)
                                          : loom_pipeline_read_group(op);
  const loom_value_id_t result = partitioned ? loom_pipeline_scatter_result(op)
                                             : loom_pipeline_read_result(op);

  uint32_t group_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_pipeline_plan_lookup_group(builder, group_value, &group_index));
  uint32_t binding_index = 0;
  uint64_t binding_byte_offset = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_binding_for_view(
      builder, source_view, &binding_index, &binding_byte_offset));
  builder->bindings[binding_index].access |=
      LOOM_PIPELINE_BINDING_ACCESS_FLAG_READ;

  const loom_type_t view_type =
      loom_module_value_type(builder->module, source_view);
  if (partitioned) {
    uint32_t leading_dimension = 0;
    IREE_RETURN_IF_ERROR(loom_pipeline_plan_exact_dimension(
        builder, view_type, 0, &leading_dimension));
    if (leading_dimension != builder->groups[group_index].lane_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipeline scatter leading dimension must equal group cardinality");
    }
  }
  loom_type_t binding_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_pipeline_plan_view_tile_type(builder, view_type, &binding_type));
  const uint32_t binding_view_index = loom_pipeline_plan_define_binding_view(
      builder, binding_type, binding_byte_offset);
  if (builder->binding_next_ports[binding_index] == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pipeline binding port ordinal overflow");
  }
  const uint32_t binding_port = builder->binding_next_ports[binding_index];
  ++builder->binding_next_ports[binding_index];
  loom_type_t tile_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_pipeline_plan_flow_tile_type(builder, result, &tile_type));
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_validate_fixed_record_tile(
      builder, view_type, tile_type));
  loom_pipeline_plan_record_shape_t record_shape = {0};
  uint32_t record_count = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_exact_record_shape(
      builder, view_type, tile_type, partitioned, &record_shape,
      &record_count));
  loom_pipeline_plan_define_flow(
      builder, result,
      (loom_pipeline_plan_flow_t){
          .tile_type = tile_type,
          .record_shape = record_shape,
          .record_count = record_count,
          .group_index = group_index,
          .minimum_capacity = LOOM_PIPELINE_PLAN_IMPLICIT_MINIMUM_CAPACITY,
          .storage_flow_index = UINT32_MAX,
          .producer_kind = LOOM_PIPELINE_ENDPOINT_KIND_BINDING,
          .binding_index = binding_index,
          .instance_start = UINT32_MAX,
          .producer_stage_index = UINT32_MAX,
          .producer_stage_port = UINT32_MAX,
          .producer_port = binding_port,
          .binding_view_index = binding_view_index,
      },
      NULL);
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_materialize_group_instances(
    loom_pipeline_plan_builder_t* builder, uint32_t group_index) {
  loom_pipeline_plan_group_t* group = &builder->groups[group_index];
  if (group->instance_start != UINT32_MAX) return iree_ok_status();
  if (group->lane_count >
      builder->instance_capacity - builder->instance_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "pipeline requires more resident instances than the target supports");
  }
  group->instance_start = builder->instance_count;
  for (uint32_t lane = 0; lane < group->lane_count; ++lane) {
    builder->instances[builder->instance_count++] =
        (loom_pipeline_plan_instance_t){
            .group_index = group_index,
            .lane = lane,
            .entry = loom_symbol_ref_null(),
        };
  }
  return iree_ok_status();
}

static void loom_pipeline_plan_append_stage_port(
    loom_pipeline_plan_builder_t* builder,
    loom_pipeline_plan_stage_port_t port) {
  IREE_ASSERT_LT(builder->stage_port_count, builder->stage_port_capacity);
  builder->stage_ports[builder->stage_port_count++] = port;
}

static iree_status_t loom_pipeline_plan_append_stage(
    loom_pipeline_plan_builder_t* builder, uint32_t group_index,
    loom_symbol_ref_t entry, uint16_t input_count, uint16_t output_count,
    uint32_t* out_stage_index) {
  IREE_RETURN_IF_ERROR(
      loom_pipeline_plan_materialize_group_instances(builder, group_index));
  IREE_ASSERT_LT(builder->stage_count, builder->stage_capacity);
  loom_pipeline_plan_group_t* group = &builder->groups[group_index];
  const uint32_t stage_index = builder->stage_count++;
  builder->stages[stage_index] = (loom_pipeline_plan_stage_t){
      .group_index = group_index,
      .entry = entry,
      .port_start = builder->stage_port_count,
      .input_count = input_count,
      .output_count = output_count,
  };
  ++group->stage_count;
  *out_stage_index = stage_index;
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_append_pointwise_stage_input(
    loom_pipeline_plan_builder_t* builder, loom_value_id_t flow_value,
    uint32_t target_group_index) {
  uint32_t flow_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_pipeline_plan_use_flow(builder, flow_value, &flow_index));
  const loom_pipeline_plan_flow_t* flow = &builder->flows[flow_index];
  const loom_pipeline_plan_group_t* group =
      &builder->groups[target_group_index];
  const uint32_t producer_lane_count =
      flow->producer_kind == LOOM_PIPELINE_ENDPOINT_KIND_BINDING
          ? builder->groups[flow->group_index].lane_count
          : flow->instance_count;
  if (producer_lane_count != group->lane_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pointwise pipeline input must match its stage group cardinality");
  }
  loom_pipeline_plan_append_stage_port(builder,
                                       (loom_pipeline_plan_stage_port_t){
                                           .flow_index = flow_index,
                                           .source_lane = UINT32_MAX,
                                       });
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_define_stage_outputs(
    loom_pipeline_plan_builder_t* builder, loom_value_slice_t outputs,
    uint32_t stage_index, loom_pipeline_plan_record_shape_t record_shape,
    uint32_t record_count) {
  const loom_pipeline_plan_stage_t* stage = &builder->stages[stage_index];
  const uint32_t group_index = stage->group_index;
  const uint32_t instance_start = builder->groups[group_index].instance_start;
  const uint32_t instance_count = builder->groups[group_index].lane_count;
  for (uint16_t i = 0; i < outputs.count; ++i) {
    loom_type_t tile_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_pipeline_plan_flow_tile_type(
        builder, outputs.values[i], &tile_type));
    uint32_t flow_index = 0;
    loom_pipeline_plan_define_flow(
        builder, outputs.values[i],
        (loom_pipeline_plan_flow_t){
            .tile_type = tile_type,
            .record_shape = record_shape,
            .record_count = record_count,
            .group_index = group_index,
            .minimum_capacity = LOOM_PIPELINE_PLAN_IMPLICIT_MINIMUM_CAPACITY,
            .storage_flow_index = UINT32_MAX,
            .producer_kind = LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE,
            .binding_index = UINT32_MAX,
            .instance_start = instance_start,
            .instance_count = instance_count,
            .producer_stage_index = stage_index,
            .producer_stage_port = stage->input_count + i,
            .producer_port = stage->input_count + i,
            .binding_view_index = UINT32_MAX,
        },
        &flow_index);
    loom_pipeline_plan_append_stage_port(builder,
                                         (loom_pipeline_plan_stage_port_t){
                                             .flow_index = flow_index,
                                             .source_lane = UINT32_MAX,
                                         });
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_common_record_shape(
    const loom_pipeline_plan_builder_t* builder, loom_value_slice_t values,
    const char* operation,
    loom_pipeline_plan_record_shape_t* inout_record_shape,
    uint32_t* inout_record_count, bool* inout_has_record_shape) {
  for (uint16_t i = 0; i < values.count; ++i) {
    uint32_t flow_index = 0;
    IREE_RETURN_IF_ERROR(
        loom_pipeline_plan_lookup_flow(builder, values.values[i], &flow_index));
    const loom_pipeline_plan_flow_t* flow = &builder->flows[flow_index];
    if (!*inout_has_record_shape) {
      *inout_record_shape = flow->record_shape;
      *inout_record_count = flow->record_count;
      *inout_has_record_shape = true;
    } else if (!loom_pipeline_plan_record_shapes_equal(*inout_record_shape,
                                                       flow->record_shape)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "pipeline %s inputs must have equal "
                              "record shapes",
                              operation);
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_parse_stage(
    loom_pipeline_plan_builder_t* builder, const loom_op_t* op) {
  const loom_value_slice_t inputs = loom_pipeline_stage_inputs(op);
  loom_pipeline_plan_record_shape_t record_shape = {0};
  uint32_t record_count = 1;
  bool has_record_shape = false;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_common_record_shape(
      builder, inputs, "stage", &record_shape, &record_count,
      &has_record_shape));
  uint32_t group_index = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_lookup_group(
      builder, loom_pipeline_stage_group(op), &group_index));
  uint32_t stage_index = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_append_stage(
      builder, group_index, loom_pipeline_stage_entry(op), inputs.count,
      loom_pipeline_stage_outputs(op).count, &stage_index));
  for (uint16_t i = 0; i < inputs.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_pipeline_plan_append_pointwise_stage_input(
        builder, inputs.values[i], group_index));
  }
  return loom_pipeline_plan_define_stage_outputs(
      builder, loom_pipeline_stage_outputs(op), stage_index, record_shape,
      record_count);
}

static iree_status_t loom_pipeline_plan_parse_fold(
    loom_pipeline_plan_builder_t* builder, const loom_op_t* op) {
  uint32_t source_flow_index = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_lookup_flow(
      builder, loom_pipeline_fold_source(op), &source_flow_index));
  if (iree_any_bit_set(builder->flow_usage[source_flow_index],
                       LOOM_PIPELINE_PLAN_FLOW_USAGE_USED)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline fold must be the source flow's only consumer");
  }
  const loom_value_t* source_value =
      loom_module_value(builder->module, loom_pipeline_fold_source(op));
  const loom_op_t* source_op = loom_value_def_op(source_value);
  if (source_op == NULL || !loom_pipeline_stage_isa(source_op) ||
      source_op->result_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "concrete pipeline folds currently require a direct single-output "
        "stage source");
  }

  loom_pipeline_plan_flow_t flow = builder->flows[source_flow_index];
  IREE_ASSERT_EQ(flow.producer_kind, LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE);
  IREE_ASSERT_LT(flow.producer_stage_index, builder->stage_count);
  loom_pipeline_plan_stage_t* stage =
      &builder->stages[flow.producer_stage_index];
  const uint32_t fold_record_count =
      flow.record_shape.rank == 0
          ? 1
          : flow.record_shape.dimensions[flow.record_shape.rank - 1];
  if (stage->fold_record_count != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline stage has more than one fold");
  }
  stage->fold_record_count = fold_record_count;
  stage->fold_output_port = flow.producer_stage_port;
  stage->fold_kind = loom_pipeline_fold_kind(op);
  stage->fold_fast_math_flags = loom_pipeline_fold_fastmath(op);
  builder->flow_usage[source_flow_index] |=
      LOOM_PIPELINE_PLAN_FLOW_USAGE_USED | LOOM_PIPELINE_PLAN_FLOW_USAGE_FOLDED;

  if (flow.record_shape.rank != 0) --flow.record_shape.rank;
  flow.record_count /= fold_record_count;
  flow.minimum_capacity = LOOM_PIPELINE_PLAN_IMPLICIT_MINIMUM_CAPACITY;
  loom_pipeline_plan_define_flow(builder, loom_pipeline_fold_result(op), flow,
                                 NULL);
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_validate_entry_argument_count(
    const loom_pipeline_plan_builder_t* builder, loom_symbol_ref_t entry_ref,
    uint32_t expected_count) {
  IREE_ASSERT_EQ(entry_ref.module_id, 0u);
  IREE_ASSERT_LT(entry_ref.symbol_id, builder->module->symbols.count);
  const loom_func_like_t entry = loom_func_like_const_cast(
      builder->module,
      builder->module->symbols.entries[entry_ref.symbol_id].defining_op);
  uint16_t argument_count = 0;
  loom_func_like_arg_ids(entry, &argument_count);
  if (argument_count != expected_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline reduction entry requires %u buffer arguments but defines %u",
        expected_count, argument_count);
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_parse_reduce(
    loom_pipeline_plan_builder_t* builder, const loom_op_t* op) {
  uint32_t source_group_index = 0;
  uint32_t target_group_index = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_lookup_group(
      builder, loom_pipeline_reduce_source_group(op), &source_group_index));
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_lookup_group(
      builder, loom_pipeline_reduce_target_group(op), &target_group_index));
  const loom_pipeline_plan_group_t* source_group =
      &builder->groups[source_group_index];
  const loom_pipeline_plan_group_t* target_group =
      &builder->groups[target_group_index];
  if (target_group->lane_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "concrete pipeline reductions currently require one target lane");
  }

  const loom_value_slice_t source_inputs =
      loom_pipeline_reduce_source_inputs(op);
  const loom_value_slice_t target_inputs =
      loom_pipeline_reduce_target_inputs(op);
  const loom_value_slice_t outputs = loom_pipeline_reduce_outputs(op);
  loom_pipeline_plan_record_shape_t record_shape = {0};
  uint32_t record_count = 1;
  bool has_record_shape = false;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_common_record_shape(
      builder, source_inputs, "reduction", &record_shape, &record_count,
      &has_record_shape));
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_common_record_shape(
      builder, target_inputs, "reduction", &record_shape, &record_count,
      &has_record_shape));
  const uint64_t expanded_source_count =
      (uint64_t)source_inputs.count * source_group->lane_count;
  const uint64_t expected_argument_count =
      expanded_source_count + target_inputs.count + outputs.count;
  if (expected_argument_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pipeline reduction argument count exceeds the "
                            "callable ABI limit");
  }
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_validate_entry_argument_count(
      builder, loom_pipeline_reduce_entry(op),
      (uint32_t)expected_argument_count));

  const uint16_t input_count =
      (uint16_t)(expanded_source_count + target_inputs.count);
  uint32_t stage_index = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_append_stage(
      builder, target_group_index, loom_pipeline_reduce_entry(op), input_count,
      outputs.count, &stage_index));
  for (uint16_t i = 0; i < source_inputs.count; ++i) {
    uint32_t flow_index = 0;
    IREE_RETURN_IF_ERROR(loom_pipeline_plan_use_flow(
        builder, source_inputs.values[i], &flow_index));
    const loom_pipeline_plan_flow_t* flow = &builder->flows[flow_index];
    if (flow->group_index != source_group_index ||
        flow->producer_kind != LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE ||
        flow->instance_count != source_group->lane_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipeline reduction source must be a resident source-group flow");
    }
    for (uint32_t lane = 0; lane < source_group->lane_count; ++lane) {
      loom_pipeline_plan_append_stage_port(builder,
                                           (loom_pipeline_plan_stage_port_t){
                                               .flow_index = flow_index,
                                               .source_lane = lane,
                                           });
    }
  }
  for (uint16_t i = 0; i < target_inputs.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_pipeline_plan_append_pointwise_stage_input(
        builder, target_inputs.values[i], target_group_index));
  }
  return loom_pipeline_plan_define_stage_outputs(builder, outputs, stage_index,
                                                 record_shape, record_count);
}

static iree_status_t loom_pipeline_plan_parse_buffer(
    loom_pipeline_plan_builder_t* builder, const loom_op_t* op) {
  uint32_t source_flow_index = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_use_flow(
      builder, loom_pipeline_buffer_source(op), &source_flow_index));
  uint32_t capacity = 0;
  IREE_RETURN_IF_ERROR(
      loom_pipeline_plan_exact_u32(builder, loom_pipeline_buffer_capacity(op),
                                   "buffer capacity", &capacity));
  if (capacity == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline buffer capacity cannot be zero");
  }
  loom_pipeline_plan_flow_t flow = builder->flows[source_flow_index];
  flow.minimum_capacity = capacity;
  loom_pipeline_plan_define_flow(builder, loom_pipeline_buffer_result(op), flow,
                                 NULL);
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_parse_write(
    loom_pipeline_plan_builder_t* builder, const loom_op_t* op) {
  uint32_t flow_index = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_use_flow(
      builder, loom_pipeline_write_source(op), &flow_index));
  const loom_pipeline_plan_flow_t* flow = &builder->flows[flow_index];
  if (flow->producer_kind != LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "concrete pipeline writes require a resident source-group flow");
  }
  uint32_t binding_index = 0;
  uint64_t binding_byte_offset = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_binding_for_view(
      builder, loom_pipeline_write_target(op), &binding_index,
      &binding_byte_offset));
  const loom_type_t target_type =
      loom_module_value_type(builder->module, loom_pipeline_write_target(op));
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_validate_fixed_record_tile(
      builder, target_type, flow->tile_type));
  const uint32_t lane_count = builder->groups[flow->group_index].lane_count;
  const bool partitioned = lane_count > 1;
  if (partitioned) {
    if (loom_type_rank(target_type) <= loom_type_rank(flow->tile_type)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "multi-lane pipeline output requires a leading lane dimension");
    }
    uint32_t leading_dimension = 0;
    IREE_RETURN_IF_ERROR(loom_pipeline_plan_exact_dimension(
        builder, target_type, 0, &leading_dimension));
    if (leading_dimension != lane_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipeline output leading dimension must equal source-group "
          "cardinality");
    }
  }
  loom_type_t binding_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_pipeline_plan_view_tile_type(builder, target_type, &binding_type));
  const uint32_t binding_view_index = loom_pipeline_plan_define_binding_view(
      builder, binding_type, binding_byte_offset);
  loom_pipeline_plan_record_shape_t target_record_shape = {0};
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_exact_record_shape(
      builder, target_type, flow->tile_type, partitioned, &target_record_shape,
      /*out_record_count=*/NULL));
  if (!loom_pipeline_plan_record_shapes_equal(target_record_shape,
                                              flow->record_shape)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline output view and flow must have equal record shapes");
  }
  builder->bindings[binding_index].access |=
      LOOM_PIPELINE_BINDING_ACCESS_FLAG_WRITE;
  if (builder->binding_next_ports[binding_index] == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pipeline binding port ordinal overflow");
  }
  const uint32_t target_port = builder->binding_next_ports[binding_index]++;
  IREE_ASSERT_LT(builder->write_count, builder->write_capacity);
  builder->writes[builder->write_count++] = (loom_pipeline_plan_write_t){
      .flow_index = flow_index,
      .binding_index = binding_index,
      .binding_view_index = binding_view_index,
      .binding_port = target_port,
      .partitioned = partitioned,
  };
  return iree_ok_status();
}

static bool loom_pipeline_plan_flow_is_internal_to_group(
    const loom_pipeline_plan_flow_t* flow, uint32_t group_index) {
  return flow->producer_kind == LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE &&
         flow->group_index == group_index;
}

static uint32_t loom_pipeline_plan_group_next_port(
    const loom_pipeline_plan_builder_t* builder, uint32_t group_index) {
  uint32_t next_port = 0;
  for (uint32_t i = 0; i < builder->group_port_count; ++i) {
    const loom_pipeline_plan_group_port_t* port = &builder->group_ports[i];
    if (port->group_index == group_index && port->port >= next_port) {
      next_port = port->port + 1;
    }
  }
  return next_port;
}

static bool loom_pipeline_plan_group_port_matches_flow(
    const loom_pipeline_plan_builder_t* builder,
    const loom_pipeline_plan_group_port_t* port, uint32_t flow_index,
    uint32_t source_lane, loom_pipeline_plan_group_port_direction_t direction) {
  if (port->direction != direction) return false;
  if (direction == LOOM_PIPELINE_PLAN_GROUP_PORT_DIRECTION_RECEIVE) {
    return port->flow_index == flow_index && port->source_lane == source_lane;
  }
  return builder->flows[port->flow_index].storage_flow_index ==
         builder->flows[flow_index].storage_flow_index;
}

static bool loom_pipeline_plan_find_group_port(
    const loom_pipeline_plan_builder_t* builder, uint32_t group_index,
    uint32_t flow_index, uint32_t source_lane,
    loom_pipeline_plan_group_port_direction_t direction, uint32_t* out_port) {
  for (uint32_t i = 0; i < builder->group_port_count; ++i) {
    const loom_pipeline_plan_group_port_t* port = &builder->group_ports[i];
    if (port->group_index == group_index &&
        loom_pipeline_plan_group_port_matches_flow(builder, port, flow_index,
                                                   source_lane, direction)) {
      *out_port = port->port;
      return true;
    }
  }
  return false;
}

static uint32_t loom_pipeline_plan_add_group_port(
    loom_pipeline_plan_builder_t* builder, uint32_t group_index,
    uint32_t flow_index, uint32_t source_lane,
    loom_pipeline_plan_group_port_direction_t direction,
    uint32_t requested_port, bool deduplicate) {
  uint32_t port = 0;
  if (deduplicate &&
      loom_pipeline_plan_find_group_port(builder, group_index, flow_index,
                                         source_lane, direction, &port)) {
    return port;
  }
  IREE_ASSERT_LT(builder->group_port_count, builder->group_port_capacity);
  port = requested_port != UINT32_MAX
             ? requested_port
             : loom_pipeline_plan_group_next_port(builder, group_index);
  builder->group_ports[builder->group_port_count++] =
      (loom_pipeline_plan_group_port_t){
          .group_index = group_index,
          .flow_index = flow_index,
          .source_lane = source_lane,
          .port = port,
          .direction = direction,
      };
  return port;
}

static uint32_t loom_pipeline_plan_add_group_receive_port(
    loom_pipeline_plan_builder_t* builder, uint32_t stage_index,
    uint16_t input_index) {
  const loom_pipeline_plan_stage_t* stage = &builder->stages[stage_index];
  const loom_pipeline_plan_stage_port_t* stage_port =
      &builder->stage_ports[stage->port_start + input_index];
  const bool composite = builder->groups[stage->group_index].stage_count > 1;
  return loom_pipeline_plan_add_group_port(
      builder, stage->group_index, stage_port->flow_index,
      stage_port->source_lane, LOOM_PIPELINE_PLAN_GROUP_PORT_DIRECTION_RECEIVE,
      composite ? UINT32_MAX : input_index, composite);
}

static uint32_t loom_pipeline_plan_add_group_send_port(
    loom_pipeline_plan_builder_t* builder, uint32_t flow_index) {
  loom_pipeline_plan_flow_t* flow = &builder->flows[flow_index];
  IREE_ASSERT_EQ(flow->producer_kind, LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE);
  IREE_ASSERT_LT(flow->producer_stage_index, builder->stage_count);
  const bool composite = builder->groups[flow->group_index].stage_count > 1;
  const uint32_t port = loom_pipeline_plan_add_group_port(
      builder, flow->group_index, flow_index, UINT32_MAX,
      LOOM_PIPELINE_PLAN_GROUP_PORT_DIRECTION_SEND,
      composite ? UINT32_MAX : flow->producer_stage_port,
      /*deduplicate=*/true);
  flow->producer_port = port;
  return port;
}

static iree_status_t loom_pipeline_plan_prepare_group_ports(
    loom_pipeline_plan_builder_t* builder) {
  for (uint32_t stage_index = 0; stage_index < builder->stage_count;
       ++stage_index) {
    const loom_pipeline_plan_stage_t* stage = &builder->stages[stage_index];
    for (uint16_t input_index = 0; input_index < stage->input_count;
         ++input_index) {
      const uint32_t flow_index =
          builder->stage_ports[stage->port_start + input_index].flow_index;
      const loom_pipeline_plan_flow_t* flow = &builder->flows[flow_index];
      if (loom_pipeline_plan_flow_is_internal_to_group(flow,
                                                       stage->group_index)) {
        continue;
      }
      loom_pipeline_plan_add_group_receive_port(builder, stage_index,
                                                input_index);
    }
  }

  for (uint32_t stage_index = 0; stage_index < builder->stage_count;
       ++stage_index) {
    const loom_pipeline_plan_stage_t* stage = &builder->stages[stage_index];
    for (uint16_t input_index = 0; input_index < stage->input_count;
         ++input_index) {
      const uint32_t flow_index =
          builder->stage_ports[stage->port_start + input_index].flow_index;
      const loom_pipeline_plan_flow_t* flow = &builder->flows[flow_index];
      if (flow->producer_kind == LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE &&
          flow->group_index != stage->group_index) {
        loom_pipeline_plan_add_group_send_port(builder, flow_index);
      }
    }
  }
  for (uint32_t i = 0; i < builder->write_count; ++i) {
    loom_pipeline_plan_add_group_send_port(builder,
                                           builder->writes[i].flow_index);
  }
  return iree_ok_status();
}

static bool loom_pipeline_plan_group_has_parallel_folds(
    const loom_pipeline_plan_builder_t* builder, uint32_t group_index,
    uint32_t* out_record_count, uint32_t* out_output_port,
    uint16_t* out_output_count, loom_combining_kind_t* out_kind,
    uint8_t* out_fast_math_flags) {
  const loom_pipeline_plan_group_t* group = &builder->groups[group_index];
  bool has_behavior = false;
  for (uint32_t stage_index = 0; stage_index < builder->stage_count;
       ++stage_index) {
    const loom_pipeline_plan_stage_t* stage = &builder->stages[stage_index];
    if (stage->group_index != group_index) continue;
    if (stage->fold_record_count == 0) return false;
    if (!has_behavior) {
      *out_record_count = stage->fold_record_count;
      *out_kind = stage->fold_kind;
      *out_fast_math_flags = stage->fold_fast_math_flags;
      has_behavior = true;
    } else if (stage->fold_record_count != *out_record_count ||
               stage->fold_kind != *out_kind ||
               stage->fold_fast_math_flags != *out_fast_math_flags) {
      return false;
    }
  }
  if (!has_behavior) return false;

  uint32_t first_output_port = UINT32_MAX;
  uint32_t last_output_port = 0;
  uint32_t output_count = 0;
  for (uint32_t i = 0; i < builder->group_port_count; ++i) {
    const loom_pipeline_plan_group_port_t* port = &builder->group_ports[i];
    if (port->group_index != group_index ||
        port->direction != LOOM_PIPELINE_PLAN_GROUP_PORT_DIRECTION_SEND) {
      continue;
    }
    const loom_pipeline_plan_flow_t* flow = &builder->flows[port->flow_index];
    if (flow->producer_stage_index >= builder->stage_count ||
        builder->stages[flow->producer_stage_index].fold_record_count == 0) {
      return false;
    }
    first_output_port = iree_min(first_output_port, port->port);
    last_output_port = iree_max(last_output_port, port->port);
    ++output_count;
  }
  if (output_count != group->stage_count || output_count > UINT16_MAX ||
      last_output_port - first_output_port + 1 != output_count) {
    return false;
  }
  *out_output_port = first_output_port;
  *out_output_count = (uint16_t)output_count;
  return true;
}

static void loom_pipeline_plan_finalize_instance_behaviors(
    loom_pipeline_plan_builder_t* builder) {
  for (uint32_t group_index = 0; group_index < builder->group_count;
       ++group_index) {
    const loom_pipeline_plan_group_t* group = &builder->groups[group_index];
    uint32_t fold_record_count = 0;
    uint32_t fold_output_port = 0;
    uint16_t fold_output_count = 0;
    loom_combining_kind_t fold_kind = LOOM_COMBINING_KIND_ADDI;
    uint8_t fold_fast_math_flags = 0;
    loom_symbol_ref_t entry = loom_symbol_ref_null();
    if (group->stage_count == 1) {
      for (uint32_t stage_index = 0; stage_index < builder->stage_count;
           ++stage_index) {
        const loom_pipeline_plan_stage_t* stage = &builder->stages[stage_index];
        if (stage->group_index != group_index) continue;
        entry = stage->entry;
        fold_record_count = stage->fold_record_count;
        fold_output_port = stage->fold_output_port;
        fold_output_count = stage->fold_record_count != 0 ? 1 : 0;
        fold_kind = stage->fold_kind;
        fold_fast_math_flags = stage->fold_fast_math_flags;
        break;
      }
    } else {
      loom_pipeline_plan_group_has_parallel_folds(
          builder, group_index, &fold_record_count, &fold_output_port,
          &fold_output_count, &fold_kind, &fold_fast_math_flags);
    }
    for (uint32_t lane = 0; lane < group->lane_count; ++lane) {
      loom_pipeline_plan_instance_t* instance =
          &builder->instances[group->instance_start + lane];
      instance->entry = entry;
      instance->fold_record_count = fold_record_count;
      instance->fold_output_port = fold_output_port;
      instance->fold_output_count = fold_output_count;
      instance->fold_kind = fold_kind;
      instance->fold_fast_math_flags = fold_fast_math_flags;
    }
  }
}

static bool loom_pipeline_plan_has_edge_target(
    const loom_pipeline_plan_builder_t* builder, uint32_t flow_index,
    uint32_t target_index, uint32_t target_port) {
  for (uint32_t i = 0; i < builder->edge_count; ++i) {
    const loom_pipeline_plan_edge_t* edge = &builder->edges[i];
    if (edge->flow_index == flow_index &&
        edge->target_kind == LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE &&
        edge->target_index == target_index &&
        edge->target_port == target_port) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_pipeline_plan_emit_stage_input_edges(
    loom_pipeline_plan_builder_t* builder, uint32_t stage_index) {
  const loom_pipeline_plan_stage_t* stage = &builder->stages[stage_index];
  const loom_pipeline_plan_group_t* target_group =
      &builder->groups[stage->group_index];
  const bool composite = target_group->stage_count > 1;
  for (uint16_t input_index = 0; input_index < stage->input_count;
       ++input_index) {
    const loom_pipeline_plan_stage_port_t* port =
        &builder->stage_ports[stage->port_start + input_index];
    const loom_pipeline_plan_flow_t* flow = &builder->flows[port->flow_index];
    if (loom_pipeline_plan_flow_is_internal_to_group(flow,
                                                     stage->group_index)) {
      continue;
    }

    uint32_t target_port = input_index;
    if (composite &&
        !loom_pipeline_plan_find_group_port(
            builder, stage->group_index, port->flow_index, port->source_lane,
            LOOM_PIPELINE_PLAN_GROUP_PORT_DIRECTION_RECEIVE, &target_port)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "composite pipeline input port was not planned");
    }
    uint32_t source_port = flow->producer_port;
    if (flow->producer_kind == LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE &&
        !loom_pipeline_plan_find_group_port(
            builder, flow->group_index, port->flow_index, UINT32_MAX,
            LOOM_PIPELINE_PLAN_GROUP_PORT_DIRECTION_SEND, &source_port)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "pipeline output port was not planned");
    }
    const bool binding_is_partitioned =
        loom_pipeline_plan_flow_binding_is_partitioned(builder, flow);
    const uint32_t target_lane_count =
        port->source_lane == UINT32_MAX ? target_group->lane_count : 1;
    for (uint32_t target_lane = 0; target_lane < target_lane_count;
         ++target_lane) {
      const uint32_t source_lane =
          port->source_lane == UINT32_MAX ? target_lane : port->source_lane;
      const uint32_t target_instance =
          target_group->instance_start + target_lane;
      if (composite &&
          loom_pipeline_plan_has_edge_target(builder, port->flow_index,
                                             target_instance, target_port)) {
        continue;
      }
      loom_pipeline_plan_append_edge(
          builder,
          (loom_pipeline_plan_edge_t){
              .flow_index = port->flow_index,
              .binding_view_index = flow->binding_view_index,
              .source_kind = flow->producer_kind,
              .source_index =
                  flow->producer_kind == LOOM_PIPELINE_ENDPOINT_KIND_BINDING
                      ? flow->binding_index
                      : flow->instance_start + source_lane,
              .source_port = source_port,
              .binding_view_lane = binding_is_partitioned ? source_lane : 0,
              .target_kind = LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE,
              .target_index = target_instance,
              .target_port = target_port,
          });
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_emit_write_edges(
    loom_pipeline_plan_builder_t* builder) {
  for (uint32_t i = 0; i < builder->write_count; ++i) {
    const loom_pipeline_plan_write_t* write = &builder->writes[i];
    const loom_pipeline_plan_flow_t* flow = &builder->flows[write->flow_index];
    uint32_t source_port = flow->producer_port;
    if (!loom_pipeline_plan_find_group_port(
            builder, flow->group_index, write->flow_index, UINT32_MAX,
            LOOM_PIPELINE_PLAN_GROUP_PORT_DIRECTION_SEND, &source_port)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "pipeline write port was not planned");
    }
    const uint32_t lane_count = builder->groups[flow->group_index].lane_count;
    for (uint32_t lane = 0; lane < lane_count; ++lane) {
      loom_pipeline_plan_append_edge(
          builder, (loom_pipeline_plan_edge_t){
                       .flow_index = write->flow_index,
                       .binding_view_index = write->binding_view_index,
                       .source_kind = LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE,
                       .source_index = flow->instance_start + lane,
                       .source_port = source_port,
                       .binding_view_lane = write->partitioned ? lane : 0,
                       .target_kind = LOOM_PIPELINE_ENDPOINT_KIND_BINDING,
                       .target_index = write->binding_index,
                       .target_port = write->binding_port,
                   });
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_finalize_graph(
    loom_pipeline_plan_builder_t* builder) {
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_prepare_group_ports(builder));
  loom_pipeline_plan_finalize_instance_behaviors(builder);
  for (uint32_t stage_index = 0; stage_index < builder->stage_count;
       ++stage_index) {
    IREE_RETURN_IF_ERROR(
        loom_pipeline_plan_emit_stage_input_edges(builder, stage_index));
  }
  return loom_pipeline_plan_emit_write_edges(builder);
}

static bool loom_pipeline_plan_op_is_compile_time(
    const loom_pipeline_plan_builder_t* builder, const loom_op_t* op) {
  if (op->result_count == 0 || loom_op_may_write(builder->module, op)) {
    return false;
  }
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_value_id_t result = loom_op_results(op)[i];
    const loom_type_t result_type =
        loom_module_value_type(builder->module, result);
    const loom_value_facts_t result_facts =
        loom_value_fact_table_lookup(builder->facts, result);
    if (loom_type_is_encoding(result_type)) {
      loom_value_fact_encoding_summary_t encoding_summary = {0};
      if (!loom_value_facts_query_encoding_summary(
              &builder->facts->context, result_facts, &encoding_summary)) {
        return false;
      }
      continue;
    }
    loom_value_facts_t element_facts = loom_value_facts_unknown();
    int64_t value = 0;
    if (!loom_value_facts_query_all_equal_element(
            &builder->facts->context, result_facts, &element_facts) ||
        !loom_value_facts_as_exact_i64(element_facts, &value)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_pipeline_plan_parse_graph(
    loom_pipeline_plan_builder_t* builder) {
  loom_region_t* body = loom_func_like_body(builder->pipeline);
  IREE_ASSERT(body != NULL && body->block_count == 1);
  const loom_block_t* block = loom_region_const_entry_block(body);
  const uint32_t specialization_count =
      (uint32_t)loom_func_like_specialization_count(builder->pipeline);

  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_buffer_view_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_pipeline_plan_parse_view(builder, op, block,
                                                         specialization_count));
    } else if (loom_group_create_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_pipeline_plan_define_group(builder, op));
    } else if (loom_pipeline_scatter_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_pipeline_plan_define_external_flow(
          builder, op, LOOM_PIPELINE_EXTERNAL_FLOW_KIND_SCATTER));
    } else if (loom_pipeline_read_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_pipeline_plan_define_external_flow(
          builder, op, LOOM_PIPELINE_EXTERNAL_FLOW_KIND_READ));
    } else if (loom_pipeline_stage_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_pipeline_plan_parse_stage(builder, op));
    } else if (loom_pipeline_buffer_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_pipeline_plan_parse_buffer(builder, op));
    } else if (loom_pipeline_fold_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_pipeline_plan_parse_fold(builder, op));
    } else if (loom_pipeline_reduce_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_pipeline_plan_parse_reduce(builder, op));
    } else if (loom_pipeline_write_isa(op)) {
      IREE_RETURN_IF_ERROR(loom_pipeline_plan_parse_write(builder, op));
    } else if (loom_pipeline_return_isa(op)) {
      continue;
    } else if (!loom_pipeline_plan_op_is_compile_time(builder, op)) {
      const iree_string_view_t name =
          loom_op_vtable_name(loom_op_vtable(builder->module, op));
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "concrete pipeline planning cannot lower operation '%.*s'",
          (int)name.size, name.data);
    }
  }

  IREE_RETURN_IF_ERROR(loom_pipeline_plan_finalize_graph(builder));

  if (builder->instance_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline has no resident instances");
  }
  for (uint32_t i = 0; i < builder->group_count; ++i) {
    if (builder->groups[i].stage_count == 0) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipeline contains a group with no resident stage");
    }
  }
  for (uint32_t i = 0; i < builder->flow_count; ++i) {
    if (!iree_any_bit_set(builder->flow_usage[i],
                          LOOM_PIPELINE_PLAN_FLOW_USAGE_USED)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipeline contains a flow with no physical consumer");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_measure_graph(
    loom_func_like_t pipeline, uint32_t maximum_instance_count,
    uint32_t* out_group_count, uint32_t* out_flow_count,
    uint32_t* out_stage_count, uint32_t* out_stage_port_count,
    uint32_t* out_write_count, uint32_t* out_view_count,
    uint32_t* out_binding_view_count) {
  *out_group_count = 0;
  *out_flow_count = 0;
  *out_stage_count = 0;
  *out_stage_port_count = 0;
  *out_write_count = 0;
  *out_view_count = 0;
  *out_binding_view_count = 0;
  loom_region_t* body = loom_func_like_body(pipeline);
  if (body == NULL || body->block_count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "concrete pipeline planning requires one graph block");
  }

  uint64_t group_count = 0;
  uint64_t flow_count = 0;
  uint64_t stage_count = 0;
  uint64_t stage_port_count = 0;
  uint64_t write_count = 0;
  uint64_t view_count = 0;
  uint64_t binding_view_count = 0;
  const loom_block_t* block = loom_region_const_entry_block(body);
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_group_create_isa(op)) ++group_count;
    if (loom_buffer_view_isa(op)) ++view_count;
    if (loom_pipeline_scatter_isa(op) || loom_pipeline_read_isa(op) ||
        loom_pipeline_write_isa(op)) {
      ++binding_view_count;
    }
    if (loom_pipeline_scatter_isa(op) || loom_pipeline_read_isa(op) ||
        loom_pipeline_stage_isa(op) || loom_pipeline_buffer_isa(op) ||
        loom_pipeline_fold_isa(op) || loom_pipeline_reduce_isa(op)) {
      flow_count += op->result_count;
    }
    if (loom_pipeline_stage_isa(op)) {
      ++stage_count;
      stage_port_count += loom_pipeline_stage_inputs(op).count;
      stage_port_count += loom_pipeline_stage_outputs(op).count;
    } else if (loom_pipeline_reduce_isa(op)) {
      ++stage_count;
      stage_port_count +=
          (uint64_t)loom_pipeline_reduce_source_inputs(op).count *
          maximum_instance_count;
      stage_port_count += loom_pipeline_reduce_target_inputs(op).count;
      stage_port_count += loom_pipeline_reduce_outputs(op).count;
    } else if (loom_pipeline_write_isa(op)) {
      ++write_count;
    }
  }
  if (group_count > UINT32_MAX || flow_count > UINT32_MAX ||
      stage_count > UINT32_MAX || stage_port_count > UINT32_MAX ||
      write_count > UINT32_MAX || view_count > UINT32_MAX ||
      binding_view_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pipeline graph entity count is too large");
  }
  *out_group_count = (uint32_t)group_count;
  *out_flow_count = (uint32_t)flow_count;
  *out_stage_count = (uint32_t)stage_count;
  *out_stage_port_count = (uint32_t)stage_port_count;
  *out_write_count = (uint32_t)write_count;
  *out_view_count = (uint32_t)view_count;
  *out_binding_view_count = (uint32_t)binding_view_count;
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_builder_initialize(
    const loom_module_t* module, loom_func_like_t pipeline,
    const loom_value_fact_table_t* facts, loom_pipeline_plan_limits_t limits,
    iree_arena_allocator_t* arena, loom_pipeline_plan_builder_t* out_builder) {
  if (limits.instance_count == 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "pipeline materializer has no resident instances");
  }
  uint16_t argument_count = 0;
  loom_func_like_arg_ids(pipeline, &argument_count);
  const int64_t specialization_count =
      loom_func_like_specialization_count(pipeline);
  IREE_ASSERT_GE(specialization_count, 0);
  IREE_ASSERT_LE(specialization_count, argument_count);

  uint32_t group_capacity = 0;
  uint32_t flow_capacity = 0;
  uint32_t stage_capacity = 0;
  uint32_t stage_port_capacity = 0;
  uint32_t write_capacity = 0;
  uint32_t view_binding_capacity = 0;
  uint32_t binding_view_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_measure_graph(
      pipeline, limits.instance_count, &group_capacity, &flow_capacity,
      &stage_capacity, &stage_port_capacity, &write_capacity,
      &view_binding_capacity, &binding_view_capacity));
  *out_builder = (loom_pipeline_plan_builder_t){
      .module = module,
      .pipeline = pipeline,
      .facts = facts,
      .arena = arena,
      .binding_count = argument_count - (uint32_t)specialization_count,
      .binding_view_capacity = binding_view_capacity,
      .group_capacity = group_capacity,
      .instance_capacity = limits.instance_count,
      .stage_capacity = stage_capacity,
      .stage_port_capacity = stage_port_capacity,
      .flow_capacity = flow_capacity,
      .write_capacity = write_capacity,
      .view_binding_capacity = view_binding_capacity,
  };
  loom_pipeline_plan_builder_t* builder = out_builder;

  iree_host_size_t edge_source_count = 0;
  if (!iree_host_size_checked_add(stage_port_capacity, write_capacity,
                                  &edge_source_count)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pipeline edge domain is too large");
  }
  iree_host_size_t edge_capacity = 0;
  if (!iree_host_size_checked_mul(edge_source_count, limits.instance_count,
                                  &edge_capacity) ||
      edge_capacity > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pipeline edge domain is too large");
  }
  builder->edge_capacity = (uint32_t)edge_capacity;
  iree_host_size_t group_port_capacity = 0;
  if (!iree_host_size_checked_mul(stage_port_capacity, 2,
                                  &group_port_capacity) ||
      !iree_host_size_checked_add(group_port_capacity, write_capacity,
                                  &group_port_capacity) ||
      group_port_capacity > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pipeline group-port domain is too large");
  }
  builder->group_port_capacity = (uint32_t)group_port_capacity;

#define LOOM_PIPELINE_PLAN_ALLOCATE(field, count)         \
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_allocate_array( \
      arena, (count), sizeof(*builder->field), (void**)&builder->field))
  LOOM_PIPELINE_PLAN_ALLOCATE(bindings, builder->binding_count);
  LOOM_PIPELINE_PLAN_ALLOCATE(binding_next_ports, builder->binding_count);
  LOOM_PIPELINE_PLAN_ALLOCATE(binding_views, builder->binding_view_capacity);
  LOOM_PIPELINE_PLAN_ALLOCATE(groups, builder->group_capacity);
  LOOM_PIPELINE_PLAN_ALLOCATE(instances, builder->instance_capacity);
  LOOM_PIPELINE_PLAN_ALLOCATE(stages, builder->stage_capacity);
  LOOM_PIPELINE_PLAN_ALLOCATE(stage_ports, builder->stage_port_capacity);
  LOOM_PIPELINE_PLAN_ALLOCATE(group_ports, builder->group_port_capacity);
  LOOM_PIPELINE_PLAN_ALLOCATE(flows, builder->flow_capacity);
  LOOM_PIPELINE_PLAN_ALLOCATE(flow_usage, builder->flow_capacity);
  LOOM_PIPELINE_PLAN_ALLOCATE(edges, builder->edge_capacity);
  LOOM_PIPELINE_PLAN_ALLOCATE(writes, builder->write_capacity);
  LOOM_PIPELINE_PLAN_ALLOCATE(view_bindings, builder->view_binding_capacity);
#undef LOOM_PIPELINE_PLAN_ALLOCATE

  if (builder->binding_count != 0) {
    memset(builder->bindings, 0,
           builder->binding_count * sizeof(*builder->bindings));
    memset(builder->binding_next_ports, 0,
           builder->binding_count * sizeof(*builder->binding_next_ports));
  }
  if (builder->flow_capacity != 0) {
    memset(builder->flow_usage, 0,
           builder->flow_capacity * sizeof(*builder->flow_usage));
  }
  return iree_ok_status();
}

iree_status_t loom_pipeline_plan_build(const loom_module_t* module,
                                       loom_func_like_t pipeline,
                                       const loom_value_fact_table_t* facts,
                                       loom_pipeline_plan_limits_t limits,
                                       iree_arena_allocator_t* arena,
                                       loom_pipeline_plan_t* out_plan) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT(loom_func_like_isa(pipeline));
  IREE_ASSERT_ARGUMENT(facts);
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_plan);
  *out_plan = (loom_pipeline_plan_t){0};
  IREE_ASSERT(loom_pipeline_def_isa(pipeline.op));

  loom_pipeline_plan_builder_t builder = {0};
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_builder_initialize(
      module, pipeline, facts, limits, arena, &builder));
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_parse_graph(&builder));
  *out_plan = (loom_pipeline_plan_t){
      .pipeline = pipeline,
      .bindings = builder.bindings,
      .binding_count = builder.binding_count,
      .binding_views = builder.binding_views,
      .binding_view_count = builder.binding_view_count,
      .groups = builder.groups,
      .group_count = builder.group_count,
      .instances = builder.instances,
      .instance_count = builder.instance_count,
      .stages = builder.stages,
      .stage_count = builder.stage_count,
      .stage_ports = builder.stage_ports,
      .stage_port_count = builder.stage_port_count,
      .group_ports = builder.group_ports,
      .group_port_count = builder.group_port_count,
      .flows = builder.flows,
      .flow_count = builder.flow_count,
      .edges = builder.edges,
      .edge_count = builder.edge_count,
  };
  return iree_ok_status();
}
