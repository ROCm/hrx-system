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
#include "loom/ops/group/ops.h"
#include "loom/ops/pipeline/ops.h"
#include "loom/ops/type_registry.h"

enum { LOOM_PIPELINE_PLAN_IMPLICIT_MINIMUM_CAPACITY = 1 };

typedef enum loom_pipeline_external_flow_kind_e {
  LOOM_PIPELINE_EXTERNAL_FLOW_KIND_READ = 0,
  LOOM_PIPELINE_EXTERNAL_FLOW_KIND_SCATTER = 1,
} loom_pipeline_external_flow_kind_t;

typedef struct loom_pipeline_plan_view_binding_t {
  // Source view SSA identity.
  loom_value_id_t source_value;

  // Launch binding ordinal referenced by the view.
  uint32_t binding_index;
} loom_pipeline_plan_view_binding_t;

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

  // Scheduling group table.
  loom_pipeline_plan_group_t* groups;

  // Number of defined scheduling groups.
  uint32_t group_count;

  // Maximum scheduling groups allocated.
  uint32_t group_capacity;

  // Whether each group has been claimed by a physical stage.
  bool* group_materialized;

  // Resident callable instance table.
  loom_pipeline_plan_instance_t* instances;

  // Number of defined resident instances.
  uint32_t instance_count;

  // Maximum resident instances accepted by the materializer.
  uint32_t instance_capacity;

  // Typed logical flow table.
  loom_pipeline_plan_flow_t* flows;

  // Number of defined logical flows.
  uint32_t flow_count;

  // Maximum logical flows allocated.
  uint32_t flow_capacity;

  // Whether each flow version has acquired its physical consumer.
  bool* flow_consumed;

  // Concrete point-to-point edge table.
  loom_pipeline_plan_edge_t* edges;

  // Number of defined edges.
  uint32_t edge_count;

  // Maximum concrete edges allocated.
  uint32_t edge_capacity;

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
                                               "partition source", out_type);
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
  builder->flows[index] = flow;
  if (out_index != NULL) *out_index = index;
}

static iree_status_t loom_pipeline_plan_claim_flow(
    loom_pipeline_plan_builder_t* builder, loom_value_id_t value_id,
    uint32_t* out_index) {
  IREE_RETURN_IF_ERROR(
      loom_pipeline_plan_lookup_flow(builder, value_id, out_index));
  if (builder->flow_consumed[*out_index]) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pipeline flow has more than one physical consumer");
  }
  builder->flow_consumed[*out_index] = true;
  return iree_ok_status();
}

static void loom_pipeline_plan_append_edge(
    loom_pipeline_plan_builder_t* builder, loom_pipeline_plan_edge_t edge) {
  IREE_ASSERT_LT(builder->edge_count, builder->edge_capacity);
  builder->edges[builder->edge_count++] = edge;
}

static iree_status_t loom_pipeline_plan_binding_for_view(
    const loom_pipeline_plan_builder_t* builder, loom_value_id_t view_value,
    uint32_t* out_binding_index) {
  for (uint32_t i = 0; i < builder->view_binding_count; ++i) {
    if (builder->view_bindings[i].source_value == view_value) {
      *out_binding_index = builder->view_bindings[i].binding_index;
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "pipeline external flow must use a zero-offset launch view");
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
  uint32_t byte_offset = 0;
  IREE_RETURN_IF_ERROR(
      loom_pipeline_plan_exact_u32(builder, loom_buffer_view_byte_offset(op),
                                   "view byte offset", &byte_offset));
  if (byte_offset != 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "concrete pipeline plans currently require zero-offset launch views");
  }
  IREE_ASSERT_LT(builder->view_binding_count, builder->view_binding_capacity);
  builder->view_bindings[builder->view_binding_count++] =
      (loom_pipeline_plan_view_binding_t){
          .source_value = loom_buffer_view_result(op),
          .binding_index = argument_index - specialization_count,
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
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_binding_for_view(builder, source_view,
                                                           &binding_index));
  builder->bindings[binding_index].access |=
      LOOM_PIPELINE_BINDING_ACCESS_FLAG_READ;

  loom_type_t partition_source_type = loom_type_none();
  uint32_t port_count = builder->groups[group_index].lane_count;
  if (partitioned) {
    const loom_type_t view_type =
        loom_module_value_type(builder->module, source_view);
    uint32_t leading_dimension = 0;
    IREE_RETURN_IF_ERROR(loom_pipeline_plan_exact_dimension(
        builder, view_type, 0, &leading_dimension));
    if (leading_dimension != builder->groups[group_index].lane_count) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipeline scatter leading dimension must equal group cardinality");
    }
    IREE_RETURN_IF_ERROR(loom_pipeline_plan_view_tile_type(
        builder, view_type, &partition_source_type));
    port_count = 1;
  }
  if (builder->binding_next_ports[binding_index] > UINT32_MAX - port_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pipeline binding port ordinal overflow");
  }
  const uint32_t first_port = builder->binding_next_ports[binding_index];
  builder->binding_next_ports[binding_index] += port_count;
  loom_type_t tile_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_pipeline_plan_flow_tile_type(builder, result, &tile_type));
  loom_pipeline_plan_define_flow(
      builder, result,
      (loom_pipeline_plan_flow_t){
          .tile_type = tile_type,
          .group_index = group_index,
          .minimum_capacity = LOOM_PIPELINE_PLAN_IMPLICIT_MINIMUM_CAPACITY,
          .producer_kind = LOOM_PIPELINE_ENDPOINT_KIND_BINDING,
          .binding_index = binding_index,
          .instance_start = UINT32_MAX,
          .producer_port = first_port,
          .partition_source_type = partition_source_type,
          .partitioned = partitioned,
      },
      NULL);
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_materialize_group_instances(
    loom_pipeline_plan_builder_t* builder, uint32_t group_index,
    loom_symbol_ref_t entry, uint32_t* out_instance_start) {
  loom_pipeline_plan_group_t* group = &builder->groups[group_index];
  if (builder->group_materialized[group_index]) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "concrete pipeline plans currently map each group to one stage");
  }
  if (group->lane_count >
      builder->instance_capacity - builder->instance_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "pipeline requires more resident instances than the target supports");
  }
  group->instance_start = builder->instance_count;
  builder->group_materialized[group_index] = true;
  for (uint32_t lane = 0; lane < group->lane_count; ++lane) {
    builder->instances[builder->instance_count++] =
        (loom_pipeline_plan_instance_t){
            .group_index = group_index,
            .lane = lane,
            .entry = entry,
        };
  }
  *out_instance_start = group->instance_start;
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_connect_pointwise_flow(
    loom_pipeline_plan_builder_t* builder, loom_value_id_t flow_value,
    uint32_t target_group_index, uint32_t target_instance_start,
    uint32_t target_port) {
  uint32_t flow_index = 0;
  IREE_RETURN_IF_ERROR(
      loom_pipeline_plan_claim_flow(builder, flow_value, &flow_index));
  const loom_pipeline_plan_flow_t* flow = &builder->flows[flow_index];
  const loom_pipeline_plan_group_t* group =
      &builder->groups[target_group_index];
  if (flow->group_index != target_group_index ||
      (flow->instance_count != 0 &&
       flow->instance_count != group->lane_count)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "pointwise pipeline input must belong to its stage group");
  }
  for (uint32_t lane = 0; lane < group->lane_count; ++lane) {
    loom_pipeline_plan_append_edge(
        builder,
        (loom_pipeline_plan_edge_t){
            .flow_index = flow_index,
            .source_kind = flow->producer_kind,
            .source_index =
                flow->producer_kind == LOOM_PIPELINE_ENDPOINT_KIND_BINDING
                    ? flow->binding_index
                    : flow->instance_start + lane,
            .source_port =
                flow->producer_port +
                (flow->producer_kind == LOOM_PIPELINE_ENDPOINT_KIND_BINDING &&
                         !flow->partitioned
                     ? lane
                     : 0),
            .partition_lane = lane,
            .target_kind = LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE,
            .target_index = target_instance_start + lane,
            .target_port = target_port,
        });
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_define_instance_outputs(
    loom_pipeline_plan_builder_t* builder, loom_value_slice_t outputs,
    uint32_t group_index, uint32_t instance_start, uint32_t first_output_port) {
  const uint32_t instance_count = builder->groups[group_index].lane_count;
  for (uint16_t i = 0; i < outputs.count; ++i) {
    loom_type_t tile_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_pipeline_plan_flow_tile_type(
        builder, outputs.values[i], &tile_type));
    loom_pipeline_plan_define_flow(
        builder, outputs.values[i],
        (loom_pipeline_plan_flow_t){
            .tile_type = tile_type,
            .group_index = group_index,
            .minimum_capacity = LOOM_PIPELINE_PLAN_IMPLICIT_MINIMUM_CAPACITY,
            .producer_kind = LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE,
            .binding_index = UINT32_MAX,
            .instance_start = instance_start,
            .instance_count = instance_count,
            .producer_port = first_output_port + i,
            .partition_source_type = loom_type_none(),
        },
        NULL);
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_parse_stage(
    loom_pipeline_plan_builder_t* builder, const loom_op_t* op) {
  uint32_t group_index = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_lookup_group(
      builder, loom_pipeline_stage_group(op), &group_index));
  uint32_t instance_start = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_materialize_group_instances(
      builder, group_index, loom_pipeline_stage_entry(op), &instance_start));
  const loom_value_slice_t inputs = loom_pipeline_stage_inputs(op);
  for (uint16_t i = 0; i < inputs.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_pipeline_plan_connect_pointwise_flow(
        builder, inputs.values[i], group_index, instance_start, i));
  }
  return loom_pipeline_plan_define_instance_outputs(
      builder, loom_pipeline_stage_outputs(op), group_index, instance_start,
      inputs.count);
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
  const uint64_t expanded_source_count =
      (uint64_t)source_inputs.count * source_group->lane_count;
  const uint64_t expected_argument_count =
      expanded_source_count + target_inputs.count + outputs.count;
  if (expected_argument_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pipeline reduction argument count is too large");
  }
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_validate_entry_argument_count(
      builder, loom_pipeline_reduce_entry(op),
      (uint32_t)expected_argument_count));

  uint32_t target_instance_start = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_materialize_group_instances(
      builder, target_group_index, loom_pipeline_reduce_entry(op),
      &target_instance_start));
  uint32_t target_port = 0;
  for (uint16_t i = 0; i < source_inputs.count; ++i) {
    uint32_t flow_index = 0;
    IREE_RETURN_IF_ERROR(loom_pipeline_plan_claim_flow(
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
      loom_pipeline_plan_append_edge(
          builder, (loom_pipeline_plan_edge_t){
                       .flow_index = flow_index,
                       .source_kind = LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE,
                       .source_index = flow->instance_start + lane,
                       .source_port = flow->producer_port,
                       .target_kind = LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE,
                       .target_index = target_instance_start,
                       .target_port = target_port++,
                   });
    }
  }
  for (uint16_t i = 0; i < target_inputs.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_pipeline_plan_connect_pointwise_flow(
        builder, target_inputs.values[i], target_group_index,
        target_instance_start, target_port++));
  }
  return loom_pipeline_plan_define_instance_outputs(
      builder, outputs, target_group_index, target_instance_start, target_port);
}

static iree_status_t loom_pipeline_plan_parse_buffer(
    loom_pipeline_plan_builder_t* builder, const loom_op_t* op) {
  uint32_t source_flow_index = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_claim_flow(
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
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_claim_flow(
      builder, loom_pipeline_write_source(op), &flow_index));
  const loom_pipeline_plan_flow_t* flow = &builder->flows[flow_index];
  if (flow->producer_kind != LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE ||
      flow->instance_count != 1 ||
      builder->groups[flow->group_index].lane_count != 1) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "concrete pipeline writes currently require one source lane");
  }
  uint32_t binding_index = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_binding_for_view(
      builder, loom_pipeline_write_target(op), &binding_index));
  builder->bindings[binding_index].access |=
      LOOM_PIPELINE_BINDING_ACCESS_FLAG_WRITE;
  if (builder->binding_next_ports[binding_index] == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pipeline binding port ordinal overflow");
  }
  const uint32_t target_port = builder->binding_next_ports[binding_index]++;
  loom_pipeline_plan_append_edge(
      builder, (loom_pipeline_plan_edge_t){
                   .flow_index = flow_index,
                   .source_kind = LOOM_PIPELINE_ENDPOINT_KIND_INSTANCE,
                   .source_index = flow->instance_start,
                   .source_port = flow->producer_port,
                   .target_kind = LOOM_PIPELINE_ENDPOINT_KIND_BINDING,
                   .target_index = binding_index,
                   .target_port = target_port,
               });
  return iree_ok_status();
}

static bool loom_pipeline_plan_op_is_compile_time(
    const loom_pipeline_plan_builder_t* builder, const loom_op_t* op) {
  if (op->result_count == 0 || loom_op_may_write(builder->module, op)) {
    return false;
  }
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_facts_t element_facts = loom_value_facts_unknown();
    int64_t value = 0;
    if (!loom_value_facts_query_all_equal_element(
            &builder->facts->context,
            loom_value_fact_table_lookup(builder->facts,
                                         loom_op_results(op)[i]),
            &element_facts) ||
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

  if (builder->instance_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "pipeline has no resident instances");
  }
  for (uint32_t i = 0; i < builder->group_count; ++i) {
    if (!builder->group_materialized[i]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipeline contains a group with no resident stage");
    }
  }
  for (uint32_t i = 0; i < builder->flow_count; ++i) {
    if (!builder->flow_consumed[i]) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "pipeline contains a flow with no physical consumer");
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_pipeline_plan_measure_graph(
    loom_func_like_t pipeline, uint32_t* out_group_count,
    uint32_t* out_flow_count, uint32_t* out_view_count) {
  *out_group_count = 0;
  *out_flow_count = 0;
  *out_view_count = 0;
  loom_region_t* body = loom_func_like_body(pipeline);
  if (body == NULL || body->block_count != 1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "concrete pipeline planning requires one graph block");
  }

  uint64_t group_count = 0;
  uint64_t flow_count = 0;
  uint64_t view_count = 0;
  const loom_block_t* block = loom_region_const_entry_block(body);
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_group_create_isa(op)) ++group_count;
    if (loom_buffer_view_isa(op)) ++view_count;
    if (loom_pipeline_scatter_isa(op) || loom_pipeline_read_isa(op) ||
        loom_pipeline_stage_isa(op) || loom_pipeline_buffer_isa(op) ||
        loom_pipeline_reduce_isa(op)) {
      flow_count += op->result_count;
    }
  }
  if (group_count > UINT32_MAX || flow_count > UINT32_MAX ||
      view_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pipeline graph entity count is too large");
  }
  *out_group_count = (uint32_t)group_count;
  *out_flow_count = (uint32_t)flow_count;
  *out_view_count = (uint32_t)view_count;
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
  uint32_t view_binding_capacity = 0;
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_measure_graph(
      pipeline, &group_capacity, &flow_capacity, &view_binding_capacity));
  *out_builder = (loom_pipeline_plan_builder_t){
      .module = module,
      .pipeline = pipeline,
      .facts = facts,
      .arena = arena,
      .binding_count = argument_count - (uint32_t)specialization_count,
      .group_capacity = group_capacity,
      .instance_capacity = limits.instance_count,
      .flow_capacity = flow_capacity,
      .view_binding_capacity = view_binding_capacity,
  };
  loom_pipeline_plan_builder_t* builder = out_builder;

  iree_host_size_t edge_capacity = 0;
  if (!iree_host_size_checked_mul(flow_capacity, limits.instance_count,
                                  &edge_capacity) ||
      edge_capacity > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pipeline edge domain is too large");
  }
  builder->edge_capacity = (uint32_t)edge_capacity;

#define LOOM_PIPELINE_PLAN_ALLOCATE(field, count)         \
  IREE_RETURN_IF_ERROR(loom_pipeline_plan_allocate_array( \
      arena, (count), sizeof(*builder->field), (void**)&builder->field))
  LOOM_PIPELINE_PLAN_ALLOCATE(bindings, builder->binding_count);
  LOOM_PIPELINE_PLAN_ALLOCATE(binding_next_ports, builder->binding_count);
  LOOM_PIPELINE_PLAN_ALLOCATE(groups, builder->group_capacity);
  LOOM_PIPELINE_PLAN_ALLOCATE(group_materialized, builder->group_capacity);
  LOOM_PIPELINE_PLAN_ALLOCATE(instances, builder->instance_capacity);
  LOOM_PIPELINE_PLAN_ALLOCATE(flows, builder->flow_capacity);
  LOOM_PIPELINE_PLAN_ALLOCATE(flow_consumed, builder->flow_capacity);
  LOOM_PIPELINE_PLAN_ALLOCATE(edges, builder->edge_capacity);
  LOOM_PIPELINE_PLAN_ALLOCATE(view_bindings, builder->view_binding_capacity);
#undef LOOM_PIPELINE_PLAN_ALLOCATE

  if (builder->binding_count != 0) {
    memset(builder->bindings, 0,
           builder->binding_count * sizeof(*builder->bindings));
    memset(builder->binding_next_ports, 0,
           builder->binding_count * sizeof(*builder->binding_next_ports));
  }
  if (builder->group_capacity != 0) {
    memset(builder->group_materialized, 0,
           builder->group_capacity * sizeof(*builder->group_materialized));
  }
  if (builder->flow_capacity != 0) {
    memset(builder->flow_consumed, 0,
           builder->flow_capacity * sizeof(*builder->flow_consumed));
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
      .groups = builder.groups,
      .group_count = builder.group_count,
      .instances = builder.instances,
      .instance_count = builder.instance_count,
      .flows = builder.flows,
      .flow_count = builder.flow_count,
      .edges = builder.edges,
      .edge_count = builder.edge_count,
  };
  return iree_ok_status();
}
