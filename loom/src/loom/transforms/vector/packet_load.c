// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/packet_load.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/module.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/util/fact_vector_partition.h"

static iree_status_t loom_vector_packet_exact_memory_load_plan(
    loom_target_legalization_context_t* context, const loom_op_t* load_op,
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    const loom_vector_packet_shape_t* shape, uint32_t* lane_counts,
    uint32_t* out_component_count, uint32_t* out_scalar_lane_count) {
  *out_component_count = 0;
  *out_scalar_lane_count = 0;
  iree_host_size_t aggregate_component_count = 0;
  iree_host_size_t fact_component_count = 0;
  iree_host_size_t operation_count = 0;
  bool has_vector_packet = false;
  uint32_t lane_offset = 0;
  while (lane_offset < shape->lane_count) {
    const uint32_t remaining_lane_count = shape->lane_count - lane_offset;
    uint32_t packet_lane_count =
        iree_min(remaining_lane_count, shape->chunk_lane_count);
    IREE_RETURN_IF_ERROR(loom_vector_packet_memory_chunk_lane_count_select(
        context, load_op, policy, vector_type, packet_lane_count,
        &packet_lane_count));
    if (packet_lane_count <= 1u) {
      iree_host_size_t scalar_op_count = 0;
      if (!iree_host_size_checked_add(fact_component_count,
                                      remaining_lane_count,
                                      &fact_component_count) ||
          !iree_host_size_checked_add(remaining_lane_count, 1u,
                                      &scalar_op_count) ||
          !iree_host_size_checked_add(operation_count, scalar_op_count,
                                      &operation_count) ||
          !iree_host_size_checked_add(aggregate_component_count, 1u,
                                      &aggregate_component_count)) {
        return iree_ok_status();
      }
      if (fact_component_count >
              LOOM_VALUE_FACT_VECTOR_PARTITION_EXPANSION_LIMIT ||
          operation_count > LOOM_VECTOR_PACKET_STATIC_OP_LIMIT) {
        return iree_ok_status();
      }
      lane_counts[aggregate_component_count - 1u] = 0;
      *out_scalar_lane_count = remaining_lane_count;
      lane_offset = shape->lane_count;
      break;
    }
    has_vector_packet |= packet_lane_count > 1u;
    if (!iree_host_size_checked_add(fact_component_count, 1u,
                                    &fact_component_count) ||
        !iree_host_size_checked_add(operation_count, 1u, &operation_count) ||
        !iree_host_size_checked_add(aggregate_component_count, 1u,
                                    &aggregate_component_count)) {
      return iree_ok_status();
    }
    if (fact_component_count >
            LOOM_VALUE_FACT_VECTOR_PARTITION_EXPANSION_LIMIT ||
        operation_count > LOOM_VECTOR_PACKET_STATIC_OP_LIMIT) {
      return iree_ok_status();
    }
    lane_counts[aggregate_component_count - 1u] = packet_lane_count;
    lane_offset += packet_lane_count;
  }
  if (aggregate_component_count > 1u &&
      !iree_host_size_checked_add(operation_count, 1u, &operation_count)) {
    return iree_ok_status();
  }
  if (!has_vector_packet ||
      fact_component_count > LOOM_VALUE_FACT_VECTOR_PARTITION_EXPANSION_LIMIT ||
      operation_count > LOOM_VECTOR_PACKET_STATIC_OP_LIMIT) {
    return iree_ok_status();
  }
  *out_component_count = (uint32_t)aggregate_component_count;
  return iree_ok_status();
}

iree_status_t loom_vector_packet_load_plan_allocate(
    loom_target_legalization_context_t* context, loom_op_t* source_op,
    const loom_target_vector_packet_policy_t* policy,
    const loom_vector_memory_footprint_t* footprint,
    const loom_vector_packet_shape_t* shape,
    loom_vector_memory_cache_policy_t cache_policy,
    loom_vector_packet_load_plan_t** out_plan) {
  *out_plan = NULL;
  uint32_t lane_counts[LOOM_VECTOR_PACKET_STATIC_OP_LIMIT];
  uint32_t component_count = shape->chunk_count;
  uint32_t scalar_lane_count = 0;
  if (policy != NULL) {
    IREE_RETURN_IF_ERROR(loom_vector_packet_exact_memory_load_plan(
        context, source_op, policy, footprint->vector_type, shape, lane_counts,
        &component_count, &scalar_lane_count));
    if (component_count == 0) {
      return iree_ok_status();
    }
  } else if (shape->chunk_lane_count == 1u) {
    component_count = 1u;
    scalar_lane_count = shape->lane_count;
  } else if (shape->lane_count % shape->chunk_lane_count == 1u) {
    scalar_lane_count = 1u;
  }

  iree_host_size_t allocation_size = 0;
  iree_host_size_t lane_counts_offset = 0;
  iree_host_size_t components_offset = 0;
  iree_host_size_t scalar_values_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_vector_packet_load_plan_t), &allocation_size,
      IREE_STRUCT_FIELD(component_count, uint32_t, &lane_counts_offset),
      IREE_STRUCT_FIELD(component_count, loom_value_id_t, &components_offset),
      IREE_STRUCT_FIELD(scalar_lane_count, loom_value_id_t,
                        &scalar_values_offset)));
  loom_vector_packet_load_plan_t* plan = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(context->arena, allocation_size, (void**)&plan));
  uint32_t* retained_lane_counts =
      (uint32_t*)((uint8_t*)plan + lane_counts_offset);
  *plan = (loom_vector_packet_load_plan_t){
      .source_op = source_op,
      .view = footprint->view,
      .vector_type = footprint->vector_type,
      .dynamic_indices = footprint->dynamic_indices,
      .static_indices = footprint->static_indices,
      .cache_policy = cache_policy,
      .component_count = component_count,
      .scalar_lane_count = scalar_lane_count,
      .lane_counts = retained_lane_counts,
      .components = (loom_value_id_t*)((uint8_t*)plan + components_offset),
      .scalar_values =
          (loom_value_id_t*)((uint8_t*)plan + scalar_values_offset),
  };
  if (policy != NULL) {
    memcpy(retained_lane_counts, lane_counts,
           component_count * sizeof(*lane_counts));
  } else {
    for (uint32_t i = 0; i < component_count; ++i) {
      retained_lane_counts[i] =
          iree_min(shape->lane_count - i * shape->chunk_lane_count,
                   shape->chunk_lane_count);
      if (retained_lane_counts[i] == 1u) {
        retained_lane_counts[i] = 0;
      }
    }
  }
  *out_plan = plan;
  return iree_ok_status();
}

iree_status_t loom_vector_packet_load_plan_select(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy,
    uint32_t maximum_lane_count, loom_vector_packet_load_plan_t** out_plan) {
  *out_plan = NULL;

  loom_vector_memory_footprint_t footprint = {0};
  if (!loom_vector_memory_footprint_describe(
          loom_vector_packet_fact_context(context), context->module, op,
          &footprint) ||
      footprint.kind != LOOM_VECTOR_MEMORY_FOOTPRINT_DENSE) {
    return iree_ok_status();
  }
  loom_vector_packet_shape_t shape = {0};
  loom_vector_memory_cache_policy_t cache_policy = {0};
  if (!loom_vector_packet_shape_from_type(context, policy,
                                          footprint.vector_type, &shape)) {
    uint32_t lane_count = 0;
    if (maximum_lane_count == 0 ||
        !loom_vector_packet_static_lane_count(footprint.vector_type,
                                              &lane_count) ||
        maximum_lane_count >= lane_count) {
      return iree_ok_status();
    }
    shape = (loom_vector_packet_shape_t){
        .lane_count = lane_count,
        .chunk_lane_count = maximum_lane_count,
        .chunk_count = (lane_count - 1u) / maximum_lane_count + 1u,
    };
  } else if (maximum_lane_count != 0 &&
             maximum_lane_count < shape.chunk_lane_count) {
    shape.chunk_lane_count = maximum_lane_count;
    shape.chunk_count = (shape.lane_count - 1u) / shape.chunk_lane_count + 1u;
  }
  bool packet_supported = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_memory_shape_constrain(
      context, op, policy, footprint.vector_type, &shape, &packet_supported));
  if (!packet_supported ||
      !loom_vector_memory_cache_policy_from_op(context->module, op,
                                               &cache_policy) ||
      !loom_vector_packet_memory_can_build_chunk_origins(context, &footprint,
                                                         &shape)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_vector_packet_load_plan_allocate(
      context, op, policy, &footprint, &shape, cache_policy, out_plan));
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_load_build_origin(
    loom_target_legalization_context_t* context,
    const loom_vector_packet_load_plan_t* plan, uint32_t lane_offset,
    const loom_value_id_t** out_dynamic_indices,
    iree_host_size_t* out_dynamic_index_count,
    const int64_t** out_static_indices,
    iree_host_size_t* out_static_index_count) {
  bool origin_built = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_memory_build_chunk_origin(
      context, plan->dynamic_indices, plan->static_indices, plan->source_op,
      lane_offset, out_dynamic_indices, out_dynamic_index_count,
      out_static_indices, out_static_index_count, &origin_built));
  IREE_ASSERT_TRUE(origin_built);
  return iree_ok_status();
}

iree_status_t loom_vector_packet_load_plan_emit(
    loom_target_legalization_context_t* context,
    loom_vector_packet_load_plan_t* plan) {
  loom_op_t* load_op = plan->source_op;
  const loom_vector_memory_cache_policy_t cache_policy = plan->cache_policy;
  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, load_op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t* components = plan->components;
  loom_value_id_t* scalar_tail = plan->scalar_values;
  uint32_t component_count = 0;
  uint32_t lane_offset = 0;
  while (component_count < plan->component_count) {
    const uint32_t packet_lane_count = plan->lane_counts[component_count];
    if (packet_lane_count == 0) {
      const loom_type_t scalar_type =
          loom_type_scalar(loom_type_element_type(plan->vector_type));
      for (uint32_t i = 0; i < plan->scalar_lane_count; ++i) {
        const loom_value_id_t* dynamic_indices = NULL;
        iree_host_size_t dynamic_index_count = 0;
        const int64_t* static_indices = NULL;
        iree_host_size_t static_index_count = 0;
        IREE_RETURN_IF_ERROR(loom_vector_packet_load_build_origin(
            context, plan, lane_offset + i, &dynamic_indices,
            &dynamic_index_count, &static_indices, &static_index_count));
        loom_op_t* scalar_load_op = NULL;
        IREE_RETURN_IF_ERROR(loom_view_load_build(
            builder, cache_policy.build_flags,
            loom_vector_load_memory_flags(load_op), plan->view, dynamic_indices,
            dynamic_index_count, static_indices, static_index_count,
            cache_policy.cache_scope, cache_policy.cache_temporal, scalar_type,
            load_op->location, &scalar_load_op));
        scalar_tail[i] = loom_view_load_result(scalar_load_op);
      }
      const loom_type_t tail_type =
          loom_vector_packet_type(plan->vector_type, plan->scalar_lane_count);
      loom_op_t* tail_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
          builder, scalar_tail, plan->scalar_lane_count, tail_type,
          load_op->location, &tail_op));
      components[component_count++] = loom_vector_from_elements_result(tail_op);
      break;
    }

    const loom_value_id_t* dynamic_indices = NULL;
    iree_host_size_t dynamic_index_count = 0;
    const int64_t* static_indices = NULL;
    iree_host_size_t static_index_count = 0;
    IREE_RETURN_IF_ERROR(loom_vector_packet_load_build_origin(
        context, plan, lane_offset, &dynamic_indices, &dynamic_index_count,
        &static_indices, &static_index_count));
    const loom_type_t packet_type =
        loom_vector_packet_type(plan->vector_type, packet_lane_count);
    loom_op_t* packet_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_load_build(
        builder, cache_policy.build_flags,
        loom_vector_load_memory_flags(load_op), plan->view, dynamic_indices,
        dynamic_index_count, static_indices, static_index_count,
        cache_policy.cache_scope, cache_policy.cache_temporal, packet_type,
        load_op->location, &packet_op));
    components[component_count++] = loom_vector_load_result(packet_op);
    lane_offset += packet_lane_count;
  }

  loom_value_id_t replacement = components[0];
  if (component_count > 1) {
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_concat_build(
        builder, /*axis=*/0, components, component_count, plan->vector_type,
        load_op->location, &concat_op));
    replacement = loom_vector_concat_result(concat_op);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, load_op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, load_op, &replacement, 1));
  return iree_ok_status();
}
