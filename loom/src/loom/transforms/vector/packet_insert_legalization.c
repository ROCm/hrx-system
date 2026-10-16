// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/packet_insert_legalization.h"

#include <stdint.h>

#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/combining.h"
#include "loom/ops/vector/ops.h"
#include "loom/transforms/vector/packet_materialization.h"
#include "loom/transforms/vector/packet_plan.h"
#include "loom/util/fact_table.h"

// Rewrites an insertion over a retained exact partition without reconstructing
// the logical vector through slices. Vector components stay in native target
// carriers, and adjacent scalar components are rebuilt as one structural tail.
// Planning completes before mutation so an unavailable carrier or excessive
// expansion leaves the authored operation untouched.
static iree_status_t loom_vector_packet_legalize_partition_insert(
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_value_fact_vector_partition_t partition, bool has_static_index,
    uint32_t static_index, bool has_dynamic_index,
    const loom_vector_packet_insert_materializer_t* insert_materializer,
    bool* out_rewritten) {
  *out_rewritten = false;
  const loom_value_id_t value = loom_vector_insert_value(op);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_insert_result(op));
  const loom_type_t element_type =
      loom_type_scalar(loom_type_element_type(result_type));
  uint32_t result_lane_count = 0;
  if (partition.count == 0 ||
      !loom_vector_packet_static_lane_count(result_type, &result_lane_count) ||
      partition.lane_count != result_lane_count ||
      (has_dynamic_index &&
       (insert_materializer == NULL ||
        insert_materializer->operation_count == NULL ||
        insert_materializer->materialize_dynamic == NULL))) {
    return iree_ok_status();
  }

  iree_host_size_t operation_count = has_static_index ? 1u : 0u;
  uint32_t aggregate_count = 0;
  uint32_t scalar_run_count = 0;
  uint32_t lane_offset = 0;
  bool previous_component_was_scalar = false;
  for (uint32_t i = 0; i < partition.count; ++i) {
    const loom_value_fact_vector_partition_component_t component =
        partition.components[i];
    const loom_type_t component_type =
        loom_module_value_type(context->module, component.value);
    const bool component_is_scalar = loom_type_is_scalar(component_type);
    if (component_is_scalar) {
      if (component.lane_count != 1 ||
          !loom_type_equal(component_type, element_type)) {
        return iree_ok_status();
      }
      if (!previous_component_was_scalar) {
        ++aggregate_count;
        ++scalar_run_count;
      }
    } else {
      uint32_t component_lane_count = 0;
      if (!loom_vector_packet_static_lane_count(component_type,
                                                &component_lane_count) ||
          component_lane_count != component.lane_count ||
          !loom_type_element_type_equals(component_type, result_type) ||
          !loom_target_legalization_has_source_vector_carrier(
              context, component_type, component.value)) {
        return iree_ok_status();
      }
      ++aggregate_count;
    }
    if (has_dynamic_index) {
      const uint8_t component_operation_count =
          insert_materializer->operation_count(insert_materializer->user_data,
                                               context, component_type,
                                               lane_offset);
      if (component_operation_count == 0 ||
          !iree_host_size_checked_add(
              operation_count, component_operation_count, &operation_count)) {
        return iree_ok_status();
      }
    }
    if (lane_offset > result_lane_count ||
        component.lane_count > result_lane_count - lane_offset) {
      return iree_ok_status();
    }
    lane_offset += component.lane_count;
    previous_component_was_scalar = component_is_scalar;
  }
  if (lane_offset != result_lane_count || aggregate_count == 0 ||
      !iree_host_size_checked_add(operation_count, scalar_run_count,
                                  &operation_count) ||
      (aggregate_count > 1 &&
       !iree_host_size_checked_add(operation_count, 1u, &operation_count)) ||
      operation_count > LOOM_VECTOR_PACKET_STATIC_OP_LIMIT) {
    return iree_ok_status();
  }

  loom_value_id_t* component_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, partition.count, sizeof(*component_values),
      (void**)&component_values));
  loom_value_id_t* aggregate_values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, aggregate_count, sizeof(*aggregate_values),
      (void**)&aggregate_values));

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  lane_offset = 0;
  for (uint32_t i = 0; i < partition.count; ++i) {
    const loom_value_fact_vector_partition_component_t component =
        partition.components[i];
    const loom_type_t component_type =
        loom_module_value_type(context->module, component.value);
    loom_value_id_t component_result = component.value;
    if (has_dynamic_index) {
      IREE_RETURN_IF_ERROR(insert_materializer->materialize_dynamic(
          insert_materializer->user_data, context, value, component.value,
          loom_vector_insert_indices(op).values[0], lane_offset, component_type,
          op->location, &component_result));
    } else if (static_index >= lane_offset &&
               static_index - lane_offset < component.lane_count) {
      if (loom_type_is_scalar(component_type)) {
        component_result = value;
      } else {
        const int64_t component_index = static_index - lane_offset;
        loom_op_t* insert_op = NULL;
        IREE_RETURN_IF_ERROR(loom_vector_insert_build(
            builder, value, component.value, /*indices=*/NULL,
            /*indices_count=*/0, &component_index,
            /*static_indices_count=*/1, component_type, op->location,
            &insert_op));
        component_result = loom_vector_insert_result(insert_op);
      }
    }
    component_values[i] = component_result;
    lane_offset += component.lane_count;
  }

  uint32_t aggregate_index = 0;
  for (uint32_t component_index = 0; component_index < partition.count;) {
    const loom_type_t component_type = loom_module_value_type(
        context->module, partition.components[component_index].value);
    if (!loom_type_is_scalar(component_type)) {
      aggregate_values[aggregate_index++] = component_values[component_index++];
      continue;
    }
    const uint32_t scalar_run_start = component_index;
    do {
      ++component_index;
    } while (
        component_index < partition.count &&
        loom_type_is_scalar(loom_module_value_type(
            context->module, partition.components[component_index].value)));
    const uint32_t scalar_run_length = component_index - scalar_run_start;
    const loom_type_t scalar_run_type = loom_type_shaped_1d(
        LOOM_TYPE_VECTOR, loom_type_element_type(result_type),
        loom_dim_pack_static(scalar_run_length), /*encoding_id=*/0);
    loom_op_t* from_elements_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
        builder, &component_values[scalar_run_start], scalar_run_length,
        scalar_run_type, op->location, &from_elements_op));
    aggregate_values[aggregate_index++] =
        loom_vector_from_elements_result(from_elements_op);
  }
  IREE_ASSERT_EQ(aggregate_index, aggregate_count);

  loom_value_id_t replacement = aggregate_values[0];
  if (aggregate_count > 1) {
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_concat_build(
        builder, /*axis=*/0, aggregate_values, aggregate_count, result_type,
        op->location, &concat_op));
    replacement = loom_vector_concat_result(concat_op);
  }
  loom_value_fact_vector_partition_component_t* replacement_components = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_vector_partition_component_scratch(
      rewriter->fact_table, partition.count, &replacement_components));
  for (uint32_t i = 0; i < partition.count; ++i) {
    replacement_components[i] = (loom_value_fact_vector_partition_component_t){
        .value = component_values[i],
        .lane_count = partition.components[i].lane_count,
    };
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_table_define_vector_partition(
      rewriter->fact_table, replacement,
      (loom_value_fact_vector_partition_t){
          .components = replacement_components,
          .count = partition.count,
          .lane_count = partition.lane_count,
      },
      /*inout_changed=*/NULL));
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}

iree_status_t loom_vector_packet_legalize_insert(
    loom_target_legalization_context_t* context, loom_op_t* op,
    const loom_target_vector_packet_policy_t* policy,
    const loom_vector_packet_insert_materializer_t* insert_materializer,
    bool* out_rewritten) {
  *out_rewritten = false;

  const loom_value_id_t value = loom_vector_insert_value(op);
  const loom_value_id_t dest = loom_vector_insert_dest(op);
  const loom_type_t value_type = loom_module_value_type(context->module, value);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_insert_result(op));
  loom_vector_packet_shape_t shape = {0};
  if (!loom_type_is_scalar(value_type) ||
      !loom_vector_packet_shape_from_type(context, policy, result_type,
                                          &shape)) {
    return iree_ok_status();
  }

  const loom_attribute_t static_indices = loom_vector_insert_static_indices(op);
  const loom_value_slice_t dynamic_indices = loom_vector_insert_indices(op);
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1) {
    return iree_ok_status();
  }
  const bool has_static_index =
      dynamic_indices.count == 0 && static_indices.i64_array[0] >= 0 &&
      (uint64_t)static_indices.i64_array[0] < shape.lane_count;
  const bool has_dynamic_index =
      dynamic_indices.count == 1 && static_indices.i64_array[0] == INT64_MIN;
  if (!has_static_index && !has_dynamic_index) {
    return iree_ok_status();
  }

  const uint32_t static_index =
      has_static_index ? (uint32_t)static_indices.i64_array[0] : 0u;
  if (!loom_vector_packet_shape_has_source_carriers(context, result_type,
                                                    &shape)) {
    const loom_value_fact_vector_partition_t partition =
        loom_value_fact_table_query_vector_partition(context->fact_table, dest);
    return loom_vector_packet_legalize_partition_insert(
        context, op, partition, has_static_index, static_index,
        has_dynamic_index, insert_materializer, out_rewritten);
  }

  if (has_dynamic_index && insert_materializer != NULL) {
    if (insert_materializer->operation_count == NULL ||
        insert_materializer->materialize_dynamic == NULL) {
      return iree_ok_status();
    }
    iree_host_size_t operation_count = 1u;
    for (uint32_t lane_offset = 0; lane_offset < shape.lane_count;
         lane_offset += shape.chunk_lane_count) {
      const uint32_t packet_lane_count =
          iree_min(shape.lane_count - lane_offset, shape.chunk_lane_count);
      const loom_type_t packet_type =
          loom_vector_packet_type(result_type, packet_lane_count);
      const uint8_t packet_operation_count =
          insert_materializer->operation_count(insert_materializer->user_data,
                                               context, packet_type,
                                               lane_offset);
      if (packet_operation_count == 0 ||
          !iree_host_size_checked_add(
              operation_count, 1u + packet_operation_count, &operation_count)) {
        return iree_ok_status();
      }
    }
    if (operation_count > LOOM_VECTOR_PACKET_STATIC_OP_LIMIT) {
      return iree_ok_status();
    }
  } else if (!loom_vector_packet_static_operation_count_is_bounded(
                 has_dynamic_index ? 9u : 3u, &shape)) {
    return iree_ok_status();
  }

  loom_value_id_t* packets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, shape.chunk_count, sizeof(*packets), (void**)&packets));

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);

  loom_value_id_t* packet_upper_bounds = NULL;
  loom_value_id_t* packet_conditions = NULL;
  loom_value_id_t first_packet_lower_bound = LOOM_VALUE_ID_INVALID;
  if (has_dynamic_index && insert_materializer == NULL) {
    loom_value_id_t* boundary_storage = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->arena, 2u * shape.chunk_count, sizeof(*boundary_storage),
        (void**)&boundary_storage));
    packet_upper_bounds = boundary_storage;
    packet_conditions = boundary_storage + shape.chunk_count;
    const loom_value_id_t dynamic_index = dynamic_indices.values[0];
    const loom_type_t index_type =
        loom_module_value_type(context->module, dynamic_index);
    loom_op_t* first_packet_lower_bound_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_index_constant_build(builder, loom_attr_i64(0), index_type,
                                  op->location, &first_packet_lower_bound_op));
    first_packet_lower_bound =
        loom_index_constant_result(first_packet_lower_bound_op);
    uint32_t lower_bound = 0;
    for (uint32_t chunk_index = 0; chunk_index < shape.chunk_count;
         ++chunk_index) {
      const uint32_t upper_bound =
          lower_bound +
          iree_min(shape.chunk_lane_count, shape.lane_count - lower_bound);
      loom_op_t* upper_bound_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_index_constant_build(builder, loom_attr_i64(upper_bound),
                                    index_type, op->location, &upper_bound_op));
      packet_upper_bounds[chunk_index] =
          loom_index_constant_result(upper_bound_op);
      loom_op_t* below_upper_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_cmp_build(
          builder, LOOM_INDEX_CMP_PREDICATE_ULT, dynamic_index,
          packet_upper_bounds[chunk_index], op->location, &below_upper_op));
      packet_conditions[chunk_index] = loom_index_cmp_result(below_upper_op);
      if (chunk_index != 0) {
        loom_op_t* at_or_above_lower_op = NULL;
        IREE_RETURN_IF_ERROR(loom_index_cmp_build(
            builder, LOOM_INDEX_CMP_PREDICATE_UGE, dynamic_index,
            packet_upper_bounds[chunk_index - 1u], op->location,
            &at_or_above_lower_op));
        loom_op_t* condition_op = NULL;
        IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
            builder, loom_index_cmp_result(at_or_above_lower_op),
            packet_conditions[chunk_index],
            loom_type_scalar(LOOM_SCALAR_TYPE_I1), op->location,
            &condition_op));
        packet_conditions[chunk_index] = loom_scalar_andi_result(condition_op);
      }
      lower_bound = upper_bound;
    }
  }

  const uint32_t static_packet_index = static_index / shape.chunk_lane_count;
  for (uint32_t chunk_index = 0; chunk_index < shape.chunk_count;
       ++chunk_index) {
    const uint32_t lane_offset = chunk_index * shape.chunk_lane_count;
    const uint32_t packet_lane_count =
        iree_min(shape.lane_count - lane_offset, shape.chunk_lane_count);
    const loom_type_t packet_type =
        loom_vector_packet_type(result_type, packet_lane_count);
    const int64_t static_offset = lane_offset;
    loom_op_t* slice_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_slice_build(
        builder, dest, /*offsets=*/NULL, /*offsets_count=*/0, &static_offset,
        /*static_offsets_count=*/1, packet_type, op->location, &slice_op));
    const loom_value_id_t original_packet = loom_vector_slice_result(slice_op);

    if (has_static_index) {
      packets[chunk_index] = original_packet;
      if (chunk_index == static_packet_index) {
        const int64_t packet_index = static_index - lane_offset;
        loom_op_t* insert_op = NULL;
        IREE_RETURN_IF_ERROR(loom_vector_insert_build(
            builder, value, original_packet, /*indices=*/NULL,
            /*indices_count=*/0, &packet_index,
            /*static_indices_count=*/1, packet_type, op->location, &insert_op));
        packets[chunk_index] = loom_vector_insert_result(insert_op);
      }
      continue;
    }

    loom_value_id_t packet_index = dynamic_indices.values[0];
    if (insert_materializer == NULL) {
      const loom_type_t index_type =
          loom_module_value_type(context->module, packet_index);
      const loom_value_id_t packet_lower_bound =
          chunk_index == 0 ? first_packet_lower_bound
                           : packet_upper_bounds[chunk_index - 1u];
      loom_op_t* safe_index_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scf_select_build(
          builder, packet_conditions[chunk_index], packet_index,
          packet_lower_bound, index_type, op->location, &safe_index_op));
      packet_index = loom_scf_select_result(safe_index_op);
      if (chunk_index > 0) {
        loom_op_t* local_index_op = NULL;
        IREE_RETURN_IF_ERROR(
            loom_index_sub_build(builder, packet_index, packet_lower_bound,
                                 index_type, op->location, &local_index_op));
        packet_index = loom_index_sub_result(local_index_op);
      }
    }
    loom_value_id_t selected_packet = LOOM_VALUE_ID_INVALID;
    if (insert_materializer != NULL) {
      IREE_RETURN_IF_ERROR(insert_materializer->materialize_dynamic(
          insert_materializer->user_data, context, value, original_packet,
          packet_index, lane_offset, packet_type, op->location,
          &selected_packet));
    } else {
      const int64_t dynamic_sentinel = INT64_MIN;
      loom_op_t* insert_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_insert_build(
          builder, value, original_packet, &packet_index,
          /*indices_count=*/1, &dynamic_sentinel,
          /*static_indices_count=*/1, packet_type, op->location, &insert_op));
      selected_packet = loom_vector_insert_result(insert_op);
    }

    if (insert_materializer == NULL) {
      loom_op_t* packet_select_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scf_select_build(
          builder, packet_conditions[chunk_index], selected_packet,
          original_packet, packet_type, op->location, &packet_select_op));
      selected_packet = loom_scf_select_result(packet_select_op);
    }
    packets[chunk_index] = selected_packet;
  }

  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_concat_build(builder, /*axis=*/0, packets,
                                                shape.chunk_count, result_type,
                                                op->location, &concat_op));
  const loom_value_id_t replacement = loom_vector_concat_result(concat_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}
