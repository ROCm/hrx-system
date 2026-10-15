// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/packet_materialization.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/analysis/motion.h"
#include "loom/analysis/view_regions.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scalar/combining.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/combining.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/util/fact_vector_partition.h"

// Intersects a dynamic lane offset's retained facts with the source vector's
// valid packet starts. Divisibility facts remove starts the offset cannot take,
// keeping register-authored packet selection proportional to the actual loop
// domain instead of the complete source width.
bool loom_vector_packet_dynamic_lane_domain(
    uint32_t source_lane_count, uint32_t packet_lane_count,
    loom_value_facts_t offset_facts,
    loom_vector_packet_dynamic_lane_domain_t* out_domain) {
  *out_domain = (loom_vector_packet_dynamic_lane_domain_t){0};
  if (packet_lane_count == 0 || packet_lane_count > source_lane_count) {
    return false;
  }

  const uint32_t maximum_lane_offset = source_lane_count - packet_lane_count;
  const int64_t range_lo = iree_max(offset_facts.range_lo, INT64_C(0));
  const int64_t range_hi =
      iree_min(offset_facts.range_hi, (int64_t)maximum_lane_offset);
  if (range_lo > range_hi) {
    return false;
  }

  const uint64_t divisor = (uint64_t)offset_facts.known_divisor;
  uint64_t first_lane_offset = (uint64_t)range_lo;
  if (first_lane_offset != 0) {
    if (divisor > (uint64_t)range_hi) {
      return false;
    }
    const uint64_t remainder = first_lane_offset % divisor;
    if (remainder != 0) {
      const uint64_t adjustment = divisor - remainder;
      if (adjustment > (uint64_t)range_hi - first_lane_offset) {
        return false;
      }
      first_lane_offset += adjustment;
    }
  }

  const uint64_t lane_offset_count =
      ((uint64_t)range_hi - first_lane_offset) / divisor + 1u;
  IREE_ASSERT_LE(first_lane_offset, UINT32_MAX);
  IREE_ASSERT_LE(lane_offset_count, UINT32_MAX);
  *out_domain = (loom_vector_packet_dynamic_lane_domain_t){
      .first_lane_offset = (uint32_t)first_lane_offset,
      .lane_offset_step = lane_offset_count == 1u ? 0u : (uint32_t)divisor,
      .lane_offset_count = (uint32_t)lane_offset_count,
  };
  return true;
}

static bool loom_vector_packet_dynamic_element_operation_count(
    uint32_t source_lane_count, uint32_t packet_lane_count,
    loom_value_facts_t offset_facts, iree_host_size_t* out_operation_count) {
  *out_operation_count = 0;
  loom_vector_packet_dynamic_lane_domain_t domain = {0};
  if (!loom_vector_packet_dynamic_lane_domain(
          source_lane_count, packet_lane_count, offset_facts, &domain)) {
    return false;
  }
  iree_host_size_t selection_operation_count = 0;
  const iree_host_size_t materialization_operation_count =
      packet_lane_count == 1u ? 1u : domain.lane_offset_count;
  return iree_host_size_checked_mul(domain.lane_offset_count - 1u, 3u,
                                    &selection_operation_count) &&
         iree_host_size_checked_add(materialization_operation_count,
                                    selection_operation_count,
                                    out_operation_count);
}

static iree_status_t loom_vector_packet_build_memory_origin(
    loom_target_legalization_context_t* context,
    const loom_vector_memory_footprint_t* footprint, const loom_op_t* source_op,
    const loom_vector_packet_slice_t* slice,
    const loom_value_id_t** out_dynamic_indices,
    iree_host_size_t* out_dynamic_index_count,
    const int64_t** out_static_indices,
    iree_host_size_t* out_static_index_count) {
  if (slice->dynamic_lane_offset != LOOM_VALUE_ID_INVALID) {
    return loom_vector_packet_memory_build_dynamic_chunk_origin(
        context, footprint, source_op, slice->dynamic_lane_offset,
        out_dynamic_indices, out_dynamic_index_count, out_static_indices,
        out_static_index_count);
  }
  bool origin_built = false;
  IREE_RETURN_IF_ERROR(loom_vector_packet_memory_build_chunk_origin(
      context, footprint, source_op, slice->static_lane_offset,
      out_dynamic_indices, out_dynamic_index_count, out_static_indices,
      out_static_index_count, &origin_built));
  IREE_ASSERT_TRUE(origin_built);
  return iree_ok_status();
}

// Plans the exact component expansion before mutating IR. The old packet shape
// is only a geometric upper bound: target queries may select much narrower
// loads, so sizing storage from that shape can under-allocate and can hide a
// scalar-width expansion behind a nominally wide packet.
static iree_status_t loom_vector_packet_exact_memory_load_plan(
    loom_target_legalization_context_t* context, const loom_op_t* load_op,
    const loom_target_vector_packet_policy_t* policy, loom_type_t vector_type,
    const loom_vector_packet_shape_t* shape, uint32_t* out_component_capacity,
    bool* out_bounded) {
  *out_component_capacity = 0;
  *out_bounded = false;
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
    if (packet_lane_count == 0) {
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
  *out_component_capacity = (uint32_t)aggregate_component_count;
  *out_bounded = true;
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_rewrite_memory_load_impl(
    loom_target_legalization_context_t* context, loom_op_t* load_op,
    const loom_target_vector_packet_policy_t* policy,
    const loom_vector_memory_footprint_t* footprint,
    const loom_vector_packet_shape_t* shape,
    loom_vector_memory_cache_policy_t cache_policy, bool* out_rewritten) {
  if (out_rewritten != NULL) {
    *out_rewritten = false;
  }
  uint32_t component_capacity = shape->chunk_count;
  if (policy != NULL) {
    bool exact_plan_is_bounded = false;
    IREE_RETURN_IF_ERROR(loom_vector_packet_exact_memory_load_plan(
        context, load_op, policy, footprint->vector_type, shape,
        &component_capacity, &exact_plan_is_bounded));
    if (!exact_plan_is_bounded) {
      return iree_ok_status();
    }
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, load_op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_value_id_t* components = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->arena, component_capacity,
                                sizeof(*components), (void**)&components));
  loom_value_id_t* scalar_tail = NULL;
  if (policy != NULL) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(context->arena, shape->chunk_lane_count,
                                  sizeof(*scalar_tail), (void**)&scalar_tail));
  }

  uint32_t component_count = 0;
  uint32_t lane_offset = 0;
  while (lane_offset < shape->lane_count) {
    const uint32_t remaining_lane_count = shape->lane_count - lane_offset;
    uint32_t packet_lane_count =
        iree_min(remaining_lane_count, shape->chunk_lane_count);
    if (policy != NULL) {
      IREE_RETURN_IF_ERROR(loom_vector_packet_memory_chunk_lane_count_select(
          context, load_op, policy, footprint->vector_type, packet_lane_count,
          &packet_lane_count));
    }
    if (packet_lane_count == 0) {
      IREE_ASSERT_LE(remaining_lane_count, shape->chunk_lane_count);
      const loom_type_t scalar_type =
          loom_type_scalar(loom_type_element_type(footprint->vector_type));
      for (uint32_t i = 0; i < remaining_lane_count; ++i) {
        const loom_vector_packet_slice_t slice = {
            .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
            .static_lane_offset = lane_offset + i,
            .lane_count = 1,
        };
        const loom_value_id_t* dynamic_indices = NULL;
        iree_host_size_t dynamic_index_count = 0;
        const int64_t* static_indices = NULL;
        iree_host_size_t static_index_count = 0;
        IREE_RETURN_IF_ERROR(loom_vector_packet_build_memory_origin(
            context, footprint, load_op, &slice, &dynamic_indices,
            &dynamic_index_count, &static_indices, &static_index_count));
        loom_op_t* scalar_load_op = NULL;
        IREE_RETURN_IF_ERROR(loom_view_load_build(
            builder, cache_policy.build_flags,
            loom_vector_load_memory_flags(load_op), footprint->view,
            dynamic_indices, dynamic_index_count, static_indices,
            static_index_count, cache_policy.cache_scope,
            cache_policy.cache_temporal, scalar_type, load_op->location,
            &scalar_load_op));
        scalar_tail[i] = loom_view_load_result(scalar_load_op);
      }
      const loom_type_t tail_type =
          loom_vector_packet_type(footprint->vector_type, remaining_lane_count);
      loom_op_t* tail_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
          builder, scalar_tail, remaining_lane_count, tail_type,
          load_op->location, &tail_op));
      IREE_ASSERT_LT(component_count, component_capacity);
      components[component_count++] = loom_vector_from_elements_result(tail_op);
      lane_offset = shape->lane_count;
      break;
    }

    const loom_vector_packet_slice_t slice = {
        .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
        .static_lane_offset = lane_offset,
        .lane_count = packet_lane_count,
    };
    const loom_value_id_t* dynamic_indices = NULL;
    iree_host_size_t dynamic_index_count = 0;
    const int64_t* static_indices = NULL;
    iree_host_size_t static_index_count = 0;
    IREE_RETURN_IF_ERROR(loom_vector_packet_build_memory_origin(
        context, footprint, load_op, &slice, &dynamic_indices,
        &dynamic_index_count, &static_indices, &static_index_count));
    const loom_type_t packet_type =
        loom_vector_packet_type(footprint->vector_type, slice.lane_count);
    loom_op_t* packet_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_load_build(
        builder, cache_policy.build_flags,
        loom_vector_load_memory_flags(load_op), footprint->view,
        dynamic_indices, dynamic_index_count, static_indices,
        static_index_count, cache_policy.cache_scope,
        cache_policy.cache_temporal, packet_type, load_op->location,
        &packet_op));
    IREE_ASSERT_LT(component_count, component_capacity);
    components[component_count++] = loom_vector_load_result(packet_op);
    lane_offset += packet_lane_count;
  }

  IREE_ASSERT_EQ(component_count, component_capacity);
  loom_value_id_t replacement = components[0];
  if (component_count > 1) {
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_concat_build(
        builder, /*axis=*/0, components, component_count,
        footprint->vector_type, load_op->location, &concat_op));
    replacement = loom_vector_concat_result(concat_op);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, load_op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, load_op, &replacement, 1));
  if (out_rewritten != NULL) {
    *out_rewritten = true;
  }
  return iree_ok_status();
}

iree_status_t loom_vector_packet_rewrite_memory_load(
    loom_target_legalization_context_t* context, loom_op_t* load_op,
    const loom_vector_memory_footprint_t* footprint,
    const loom_vector_packet_shape_t* shape,
    loom_vector_memory_cache_policy_t cache_policy) {
  return loom_vector_packet_rewrite_memory_load_impl(
      context, load_op, /*policy=*/NULL, footprint, shape, cache_policy,
      /*out_rewritten=*/NULL);
}

iree_status_t loom_vector_packet_rewrite_memory_load_components(
    loom_target_legalization_context_t* context, loom_op_t* load_op,
    const loom_target_vector_packet_policy_t* policy,
    const loom_vector_memory_footprint_t* footprint,
    const loom_vector_packet_shape_t* shape,
    loom_vector_memory_cache_policy_t cache_policy, bool* out_rewritten) {
  return loom_vector_packet_rewrite_memory_load_impl(
      context, load_op, policy, footprint, shape, cache_policy, out_rewritten);
}

static iree_status_t loom_vector_packet_preserve_memory_loads_impl(
    loom_vector_packetization_t* packetization,
    const loom_vector_packet_shape_t* shape, bool exact_components,
    bool* out_preserved) {
  *out_preserved = false;
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    loom_vector_packetized_value_t* packetized_value =
        &packetization->values[i];
    if (packetized_value->source_mode !=
        LOOM_VECTOR_PACKET_SOURCE_MODE_PRESERVE_LOAD) {
      continue;
    }
    const loom_value_t* value = loom_module_value(
        packetization->context->module, packetized_value->source);
    loom_op_t* load_op = loom_value_def_op(value);
    IREE_ASSERT_TRUE(loom_vector_load_isa(load_op));
    if (exact_components) {
      loom_target_contract_query_result_t query_result =
          loom_target_contract_query_result_empty();
      IREE_RETURN_IF_ERROR(loom_target_legalization_query_contract(
          packetization->context, load_op, &query_result));
      if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_LEGAL) {
        continue;
      }
      if (query_result.outcome == LOOM_TARGET_CONTRACT_QUERY_INVALID_IR) {
        continue;
      }
    }
    loom_vector_memory_footprint_t footprint = {0};
    const bool footprint_described = loom_vector_memory_footprint_describe(
        loom_vector_packet_fact_context(packetization->context),
        packetization->context->module, load_op, &footprint);
    IREE_ASSERT_TRUE(footprint_described);
    (void)footprint_described;
    loom_vector_memory_cache_policy_t cache_policy = {0};
    const bool cache_policy_described = loom_vector_memory_cache_policy_from_op(
        packetization->context->module, load_op, &cache_policy);
    IREE_ASSERT_TRUE(cache_policy_described);
    (void)cache_policy_described;
    if (exact_components) {
      bool rewritten = false;
      IREE_RETURN_IF_ERROR(loom_vector_packet_rewrite_memory_load_components(
          packetization->context, load_op, packetization->policy, &footprint,
          shape, cache_policy, &rewritten));
      *out_preserved |= rewritten;
    } else {
      IREE_RETURN_IF_ERROR(loom_vector_packet_rewrite_memory_load(
          packetization->context, load_op, &footprint, shape, cache_policy));
      *out_preserved = true;
    }
  }
  return iree_ok_status();
}

// Legalizes source reads that cannot be streamed at a later consumer. The
// greedy driver then rebuilds its analyses and revisits that consumer, where
// the load's regular vector partition is consumed as an immutable SSA snapshot.
iree_status_t loom_vector_packet_preserve_memory_loads(
    loom_vector_packetization_t* packetization,
    const loom_vector_packet_shape_t* shape, bool* out_preserved) {
  return loom_vector_packet_preserve_memory_loads_impl(
      packetization, shape, /*exact_components=*/false, out_preserved);
}

iree_status_t loom_vector_packet_preserve_memory_load_components(
    loom_vector_packetization_t* packetization,
    const loom_vector_packet_shape_t* shape, bool* out_preserved) {
  return loom_vector_packet_preserve_memory_loads_impl(
      packetization, shape, /*exact_components=*/true, out_preserved);
}

iree_status_t loom_vector_packet_staging_load(
    loom_vector_packetization_t* packetization, loom_value_id_t staging_view,
    loom_type_t source_type, const loom_vector_packet_slice_t* slice,
    const loom_op_t* source_op, loom_vector_packet_value_t* out_packet) {
  *out_packet = (loom_vector_packet_value_t){
      .vector = LOOM_VALUE_ID_INVALID,
      .scalar = LOOM_VALUE_ID_INVALID,
  };
  const loom_value_id_t* dynamic_indices = NULL;
  iree_host_size_t dynamic_index_count = 0;
  int64_t static_index = slice->static_lane_offset;
  if (slice->dynamic_lane_offset != LOOM_VALUE_ID_INVALID) {
    dynamic_indices = &slice->dynamic_lane_offset;
    dynamic_index_count = 1;
    static_index = INT64_MIN;
  }
  const loom_type_t packet_type =
      loom_vector_packet_type(source_type, slice->lane_count);
  if (slice->lane_count == 1u) {
    const loom_type_t scalar_type =
        loom_type_scalar(loom_type_element_type(source_type));
    loom_op_t* scalar_load_op = NULL;
    IREE_RETURN_IF_ERROR(loom_view_load_build(
        &packetization->context->rewriter->builder, /*build_flags=*/0,
        /*instance_flags=*/0, staging_view, dynamic_indices,
        dynamic_index_count, &static_index,
        /*static_indices_count=*/1, /*cache_scope=*/0,
        /*cache_temporal=*/0, scalar_type, source_op->location,
        &scalar_load_op));
    loom_op_t* splat_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_vector_splat_build(&packetization->context->rewriter->builder,
                                loom_view_load_result(scalar_load_op),
                                packet_type, source_op->location, &splat_op));
    out_packet->vector = loom_vector_splat_result(splat_op);
    out_packet->scalar = loom_view_load_result(scalar_load_op);
    return iree_ok_status();
  }
  loom_op_t* staging_load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_load_build(
      &packetization->context->rewriter->builder, /*build_flags=*/0,
      /*instance_flags=*/0, staging_view, dynamic_indices, dynamic_index_count,
      &static_index,
      /*static_indices_count=*/1, /*cache_scope=*/0, /*cache_temporal=*/0,
      packet_type, source_op->location, &staging_load_op));
  out_packet->vector = loom_vector_load_result(staging_load_op);
  return iree_ok_status();
}

static void loom_vector_packet_reset_value(
    loom_vector_packetized_value_t* packetized_value) {
  packetized_value->packet = (loom_vector_packet_value_t){
      .vector = LOOM_VALUE_ID_INVALID,
      .scalar = LOOM_VALUE_ID_INVALID,
  };
}

static iree_status_t loom_vector_packet_materialize_snapshot(
    loom_vector_packetization_t* packetization,
    const loom_vector_packet_slice_t* slice,
    loom_vector_packetized_value_t* packetized_value) {
  IREE_ASSERT_EQ(slice->dynamic_lane_offset, LOOM_VALUE_ID_INVALID);

  const loom_value_t* snapshot_value = loom_module_value(
      packetization->context->module, packetized_value->source);
  const loom_op_t* snapshot_op = loom_value_def_op(snapshot_value);
  IREE_ASSERT(snapshot_op != NULL);
  const loom_value_fact_vector_partition_t partition =
      packetized_value->source_partition;
  const uint32_t snapshot_packet_lane_count = partition.snapshot_lane_count;
  uint32_t packet_index =
      slice->static_lane_offset / snapshot_packet_lane_count;
  uint32_t packet_lane_offset =
      slice->static_lane_offset % snapshot_packet_lane_count;
  const loom_type_t packet_type =
      loom_vector_packet_type(packetized_value->source_type, slice->lane_count);
  IREE_ASSERT_LE(slice->lane_count, snapshot_packet_lane_count);

  // The selected packet width never exceeds the regular snapshot width, so a
  // static packet is either contained in one component or joins the suffix and
  // prefix of two adjacent components.
  loom_value_id_t fragments[2] = {0};
  uint32_t fragment_count = 0;
  uint32_t remaining_lane_count = slice->lane_count;
  while (remaining_lane_count != 0) {
    IREE_ASSERT_LT(packet_index, partition.snapshot_count);
    IREE_ASSERT_LT(fragment_count, IREE_ARRAYSIZE(fragments));
    const loom_value_fact_vector_partition_component_t component =
        partition.snapshot_components[packet_index];
    IREE_ASSERT_LT(packet_lane_offset, component.lane_count);
    const uint32_t fragment_lane_count = iree_min(
        remaining_lane_count, component.lane_count - packet_lane_offset);
    loom_value_id_t fragment = component.value;
    if (packet_lane_offset != 0 ||
        fragment_lane_count != component.lane_count) {
      const int64_t static_offset = packet_lane_offset;
      loom_op_t* fragment_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_slice_build(
          &packetization->context->rewriter->builder, component.value,
          /*offsets=*/NULL, /*offsets_count=*/0, &static_offset,
          /*static_offsets_count=*/1,
          loom_vector_packet_type(packetized_value->source_type,
                                  fragment_lane_count),
          snapshot_op->location, &fragment_op));
      fragment = loom_vector_slice_result(fragment_op);
    }
    fragments[fragment_count++] = fragment;
    remaining_lane_count -= fragment_lane_count;
    ++packet_index;
    packet_lane_offset = 0;
  }

  loom_value_id_t packet = fragments[0];
  if (fragment_count > 1) {
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_concat_build(
        &packetization->context->rewriter->builder, /*axis=*/0, fragments,
        fragment_count, packet_type, snapshot_op->location, &concat_op));
    packet = loom_vector_concat_result(concat_op);
  }
  packetized_value->packet.vector = packet;
  if (slice->lane_count == 1u) {
    const int64_t static_index = 0;
    loom_op_t* extract_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_extract_build(
        &packetization->context->rewriter->builder, packet,
        /*indices=*/NULL, /*indices_count=*/0, &static_index,
        /*static_indices_count=*/1,
        loom_type_scalar(loom_type_element_type(packetized_value->source_type)),
        snapshot_op->location, &extract_op));
    packetized_value->packet.scalar = loom_vector_extract_result(extract_op);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_materialize_memory_load(
    loom_vector_packetization_t* packetization, loom_op_t* op,
    const loom_vector_packet_slice_t* slice,
    loom_vector_packetized_value_t* packetized_value) {
  loom_target_legalization_context_t* context = packetization->context;
  loom_vector_memory_footprint_t footprint = {0};
  const bool footprint_described = loom_vector_memory_footprint_describe(
      loom_vector_packet_fact_context(context), context->module, op,
      &footprint);
  IREE_ASSERT_TRUE(footprint_described);
  (void)footprint_described;
  loom_vector_memory_cache_policy_t cache_policy = {0};
  const bool cache_policy_described = loom_vector_memory_cache_policy_from_op(
      context->module, op, &cache_policy);
  IREE_ASSERT_TRUE(cache_policy_described);
  (void)cache_policy_described;

  const loom_value_id_t* dynamic_indices = NULL;
  iree_host_size_t dynamic_index_count = 0;
  const int64_t* static_indices = NULL;
  iree_host_size_t static_index_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_packet_build_memory_origin(
      context, &footprint, op, slice, &dynamic_indices, &dynamic_index_count,
      &static_indices, &static_index_count));
  const loom_type_t packet_type =
      loom_vector_packet_type(packetized_value->source_type, slice->lane_count);
  if (slice->lane_count == 1u) {
    const loom_type_t scalar_type =
        loom_type_scalar(loom_type_element_type(packetized_value->source_type));
    loom_op_t* scalar_load_op = NULL;
    IREE_RETURN_IF_ERROR(loom_view_load_build(
        &context->rewriter->builder, cache_policy.build_flags,
        loom_vector_load_memory_flags(op), footprint.view, dynamic_indices,
        dynamic_index_count, static_indices, static_index_count,
        cache_policy.cache_scope, cache_policy.cache_temporal, scalar_type,
        op->location, &scalar_load_op));
    loom_op_t* splat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_splat_build(
        &context->rewriter->builder, loom_view_load_result(scalar_load_op),
        packet_type, op->location, &splat_op));
    packetized_value->packet.vector = loom_vector_splat_result(splat_op);
    packetized_value->packet.scalar = loom_view_load_result(scalar_load_op);
    return iree_ok_status();
  }
  loom_op_t* packet_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_load_build(
      &context->rewriter->builder, cache_policy.build_flags,
      loom_vector_load_memory_flags(op), footprint.view, dynamic_indices,
      dynamic_index_count, static_indices, static_index_count,
      cache_policy.cache_scope, cache_policy.cache_temporal, packet_type,
      op->location, &packet_op));
  packetized_value->packet.vector = loom_vector_load_result(packet_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_materialize_simple_op(
    loom_vector_packetization_t* packetization, loom_op_t* op,
    const loom_vector_packet_slice_t* slice,
    loom_vector_packetized_value_t* packetized_value) {
  loom_builder_t* builder = &packetization->context->rewriter->builder;
  loom_op_t* packet_op = NULL;
  IREE_RETURN_IF_ERROR(loom_builder_allocate_op(
      builder, op->kind, op->operand_count, op->result_count,
      /*region_count=*/0, /*tied_result_count=*/0, op->attribute_count,
      op->location, &packet_op));
  packet_op->instance_flags = op->instance_flags;
  const loom_value_id_t* source_operands = loom_op_const_operands(op);
  for (uint16_t operand_index = 0; operand_index < op->operand_count;
       ++operand_index) {
    loom_vector_packetized_value_t* operand =
        loom_vector_packet_find(packetization, source_operands[operand_index]);
    IREE_ASSERT(operand != NULL);
    IREE_ASSERT(operand->packet.vector != LOOM_VALUE_ID_INVALID);
    loom_op_operands(packet_op)[operand_index] = operand->packet.vector;
  }
  if (op->attribute_count != 0) {
    memcpy(loom_op_attrs(packet_op), loom_op_const_attrs(op),
           op->attribute_count * sizeof(loom_attribute_t));
  }
  const loom_type_t packet_type =
      loom_vector_packet_type(packetized_value->source_type, slice->lane_count);
  IREE_RETURN_IF_ERROR(loom_builder_define_value(
      builder, packet_type, &packetized_value->packet.vector));
  loom_op_results(packet_op)[0] = packetized_value->packet.vector;
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, packet_op));
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_materialize_constant(
    loom_vector_packetization_t* packetization, loom_op_t* op,
    const loom_vector_packet_slice_t* slice,
    loom_vector_packetized_value_t* packetized_value) {
  const loom_type_t packet_type =
      loom_vector_packet_type(packetized_value->source_type, slice->lane_count);
  loom_op_t* packet_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      &packetization->context->rewriter->builder,
      loom_vector_constant_value(op), packet_type, op->location, &packet_op));
  packetized_value->packet.vector = loom_vector_constant_result(packet_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_materialize_poison(
    loom_vector_packetization_t* packetization, loom_op_t* op,
    const loom_vector_packet_slice_t* slice,
    loom_vector_packetized_value_t* packetized_value) {
  const loom_type_t packet_type =
      loom_vector_packet_type(packetized_value->source_type, slice->lane_count);
  loom_op_t* packet_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_vector_poison_build(&packetization->context->rewriter->builder,
                               packet_type, op->location, &packet_op));
  packetized_value->packet.vector = loom_vector_poison_result(packet_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_materialize_splat(
    loom_vector_packetization_t* packetization, loom_op_t* op,
    const loom_vector_packet_slice_t* slice,
    loom_vector_packetized_value_t* packetized_value) {
  const loom_type_t packet_type =
      loom_vector_packet_type(packetized_value->source_type, slice->lane_count);
  loom_op_t* packet_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(
      &packetization->context->rewriter->builder, loom_vector_splat_scalar(op),
      packet_type, op->location, &packet_op));
  packetized_value->packet.vector = loom_vector_splat_result(packet_op);
  if (slice->lane_count == 1u) {
    packetized_value->packet.scalar = loom_vector_splat_scalar(op);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_materialize_from_elements(
    loom_vector_packetization_t* packetization, loom_op_t* op,
    const loom_vector_packet_slice_t* slice,
    loom_vector_packetized_value_t* packetized_value) {
  const loom_value_slice_t elements = loom_vector_from_elements_elements(op);
  const loom_type_t packet_type =
      loom_vector_packet_type(packetized_value->source_type, slice->lane_count);

  if (slice->dynamic_lane_offset != LOOM_VALUE_ID_INVALID) {
    loom_target_legalization_context_t* context = packetization->context;
    const loom_value_facts_t offset_facts =
        context->fact_table == NULL
            ? loom_value_facts_unknown()
            : loom_value_fact_table_lookup(context->fact_table,
                                           slice->dynamic_lane_offset);
    loom_vector_packet_dynamic_lane_domain_t domain = {0};
    const bool domain_available = loom_vector_packet_dynamic_lane_domain(
        (uint32_t)elements.count, slice->lane_count, offset_facts, &domain);
    IREE_ASSERT_TRUE(domain_available);
    (void)domain_available;

    loom_builder_t* builder = &context->rewriter->builder;
    const loom_type_t offset_type =
        loom_module_value_type(context->module, slice->dynamic_lane_offset);
    // A one-lane packet selects the source scalar directly and constructs its
    // vector projection once. Wider packets select complete vector values.
    // Both paths therefore retain the scalar projection without building a
    // candidate vector merely to extract it again at a scalar consumer.
    const bool selects_scalar = slice->lane_count == 1u;
    const loom_type_t selection_type =
        selects_scalar
            ? loom_module_value_type(context->module, elements.values[0])
            : packet_type;
    loom_value_id_t selected_value = LOOM_VALUE_ID_INVALID;
    for (uint32_t candidate_index = 0;
         candidate_index < domain.lane_offset_count; ++candidate_index) {
      const uint32_t candidate_lane_offset =
          domain.first_lane_offset + candidate_index * domain.lane_offset_step;
      loom_value_id_t candidate_value = elements.values[candidate_lane_offset];
      if (!selects_scalar) {
        loom_op_t* candidate_op = NULL;
        IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
            builder, elements.values + candidate_lane_offset, slice->lane_count,
            packet_type, op->location, &candidate_op));
        candidate_value = loom_vector_from_elements_result(candidate_op);
      }
      if (candidate_index == 0) {
        selected_value = candidate_value;
        continue;
      }

      loom_op_t* expected_offset_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_constant_build(
          builder, loom_attr_i64(candidate_lane_offset), offset_type,
          op->location, &expected_offset_op));
      loom_op_t* matches_offset_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_cmp_build(
          builder, LOOM_INDEX_CMP_PREDICATE_EQ, slice->dynamic_lane_offset,
          loom_index_constant_result(expected_offset_op), op->location,
          &matches_offset_op));
      loom_op_t* select_op = NULL;
      IREE_RETURN_IF_ERROR(loom_scf_select_build(
          builder, loom_index_cmp_result(matches_offset_op), candidate_value,
          selected_value, selection_type, op->location, &select_op));
      selected_value = loom_scf_select_result(select_op);
    }
    loom_value_id_t selected_packet = selected_value;
    if (selects_scalar) {
      loom_op_t* packet_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
          builder, &selected_value, 1u, packet_type, op->location, &packet_op));
      selected_packet = loom_vector_from_elements_result(packet_op);
    }
    packetized_value->packet = (loom_vector_packet_value_t){
        .vector = selected_packet,
        .scalar = selects_scalar ? selected_value : LOOM_VALUE_ID_INVALID,
    };
    return iree_ok_status();
  }

  IREE_ASSERT_LE(
      (iree_host_size_t)slice->static_lane_offset + slice->lane_count,
      elements.count);
  loom_op_t* packet_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      &packetization->context->rewriter->builder,
      elements.values + slice->static_lane_offset, slice->lane_count,
      packet_type, op->location, &packet_op));
  packetized_value->packet.vector = loom_vector_from_elements_result(packet_op);
  if (slice->lane_count == 1u) {
    packetized_value->packet.scalar =
        elements.values[slice->static_lane_offset];
  }
  return iree_ok_status();
}

bool loom_vector_packet_dynamic_materialization_operation_count(
    const loom_vector_packetization_t* packetization,
    loom_value_facts_t lane_offset_facts, uint32_t lane_count,
    iree_host_size_t* out_operation_count) {
  *out_operation_count = 0;
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    const loom_vector_packetized_value_t* packetized_value =
        &packetization->values[i];
    iree_host_size_t value_operation_count = 1u;
    if (packetized_value->source_mode ==
        LOOM_VECTOR_PACKET_SOURCE_MODE_STATIC_ELEMENTS) {
      uint32_t source_lane_count = 0;
      if (!loom_vector_packet_static_lane_count(packetized_value->source_type,
                                                &source_lane_count) ||
          !loom_vector_packet_dynamic_element_operation_count(
              source_lane_count, lane_count, lane_offset_facts,
              &value_operation_count)) {
        return false;
      }
    }
    if (!iree_host_size_checked_add(*out_operation_count, value_operation_count,
                                    out_operation_count)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_vector_packet_materialize_captured(
    loom_vector_packetization_t* packetization,
    const loom_vector_packet_slice_t* slice,
    loom_vector_packetized_value_t* packetized_value) {
  const loom_value_id_t* offsets = NULL;
  iree_host_size_t offset_count = 0;
  int64_t static_offset = slice->static_lane_offset;
  if (slice->dynamic_lane_offset != LOOM_VALUE_ID_INVALID) {
    offsets = &slice->dynamic_lane_offset;
    offset_count = 1;
    static_offset = INT64_MIN;
  }
  const loom_type_t packet_type =
      loom_vector_packet_type(packetized_value->source_type, slice->lane_count);
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_slice_build(
      &packetization->context->rewriter->builder, packetized_value->source,
      offsets, offset_count, &static_offset, /*static_offsets_count=*/1,
      packet_type, packetization->captured_value_location, &slice_op));
  packetized_value->packet.vector = loom_vector_slice_result(slice_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_materialize_value(
    loom_vector_packetization_t* packetization,
    const loom_vector_packet_slice_t* slice,
    loom_vector_packetized_value_t* packetized_value) {
  loom_target_legalization_record_component_packet_materialization(
      packetization->context);
  loom_vector_packet_reset_value(packetized_value);
  if (packetized_value->source_mode ==
      LOOM_VECTOR_PACKET_SOURCE_MODE_SNAPSHOT) {
    return loom_vector_packet_materialize_snapshot(packetization, slice,
                                                   packetized_value);
  }
  if (packetized_value->source_mode ==
      LOOM_VECTOR_PACKET_SOURCE_MODE_CAPTURED) {
    return loom_vector_packet_materialize_captured(packetization, slice,
                                                   packetized_value);
  }
  loom_target_legalization_context_t* context = packetization->context;
  const loom_value_t* value =
      loom_module_value(context->module, packetized_value->source);
  loom_op_t* op = loom_value_def_op(value);
  if (loom_vector_load_isa(op)) {
    return loom_vector_packet_materialize_memory_load(packetization, op, slice,
                                                      packetized_value);
  } else if (loom_vector_constant_isa(op)) {
    return loom_vector_packet_materialize_constant(packetization, op, slice,
                                                   packetized_value);
  } else if (loom_vector_poison_isa(op)) {
    return loom_vector_packet_materialize_poison(packetization, op, slice,
                                                 packetized_value);
  } else if (loom_vector_splat_isa(op)) {
    return loom_vector_packet_materialize_splat(packetization, op, slice,
                                                packetized_value);
  } else if (loom_vector_from_elements_isa(op)) {
    return loom_vector_packet_materialize_from_elements(
        packetization, op, slice, packetized_value);
  }
  return loom_vector_packet_materialize_simple_op(packetization, op, slice,
                                                  packetized_value);
}

iree_status_t loom_vector_packet_materialize(
    loom_vector_packetization_t* packetization, loom_value_id_t source,
    const loom_vector_packet_slice_t* slice,
    loom_vector_packetized_value_t** out_value) {
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_vector_packet_materialize_value(
        packetization, slice, &packetization->values[i]));
  }
  *out_value = loom_vector_packet_find(packetization, source);
  IREE_ASSERT(*out_value != NULL);
  IREE_ASSERT((*out_value)->packet.vector != LOOM_VALUE_ID_INVALID);
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_extract_single_lane(
    loom_vector_packetization_t* packetization,
    loom_vector_packet_value_t packet, loom_location_id_t location,
    loom_value_id_t* out_scalar) {
  if (packet.scalar != LOOM_VALUE_ID_INVALID) {
    *out_scalar = packet.scalar;
    return iree_ok_status();
  }
  const loom_value_t* packet_value =
      loom_module_value(packetization->context->module, packet.vector);

  const int64_t static_index = 0;
  const loom_type_t packet_type = packet_value->type;
  const loom_type_t scalar_type =
      loom_type_scalar(loom_type_element_type(packet_type));
  loom_op_t* extract_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      &packetization->context->rewriter->builder, packet.vector,
      /*indices=*/NULL,
      /*indices_count=*/0, &static_index, /*static_indices_count=*/1,
      scalar_type, location, &extract_op));
  *out_scalar = loom_vector_extract_result(extract_op);
  return iree_ok_status();
}

iree_status_t loom_vector_packet_store(
    loom_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint,
    const loom_vector_packet_slice_t* slice, loom_vector_packet_value_t packet,
    loom_vector_memory_cache_policy_t store_cache_policy, loom_op_t* store_op) {
  loom_target_legalization_context_t* context = packetization->context;
  const loom_value_id_t* dynamic_indices = NULL;
  iree_host_size_t dynamic_index_count = 0;
  const int64_t* static_indices = NULL;
  iree_host_size_t static_index_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_packet_build_memory_origin(
      context, store_footprint, store_op, slice, &dynamic_indices,
      &dynamic_index_count, &static_indices, &static_index_count));
  if (slice->lane_count == 1u) {
    loom_value_id_t scalar = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_packet_extract_single_lane(
        packetization, packet, store_op->location, &scalar));
    loom_op_t* scalar_store_op = NULL;
    return loom_view_store_build(
        &context->rewriter->builder, store_cache_policy.build_flags,
        loom_vector_store_memory_flags(store_op), scalar, store_footprint->view,
        dynamic_indices, dynamic_index_count, static_indices,
        static_index_count, store_cache_policy.cache_scope,
        store_cache_policy.cache_temporal, store_op->location,
        &scalar_store_op);
  }
  loom_op_t* packet_store_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_store_build(
      &context->rewriter->builder, store_cache_policy.build_flags,
      loom_vector_store_memory_flags(store_op), packet.vector,
      store_footprint->view, dynamic_indices, dynamic_index_count,
      static_indices, static_index_count, store_cache_policy.cache_scope,
      store_cache_policy.cache_temporal, store_op->location, &packet_store_op));
  return iree_ok_status();
}

bool loom_vector_packet_static_operation_count_is_bounded(
    iree_host_size_t operations_per_chunk,
    const loom_vector_packet_shape_t* shape) {
  iree_host_size_t operation_count = 0;
  return iree_host_size_checked_mul(operations_per_chunk, shape->chunk_count,
                                    &operation_count) &&
         operation_count <= LOOM_VECTOR_PACKET_STATIC_OP_LIMIT;
}

iree_status_t loom_vector_packet_store_captured_value(
    loom_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint,
    const loom_vector_packet_shape_t* shape,
    loom_vector_memory_cache_policy_t store_cache_policy, loom_op_t* store_op) {
  loom_builder_t* builder = &packetization->context->rewriter->builder;
  for (uint32_t chunk_index = 0; chunk_index < shape->chunk_count;
       ++chunk_index) {
    const uint32_t lane_offset = chunk_index * shape->chunk_lane_count;
    const loom_vector_packet_slice_t slice = {
        .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
        .static_lane_offset = lane_offset,
        .lane_count =
            iree_min(shape->lane_count - lane_offset, shape->chunk_lane_count),
    };
    const int64_t static_offset = lane_offset;
    const loom_type_t packet_type =
        loom_vector_packet_type(store_footprint->vector_type, slice.lane_count);
    loom_op_t* slice_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_slice_build(
        builder, store_footprint->value, /*offsets=*/NULL, /*offsets_count=*/0,
        &static_offset, /*static_offsets_count=*/1, packet_type,
        store_op->location, &slice_op));
    IREE_RETURN_IF_ERROR(loom_vector_packet_store(
        packetization, store_footprint, &slice,
        (loom_vector_packet_value_t){
            .vector = loom_vector_slice_result(slice_op),
            .scalar = LOOM_VALUE_ID_INVALID,
        },
        store_cache_policy, store_op));
  }
  return iree_ok_status();
}

bool loom_vector_packet_static_expansion_is_bounded(
    const loom_vector_packetization_t* packetization,
    const loom_vector_packet_shape_t* shape) {
  iree_host_size_t operations_per_chunk = 0;
  return iree_host_size_checked_add(packetization->value_count, 1,
                                    &operations_per_chunk) &&
         loom_vector_packet_static_operation_count_is_bounded(
             operations_per_chunk, shape);
}

iree_status_t loom_vector_packet_static_interleaved_store(
    loom_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint,
    const loom_vector_packet_shape_t* shape,
    loom_vector_memory_cache_policy_t store_cache_policy, loom_op_t* store_op) {
  for (uint32_t chunk_index = 0; chunk_index < shape->chunk_count;
       ++chunk_index) {
    const uint32_t lane_offset = chunk_index * shape->chunk_lane_count;
    const loom_vector_packet_slice_t slice = {
        .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
        .static_lane_offset = lane_offset,
        .lane_count =
            iree_min(shape->lane_count - lane_offset, shape->chunk_lane_count),
    };
    loom_vector_packetized_value_t* root = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_packet_materialize(
        packetization, store_footprint->value, &slice, &root));
    IREE_RETURN_IF_ERROR(
        loom_vector_packet_store(packetization, store_footprint, &slice,
                                 root->packet, store_cache_policy, store_op));
  }
  return iree_ok_status();
}

iree_status_t loom_vector_packet_static_store(
    loom_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint,
    const loom_vector_packet_shape_t* shape,
    loom_vector_memory_cache_policy_t store_cache_policy, loom_op_t* store_op) {
  const iree_host_size_t packet_count =
      (iree_host_size_t)packetization->value_count * shape->chunk_count;
  IREE_ASSERT_LE(packet_count, LOOM_VECTOR_PACKET_STATIC_OP_LIMIT);
  loom_vector_packet_value_t* packets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(packetization->context->arena,
                                                 packet_count, sizeof(*packets),
                                                 (void**)&packets));
  for (uint32_t value_index = 0; value_index < packetization->value_count;
       ++value_index) {
    loom_vector_packetized_value_t* packetized_value =
        &packetization->values[value_index];
    const loom_value_t* value = loom_module_value(
        packetization->context->module, packetized_value->source);
    const loom_op_t* source_op = loom_value_def_op(value);
    for (uint32_t chunk_index = 0; chunk_index < shape->chunk_count;
         ++chunk_index) {
      const loom_value_id_t* source_operands =
          loom_op_const_operands(source_op);
      for (uint16_t operand_index = 0; operand_index < source_op->operand_count;
           ++operand_index) {
        loom_vector_packetized_value_t* operand = loom_vector_packet_find(
            packetization, source_operands[operand_index]);
        if (operand == NULL) {
          continue;
        }
        const uint32_t operand_value_index =
            (uint32_t)(operand - packetization->values);
        operand->packet =
            packets[operand_value_index * shape->chunk_count + chunk_index];
      }
      const uint32_t lane_offset = chunk_index * shape->chunk_lane_count;
      const loom_vector_packet_slice_t slice = {
          .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
          .static_lane_offset = lane_offset,
          .lane_count = iree_min(shape->lane_count - lane_offset,
                                 shape->chunk_lane_count),
      };
      IREE_RETURN_IF_ERROR(loom_vector_packet_materialize_value(
          packetization, &slice, packetized_value));
      packets[value_index * shape->chunk_count + chunk_index] =
          packetized_value->packet;
    }
  }
  loom_vector_packetized_value_t* root =
      loom_vector_packet_find(packetization, store_footprint->value);
  IREE_ASSERT(root != NULL);
  const uint32_t root_value_index = (uint32_t)(root - packetization->values);
  for (uint32_t chunk_index = 0; chunk_index < shape->chunk_count;
       ++chunk_index) {
    const uint32_t lane_offset = chunk_index * shape->chunk_lane_count;
    const loom_vector_packet_slice_t slice = {
        .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
        .static_lane_offset = lane_offset,
        .lane_count =
            iree_min(shape->lane_count - lane_offset, shape->chunk_lane_count),
    };
    IREE_RETURN_IF_ERROR(loom_vector_packet_store(
        packetization, store_footprint, &slice,
        packets[root_value_index * shape->chunk_count + chunk_index],
        store_cache_policy, store_op));
  }
  return iree_ok_status();
}

iree_status_t loom_vector_packet_build_staging_view(
    loom_vector_packetization_t* packetization, loom_type_t vector_type,
    const loom_vector_packet_shape_t* shape, const loom_op_t* source_op,
    loom_value_id_t* out_staging_view) {
  *out_staging_view = LOOM_VALUE_ID_INVALID;
  const int32_t element_bit_count = loom_vector_packet_element_bit_count(
      packetization->policy, loom_type_element_type(vector_type));
  IREE_ASSERT_GT(element_bit_count, 0);
  IREE_ASSERT_EQ(element_bit_count % 8, 0);
  const int64_t byte_count =
      (int64_t)shape->lane_count * (element_bit_count / 8);
  const int64_t packet_byte_count =
      (int64_t)shape->chunk_lane_count * (element_bit_count / 8);
  IREE_ASSERT(iree_math_is_power_of_two_i64(packet_byte_count));
  // Preserve the selected native packet access through private staging while
  // retaining the historical 16-byte floor for narrower packets.
  const int64_t staging_alignment = iree_max((int64_t)16, packet_byte_count);
  loom_builder_t* builder = &packetization->context->rewriter->builder;
  const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);

  loom_op_t* byte_count_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(builder, loom_attr_i64(byte_count), offset_type,
                                source_op->location, &byte_count_op));
  loom_op_t* staging_buffer_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_alloca_build(
      builder, /*build_flags=*/0, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE,
      LOOM_VALUE_ID_INVALID, staging_alignment,
      loom_index_constant_result(byte_count_op), loom_type_buffer(),
      source_op->location, &staging_buffer_op));
  loom_op_t* zero_offset_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(builder, loom_attr_i64(0), offset_type,
                                source_op->location, &zero_offset_op));
  const loom_type_t staging_view_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, loom_type_element_type(vector_type),
      loom_dim_pack_static(shape->lane_count), /*encoding_id=*/0);
  loom_op_t* staging_view_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_view_build(
      builder, loom_buffer_alloca_result(staging_buffer_op),
      loom_index_constant_result(zero_offset_op), staging_view_type,
      source_op->location, &staging_view_op));
  *out_staging_view = loom_buffer_view_result(staging_view_op);
  return iree_ok_status();
}

iree_status_t loom_vector_packet_staging_store(
    loom_vector_packetization_t* packetization, loom_value_id_t staging_view,
    const loom_vector_packet_slice_t* slice, loom_vector_packet_value_t packet,
    const loom_op_t* source_op) {
  const loom_value_id_t* dynamic_indices = NULL;
  iree_host_size_t dynamic_index_count = 0;
  int64_t static_index = slice->static_lane_offset;
  if (slice->dynamic_lane_offset != LOOM_VALUE_ID_INVALID) {
    dynamic_indices = &slice->dynamic_lane_offset;
    dynamic_index_count = 1;
    static_index = INT64_MIN;
  }
  if (slice->lane_count == 1u) {
    loom_value_id_t scalar = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_packet_extract_single_lane(
        packetization, packet, source_op->location, &scalar));
    loom_op_t* staging_store_op = NULL;
    return loom_view_store_build(
        &packetization->context->rewriter->builder, /*build_flags=*/0,
        /*instance_flags=*/0, scalar, staging_view, dynamic_indices,
        dynamic_index_count, &static_index,
        /*static_indices_count=*/1, /*cache_scope=*/0,
        /*cache_temporal=*/0, source_op->location, &staging_store_op);
  }
  loom_op_t* staging_store_op = NULL;
  return loom_vector_store_build(
      &packetization->context->rewriter->builder, /*build_flags=*/0,
      /*instance_flags=*/0, packet.vector, staging_view, dynamic_indices,
      dynamic_index_count, &static_index,
      /*static_indices_count=*/1, /*cache_scope=*/0, /*cache_temporal=*/0,
      source_op->location, &staging_store_op);
}

// Reifies packet snapshots as immutable private storage at their definition.
// Each direct consumer receives its own oversized load so a later packetization
// can stream from the shared storage without retaining the original packet
// tuple across an unrelated consumer.
iree_status_t loom_vector_packet_canonicalize_snapshots(
    loom_vector_packetization_t* packetization,
    const loom_vector_packet_shape_t* shape) {
  loom_rewriter_t* rewriter = packetization->context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    loom_vector_packetized_value_t* packetized_value =
        &packetization->values[i];
    if (packetized_value->source_mode !=
        LOOM_VECTOR_PACKET_SOURCE_MODE_SNAPSHOT) {
      continue;
    }

    loom_value_t* snapshot_value = loom_module_value(
        packetization->context->module, packetized_value->source);
    loom_op_t* snapshot_op = loom_value_def_op(snapshot_value);
    IREE_ASSERT(snapshot_op != NULL);
    loom_builder_set_after(builder, snapshot_op);
    loom_value_id_t staging_view = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_packet_build_staging_view(
        packetization, packetized_value->source_type, shape, snapshot_op,
        &staging_view));
    for (uint32_t chunk_index = 0; chunk_index < shape->chunk_count;
         ++chunk_index) {
      const uint32_t lane_offset = chunk_index * shape->chunk_lane_count;
      const loom_vector_packet_slice_t slice = {
          .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
          .static_lane_offset = lane_offset,
          .lane_count = iree_min(shape->lane_count - lane_offset,
                                 shape->chunk_lane_count),
      };
      loom_vector_packet_reset_value(packetized_value);
      IREE_RETURN_IF_ERROR(loom_vector_packet_materialize_snapshot(
          packetization, &slice, packetized_value));
      IREE_RETURN_IF_ERROR(loom_vector_packet_staging_store(
          packetization, staging_view, &slice, packetized_value->packet,
          snapshot_op));
    }

    const uint32_t use_count = snapshot_value->use_count;
    loom_op_t** consumers = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(packetization->context->arena, use_count,
                                  sizeof(*consumers), (void**)&consumers));
    uint32_t consumer_count = 0;
    const loom_use_t* use = NULL;
    loom_value_for_each_use(snapshot_value, use) {
      loom_op_t* consumer = loom_use_user_op(*use);
      bool already_recorded = false;
      for (uint32_t j = 0; j < consumer_count; ++j) {
        if (consumers[j] == consumer) {
          already_recorded = true;
          break;
        }
      }
      if (!already_recorded) {
        consumers[consumer_count++] = consumer;
      }
    }

    const loom_vector_packet_slice_t snapshot_slice = {
        .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
        .static_lane_offset = 0,
        .lane_count = shape->lane_count,
    };
    for (uint32_t j = 0; j < consumer_count; ++j) {
      loom_op_t* consumer = consumers[j];
      loom_builder_set_before(builder, consumer);
      loom_vector_packet_value_t snapshot_load = {0};
      IREE_RETURN_IF_ERROR(loom_vector_packet_staging_load(
          packetization, staging_view, packetized_value->source_type,
          &snapshot_slice, snapshot_op, &snapshot_load));
      const loom_value_id_t* operands = loom_op_const_operands(consumer);
      for (uint16_t operand_index = 0; operand_index < consumer->operand_count;
           ++operand_index) {
        if (operands[operand_index] == packetized_value->source) {
          IREE_RETURN_IF_ERROR(loom_rewriter_set_operand(
              rewriter, consumer, operand_index, snapshot_load.vector));
        }
      }
    }
  }
  loom_builder_restore(builder, saved_ip);

  for (uint32_t i = packetization->value_count; i > 0; --i) {
    const loom_vector_packetized_value_t* packetized_value =
        &packetization->values[i - 1u];
    if (packetized_value->source_mode !=
        LOOM_VECTOR_PACKET_SOURCE_MODE_SNAPSHOT) {
      continue;
    }
    loom_value_t* snapshot_value = loom_module_value(
        packetization->context->module, packetized_value->source);
    loom_op_t* snapshot_op = loom_value_def_op(snapshot_value);
    bool erased = false;
    IREE_RETURN_IF_ERROR(
        loom_rewriter_erase_if_dead(rewriter, snapshot_op, &erased));
  }
  return iree_ok_status();
}

iree_status_t loom_vector_packet_staged_store(
    loom_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint,
    const loom_vector_packet_shape_t* shape,
    loom_vector_memory_cache_policy_t store_cache_policy, loom_op_t* store_op) {
  loom_builder_t* builder = &packetization->context->rewriter->builder;
  loom_value_id_t staging_view = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_packet_build_staging_view(
      packetization, store_footprint->vector_type, shape, store_op,
      &staging_view));

  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const uint32_t loop_lane_count =
      (shape->lane_count / shape->chunk_lane_count) * shape->chunk_lane_count;
  loom_op_t* lower_bound_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(builder, loom_attr_i64(0),
                                                 index_type, store_op->location,
                                                 &lower_bound_op));
  loom_op_t* upper_bound_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(loop_lane_count), index_type, store_op->location,
      &upper_bound_op));
  loom_op_t* step_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(builder, loom_attr_i64(shape->chunk_lane_count),
                                index_type, store_op->location, &step_op));

  loom_op_t* stage_loop_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      builder, /*build_flags=*/0, loom_index_constant_result(lower_bound_op),
      loom_index_constant_result(upper_bound_op),
      loom_index_constant_result(step_op), /*iter_args=*/NULL,
      /*iter_args_count=*/0, /*result_types=*/NULL, /*tied_results=*/NULL,
      /*tied_result_count=*/0,
      /*pipeline_depth=*/LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, store_op->location,
      &stage_loop_op));
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      builder, stage_loop_op, loom_scf_for_body(stage_loop_op));
  const loom_vector_packet_slice_t loop_slice = {
      .dynamic_lane_offset =
          loom_region_entry_arg_id(loom_scf_for_body(stage_loop_op), 0),
      .lane_count = shape->chunk_lane_count,
  };
  loom_vector_packetized_value_t* packetized_value = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_packet_materialize(
      packetization, store_footprint->value, &loop_slice, &packetized_value));
  IREE_RETURN_IF_ERROR(
      loom_vector_packet_staging_store(packetization, staging_view, &loop_slice,
                                       packetized_value->packet, store_op));
  loom_op_t* stage_yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(
      builder, NULL, 0, store_op->location, &stage_yield_op));
  loom_builder_restore(builder, saved_ip);

  if (loop_lane_count < shape->lane_count) {
    const loom_vector_packet_slice_t tail_slice = {
        .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
        .static_lane_offset = loop_lane_count,
        .lane_count = shape->lane_count - loop_lane_count,
    };
    IREE_RETURN_IF_ERROR(loom_vector_packet_materialize(
        packetization, store_footprint->value, &tail_slice, &packetized_value));
    IREE_RETURN_IF_ERROR(loom_vector_packet_staging_store(
        packetization, staging_view, &tail_slice, packetized_value->packet,
        store_op));
  }

  loom_op_t* commit_loop_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      builder, /*build_flags=*/0, loom_index_constant_result(lower_bound_op),
      loom_index_constant_result(upper_bound_op),
      loom_index_constant_result(step_op), /*iter_args=*/NULL,
      /*iter_args_count=*/0, /*result_types=*/NULL, /*tied_results=*/NULL,
      /*tied_result_count=*/0,
      /*pipeline_depth=*/LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, store_op->location,
      &commit_loop_op));
  saved_ip = loom_builder_enter_region(builder, commit_loop_op,
                                       loom_scf_for_body(commit_loop_op));
  const loom_vector_packet_slice_t commit_loop_slice = {
      .dynamic_lane_offset =
          loom_region_entry_arg_id(loom_scf_for_body(commit_loop_op), 0),
      .lane_count = shape->chunk_lane_count,
  };
  loom_vector_packet_value_t staged_packet = {0};
  IREE_RETURN_IF_ERROR(loom_vector_packet_staging_load(
      packetization, staging_view, store_footprint->vector_type,
      &commit_loop_slice, store_op, &staged_packet));
  IREE_RETURN_IF_ERROR(loom_vector_packet_store(
      packetization, store_footprint, &commit_loop_slice, staged_packet,
      store_cache_policy, store_op));
  loom_op_t* commit_yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(
      builder, NULL, 0, store_op->location, &commit_yield_op));
  loom_builder_restore(builder, saved_ip);

  if (loop_lane_count < shape->lane_count) {
    const loom_vector_packet_slice_t tail_slice = {
        .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
        .static_lane_offset = loop_lane_count,
        .lane_count = shape->lane_count - loop_lane_count,
    };
    IREE_RETURN_IF_ERROR(loom_vector_packet_staging_load(
        packetization, staging_view, store_footprint->vector_type, &tail_slice,
        store_op, &staged_packet));
    IREE_RETURN_IF_ERROR(
        loom_vector_packet_store(packetization, store_footprint, &tail_slice,
                                 staged_packet, store_cache_policy, store_op));
  }
  return iree_ok_status();
}

bool loom_vector_packet_has_snapshots(
    const loom_vector_packetization_t* packetization) {
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    if (packetization->values[i].source_mode ==
        LOOM_VECTOR_PACKET_SOURCE_MODE_SNAPSHOT) {
      return true;
    }
  }
  return false;
}

bool loom_vector_packet_has_captured_values(
    const loom_vector_packetization_t* packetization) {
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    if (packetization->values[i].source_mode ==
        LOOM_VECTOR_PACKET_SOURCE_MODE_CAPTURED) {
      return true;
    }
  }
  return false;
}

bool loom_vector_packet_requires_static_slices(
    const loom_vector_packetization_t* packetization) {
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    if (packetization->values[i].source_mode ==
        LOOM_VECTOR_PACKET_SOURCE_MODE_STATIC_ELEMENTS) {
      return true;
    }
  }
  return false;
}

static bool loom_vector_packet_reduce_can_fold_packets(
    loom_combining_kind_t kind, uint8_t fastmath_flags) {
  return loom_combining_kind_accepts_integer(kind) ||
         (loom_combining_kind_accepts_float(kind) &&
          iree_any_bit_set(fastmath_flags, LOOM_VECTOR_FASTMATHFLAGS_REASSOC));
}

static iree_status_t loom_vector_packet_reduce_vector_component(
    loom_target_legalization_context_t* context, loom_op_t* reduce_op,
    loom_value_id_t input, loom_value_id_t* inout_accumulator) {
  loom_op_t* component_reduce_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_reduce_build(
      &context->rewriter->builder, loom_vector_reduce_kind(reduce_op),
      loom_vector_reduce_fastmath(reduce_op), input, *inout_accumulator,
      loom_module_value_type(context->module,
                             loom_vector_reduce_result(reduce_op)),
      reduce_op->location, &component_reduce_op));
  *inout_accumulator = loom_vector_reduce_result(component_reduce_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_reduce_scalar_component(
    loom_target_legalization_context_t* context, loom_op_t* reduce_op,
    loom_value_id_t component, loom_value_id_t* inout_accumulator) {
  const loom_combining_kind_t kind = loom_vector_reduce_kind(reduce_op);
  const uint8_t fastmath_flags = loom_vector_reduce_fastmath(reduce_op);
  const loom_type_t result_type = loom_module_value_type(
      context->module, loom_vector_reduce_result(reduce_op));
  return loom_scalar_combining_build(
      &context->rewriter->builder, kind, fastmath_flags, *inout_accumulator,
      component, result_type, reduce_op->location, inout_accumulator);
}

static iree_status_t loom_vector_packet_partition_reduce_hybrid(
    loom_target_legalization_context_t* context,
    loom_value_fact_vector_partition_t partition, loom_op_t* reduce_op,
    loom_value_id_t* out_accumulator) {
  loom_value_id_t accumulator = loom_vector_reduce_init(reduce_op);
  for (uint32_t i = 0; i < partition.count; ++i) {
    const loom_value_fact_vector_partition_component_t component =
        partition.components[i];
    const loom_type_t component_type =
        loom_module_value_type(context->module, component.value);
    if (loom_type_is_vector(component_type)) {
      IREE_RETURN_IF_ERROR(loom_vector_packet_reduce_vector_component(
          context, reduce_op, component.value, &accumulator));
    } else {
      IREE_RETURN_IF_ERROR(loom_vector_packet_reduce_scalar_component(
          context, reduce_op, component.value, &accumulator));
    }
  }
  *out_accumulator = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_combine_balanced(
    loom_target_legalization_context_t* context, loom_op_t* reduce_op,
    loom_type_t value_type, loom_value_id_t* values, uint32_t value_count,
    bool vector_values, loom_value_id_t* out_value) {
  const loom_combining_kind_t kind = loom_vector_reduce_kind(reduce_op);
  const uint8_t fastmath_flags = loom_vector_reduce_fastmath(reduce_op);
  while (value_count > 1) {
    uint32_t next_count = 0;
    uint32_t value_index = 0;
    for (; value_index + 1u < value_count; value_index += 2u) {
      if (vector_values) {
        IREE_RETURN_IF_ERROR(loom_vector_combining_build(
            &context->rewriter->builder, kind, fastmath_flags,
            values[value_index], values[value_index + 1u], value_type,
            reduce_op->location, &values[next_count++]));
      } else {
        IREE_RETURN_IF_ERROR(loom_scalar_combining_build(
            &context->rewriter->builder, kind, fastmath_flags,
            values[value_index], values[value_index + 1u], value_type,
            reduce_op->location, &values[next_count++]));
      }
    }
    if (value_index < value_count) {
      values[next_count++] = values[value_index];
    }
    value_count = next_count;
  }
  *out_value = values[0];
  return iree_ok_status();
}

// Finishes a packed fold when the target has native lanewise combining but no
// native horizontal reduction at the selected leaf width. Each level combines
// the low and high halves into a narrower vector; only the final pair crosses
// into scalar form.
static iree_status_t loom_vector_packet_reduce_folded_leaf(
    loom_target_legalization_context_t* context, loom_op_t* reduce_op,
    loom_value_id_t leaf, uint32_t leaf_lane_count,
    loom_value_id_t* out_accumulator) {
  IREE_ASSERT_GE(leaf_lane_count, 2u);
  IREE_ASSERT_EQ(leaf_lane_count & (leaf_lane_count - 1u), 0u);
  loom_value_id_t folded_leaf = leaf;
  const loom_type_t leaf_type = loom_module_value_type(context->module, leaf);
  while (leaf_lane_count > 2) {
    const uint32_t half_lane_count = leaf_lane_count / 2u;
    const loom_type_t half_type =
        loom_vector_packet_type(leaf_type, half_lane_count);
    loom_value_id_t halves[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
    for (uint32_t half_index = 0; half_index < IREE_ARRAYSIZE(halves);
         ++half_index) {
      const int64_t static_offset = half_index * half_lane_count;
      loom_op_t* slice_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_slice_build(
          &context->rewriter->builder, folded_leaf, /*indices=*/NULL,
          /*indices_count=*/0, &static_offset, /*static_indices_count=*/1,
          half_type, reduce_op->location, &slice_op));
      halves[half_index] = loom_vector_slice_result(slice_op);
    }
    IREE_RETURN_IF_ERROR(loom_vector_packet_combine_balanced(
        context, reduce_op, half_type, halves, IREE_ARRAYSIZE(halves),
        /*vector_values=*/true, &folded_leaf));
    leaf_lane_count = half_lane_count;
  }

  const loom_type_t result_type = loom_module_value_type(
      context->module, loom_vector_reduce_result(reduce_op));
  loom_value_id_t lanes[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  for (uint32_t lane_index = 0; lane_index < IREE_ARRAYSIZE(lanes);
       ++lane_index) {
    const int64_t static_index = lane_index;
    loom_op_t* extract_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_extract_build(
        &context->rewriter->builder, folded_leaf, /*indices=*/NULL,
        /*indices_count=*/0, &static_index, /*static_indices_count=*/1,
        result_type, reduce_op->location, &extract_op));
    lanes[lane_index] = loom_vector_extract_result(extract_op);
  }

  loom_value_id_t folded_scalar = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_packet_combine_balanced(
      context, reduce_op, result_type, lanes, IREE_ARRAYSIZE(lanes),
      /*vector_values=*/false, &folded_scalar));
  loom_value_id_t accumulator = loom_vector_reduce_init(reduce_op);
  IREE_RETURN_IF_ERROR(loom_vector_packet_reduce_scalar_component(
      context, reduce_op, folded_scalar, &accumulator));
  *out_accumulator = accumulator;
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_build_identity(
    loom_target_legalization_context_t* context, loom_op_t* reduce_op,
    loom_type_t vector_type, loom_value_id_t* out_value) {
  const loom_scalar_type_t element_type = loom_type_element_type(vector_type);
  loom_attribute_t identity_attr = loom_attr_absent();
  int64_t identity = 0;
  double float_identity = 0.0;
  bool has_identity = loom_combining_kind_integer_identity(
      loom_vector_reduce_kind(reduce_op), element_type, &identity);
  if (has_identity) {
    identity_attr = element_type == LOOM_SCALAR_TYPE_I1
                        ? loom_attr_bool(identity != 0)
                        : loom_attr_i64(identity);
  } else {
    has_identity = loom_combining_kind_float_identity(
        loom_vector_reduce_kind(reduce_op), &float_identity);
    if (has_identity) {
      identity_attr = loom_attr_f64(float_identity);
    }
  }
  IREE_ASSERT_TRUE(has_identity);
  (void)has_identity;
  loom_op_t* identity_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      &context->rewriter->builder, identity_attr, vector_type,
      reduce_op->location, &identity_op));
  *out_value = loom_vector_constant_result(identity_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_packet_normalize_short_component(
    loom_target_legalization_context_t* context, loom_op_t* reduce_op,
    loom_value_id_t component, uint32_t component_lane_count,
    loom_type_t leaf_type, uint32_t leaf_lane_count,
    loom_value_id_t* repeated_components, loom_value_id_t* out_packet) {
  IREE_ASSERT_LT(component_lane_count, leaf_lane_count);
  loom_builder_t* builder = &context->rewriter->builder;
  const bool idempotent =
      loom_combining_kind_is_idempotent(loom_vector_reduce_kind(reduce_op));
  if (idempotent && leaf_lane_count % component_lane_count == 0) {
    const uint32_t repeat_count = leaf_lane_count / component_lane_count;
    for (uint32_t i = 0; i < repeat_count; ++i) {
      repeated_components[i] = component;
    }
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_concat_build(
        builder, /*axis=*/0, repeated_components, repeat_count, leaf_type,
        reduce_op->location, &concat_op));
    *out_packet = loom_vector_concat_result(concat_op);
    return iree_ok_status();
  }

  const uint32_t padding_lane_count = leaf_lane_count - component_lane_count;
  const loom_type_t padding_type =
      loom_vector_packet_type(leaf_type, padding_lane_count);
  loom_value_id_t padding = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_packet_build_identity(
      context, reduce_op, padding_type, &padding));
  const loom_value_id_t concat_operands[] = {component, padding};
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_concat_build(
      builder, /*axis=*/0, concat_operands, IREE_ARRAYSIZE(concat_operands),
      leaf_type, reduce_op->location, &concat_op));
  *out_packet = loom_vector_concat_result(concat_op);
  return iree_ok_status();
}

// Returns the next scalar producer as an insertion anchor when |current|
// reaches it in the same block. Building the pure pack operation there keeps a
// scalar load live only until it enters the rolling packet instead of retaining
// every terminal lane until the reduction.
static loom_op_t* loom_vector_packet_scalar_pack_anchor(
    const loom_module_t* module, loom_value_id_t current, loom_value_id_t next,
    loom_op_t* reduce_op) {
  if (next == LOOM_VALUE_ID_INVALID) {
    return reduce_op;
  }
  const loom_value_t* next_value = loom_module_value(module, next);
  if (next_value == NULL || loom_value_is_block_arg(next_value)) {
    return reduce_op;
  }
  loom_op_t* next_op = loom_value_def_op(next_value);
  if (next_op == NULL || next_op->parent_block != reduce_op->parent_block ||
      next_op->block_ordinal >= reduce_op->block_ordinal) {
    return reduce_op;
  }
  const loom_value_t* current_value = loom_module_value(module, current);
  if (current_value == NULL) {
    return reduce_op;
  }
  if (!loom_value_is_block_arg(current_value)) {
    const loom_op_t* current_op = loom_value_def_op(current_value);
    if (current_op == NULL ||
        current_op->parent_block != next_op->parent_block ||
        current_op->block_ordinal >= next_op->block_ordinal) {
      return reduce_op;
    }
  }
  return next_op;
}

static iree_status_t loom_vector_packet_partition_reduce_packed(
    loom_target_legalization_context_t* context,
    const loom_vector_packet_partition_reduce_plan_t* plan,
    loom_value_fact_vector_partition_t partition, loom_op_t* reduce_op,
    loom_value_id_t* out_accumulator) {
  const loom_value_id_t input = loom_vector_reduce_input(reduce_op);
  const loom_type_t input_type = loom_module_value_type(context->module, input);
  const loom_scalar_type_t element_type = loom_type_element_type(input_type);
  const uint32_t leaf_lane_count = plan->leaf_shape.chunk_lane_count;
  const loom_type_t leaf_type =
      loom_vector_packet_type(input_type, leaf_lane_count);
  const loom_combining_kind_t kind = loom_vector_reduce_kind(reduce_op);
  const bool idempotent = loom_combining_kind_is_idempotent(kind);
  IREE_ASSERT_NE(plan->packed_value_count, 0u);
  loom_value_id_t* packets = NULL;
  loom_value_id_t* scalars = NULL;
  loom_value_id_t* repeated_components = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(context->arena, plan->packed_value_count,
                                sizeof(*packets), (void**)&packets));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, partition.count, sizeof(*scalars), (void**)&scalars));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, leaf_lane_count, sizeof(*repeated_components),
      (void**)&repeated_components));

  loom_builder_t* builder = &context->rewriter->builder;
  uint32_t packet_count = 0;
  uint32_t scalar_count = 0;
  for (uint32_t i = 0; i < partition.count; ++i) {
    const loom_value_fact_vector_partition_component_t component =
        partition.components[i];
    const loom_type_t component_type =
        loom_module_value_type(context->module, component.value);
    if (loom_type_is_scalar(component_type)) {
      scalars[scalar_count++] = component.value;
      continue;
    }
    for (uint32_t lane_offset = 0; lane_offset < component.lane_count;) {
      const uint32_t packet_lane_count =
          iree_min(component.lane_count - lane_offset, leaf_lane_count);
      loom_value_id_t packet = component.value;
      if (lane_offset != 0 || packet_lane_count != component.lane_count) {
        const loom_type_t packet_type =
            loom_vector_packet_type(component_type, packet_lane_count);
        const int64_t static_offset = lane_offset;
        loom_op_t* slice_op = NULL;
        IREE_RETURN_IF_ERROR(loom_vector_slice_build(
            builder, component.value, NULL, 0, &static_offset, 1, packet_type,
            reduce_op->location, &slice_op));
        packet = loom_vector_slice_result(slice_op);
      }
      if (packet_lane_count < leaf_lane_count) {
        IREE_RETURN_IF_ERROR(loom_vector_packet_normalize_short_component(
            context, reduce_op, packet, packet_lane_count, leaf_type,
            leaf_lane_count, repeated_components, &packet));
      }
      packets[packet_count++] = packet;
      lane_offset += packet_lane_count;
    }
  }

  if (idempotent) {
    if (scalar_count != 0) {
      IREE_ASSERT_NE(plan->scalar_pack_lane_count, 0u);
    }
    const loom_builder_ip_t saved_ip = loom_builder_save(builder);
    iree_status_t status = iree_ok_status();
    for (uint32_t scalar_index = 0;
         scalar_index < scalar_count && iree_status_is_ok(status);) {
      const uint32_t group_count =
          iree_min(scalar_count - scalar_index, plan->scalar_pack_lane_count);
      loom_builder_set_before(
          builder,
          loom_vector_packet_scalar_pack_anchor(
              context->module, scalars[scalar_index],
              scalar_index + 1u < scalar_count ? scalars[scalar_index + 1u]
                                               : LOOM_VALUE_ID_INVALID,
              reduce_op));
      loom_op_t* splat_op = NULL;
      status =
          loom_vector_splat_build(builder, scalars[scalar_index], leaf_type,
                                  reduce_op->location, &splat_op);
      if (!iree_status_is_ok(status)) {
        break;
      }
      loom_value_id_t packet = loom_vector_splat_result(splat_op);
      for (uint32_t group_index = 1;
           group_index < group_count && iree_status_is_ok(status);
           ++group_index) {
        const int64_t static_index = group_index - 1u;
        const uint32_t absolute_index = scalar_index + group_index;
        loom_builder_set_before(builder,
                                loom_vector_packet_scalar_pack_anchor(
                                    context->module, scalars[absolute_index],
                                    absolute_index + 1u < scalar_count
                                        ? scalars[absolute_index + 1u]
                                        : LOOM_VALUE_ID_INVALID,
                                    reduce_op));
        loom_op_t* insert_op = NULL;
        status = loom_vector_insert_build(
            builder, scalars[absolute_index], packet, NULL, 0, &static_index, 1,
            leaf_type, reduce_op->location, &insert_op);
        if (iree_status_is_ok(status)) {
          packet = loom_vector_insert_result(insert_op);
        }
      }
      if (iree_status_is_ok(status)) {
        packets[packet_count++] = packet;
        scalar_index += group_count;
      }
    }
    loom_builder_restore(builder, saved_ip);
    IREE_RETURN_IF_ERROR(status);
  } else if (scalar_count != 0) {
    loom_value_id_t scalar = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_packet_combine_balanced(
        context, reduce_op, loom_type_scalar(element_type), scalars,
        scalar_count, /*vector_values=*/false, &scalar));
    loom_value_id_t identity = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_packet_build_identity(
        context, reduce_op, leaf_type, &identity));
    const int64_t static_index = 0;
    loom_op_t* insert_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_insert_build(
        builder, scalar, identity, NULL, 0, &static_index, 1, leaf_type,
        reduce_op->location, &insert_op));
    packets[packet_count++] = loom_vector_insert_result(insert_op);
  }

  IREE_ASSERT_EQ(packet_count, plan->packed_value_count);
  IREE_ASSERT_NE(packet_count, 0u);
  loom_value_id_t folded_packet = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_packet_combine_balanced(
      context, reduce_op, leaf_type, packets, packet_count,
      /*vector_values=*/true, &folded_packet));
  if (!plan->leaf_reduction_is_native) {
    return loom_vector_packet_reduce_folded_leaf(
        context, reduce_op, folded_packet, leaf_lane_count, out_accumulator);
  }
  loom_value_id_t accumulator = loom_vector_reduce_init(reduce_op);
  IREE_RETURN_IF_ERROR(loom_vector_packet_reduce_vector_component(
      context, reduce_op, folded_packet, &accumulator));
  *out_accumulator = accumulator;
  return iree_ok_status();
}

iree_status_t loom_vector_packet_partition_reduce(
    loom_target_legalization_context_t* context,
    const loom_vector_packet_partition_reduce_plan_t* plan,
    loom_value_fact_vector_partition_t partition, loom_op_t* reduce_op,
    loom_value_id_t* out_accumulator) {
  if (plan->mode == LOOM_VECTOR_PACKET_PARTITION_REDUCE_MODE_PACKED) {
    return loom_vector_packet_partition_reduce_packed(
        context, plan, partition, reduce_op, out_accumulator);
  }
  IREE_ASSERT_EQ(plan->mode, LOOM_VECTOR_PACKET_PARTITION_REDUCE_MODE_HYBRID);
  return loom_vector_packet_partition_reduce_hybrid(context, partition,
                                                    reduce_op, out_accumulator);
}

iree_status_t loom_vector_packet_static_reduce(
    loom_vector_packetization_t* packetization, loom_value_id_t input,
    const loom_vector_packet_shape_t* shape, loom_op_t* reduce_op,
    bool fold_packet_inputs, loom_value_id_t* out_accumulator) {
  loom_builder_t* builder = &packetization->context->rewriter->builder;
  const loom_combining_kind_t kind = loom_vector_reduce_kind(reduce_op);
  const uint8_t fastmath_flags = loom_vector_reduce_fastmath(reduce_op);
  const loom_type_t result_type = loom_module_value_type(
      packetization->context->module, loom_vector_reduce_result(reduce_op));
  const loom_value_id_t initial_accumulator =
      loom_vector_reduce_init(reduce_op);
  const bool fold_packets =
      fold_packet_inputs &&
      loom_vector_packet_reduce_can_fold_packets(kind, fastmath_flags);
  loom_value_id_t accumulator = initial_accumulator;
  loom_value_id_t folded_packet = LOOM_VALUE_ID_INVALID;
  loom_type_t folded_packet_type = {0};
  for (uint32_t chunk_index = 0; chunk_index < shape->chunk_count;
       ++chunk_index) {
    const uint32_t lane_offset = chunk_index * shape->chunk_lane_count;
    const loom_vector_packet_slice_t slice = {
        .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
        .static_lane_offset = lane_offset,
        .lane_count =
            iree_min(shape->lane_count - lane_offset, shape->chunk_lane_count),
    };
    loom_vector_packetized_value_t* packetized_input = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_packet_materialize(
        packetization, input, &slice, &packetized_input));
    const loom_type_t packet_type = loom_module_value_type(
        packetization->context->module, packetized_input->packet.vector);

    if (!fold_packets) {
      loom_op_t* packet_reduce_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_reduce_build(
          builder, kind, fastmath_flags, packetized_input->packet.vector,
          accumulator, result_type, reduce_op->location, &packet_reduce_op));
      accumulator = loom_vector_reduce_result(packet_reduce_op);
      continue;
    }

    if (folded_packet != LOOM_VALUE_ID_INVALID &&
        loom_type_equal(folded_packet_type, packet_type)) {
      IREE_RETURN_IF_ERROR(loom_vector_combining_build(
          builder, kind, fastmath_flags, folded_packet,
          packetized_input->packet.vector, packet_type, reduce_op->location,
          &folded_packet));
      continue;
    }

    if (folded_packet != LOOM_VALUE_ID_INVALID) {
      loom_op_t* packet_reduce_op = NULL;
      IREE_RETURN_IF_ERROR(loom_vector_reduce_build(
          builder, kind, fastmath_flags, folded_packet, accumulator,
          result_type, reduce_op->location, &packet_reduce_op));
      accumulator = loom_vector_reduce_result(packet_reduce_op);
    }
    folded_packet = packetized_input->packet.vector;
    folded_packet_type = packet_type;
  }

  if (folded_packet != LOOM_VALUE_ID_INVALID) {
    loom_op_t* packet_reduce_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_reduce_build(
        builder, kind, fastmath_flags, folded_packet, accumulator, result_type,
        reduce_op->location, &packet_reduce_op));
    accumulator = loom_vector_reduce_result(packet_reduce_op);
  }
  *out_accumulator = accumulator;
  return iree_ok_status();
}

iree_status_t loom_vector_packet_erase_dead_sources(
    loom_vector_packetization_t* packetization, const loom_op_t* excluded_op) {
  loom_rewriter_t* rewriter = packetization->context->rewriter;
  for (uint32_t i = packetization->value_count; i > 0; --i) {
    const loom_vector_packetized_value_t* packetized_value =
        &packetization->values[i - 1];
    const loom_value_t* value = loom_module_value(
        packetization->context->module, packetized_value->source);
    if (value == NULL || loom_value_is_block_arg(value)) {
      continue;
    }
    loom_op_t* op = loom_value_def_op(value);
    if (op == NULL || op == excluded_op) {
      continue;
    }
    bool erased = false;
    IREE_RETURN_IF_ERROR(loom_rewriter_erase_if_dead(rewriter, op, &erased));
    if (!erased || packetized_value->source_mode !=
                       LOOM_VECTOR_PACKET_SOURCE_MODE_SNAPSHOT) {
      continue;
    }
    // A retained partition may itself be a balanced concat tree. Remove its
    // direct packet producers with the snapshot root so a legalization
    // worklist cannot revisit dead intermediate carriers before ordinary DCE.
    // Shared packet producers remain protected by erase-if-dead.
    for (uint32_t packet_index =
             packetized_value->source_partition.snapshot_count;
         packet_index > 0; --packet_index) {
      const loom_value_t* packet_value =
          loom_module_value(packetization->context->module,
                            packetized_value->source_partition
                                .snapshot_components[packet_index - 1u]
                                .value);
      if (packet_value == NULL || loom_value_is_block_arg(packet_value)) {
        continue;
      }
      loom_op_t* packet_op = loom_value_def_op(packet_value);
      if (packet_op == NULL || packet_op == excluded_op) {
        continue;
      }
      bool packet_erased = false;
      IREE_RETURN_IF_ERROR(
          loom_rewriter_erase_if_dead(rewriter, packet_op, &packet_erased));
    }
  }
  return iree_ok_status();
}
