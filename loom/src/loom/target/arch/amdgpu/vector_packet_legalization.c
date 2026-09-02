// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/vector_packet_legalization.h"

#include <string.h>

#include "loom/analysis/view_regions.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amdgpu/lower/kinds.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

static bool loom_amdgpu_vector_packet_descriptor_set_is_amdgpu(
    const loom_low_descriptor_set_t* descriptor_set) {
  return descriptor_set != NULL &&
         descriptor_set->target_stable_id == LOOM_AMDGPU_TARGET_STABLE_ID;
}

typedef struct loom_amdgpu_vector_memory_chunk_shape_t {
  // Number of logical lanes in every vector carried by the producer graph.
  uint32_t lane_count;
  // Number of matching logical lanes materialized per native memory packet.
  uint32_t chunk_lane_count;
  // Number of chunks needed to cover lane_count.
  uint32_t chunk_count;
} loom_amdgpu_vector_memory_chunk_shape_t;

// Maximum source packet operations cloned by the alias-preserving static
// fallback. Larger producer graphs stage through one private vector so compile
// time and emitted code remain proportional to the graph instead of the
// product of graph size and packet count.
#define LOOM_AMDGPU_VECTOR_PACKET_STATIC_OP_LIMIT 64u

static bool loom_amdgpu_vector_static_lane_count(loom_type_t vector_type,
                                                 uint32_t* out_lane_count) {
  *out_lane_count = 0;
  if (!loom_type_is_vector(vector_type) || loom_type_rank(vector_type) != 1 ||
      !loom_type_is_all_static(vector_type)) {
    return false;
  }
  const int64_t lane_count_i64 = loom_type_dim_static_size_at(vector_type, 0);
  if (lane_count_i64 < 1 || lane_count_i64 > UINT32_MAX) {
    return false;
  }
  *out_lane_count = (uint32_t)lane_count_i64;
  return true;
}

static bool loom_amdgpu_vector_payload_exceeds_scalarized_limit(
    loom_type_t vector_type, uint32_t lane_count) {
  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(vector_type));
  if (element_bit_count != 8 && element_bit_count != 16 &&
      element_bit_count != 32) {
    return false;
  }
  const uint64_t payload_bit_count =
      (uint64_t)lane_count * (uint32_t)element_bit_count;
  const uint64_t required_32bit_lane_count = (payload_bit_count + 31u) / 32u;
  return required_32bit_lane_count > LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES;
}

static bool loom_amdgpu_vector_memory_chunk_lane_limit(
    loom_type_t vector_type, uint32_t* out_chunk_lane_count) {
  *out_chunk_lane_count = 0;
  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(vector_type));
  if (element_bit_count != 8 && element_bit_count != 16 &&
      element_bit_count != 32) {
    return false;
  }
  *out_chunk_lane_count =
      (LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES * 32u) / (uint32_t)element_bit_count;
  return true;
}

static bool loom_amdgpu_vector_memory_chunk_shape(
    loom_type_t vector_type,
    loom_amdgpu_vector_memory_chunk_shape_t* out_shape) {
  *out_shape = (loom_amdgpu_vector_memory_chunk_shape_t){0};
  uint32_t lane_count = 0;
  uint32_t chunk_lane_count = 0;
  if (!loom_amdgpu_vector_static_lane_count(vector_type, &lane_count) ||
      !loom_amdgpu_vector_memory_chunk_lane_limit(vector_type,
                                                  &chunk_lane_count)) {
    return false;
  }

  // Existing source-to-low memory and vector lowering owns payloads through
  // the bounded scalarized storage limit, including specialized packed and
  // b128 paths. This legalizer extends that boundary for oversized payloads;
  // taking over smaller vectors would replace those established fast paths
  // with loop control.
  if (!loom_amdgpu_vector_payload_exceeds_scalarized_limit(vector_type,
                                                           lane_count)) {
    return false;
  }
  const uint32_t chunk_count =
      (lane_count + chunk_lane_count - 1u) / chunk_lane_count;
  *out_shape = (loom_amdgpu_vector_memory_chunk_shape_t){
      .lane_count = lane_count,
      .chunk_lane_count = chunk_lane_count,
      .chunk_count = chunk_count,
  };
  return true;
}

static bool loom_amdgpu_vector_memory_chunk_shape_constrain(
    loom_type_t vector_type,
    loom_amdgpu_vector_memory_chunk_shape_t* inout_shape) {
  uint32_t lane_count = 0;
  uint32_t chunk_lane_count = 0;
  if (!loom_amdgpu_vector_static_lane_count(vector_type, &lane_count) ||
      lane_count != inout_shape->lane_count ||
      !loom_amdgpu_vector_memory_chunk_lane_limit(vector_type,
                                                  &chunk_lane_count)) {
    return false;
  }
  inout_shape->chunk_lane_count =
      iree_min(inout_shape->chunk_lane_count, chunk_lane_count);
  inout_shape->chunk_count = (lane_count + inout_shape->chunk_lane_count - 1u) /
                             inout_shape->chunk_lane_count;
  return true;
}

static loom_type_t loom_amdgpu_vector_memory_chunk_type(loom_type_t vector_type,
                                                        uint32_t lane_count) {
  return loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, loom_type_element_type(vector_type),
      loom_dim_pack_static(lane_count), vector_type.encoding_id);
}

typedef struct loom_amdgpu_vector_packetized_value_t {
  // Original oversized source value represented by a native packet.
  loom_value_id_t source;
  // Original oversized source type.
  loom_type_t source_type;
  // Native packet materialized for the current physical loop iteration.
  loom_value_id_t packet;
} loom_amdgpu_vector_packetized_value_t;

typedef struct loom_amdgpu_vector_packetization_t {
  // Target legalization context that owns the rewrite.
  loom_target_legalization_context_t* context;
  // Function-local value domain providing direct value-to-ordinal mapping.
  const loom_local_value_domain_t* value_domain;
  // One-based compact value record index keyed directly by value ordinal.
  uint32_t* value_indices;
  // Arena-backed packetized values materialized for this root rewrite.
  loom_amdgpu_vector_packetized_value_t* values;
  // Number of populated values.
  uint32_t value_count;
  // Number of allocated value entries.
  uint32_t value_capacity;
} loom_amdgpu_vector_packetization_t;

static bool loom_amdgpu_vector_memory_find_dynamic_axis_index(
    loom_attribute_t static_indices, iree_host_size_t axis,
    iree_host_size_t* out_dynamic_index) {
  iree_host_size_t dynamic_index = 0;
  for (iree_host_size_t i = 0; i < static_indices.count; ++i) {
    if (static_indices.i64_array[i] != INT64_MIN) {
      continue;
    }
    if (i == axis) {
      *out_dynamic_index = dynamic_index;
      return true;
    }
    ++dynamic_index;
  }
  return false;
}

static bool loom_amdgpu_vector_memory_can_build_chunk_origins(
    const loom_target_legalization_context_t* context,
    const loom_vector_memory_footprint_t* footprint,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape) {
  if (shape->chunk_count <= 1) {
    return true;
  }
  if (footprint->static_indices.count == 0) {
    return false;
  }

  const uint32_t max_lane_offset =
      (shape->chunk_count - 1u) * shape->chunk_lane_count;
  const iree_host_size_t last_axis = footprint->static_indices.count - 1u;
  const int64_t last_static_index =
      footprint->static_indices.i64_array[last_axis];
  if (last_static_index != INT64_MIN) {
    int64_t last_chunk_static_index = 0;
    return iree_checked_add_i64(last_static_index, (int64_t)max_lane_offset,
                                &last_chunk_static_index);
  }

  iree_host_size_t dynamic_index = 0;
  if (!loom_amdgpu_vector_memory_find_dynamic_axis_index(
          footprint->static_indices, last_axis, &dynamic_index) ||
      dynamic_index >= footprint->dynamic_indices.count) {
    return false;
  }
  const loom_value_id_t dynamic_value =
      footprint->dynamic_indices.values[dynamic_index];
  const loom_type_t dynamic_type =
      loom_module_value_type(context->module, dynamic_value);
  return loom_type_is_scalar(dynamic_type) &&
         loom_type_element_type(dynamic_type) == LOOM_SCALAR_TYPE_INDEX;
}

static iree_status_t loom_amdgpu_vector_memory_build_chunk_origin(
    loom_target_legalization_context_t* context,
    const loom_vector_memory_footprint_t* footprint, const loom_op_t* source_op,
    uint32_t chunk_lane_offset, const loom_value_id_t** out_dynamic_indices,
    iree_host_size_t* out_dynamic_index_count,
    const int64_t** out_static_indices,
    iree_host_size_t* out_static_index_count, bool* out_built) {
  *out_dynamic_indices = footprint->dynamic_indices.values;
  *out_dynamic_index_count = footprint->dynamic_indices.count;
  *out_static_indices = footprint->static_indices.i64_array;
  *out_static_index_count = footprint->static_indices.count;
  *out_built = true;
  if (chunk_lane_offset == 0) {
    return iree_ok_status();
  }
  if (footprint->static_indices.count == 0) {
    *out_built = false;
    return iree_ok_status();
  }

  int64_t* static_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->rewriter->builder.arena, footprint->static_indices.count,
      sizeof(*static_indices), (void**)&static_indices));
  memcpy(static_indices, footprint->static_indices.i64_array,
         footprint->static_indices.count * sizeof(*static_indices));

  const iree_host_size_t last_axis = footprint->static_indices.count - 1u;
  if (static_indices[last_axis] != INT64_MIN) {
    if (!iree_checked_add_i64(static_indices[last_axis],
                              (int64_t)chunk_lane_offset,
                              &static_indices[last_axis])) {
      *out_built = false;
      return iree_ok_status();
    }
    *out_static_indices = static_indices;
    return iree_ok_status();
  }

  iree_host_size_t dynamic_index = 0;
  if (!loom_amdgpu_vector_memory_find_dynamic_axis_index(
          footprint->static_indices, last_axis, &dynamic_index) ||
      dynamic_index >= footprint->dynamic_indices.count) {
    *out_built = false;
    return iree_ok_status();
  }
  loom_value_id_t* dynamic_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->rewriter->builder.arena, footprint->dynamic_indices.count,
      sizeof(*dynamic_indices), (void**)&dynamic_indices));
  memcpy(dynamic_indices, footprint->dynamic_indices.values,
         footprint->dynamic_indices.count * sizeof(*dynamic_indices));

  const loom_value_id_t dynamic_value = dynamic_indices[dynamic_index];
  const loom_type_t dynamic_type =
      loom_module_value_type(context->module, dynamic_value);
  if (!loom_type_is_scalar(dynamic_type) ||
      loom_type_element_type(dynamic_type) != LOOM_SCALAR_TYPE_INDEX) {
    *out_built = false;
    return iree_ok_status();
  }
  loom_op_t* offset_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      &context->rewriter->builder, loom_attr_i64(chunk_lane_offset),
      dynamic_type, source_op->location, &offset_op));
  loom_op_t* add_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_add_build(&context->rewriter->builder, dynamic_value,
                           loom_index_constant_result(offset_op), dynamic_type,
                           source_op->location, &add_op));
  dynamic_indices[dynamic_index] = loom_index_add_result(add_op);

  *out_dynamic_indices = dynamic_indices;
  *out_static_indices = static_indices;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_memory_build_dynamic_chunk_origin(
    loom_target_legalization_context_t* context,
    const loom_vector_memory_footprint_t* footprint, const loom_op_t* source_op,
    loom_value_id_t lane_offset, const loom_value_id_t** out_dynamic_indices,
    iree_host_size_t* out_dynamic_index_count,
    const int64_t** out_static_indices,
    iree_host_size_t* out_static_index_count) {
  const iree_host_size_t last_axis = footprint->static_indices.count - 1u;
  const int64_t last_static_index =
      footprint->static_indices.i64_array[last_axis];
  loom_builder_t* builder = &context->rewriter->builder;
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);

  if (last_static_index != INT64_MIN) {
    int64_t* static_indices = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, footprint->static_indices.count,
        sizeof(*static_indices), (void**)&static_indices));
    memcpy(static_indices, footprint->static_indices.i64_array,
           footprint->static_indices.count * sizeof(*static_indices));
    static_indices[last_axis] = INT64_MIN;

    const iree_host_size_t dynamic_index_count =
        footprint->dynamic_indices.count + 1u;
    loom_value_id_t* dynamic_indices = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->arena, dynamic_index_count, sizeof(*dynamic_indices),
        (void**)&dynamic_indices));
    if (footprint->dynamic_indices.count != 0) {
      memcpy(dynamic_indices, footprint->dynamic_indices.values,
             footprint->dynamic_indices.count * sizeof(*dynamic_indices));
    }
    loom_value_id_t dynamic_last_index = lane_offset;
    if (last_static_index != 0) {
      loom_op_t* base_op = NULL;
      IREE_RETURN_IF_ERROR(
          loom_index_constant_build(builder, loom_attr_i64(last_static_index),
                                    index_type, source_op->location, &base_op));
      loom_op_t* add_op = NULL;
      IREE_RETURN_IF_ERROR(loom_index_add_build(
          builder, loom_index_constant_result(base_op), lane_offset, index_type,
          source_op->location, &add_op));
      dynamic_last_index = loom_index_add_result(add_op);
    }
    dynamic_indices[dynamic_index_count - 1u] = dynamic_last_index;

    *out_dynamic_indices = dynamic_indices;
    *out_dynamic_index_count = dynamic_index_count;
    *out_static_indices = static_indices;
    *out_static_index_count = footprint->static_indices.count;
    return iree_ok_status();
  }

  loom_value_id_t* dynamic_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->arena, footprint->dynamic_indices.count,
      sizeof(*dynamic_indices), (void**)&dynamic_indices));
  memcpy(dynamic_indices, footprint->dynamic_indices.values,
         footprint->dynamic_indices.count * sizeof(*dynamic_indices));
  iree_host_size_t dynamic_index = 0;
  const bool found_dynamic_index =
      loom_amdgpu_vector_memory_find_dynamic_axis_index(
          footprint->static_indices, last_axis, &dynamic_index);
  IREE_ASSERT_TRUE(found_dynamic_index);
  (void)found_dynamic_index;
  IREE_ASSERT_LT(dynamic_index, footprint->dynamic_indices.count);
  loom_op_t* add_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_add_build(builder, dynamic_indices[dynamic_index], lane_offset,
                           index_type, source_op->location, &add_op));
  dynamic_indices[dynamic_index] = loom_index_add_result(add_op);

  *out_dynamic_indices = dynamic_indices;
  *out_dynamic_index_count = footprint->dynamic_indices.count;
  *out_static_indices = footprint->static_indices.i64_array;
  *out_static_index_count = footprint->static_indices.count;
  return iree_ok_status();
}

static const loom_fact_context_t* loom_amdgpu_vector_packet_fact_context(
    const loom_target_legalization_context_t* context) {
  return context->fact_table ? &context->fact_table->context : NULL;
}

static bool loom_amdgpu_vector_packet_type_shape_matches(
    loom_type_t type, const loom_amdgpu_vector_memory_chunk_shape_t* shape) {
  uint32_t lane_count = 0;
  return loom_amdgpu_vector_static_lane_count(type, &lane_count) &&
         lane_count == shape->lane_count;
}

static iree_status_t loom_amdgpu_vector_packetization_initialize(
    loom_target_legalization_context_t* context,
    loom_amdgpu_vector_packetization_t* out_packetization) {
  *out_packetization = (loom_amdgpu_vector_packetization_t){
      .context = context,
      .value_domain = context->value_domain,
  };
  const iree_host_size_t value_count = context->value_domain->value_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->arena, value_count, sizeof(*out_packetization->value_indices),
      (void**)&out_packetization->value_indices));
  memset(out_packetization->value_indices, 0,
         value_count * sizeof(*out_packetization->value_indices));
  return iree_ok_status();
}

static loom_amdgpu_vector_packetized_value_t* loom_amdgpu_vector_packet_find(
    loom_amdgpu_vector_packetization_t* packetization, loom_value_id_t source) {
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_ordinal(packetization->value_domain, source);
  const uint32_t value_index = packetization->value_indices[value_ordinal];
  return value_index == 0 ? NULL : &packetization->values[value_index - 1u];
}

static iree_status_t loom_amdgpu_vector_packet_reserve(
    loom_amdgpu_vector_packetization_t* packetization, uint32_t capacity) {
  if (capacity <= packetization->value_capacity) {
    return iree_ok_status();
  }
  uint32_t new_capacity =
      packetization->value_capacity == 0 ? 8u : packetization->value_capacity;
  while (new_capacity < capacity) {
    if (new_capacity > UINT32_MAX / 2u) {
      new_capacity = capacity;
      break;
    }
    new_capacity *= 2u;
  }
  loom_amdgpu_vector_packetized_value_t* new_values = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(packetization->context->arena, new_capacity,
                                sizeof(*new_values), (void**)&new_values));
  if (packetization->value_count != 0) {
    memcpy(new_values, packetization->values,
           packetization->value_count * sizeof(*new_values));
  }
  packetization->values = new_values;
  packetization->value_capacity = new_capacity;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_record(
    loom_amdgpu_vector_packetization_t* packetization, loom_value_id_t source,
    loom_type_t source_type) {
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_reserve(
      packetization, packetization->value_count + 1u));
  const uint32_t value_index = packetization->value_count++;
  packetization->values[value_index] = (loom_amdgpu_vector_packetized_value_t){
      .source = source,
      .source_type = source_type,
      .packet = LOOM_VALUE_ID_INVALID,
  };
  const loom_value_ordinal_t value_ordinal =
      loom_local_value_domain_ordinal(packetization->value_domain, source);
  IREE_ASSERT_EQ(packetization->value_indices[value_ordinal], 0u);
  packetization->value_indices[value_ordinal] = value_index + 1u;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_select_value_shape(
    loom_amdgpu_vector_packetization_t* packetization, loom_value_id_t source,
    loom_amdgpu_vector_memory_chunk_shape_t* inout_shape, bool* out_selected);

static bool loom_amdgpu_vector_packet_select_memory_load_shape(
    const loom_target_legalization_context_t* context, const loom_op_t* op,
    loom_amdgpu_vector_memory_chunk_shape_t* inout_shape) {
  loom_vector_memory_footprint_t footprint = {0};
  if (!loom_vector_memory_footprint_describe(
          loom_amdgpu_vector_packet_fact_context(context), context->module, op,
          &footprint) ||
      footprint.kind != LOOM_VECTOR_MEMORY_FOOTPRINT_DENSE ||
      !loom_type_equal(footprint.vector_type,
                       loom_module_value_type(context->module,
                                              loom_vector_load_result(op)))) {
    return false;
  }
  loom_vector_memory_cache_policy_t cache_policy = {0};
  return loom_vector_memory_cache_policy_from_op(context->module, op,
                                                 &cache_policy) &&
         loom_amdgpu_vector_memory_chunk_shape_constrain(footprint.vector_type,
                                                         inout_shape);
}

static iree_status_t loom_amdgpu_vector_packet_select_op_shape(
    loom_amdgpu_vector_packetization_t* packetization, const loom_op_t* op,
    loom_amdgpu_vector_memory_chunk_shape_t* inout_shape, bool* out_selected) {
  *out_selected = false;
  const loom_target_legalization_context_t* context = packetization->context;
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, op);
  if (!iree_all_bits_set(traits, LOOM_TRAIT_DECOMPOSABLE)) {
    return iree_ok_status();
  }
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_op_results(op)[0]);
  if (!loom_amdgpu_vector_packet_type_shape_matches(result_type, inout_shape)) {
    return iree_ok_status();
  }
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_type_t operand_type =
        loom_module_value_type(context->module, operands[i]);
    if (!loom_amdgpu_vector_packet_type_shape_matches(operand_type,
                                                      inout_shape)) {
      return iree_ok_status();
    }
    bool operand_selected = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_select_value_shape(
        packetization, operands[i], inout_shape, &operand_selected));
    if (!operand_selected) {
      return iree_ok_status();
    }
  }
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_select_value_shape(
    loom_amdgpu_vector_packetization_t* packetization, loom_value_id_t source,
    loom_amdgpu_vector_memory_chunk_shape_t* inout_shape, bool* out_selected) {
  *out_selected = false;
  if (loom_amdgpu_vector_packet_find(packetization, source) != NULL) {
    *out_selected = true;
    return iree_ok_status();
  }
  const loom_target_legalization_context_t* context = packetization->context;
  const loom_value_t* value = loom_module_value(context->module, source);
  if (value == NULL || loom_value_is_block_arg(value)) {
    return iree_ok_status();
  }
  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
  if (!loom_amdgpu_vector_packet_type_shape_matches(source_type, inout_shape)) {
    return iree_ok_status();
  }
  const loom_op_t* op = loom_value_def_op(value);
  if (op == NULL) {
    return iree_ok_status();
  }
  if (loom_vector_load_isa(op)) {
    *out_selected = loom_amdgpu_vector_packet_select_memory_load_shape(
        context, op, inout_shape);
  } else if (loom_vector_constant_isa(op) || loom_vector_poison_isa(op) ||
             loom_vector_splat_isa(op)) {
    *out_selected = true;
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_select_op_shape(
        packetization, op, inout_shape, out_selected));
  }
  if (*out_selected) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vector_packet_record(packetization, source, source_type));
  }
  return iree_ok_status();
}

static bool loom_amdgpu_vector_packet_can_materialize_memory_load(
    const loom_target_legalization_context_t* context, const loom_op_t* op,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape) {
  loom_vector_memory_footprint_t footprint = {0};
  if (!loom_vector_memory_footprint_describe(
          loom_amdgpu_vector_packet_fact_context(context), context->module, op,
          &footprint) ||
      footprint.kind != LOOM_VECTOR_MEMORY_FOOTPRINT_DENSE ||
      !loom_type_equal(footprint.vector_type,
                       loom_module_value_type(context->module,
                                              loom_vector_load_result(op)))) {
    return false;
  }
  loom_vector_memory_cache_policy_t cache_policy = {0};
  return loom_vector_memory_cache_policy_from_op(context->module, op,
                                                 &cache_policy) &&
         loom_amdgpu_vector_memory_can_build_chunk_origins(context, &footprint,
                                                           shape);
}

static bool loom_amdgpu_vector_packet_can_materialize(
    const loom_amdgpu_vector_packetization_t* packetization,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape) {
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    const loom_amdgpu_vector_packetized_value_t* value =
        &packetization->values[i];
    const loom_value_t* source_value =
        loom_module_value(packetization->context->module, value->source);
    const loom_op_t* op = loom_value_def_op(source_value);
    if (loom_vector_load_isa(op) &&
        !loom_amdgpu_vector_packet_can_materialize_memory_load(
            packetization->context, op, shape)) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_vector_packet_footprints_match(
    const loom_vector_memory_footprint_t* left,
    const loom_vector_memory_footprint_t* right) {
  return left->view == right->view &&
         loom_type_equal(left->vector_type, right->vector_type) &&
         loom_attribute_equal(&left->static_indices, &right->static_indices) &&
         left->dynamic_indices.count == right->dynamic_indices.count &&
         (left->dynamic_indices.count == 0 ||
          memcmp(left->dynamic_indices.values, right->dynamic_indices.values,
                 left->dynamic_indices.count * sizeof(loom_value_id_t)) == 0);
}

static bool loom_amdgpu_vector_packet_views_are_disjoint(
    const loom_target_legalization_context_t* context, loom_value_id_t left,
    loom_value_id_t right) {
  loom_value_fact_view_reference_t left_reference = {0};
  loom_value_fact_view_reference_t right_reference = {0};
  if (!loom_value_facts_query_view_reference(
          &context->fact_table->context,
          loom_value_fact_table_lookup(context->fact_table, left),
          &left_reference) ||
      !loom_value_facts_query_view_reference(
          &context->fact_table->context,
          loom_value_fact_table_lookup(context->fact_table, right),
          &right_reference) ||
      left_reference.root_value_id == LOOM_VALUE_ID_INVALID ||
      right_reference.root_value_id == LOOM_VALUE_ID_INVALID ||
      left_reference.root_value_id == right_reference.root_value_id) {
    return false;
  }
  return loom_view_memory_spaces_are_disjoint(left_reference.memory_space,
                                              right_reference.memory_space) ||
         (left_reference.alias_scope_id !=
              LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE &&
          right_reference.alias_scope_id !=
              LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE &&
          left_reference.alias_scope_id != right_reference.alias_scope_id);
}

static bool loom_amdgpu_vector_packet_store_can_interleave(
    const loom_amdgpu_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint) {
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    const loom_amdgpu_vector_packetized_value_t* value =
        &packetization->values[i];
    const loom_value_t* source_value =
        loom_module_value(packetization->context->module, value->source);
    const loom_op_t* op = loom_value_def_op(source_value);
    if (!loom_vector_load_isa(op)) continue;

    loom_vector_memory_footprint_t load_footprint = {0};
    const bool footprint_described = loom_vector_memory_footprint_describe(
        loom_amdgpu_vector_packet_fact_context(packetization->context),
        packetization->context->module, op, &load_footprint);
    IREE_ASSERT_TRUE(footprint_described);
    if (!loom_amdgpu_vector_packet_footprints_match(&load_footprint,
                                                    store_footprint) &&
        !loom_amdgpu_vector_packet_views_are_disjoint(packetization->context,
                                                      load_footprint.view,
                                                      store_footprint->view)) {
      return false;
    }
  }
  return true;
}

typedef struct loom_amdgpu_vector_packet_slice_t {
  // Dynamic logical lane offset, or invalid when the offset is static.
  loom_value_id_t dynamic_lane_offset;
  // Static logical lane offset used when dynamic_lane_offset is invalid.
  uint32_t static_lane_offset;
  // Number of logical lanes represented by the packet.
  uint32_t lane_count;
} loom_amdgpu_vector_packet_slice_t;

static iree_status_t loom_amdgpu_vector_packet_build_memory_origin(
    loom_target_legalization_context_t* context,
    const loom_vector_memory_footprint_t* footprint, const loom_op_t* source_op,
    const loom_amdgpu_vector_packet_slice_t* slice,
    const loom_value_id_t** out_dynamic_indices,
    iree_host_size_t* out_dynamic_index_count,
    const int64_t** out_static_indices,
    iree_host_size_t* out_static_index_count) {
  if (slice->dynamic_lane_offset != LOOM_VALUE_ID_INVALID) {
    return loom_amdgpu_vector_memory_build_dynamic_chunk_origin(
        context, footprint, source_op, slice->dynamic_lane_offset,
        out_dynamic_indices, out_dynamic_index_count, out_static_indices,
        out_static_index_count);
  }
  bool origin_built = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_memory_build_chunk_origin(
      context, footprint, source_op, slice->static_lane_offset,
      out_dynamic_indices, out_dynamic_index_count, out_static_indices,
      out_static_index_count, &origin_built));
  IREE_ASSERT_TRUE(origin_built);
  return iree_ok_status();
}

static void loom_amdgpu_vector_packet_reset(
    loom_amdgpu_vector_packetization_t* packetization) {
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    packetization->values[i].packet = LOOM_VALUE_ID_INVALID;
  }
}

static iree_status_t loom_amdgpu_vector_packet_materialize_memory_load(
    loom_amdgpu_vector_packetization_t* packetization, loom_op_t* op,
    const loom_amdgpu_vector_packet_slice_t* slice,
    loom_amdgpu_vector_packetized_value_t* packetized_value) {
  loom_target_legalization_context_t* context = packetization->context;
  loom_vector_memory_footprint_t footprint = {0};
  const bool footprint_described = loom_vector_memory_footprint_describe(
      loom_amdgpu_vector_packet_fact_context(context), context->module, op,
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
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_build_memory_origin(
      context, &footprint, op, slice, &dynamic_indices, &dynamic_index_count,
      &static_indices, &static_index_count));
  const loom_type_t packet_type = loom_amdgpu_vector_memory_chunk_type(
      packetized_value->source_type, slice->lane_count);
  loom_op_t* packet_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_load_build(
      &context->rewriter->builder, cache_policy.build_flags,
      loom_vector_load_memory_flags(op), footprint.view, dynamic_indices,
      dynamic_index_count, static_indices, static_index_count,
      cache_policy.cache_scope, cache_policy.cache_temporal, packet_type,
      op->location, &packet_op));
  packetized_value->packet = loom_vector_load_result(packet_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_materialize_simple_op(
    loom_amdgpu_vector_packetization_t* packetization, loom_op_t* op,
    const loom_amdgpu_vector_packet_slice_t* slice,
    loom_amdgpu_vector_packetized_value_t* packetized_value) {
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
    loom_amdgpu_vector_packetized_value_t* operand =
        loom_amdgpu_vector_packet_find(packetization,
                                       source_operands[operand_index]);
    IREE_ASSERT(operand != NULL);
    IREE_ASSERT(operand->packet != LOOM_VALUE_ID_INVALID);
    loom_op_operands(packet_op)[operand_index] = operand->packet;
  }
  if (op->attribute_count != 0) {
    memcpy(loom_op_attrs(packet_op), loom_op_const_attrs(op),
           op->attribute_count * sizeof(loom_attribute_t));
  }
  const loom_type_t packet_type = loom_amdgpu_vector_memory_chunk_type(
      packetized_value->source_type, slice->lane_count);
  IREE_RETURN_IF_ERROR(loom_builder_define_value(builder, packet_type,
                                                 &packetized_value->packet));
  loom_op_results(packet_op)[0] = packetized_value->packet;
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(builder, packet_op));
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_materialize_constant(
    loom_amdgpu_vector_packetization_t* packetization, loom_op_t* op,
    const loom_amdgpu_vector_packet_slice_t* slice,
    loom_amdgpu_vector_packetized_value_t* packetized_value) {
  const loom_type_t packet_type = loom_amdgpu_vector_memory_chunk_type(
      packetized_value->source_type, slice->lane_count);
  loom_op_t* packet_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_constant_build(
      &packetization->context->rewriter->builder,
      loom_vector_constant_value(op), packet_type, op->location, &packet_op));
  packetized_value->packet = loom_vector_constant_result(packet_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_materialize_poison(
    loom_amdgpu_vector_packetization_t* packetization, loom_op_t* op,
    const loom_amdgpu_vector_packet_slice_t* slice,
    loom_amdgpu_vector_packetized_value_t* packetized_value) {
  const loom_type_t packet_type = loom_amdgpu_vector_memory_chunk_type(
      packetized_value->source_type, slice->lane_count);
  loom_op_t* packet_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_vector_poison_build(&packetization->context->rewriter->builder,
                               packet_type, op->location, &packet_op));
  packetized_value->packet = loom_vector_poison_result(packet_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_materialize_splat(
    loom_amdgpu_vector_packetization_t* packetization, loom_op_t* op,
    const loom_amdgpu_vector_packet_slice_t* slice,
    loom_amdgpu_vector_packetized_value_t* packetized_value) {
  const loom_type_t packet_type = loom_amdgpu_vector_memory_chunk_type(
      packetized_value->source_type, slice->lane_count);
  loom_op_t* packet_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(
      &packetization->context->rewriter->builder, loom_vector_splat_scalar(op),
      packet_type, op->location, &packet_op));
  packetized_value->packet = loom_vector_splat_result(packet_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_materialize_value(
    loom_amdgpu_vector_packetization_t* packetization,
    const loom_amdgpu_vector_packet_slice_t* slice,
    loom_amdgpu_vector_packetized_value_t* packetized_value) {
  loom_target_legalization_context_t* context = packetization->context;
  const loom_value_t* value =
      loom_module_value(context->module, packetized_value->source);
  loom_op_t* op = loom_value_def_op(value);
  if (loom_vector_load_isa(op)) {
    return loom_amdgpu_vector_packet_materialize_memory_load(
        packetization, op, slice, packetized_value);
  } else if (loom_vector_constant_isa(op)) {
    return loom_amdgpu_vector_packet_materialize_constant(
        packetization, op, slice, packetized_value);
  } else if (loom_vector_poison_isa(op)) {
    return loom_amdgpu_vector_packet_materialize_poison(
        packetization, op, slice, packetized_value);
  } else if (loom_vector_splat_isa(op)) {
    return loom_amdgpu_vector_packet_materialize_splat(packetization, op, slice,
                                                       packetized_value);
  }
  return loom_amdgpu_vector_packet_materialize_simple_op(
      packetization, op, slice, packetized_value);
}

static iree_status_t loom_amdgpu_vector_packet_materialize(
    loom_amdgpu_vector_packetization_t* packetization, loom_value_id_t source,
    const loom_amdgpu_vector_packet_slice_t* slice,
    loom_amdgpu_vector_packetized_value_t** out_value) {
  loom_amdgpu_vector_packet_reset(packetization);
  for (uint32_t i = 0; i < packetization->value_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize_value(
        packetization, slice, &packetization->values[i]));
  }
  *out_value = loom_amdgpu_vector_packet_find(packetization, source);
  IREE_ASSERT(*out_value != NULL);
  IREE_ASSERT((*out_value)->packet != LOOM_VALUE_ID_INVALID);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_store(
    loom_amdgpu_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint,
    const loom_amdgpu_vector_packet_slice_t* slice, loom_value_id_t packet,
    loom_vector_memory_cache_policy_t store_cache_policy, loom_op_t* store_op) {
  loom_target_legalization_context_t* context = packetization->context;
  const loom_value_id_t* dynamic_indices = NULL;
  iree_host_size_t dynamic_index_count = 0;
  const int64_t* static_indices = NULL;
  iree_host_size_t static_index_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_build_memory_origin(
      context, store_footprint, store_op, slice, &dynamic_indices,
      &dynamic_index_count, &static_indices, &static_index_count));
  loom_op_t* packet_store_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_store_build(
      &context->rewriter->builder, store_cache_policy.build_flags,
      loom_vector_store_memory_flags(store_op), packet, store_footprint->view,
      dynamic_indices, dynamic_index_count, static_indices, static_index_count,
      store_cache_policy.cache_scope, store_cache_policy.cache_temporal,
      store_op->location, &packet_store_op));
  return iree_ok_status();
}

static bool loom_amdgpu_vector_packet_static_store_is_bounded(
    const loom_amdgpu_vector_packetization_t* packetization,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape) {
  iree_host_size_t operations_per_chunk = 0;
  iree_host_size_t operation_count = 0;
  return iree_host_size_checked_add(packetization->value_count, 1,
                                    &operations_per_chunk) &&
         iree_host_size_checked_mul(operations_per_chunk, shape->chunk_count,
                                    &operation_count) &&
         operation_count <= LOOM_AMDGPU_VECTOR_PACKET_STATIC_OP_LIMIT;
}

static iree_status_t loom_amdgpu_vector_packet_static_store(
    loom_amdgpu_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape,
    loom_vector_memory_cache_policy_t store_cache_policy, loom_op_t* store_op) {
  const iree_host_size_t packet_count =
      (iree_host_size_t)packetization->value_count * shape->chunk_count;
  IREE_ASSERT_LE(packet_count, LOOM_AMDGPU_VECTOR_PACKET_STATIC_OP_LIMIT);
  loom_value_id_t* packets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(packetization->context->arena,
                                                 packet_count, sizeof(*packets),
                                                 (void**)&packets));
  for (uint32_t value_index = 0; value_index < packetization->value_count;
       ++value_index) {
    loom_amdgpu_vector_packetized_value_t* packetized_value =
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
        loom_amdgpu_vector_packetized_value_t* operand =
            loom_amdgpu_vector_packet_find(packetization,
                                           source_operands[operand_index]);
        if (operand == NULL) continue;
        const uint32_t operand_value_index =
            (uint32_t)(operand - packetization->values);
        operand->packet =
            packets[operand_value_index * shape->chunk_count + chunk_index];
      }
      const uint32_t lane_offset = chunk_index * shape->chunk_lane_count;
      const loom_amdgpu_vector_packet_slice_t slice = {
          .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
          .static_lane_offset = lane_offset,
          .lane_count = iree_min(shape->lane_count - lane_offset,
                                 shape->chunk_lane_count),
      };
      IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize_value(
          packetization, &slice, packetized_value));
      packets[value_index * shape->chunk_count + chunk_index] =
          packetized_value->packet;
    }
  }
  loom_amdgpu_vector_packetized_value_t* root =
      loom_amdgpu_vector_packet_find(packetization, store_footprint->value);
  IREE_ASSERT(root != NULL);
  const uint32_t root_value_index = (uint32_t)(root - packetization->values);
  for (uint32_t chunk_index = 0; chunk_index < shape->chunk_count;
       ++chunk_index) {
    const uint32_t lane_offset = chunk_index * shape->chunk_lane_count;
    const loom_amdgpu_vector_packet_slice_t slice = {
        .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
        .static_lane_offset = lane_offset,
        .lane_count =
            iree_min(shape->lane_count - lane_offset, shape->chunk_lane_count),
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_store(
        packetization, store_footprint, &slice,
        packets[root_value_index * shape->chunk_count + chunk_index],
        store_cache_policy, store_op));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_build_staging_view(
    loom_amdgpu_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape, loom_op_t* store_op,
    loom_value_id_t* out_staging_view) {
  *out_staging_view = LOOM_VALUE_ID_INVALID;
  const int32_t element_bit_count = loom_scalar_type_bitwidth(
      loom_type_element_type(store_footprint->vector_type));
  IREE_ASSERT_GT(element_bit_count, 0);
  IREE_ASSERT_EQ(element_bit_count % 8, 0);
  const int64_t byte_count =
      (int64_t)shape->lane_count * (element_bit_count / 8);
  loom_builder_t* builder = &packetization->context->rewriter->builder;
  const loom_type_t offset_type = loom_type_scalar(LOOM_SCALAR_TYPE_OFFSET);

  loom_op_t* byte_count_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(builder, loom_attr_i64(byte_count), offset_type,
                                store_op->location, &byte_count_op));
  loom_op_t* staging_buffer_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_alloca_build(
      builder, LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE,
      /*base_alignment=*/16, loom_index_constant_result(byte_count_op),
      loom_type_buffer(), store_op->location, &staging_buffer_op));
  loom_op_t* zero_offset_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(builder, loom_attr_i64(0), offset_type,
                                store_op->location, &zero_offset_op));
  const loom_type_t staging_view_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, loom_type_element_type(store_footprint->vector_type),
      loom_dim_pack_static(shape->lane_count), /*encoding_id=*/0);
  loom_op_t* staging_view_op = NULL;
  IREE_RETURN_IF_ERROR(loom_buffer_view_build(
      builder, loom_buffer_alloca_result(staging_buffer_op),
      loom_index_constant_result(zero_offset_op), staging_view_type,
      store_op->location, &staging_view_op));
  *out_staging_view = loom_buffer_view_result(staging_view_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_staging_store(
    loom_amdgpu_vector_packetization_t* packetization,
    loom_value_id_t staging_view,
    const loom_amdgpu_vector_packet_slice_t* slice, loom_value_id_t packet,
    loom_op_t* store_op) {
  const loom_value_id_t* dynamic_indices = NULL;
  iree_host_size_t dynamic_index_count = 0;
  int64_t static_index = slice->static_lane_offset;
  if (slice->dynamic_lane_offset != LOOM_VALUE_ID_INVALID) {
    dynamic_indices = &slice->dynamic_lane_offset;
    dynamic_index_count = 1;
    static_index = INT64_MIN;
  }
  loom_op_t* staging_store_op = NULL;
  return loom_vector_store_build(
      &packetization->context->rewriter->builder, /*build_flags=*/0,
      /*instance_flags=*/0, packet, staging_view, dynamic_indices,
      dynamic_index_count, &static_index,
      /*static_indices_count=*/1, /*cache_scope=*/0, /*cache_temporal=*/0,
      store_op->location, &staging_store_op);
}

static iree_status_t loom_amdgpu_vector_packet_staging_load(
    loom_amdgpu_vector_packetization_t* packetization,
    loom_value_id_t staging_view, loom_type_t source_type,
    const loom_amdgpu_vector_packet_slice_t* slice, loom_op_t* store_op,
    loom_value_id_t* out_packet) {
  *out_packet = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t* dynamic_indices = NULL;
  iree_host_size_t dynamic_index_count = 0;
  int64_t static_index = slice->static_lane_offset;
  if (slice->dynamic_lane_offset != LOOM_VALUE_ID_INVALID) {
    dynamic_indices = &slice->dynamic_lane_offset;
    dynamic_index_count = 1;
    static_index = INT64_MIN;
  }
  const loom_type_t packet_type =
      loom_amdgpu_vector_memory_chunk_type(source_type, slice->lane_count);
  loom_op_t* staging_load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_load_build(
      &packetization->context->rewriter->builder, /*build_flags=*/0,
      /*instance_flags=*/0, staging_view, dynamic_indices, dynamic_index_count,
      &static_index,
      /*static_indices_count=*/1, /*cache_scope=*/0, /*cache_temporal=*/0,
      packet_type, store_op->location, &staging_load_op));
  *out_packet = loom_vector_load_result(staging_load_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_staged_store(
    loom_amdgpu_vector_packetization_t* packetization,
    const loom_vector_memory_footprint_t* store_footprint,
    const loom_amdgpu_vector_memory_chunk_shape_t* shape,
    loom_vector_memory_cache_policy_t store_cache_policy, loom_op_t* store_op) {
  loom_builder_t* builder = &packetization->context->rewriter->builder;
  loom_value_id_t staging_view = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_build_staging_view(
      packetization, store_footprint, shape, store_op, &staging_view));

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
      /*iter_args_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, store_op->location,
      &stage_loop_op));
  loom_builder_ip_t saved_ip = loom_builder_enter_region(
      builder, stage_loop_op, loom_scf_for_body(stage_loop_op));
  const loom_amdgpu_vector_packet_slice_t loop_slice = {
      .dynamic_lane_offset =
          loom_region_entry_arg_id(loom_scf_for_body(stage_loop_op), 0),
      .lane_count = shape->chunk_lane_count,
  };
  loom_amdgpu_vector_packetized_value_t* packetized_value = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize(
      packetization, store_footprint->value, &loop_slice, &packetized_value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_staging_store(
      packetization, staging_view, &loop_slice, packetized_value->packet,
      store_op));
  loom_op_t* stage_yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(
      builder, NULL, 0, store_op->location, &stage_yield_op));
  loom_builder_restore(builder, saved_ip);

  if (loop_lane_count < shape->lane_count) {
    const loom_amdgpu_vector_packet_slice_t tail_slice = {
        .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
        .static_lane_offset = loop_lane_count,
        .lane_count = shape->lane_count - loop_lane_count,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize(
        packetization, store_footprint->value, &tail_slice, &packetized_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_staging_store(
        packetization, staging_view, &tail_slice, packetized_value->packet,
        store_op));
  }

  loom_op_t* commit_loop_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      builder, /*build_flags=*/0, loom_index_constant_result(lower_bound_op),
      loom_index_constant_result(upper_bound_op),
      loom_index_constant_result(step_op), /*iter_args=*/NULL,
      /*iter_args_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, store_op->location,
      &commit_loop_op));
  saved_ip = loom_builder_enter_region(builder, commit_loop_op,
                                       loom_scf_for_body(commit_loop_op));
  const loom_amdgpu_vector_packet_slice_t commit_loop_slice = {
      .dynamic_lane_offset =
          loom_region_entry_arg_id(loom_scf_for_body(commit_loop_op), 0),
      .lane_count = shape->chunk_lane_count,
  };
  loom_value_id_t staged_packet = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_staging_load(
      packetization, staging_view, store_footprint->vector_type,
      &commit_loop_slice, store_op, &staged_packet));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_store(
      packetization, store_footprint, &commit_loop_slice, staged_packet,
      store_cache_policy, store_op));
  loom_op_t* commit_yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(
      builder, NULL, 0, store_op->location, &commit_yield_op));
  loom_builder_restore(builder, saved_ip);

  if (loop_lane_count < shape->lane_count) {
    const loom_amdgpu_vector_packet_slice_t tail_slice = {
        .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
        .static_lane_offset = loop_lane_count,
        .lane_count = shape->lane_count - loop_lane_count,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_staging_load(
        packetization, staging_view, store_footprint->vector_type, &tail_slice,
        store_op, &staged_packet));
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_store(
        packetization, store_footprint, &tail_slice, staged_packet,
        store_cache_policy, store_op));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_vector_packet_erase_dead_sources(
    loom_amdgpu_vector_packetization_t* packetization) {
  loom_rewriter_t* rewriter = packetization->context->rewriter;
  for (uint32_t i = packetization->value_count; i > 0; --i) {
    const loom_amdgpu_vector_packetized_value_t* packetized_value =
        &packetization->values[i - 1];
    const loom_value_t* value = loom_module_value(
        packetization->context->module, packetized_value->source);
    if (value == NULL || loom_value_is_block_arg(value)) continue;
    loom_op_t* op = loom_value_def_op(value);
    if (op == NULL) continue;
    bool erased = false;
    IREE_RETURN_IF_ERROR(loom_rewriter_erase_if_dead(rewriter, op, &erased));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_legalize_oversized_vector_store(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_amdgpu_vector_packet_descriptor_set_is_amdgpu(
          context->descriptor_set)) {
    return iree_ok_status();
  }

  loom_vector_memory_footprint_t store_footprint = {0};
  if (!loom_vector_memory_footprint_describe(
          loom_amdgpu_vector_packet_fact_context(context), context->module, op,
          &store_footprint) ||
      store_footprint.kind != LOOM_VECTOR_MEMORY_FOOTPRINT_DENSE) {
    return iree_ok_status();
  }
  loom_amdgpu_vector_memory_chunk_shape_t shape = {0};
  if (!loom_amdgpu_vector_memory_chunk_shape(store_footprint.vector_type,
                                             &shape)) {
    return iree_ok_status();
  }

  loom_amdgpu_vector_packetization_t packetization = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_packetization_initialize(context, &packetization));
  bool producer_selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_select_value_shape(
      &packetization, store_footprint.value, &shape, &producer_selected));

  loom_vector_memory_cache_policy_t store_cache_policy = {0};
  if (!producer_selected ||
      !loom_vector_memory_cache_policy_from_op(context->module, op,
                                               &store_cache_policy) ||
      !loom_amdgpu_vector_memory_can_build_chunk_origins(
          context, &store_footprint, &shape) ||
      !loom_amdgpu_vector_packet_can_materialize(&packetization, &shape)) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  if (!loom_amdgpu_vector_packet_store_can_interleave(&packetization,
                                                      &store_footprint)) {
    if (loom_amdgpu_vector_packet_static_store_is_bounded(&packetization,
                                                          &shape)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_static_store(
          &packetization, &store_footprint, &shape, store_cache_policy, op));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_staged_store(
          &packetization, &store_footprint, &shape, store_cache_policy, op));
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_vector_packet_erase_dead_sources(&packetization));
    *out_result = (loom_target_legalizer_result_t){
        .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
    };
    return iree_ok_status();
  }

  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const uint32_t loop_lane_count =
      (shape.lane_count / shape.chunk_lane_count) * shape.chunk_lane_count;
  loom_op_t* lower_bound_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(0), index_type, op->location, &lower_bound_op));
  loom_op_t* upper_bound_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(builder, loom_attr_i64(loop_lane_count),
                                index_type, op->location, &upper_bound_op));
  loom_op_t* step_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(builder, loom_attr_i64(shape.chunk_lane_count),
                                index_type, op->location, &step_op));
  loom_op_t* loop_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      builder, /*build_flags=*/0, loom_index_constant_result(lower_bound_op),
      loom_index_constant_result(upper_bound_op),
      loom_index_constant_result(step_op), /*iter_args=*/NULL,
      /*iter_args_count=*/0, /*tied_results=*/NULL,
      /*tied_result_count=*/0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, op->location, &loop_op));

  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(builder, loop_op, loom_scf_for_body(loop_op));
  const loom_amdgpu_vector_packet_slice_t loop_slice = {
      .dynamic_lane_offset =
          loom_region_entry_arg_id(loom_scf_for_body(loop_op), 0),
      .lane_count = shape.chunk_lane_count,
  };
  loom_amdgpu_vector_packetized_value_t* packetized_value = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize(
      &packetization, store_footprint.value, &loop_slice, &packetized_value));
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_store(
      &packetization, &store_footprint, &loop_slice, packetized_value->packet,
      store_cache_policy, op));
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scf_yield_build(builder, NULL, 0, op->location, &yield_op));
  loom_builder_restore(builder, saved_ip);

  if (loop_lane_count < shape.lane_count) {
    const loom_amdgpu_vector_packet_slice_t tail_slice = {
        .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
        .static_lane_offset = loop_lane_count,
        .lane_count = shape.lane_count - loop_lane_count,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize(
        &packetization, store_footprint.value, &tail_slice, &packetized_value));
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_store(
        &packetization, &store_footprint, &tail_slice, packetized_value->packet,
        store_cache_policy, op));
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_packet_erase_dead_sources(&packetization));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}

iree_status_t loom_amdgpu_legalize_oversized_vector_reduce(
    const loom_target_legalizer_entry_t* entry,
    loom_target_legalization_context_t* context, loom_op_t* op,
    loom_target_legalizer_result_t* out_result) {
  (void)entry;
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_NO_COMMENT,
  };
  if (!loom_amdgpu_vector_packet_descriptor_set_is_amdgpu(
          context->descriptor_set)) {
    return iree_ok_status();
  }

  const loom_value_id_t input = loom_vector_reduce_input(op);
  const loom_type_t input_type = loom_module_value_type(context->module, input);
  loom_amdgpu_vector_memory_chunk_shape_t shape = {0};
  if (!loom_amdgpu_vector_memory_chunk_shape(input_type, &shape)) {
    return iree_ok_status();
  }

  loom_amdgpu_vector_packetization_t packetization = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_packetization_initialize(context, &packetization));
  bool producer_selected = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_select_value_shape(
      &packetization, input, &shape, &producer_selected));
  if (!producer_selected ||
      !loom_amdgpu_vector_packet_can_materialize(&packetization, &shape)) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  const loom_combining_kind_t kind = loom_vector_reduce_kind(op);
  const uint8_t fastmath_flags = loom_vector_reduce_fastmath(op);
  const loom_type_t result_type =
      loom_module_value_type(context->module, loom_vector_reduce_result(op));
  const loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  const uint32_t loop_lane_count =
      (shape.lane_count / shape.chunk_lane_count) * shape.chunk_lane_count;
  loom_op_t* lower_bound_op = NULL;
  IREE_RETURN_IF_ERROR(loom_index_constant_build(
      builder, loom_attr_i64(0), index_type, op->location, &lower_bound_op));
  loom_op_t* upper_bound_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(builder, loom_attr_i64(loop_lane_count),
                                index_type, op->location, &upper_bound_op));
  loom_op_t* step_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_index_constant_build(builder, loom_attr_i64(shape.chunk_lane_count),
                                index_type, op->location, &step_op));
  const loom_value_id_t initial_accumulator = loom_vector_reduce_init(op);
  loom_op_t* loop_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_for_build(
      builder, /*build_flags=*/0, loom_index_constant_result(lower_bound_op),
      loom_index_constant_result(upper_bound_op),
      loom_index_constant_result(step_op), &initial_accumulator, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, op->location, &loop_op));

  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(builder, loop_op, loom_scf_for_body(loop_op));
  const loom_amdgpu_vector_packet_slice_t loop_slice = {
      .dynamic_lane_offset =
          loom_region_entry_arg_id(loom_scf_for_body(loop_op), 0),
      .lane_count = shape.chunk_lane_count,
  };
  loom_amdgpu_vector_packetized_value_t* packetized_input = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize(
      &packetization, input, &loop_slice, &packetized_input));
  loom_op_t* packet_reduce_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_reduce_build(
      builder, kind, fastmath_flags, packetized_input->packet,
      loom_region_entry_arg_id(loom_scf_for_body(loop_op), 1), result_type,
      op->location, &packet_reduce_op));
  const loom_value_id_t next_accumulator =
      loom_vector_reduce_result(packet_reduce_op);
  loom_op_t* yield_op = NULL;
  IREE_RETURN_IF_ERROR(loom_scf_yield_build(builder, &next_accumulator, 1,
                                            op->location, &yield_op));
  loom_builder_restore(builder, saved_ip);

  loom_value_id_t accumulator = loom_scf_for_results(loop_op).values[0];
  if (loop_lane_count < shape.lane_count) {
    const loom_amdgpu_vector_packet_slice_t tail_slice = {
        .dynamic_lane_offset = LOOM_VALUE_ID_INVALID,
        .static_lane_offset = loop_lane_count,
        .lane_count = shape.lane_count - loop_lane_count,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_vector_packet_materialize(
        &packetization, input, &tail_slice, &packetized_input));
    loom_op_t* tail_reduce_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_reduce_build(
        builder, kind, fastmath_flags, packetized_input->packet, accumulator,
        result_type, op->location, &tail_reduce_op));
    accumulator = loom_vector_reduce_result(tail_reduce_op);
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &accumulator, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &accumulator, 1));
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_vector_packet_erase_dead_sources(&packetization));
  *out_result = (loom_target_legalizer_result_t){
      .action = LOOM_TARGET_LEGALIZER_ACTION_REWRITTEN,
  };
  return iree_ok_status();
}
