// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/lower/vector_load.h"

#include "iree/base/internal/math.h"
#include "loom/codegen/low/builder.h"
#include "loom/codegen/low/lower/source_memory.h"
#include "loom/codegen/low/source_memory_plan.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/contracts/core_lower_rules.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/vector_packet.h"
#include "loom/target/arch/amd/xdna/aie2p/vector_carrier.h"

enum {
  LOOM_AIE2P_VECTOR_LOAD_PLAN_SCALAR_PACKETS = 0x405,
  LOOM_AIE2P_VECTOR_LOAD_OFFSET_CHUNK_MAXIMUM = 127,
};

typedef struct loom_aie2p_vector_load_plan_t {
  // Function-owned canonical source access selected with this target plan.
  const loom_low_source_memory_access_plan_t* access;
  // Function-owned dynamic memory-effect proof for the canonical access.
  const loom_low_lower_memory_origin_t* origin;
  // Number of logical source lanes.
  uint16_t lane_count;
  // Physical width of one source lane.
  uint8_t element_bit_count;
  // Logical source lanes represented by one X packet.
  uint8_t packet_lane_count;
  // Number of packets containing logical source lanes.
  uint8_t logical_packet_count;
  // Native result carrier family.
  uint8_t carrier_kind;
  // Number of allocation units in the complete result carrier.
  uint8_t carrier_unit_count;
} loom_aie2p_vector_load_plan_t;

static bool loom_aie2p_vector_load_memory_space_supported(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
      return true;
    default:
      return false;
  }
}

static bool loom_aie2p_vector_load_element_supported(
    loom_scalar_type_t element_type) {
  switch (element_type) {
    case LOOM_SCALAR_TYPE_I8:
    case LOOM_SCALAR_TYPE_F8E4M3:
    case LOOM_SCALAR_TYPE_F8E5M2:
    case LOOM_SCALAR_TYPE_I16:
    case LOOM_SCALAR_TYPE_F16:
    case LOOM_SCALAR_TYPE_BF16:
    case LOOM_SCALAR_TYPE_I32:
    case LOOM_SCALAR_TYPE_F32:
    case LOOM_SCALAR_TYPE_I64:
    case LOOM_SCALAR_TYPE_F64:
      return true;
    default:
      return false;
  }
}

static bool loom_aie2p_vector_load_plan_match(
    const loom_module_t* module, const loom_op_t* source_op,
    const loom_low_source_memory_access_plan_t* access,
    loom_aie2p_vector_load_plan_t* out_plan) {
  *out_plan = (loom_aie2p_vector_load_plan_t){0};
  if (!loom_vector_load_isa(source_op) || access == NULL ||
      access->operation_kind != LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD ||
      access->vector_offset_kind != LOOM_LOW_SOURCE_MEMORY_VECTOR_OFFSET_NONE ||
      access->cache_policy.build_flags != 0 ||
      access->minimum_alignment >= 16 ||
      !loom_aie2p_vector_load_memory_space_supported(access->memory_space)) {
    return false;
  }

  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_load_result(source_op));
  if (!loom_type_is_vector(result_type) || loom_type_rank(result_type) != 1 ||
      loom_type_dim_is_dynamic_at(result_type, 0)) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(result_type);
  if (!loom_aie2p_vector_load_element_supported(element_type)) {
    return false;
  }
  const uint16_t element_bit_count =
      loom_aie2p_scalar_type_physical_bit_count(element_type);
  const int64_t lane_count = loom_type_dim_static_size_at(result_type, 0);
  if (lane_count <= 0 || lane_count > UINT16_MAX ||
      access->element_byte_count != element_bit_count / 8 ||
      access->vector_lane_count != (uint32_t)lane_count ||
      access->vector_lane_byte_stride != access->element_byte_count) {
    return false;
  }

  const uint32_t scalar_unit_count =
      (uint32_t)lane_count * (element_bit_count == 64 ? 2u : 1u);
  if (scalar_unit_count <= 16) {
    return false;
  }
  const loom_aie2p_vector_carrier_t carrier =
      loom_aie2p_vector_carrier_for_type(result_type);
  if (carrier.kind == LOOM_AIE2P_VECTOR_CARRIER_NONE ||
      carrier.unit_count > UINT8_MAX) {
    return false;
  }

  int64_t access_byte_count = 0;
  int64_t final_byte_offset = 0;
  if (!iree_checked_mul_i64(lane_count, access->element_byte_count,
                            &access_byte_count) ||
      !iree_checked_add_i64(access->static_byte_offset, access_byte_count - 1,
                            &final_byte_offset) ||
      !loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
          access, access->static_byte_offset, 32) ||
      !loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
          access, final_byte_offset, 32)) {
    return false;
  }

  const uint8_t packet_lane_count = (uint8_t)(512 / element_bit_count);
  const uint8_t logical_packet_count =
      (uint8_t)((lane_count + packet_lane_count - 1) / packet_lane_count);
  if (logical_packet_count == 0 ||
      logical_packet_count > LOOM_AIE2P_VECTOR_PACKET_MAX_COUNT) {
    return false;
  }
  *out_plan = (loom_aie2p_vector_load_plan_t){
      .lane_count = (uint16_t)lane_count,
      .element_bit_count = (uint8_t)element_bit_count,
      .packet_lane_count = packet_lane_count,
      .logical_packet_count = logical_packet_count,
      .carrier_kind = carrier.kind,
      .carrier_unit_count = (uint8_t)carrier.unit_count,
  };
  return true;
}

iree_status_t loom_aie2p_query_vector_load_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  (void)user_data;
  *out_result = loom_target_contract_query_result_empty();
  if (!loom_vector_load_isa(source_op) ||
      environment->vector_lane_projection.source_lane_count != 0) {
    return iree_ok_status();
  }

  loom_target_contract_query_environment_t analyzed_environment = *environment;
  IREE_RETURN_IF_ERROR(loom_target_contract_query_resolve_function_analysis(
      &analyzed_environment));
  loom_low_source_memory_access_plan_t access = {0};
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  loom_aie2p_vector_load_plan_t plan = {0};
  if (loom_low_source_memory_access_plan_build(
          analyzed_environment.view_regions, source_op,
          /*axis_byte_strides=*/NULL, &access, &diagnostic) &&
      loom_aie2p_vector_load_plan_match(analyzed_environment.module, source_op,
                                        &access, &plan)) {
    out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  }
  return iree_ok_status();
}

bool loom_aie2p_vector_load_plan_isa(loom_low_lower_plan_t plan) {
  return plan.id == LOOM_AIE2P_VECTOR_LOAD_PLAN_SCALAR_PACKETS;
}

iree_status_t loom_aie2p_prepare_vector_load_plan(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  *out_plan = loom_low_lower_plan_empty();
  if (!loom_vector_load_isa(source_op)) {
    return iree_ok_status();
  }
  loom_low_source_memory_access_diagnostic_t diagnostic = {0};
  const loom_low_source_memory_access_plan_t* access =
      loom_low_lower_source_memory_access(context, source_op, &diagnostic);
  loom_aie2p_vector_load_plan_t matched_plan = {0};
  if (!loom_aie2p_vector_load_plan_match(loom_low_lower_context_module(context),
                                         source_op, access, &matched_plan)) {
    return iree_ok_status();
  }

  loom_aie2p_vector_load_plan_t* retained_plan = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*retained_plan), (void**)&retained_plan));
  *retained_plan = matched_plan;
  retained_plan->access = access;
  retained_plan->origin = loom_low_lower_source_memory_origin(context);
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_make_carrier_type(
      context, retained_plan->carrier_kind, retained_plan->carrier_unit_count,
      &result_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
      context, loom_vector_load_result(source_op), result_type));
  *out_plan = loom_low_lower_plan_make(
      LOOM_AIE2P_VECTOR_LOAD_PLAN_SCALAR_PACKETS, retained_plan);
  out_plan->access_flags = access->access_flags;
  return iree_ok_status();
}

void loom_aie2p_mark_vector_load_plan_demands(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t plan) {
  (void)source_op;
  const loom_aie2p_vector_load_plan_t* load_plan = plan.target_data;
  loom_low_lower_require_source_memory_complete_byte_offset_storage(
      context, load_plan->access);
}

void loom_aie2p_describe_vector_load_plan(
    loom_low_lower_plan_report_t* out_report) {
  *out_report = (loom_low_lower_plan_report_t){
      .plan_key = IREE_SV("memory.load.underaligned.scalar-packets"),
  };
}

static iree_status_t loom_aie2p_vector_load_immediate_attr(
    loom_aie2p_vector_packet_emitter_t* emitter,
    const loom_low_descriptor_t* descriptor, int64_t value,
    loom_named_attr_t* out_attr) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(emitter->context);
  const loom_low_descriptor_layout_t* layout =
      loom_low_descriptor_layout(descriptor_set, descriptor);
  IREE_ASSERT_EQ(layout->immediate_count, 1u);
  const loom_low_immediate_t* immediate =
      &descriptor_set->immediates[layout->immediate_start];
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(
      loom_low_lower_context_builder(emitter->context),
      loom_low_descriptor_set_string(descriptor_set,
                                     immediate->field_name_string_ref),
      &name_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_i64(value),
  };
  return iree_ok_status();
}

static iree_status_t loom_aie2p_vector_load_emit_offset_add(
    loom_aie2p_vector_packet_emitter_t* emitter, loom_value_id_t source,
    int64_t immediate, loom_value_id_t* out_result) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(emitter->context);
  const loom_low_descriptor_t* descriptor =
      &descriptor_set
           ->descriptors[AIE2P_CORE_DESCRIPTOR_REF_ADD_NO_CARRY_I32_IMMEDIATE];
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_load_immediate_attr(
      emitter, descriptor, immediate, &attr));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      loom_low_lower_context_builder(emitter->context), descriptor_set,
      descriptor, /*access_flags=*/0, &source, 1,
      loom_make_named_attr_slice(&attr, 1), &emitter->scalar_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0,
      emitter->source_op->location, &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_aie2p_vector_load_emit_cursor(
    loom_aie2p_vector_packet_emitter_t* emitter, loom_value_id_t root,
    loom_value_id_t complete_byte_offset, uint32_t packet_byte_offset,
    loom_type_t pointer_type, loom_type_t modifier_type,
    loom_value_id_t* out_cursor) {
  loom_value_id_t packet_offset = complete_byte_offset;
  uint32_t remaining_offset = packet_byte_offset;
  while (remaining_offset != 0) {
    const int64_t chunk =
        iree_min(remaining_offset,
                 (uint32_t)LOOM_AIE2P_VECTOR_LOAD_OFFSET_CHUNK_MAXIMUM);
    IREE_RETURN_IF_ERROR(loom_aie2p_vector_load_emit_offset_add(
        emitter, packet_offset, chunk, &packet_offset));
    remaining_offset -= (uint32_t)chunk;
  }

  loom_value_id_t modifier = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_emit_descriptor_op(
      emitter, AIE2P_CORE_DESCRIPTOR_REF_MOVE_ER_TO_EM, &packet_offset, 1,
      loom_named_attr_slice_empty(), modifier_type,
      /*tied_results=*/NULL, /*tied_result_count=*/0, &modifier));
  loom_op_t* root_copy = NULL;
  IREE_RETURN_IF_ERROR(loom_low_copy_build(
      loom_low_lower_context_builder(emitter->context), root,
      /*detached=*/false, pointer_type, emitter->source_op->location,
      &root_copy));
  const loom_value_id_t operands[] = {loom_low_copy_result(root_copy),
                                      modifier};
  const loom_tied_result_t tied_result = {
      .result_index = 0,
      .operand_index = 0,
  };
  return loom_aie2p_vector_packet_emit_descriptor_op(
      emitter, AIE2P_CORE_DESCRIPTOR_REF_ADDRESS_ADD_A_REGISTER, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), pointer_type,
      &tied_result, 1, out_cursor);
}

static uint32_t loom_aie2p_vector_load_descriptor(uint8_t byte_count,
                                                  bool is_volatile) {
  switch (byte_count) {
    case 1:
      return is_volatile
                 ? AIE2P_CORE_DESCRIPTOR_REF_LOAD_SCALAR_U8_POSTINCREMENT_IMMEDIATE_VOLATILE
                 : AIE2P_CORE_DESCRIPTOR_REF_LOAD_SCALAR_U8_POSTINCREMENT_IMMEDIATE;
    case 2:
      return is_volatile
                 ? AIE2P_CORE_DESCRIPTOR_REF_LOAD_SCALAR_U16_POSTINCREMENT_IMMEDIATE_VOLATILE
                 : AIE2P_CORE_DESCRIPTOR_REF_LOAD_SCALAR_U16_POSTINCREMENT_IMMEDIATE;
    case 4:
      return is_volatile
                 ? AIE2P_CORE_DESCRIPTOR_REF_LOAD_SCALAR_I32_POSTINCREMENT_IMMEDIATE_VOLATILE
                 : AIE2P_CORE_DESCRIPTOR_REF_LOAD_SCALAR_I32_POSTINCREMENT_IMMEDIATE;
    default:
      IREE_ASSERT_UNREACHABLE("selected AIE2P scalar load width");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t loom_aie2p_vector_load_emit_word(
    loom_aie2p_vector_packet_emitter_t* emitter,
    const loom_low_source_memory_access_plan_t* access,
    const loom_low_lower_memory_origin_t* origin, uint16_t lane,
    uint8_t word_byte_offset, uint8_t word_byte_count, int64_t increment,
    loom_type_t pointer_type, loom_value_id_t* cursor,
    loom_value_id_t* out_word) {
  const bool is_volatile =
      iree_any_bit_set(access->access_flags, LOOM_MEMORY_ACCESS_FLAG_VOLATILE);
  const uint32_t descriptor_ordinal =
      loom_aie2p_vector_load_descriptor(word_byte_count, is_volatile);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(emitter->context);
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];
  loom_named_attr_t attr = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_load_immediate_attr(
      emitter, descriptor, increment, &attr));
  const loom_type_t result_types[] = {emitter->scalar_type, pointer_type};
  const loom_tied_result_t tied_result = {
      .result_index = 1,
      .operand_index = 0,
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_build_resolved_descriptor_op(
      loom_low_lower_context_builder(emitter->context), descriptor_set,
      descriptor, access->access_flags & LOOM_MEMORY_ACCESS_FLAG_VOLATILE,
      cursor, 1, loom_make_named_attr_slice(&attr, 1), result_types,
      IREE_ARRAYSIZE(result_types), &tied_result, 1,
      emitter->source_op->location, &low_op));
  const loom_value_slice_t results = loom_low_op_results(low_op);
  *out_word = results.values[0];
  *cursor = results.values[1];

  const loom_low_lower_memory_packet_geometry_t geometry = {
      .static_byte_offset = access->static_byte_offset +
                            (int64_t)lane * access->element_byte_count +
                            word_byte_offset,
      .element_byte_count = word_byte_count,
      .vector_lane_count = 1,
      .vector_lane_byte_stride = word_byte_count,
  };
  return loom_low_lower_record_memory_packet(
      emitter->context, low_op, descriptor, origin, access, &geometry,
      loom_value_facts_exact_i64(0));
}

static iree_status_t loom_aie2p_vector_load_emit_scalar(
    loom_aie2p_vector_packet_emitter_t* emitter,
    const loom_low_source_memory_access_plan_t* access,
    const loom_low_lower_memory_origin_t* origin,
    const loom_aie2p_vector_load_plan_t* plan, uint16_t lane, bool reverse,
    loom_type_t pointer_type, loom_type_t wide_scalar_type,
    loom_value_id_t* cursor, loom_value_id_t* out_scalar) {
  const uint8_t element_byte_count = (uint8_t)access->element_byte_count;
  if (plan->element_bit_count != 64) {
    const int64_t increment =
        reverse ? -(int64_t)element_byte_count : element_byte_count;
    return loom_aie2p_vector_load_emit_word(
        emitter, access, origin, lane, /*word_byte_offset=*/0,
        element_byte_count, increment, pointer_type, cursor, out_scalar);
  }

  loom_value_id_t words[2] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_load_emit_word(
      emitter, access, origin, lane, /*word_byte_offset=*/0,
      /*word_byte_count=*/4, /*increment=*/4, pointer_type, cursor, &words[0]));
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_load_emit_word(
      emitter, access, origin, lane, /*word_byte_offset=*/4,
      /*word_byte_count=*/4, reverse ? -12 : 4, pointer_type, cursor,
      &words[1]));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(loom_low_lower_context_builder(emitter->context),
                            words, IREE_ARRAYSIZE(words), wide_scalar_type,
                            emitter->source_op->location, &concat_op));
  *out_scalar = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t loom_aie2p_emit_vector_load_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan) {
  const loom_aie2p_vector_load_plan_t* load_plan = plan.target_data;
  const loom_low_source_memory_access_plan_t* access = load_plan->access;
  IREE_ASSERT_EQ(loom_amd_xdna_aie2p_core_lower_rule_set
                     .source_memory_byte_offset_materializer_count,
                 1u);

  loom_aie2p_vector_packet_emitter_t emitter;
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_emitter_initialize(
      context, source_op, &emitter));
  loom_type_t pointer_type = loom_type_none();
  loom_type_t modifier_type = loom_type_none();
  loom_type_t wide_scalar_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, AIE2P_CORE_REG_CLASS_ID_AIE2P_EP, 1, &pointer_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, AIE2P_CORE_REG_CLASS_ID_AIE2P_EM, 1, &modifier_type));
  if (load_plan->element_bit_count == 64) {
    IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
        context, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 2, &wide_scalar_type));
  }

  loom_value_id_t complete_byte_offset = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_materialize_source_memory_complete_byte_offset(
          context, &loom_amd_xdna_aie2p_core_lower_rule_set,
          &loom_amd_xdna_aie2p_core_lower_rule_set
               .source_memory_byte_offset_materializers[0],
          source_op, access, &complete_byte_offset));
  const loom_value_id_t root =
      loom_low_lower_lookup_value(context, access->root_value_id);
  const loom_low_lower_memory_origin_t* origin = load_plan->origin;
  loom_value_id_t packets[LOOM_AIE2P_VECTOR_PACKET_MAX_COUNT] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint8_t packet = 0; packet < load_plan->logical_packet_count; ++packet) {
    const uint16_t packet_lane_begin =
        (uint16_t)(packet * load_plan->packet_lane_count);
    const uint16_t packet_lane_count =
        (uint16_t)iree_min(load_plan->lane_count - packet_lane_begin,
                           load_plan->packet_lane_count);
    const bool reverse = packet_lane_count != load_plan->packet_lane_count;
    const uint16_t seed_lane =
        (uint16_t)(packet_lane_begin + (reverse ? packet_lane_count - 1 : 0));
    const uint32_t packet_byte_offset =
        (uint32_t)seed_lane * access->element_byte_count;
    loom_value_id_t cursor = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_aie2p_vector_load_emit_cursor(
        &emitter, root, complete_byte_offset, packet_byte_offset, pointer_type,
        modifier_type, &cursor));

    loom_value_id_t packet_value = LOOM_VALUE_ID_INVALID;
    for (uint16_t packet_lane = 0; packet_lane < packet_lane_count;
         ++packet_lane) {
      const uint16_t lane = reverse ? (uint16_t)(seed_lane - packet_lane)
                                    : (uint16_t)(seed_lane + packet_lane);
      loom_value_id_t scalar = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_aie2p_vector_load_emit_scalar(
          &emitter, access, origin, load_plan, lane, reverse, pointer_type,
          wide_scalar_type, &cursor, &scalar));
      if (packet_lane == 0) {
        IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_emit_scalar_splat(
            &emitter, load_plan->element_bit_count, scalar, &packet_value));
      } else {
        IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_emit_scalar_push(
            &emitter, load_plan->element_bit_count, /*push_high=*/!reverse,
            packet_value, scalar, &packet_value));
      }
    }
    packets[packet] = packet_value;
  }

  const loom_aie2p_vector_carrier_t carrier = {
      .kind = load_plan->carrier_kind,
      .unit_count = load_plan->carrier_unit_count,
  };
  return loom_aie2p_vector_packet_bind_vector_packets(
      &emitter, carrier, load_plan->logical_packet_count, packets,
      loom_vector_load_result(source_op));
}
