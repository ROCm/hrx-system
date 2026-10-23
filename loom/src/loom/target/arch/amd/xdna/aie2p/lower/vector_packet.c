// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/lower/vector_packet.h"

#include <limits.h>

#include "loom/ops/low/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"

typedef struct loom_aie2p_vector_packet_scalar_width_t {
  // Broadcast that seeds one X packet from a scalar payload.
  uint32_t splat_descriptor;
  // Push that appends a scalar at the high end of an X packet.
  uint32_t push_high_descriptor;
  // Push that prepends a scalar at the low end of an X packet.
  uint32_t push_low_descriptor;
  // Logical scalar elements represented by one X packet.
  uint8_t lane_count;
} loom_aie2p_vector_packet_scalar_width_t;

static const loom_aie2p_vector_packet_scalar_width_t kScalarWidths[] = {
    {AIE2P_CORE_DESCRIPTOR_REF_SPLAT_I8X64,
     AIE2P_CORE_DESCRIPTOR_REF_PUSH_HI_I8, AIE2P_CORE_DESCRIPTOR_REF_PUSH_LO_I8,
     64},
    {AIE2P_CORE_DESCRIPTOR_REF_SPLAT_I16X32,
     AIE2P_CORE_DESCRIPTOR_REF_PUSH_HI_I16,
     AIE2P_CORE_DESCRIPTOR_REF_PUSH_LO_I16, 32},
    {AIE2P_CORE_DESCRIPTOR_REF_SPLAT_I32X16,
     AIE2P_CORE_DESCRIPTOR_REF_PUSH_HI_I32,
     AIE2P_CORE_DESCRIPTOR_REF_PUSH_LO_I32, 16},
    {AIE2P_CORE_DESCRIPTOR_REF_SPLAT_I64X8,
     AIE2P_CORE_DESCRIPTOR_REF_PUSH_HI_I64,
     AIE2P_CORE_DESCRIPTOR_REF_PUSH_LO_I64, 8},
};

static const loom_aie2p_vector_packet_scalar_width_t*
loom_aie2p_vector_packet_scalar_width(uint16_t element_bit_count) {
  uint8_t width_index = 0;
  switch (element_bit_count) {
    case 1:
    case 8:
      width_index = 0;
      break;
    case 16:
      width_index = 1;
      break;
    case 32:
      width_index = 2;
      break;
    case 64:
      width_index = 3;
      break;
    default:
      return NULL;
  }
  return &kScalarWidths[width_index];
}

uint8_t loom_aie2p_vector_packet_carrier_unit_count(
    loom_aie2p_vector_carrier_kind_t carrier_kind) {
  return carrier_kind == LOOM_AIE2P_VECTOR_CARRIER_ORDINARY ? 2 : 1;
}

uint8_t loom_aie2p_vector_packet_scalar_lane_count(uint16_t element_bit_count) {
  const loom_aie2p_vector_packet_scalar_width_t* width =
      loom_aie2p_vector_packet_scalar_width(element_bit_count);
  return width != NULL ? width->lane_count : 0;
}

iree_status_t loom_aie2p_vector_packet_emit_scalar_splat(
    loom_aie2p_vector_packet_emitter_t* emitter, uint16_t element_bit_count,
    loom_value_id_t scalar, loom_value_id_t* out_packet) {
  const loom_aie2p_vector_packet_scalar_width_t* width =
      loom_aie2p_vector_packet_scalar_width(element_bit_count);
  IREE_ASSERT(width != NULL);
  return loom_aie2p_vector_packet_emit_descriptor_op(
      emitter, width->splat_descriptor, &scalar, 1,
      loom_named_attr_slice_empty(), emitter->vector_type,
      /*tied_results=*/NULL, /*tied_result_count=*/0, out_packet);
}

iree_status_t loom_aie2p_vector_packet_emit_scalar_push(
    loom_aie2p_vector_packet_emitter_t* emitter, uint16_t element_bit_count,
    bool push_high, loom_value_id_t packet, loom_value_id_t scalar,
    loom_value_id_t* out_packet) {
  const loom_aie2p_vector_packet_scalar_width_t* width =
      loom_aie2p_vector_packet_scalar_width(element_bit_count);
  IREE_ASSERT(width != NULL);
  const loom_value_id_t operands[] = {
      push_high ? packet : scalar,
      push_high ? scalar : packet,
  };
  return loom_aie2p_vector_packet_emit_descriptor_op(
      emitter,
      push_high ? width->push_high_descriptor : width->push_low_descriptor,
      operands, IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(),
      emitter->vector_type, /*tied_results=*/NULL,
      /*tied_result_count=*/0, out_packet);
}

iree_status_t loom_aie2p_vector_packet_make_carrier_type(
    loom_low_lower_context_t* context,
    loom_aie2p_vector_carrier_kind_t carrier_kind, uint32_t unit_count,
    loom_type_t* out_type) {
  uint16_t register_class = 0;
  switch (carrier_kind) {
    case LOOM_AIE2P_VECTOR_CARRIER_ORDINARY:
      register_class = AIE2P_CORE_REG_CLASS_ID_AIE2P_VEC256;
      break;
    case LOOM_AIE2P_VECTOR_CARRIER_PREDICATE:
      register_class = AIE2P_CORE_REG_CLASS_ID_AIE2P_ELPREDICATE;
      break;
    case LOOM_AIE2P_VECTOR_CARRIER_ACCUMULATOR:
      register_class = AIE2P_CORE_REG_CLASS_ID_AIE2P_MBMS;
      break;
    case LOOM_AIE2P_VECTOR_CARRIER_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("selected AIE2P vector carrier");
      break;
  }
  return loom_low_lower_make_register_type(context, register_class, unit_count,
                                           out_type);
}

iree_status_t loom_aie2p_vector_packet_emitter_initialize(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_aie2p_vector_packet_emitter_t* out_emitter) {
  *out_emitter = (loom_aie2p_vector_packet_emitter_t){
      .context = context,
      .source_op = source_op,
      .scalar_type = loom_type_none(),
      .vector_type = loom_type_none(),
      .predicate_type = loom_type_none(),
      .accumulator_type = loom_type_none(),
      .scalar_immediate_name = LOOM_STRING_ID_INVALID,
      .zero_bytes = LOOM_VALUE_ID_INVALID,
      .one_bytes = LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 1, &out_emitter->scalar_type));
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_make_carrier_type(
      context, LOOM_AIE2P_VECTOR_CARRIER_ORDINARY, 2,
      &out_emitter->vector_type));
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_make_carrier_type(
      context, LOOM_AIE2P_VECTOR_CARRIER_PREDICATE, 1,
      &out_emitter->predicate_type));
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_make_carrier_type(
      context, LOOM_AIE2P_VECTOR_CARRIER_ACCUMULATOR, 1,
      &out_emitter->accumulator_type));
  return loom_builder_intern_string(loom_low_lower_context_builder(context),
                                    IREE_SV("i"),
                                    &out_emitter->scalar_immediate_name);
}

iree_status_t loom_aie2p_vector_packet_emit_descriptor_op(
    loom_aie2p_vector_packet_emitter_t* emitter, uint32_t descriptor_ordinal,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, loom_type_t result_type,
    const loom_tied_result_t* tied_results, iree_host_size_t tied_result_count,
    loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor = &loom_low_lower_context_descriptor_set(emitter->context)
                         ->descriptors[descriptor_ordinal],
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      emitter->context, &descriptor, operands, operand_count, attrs,
      &result_type, 1, tied_results, tied_result_count,
      emitter->source_op->location, &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

iree_status_t loom_aie2p_vector_packet_emit_constant(
    loom_aie2p_vector_packet_emitter_t* emitter, uint32_t descriptor_ordinal,
    int64_t value, loom_type_t result_type, loom_value_id_t* out_result) {
  *out_result = LOOM_VALUE_ID_INVALID;
  const loom_named_attr_t immediate = {
      .name_id = emitter->scalar_immediate_name,
      .value = loom_attr_i64(value),
  };
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor = &loom_low_lower_context_descriptor_set(emitter->context)
                         ->descriptors[descriptor_ordinal],
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      emitter->context, &descriptor, loom_make_named_attr_slice(&immediate, 1),
      result_type, emitter->source_op->location, &low_op));
  *out_result = loom_low_const_result(low_op);
  return iree_ok_status();
}

static int64_t loom_aie2p_vector_packet_signed_i32_bits(uint32_t value) {
  return value <= INT32_MAX ? (int64_t)value
                            : (int64_t)value - (INT64_C(1) << 32);
}

iree_status_t loom_aie2p_vector_packet_emit_byte_selector(
    loom_aie2p_vector_packet_emitter_t* emitter, uint64_t mask,
    loom_value_id_t* out_selector) {
  const uint32_t low_word = (uint32_t)mask;
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_emit_constant(
      emitter, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_PREDICATE_LOW32,
      loom_aie2p_vector_packet_signed_i32_bits(low_word),
      emitter->predicate_type, out_selector));

  const uint32_t high_word = (uint32_t)(mask >> 32);
  uint32_t descriptor_ordinal =
      AIE2P_CORE_DESCRIPTOR_REF_PREDICATE_COMPLETE_ZERO_HIGH32;
  int64_t immediate_value = 0;
  if (high_word != 0) {
    descriptor_ordinal =
        AIE2P_CORE_DESCRIPTOR_REF_PREDICATE_COMPLETE_CONSTANT_HIGH32;
    immediate_value = loom_aie2p_vector_packet_signed_i32_bits(high_word);
  }
  const loom_named_attr_t immediate = {
      .name_id = emitter->scalar_immediate_name,
      .value = loom_attr_i64(immediate_value),
  };
  const loom_tied_result_t tied_result = {
      .result_index = 0,
      .operand_index = 0,
  };
  const loom_value_id_t low_selector = *out_selector;
  return loom_aie2p_vector_packet_emit_descriptor_op(
      emitter, descriptor_ordinal, &low_selector, 1,
      loom_make_named_attr_slice(&immediate, 1), emitter->predicate_type,
      &tied_result, 1, out_selector);
}

static iree_status_t loom_aie2p_vector_packet_ensure_boolean_bytes(
    loom_aie2p_vector_packet_emitter_t* emitter) {
  if (emitter->zero_bytes != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  loom_value_id_t one = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_emit_constant(
      emitter, AIE2P_CORE_DESCRIPTOR_REF_CONSTANT_I32_SHORT, 1,
      emitter->scalar_type, &one));
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_emit_descriptor_op(
      emitter, AIE2P_CORE_DESCRIPTOR_REF_SPLAT_I8X64, &one, 1,
      loom_named_attr_slice_empty(), emitter->vector_type,
      /*tied_results=*/NULL, /*tied_result_count=*/0, &emitter->one_bytes));
  const loom_value_id_t subtract_operands[] = {emitter->one_bytes,
                                               emitter->one_bytes};
  return loom_aie2p_vector_packet_emit_descriptor_op(
      emitter, AIE2P_CORE_DESCRIPTOR_REF_SUB_I8X64, subtract_operands,
      IREE_ARRAYSIZE(subtract_operands), loom_named_attr_slice_empty(),
      emitter->vector_type, /*tied_results=*/NULL, /*tied_result_count=*/0,
      &emitter->zero_bytes);
}

iree_status_t loom_aie2p_vector_packet_read_native(
    loom_aie2p_vector_packet_emitter_t* emitter, loom_value_id_t low_value,
    loom_aie2p_vector_carrier_t carrier, uint8_t packet_index,
    loom_value_id_t* out_packet) {
  const uint8_t units_per_packet =
      loom_aie2p_vector_packet_carrier_unit_count(carrier.kind);
  if (carrier.unit_count == units_per_packet) {
    *out_packet = low_value;
    return iree_ok_status();
  }

  loom_type_t packet_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_make_carrier_type(
      emitter->context, carrier.kind, units_per_packet, &packet_type));
  loom_op_t* slice_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_slice_build(
      loom_low_lower_context_builder(emitter->context), low_value,
      packet_index * units_per_packet, packet_type,
      emitter->source_op->location, &slice_op));
  *out_packet = loom_low_slice_result(slice_op);
  return iree_ok_status();
}

iree_status_t loom_aie2p_vector_packet_read_vector(
    loom_aie2p_vector_packet_emitter_t* emitter, loom_value_id_t low_value,
    loom_aie2p_vector_carrier_t carrier, uint8_t packet_index,
    loom_value_id_t* out_packet) {
  loom_value_id_t native_packet = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_read_native(
      emitter, low_value, carrier, packet_index, &native_packet));
  return loom_aie2p_vector_packet_native_to_vector(emitter, carrier.kind,
                                                   native_packet, out_packet);
}

iree_status_t loom_aie2p_vector_packet_native_to_vector(
    loom_aie2p_vector_packet_emitter_t* emitter,
    loom_aie2p_vector_carrier_kind_t carrier_kind,
    loom_value_id_t native_packet, loom_value_id_t* out_packet) {
  switch (carrier_kind) {
    case LOOM_AIE2P_VECTOR_CARRIER_ORDINARY:
      *out_packet = native_packet;
      return iree_ok_status();
    case LOOM_AIE2P_VECTOR_CARRIER_PREDICATE: {
      IREE_RETURN_IF_ERROR(
          loom_aie2p_vector_packet_ensure_boolean_bytes(emitter));
      const loom_value_id_t select_operands[] = {
          emitter->zero_bytes,
          emitter->one_bytes,
          native_packet,
      };
      return loom_aie2p_vector_packet_emit_descriptor_op(
          emitter, AIE2P_CORE_DESCRIPTOR_REF_SELECT_I8X64, select_operands,
          IREE_ARRAYSIZE(select_operands), loom_named_attr_slice_empty(),
          emitter->vector_type, /*tied_results=*/NULL,
          /*tied_result_count=*/0, out_packet);
    }
    case LOOM_AIE2P_VECTOR_CARRIER_ACCUMULATOR:
      return loom_aie2p_vector_packet_emit_descriptor_op(
          emitter, AIE2P_CORE_DESCRIPTOR_REF_MOVE_ACCUMULATOR512_TO_VECTOR512,
          &native_packet, 1, loom_named_attr_slice_empty(),
          emitter->vector_type, /*tied_results=*/NULL,
          /*tied_result_count=*/0, out_packet);
    case LOOM_AIE2P_VECTOR_CARRIER_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("selected AIE2P vector carrier");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_aie2p_vector_packet_write_native(
    loom_aie2p_vector_packet_emitter_t* emitter,
    loom_aie2p_vector_carrier_kind_t carrier_kind,
    loom_value_id_t vector_packet, loom_value_id_t* out_packet) {
  switch (carrier_kind) {
    case LOOM_AIE2P_VECTOR_CARRIER_ORDINARY:
      *out_packet = vector_packet;
      return iree_ok_status();
    case LOOM_AIE2P_VECTOR_CARRIER_PREDICATE: {
      IREE_RETURN_IF_ERROR(
          loom_aie2p_vector_packet_ensure_boolean_bytes(emitter));
      const loom_value_id_t compare_operands[] = {emitter->zero_bytes,
                                                  vector_packet};
      return loom_aie2p_vector_packet_emit_descriptor_op(
          emitter, AIE2P_CORE_DESCRIPTOR_REF_CMP_LT_UNSIGNED_I8X64,
          compare_operands, IREE_ARRAYSIZE(compare_operands),
          loom_named_attr_slice_empty(), emitter->predicate_type,
          /*tied_results=*/NULL, /*tied_result_count=*/0, out_packet);
    }
    case LOOM_AIE2P_VECTOR_CARRIER_ACCUMULATOR:
      return loom_aie2p_vector_packet_emit_descriptor_op(
          emitter, AIE2P_CORE_DESCRIPTOR_REF_MOVE_VECTOR512_TO_ACCUMULATOR512,
          &vector_packet, 1, loom_named_attr_slice_empty(),
          emitter->accumulator_type, /*tied_results=*/NULL,
          /*tied_result_count=*/0, out_packet);
    case LOOM_AIE2P_VECTOR_CARRIER_NONE:
    default:
      IREE_ASSERT_UNREACHABLE("selected AIE2P vector carrier");
      IREE_BUILTIN_UNREACHABLE();
  }
}

iree_status_t loom_aie2p_vector_packet_bind_native_packets(
    loom_aie2p_vector_packet_emitter_t* emitter,
    loom_aie2p_vector_carrier_t carrier, uint8_t logical_packet_count,
    loom_value_id_t* native_packets, loom_value_id_t result_value) {
  const uint8_t units_per_packet =
      loom_aie2p_vector_packet_carrier_unit_count(carrier.kind);
  const uint8_t physical_packet_count =
      (uint8_t)(carrier.unit_count / units_per_packet);
  for (uint8_t packet = logical_packet_count; packet < physical_packet_count;
       ++packet) {
    native_packets[packet] = native_packets[logical_packet_count - 1];
  }

  loom_value_id_t low_result = native_packets[0];
  if (physical_packet_count > 1) {
    loom_type_t result_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_make_carrier_type(
        emitter->context, carrier.kind, carrier.unit_count, &result_type));
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_concat_build(
        loom_low_lower_context_builder(emitter->context), native_packets,
        physical_packet_count, result_type, emitter->source_op->location,
        &concat_op));
    low_result = loom_low_concat_result(concat_op);
  }
  return loom_low_lower_bind_value(emitter->context, result_value, low_result);
}

iree_status_t loom_aie2p_vector_packet_bind_vector_packets(
    loom_aie2p_vector_packet_emitter_t* emitter,
    loom_aie2p_vector_carrier_t carrier, uint8_t logical_packet_count,
    const loom_value_id_t* vector_packets, loom_value_id_t result_value) {
  loom_value_id_t native_packets[LOOM_AIE2P_VECTOR_PACKET_MAX_COUNT] = {
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
      LOOM_VALUE_ID_INVALID,
  };
  for (uint8_t packet = 0; packet < logical_packet_count; ++packet) {
    IREE_RETURN_IF_ERROR(loom_aie2p_vector_packet_write_native(
        emitter, carrier.kind, vector_packets[packet],
        &native_packets[packet]));
  }
  return loom_aie2p_vector_packet_bind_native_packets(
      emitter, carrier, logical_packet_count, native_packets, result_value);
}
