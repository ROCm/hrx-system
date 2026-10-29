// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/bitstream_packet_legalization.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/arena.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ops/scalar/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/vector/packet_legalization.h"
#include "loom/util/fact_vector_partition.h"

#define LOOM_VECTOR_BITSTREAM_CONSTANT_LIMIT 32u

typedef struct loom_vector_bitstream_constant_t {
  // Canonical low bits of the scalar constant.
  uint64_t bits;
  // Scalar integer type of the constant.
  loom_type_t type;
  // Interned SSA value containing the constant.
  loom_value_id_t value;
} loom_vector_bitstream_constant_t;

typedef struct loom_vector_bitstream_builder_t {
  // Active target legalization invocation.
  loom_target_legalization_context_t* context;
  // Builder positioned immediately before the packed transform.
  loom_builder_t* builder;
  // Source location copied onto replacement operations.
  loom_location_id_t location;
  // Function-local scalar constants reused by the replacement sequence.
  loom_vector_bitstream_constant_t
      constants[LOOM_VECTOR_BITSTREAM_CONSTANT_LIMIT];
  // Number of populated entries in |constants|.
  uint8_t constant_count;
} loom_vector_bitstream_builder_t;

typedef struct loom_vector_bitstream_plan_t {
  // Source module owning every type and value in the plan.
  const loom_module_t* module;
  // Authored packed-transform source value.
  loom_value_id_t source;
  // Authored packed-transform result value.
  loom_value_id_t result;
  // Authored source vector type.
  loom_type_t source_type;
  // Authored result vector type.
  loom_type_t result_type;
  // Unpacked logical vector type.
  loom_type_t logical_type;
  // Packed storage vector type.
  loom_type_t storage_type;
  // Number of fields in the logical bitstream.
  uint32_t logical_lane_count;
  // Number of elements in the packed storage vector.
  uint32_t storage_lane_count;
  // Physical width of each authored logical element.
  uint32_t logical_element_bit_count;
  // Physical width of each authored storage element.
  uint32_t storage_element_bit_count;
  // Number of low logical bits retained in each field.
  uint32_t field_bit_count;
  // Power-of-two scalar width used by logical SWAR packets.
  uint32_t work_element_bit_count;
  // Power-of-two scalar width used by storage SWAR packets.
  uint32_t storage_work_element_bit_count;
  // Width of each scalar word in the shared packed stream.
  uint32_t word_bit_count;
  // Native scalar width used to bridge packets and stream words.
  uint32_t packet_bit_count;
  // Number of scalar words covering the complete packed stream.
  uint32_t word_count;
  // Admitted logical packet lane counts, indexed by lane count.
  uint16_t logical_packet_lane_mask;
  // Admitted storage packet lane counts, indexed by lane count.
  uint16_t storage_packet_lane_mask;
  // Retained source decomposition produced by earlier packet legalization.
  loom_value_fact_vector_partition_t source_partition;
  // Exact number of meaningful bits in the packed stream.
  uint64_t total_bit_count;
  // Whether adjacent scalar partition components may form one vector packet.
  bool group_scalar_partition_components;
  // True for logical-to-storage transforms.
  bool is_bitpack;
  // True when unpacked fields require signed extension.
  bool is_signed_unpack;
} loom_vector_bitstream_plan_t;

typedef struct loom_vector_bitstream_source_cursor_t {
  // Current component in the retained source partition.
  uint32_t component_index;
  // First unread lane within the current vector component.
  uint32_t component_lane_offset;
} loom_vector_bitstream_source_cursor_t;

static bool loom_vector_bitstream_scalar_type(uint32_t bit_count,
                                              loom_scalar_type_t* out_type) {
  switch (bit_count) {
    case 8:
      *out_type = LOOM_SCALAR_TYPE_I8;
      return true;
    case 16:
      *out_type = LOOM_SCALAR_TYPE_I16;
      return true;
    case 32:
      *out_type = LOOM_SCALAR_TYPE_I32;
      return true;
    case 64:
      *out_type = LOOM_SCALAR_TYPE_I64;
      return true;
    default:
      *out_type = LOOM_SCALAR_TYPE_NONE;
      return false;
  }
}

static loom_scalar_type_t loom_vector_bitstream_required_scalar_type(
    uint32_t bit_count) {
  loom_scalar_type_t type = LOOM_SCALAR_TYPE_NONE;
  const bool has_type = loom_vector_bitstream_scalar_type(bit_count, &type);
  IREE_ASSERT_TRUE(has_type);
  return type;
}

static uint64_t loom_vector_bitstream_low_mask(uint32_t bit_count) {
  return bit_count == 64 ? UINT64_MAX : ((uint64_t)1u << bit_count) - 1u;
}

static int64_t loom_vector_bitstream_constant_i64(uint64_t bits,
                                                  uint32_t bit_count) {
  const uint64_t mask = loom_vector_bitstream_low_mask(bit_count);
  bits &= mask;
  if (bit_count < 64 &&
      iree_any_bit_set(bits, (uint64_t)1u << (bit_count - 1u))) {
    bits |= ~mask;
  }
  return (int64_t)bits;
}

static loom_type_t loom_vector_bitstream_vector_type(loom_scalar_type_t element,
                                                     uint32_t lane_count) {
  return loom_type_shaped_1d(LOOM_TYPE_VECTOR, element,
                             loom_dim_pack_static(lane_count),
                             /*encoding_id=*/0);
}

static bool loom_vector_bitstream_static_lane_count(loom_type_t type,
                                                    uint32_t* out_lane_count) {
  *out_lane_count = 0;
  if (!loom_type_is_vector(type) || loom_type_rank(type) != 1 ||
      !loom_type_is_all_static(type)) {
    return false;
  }
  const int64_t lane_count = loom_type_dim_static_size_at(type, 0);
  if (lane_count < 1 || lane_count > UINT32_MAX) {
    return false;
  }
  *out_lane_count = (uint32_t)lane_count;
  return true;
}

static iree_status_t loom_vector_bitstream_constant(
    loom_vector_bitstream_builder_t* builder, loom_type_t type, uint64_t bits,
    loom_value_id_t* out_value) {
  const uint32_t bit_count =
      (uint32_t)loom_scalar_type_bitwidth(loom_type_element_type(type));
  const uint64_t canonical_bits =
      bits & loom_vector_bitstream_low_mask(bit_count);
  for (uint8_t i = 0; i < builder->constant_count; ++i) {
    const loom_vector_bitstream_constant_t* constant = &builder->constants[i];
    if (constant->bits == canonical_bits &&
        loom_type_equal(constant->type, type)) {
      *out_value = constant->value;
      return iree_ok_status();
    }
  }
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder->builder,
      loom_attr_i64(
          loom_vector_bitstream_constant_i64(canonical_bits, bit_count)),
      type, builder->location, &op));
  const loom_value_id_t value = loom_scalar_constant_result(op);
  if (builder->constant_count < LOOM_VECTOR_BITSTREAM_CONSTANT_LIMIT) {
    builder->constants[builder->constant_count++] =
        (loom_vector_bitstream_constant_t){
            .bits = canonical_bits,
            .type = type,
            .value = value,
        };
  }
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_andi(
    loom_vector_bitstream_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t type, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_andi_build(builder->builder, lhs, rhs, type,
                                              builder->location, &op));
  *out_value = loom_scalar_andi_result(op);
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_ori(
    loom_vector_bitstream_builder_t* builder, loom_value_id_t lhs,
    loom_value_id_t rhs, loom_type_t type, loom_value_id_t* out_value) {
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_ori_build(builder->builder, lhs, rhs, type,
                                             builder->location, &op));
  *out_value = loom_scalar_ori_result(op);
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_shift_left(
    loom_vector_bitstream_builder_t* builder, loom_value_id_t value,
    uint32_t shift, loom_type_t type, loom_value_id_t* out_value) {
  if (shift == 0) {
    *out_value = value;
    return iree_ok_status();
  }
  loom_value_id_t amount = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_bitstream_constant(builder, type, shift, &amount));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_scalar_shli_build(builder->builder, /*instance_flags=*/0, value,
                             amount, type, builder->location, &op));
  *out_value = loom_scalar_shli_result(op);
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_shift_right(
    loom_vector_bitstream_builder_t* builder, loom_value_id_t value,
    uint32_t shift, loom_type_t type, loom_value_id_t* out_value) {
  if (shift == 0) {
    *out_value = value;
    return iree_ok_status();
  }
  loom_value_id_t amount = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_bitstream_constant(builder, type, shift, &amount));
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_scalar_shrui_build(builder->builder, value, amount,
                                               type, builder->location, &op));
  *out_value = loom_scalar_shrui_result(op);
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_mask(
    loom_vector_bitstream_builder_t* builder, loom_value_id_t value,
    uint64_t mask, loom_type_t type, loom_value_id_t* out_value) {
  const uint32_t type_bit_count =
      (uint32_t)loom_scalar_type_bitwidth(loom_type_element_type(type));
  if (mask == loom_vector_bitstream_low_mask(type_bit_count)) {
    *out_value = value;
    return iree_ok_status();
  }
  loom_value_id_t mask_value = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_bitstream_constant(builder, type, mask, &mask_value));
  return loom_vector_bitstream_andi(builder, value, mask_value, type,
                                    out_value);
}

static iree_status_t loom_vector_bitstream_scalar_convert_unsigned(
    loom_vector_bitstream_builder_t* builder, loom_value_id_t value,
    loom_type_t source_type, loom_type_t result_type,
    loom_value_id_t* out_value) {
  const int32_t source_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(source_type));
  const int32_t result_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  if (source_bit_count == result_bit_count) {
    *out_value = value;
    return iree_ok_status();
  }
  loom_op_t* op = NULL;
  if (source_bit_count < result_bit_count) {
    IREE_RETURN_IF_ERROR(loom_scalar_extui_build(builder->builder, value,
                                                 source_type, result_type,
                                                 builder->location, &op));
    *out_value = loom_scalar_extui_result(op);
  } else {
    IREE_RETURN_IF_ERROR(loom_scalar_trunci_build(builder->builder, value,
                                                  source_type, result_type,
                                                  builder->location, &op));
    *out_value = loom_scalar_trunci_result(op);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_vector_convert_unsigned(
    loom_vector_bitstream_builder_t* builder, loom_value_id_t value,
    loom_type_t source_type, loom_type_t result_type,
    loom_value_id_t* out_value) {
  const int32_t source_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(source_type));
  const int32_t result_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  if (source_bit_count == result_bit_count) {
    *out_value = value;
    return iree_ok_status();
  }
  if (result_bit_count == 1) {
    loom_op_t* one_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_splati_build(
        builder->builder, 1, source_type, builder->location, &one_op));
    loom_op_t* compare_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_cmpi_build(
        builder->builder, LOOM_VECTOR_CMPI_PREDICATE_EQ, value,
        loom_vector_splati_result(one_op), source_type, result_type,
        builder->location, &compare_op));
    *out_value = loom_vector_cmpi_result(compare_op);
    return iree_ok_status();
  }
  loom_op_t* op = NULL;
  if (source_bit_count < result_bit_count) {
    IREE_RETURN_IF_ERROR(loom_vector_extui_build(builder->builder, value,
                                                 source_type, result_type,
                                                 builder->location, &op));
    *out_value = loom_vector_extui_result(op);
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_trunci_build(builder->builder, value,
                                                  source_type, result_type,
                                                  builder->location, &op));
    *out_value = loom_vector_trunci_result(op);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_vector_convert_signed(
    loom_vector_bitstream_builder_t* builder, loom_value_id_t value,
    loom_type_t source_type, loom_type_t result_type,
    loom_value_id_t* out_value) {
  const int32_t source_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(source_type));
  const int32_t result_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(result_type));
  if (result_bit_count == 1) {
    return loom_vector_bitstream_vector_convert_unsigned(
        builder, value, source_type, result_type, out_value);
  }
  if (source_bit_count == result_bit_count) {
    *out_value = value;
    return iree_ok_status();
  }
  loom_op_t* op = NULL;
  if (source_bit_count < result_bit_count) {
    IREE_RETURN_IF_ERROR(loom_vector_extsi_build(builder->builder, value,
                                                 source_type, result_type,
                                                 builder->location, &op));
    *out_value = loom_vector_extsi_result(op);
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_trunci_build(builder->builder, value,
                                                  source_type, result_type,
                                                  builder->location, &op));
    *out_value = loom_vector_trunci_result(op);
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_slice(
    loom_vector_bitstream_builder_t* builder, loom_value_id_t source,
    loom_type_t source_type, uint32_t lane_offset, uint32_t lane_count,
    loom_value_id_t* out_value, loom_type_t* out_type) {
  *out_type = loom_vector_bitstream_vector_type(
      loom_type_element_type(source_type), lane_count);
  const int64_t static_offset = lane_offset;
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_slice_build(
      builder->builder, source, /*offsets=*/NULL, /*offsets_count=*/0,
      &static_offset, /*static_offsets_count=*/1, *out_type, builder->location,
      &op));
  *out_value = loom_vector_slice_result(op);
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_pad_packet(
    loom_vector_bitstream_builder_t* builder, loom_value_id_t packet,
    loom_type_t packet_type, uint32_t carrier_lane_count,
    loom_value_id_t* out_packet, loom_type_t* out_type) {
  const int64_t packet_lane_count_i64 =
      loom_type_dim_static_size_at(packet_type, 0);
  IREE_ASSERT_GE(packet_lane_count_i64, 1);
  IREE_ASSERT_LE(packet_lane_count_i64, UINT32_MAX);
  const uint32_t packet_lane_count = (uint32_t)packet_lane_count_i64;
  IREE_ASSERT_LE(packet_lane_count, carrier_lane_count);
  if (packet_lane_count == carrier_lane_count) {
    *out_packet = packet;
    *out_type = packet_type;
    return iree_ok_status();
  }

  const loom_scalar_type_t element_type = loom_type_element_type(packet_type);
  *out_type =
      loom_vector_bitstream_vector_type(element_type, carrier_lane_count);
  loom_op_t* zero_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splati_build(builder->builder, 0, *out_type,
                                                builder->location, &zero_op));
  loom_value_id_t padded_packet = loom_vector_splati_result(zero_op);
  for (uint32_t i = 0; i < packet_lane_count; ++i) {
    const int64_t static_index = i;
    loom_op_t* extract_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_extract_build(
        builder->builder, packet, /*indices=*/NULL, /*indices_count=*/0,
        &static_index, /*static_indices_count=*/1,
        loom_type_scalar(element_type), builder->location, &extract_op));
    loom_op_t* insert_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_insert_build(
        builder->builder, loom_vector_extract_result(extract_op), padded_packet,
        /*indices=*/NULL, /*indices_count=*/0, &static_index,
        /*static_indices_count=*/1, *out_type, builder->location, &insert_op));
    padded_packet = loom_vector_insert_result(insert_op);
  }
  *out_packet = padded_packet;
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_extract_lane_zero(
    loom_vector_bitstream_builder_t* builder, loom_value_id_t source,
    loom_type_t source_type, loom_value_id_t* out_value) {
  const int64_t static_index = 0;
  loom_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      builder->builder, source, /*indices=*/NULL, /*indices_count=*/0,
      &static_index, /*static_indices_count=*/1,
      loom_type_scalar(loom_type_element_type(source_type)), builder->location,
      &op));
  *out_value = loom_vector_extract_result(op);
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_packet_to_scalar(
    loom_vector_bitstream_builder_t* builder, loom_value_id_t packet,
    loom_type_t packet_type, uint32_t packet_lane_count,
    uint32_t work_element_bit_count, uint32_t packet_bit_count,
    loom_value_id_t* out_scalar, loom_type_t* out_scalar_type) {
  if (packet_lane_count == 1u) {
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_extract_lane_zero(
        builder, packet, packet_type, out_scalar));
    *out_scalar_type = loom_type_scalar(loom_type_element_type(packet_type));
    return iree_ok_status();
  }

  const loom_scalar_type_t packet_element =
      loom_vector_bitstream_required_scalar_type(packet_bit_count);
  loom_value_id_t padded_packet = LOOM_VALUE_ID_INVALID;
  loom_type_t padded_packet_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_vector_bitstream_pad_packet(
      builder, packet, packet_type, packet_bit_count / work_element_bit_count,
      &padded_packet, &padded_packet_type));

  const loom_type_t carrier_type =
      loom_vector_bitstream_vector_type(packet_element, 1u);
  if (!loom_type_equal(padded_packet_type, carrier_type)) {
    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
        builder->builder, padded_packet, padded_packet_type, carrier_type,
        builder->location, &bitcast_op));
    padded_packet = loom_vector_bitcast_result(bitcast_op);
  }
  IREE_RETURN_IF_ERROR(loom_vector_bitstream_extract_lane_zero(
      builder, padded_packet, carrier_type, out_scalar));
  *out_scalar_type = loom_type_scalar(packet_element);
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_scalar_to_packet(
    loom_vector_bitstream_builder_t* builder, loom_value_id_t scalar,
    loom_type_t scalar_type, loom_scalar_type_t work_element,
    uint32_t packet_lane_count, uint32_t work_element_bit_count,
    uint32_t packet_bit_count, loom_value_id_t* out_packet,
    loom_type_t* out_packet_type) {
  if (packet_lane_count == 1u) {
    const loom_type_t work_scalar_type = loom_type_scalar(work_element);
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_scalar_convert_unsigned(
        builder, scalar, scalar_type, work_scalar_type, &scalar));
    *out_packet_type = loom_vector_bitstream_vector_type(work_element, 1u);
    loom_op_t* splat_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_splat_build(builder->builder, scalar,
                                                 *out_packet_type,
                                                 builder->location, &splat_op));
    *out_packet = loom_vector_splat_result(splat_op);
    return iree_ok_status();
  }

  const loom_scalar_type_t packet_element =
      loom_vector_bitstream_required_scalar_type(packet_bit_count);
  const loom_type_t packet_scalar_type = loom_type_scalar(packet_element);
  IREE_RETURN_IF_ERROR(loom_vector_bitstream_scalar_convert_unsigned(
      builder, scalar, scalar_type, packet_scalar_type, &scalar));
  const loom_type_t carrier_type =
      loom_vector_bitstream_vector_type(packet_element, 1u);
  loom_op_t* splat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_splat_build(
      builder->builder, scalar, carrier_type, builder->location, &splat_op));
  loom_value_id_t carrier = loom_vector_splat_result(splat_op);

  const uint32_t carrier_lane_count = packet_bit_count / work_element_bit_count;
  const loom_type_t padded_packet_type =
      loom_vector_bitstream_vector_type(work_element, carrier_lane_count);
  loom_value_id_t padded_packet = carrier;
  if (!loom_type_equal(carrier_type, padded_packet_type)) {
    loom_op_t* bitcast_op = NULL;
    IREE_RETURN_IF_ERROR(loom_vector_bitcast_build(
        builder->builder, carrier, carrier_type, padded_packet_type,
        builder->location, &bitcast_op));
    padded_packet = loom_vector_bitcast_result(bitcast_op);
  }

  *out_packet_type =
      loom_vector_bitstream_vector_type(work_element, packet_lane_count);
  if (packet_lane_count == carrier_lane_count) {
    *out_packet = padded_packet;
    return iree_ok_status();
  }
  return loom_vector_bitstream_slice(builder, padded_packet, padded_packet_type,
                                     /*lane_offset=*/0, packet_lane_count,
                                     out_packet, out_packet_type);
}

static uint16_t loom_vector_bitstream_packet_lane_mask(
    const loom_target_legalization_context_t* context,
    loom_scalar_type_t source_element, loom_scalar_type_t work_element,
    uint32_t work_element_bit_count, uint32_t packet_bit_count) {
  static const uint8_t kCandidateLaneCounts[] = {8, 4, 2, 1};
  uint16_t lane_mask = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kCandidateLaneCounts); ++i) {
    const uint32_t lane_count = kCandidateLaneCounts[i];
    if (lane_count != 1u &&
        lane_count * work_element_bit_count > packet_bit_count) {
      continue;
    }
    const loom_type_t source_packet_type =
        loom_vector_bitstream_vector_type(source_element, lane_count);
    const loom_type_t work_packet_type =
        loom_vector_bitstream_vector_type(work_element, lane_count);
    bool has_bridge = lane_count == 1u;
    if (!has_bridge) {
      const loom_type_t padded_work_packet_type =
          loom_vector_bitstream_vector_type(
              work_element, packet_bit_count / work_element_bit_count);
      const loom_scalar_type_t packet_element =
          loom_vector_bitstream_required_scalar_type(packet_bit_count);
      const loom_type_t carrier_type =
          loom_vector_bitstream_vector_type(packet_element, 1u);
      has_bridge = loom_target_legalization_has_bitstream_packet_carrier(
                       context, padded_work_packet_type) &&
                   loom_target_legalization_has_bitstream_packet_carrier(
                       context, carrier_type);
    }
    if (has_bridge &&
        loom_target_legalization_has_bitstream_packet_carrier(
            context, source_packet_type) &&
        loom_target_legalization_has_bitstream_packet_carrier(
            context, work_packet_type)) {
      lane_mask |= (uint16_t)(1u << lane_count);
    }
  }
  return lane_mask;
}

static uint32_t loom_vector_bitstream_select_lane_count(
    uint16_t packet_lane_mask, uint32_t remaining_lane_count) {
  static const uint8_t kCandidateLaneCounts[] = {8, 4, 2, 1};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kCandidateLaneCounts); ++i) {
    const uint32_t lane_count = kCandidateLaneCounts[i];
    if (lane_count <= remaining_lane_count &&
        iree_any_bit_set(packet_lane_mask, (uint16_t)(1u << lane_count))) {
      return lane_count;
    }
  }
  IREE_ASSERT_UNREACHABLE("bitstream plan has no terminal packet lane");
  return 1;
}

static uint32_t loom_vector_bitstream_packet_scalar_bit_count(
    const loom_vector_bitstream_plan_t* plan, uint32_t packet_lane_count,
    uint32_t work_element_bit_count) {
  if (packet_lane_count == 1u &&
      work_element_bit_count > plan->packet_bit_count) {
    return work_element_bit_count;
  }
  return plan->packet_bit_count;
}

static bool loom_vector_bitstream_source_partition_is_usable(
    const loom_target_legalization_context_t* context,
    loom_value_fact_vector_partition_t partition, loom_type_t source_type,
    uint32_t source_lane_count) {
  if (partition.count == 0 || partition.lane_count != source_lane_count) {
    return false;
  }
  const loom_scalar_type_t source_element = loom_type_element_type(source_type);
  uint64_t lane_count = 0;
  for (uint32_t i = 0; i < partition.count; ++i) {
    const loom_value_fact_vector_partition_component_t component =
        partition.components[i];
    if (component.lane_count == 0) {
      return false;
    }
    const loom_type_t component_type =
        loom_module_value_type(context->module, component.value);
    if (loom_type_is_scalar(component_type)) {
      if (component.lane_count != 1u ||
          loom_type_element_type(component_type) != source_element) {
        return false;
      }
    } else {
      uint32_t component_lane_count = 0;
      if (!loom_vector_bitstream_static_lane_count(component_type,
                                                   &component_lane_count) ||
          component_lane_count != component.lane_count ||
          loom_type_element_type(component_type) != source_element ||
          !loom_target_legalization_has_bitstream_packet_carrier(
              context, component_type)) {
        return false;
      }
    }
    lane_count += component.lane_count;
  }
  return lane_count == source_lane_count;
}

static loom_value_fact_vector_partition_t
loom_vector_bitstream_source_partition_select(
    const loom_target_legalization_context_t* context,
    loom_value_fact_vector_partition_t partition, loom_type_t source_type,
    uint32_t source_lane_count) {
  if (partition.snapshot_count != 0) {
    loom_value_fact_vector_partition_t snapshot = partition;
    snapshot.components = partition.snapshot_components;
    snapshot.count = partition.snapshot_count;
    snapshot.regular_lane_count = partition.snapshot_lane_count;
    if (loom_vector_bitstream_source_partition_is_usable(
            context, snapshot, source_type, source_lane_count)) {
      return snapshot;
    }
  }
  if (loom_vector_bitstream_source_partition_is_usable(
          context, partition, source_type, source_lane_count)) {
    return partition;
  }
  return (loom_value_fact_vector_partition_t){0};
}

static uint32_t loom_vector_bitstream_source_component_remaining_lanes(
    const loom_vector_bitstream_plan_t* plan,
    const loom_vector_bitstream_source_cursor_t* cursor) {
  if (plan->source_partition.count == 0) {
    return UINT32_MAX;
  }
  IREE_ASSERT_LT(cursor->component_index, plan->source_partition.count);
  const uint32_t component_lane_count =
      plan->source_partition.components[cursor->component_index].lane_count;
  IREE_ASSERT_LT(cursor->component_lane_offset, component_lane_count);
  const loom_type_t component_type = loom_module_value_type(
      plan->module,
      plan->source_partition.components[cursor->component_index].value);
  if (loom_type_is_scalar(component_type) &&
      plan->group_scalar_partition_components) {
    IREE_ASSERT_EQ(cursor->component_lane_offset, 0u);
    uint32_t scalar_count = 0;
    for (uint32_t i = cursor->component_index; i < plan->source_partition.count;
         ++i) {
      const loom_value_fact_vector_partition_component_t component =
          plan->source_partition.components[i];
      const loom_type_t type =
          loom_module_value_type(plan->module, component.value);
      if (!loom_type_is_scalar(type)) {
        break;
      }
      IREE_ASSERT_EQ(component.lane_count, 1u);
      ++scalar_count;
    }
    return scalar_count;
  }
  return component_lane_count - cursor->component_lane_offset;
}

static void loom_vector_bitstream_advance_source_cursor(
    const loom_vector_bitstream_plan_t* plan, uint32_t packet_lane_count,
    loom_vector_bitstream_source_cursor_t* cursor) {
  if (plan->source_partition.count == 0) {
    return;
  }
  const loom_type_t component_type = loom_module_value_type(
      plan->module,
      plan->source_partition.components[cursor->component_index].value);
  if (loom_type_is_scalar(component_type)) {
    IREE_ASSERT_EQ(cursor->component_lane_offset, 0u);
    IREE_ASSERT_LE(cursor->component_index + packet_lane_count,
                   plan->source_partition.count);
    cursor->component_index += packet_lane_count;
    return;
  }
  const uint32_t remaining_lane_count =
      loom_vector_bitstream_source_component_remaining_lanes(plan, cursor);
  IREE_ASSERT_LE(packet_lane_count, remaining_lane_count);
  cursor->component_lane_offset += packet_lane_count;
  if (cursor->component_lane_offset ==
      plan->source_partition.components[cursor->component_index].lane_count) {
    ++cursor->component_index;
    cursor->component_lane_offset = 0;
  }
}

static iree_status_t loom_vector_bitstream_source_packet(
    loom_vector_bitstream_builder_t* builder,
    const loom_vector_bitstream_plan_t* plan, loom_type_t source_type,
    uint32_t lane_offset, uint32_t packet_lane_count,
    loom_vector_bitstream_source_cursor_t* source_cursor,
    loom_value_id_t* out_packet, loom_type_t* out_packet_type) {
  if (plan->source_partition.count == 0) {
    return loom_vector_bitstream_slice(builder, plan->source, source_type,
                                       lane_offset, packet_lane_count,
                                       out_packet, out_packet_type);
  }

  const loom_value_fact_vector_partition_component_t component =
      plan->source_partition.components[source_cursor->component_index];
  const uint32_t component_lane_offset = source_cursor->component_lane_offset;
  const loom_type_t component_type =
      loom_module_value_type(builder->context->module, component.value);
  *out_packet_type = loom_vector_bitstream_vector_type(
      loom_type_element_type(source_type), packet_lane_count);
  if (loom_type_is_vector(component_type)) {
    IREE_ASSERT_LE(packet_lane_count,
                   loom_vector_bitstream_source_component_remaining_lanes(
                       plan, source_cursor));
    if (component_lane_offset == 0 &&
        component.lane_count == packet_lane_count) {
      IREE_ASSERT_TRUE(loom_type_equal(component_type, *out_packet_type));
      *out_packet = component.value;
    } else {
      IREE_RETURN_IF_ERROR(loom_vector_bitstream_slice(
          builder, component.value, component_type, component_lane_offset,
          packet_lane_count, out_packet, out_packet_type));
    }
    loom_vector_bitstream_advance_source_cursor(plan, packet_lane_count,
                                                source_cursor);
    return iree_ok_status();
  }

  IREE_ASSERT_EQ(component_lane_offset, 0u);
  loom_value_id_t elements[8];
  IREE_ASSERT_LE(packet_lane_count, IREE_ARRAYSIZE(elements));
  for (uint32_t i = 0; i < packet_lane_count; ++i) {
    const loom_value_fact_vector_partition_component_t scalar_component =
        plan->source_partition.components[source_cursor->component_index + i];
    IREE_ASSERT_EQ(scalar_component.lane_count, 1u);
    IREE_ASSERT_TRUE(loom_type_is_scalar(loom_module_value_type(
        builder->context->module, scalar_component.value)));
    elements[i] = scalar_component.value;
  }
  loom_op_t* from_elements_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_from_elements_build(
      builder->builder, elements, packet_lane_count, *out_packet_type,
      builder->location, &from_elements_op));
  *out_packet = loom_vector_from_elements_result(from_elements_op);
  loom_vector_bitstream_advance_source_cursor(plan, packet_lane_count,
                                              source_cursor);
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_source_packet_scalar(
    loom_vector_bitstream_builder_t* builder,
    const loom_vector_bitstream_plan_t* plan, loom_type_t source_type,
    uint32_t lane_offset, uint32_t packet_lane_count,
    loom_scalar_type_t work_element, uint32_t work_element_bit_count,
    loom_vector_bitstream_source_cursor_t* source_cursor,
    loom_value_id_t* out_scalar, loom_type_t* out_scalar_type) {
  if (plan->source_partition.count != 0 && packet_lane_count == 1u) {
    const loom_value_fact_vector_partition_component_t component =
        plan->source_partition.components[source_cursor->component_index];
    const loom_type_t component_type =
        loom_module_value_type(builder->context->module, component.value);
    if (loom_type_is_scalar(component_type)) {
      *out_scalar_type = loom_type_scalar(work_element);
      IREE_RETURN_IF_ERROR(loom_vector_bitstream_scalar_convert_unsigned(
          builder, component.value, component_type, *out_scalar_type,
          out_scalar));
      loom_vector_bitstream_advance_source_cursor(plan, packet_lane_count,
                                                  source_cursor);
      return iree_ok_status();
    }
  }

  loom_value_id_t packet = LOOM_VALUE_ID_INVALID;
  loom_type_t packet_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_vector_bitstream_source_packet(
      builder, plan, source_type, lane_offset, packet_lane_count, source_cursor,
      &packet, &packet_type));
  const loom_type_t work_packet_type =
      loom_vector_bitstream_vector_type(work_element, packet_lane_count);
  IREE_RETURN_IF_ERROR(loom_vector_bitstream_vector_convert_unsigned(
      builder, packet, packet_type, work_packet_type, &packet));
  return loom_vector_bitstream_packet_to_scalar(
      builder, packet, work_packet_type, packet_lane_count,
      work_element_bit_count, plan->packet_bit_count, out_scalar,
      out_scalar_type);
}

static iree_status_t loom_vector_bitstream_compress(
    loom_vector_bitstream_builder_t* builder, loom_value_id_t value,
    uint32_t lane_count, uint32_t element_bit_count, uint32_t field_bit_count,
    loom_type_t type, loom_value_id_t* out_value) {
  uint64_t lane_mask = 0;
  const uint64_t field_mask = loom_vector_bitstream_low_mask(field_bit_count);
  for (uint32_t lane = 0; lane < lane_count; ++lane) {
    lane_mask |= field_mask << (lane * element_bit_count);
  }
  IREE_RETURN_IF_ERROR(
      loom_vector_bitstream_mask(builder, value, lane_mask, type, &value));
  if (element_bit_count == field_bit_count || lane_count == 1) {
    *out_value = value;
    return iree_ok_status();
  }

  for (uint32_t group_lane_count = 1; group_lane_count < lane_count;
       group_lane_count *= 2u) {
    const uint32_t shift =
        group_lane_count * (element_bit_count - field_bit_count);
    loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_shift_right(
        builder, value, shift, type, &shifted));
    IREE_RETURN_IF_ERROR(
        loom_vector_bitstream_ori(builder, value, shifted, type, &value));

    const uint32_t merged_lane_count = group_lane_count * 2u;
    uint64_t stage_mask = 0;
    const uint64_t merged_mask =
        loom_vector_bitstream_low_mask(merged_lane_count * field_bit_count);
    for (uint32_t lane = 0; lane < lane_count; lane += merged_lane_count) {
      stage_mask |= merged_mask << (lane * element_bit_count);
    }
    IREE_RETURN_IF_ERROR(
        loom_vector_bitstream_mask(builder, value, stage_mask, type, &value));
  }
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_expand(
    loom_vector_bitstream_builder_t* builder, loom_value_id_t value,
    uint32_t lane_count, uint32_t element_bit_count, uint32_t field_bit_count,
    loom_type_t type, loom_value_id_t* out_value) {
  if (element_bit_count == field_bit_count || lane_count == 1) {
    *out_value = value;
    return iree_ok_status();
  }
  for (uint32_t group_lane_count = lane_count / 2u; group_lane_count != 0;
       group_lane_count /= 2u) {
    const uint32_t shift =
        group_lane_count * (element_bit_count - field_bit_count);
    loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_shift_left(builder, value, shift,
                                                          type, &shifted));
    IREE_RETURN_IF_ERROR(
        loom_vector_bitstream_ori(builder, value, shifted, type, &value));

    const uint32_t merged_lane_count = group_lane_count * 2u;
    const uint64_t group_mask =
        loom_vector_bitstream_low_mask(group_lane_count * field_bit_count);
    uint64_t stage_mask = 0;
    for (uint32_t lane = 0; lane < lane_count; lane += merged_lane_count) {
      stage_mask |= group_mask << (lane * element_bit_count);
      stage_mask |= group_mask
                    << ((lane + group_lane_count) * element_bit_count);
    }
    IREE_RETURN_IF_ERROR(
        loom_vector_bitstream_mask(builder, value, stage_mask, type, &value));
  }
  *out_value = value;
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_sign_extend_packed(
    loom_vector_bitstream_builder_t* builder, loom_value_id_t value,
    uint32_t lane_count, uint32_t element_bit_count, uint32_t field_bit_count,
    loom_type_t type, loom_value_id_t* out_value) {
  if (field_bit_count == element_bit_count) {
    *out_value = value;
    return iree_ok_status();
  }

  uint64_t sign_mask = 0;
  uint64_t extension_mask = 0;
  const uint64_t lane_extension_mask =
      loom_vector_bitstream_low_mask(element_bit_count - field_bit_count)
      << field_bit_count;
  for (uint32_t lane = 0; lane < lane_count; ++lane) {
    const uint32_t lane_bit_offset = lane * element_bit_count;
    sign_mask |= (uint64_t)1u << (lane_bit_offset + field_bit_count - 1u);
    extension_mask |= lane_extension_mask << lane_bit_offset;
  }

  loom_value_id_t sign_bits = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_vector_bitstream_mask(builder, value, sign_mask, type, &sign_bits));
  for (uint32_t shift = 1u; shift < element_bit_count; shift *= 2u) {
    loom_value_id_t shifted = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_shift_left(
        builder, sign_bits, shift, type, &shifted));
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_ori(builder, sign_bits, shifted,
                                                   type, &sign_bits));
  }
  IREE_RETURN_IF_ERROR(loom_vector_bitstream_mask(
      builder, sign_bits, extension_mask, type, &sign_bits));
  return loom_vector_bitstream_ori(builder, value, sign_bits, type, out_value);
}

static iree_status_t loom_vector_bitstream_plan(
    loom_target_legalization_context_t* context, loom_op_t* op,
    uint32_t native_word_bit_count, loom_vector_bitstream_plan_t* out_plan) {
  *out_plan = (loom_vector_bitstream_plan_t){0};
  IREE_ASSERT(native_word_bit_count == 32u || native_word_bit_count == 64u);
  const bool is_bitpack = loom_vector_bitpack_isa(op);
  const bool is_unsigned_unpack = loom_vector_bitunpacku_isa(op);
  const bool is_signed_unpack = loom_vector_bitunpacks_isa(op);
  if (!is_bitpack && !is_unsigned_unpack && !is_signed_unpack) {
    return iree_ok_status();
  }

  const loom_value_id_t source =
      is_bitpack ? loom_vector_bitpack_source(op)
                 : (is_unsigned_unpack ? loom_vector_bitunpacku_source(op)
                                       : loom_vector_bitunpacks_source(op));
  const loom_value_id_t result =
      is_bitpack ? loom_vector_bitpack_result(op)
                 : (is_unsigned_unpack ? loom_vector_bitunpacku_result(op)
                                       : loom_vector_bitunpacks_result(op));
  const int64_t field_bit_count =
      is_bitpack ? loom_vector_bitpack_width(op)
                 : (is_unsigned_unpack ? loom_vector_bitunpacku_width(op)
                                       : loom_vector_bitunpacks_width(op));
  if (field_bit_count <= 0 || field_bit_count > 64) {
    return iree_ok_status();
  }

  const loom_type_t source_type =
      loom_module_value_type(context->module, source);
  const loom_type_t result_type =
      loom_module_value_type(context->module, result);
  const loom_type_t logical_type = is_bitpack ? source_type : result_type;
  const loom_type_t storage_type = is_bitpack ? result_type : source_type;
  uint32_t logical_lane_count = 0;
  uint32_t storage_lane_count = 0;
  if (!loom_vector_bitstream_static_lane_count(logical_type,
                                               &logical_lane_count) ||
      !loom_vector_bitstream_static_lane_count(storage_type,
                                               &storage_lane_count)) {
    return iree_ok_status();
  }
  const int32_t logical_element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(logical_type));
  const int32_t storage_element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(storage_type));
  if (logical_element_bit_count <= 0 || storage_element_bit_count <= 0 ||
      field_bit_count > logical_element_bit_count) {
    return iree_ok_status();
  }
  const uint64_t total_bit_count =
      (uint64_t)logical_lane_count * (uint32_t)field_bit_count;
  if (total_bit_count == 0 ||
      total_bit_count !=
          (uint64_t)storage_lane_count * (uint32_t)storage_element_bit_count) {
    return iree_ok_status();
  }

  uint32_t work_element_bit_count = 8;
  while (work_element_bit_count < (uint32_t)field_bit_count) {
    work_element_bit_count *= 2u;
  }
  uint32_t storage_work_element_bit_count = 8;
  while (storage_work_element_bit_count < (uint32_t)storage_element_bit_count) {
    storage_work_element_bit_count *= 2u;
  }
  uint32_t word_bit_count = native_word_bit_count;
  const uint32_t minimum_word_bit_count =
      iree_max((uint32_t)field_bit_count, (uint32_t)storage_element_bit_count);
  while (word_bit_count < minimum_word_bit_count && word_bit_count < 64) {
    word_bit_count *= 2u;
  }
  if (word_bit_count < minimum_word_bit_count) {
    return iree_ok_status();
  }
  const uint64_t word_count = (total_bit_count - 1u) / word_bit_count + 1u;
  if (word_count > UINT32_MAX) {
    return iree_ok_status();
  }

  loom_scalar_type_t work_element = LOOM_SCALAR_TYPE_NONE;
  if (!loom_vector_bitstream_scalar_type(work_element_bit_count,
                                         &work_element)) {
    return iree_ok_status();
  }
  loom_scalar_type_t storage_work_element = LOOM_SCALAR_TYPE_NONE;
  if (!loom_vector_bitstream_scalar_type(storage_work_element_bit_count,
                                         &storage_work_element)) {
    return iree_ok_status();
  }
  const loom_scalar_type_t storage_element =
      loom_type_element_type(storage_type);
  const uint32_t packet_bit_count = native_word_bit_count;
  const uint16_t logical_packet_lane_mask =
      loom_vector_bitstream_packet_lane_mask(
          context, loom_type_element_type(logical_type), work_element,
          work_element_bit_count, packet_bit_count);
  const uint16_t storage_packet_lane_mask =
      loom_vector_bitstream_packet_lane_mask(
          context, storage_element, storage_work_element,
          storage_work_element_bit_count, packet_bit_count);
  if (!iree_all_bits_set(logical_packet_lane_mask, (uint16_t)(1u << 1)) ||
      !iree_all_bits_set(storage_packet_lane_mask, (uint16_t)(1u << 1))) {
    return iree_ok_status();
  }
  const uint32_t source_lane_count =
      is_bitpack ? logical_lane_count : storage_lane_count;
  const loom_value_fact_vector_partition_t source_partition =
      loom_vector_bitstream_source_partition_select(
          context,
          loom_value_fact_table_query_vector_partition(context->fact_table,
                                                       source),
          source_type, source_lane_count);
  if (source_partition.count == 0 &&
      !loom_target_legalization_has_bitstream_packet_carrier(context,
                                                             source_type)) {
    return iree_ok_status();
  }

  *out_plan = (loom_vector_bitstream_plan_t){
      .module = context->module,
      .source = source,
      .result = result,
      .source_type = source_type,
      .result_type = result_type,
      .logical_type = logical_type,
      .storage_type = storage_type,
      .logical_lane_count = logical_lane_count,
      .storage_lane_count = storage_lane_count,
      .logical_element_bit_count = (uint32_t)logical_element_bit_count,
      .storage_element_bit_count = (uint32_t)storage_element_bit_count,
      .field_bit_count = (uint32_t)field_bit_count,
      .work_element_bit_count = work_element_bit_count,
      .storage_work_element_bit_count = storage_work_element_bit_count,
      .word_bit_count = word_bit_count,
      .packet_bit_count = packet_bit_count,
      .word_count = (uint32_t)word_count,
      .logical_packet_lane_mask = logical_packet_lane_mask,
      .storage_packet_lane_mask = storage_packet_lane_mask,
      .source_partition = source_partition,
      .total_bit_count = total_bit_count,
      .group_scalar_partition_components =
          context->vector_packet_policy == NULL,
      .is_bitpack = is_bitpack,
      .is_signed_unpack = is_signed_unpack,
  };
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_build_words_from_storage(
    loom_vector_bitstream_builder_t* builder,
    const loom_vector_bitstream_plan_t* plan, loom_value_id_t* words) {
  const loom_scalar_type_t word_element =
      loom_vector_bitstream_required_scalar_type(plan->word_bit_count);
  const loom_type_t word_type = loom_type_scalar(word_element);
  const loom_scalar_type_t storage_work_element =
      loom_vector_bitstream_required_scalar_type(
          plan->storage_work_element_bit_count);
  uint32_t storage_lane_offset = 0;
  loom_vector_bitstream_source_cursor_t source_cursor = {0};
  for (uint32_t word_index = 0; word_index < plan->word_count; ++word_index) {
    const uint64_t word_bit_offset =
        (uint64_t)word_index * plan->word_bit_count;
    const uint32_t valid_bit_count = (uint32_t)iree_min(
        plan->total_bit_count - word_bit_offset, plan->word_bit_count);
    const uint32_t word_storage_lane_count =
        valid_bit_count / plan->storage_element_bit_count;
    uint32_t word_storage_lane_offset = 0;
    loom_value_id_t word = LOOM_VALUE_ID_INVALID;
    while (word_storage_lane_offset < word_storage_lane_count) {
      uint32_t available_lane_count =
          word_storage_lane_count - word_storage_lane_offset;
      available_lane_count =
          iree_min(available_lane_count,
                   loom_vector_bitstream_source_component_remaining_lanes(
                       plan, &source_cursor));
      const uint32_t packet_lane_count =
          loom_vector_bitstream_select_lane_count(
              plan->storage_packet_lane_mask, available_lane_count);
      loom_value_id_t packet_scalar = LOOM_VALUE_ID_INVALID;
      loom_type_t packet_scalar_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_vector_bitstream_source_packet_scalar(
          builder, plan, plan->storage_type, storage_lane_offset,
          packet_lane_count, storage_work_element,
          plan->storage_work_element_bit_count, &source_cursor, &packet_scalar,
          &packet_scalar_type));
      IREE_RETURN_IF_ERROR(loom_vector_bitstream_compress(
          builder, packet_scalar, packet_lane_count,
          plan->storage_work_element_bit_count, plan->storage_element_bit_count,
          packet_scalar_type, &packet_scalar));
      IREE_RETURN_IF_ERROR(loom_vector_bitstream_scalar_convert_unsigned(
          builder, packet_scalar, packet_scalar_type, word_type,
          &packet_scalar));
      IREE_RETURN_IF_ERROR(loom_vector_bitstream_shift_left(
          builder, packet_scalar,
          word_storage_lane_offset * plan->storage_element_bit_count, word_type,
          &packet_scalar));
      if (word == LOOM_VALUE_ID_INVALID) {
        word = packet_scalar;
      } else {
        IREE_RETURN_IF_ERROR(loom_vector_bitstream_ori(
            builder, word, packet_scalar, word_type, &word));
      }
      storage_lane_offset += packet_lane_count;
      word_storage_lane_offset += packet_lane_count;
    }
    words[word_index] = word;
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_store_chunk(
    loom_vector_bitstream_builder_t* builder,
    const loom_vector_bitstream_plan_t* plan, loom_value_id_t chunk,
    loom_type_t chunk_type, uint32_t chunk_bit_count, uint64_t bit_offset,
    loom_value_id_t* words, bool* word_initialized) {
  const loom_scalar_type_t word_element =
      loom_vector_bitstream_required_scalar_type(plan->word_bit_count);
  const loom_type_t word_type = loom_type_scalar(word_element);
  IREE_RETURN_IF_ERROR(loom_vector_bitstream_scalar_convert_unsigned(
      builder, chunk, chunk_type, word_type, &chunk));

  const uint32_t word_index = (uint32_t)(bit_offset / plan->word_bit_count);
  const uint32_t word_bit_offset =
      (uint32_t)(bit_offset % plan->word_bit_count);
  loom_value_id_t low = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_vector_bitstream_shift_left(
      builder, chunk, word_bit_offset, word_type, &low));
  if (!word_initialized[word_index]) {
    words[word_index] = low;
    word_initialized[word_index] = true;
  } else {
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_ori(
        builder, words[word_index], low, word_type, &words[word_index]));
  }
  if (word_bit_offset + chunk_bit_count > plan->word_bit_count) {
    loom_value_id_t high = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_shift_right(
        builder, chunk, plan->word_bit_count - word_bit_offset, word_type,
        &high));
    if (!word_initialized[word_index + 1u]) {
      words[word_index + 1u] = high;
      word_initialized[word_index + 1u] = true;
    } else {
      IREE_RETURN_IF_ERROR(
          loom_vector_bitstream_ori(builder, words[word_index + 1u], high,
                                    word_type, &words[word_index + 1u]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_load_chunk(
    loom_vector_bitstream_builder_t* builder,
    const loom_vector_bitstream_plan_t* plan, const loom_value_id_t* words,
    uint64_t bit_offset, uint32_t chunk_bit_count, loom_type_t chunk_type,
    loom_value_id_t* out_chunk) {
  const loom_scalar_type_t word_element =
      loom_vector_bitstream_required_scalar_type(plan->word_bit_count);
  const loom_type_t word_type = loom_type_scalar(word_element);
  const uint32_t word_index = (uint32_t)(bit_offset / plan->word_bit_count);
  const uint32_t word_bit_offset =
      (uint32_t)(bit_offset % plan->word_bit_count);
  loom_value_id_t chunk = words[word_index];
  IREE_RETURN_IF_ERROR(loom_vector_bitstream_shift_right(
      builder, chunk, word_bit_offset, word_type, &chunk));
  if (word_bit_offset + chunk_bit_count > plan->word_bit_count) {
    loom_value_id_t high = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_shift_left(
        builder, words[word_index + 1u], plan->word_bit_count - word_bit_offset,
        word_type, &high));
    IREE_RETURN_IF_ERROR(
        loom_vector_bitstream_ori(builder, chunk, high, word_type, &chunk));
  }
  IREE_RETURN_IF_ERROR(loom_vector_bitstream_mask(
      builder, chunk, loom_vector_bitstream_low_mask(chunk_bit_count),
      word_type, &chunk));
  return loom_vector_bitstream_scalar_convert_unsigned(
      builder, chunk, word_type, chunk_type, out_chunk);
}

static iree_status_t loom_vector_bitstream_build_storage_packets(
    loom_vector_bitstream_builder_t* builder,
    const loom_vector_bitstream_plan_t* plan, const loom_value_id_t* words,
    loom_value_id_t** out_packets, uint32_t* out_packet_count) {
  *out_packets = NULL;
  *out_packet_count = 0;
  loom_value_id_t* packets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->context->arena, plan->storage_lane_count, sizeof(*packets),
      (void**)&packets));
  const loom_scalar_type_t storage_work_element =
      loom_vector_bitstream_required_scalar_type(
          plan->storage_work_element_bit_count);
  uint32_t storage_lane_offset = 0;
  uint32_t packet_count = 0;
  while (storage_lane_offset < plan->storage_lane_count) {
    const uint64_t bit_offset =
        (uint64_t)storage_lane_offset * plan->storage_element_bit_count;
    const uint32_t word_bit_offset =
        (uint32_t)(bit_offset % plan->word_bit_count);
    const uint32_t lanes_before_word_end =
        (plan->word_bit_count - word_bit_offset) /
        plan->storage_element_bit_count;
    const uint32_t packet_lane_count = loom_vector_bitstream_select_lane_count(
        plan->storage_packet_lane_mask,
        iree_min(plan->storage_lane_count - storage_lane_offset,
                 lanes_before_word_end));
    const uint32_t packet_scalar_bit_count =
        loom_vector_bitstream_packet_scalar_bit_count(
            plan, packet_lane_count, plan->storage_work_element_bit_count);
    const loom_type_t packet_scalar_type = loom_type_scalar(
        loom_vector_bitstream_required_scalar_type(packet_scalar_bit_count));
    loom_value_id_t packet_scalar = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_load_chunk(
        builder, plan, words, bit_offset,
        packet_lane_count * plan->storage_element_bit_count, packet_scalar_type,
        &packet_scalar));
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_expand(
        builder, packet_scalar, packet_lane_count,
        plan->storage_work_element_bit_count, plan->storage_element_bit_count,
        packet_scalar_type, &packet_scalar));
    loom_value_id_t work_packet = LOOM_VALUE_ID_INVALID;
    loom_type_t work_packet_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_scalar_to_packet(
        builder, packet_scalar, packet_scalar_type, storage_work_element,
        packet_lane_count, plan->storage_work_element_bit_count,
        plan->packet_bit_count, &work_packet, &work_packet_type));
    const loom_type_t packet_type = loom_vector_bitstream_vector_type(
        loom_type_element_type(plan->storage_type), packet_lane_count);
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_vector_convert_unsigned(
        builder, work_packet, work_packet_type, packet_type,
        &packets[packet_count++]));
    storage_lane_offset += packet_lane_count;
  }
  *out_packets = packets;
  *out_packet_count = packet_count;
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_pack(
    loom_vector_bitstream_builder_t* builder,
    const loom_vector_bitstream_plan_t* plan,
    loom_value_id_t* out_replacement) {
  loom_value_id_t* words = NULL;
  bool* word_initialized = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->context->arena, plan->word_count,
                                sizeof(*words), (void**)&words));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->context->arena, plan->word_count, sizeof(*word_initialized),
      (void**)&word_initialized));
  memset(word_initialized, 0, plan->word_count * sizeof(*word_initialized));

  const loom_scalar_type_t work_element =
      loom_vector_bitstream_required_scalar_type(plan->work_element_bit_count);
  uint32_t logical_lane_offset = 0;
  loom_vector_bitstream_source_cursor_t source_cursor = {0};
  while (logical_lane_offset < plan->logical_lane_count) {
    uint32_t available_lane_count =
        plan->logical_lane_count - logical_lane_offset;
    available_lane_count =
        iree_min(available_lane_count,
                 loom_vector_bitstream_source_component_remaining_lanes(
                     plan, &source_cursor));
    const uint32_t packet_lane_count = loom_vector_bitstream_select_lane_count(
        plan->logical_packet_lane_mask, available_lane_count);
    loom_value_id_t packed = LOOM_VALUE_ID_INVALID;
    loom_type_t packed_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_source_packet_scalar(
        builder, plan, plan->logical_type, logical_lane_offset,
        packet_lane_count, work_element, plan->work_element_bit_count,
        &source_cursor, &packed, &packed_type));
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_compress(
        builder, packed, packet_lane_count, plan->work_element_bit_count,
        plan->field_bit_count, packed_type, &packed));
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_store_chunk(
        builder, plan, packed, packed_type,
        packet_lane_count * plan->field_bit_count,
        (uint64_t)logical_lane_offset * plan->field_bit_count, words,
        word_initialized));
    logical_lane_offset += packet_lane_count;
  }

  loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
  const loom_scalar_type_t word_element =
      loom_vector_bitstream_required_scalar_type(plan->word_bit_count);
  const loom_type_t word_type = loom_type_scalar(word_element);
  IREE_RETURN_IF_ERROR(
      loom_vector_bitstream_constant(builder, word_type, 0, &zero));
  for (uint32_t i = 0; i < plan->word_count; ++i) {
    if (!word_initialized[i]) {
      words[i] = zero;
    }
  }

  loom_value_id_t* packets = NULL;
  uint32_t packet_count = 0;
  IREE_RETURN_IF_ERROR(loom_vector_bitstream_build_storage_packets(
      builder, plan, words, &packets, &packet_count));
  if (packet_count == 0) {
    return iree_ok_status();
  }
  if (packet_count == 1 &&
      loom_type_equal(
          loom_module_value_type(builder->context->module, packets[0]),
          plan->result_type)) {
    *out_replacement = packets[0];
    return iree_ok_status();
  }
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_concat_build(
      builder->builder, /*axis=*/0, packets, packet_count, plan->result_type,
      builder->location, &concat_op));
  *out_replacement = loom_vector_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_vector_bitstream_unpack(
    loom_vector_bitstream_builder_t* builder,
    const loom_vector_bitstream_plan_t* plan,
    loom_value_id_t* out_replacement) {
  loom_value_id_t* words = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(builder->context->arena, plan->word_count,
                                sizeof(*words), (void**)&words));
  IREE_RETURN_IF_ERROR(
      loom_vector_bitstream_build_words_from_storage(builder, plan, words));

  loom_value_id_t* packets = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->context->arena, plan->logical_lane_count, sizeof(*packets),
      (void**)&packets));
  uint32_t packet_count = 0;
  uint32_t logical_lane_offset = 0;
  const loom_scalar_type_t logical_element =
      loom_type_element_type(plan->logical_type);
  const loom_scalar_type_t work_element =
      loom_vector_bitstream_required_scalar_type(plan->work_element_bit_count);
  while (logical_lane_offset < plan->logical_lane_count) {
    const uint32_t packet_lane_count = loom_vector_bitstream_select_lane_count(
        plan->logical_packet_lane_mask,
        plan->logical_lane_count - logical_lane_offset);
    const uint32_t carrier_scalar_bit_count =
        loom_vector_bitstream_packet_scalar_bit_count(
            plan, packet_lane_count, plan->work_element_bit_count);
    const loom_type_t carrier_scalar_type = loom_type_scalar(
        loom_vector_bitstream_required_scalar_type(carrier_scalar_bit_count));
    loom_value_id_t carrier = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_load_chunk(
        builder, plan, words,
        (uint64_t)logical_lane_offset * plan->field_bit_count,
        packet_lane_count * plan->field_bit_count, carrier_scalar_type,
        &carrier));
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_expand(
        builder, carrier, packet_lane_count, plan->work_element_bit_count,
        plan->field_bit_count, carrier_scalar_type, &carrier));
    if (plan->is_signed_unpack && plan->logical_element_bit_count != 1u) {
      IREE_RETURN_IF_ERROR(loom_vector_bitstream_sign_extend_packed(
          builder, carrier, packet_lane_count, plan->work_element_bit_count,
          plan->field_bit_count, carrier_scalar_type, &carrier));
    }

    loom_value_id_t work_packet = LOOM_VALUE_ID_INVALID;
    loom_type_t work_packet_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_vector_bitstream_scalar_to_packet(
        builder, carrier, carrier_scalar_type, work_element, packet_lane_count,
        plan->work_element_bit_count, plan->packet_bit_count, &work_packet,
        &work_packet_type));
    const loom_type_t result_packet_type =
        loom_vector_bitstream_vector_type(logical_element, packet_lane_count);
    if (plan->is_signed_unpack) {
      IREE_RETURN_IF_ERROR(loom_vector_bitstream_vector_convert_signed(
          builder, work_packet, work_packet_type, result_packet_type,
          &packets[packet_count]));
    } else {
      IREE_RETURN_IF_ERROR(loom_vector_bitstream_vector_convert_unsigned(
          builder, work_packet, work_packet_type, result_packet_type,
          &packets[packet_count]));
    }
    ++packet_count;
    logical_lane_offset += packet_lane_count;
  }

  if (packet_count == 1 &&
      loom_type_equal(
          loom_module_value_type(builder->context->module, packets[0]),
          plan->result_type)) {
    *out_replacement = packets[0];
    return iree_ok_status();
  }
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_concat_build(
      builder->builder, /*axis=*/0, packets, packet_count, plan->result_type,
      builder->location, &concat_op));
  *out_replacement = loom_vector_concat_result(concat_op);
  return iree_ok_status();
}

iree_status_t loom_vector_packet_legalize_bitstream_words(
    loom_target_legalization_context_t* context, loom_op_t* op,
    bool* out_rewritten) {
  *out_rewritten = false;
  const loom_target_bitstream_policy_t* policy = context->bitstream_policy;
  if (policy == NULL) {
    return iree_ok_status();
  }
  const uint32_t native_word_bit_count = policy->word_bit_count;
  loom_vector_bitstream_plan_t plan = {0};
  IREE_RETURN_IF_ERROR(
      loom_vector_bitstream_plan(context, op, native_word_bit_count, &plan));
  if (plan.word_count == 0) {
    return iree_ok_status();
  }

  loom_rewriter_t* rewriter = context->rewriter;
  loom_builder_set_before(&rewriter->builder, op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_vector_bitstream_builder_t builder = {
      .context = context,
      .builder = &rewriter->builder,
      .location = op->location,
  };
  loom_value_id_t replacement = LOOM_VALUE_ID_INVALID;
  if (plan.is_bitpack) {
    IREE_RETURN_IF_ERROR(
        loom_vector_bitstream_pack(&builder, &plan, &replacement));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_vector_bitstream_unpack(&builder, &plan, &replacement));
  }
  if (replacement == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, &replacement, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_replace_all_uses_and_erase(rewriter, op, &replacement, 1));
  *out_rewritten = true;
  return iree_ok_status();
}

iree_status_t loom_vector_packet_prepare_bitstream_source_load(
    loom_target_legalization_context_t* context, loom_op_t* load_op,
    bool* out_rewritten) {
  *out_rewritten = false;
  if (!loom_vector_load_isa(load_op)) {
    return iree_ok_status();
  }
  const loom_type_t source_type =
      loom_module_value_type(context->module, loom_vector_load_result(load_op));
  uint32_t source_lane_count = 0;
  if (!loom_vector_bitstream_static_lane_count(source_type,
                                               &source_lane_count) ||
      source_lane_count <= 4u) {
    return iree_ok_status();
  }
  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(source_type));
  if (element_bit_count <= 0) {
    return iree_ok_status();
  }

  static const uint16_t kCandidateLaneCounts[] = {8u, 4u, 2u};
  uint16_t native_bit_counts[IREE_ARRAYSIZE(kCandidateLaneCounts)] = {0};
  uint16_t native_lane_counts[IREE_ARRAYSIZE(kCandidateLaneCounts)] = {0};
  uint8_t native_bit_count_count = 0;
  uint8_t native_lane_count_count = 0;
  uint16_t maximum_lane_count = 0;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(kCandidateLaneCounts); ++i) {
    const uint16_t lane_count = kCandidateLaneCounts[i];
    const loom_type_t candidate_type = loom_vector_bitstream_vector_type(
        loom_type_element_type(source_type), lane_count);
    if (!loom_target_legalization_has_bitstream_packet_carrier(
            context, candidate_type)) {
      continue;
    }
    native_lane_counts[native_lane_count_count++] = lane_count;
    maximum_lane_count = iree_max(maximum_lane_count, lane_count);
    uint16_t bit_count = 8u;
    const uint32_t required_bit_count =
        lane_count * (uint32_t)element_bit_count;
    while (bit_count < required_bit_count) {
      bit_count *= 2u;
    }
    if (native_bit_count_count == 0 ||
        native_bit_counts[native_bit_count_count - 1u] != bit_count) {
      native_bit_counts[native_bit_count_count++] = bit_count;
    }
  }
  if (native_bit_count_count == 0 || maximum_lane_count == 0) {
    return iree_ok_status();
  }

  const loom_target_vector_packet_lane_limit_t structural_lane_limit = {
      .element_type = loom_type_element_type(source_type),
      .maximum_lane_count = maximum_lane_count,
  };
  const loom_target_vector_packet_policy_t policy = {
      .native_bit_counts = native_bit_counts,
      .native_lane_counts = native_lane_counts,
      .structural_lane_limits = &structural_lane_limit,
      .maximum_unpacketized_bit_count = native_bit_counts[0],
      .native_bit_count_count = native_bit_count_count,
      .native_lane_count_count = native_lane_count_count,
      .structural_lane_limit_count = 1u,
  };
  return loom_vector_packet_legalize_load(context, load_op, &policy,
                                          out_rewritten);
}
