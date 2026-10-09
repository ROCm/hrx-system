// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/wasm/lower/vector_structural.h"

#include <string.h>

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/interleave.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/wasm/descriptors/descriptors.h"
#include "loom/target/emit/wasm/lower/predicate_representation.h"
#include "loom/target/emit/wasm/lower/vector_carrier.h"

enum {
  LOOM_WASM_VECTOR_STRUCTURAL_PLAN_INTERLEAVE = 0x501,
  LOOM_WASM_VECTOR_STRUCTURAL_PLAN_CONCAT = 0x502,
  LOOM_WASM_VECTOR_STRUCTURAL_PLAN_SLICE = 0x503,
  LOOM_WASM_VECTOR_STRUCTURAL_PLAN_BITCAST = 0x504,
  LOOM_WASM_VECTOR_STRUCTURAL_PLAN_FROM_ELEMENTS = 0x505,
  LOOM_WASM_VECTOR_STRUCTURAL_PLAN_SHUFFLE = 0x506,
};

typedef struct loom_wasm_vector_slice_plan_t {
  // Physical byte offset of the result payload in the source payload.
  uint16_t source_byte_offset;
  // Selected source carrier retained before source IR mutation.
  loom_wasm_vector_carrier_t source_carrier;
  // Selected result carrier retained before source IR mutation.
  loom_wasm_vector_carrier_t result_carrier;
} loom_wasm_vector_slice_plan_t;

typedef struct loom_wasm_vector_constructor_descriptors_t {
  // Descriptor reference for broadcasting the first packet lane.
  uint16_t splat;
  // Descriptor reference for replacing each remaining packet lane.
  uint16_t replace_lane;
  // Number of physical lanes in one v128 packet.
  uint8_t lane_count;
} loom_wasm_vector_constructor_descriptors_t;

typedef struct loom_wasm_vector_from_elements_plan_t {
  // Selected result carrier retained before source IR mutation.
  loom_wasm_vector_carrier_t result_carrier;
  // Lane construction descriptors matching the selected physical width.
  loom_wasm_vector_constructor_descriptors_t descriptors;
  // Whether scalar i1 lanes require conversion from 0/1 to 0/-1.
  bool normalize_predicate_lanes;
} loom_wasm_vector_from_elements_plan_t;

typedef struct loom_wasm_vector_interleave_plan_t {
  // Shared byte-routing plan for the selected physical lane width.
  loom_vector_interleave_packet_plan_t packet_plan;
  // Selected carriers for the semantic source values.
  loom_wasm_vector_carrier_t source_carriers[2];
  // Selected carriers for the semantic result values.
  loom_wasm_vector_carrier_t result_carriers[2];
  // Number of populated source carriers.
  uint8_t source_count;
  // Number of populated result carriers.
  uint8_t result_count;
} loom_wasm_vector_interleave_plan_t;

typedef struct loom_wasm_vector_concat_plan_t {
  // Selected result carrier retained before source IR mutation.
  loom_wasm_vector_carrier_t result_carrier;
  // Number of selected input carriers stored after this header.
  uint16_t input_count;
  // Selected carriers for each concatenated input.
  loom_wasm_vector_carrier_t input_carriers[];
} loom_wasm_vector_concat_plan_t;

typedef struct loom_wasm_vector_shuffle_plan_t {
  // Selected source/result carrier retained before source IR mutation.
  loom_wasm_vector_carrier_t carrier;
} loom_wasm_vector_shuffle_plan_t;

static loom_wasm_vector_carrier_t
loom_wasm_vector_carrier_for_source_value_with_representation(
    const loom_module_t* module, loom_low_lower_context_t* context,
    loom_value_id_t value_id,
    loom_low_representation_id_t query_representation) {
  const loom_type_t source_type = loom_module_value_type(module, value_id);
  if (!loom_wasm_predicate_type(source_type, NULL)) {
    return loom_wasm_vector_carrier_for_type(source_type);
  }
  if (context == NULL) {
    return loom_wasm_predicate_carrier(source_type, query_representation);
  }
  loom_low_representation_id_t representation = LOOM_LOW_REPRESENTATION_ID_NONE;
  loom_low_lower_representation_lookup(context, value_id, &representation);
  return loom_wasm_predicate_carrier(source_type, representation);
}

static loom_wasm_vector_carrier_t loom_wasm_vector_carrier_for_source_value(
    const loom_module_t* module, loom_low_lower_context_t* context,
    loom_value_id_t value_id) {
  return loom_wasm_vector_carrier_for_source_value_with_representation(
      module, context, value_id, LOOM_LOW_REPRESENTATION_ID_NONE);
}

static bool loom_wasm_interleave_plan_from_op(
    const loom_module_t* module, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_wasm_vector_interleave_plan_t* out_plan) {
  *out_plan = (loom_wasm_vector_interleave_plan_t){0};
  loom_type_t source_type = loom_type_none();
  loom_type_t result_type = loom_type_none();
  loom_vector_interleave_kind_t kind = LOOM_VECTOR_INTERLEAVE_KIND_ZIP;
  int64_t axis = 0;
  if (loom_vector_interleave_isa(source_op)) {
    source_type =
        loom_module_value_type(module, loom_vector_interleave_even(source_op));
    const loom_type_t odd_type =
        loom_module_value_type(module, loom_vector_interleave_odd(source_op));
    result_type = loom_module_value_type(
        module, loom_vector_interleave_result(source_op));
    if (!loom_type_equal(source_type, odd_type)) {
      return false;
    }
    axis = loom_vector_interleave_axis(source_op);
  } else if (loom_vector_deinterleave_isa(source_op)) {
    source_type = loom_module_value_type(
        module, loom_vector_deinterleave_source(source_op));
    result_type = loom_module_value_type(
        module, loom_vector_deinterleave_even(source_op));
    const loom_type_t odd_type =
        loom_module_value_type(module, loom_vector_deinterleave_odd(source_op));
    if (!loom_type_equal(result_type, odd_type)) {
      return false;
    }
    kind = LOOM_VECTOR_INTERLEAVE_KIND_UNZIP;
    axis = loom_vector_deinterleave_axis(source_op);
  } else {
    return false;
  }

  const loom_type_t half_type =
      kind == LOOM_VECTOR_INTERLEAVE_KIND_ZIP ? source_type : result_type;
  const loom_type_t combined_type =
      kind == LOOM_VECTOR_INTERLEAVE_KIND_ZIP ? result_type : source_type;
  const loom_value_id_t half_value =
      kind == LOOM_VECTOR_INTERLEAVE_KIND_ZIP
          ? loom_vector_interleave_even(source_op)
          : loom_vector_deinterleave_even(source_op);
  const loom_value_id_t combined_value =
      kind == LOOM_VECTOR_INTERLEAVE_KIND_ZIP
          ? loom_vector_interleave_result(source_op)
          : loom_vector_deinterleave_source(source_op);
  const loom_wasm_vector_carrier_t combined_carrier =
      loom_wasm_vector_carrier_for_source_value(module, context,
                                                combined_value);
  const loom_low_representation_id_t query_representation =
      context == NULL && loom_wasm_predicate_type(combined_type, NULL)
          ? combined_carrier.element_bit_count
          : LOOM_LOW_REPRESENTATION_ID_NONE;
  const loom_wasm_vector_carrier_t half_carrier =
      loom_wasm_vector_carrier_for_source_value_with_representation(
          module, context, half_value, query_representation);
  if (half_carrier.packet_count == 0 || combined_carrier.packet_count == 0 ||
      half_carrier.element_bit_count != combined_carrier.element_bit_count ||
      !loom_vector_interleave_packet_plan_initialize(
          kind, half_type, combined_type, axis, half_carrier.element_bit_count,
          LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT,
          /*maximum_packet_count=*/UINT16_MAX, &out_plan->packet_plan)) {
    return false;
  }
  if (kind == LOOM_VECTOR_INTERLEAVE_KIND_ZIP) {
    out_plan->source_count = 2;
    out_plan->result_count = 1;
    out_plan->source_carriers[0] = half_carrier;
    out_plan->source_carriers[1] =
        loom_wasm_vector_carrier_for_source_value_with_representation(
            module, context, loom_vector_interleave_odd(source_op),
            query_representation);
    out_plan->result_carriers[0] = combined_carrier;
  } else {
    out_plan->source_count = 1;
    out_plan->result_count = 2;
    out_plan->source_carriers[0] = combined_carrier;
    out_plan->result_carriers[0] = half_carrier;
    out_plan->result_carriers[1] =
        loom_wasm_vector_carrier_for_source_value_with_representation(
            module, context, loom_vector_deinterleave_odd(source_op),
            query_representation);
  }
  for (uint8_t i = 0; i < out_plan->source_count; ++i) {
    if (out_plan->source_carriers[i].element_bit_count !=
        half_carrier.element_bit_count) {
      return false;
    }
  }
  for (uint8_t i = 0; i < out_plan->result_count; ++i) {
    if (out_plan->result_carriers[i].element_bit_count !=
        half_carrier.element_bit_count) {
      return false;
    }
  }
  return out_plan->packet_plan.source_packet_count ==
             out_plan->source_carriers[0].packet_count &&
         out_plan->packet_plan.result_packet_count ==
             out_plan->result_carriers[0].packet_count;
}

static bool loom_wasm_concat_plan_from_op(
    const loom_module_t* module, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_wasm_vector_carrier_t* out_result) {
  if (!loom_vector_concat_isa(source_op) ||
      loom_vector_concat_axis(source_op) != 0) {
    return false;
  }
  const loom_value_slice_t inputs = loom_vector_concat_inputs(source_op);
  if (inputs.count == 0) {
    return false;
  }
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_concat_result(source_op));
  const loom_wasm_vector_carrier_t result_carrier =
      loom_wasm_vector_carrier_for_source_value(
          module, context, loom_vector_concat_result(source_op));
  if (loom_type_rank(result_type) != 1 || result_carrier.packet_count == 0) {
    return false;
  }
  const loom_low_representation_id_t query_representation =
      context == NULL && loom_wasm_predicate_type(result_type, NULL)
          ? result_carrier.element_bit_count
          : LOOM_LOW_REPRESENTATION_ID_NONE;
  uint32_t payload_byte_count = 0;
  for (uint16_t i = 0; i < inputs.count; ++i) {
    const loom_type_t input_type =
        loom_module_value_type(module, inputs.values[i]);
    const loom_wasm_vector_carrier_t input_carrier =
        loom_wasm_vector_carrier_for_source_value_with_representation(
            module, context, inputs.values[i], query_representation);
    if (loom_type_rank(input_type) != 1 ||
        !loom_type_element_type_equals(input_type, result_type) ||
        input_carrier.packet_count == 0 ||
        input_carrier.element_bit_count != result_carrier.element_bit_count ||
        input_carrier.payload_byte_count > result_carrier.payload_byte_count ||
        payload_byte_count > (uint32_t)result_carrier.payload_byte_count -
                                 input_carrier.payload_byte_count) {
      return false;
    }
    payload_byte_count += input_carrier.payload_byte_count;
  }
  if (payload_byte_count != result_carrier.payload_byte_count) {
    return false;
  }
  *out_result = result_carrier;
  return true;
}

static bool loom_wasm_slice_plan_from_op(
    const loom_module_t* module, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_wasm_vector_slice_plan_t* out_plan) {
  *out_plan = (loom_wasm_vector_slice_plan_t){0};
  if (!loom_vector_slice_isa(source_op) ||
      loom_vector_slice_offsets(source_op).count != 0) {
    return false;
  }
  const loom_attribute_t offsets = loom_vector_slice_static_offsets(source_op);
  if (offsets.kind != LOOM_ATTR_I64_ARRAY || offsets.count != 1 ||
      offsets.i64_array[0] < 0 || offsets.i64_array[0] == INT64_MIN) {
    return false;
  }

  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_slice_source(source_op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_slice_result(source_op));
  const loom_wasm_vector_carrier_t source_carrier =
      loom_wasm_vector_carrier_for_source_value(
          module, context, loom_vector_slice_source(source_op));
  const loom_low_representation_id_t query_representation =
      context == NULL && loom_wasm_predicate_type(source_type, NULL)
          ? source_carrier.element_bit_count
          : LOOM_LOW_REPRESENTATION_ID_NONE;
  const loom_wasm_vector_carrier_t result_carrier =
      loom_wasm_vector_carrier_for_source_value_with_representation(
          module, context, loom_vector_slice_result(source_op),
          query_representation);
  if (loom_type_rank(source_type) != 1 || loom_type_rank(result_type) != 1 ||
      !loom_type_element_type_equals(source_type, result_type) ||
      source_carrier.packet_count == 0 || result_carrier.packet_count == 0 ||
      source_carrier.element_bit_count != result_carrier.element_bit_count) {
    return false;
  }
  const uint64_t source_byte_offset =
      (uint64_t)offsets.i64_array[0] * source_carrier.element_bit_count / 8u;
  if (source_byte_offset > UINT16_MAX ||
      source_byte_offset + result_carrier.payload_byte_count >
          source_carrier.payload_byte_count) {
    return false;
  }
  out_plan->source_byte_offset = (uint16_t)source_byte_offset;
  out_plan->source_carrier = source_carrier;
  out_plan->result_carrier = result_carrier;
  return true;
}

static bool loom_wasm_element_type_is_numeric_payload(
    loom_scalar_type_t element_type) {
  return element_type >= LOOM_SCALAR_TYPE_I8 &&
         element_type <= LOOM_SCALAR_TYPE_F64;
}

static bool loom_wasm_element_type_is_integer_payload(
    loom_scalar_type_t element_type) {
  return loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD,
                                       element_type);
}

static bool loom_wasm_vector_constructor_descriptors_for_element(
    loom_scalar_type_t element_type, uint16_t physical_element_bit_count,
    loom_wasm_vector_constructor_descriptors_t* out_descriptors) {
  *out_descriptors = (loom_wasm_vector_constructor_descriptors_t){0};
  switch (physical_element_bit_count) {
    case 8:
      *out_descriptors = (loom_wasm_vector_constructor_descriptors_t){
          .splat = WASM_CORE_SIMD128_DESCRIPTOR_REF_I8X16_SPLAT,
          .replace_lane = WASM_CORE_SIMD128_DESCRIPTOR_REF_I8X16_REPLACE_LANE,
          .lane_count = 16,
      };
      return true;
    case 16:
      *out_descriptors = (loom_wasm_vector_constructor_descriptors_t){
          .splat = WASM_CORE_SIMD128_DESCRIPTOR_REF_I16X8_SPLAT,
          .replace_lane = WASM_CORE_SIMD128_DESCRIPTOR_REF_I16X8_REPLACE_LANE,
          .lane_count = 8,
      };
      return true;
    case 32:
      if (element_type == LOOM_SCALAR_TYPE_F32) {
        *out_descriptors = (loom_wasm_vector_constructor_descriptors_t){
            .splat = WASM_CORE_SIMD128_DESCRIPTOR_REF_F32X4_SPLAT,
            .replace_lane = WASM_CORE_SIMD128_DESCRIPTOR_REF_F32X4_REPLACE_LANE,
            .lane_count = 4,
        };
      } else {
        *out_descriptors = (loom_wasm_vector_constructor_descriptors_t){
            .splat = WASM_CORE_SIMD128_DESCRIPTOR_REF_I32X4_SPLAT,
            .replace_lane = WASM_CORE_SIMD128_DESCRIPTOR_REF_I32X4_REPLACE_LANE,
            .lane_count = 4,
        };
      }
      return true;
    case 64:
      if (element_type == LOOM_SCALAR_TYPE_F64) {
        *out_descriptors = (loom_wasm_vector_constructor_descriptors_t){
            .splat = WASM_CORE_SIMD128_DESCRIPTOR_REF_F64X2_SPLAT,
            .replace_lane = WASM_CORE_SIMD128_DESCRIPTOR_REF_F64X2_REPLACE_LANE,
            .lane_count = 2,
        };
      } else {
        *out_descriptors = (loom_wasm_vector_constructor_descriptors_t){
            .splat = WASM_CORE_SIMD128_DESCRIPTOR_REF_I64X2_SPLAT,
            .replace_lane = WASM_CORE_SIMD128_DESCRIPTOR_REF_I64X2_REPLACE_LANE,
            .lane_count = 2,
        };
      }
      return true;
    default:
      return false;
  }
}

static bool loom_wasm_from_elements_plan_from_op(
    const loom_module_t* module, loom_low_lower_context_t* context,
    const loom_op_t* source_op,
    loom_wasm_vector_from_elements_plan_t* out_plan) {
  *out_plan = (loom_wasm_vector_from_elements_plan_t){0};
  if (!loom_vector_from_elements_isa(source_op)) {
    return false;
  }
  const loom_value_id_t result = loom_vector_from_elements_result(source_op);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const loom_wasm_vector_carrier_t carrier =
      loom_wasm_vector_carrier_for_source_value(module, context, result);
  loom_wasm_vector_constructor_descriptors_t descriptors;
  if (carrier.packet_count == 0 ||
      !loom_wasm_vector_constructor_descriptors_for_element(
          loom_type_element_type(result_type), carrier.element_bit_count,
          &descriptors) ||
      carrier.element_bit_count * descriptors.lane_count != 128u) {
    return false;
  }

  // The generated scalar-to-vector rules already produce the optimal chain
  // for rank-one integer partials and exact physical vectors. Keep those on
  // the ordinary descriptor path; this plan owns tuple carriers, logical
  // shapes, address vectors, and partial floating-point/predicate vectors.
  if (loom_type_rank(result_type) == 1 && carrier.packet_count == 1) {
    const loom_scalar_type_t element_type = loom_type_element_type(result_type);
    if (loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_INTEGER_PAYLOAD,
                                      element_type) ||
        (loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_FLOAT,
                                       element_type) &&
         (uint64_t)loom_type_dim_static_size_at(result_type, 0) ==
             descriptors.lane_count)) {
      return false;
    }
  }
  out_plan->result_carrier = carrier;
  out_plan->descriptors = descriptors;
  out_plan->normalize_predicate_lanes =
      loom_type_element_type(result_type) == LOOM_SCALAR_TYPE_I1;
  return true;
}

static bool loom_wasm_bitcast_has_generated_rule(
    loom_type_t source_type, loom_type_t result_type,
    const loom_wasm_vector_carrier_t* source_carrier,
    const loom_wasm_vector_carrier_t* result_carrier) {
  if (loom_type_rank(source_type) != 1 || loom_type_rank(result_type) != 1 ||
      source_carrier->packet_count != 1 || result_carrier->packet_count != 1) {
    return false;
  }
  const loom_scalar_type_t source_element = loom_type_element_type(source_type);
  const loom_scalar_type_t result_element = loom_type_element_type(result_type);
  if (source_carrier->payload_byte_count ==
          LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT &&
      result_carrier->payload_byte_count ==
          LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT) {
    return loom_wasm_element_type_is_numeric_payload(source_element) &&
           loom_wasm_element_type_is_numeric_payload(result_element);
  }
  return loom_wasm_element_type_is_integer_payload(source_element) &&
         loom_wasm_element_type_is_integer_payload(result_element);
}

static bool loom_wasm_bitcast_plan_from_op(const loom_module_t* module,
                                           const loom_op_t* source_op) {
  if (!loom_vector_bitcast_isa(source_op)) {
    return false;
  }
  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_bitcast_input(source_op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_bitcast_result(source_op));
  const loom_wasm_vector_carrier_t source_carrier =
      loom_wasm_vector_carrier_for_type(source_type);
  const loom_wasm_vector_carrier_t result_carrier =
      loom_wasm_vector_carrier_for_type(result_type);
  return source_carrier.packet_count != 0 &&
         source_carrier.payload_byte_count ==
             result_carrier.payload_byte_count &&
         source_carrier.packet_count == result_carrier.packet_count &&
         !loom_wasm_bitcast_has_generated_rule(
             source_type, result_type, &source_carrier, &result_carrier);
}

static bool loom_wasm_shuffle_plan_from_op(
    const loom_module_t* module, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_wasm_vector_shuffle_plan_t* out_plan) {
  *out_plan = (loom_wasm_vector_shuffle_plan_t){0};
  if (!loom_vector_shuffle_isa(source_op)) {
    return false;
  }
  const loom_value_id_t source = loom_vector_shuffle_source(source_op);
  const loom_value_id_t result = loom_vector_shuffle_result(source_op);
  if (!loom_wasm_predicate_type(loom_module_value_type(module, source), NULL)) {
    return false;
  }
  const loom_wasm_vector_carrier_t source_carrier =
      loom_wasm_vector_carrier_for_source_value(module, context, source);
  const loom_wasm_vector_carrier_t result_carrier =
      loom_wasm_vector_carrier_for_source_value(module, context, result);
  if (source_carrier.packet_count != 1 || result_carrier.packet_count != 1 ||
      source_carrier.element_bit_count != result_carrier.element_bit_count) {
    return false;
  }
  out_plan->carrier = source_carrier;
  return true;
}

iree_status_t loom_wasm_query_vector_structural_contract(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  (void)user_data;
  *out_result = loom_target_contract_query_result_empty();
  if (environment->vector_lane_projection.source_lane_count != 0) {
    return iree_ok_status();
  }
  loom_wasm_vector_from_elements_plan_t from_elements_plan;
  loom_wasm_vector_interleave_plan_t interleave_plan;
  loom_wasm_vector_carrier_t concat_result_carrier;
  loom_wasm_vector_slice_plan_t slice_plan;
  loom_wasm_vector_shuffle_plan_t shuffle_plan;
  if (loom_wasm_from_elements_plan_from_op(environment->module,
                                           /*context=*/NULL, source_op,
                                           &from_elements_plan) ||
      loom_wasm_interleave_plan_from_op(environment->module, /*context=*/NULL,
                                        source_op, &interleave_plan) ||
      loom_wasm_concat_plan_from_op(environment->module, /*context=*/NULL,
                                    source_op, &concat_result_carrier) ||
      loom_wasm_slice_plan_from_op(environment->module, /*context=*/NULL,
                                   source_op, &slice_plan) ||
      loom_wasm_shuffle_plan_from_op(environment->module, /*context=*/NULL,
                                     source_op, &shuffle_plan) ||
      loom_wasm_bitcast_plan_from_op(environment->module, source_op)) {
    out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  }
  return iree_ok_status();
}

bool loom_wasm_vector_structural_plan_isa(loom_low_lower_plan_t plan) {
  return plan.id >= LOOM_WASM_VECTOR_STRUCTURAL_PLAN_INTERLEAVE &&
         plan.id <= LOOM_WASM_VECTOR_STRUCTURAL_PLAN_SHUFFLE;
}

iree_status_t loom_wasm_select_vector_structural_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  const loom_module_t* module = loom_low_lower_context_module(context);
  loom_wasm_vector_from_elements_plan_t from_elements_plan;
  if (loom_wasm_from_elements_plan_from_op(module, context, source_op,
                                           &from_elements_plan)) {
    loom_wasm_vector_from_elements_plan_t* retained_plan = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
        context, sizeof(*retained_plan), (void**)&retained_plan));
    *retained_plan = from_elements_plan;
    *out_plan = loom_low_lower_plan_make(
        LOOM_WASM_VECTOR_STRUCTURAL_PLAN_FROM_ELEMENTS, retained_plan);
    return iree_ok_status();
  }
  loom_wasm_vector_interleave_plan_t interleave_plan;
  if (loom_wasm_interleave_plan_from_op(module, context, source_op,
                                        &interleave_plan)) {
    loom_wasm_vector_interleave_plan_t* retained_plan = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
        context, sizeof(*retained_plan), (void**)&retained_plan));
    *retained_plan = interleave_plan;
    *out_plan = loom_low_lower_plan_make(
        LOOM_WASM_VECTOR_STRUCTURAL_PLAN_INTERLEAVE, retained_plan);
    return iree_ok_status();
  }
  loom_wasm_vector_carrier_t concat_result_carrier;
  if (loom_wasm_concat_plan_from_op(module, context, source_op,
                                    &concat_result_carrier)) {
    const loom_value_slice_t inputs = loom_vector_concat_inputs(source_op);
    const iree_host_size_t plan_size =
        sizeof(loom_wasm_vector_concat_plan_t) +
        inputs.count * sizeof(loom_wasm_vector_carrier_t);
    loom_wasm_vector_concat_plan_t* retained_plan = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
        context, plan_size, (void**)&retained_plan));
    retained_plan->result_carrier = concat_result_carrier;
    for (uint16_t i = 0; i < inputs.count; ++i) {
      retained_plan->input_carriers[i] =
          loom_wasm_vector_carrier_for_source_value(module, context,
                                                    inputs.values[i]);
    }
    retained_plan->input_count = inputs.count;
    *out_plan = loom_low_lower_plan_make(
        LOOM_WASM_VECTOR_STRUCTURAL_PLAN_CONCAT, retained_plan);
    return iree_ok_status();
  }
  loom_wasm_vector_slice_plan_t slice_plan;
  if (loom_wasm_slice_plan_from_op(module, context, source_op, &slice_plan)) {
    loom_wasm_vector_slice_plan_t* retained_plan = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
        context, sizeof(*retained_plan), (void**)&retained_plan));
    *retained_plan = slice_plan;
    *out_plan = loom_low_lower_plan_make(LOOM_WASM_VECTOR_STRUCTURAL_PLAN_SLICE,
                                         retained_plan);
    return iree_ok_status();
  }
  loom_wasm_vector_shuffle_plan_t shuffle_plan;
  if (loom_wasm_shuffle_plan_from_op(module, context, source_op,
                                     &shuffle_plan)) {
    loom_wasm_vector_shuffle_plan_t* retained_plan = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
        context, sizeof(*retained_plan), (void**)&retained_plan));
    *retained_plan = shuffle_plan;
    *out_plan = loom_low_lower_plan_make(
        LOOM_WASM_VECTOR_STRUCTURAL_PLAN_SHUFFLE, retained_plan);
    return iree_ok_status();
  }
  if (loom_wasm_bitcast_plan_from_op(module, source_op)) {
    *out_plan = loom_low_lower_plan_make(
        LOOM_WASM_VECTOR_STRUCTURAL_PLAN_BITCAST, NULL);
  }
  return iree_ok_status();
}

void loom_wasm_mark_vector_structural_plan_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan) {
  (void)plan;
  const loom_value_id_t* operands = loom_op_const_operands(source_op);
  for (uint16_t i = 0; i < source_op->operand_count; ++i) {
    loom_low_lower_require_source_value_storage(context, operands[i]);
  }
}

void loom_wasm_describe_vector_structural_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan, loom_low_lower_plan_report_t* out_report) {
  (void)context;
  (void)source_op;
  switch (plan.id) {
    case LOOM_WASM_VECTOR_STRUCTURAL_PLAN_INTERLEAVE: {
      const loom_wasm_vector_interleave_plan_t* interleave_plan =
          (const loom_wasm_vector_interleave_plan_t*)plan.target_data;
      out_report->plan_key =
          interleave_plan->packet_plan.kind == LOOM_VECTOR_INTERLEAVE_KIND_ZIP
              ? IREE_SV("interleave.i8x16-shuffle")
              : IREE_SV("deinterleave.i8x16-shuffle");
      return;
    }
    case LOOM_WASM_VECTOR_STRUCTURAL_PLAN_CONCAT:
      out_report->plan_key = IREE_SV("concat.v128-packets");
      return;
    case LOOM_WASM_VECTOR_STRUCTURAL_PLAN_SLICE:
      out_report->plan_key = IREE_SV("slice.v128-packets");
      return;
    case LOOM_WASM_VECTOR_STRUCTURAL_PLAN_BITCAST:
      out_report->plan_key = IREE_SV("bitcast.v128-packets");
      return;
    case LOOM_WASM_VECTOR_STRUCTURAL_PLAN_FROM_ELEMENTS:
      out_report->plan_key = IREE_SV("from-elements.v128-packets");
      return;
    case LOOM_WASM_VECTOR_STRUCTURAL_PLAN_SHUFFLE:
      out_report->plan_key = IREE_SV("shuffle.i8x16");
      return;
    default:
      IREE_ASSERT_UNREACHABLE("unknown Wasm vector structural plan");
  }
}

typedef struct loom_wasm_packet_byte_source_t {
  // Source vector ordinal for this result byte.
  uint16_t source_index;
  // v128 packet ordinal within the source vector.
  uint16_t source_packet;
  // Byte ordinal within the source v128 packet.
  uint8_t source_byte;
} loom_wasm_packet_byte_source_t;

typedef struct loom_wasm_packet_emitter_t {
  // Active source-to-low lowering context.
  loom_low_lower_context_t* context;
  // Source operation owning emitted Low instructions.
  const loom_op_t* source_op;
  // Low values corresponding to each semantic source value.
  loom_value_id_t* low_sources;
  // Physical packet count for each semantic source value.
  uint16_t* source_packet_counts;
  // First cache entry for each semantic source value.
  uint32_t* source_packet_starts;
  // Lazily projected one-unit packet values.
  loom_value_id_t* source_packets;
  // Number of semantic source values.
  uint16_t source_count;
  // One-unit wasm.v128 Low register type.
  loom_type_t packet_type;
  // Interned i8x16.shuffle immediate names.
  loom_string_id_t lane_names[LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT];
  // Whether |lane_names| has been initialized.
  bool has_lane_names;
} loom_wasm_packet_emitter_t;

static iree_status_t loom_wasm_packet_emitter_initialize(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_id_t* source_values,
    const loom_wasm_vector_carrier_t* source_carriers, uint16_t source_count,
    loom_wasm_packet_emitter_t* out_emitter) {
  *out_emitter = (loom_wasm_packet_emitter_t){
      .context = context,
      .source_op = source_op,
      .source_count = source_count,
      .packet_type = loom_type_none(),
  };
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_V128, 1,
      &out_emitter->packet_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, source_count, sizeof(*out_emitter->low_sources),
      (void**)&out_emitter->low_sources));
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, source_count, sizeof(*out_emitter->source_packet_counts),
      (void**)&out_emitter->source_packet_counts));
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, source_count, sizeof(*out_emitter->source_packet_starts),
      (void**)&out_emitter->source_packet_starts));

  uint32_t total_packet_count = 0;
  for (uint16_t i = 0; i < source_count; ++i) {
    out_emitter->low_sources[i] =
        loom_low_lower_lookup_value(context, source_values[i]);
    IREE_ASSERT_NE(source_carriers[i].packet_count, 0);
    out_emitter->source_packet_counts[i] = source_carriers[i].packet_count;
    out_emitter->source_packet_starts[i] = total_packet_count;
    total_packet_count += source_carriers[i].packet_count;
  }
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, total_packet_count, sizeof(*out_emitter->source_packets),
      (void**)&out_emitter->source_packets));
  for (uint32_t i = 0; i < total_packet_count; ++i) {
    out_emitter->source_packets[i] = LOOM_VALUE_ID_INVALID;
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_packet_emitter_source_packet(
    loom_wasm_packet_emitter_t* emitter, uint16_t source_index,
    uint16_t packet_index, loom_value_id_t* out_packet) {
  IREE_ASSERT_LT(source_index, emitter->source_count);
  IREE_ASSERT_LT(packet_index, emitter->source_packet_counts[source_index]);
  loom_value_id_t* cached =
      &emitter->source_packets[emitter->source_packet_starts[source_index] +
                               packet_index];
  if (*cached == LOOM_VALUE_ID_INVALID) {
    if (emitter->source_packet_counts[source_index] == 1) {
      *cached = emitter->low_sources[source_index];
    } else {
      loom_op_t* slice_op = NULL;
      IREE_RETURN_IF_ERROR(loom_low_slice_build(
          loom_low_lower_context_builder(emitter->context),
          emitter->low_sources[source_index], packet_index,
          emitter->packet_type, emitter->source_op->location, &slice_op));
      *cached = loom_low_slice_result(slice_op);
    }
  }
  *out_packet = *cached;
  return iree_ok_status();
}

static bool loom_wasm_packet_byte_source_same_packet(
    loom_wasm_packet_byte_source_t lhs, loom_wasm_packet_byte_source_t rhs) {
  return lhs.source_index == rhs.source_index &&
         lhs.source_packet == rhs.source_packet;
}

static uint8_t loom_wasm_packet_source_ordinal(
    const loom_wasm_packet_byte_source_t* sources, uint8_t source_count,
    loom_wasm_packet_byte_source_t source) {
  for (uint8_t i = 0; i < source_count; ++i) {
    if (loom_wasm_packet_byte_source_same_packet(sources[i], source)) {
      return i;
    }
  }
  return UINT8_MAX;
}

static iree_status_t loom_wasm_packet_emitter_initialize_lane_names(
    loom_wasm_packet_emitter_t* emitter) {
  if (emitter->has_lane_names) {
    return iree_ok_status();
  }
  static const char* kLaneNames[] = {
      "lane0",  "lane1",  "lane2",  "lane3",  "lane4",  "lane5",
      "lane6",  "lane7",  "lane8",  "lane9",  "lane10", "lane11",
      "lane12", "lane13", "lane14", "lane15",
  };
  static_assert(
      IREE_ARRAYSIZE(kLaneNames) == LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT,
      "Wasm shuffle names must cover every result byte");
  loom_builder_t* builder = loom_low_lower_context_builder(emitter->context);
  for (uint8_t i = 0; i < IREE_ARRAYSIZE(kLaneNames); ++i) {
    IREE_RETURN_IF_ERROR(loom_builder_intern_string(
        builder, iree_make_cstring_view(kLaneNames[i]),
        &emitter->lane_names[i]));
  }
  emitter->has_lane_names = true;
  return iree_ok_status();
}

static iree_status_t loom_wasm_packet_emitter_shuffle(
    loom_wasm_packet_emitter_t* emitter, loom_value_id_t lhs,
    loom_value_id_t rhs,
    const uint8_t lanes[LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT],
    loom_value_id_t* out_result) {
  IREE_RETURN_IF_ERROR(loom_wasm_packet_emitter_initialize_lane_names(emitter));
  loom_named_attr_t attrs[LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT];
  for (uint8_t i = 0; i < IREE_ARRAYSIZE(attrs); ++i) {
    attrs[i] = (loom_named_attr_t){
        .name_id = emitter->lane_names[i],
        .value = loom_attr_i64(lanes[i]),
    };
  }
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor =
          &loom_low_lower_context_descriptor_set(emitter->context)
               ->descriptors[WASM_CORE_SIMD128_DESCRIPTOR_REF_I8X16_SHUFFLE],
  };
  const loom_value_id_t operands[] = {lhs, rhs};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      emitter->context, &descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)),
      &emitter->packet_type, 1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, emitter->source_op->location, &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_wasm_packet_emitter_route_packet(
    loom_wasm_packet_emitter_t* emitter,
    const loom_wasm_packet_byte_source_t
        byte_sources[LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT],
    uint8_t live_byte_count, loom_value_id_t* out_packet) {
  IREE_ASSERT_GT(live_byte_count, 0);
  IREE_ASSERT_LE(live_byte_count, LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT);
  loom_wasm_packet_byte_source_t
      packet_sources[LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT];
  uint8_t packet_source_count = 0;
  for (uint8_t byte = 0; byte < live_byte_count; ++byte) {
    if (loom_wasm_packet_source_ordinal(packet_sources, packet_source_count,
                                        byte_sources[byte]) == UINT8_MAX) {
      packet_sources[packet_source_count++] = byte_sources[byte];
    }
  }
  IREE_ASSERT_GT(packet_source_count, 0);

  loom_value_id_t first_packet = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_packet_emitter_source_packet(
      emitter, packet_sources[0].source_index, packet_sources[0].source_packet,
      &first_packet));
  bool is_identity = packet_source_count == 1;
  for (uint8_t byte = 0; is_identity && byte < live_byte_count; ++byte) {
    is_identity = byte_sources[byte].source_byte == byte;
  }
  if (is_identity) {
    *out_packet = first_packet;
    return iree_ok_status();
  }

  loom_value_id_t second_packet = first_packet;
  if (packet_source_count > 1) {
    IREE_RETURN_IF_ERROR(loom_wasm_packet_emitter_source_packet(
        emitter, packet_sources[1].source_index,
        packet_sources[1].source_packet, &second_packet));
  }
  uint8_t lanes[LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT] = {0};
  for (uint8_t byte = 0; byte < live_byte_count; ++byte) {
    const uint8_t source_ordinal = loom_wasm_packet_source_ordinal(
        packet_sources, packet_source_count, byte_sources[byte]);
    if (source_ordinal == 0) {
      lanes[byte] = byte_sources[byte].source_byte;
    } else if (source_ordinal == 1) {
      lanes[byte] = LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT +
                    byte_sources[byte].source_byte;
    }
  }
  IREE_RETURN_IF_ERROR(loom_wasm_packet_emitter_shuffle(
      emitter, first_packet, second_packet, lanes, out_packet));

  for (uint8_t source_ordinal = 2; source_ordinal < packet_source_count;
       ++source_ordinal) {
    loom_value_id_t source_packet = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_wasm_packet_emitter_source_packet(
        emitter, packet_sources[source_ordinal].source_index,
        packet_sources[source_ordinal].source_packet, &source_packet));
    for (uint8_t byte = 0; byte < IREE_ARRAYSIZE(lanes); ++byte) {
      lanes[byte] = byte;
    }
    for (uint8_t byte = 0; byte < live_byte_count; ++byte) {
      if (loom_wasm_packet_byte_source_same_packet(
              packet_sources[source_ordinal], byte_sources[byte])) {
        lanes[byte] = LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT +
                      byte_sources[byte].source_byte;
      }
    }
    IREE_RETURN_IF_ERROR(loom_wasm_packet_emitter_shuffle(
        emitter, *out_packet, source_packet, lanes, out_packet));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_bind_packets(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_value_id_t* packets, loom_wasm_vector_carrier_t result_carrier,
    loom_value_id_t result_value) {
  IREE_ASSERT_NE(result_carrier.packet_count, 0);
  loom_value_id_t low_result = packets[0];
  if (result_carrier.packet_count > 1) {
    loom_type_t result_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
        context, WASM_CORE_SIMD128_REG_CLASS_ID_V128,
        result_carrier.packet_count, &result_type));
    loom_op_t* concat_op = NULL;
    IREE_RETURN_IF_ERROR(
        loom_low_concat_build(loom_low_lower_context_builder(context), packets,
                              result_carrier.packet_count, result_type,
                              source_op->location, &concat_op));
    low_result = loom_low_concat_result(concat_op);
  }
  return loom_low_lower_bind_value(context, result_value, low_result);
}

static iree_status_t loom_wasm_emit_shuffle(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_wasm_vector_shuffle_plan_t* plan) {
  const loom_value_id_t source = loom_vector_shuffle_source(source_op);
  loom_wasm_packet_emitter_t emitter;
  IREE_RETURN_IF_ERROR(loom_wasm_packet_emitter_initialize(
      context, source_op, &source, &plan->carrier, 1, &emitter));
  loom_value_id_t source_packet = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_packet_emitter_source_packet(
      &emitter, /*source_index=*/0, /*packet_index=*/0, &source_packet));

  const loom_attribute_t source_lanes =
      loom_vector_shuffle_source_lanes(source_op);
  const uint8_t bytes_per_lane = plan->carrier.element_bit_count / 8u;
  uint8_t lanes[LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT] = {0};
  for (iree_host_size_t result_lane = 0; result_lane < source_lanes.count;
       ++result_lane) {
    const uint8_t source_byte =
        (uint8_t)(source_lanes.i64_array[result_lane] * bytes_per_lane);
    for (uint8_t byte = 0; byte < bytes_per_lane; ++byte) {
      lanes[result_lane * bytes_per_lane + byte] = source_byte + byte;
    }
  }

  loom_value_id_t low_result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_wasm_packet_emitter_shuffle(
      &emitter, source_packet, source_packet, lanes, &low_result));
  return loom_low_lower_bind_value(
      context, loom_vector_shuffle_result(source_op), low_result);
}

static iree_status_t loom_wasm_emit_predicate_zero(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t i32_type, loom_value_id_t* out_zero) {
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor =
          &loom_low_lower_context_descriptor_set(context)
               ->descriptors[WASM_CORE_SIMD128_DESCRIPTOR_REF_I32_CONST],
  };
  loom_string_id_t value_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_builder_intern_string(loom_low_lower_context_builder(context),
                                 IREE_SV("i32_value"), &value_name));
  const loom_named_attr_t value_attr = {
      .name_id = value_name,
      .value = loom_attr_i64(0),
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_const(
      context, &descriptor, loom_make_named_attr_slice(&value_attr, 1),
      i32_type, source_op->location, &low_op));
  *out_zero = loom_low_const_result(low_op);
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_predicate_lane_mask(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t zero, loom_value_id_t predicate,
    uint8_t physical_element_bit_count, loom_type_t i32_type,
    loom_type_t i64_type, loom_value_id_t* out_mask) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_lower_resolved_descriptor_t subtract_descriptor = {
      .descriptor =
          &descriptor_set
               ->descriptors[WASM_CORE_SIMD128_DESCRIPTOR_REF_I32_SUB],
  };
  const loom_value_id_t subtract_operands[] = {zero, predicate};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &subtract_descriptor, subtract_operands,
      IREE_ARRAYSIZE(subtract_operands), (loom_named_attr_slice_t){0},
      &i32_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &low_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(low_op), 0);
  if (physical_element_bit_count != 64) {
    return iree_ok_status();
  }

  const loom_low_lower_resolved_descriptor_t extend_descriptor = {
      .descriptor =
          &descriptor_set
               ->descriptors[WASM_CORE_SIMD128_DESCRIPTOR_REF_I64_EXTEND_I32_S],
  };
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &extend_descriptor, out_mask, 1, (loom_named_attr_slice_t){0},
      &i64_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &low_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_from_elements(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_wasm_vector_from_elements_plan_t* plan) {
  const loom_value_slice_t elements =
      loom_vector_from_elements_elements(source_op);
  const loom_value_id_t result = loom_vector_from_elements_result(source_op);
  const loom_wasm_vector_carrier_t result_carrier = plan->result_carrier;
  const loom_wasm_vector_constructor_descriptors_t descriptors =
      plan->descriptors;

  loom_value_id_t* packets = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, result_carrier.packet_count, sizeof(*packets),
      (void**)&packets));
  loom_type_t packet_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, WASM_CORE_SIMD128_REG_CLASS_ID_V128, 1, &packet_type));
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_lower_resolved_descriptor_t splat_descriptor = {
      .descriptor = &descriptor_set->descriptors[descriptors.splat],
  };
  const loom_low_lower_resolved_descriptor_t replace_descriptor = {
      .descriptor = &descriptor_set->descriptors[descriptors.replace_lane],
  };
  loom_type_t i32_type = loom_type_none();
  loom_type_t i64_type = loom_type_none();
  loom_value_id_t predicate_zero = LOOM_VALUE_ID_INVALID;
  if (plan->normalize_predicate_lanes) {
    IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
        context, WASM_CORE_SIMD128_REG_CLASS_ID_I32, 1, &i32_type));
    if (result_carrier.element_bit_count == 64) {
      IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
          context, WASM_CORE_SIMD128_REG_CLASS_ID_I64, 1, &i64_type));
    }
    IREE_RETURN_IF_ERROR(loom_wasm_emit_predicate_zero(
        context, source_op, i32_type, &predicate_zero));
  }
  loom_string_id_t lane_name = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(
      loom_low_lower_context_builder(context), IREE_SV("lane"), &lane_name));

  for (uint16_t packet = 0; packet < result_carrier.packet_count; ++packet) {
    const iree_host_size_t element_start =
        (iree_host_size_t)packet * descriptors.lane_count;
    const uint8_t live_lane_count =
        (uint8_t)iree_min((iree_host_size_t)descriptors.lane_count,
                          elements.count - element_start);
    IREE_ASSERT_GT(live_lane_count, 0);

    loom_value_id_t scalar =
        loom_low_lower_lookup_value(context, elements.values[element_start]);
    if (plan->normalize_predicate_lanes) {
      IREE_RETURN_IF_ERROR(loom_wasm_emit_predicate_lane_mask(
          context, source_op, predicate_zero, scalar,
          result_carrier.element_bit_count, i32_type, i64_type, &scalar));
    }
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &splat_descriptor, &scalar, 1, (loom_named_attr_slice_t){0},
        &packet_type, 1,
        /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
        &low_op));
    packets[packet] = loom_value_slice_get(loom_low_op_results(low_op), 0);

    for (uint8_t lane = 1; lane < live_lane_count; ++lane) {
      scalar = loom_low_lower_lookup_value(
          context, elements.values[element_start + lane]);
      if (plan->normalize_predicate_lanes) {
        IREE_RETURN_IF_ERROR(loom_wasm_emit_predicate_lane_mask(
            context, source_op, predicate_zero, scalar,
            result_carrier.element_bit_count, i32_type, i64_type, &scalar));
      }
      const loom_value_id_t operands[] = {packets[packet], scalar};
      const loom_named_attr_t lane_attr = {
          .name_id = lane_name,
          .value = loom_attr_i64(lane),
      };
      IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
          context, &replace_descriptor, operands, IREE_ARRAYSIZE(operands),
          loom_make_named_attr_slice(&lane_attr, 1), &packet_type, 1,
          /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
          &low_op));
      packets[packet] = loom_value_slice_get(loom_low_op_results(low_op), 0);
    }
  }
  return loom_wasm_bind_packets(context, source_op, packets, result_carrier,
                                result);
}

static iree_status_t loom_wasm_emit_interleave_result(
    loom_wasm_packet_emitter_t* emitter,
    const loom_vector_interleave_packet_plan_t* plan, uint8_t result_index,
    loom_value_id_t result_value, loom_wasm_vector_carrier_t result_carrier) {
  loom_value_id_t* packets = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      emitter->context, result_carrier.packet_count, sizeof(*packets),
      (void**)&packets));
  for (uint16_t packet = 0; packet < result_carrier.packet_count; ++packet) {
    const uint8_t live_byte_count =
        (uint8_t)loom_vector_interleave_packet_plan_result_live_byte_count(
            plan, result_index, packet);
    loom_wasm_packet_byte_source_t
        byte_sources[LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT] = {{0}};
    uint16_t cursor = 0;
    loom_vector_interleave_packet_segment_t segment;
    while (loom_vector_interleave_packet_plan_next_segment(
        plan, result_index, packet, &cursor, &segment)) {
      for (uint16_t byte = 0; byte < segment.byte_count; ++byte) {
        const uint16_t result_byte = segment.result_packet_byte_offset + byte;
        byte_sources[result_byte] = (loom_wasm_packet_byte_source_t){
            .source_index = segment.source_index,
            .source_packet = segment.source_packet,
            .source_byte = (uint8_t)(segment.source_packet_byte_offset + byte),
        };
      }
    }
    IREE_RETURN_IF_ERROR(loom_wasm_packet_emitter_route_packet(
        emitter, byte_sources, live_byte_count, &packets[packet]));
  }
  return loom_wasm_bind_packets(emitter->context, emitter->source_op, packets,
                                result_carrier, result_value);
}

static iree_status_t loom_wasm_emit_interleave(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_wasm_vector_interleave_plan_t* plan) {
  loom_value_id_t source_values[2];
  uint16_t source_count = 1;
  if (plan->packet_plan.kind == LOOM_VECTOR_INTERLEAVE_KIND_ZIP) {
    source_values[0] = loom_vector_interleave_even(source_op);
    source_values[1] = loom_vector_interleave_odd(source_op);
    source_count = 2;
  } else {
    source_values[0] = loom_vector_deinterleave_source(source_op);
  }
  loom_wasm_packet_emitter_t emitter;
  IREE_RETURN_IF_ERROR(loom_wasm_packet_emitter_initialize(
      context, source_op, source_values, plan->source_carriers, source_count,
      &emitter));
  if (plan->packet_plan.kind == LOOM_VECTOR_INTERLEAVE_KIND_ZIP) {
    const loom_value_id_t result = loom_vector_interleave_result(source_op);
    return loom_wasm_emit_interleave_result(&emitter, &plan->packet_plan,
                                            /*result_index=*/0, result,
                                            plan->result_carriers[0]);
  }
  const loom_value_id_t results[2] = {
      loom_vector_deinterleave_even(source_op),
      loom_vector_deinterleave_odd(source_op),
  };
  for (uint8_t result_index = 0; result_index < 2; ++result_index) {
    IREE_RETURN_IF_ERROR(loom_wasm_emit_interleave_result(
        &emitter, &plan->packet_plan, result_index, results[result_index],
        plan->result_carriers[result_index]));
  }
  return iree_ok_status();
}

static iree_status_t loom_wasm_emit_concat(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_wasm_vector_concat_plan_t* plan) {
  const loom_value_slice_t inputs = loom_vector_concat_inputs(source_op);
  const loom_value_id_t result = loom_vector_concat_result(source_op);
  IREE_ASSERT_EQ(inputs.count, plan->input_count);
  if (inputs.count == 1) {
    return loom_low_lower_bind_value_alias(context, inputs.values[0], result);
  }

  loom_wasm_packet_emitter_t emitter;
  IREE_RETURN_IF_ERROR(loom_wasm_packet_emitter_initialize(
      context, source_op, inputs.values, plan->input_carriers, inputs.count,
      &emitter));
  const loom_wasm_vector_carrier_t result_carrier = plan->result_carrier;
  uint16_t* source_byte_starts = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, inputs.count, sizeof(*source_byte_starts),
      (void**)&source_byte_starts));
  uint16_t next_source_byte = 0;
  for (uint16_t i = 0; i < inputs.count; ++i) {
    source_byte_starts[i] = next_source_byte;
    next_source_byte += plan->input_carriers[i].payload_byte_count;
  }

  loom_value_id_t* packets = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, result_carrier.packet_count, sizeof(*packets),
      (void**)&packets));
  for (uint16_t packet = 0; packet < result_carrier.packet_count; ++packet) {
    const uint16_t result_byte_start =
        packet * LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT;
    const uint8_t live_byte_count = (uint8_t)iree_min(
        LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT,
        result_carrier.payload_byte_count - result_byte_start);
    loom_wasm_packet_byte_source_t
        byte_sources[LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT] = {{0}};
    uint16_t source_index = 0;
    for (uint8_t byte = 0; byte < live_byte_count; ++byte) {
      const uint16_t result_byte = result_byte_start + byte;
      while (source_index + 1u < inputs.count &&
             result_byte >= source_byte_starts[source_index + 1u]) {
        ++source_index;
      }
      const uint16_t source_byte =
          result_byte - source_byte_starts[source_index];
      byte_sources[byte] = (loom_wasm_packet_byte_source_t){
          .source_index = source_index,
          .source_packet =
              (uint16_t)(source_byte /
                         LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT),
          .source_byte = (uint8_t)(source_byte %
                                   LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT),
      };
    }
    IREE_RETURN_IF_ERROR(loom_wasm_packet_emitter_route_packet(
        &emitter, byte_sources, live_byte_count, &packets[packet]));
  }
  return loom_wasm_bind_packets(context, source_op, packets, result_carrier,
                                result);
}

static iree_status_t loom_wasm_emit_slice(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_wasm_vector_slice_plan_t* plan) {
  const loom_value_id_t source = loom_vector_slice_source(source_op);
  const loom_value_id_t result = loom_vector_slice_result(source_op);
  const loom_wasm_vector_carrier_t source_carrier = plan->source_carrier;
  const loom_wasm_vector_carrier_t result_carrier = plan->result_carrier;
  if (plan->source_byte_offset == 0 &&
      source_carrier.packet_count == result_carrier.packet_count) {
    return loom_low_lower_bind_value_alias(context, source, result);
  }

  loom_wasm_packet_emitter_t emitter;
  IREE_RETURN_IF_ERROR(loom_wasm_packet_emitter_initialize(
      context, source_op, &source, &source_carrier, 1, &emitter));
  loom_value_id_t* packets = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, result_carrier.packet_count, sizeof(*packets),
      (void**)&packets));
  for (uint16_t packet = 0; packet < result_carrier.packet_count; ++packet) {
    const uint16_t result_byte_start =
        packet * LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT;
    const uint8_t live_byte_count = (uint8_t)iree_min(
        LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT,
        result_carrier.payload_byte_count - result_byte_start);
    loom_wasm_packet_byte_source_t
        byte_sources[LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT] = {{0}};
    for (uint8_t byte = 0; byte < live_byte_count; ++byte) {
      const uint16_t source_byte =
          plan->source_byte_offset + result_byte_start + byte;
      byte_sources[byte] = (loom_wasm_packet_byte_source_t){
          .source_index = 0,
          .source_packet =
              (uint16_t)(source_byte /
                         LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT),
          .source_byte = (uint8_t)(source_byte %
                                   LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT),
      };
    }
    IREE_RETURN_IF_ERROR(loom_wasm_packet_emitter_route_packet(
        &emitter, byte_sources, live_byte_count, &packets[packet]));
  }
  return loom_wasm_bind_packets(context, source_op, packets, result_carrier,
                                result);
}

static iree_status_t loom_wasm_emit_bitcast(loom_low_lower_context_t* context,
                                            const loom_op_t* source_op) {
  return loom_low_lower_bind_value_alias(context,
                                         loom_vector_bitcast_input(source_op),
                                         loom_vector_bitcast_result(source_op));
}

iree_status_t loom_wasm_emit_vector_structural_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan) {
  switch (plan.id) {
    case LOOM_WASM_VECTOR_STRUCTURAL_PLAN_FROM_ELEMENTS:
      return loom_wasm_emit_from_elements(
          context, source_op,
          (const loom_wasm_vector_from_elements_plan_t*)plan.target_data);
    case LOOM_WASM_VECTOR_STRUCTURAL_PLAN_INTERLEAVE:
      return loom_wasm_emit_interleave(
          context, source_op,
          (const loom_wasm_vector_interleave_plan_t*)plan.target_data);
    case LOOM_WASM_VECTOR_STRUCTURAL_PLAN_CONCAT:
      return loom_wasm_emit_concat(
          context, source_op,
          (const loom_wasm_vector_concat_plan_t*)plan.target_data);
    case LOOM_WASM_VECTOR_STRUCTURAL_PLAN_SLICE:
      return loom_wasm_emit_slice(
          context, source_op,
          (const loom_wasm_vector_slice_plan_t*)plan.target_data);
    case LOOM_WASM_VECTOR_STRUCTURAL_PLAN_BITCAST:
      return loom_wasm_emit_bitcast(context, source_op);
    case LOOM_WASM_VECTOR_STRUCTURAL_PLAN_SHUFFLE:
      return loom_wasm_emit_shuffle(
          context, source_op,
          (const loom_wasm_vector_shuffle_plan_t*)plan.target_data);
    default:
      IREE_ASSERT_UNREACHABLE("unknown Wasm vector structural plan");
      IREE_BUILTIN_UNREACHABLE();
  }
}
