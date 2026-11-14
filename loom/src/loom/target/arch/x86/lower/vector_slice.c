// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/lower/vector_slice.h"

#include "loom/ops/low/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/target/arch/x86/descriptors/vector_slice.h"
#include "loom/target/arch/x86/feature_bits.h"
#include "loom/target/arch/x86/lower/predicate_representation.h"
#include "loom/target/arch/x86/register_classes.h"

enum {
  LOOM_X86_VECTOR_SLICE_PLAN = 0x7838736Cu,
  LOOM_X86_VECTOR_SLICE_QUARTER_BYTE_COUNT = 16,
};

typedef enum loom_x86_vector_slice_mechanism_e {
  // The result aliases the source or projects its low register class.
  LOOM_X86_VECTOR_SLICE_PROJECT = 0,
  // One XMM quarter is shifted to expose a window in its low bytes.
  LOOM_X86_VECTOR_SLICE_SHIFT_XMM = 1,
  // Adjacent XMM quarters are aligned to expose a crossing window.
  LOOM_X86_VECTOR_SLICE_ALIGN_XMM = 2,
  // One aligned XMM quarter is extracted from a wide source.
  LOOM_X86_VECTOR_SLICE_EXTRACT_XMM = 3,
  // A ZMM source is aligned at a 32-bit lane and projected to YMM.
  LOOM_X86_VECTOR_SLICE_ALIGN_DWORD_ZMM = 4,
  // A ZMM source is aligned at a 64-bit lane and projected to YMM.
  LOOM_X86_VECTOR_SLICE_ALIGN_QWORD_ZMM = 5,
  // The high YMM half is extracted from a ZMM source.
  LOOM_X86_VECTOR_SLICE_EXTRACT_YMM = 6,
  // Two byte-aligned XMM windows are assembled into one YMM result.
  LOOM_X86_VECTOR_SLICE_ASSEMBLE_YMM = 7,
  // A packed predicate mask is shifted by its logical lane offset.
  LOOM_X86_VECTOR_SLICE_SHIFT_MASK = 8,
} loom_x86_vector_slice_mechanism_t;

typedef enum loom_x86_vector_slice_conversion_e {
  // The sliced carrier already has the selected result representation.
  LOOM_X86_VECTOR_SLICE_CONVERSION_NONE = 0,
  // SIMD predicate lanes are sign-extended to the selected width.
  LOOM_X86_VECTOR_SLICE_CONVERSION_WIDEN = 1,
  // SIMD predicate lanes are packed to the selected width.
  LOOM_X86_VECTOR_SLICE_CONVERSION_NARROW = 2,
  // Packed K bits are expanded into all-zero/all-one SIMD lanes.
  LOOM_X86_VECTOR_SLICE_CONVERSION_EXPAND_MASK = 3,
  // SIMD lane sign bits are compressed into a packed K mask.
  LOOM_X86_VECTOR_SLICE_CONVERSION_COMPRESS_MASK = 4,
} loom_x86_vector_slice_conversion_t;

typedef struct loom_x86_vector_slice_shape_t {
  // True when logical i1 lanes require selected physical representations.
  bool is_predicate;
  // Declared bit width of one logical vector element.
  uint8_t element_bit_width;
  // Number of logical lanes in the source vector.
  uint8_t source_lane_count;
  // Number of logical lanes in the result vector.
  uint8_t result_lane_count;
  // Logical lane offset of the result window in the source.
  uint8_t source_lane_offset;
  // Physical register class carrying the source value.
  loom_x86_register_class_t source_register_class;
  // Physical register class carrying the sliced source representation.
  loom_x86_register_class_t window_register_class;
  // Physical register class carrying the selected result representation.
  loom_x86_register_class_t result_register_class;
  // Logical byte offset of the result window in the source.
  uint8_t source_byte_offset;
  // Logical result payload length in bytes.
  uint8_t result_byte_count;
} loom_x86_vector_slice_shape_t;

typedef struct loom_x86_vector_slice_plan_t {
  // Selected physical register-window mechanism.
  loom_x86_vector_slice_mechanism_t mechanism;
  // Selected predicate representation conversion after slicing.
  loom_x86_vector_slice_conversion_t conversion;
  // Physical register class carrying the source value.
  loom_x86_register_class_t source_register_class;
  // Physical register class carrying the sliced source representation.
  loom_x86_register_class_t window_register_class;
  // Physical register class carrying the selected result representation.
  loom_x86_register_class_t result_register_class;
  // Physical bit width of one source predicate lane, or zero for numeric data.
  uint8_t source_predicate_lane_bit_width;
  // Physical bit width of one result predicate lane, or zero for numeric data.
  uint8_t result_predicate_lane_bit_width;
  // Logical lane offset consumed by a packed K-mask slice.
  uint8_t predicate_lane_offset;
  // True when a no-AVX512VL mask conversion must use a full ZMM bridge.
  bool use_full_width_mask_conversion;
  // First 128-bit source quarter consumed by an XMM window.
  uint8_t source_quarter;
  // Byte shift within the first source quarter.
  uint8_t quarter_byte_offset;
  // Immediate consumed by a full-width ZMM alignment.
  uint8_t alignment_lane_offset;
} loom_x86_vector_slice_plan_t;

typedef struct loom_x86_vector_slice_emitter_t {
  // Active lowering context.
  loom_low_lower_context_t* context;
  // Source slice supplying the result binding and source location.
  const loom_op_t* source_op;
  // Descriptor vocabulary for the selected x86 view.
  const loom_x86_vector_slice_descriptors_t* descriptors;
  // XMM intermediate type.
  loom_type_t xmm_type;
  // YMM intermediate type.
  loom_type_t ymm_type;
  // ZMM intermediate type when available.
  loom_type_t zmm_type;
  // K intermediate type when available.
  loom_type_t mask_type;
  // Interned immediate field name "bytes".
  loom_string_id_t bytes_name;
  // Interned immediate field name "lane".
  loom_string_id_t lane_name;
  // Interned immediate field name "lanes".
  loom_string_id_t lanes_name;
  // Interned immediate field name "shift".
  loom_string_id_t shift_name;
  // Interned immediate field name "control".
  loom_string_id_t control_name;
} loom_x86_vector_slice_emitter_t;

static bool loom_x86_vector_slice_match_shape(
    const loom_module_t* module, const loom_op_t* source_op,
    const loom_x86_vector_slice_descriptors_t* descriptors,
    loom_x86_vector_slice_shape_t* out_shape) {
  *out_shape = (loom_x86_vector_slice_shape_t){0};
  if (descriptors == NULL || !loom_vector_slice_isa(source_op) ||
      loom_vector_slice_offsets(source_op).count != 0) {
    return false;
  }
  const loom_attribute_t offsets = loom_vector_slice_static_offsets(source_op);
  if (offsets.kind != LOOM_ATTR_I64_ARRAY || offsets.count != 1 ||
      offsets.i64_array[0] < 0) {
    return false;
  }

  const loom_type_t source_type =
      loom_module_value_type(module, loom_vector_slice_source(source_op));
  const loom_type_t result_type =
      loom_module_value_type(module, loom_vector_slice_result(source_op));
  if (!loom_type_is_vector(source_type) || !loom_type_is_vector(result_type) ||
      loom_type_rank(source_type) != 1 || loom_type_rank(result_type) != 1 ||
      !loom_type_is_all_static(source_type) ||
      !loom_type_is_all_static(result_type) ||
      !loom_type_element_type_equals(source_type, result_type)) {
    return false;
  }

  const int64_t source_lane_count =
      loom_type_dim_static_size_at(source_type, 0);
  const int64_t result_lane_count =
      loom_type_dim_static_size_at(result_type, 0);
  const uint64_t source_lane_offset = (uint64_t)offsets.i64_array[0];
  if (source_lane_count <= 0 || source_lane_count > UINT8_MAX ||
      result_lane_count <= 0 || result_lane_count > UINT8_MAX ||
      source_lane_offset > UINT8_MAX ||
      source_lane_offset + (uint64_t)result_lane_count >
          (uint64_t)source_lane_count) {
    return false;
  }

  if (loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_I1) {
    const bool supported =
        descriptors->supports_mask
            ? loom_x86_avx512_predicate_type(source_type, NULL) &&
                  loom_x86_avx512_predicate_type(result_type, NULL)
            : loom_x86_avx2_predicate_type(source_type, NULL) &&
                  loom_x86_avx2_predicate_type(result_type, NULL);
    if (!supported) {
      return false;
    }
    *out_shape = (loom_x86_vector_slice_shape_t){
        .is_predicate = true,
        .element_bit_width = 1,
        .source_lane_count = (uint8_t)source_lane_count,
        .result_lane_count = (uint8_t)result_lane_count,
        .source_lane_offset = (uint8_t)source_lane_offset,
    };
    return true;
  }

  uint32_t element_bit_width = 0;
  if (!loom_x86_register_vector_element_bit_width(
          loom_type_element_type(source_type), &element_bit_width)) {
    return false;
  }
  const uint64_t source_bit_width =
      (uint64_t)source_lane_count * element_bit_width;
  const uint64_t result_bit_width =
      (uint64_t)result_lane_count * element_bit_width;
  if (source_bit_width > descriptors->maximum_vector_bit_width ||
      result_bit_width > descriptors->maximum_vector_bit_width) {
    return false;
  }

  loom_x86_register_class_t source_register_class;
  loom_x86_register_class_t result_register_class;
  if (!loom_x86_register_class_for_vector_bit_width((uint32_t)source_bit_width,
                                                    &source_register_class) ||
      !loom_x86_register_class_for_vector_bit_width((uint32_t)result_bit_width,
                                                    &result_register_class)) {
    return false;
  }

  const uint64_t source_byte_offset =
      source_lane_offset * element_bit_width / 8u;
  const uint64_t result_byte_count = result_bit_width / 8u;
  if (source_byte_offset > UINT8_MAX || result_byte_count > UINT8_MAX) {
    return false;
  }

  *out_shape = (loom_x86_vector_slice_shape_t){
      .element_bit_width = (uint8_t)element_bit_width,
      .source_lane_count = (uint8_t)source_lane_count,
      .result_lane_count = (uint8_t)result_lane_count,
      .source_lane_offset = (uint8_t)source_lane_offset,
      .source_register_class = source_register_class,
      .window_register_class = result_register_class,
      .result_register_class = result_register_class,
      .source_byte_offset = (uint8_t)source_byte_offset,
      .result_byte_count = (uint8_t)result_byte_count,
  };
  return true;
}

static loom_x86_vector_slice_plan_t loom_x86_vector_slice_select_mechanism(
    const loom_x86_vector_slice_shape_t* shape) {
  loom_x86_vector_slice_plan_t plan = {
      .mechanism = LOOM_X86_VECTOR_SLICE_PROJECT,
      .conversion = LOOM_X86_VECTOR_SLICE_CONVERSION_NONE,
      .source_register_class = shape->source_register_class,
      .window_register_class = shape->window_register_class,
      .result_register_class = shape->result_register_class,
      .source_quarter =
          shape->source_byte_offset / LOOM_X86_VECTOR_SLICE_QUARTER_BYTE_COUNT,
      .quarter_byte_offset =
          shape->source_byte_offset % LOOM_X86_VECTOR_SLICE_QUARTER_BYTE_COUNT,
  };
  if (shape->source_byte_offset == 0) {
    return plan;
  }
  if (shape->source_register_class == LOOM_X86_REGISTER_CLASS_XMM) {
    plan.mechanism = LOOM_X86_VECTOR_SLICE_SHIFT_XMM;
    return plan;
  }
  if (shape->window_register_class == LOOM_X86_REGISTER_CLASS_XMM) {
    if (plan.quarter_byte_offset == 0) {
      plan.mechanism = LOOM_X86_VECTOR_SLICE_EXTRACT_XMM;
    } else if (plan.quarter_byte_offset + shape->result_byte_count <=
               LOOM_X86_VECTOR_SLICE_QUARTER_BYTE_COUNT) {
      plan.mechanism = LOOM_X86_VECTOR_SLICE_SHIFT_XMM;
    } else {
      plan.mechanism = LOOM_X86_VECTOR_SLICE_ALIGN_XMM;
    }
    return plan;
  }

  IREE_ASSERT_EQ(shape->source_register_class, LOOM_X86_REGISTER_CLASS_ZMM);
  IREE_ASSERT_EQ(shape->window_register_class, LOOM_X86_REGISTER_CLASS_YMM);
  if (shape->source_byte_offset == 32) {
    plan.mechanism = LOOM_X86_VECTOR_SLICE_EXTRACT_YMM;
  } else if (shape->element_bit_width == 64) {
    plan.mechanism = LOOM_X86_VECTOR_SLICE_ALIGN_QWORD_ZMM;
    plan.alignment_lane_offset = shape->source_byte_offset / 8u;
  } else if (shape->element_bit_width >= 32 ||
             shape->source_byte_offset % 4u == 0) {
    plan.mechanism = LOOM_X86_VECTOR_SLICE_ALIGN_DWORD_ZMM;
    plan.alignment_lane_offset = shape->source_byte_offset / 4u;
  } else {
    plan.mechanism = LOOM_X86_VECTOR_SLICE_ASSEMBLE_YMM;
  }
  return plan;
}

static bool loom_x86_vector_slice_predicate_register_class(
    const loom_x86_vector_slice_descriptors_t* descriptors, loom_type_t type,
    loom_low_representation_id_t representation,
    loom_x86_register_class_t* out_register_class) {
  return descriptors->supports_mask
             ? loom_x86_avx512_predicate_register_class(type, representation,
                                                        out_register_class)
             : loom_x86_avx2_predicate_register_class(type, representation,
                                                      out_register_class);
}

static loom_x86_vector_slice_conversion_t
loom_x86_vector_slice_select_conversion(
    loom_low_representation_id_t source_representation,
    loom_low_representation_id_t result_representation) {
  if (source_representation == result_representation) {
    return LOOM_X86_VECTOR_SLICE_CONVERSION_NONE;
  }
  if (source_representation == LOOM_X86_PREDICATE_REPRESENTATION_MASK) {
    return LOOM_X86_VECTOR_SLICE_CONVERSION_EXPAND_MASK;
  }
  if (result_representation == LOOM_X86_PREDICATE_REPRESENTATION_MASK) {
    return LOOM_X86_VECTOR_SLICE_CONVERSION_COMPRESS_MASK;
  }
  return source_representation < result_representation
             ? LOOM_X86_VECTOR_SLICE_CONVERSION_WIDEN
             : LOOM_X86_VECTOR_SLICE_CONVERSION_NARROW;
}

IREE_ATTRIBUTE_NOINLINE static loom_x86_vector_slice_plan_t
loom_x86_vector_slice_select_predicate(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_x86_vector_slice_descriptors_t* descriptors,
    const loom_x86_vector_slice_shape_t* logical_shape) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_id_t source_value = loom_vector_slice_source(source_op);
  const loom_value_id_t result_value = loom_vector_slice_result(source_op);
  loom_low_representation_id_t source_representation =
      LOOM_LOW_REPRESENTATION_ID_NONE;
  loom_low_representation_id_t result_representation =
      LOOM_LOW_REPRESENTATION_ID_NONE;
  loom_low_lower_representation_lookup_if_ready(context, source_value,
                                                &source_representation);
  loom_low_lower_representation_lookup_if_ready(context, result_value,
                                                &result_representation);
  if (source_representation == LOOM_LOW_REPRESENTATION_ID_NONE) {
    source_representation = loom_x86_predicate_callable_representation(
        logical_shape->source_lane_count);
  }
  if (result_representation == LOOM_LOW_REPRESENTATION_ID_NONE) {
    result_representation = loom_x86_predicate_callable_representation(
        logical_shape->result_lane_count);
  }

  loom_x86_register_class_t source_register_class = 0;
  loom_x86_register_class_t result_register_class = 0;
  const bool source_mapped = loom_x86_vector_slice_predicate_register_class(
      descriptors, loom_module_value_type(module, source_value),
      source_representation, &source_register_class);
  const bool result_mapped = loom_x86_vector_slice_predicate_register_class(
      descriptors, loom_module_value_type(module, result_value),
      result_representation, &result_register_class);
  IREE_ASSERT(source_mapped && result_mapped);

  loom_x86_vector_slice_plan_t plan = {0};
  if (source_representation == LOOM_X86_PREDICATE_REPRESENTATION_MASK) {
    plan = (loom_x86_vector_slice_plan_t){
        .mechanism = logical_shape->source_lane_offset == 0
                         ? LOOM_X86_VECTOR_SLICE_PROJECT
                         : LOOM_X86_VECTOR_SLICE_SHIFT_MASK,
        .source_register_class = LOOM_X86_REGISTER_CLASS_K,
        .window_register_class = LOOM_X86_REGISTER_CLASS_K,
        .result_register_class = result_register_class,
        .predicate_lane_offset = logical_shape->source_lane_offset,
    };
  } else {
    const uint32_t source_lane_bit_width = (uint32_t)source_representation;
    const uint32_t source_bit_width =
        logical_shape->source_lane_count * source_lane_bit_width;
    const uint32_t window_bit_width =
        logical_shape->result_lane_count * source_lane_bit_width;
    loom_x86_register_class_t window_register_class = 0;
    const bool source_class_mapped =
        loom_x86_register_class_for_vector_bit_width(source_bit_width,
                                                     &source_register_class);
    const bool window_class_mapped =
        loom_x86_register_class_for_vector_bit_width(window_bit_width,
                                                     &window_register_class);
    IREE_ASSERT(source_class_mapped && window_class_mapped);
    const loom_x86_vector_slice_shape_t physical_shape = {
        .element_bit_width = (uint8_t)source_lane_bit_width,
        .source_register_class = source_register_class,
        .window_register_class = window_register_class,
        .result_register_class = result_register_class,
        .source_byte_offset = (uint8_t)(logical_shape->source_lane_offset *
                                        source_lane_bit_width / 8u),
        .result_byte_count = (uint8_t)(logical_shape->result_lane_count *
                                       source_lane_bit_width / 8u),
    };
    plan = loom_x86_vector_slice_select_mechanism(&physical_shape);
  }

  plan.conversion = loom_x86_vector_slice_select_conversion(
      source_representation, result_representation);
  plan.source_predicate_lane_bit_width =
      source_representation == LOOM_X86_PREDICATE_REPRESENTATION_MASK
          ? 1
          : (uint8_t)source_representation;
  plan.result_predicate_lane_bit_width =
      result_representation == LOOM_X86_PREDICATE_REPRESENTATION_MASK
          ? 1
          : (uint8_t)result_representation;

  const bool supports_avx512_vl = iree_any_bit_set(
      loom_low_lower_context_bundle(context)->config->contract_feature_bits,
      LOOM_X86_FEATURE_AVX512_VL);
  plan.use_full_width_mask_conversion =
      descriptors->supports_mask && !supports_avx512_vl &&
      ((plan.conversion == LOOM_X86_VECTOR_SLICE_CONVERSION_EXPAND_MASK &&
        result_register_class != LOOM_X86_REGISTER_CLASS_ZMM) ||
       (plan.conversion == LOOM_X86_VECTOR_SLICE_CONVERSION_COMPRESS_MASK &&
        plan.window_register_class != LOOM_X86_REGISTER_CLASS_ZMM));
  return plan;
}

iree_status_t loom_x86_query_vector_slice_contract(
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op,
    loom_target_contract_query_result_t* out_result) {
  *out_result = loom_target_contract_query_result_empty();
  if (!loom_vector_slice_isa(source_op) ||
      environment->vector_lane_projection.source_lane_count != 0) {
    return iree_ok_status();
  }
  loom_x86_vector_slice_shape_t shape;
  if (loom_x86_vector_slice_match_shape(
          environment->module, source_op,
          loom_x86_vector_slice_descriptors(environment->descriptor_set),
          &shape)) {
    out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  } else {
    // This provider owns native register windows. Rejected shapes require
    // shared packet legalization instead of passing as an unanswered query.
    out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED;
  }
  return iree_ok_status();
}

bool loom_x86_vector_slice_plan_isa(loom_low_lower_plan_t plan) {
  return plan.id == LOOM_X86_VECTOR_SLICE_PLAN;
}

iree_status_t loom_x86_select_vector_slice_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t* out_plan) {
  *out_plan = loom_low_lower_plan_empty();
  const loom_x86_vector_slice_descriptors_t* descriptors =
      loom_x86_vector_slice_descriptors(
          loom_low_lower_context_descriptor_set(context));
  loom_x86_vector_slice_shape_t shape;
  if (!loom_x86_vector_slice_match_shape(loom_low_lower_context_module(context),
                                         source_op, descriptors, &shape)) {
    return iree_ok_status();
  }

  const loom_x86_vector_slice_plan_t selected =
      shape.is_predicate ? loom_x86_vector_slice_select_predicate(
                               context, source_op, descriptors, &shape)
                         : loom_x86_vector_slice_select_mechanism(&shape);
  loom_x86_vector_slice_plan_t* retained_plan = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_plan_data(
      context, sizeof(*retained_plan), (void**)&retained_plan));
  *retained_plan = selected;
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, selected.result_register_class, 1, &result_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_plan_value_type(
      context, loom_vector_slice_result(source_op), result_type));
  *out_plan =
      loom_low_lower_plan_make(LOOM_X86_VECTOR_SLICE_PLAN, retained_plan);
  return iree_ok_status();
}

void loom_x86_mark_vector_slice_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan) {
  (void)plan;
  loom_low_lower_require_source_value_storage(
      context, loom_vector_slice_source(source_op));
}

void loom_x86_describe_vector_slice_plan(
    loom_low_lower_plan_t plan, loom_low_lower_plan_report_t* out_report) {
  const loom_x86_vector_slice_plan_t* slice = plan.target_data;
  const bool zero_instruction =
      slice->mechanism == LOOM_X86_VECTOR_SLICE_PROJECT &&
      slice->conversion == LOOM_X86_VECTOR_SLICE_CONVERSION_NONE;
  *out_report = (loom_low_lower_plan_report_t){
      .plan_key = zero_instruction ? IREE_SV("vector-slice.register-alias")
                  : slice->conversion != LOOM_X86_VECTOR_SLICE_CONVERSION_NONE
                      ? IREE_SV("vector-slice.predicate-conversion")
                      : IREE_SV("vector-slice.register-window"),
  };
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_x86_vector_slice_emitter_initialize(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_x86_vector_slice_emitter_t* out_emitter) {
  *out_emitter = (loom_x86_vector_slice_emitter_t){
      .context = context,
      .source_op = source_op,
      .descriptors = loom_x86_vector_slice_descriptors(
          loom_low_lower_context_descriptor_set(context)),
      .zmm_type = loom_type_none(),
      .mask_type = loom_type_none(),
  };
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, LOOM_X86_REGISTER_CLASS_XMM, 1, &out_emitter->xmm_type));
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, LOOM_X86_REGISTER_CLASS_YMM, 1, &out_emitter->ymm_type));
  if (out_emitter->descriptors->supports_mask) {
    IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
        context, LOOM_X86_REGISTER_CLASS_ZMM, 1, &out_emitter->zmm_type));
    IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
        context, LOOM_X86_REGISTER_CLASS_K, 1, &out_emitter->mask_type));
  }
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, IREE_SV("bytes"),
                                                  &out_emitter->bytes_name));
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, IREE_SV("lane"),
                                                  &out_emitter->lane_name));
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, IREE_SV("lanes"),
                                                  &out_emitter->lanes_name));
  IREE_RETURN_IF_ERROR(loom_builder_intern_string(builder, IREE_SV("shift"),
                                                  &out_emitter->shift_name));
  return loom_builder_intern_string(builder, IREE_SV("control"),
                                    &out_emitter->control_name);
}

static const loom_low_descriptor_t* loom_x86_vector_slice_descriptor(
    const loom_x86_vector_slice_emitter_t* emitter, uint32_t ordinal) {
  IREE_ASSERT_NE(ordinal, LOOM_LOW_DESCRIPTOR_ORDINAL_NONE);
  return &loom_low_lower_context_descriptor_set(emitter->context)
              ->descriptors[ordinal];
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t loom_x86_vector_slice_emit_op(
    loom_x86_vector_slice_emitter_t* emitter, uint32_t descriptor_ordinal,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, loom_type_t result_type,
    loom_value_id_t* out_result) {
  const loom_low_lower_resolved_descriptor_t descriptor = {
      .descriptor =
          loom_x86_vector_slice_descriptor(emitter, descriptor_ordinal),
  };
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      emitter->context, &descriptor, operands, operand_count, attrs,
      &result_type, 1, NULL, 0, emitter->source_op->location, &low_op));
  *out_result = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_x86_vector_slice_emit_descriptor(
    loom_x86_vector_slice_emitter_t* emitter, uint32_t descriptor_ordinal,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_string_id_t immediate_name, uint8_t immediate, loom_type_t result_type,
    loom_value_id_t* out_result) {
  const loom_named_attr_t attr = {
      .name_id = immediate_name,
      .value = loom_attr_i64(immediate),
  };
  return loom_x86_vector_slice_emit_op(
      emitter, descriptor_ordinal, operands, operand_count,
      loom_make_named_attr_slice(&attr, 1), result_type, out_result);
}

static iree_status_t loom_x86_vector_slice_emit_plain_descriptor(
    loom_x86_vector_slice_emitter_t* emitter, uint32_t descriptor_ordinal,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_type_t result_type, loom_value_id_t* out_result) {
  return loom_x86_vector_slice_emit_op(
      emitter, descriptor_ordinal, operands, operand_count,
      loom_named_attr_slice_empty(), result_type, out_result);
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t loom_x86_vector_slice_project(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source, loom_x86_register_class_t source_register_class,
    loom_x86_register_class_t result_register_class,
    loom_value_id_t* out_result) {
  if (source_register_class == result_register_class) {
    *out_result = source;
    return iree_ok_status();
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
      context, result_register_class, 1, &result_type));
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_slice_build(loom_low_lower_context_builder(context), source, 0,
                           result_type, source_op->location, &low_op));
  *out_result = loom_low_slice_result(low_op);
  return iree_ok_status();
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t loom_x86_vector_slice_extract_xmm(
    loom_x86_vector_slice_emitter_t* emitter, loom_value_id_t source,
    loom_x86_register_class_t source_register_class, uint8_t source_quarter,
    loom_value_id_t* out_result) {
  if (source_quarter == 0) {
    return loom_x86_vector_slice_project(
        emitter->context, emitter->source_op, source, source_register_class,
        LOOM_X86_REGISTER_CLASS_XMM, out_result);
  }
  const uint32_t descriptor_ordinal =
      source_register_class == LOOM_X86_REGISTER_CLASS_YMM
          ? emitter->descriptors->extract_xmm_from_ymm
          : emitter->descriptors->extract_xmm_from_zmm;
  return loom_x86_vector_slice_emit_descriptor(
      emitter, descriptor_ordinal, &source, 1, emitter->lane_name,
      source_quarter, emitter->xmm_type, out_result);
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_x86_vector_slice_emit_xmm_window(loom_x86_vector_slice_emitter_t* emitter,
                                      loom_value_id_t source,
                                      const loom_x86_vector_slice_plan_t* slice,
                                      bool crosses_quarter,
                                      loom_value_id_t* out_result) {
  loom_value_id_t lower = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_x86_vector_slice_extract_xmm(
      emitter, source, slice->source_register_class, slice->source_quarter,
      &lower));
  if (!crosses_quarter) {
    return loom_x86_vector_slice_emit_descriptor(
        emitter, emitter->descriptors->shift_xmm, &lower, 1,
        emitter->bytes_name, slice->quarter_byte_offset, emitter->xmm_type,
        out_result);
  }

  loom_value_id_t upper = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_x86_vector_slice_extract_xmm(
      emitter, source, slice->source_register_class, slice->source_quarter + 1,
      &upper));
  const loom_value_id_t operands[] = {upper, lower};
  return loom_x86_vector_slice_emit_descriptor(
      emitter, emitter->descriptors->align_xmm, operands,
      IREE_ARRAYSIZE(operands), emitter->bytes_name, slice->quarter_byte_offset,
      emitter->xmm_type, out_result);
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_x86_vector_slice_emit_assembled_ymm(
    loom_x86_vector_slice_emitter_t* emitter, loom_value_id_t source,
    const loom_x86_vector_slice_plan_t* slice, loom_value_id_t* out_result) {
  loom_value_id_t quarters[3] = {LOOM_VALUE_ID_INVALID};
  for (uint8_t i = 0; i < IREE_ARRAYSIZE(quarters); ++i) {
    IREE_RETURN_IF_ERROR(loom_x86_vector_slice_extract_xmm(
        emitter, source, LOOM_X86_REGISTER_CLASS_ZMM, slice->source_quarter + i,
        &quarters[i]));
  }

  loom_value_id_t windows[2] = {LOOM_VALUE_ID_INVALID};
  for (uint8_t i = 0; i < IREE_ARRAYSIZE(windows); ++i) {
    const loom_value_id_t operands[] = {quarters[i + 1], quarters[i]};
    IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_descriptor(
        emitter, emitter->descriptors->align_xmm, operands,
        IREE_ARRAYSIZE(operands), emitter->bytes_name,
        slice->quarter_byte_offset, emitter->xmm_type, &windows[i]));
  }

  loom_value_id_t assembled = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_x86_vector_slice_project(
      emitter->context, emitter->source_op, source, LOOM_X86_REGISTER_CLASS_ZMM,
      LOOM_X86_REGISTER_CLASS_YMM, &assembled));
  for (uint8_t lane = 0; lane < IREE_ARRAYSIZE(windows); ++lane) {
    const loom_value_id_t operands[] = {assembled, windows[lane]};
    IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_descriptor(
        emitter, emitter->descriptors->insert_xmm_into_ymm, operands,
        IREE_ARRAYSIZE(operands), emitter->lane_name, lane, emitter->ymm_type,
        &assembled));
  }
  *out_result = assembled;
  return iree_ok_status();
}

static loom_type_t loom_x86_vector_slice_type_for_class(
    const loom_x86_vector_slice_emitter_t* emitter,
    loom_x86_register_class_t register_class) {
  switch (register_class) {
    case LOOM_X86_REGISTER_CLASS_XMM:
      return emitter->xmm_type;
    case LOOM_X86_REGISTER_CLASS_YMM:
      return emitter->ymm_type;
    case LOOM_X86_REGISTER_CLASS_ZMM:
      return emitter->zmm_type;
    case LOOM_X86_REGISTER_CLASS_K:
      return emitter->mask_type;
    default:
      IREE_ASSERT_UNREACHABLE("selected x86 vector slice register class");
      return loom_type_none();
  }
}

static uint8_t loom_x86_vector_slice_width_index(uint8_t lane_bit_width) {
  switch (lane_bit_width) {
    case 8:
      return LOOM_X86_VECTOR_SLICE_WIDTH_I8;
    case 16:
      return LOOM_X86_VECTOR_SLICE_WIDTH_I16;
    case 32:
      return LOOM_X86_VECTOR_SLICE_WIDTH_I32;
    case 64:
      return LOOM_X86_VECTOR_SLICE_WIDTH_I64;
    default:
      IREE_ASSERT_UNREACHABLE("selected x86 predicate lane width");
      return LOOM_X86_VECTOR_SLICE_WIDTH_I8;
  }
}

static uint8_t loom_x86_vector_slice_class_index(
    loom_x86_register_class_t register_class) {
  switch (register_class) {
    case LOOM_X86_REGISTER_CLASS_XMM:
      return LOOM_X86_VECTOR_SLICE_CLASS_XMM;
    case LOOM_X86_REGISTER_CLASS_YMM:
      return LOOM_X86_VECTOR_SLICE_CLASS_YMM;
    case LOOM_X86_REGISTER_CLASS_ZMM:
      return LOOM_X86_VECTOR_SLICE_CLASS_ZMM;
    default:
      IREE_ASSERT_UNREACHABLE("selected x86 SIMD predicate carrier");
      return LOOM_X86_VECTOR_SLICE_CLASS_XMM;
  }
}

static uint8_t loom_x86_vector_slice_widen_index(uint8_t source_bit_width,
                                                 uint8_t result_bit_width) {
  if (source_bit_width == 8) {
    if (result_bit_width == 16) {
      return LOOM_X86_VECTOR_SLICE_WIDEN_I8_I16;
    }
    return result_bit_width == 32 ? LOOM_X86_VECTOR_SLICE_WIDEN_I8_I32
                                  : LOOM_X86_VECTOR_SLICE_WIDEN_I8_I64;
  }
  if (source_bit_width == 16) {
    return result_bit_width == 32 ? LOOM_X86_VECTOR_SLICE_WIDEN_I16_I32
                                  : LOOM_X86_VECTOR_SLICE_WIDEN_I16_I64;
  }
  IREE_ASSERT(source_bit_width == 32 && result_bit_width == 64);
  return LOOM_X86_VECTOR_SLICE_WIDEN_I32_I64;
}

static uint8_t loom_x86_vector_slice_pack_index(uint8_t source_bit_width) {
  switch (source_bit_width) {
    case 64:
      return 0;
    case 32:
      return 1;
    case 16:
      return 2;
    default:
      IREE_ASSERT_UNREACHABLE("selected x86 predicate narrowing width");
      return 0;
  }
}

static iree_status_t loom_x86_vector_slice_emit_pack(
    loom_x86_vector_slice_emitter_t* emitter, uint8_t source_bit_width,
    loom_value_id_t low, loom_value_id_t high, loom_value_id_t* out_result) {
  const uint8_t pack_index = loom_x86_vector_slice_pack_index(source_bit_width);
  const loom_value_id_t operands[] = {low, high};
  if (source_bit_width == 64) {
    return loom_x86_vector_slice_emit_descriptor(
        emitter, emitter->descriptors->simd_conversion.pack[pack_index],
        operands, IREE_ARRAYSIZE(operands), emitter->control_name, 0x88,
        emitter->xmm_type, out_result);
  }
  return loom_x86_vector_slice_emit_plain_descriptor(
      emitter, emitter->descriptors->simd_conversion.pack[pack_index], operands,
      IREE_ARRAYSIZE(operands), emitter->xmm_type, out_result);
}

static iree_status_t loom_x86_vector_slice_emit_widen(
    loom_x86_vector_slice_emitter_t* emitter,
    const loom_x86_vector_slice_plan_t* slice, loom_value_id_t source,
    loom_value_id_t* out_result) {
  IREE_ASSERT_EQ(slice->window_register_class, LOOM_X86_REGISTER_CLASS_XMM);
  IREE_ASSERT(slice->result_register_class == LOOM_X86_REGISTER_CLASS_XMM ||
              slice->result_register_class == LOOM_X86_REGISTER_CLASS_YMM);
  const uint8_t widen_index =
      loom_x86_vector_slice_widen_index(slice->source_predicate_lane_bit_width,
                                        slice->result_predicate_lane_bit_width);
  const uint8_t class_index =
      slice->result_register_class == LOOM_X86_REGISTER_CLASS_XMM ? 0 : 1;
  return loom_x86_vector_slice_emit_plain_descriptor(
      emitter,
      emitter->descriptors->simd_conversion.widen[widen_index][class_index],
      &source, 1,
      loom_x86_vector_slice_type_for_class(emitter,
                                           slice->result_register_class),
      out_result);
}

static iree_status_t loom_x86_vector_slice_emit_narrow(
    loom_x86_vector_slice_emitter_t* emitter,
    const loom_x86_vector_slice_plan_t* slice, loom_value_id_t source,
    loom_value_id_t* out_result) {
  IREE_ASSERT_EQ(slice->result_register_class, LOOM_X86_REGISTER_CLASS_XMM);
  IREE_ASSERT(slice->window_register_class == LOOM_X86_REGISTER_CLASS_XMM ||
              slice->window_register_class == LOOM_X86_REGISTER_CLASS_YMM);
  uint8_t current_bit_width = slice->source_predicate_lane_bit_width;
  loom_value_id_t current = source;
  if (slice->window_register_class == LOOM_X86_REGISTER_CLASS_YMM) {
    loom_value_id_t low = LOOM_VALUE_ID_INVALID;
    loom_value_id_t high = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_x86_vector_slice_project(
        emitter->context, emitter->source_op, source,
        LOOM_X86_REGISTER_CLASS_YMM, LOOM_X86_REGISTER_CLASS_XMM, &low));
    IREE_RETURN_IF_ERROR(loom_x86_vector_slice_extract_xmm(
        emitter, source, LOOM_X86_REGISTER_CLASS_YMM, 1, &high));
    IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_pack(
        emitter, current_bit_width, low, high, &current));
    current_bit_width /= 2;
  }
  while (current_bit_width > slice->result_predicate_lane_bit_width) {
    IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_pack(
        emitter, current_bit_width, current, current, &current));
    current_bit_width /= 2;
  }
  *out_result = current;
  return iree_ok_status();
}

static iree_status_t loom_x86_vector_slice_emit_expand_mask(
    loom_x86_vector_slice_emitter_t* emitter,
    const loom_x86_vector_slice_plan_t* slice, loom_value_id_t source,
    loom_value_id_t* out_result) {
  const uint8_t width_index =
      loom_x86_vector_slice_width_index(slice->result_predicate_lane_bit_width);
  const loom_x86_register_class_t expansion_class =
      slice->use_full_width_mask_conversion ? LOOM_X86_REGISTER_CLASS_ZMM
                                            : slice->result_register_class;
  loom_value_id_t expanded = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_plain_descriptor(
      emitter,
      emitter->descriptors->mask_conversion
          .expand[width_index]
                 [loom_x86_vector_slice_class_index(expansion_class)],
      &source, 1,
      loom_x86_vector_slice_type_for_class(emitter, expansion_class),
      &expanded));
  if (!slice->use_full_width_mask_conversion) {
    *out_result = expanded;
    return iree_ok_status();
  }
  return loom_x86_vector_slice_project(
      emitter->context, emitter->source_op, expanded,
      LOOM_X86_REGISTER_CLASS_ZMM, slice->result_register_class, out_result);
}

static iree_status_t loom_x86_vector_slice_emit_compress_mask(
    loom_x86_vector_slice_emitter_t* emitter,
    const loom_x86_vector_slice_plan_t* slice, loom_value_id_t source,
    loom_value_id_t* out_result) {
  loom_x86_register_class_t compression_class = slice->window_register_class;
  loom_value_id_t carrier = source;
  if (slice->use_full_width_mask_conversion) {
    loom_value_id_t zero = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_plain_descriptor(
        emitter, emitter->descriptors->mask_conversion.zero_zmm,
        /*operands=*/NULL, /*operand_count=*/0, emitter->zmm_type, &zero));
    const uint32_t insert_descriptor =
        compression_class == LOOM_X86_REGISTER_CLASS_XMM
            ? emitter->descriptors->mask_conversion.insert_xmm_into_zmm
            : emitter->descriptors->mask_conversion.insert_ymm_into_zmm;
    const loom_value_id_t operands[] = {zero, source};
    IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_descriptor(
        emitter, insert_descriptor, operands, IREE_ARRAYSIZE(operands),
        emitter->lane_name, 0, emitter->zmm_type, &carrier));
    compression_class = LOOM_X86_REGISTER_CLASS_ZMM;
  }
  const uint8_t width_index =
      loom_x86_vector_slice_width_index(slice->source_predicate_lane_bit_width);
  return loom_x86_vector_slice_emit_plain_descriptor(
      emitter,
      emitter->descriptors->mask_conversion
          .compress[width_index]
                   [loom_x86_vector_slice_class_index(compression_class)],
      &carrier, 1, emitter->mask_type, out_result);
}

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_x86_vector_slice_emit_conversion(loom_x86_vector_slice_emitter_t* emitter,
                                      const loom_x86_vector_slice_plan_t* slice,
                                      loom_value_id_t source,
                                      loom_value_id_t* out_result) {
  switch (slice->conversion) {
    case LOOM_X86_VECTOR_SLICE_CONVERSION_NONE:
      *out_result = source;
      return iree_ok_status();
    case LOOM_X86_VECTOR_SLICE_CONVERSION_WIDEN:
      return loom_x86_vector_slice_emit_widen(emitter, slice, source,
                                              out_result);
    case LOOM_X86_VECTOR_SLICE_CONVERSION_NARROW:
      return loom_x86_vector_slice_emit_narrow(emitter, slice, source,
                                               out_result);
    case LOOM_X86_VECTOR_SLICE_CONVERSION_EXPAND_MASK:
      return loom_x86_vector_slice_emit_expand_mask(emitter, slice, source,
                                                    out_result);
    case LOOM_X86_VECTOR_SLICE_CONVERSION_COMPRESS_MASK:
      return loom_x86_vector_slice_emit_compress_mask(emitter, slice, source,
                                                      out_result);
  }
  IREE_ASSERT_UNREACHABLE("unknown x86 predicate slice conversion");
  return iree_ok_status();
}

iree_status_t loom_x86_emit_vector_slice_plan(loom_low_lower_context_t* context,
                                              const loom_op_t* source_op,
                                              loom_low_lower_plan_t plan) {
  const loom_x86_vector_slice_plan_t* slice = plan.target_data;
  const loom_value_id_t source =
      loom_low_lower_lookup_value(context, loom_vector_slice_source(source_op));
  if (slice->mechanism == LOOM_X86_VECTOR_SLICE_PROJECT &&
      slice->conversion == LOOM_X86_VECTOR_SLICE_CONVERSION_NONE) {
    loom_value_id_t result = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_x86_vector_slice_project(
        context, source_op, source, slice->source_register_class,
        slice->window_register_class, &result));
    return loom_low_lower_bind_value(
        context, loom_vector_slice_result(source_op), result);
  }

  loom_x86_vector_slice_emitter_t emitter;
  IREE_RETURN_IF_ERROR(
      loom_x86_vector_slice_emitter_initialize(context, source_op, &emitter));
  loom_value_id_t window = LOOM_VALUE_ID_INVALID;
  switch (slice->mechanism) {
    case LOOM_X86_VECTOR_SLICE_PROJECT: {
      IREE_RETURN_IF_ERROR(loom_x86_vector_slice_project(
          context, source_op, source, slice->source_register_class,
          slice->window_register_class, &window));
      break;
    }
    case LOOM_X86_VECTOR_SLICE_SHIFT_XMM: {
      IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_xmm_window(
          &emitter, source, slice, /*crosses_quarter=*/false, &window));
      break;
    }
    case LOOM_X86_VECTOR_SLICE_ALIGN_XMM: {
      IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_xmm_window(
          &emitter, source, slice, /*crosses_quarter=*/true, &window));
      break;
    }
    case LOOM_X86_VECTOR_SLICE_EXTRACT_XMM: {
      IREE_RETURN_IF_ERROR(loom_x86_vector_slice_extract_xmm(
          &emitter, source, slice->source_register_class, slice->source_quarter,
          &window));
      break;
    }
    case LOOM_X86_VECTOR_SLICE_ALIGN_DWORD_ZMM:
    case LOOM_X86_VECTOR_SLICE_ALIGN_QWORD_ZMM: {
      const uint32_t descriptor_ordinal =
          slice->mechanism == LOOM_X86_VECTOR_SLICE_ALIGN_DWORD_ZMM
              ? emitter.descriptors->align_dword_zmm
              : emitter.descriptors->align_qword_zmm;
      const loom_value_id_t operands[] = {source, source};
      loom_value_id_t aligned = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_descriptor(
          &emitter, descriptor_ordinal, operands, IREE_ARRAYSIZE(operands),
          emitter.lanes_name, slice->alignment_lane_offset, emitter.zmm_type,
          &aligned));
      IREE_RETURN_IF_ERROR(loom_x86_vector_slice_project(
          context, source_op, aligned, LOOM_X86_REGISTER_CLASS_ZMM,
          LOOM_X86_REGISTER_CLASS_YMM, &window));
      break;
    }
    case LOOM_X86_VECTOR_SLICE_EXTRACT_YMM: {
      IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_descriptor(
          &emitter, emitter.descriptors->extract_ymm_from_zmm, &source, 1,
          emitter.lane_name, 1, emitter.ymm_type, &window));
      break;
    }
    case LOOM_X86_VECTOR_SLICE_ASSEMBLE_YMM: {
      IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_assembled_ymm(
          &emitter, source, slice, &window));
      break;
    }
    case LOOM_X86_VECTOR_SLICE_SHIFT_MASK: {
      IREE_RETURN_IF_ERROR(loom_x86_vector_slice_emit_descriptor(
          &emitter, emitter.descriptors->mask_conversion.shift, &source, 1,
          emitter.shift_name, slice->predicate_lane_offset, emitter.mask_type,
          &window));
      break;
    }
  }
  loom_value_id_t result = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_x86_vector_slice_emit_conversion(&emitter, slice, window, &result));
  return loom_low_lower_bind_value(context, loom_vector_slice_result(source_op),
                                   result);
}
