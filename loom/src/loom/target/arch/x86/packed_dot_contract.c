// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/packed_dot_contract.h"

#include "loom/target/arch/x86/packed_dot_contract_data.h"

#define AVX512_VNNI_FEATURES_128_256 \
  (LOOM_X86_FEATURE_AVX512_VNNI | LOOM_X86_FEATURE_AVX512_VL)
#define AVX512_BF16_FEATURES_128_256 \
  (LOOM_X86_FEATURE_AVX512_BF16 | LOOM_X86_FEATURE_AVX512_VL)

iree_string_view_t loom_x86_packed_dot_family_name(
    loom_x86_packed_dot_family_t family) {
  switch (family) {
    case LOOM_X86_PACKED_DOT_FAMILY_AVX512_VNNI:
      return IREE_SV("x86-avx512-vnni");
    case LOOM_X86_PACKED_DOT_FAMILY_AVX_VNNI:
      return IREE_SV("x86-avx-vnni");
    case LOOM_X86_PACKED_DOT_FAMILY_AVX_VNNI_INT8:
      return IREE_SV("x86-avx-vnni-int8");
    case LOOM_X86_PACKED_DOT_FAMILY_AVX_VNNI_INT16:
      return IREE_SV("x86-avx-vnni-int16");
    case LOOM_X86_PACKED_DOT_FAMILY_AVX10_2:
      return IREE_SV("x86-avx10.2");
    case LOOM_X86_PACKED_DOT_FAMILY_AVX512_BF16:
      return IREE_SV("x86-avx512-bf16");
    case LOOM_X86_PACKED_DOT_FAMILY_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_x86_packed_dot_numeric_type_name(
    loom_x86_packed_dot_numeric_type_t numeric_type) {
  switch (numeric_type) {
    case LOOM_X86_PACKED_DOT_NUMERIC_I8:
      return IREE_SV("i8");
    case LOOM_X86_PACKED_DOT_NUMERIC_U8:
      return IREE_SV("u8");
    case LOOM_X86_PACKED_DOT_NUMERIC_I16:
      return IREE_SV("i16");
    case LOOM_X86_PACKED_DOT_NUMERIC_U16:
      return IREE_SV("u16");
    case LOOM_X86_PACKED_DOT_NUMERIC_F16:
      return IREE_SV("f16");
    case LOOM_X86_PACKED_DOT_NUMERIC_BF16:
      return IREE_SV("bf16");
    case LOOM_X86_PACKED_DOT_NUMERIC_I32:
      return IREE_SV("i32");
    case LOOM_X86_PACKED_DOT_NUMERIC_F32:
      return IREE_SV("f32");
    case LOOM_X86_PACKED_DOT_NUMERIC_UNKNOWN:
    default:
      return IREE_SV("unknown");
  }
}

iree_status_t loom_x86_packed_dot_feature_bits_for_name(
    iree_string_view_t name,
    loom_x86_packed_dot_feature_bits_t* out_feature_bits) {
  if (out_feature_bits == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_feature_bits must not be NULL");
  }
  *out_feature_bits = 0;
  if (iree_string_view_equal(name, IREE_SV("x86-avx512-vnni"))) {
    *out_feature_bits = LOOM_X86_FEATURE_AVX512_VNNI;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("x86-avx512-vnni-vl"))) {
    *out_feature_bits = AVX512_VNNI_FEATURES_128_256;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("x86-avx-vnni"))) {
    *out_feature_bits = LOOM_X86_FEATURE_AVX_VNNI;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("x86-avx-vnni-int8"))) {
    *out_feature_bits = LOOM_X86_FEATURE_AVX_VNNI_INT8;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("x86-avx-vnni-int16"))) {
    *out_feature_bits = LOOM_X86_FEATURE_AVX_VNNI_INT16;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("x86-avx10.2"))) {
    *out_feature_bits = LOOM_X86_FEATURE_AVX10_2;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("x86-avx512-bf16"))) {
    *out_feature_bits = LOOM_X86_FEATURE_AVX512_BF16;
    return iree_ok_status();
  }
  if (iree_string_view_equal(name, IREE_SV("x86-avx512-bf16-vl"))) {
    *out_feature_bits = AVX512_BF16_FEATURES_128_256;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unknown x86 packed-dot feature set '%.*s'",
                          (int)name.size, name.data);
}

iree_host_size_t loom_x86_packed_dot_descriptor_count(void) {
  return loom_x86_packed_dot_builtin_descriptor_count;
}

const loom_x86_packed_dot_descriptor_t* loom_x86_packed_dot_descriptor_at(
    iree_host_size_t index) {
  if (index >= loom_x86_packed_dot_builtin_descriptor_count) return NULL;
  return &loom_x86_packed_dot_builtin_descriptors[index];
}

bool loom_x86_packed_dot_is_available(
    const loom_x86_packed_dot_descriptor_t* descriptor,
    loom_x86_packed_dot_feature_bits_t feature_bits) {
  if (descriptor == NULL) {
    return false;
  }
  return (feature_bits & descriptor->required_feature_bits) ==
         descriptor->required_feature_bits;
}

static bool loom_x86_packed_dot_family_matches(
    const loom_x86_packed_dot_descriptor_t* descriptor,
    const loom_x86_packed_dot_match_request_t* request) {
  return request->family == LOOM_X86_PACKED_DOT_FAMILY_UNKNOWN ||
         descriptor->family == request->family;
}

static bool loom_x86_packed_dot_shape_matches(
    const loom_x86_packed_dot_descriptor_t* descriptor,
    const loom_x86_packed_dot_match_request_t* request) {
  return descriptor->shape.vector_bit_width ==
             request->shape.vector_bit_width &&
         descriptor->shape.input_lane_count ==
             request->shape.input_lane_count &&
         descriptor->shape.result_lane_count ==
             request->shape.result_lane_count &&
         descriptor->shape.reduction_group_size ==
             request->shape.reduction_group_size;
}

static loom_x86_packed_dot_rejection_bits_t
loom_x86_packed_dot_payload_rejection_bits(
    const loom_x86_packed_dot_descriptor_t* descriptor,
    const loom_x86_packed_dot_match_request_t* request) {
  loom_x86_packed_dot_rejection_bits_t rejection_bits =
      LOOM_X86_PACKED_DOT_REJECTION_NONE;
  if (request->lhs_numeric_type == LOOM_X86_PACKED_DOT_NUMERIC_UNKNOWN ||
      descriptor->lhs_numeric_type != request->lhs_numeric_type) {
    rejection_bits |= LOOM_X86_PACKED_DOT_REJECTION_PAYLOAD;
  }
  if (request->rhs_numeric_type == LOOM_X86_PACKED_DOT_NUMERIC_UNKNOWN ||
      descriptor->rhs_numeric_type != request->rhs_numeric_type) {
    rejection_bits |= LOOM_X86_PACKED_DOT_REJECTION_PAYLOAD;
  }
  if (request->accumulator_numeric_type ==
          LOOM_X86_PACKED_DOT_NUMERIC_UNKNOWN ||
      descriptor->accumulator_numeric_type !=
          request->accumulator_numeric_type) {
    rejection_bits |= LOOM_X86_PACKED_DOT_REJECTION_PAYLOAD;
  }
  if (request->result_numeric_type == LOOM_X86_PACKED_DOT_NUMERIC_UNKNOWN ||
      descriptor->result_numeric_type != request->result_numeric_type) {
    rejection_bits |= LOOM_X86_PACKED_DOT_REJECTION_PAYLOAD;
  }
  return rejection_bits;
}

static bool loom_x86_packed_dot_flags_match(
    const loom_x86_packed_dot_descriptor_t* descriptor,
    const loom_x86_packed_dot_match_request_t* request) {
  return (request->required_flags & ~descriptor->flags) == 0;
}

const loom_x86_packed_dot_descriptor_t* loom_x86_packed_dot_select(
    const loom_x86_packed_dot_match_request_t* request,
    loom_x86_packed_dot_match_diagnostic_t* out_diagnostic) {
  loom_x86_packed_dot_match_diagnostic_t diagnostic = {
      .descriptor_count = loom_x86_packed_dot_builtin_descriptor_count,
  };
  if (request == NULL) {
    diagnostic.rejection_bits = LOOM_X86_PACKED_DOT_REJECTION_INVALID_REQUEST;
    if (out_diagnostic != NULL) *out_diagnostic = diagnostic;
    return NULL;
  }

  for (iree_host_size_t i = 0; i < loom_x86_packed_dot_builtin_descriptor_count;
       ++i) {
    const loom_x86_packed_dot_descriptor_t* descriptor =
        &loom_x86_packed_dot_builtin_descriptors[i];
    if (!loom_x86_packed_dot_family_matches(descriptor, request)) {
      continue;
    }
    ++diagnostic.family_candidate_count;

    if (!loom_x86_packed_dot_shape_matches(descriptor, request)) {
      continue;
    }
    ++diagnostic.shape_candidate_count;

    if (loom_x86_packed_dot_payload_rejection_bits(descriptor, request) !=
        LOOM_X86_PACKED_DOT_REJECTION_NONE) {
      continue;
    }
    ++diagnostic.payload_candidate_count;

    if (!loom_x86_packed_dot_flags_match(descriptor, request)) {
      continue;
    }
    ++diagnostic.flag_candidate_count;

    if (!loom_x86_packed_dot_is_available(descriptor, request->feature_bits)) {
      continue;
    }
    ++diagnostic.feature_candidate_count;

    if (out_diagnostic != NULL) *out_diagnostic = diagnostic;
    return descriptor;
  }

  if (diagnostic.family_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_X86_PACKED_DOT_REJECTION_FAMILY;
  } else if (diagnostic.shape_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_X86_PACKED_DOT_REJECTION_SHAPE;
  } else if (diagnostic.payload_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_X86_PACKED_DOT_REJECTION_PAYLOAD;
  } else if (diagnostic.flag_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_X86_PACKED_DOT_REJECTION_FLAGS;
  } else if (diagnostic.feature_candidate_count == 0) {
    diagnostic.rejection_bits = LOOM_X86_PACKED_DOT_REJECTION_FEATURES;
  }
  if (out_diagnostic != NULL) *out_diagnostic = diagnostic;
  return NULL;
}

#undef AVX512_VNNI_FEATURES_128_256
#undef AVX512_BF16_FEATURES_128_256
