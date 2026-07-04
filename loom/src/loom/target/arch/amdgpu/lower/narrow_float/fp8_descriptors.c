// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>

#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/narrow_float/fp8.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

enum {
  LOOM_AMDGPU_FP8_DESCRIPTOR_SOURCE_TYPE_COUNT =
      LOOM_SCALAR_TYPE_F8E5M2 - LOOM_SCALAR_TYPE_F8E4M3 + 1,
  LOOM_AMDGPU_FP8_DESCRIPTOR_RESULT_TYPE_COUNT =
      LOOM_SCALAR_TYPE_F32 - LOOM_SCALAR_TYPE_F16 + 1,
};

static_assert(LOOM_SCALAR_TYPE_F8E5M2 == LOOM_SCALAR_TYPE_F8E4M3 + 1,
              "FP8 descriptor lookup assumes dense FP8/BF8 scalar types");
static_assert(LOOM_SCALAR_TYPE_BF16 == LOOM_SCALAR_TYPE_F16 + 1 &&
                  LOOM_SCALAR_TYPE_F32 == LOOM_SCALAR_TYPE_F16 + 2,
              "FP8 descriptor lookup assumes dense 16/32-bit float results");

static bool loom_amdgpu_fp8_descriptor_source_type_index(
    loom_scalar_type_t source_element_type, iree_host_size_t* out_index) {
  if (source_element_type < LOOM_SCALAR_TYPE_F8E4M3 ||
      source_element_type > LOOM_SCALAR_TYPE_F8E5M2) {
    return false;
  }
  *out_index = source_element_type - LOOM_SCALAR_TYPE_F8E4M3;
  return true;
}

static bool loom_amdgpu_fp8_descriptor_result_type_index(
    loom_scalar_type_t result_element_type, iree_host_size_t* out_index) {
  if (result_element_type < LOOM_SCALAR_TYPE_F16 ||
      result_element_type > LOOM_SCALAR_TYPE_F32) {
    return false;
  }
  *out_index = result_element_type - LOOM_SCALAR_TYPE_F16;
  return true;
}

typedef struct loom_amdgpu_fp8_native_descriptor_ref_row_t {
  // Encoded FP8/BF8 source scalar type.
  loom_scalar_type_t source_element_type;
  // Decoded result scalar type.
  loom_scalar_type_t result_element_type;
  // Native unscaled descriptor refs for the type pair.
  loom_amdgpu_fp8_native_descriptor_refs_t refs;
} loom_amdgpu_fp8_native_descriptor_ref_row_t;

static const loom_amdgpu_fp8_native_descriptor_ref_row_t
    kLoomAmdgpuFp8NativeDescriptorRefRows[] = {
#define LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_REF_ROW(           \
    row_index, source_type, result_type, lane_ref, pair_ref) \
  [row_index] = {source_type, result_type, {lane_ref, pair_ref}}
#include "loom/target/arch/amdgpu/lower/narrow_float/fp8_native_descriptor_ref_rows.inl"
#undef LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_REF_ROW
};
static_assert(IREE_ARRAYSIZE(kLoomAmdgpuFp8NativeDescriptorRefRows) <= 64,
              "native descriptor cache stores row state in one u64 bitset");

static const uint8_t kLoomAmdgpuFp8NativeDescriptorRefRowIndex
    [LOOM_AMDGPU_FP8_DESCRIPTOR_SOURCE_TYPE_COUNT]
    [LOOM_AMDGPU_FP8_DESCRIPTOR_RESULT_TYPE_COUNT] = {
#define LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_REF_ROW(                   \
    row_index, source_type, result_type, lane_ref, pair_ref)         \
  [source_type -                                                     \
      LOOM_SCALAR_TYPE_F8E4M3][result_type - LOOM_SCALAR_TYPE_F16] = \
      (row_index) + 1
#include "loom/target/arch/amdgpu/lower/narrow_float/fp8_native_descriptor_ref_rows.inl"
#undef LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_REF_ROW
};

static bool loom_amdgpu_fp8_native_descriptor_ref_row(
    loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type, size_t* out_row_index,
    loom_amdgpu_fp8_native_descriptor_refs_t* out_refs) {
  *out_row_index = 0;
  *out_refs = (loom_amdgpu_fp8_native_descriptor_refs_t){
      .lane = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      .pair = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
  };
  iree_host_size_t source_index = 0;
  iree_host_size_t result_index = 0;
  if (!loom_amdgpu_fp8_descriptor_source_type_index(source_element_type,
                                                    &source_index) ||
      !loom_amdgpu_fp8_descriptor_result_type_index(result_element_type,
                                                    &result_index)) {
    return false;
  }
  const uint8_t encoded_row_index =
      kLoomAmdgpuFp8NativeDescriptorRefRowIndex[source_index][result_index];
  if (encoded_row_index == 0) {
    return false;
  }
  *out_row_index = encoded_row_index - 1;
  IREE_ASSERT_LT(*out_row_index,
                 IREE_ARRAYSIZE(kLoomAmdgpuFp8NativeDescriptorRefRows));
  const loom_amdgpu_fp8_native_descriptor_ref_row_t* row =
      &kLoomAmdgpuFp8NativeDescriptorRefRows[*out_row_index];
  IREE_ASSERT_EQ(row->source_element_type, source_element_type);
  IREE_ASSERT_EQ(row->result_element_type, result_element_type);
  *out_refs = row->refs;
  return true;
}

bool loom_amdgpu_fp8_native_descriptor_refs(
    loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type,
    loom_amdgpu_fp8_native_descriptor_refs_t* out_refs) {
  size_t row_index = 0;
  return loom_amdgpu_fp8_native_descriptor_ref_row(
      source_element_type, result_element_type, &row_index, out_refs);
}

typedef struct loom_amdgpu_fp8_native_descriptor_cache_t {
  // Generated-row bits whose descriptor availability has been resolved.
  uint64_t initialized_row_bits;
  // Function-local resolved descriptors keyed by generated row index.
  loom_amdgpu_fp8_native_descriptors_t
      descriptors[IREE_ARRAYSIZE(kLoomAmdgpuFp8NativeDescriptorRefRows)];
} loom_amdgpu_fp8_native_descriptor_cache_t;

static int loom_amdgpu_fp8_native_descriptor_cache_state_key;

iree_status_t loom_amdgpu_get_fp8_native_descriptors(
    loom_low_lower_context_t* context, loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type,
    const loom_amdgpu_fp8_native_descriptors_t** out_descriptors) {
  *out_descriptors = NULL;
  size_t row_index = 0;
  loom_amdgpu_fp8_native_descriptor_refs_t descriptor_refs = {0};
  if (!loom_amdgpu_fp8_native_descriptor_ref_row(
          source_element_type, result_element_type, &row_index,
          &descriptor_refs)) {
    return iree_ok_status();
  }
  loom_amdgpu_fp8_native_descriptor_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_fp8_native_descriptor_cache_state_key,
      sizeof(*cache), (void**)&cache));
  const uint64_t row_bit = UINT64_C(1) << row_index;
  if ((cache->initialized_row_bits & row_bit) == 0) {
    loom_amdgpu_fp8_native_descriptors_t* descriptors =
        &cache->descriptors[row_index];
    if (descriptor_refs.lane != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
      bool descriptor_is_present = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
          context, descriptor_refs.lane, &descriptors->lane_descriptor,
          &descriptor_is_present));
      if (descriptor_is_present) {
        descriptors->flags |= LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_LANE;
      }
    }
    if (descriptor_refs.pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
      bool descriptor_is_present = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
          context, descriptor_refs.pair, &descriptors->pair_descriptor,
          &descriptor_is_present));
      if (descriptor_is_present) {
        descriptors->flags |= LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR;
      }
    }
    cache->initialized_row_bits |= row_bit;
  }
  *out_descriptors = &cache->descriptors[row_index];
  return iree_ok_status();
}

typedef struct loom_amdgpu_fp8_scaled_descriptor_ref_row_t {
  // Encoded FP8/BF8 source scalar type.
  loom_scalar_type_t source_element_type;
  // Decoded result scalar type.
  loom_scalar_type_t result_element_type;
  // Descriptor ref for the native scale-f32 pair packet.
  loom_amdgpu_descriptor_ref_t scalef32_pair_descriptor_ref;
  // Descriptor ref for the native E8M0 scale-pk8 packet.
  loom_amdgpu_descriptor_ref_t e8m0_pk8_descriptor_ref;
} loom_amdgpu_fp8_scaled_descriptor_ref_row_t;

static const loom_amdgpu_fp8_scaled_descriptor_ref_row_t
    kLoomAmdgpuFp8ScaledDescriptorRefRows[] = {
#define LOOM_AMDGPU_FP8_SCALED_DESCRIPTOR_REF_ROW(                        \
    row_index, source_type, result_type, scalef32_pair_ref, e8m0_pk8_ref) \
  [row_index] = {source_type, result_type, scalef32_pair_ref, e8m0_pk8_ref}
#include "loom/target/arch/amdgpu/lower/narrow_float/fp8_scaled_descriptor_ref_rows.inl"
#undef LOOM_AMDGPU_FP8_SCALED_DESCRIPTOR_REF_ROW
};
static_assert(IREE_ARRAYSIZE(kLoomAmdgpuFp8ScaledDescriptorRefRows) <= 64,
              "scaled descriptor cache stores row state in one u64 bitset");

static const uint8_t kLoomAmdgpuFp8ScaledDescriptorRefRowIndex
    [LOOM_AMDGPU_FP8_DESCRIPTOR_SOURCE_TYPE_COUNT]
    [LOOM_AMDGPU_FP8_DESCRIPTOR_RESULT_TYPE_COUNT] = {
#define LOOM_AMDGPU_FP8_SCALED_DESCRIPTOR_REF_ROW(                        \
    row_index, source_type, result_type, scalef32_pair_ref, e8m0_pk8_ref) \
  [source_type -                                                          \
      LOOM_SCALAR_TYPE_F8E4M3][result_type - LOOM_SCALAR_TYPE_F16] =      \
      (row_index) + 1
#include "loom/target/arch/amdgpu/lower/narrow_float/fp8_scaled_descriptor_ref_rows.inl"
#undef LOOM_AMDGPU_FP8_SCALED_DESCRIPTOR_REF_ROW
};

static bool loom_amdgpu_fp8_scaled_descriptor_ref_row(
    loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type, size_t* out_row_index,
    const loom_amdgpu_fp8_scaled_descriptor_ref_row_t** out_row) {
  *out_row_index = 0;
  *out_row = NULL;
  iree_host_size_t source_index = 0;
  iree_host_size_t result_index = 0;
  if (!loom_amdgpu_fp8_descriptor_source_type_index(source_element_type,
                                                    &source_index) ||
      !loom_amdgpu_fp8_descriptor_result_type_index(result_element_type,
                                                    &result_index)) {
    return false;
  }
  const uint8_t encoded_row_index =
      kLoomAmdgpuFp8ScaledDescriptorRefRowIndex[source_index][result_index];
  if (encoded_row_index == 0) {
    return false;
  }
  *out_row_index = encoded_row_index - 1;
  IREE_ASSERT_LT(*out_row_index,
                 IREE_ARRAYSIZE(kLoomAmdgpuFp8ScaledDescriptorRefRows));
  *out_row = &kLoomAmdgpuFp8ScaledDescriptorRefRows[*out_row_index];
  IREE_ASSERT_EQ((*out_row)->source_element_type, source_element_type);
  IREE_ASSERT_EQ((*out_row)->result_element_type, result_element_type);
  return true;
}

bool loom_amdgpu_fp8_scalef32_descriptor_ref(
    loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type,
    loom_amdgpu_descriptor_ref_t* out_ref) {
  *out_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  size_t row_index = 0;
  const loom_amdgpu_fp8_scaled_descriptor_ref_row_t* row = NULL;
  if (!loom_amdgpu_fp8_scaled_descriptor_ref_row(
          source_element_type, result_element_type, &row_index, &row)) {
    return false;
  }
  *out_ref = row->scalef32_pair_descriptor_ref;
  return true;
}

bool loom_amdgpu_fp8_e8m0_pk8_descriptor_ref(
    loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type,
    loom_amdgpu_descriptor_ref_t* out_ref) {
  *out_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  size_t row_index = 0;
  const loom_amdgpu_fp8_scaled_descriptor_ref_row_t* row = NULL;
  if (!loom_amdgpu_fp8_scaled_descriptor_ref_row(
          source_element_type, result_element_type, &row_index, &row)) {
    return false;
  }
  *out_ref = row->e8m0_pk8_descriptor_ref;
  return true;
}

typedef struct loom_amdgpu_fp8_scaled_descriptor_cache_t {
  // Generated-row bits whose scale-f32 descriptor availability is resolved.
  uint64_t initialized_scalef32_pair_row_bits;
  // Generated-row bits with scale-f32 descriptors in the active descriptor set.
  uint64_t present_scalef32_pair_row_bits;
  // Generated-row bits whose E8M0 pk8 descriptor availability is resolved.
  uint64_t initialized_e8m0_pk8_row_bits;
  // Generated-row bits with E8M0 pk8 descriptors in the active descriptor set.
  uint64_t present_e8m0_pk8_row_bits;
  // Function-local resolved descriptors keyed by generated row index.
  loom_low_lower_resolved_descriptor_t scalef32_pair_descriptors[IREE_ARRAYSIZE(
      kLoomAmdgpuFp8ScaledDescriptorRefRows)];
  // Function-local resolved E8M0 pk8 descriptors keyed by generated row index.
  loom_low_lower_resolved_descriptor_t e8m0_pk8_descriptors[IREE_ARRAYSIZE(
      kLoomAmdgpuFp8ScaledDescriptorRefRows)];
} loom_amdgpu_fp8_scaled_descriptor_cache_t;

static int loom_amdgpu_fp8_scaled_descriptor_cache_state_key;

static iree_status_t loom_amdgpu_get_fp8_scaled_descriptor(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint64_t* initialized_row_bits,
    uint64_t* present_row_bits,
    loom_low_lower_resolved_descriptor_t* descriptors, size_t row_index,
    const loom_low_lower_resolved_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  if (descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    return iree_ok_status();
  }
  const uint64_t row_bit = UINT64_C(1) << row_index;
  if ((*initialized_row_bits & row_bit) == 0) {
    bool descriptor_is_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, descriptor_ref, &descriptors[row_index],
        &descriptor_is_present));
    if (descriptor_is_present) {
      *present_row_bits |= row_bit;
    }
    *initialized_row_bits |= row_bit;
  }
  if ((*present_row_bits & row_bit) != 0) {
    *out_descriptor = &descriptors[row_index];
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_get_fp8_scalef32_descriptor(
    loom_low_lower_context_t* context, loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type,
    const loom_low_lower_resolved_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  size_t row_index = 0;
  const loom_amdgpu_fp8_scaled_descriptor_ref_row_t* row = NULL;
  if (!loom_amdgpu_fp8_scaled_descriptor_ref_row(
          source_element_type, result_element_type, &row_index, &row)) {
    return iree_ok_status();
  }
  loom_amdgpu_fp8_scaled_descriptor_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_fp8_scaled_descriptor_cache_state_key,
      sizeof(*cache), (void**)&cache));
  return loom_amdgpu_get_fp8_scaled_descriptor(
      context, row->scalef32_pair_descriptor_ref,
      &cache->initialized_scalef32_pair_row_bits,
      &cache->present_scalef32_pair_row_bits, cache->scalef32_pair_descriptors,
      row_index, out_descriptor);
}

iree_status_t loom_amdgpu_get_fp8_e8m0_pk8_descriptor(
    loom_low_lower_context_t* context, loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type,
    const loom_low_lower_resolved_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  size_t row_index = 0;
  const loom_amdgpu_fp8_scaled_descriptor_ref_row_t* row = NULL;
  if (!loom_amdgpu_fp8_scaled_descriptor_ref_row(
          source_element_type, result_element_type, &row_index, &row)) {
    return iree_ok_status();
  }
  loom_amdgpu_fp8_scaled_descriptor_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_fp8_scaled_descriptor_cache_state_key,
      sizeof(*cache), (void**)&cache));
  return loom_amdgpu_get_fp8_scaled_descriptor(
      context, row->e8m0_pk8_descriptor_ref,
      &cache->initialized_e8m0_pk8_row_bits, &cache->present_e8m0_pk8_row_bits,
      cache->e8m0_pk8_descriptors, row_index, out_descriptor);
}
