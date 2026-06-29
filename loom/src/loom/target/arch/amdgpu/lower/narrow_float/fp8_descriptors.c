// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>

#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/narrow_float/fp8.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

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
#define LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_REF_ROW(source_type, result_type, \
                                                  lane_ref, pair_ref)       \
  {                                                                         \
    source_type, result_type, { lane_ref, pair_ref }                        \
  }
#include "loom/target/arch/amdgpu/lower/fp8_native_descriptor_ref_rows.inl"
#undef LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_REF_ROW
};
static_assert(IREE_ARRAYSIZE(kLoomAmdgpuFp8NativeDescriptorRefRows) <= 64,
              "native descriptor cache stores row state in one u64 bitset");

static bool loom_amdgpu_fp8_native_descriptor_ref_row(
    loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type, size_t* out_row_index,
    loom_amdgpu_fp8_native_descriptor_refs_t* out_refs) {
  *out_row_index = 0;
  *out_refs = (loom_amdgpu_fp8_native_descriptor_refs_t){
      .lane = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
      .pair = LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
  };
  for (size_t i = 0; i < IREE_ARRAYSIZE(kLoomAmdgpuFp8NativeDescriptorRefRows);
       ++i) {
    const loom_amdgpu_fp8_native_descriptor_ref_row_t* row =
        &kLoomAmdgpuFp8NativeDescriptorRefRows[i];
    if (row->source_element_type == source_element_type &&
        row->result_element_type == result_element_type) {
      *out_row_index = i;
      *out_refs = row->refs;
      return true;
    }
  }
  return false;
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

typedef struct loom_amdgpu_fp8_scalef32_descriptor_ref_row_t {
  // Encoded FP8/BF8 source scalar type.
  loom_scalar_type_t source_element_type;
  // Decoded result scalar type.
  loom_scalar_type_t result_element_type;
  // Descriptor ref for the native scale-f32 packet.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
} loom_amdgpu_fp8_scalef32_descriptor_ref_row_t;

static const loom_amdgpu_fp8_scalef32_descriptor_ref_row_t
    kLoomAmdgpuFp8ScaleF32DescriptorRefRows[] = {
#define LOOM_AMDGPU_FP8_SCALEF32_DESCRIPTOR_REF_ROW(source_type, result_type, \
                                                    ref)                      \
  {source_type, result_type, ref}
#include "loom/target/arch/amdgpu/lower/fp8_scalef32_descriptor_ref_rows.inl"
#undef LOOM_AMDGPU_FP8_SCALEF32_DESCRIPTOR_REF_ROW
};
static_assert(IREE_ARRAYSIZE(kLoomAmdgpuFp8ScaleF32DescriptorRefRows) <= 64,
              "scale-f32 descriptor cache stores row state in one u64 bitset");

static bool loom_amdgpu_fp8_scalef32_descriptor_ref_row(
    loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type, size_t* out_row_index,
    loom_amdgpu_descriptor_ref_t* out_ref) {
  *out_row_index = 0;
  *out_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  for (size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuFp8ScaleF32DescriptorRefRows); ++i) {
    const loom_amdgpu_fp8_scalef32_descriptor_ref_row_t* row =
        &kLoomAmdgpuFp8ScaleF32DescriptorRefRows[i];
    if (row->source_element_type == source_element_type &&
        row->result_element_type == result_element_type) {
      *out_row_index = i;
      *out_ref = row->descriptor_ref;
      return true;
    }
  }
  return false;
}

bool loom_amdgpu_fp8_scalef32_descriptor_ref(
    loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type,
    loom_amdgpu_descriptor_ref_t* out_ref) {
  size_t row_index = 0;
  return loom_amdgpu_fp8_scalef32_descriptor_ref_row(
      source_element_type, result_element_type, &row_index, out_ref);
}

typedef struct loom_amdgpu_fp8_scalef32_descriptor_cache_t {
  // Generated-row bits whose descriptor availability has been resolved.
  uint64_t initialized_row_bits;
  // Generated-row bits with a descriptor available in the active descriptor
  // set.
  uint64_t present_row_bits;
  // Function-local resolved descriptors keyed by generated row index.
  loom_low_lower_resolved_descriptor_t
      descriptors[IREE_ARRAYSIZE(kLoomAmdgpuFp8ScaleF32DescriptorRefRows)];
} loom_amdgpu_fp8_scalef32_descriptor_cache_t;

static int loom_amdgpu_fp8_scalef32_descriptor_cache_state_key;

iree_status_t loom_amdgpu_get_fp8_scalef32_descriptor(
    loom_low_lower_context_t* context, loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type,
    const loom_low_lower_resolved_descriptor_t** out_descriptor) {
  *out_descriptor = NULL;
  size_t row_index = 0;
  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_fp8_scalef32_descriptor_ref_row(
          source_element_type, result_element_type, &row_index,
          &descriptor_ref)) {
    return iree_ok_status();
  }
  loom_amdgpu_fp8_scalef32_descriptor_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_fp8_scalef32_descriptor_cache_state_key,
      sizeof(*cache), (void**)&cache));
  const uint64_t row_bit = UINT64_C(1) << row_index;
  if ((cache->initialized_row_bits & row_bit) == 0) {
    bool descriptor_is_present = false;
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
        context, descriptor_ref, &cache->descriptors[row_index],
        &descriptor_is_present));
    if (descriptor_is_present) {
      cache->present_row_bits |= row_bit;
    }
    cache->initialized_row_bits |= row_bit;
  }
  if ((cache->present_row_bits & row_bit) != 0) {
    *out_descriptor = &cache->descriptors[row_index];
  }
  return iree_ok_status();
}
