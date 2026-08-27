// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>

#include "iree/base/api.h"
#include "loom/ops/cache.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/target_info.h"

bool loom_amdgpu_memory_cache_policy_is_present(
    const loom_vector_memory_cache_policy_t* policy) {
  return iree_any_bit_set(
      policy->build_flags,
      LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_SCOPE |
          LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_TEMPORAL);
}

iree_string_view_t loom_amdgpu_cache_scope_name(uint8_t scope) {
  switch (scope) {
    case LOOM_CACHE_SCOPE_CU:
      return IREE_SV("cu");
    case LOOM_CACHE_SCOPE_SE:
      return IREE_SV("se");
    case LOOM_CACHE_SCOPE_DEVICE:
      return IREE_SV("device");
    case LOOM_CACHE_SCOPE_SYSTEM:
      return IREE_SV("system");
  }
  return IREE_SV("invalid");
}

iree_string_view_t loom_amdgpu_cache_temporal_name(uint8_t temporal) {
  switch (temporal) {
    case LOOM_CACHE_TEMPORAL_REGULAR:
      return IREE_SV("regular");
    case LOOM_CACHE_TEMPORAL_NON_TEMPORAL:
      return IREE_SV("non_temporal");
    case LOOM_CACHE_TEMPORAL_HIGH_TEMPORAL:
      return IREE_SV("high_temporal");
    case LOOM_CACHE_TEMPORAL_LAST_USE:
      return IREE_SV("last_use");
    case LOOM_CACHE_TEMPORAL_WRITEBACK:
      return IREE_SV("writeback");
    case LOOM_CACHE_TEMPORAL_NON_TEMPORAL_REGULAR:
      return IREE_SV("non_temporal_regular");
    case LOOM_CACHE_TEMPORAL_REGULAR_NON_TEMPORAL:
      return IREE_SV("regular_non_temporal");
    case LOOM_CACHE_TEMPORAL_NON_TEMPORAL_HIGH_TEMPORAL:
      return IREE_SV("non_temporal_high_temporal");
    case LOOM_CACHE_TEMPORAL_NON_TEMPORAL_WRITEBACK:
      return IREE_SV("non_temporal_writeback");
    case LOOM_CACHE_TEMPORAL_BYPASS:
      return IREE_SV("bypass");
  }
  return IREE_SV("invalid");
}

static bool loom_amdgpu_memory_cache_policy_is_complete(
    const loom_vector_memory_cache_policy_t* policy) {
  const uint32_t required_flags =
      LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_SCOPE |
      LOOM_VECTOR_MEMORY_CACHE_POLICY_BUILD_FLAG_TEMPORAL;
  return policy->build_flags == 0 || policy->build_flags == required_flags;
}

typedef struct loom_amdgpu_memory_cache_policy_encoding_row_t {
  // Descriptor-set cache-policy encoding selected by target info.
  loom_amdgpu_vector_memory_cache_policy_encoding_t encoding;
  // Stable key naming the encoding.
  iree_string_view_t encoding_key;
  // Stable key naming the selected encoding decision.
  iree_string_view_t selected_key;
  // Accepted cache-scope bits indexed by loom_cache_scope_t.
  uint32_t scope_bits;
  // Accepted cache-temporal bits indexed by loom_cache_temporal_t.
  uint32_t temporal_bits;
  // Attribute fields materialized by this encoding.
  loom_amdgpu_memory_cache_policy_attr_flags_t attr_flags;
} loom_amdgpu_memory_cache_policy_encoding_row_t;

static const int64_t kAmdgpuGfx12CacheTemporalTh[LOOM_CACHE_TEMPORAL_COUNT_] = {
#include "loom/target/arch/amdgpu/memory_cache_policy_temporal_th.inl"
};

static const loom_amdgpu_memory_cache_policy_encoding_row_t
    kAmdgpuMemoryCachePolicyEncodingRows[] = {
#include "loom/target/arch/amdgpu/memory_cache_policy_encoding_rows.inl"
};

static const loom_amdgpu_memory_cache_policy_encoding_row_t*
loom_amdgpu_memory_cache_policy_encoding_row(
    loom_amdgpu_vector_memory_cache_policy_encoding_t encoding) {
  if (encoding == LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE) {
    return NULL;
  }
  const iree_host_size_t row_index = (iree_host_size_t)encoding - 1u;
  if (row_index < IREE_ARRAYSIZE(kAmdgpuMemoryCachePolicyEncodingRows) &&
      kAmdgpuMemoryCachePolicyEncodingRows[row_index].encoding == encoding) {
    return &kAmdgpuMemoryCachePolicyEncodingRows[row_index];
  }
  return NULL;
}

static iree_string_view_t loom_amdgpu_memory_cache_policy_encoding_name(
    loom_amdgpu_vector_memory_cache_policy_encoding_t encoding) {
  const loom_amdgpu_memory_cache_policy_encoding_row_t* row =
      loom_amdgpu_memory_cache_policy_encoding_row(encoding);
  if (row != NULL) {
    return row->encoding_key;
  }
  if (encoding == LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE) {
    return IREE_SV("none");
  }
  return IREE_SV("invalid");
}

static iree_string_view_t loom_amdgpu_memory_cache_policy_selected_name(
    loom_amdgpu_vector_memory_cache_policy_encoding_t encoding) {
  const loom_amdgpu_memory_cache_policy_encoding_row_t* row =
      loom_amdgpu_memory_cache_policy_encoding_row(encoding);
  if (row != NULL) {
    return row->selected_key;
  }
  if (encoding == LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE) {
    return IREE_SV("memory_cache_policy.none");
  }
  return IREE_SV("memory_cache_policy.invalid");
}

loom_amdgpu_vector_memory_cache_policy_encoding_t
loom_amdgpu_memory_cache_policy_descriptor_encoding(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  return descriptor_set_info
             ? descriptor_set_info->vector_memory.cache_policy_encoding
             : LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE;
}

iree_string_view_t loom_amdgpu_memory_cache_policy_encoding_key(
    const loom_low_descriptor_set_t* descriptor_set) {
  return loom_amdgpu_memory_cache_policy_encoding_name(
      loom_amdgpu_memory_cache_policy_descriptor_encoding(descriptor_set));
}

iree_string_view_t loom_amdgpu_memory_cache_policy_selected_key(
    const loom_low_descriptor_set_t* descriptor_set) {
  return loom_amdgpu_memory_cache_policy_selected_name(
      loom_amdgpu_memory_cache_policy_descriptor_encoding(descriptor_set));
}

static bool loom_amdgpu_memory_cache_policy_bits_contain(uint32_t bits,
                                                         uint8_t ordinal) {
  return ordinal < 32 && iree_any_bit_set(bits, (uint32_t)1u << ordinal);
}

loom_amdgpu_memory_cache_policy_resolution_t
loom_amdgpu_memory_cache_policy_resolve(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_cache_policy_attrs_t* out_attrs) {
  *out_attrs = (loom_amdgpu_memory_cache_policy_attrs_t){0};

  const loom_vector_memory_cache_policy_t* policy =
      &access->source.cache_policy;
  if (!loom_amdgpu_memory_cache_policy_is_present(policy)) {
    return LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_ABSENT;
  }
  if (!loom_amdgpu_memory_cache_policy_is_complete(policy) ||
      access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_REJECTED;
  }

  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  if (descriptor_set_info == NULL) {
    return LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_REJECTED;
  }
  const loom_amdgpu_memory_cache_policy_encoding_row_t* row =
      loom_amdgpu_memory_cache_policy_encoding_row(
          descriptor_set_info->vector_memory.cache_policy_encoding);
  if (row == NULL) {
    return LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_REJECTED;
  }
  if (!loom_amdgpu_memory_cache_policy_bits_contain(row->scope_bits,
                                                    policy->cache_scope) ||
      !loom_amdgpu_memory_cache_policy_bits_contain(row->temporal_bits,
                                                    policy->cache_temporal)) {
    return LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_DROPPED;
  }

  if (iree_any_bit_set(row->attr_flags,
                       LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_SCOPE)) {
    out_attrs->flags |= LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_SCOPE;
    out_attrs->scope = policy->cache_scope;
  }
  if (iree_any_bit_set(row->attr_flags,
                       LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_TH)) {
    IREE_ASSERT(policy->cache_temporal <
                IREE_ARRAYSIZE(kAmdgpuGfx12CacheTemporalTh));
    out_attrs->flags |= LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_TH;
    out_attrs->th = kAmdgpuGfx12CacheTemporalTh[policy->cache_temporal];
  }
  if (iree_any_bit_set(row->attr_flags,
                       LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_NT) &&
      policy->cache_temporal == LOOM_CACHE_TEMPORAL_NON_TEMPORAL) {
    out_attrs->flags |= LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_NT;
    out_attrs->nt = 1;
  }
  return LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_ENCODED;
}

bool loom_amdgpu_memory_cache_policy_can_lower(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access) {
  loom_amdgpu_memory_cache_policy_attrs_t attrs;
  return loom_amdgpu_memory_cache_policy_resolve(descriptor_set, access,
                                                 &attrs) !=
         LOOM_AMDGPU_MEMORY_CACHE_POLICY_RESOLUTION_REJECTED;
}

iree_string_view_t loom_amdgpu_memory_cache_policy_rejection_key(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    const loom_vector_memory_cache_policy_t* policy) {
  if (!loom_amdgpu_memory_cache_policy_is_complete(policy)) {
    return IREE_SV("memory_cache_policy.incomplete");
  }
  if (access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return IREE_SV("memory_cache_policy.workgroup");
  }
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  if (descriptor_set_info == NULL) {
    return IREE_SV("memory_cache_policy.descriptor_set_info");
  }
  return IREE_SV("memory_cache_policy.descriptor_encoding");
}
