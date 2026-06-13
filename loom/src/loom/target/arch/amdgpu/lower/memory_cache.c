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
  // Accepted cache-scope bits indexed by loom_cache_scope_t.
  uint32_t scope_bits;
  // Accepted cache-temporal bits indexed by loom_cache_temporal_t.
  uint32_t temporal_bits;
  // Attribute fields materialized by this encoding.
  loom_amdgpu_memory_cache_policy_attr_flags_t attr_flags;
} loom_amdgpu_memory_cache_policy_encoding_row_t;

#define LOOM_AMDGPU_CACHE_SCOPE_BIT(scope) \
  ((uint32_t)1u << LOOM_CACHE_SCOPE_##scope)
#define LOOM_AMDGPU_CACHE_TEMPORAL_BIT(temporal) \
  ((uint32_t)1u << LOOM_CACHE_TEMPORAL_##temporal)

#define LOOM_AMDGPU_ALL_CACHE_SCOPE_BITS                               \
  (LOOM_AMDGPU_CACHE_SCOPE_BIT(CU) | LOOM_AMDGPU_CACHE_SCOPE_BIT(SE) | \
   LOOM_AMDGPU_CACHE_SCOPE_BIT(DEVICE) | LOOM_AMDGPU_CACHE_SCOPE_BIT(SYSTEM))

#define LOOM_AMDGPU_ALL_CACHE_TEMPORAL_BITS                     \
  (LOOM_AMDGPU_CACHE_TEMPORAL_BIT(REGULAR) |                    \
   LOOM_AMDGPU_CACHE_TEMPORAL_BIT(NON_TEMPORAL) |               \
   LOOM_AMDGPU_CACHE_TEMPORAL_BIT(HIGH_TEMPORAL) |              \
   LOOM_AMDGPU_CACHE_TEMPORAL_BIT(LAST_USE) |                   \
   LOOM_AMDGPU_CACHE_TEMPORAL_BIT(WRITEBACK) |                  \
   LOOM_AMDGPU_CACHE_TEMPORAL_BIT(NON_TEMPORAL_REGULAR) |       \
   LOOM_AMDGPU_CACHE_TEMPORAL_BIT(REGULAR_NON_TEMPORAL) |       \
   LOOM_AMDGPU_CACHE_TEMPORAL_BIT(NON_TEMPORAL_HIGH_TEMPORAL) | \
   LOOM_AMDGPU_CACHE_TEMPORAL_BIT(NON_TEMPORAL_WRITEBACK) |     \
   LOOM_AMDGPU_CACHE_TEMPORAL_BIT(BYPASS))

static const int64_t kAmdgpuGfx12CacheTemporalTh[] = {
    [LOOM_CACHE_TEMPORAL_REGULAR] = 0,
    [LOOM_CACHE_TEMPORAL_NON_TEMPORAL] = 1,
    [LOOM_CACHE_TEMPORAL_HIGH_TEMPORAL] = 2,
    [LOOM_CACHE_TEMPORAL_LAST_USE] = 3,
    [LOOM_CACHE_TEMPORAL_WRITEBACK] = 3,
    [LOOM_CACHE_TEMPORAL_NON_TEMPORAL_REGULAR] = 4,
    [LOOM_CACHE_TEMPORAL_REGULAR_NON_TEMPORAL] = 5,
    [LOOM_CACHE_TEMPORAL_NON_TEMPORAL_HIGH_TEMPORAL] = 6,
    [LOOM_CACHE_TEMPORAL_NON_TEMPORAL_WRITEBACK] = 7,
    [LOOM_CACHE_TEMPORAL_BYPASS] = 3,
};

static const loom_amdgpu_memory_cache_policy_encoding_row_t
    kAmdgpuMemoryCachePolicyEncodingRows[] = {
        {
            .encoding =
                LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH,
            .scope_bits = LOOM_AMDGPU_ALL_CACHE_SCOPE_BITS,
            .temporal_bits = LOOM_AMDGPU_ALL_CACHE_TEMPORAL_BITS,
            .attr_flags = LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_SCOPE |
                          LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_TH,
        },
        {
            .encoding =
                LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1,
            .scope_bits = LOOM_AMDGPU_CACHE_SCOPE_BIT(DEVICE),
            .temporal_bits = LOOM_AMDGPU_CACHE_TEMPORAL_BIT(REGULAR) |
                             LOOM_AMDGPU_CACHE_TEMPORAL_BIT(NON_TEMPORAL),
            .attr_flags = LOOM_AMDGPU_MEMORY_CACHE_POLICY_ATTR_NT,
        },
        {
            .encoding =
                LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC,
            .scope_bits = LOOM_AMDGPU_CACHE_SCOPE_BIT(DEVICE),
            .temporal_bits = LOOM_AMDGPU_CACHE_TEMPORAL_BIT(REGULAR),
            .attr_flags = 0,
        },
};

static const loom_amdgpu_memory_cache_policy_encoding_row_t*
loom_amdgpu_memory_cache_policy_encoding_row(
    loom_amdgpu_vector_memory_cache_policy_encoding_t encoding) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kAmdgpuMemoryCachePolicyEncodingRows); ++i) {
    const loom_amdgpu_memory_cache_policy_encoding_row_t* row =
        &kAmdgpuMemoryCachePolicyEncodingRows[i];
    if (row->encoding == encoding) {
      return row;
    }
  }
  return NULL;
}

static iree_string_view_t loom_amdgpu_memory_cache_policy_encoding_name(
    loom_amdgpu_vector_memory_cache_policy_encoding_t encoding) {
  switch (encoding) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE:
      return IREE_SV("none");
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
      return IREE_SV("gfx9_11_glc_slc_dlc");
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH:
      return IREE_SV("gfx12_nv_scope_th");
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return IREE_SV("gfx950_nt_sc0_sc1");
  }
  return IREE_SV("invalid");
}

static iree_string_view_t loom_amdgpu_memory_cache_policy_selected_name(
    loom_amdgpu_vector_memory_cache_policy_encoding_t encoding) {
  switch (encoding) {
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_NONE:
      return IREE_SV("memory_cache_policy.none");
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX9_11_GLC_SLC_DLC:
      return IREE_SV("memory_cache_policy.gfx9_11_glc_slc_dlc");
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX12_NV_SCOPE_TH:
      return IREE_SV("memory_cache_policy.gfx12_nv_scope_th");
    case LOOM_AMDGPU_VECTOR_MEMORY_CACHE_POLICY_ENCODING_GFX950_NT_SC0_SC1:
      return IREE_SV("memory_cache_policy.gfx950_nt_sc0_sc1");
  }
  return IREE_SV("memory_cache_policy.invalid");
}

static loom_amdgpu_vector_memory_cache_policy_encoding_t
loom_amdgpu_memory_cache_policy_descriptor_encoding(
    const loom_low_descriptor_set_t* descriptor_set) {
  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  return descriptor_set_info
             ? descriptor_set_info->vector_memory_cache_policy_encoding
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

bool loom_amdgpu_memory_cache_policy_encode(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_cache_policy_attrs_t* out_attrs) {
  *out_attrs = (loom_amdgpu_memory_cache_policy_attrs_t){0};

  const loom_vector_memory_cache_policy_t* policy =
      &access->source.cache_policy;
  if (!loom_amdgpu_memory_cache_policy_is_present(policy)) {
    return true;
  }
  if (!loom_amdgpu_memory_cache_policy_is_complete(policy) ||
      access->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return false;
  }

  const loom_amdgpu_descriptor_set_info_t* descriptor_set_info =
      loom_amdgpu_target_info_descriptor_set_at(
          descriptor_set->descriptor_set_ordinal);
  if (descriptor_set_info == NULL) {
    return false;
  }
  const loom_amdgpu_memory_cache_policy_encoding_row_t* row =
      loom_amdgpu_memory_cache_policy_encoding_row(
          descriptor_set_info->vector_memory_cache_policy_encoding);
  if (row == NULL ||
      !loom_amdgpu_memory_cache_policy_bits_contain(row->scope_bits,
                                                    policy->cache_scope) ||
      !loom_amdgpu_memory_cache_policy_bits_contain(row->temporal_bits,
                                                    policy->cache_temporal)) {
    return false;
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
  return true;
}

bool loom_amdgpu_memory_cache_policy_can_lower(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access) {
  loom_amdgpu_memory_cache_policy_attrs_t attrs;
  return loom_amdgpu_memory_cache_policy_encode(descriptor_set, access, &attrs);
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
