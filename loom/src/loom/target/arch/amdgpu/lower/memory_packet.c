// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/memory_packet.h"

#include <stdint.h>

#include "loom/ir/context.h"
#include "loom/ops/kernel/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/source_alloca_layout.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/util/fact_table.h"

static bool loom_amdgpu_memory_vaddr_offset_fits_u32(
    const loom_amdgpu_memory_access_t* access, int64_t static_byte_offset) {
  bool all_dynamic_terms_are_vaddr = true;
  for (uint8_t i = 0; i < access->source->dynamic_term_count; ++i) {
    all_dynamic_terms_are_vaddr &=
        access->dynamic_term_kinds[i] == LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
  }
  if (all_dynamic_terms_are_vaddr) {
    return loom_low_source_memory_dynamic_offset_fits_unsigned_bit_count(
        access->source, static_byte_offset, 32);
  }

  loom_value_facts_t offset_facts =
      loom_value_facts_exact_i64(static_byte_offset);
  for (uint8_t i = 0; i < access->source->dynamic_term_count; ++i) {
    if (access->dynamic_term_kinds[i] !=
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR) {
      continue;
    }
    loom_value_facts_addi(&offset_facts,
                          &access->source->dynamic_terms[i].byte_facts,
                          &offset_facts);
  }
  return loom_value_facts_fit_unsigned_bit_count(offset_facts, 32);
}

bool loom_amdgpu_memory_access_has_contiguous_vector_lanes(
    const loom_amdgpu_memory_access_t* access) {
  return access->vector_lane_count <= 1 ||
         access->source->vector_lane_byte_stride ==
             access->source->element_byte_count;
}

bool loom_amdgpu_memory_access_is_ds_pair(
    const loom_amdgpu_memory_access_t* access) {
  if (access->vector_lane_count != 2u ||
      (access->source->element_byte_count != 4u &&
       access->source->element_byte_count != 8u)) {
    return false;
  }
  const uint32_t lane_register_count = access->source->element_byte_count / 4u;
  return access->payload_register_count == lane_register_count * 2u &&
         access->packet_byte_count == access->source->element_byte_count * 2u;
}

bool loom_amdgpu_memory_access_static_byte_offset_is_usable(
    int64_t static_byte_offset,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (static_byte_offset < 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET;
    return false;
  }
  return true;
}

typedef enum loom_amdgpu_memory_descriptor_domain_e {
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE = 0,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_SADDR = 1,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS = 2,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_FLAT = 3,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_SMEM = 4,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_SCRATCH = 5,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GENERIC_FLAT = 6,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_COUNT_,
} loom_amdgpu_memory_descriptor_domain_t;

typedef enum loom_amdgpu_memory_address_attempt_kind_e {
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_ADDTID = 0,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_2ADDR = 1,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_LDS_DEFAULT = 2,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_BUFFER_RESOURCE = 3,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_SADDR = 4,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_FLAT = 5,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_SMEM = 6,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SCRATCH_VADDR = 7,
} loom_amdgpu_memory_address_attempt_kind_t;

typedef enum loom_amdgpu_memory_address_attempt_result_e {
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE = 0,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED = 1,
  LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED = 2,
} loom_amdgpu_memory_address_attempt_result_t;

typedef struct loom_amdgpu_memory_address_attempt_t {
  // Address-form selector attempted by this row.
  loom_amdgpu_memory_address_attempt_kind_t kind;
} loom_amdgpu_memory_address_attempt_t;

static const loom_amdgpu_memory_address_attempt_t kAmdgpuLdsAddressAttempts[] =
    {
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_ADDTID,
        },
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_2ADDR,
        },
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_LDS_DEFAULT,
        },
};

static const loom_amdgpu_memory_address_attempt_t
    kAmdgpuGlobalAddressAttempts[] = {
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_SMEM,
        },
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_SADDR,
        },
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_FLAT,
        },
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_BUFFER_RESOURCE,
        },
};

static const loom_amdgpu_memory_address_attempt_t kAmdgpuFlatAddressAttempts[] =
    {
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_FLAT,
        },
};

static const loom_amdgpu_memory_address_attempt_t
    kAmdgpuDescriptorAddressAttempts[] = {
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_BUFFER_RESOURCE,
        },
};

static const loom_amdgpu_memory_address_attempt_t
    kAmdgpuScratchAddressAttempts[] = {
        {
            .kind = LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SCRATCH_VADDR,
        },
};

static const uint8_t kAmdgpuMemorySpaceDescriptorDomainMap[] = {
    [LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN] =
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
    [LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL] =
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
    [LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP] =
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
    [LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE] =
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_SCRATCH,
    [LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT] =
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
    [LOOM_VALUE_FACT_MEMORY_SPACE_HOST] =
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_COUNT_,
    [LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR] =
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
    [LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC] =
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GENERIC_FLAT,
};
static_assert(IREE_ARRAYSIZE(kAmdgpuMemorySpaceDescriptorDomainMap) ==
                  LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC + 1u,
              "AMDGPU memory-space map must cover every memory-space fact");

static bool loom_amdgpu_memory_descriptor_domain_from_memory_space(
    loom_value_fact_memory_space_t memory_space,
    loom_amdgpu_memory_descriptor_domain_t* out_descriptor_domain) {
  *out_descriptor_domain = LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE;
  const uint32_t memory_space_ordinal = (uint32_t)memory_space;
  if (memory_space_ordinal >=
      IREE_ARRAYSIZE(kAmdgpuMemorySpaceDescriptorDomainMap)) {
    return false;
  }
  const uint8_t descriptor_domain =
      kAmdgpuMemorySpaceDescriptorDomainMap[memory_space_ordinal];
  if (descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_COUNT_) {
    return false;
  }
  *out_descriptor_domain =
      (loom_amdgpu_memory_descriptor_domain_t)descriptor_domain;
  return true;
}

typedef uint32_t loom_amdgpu_memory_descriptor_candidate_key_t;

enum {
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_DOMAIN_BITS = 3u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_ADDRESS_FORM_BITS = 3u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_OPERATION_BITS = 1u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_BITS = 5u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_CLASS_BITS = 1u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_FORMAT_BITS = 3u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_BITS = 3u,

  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_DOMAIN_SHIFT = 0u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_ADDRESS_FORM_SHIFT =
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_DOMAIN_SHIFT +
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_DOMAIN_BITS,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_OPERATION_SHIFT =
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_ADDRESS_FORM_SHIFT +
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_ADDRESS_FORM_BITS,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_SHIFT =
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_OPERATION_SHIFT +
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_OPERATION_BITS,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_CLASS_SHIFT =
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_SHIFT +
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_BITS,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_FORMAT_SHIFT =
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_CLASS_SHIFT +
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_CLASS_BITS,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_SHIFT =
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_FORMAT_SHIFT +
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_FORMAT_BITS,

  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_DOMAIN_MAX =
      (1u << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_DOMAIN_BITS) - 1u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_ADDRESS_FORM_MAX =
      (1u << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_ADDRESS_FORM_BITS) - 1u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_MAX =
      (1u << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_BITS) -
      1u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_CLASS_MAX =
      (1u
       << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_CLASS_BITS) -
      1u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_FORMAT_MAX =
      (1u << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_FORMAT_BITS) - 1u,
  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_MAX =
      (1u
       << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_BITS) -
      1u,
};

static_assert(LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_COUNT_ - 1u <=
                  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_DOMAIN_MAX,
              "memory descriptor domain key field is too small");
static_assert(LOOM_AMDGPU_MEMORY_ADDRESS_FORM_COUNT_ - 1u <=
                  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_ADDRESS_FORM_MAX,
              "memory address form key field is too small");
static_assert(4u * LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES <=
                  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_MAX,
              "memory packet byte count key field is too small");
static_assert(
    LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_COUNT_ - 1u <=
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_CLASS_MAX,
    "memory payload register class key field is too small");
static_assert(LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_COUNT_ - 1u <=
                  LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_FORMAT_MAX,
              "memory payload format key field is too small");
static_assert(
    LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES <=
        LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_MAX,
    "memory payload register count key field is too small");
static_assert(
    LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_SHIFT +
            LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_BITS <=
        8u * sizeof(loom_amdgpu_memory_descriptor_candidate_key_t),
    "memory descriptor candidate key storage is too small");

#define LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_OPERATION_KEY(kind) \
  ((loom_amdgpu_memory_descriptor_candidate_key_t)((kind) ==        \
                                                   LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE))

#define LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KEY(                           \
    domain, address_form, operation_key, packet_byte_count,                    \
    payload_register_class, payload_format, payload_register_count)            \
  ((((loom_amdgpu_memory_descriptor_candidate_key_t)(domain))                  \
    << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_DOMAIN_SHIFT) |                 \
   (((loom_amdgpu_memory_descriptor_candidate_key_t)(address_form))            \
    << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_ADDRESS_FORM_SHIFT) |           \
   (((loom_amdgpu_memory_descriptor_candidate_key_t)(operation_key))           \
    << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_OPERATION_SHIFT) |              \
   (((loom_amdgpu_memory_descriptor_candidate_key_t)(packet_byte_count))       \
    << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_SHIFT) |      \
   (((loom_amdgpu_memory_descriptor_candidate_key_t)(payload_register_class))  \
    << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_CLASS_SHIFT) | \
   (((loom_amdgpu_memory_descriptor_candidate_key_t)(payload_format))          \
    << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_FORMAT_SHIFT) |         \
   (((loom_amdgpu_memory_descriptor_candidate_key_t)(payload_register_count))  \
    << LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_SHIFT))

#define LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE(                           \
    domain, address_form, kind, packet_byte_count, payload_register_class, \
    payload_format, payload_register_count, descriptor_ref_value,          \
    preserving_ref)                                                        \
  {                                                                        \
      .key = LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KEY(                  \
          domain, address_form,                                            \
          LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_OPERATION_KEY(kind),     \
          (packet_byte_count), payload_register_class, payload_format,     \
          (payload_register_count)),                                       \
      .descriptor_ref = descriptor_ref_value,                              \
      .preserving_descriptor_ref = preserving_ref,                         \
  },

#define LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_RANGE(                     \
    domain, address_form, kind, packet_byte_count, payload_register_class, \
    payload_format, payload_register_count, first_candidate_index,         \
    candidate_count_value)                                                 \
  {                                                                        \
      .key = LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KEY(                  \
          domain, address_form,                                            \
          LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_OPERATION_KEY(kind),     \
          (packet_byte_count), payload_register_class, payload_format,     \
          (payload_register_count)),                                       \
      .first_candidate = (first_candidate_index),                          \
      .candidate_count = (candidate_count_value),                          \
  },

static bool loom_amdgpu_memory_descriptor_candidate_key(
    loom_amdgpu_memory_descriptor_domain_t domain,
    loom_amdgpu_memory_address_form_t address_form,
    loom_low_source_memory_operation_kind_t kind, uint32_t packet_byte_count,
    loom_amdgpu_memory_payload_register_class_t payload_register_class,
    loom_amdgpu_memory_payload_format_t payload_format,
    uint32_t payload_register_count,
    loom_amdgpu_memory_descriptor_candidate_key_t* out_key) {
  if (packet_byte_count >
          LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PACKET_BYTE_COUNT_MAX ||
      payload_register_count >
          LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_PAYLOAD_REGISTER_COUNT_MAX ||
      (kind != LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD &&
       kind != LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE)) {
    *out_key = 0;
    return false;
  }
  *out_key = LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KEY(
      domain, address_form,
      LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_OPERATION_KEY(kind),
      packet_byte_count, payload_register_class, payload_format,
      payload_register_count);
  return true;
}

typedef struct loom_amdgpu_memory_descriptor_candidate_t {
  // Packed selector key for memory operation shape and payload format.
  loom_amdgpu_memory_descriptor_candidate_key_t key;
  // Stable descriptor ref selected when present in the descriptor set.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
  // Tied load form, or NONE for scalar memory and stores. Fits existing
  // padding.
  loom_amdgpu_descriptor_ref_t preserving_descriptor_ref;
} loom_amdgpu_memory_descriptor_candidate_t;
static_assert(sizeof(loom_amdgpu_memory_descriptor_candidate_t) == 8,
              "memory descriptor candidate rows must stay compact");

typedef struct loom_amdgpu_memory_descriptor_candidate_range_t {
  // Packed selector key shared by a contiguous fallback candidate range.
  loom_amdgpu_memory_descriptor_candidate_key_t key;
  // First candidate row for the selector key.
  uint16_t first_candidate;
  // Number of candidate rows for the selector key.
  uint16_t candidate_count;
} loom_amdgpu_memory_descriptor_candidate_range_t;
static_assert(sizeof(loom_amdgpu_memory_descriptor_candidate_range_t) == 8,
              "memory descriptor candidate ranges must stay compact");

static const loom_amdgpu_memory_descriptor_candidate_t
    kAmdgpuMemoryDescriptorCandidates[] = {
#include "loom/target/arch/amdgpu/lower/candidates/memory_descriptor_candidates.inl"
};

static const loom_amdgpu_memory_descriptor_candidate_range_t
    kAmdgpuMemoryDescriptorCandidateRanges[] = {
#include "loom/target/arch/amdgpu/lower/candidates/memory_descriptor_candidate_ranges.inl"
};

#undef LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_RANGE

#undef LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE

#undef LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_KEY

#undef LOOM_AMDGPU_MEMORY_DESCRIPTOR_CANDIDATE_OPERATION_KEY

static const loom_amdgpu_memory_descriptor_candidate_range_t*
loom_amdgpu_find_memory_descriptor_candidate_range(
    loom_amdgpu_memory_descriptor_candidate_key_t candidate_key) {
  iree_host_size_t lower_bound = 0;
  iree_host_size_t upper_bound =
      IREE_ARRAYSIZE(kAmdgpuMemoryDescriptorCandidateRanges);
  while (lower_bound < upper_bound) {
    const iree_host_size_t middle =
        lower_bound + ((upper_bound - lower_bound) / 2);
    const loom_amdgpu_memory_descriptor_candidate_range_t* range =
        &kAmdgpuMemoryDescriptorCandidateRanges[middle];
    if (candidate_key < range->key) {
      upper_bound = middle;
    } else if (candidate_key > range->key) {
      lower_bound = middle + 1;
    } else {
      return range;
    }
  }
  return NULL;
}

static bool loom_amdgpu_select_memory_descriptor_candidate(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_descriptor_domain_t domain,
    loom_amdgpu_memory_address_form_t address_form,
    loom_low_source_memory_operation_kind_t kind, uint32_t packet_byte_count,
    loom_amdgpu_memory_payload_register_class_t payload_register_class,
    loom_amdgpu_memory_payload_format_t payload_format,
    uint32_t payload_register_count, bool preserve_inactive,
    const loom_low_descriptor_t** out_descriptor,
    uint32_t* out_descriptor_ordinal) {
  *out_descriptor = NULL;
  *out_descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  loom_amdgpu_memory_descriptor_candidate_key_t candidate_key = 0;
  if (!loom_amdgpu_memory_descriptor_candidate_key(
          domain, address_form, kind, packet_byte_count, payload_register_class,
          payload_format, payload_register_count, &candidate_key)) {
    return false;
  }
  const loom_amdgpu_memory_descriptor_candidate_range_t* range =
      loom_amdgpu_find_memory_descriptor_candidate_range(candidate_key);
  if (range == NULL) {
    return false;
  }
  IREE_ASSERT((iree_host_size_t)range->first_candidate +
                  range->candidate_count <=
              IREE_ARRAYSIZE(kAmdgpuMemoryDescriptorCandidates));
  const iree_host_size_t candidate_end =
      (iree_host_size_t)range->first_candidate + range->candidate_count;
  for (iree_host_size_t i = range->first_candidate; i < candidate_end; ++i) {
    const loom_amdgpu_memory_descriptor_candidate_t* candidate =
        &kAmdgpuMemoryDescriptorCandidates[i];
    const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
        descriptor_set, preserve_inactive ? candidate->preserving_descriptor_ref
                                          : candidate->descriptor_ref);
    if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
      continue;
    }
    const loom_low_descriptor_t* descriptor =
        loom_low_descriptor_set_descriptor_at(descriptor_set,
                                              descriptor_ordinal);
    IREE_ASSERT(descriptor != NULL);
    *out_descriptor = descriptor;
    *out_descriptor_ordinal = descriptor_ordinal;
    return true;
  }
  return false;
}

static bool loom_amdgpu_select_buffer_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_address_form_t address_form,
    loom_low_source_memory_operation_kind_t kind,
    const loom_low_descriptor_t** out_descriptor,
    uint32_t* out_descriptor_ordinal) {
  return loom_amdgpu_select_memory_descriptor_candidate(
      descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_BUFFER_RESOURCE,
      address_form, kind, access->packet_byte_count,
      access->payload_register_class, access->payload_format,
      access->payload_register_count, access->preserve_inactive, out_descriptor,
      out_descriptor_ordinal);
}

static bool loom_amdgpu_select_global_saddr_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_low_source_memory_operation_kind_t kind,
    const loom_low_descriptor_t** out_descriptor,
    uint32_t* out_descriptor_ordinal) {
  return loom_amdgpu_select_memory_descriptor_candidate(
      descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_SADDR,
      LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR, kind,
      access->packet_byte_count, access->payload_register_class,
      access->payload_format, access->payload_register_count,
      access->preserve_inactive, out_descriptor, out_descriptor_ordinal);
}

static bool loom_amdgpu_select_global_smem_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_low_source_memory_operation_kind_t kind,
    const loom_low_descriptor_t** out_descriptor,
    uint32_t* out_descriptor_ordinal) {
  return loom_amdgpu_select_memory_descriptor_candidate(
      descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_SMEM,
      LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM, kind,
      access->packet_byte_count, access->payload_register_class,
      access->payload_format, access->payload_register_count,
      access->preserve_inactive, out_descriptor, out_descriptor_ordinal);
}

static bool loom_amdgpu_select_flat_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_low_source_memory_operation_kind_t kind,
    const loom_low_descriptor_t** out_descriptor,
    uint32_t* out_descriptor_ordinal) {
  const loom_amdgpu_memory_descriptor_domain_t domain =
      access->source->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC
          ? LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GENERIC_FLAT
          : LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GLOBAL_FLAT;
  return loom_amdgpu_select_memory_descriptor_candidate(
      descriptor_set, domain, LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT, kind,
      access->packet_byte_count, access->payload_register_class,
      access->payload_format, access->payload_register_count,
      access->preserve_inactive, out_descriptor, out_descriptor_ordinal);
}

static bool loom_amdgpu_select_scratch_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_low_source_memory_operation_kind_t kind,
    const loom_low_descriptor_t** out_descriptor,
    uint32_t* out_descriptor_ordinal) {
  return loom_amdgpu_select_memory_descriptor_candidate(
      descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_SCRATCH,
      LOOM_AMDGPU_MEMORY_ADDRESS_FORM_SCRATCH_VADDR, kind,
      access->packet_byte_count, access->payload_register_class,
      access->payload_format, access->payload_register_count,
      access->preserve_inactive, out_descriptor, out_descriptor_ordinal);
}

static bool loom_amdgpu_select_ds_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_low_source_memory_operation_kind_t kind,
    const loom_low_descriptor_t** out_descriptor,
    uint32_t* out_descriptor_ordinal) {
  return loom_amdgpu_select_memory_descriptor_candidate(
      descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
      LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT, kind, access->packet_byte_count,
      access->payload_register_class, access->payload_format,
      access->payload_register_count, access->preserve_inactive, out_descriptor,
      out_descriptor_ordinal);
}

static bool loom_amdgpu_select_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access,
    loom_low_source_memory_operation_kind_t kind,
    const loom_low_descriptor_t** out_descriptor,
    uint32_t* out_descriptor_ordinal) {
  loom_amdgpu_memory_descriptor_domain_t descriptor_domain;
  if (!loom_amdgpu_memory_descriptor_domain_from_memory_space(
          access->source->memory_space, &descriptor_domain)) {
    return false;
  }
  if (descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS) {
    return loom_amdgpu_select_ds_memory_descriptor(
        descriptor_set, access, kind, out_descriptor, out_descriptor_ordinal);
  }
  if (descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_SCRATCH) {
    return loom_amdgpu_select_scratch_memory_descriptor(
        descriptor_set, access, kind, out_descriptor, out_descriptor_ordinal);
  }
  if (descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GENERIC_FLAT) {
    return loom_amdgpu_select_flat_memory_descriptor(
        descriptor_set, access, kind, out_descriptor, out_descriptor_ordinal);
  }
  return loom_amdgpu_select_buffer_memory_descriptor(
      descriptor_set, access, access->address_form, kind, out_descriptor,
      out_descriptor_ordinal);
}

typedef struct loom_amdgpu_offset_immediate_encoding_t {
  // Target immediate encoding ID.
  uint16_t encoding_id;
  // Byte count represented by one encoded offset unit.
  uint32_t unit_byte_count;
} loom_amdgpu_offset_immediate_encoding_t;

static const loom_amdgpu_offset_immediate_encoding_t
    kAmdgpuOffsetImmediateEncodings[] = {
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_BYTE,
            .unit_byte_count = 1,
        },
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DWORD,
            .unit_byte_count = 4,
        },
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_QWORD,
            .unit_byte_count = 8,
        },
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DWORD_STRIDE64,
            .unit_byte_count = 4 * 64,
        },
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_QWORD_STRIDE64,
            .unit_byte_count = 8 * 64,
        },
        {
            .encoding_id =
                LOOM_AMDGPU_IMMEDIATE_ENCODING_ID_ADDRESS_OFFSET_DS16,
            .unit_byte_count = 1,
        },
};

static bool loom_amdgpu_immediate_encoding_address_unit_byte_count(
    uint16_t encoding_id, uint32_t* out_unit_byte_count) {
  *out_unit_byte_count = 0;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kAmdgpuOffsetImmediateEncodings); ++i) {
    const loom_amdgpu_offset_immediate_encoding_t* encoding =
        &kAmdgpuOffsetImmediateEncodings[i];
    if (encoding->encoding_id == encoding_id) {
      *out_unit_byte_count = encoding->unit_byte_count;
      return true;
    }
  }
  return false;
}

bool loom_amdgpu_descriptor_offset_immediate_info(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal, uint16_t expected_offset_immediate_count,
    loom_low_immediate_kind_t expected_kind,
    loom_amdgpu_descriptor_offset_immediate_info_t* out_info) {
  *out_info = (loom_amdgpu_descriptor_offset_immediate_info_t){
      .kind = expected_kind,
      .signed_min = INT64_MIN,
      .unsigned_max = UINT64_MAX,
  };
  if (!loom_low_descriptor_set_has_descriptor(descriptor_set,
                                              descriptor_ordinal)) {
    return false;
  }
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];
  const loom_low_descriptor_layout_t* descriptor_layout =
      loom_low_descriptor_layout(descriptor_set, descriptor);
  if (descriptor_layout->immediate_count != 0 &&
      descriptor_layout->immediate_start >= descriptor_set->immediate_count) {
    return false;
  }
  uint16_t offset_immediate_count = 0;
  uint64_t unsigned_max = UINT64_MAX;
  for (uint16_t i = 0; i < descriptor_layout->immediate_count; ++i) {
    const uint32_t immediate_index = descriptor_layout->immediate_start + i;
    if (immediate_index >= descriptor_set->immediate_count) {
      return false;
    }
    const loom_low_immediate_t* immediate =
        &descriptor_set->immediates[immediate_index];
    uint32_t unit_byte_count = 0;
    if (!loom_amdgpu_immediate_encoding_address_unit_byte_count(
            immediate->encoding_id, &unit_byte_count)) {
      continue;
    }
    if (immediate->kind != expected_kind) {
      return false;
    }
    if (offset_immediate_count == 0) {
      out_info->unit_byte_count = unit_byte_count;
    } else if (out_info->unit_byte_count != unit_byte_count) {
      return false;
    }
    if (expected_kind == LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
      out_info->signed_min =
          iree_max(out_info->signed_min, immediate->signed_min);
    }
    unsigned_max = iree_min(unsigned_max, immediate->unsigned_max);
    ++offset_immediate_count;
  }
  if (offset_immediate_count != expected_offset_immediate_count) {
    return false;
  }
  out_info->unsigned_max = unsigned_max;
  return true;
}

static bool loom_amdgpu_memory_access_try_select_buffer_off_zero(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_low_source_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access) {
  if (access->source->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP ||
      loom_low_source_memory_access_is_dynamic(access->source) ||
      access->scalar_byte_offset != 0) {
    return false;
  }
  const loom_low_descriptor_t* descriptor = NULL;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_buffer_memory_descriptor(
          descriptor_set, access,
          LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO, kind, &descriptor,
          &descriptor_ordinal)) {
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info) ||
      offset_info.unit_byte_count != 1 ||
      (uint64_t)access->immediate_offset > offset_info.unsigned_max) {
    return false;
  }
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_BUFFER_OFF_ZERO;
  access->descriptor = descriptor;
  return true;
}

static bool loom_amdgpu_memory_access_split_static_offset(
    loom_amdgpu_memory_access_t* access, uint32_t offset_unit_byte_count,
    uint64_t offset_unsigned_max,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (access->static_byte_offset < 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET;
    return false;
  }
  if (offset_unit_byte_count == 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }

  const uint64_t static_byte_offset = (uint64_t)access->static_byte_offset;
  if (access->source->memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if ((static_byte_offset % offset_unit_byte_count) != 0) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE;
      return false;
    }
    const uint64_t encoded_offset = static_byte_offset / offset_unit_byte_count;
    if (encoded_offset > offset_unsigned_max) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE;
      return false;
    }
    access->immediate_offset = (uint32_t)encoded_offset;
    access->scalar_byte_offset = 0;
    if (!loom_amdgpu_source_memory_offset_fits_u32(
            access->source, access->static_byte_offset)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
      return false;
    }
    return true;
  }
  if (offset_unit_byte_count != 1) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }

  uint64_t immediate_offset = iree_min(static_byte_offset, offset_unsigned_max);
  if (access->payload_register_count == 4) {
    immediate_offset &= ~UINT64_C(15);
  }
  const uint64_t scalar_byte_offset = static_byte_offset - immediate_offset;
  if (scalar_byte_offset > UINT32_MAX) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE;
    return false;
  }
  access->immediate_offset = (uint32_t)immediate_offset;
  access->scalar_byte_offset = (uint32_t)scalar_byte_offset;
  if (!loom_amdgpu_source_memory_offset_fits_u32(access->source,
                                                 access->static_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }
  return true;
}

static bool loom_amdgpu_memory_access_split_lds_default_static_offset(
    loom_amdgpu_memory_access_t* access, uint32_t offset_unit_byte_count,
    uint64_t offset_unsigned_max,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (access->static_byte_offset < 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET;
    return false;
  }
  if (offset_unit_byte_count == 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  if (!loom_amdgpu_source_memory_offset_fits_u32(access->source,
                                                 access->static_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }

  const uint64_t static_byte_offset = (uint64_t)access->static_byte_offset;
  const bool static_fits_immediate =
      (static_byte_offset % offset_unit_byte_count) == 0 &&
      (static_byte_offset / offset_unit_byte_count) <= offset_unsigned_max;
  access->immediate_offset =
      static_fits_immediate
          ? (int64_t)(static_byte_offset / offset_unit_byte_count)
          : 0;
  access->secondary_immediate_offset = 0;
  access->vaddr_static_byte_offset =
      static_fits_immediate ? 0 : (uint32_t)static_byte_offset;
  access->scalar_byte_offset = 0;
  return true;
}

static bool loom_amdgpu_memory_access_split_global_saddr_static_offset(
    loom_amdgpu_memory_access_t* access,
    const loom_amdgpu_descriptor_offset_immediate_info_t* offset_info,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (offset_info->unit_byte_count != 1 ||
      offset_info->kind != LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE |
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_UNAVAILABLE;
    return false;
  }
  const int64_t static_byte_offset = access->static_byte_offset;
  const int64_t signed_max = offset_info->unsigned_max > INT64_MAX
                                 ? INT64_MAX
                                 : (int64_t)offset_info->unsigned_max;
  if (static_byte_offset < offset_info->signed_min) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE |
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_OFFSET_RANGE;
    return false;
  }
  if (!loom_amdgpu_memory_vaddr_offset_fits_u32(access,
                                                /*static_byte_offset=*/0)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }
  if (static_byte_offset <= signed_max) {
    access->immediate_offset = static_byte_offset;
    access->scalar_byte_offset = 0;
    access->scalar_base_byte_offset = 0;
    access->scalar_offset_placement =
        LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_SOFFSET;
    return true;
  }

  uint64_t immediate_offset = (uint64_t)signed_max;
  if (access->payload_register_count == 4) {
    immediate_offset &= ~UINT64_C(15);
  }
  const uint64_t scalar_byte_offset =
      (uint64_t)static_byte_offset - immediate_offset;
  access->immediate_offset = (int64_t)immediate_offset;
  if (scalar_byte_offset <= UINT32_MAX) {
    access->scalar_byte_offset = (uint32_t)scalar_byte_offset;
    access->scalar_base_byte_offset = 0;
    access->scalar_offset_placement =
        LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_SOFFSET;
  } else {
    access->scalar_byte_offset = 0;
    access->scalar_base_byte_offset = scalar_byte_offset;
    access->scalar_offset_placement =
        LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_BASE;
  }
  return true;
}

static bool loom_amdgpu_memory_access_split_global_smem_static_offset(
    loom_amdgpu_memory_access_t* access,
    const loom_amdgpu_descriptor_offset_immediate_info_t* offset_info,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (offset_info->unit_byte_count != 1 ||
      offset_info->kind != LOOM_LOW_IMMEDIATE_KIND_UNSIGNED) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  if (access->static_byte_offset < 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_NEGATIVE_STATIC_OFFSET;
    return false;
  }

  const uint64_t static_byte_offset = (uint64_t)access->static_byte_offset;
  const uint64_t immediate_offset =
      iree_min(static_byte_offset, offset_info->unsigned_max);
  const uint64_t scalar_byte_offset = static_byte_offset - immediate_offset;
  if (scalar_byte_offset > UINT32_MAX) {
    access->immediate_offset = 0;
    access->scalar_byte_offset = 0;
    access->scalar_base_byte_offset = static_byte_offset;
    access->scalar_offset_placement =
        LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_BASE;
    return true;
  }
  access->immediate_offset = (int64_t)immediate_offset;
  access->scalar_byte_offset = (uint32_t)scalar_byte_offset;
  if (!loom_amdgpu_source_memory_offset_fits_u32(access->source,
                                                 access->scalar_byte_offset)) {
    access->immediate_offset = 0;
    access->scalar_byte_offset = 0;
    access->scalar_base_byte_offset = static_byte_offset;
    access->scalar_offset_placement =
        LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_BASE;
    return true;
  }
  access->scalar_base_byte_offset = 0;
  access->scalar_offset_placement =
      LOOM_AMDGPU_MEMORY_SCALAR_OFFSET_PLACEMENT_SOFFSET;
  return true;
}

bool loom_amdgpu_memory_access_select_flat_offset(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal, loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  const loom_low_descriptor_t* descriptor =
      &descriptor_set->descriptors[descriptor_ordinal];
  // Flat descriptors put their byte offset first. CDNA generic offsets are
  // unsigned; global and RDNA offsets use the descriptor's signed range.
  const loom_low_immediate_kind_t offset_kind =
      descriptor_set
          ->immediates[loom_low_descriptor_layout(descriptor_set, descriptor)
                           ->immediate_start]
          .kind;
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1, offset_kind, &offset_info) ||
      offset_info.unit_byte_count != 1) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  const int64_t static_byte_offset = access->static_byte_offset;
  const int64_t signed_max = offset_info.unsigned_max > INT64_MAX
                                 ? INT64_MAX
                                 : (int64_t)offset_info.unsigned_max;
  const int64_t minimum = offset_info.kind == LOOM_LOW_IMMEDIATE_KIND_SIGNED
                              ? offset_info.signed_min
                              : 0;
  if (static_byte_offset < minimum) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE;
    return false;
  }
  if (static_byte_offset <= signed_max) {
    access->immediate_offset = static_byte_offset;
    access->vaddr_static_byte_offset = 0;
  } else {
    access->immediate_offset = 0;
    access->vaddr_static_byte_offset = (uint64_t)static_byte_offset;
  }
  access->scalar_byte_offset = 0;
  return true;
}

static bool loom_amdgpu_memory_access_split_scratch_vaddr_static_offset(
    loom_amdgpu_memory_access_t* access,
    const loom_amdgpu_descriptor_offset_immediate_info_t* offset_info,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (offset_info->unit_byte_count != 1 ||
      offset_info->kind != LOOM_LOW_IMMEDIATE_KIND_SIGNED) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  const int64_t static_byte_offset = access->static_byte_offset;
  const int64_t signed_max = offset_info->unsigned_max > INT64_MAX
                                 ? INT64_MAX
                                 : (int64_t)offset_info->unsigned_max;
  // The instruction adds its signed immediate to an unsigned u32 VADDR. Keep
  // the static term in the immediate only when the remaining VADDR is valid;
  // otherwise materialize the complete proven address through VADDR.
  if (!loom_amdgpu_memory_vaddr_offset_fits_u32(access, static_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }
  if (static_byte_offset >= offset_info->signed_min &&
      static_byte_offset <= signed_max &&
      loom_amdgpu_memory_vaddr_offset_fits_u32(access,
                                               /*static_byte_offset=*/0)) {
    access->immediate_offset = static_byte_offset;
    access->vaddr_static_byte_offset = 0;
  } else {
    access->immediate_offset = 0;
    access->vaddr_static_byte_offset = (uint64_t)static_byte_offset;
  }
  access->scalar_byte_offset = 0;
  return true;
}

static bool loom_amdgpu_memory_access_split_ds2_static_offset(
    loom_amdgpu_memory_access_t* access, uint32_t offset_unit_byte_count,
    uint64_t offset_unsigned_max,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (access->static_byte_offset < 0) {
    return false;
  }
  int64_t secondary_byte_offset = 0;
  if (!iree_checked_add_i64(access->static_byte_offset,
                            access->source->vector_lane_byte_stride,
                            &secondary_byte_offset) ||
      secondary_byte_offset < 0) {
    return false;
  }

  const uint64_t byte_offsets[] = {
      (uint64_t)access->static_byte_offset,
      (uint64_t)secondary_byte_offset,
  };
  uint32_t encoded_offsets[IREE_ARRAYSIZE(byte_offsets)] = {0};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(byte_offsets); ++i) {
    if ((byte_offsets[i] % offset_unit_byte_count) != 0) {
      return false;
    }
    const uint64_t encoded_offset = byte_offsets[i] / offset_unit_byte_count;
    if (encoded_offset > offset_unsigned_max) {
      return false;
    }
    encoded_offsets[i] = (uint32_t)encoded_offset;
  }

  access->immediate_offset = encoded_offsets[0];
  access->secondary_immediate_offset = encoded_offsets[1];
  access->scalar_byte_offset = 0;
  if (!loom_amdgpu_source_memory_offset_fits_u32(access->source,
                                                 secondary_byte_offset)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }
  return true;
}

static bool loom_amdgpu_select_ds2_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_access_t* access,
    loom_low_source_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  loom_amdgpu_memory_descriptor_candidate_key_t candidate_key = 0;
  if (!loom_amdgpu_memory_descriptor_candidate_key(
          LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
          LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR, kind,
          access->packet_byte_count, access->payload_register_class,
          access->payload_format, access->payload_register_count,
          &candidate_key)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING;
    return false;
  }
  bool found_kind_descriptor = false;
  const loom_amdgpu_memory_descriptor_candidate_range_t* range =
      loom_amdgpu_find_memory_descriptor_candidate_range(candidate_key);
  if (range == NULL) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING;
    return false;
  }
  IREE_ASSERT((iree_host_size_t)range->first_candidate +
                  range->candidate_count <=
              IREE_ARRAYSIZE(kAmdgpuMemoryDescriptorCandidates));
  const iree_host_size_t candidate_end =
      (iree_host_size_t)range->first_candidate + range->candidate_count;
  for (iree_host_size_t i = range->first_candidate; i < candidate_end; ++i) {
    const loom_amdgpu_memory_descriptor_candidate_t* candidate =
        &kAmdgpuMemoryDescriptorCandidates[i];
    const uint32_t descriptor_ordinal = loom_amdgpu_descriptor_ref_ordinal(
        descriptor_set, access->preserve_inactive
                            ? candidate->preserving_descriptor_ref
                            : candidate->descriptor_ref);
    if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
      continue;
    }
    found_kind_descriptor = true;
    loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
    if (!loom_amdgpu_descriptor_offset_immediate_info(
            descriptor_set, descriptor_ordinal, 2,
            LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
      return false;
    }
    if (!loom_amdgpu_memory_access_split_ds2_static_offset(
            access, offset_info.unit_byte_count, offset_info.unsigned_max,
            diagnostic)) {
      continue;
    }
    access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_2ADDR;
    access->descriptor = loom_low_descriptor_set_descriptor_at(
        descriptor_set, descriptor_ordinal);
    IREE_ASSERT(access->descriptor != NULL);
    return true;
  }

  diagnostic->rejection_bits |=
      found_kind_descriptor
          ? LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_RANGE
          : LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING;
  return false;
}

static bool loom_amdgpu_try_select_ds_addtid_memory_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_module_t* module, loom_func_like_t source_function,
    const loom_target_bundle_t* bundle, loom_amdgpu_memory_access_t* access,
    loom_low_source_memory_operation_kind_t kind) {
  loom_target_workgroup_size_t workgroup_size = {0};
  const uint32_t wavefront_size = bundle != NULL && bundle->snapshot != NULL
                                      ? bundle->snapshot->subgroup_size
                                      : 0;
  if (wavefront_size == 0 ||
      !loom_amdgpu_required_workgroup_size(module, source_function, bundle,
                                           &workgroup_size) ||
      workgroup_size.x == 0 || workgroup_size.x > wavefront_size ||
      workgroup_size.y != 1 || workgroup_size.z != 1) {
    return false;
  }
  if (access->payload_register_count != 1 ||
      access->source->dynamic_term_count != 1 ||
      access->dynamic_term_kinds[0] != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR ||
      access->source->dynamic_terms[0].source !=
          LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID ||
      access->source->dynamic_terms[0].stride_value_count != 0 ||
      access->source->dynamic_terms[0].dimension != LOOM_KERNEL_DIMENSION_X ||
      access->source->dynamic_terms[0].byte_stride != 1 ||
      !loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
    return false;
  }

  const loom_low_descriptor_t* descriptor = NULL;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_memory_descriptor_candidate(
          descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
          LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID, kind,
          access->packet_byte_count, access->payload_register_class,
          access->payload_format, access->payload_register_count,
          access->preserve_inactive, &descriptor, &descriptor_ordinal)) {
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
    return false;
  }

  loom_amdgpu_memory_access_diagnostic_t ignored_diagnostic = {0};
  if (!loom_amdgpu_memory_access_split_static_offset(
          access, offset_info.unit_byte_count, offset_info.unsigned_max,
          &ignored_diagnostic)) {
    return false;
  }
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID;
  access->descriptor = descriptor;
  return true;
}

static iree_string_view_t loom_amdgpu_memory_ds_addtid_topology_reason_key(
    const loom_module_t* module, loom_func_like_t source_function,
    const loom_target_bundle_t* bundle) {
  const uint32_t wavefront_size = bundle != NULL && bundle->snapshot != NULL
                                      ? bundle->snapshot->subgroup_size
                                      : 0;
  if (wavefront_size == 0) {
    return IREE_SV("wavefront_size_unknown");
  }
  loom_target_workgroup_size_t workgroup_size = {0};
  if (!loom_amdgpu_required_workgroup_size(module, source_function, bundle,
                                           &workgroup_size) ||
      workgroup_size.x == 0) {
    return IREE_SV("workgroup_size_unknown");
  }
  if (workgroup_size.x > wavefront_size) {
    return IREE_SV("cross_wave_workgroup");
  }
  if (workgroup_size.y != 1 || workgroup_size.z != 1) {
    return IREE_SV("multidimensional_workgroup");
  }
  return iree_string_view_empty();
}

iree_string_view_t loom_amdgpu_memory_ds_addtid_reason_key(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_module_t* module, loom_func_like_t source_function,
    const loom_target_bundle_t* bundle,
    const loom_amdgpu_memory_access_t* access,
    loom_low_source_memory_operation_kind_t kind) {
  if (access->address_form == LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID) {
    return IREE_SV("selected");
  }
  if (access->source->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    return IREE_SV("not_applicable");
  }
  const iree_string_view_t topology_reason =
      loom_amdgpu_memory_ds_addtid_topology_reason_key(module, source_function,
                                                       bundle);
  if (!iree_string_view_is_empty(topology_reason)) {
    return topology_reason;
  }
  if (access->payload_register_count != 1) {
    return IREE_SV("payload_width");
  }
  if (access->source->dynamic_term_count != 1) {
    return IREE_SV("dynamic_term_count");
  }
  const loom_low_source_memory_dynamic_term_t* term =
      &access->source->dynamic_terms[0];
  if (access->dynamic_term_kinds[0] != LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR) {
    return IREE_SV("dynamic_term_kind");
  }
  if (term->source != LOOM_LOW_SOURCE_MEMORY_DYNAMIC_INDEX_SOURCE_WORKITEM_ID) {
    return IREE_SV("dynamic_term_source");
  }
  if (term->stride_value_count != 0) {
    return IREE_SV("dynamic_stride_value");
  }
  if (term->dimension != LOOM_KERNEL_DIMENSION_X) {
    return IREE_SV("dynamic_dimension");
  }
  if (term->byte_stride != 1) {
    return IREE_SV("dynamic_stride_bytes");
  }
  if (!loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
    return IREE_SV("vector_lane_layout");
  }

  const loom_low_descriptor_t* descriptor = NULL;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_memory_descriptor_candidate(
          descriptor_set, LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS,
          LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DS_ADDTID, kind,
          access->packet_byte_count, access->payload_register_class,
          access->payload_format, access->payload_register_count,
          access->preserve_inactive, &descriptor, &descriptor_ordinal)) {
    return IREE_SV("descriptor_missing");
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
    return IREE_SV("descriptor_offset_immediate");
  }
  loom_amdgpu_memory_access_t access_copy = *access;
  loom_amdgpu_memory_access_diagnostic_t ignored_diagnostic = {0};
  if (!loom_amdgpu_memory_access_split_static_offset(
          &access_copy, offset_info.unit_byte_count, offset_info.unsigned_max,
          &ignored_diagnostic)) {
    return IREE_SV("static_offset_range");
  }
  return IREE_SV("not_selected");
}

static bool loom_amdgpu_memory_access_offset_is_aligned(
    const loom_amdgpu_memory_access_t* access, uint32_t byte_alignment,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (byte_alignment == 0) {
    return false;
  }
  if ((access->static_byte_offset % (int64_t)byte_alignment) != 0) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_STATIC_ALIGNMENT;
    return false;
  }
  for (uint8_t i = 0; i < access->source->dynamic_term_count; ++i) {
    const loom_value_facts_t byte_facts =
        access->source->dynamic_terms[i].byte_facts;
    if (!loom_value_facts_is_zero(byte_facts) &&
        !loom_value_facts_divisible_by(byte_facts, byte_alignment)) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_B128_DYNAMIC_ALIGNMENT;
      diagnostic->dynamic_term_index = i;
      return false;
    }
  }
  return true;
}

static void loom_amdgpu_memory_access_record_descriptor_missing(
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  diagnostic->rejection_bits |=
      access->payload_format ==
              LOOM_AMDGPU_MEMORY_PAYLOAD_FORMAT_LOW_16BIT_FLOAT
          ? LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_LOW16_DESCRIPTOR_MISSING
          : LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_MISSING;
}

static bool loom_amdgpu_memory_access_signed_narrow_repair_is_available(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_memory_access_t* access) {
  if (loom_amdgpu_memory_access_signed_narrow_lds_repair_bit_count(access) ==
      0) {
    return true;
  }
  return loom_amdgpu_descriptor_set_has_ref(
      descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_I32_OFFSET_WIDTH_INLINE);
}

bool loom_amdgpu_memory_access_try_select_buffer(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_low_source_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (access->payload_register_count == 4 &&
      !loom_amdgpu_memory_access_offset_is_aligned(
          access, /*byte_alignment=*/16, diagnostic)) {
    return false;
  }

  const loom_low_descriptor_t* descriptor = NULL;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_buffer_memory_descriptor(
          descriptor_set, access, LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT, kind,
          &descriptor, &descriptor_ordinal)) {
    loom_amdgpu_memory_access_record_descriptor_missing(access, diagnostic);
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  if (!loom_amdgpu_memory_access_split_static_offset(
          access, offset_info.unit_byte_count, offset_info.unsigned_max,
          diagnostic)) {
    return false;
  }
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
  access->descriptor = descriptor;
  loom_amdgpu_memory_access_try_select_buffer_off_zero(descriptor_set, kind,
                                                       access);
  return true;
}

static bool loom_amdgpu_memory_access_uses_only_scalar_address_terms(
    const loom_amdgpu_memory_access_t* access) {
  for (uint8_t i = 0; i < access->source->dynamic_term_count; ++i) {
    if (access->dynamic_term_kinds[i] !=
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET) {
      return false;
    }
  }
  return true;
}

static bool
loom_amdgpu_memory_access_promote_scalar_materializable_terms_to_soffset(
    const loom_amdgpu_memory_dynamic_term_materialization_plan_t*
        materialization_plan,
    loom_amdgpu_memory_access_t* access) {
  for (uint8_t i = 0; i < access->source->dynamic_term_count; ++i) {
    if (access->dynamic_term_kinds[i] ==
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET) {
      continue;
    }
    if (access->dynamic_term_kinds[i] !=
        LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR) {
      return false;
    }
    if (!materialization_plan->terms[i].can_materialize_soffset) {
      return false;
    }
    access->dynamic_term_kinds[i] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_SOFFSET;
  }
  return true;
}

static bool loom_amdgpu_memory_access_try_select_global_smem(
    const loom_amdgpu_memory_packet_selection_context_t* selection_context,
    loom_low_source_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access) {
  loom_amdgpu_memory_access_t candidate = *access;
  // Scalar memory discards the address's two low bits. Packed element types
  // may require less alignment than the register-sized packets carrying them.
  if (kind != LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD ||
      candidate.source->minimum_alignment < 4 ||
      (candidate.source->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
       candidate.source->memory_space !=
           LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT) ||
      loom_amdgpu_memory_cache_policy_is_present(
          &candidate.source->cache_policy) ||
      candidate.source->read_visibility_scope != LOOM_ATOMIC_SCOPE_THREAD ||
      !loom_amdgpu_memory_access_promote_scalar_materializable_terms_to_soffset(
          selection_context->materialization_plan, &candidate) ||
      !loom_amdgpu_memory_access_uses_only_scalar_address_terms(&candidate) ||
      !selection_context->view_regions ||
      !loom_view_region_table_root_is_stable(
          selection_context->view_regions, candidate.source->root_value_id,
          candidate.source->alias_scope_id, candidate.source->memory_space)) {
    return false;
  }

  candidate.payload_register_class =
      LOOM_AMDGPU_MEMORY_PAYLOAD_REGISTER_CLASS_SGPR;
  const loom_low_descriptor_t* descriptor = NULL;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_global_smem_memory_descriptor(
          selection_context->descriptor_set, &candidate, kind, &descriptor,
          &descriptor_ordinal)) {
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          selection_context->descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
    return false;
  }
  loom_amdgpu_memory_access_diagnostic_t ignored_diagnostic = {0};
  if (!loom_amdgpu_memory_access_split_global_smem_static_offset(
          &candidate, &offset_info, &ignored_diagnostic)) {
    return false;
  }
  candidate.address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SMEM;
  candidate.descriptor = descriptor;
  *access = candidate;
  return true;
}

static bool loom_amdgpu_memory_access_try_select_global_saddr(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_low_source_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  const loom_low_descriptor_t* descriptor = NULL;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_global_saddr_memory_descriptor(
          descriptor_set, access, kind, &descriptor, &descriptor_ordinal)) {
    loom_amdgpu_memory_access_record_descriptor_missing(access, diagnostic);
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_UNAVAILABLE;
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1, LOOM_LOW_IMMEDIATE_KIND_SIGNED,
          &offset_info)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE |
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_GLOBAL_FALLBACK_UNAVAILABLE;
    return false;
  }
  if (!loom_amdgpu_memory_access_split_global_saddr_static_offset(
          access, &offset_info, diagnostic)) {
    return false;
  }
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR;
  access->descriptor = descriptor;
  return true;
}

static bool loom_amdgpu_memory_dynamic_term_can_flat_address(
    const loom_module_t* module,
    const loom_low_source_memory_dynamic_term_t* term) {
  return term->byte_stride != 0 &&
         loom_amdgpu_memory_dynamic_term_can_materialize_vaddr(module, term);
}

void loom_amdgpu_memory_access_record_flat_dynamic_address_rejection(
    const loom_module_t* module,
    const loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  diagnostic->rejection_bits |=
      LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_FLAT_DYNAMIC_ADDRESS;
  diagnostic->dynamic_term_index = 0;
  for (uint8_t i = 0; i < source->dynamic_term_count; ++i) {
    if (!loom_amdgpu_memory_dynamic_term_can_flat_address(
            module, &source->dynamic_terms[i])) {
      diagnostic->dynamic_term_index = i;
      return;
    }
  }
}

static bool loom_amdgpu_memory_access_dynamic_terms_can_flat_address(
    const loom_module_t* module, const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  for (uint8_t i = 0; i < access->source->dynamic_term_count; ++i) {
    if (!loom_amdgpu_memory_dynamic_term_can_flat_address(
            module, &access->source->dynamic_terms[i])) {
      loom_amdgpu_memory_access_record_flat_dynamic_address_rejection(
          module, access->source, diagnostic);
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_memory_access_dynamic_terms_can_vaddr_from_plan(
    const loom_amdgpu_memory_dynamic_term_materialization_plan_t*
        materialization_plan,
    const loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  for (uint8_t i = 0; i < access->source->dynamic_term_count; ++i) {
    if (!materialization_plan->terms[i].can_materialize_vaddr) {
      diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_INDEX_SOURCE;
      diagnostic->dynamic_term_index = i;
      return false;
    }
  }
  return true;
}

void loom_amdgpu_memory_access_route_dynamic_terms_through_vaddr(
    loom_amdgpu_memory_access_t* access) {
  for (uint8_t i = 0; i < access->source->dynamic_term_count; ++i) {
    access->dynamic_term_kinds[i] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_VADDR;
  }
}

bool loom_amdgpu_memory_access_select_u32_vaddr_byte_offset(
    const loom_module_t* module,
    const loom_amdgpu_source_alloca_layout_t* alloca_layout,
    loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_memory_access_t* out_access,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  *out_access = loom_amdgpu_memory_access_from_source(source);
  *out_diagnostic = (loom_amdgpu_memory_access_diagnostic_t){0};

  if (!loom_amdgpu_memory_access_static_byte_offset_is_usable(
          source->static_byte_offset, out_diagnostic)) {
    return false;
  }
  if (!loom_amdgpu_memory_access_include_alloca_root_byte_offset(
          alloca_layout, source, out_diagnostic)) {
    return false;
  }
  out_access->static_byte_offset = source->static_byte_offset;
  if (!loom_amdgpu_memory_access_select_vaddr_dynamic_terms(module, out_access,
                                                            out_diagnostic)) {
    return false;
  }
  if ((uint64_t)out_access->static_byte_offset > UINT32_MAX ||
      !loom_amdgpu_memory_vaddr_offset_fits_u32(
          out_access, out_access->static_byte_offset)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DYNAMIC_OFFSET_RANGE;
    return false;
  }

  out_access->vaddr_static_byte_offset =
      (uint64_t)out_access->static_byte_offset;
  return true;
}

static bool loom_amdgpu_memory_access_select_flat_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_low_source_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  const loom_low_descriptor_t* descriptor = NULL;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_flat_memory_descriptor(
          descriptor_set, access, kind, &descriptor, &descriptor_ordinal)) {
    loom_amdgpu_memory_access_record_descriptor_missing(access, diagnostic);
    return false;
  }
  if (!loom_amdgpu_memory_access_select_flat_offset(
          descriptor_set, descriptor_ordinal, access, diagnostic)) {
    return false;
  }
  access->descriptor = descriptor;
  return true;
}

static bool loom_amdgpu_memory_access_try_select_flat(
    const loom_module_t* module,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_low_source_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (!loom_amdgpu_memory_access_dynamic_terms_can_flat_address(module, access,
                                                                diagnostic)) {
    return false;
  }

  if (!loom_amdgpu_memory_access_select_flat_descriptor(descriptor_set, kind,
                                                        access, diagnostic)) {
    return false;
  }
  loom_amdgpu_memory_access_route_dynamic_terms_through_vaddr(access);
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT;
  return true;
}

static bool loom_amdgpu_memory_access_try_select_scratch_vaddr(
    const loom_amdgpu_memory_packet_selection_context_t* selection_context,
    loom_low_source_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (!loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }
  if (!loom_amdgpu_memory_access_dynamic_terms_can_vaddr_from_plan(
          selection_context->materialization_plan, access, diagnostic)) {
    return false;
  }

  const loom_low_descriptor_t* descriptor = NULL;
  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_scratch_memory_descriptor(
          selection_context->descriptor_set, access, kind, &descriptor,
          &descriptor_ordinal)) {
    loom_amdgpu_memory_access_record_descriptor_missing(access, diagnostic);
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          selection_context->descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_SIGNED, &offset_info)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  if (!loom_amdgpu_memory_access_split_scratch_vaddr_static_offset(
          access, &offset_info, diagnostic)) {
    return false;
  }
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_SCRATCH_VADDR;
  access->descriptor = descriptor;
  return true;
}

static bool loom_amdgpu_memory_access_try_select_lds_default(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_low_source_memory_operation_kind_t kind,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  if (!loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }

  uint32_t descriptor_ordinal = LOOM_LOW_DESCRIPTOR_ORDINAL_NONE;
  if (!loom_amdgpu_select_memory_descriptor(descriptor_set, access, kind,
                                            &access->descriptor,
                                            &descriptor_ordinal)) {
    loom_amdgpu_memory_access_record_descriptor_missing(access, diagnostic);
    return false;
  }
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info;
  if (!loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1,
          LOOM_LOW_IMMEDIATE_KIND_UNSIGNED, &offset_info)) {
    diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_DESCRIPTOR_OFFSET_IMMEDIATE;
    return false;
  }
  return loom_amdgpu_memory_access_split_lds_default_static_offset(
      access, offset_info.unit_byte_count, offset_info.unsigned_max,
      diagnostic);
}

static loom_amdgpu_memory_address_attempt_result_t
loom_amdgpu_memory_address_attempt_apply(
    const loom_amdgpu_memory_address_attempt_t* attempt,
    const loom_amdgpu_memory_packet_selection_context_t* selection_context,
    loom_low_source_memory_operation_kind_t kind, bool allow_global_smem,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  switch (attempt->kind) {
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_ADDTID:
      if (iree_any_bit_set(
              selection_context->instruction_constraints,
              LOOM_AMDGPU_INSTRUCTION_CONSTRAINT_DS_ADDTID_ADDRESS_MATERIALIZATION)) {
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
      }
      return loom_amdgpu_try_select_ds_addtid_memory_descriptor(
                 selection_context->descriptor_set, selection_context->module,
                 selection_context->source_function, selection_context->bundle,
                 access, kind)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_DS_2ADDR:
      if (loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
      }
      if (!loom_amdgpu_memory_access_is_ds_pair(access)) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
      }
      return loom_amdgpu_select_ds2_memory_descriptor(
                 selection_context->descriptor_set, access, kind, diagnostic)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_LDS_DEFAULT:
      return loom_amdgpu_memory_access_try_select_lds_default(
                 selection_context->descriptor_set, kind, access, diagnostic)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_BUFFER_RESOURCE:
      if (selection_context->root_prefers_vgpr) {
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
      }
      if (!loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
      }
      return loom_amdgpu_memory_access_try_select_buffer(
                 selection_context->descriptor_set, kind, access, diagnostic)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_SMEM:
      if (!allow_global_smem || access->preserve_inactive ||
          selection_context->root_prefers_vgpr) {
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
      }
      if (!loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
      }
      return loom_amdgpu_memory_access_try_select_global_smem(selection_context,
                                                              kind, access)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_GLOBAL_SADDR:
      if (selection_context->root_prefers_vgpr) {
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
      }
      if (!loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
      }
      return loom_amdgpu_memory_access_try_select_global_saddr(
                 selection_context->descriptor_set, kind, access, diagnostic)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_FLAT:
      if (!loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
        diagnostic->rejection_bits |=
            LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
        return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
      }
      return loom_amdgpu_memory_access_try_select_flat(
                 selection_context->module, selection_context->descriptor_set,
                 kind, access, diagnostic)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_NOT_APPLICABLE;
    case LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SCRATCH_VADDR:
      return loom_amdgpu_memory_access_try_select_scratch_vaddr(
                 selection_context, kind, access, diagnostic)
                 ? LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED
                 : LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
  }
  return LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED;
}

static bool loom_amdgpu_memory_access_select_address_form(
    const loom_amdgpu_memory_packet_selection_context_t* selection_context,
    loom_amdgpu_memory_descriptor_domain_t descriptor_domain,
    loom_low_source_memory_operation_kind_t kind, bool allow_global_smem,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* diagnostic) {
  const loom_amdgpu_memory_address_attempt_t* attempts =
      kAmdgpuGlobalAddressAttempts;
  iree_host_size_t attempt_count = IREE_ARRAYSIZE(kAmdgpuGlobalAddressAttempts);
  if (descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS) {
    attempts = kAmdgpuLdsAddressAttempts;
    attempt_count = IREE_ARRAYSIZE(kAmdgpuLdsAddressAttempts);
  } else if (descriptor_domain ==
             LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_SCRATCH) {
    attempts = kAmdgpuScratchAddressAttempts;
    attempt_count = IREE_ARRAYSIZE(kAmdgpuScratchAddressAttempts);
  } else if (descriptor_domain ==
             LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_GENERIC_FLAT) {
    attempts = kAmdgpuFlatAddressAttempts;
    attempt_count = IREE_ARRAYSIZE(kAmdgpuFlatAddressAttempts);
  } else if (access->source->memory_space ==
             LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR) {
    attempts = kAmdgpuDescriptorAddressAttempts;
    attempt_count = IREE_ARRAYSIZE(kAmdgpuDescriptorAddressAttempts);
  }
  for (iree_host_size_t i = 0; i < attempt_count; ++i) {
    const loom_amdgpu_memory_address_attempt_result_t result =
        loom_amdgpu_memory_address_attempt_apply(
            &attempts[i], selection_context, kind, allow_global_smem, access,
            diagnostic);
    if (result == LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_SELECTED) {
      return true;
    }
    if (result == LOOM_AMDGPU_MEMORY_ADDRESS_ATTEMPT_REJECTED) {
      return false;
    }
  }
  return false;
}

bool loom_amdgpu_memory_access_select_packet(
    const loom_amdgpu_memory_packet_selection_context_t* selection_context,
    loom_low_source_memory_operation_kind_t kind, bool allow_global_smem,
    loom_amdgpu_memory_access_t* access,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT;
  access->descriptor = NULL;
  for (uint8_t i = 0; i < IREE_ARRAYSIZE(access->dynamic_term_kinds); ++i) {
    access->dynamic_term_kinds[i] = LOOM_AMDGPU_MEMORY_DYNAMIC_INDEX_NONE;
  }
  if (access->payload_register_count == 0 ||
      access->payload_register_count > LOOM_AMDGPU_MAX_MEMORY_32BIT_LANES) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_TYPE;
    return false;
  }

  if (access->source->vector_lane_byte_stride <= 0 ||
      access->source->vector_lane_byte_stride > UINT32_MAX) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }
  if (!loom_amdgpu_memory_access_static_byte_offset_is_usable(
          access->static_byte_offset, out_diagnostic)) {
    return false;
  }

  if (!loom_amdgpu_memory_access_select_dynamic_term_kinds_from_plan(
          selection_context->materialization_plan, access, out_diagnostic)) {
    return false;
  }

  if (selection_context->requires_scalar_payload) {
    if (!loom_amdgpu_memory_access_has_contiguous_vector_lanes(access)) {
      out_diagnostic->rejection_bits |=
          LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
      return false;
    }
    if (loom_amdgpu_memory_access_try_select_global_smem(selection_context,
                                                         kind, access)) {
      return true;
    }
    loom_amdgpu_memory_access_record_descriptor_missing(access, out_diagnostic);
    return false;
  }

  loom_amdgpu_memory_descriptor_domain_t descriptor_domain;
  if (!loom_amdgpu_memory_descriptor_domain_from_memory_space(
          access->source->memory_space, &descriptor_domain)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE;
    return false;
  }

  if (descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_LDS ||
      descriptor_domain == LOOM_AMDGPU_MEMORY_DESCRIPTOR_DOMAIN_SCRATCH) {
    loom_amdgpu_memory_access_route_dynamic_terms_through_vaddr(access);
  }
  if (!loom_amdgpu_memory_access_signed_narrow_repair_is_available(
          selection_context->descriptor_set, access)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_SIGNED_NARROW_REPAIR_DESCRIPTOR_MISSING;
    return false;
  }

  return loom_amdgpu_memory_access_select_address_form(
      selection_context, descriptor_domain, kind, allow_global_smem, access,
      out_diagnostic);
}

bool loom_amdgpu_memory_access_select_flat_address(
    const loom_module_t* module,
    const loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_memory_access_t* out_access,
    loom_amdgpu_memory_access_diagnostic_t* out_diagnostic) {
  *out_access = loom_amdgpu_memory_access_from_source(source);
  *out_diagnostic = (loom_amdgpu_memory_access_diagnostic_t){0};
  if (source->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL &&
      source->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC &&
      source->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT &&
      source->memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_MEMORY_SPACE;
    return false;
  }
  if (out_access->source->vector_lane_byte_stride <= 0 ||
      out_access->source->vector_lane_byte_stride > UINT32_MAX ||
      !loom_amdgpu_memory_access_has_contiguous_vector_lanes(out_access)) {
    out_diagnostic->rejection_bits |=
        LOOM_AMDGPU_MEMORY_ACCESS_REJECTION_VECTOR_AXIS_STRIDE;
    return false;
  }
  if (!loom_amdgpu_memory_access_static_byte_offset_is_usable(
          out_access->static_byte_offset, out_diagnostic)) {
    return false;
  }
  if (!loom_amdgpu_memory_access_dynamic_terms_can_flat_address(
          module, out_access, out_diagnostic)) {
    return false;
  }
  loom_amdgpu_memory_access_route_dynamic_terms_through_vaddr(out_access);
  out_access->immediate_offset = 0;
  out_access->secondary_immediate_offset = 0;
  out_access->vaddr_static_byte_offset =
      (uint64_t)out_access->static_byte_offset;
  out_access->scalar_byte_offset = 0;
  out_access->address_form = LOOM_AMDGPU_MEMORY_ADDRESS_FORM_FLAT;
  return true;
}
