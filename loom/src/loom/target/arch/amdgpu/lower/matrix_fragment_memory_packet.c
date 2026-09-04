// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_packet.h"

#include <stdint.h>

#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/encoding/float16.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_address.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_publication_cost.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/util/fact_table.h"

enum {
  LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_PACKET_REGISTERS = 4,
};

static const uint16_t kLoomAmdgpuFragmentMemoryPacketCandidates[] = {4, 3, 2,
                                                                     1};

static const uint16_t kLoomAmdgpuFragmentMemoryNarrowedStoreCandidates[] = {
    8, 6, 4, 2, 1};

typedef enum loom_amdgpu_fragment_memory_domain_e {
  LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_GLOBAL = 0,
  LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_DESCRIPTOR = 1,
  LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_WORKGROUP = 2,
  LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_COUNT_,
} loom_amdgpu_fragment_memory_domain_t;

typedef struct loom_amdgpu_fragment_memory_descriptor_pair_t {
  // Descriptor writing the low 16-bit destination register half.
  loom_amdgpu_descriptor_ref_t low_ref;
  // Tied descriptor writing the high 16-bit destination register half.
  loom_amdgpu_descriptor_ref_t high_ref;
} loom_amdgpu_fragment_memory_descriptor_pair_t;

typedef struct loom_amdgpu_fragment_memory_descriptor_table_t {
  // Descriptor refs for normal 32-bit-register packet payloads, indexed by
  // operation kind and packet register count.
  loom_amdgpu_descriptor_ref_t
      packet_refs[LOOM_LOW_SOURCE_MEMORY_OPERATION_COUNT_]
                 [LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_PACKET_REGISTERS + 1u];
  // Descriptor refs for scalar 16-bit packets, indexed by operation kind.
  loom_amdgpu_descriptor_ref_t
      b16_refs[LOOM_LOW_SOURCE_MEMORY_OPERATION_COUNT_];
  // Paired D16 descriptors that directly construct a packed B16 load result.
  loom_amdgpu_fragment_memory_descriptor_pair_t packed_b16_load;
} loom_amdgpu_fragment_memory_descriptor_table_t;

static const loom_amdgpu_fragment_memory_descriptor_table_t
    kFragmentMemoryDescriptorTables[LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_COUNT_] = {
        [LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_GLOBAL] =
            {
                .packet_refs =
                    {
                        [LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD] =
                            {
                                LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B32_SADDR,
                                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR,
                                LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B128_SADDR,
                            },
                        [LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE] =
                            {
                                LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B32_SADDR,
                                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B64_SADDR,
                                LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B128_SADDR,
                            },
                    },
                .b16_refs =
                    {
                        [LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD] =
                            LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B16_D16_SADDR,
                        [LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE] =
                            LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B16_SADDR,
                    },
                .packed_b16_load =
                    {
                        .low_ref =
                            LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B16_D16_SADDR,
                        .high_ref =
                            LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B16_D16_HI_SADDR,
                    },
            },
        [LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_DESCRIPTOR] =
            {
                .packet_refs =
                    {
                        [LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD] =
                            {
                                LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                                LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_LOAD_DWORD,
                                LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_LOAD_B64,
                                LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_LOAD_B96,
                                LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_LOAD_B128,
                            },
                        [LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE] =
                            {
                                LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                                LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_STORE_DWORD,
                                LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_STORE_B64,
                                LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_STORE_B96,
                                LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_STORE_B128,
                            },
                    },
                .b16_refs =
                    {
                        [LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD] =
                            LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_LOAD_B16_D16,
                        [LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE] =
                            LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_STORE_B16,
                    },
                .packed_b16_load =
                    {
                        .low_ref =
                            LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_LOAD_B16_D16,
                        .high_ref =
                            LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_LOAD_B16_D16_HI,
                    },
            },
        [LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_WORKGROUP] =
            {
                .packet_refs =
                    {
                        [LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD] =
                            {
                                LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                                LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B32,
                                LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B64,
                                LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B96,
                                LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B128,
                            },
                        [LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE] =
                            {
                                LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                                LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B32,
                                LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B64,
                                LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B96,
                                LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B128,
                            },
                    },
                .b16_refs =
                    {
                        [LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD] =
                            LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_U16,
                        [LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE] =
                            LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B16,
                    },
                .packed_b16_load =
                    {
                        .low_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_LOAD_U16_D16,
                        .high_ref =
                            LOOM_AMDGPU_DESCRIPTOR_REF_DS_LOAD_U16_D16_HI,
                    },
            },
};

bool loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
    loom_amdgpu_fragment_memory_payload_form_t payload_form) {
  return payload_form ==
             LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_BF16 ||
         payload_form ==
             LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_F16;
}

bool loom_amdgpu_fragment_memory_payload_form_is_store_narrow_f32_to_16bit(
    loom_amdgpu_fragment_memory_payload_form_t payload_form) {
  return payload_form ==
             LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16 ||
         payload_form ==
             LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_F16;
}

loom_scalar_type_t loom_amdgpu_fragment_memory_store_narrow_result_element_type(
    loom_amdgpu_fragment_memory_payload_form_t payload_form) {
  switch (payload_form) {
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16:
      return LOOM_SCALAR_TYPE_BF16;
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_F16:
      return LOOM_SCALAR_TYPE_F16;
    default:
      return LOOM_SCALAR_TYPE_NONE;
  }
}

loom_scalar_type_t loom_amdgpu_fragment_memory_load_fp8_result_element_type(
    loom_amdgpu_fragment_memory_payload_form_t payload_form) {
  switch (payload_form) {
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_BF16:
      return LOOM_SCALAR_TYPE_BF16;
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_F16:
      return LOOM_SCALAR_TYPE_F16;
    default:
      return LOOM_SCALAR_TYPE_NONE;
  }
}

static bool loom_amdgpu_fragment_memory_packets_support_fp8_e8m0_pk8(
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  for (uint16_t i = 0; i < plan->packet_count; ++i) {
    const loom_amdgpu_fragment_memory_packet_plan_t* packet = &plan->packets[i];
    if (packet->result_register_count == 0 ||
        packet->result_register_count % 4u != 0 ||
        packet->packet_register_count < packet->result_register_count / 2u) {
      return false;
    }
  }
  return true;
}

typedef uint32_t loom_amdgpu_fp8_16bit_capabilities_t;

enum loom_amdgpu_fp8_16bit_capability_bits_e {
  LOOM_AMDGPU_FP8_16BIT_CAPABILITY_NONE = 0u,
  LOOM_AMDGPU_FP8_16BIT_CAPABILITY_IDENTITY_E8M0_PK8_BF16 = 1u << 0,
  LOOM_AMDGPU_FP8_16BIT_CAPABILITY_SCALEF32_BF16_PAIR = 1u << 1,
  LOOM_AMDGPU_FP8_16BIT_CAPABILITY_IDENTITY_E8M0_PK8_F16 = 1u << 2,
  LOOM_AMDGPU_FP8_16BIT_CAPABILITY_SCALEF32_F16_PAIR = 1u << 3,
  LOOM_AMDGPU_FP8_16BIT_CAPABILITY_NATIVE_F16_PAIR = 1u << 4,
};

static loom_amdgpu_fp8_decode_action_t
loom_amdgpu_fragment_memory_select_fp8_to_bf16_decode_action(
    const loom_amdgpu_fp8_decode_plan_t* decode_plan,
    loom_amdgpu_fp8_16bit_capabilities_t capabilities,
    loom_amdgpu_fp8_decode_value_flags_t decode_value_flags) {
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_FP8_16BIT_CAPABILITY_IDENTITY_E8M0_PK8_BF16)) {
    return (loom_amdgpu_fp8_decode_action_t){
        .kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_BF16,
        .value_flags = decode_value_flags,
    };
  }
  if (iree_any_bit_set(capabilities,
                       LOOM_AMDGPU_FP8_16BIT_CAPABILITY_SCALEF32_BF16_PAIR)) {
    return (loom_amdgpu_fp8_decode_action_t){
        .kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_BF16_PAIR,
        .value_flags = decode_value_flags,
    };
  }
  const bool prefer_packed_bf16 =
      loom_amdgpu_fp8_prefers_packed_bf16_pair_decode(decode_plan,
                                                      decode_value_flags);
  if (prefer_packed_bf16) {
    return loom_amdgpu_select_fp8_packed_bf16_decode_action(decode_plan,
                                                            decode_value_flags);
  }
  const loom_amdgpu_fp8_packed_bf16_missing_requirements_t
      missing_requirements =
          loom_amdgpu_fp8_pair_to_packed_bf16_missing_requirements(
              decode_plan, decode_value_flags);
  if (iree_any_bit_set(decode_plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_F32_PAIR)) {
    const bool has_native_bf16_pack =
        iree_any_bit_set(decode_plan->flags,
                         LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_BF16_PACK);
    return (loom_amdgpu_fp8_decode_action_t){
        .kind =
            has_native_bf16_pack
                ? LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR_BF16_PACK
                : LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR,
        .value_flags = decode_value_flags,
    };
  }
  if (missing_requirements ==
      LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_NONE) {
    return loom_amdgpu_select_fp8_packed_bf16_decode_action(decode_plan,
                                                            decode_value_flags);
  }
  loom_amdgpu_fp8_decode_action_t action = {
      .kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_BF16,
      .value_flags = decode_value_flags,
  };
  if (iree_any_bit_set(
          missing_requirements,
          LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_VALUE_FINITE)) {
    action.detail_flags |=
        LOOM_AMDGPU_FP8_DECODE_ACTION_DETAIL_FLAG_MISSING_VALUE_FINITE;
  }
  if (iree_any_bit_set(
          missing_requirements,
          LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_VALUE_NOT_SUBNORMAL)) {
    action.detail_flags |=
        LOOM_AMDGPU_FP8_DECODE_ACTION_DETAIL_FLAG_MISSING_NOT_SUBNORMAL;
  }
  const loom_amdgpu_fp8_packed_bf16_missing_requirements_t target_requirements =
      LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_PERMUTE_PACKET |
      LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_PACKED_SHIFT_PACKET |
      LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_ZERO_REPAIR_PACKETS;
  if (iree_any_bit_set(missing_requirements, target_requirements)) {
    action.detail_flags |=
        LOOM_AMDGPU_FP8_DECODE_ACTION_DETAIL_FLAG_MISSING_TARGET_PACKETS;
  }
  return action;
}

static loom_amdgpu_fp8_decode_action_t
loom_amdgpu_fragment_memory_select_fp8_to_f16_decode_action(
    const loom_amdgpu_fp8_decode_plan_t* decode_plan,
    loom_amdgpu_fp8_16bit_capabilities_t capabilities,
    loom_amdgpu_fp8_decode_value_flags_t decode_value_flags) {
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_FP8_16BIT_CAPABILITY_IDENTITY_E8M0_PK8_F16)) {
    return (loom_amdgpu_fp8_decode_action_t){
        .kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F16,
        .value_flags = decode_value_flags,
    };
  }
  if (iree_any_bit_set(capabilities,
                       LOOM_AMDGPU_FP8_16BIT_CAPABILITY_SCALEF32_F16_PAIR)) {
    return (loom_amdgpu_fp8_decode_action_t){
        .kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_F16_PAIR,
        .value_flags = decode_value_flags,
    };
  }
  if (iree_any_bit_set(capabilities,
                       LOOM_AMDGPU_FP8_16BIT_CAPABILITY_NATIVE_F16_PAIR)) {
    return (loom_amdgpu_fp8_decode_action_t){
        .kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F16_PAIR,
        .value_flags = decode_value_flags,
    };
  }
  if (loom_amdgpu_can_emit_fp8_pair_to_packed_f16_finite(decode_plan,
                                                         decode_value_flags)) {
    return loom_amdgpu_select_fp8_packed_f16_decode_action(decode_plan,
                                                           decode_value_flags);
  }
  return (loom_amdgpu_fp8_decode_action_t){
      .kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NONE,
      .value_flags = decode_value_flags,
  };
}

static bool loom_amdgpu_fragment_memory_domain_from_space(
    loom_value_fact_memory_space_t memory_space,
    loom_amdgpu_fragment_memory_domain_t* out_domain) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
      *out_domain = LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_GLOBAL;
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      *out_domain = LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_DESCRIPTOR;
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      *out_domain = LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_WORKGROUP;
      return true;
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_descriptor_ref(
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_value_fact_memory_space_t memory_space, uint16_t packet_register_count,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (operation_kind >= LOOM_LOW_SOURCE_MEMORY_OPERATION_COUNT_ ||
      packet_register_count >
          LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_PACKET_REGISTERS) {
    return false;
  }
  loom_amdgpu_fragment_memory_domain_t domain =
      LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_COUNT_;
  if (!loom_amdgpu_fragment_memory_domain_from_space(memory_space, &domain)) {
    return false;
  }
  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      kFragmentMemoryDescriptorTables[domain]
          .packet_refs[operation_kind][packet_register_count];
  if (descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    return false;
  }
  *out_descriptor_ref = descriptor_ref;
  return true;
}

static bool loom_amdgpu_fragment_memory_16bit_descriptor_ref(
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_value_fact_memory_space_t memory_space,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (operation_kind >= LOOM_LOW_SOURCE_MEMORY_OPERATION_COUNT_) {
    return false;
  }
  loom_amdgpu_fragment_memory_domain_t domain =
      LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_COUNT_;
  if (!loom_amdgpu_fragment_memory_domain_from_space(memory_space, &domain)) {
    return false;
  }
  const loom_amdgpu_descriptor_ref_t descriptor_ref =
      kFragmentMemoryDescriptorTables[domain].b16_refs[operation_kind];
  if (descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    return false;
  }
  *out_descriptor_ref = descriptor_ref;
  return true;
}

static bool loom_amdgpu_fragment_memory_packed_b16_load_descriptor_refs(
    loom_value_fact_memory_space_t memory_space,
    loom_amdgpu_descriptor_ref_t* out_low_descriptor_ref,
    loom_amdgpu_descriptor_ref_t* out_high_descriptor_ref) {
  *out_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  *out_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  loom_amdgpu_fragment_memory_domain_t domain =
      LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_COUNT_;
  if (!loom_amdgpu_fragment_memory_domain_from_space(memory_space, &domain)) {
    return false;
  }
  const loom_amdgpu_fragment_memory_descriptor_pair_t pair =
      kFragmentMemoryDescriptorTables[domain].packed_b16_load;
  if (pair.low_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE ||
      pair.high_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    return false;
  }
  *out_low_descriptor_ref = pair.low_ref;
  *out_high_descriptor_ref = pair.high_ref;
  return true;
}

static bool loom_amdgpu_fragment_memory_narrowed_store_descriptor_ref(
    loom_value_fact_memory_space_t memory_space, uint16_t result_register_count,
    uint16_t* out_packet_register_count,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_packet_register_count = 0;
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (result_register_count == 1) {
    *out_packet_register_count = 1;
    return loom_amdgpu_fragment_memory_16bit_descriptor_ref(
        LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE, memory_space,
        out_descriptor_ref);
  }
  if ((result_register_count & 1u) != 0) {
    return false;
  }
  *out_packet_register_count = result_register_count / 2u;
  return loom_amdgpu_fragment_memory_descriptor_ref(
      LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE, memory_space,
      *out_packet_register_count, out_descriptor_ref);
}

bool loom_amdgpu_fragment_memory_space_supports_access(
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_value_fact_memory_space_t memory_space,
    loom_amdgpu_fragment_memory_packetization_t packetization,
    loom_amdgpu_fragment_memory_payload_form_t payload_form) {
  if (loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
          payload_form)) {
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    return operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD &&
           (loom_amdgpu_fragment_memory_descriptor_ref(
                LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD, memory_space,
                /*packet_register_count=*/1, &descriptor_ref) ||
            loom_amdgpu_fragment_memory_16bit_descriptor_ref(
                LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD, memory_space,
                &descriptor_ref));
  }

  if (packetization != LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_NATIVE) {
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    return loom_amdgpu_fragment_memory_16bit_descriptor_ref(
        operation_kind, memory_space, &descriptor_ref);
  }

  if (payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT) {
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    return loom_amdgpu_fragment_memory_16bit_descriptor_ref(
        LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD, memory_space, &descriptor_ref);
  }

  if (loom_amdgpu_fragment_memory_payload_form_is_store_narrow_f32_to_16bit(
          payload_form)) {
    for (iree_host_size_t i = 0;
         i < IREE_ARRAYSIZE(kLoomAmdgpuFragmentMemoryNarrowedStoreCandidates);
         ++i) {
      uint16_t packet_register_count = 0;
      loom_amdgpu_descriptor_ref_t descriptor_ref =
          LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
      if (loom_amdgpu_fragment_memory_narrowed_store_descriptor_ref(
              memory_space, kLoomAmdgpuFragmentMemoryNarrowedStoreCandidates[i],
              &packet_register_count, &descriptor_ref)) {
        return true;
      }
    }
    return false;
  }

  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuFragmentMemoryPacketCandidates); ++i) {
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (loom_amdgpu_fragment_memory_descriptor_ref(
            operation_kind, memory_space,
            kLoomAmdgpuFragmentMemoryPacketCandidates[i], &descriptor_ref)) {
      return true;
    }
  }
  return false;
}

bool loom_amdgpu_fragment_memory_payload_form_has_descriptors(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_fragment_memory_payload_form_t payload_form) {
  switch (payload_form) {
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE:
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT:
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_BF16:
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_F16:
      return true;
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16:
      return loom_amdgpu_bf16_descriptor_set_can_emit_f32_to_bf16_lane(
          descriptor_set);
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_F16:
      return loom_amdgpu_f16_descriptor_set_can_emit_f32_to_f16_lane(
          descriptor_set);
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_EXTEND_F16_TO_F32:
      return loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_F16);
  }
  return false;
}

static bool loom_amdgpu_fragment_memory_packet_addresses_fit_u32(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    iree_string_view_t* out_constraint_key) {
  const loom_matrix_fragment_role_layout_t* role_layout =
      loom_matrix_fragment_role_layout(layout, plan->role);
  const uint16_t register_address_count =
      plan->payload_form ==
              LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT
          ? packet->result_register_count
          : 1;
  const uint16_t element_address_count =
      loom_amdgpu_matrix_fragment_role_layout_uses_packed_b16_elements(
          plan->role, role_layout)
          ? LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT
          : 1;
  for (uint16_t register_offset = 0; register_offset < register_address_count;
       ++register_offset) {
    for (uint16_t element_index = 0; element_index < element_address_count;
         ++element_index) {
      uint64_t unused_static_byte_offset = 0;
      if (!loom_amdgpu_fragment_memory_vaddr_static_offset_u32(
              plan, packet->register_index + register_offset, element_index,
              &unused_static_byte_offset)) {
        if (out_constraint_key != NULL &&
            iree_string_view_is_empty(*out_constraint_key)) {
          *out_constraint_key = IREE_SV("fragment_memory.base_offset");
        }
        return false;
      }
    }
  }
  return true;
}

static const loom_matrix_fragment_packed_b16_publication_t*
loom_amdgpu_fragment_memory_crosslane_packed_b16_store_publication(
    const loom_amdgpu_fragment_memory_publication_query_t* query) {
  if (query->layout == NULL ||
      !loom_amdgpu_fragment_memory_payload_form_is_store_narrow_f32_to_16bit(
          query->payload_form) ||
      !loom_amdgpu_matrix_fragment_role_is_result_like(query->role) ||
      query->view_rank != LOOM_AMDGPU_FRAGMENT_UNBLOCKED_VIEW_RANK ||
      query->element_byte_count != 2) {
    return NULL;
  }

  const loom_matrix_fragment_role_layout_t* role_layout =
      loom_matrix_fragment_role_layout(query->layout, query->role);
  if (role_layout == NULL || role_layout->element_bit_count != 32) {
    return NULL;
  }

  const loom_matrix_fragment_packed_b16_publication_t* publication = NULL;
  const uint8_t row_view_axis = loom_amdgpu_matrix_fragment_role_view_axis(
      query->role, query->view_rank, query->representation_flags,
      LOOM_MATRIX_FRAGMENT_AXIS_ROW);
  const uint8_t column_view_axis = loom_amdgpu_matrix_fragment_role_view_axis(
      query->role, query->view_rank, query->representation_flags,
      LOOM_MATRIX_FRAGMENT_AXIS_COLUMN);
  if (row_view_axis != UINT8_MAX &&
      query->static_axis_byte_strides[row_view_axis] ==
          query->element_byte_count &&
      role_layout->packed_b16_publications.row
              .publishing_participant_and_mask != 0) {
    publication = &role_layout->packed_b16_publications.row;
  }
  if (column_view_axis != UINT8_MAX &&
      query->static_axis_byte_strides[column_view_axis] ==
          query->element_byte_count &&
      role_layout->packed_b16_publications.column
              .publishing_participant_and_mask != 0) {
    if (publication != NULL) {
      return NULL;
    }
    publication = &role_layout->packed_b16_publications.column;
  }
  return publication;
}

static bool loom_amdgpu_fragment_memory_select_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    loom_amdgpu_fragment_memory_packet_plan_t* out_packet) {
  *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){0};
  const uint16_t remaining = plan->register_count - register_index;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuFragmentMemoryPacketCandidates); ++i) {
    const uint16_t candidate = kLoomAmdgpuFragmentMemoryPacketCandidates[i];
    if (candidate > remaining ||
        (candidate % plan->address_layout.payload_registers_per_element) != 0 ||
        (register_index % plan->address_layout.payload_registers_per_element) !=
            0) {
      continue;
    }
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (!loom_amdgpu_fragment_memory_descriptor_ref(
            plan->operation_kind, plan->source.memory_space, candidate,
            &descriptor_ref) ||
        !loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref) ||
        !loom_amdgpu_fragment_memory_register_group_is_contiguous(
            plan, register_index, candidate,
            LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT)) {
      continue;
    }
    *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){
        .register_index = register_index,
        .result_register_count = candidate,
        .packet_register_count = candidate,
        .descriptor_ref = descriptor_ref,
    };
    return true;
  }
  return false;
}

static bool loom_amdgpu_fragment_memory_select_low_subword_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    loom_amdgpu_descriptor_ref_t preferred_descriptor_ref,
    loom_amdgpu_fragment_memory_packet_plan_t* out_packet) {
  *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){0};
  loom_amdgpu_descriptor_ref_t descriptor_ref = preferred_descriptor_ref;
  if ((descriptor_ref == LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
       !loom_amdgpu_fragment_memory_16bit_descriptor_ref(
           plan->operation_kind, plan->source.memory_space, &descriptor_ref)) ||
      !loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref)) {
    return false;
  }
  *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){
      .register_index = register_index,
      .result_register_count = 1,
      .packet_register_count = 1,
      .descriptor_ref = descriptor_ref,
  };
  return true;
}

static bool loom_amdgpu_fragment_memory_select_packed_16bit_result_load_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    loom_amdgpu_fragment_memory_packet_plan_t* out_packet) {
  *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){0};
  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_fragment_memory_16bit_descriptor_ref(
          LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD, plan->source.memory_space,
          &descriptor_ref) ||
      !loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref)) {
    return false;
  }
  const uint16_t remaining = plan->register_count - register_index;
  *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){
      .register_index = register_index,
      .result_register_count = remaining >= 2 ? 2 : 1,
      .packet_register_count = 1,
      .descriptor_ref = descriptor_ref,
  };
  return true;
}

static bool loom_amdgpu_fragment_memory_select_fp8_to_16bit_load_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    loom_amdgpu_fragment_memory_packet_plan_t* out_packet) {
  *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){0};
  const uint16_t remaining = plan->register_count - register_index;
  const uint32_t source_register_byte_count =
      (uint32_t)plan->address_layout.payload_elements_per_register *
      plan->element_byte_count;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuFragmentMemoryPacketCandidates); ++i) {
    const uint16_t packet_register_count =
        kLoomAmdgpuFragmentMemoryPacketCandidates[i];
    const uint16_t result_register_count = packet_register_count * 2u;
    if (result_register_count > remaining) {
      continue;
    }
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (!loom_amdgpu_fragment_memory_descriptor_ref(
            LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD, plan->source.memory_space,
            packet_register_count, &descriptor_ref) ||
        !loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref) ||
        !loom_amdgpu_fragment_memory_register_group_is_contiguous(
            plan, register_index, result_register_count,
            source_register_byte_count)) {
      continue;
    }
    *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){
        .register_index = register_index,
        .result_register_count = result_register_count,
        .packet_register_count = packet_register_count,
        .descriptor_ref = descriptor_ref,
    };
    return true;
  }

  if (remaining == 0) {
    return false;
  }
  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_fragment_memory_16bit_descriptor_ref(
          LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD, plan->source.memory_space,
          &descriptor_ref) ||
      !loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref)) {
    return false;
  }
  *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){
      .register_index = register_index,
      .result_register_count = 1,
      .packet_register_count = 1,
      .descriptor_ref = descriptor_ref,
  };
  return true;
}

static bool loom_amdgpu_fragment_memory_can_emit_narrowed_store_pair(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_fragment_memory_payload_form_t payload_form,
    loom_amdgpu_fragment_publication_source_flags_t source_flags) {
  if (iree_any_bit_set(source_flags,
                       LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_PACKED)) {
    return loom_amdgpu_descriptor_set_can_emit_packed_u16_lane_pair(
        descriptor_set);
  }
  switch (payload_form) {
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16:
      return loom_amdgpu_bf16_descriptor_set_can_emit_f32_pair_to_packed_bf16(
          descriptor_set);
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_F16:
      return loom_amdgpu_f16_descriptor_set_can_emit_f32_pair_to_packed_f16(
          descriptor_set);
    default:
      IREE_ASSERT_UNREACHABLE("selected AMDGPU narrowed store payload form");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static bool loom_amdgpu_fragment_memory_publication_group_is_contiguous(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    uint16_t register_index, uint16_t register_count) {
  if (register_count == 0 ||
      register_index + register_count > query->register_count) {
    return false;
  }
  const uint32_t* register_byte_offsets =
      query->address_layout->register_byte_offsets;
  const uint64_t base_byte_offset = register_byte_offsets[register_index];
  for (uint16_t i = 1; i < register_count; ++i) {
    if (register_byte_offsets[register_index + i] !=
        base_byte_offset + (uint64_t)i * query->element_byte_count) {
      return false;
    }
    for (uint8_t view_axis = 0; view_axis < query->view_rank; ++view_axis) {
      const loom_amdgpu_fragment_memory_runtime_axis_t* runtime_axis =
          &query->runtime_axes[view_axis];
      if (runtime_axis->register_coordinates[register_index + i] !=
          runtime_axis->register_coordinates[register_index]) {
        return false;
      }
    }
  }
  return true;
}

static bool loom_amdgpu_fragment_memory_publication_cost_less(
    loom_low_representation_cost_t left, loom_low_representation_cost_t right) {
  return left.runtime < right.runtime ||
         (left.runtime == right.runtime && left.code_size < right.code_size);
}

typedef enum loom_amdgpu_fragment_memory_direct_publication_mode_e {
  LOOM_AMDGPU_FRAGMENT_MEMORY_DIRECT_PUBLICATION_MODE_SCALAR = 0,
  LOOM_AMDGPU_FRAGMENT_MEMORY_DIRECT_PUBLICATION_MODE_PACKED = 1,
} loom_amdgpu_fragment_memory_direct_publication_mode_t;

static bool loom_amdgpu_fragment_memory_select_direct_publication_packet(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    loom_amdgpu_fragment_memory_direct_publication_mode_t mode,
    uint16_t register_index,
    loom_amdgpu_fragment_memory_packet_plan_t* out_packet) {
  *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){0};
  const bool can_emit_packed_pair =
      mode == LOOM_AMDGPU_FRAGMENT_MEMORY_DIRECT_PUBLICATION_MODE_PACKED &&
      loom_amdgpu_fragment_memory_can_emit_narrowed_store_pair(
          query->descriptor_set, query->payload_form, query->source_flags);
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuFragmentMemoryNarrowedStoreCandidates);
       ++i) {
    const uint16_t candidate =
        kLoomAmdgpuFragmentMemoryNarrowedStoreCandidates[i];
    if (candidate > query->register_count - register_index ||
        (candidate > 1 &&
         (!can_emit_packed_pair ||
          (!iree_any_bit_set(
               query->source_flags,
               LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_ROUNDED) &&
           ((register_index | candidate) & 1u) != 0) ||
          !loom_amdgpu_fragment_memory_publication_group_is_contiguous(
              query, register_index, candidate)))) {
      continue;
    }
    uint16_t packet_register_count = 0;
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (!loom_amdgpu_fragment_memory_narrowed_store_descriptor_ref(
            query->memory_space, candidate, &packet_register_count,
            &descriptor_ref) ||
        !loom_amdgpu_descriptor_set_has_ref(query->descriptor_set,
                                            descriptor_ref)) {
      continue;
    }
    *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){
        .register_index = register_index,
        .result_register_count = candidate,
        .packet_register_count = packet_register_count,
        .descriptor_ref = descriptor_ref,
    };
    return true;
  }
  return false;
}

static bool loom_amdgpu_fragment_memory_build_direct_publication(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    loom_amdgpu_fragment_memory_direct_publication_mode_t mode,
    loom_amdgpu_fragment_memory_packet_plan_t* out_packets,
    uint16_t* out_packet_count, bool* out_has_packed_packet) {
  *out_packet_count = 0;
  *out_has_packed_packet = false;
  for (uint16_t register_index = 0; register_index < query->register_count;) {
    loom_amdgpu_fragment_memory_packet_plan_t* packet =
        &out_packets[*out_packet_count];
    if (!loom_amdgpu_fragment_memory_select_direct_publication_packet(
            query, mode, register_index, packet)) {
      return false;
    }
    ++*out_packet_count;
    *out_has_packed_packet |= packet->result_register_count > 1;
    register_index += packet->result_register_count;
  }
  return true;
}

bool loom_amdgpu_fragment_memory_select_publication(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    loom_amdgpu_fragment_memory_publication_choice_t* out_choice) {
  if (out_choice == NULL) return false;
  *out_choice = (loom_amdgpu_fragment_memory_publication_choice_t){0};
  if (query == NULL || query->descriptor_set == NULL || query->layout == NULL ||
      query->address_layout == NULL || query->runtime_axes == NULL ||
      query->static_axis_byte_strides == NULL || query->register_count == 0 ||
      query->register_count > LOOM_AMDGPU_MAX_MATRIX_FRAGMENT_32BIT_REGISTERS ||
      !loom_amdgpu_fragment_memory_payload_form_is_store_narrow_f32_to_16bit(
          query->payload_form)) {
    return false;
  }
  const bool has_rounded_source =
      iree_any_bit_set(query->source_flags,
                       LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_ROUNDED);
  const bool has_scaled_source = iree_any_bit_set(
      query->source_flags, LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_SCALED);
  const bool has_packed_source = iree_any_bit_set(
      query->source_flags, LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_PACKED);
  if ((has_scaled_source && !has_rounded_source) ||
      (has_packed_source && (has_rounded_source || has_scaled_source))) {
    return false;
  }

  bool selected = false;

  loom_amdgpu_fragment_memory_packet_plan_t
      direct_packets[LOOM_AMDGPU_MAX_MATRIX_FRAGMENT_32BIT_REGISTERS] = {0};
  uint16_t direct_packet_count = 0;
  bool has_packed_packet = false;
  loom_low_representation_cost_t direct_cost = {0};
  if (loom_amdgpu_fragment_memory_build_direct_publication(
          query, LOOM_AMDGPU_FRAGMENT_MEMORY_DIRECT_PUBLICATION_MODE_SCALAR,
          direct_packets, &direct_packet_count, &has_packed_packet) &&
      loom_amdgpu_fragment_publication_cost_direct(
          query, direct_packets, direct_packet_count, &direct_cost)) {
    *out_choice = (loom_amdgpu_fragment_memory_publication_choice_t){
        .strategy =
            LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_SCALAR_B16_STORE,
        .cost = direct_cost,
    };
    selected = true;
  }

  if (loom_amdgpu_fragment_memory_build_direct_publication(
          query, LOOM_AMDGPU_FRAGMENT_MEMORY_DIRECT_PUBLICATION_MODE_PACKED,
          direct_packets, &direct_packet_count, &has_packed_packet) &&
      has_packed_packet &&
      loom_amdgpu_fragment_publication_cost_direct(
          query, direct_packets, direct_packet_count, &direct_cost) &&
      (!selected || loom_amdgpu_fragment_memory_publication_cost_less(
                        direct_cost, out_choice->cost))) {
    *out_choice = (loom_amdgpu_fragment_memory_publication_choice_t){
        .strategy =
            LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_PACKED_B16_STORE,
        .cost = direct_cost,
    };
    selected = true;
  }

  const bool can_emit_packed_pair =
      loom_amdgpu_fragment_memory_can_emit_narrowed_store_pair(
          query->descriptor_set, query->payload_form, query->source_flags);
  const loom_matrix_fragment_packed_b16_publication_t* publication =
      can_emit_packed_pair
          ? loom_amdgpu_fragment_memory_crosslane_packed_b16_store_publication(
                query)
          : NULL;
  uint16_t crosslane_packet_register_count = 0;
  loom_amdgpu_descriptor_ref_t crosslane_store_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (publication != NULL &&
      loom_amdgpu_fragment_memory_narrowed_store_descriptor_ref(
          query->memory_space, LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT,
          &crosslane_packet_register_count, &crosslane_store_descriptor_ref) &&
      crosslane_packet_register_count == 1 &&
      loom_amdgpu_descriptor_set_has_ref(query->descriptor_set,
                                         crosslane_store_descriptor_ref)) {
    for (uint32_t strategy_index = 0; strategy_index < 2; ++strategy_index) {
      const bool uses_dpp = strategy_index == 0;
      const loom_amdgpu_fragment_memory_epilogue_strategy_t strategy =
          uses_dpp
              ? LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DPP_PACKED_B16_STORE
              : LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DS_PACKED_B16_STORE;
      if (uses_dpp && publication->paired_participant_xor_mask != 1) continue;
      const loom_amdgpu_fragment_memory_packet_flags_t crosslane_flags =
          LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE |
          (uses_dpp
               ? LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE_DPP
               : 0);
      loom_low_representation_cost_t cost = {0};
      if (!loom_amdgpu_fragment_publication_cost_crosslane(
              query, publication, crosslane_flags,
              crosslane_store_descriptor_ref, &cost) ||
          (selected && !loom_amdgpu_fragment_memory_publication_cost_less(
                           cost, out_choice->cost))) {
        continue;
      }
      *out_choice = (loom_amdgpu_fragment_memory_publication_choice_t){
          .strategy = strategy,
          .cost = cost,
          .packed_b16_publication = publication,
          .packet_flags = crosslane_flags,
      };
      selected = true;
    }
  }
  return selected;
}

static bool loom_amdgpu_fragment_memory_select_narrowed_store_packet(
    const loom_amdgpu_fragment_memory_publication_query_t* query,
    uint16_t register_index,
    loom_amdgpu_fragment_memory_packet_flags_t crosslane_packed_b16_store_flags,
    loom_amdgpu_fragment_memory_epilogue_strategy_t selected_strategy,
    loom_amdgpu_fragment_memory_packet_plan_t* out_packet) {
  if (selected_strategy ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_PACKED_B16_STORE) {
    return loom_amdgpu_fragment_memory_select_direct_publication_packet(
        query, LOOM_AMDGPU_FRAGMENT_MEMORY_DIRECT_PUBLICATION_MODE_PACKED,
        register_index, out_packet);
  }
  if (selected_strategy ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_SCALAR_B16_STORE) {
    return loom_amdgpu_fragment_memory_select_direct_publication_packet(
        query, LOOM_AMDGPU_FRAGMENT_MEMORY_DIRECT_PUBLICATION_MODE_SCALAR,
        register_index, out_packet);
  }

  *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){0};
  if (loom_amdgpu_fragment_memory_epilogue_strategy_is_crosslane_packed_b16(
          selected_strategy) &&
      register_index < query->register_count &&
      crosslane_packed_b16_store_flags != 0) {
    uint16_t packet_register_count = 0;
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (loom_amdgpu_fragment_memory_narrowed_store_descriptor_ref(
            query->memory_space, LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT,
            &packet_register_count, &descriptor_ref) &&
        packet_register_count == 1 &&
        loom_amdgpu_descriptor_set_has_ref(query->descriptor_set,
                                           descriptor_ref)) {
      *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){
          .register_index = register_index,
          .result_register_count = 1,
          .packet_register_count = packet_register_count,
          .flags = crosslane_packed_b16_store_flags,
          .descriptor_ref = descriptor_ref,
      };
      return true;
    }
  }
  return false;
}

static void loom_amdgpu_fragment_memory_plan_push_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet) {
  IREE_ASSERT_GT(packet->result_register_count, 0);
  IREE_ASSERT_LT(plan->packet_count, IREE_ARRAYSIZE(plan->packets));
  plan->packets[plan->packet_count++] = *packet;
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_lookup_descriptor_ref(descriptor_set, packet->descriptor_ref);
  plan->packet_address_sources_consumed_at_issue &= !iree_any_bit_set(
      loom_amdgpu_descriptor_traits(descriptor_set, descriptor),
      LOOM_AMDGPU_DESCRIPTOR_TRAIT_ADDRESS_SOURCE_RETAINED);
  plan->packet_flags |= packet->flags;
  if (loom_amdgpu_fragment_memory_payload_form_is_store_narrow_f32_to_16bit(
          plan->payload_form) &&
      packet->result_register_count > 1) {
    plan->packet_flags |=
        LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_PACKED_B16_STORE;
  }
}

static loom_amdgpu_fragment_memory_epilogue_strategy_t
loom_amdgpu_fragment_memory_plan_epilogue_strategy(
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  if (!loom_amdgpu_fragment_memory_payload_form_is_store_narrow_f32_to_16bit(
          plan->payload_form)) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_NONE;
  }
  if (iree_any_bit_set(
          plan->packet_flags,
          LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE)) {
    return iree_any_bit_set(
               plan->packet_flags,
               LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE_DPP)
               ? LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DPP_PACKED_B16_STORE
               : LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DS_PACKED_B16_STORE;
  }
  if (iree_any_bit_set(
          plan->packet_flags,
          LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_PACKED_B16_STORE)) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_PACKED_B16_STORE;
  }
  return plan->packet_count != 0
             ? LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_SCALAR_B16_STORE
             : LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_NONE;
}

bool loom_amdgpu_fragment_memory_epilogue_strategy_is_crosslane_packed_b16(
    loom_amdgpu_fragment_memory_epilogue_strategy_t strategy) {
  return strategy ==
             LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DS_PACKED_B16_STORE ||
         strategy ==
             LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DPP_PACKED_B16_STORE;
}

bool loom_amdgpu_fragment_memory_epilogue_strategy_uses_dpp(
    loom_amdgpu_fragment_memory_epilogue_strategy_t strategy) {
  return strategy ==
         LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DPP_PACKED_B16_STORE;
}

bool loom_amdgpu_fragment_memory_plan_packets(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_amdgpu_fragment_memory_plan_t* plan,
    iree_string_view_t* out_constraint_key) {
  if (out_constraint_key != NULL) {
    *out_constraint_key = iree_string_view_empty();
  }
  plan->packet_count = 0;
  plan->packet_address_sources_consumed_at_issue = true;
  plan->packet_flags = 0;
  plan->packed_b16_high_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  plan->packed_b16_publication = NULL;
  const bool load_packed_16bit_result =
      plan->payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT;
  const bool load_fp8_to_16bit =
      loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
          plan->payload_form);
  const bool store_narrow_f32_to_16bit =
      loom_amdgpu_fragment_memory_payload_form_is_store_narrow_f32_to_16bit(
          plan->payload_form);
  loom_amdgpu_fragment_publication_source_flags_t publication_source_flags =
      LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_NONE;
  if (plan->narrowed_result_round_source != LOOM_VALUE_ID_INVALID) {
    publication_source_flags |=
        LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_ROUNDED;
  }
  if (plan->narrowed_result_scale_source != LOOM_VALUE_ID_INVALID) {
    publication_source_flags |=
        LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_SCALED;
  }
  if (plan->narrowed_result_packed_source != LOOM_VALUE_ID_INVALID) {
    publication_source_flags |=
        LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_PACKED;
  }
  loom_amdgpu_fragment_memory_publication_query_t publication_query = {0};
  loom_amdgpu_fragment_memory_publication_choice_t publication_choice = {0};
  if (store_narrow_f32_to_16bit) {
    publication_query = (loom_amdgpu_fragment_memory_publication_query_t){
        .descriptor_set = descriptor_set,
        .layout = layout,
        .address_layout = &plan->address_layout,
        .runtime_axes = plan->runtime_axes,
        .static_axis_byte_strides = plan->static_axis_byte_strides,
        .memory_space = plan->source.memory_space,
        .role = plan->role,
        .representation_flags = plan->representation_flags,
        .payload_form = plan->payload_form,
        .register_count = plan->register_count,
        .element_byte_count = plan->element_byte_count,
        .view_rank = plan->view_rank,
        .source_flags = publication_source_flags,
    };
    if (!loom_amdgpu_fragment_memory_select_publication(&publication_query,
                                                        &publication_choice)) {
      if (out_constraint_key != NULL) {
        *out_constraint_key = IREE_SV("fragment_memory.packet");
      }
      return false;
    }
  }
  const loom_matrix_fragment_packed_b16_publication_t*
      crosslane_packed_b16_store_publication =
          publication_choice.packed_b16_publication;
  const loom_amdgpu_fragment_memory_packet_flags_t
      crosslane_packed_b16_store_flags = publication_choice.packet_flags;
  if (crosslane_packed_b16_store_flags != 0) {
    plan->packed_b16_publication = crosslane_packed_b16_store_publication;
  }
  loom_amdgpu_descriptor_ref_t packed_b16_low_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  loom_amdgpu_descriptor_ref_t packed_b16_high_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (plan->operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD &&
      plan->packetization ==
          LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_PACKED_B16 &&
      loom_amdgpu_fragment_memory_packed_b16_load_descriptor_refs(
          plan->source.memory_space, &packed_b16_low_descriptor_ref,
          &packed_b16_high_descriptor_ref) &&
      loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                         packed_b16_low_descriptor_ref) &&
      loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                         packed_b16_high_descriptor_ref)) {
    plan->packed_b16_high_descriptor_ref = packed_b16_high_descriptor_ref;
    const loom_low_descriptor_t* descriptor = loom_amdgpu_lookup_descriptor_ref(
        descriptor_set, packed_b16_high_descriptor_ref);
    plan->packet_address_sources_consumed_at_issue &= !iree_any_bit_set(
        loom_amdgpu_descriptor_traits(descriptor_set, descriptor),
        LOOM_AMDGPU_DESCRIPTOR_TRAIT_ADDRESS_SOURCE_RETAINED);
  } else {
    packed_b16_low_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  }
  for (uint16_t register_index = 0; register_index < plan->register_count;) {
    loom_amdgpu_fragment_memory_packet_plan_t packet = {0};
    const bool selected =
        load_fp8_to_16bit
            ? loom_amdgpu_fragment_memory_select_fp8_to_16bit_load_packet(
                  descriptor_set, layout, plan, register_index, &packet)
        : plan->packetization !=
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_NATIVE
            ? loom_amdgpu_fragment_memory_select_low_subword_packet(
                  descriptor_set, plan, register_index,
                  packed_b16_low_descriptor_ref, &packet)
        : load_packed_16bit_result
            ? loom_amdgpu_fragment_memory_select_packed_16bit_result_load_packet(
                  descriptor_set, plan, register_index, &packet)
        : store_narrow_f32_to_16bit
            ? loom_amdgpu_fragment_memory_select_narrowed_store_packet(
                  &publication_query, register_index,
                  crosslane_packed_b16_store_flags, publication_choice.strategy,
                  &packet)
            : loom_amdgpu_fragment_memory_select_packet(
                  descriptor_set, layout, plan, register_index, &packet);
    if (!selected || !loom_amdgpu_fragment_memory_packet_addresses_fit_u32(
                         layout, plan, &packet, out_constraint_key)) {
      if (out_constraint_key != NULL &&
          iree_string_view_is_empty(*out_constraint_key)) {
        *out_constraint_key = IREE_SV("fragment_memory.packet");
      }
      return false;
    }
    loom_amdgpu_fragment_memory_plan_push_packet(descriptor_set, plan, &packet);
    register_index += packet.result_register_count;
  }
  if (plan->packet_count == 0) {
    if (out_constraint_key != NULL) {
      *out_constraint_key = IREE_SV("fragment_memory.packet");
    }
    return false;
  }
  plan->epilogue_strategy =
      loom_amdgpu_fragment_memory_plan_epilogue_strategy(plan);
  IREE_ASSERT_TRUE(!store_narrow_f32_to_16bit ||
                   plan->epilogue_strategy == publication_choice.strategy);
  return true;
}

bool loom_amdgpu_fragment_memory_select_fp8_load_decode_plan(
    const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* plan) {
  plan->fp8_load_decode = (loom_amdgpu_fp8_decode_action_t){0};
  if (!loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
          plan->payload_form)) {
    return true;
  }

  const loom_scalar_type_t result_element_type =
      loom_amdgpu_fragment_memory_load_fp8_result_element_type(
          plan->payload_form);
  const bool result_is_f16 = result_element_type == LOOM_SCALAR_TYPE_F16;
  loom_amdgpu_fp8_decode_plan_t decode_plan = {0};
  loom_amdgpu_initialize_fp8_decode_plan_from_descriptor_set(
      descriptor_set, plan->view_element_format, plan->descriptor_source_format,
      &decode_plan);
  loom_amdgpu_descriptor_ref_t scalef32_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  loom_amdgpu_fp8_16bit_capabilities_t capabilities =
      LOOM_AMDGPU_FP8_16BIT_CAPABILITY_NONE;
  const bool has_scalef32_descriptor =
      descriptor_set != NULL &&
      loom_amdgpu_fp8_scalef32_descriptor_ref(plan->descriptor_source_format,
                                              result_element_type,
                                              &scalef32_descriptor_ref) &&
      loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                         scalef32_descriptor_ref);
  if (has_scalef32_descriptor) {
    capabilities |= result_is_f16
                        ? LOOM_AMDGPU_FP8_16BIT_CAPABILITY_SCALEF32_F16_PAIR
                        : LOOM_AMDGPU_FP8_16BIT_CAPABILITY_SCALEF32_BF16_PAIR;
  }
  loom_amdgpu_descriptor_ref_t e8m0_pk8_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const bool has_identity_e8m0_pk8_descriptor =
      descriptor_set != NULL &&
      loom_amdgpu_fragment_memory_packets_support_fp8_e8m0_pk8(plan) &&
      loom_amdgpu_fp8_e8m0_pk8_descriptor_ref(plan->descriptor_source_format,
                                              result_element_type,
                                              &e8m0_pk8_descriptor_ref) &&
      loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                         e8m0_pk8_descriptor_ref);
  if (has_identity_e8m0_pk8_descriptor) {
    capabilities |=
        result_is_f16 ? LOOM_AMDGPU_FP8_16BIT_CAPABILITY_IDENTITY_E8M0_PK8_F16
                      : LOOM_AMDGPU_FP8_16BIT_CAPABILITY_IDENTITY_E8M0_PK8_BF16;
  }

  if (result_is_f16) {
    loom_amdgpu_fp8_native_descriptor_refs_t native_refs = {0};
    if (descriptor_set != NULL &&
        loom_amdgpu_fp8_native_descriptor_refs(plan->descriptor_source_format,
                                               LOOM_SCALAR_TYPE_F16,
                                               &native_refs) &&
        native_refs.pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
        loom_amdgpu_descriptor_set_has_ref(descriptor_set, native_refs.pair)) {
      capabilities |= LOOM_AMDGPU_FP8_16BIT_CAPABILITY_NATIVE_F16_PAIR;
    }
  }

  loom_amdgpu_fp8_decode_value_flags_t decode_value_flags =
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  if (fact_table != NULL) {
    const loom_value_facts_t source_facts = loom_value_fact_table_lookup(
        fact_table, loom_vector_fragment_load_result(source_op));
    decode_value_flags =
        loom_amdgpu_fp8_decode_value_flags_from_facts(source_facts);
  }

  loom_amdgpu_fp8_decode_action_t fp8_load_decode = {0};
  if (plan->fp8_load_scale_source != LOOM_VALUE_ID_INVALID) {
    if (has_scalef32_descriptor) {
      fp8_load_decode = (loom_amdgpu_fp8_decode_action_t){
          .kind = LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_BF16_PAIR,
          .value_flags = decode_value_flags,
      };
    } else if (iree_any_bit_set(
                   decode_plan.flags,
                   LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_F32_PAIR)) {
      const bool has_native_bf16_pack = iree_any_bit_set(
          decode_plan.flags,
          LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_BF16_PACK);
      fp8_load_decode = (loom_amdgpu_fp8_decode_action_t){
          .kind =
              has_native_bf16_pack
                  ? LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR_BF16_PACK
                  : LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR,
          .value_flags = decode_value_flags,
      };
    } else {
      fp8_load_decode =
          loom_amdgpu_fragment_memory_select_fp8_to_bf16_decode_action(
              &decode_plan, LOOM_AMDGPU_FP8_16BIT_CAPABILITY_NONE,
              decode_value_flags);
    }
  } else {
    fp8_load_decode =
        result_is_f16
            ? loom_amdgpu_fragment_memory_select_fp8_to_f16_decode_action(
                  &decode_plan, capabilities, decode_value_flags)
            : loom_amdgpu_fragment_memory_select_fp8_to_bf16_decode_action(
                  &decode_plan, capabilities, decode_value_flags);
  }
  plan->fp8_load_decode = fp8_load_decode;
  return fp8_load_decode.kind != LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NONE;
}

static iree_string_view_t
loom_amdgpu_fragment_memory_fp8_full_decode_fallback_reason(
    const loom_amdgpu_fp8_decode_action_t* action) {
  const bool missing_finite = iree_any_bit_set(
      action->detail_flags,
      LOOM_AMDGPU_FP8_DECODE_ACTION_DETAIL_FLAG_MISSING_VALUE_FINITE);
  const bool missing_not_subnormal = iree_any_bit_set(
      action->detail_flags,
      LOOM_AMDGPU_FP8_DECODE_ACTION_DETAIL_FLAG_MISSING_NOT_SUBNORMAL);
  if (missing_finite && missing_not_subnormal) {
    return IREE_SV("missing_finite_not_subnormal");
  }
  if (missing_finite) return IREE_SV("missing_finite");
  if (missing_not_subnormal) return IREE_SV("missing_not_subnormal");
  if (iree_any_bit_set(
          action->detail_flags,
          LOOM_AMDGPU_FP8_DECODE_ACTION_DETAIL_FLAG_MISSING_TARGET_PACKETS)) {
    return IREE_SV("missing_target_packets");
  }
  return iree_string_view_empty();
}

static iree_string_view_t loom_amdgpu_fragment_memory_fp8_strategy_key(
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  const loom_amdgpu_fp8_decode_action_t* action = &plan->fp8_load_decode;
  const bool scaled = plan->fp8_load_scale_source != LOOM_VALUE_ID_INVALID;
  if (scaled) {
    switch (action->kind) {
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_BF16_PAIR:
        return IREE_SV("fp8_scalef32_native_bf16_pair");
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR:
        return IREE_SV("fp8_scalef32_native_f32_pair_manual_bf16_pack");
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR_BF16_PACK:
        return IREE_SV("fp8_scalef32_native_f32_pair_native_bf16_pack");
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_NORMAL:
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_EXACT_REPAIR:
        return IREE_SV("fp8_scalef32_software_packed_bf16_decode");
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_EXACT_VIA_F16:
        return IREE_SV(
            "fp8_scalef32_software_packed_bf16_decode_exact_via_f16");
      case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_BF16:
        return IREE_SV("fp8_scalef32_software_full_bf16_decode");
      default:
        IREE_ASSERT_UNREACHABLE("selected scaled FP8 fragment decode kind");
        IREE_BUILTIN_UNREACHABLE();
    }
  }

  switch (action->kind) {
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_BF16:
      return IREE_SV("fp8_identity_e8m0_pk8_bf16");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_IDENTITY_E8M0_PK8_F16:
      return IREE_SV("fp8_identity_e8m0_pk8_f16");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_BF16_PAIR:
      return IREE_SV("fp8_identity_scalef32_bf16_pair");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_SCALEF32_F16_PAIR:
      return IREE_SV("fp8_identity_scalef32_f16_pair");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F16_PAIR:
      return IREE_SV("fp8_native_f16_pair");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR:
      return IREE_SV("fp8_native_f32_pair_manual_bf16_pack");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_NATIVE_F32_PAIR_BF16_PACK:
      return IREE_SV("fp8_native_f32_pair_native_bf16_pack");
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_NORMAL:
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_EXACT_REPAIR:
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_BF16_EXACT_VIA_F16:
      return loom_amdgpu_fp8_packed_bf16_strategy_key(
          loom_amdgpu_fp8_decode_action_packed_bf16_strategy(action),
          loom_amdgpu_fp8_decode_action_repairs(action));
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_F16_NORMAL:
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_PACKED_F16_EXACT_REPAIR:
      return loom_amdgpu_fp8_packed_f16_repair_reason_key(
          loom_amdgpu_fp8_decode_action_repairs(action));
    case LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_BF16:
      return IREE_SV("fp8_full_bf16_decode");
    default:
      IREE_ASSERT_UNREACHABLE("selected FP8 fragment decode kind");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_string_view_t loom_amdgpu_fragment_memory_packet_strategy_key(
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet) {
  if (loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
          plan->payload_form)) {
    return loom_amdgpu_fragment_memory_fp8_strategy_key(plan);
  }

  if (plan->packetization ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_SCALAR_B16) {
    return plan->operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD
               ? IREE_SV("scalar_b16_fragment_load")
               : IREE_SV("scalar_b16_fragment_store");
  }
  if (plan->packetization ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_PACKED_B16) {
    const bool strided = plan->address_layout.packed_element_byte_stride !=
                         plan->element_byte_count;
    if (plan->operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD) {
      if (plan->packed_b16_high_descriptor_ref !=
          LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
        return strided ? IREE_SV("strided_d16_packed_b16_fragment_load")
                       : IREE_SV("d16_packed_b16_fragment_load");
      }
      return strided ? IREE_SV("strided_packed_b16_fragment_load")
                     : IREE_SV("packed_b16_fragment_load");
    }
    return strided ? IREE_SV("strided_packed_b16_fragment_store")
                   : IREE_SV("packed_b16_fragment_store");
  }

  if (!loom_amdgpu_fragment_memory_payload_form_is_store_narrow_f32_to_16bit(
          plan->payload_form)) {
    return iree_string_view_empty();
  }
  const bool result_is_f16 =
      plan->payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_F16;
  if (iree_all_bits_set(
          packet->flags,
          LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE)) {
    if (loom_amdgpu_fragment_memory_epilogue_strategy_uses_dpp(
            plan->epilogue_strategy)) {
      return result_is_f16 ? IREE_SV("dpp_packed_f16_store")
                           : IREE_SV("dpp_packed_bf16_store");
    }
    return result_is_f16 ? IREE_SV("ds_bpermute_packed_f16_store")
                         : IREE_SV("ds_bpermute_packed_bf16_store");
  }
  if (packet->result_register_count > 1) {
    return result_is_f16 ? IREE_SV("packed_f16_store")
                         : IREE_SV("packed_bf16_store");
  }
  if (plan->epilogue_strategy ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_SCALAR_B16_STORE) {
    return result_is_f16 ? IREE_SV("scalar_f16_store")
                         : IREE_SV("scalar_bf16_store");
  }
  return iree_string_view_empty();
}

iree_string_view_t loom_amdgpu_fragment_memory_plan_key(
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  if (plan->packet_count == 0) {
    return iree_string_view_empty();
  }
  if (loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
          plan->payload_form)) {
    return loom_amdgpu_fragment_memory_fp8_strategy_key(plan);
  }
  const iree_string_view_t first_key =
      loom_amdgpu_fragment_memory_packet_strategy_key(plan, &plan->packets[0]);
  if (iree_string_view_is_empty(first_key)) {
    return iree_string_view_empty();
  }
  for (uint16_t i = 1; i < plan->packet_count; ++i) {
    const iree_string_view_t packet_key =
        loom_amdgpu_fragment_memory_packet_strategy_key(plan,
                                                        &plan->packets[i]);
    if (!iree_string_view_equal(first_key, packet_key)) {
      return IREE_SV("mixed_fragment_memory_strategy");
    }
  }
  return first_key;
}

static iree_string_view_t loom_amdgpu_fragment_memory_packet_fallback_reason(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet) {
  if (loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
          plan->payload_form)) {
    if (plan->fp8_load_decode.kind ==
        LOOM_AMDGPU_FP8_DECODE_ACTION_KIND_FULL_BF16) {
      return loom_amdgpu_fragment_memory_fp8_full_decode_fallback_reason(
          &plan->fp8_load_decode);
    }
    return iree_string_view_empty();
  }

  if (plan->payload_form ==
          LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT &&
      packet->result_register_count == 1 && plan->register_count > 1) {
    const uint16_t remaining = plan->register_count - packet->register_index;
    if (remaining > 1 &&
        !loom_amdgpu_fragment_memory_register_group_is_contiguous(
            plan, packet->register_index, 2, plan->element_byte_count)) {
      return IREE_SV("fragment_noncontiguous_registers");
    }
    return iree_string_view_empty();
  }

  if (!loom_amdgpu_fragment_memory_payload_form_is_store_narrow_f32_to_16bit(
          plan->payload_form) ||
      packet->result_register_count != 1 || plan->register_count <= 1) {
    return iree_string_view_empty();
  }
  if (iree_all_bits_set(
          packet->flags,
          LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE)) {
    return iree_string_view_empty();
  }
  const uint16_t remaining = plan->register_count - packet->register_index;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuFragmentMemoryNarrowedStoreCandidates);
       ++i) {
    const uint16_t candidate =
        kLoomAmdgpuFragmentMemoryNarrowedStoreCandidates[i];
    const bool can_use_packed_payload_slice =
        plan->narrowed_result_round_source != LOOM_VALUE_ID_INVALID ||
        ((packet->register_index | candidate) & 1u) == 0;
    if (candidate <= 1 || candidate > remaining ||
        !can_use_packed_payload_slice) {
      continue;
    }
    uint16_t packet_register_count = 0;
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (!loom_amdgpu_fragment_memory_narrowed_store_descriptor_ref(
            plan->source.memory_space, candidate, &packet_register_count,
            &descriptor_ref) ||
        !loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref)) {
      continue;
    }
    if (!loom_amdgpu_fragment_memory_register_group_is_contiguous(
            plan, packet->register_index, candidate,
            plan->element_byte_count)) {
      return IREE_SV("fragment_noncontiguous_registers");
    }
  }
  return iree_string_view_empty();
}

void loom_amdgpu_fragment_memory_query_packet_report(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    loom_amdgpu_fragment_memory_packet_report_t* out_report) {
  out_report->strategy_key =
      loom_amdgpu_fragment_memory_packet_strategy_key(plan, packet);
  out_report->fallback_reason =
      loom_amdgpu_fragment_memory_packet_fallback_reason(descriptor_set, plan,
                                                         packet);
}
