// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment.h"

#include <stdint.h>

#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/encoding/storage.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/error_catalog.h"
#include "loom/target/arch/amdgpu/lower/candidates/compare_candidates.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/narrow_float/bf16.h"
#include "loom/target/arch/amdgpu/lower/narrow_float/fp8.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_id/target_id.h"
#include "loom/util/fact_table.h"
#include "loom/util/numeric_format.h"

enum {
  LOOM_AMDGPU_FRAGMENT_VIEW_RANK = 2,
  LOOM_AMDGPU_FRAGMENT_LANE_MODULUS = 16,
  LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT = 4,
  LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_PACKET_REGISTERS = 4,
  LOOM_AMDGPU_FRAGMENT_PACKED_B8_ELEMENT_BIT_COUNT = 8,
  LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT = 2,
  LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_BIT_COUNT = 16,
  LOOM_AMDGPU_FRAGMENT_FP8_REPAIR_PACKET_FLAG_SHIFT = 8,
  LOOM_AMDGPU_FRAGMENT_FP8_REPAIR_PACKET_FLAGS =
      LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_REPAIR_ZERO |
      LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_REPAIR_SUBNORMAL |
      LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_REPAIR_NAN |
      LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_REPAIR_INF,
};

static const uint16_t kLoomAmdgpuFragmentMemoryPacketCandidates[] = {4, 3, 2,
                                                                     1};

static const uint16_t kLoomAmdgpuFragmentMemoryNarrowedStoreCandidates[] = {
    8, 6, 4, 2, 1};

static_assert(LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_ZERO ==
                  (LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_REPAIR_ZERO >>
                   LOOM_AMDGPU_FRAGMENT_FP8_REPAIR_PACKET_FLAG_SHIFT),
              "FP8 zero repair packet flags mirror repair bits");
static_assert(
    LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_SUBNORMAL ==
        (LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_REPAIR_SUBNORMAL >>
         LOOM_AMDGPU_FRAGMENT_FP8_REPAIR_PACKET_FLAG_SHIFT),
    "FP8 subnormal repair packet flags mirror repair bits");
static_assert(LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_NAN ==
                  (LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_REPAIR_NAN >>
                   LOOM_AMDGPU_FRAGMENT_FP8_REPAIR_PACKET_FLAG_SHIFT),
              "FP8 NaN repair packet flags mirror repair bits");
static_assert(LOOM_AMDGPU_FP8_PACKED_U16_REPAIR_INF ==
                  (LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_REPAIR_INF >>
                   LOOM_AMDGPU_FRAGMENT_FP8_REPAIR_PACKET_FLAG_SHIFT),
              "FP8 infinity repair packet flags mirror repair bits");

typedef enum loom_amdgpu_fragment_memory_domain_e {
  LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_GLOBAL = 0,
  LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_DESCRIPTOR = 1,
  LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_WORKGROUP = 2,
  LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_COUNT_,
} loom_amdgpu_fragment_memory_domain_t;

typedef struct loom_amdgpu_fragment_memory_descriptor_table_t {
  // Descriptor refs for normal 32-bit-register packet payloads, indexed by
  // operation kind and packet register count.
  loom_amdgpu_descriptor_ref_t
      packet_refs[LOOM_AMDGPU_MEMORY_OPERATION_COUNT_]
                 [LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_PACKET_REGISTERS + 1u];
  // Descriptor refs for scalar 16-bit packets, indexed by operation kind.
  loom_amdgpu_descriptor_ref_t b16_refs[LOOM_AMDGPU_MEMORY_OPERATION_COUNT_];
} loom_amdgpu_fragment_memory_descriptor_table_t;

static_assert(LOOM_AMDGPU_MEMORY_OPERATION_COUNT_ == 2,
              "AMDGPU fragment memory descriptor tables cover load/store");

static const loom_amdgpu_fragment_memory_descriptor_table_t
    kFragmentMemoryDescriptorTables[LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_COUNT_] = {
        [LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_GLOBAL] =
            {
                .packet_refs =
                    {
                        [LOOM_AMDGPU_MEMORY_OPERATION_LOAD] =
                            {
                                LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B32_SADDR,
                                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B64_SADDR,
                                LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                                LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B128_SADDR,
                            },
                        [LOOM_AMDGPU_MEMORY_OPERATION_STORE] =
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
                        [LOOM_AMDGPU_MEMORY_OPERATION_LOAD] =
                            LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B16_D16_SADDR,
                        [LOOM_AMDGPU_MEMORY_OPERATION_STORE] =
                            LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B16_SADDR,
                    },
            },
        [LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_DESCRIPTOR] =
            {
                .packet_refs =
                    {
                        [LOOM_AMDGPU_MEMORY_OPERATION_LOAD] =
                            {
                                LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                                LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_LOAD_DWORD,
                                LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_LOAD_B64,
                                LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_LOAD_B96,
                                LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_LOAD_B128,
                            },
                        [LOOM_AMDGPU_MEMORY_OPERATION_STORE] =
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
                        [LOOM_AMDGPU_MEMORY_OPERATION_LOAD] =
                            LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_LOAD_B16_D16,
                        [LOOM_AMDGPU_MEMORY_OPERATION_STORE] =
                            LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_STORE_B16,
                    },
            },
        [LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_WORKGROUP] =
            {
                .packet_refs =
                    {
                        [LOOM_AMDGPU_MEMORY_OPERATION_LOAD] =
                            {
                                LOOM_AMDGPU_DESCRIPTOR_REF_NONE,
                                LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B32,
                                LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B64,
                                LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B96,
                                LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_B128,
                            },
                        [LOOM_AMDGPU_MEMORY_OPERATION_STORE] =
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
                        [LOOM_AMDGPU_MEMORY_OPERATION_LOAD] =
                            LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_U16,
                        [LOOM_AMDGPU_MEMORY_OPERATION_STORE] =
                            LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B16,
                    },
            },
};

static const loom_contract_operand_role_t kFragmentMemoryContractRoles[] = {
    [LOOM_VECTOR_ROLE_LHS] = LOOM_CONTRACT_OPERAND_ROLE_LHS,
    [LOOM_VECTOR_ROLE_RHS] = LOOM_CONTRACT_OPERAND_ROLE_RHS,
    [LOOM_VECTOR_ROLE_INIT] = LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR,
    [LOOM_VECTOR_ROLE_RESULT] = LOOM_CONTRACT_OPERAND_ROLE_RESULT,
};

static_assert(IREE_ARRAYSIZE(kFragmentMemoryContractRoles) ==
                  LOOM_VECTOR_ROLE_COUNT_,
              "fragment memory contract roles cover vector roles");
static_assert((int)LOOM_AMDGPU_MEMORY_OPERATION_LOAD ==
                  (int)LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD,
              "AMDGPU load operation kind matches source memory planning");
static_assert((int)LOOM_AMDGPU_MEMORY_OPERATION_STORE ==
                  (int)LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE,
              "AMDGPU store operation kind matches source memory planning");

typedef struct loom_amdgpu_fragment_memory_contract_candidate_list_t {
  // Descriptor set used to filter descriptors in this list.
  const loom_low_descriptor_set_t* descriptor_set;
  // Matrix features used to filter descriptors in this list.
  loom_amdgpu_matrix_feature_bits_t feature_bits;
  // Wave size used to filter descriptors in this list.
  uint32_t wave_size;
  // Matrix contract descriptors available to this descriptor set and target.
  const loom_amdgpu_matrix_contract_descriptor_t** descriptors;
  // Number of entries in descriptors.
  iree_host_size_t descriptor_count;
} loom_amdgpu_fragment_memory_contract_candidate_list_t;

typedef struct loom_amdgpu_fragment_memory_environment_t {
  // Source module being checked or lowered.
  const loom_module_t* module;
  // Source facts available for shape, view, and address reasoning.
  const loom_value_fact_table_t* fact_table;
  // Precomputed source view summaries available for address planning.
  const loom_view_region_table_t* view_regions;
  // Target bundle selected for this source-to-low attempt.
  const loom_target_bundle_t* bundle;
  // Low descriptor set selected by the target bundle.
  const loom_low_descriptor_set_t* descriptor_set;
  // Optional function-local matrix contract candidate list.
  const loom_amdgpu_fragment_memory_contract_candidate_list_t*
      contract_candidates;
  // Function-local source allocation layout analysis.
  const loom_amdgpu_source_alloca_layout_t* alloca_layout;
  // Matrix feature bits available on the selected processor.
  loom_amdgpu_matrix_feature_bits_t feature_bits;
  // Source function owning the fragment movement op.
  loom_func_like_t source_function;
} loom_amdgpu_fragment_memory_environment_t;

typedef struct loom_amdgpu_fragment_memory_source_t {
  // Source vector fragment role.
  loom_vector_role_t vector_role;
  // View value read or written by the fragment movement op.
  loom_value_id_t view;
  // Vector payload result for loads or stored payload for stores.
  loom_value_id_t payload;
  // Source fragment row count value.
  loom_value_id_t rows;
  // Source fragment column count value.
  loom_value_id_t columns;
  // Static index array spelling the base view indices.
  loom_attribute_t static_indices;
  // Dynamic index operands referenced by the static index sentinel slots.
  loom_value_slice_t dynamic_indices;
  // Optional cache scope attr on the source op.
  loom_attribute_t cache_scope;
  // Optional cache temporal attr on the source op.
  loom_attribute_t cache_temporal;
} loom_amdgpu_fragment_memory_source_t;

typedef struct loom_amdgpu_fragment_memory_diagnostic_t {
  // Stable constraint key identifying the first failed representation contract.
  iree_string_view_t constraint_key;
} loom_amdgpu_fragment_memory_diagnostic_t;

typedef struct loom_amdgpu_fragment_memory_narrowed_result_sources_t {
  // F32 fragment source rounded directly for narrowed stores.
  loom_value_id_t round_source;
  // Optional scalar scale applied before narrowed stores.
  loom_value_id_t scale_source;
  // Packed BF16 fragment source copied directly for narrowed stores.
  loom_value_id_t packed_source;
  // Number of 32-bit registers in the packed source.
  uint16_t packed_register_count;
} loom_amdgpu_fragment_memory_narrowed_result_sources_t;

typedef struct loom_amdgpu_fragment_lane_ids_t {
  // Full subgroup lane id.
  loom_value_id_t lane;
  // Lane id modulo the fragment row/column tile when materialized.
  loom_value_id_t lane_mod;
  // Lane id divided by the fragment row/column tile when materialized.
  loom_value_id_t lane_div;
} loom_amdgpu_fragment_lane_ids_t;

typedef struct loom_amdgpu_fragment_lane_id_cache_t {
  // Low block where lane ids were materialized.
  loom_block_t* block;
  // Register type used for the materialized lane ids.
  loom_type_t vgpr_type;
  // Tile row/column divisor used to derive lane_mod and lane_div.
  uint16_t lane_divisor;
  // Cached lane ids valid in block append order.
  loom_amdgpu_fragment_lane_ids_t lane_ids;
} loom_amdgpu_fragment_lane_id_cache_t;

typedef struct loom_amdgpu_fragment_memory_address_t {
  // Low packet address operand after static offset immediates are split out.
  loom_value_id_t low_vaddr;
  // Encoded descriptor offset immediate value.
  int64_t immediate_offset;
} loom_amdgpu_fragment_memory_address_t;

typedef enum loom_amdgpu_fragment_memory_pending_store_payload_form_e {
  LOOM_AMDGPU_FRAGMENT_MEMORY_PENDING_STORE_PAYLOAD_FORM_F32 = 0,
  LOOM_AMDGPU_FRAGMENT_MEMORY_PENDING_STORE_PAYLOAD_FORM_BF16 = 1,
} loom_amdgpu_fragment_memory_pending_store_payload_form_t;

typedef struct loom_amdgpu_fragment_memory_pending_store_t {
  // Packet plan copied from the immutable fragment memory plan.
  loom_amdgpu_fragment_memory_packet_plan_t packet;
  // Local result-fragment lane payload before final packing.
  loom_value_id_t low_source_register;
  // Cross-lane result-fragment payload paired with low_source_register.
  loom_value_id_t low_paired_source_register;
} loom_amdgpu_fragment_memory_pending_store_t;

typedef enum loom_amdgpu_fragment_memory_address_register_kind_e {
  LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE = 0,
  LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR,
  LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR,
} loom_amdgpu_fragment_memory_address_register_kind_t;

typedef struct loom_amdgpu_fragment_memory_address_accumulator_t {
  // Low scalar register containing the accumulated byte address.
  loom_value_id_t value;
  // Register class of value.
  loom_amdgpu_fragment_memory_address_register_kind_t register_kind;
} loom_amdgpu_fragment_memory_address_accumulator_t;

typedef struct loom_amdgpu_fragment_memory_address_base_key_t {
  // Low block where the base address was materialized.
  loom_block_t* block;
  // Register type used for any VGPR address terms.
  loom_type_t vgpr_type;
  // Static lane-mod byte stride in the fragment register map.
  uint32_t lane_mod_stride;
  // Static lane-div byte stride in the fragment register map.
  uint32_t lane_div_stride;
  // Number of dynamic source terms in the key.
  uint8_t dynamic_term_count;
  // Source SSA values used for dynamic byte-address terms.
  loom_value_id_t dynamic_values[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
  // Static byte strides paired with dynamic_values.
  int64_t dynamic_byte_strides[LOOM_LOW_SOURCE_MEMORY_DYNAMIC_TERM_CAPACITY];
} loom_amdgpu_fragment_memory_address_base_key_t;

typedef struct loom_amdgpu_fragment_memory_address_base_cache_t {
  // Key for the cached base accumulator.
  loom_amdgpu_fragment_memory_address_base_key_t key;
  // Cached low address accumulator. register_kind NONE means empty.
  loom_amdgpu_fragment_memory_address_accumulator_t accumulator;
} loom_amdgpu_fragment_memory_address_base_cache_t;

typedef struct loom_amdgpu_fragment_memory_cache_t {
  // Cached subgroup lane ids for adjacent fragment memory operations.
  loom_amdgpu_fragment_lane_id_cache_t lane_id_cache;
  // Cached base address shared by adjacent fragment memory operations.
  loom_amdgpu_fragment_memory_address_base_cache_t address_base;
  // Cached matrix contract descriptors available to this source function.
  loom_amdgpu_fragment_memory_contract_candidate_list_t contract_candidates;
} loom_amdgpu_fragment_memory_cache_t;

static bool loom_amdgpu_fragment_memory_reject(
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic,
    iree_string_view_t constraint_key) {
  if (diagnostic != NULL &&
      iree_string_view_is_empty(diagnostic->constraint_key)) {
    diagnostic->constraint_key = constraint_key;
  }
  return false;
}

static bool loom_amdgpu_fragment_memory_role_from_vector_role(
    loom_vector_role_t role, loom_contract_operand_role_t* out_role) {
  *out_role = LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN;
  if (role >= IREE_ARRAYSIZE(kFragmentMemoryContractRoles) ||
      kFragmentMemoryContractRoles[role] ==
          LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN) {
    return false;
  }
  *out_role = kFragmentMemoryContractRoles[role];
  return true;
}

static bool loom_amdgpu_fragment_memory_source_operation_kind(
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_low_source_memory_operation_kind_t* out_operation_kind) {
  *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  if (operation_kind >= LOOM_AMDGPU_MEMORY_OPERATION_COUNT_) {
    return false;
  }
  *out_operation_kind = (loom_low_source_memory_operation_kind_t)operation_kind;
  return true;
}

static bool loom_amdgpu_fragment_memory_exact_nonnegative_i64(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    int64_t* out_value) {
  return loom_amdgpu_value_facts_as_exact_non_negative_i64(
      loom_value_fact_table_lookup(fact_table, value_id), out_value);
}

static bool loom_amdgpu_fragment_repack_shape_value_matches(
    const loom_value_fact_table_t* fact_table, loom_value_id_t source_value,
    loom_value_id_t result_value) {
  if (source_value == result_value) {
    return true;
  }
  int64_t source_exact = 0;
  int64_t result_exact = 0;
  return loom_amdgpu_fragment_memory_exact_nonnegative_i64(
             fact_table, source_value, &source_exact) &&
         loom_amdgpu_fragment_memory_exact_nonnegative_i64(
             fact_table, result_value, &result_exact) &&
         source_exact == result_exact;
}

static bool loom_amdgpu_fragment_repack_shape_matches(
    const loom_value_fact_table_t* fact_table,
    loom_vector_fragment_fact_t source_fact, loom_value_id_t rows,
    loom_value_id_t columns) {
  return source_fact.shape_rank == 2 &&
         loom_amdgpu_fragment_repack_shape_value_matches(
             fact_table, source_fact.shape_value_ids[0], rows) &&
         loom_amdgpu_fragment_repack_shape_value_matches(
             fact_table, source_fact.shape_value_ids[1], columns);
}

static iree_string_view_t loom_amdgpu_fragment_repack_role_flags_key(
    loom_vector_fragment_role_flags_t role_flags) {
  switch (role_flags) {
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_LHS:
      return IREE_SV("lhs");
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RHS:
      return IREE_SV("rhs");
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT:
      return IREE_SV("init");
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT:
      return IREE_SV("result");
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
        LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT:
      return IREE_SV("accumulator_result");
    default:
      return IREE_SV("unknown");
  }
}

static iree_string_view_t loom_amdgpu_fragment_repack_reason_key(
    loom_amdgpu_fragment_repack_reason_t reason) {
  switch (reason) {
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_NONE:
      return IREE_SV("none");
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_SOURCE_FACTS:
      return IREE_SV("source_fragment_facts");
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_SHAPE:
      return IREE_SV("fragment_shape");
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_ROLE_TRANSITION:
      return IREE_SV("role_transition");
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TYPE_TRANSITION:
      return IREE_SV("type_transition");
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_ROLE_TYPE_TRANSITION:
      return IREE_SV("role_type_transition");
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TARGET_LAYOUT:
      return IREE_SV("target_layout");
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_LAYOUT_STRATEGY:
      return IREE_SV("layout_strategy");
    case LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TARGET_PACKETS:
      return IREE_SV("target_packets");
    default:
      return IREE_SV("unknown");
  }
}

iree_string_view_t loom_amdgpu_fragment_repack_plan_key(
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  switch (plan->strategy) {
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_ALIAS:
      return IREE_SV("amdgpu.fragment_repack.strategy.alias");
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_DPP_BPERMUTE:
      return IREE_SV(
          "amdgpu.fragment_repack.strategy.result_to_lhs_bf16_dpp_bpermute");
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_BPERMUTE:
      return IREE_SV(
          "amdgpu.fragment_repack.strategy.result_to_lhs_bf16_bpermute");
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_DIAGNOSTIC:
      return IREE_SV("amdgpu.fragment_repack.strategy.diagnostic");
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_NONE:
    default:
      return iree_string_view_empty();
  }
}

static loom_amdgpu_fragment_repack_reason_t
loom_amdgpu_fragment_repack_transition_reason(
    loom_vector_fragment_role_flags_t source_role_flags,
    loom_vector_fragment_role_flags_t result_role_flags,
    loom_type_t source_type, loom_type_t result_type) {
  const bool role_matches = source_role_flags == result_role_flags;
  const bool type_matches = loom_type_equal(source_type, result_type);
  if (!role_matches && !type_matches) {
    return LOOM_AMDGPU_FRAGMENT_REPACK_REASON_ROLE_TYPE_TRANSITION;
  }
  if (!role_matches) {
    return LOOM_AMDGPU_FRAGMENT_REPACK_REASON_ROLE_TRANSITION;
  }
  if (!type_matches) {
    return LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TYPE_TRANSITION;
  }
  return LOOM_AMDGPU_FRAGMENT_REPACK_REASON_NONE;
}

static bool loom_amdgpu_fragment_repack_contract_role(
    loom_vector_fragment_role_flags_t role_flags,
    loom_contract_operand_role_t* out_role) {
  *out_role = LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN;
  switch (role_flags) {
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_LHS:
      *out_role = LOOM_CONTRACT_OPERAND_ROLE_LHS;
      return true;
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RHS:
      *out_role = LOOM_CONTRACT_OPERAND_ROLE_RHS;
      return true;
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT:
      *out_role = LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR;
      return true;
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT:
    case LOOM_VECTOR_FRAGMENT_ROLE_FLAG_INIT |
        LOOM_VECTOR_FRAGMENT_ROLE_FLAG_RESULT:
      *out_role = LOOM_CONTRACT_OPERAND_ROLE_RESULT;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_tile_shape_matches(
    const loom_value_fact_table_t* fact_table,
    loom_amdgpu_matrix_tile_shape_t shape, loom_contract_operand_role_t role,
    loom_value_id_t rows, loom_value_id_t columns) {
  int64_t row_count = 0;
  int64_t column_count = 0;
  if (!loom_amdgpu_fragment_memory_exact_nonnegative_i64(fact_table, rows,
                                                         &row_count) ||
      !loom_amdgpu_fragment_memory_exact_nonnegative_i64(fact_table, columns,
                                                         &column_count)) {
    return false;
  }

  switch (role) {
    case LOOM_CONTRACT_OPERAND_ROLE_LHS:
      return row_count == shape.result_row_count &&
             column_count == shape.reduction_count;
    case LOOM_CONTRACT_OPERAND_ROLE_RHS:
      return row_count == shape.reduction_count &&
             column_count == shape.result_column_count;
    case LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR:
    case LOOM_CONTRACT_OPERAND_ROLE_RESULT:
      return row_count == shape.result_row_count &&
             column_count == shape.result_column_count;
    case LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN:
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_shape_matches(
    const loom_value_fact_table_t* fact_table,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role, loom_value_id_t rows,
    loom_value_id_t columns) {
  return loom_amdgpu_fragment_memory_tile_shape_matches(
      fact_table, layout->tile_shape, role, rows, columns);
}

static bool loom_amdgpu_fragment_memory_descriptor_shape_matches(
    const loom_value_fact_table_t* fact_table,
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    loom_contract_operand_role_t role, loom_value_id_t rows,
    loom_value_id_t columns) {
  return loom_amdgpu_fragment_memory_tile_shape_matches(
      fact_table, descriptor->tile_shape, role, rows, columns);
}

static bool loom_amdgpu_fragment_memory_role_is_result_like(
    loom_contract_operand_role_t role) {
  return role == LOOM_CONTRACT_OPERAND_ROLE_RESULT ||
         role == LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR;
}

static bool loom_amdgpu_fragment_memory_can_narrow_result_store(
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_scalar_type_t expected_element_type,
    loom_scalar_type_t storage_element_type) {
  return operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_STORE &&
         loom_amdgpu_fragment_memory_role_is_result_like(role) &&
         expected_element_type == LOOM_SCALAR_TYPE_F32 &&
         storage_element_type == LOOM_SCALAR_TYPE_BF16;
}

static bool loom_amdgpu_fragment_memory_can_extend_result_store(
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_scalar_type_t expected_element_type,
    loom_scalar_type_t payload_element_type) {
  return operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_STORE &&
         loom_amdgpu_fragment_memory_role_is_result_like(role) &&
         expected_element_type == LOOM_SCALAR_TYPE_F32 &&
         payload_element_type == LOOM_SCALAR_TYPE_F16;
}

static bool loom_amdgpu_fragment_memory_scalar_type_is_16bit_float(
    loom_scalar_type_t element_type) {
  return loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_16BIT_FLOAT,
                                       element_type);
}

static bool loom_amdgpu_fragment_memory_role_is_matrix_input(
    loom_contract_operand_role_t role) {
  return role == LOOM_CONTRACT_OPERAND_ROLE_LHS ||
         role == LOOM_CONTRACT_OPERAND_ROLE_RHS;
}

static bool loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
    loom_amdgpu_fragment_memory_payload_form_t payload_form) {
  return payload_form ==
             LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_BF16 ||
         payload_form ==
             LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_F16;
}

static loom_scalar_type_t
loom_amdgpu_fragment_memory_load_fp8_result_element_type(
    loom_amdgpu_fragment_memory_payload_form_t payload_form) {
  switch (payload_form) {
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_BF16:
      return LOOM_SCALAR_TYPE_BF16;
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_F16:
      return LOOM_SCALAR_TYPE_F16;
    default:
      return LOOM_SCALAR_TYPE_COUNT_;
  }
}

static bool loom_amdgpu_fragment_memory_descriptor_set_has_fp8_to_16bit_native(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_scalar_type_t source_element_type,
    loom_scalar_type_t result_element_type) {
  if (descriptor_set == NULL) {
    return false;
  }
  loom_amdgpu_fp8_native_descriptor_refs_t native_refs = {0};
  if (loom_amdgpu_fp8_native_descriptor_refs(
          source_element_type, result_element_type, &native_refs) &&
      native_refs.pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
      loom_amdgpu_descriptor_set_has_ref(descriptor_set, native_refs.pair)) {
    return true;
  }
  loom_amdgpu_descriptor_ref_t scale_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (loom_amdgpu_fp8_scalef32_descriptor_ref(
          source_element_type, result_element_type, &scale_descriptor_ref) &&
      loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                         scale_descriptor_ref)) {
    return true;
  }
  return loom_amdgpu_fp8_e8m0_pk8_descriptor_ref(
             source_element_type, result_element_type, &scale_descriptor_ref) &&
         loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                            scale_descriptor_ref);
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

static loom_amdgpu_fragment_memory_packet_flags_t
loom_amdgpu_fragment_memory_fp8_repair_packet_flags(
    loom_amdgpu_fp8_packed_u16_repairs_t repairs) {
  return ((loom_amdgpu_fragment_memory_packet_flags_t)repairs
          << LOOM_AMDGPU_FRAGMENT_FP8_REPAIR_PACKET_FLAG_SHIFT) &
         LOOM_AMDGPU_FRAGMENT_FP8_REPAIR_PACKET_FLAGS;
}

typedef uint32_t loom_amdgpu_fp8_16bit_capabilities_t;

enum loom_amdgpu_fp8_16bit_capability_bits_e {
  LOOM_AMDGPU_FP8_16BIT_CAPABILITY_NONE = 0u,
  LOOM_AMDGPU_FP8_16BIT_CAPABILITY_IDENTITY_E8M0_PK8_BF16 = 1u << 0,
  LOOM_AMDGPU_FP8_16BIT_CAPABILITY_IDENTITY_SCALEF32_BF16_PAIR = 1u << 1,
  LOOM_AMDGPU_FP8_16BIT_CAPABILITY_IDENTITY_E8M0_PK8_F16 = 1u << 2,
  LOOM_AMDGPU_FP8_16BIT_CAPABILITY_IDENTITY_SCALEF32_F16_PAIR = 1u << 3,
  LOOM_AMDGPU_FP8_16BIT_CAPABILITY_NATIVE_F16_PAIR = 1u << 4,
};

static loom_amdgpu_fragment_memory_packet_flags_t
loom_amdgpu_fragment_memory_fp8_decode_packet_flags(
    const loom_amdgpu_fp8_decode_plan_t* decode_plan,
    loom_amdgpu_fp8_16bit_capabilities_t capabilities,
    loom_amdgpu_fp8_decode_value_flags_t decode_value_flags) {
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_FP8_16BIT_CAPABILITY_IDENTITY_E8M0_PK8_BF16)) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_IDENTITY_E8M0_PK8_BF16;
  }
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_FP8_16BIT_CAPABILITY_IDENTITY_SCALEF32_BF16_PAIR)) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_IDENTITY_SCALEF32_BF16_PAIR;
  }
  const bool prefer_packed_bf16 =
      loom_amdgpu_fp8_prefers_packed_bf16_pair_decode(decode_plan,
                                                      decode_value_flags);
  if (prefer_packed_bf16) {
    const loom_amdgpu_fp8_packed_u16_repairs_t repairs =
        loom_amdgpu_fp8_pair_to_packed_bf16_repairs(decode_plan,
                                                    decode_value_flags);
    return LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_PACKED_BF16_DECODE |
           loom_amdgpu_fragment_memory_fp8_repair_packet_flags(repairs);
  }
  const loom_amdgpu_fp8_packed_bf16_missing_requirements_t
      missing_requirements =
          loom_amdgpu_fp8_pair_to_packed_bf16_missing_requirements(
              decode_plan, decode_value_flags);
  if (iree_any_bit_set(decode_plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_F32_PAIR)) {
    loom_amdgpu_fragment_memory_packet_flags_t packet_flags =
        LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_NATIVE_F32_PAIR;
    if (iree_any_bit_set(
            decode_plan->flags,
            LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_BF16_PACK)) {
      packet_flags |=
          LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_NATIVE_BF16_PACK;
    }
    return packet_flags;
  }
  if (missing_requirements ==
      LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_NONE) {
    const loom_amdgpu_fp8_packed_u16_repairs_t repairs =
        loom_amdgpu_fp8_pair_to_packed_bf16_repairs(decode_plan,
                                                    decode_value_flags);
    return LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_PACKED_BF16_DECODE |
           loom_amdgpu_fragment_memory_fp8_repair_packet_flags(repairs);
  }
  loom_amdgpu_fragment_memory_packet_flags_t packet_flags =
      LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_FULL_BF16_DECODE;
  if (iree_any_bit_set(
          missing_requirements,
          LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_VALUE_FINITE)) {
    packet_flags |=
        LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_MISSING_VALUE_FINITE;
  }
  if (iree_any_bit_set(
          missing_requirements,
          LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_VALUE_NOT_SUBNORMAL)) {
    packet_flags |=
        LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_MISSING_VALUE_NOT_SUBNORMAL;
  }
  const loom_amdgpu_fp8_packed_bf16_missing_requirements_t target_requirements =
      LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_PERMUTE_PACKET |
      LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_PACKED_SHIFT_PACKET |
      LOOM_AMDGPU_FP8_PACKED_BF16_MISSING_REQUIREMENT_ZERO_REPAIR_PACKETS;
  if (iree_any_bit_set(missing_requirements, target_requirements)) {
    packet_flags |=
        LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_MISSING_TARGET_PACKETS;
  }
  return packet_flags;
}

static loom_amdgpu_fragment_memory_packet_flags_t
loom_amdgpu_fragment_memory_fp8_to_f16_decode_packet_flags(
    const loom_amdgpu_fp8_decode_plan_t* decode_plan,
    loom_amdgpu_fp8_16bit_capabilities_t capabilities,
    loom_amdgpu_fp8_decode_value_flags_t decode_value_flags) {
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_FP8_16BIT_CAPABILITY_IDENTITY_E8M0_PK8_F16)) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_IDENTITY_E8M0_PK8_F16;
  }
  if (iree_any_bit_set(
          capabilities,
          LOOM_AMDGPU_FP8_16BIT_CAPABILITY_IDENTITY_SCALEF32_F16_PAIR)) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_IDENTITY_SCALEF32_F16_PAIR;
  }
  if (iree_any_bit_set(capabilities,
                       LOOM_AMDGPU_FP8_16BIT_CAPABILITY_NATIVE_F16_PAIR)) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_NATIVE_F16_PAIR;
  }
  if (loom_amdgpu_can_emit_fp8_pair_to_packed_f16_finite(decode_plan,
                                                         decode_value_flags)) {
    const loom_amdgpu_fp8_packed_u16_repairs_t repairs =
        loom_amdgpu_fp8_pair_to_packed_f16_repairs(decode_plan,
                                                   decode_value_flags);
    return LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_PACKED_F16_DECODE |
           loom_amdgpu_fragment_memory_fp8_repair_packet_flags(repairs);
  }
  return 0;
}

static bool loom_amdgpu_fragment_memory_schema_proves_fp8_finite(
    const loom_value_fact_storage_schema_t* view_storage_schema,
    loom_scalar_type_t view_element_type) {
  loom_value_facts_t content_facts = loom_value_facts_unknown();
  return loom_encoding_query_storage_schema_content_facts(
             view_storage_schema, view_element_type, &content_facts) &&
         loom_value_facts_is_finite(content_facts);
}

static bool loom_amdgpu_fragment_memory_can_load_fp8_to_16bit(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_scalar_type_t expected_element_type,
    loom_scalar_type_t payload_element_type,
    loom_scalar_type_t view_element_type,
    const loom_value_fact_storage_schema_t* view_storage_schema,
    loom_amdgpu_fragment_memory_payload_form_t* out_payload_form) {
  *out_payload_form = LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
  loom_scalar_type_fp8_format_t unused_format = {0};
  if (operation_kind != LOOM_AMDGPU_MEMORY_OPERATION_LOAD ||
      !loom_amdgpu_fragment_memory_role_is_matrix_input(role) ||
      expected_element_type != payload_element_type ||
      !loom_scalar_type_fp8_format(view_element_type, &unused_format)) {
    return false;
  }
  switch (expected_element_type) {
    case LOOM_SCALAR_TYPE_BF16:
      *out_payload_form =
          LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_BF16;
      return true;
    case LOOM_SCALAR_TYPE_F16:
      if (!loom_amdgpu_fragment_memory_descriptor_set_has_fp8_to_16bit_native(
              descriptor_set, view_element_type, expected_element_type) &&
          !loom_amdgpu_fragment_memory_schema_proves_fp8_finite(
              view_storage_schema, view_element_type)) {
        return false;
      }
      *out_payload_form =
          LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_F16;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_can_load_packed_16bit_result(
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_scalar_type_t expected_element_type,
    loom_scalar_type_t payload_element_type,
    loom_scalar_type_t view_element_type) {
  return operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_LOAD &&
         loom_amdgpu_fragment_memory_role_is_result_like(role) &&
         expected_element_type == LOOM_SCALAR_TYPE_F32 &&
         payload_element_type == view_element_type &&
         loom_amdgpu_fragment_memory_scalar_type_is_16bit_float(
             payload_element_type);
}

static bool loom_amdgpu_fragment_memory_payload_form_select(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_scalar_type_t expected_element_type,
    loom_scalar_type_t payload_element_type,
    loom_scalar_type_t view_element_type,
    const loom_value_fact_storage_schema_t* view_storage_schema,
    loom_amdgpu_fragment_memory_payload_form_t* out_payload_form) {
  *out_payload_form = LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
  if (payload_element_type == expected_element_type &&
      view_element_type == expected_element_type) {
    return true;
  }
  if (loom_amdgpu_fragment_memory_can_load_packed_16bit_result(
          operation_kind, role, expected_element_type, payload_element_type,
          view_element_type)) {
    *out_payload_form =
        LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT;
    return true;
  }
  if (loom_amdgpu_fragment_memory_can_load_fp8_to_16bit(
          descriptor_set, operation_kind, role, expected_element_type,
          payload_element_type, view_element_type, view_storage_schema,
          out_payload_form)) {
    return true;
  }
  if ((payload_element_type == expected_element_type ||
       payload_element_type == view_element_type) &&
      loom_amdgpu_fragment_memory_can_narrow_result_store(
          operation_kind, role, expected_element_type, view_element_type)) {
    *out_payload_form =
        LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16;
    return true;
  }
  if (view_element_type == expected_element_type &&
      loom_amdgpu_fragment_memory_can_extend_result_store(
          operation_kind, role, expected_element_type, payload_element_type)) {
    *out_payload_form =
        LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_EXTEND_F16_TO_F32;
    return true;
  }
  return false;
}

static bool loom_amdgpu_fragment_memory_role_uses_low_subword(
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout) {
  if (role_layout == NULL) {
    return false;
  }
  switch (role_layout->map_kind) {
    case LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN_LOW_SUBWORD:
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN_LOW_SUBWORD:
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_role_uses_packed_b16_elements(
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout) {
  return role_layout != NULL &&
         role_layout->map_kind ==
             LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_PACKED_ROW_COLUMN &&
         role_layout->element_bit_count ==
             LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_BIT_COUNT &&
         role_layout->elements_per_register ==
             LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
}

static bool loom_amdgpu_fragment_memory_packed_float_storage(
    loom_type_t type, uint16_t element_bit_count,
    uint32_t* out_payload_bit_count, uint32_t* out_register_count) {
  *out_payload_bit_count = 0;
  *out_register_count = 0;
  switch (element_bit_count) {
    case LOOM_AMDGPU_FRAGMENT_PACKED_B8_ELEMENT_BIT_COUNT:
      return loom_amdgpu_type_packed_8bit_float_storage(
          type, out_payload_bit_count, out_register_count);
    case LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_BIT_COUNT:
      return loom_amdgpu_type_packed_16bit_float_storage(
          type, out_payload_bit_count, out_register_count);
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_role_packed_element_axis(
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout,
    uint8_t* out_axis) {
  *out_axis = UINT8_MAX;
  if (role_layout == NULL || role_layout->elements_per_register <= 1) {
    return false;
  }
  switch (role_layout->map_kind) {
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION:
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION:
      *out_axis = 1;
      return true;
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION:
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION:
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_GROUP_PACKED_ROW_COLUMN:
      *out_axis = 0;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_role_uses_scalar_b16_packets(
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout) {
  return loom_amdgpu_fragment_memory_role_uses_low_subword(role_layout) ||
         loom_amdgpu_fragment_memory_role_uses_packed_b16_elements(role_layout);
}

static bool loom_amdgpu_fragment_memory_requires_native_payload_storage(
    loom_amdgpu_memory_operation_kind_t operation_kind,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout,
    loom_amdgpu_fragment_memory_payload_form_t payload_form) {
  return operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_STORE &&
         (loom_amdgpu_fragment_memory_role_uses_low_subword(role_layout) ||
          payload_form ==
              LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_EXTEND_F16_TO_F32);
}

static bool loom_amdgpu_fragment_memory_payload_has_native_storage(
    const loom_value_fact_table_t* fact_table, loom_value_id_t payload) {
  if (fact_table == NULL || payload == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  loom_vector_fragment_fact_t fragment;
  return loom_vector_fragment_fact_query_value_facts(
             &fact_table->context,
             loom_value_fact_table_lookup(fact_table, payload), &fragment) &&
         iree_all_bits_set(fragment.flags,
                           LOOM_VECTOR_FRAGMENT_FACT_FLAG_HAS_NATIVE_STORAGE);
}

static bool loom_amdgpu_fragment_memory_payload_matches_role_storage(
    loom_type_t payload_type, loom_scalar_type_t expected_element_type,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout) {
  if (loom_type_element_type(payload_type) != expected_element_type ||
      !loom_scalar_type_is_float(expected_element_type)) {
    return false;
  }
  const int32_t element_bit_count =
      loom_scalar_type_bitwidth(expected_element_type);
  if (element_bit_count != role_layout->element_bit_count) {
    return false;
  }
  if (element_bit_count == 32) {
    return loom_amdgpu_vector_f32_lane_count(payload_type) ==
           role_layout->register_count;
  }
  if (element_bit_count != LOOM_AMDGPU_FRAGMENT_PACKED_B8_ELEMENT_BIT_COUNT &&
      element_bit_count != LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_BIT_COUNT) {
    return false;
  }
  uint32_t payload_bit_count = 0;
  uint32_t register_count = 0;
  return loom_amdgpu_fragment_memory_packed_float_storage(
             payload_type, (uint16_t)element_bit_count, &payload_bit_count,
             &register_count) &&
         register_count == role_layout->register_count &&
         payload_bit_count == (uint32_t)role_layout->register_count *
                                  role_layout->elements_per_register *
                                  (uint32_t)role_layout->element_bit_count;
}

static bool loom_amdgpu_fragment_memory_schema_format_matches_element_type(
    loom_value_fact_numeric_format_flags_t format,
    loom_scalar_type_t element_type,
    const loom_numeric_format_info_t** out_info) {
  if (out_info != NULL) {
    *out_info = NULL;
  }
  const loom_numeric_format_info_t* info = NULL;
  if (!loom_numeric_format_info(format, &info)) {
    return false;
  }
  switch (element_type) {
    case LOOM_SCALAR_TYPE_F8E4M3:
      if (info->kind != LOOM_NUMERIC_FORMAT_KIND_FLOAT ||
          info->float_family != LOOM_NUMERIC_FLOAT_FAMILY_FP8) {
        return false;
      }
      break;
    case LOOM_SCALAR_TYPE_F8E5M2:
      if (info->kind != LOOM_NUMERIC_FORMAT_KIND_FLOAT ||
          info->float_family != LOOM_NUMERIC_FLOAT_FAMILY_BF8) {
        return false;
      }
      break;
    case LOOM_SCALAR_TYPE_F16:
      if (info->kind != LOOM_NUMERIC_FORMAT_KIND_FLOAT ||
          info->float_family != LOOM_NUMERIC_FLOAT_FAMILY_IEEE ||
          info->storage_bit_count != 16) {
        return false;
      }
      break;
    case LOOM_SCALAR_TYPE_BF16:
      if (info->kind != LOOM_NUMERIC_FORMAT_KIND_FLOAT ||
          info->float_family != LOOM_NUMERIC_FLOAT_FAMILY_BFLOAT) {
        return false;
      }
      break;
    case LOOM_SCALAR_TYPE_F32:
      if (info->kind != LOOM_NUMERIC_FORMAT_KIND_FLOAT ||
          info->float_family != LOOM_NUMERIC_FLOAT_FAMILY_IEEE ||
          info->storage_bit_count != 32) {
        return false;
      }
      break;
    default:
      return false;
  }
  if (out_info != NULL) {
    *out_info = info;
  }
  return true;
}

static bool loom_amdgpu_fragment_memory_payload_matches_encoded_storage(
    loom_type_t payload_type, loom_type_t view_type,
    const loom_value_fact_storage_schema_t* view_storage_schema,
    loom_scalar_type_t expected_element_type,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout) {
  if (view_storage_schema == NULL || !loom_type_is_view(view_type)) {
    return false;
  }
  const loom_value_fact_encoded_operand_schema_t operand =
      view_storage_schema->encoded_operand;
  if (loom_value_fact_encoded_operand_schema_is_unknown(operand) ||
      !iree_any_bit_set(operand.payload_packing,
                        LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT) ||
      operand.payload_register_count != role_layout->register_count ||
      operand.payload_element_count !=
          role_layout->register_count * role_layout->elements_per_register) {
    return false;
  }

  const loom_numeric_format_info_t* element_format = NULL;
  if (!loom_amdgpu_fragment_memory_schema_format_matches_element_type(
          operand.element_format, expected_element_type, &element_format) ||
      element_format->storage_bit_count != role_layout->element_bit_count) {
    return false;
  }
  const int32_t view_element_bit_count =
      loom_scalar_type_bitwidth(loom_type_element_type(view_type));
  if (view_element_bit_count != element_format->storage_bit_count) {
    return false;
  }

  loom_amdgpu_vector_storage_t storage;
  return loom_amdgpu_type_vector_storage(payload_type, &storage) &&
         storage.element_bit_count == 32 &&
         storage.element_register_count == 1 &&
         storage.register_count == role_layout->register_count;
}

static bool
loom_amdgpu_fragment_memory_payload_matches_packed_16bit_result_load(
    loom_type_t payload_type,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_scalar_type_t expected_element_type,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout) {
  if (operation_kind != LOOM_AMDGPU_MEMORY_OPERATION_LOAD ||
      !loom_amdgpu_fragment_memory_role_is_result_like(role_layout->role) ||
      expected_element_type != LOOM_SCALAR_TYPE_F32 ||
      role_layout->element_bit_count != 32 ||
      role_layout->elements_per_register != 1) {
    return false;
  }
  uint32_t payload_bit_count = 0;
  uint32_t register_count = 0;
  return loom_amdgpu_type_packed_16bit_float_storage(
             payload_type, &payload_bit_count, &register_count) &&
         payload_bit_count == (uint32_t)role_layout->register_count * 16u &&
         register_count == ((uint32_t)role_layout->register_count + 1u) / 2u;
}

static bool loom_amdgpu_fragment_memory_payload_storage_register_count(
    loom_type_t payload_type, uint16_t* out_register_count) {
  *out_register_count = 0;
  loom_amdgpu_vector_storage_t storage;
  if (!loom_amdgpu_type_vector_storage(payload_type, &storage)) {
    return false;
  }
  if (storage.register_count == 0 || storage.register_count > UINT16_MAX) {
    return false;
  }
  *out_register_count = (uint16_t)storage.register_count;
  return true;
}

static bool loom_amdgpu_fragment_memory_payload_matches(
    loom_type_t payload_type, loom_type_t view_type,
    const loom_value_fact_storage_schema_t* view_storage_schema,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_scalar_type_t expected_element_type,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout) {
  if (role_layout == NULL || !loom_type_is_vector(payload_type) ||
      loom_type_rank(payload_type) != 1 ||
      !loom_type_is_all_static(payload_type)) {
    return false;
  }

  switch (role_layout->role) {
    case LOOM_CONTRACT_OPERAND_ROLE_LHS:
    case LOOM_CONTRACT_OPERAND_ROLE_RHS: {
      return loom_amdgpu_fragment_memory_payload_matches_role_storage(
                 payload_type, expected_element_type, role_layout) ||
             loom_amdgpu_fragment_memory_payload_matches_encoded_storage(
                 payload_type, view_type, view_storage_schema,
                 expected_element_type, role_layout);
    }
    case LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR:
    case LOOM_CONTRACT_OPERAND_ROLE_RESULT: {
      if (loom_amdgpu_fragment_memory_payload_matches_role_storage(
              payload_type, expected_element_type, role_layout)) {
        return true;
      }
      if (loom_amdgpu_fragment_memory_payload_matches_encoded_storage(
              payload_type, view_type, view_storage_schema,
              expected_element_type, role_layout)) {
        return true;
      }
      if (loom_amdgpu_fragment_memory_payload_matches_packed_16bit_result_load(
              payload_type, operation_kind, expected_element_type,
              role_layout)) {
        return true;
      }
      const loom_scalar_type_t payload_element_type =
          loom_type_element_type(payload_type);
      if (!loom_amdgpu_fragment_memory_can_narrow_result_store(
              operation_kind, role_layout->role, expected_element_type,
              loom_type_element_type(view_type))) {
        if (!loom_amdgpu_fragment_memory_can_extend_result_store(
                operation_kind, role_layout->role, expected_element_type,
                payload_element_type)) {
          return false;
        }
        uint32_t payload_bit_count = 0;
        uint32_t register_count = 0;
        return loom_amdgpu_type_packed_16bit_float_storage(
                   payload_type, &payload_bit_count, &register_count) &&
               payload_bit_count ==
                   (uint32_t)role_layout->register_count *
                       (uint32_t)role_layout->element_bit_count &&
               register_count == role_layout->register_count;
      }
      if (payload_element_type == expected_element_type) {
        return loom_amdgpu_fragment_memory_payload_matches_role_storage(
            payload_type, expected_element_type, role_layout);
      }
      uint32_t payload_bit_count = 0;
      uint32_t register_count = 0;
      return loom_amdgpu_type_packed_16bit_float_storage(
                 payload_type, &payload_bit_count, &register_count) &&
             payload_bit_count == (uint32_t)role_layout->register_count * 16u &&
             register_count ==
                 ((uint32_t)role_layout->register_count + 1u) / 2u;
    }
    case LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN:
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_scalar_type_from_numeric(
    loom_amdgpu_matrix_numeric_type_t numeric_type,
    loom_scalar_type_t* out_element_type) {
  *out_element_type = LOOM_SCALAR_TYPE_COUNT_;
  switch (numeric_type) {
    case LOOM_AMDGPU_MATRIX_NUMERIC_F16:
      *out_element_type = LOOM_SCALAR_TYPE_F16;
      return true;
    case LOOM_AMDGPU_MATRIX_NUMERIC_BF16:
      *out_element_type = LOOM_SCALAR_TYPE_BF16;
      return true;
    case LOOM_AMDGPU_MATRIX_NUMERIC_FP8:
      *out_element_type = LOOM_SCALAR_TYPE_F8E4M3;
      return true;
    case LOOM_AMDGPU_MATRIX_NUMERIC_BF8:
      *out_element_type = LOOM_SCALAR_TYPE_F8E5M2;
      return true;
    case LOOM_AMDGPU_MATRIX_NUMERIC_F32:
    case LOOM_AMDGPU_MATRIX_NUMERIC_XF32:
      *out_element_type = LOOM_SCALAR_TYPE_F32;
      return true;
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_descriptor_payload(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    loom_contract_operand_role_t role,
    loom_amdgpu_matrix_payload_shape_t* out_payload) {
  *out_payload = (loom_amdgpu_matrix_payload_shape_t){0};
  switch (role) {
    case LOOM_CONTRACT_OPERAND_ROLE_LHS:
      *out_payload = descriptor->lhs_payload;
      return true;
    case LOOM_CONTRACT_OPERAND_ROLE_RHS:
      *out_payload = descriptor->rhs_payload;
      return true;
    case LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR:
      *out_payload = descriptor->accumulator_payload;
      return true;
    case LOOM_CONTRACT_OPERAND_ROLE_RESULT:
      *out_payload = descriptor->result_payload;
      return true;
    case LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN:
    default:
      return false;
  }
}

typedef struct loom_amdgpu_fragment_memory_role_storage_t {
  // Matrix descriptor payload selected for the fragment role.
  loom_amdgpu_matrix_payload_shape_t payload;
  // Physical scalar type used to move each logical payload element.
  loom_scalar_type_t element_type;
} loom_amdgpu_fragment_memory_role_storage_t;

static bool loom_amdgpu_fragment_memory_descriptor_role_storage(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    loom_contract_operand_role_t role,
    loom_amdgpu_fragment_memory_role_storage_t* out_storage) {
  *out_storage = (loom_amdgpu_fragment_memory_role_storage_t){0};
  out_storage->element_type = LOOM_SCALAR_TYPE_COUNT_;
  return loom_amdgpu_fragment_memory_descriptor_payload(
             descriptor, role, &out_storage->payload) &&
         loom_amdgpu_fragment_memory_scalar_type_from_numeric(
             out_storage->payload.numeric_type, &out_storage->element_type);
}

static bool loom_amdgpu_fragment_memory_numeric_schema_matches(
    loom_amdgpu_matrix_numeric_type_t numeric_type,
    const loom_value_fact_storage_schema_t* storage_schema) {
  if (numeric_type != LOOM_AMDGPU_MATRIX_NUMERIC_XF32) {
    return true;
  }
  return storage_schema != NULL &&
         storage_schema->encoded_operand.element_format ==
             LOOM_VALUE_FACT_NUMERIC_FORMAT_TF32;
}

static bool loom_amdgpu_fragment_memory_payload_matches_descriptor_storage(
    loom_type_t payload_type, loom_scalar_type_t expected_element_type,
    const loom_amdgpu_matrix_payload_shape_t* descriptor_payload) {
  loom_amdgpu_vector_storage_t storage = {0};
  return loom_amdgpu_type_vector_storage(payload_type, &storage) &&
         storage.element_type == expected_element_type &&
         storage.element_count == descriptor_payload->element_count &&
         storage.register_count == descriptor_payload->register_count;
}

typedef enum loom_amdgpu_fragment_memory_element_match_e {
  LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE = 0,
  LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_ADAPTED = 1,
  LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_EXACT = 2,
} loom_amdgpu_fragment_memory_element_match_t;

typedef enum loom_amdgpu_fragment_memory_layout_match_e {
  LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NONE = 0,
  LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_ADAPTED = 1,
  LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_EXACT = 2,
  LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NARROWED_F32_RESULT = 3,
} loom_amdgpu_fragment_memory_layout_match_t;

static bool loom_amdgpu_fragment_memory_is_f32_result_source(
    const loom_module_t* module, loom_value_id_t source,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout) {
  if (source >= module->values.count || role_layout == NULL) {
    return false;
  }
  const loom_type_t source_type = loom_module_value_type(module, source);
  return loom_type_element_type(source_type) == LOOM_SCALAR_TYPE_F32 &&
         loom_amdgpu_vector_f32_lane_count(source_type) ==
             role_layout->register_count;
}

static loom_value_id_t loom_amdgpu_fragment_memory_same_lane_round_source(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t payload,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout) {
  if (payload >= module->values.count || fact_table == NULL) {
    return LOOM_VALUE_ID_INVALID;
  }

  loom_value_fact_static_lane_origin_t lane_origin = {0};
  if (!loom_value_fact_table_query_static_lane_origin(fact_table, module,
                                                      payload, &lane_origin) ||
      lane_origin.source_lane_offset != 0 ||
      lane_origin.source_lane_stride != 1) {
    return LOOM_VALUE_ID_INVALID;
  }

  return loom_amdgpu_fragment_memory_is_f32_result_source(
             module, lane_origin.source_value_id, role_layout)
             ? lane_origin.source_value_id
             : LOOM_VALUE_ID_INVALID;
}

static bool loom_amdgpu_fragment_memory_is_packed_bf16_result_source(
    const loom_module_t* module, loom_value_id_t source,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout,
    uint16_t* out_register_count) {
  *out_register_count = 0;
  if (source >= module->values.count || role_layout == NULL ||
      !loom_amdgpu_fragment_memory_role_is_result_like(role_layout->role) ||
      role_layout->element_bit_count != 32 ||
      role_layout->elements_per_register != 1) {
    return false;
  }
  const loom_type_t source_type = loom_module_value_type(module, source);
  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_amdgpu_type_vector_storage(source_type, &storage) ||
      storage.element_type != LOOM_SCALAR_TYPE_BF16 ||
      storage.element_bit_count != 16 ||
      storage.element_count != role_layout->register_count ||
      storage.register_count == 0 || storage.register_count > UINT16_MAX) {
    return false;
  }
  *out_register_count = (uint16_t)storage.register_count;
  return true;
}

static loom_value_id_t loom_amdgpu_fragment_memory_same_lane_packed_bf16_source(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t payload,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout,
    uint16_t* out_register_count) {
  *out_register_count = 0;
  if (payload >= module->values.count || fact_table == NULL) {
    return LOOM_VALUE_ID_INVALID;
  }

  loom_value_fact_static_lane_origin_t lane_origin = {0};
  if (!loom_value_fact_table_query_static_lane_origin(fact_table, module,
                                                      payload, &lane_origin) ||
      lane_origin.source_lane_offset != 0 ||
      lane_origin.source_lane_stride != 1) {
    return LOOM_VALUE_ID_INVALID;
  }

  return loom_amdgpu_fragment_memory_is_packed_bf16_result_source(
             module, lane_origin.source_value_id, role_layout,
             out_register_count)
             ? lane_origin.source_value_id
             : LOOM_VALUE_ID_INVALID;
}

static loom_amdgpu_fragment_memory_layout_match_t
loom_amdgpu_fragment_memory_layout_match_rank(
    const loom_amdgpu_fragment_memory_environment_t* environment,
    loom_value_id_t payload,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout,
    loom_amdgpu_fragment_memory_element_match_t element_match,
    loom_amdgpu_fragment_memory_payload_form_t payload_form) {
  if (element_match == LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NONE;
  }
  if (payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16) {
    uint16_t packed_register_count = 0;
    if (loom_amdgpu_fragment_memory_is_f32_result_source(
            environment->module, payload, role_layout) ||
        loom_amdgpu_fragment_memory_same_lane_round_source(
            environment->module, environment->fact_table, payload,
            role_layout) != LOOM_VALUE_ID_INVALID ||
        loom_amdgpu_fragment_memory_same_lane_packed_bf16_source(
            environment->module, environment->fact_table, payload, role_layout,
            &packed_register_count) != LOOM_VALUE_ID_INVALID) {
      return LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NARROWED_F32_RESULT;
    }
  }
  return element_match == LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_EXACT
             ? LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_EXACT
             : LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_ADAPTED;
}

static loom_amdgpu_fragment_memory_element_match_t
loom_amdgpu_fragment_memory_payload_element_match(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_type_t payload_type,
    loom_type_t view_type,
    const loom_value_fact_storage_schema_t* view_storage_schema,
    loom_scalar_type_t expected_element_type,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout,
    loom_amdgpu_fragment_memory_payload_form_t* out_payload_form) {
  *out_payload_form = LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
  if (!loom_type_is_vector(payload_type)) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_EXACT;
  }
  if (!loom_type_is_view(view_type)) {
    return loom_type_element_type(payload_type) == expected_element_type
               ? LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_EXACT
               : LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE;
  }
  if (loom_amdgpu_fragment_memory_payload_matches_encoded_storage(
          payload_type, view_type, view_storage_schema, expected_element_type,
          role_layout)) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_EXACT;
  }
  const loom_scalar_type_t payload_element_type =
      loom_type_element_type(payload_type);
  const loom_scalar_type_t view_element_type =
      loom_type_element_type(view_type);
  if (!loom_amdgpu_fragment_memory_payload_form_select(
          descriptor_set, operation_kind, role, expected_element_type,
          payload_element_type, view_element_type, view_storage_schema,
          out_payload_form)) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE;
  }
  return *out_payload_form == LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE
             ? LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_EXACT
             : LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_ADAPTED;
}

static bool loom_amdgpu_fragment_memory_contract_is_available(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_matrix_feature_bits_t feature_bits, uint32_t wave_size) {
  return loom_amdgpu_matrix_contract_is_available(descriptor, feature_bits,
                                                  wave_size) &&
         loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                            descriptor->low_descriptor_ref);
}

static iree_host_size_t loom_amdgpu_fragment_memory_contract_candidate_count(
    const loom_amdgpu_fragment_memory_contract_candidate_list_t* candidates) {
  return candidates != NULL ? candidates->descriptor_count
                            : loom_amdgpu_matrix_contract_descriptor_count();
}

static const loom_amdgpu_matrix_contract_descriptor_t*
loom_amdgpu_fragment_memory_contract_candidate_at(
    const loom_amdgpu_fragment_memory_contract_candidate_list_t* candidates,
    iree_host_size_t index) {
  return candidates != NULL ? candidates->descriptors[index]
                            : loom_amdgpu_matrix_contract_descriptor_at(index);
}

static bool loom_amdgpu_fragment_memory_target_layout(
    const loom_amdgpu_fragment_memory_environment_t* environment,
    loom_contract_operand_role_t role,
    loom_amdgpu_memory_operation_kind_t operation_kind, loom_value_id_t payload,
    loom_type_t payload_type, loom_type_t view_type, loom_value_id_t rows,
    loom_value_id_t columns,
    const loom_value_fact_storage_schema_t* view_storage_schema,
    const loom_amdgpu_matrix_fragment_layout_t** out_layout,
    loom_scalar_type_t* out_expected_element_type,
    loom_amdgpu_fragment_memory_payload_form_t* out_payload_form,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
  *out_layout = NULL;
  *out_expected_element_type = LOOM_SCALAR_TYPE_COUNT_;
  *out_payload_form = LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
  if (environment->bundle == NULL || environment->bundle->snapshot == NULL ||
      environment->descriptor_set == NULL) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.target_layout"));
  }

  const loom_amdgpu_fragment_memory_contract_candidate_list_t* candidates =
      environment->contract_candidates;
  const iree_host_size_t descriptor_count =
      loom_amdgpu_fragment_memory_contract_candidate_count(candidates);
  const uint32_t wave_size = candidates != NULL
                                 ? candidates->wave_size
                                 : environment->bundle->snapshot->subgroup_size;
  const loom_amdgpu_matrix_fragment_layout_t* best_layout = NULL;
  loom_scalar_type_t best_element_type = LOOM_SCALAR_TYPE_COUNT_;
  loom_amdgpu_fragment_memory_payload_form_t best_payload_form =
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
  loom_amdgpu_fragment_memory_layout_match_t best_match =
      LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NONE;
  bool rejected_shape = false;
  bool rejected_payload_layout = false;
  bool rejected_payload_form = false;
  bool rejected_target_layout = false;
  for (iree_host_size_t i = 0; i < descriptor_count; ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        loom_amdgpu_fragment_memory_contract_candidate_at(candidates, i);
    if (candidates == NULL &&
        !loom_amdgpu_fragment_memory_contract_is_available(
            descriptor, environment->descriptor_set, environment->feature_bits,
            wave_size)) {
      continue;
    }
    loom_amdgpu_fragment_memory_role_storage_t role_storage = {0};
    if (!loom_amdgpu_fragment_memory_descriptor_role_storage(descriptor, role,
                                                             &role_storage)) {
      continue;
    }
    if (!loom_amdgpu_fragment_memory_numeric_schema_matches(
            role_storage.payload.numeric_type, view_storage_schema)) {
      rejected_payload_layout = true;
      continue;
    }
    const loom_scalar_type_t expected_element_type = role_storage.element_type;
    const loom_amdgpu_matrix_fragment_layout_t* layout =
        loom_amdgpu_matrix_contract_descriptor_fragment_layout(descriptor);
    if (layout == NULL) {
      loom_amdgpu_matrix_payload_shape_t descriptor_payload = {0};
      if (loom_amdgpu_fragment_memory_descriptor_payload(descriptor, role,
                                                         &descriptor_payload) &&
          loom_amdgpu_fragment_memory_descriptor_shape_matches(
              environment->fact_table, descriptor, role, rows, columns) &&
          loom_amdgpu_fragment_memory_payload_matches_descriptor_storage(
              payload_type, expected_element_type, &descriptor_payload)) {
        rejected_target_layout = true;
      }
      continue;
    }
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
        loom_amdgpu_matrix_fragment_role_layout(layout, role);
    if (role_layout == NULL) {
      continue;
    }
    const bool shape_matches = loom_amdgpu_fragment_memory_shape_matches(
        environment->fact_table, layout, role, rows, columns);
    const bool payload_matches = loom_amdgpu_fragment_memory_payload_matches(
        payload_type, view_type, view_storage_schema, operation_kind,
        expected_element_type, role_layout);
    loom_amdgpu_fragment_memory_payload_form_t payload_form =
        LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
    const loom_amdgpu_fragment_memory_element_match_t match =
        loom_amdgpu_fragment_memory_payload_element_match(
            environment->descriptor_set, operation_kind, role, payload_type,
            view_type, view_storage_schema, expected_element_type, role_layout,
            &payload_form);
    const loom_amdgpu_fragment_memory_layout_match_t layout_match =
        loom_amdgpu_fragment_memory_layout_match_rank(
            environment, payload, role_layout, match, payload_form);
    if (!shape_matches) {
      if (payload_matches &&
          match != LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE) {
        rejected_shape = true;
      }
      continue;
    }
    if (!payload_matches) {
      if (match != LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE) {
        rejected_payload_layout = true;
      }
      continue;
    }
    if (match == LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE) {
      rejected_payload_form = true;
      continue;
    }
    if (layout_match <= best_match) {
      continue;
    }
    best_layout = layout;
    best_element_type = expected_element_type;
    best_payload_form = payload_form;
    best_match = layout_match;
    if (layout_match ==
        LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NARROWED_F32_RESULT) {
      break;
    }
  }
  if (best_layout != NULL) {
    *out_layout = best_layout;
    *out_expected_element_type = best_element_type;
    *out_payload_form = best_payload_form;
    return true;
  }

  return loom_amdgpu_fragment_memory_reject(
      diagnostic,
      rejected_payload_form
          ? (operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_STORE
                 ? IREE_SV("fragment_memory.store_conversion")
                 : IREE_SV("fragment_memory.payload_form"))
          : (rejected_target_layout
                 ? IREE_SV("fragment_memory.target_layout")
                 : (rejected_payload_layout
                        ? IREE_SV("fragment_memory.payload_layout")
                        : (rejected_shape
                               ? IREE_SV("fragment_memory.shape")
                               : IREE_SV("fragment_memory.target_layout")))));
}

static uint16_t loom_amdgpu_fragment_repack_log2_u16(uint16_t value) {
  uint16_t log2_value = 0;
  while (value > 1) {
    value = (uint16_t)(value >> 1);
    ++log2_value;
  }
  return log2_value;
}

static uint16_t loom_amdgpu_fragment_repack_log2_u32(uint32_t value) {
  uint16_t log2_value = 0;
  while (value > 1) {
    value >>= 1;
    ++log2_value;
  }
  return log2_value;
}

static loom_amdgpu_descriptor_ref_t
loom_amdgpu_fragment_memory_compare_i32_src1_inline_ref(
    loom_amdgpu_descriptor_ref_t descriptor_ref) {
  switch (descriptor_ref) {
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32:
      return kLoomAmdgpuVectorCmpiCompareDescriptorCandidates
          [LOOM_VECTOR_CMPI_PREDICATE_EQ]
              .src1_inline_descriptor_ref;
    case LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32:
      return kLoomAmdgpuVectorCmpiCompareDescriptorCandidates
          [LOOM_VECTOR_CMPI_PREDICATE_NE]
              .src1_inline_descriptor_ref;
    default:
      return LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  }
}

static bool loom_amdgpu_can_emit_compare_u32_immediate(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t descriptor_ref, uint32_t immediate) {
  const loom_amdgpu_descriptor_ref_t src1_inline_ref =
      loom_amdgpu_fragment_memory_compare_i32_src1_inline_ref(descriptor_ref);
  if (immediate <= LOOM_AMDGPU_SOURCE_INLINE_U32_MAX &&
      src1_inline_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
      loom_amdgpu_descriptor_set_has_ref(descriptor_set, src1_inline_ref)) {
    return true;
  }
  return loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref) &&
         loom_amdgpu_descriptor_set_has_ref(
             descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32);
}

static loom_amdgpu_descriptor_ref_t loom_amdgpu_dpp_b32_descriptor_ref(
    const loom_low_descriptor_set_t* descriptor_set) {
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP16)) {
    return LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP16;
  }
  if (loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP)) {
    return LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32_DPP;
  }
  return LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
}

static bool loom_amdgpu_fragment_repack_uses_source_register_bit_tree(
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  return plan->source_register_count >= 4 &&
         loom_amdgpu_u32_is_power_of_two(plan->source_register_count);
}

static bool loom_amdgpu_fragment_repack_has_static_zero_source_byte_base(
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  return plan->source_map_kind ==
             LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN &&
         plan->lane_group_count == 1;
}

static bool
loom_amdgpu_fragment_repack_has_result_to_lhs_bf16_bpermute_descriptors(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  static const loom_amdgpu_descriptor_ref_t kRequiredDescriptorRefs[] = {
      LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32,
  };
  if (!loom_amdgpu_descriptor_set_has_all_refs(
          descriptor_set, kRequiredDescriptorRefs,
          IREE_ARRAYSIZE(kRequiredDescriptorRefs)) ||
      !loom_amdgpu_bf16_descriptor_set_can_emit_f32_pair_to_packed_bf16(
          descriptor_set)) {
    return false;
  }
  if (plan->lane_group_count > 1 &&
      !loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
          plan->lane_group_count - 1u)) {
    return false;
  }
  switch (plan->source_map_kind) {
    case LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN:
      if (plan->lane_group_count > 1 &&
          !loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
              descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
              plan->source_lane_group_byte_shift)) {
        return false;
      }
      if (plan->lane_group_count == 1 &&
          plan->result_lane_div_byte_shift == 0 &&
          !loom_amdgpu_descriptor_set_has_ref(
              descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32)) {
        return false;
      }
      if (plan->source_register_count > 1 &&
          !loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
              descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
              loom_amdgpu_fragment_repack_log2_u16(plan->lane_group_count))) {
        return false;
      }
      break;
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN:
      if (plan->source_register_count > 1 &&
          !loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
              descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
              plan->source_register_count - 1u)) {
        return false;
      }
      if (plan->lane_group_count > 1 &&
          !loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
              descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
              loom_amdgpu_fragment_repack_log2_u16(
                  plan->source_register_count))) {
        return false;
      }
      if (!loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
              descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
              plan->source_lane_group_byte_shift)) {
        return false;
      }
      break;
    default:
      return false;
  }
  if (plan->result_lane_div_byte_shift != 0 &&
      !loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
          plan->result_lane_div_byte_shift)) {
    return false;
  }
  if (plan->result_lane_div_byte_shift != 0 &&
      !loom_amdgpu_fragment_repack_has_static_zero_source_byte_base(plan) &&
      !loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32)) {
    return false;
  }
  if (loom_amdgpu_fragment_repack_uses_source_register_bit_tree(plan)) {
    if (!loom_amdgpu_can_emit_compare_u32_immediate(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32, 0)) {
      return false;
    }
    const uint16_t bit_count =
        loom_amdgpu_fragment_repack_log2_u16(plan->source_register_count);
    for (uint16_t i = 0; i < bit_count; ++i) {
      if (!loom_amdgpu_descriptor_set_can_emit_vgpr_binary_immediate(
              descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
              UINT32_C(1) << i)) {
        return false;
      }
    }
    return true;
  }
  for (uint16_t i = 1; i < plan->source_register_count; ++i) {
    if (!loom_amdgpu_can_emit_compare_u32_immediate(
            descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, i)) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_fragment_repack_can_use_dpp_packed_source_pairs(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  const uint16_t required_even_lane_byte_base_shift = 3;
  if (loom_amdgpu_dpp_b32_descriptor_ref(descriptor_set) ==
          LOOM_AMDGPU_DESCRIPTOR_REF_NONE ||
      !loom_matrix_fragment_role_has_contiguous_lane_xor1_columns(
          layout, plan->source_role) ||
      plan->source_lane_group_byte_shift < required_even_lane_byte_base_shift ||
      (plan->result_lane_div_byte_shift != 0 &&
       plan->result_lane_div_byte_shift < required_even_lane_byte_base_shift)) {
    return false;
  }
  return loom_amdgpu_fragment_repack_has_result_to_lhs_bf16_bpermute_descriptors(
      descriptor_set, plan);
}

static loom_amdgpu_matrix_feature_bits_t
loom_amdgpu_fragment_memory_feature_bits_from_target_ref(
    const loom_module_t* module, loom_symbol_ref_t target_ref) {
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_processor_from_ref(module, target_ref);
  if (processor == NULL) {
    return 0;
  }
  loom_amdgpu_matrix_feature_bits_t feature_bits = 0;
  (void)loom_amdgpu_matrix_feature_bits_from_profile(processor->features.matrix,
                                                     &feature_bits);
  return feature_bits;
}

static bool
loom_amdgpu_fragment_repack_select_result_to_lhs_bf16_bpermute_strategy(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_matrix_fragment_role_layout_t* source_role_layout,
    const loom_amdgpu_matrix_fragment_role_layout_t* result_role_layout,
    loom_amdgpu_fragment_repack_plan_t* plan) {
  if (plan->source_role != LOOM_CONTRACT_OPERAND_ROLE_RESULT ||
      plan->result_role != LOOM_CONTRACT_OPERAND_ROLE_LHS ||
      loom_type_element_type(plan->source_type) != LOOM_SCALAR_TYPE_F32 ||
      loom_type_element_type(plan->result_type) != LOOM_SCALAR_TYPE_BF16 ||
      source_role_layout->elements_per_register != 1 ||
      source_role_layout->element_bit_count != 32 ||
      result_role_layout->elements_per_register !=
          LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT ||
      result_role_layout->element_bit_count !=
          LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_BIT_COUNT ||
      source_role_layout->register_count == 0 ||
      result_role_layout->register_count == 0) {
    return false;
  }

  switch (source_role_layout->map_kind) {
    case LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN:
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN:
      break;
    default:
      return false;
  }
  const bool result_uses_lane_div_reduction =
      result_role_layout->map_kind ==
      LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION;
  switch (result_role_layout->map_kind) {
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION:
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION:
      break;
    default:
      return false;
  }

  if (!loom_amdgpu_u32_is_power_of_two(layout->tile_shape.result_row_count) ||
      !loom_amdgpu_u32_is_power_of_two(
          layout->tile_shape.result_column_count) ||
      !loom_amdgpu_u32_is_power_of_two(layout->tile_shape.reduction_count)) {
    return false;
  }

  const uint16_t lane_group_count =
      (uint16_t)(layout->wave_size / layout->tile_shape.result_column_count);
  if (lane_group_count == 0 ||
      (layout->wave_size % layout->tile_shape.result_column_count) != 0 ||
      !loom_amdgpu_u32_is_power_of_two(lane_group_count) ||
      (layout->tile_shape.result_row_count % lane_group_count) != 0 ||
      source_role_layout->register_count !=
          layout->tile_shape.result_row_count / lane_group_count ||
      source_role_layout->register_count >
          LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES ||
      (layout->tile_shape.reduction_count %
       LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT) != 0 ||
      result_role_layout->register_count >
          LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES) {
    return false;
  }
  const uint16_t lane_div_count =
      (uint16_t)(layout->wave_size / layout->tile_shape.result_row_count);
  if (lane_div_count == 0 ||
      (layout->wave_size % layout->tile_shape.result_row_count) != 0 ||
      !loom_amdgpu_u32_is_power_of_two(lane_div_count)) {
    return false;
  }
  const uint16_t expected_result_register_count =
      (uint16_t)(layout->tile_shape.reduction_count /
                 (LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT *
                  (result_uses_lane_div_reduction ? lane_div_count : 1u)));
  if (expected_result_register_count == 0 ||
      result_role_layout->register_count != expected_result_register_count) {
    return false;
  }

  const uint32_t source_lane_group_byte_count =
      (uint32_t)layout->tile_shape.result_column_count *
      LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT;
  if (!loom_amdgpu_u32_is_power_of_two(source_lane_group_byte_count)) {
    return false;
  }
  const uint32_t result_lane_div_byte_count =
      result_uses_lane_div_reduction
          ? (uint32_t)result_role_layout->register_count *
                LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT *
                LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT
          : 0u;
  if (result_lane_div_byte_count != 0 &&
      !loom_amdgpu_u32_is_power_of_two(result_lane_div_byte_count)) {
    return false;
  }

  plan->strategy =
      LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_BPERMUTE;
  plan->reason = LOOM_AMDGPU_FRAGMENT_REPACK_REASON_NONE;
  plan->layout_kind = (loom_amdgpu_matrix_fragment_layout_kind_t)layout->kind;
  plan->source_register_count = source_role_layout->register_count;
  plan->result_register_count = result_role_layout->register_count;
  plan->lane_group_count = lane_group_count;
  plan->lane_divisor = layout->tile_shape.result_row_count;
  plan->source_lane_group_byte_shift =
      loom_amdgpu_fragment_repack_log2_u32(source_lane_group_byte_count);
  plan->result_lane_div_byte_shift =
      result_lane_div_byte_count == 0
          ? 0
          : loom_amdgpu_fragment_repack_log2_u32(result_lane_div_byte_count);
  plan->source_map_kind = source_role_layout->map_kind;

  if (loom_amdgpu_fragment_repack_can_use_dpp_packed_source_pairs(
          descriptor_set, layout, plan)) {
    plan->strategy =
        LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_DPP_BPERMUTE;
    return true;
  }

  if (!loom_amdgpu_fragment_repack_has_result_to_lhs_bf16_bpermute_descriptors(
          descriptor_set, plan)) {
    plan->reason = LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TARGET_PACKETS;
    plan->strategy = LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_DIAGNOSTIC;
    return false;
  }

  return true;
}

static bool loom_amdgpu_fragment_repack_select_target_strategy(
    const loom_module_t* module, const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_fragment_memory_contract_candidate_list_t*
        contract_candidates,
    loom_symbol_ref_t target_ref, loom_amdgpu_fragment_repack_plan_t* plan) {
  if (bundle == NULL || bundle->snapshot == NULL || descriptor_set == NULL) {
    plan->reason = LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TARGET_LAYOUT;
    return false;
  }

  if (!loom_amdgpu_fragment_repack_contract_role(plan->source_role_flags,
                                                 &plan->source_role) ||
      !loom_amdgpu_fragment_repack_contract_role(plan->result_role_flags,
                                                 &plan->result_role)) {
    return false;
  }

  const loom_amdgpu_matrix_feature_bits_t feature_bits =
      contract_candidates != NULL
          ? contract_candidates->feature_bits
          : loom_amdgpu_fragment_memory_feature_bits_from_target_ref(
                module, target_ref);
  const uint32_t wave_size = contract_candidates != NULL
                                 ? contract_candidates->wave_size
                                 : bundle->snapshot->subgroup_size;
  const iree_host_size_t descriptor_count =
      loom_amdgpu_fragment_memory_contract_candidate_count(contract_candidates);
  bool found_storage_candidate = false;
  for (iree_host_size_t i = 0; i < descriptor_count; ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        loom_amdgpu_fragment_memory_contract_candidate_at(contract_candidates,
                                                          i);
    const loom_amdgpu_matrix_fragment_layout_t* layout =
        loom_amdgpu_matrix_contract_descriptor_fragment_layout(descriptor);
    if (layout == NULL ||
        (contract_candidates == NULL &&
         !loom_amdgpu_fragment_memory_contract_is_available(
             descriptor, descriptor_set, feature_bits, wave_size))) {
      continue;
    }

    loom_amdgpu_fragment_memory_role_storage_t source_storage = {0};
    loom_amdgpu_fragment_memory_role_storage_t result_storage = {0};
    if (!loom_amdgpu_fragment_memory_descriptor_role_storage(
            descriptor, plan->source_role, &source_storage) ||
        !loom_amdgpu_fragment_memory_descriptor_role_storage(
            descriptor, plan->result_role, &result_storage)) {
      continue;
    }

    const loom_amdgpu_matrix_fragment_role_layout_t* source_role_layout =
        loom_amdgpu_matrix_fragment_role_layout(layout, plan->source_role);
    const loom_amdgpu_matrix_fragment_role_layout_t* result_role_layout =
        loom_amdgpu_matrix_fragment_role_layout(layout, plan->result_role);
    if (!loom_amdgpu_fragment_memory_payload_matches_role_storage(
            plan->source_type, source_storage.element_type,
            source_role_layout) ||
        !loom_amdgpu_fragment_memory_payload_matches_role_storage(
            plan->result_type, result_storage.element_type,
            result_role_layout)) {
      continue;
    }
    found_storage_candidate = true;

    if (loom_amdgpu_fragment_repack_select_result_to_lhs_bf16_bpermute_strategy(
            descriptor_set, layout, source_role_layout, result_role_layout,
            plan)) {
      return true;
    }
  }

  if (plan->reason != LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TARGET_PACKETS) {
    plan->reason = found_storage_candidate
                       ? LOOM_AMDGPU_FRAGMENT_REPACK_REASON_LAYOUT_STRATEGY
                       : LOOM_AMDGPU_FRAGMENT_REPACK_REASON_TARGET_LAYOUT;
  }
  return false;
}

static bool loom_amdgpu_fragment_memory_view_matches(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_type_t view_type, loom_type_t payload_type,
    const loom_value_fact_storage_schema_t* view_storage_schema,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_scalar_type_t expected_element_type,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout,
    loom_amdgpu_fragment_memory_payload_form_t payload_form,
    loom_vector_memory_access_t* out_access) {
  *out_access = (loom_vector_memory_access_t){0};
  if (!loom_type_is_view(view_type) ||
      loom_type_rank(view_type) != LOOM_AMDGPU_FRAGMENT_VIEW_RANK ||
      !loom_type_is_all_static(view_type)) {
    return false;
  }
  const loom_scalar_type_t view_element_type =
      loom_type_element_type(view_type);
  const loom_scalar_type_t payload_element_type =
      loom_type_element_type(payload_type);
  const bool view_is_narrowed =
      payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16;
  const bool view_is_extended =
      payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_EXTEND_F16_TO_F32;
  const bool view_is_packed_16bit_result =
      payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT;
  const bool view_is_fp8_to_16bit_load =
      loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
          payload_form);
  const bool view_is_encoded_native_payload =
      payload_form == LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE &&
      loom_amdgpu_fragment_memory_payload_matches_encoded_storage(
          payload_type, view_type, view_storage_schema, expected_element_type,
          role_layout);
  if (loom_type_is_vector(payload_type) &&
      payload_element_type != view_element_type &&
      !(view_is_narrowed && payload_element_type == expected_element_type) &&
      !(view_is_extended && view_element_type == expected_element_type) &&
      !(view_is_fp8_to_16bit_load &&
        payload_element_type == expected_element_type) &&
      !view_is_encoded_native_payload) {
    return false;
  }

  switch (role_layout->role) {
    case LOOM_CONTRACT_OPERAND_ROLE_LHS:
    case LOOM_CONTRACT_OPERAND_ROLE_RHS:
      if (!loom_scalar_type_is_float(expected_element_type) ||
          loom_scalar_type_bitwidth(expected_element_type) !=
              role_layout->element_bit_count) {
        return false;
      }
      break;
    case LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR:
    case LOOM_CONTRACT_OPERAND_ROLE_RESULT:
      if (!loom_scalar_type_is_float(expected_element_type) ||
          loom_scalar_type_bitwidth(expected_element_type) !=
              role_layout->element_bit_count) {
        return false;
      }
      if (view_element_type != expected_element_type && !view_is_narrowed &&
          !view_is_packed_16bit_result && !view_is_encoded_native_payload) {
        return false;
      }
      break;
    case LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN:
    default:
      return false;
  }
  if (view_element_type != expected_element_type && !view_is_narrowed &&
      !view_is_packed_16bit_result && !view_is_fp8_to_16bit_load &&
      !view_is_encoded_native_payload) {
    return false;
  }

  loom_type_t scalar_vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, view_element_type,
                          loom_dim_pack_static(1), /*encoding_id=*/0);
  const loom_fact_context_t* fact_context =
      fact_table ? &fact_table->context : NULL;
  if (!loom_vector_memory_access_describe(fact_context, module, view_type,
                                          scalar_vector_type, out_access)) {
    return false;
  }
  return out_access->static_element_byte_count > 0 &&
         out_access->static_element_byte_count <= UINT16_MAX &&
         (out_access->layout_kind == LOOM_VECTOR_MEMORY_LAYOUT_DENSE ||
          out_access->layout_kind == LOOM_VECTOR_MEMORY_LAYOUT_STRIDED);
}

static bool loom_amdgpu_fragment_memory_fill_view_strides(
    const loom_vector_memory_access_t* access,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout,
    uint32_t* out_axis_byte_strides,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
  for (uint8_t axis = 0; axis < access->view_rank; ++axis) {
    int64_t element_stride = 0;
    if (!loom_vector_memory_access_static_axis_stride(access, axis,
                                                      &element_stride) ||
        element_stride <= 0) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.view_stride"));
    }
    int64_t byte_stride = 0;
    if (!iree_checked_mul_i64(element_stride, access->static_element_byte_count,
                              &byte_stride) ||
        byte_stride <= 0 || byte_stride > UINT32_MAX) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.view_stride"));
    }
    out_axis_byte_strides[axis] = (uint32_t)byte_stride;
  }

  if (role_layout->elements_per_register > 1) {
    uint8_t packed_axis = UINT8_MAX;
    switch (role_layout->map_kind) {
      case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION:
      case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION:
        packed_axis = 1;
        break;
      case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION:
      case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION:
        packed_axis = 0;
        break;
      case LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN_LOW_SUBWORD:
      case LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN_LOW_SUBWORD:
      case LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_PACKED_ROW_COLUMN:
        return true;
      default:
        return loom_amdgpu_fragment_memory_reject(
            diagnostic, IREE_SV("fragment_memory.layout_map"));
    }
    if (packed_axis >= access->view_rank ||
        out_axis_byte_strides[packed_axis] !=
            (uint32_t)access->static_element_byte_count) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.packed_axis_stride"));
    }
  }

  return true;
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
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_value_fact_memory_space_t memory_space, uint16_t packet_register_count,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (operation_kind >= LOOM_AMDGPU_MEMORY_OPERATION_COUNT_ ||
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
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_value_fact_memory_space_t memory_space,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (operation_kind >= LOOM_AMDGPU_MEMORY_OPERATION_COUNT_) {
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

static bool loom_amdgpu_fragment_memory_narrowed_store_descriptor_ref(
    loom_value_fact_memory_space_t memory_space, uint16_t result_register_count,
    uint16_t* out_packet_register_count,
    loom_amdgpu_descriptor_ref_t* out_descriptor_ref) {
  *out_packet_register_count = 0;
  *out_descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (result_register_count == 1) {
    *out_packet_register_count = 1;
    return loom_amdgpu_fragment_memory_16bit_descriptor_ref(
        LOOM_AMDGPU_MEMORY_OPERATION_STORE, memory_space, out_descriptor_ref);
  }
  if ((result_register_count & 1u) != 0) {
    return false;
  }
  *out_packet_register_count = result_register_count / 2u;
  return loom_amdgpu_fragment_memory_descriptor_ref(
      LOOM_AMDGPU_MEMORY_OPERATION_STORE, memory_space,
      *out_packet_register_count, out_descriptor_ref);
}

static bool loom_amdgpu_fragment_memory_has_vgpr_immediate_ref(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_descriptor_ref_t literal_ref,
    loom_amdgpu_descriptor_ref_t src0_inline_ref) {
  return loom_amdgpu_descriptor_set_has_ref(descriptor_set, literal_ref) ||
         loom_amdgpu_descriptor_set_has_ref(descriptor_set, src0_inline_ref);
}

static bool loom_amdgpu_fragment_memory_space_supports_access(
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_value_fact_memory_space_t memory_space,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout,
    loom_amdgpu_fragment_memory_payload_form_t payload_form) {
  if (loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
          payload_form)) {
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    return operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_LOAD &&
           (loom_amdgpu_fragment_memory_descriptor_ref(
                LOOM_AMDGPU_MEMORY_OPERATION_LOAD, memory_space,
                /*packet_register_count=*/1, &descriptor_ref) ||
            loom_amdgpu_fragment_memory_16bit_descriptor_ref(
                LOOM_AMDGPU_MEMORY_OPERATION_LOAD, memory_space,
                &descriptor_ref));
  }

  if (loom_amdgpu_fragment_memory_role_uses_scalar_b16_packets(role_layout)) {
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
        LOOM_AMDGPU_MEMORY_OPERATION_LOAD, memory_space, &descriptor_ref);
  }

  if (payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16) {
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

static bool loom_amdgpu_fragment_memory_payload_form_has_descriptors(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_amdgpu_fragment_memory_payload_form_t payload_form) {
  switch (payload_form) {
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE:
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT:
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_BF16:
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_F16:
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16:
      return true;
    case LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_EXTEND_F16_TO_F32:
      return loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_F16);
  }
  return false;
}

static bool loom_amdgpu_fragment_memory_source_plan_supports_addressing(
    const loom_low_source_memory_access_plan_t* source,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
  if (source->static_byte_offset < 0) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.base_offset"));
  }
  for (uint8_t i = 0; i < source->dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &source->dynamic_terms[i];
    if (term->stride_value_count != 0 || term->byte_stride < 0 ||
        term->byte_stride > UINT32_MAX) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.dynamic_stride"));
    }
  }
  return true;
}

static void loom_amdgpu_fragment_memory_source_from_op(
    const loom_op_t* source_op,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_amdgpu_fragment_memory_source_t* out_source) {
  *out_source = (loom_amdgpu_fragment_memory_source_t){
      .vector_role = LOOM_VECTOR_ROLE_COUNT_,
      .view = LOOM_VALUE_ID_INVALID,
      .payload = LOOM_VALUE_ID_INVALID,
      .rows = LOOM_VALUE_ID_INVALID,
      .columns = LOOM_VALUE_ID_INVALID,
      .static_indices = loom_attr_absent(),
      .dynamic_indices = {0},
      .cache_scope = loom_attr_absent(),
      .cache_temporal = loom_attr_absent(),
  };
  if (operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_LOAD) {
    out_source->vector_role = loom_vector_fragment_load_role(source_op);
    out_source->view = loom_vector_fragment_load_view(source_op);
    out_source->payload = loom_vector_fragment_load_result(source_op);
    out_source->rows = loom_vector_fragment_load_rows(source_op);
    out_source->columns = loom_vector_fragment_load_columns(source_op);
    out_source->static_indices =
        loom_vector_fragment_load_static_indices(source_op);
    out_source->dynamic_indices = loom_vector_fragment_load_indices(source_op);
    out_source->cache_scope = loom_op_attrs(
        source_op)[loom_vector_fragment_load_cache_scope_ATTR_INDEX];
    out_source->cache_temporal = loom_op_attrs(
        source_op)[loom_vector_fragment_load_cache_temporal_ATTR_INDEX];
    return;
  }

  out_source->vector_role = loom_vector_fragment_store_role(source_op);
  out_source->view = loom_vector_fragment_store_view(source_op);
  out_source->payload = loom_vector_fragment_store_value(source_op);
  out_source->rows = loom_vector_fragment_store_rows(source_op);
  out_source->columns = loom_vector_fragment_store_columns(source_op);
  out_source->static_indices =
      loom_vector_fragment_store_static_indices(source_op);
  out_source->dynamic_indices = loom_vector_fragment_store_indices(source_op);
  out_source->cache_scope = loom_op_attrs(
      source_op)[loom_vector_fragment_store_cache_scope_ATTR_INDEX];
  out_source->cache_temporal = loom_op_attrs(
      source_op)[loom_vector_fragment_store_cache_temporal_ATTR_INDEX];
}

static loom_amdgpu_fragment_memory_narrowed_result_sources_t
loom_amdgpu_fragment_memory_narrowed_result_sources(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t payload,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout) {
  loom_amdgpu_fragment_memory_narrowed_result_sources_t sources = {
      .round_source = LOOM_VALUE_ID_INVALID,
      .scale_source = LOOM_VALUE_ID_INVALID,
      .packed_source = LOOM_VALUE_ID_INVALID,
      .packed_register_count = 0,
  };
  sources.packed_source =
      loom_amdgpu_fragment_memory_same_lane_packed_bf16_source(
          module, fact_table, payload, role_layout,
          &sources.packed_register_count);
  if (sources.packed_source != LOOM_VALUE_ID_INVALID) {
    return sources;
  }

  if (loom_amdgpu_fragment_memory_is_f32_result_source(module, payload,
                                                       role_layout)) {
    sources.round_source = payload;
  } else {
    sources.round_source = loom_amdgpu_fragment_memory_same_lane_round_source(
        module, fact_table, payload, role_layout);
  }
  if (sources.round_source == LOOM_VALUE_ID_INVALID) {
    return sources;
  }

  loom_value_fact_uniform_scale_origin_t scale_origin = {0};
  if (!loom_value_fact_table_query_uniform_scale_origin(
          fact_table, module, sources.round_source, &scale_origin) ||
      !loom_amdgpu_fragment_memory_is_f32_result_source(
          module, scale_origin.source_value_id, role_layout)) {
    return sources;
  }
  sources.round_source = scale_origin.source_value_id;
  sources.scale_source = scale_origin.scale_value_id;
  return sources;
}

static bool loom_amdgpu_fragment_memory_analyze(
    const loom_amdgpu_fragment_memory_environment_t* environment,
    const loom_amdgpu_fragment_memory_source_t* source,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_amdgpu_fragment_memory_plan_t* out_plan,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
  if (out_plan != NULL) {
    *out_plan = (loom_amdgpu_fragment_memory_plan_t){0};
  }
  if (!loom_kernel_def_isa(environment->source_function.op)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.kernel_entry"));
  }
  if (!loom_attr_is_absent(source->cache_scope) ||
      !loom_attr_is_absent(source->cache_temporal)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.cache_policy"));
  }

  loom_contract_operand_role_t role = LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN;
  if (!loom_amdgpu_fragment_memory_role_from_vector_role(source->vector_role,
                                                         &role)) {
    return loom_amdgpu_fragment_memory_reject(diagnostic,
                                              IREE_SV("fragment_memory.role"));
  }
  const loom_type_t payload_type =
      loom_module_value_type(environment->module, source->payload);
  const loom_type_t view_type =
      loom_module_value_type(environment->module, source->view);
  loom_value_fact_storage_schema_t view_storage_schema = {0};
  const loom_fact_context_t* fact_context =
      environment->fact_table != NULL ? &environment->fact_table->context
                                      : NULL;
  const bool has_view_storage_schema =
      loom_encoding_query_type_storage_schema(
          fact_context, environment->module, view_type, &view_storage_schema) &&
      !loom_value_fact_encoded_operand_schema_is_unknown(
          view_storage_schema.encoded_operand);
  if (has_view_storage_schema &&
      iree_any_bit_set(view_storage_schema.encoded_operand.sparsity_policy,
                       LOOM_VALUE_FACT_SPARSITY_POLICY_ALL)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.sparse_layout"));
  }
  const loom_amdgpu_matrix_fragment_layout_t* layout = NULL;
  loom_scalar_type_t expected_element_type = LOOM_SCALAR_TYPE_COUNT_;
  loom_amdgpu_fragment_memory_payload_form_t payload_form =
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
  if (!loom_amdgpu_fragment_memory_target_layout(
          environment, role, operation_kind, source->payload, payload_type,
          view_type, source->rows, source->columns,
          has_view_storage_schema ? &view_storage_schema : NULL, &layout,
          &expected_element_type, &payload_form, diagnostic)) {
    return false;
  }

  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, role);
  if (role_layout == NULL ||
      role_layout->register_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.role_layout"));
  }
  if (!loom_amdgpu_fragment_memory_payload_matches(
          payload_type, view_type,
          has_view_storage_schema ? &view_storage_schema : NULL, operation_kind,
          expected_element_type, role_layout)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.payload"));
  }
  uint16_t payload_register_count = 0;
  if (!loom_amdgpu_fragment_memory_payload_storage_register_count(
          payload_type, &payload_register_count)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.payload_storage"));
  }
  if (loom_amdgpu_fragment_memory_requires_native_payload_storage(
          operation_kind, role_layout, payload_form) &&
      !loom_amdgpu_fragment_memory_payload_has_native_storage(
          environment->fact_table, source->payload)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.payload_storage"));
  }

  loom_vector_memory_access_t access = {0};
  if (!loom_amdgpu_fragment_memory_view_matches(
          environment->module, environment->fact_table, view_type, payload_type,
          has_view_storage_schema ? &view_storage_schema : NULL, operation_kind,
          expected_element_type, role_layout, payload_form, &access)) {
    return loom_amdgpu_fragment_memory_reject(diagnostic,
                                              IREE_SV("fragment_memory.view"));
  }
  uint32_t axis_byte_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK] = {0};
  if (!loom_amdgpu_fragment_memory_fill_view_strides(
          &access, role_layout, axis_byte_strides, diagnostic)) {
    return false;
  }

  loom_low_source_memory_operation_kind_t source_operation_kind =
      LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  if (!loom_amdgpu_fragment_memory_source_operation_kind(
          operation_kind, &source_operation_kind)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.operation"));
  }
  const loom_type_t scalar_vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, loom_type_element_type(view_type),
                          loom_dim_pack_static(1), /*encoding_id=*/0);
  loom_low_source_memory_access_plan_t source_access = {0};
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  if (!loom_low_source_memory_access_plan_build_indexed_with_view_regions(
          environment->module, environment->fact_table,
          environment->view_regions, source_operation_kind, source->view,
          source->dynamic_indices, source->static_indices, scalar_vector_type,
          (loom_vector_memory_cache_policy_t){0}, &source_access,
          &source_diagnostic)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, loom_low_source_memory_access_rejection_key(
                        source_diagnostic.rejection_bits));
  }
  if (source_access.element_byte_count > UINT16_MAX) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("source_memory.element_width"));
  }
  if (!loom_amdgpu_fragment_memory_source_plan_supports_addressing(
          &source_access, diagnostic)) {
    return false;
  }

  const loom_amdgpu_fragment_memory_narrowed_result_sources_t
      narrowed_result_sources =
          payload_form ==
                  LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16
              ? loom_amdgpu_fragment_memory_narrowed_result_sources(
                    environment->module, environment->fact_table,
                    source->payload, role_layout)
              : (loom_amdgpu_fragment_memory_narrowed_result_sources_t){
                    .round_source = LOOM_VALUE_ID_INVALID,
                    .scale_source = LOOM_VALUE_ID_INVALID,
                    .packed_source = LOOM_VALUE_ID_INVALID,
                    .packed_register_count = 0,
                };
  if (!loom_amdgpu_fragment_memory_payload_form_has_descriptors(
          environment->descriptor_set, payload_form)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.store_conversion"));
  }

  if (source_access.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    uint64_t root_byte_offset = 0;
    if (!loom_amdgpu_source_alloca_layout_lookup_root(
            environment->alloca_layout, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
            source_access.root_value_id, &root_byte_offset) ||
        root_byte_offset > INT64_MAX) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.workgroup_root"));
    }
    if (!iree_checked_add_i64(source_access.static_byte_offset,
                              (int64_t)root_byte_offset,
                              &source_access.static_byte_offset)) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.base_offset"));
    }
  }
  if (!loom_amdgpu_fragment_memory_source_plan_supports_addressing(
          &source_access, diagnostic)) {
    return false;
  }

  if (!loom_amdgpu_fragment_memory_space_supports_access(
          operation_kind, source_access.memory_space, role_layout,
          payload_form)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.memory_space"));
  }

  if (narrowed_result_sources.packed_source != LOOM_VALUE_ID_INVALID) {
    payload_register_count = narrowed_result_sources.packed_register_count;
  }

  if (out_plan != NULL) {
    *out_plan = (loom_amdgpu_fragment_memory_plan_t){
        .operation_kind = operation_kind,
        .role = role,
        .layout_kind = layout->kind,
        .source = source_access,
        .payload = source->payload,
        .view_rank = access.view_rank,
        .register_count = role_layout->register_count,
        .payload_register_count = payload_register_count,
        .elements_per_register = role_layout->elements_per_register,
        .element_byte_count = (uint16_t)source_access.element_byte_count,
        .view_element_type = loom_type_element_type(view_type),
        .payload_form = payload_form,
        .narrowed_result_round_source = narrowed_result_sources.round_source,
        .narrowed_result_scale_source = narrowed_result_sources.scale_source,
        .narrowed_result_packed_source = narrowed_result_sources.packed_source,
    };
    for (uint8_t axis = 0; axis < access.view_rank; ++axis) {
      out_plan->axis_byte_strides[axis] = axis_byte_strides[axis];
    }
  }
  return true;
}

static loom_amdgpu_fragment_memory_address_register_kind_t
loom_amdgpu_fragment_memory_low_register_kind(loom_low_lower_context_t* context,
                                              loom_value_id_t low_value) {
  if (low_value == LOOM_VALUE_ID_INVALID) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE;
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  const bool is_sgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR);
  if (is_sgpr && loom_low_register_type_unit_count(low_type) == 1) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR;
  }

  const bool is_vgpr = loom_amdgpu_low_type_is_register_class(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR);
  if (is_vgpr && loom_low_register_type_unit_count(low_type) == 1) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR;
  }

  IREE_ASSERT_UNREACHABLE(
      "AMDGPU fragment memory address selected a non-scalar register");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_add_address_term(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_term,
    loom_amdgpu_fragment_memory_address_register_kind_t term_register_kind,
    loom_type_t sgpr_type, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_accumulator_t* inout_accumulator) {
  if (term_register_kind == LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE) {
    return iree_ok_status();
  }
  if (inout_accumulator->register_kind ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE) {
    *inout_accumulator = (loom_amdgpu_fragment_memory_address_accumulator_t){
        .value = low_term,
        .register_kind = term_register_kind,
    };
    return iree_ok_status();
  }
  if (inout_accumulator->register_kind ==
          LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR &&
      term_register_kind == LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
        inout_accumulator->value, low_term, sgpr_type,
        &inout_accumulator->value));
    return iree_ok_status();
  }

  loom_value_id_t low_lhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_rhs = LOOM_VALUE_ID_INVALID;
  if (inout_accumulator->register_kind ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR) {
    if (term_register_kind ==
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
      low_lhs = low_term;
      low_rhs = inout_accumulator->value;
    } else {
      low_lhs = inout_accumulator->value;
      low_rhs = low_term;
    }
  } else {
    low_lhs = inout_accumulator->value;
    low_rhs = low_term;
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32, low_lhs,
      low_rhs, vgpr_type, &inout_accumulator->value));
  inout_accumulator->register_kind =
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR;
  return iree_ok_status();
}

static int loom_amdgpu_fragment_memory_cache_state_key;

static iree_status_t loom_amdgpu_get_fragment_memory_cache(
    loom_low_lower_context_t* context,
    loom_amdgpu_fragment_memory_cache_t** out_cache) {
  return loom_low_lower_get_or_allocate_target_state(
      context, &loom_amdgpu_fragment_memory_cache_state_key,
      sizeof(**out_cache), (void**)out_cache);
}

static iree_status_t loom_amdgpu_get_fragment_memory_contract_candidates(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fragment_memory_contract_candidate_list_t**
        out_candidates) {
  *out_candidates = NULL;
  const loom_target_bundle_t* bundle = loom_low_lower_context_bundle(context);
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  if (bundle == NULL || bundle->snapshot == NULL || descriptor_set == NULL) {
    return iree_ok_status();
  }

  const loom_amdgpu_matrix_feature_bits_t feature_bits =
      loom_amdgpu_fragment_memory_feature_bits_from_target_ref(
          loom_low_lower_context_module(context),
          loom_low_lower_context_target_ref(context));
  const uint32_t wave_size = bundle->snapshot->subgroup_size;
  loom_amdgpu_fragment_memory_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fragment_memory_cache(context, &cache));
  loom_amdgpu_fragment_memory_contract_candidate_list_t* candidates =
      &cache->contract_candidates;
  if (candidates->descriptor_set == descriptor_set &&
      candidates->feature_bits == feature_bits &&
      candidates->wave_size == wave_size) {
    *out_candidates = candidates;
    return iree_ok_status();
  }

  const iree_host_size_t descriptor_count =
      loom_amdgpu_matrix_contract_descriptor_count();
  const loom_amdgpu_matrix_contract_descriptor_t** descriptors = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_scratch_array(
      context, descriptor_count, sizeof(*descriptors), (void**)&descriptors));
  iree_host_size_t candidate_count = 0;
  for (iree_host_size_t i = 0; i < descriptor_count; ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        loom_amdgpu_matrix_contract_descriptor_at(i);
    if (!loom_amdgpu_fragment_memory_contract_is_available(
            descriptor, descriptor_set, feature_bits, wave_size)) {
      continue;
    }
    descriptors[candidate_count++] = descriptor;
  }
  *candidates = (loom_amdgpu_fragment_memory_contract_candidate_list_t){
      .descriptor_set = descriptor_set,
      .feature_bits = feature_bits,
      .wave_size = wave_size,
      .descriptors = descriptors,
      .descriptor_count = candidate_count,
  };
  *out_candidates = candidates;
  return iree_ok_status();
}

static bool loom_amdgpu_fragment_memory_lane_id_cache_matches(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fragment_memory_cache_t* cache, uint16_t lane_divisor,
    loom_type_t vgpr_type) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  return builder->ip.block != NULL && builder->ip.before_op == NULL &&
         cache->lane_id_cache.block == builder->ip.block &&
         cache->lane_id_cache.lane_ids.lane != LOOM_VALUE_ID_INVALID &&
         loom_type_equal(cache->lane_id_cache.vgpr_type, vgpr_type) &&
         cache->lane_id_cache.lane_divisor == lane_divisor;
}

static iree_status_t loom_amdgpu_update_fragment_memory_lane_id_cache(
    loom_low_lower_context_t* context, uint16_t lane_divisor,
    loom_type_t vgpr_type, const loom_amdgpu_fragment_lane_ids_t* lane_ids) {
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  if (builder->ip.block == NULL || builder->ip.before_op != NULL) {
    return iree_ok_status();
  }
  loom_amdgpu_fragment_memory_cache_t* cache = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fragment_memory_cache(context, &cache));
  if (loom_amdgpu_fragment_memory_lane_id_cache_matches(
          context, cache, lane_divisor, vgpr_type)) {
    cache->lane_id_cache.lane_ids = *lane_ids;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_ensure_fragment_memory_lane_mod(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t lane_divisor, loom_type_t vgpr_type,
    loom_amdgpu_fragment_lane_ids_t* inout_lane_ids) {
  if (inout_lane_ids->lane_mod != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (lane_divisor == 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0, vgpr_type,
        &inout_lane_ids->lane_mod));
  } else {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        inout_lane_ids->lane, lane_divisor - 1u, vgpr_type,
        &inout_lane_ids->lane_mod));
  }
  return loom_amdgpu_update_fragment_memory_lane_id_cache(
      context, lane_divisor, vgpr_type, inout_lane_ids);
}

static iree_status_t loom_amdgpu_ensure_fragment_memory_lane_div(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t lane_divisor, loom_type_t vgpr_type,
    loom_amdgpu_fragment_lane_ids_t* inout_lane_ids) {
  if (inout_lane_ids->lane_div != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
      loom_amdgpu_fragment_repack_log2_u16(lane_divisor), inout_lane_ids->lane,
      vgpr_type, &inout_lane_ids->lane_div));
  return loom_amdgpu_update_fragment_memory_lane_id_cache(
      context, lane_divisor, vgpr_type, inout_lane_ids);
}

static bool loom_amdgpu_fragment_memory_address_base_key_is_equal(
    const loom_amdgpu_fragment_memory_address_base_key_t* lhs,
    const loom_amdgpu_fragment_memory_address_base_key_t* rhs) {
  if (lhs->block != rhs->block ||
      !loom_type_equal(lhs->vgpr_type, rhs->vgpr_type) ||
      lhs->lane_mod_stride != rhs->lane_mod_stride ||
      lhs->lane_div_stride != rhs->lane_div_stride ||
      lhs->dynamic_term_count != rhs->dynamic_term_count) {
    return false;
  }
  for (uint8_t i = 0; i < lhs->dynamic_term_count; ++i) {
    if (lhs->dynamic_values[i] != rhs->dynamic_values[i] ||
        lhs->dynamic_byte_strides[i] != rhs->dynamic_byte_strides[i]) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_fragment_memory_uses_dynamic_view_base_value(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint8_t term_index) {
  // The source plan keeps normalized dynamic terms for legality, but the
  // original byte-domain view-base value may already be materialized and shared
  // by nearby fragment ops. When using that value, emission subtracts the
  // extracted static view-base delta from the immediate side of the address.
  return term_index == 0 && plan->source.dynamic_view_base_term_count == 1 &&
         plan->source.dynamic_view_base_value_id != LOOM_VALUE_ID_INVALID;
}

static bool loom_amdgpu_fragment_memory_address_base_key_for_plan(
    loom_low_lower_context_t* context,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_type_t vgpr_type,
    uint32_t lane_mod_stride, uint32_t lane_div_stride,
    loom_amdgpu_fragment_memory_address_base_key_t* out_key) {
  memset(out_key, 0, sizeof(*out_key));
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  if (builder->ip.block == NULL || builder->ip.before_op != NULL) {
    return false;
  }
  out_key->block = builder->ip.block;
  out_key->vgpr_type = vgpr_type;
  out_key->lane_mod_stride = lane_mod_stride;
  out_key->lane_div_stride = lane_div_stride;
  out_key->dynamic_term_count = plan->source.dynamic_term_count;
  for (uint8_t i = 0; i < plan->source.dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &plan->source.dynamic_terms[i];
    if (term->stride_value_count != 0) {
      return false;
    }
    if (loom_amdgpu_fragment_memory_uses_dynamic_view_base_value(plan, i)) {
      out_key->dynamic_values[i] = plan->source.dynamic_view_base_value_id;
      out_key->dynamic_byte_strides[i] = 1;
    } else {
      out_key->dynamic_values[i] = term->index;
      out_key->dynamic_byte_strides[i] = term->byte_stride;
    }
  }
  return true;
}

static iree_status_t loom_amdgpu_emit_fragment_memory_dynamic_source_terms(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_type_t sgpr_type,
    loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_accumulator_t* inout_accumulator) {
  for (uint8_t i = 0; i < plan->source.dynamic_term_count; ++i) {
    const loom_low_source_memory_dynamic_term_t* term =
        &plan->source.dynamic_terms[i];
    IREE_ASSERT_EQ(term->stride_value_count, 0u);
    IREE_ASSERT_GE(term->byte_stride, 0);
    IREE_ASSERT_LE(term->byte_stride, UINT32_MAX);

    const bool use_view_base_value =
        loom_amdgpu_fragment_memory_uses_dynamic_view_base_value(plan, i);
    const loom_value_id_t source_value =
        use_view_base_value ? plan->source.dynamic_view_base_value_id
                            : term->index;
    const uint32_t byte_stride =
        use_view_base_value ? 1u : (uint32_t)term->byte_stride;
    loom_value_id_t low_index = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_low_lower_lookup_value(context, source_value, &low_index));
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    loom_amdgpu_fragment_memory_address_register_kind_t register_kind =
        loom_amdgpu_fragment_memory_low_register_kind(context, low_index);
    if (register_kind == LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_scale_u32(
          context, source_op, low_index, byte_stride, sgpr_type, &low_term));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
          context, source_op, low_index, byte_stride,
          LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_NONE, vgpr_type, &low_term));
      register_kind = LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_address_term(
        context, source_op, low_term, register_kind, sgpr_type, vgpr_type,
        inout_accumulator));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_lane_terms(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_lane_ids_t* lane_ids, uint32_t lane_mod_stride,
    uint32_t lane_div_stride, uint16_t lane_divisor, loom_type_t sgpr_type,
    loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_accumulator_t* inout_accumulator) {
  if (lane_mod_stride != 0 && lane_div_stride != 0 &&
      (uint64_t)lane_mod_stride * lane_divisor == lane_div_stride) {
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
        context, source_op, lane_ids->lane, lane_mod_stride,
        LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_VALUE_UNSIGNED_24, vgpr_type,
        &low_term));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_address_term(
        context, source_op, low_term,
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR, sgpr_type, vgpr_type,
        inout_accumulator));
    lane_mod_stride = 0;
    lane_div_stride = 0;
  }
  if (lane_div_stride != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fragment_memory_lane_div(
        context, source_op, lane_divisor, vgpr_type, lane_ids));
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
        context, source_op, lane_ids->lane_div, lane_div_stride,
        LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_VALUE_UNSIGNED_24, vgpr_type,
        &low_term));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_address_term(
        context, source_op, low_term,
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR, sgpr_type, vgpr_type,
        inout_accumulator));
  }
  if (lane_mod_stride != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fragment_memory_lane_mod(
        context, source_op, lane_divisor, vgpr_type, lane_ids));
    loom_value_id_t low_term = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_scale_u32(
        context, source_op, lane_ids->lane_mod, lane_mod_stride,
        LOOM_AMDGPU_VGPR_SCALE_U32_FLAG_VALUE_UNSIGNED_24, vgpr_type,
        &low_term));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_add_address_term(
        context, source_op, low_term,
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR, sgpr_type, vgpr_type,
        inout_accumulator));
  }
  return iree_ok_status();
}

static bool loom_amdgpu_fragment_memory_scale_stride_u32(
    uint32_t factor, uint32_t byte_stride, uint32_t* out_byte_stride) {
  const uint64_t scaled_byte_stride = (uint64_t)factor * byte_stride;
  if (scaled_byte_stride > UINT32_MAX) {
    return false;
  }
  *out_byte_stride = (uint32_t)scaled_byte_stride;
  return true;
}

static bool loom_amdgpu_fragment_memory_register_terms(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint32_t* out_lane_mod_stride, uint32_t* out_lane_div_stride,
    uint64_t* out_static_byte_offset) {
  const loom_amdgpu_matrix_tile_shape_t shape = layout->tile_shape;
  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, plan->role);
  *out_lane_mod_stride = 0;
  *out_lane_div_stride = 0;
  *out_static_byte_offset = 0;
  if (role_layout == NULL) {
    return false;
  }
  switch (role_layout->map_kind) {
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION:
      *out_lane_mod_stride = plan->axis_byte_strides[0];
      *out_static_byte_offset = (uint64_t)register_index *
                                plan->elements_per_register *
                                plan->axis_byte_strides[1];
      return true;
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION:
      *out_lane_mod_stride = plan->axis_byte_strides[1];
      *out_static_byte_offset = (uint64_t)register_index *
                                plan->elements_per_register *
                                plan->axis_byte_strides[0];
      return true;
    case LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN:
    case LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN_LOW_SUBWORD:
      *out_lane_mod_stride = plan->axis_byte_strides[1];
      *out_lane_div_stride = plan->axis_byte_strides[0];
      *out_static_byte_offset =
          (uint64_t)register_index *
          (shape.result_row_count / plan->register_count) *
          plan->axis_byte_strides[0];
      return true;
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION:
      *out_lane_mod_stride = plan->axis_byte_strides[0];
      if (!loom_amdgpu_fragment_memory_scale_stride_u32(
              (uint32_t)plan->register_count * plan->elements_per_register,
              plan->axis_byte_strides[1], out_lane_div_stride)) {
        return false;
      }
      *out_static_byte_offset = (uint64_t)register_index *
                                plan->elements_per_register *
                                plan->axis_byte_strides[1];
      return true;
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION:
      *out_lane_mod_stride = plan->axis_byte_strides[1];
      if (!loom_amdgpu_fragment_memory_scale_stride_u32(
              (uint32_t)plan->register_count * plan->elements_per_register,
              plan->axis_byte_strides[0], out_lane_div_stride)) {
        return false;
      }
      *out_static_byte_offset = (uint64_t)register_index *
                                plan->elements_per_register *
                                plan->axis_byte_strides[0];
      return true;
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN:
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN_LOW_SUBWORD:
      *out_lane_mod_stride = plan->axis_byte_strides[1];
      if (!loom_amdgpu_fragment_memory_scale_stride_u32(
              plan->register_count, plan->axis_byte_strides[0],
              out_lane_div_stride)) {
        return false;
      }
      *out_static_byte_offset =
          (uint64_t)register_index * plan->axis_byte_strides[0];
      return true;
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_PACKED_ROW_COLUMN:
      *out_lane_mod_stride = plan->axis_byte_strides[1];
      if (!loom_amdgpu_fragment_memory_scale_stride_u32(
              (uint32_t)plan->register_count * plan->elements_per_register,
              plan->axis_byte_strides[0], out_lane_div_stride)) {
        return false;
      }
      *out_static_byte_offset = (uint64_t)register_index *
                                plan->elements_per_register *
                                plan->axis_byte_strides[0];
      return true;
    case LOOM_MATRIX_FRAGMENT_MAP_UNKNOWN:
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_lane_divisor(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t* out_divisor) {
  *out_divisor = 0;
  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, plan->role);
  if (role_layout == NULL) {
    return false;
  }
  switch (role_layout->map_kind) {
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION:
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION:
      *out_divisor = layout->tile_shape.result_row_count;
      return true;
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION:
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN:
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION:
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN:
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN_LOW_SUBWORD:
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN_LOW_SUBWORD:
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_GROUP_PACKED_ROW_COLUMN:
      *out_divisor = layout->tile_shape.result_column_count;
      return true;
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_UNKNOWN:
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_descriptor_offset_info(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_amdgpu_descriptor_offset_immediate_info_t* out_info) {
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const uint32_t descriptor_ordinal =
      loom_amdgpu_descriptor_ref_ordinal(descriptor_set, descriptor_ref);
  if (descriptor_ordinal == LOOM_LOW_DESCRIPTOR_ORDINAL_NONE) {
    return false;
  }
  if (loom_amdgpu_descriptor_offset_immediate_info(
          descriptor_set, descriptor_ordinal, 1, LOOM_LOW_IMMEDIATE_KIND_SIGNED,
          out_info)) {
    return true;
  }
  return loom_amdgpu_descriptor_offset_immediate_info(
      descriptor_set, descriptor_ordinal, 1, LOOM_LOW_IMMEDIATE_KIND_UNSIGNED,
      out_info);
}

static bool loom_amdgpu_fragment_memory_register_group_is_contiguous(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t register_count, uint32_t register_byte_count) {
  if (register_count == 0 ||
      register_index + register_count > plan->register_count) {
    return false;
  }

  uint32_t base_lane_mod_stride = 0;
  uint32_t base_lane_div_stride = 0;
  uint64_t base_static_byte_offset = 0;
  if (!loom_amdgpu_fragment_memory_register_terms(
          layout, plan, register_index, &base_lane_mod_stride,
          &base_lane_div_stride, &base_static_byte_offset)) {
    return false;
  }
  for (uint16_t i = 1; i < register_count; ++i) {
    uint32_t lane_mod_stride = 0;
    uint32_t lane_div_stride = 0;
    uint64_t static_byte_offset = 0;
    if (!loom_amdgpu_fragment_memory_register_terms(
            layout, plan, register_index + i, &lane_mod_stride,
            &lane_div_stride, &static_byte_offset)) {
      return false;
    }
    if (lane_mod_stride != base_lane_mod_stride ||
        lane_div_stride != base_lane_div_stride ||
        static_byte_offset !=
            base_static_byte_offset + (uint64_t)i * register_byte_count) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_fragment_memory_register_static_offset(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t element_index, uint64_t* out_static_byte_offset) {
  *out_static_byte_offset = 0;
  uint32_t unused_lane_mod_stride = 0;
  uint32_t unused_lane_div_stride = 0;
  if (!loom_amdgpu_fragment_memory_register_terms(
          layout, plan, register_index, &unused_lane_mod_stride,
          &unused_lane_div_stride, out_static_byte_offset)) {
    return false;
  }
  if (element_index == 0) {
    return true;
  }

  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, plan->role);
  uint8_t element_axis = UINT8_MAX;
  if (!loom_amdgpu_fragment_memory_role_packed_element_axis(role_layout,
                                                            &element_axis) ||
      element_axis >= plan->view_rank ||
      element_index >= role_layout->elements_per_register) {
    return false;
  }
  if (plan->axis_byte_strides[element_axis] != 0 &&
      element_index > UINT64_MAX / plan->axis_byte_strides[element_axis]) {
    return false;
  }
  const uint64_t element_static_offset =
      (uint64_t)element_index * plan->axis_byte_strides[element_axis];
  if (*out_static_byte_offset > UINT64_MAX - element_static_offset) {
    return false;
  }
  *out_static_byte_offset += element_static_offset;
  return true;
}

static bool loom_amdgpu_fragment_memory_static_offset_i64(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t element_index, int64_t* out_static_byte_offset) {
  *out_static_byte_offset = plan->source.static_byte_offset;
  if (plan->source.static_byte_offset < 0) {
    return false;
  }
  uint64_t register_static_offset = 0;
  if (!loom_amdgpu_fragment_memory_register_static_offset(
          layout, plan, register_index, element_index,
          &register_static_offset) ||
      register_static_offset > INT64_MAX) {
    return false;
  }
  return iree_checked_add_i64(plan->source.static_byte_offset,
                              (int64_t)register_static_offset,
                              out_static_byte_offset);
}

static bool loom_amdgpu_fragment_memory_vaddr_static_offset_u32(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t element_index, uint64_t* out_static_byte_offset) {
  *out_static_byte_offset = 0;
  if (plan->source.static_byte_offset < 0) {
    return false;
  }
  int64_t static_byte_offset = plan->source.static_byte_offset;
  if (loom_amdgpu_fragment_memory_uses_dynamic_view_base_value(
          plan, /*term_index=*/0) &&
      !loom_checked_sub_i64(static_byte_offset,
                            plan->source.static_view_base_byte_offset,
                            &static_byte_offset)) {
    return false;
  }

  uint64_t register_static_offset = 0;
  if (!loom_amdgpu_fragment_memory_register_static_offset(
          layout, plan, register_index, element_index,
          &register_static_offset) ||
      register_static_offset > INT64_MAX ||
      !iree_checked_add_i64(static_byte_offset, (int64_t)register_static_offset,
                            &static_byte_offset) ||
      static_byte_offset < 0) {
    return false;
  }
  *out_static_byte_offset = (uint64_t)static_byte_offset;
  return *out_static_byte_offset <= UINT32_MAX;
}

static bool loom_amdgpu_fragment_memory_packet_addresses_fit_u32(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, plan->role);
  const uint16_t register_address_count =
      plan->payload_form ==
              LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT
          ? packet->result_register_count
          : 1;
  const uint16_t element_address_count =
      loom_amdgpu_fragment_memory_role_uses_packed_b16_elements(role_layout)
          ? LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT
          : 1;
  for (uint16_t register_offset = 0; register_offset < register_address_count;
       ++register_offset) {
    for (uint16_t element_index = 0; element_index < element_address_count;
         ++element_index) {
      uint64_t unused_static_byte_offset = 0;
      if (!loom_amdgpu_fragment_memory_vaddr_static_offset_u32(
              layout, plan, packet->register_index + register_offset,
              element_index, &unused_static_byte_offset)) {
        return loom_amdgpu_fragment_memory_reject(
            diagnostic, IREE_SV("fragment_memory.base_offset"));
      }
    }
  }
  return true;
}

static loom_amdgpu_descriptor_ref_t
loom_amdgpu_fragment_memory_crosslane_packed_b16_store_dpp_ref(
    const loom_low_descriptor_set_t* descriptor_set) {
  return loom_amdgpu_dpp_b32_descriptor_ref(descriptor_set);
}

static loom_amdgpu_fragment_memory_packet_flags_t
loom_amdgpu_fragment_memory_crosslane_packed_b16_store_flags(
    const loom_low_descriptor_set_t* descriptor_set) {
  const bool has_common_refs =
      loom_amdgpu_can_emit_compare_u32_immediate(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32, 0) &&
      loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64) &&
      loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC) &&
      loom_amdgpu_fragment_memory_has_vgpr_immediate_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_SRC0_INLINE);
  if (!has_common_refs) {
    return 0;
  }
  if (loom_amdgpu_fragment_memory_crosslane_packed_b16_store_dpp_ref(
          descriptor_set) != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE |
           LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE_DPP;
  }
  if (!loom_amdgpu_descriptor_set_has_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32) ||
      !loom_amdgpu_fragment_memory_has_vgpr_immediate_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_LIT,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_SRC0_INLINE) ||
      !loom_amdgpu_fragment_memory_has_vgpr_immediate_ref(
          descriptor_set, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_SRC0_INLINE)) {
    return 0;
  }
  return LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE;
}

static bool loom_amdgpu_fragment_memory_crosslane_packed_b16_store_layout(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  if (plan->operation_kind != LOOM_AMDGPU_MEMORY_OPERATION_STORE ||
      plan->payload_form !=
          LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16 ||
      !loom_amdgpu_fragment_memory_role_is_result_like(plan->role) ||
      plan->view_rank != LOOM_AMDGPU_FRAGMENT_VIEW_RANK ||
      plan->element_byte_count != 2 ||
      plan->axis_byte_strides[1] != plan->element_byte_count) {
    return false;
  }

  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, plan->role);
  if (role_layout == NULL || role_layout->elements_per_register != 1 ||
      role_layout->element_bit_count != 32 ||
      role_layout->coordinate_flags !=
          (LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_ROW |
           LOOM_AMDGPU_MATRIX_FRAGMENT_COORDINATE_COLUMN)) {
    return false;
  }

  return loom_matrix_fragment_role_has_contiguous_lane_xor1_columns(layout,
                                                                    plan->role);
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
    if (candidate > remaining) {
      continue;
    }
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (!loom_amdgpu_fragment_memory_descriptor_ref(
            plan->operation_kind, plan->source.memory_space, candidate,
            &descriptor_ref) ||
        !loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref) ||
        !loom_amdgpu_fragment_memory_register_group_is_contiguous(
            layout, plan, register_index, candidate,
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
    loom_amdgpu_fragment_memory_packet_plan_t* out_packet) {
  *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){0};
  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (!loom_amdgpu_fragment_memory_16bit_descriptor_ref(
          plan->operation_kind, plan->source.memory_space, &descriptor_ref) ||
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
          LOOM_AMDGPU_MEMORY_OPERATION_LOAD, plan->source.memory_space,
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
      (uint32_t)plan->elements_per_register * plan->element_byte_count;
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
            LOOM_AMDGPU_MEMORY_OPERATION_LOAD, plan->source.memory_space,
            packet_register_count, &descriptor_ref) ||
        !loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref) ||
        !loom_amdgpu_fragment_memory_register_group_is_contiguous(
            layout, plan, register_index, result_register_count,
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
          LOOM_AMDGPU_MEMORY_OPERATION_LOAD, plan->source.memory_space,
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

static bool loom_amdgpu_fragment_memory_can_use_packed_payload_slice(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t result_register_count) {
  return plan->narrowed_result_round_source != LOOM_VALUE_ID_INVALID ||
         result_register_count == 1 ||
         (((register_index | result_register_count) & 1u) == 0);
}

static bool loom_amdgpu_fragment_memory_select_narrowed_store_packet(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    loom_amdgpu_fragment_memory_packet_flags_t crosslane_packed_b16_store_flags,
    loom_amdgpu_fragment_memory_packet_plan_t* out_packet) {
  *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){0};
  const uint16_t remaining = plan->register_count - register_index;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuFragmentMemoryNarrowedStoreCandidates);
       ++i) {
    const uint16_t candidate =
        kLoomAmdgpuFragmentMemoryNarrowedStoreCandidates[i];
    if (candidate == 1 || candidate > remaining ||
        !loom_amdgpu_fragment_memory_can_use_packed_payload_slice(
            plan, register_index, candidate)) {
      continue;
    }
    uint16_t packet_register_count = 0;
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (!loom_amdgpu_fragment_memory_narrowed_store_descriptor_ref(
            plan->source.memory_space, candidate, &packet_register_count,
            &descriptor_ref) ||
        !loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref) ||
        !loom_amdgpu_fragment_memory_register_group_is_contiguous(
            layout, plan, register_index, candidate,
            plan->element_byte_count)) {
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
  if (remaining != 0 && crosslane_packed_b16_store_flags != 0) {
    uint16_t packet_register_count = 0;
    loom_amdgpu_descriptor_ref_t descriptor_ref =
        LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
    if (loom_amdgpu_fragment_memory_narrowed_store_descriptor_ref(
            plan->source.memory_space,
            LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT,
            &packet_register_count, &descriptor_ref) &&
        packet_register_count == 1 &&
        loom_amdgpu_descriptor_set_has_ref(descriptor_set, descriptor_ref)) {
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
  uint16_t scalar_packet_register_count = 0;
  loom_amdgpu_descriptor_ref_t scalar_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (remaining != 0 &&
      loom_amdgpu_fragment_memory_narrowed_store_descriptor_ref(
          plan->source.memory_space, 1, &scalar_packet_register_count,
          &scalar_descriptor_ref) &&
      loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                         scalar_descriptor_ref)) {
    *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){
        .register_index = register_index,
        .result_register_count = 1,
        .packet_register_count = scalar_packet_register_count,
        .descriptor_ref = scalar_descriptor_ref,
    };
    return true;
  }
  return false;
}

static bool loom_amdgpu_fragment_memory_plan_push_packet(
    loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet) {
  if (packet->result_register_count == 0 ||
      plan->packet_count >= IREE_ARRAYSIZE(plan->packets)) {
    return false;
  }
  plan->packets[plan->packet_count++] = *packet;
  plan->packet_flags |= packet->flags;
  if (plan->payload_form ==
          LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16 &&
      packet->result_register_count > 1) {
    plan->packet_flags |=
        LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_PACKED_B16_STORE;
  }
  return true;
}

static loom_amdgpu_fragment_memory_epilogue_strategy_t
loom_amdgpu_fragment_memory_plan_epilogue_strategy(
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  if (plan->payload_form !=
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16) {
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

static bool
loom_amdgpu_fragment_memory_epilogue_strategy_is_crosslane_packed_b16(
    loom_amdgpu_fragment_memory_epilogue_strategy_t strategy) {
  return strategy ==
             LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DS_PACKED_B16_STORE ||
         strategy ==
             LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DPP_PACKED_B16_STORE;
}

static bool loom_amdgpu_fragment_memory_epilogue_strategy_uses_dpp(
    loom_amdgpu_fragment_memory_epilogue_strategy_t strategy) {
  return strategy ==
         LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_DPP_PACKED_B16_STORE;
}

static bool loom_amdgpu_fragment_memory_plan_packets(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_amdgpu_fragment_memory_plan_t* plan,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
  plan->packet_count = 0;
  plan->packet_flags = 0;
  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, plan->role);
  const bool scalar_b16_packets =
      loom_amdgpu_fragment_memory_role_uses_scalar_b16_packets(role_layout);
  const bool load_packed_16bit_result =
      plan->payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT;
  const bool load_fp8_to_16bit =
      loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
          plan->payload_form);
  const bool store_narrow_f32_to_bf16 =
      plan->payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16;
  const bool crosslane_packed_b16_store_layout =
      store_narrow_f32_to_bf16 &&
      loom_amdgpu_fragment_memory_crosslane_packed_b16_store_layout(layout,
                                                                    plan);
  const loom_amdgpu_fragment_memory_packet_flags_t
      crosslane_packed_b16_store_flags =
          crosslane_packed_b16_store_layout
              ? loom_amdgpu_fragment_memory_crosslane_packed_b16_store_flags(
                    descriptor_set)
              : 0;
  for (uint16_t register_index = 0; register_index < plan->register_count;) {
    loom_amdgpu_fragment_memory_packet_plan_t packet = {0};
    const bool selected =
        load_fp8_to_16bit
            ? loom_amdgpu_fragment_memory_select_fp8_to_16bit_load_packet(
                  descriptor_set, layout, plan, register_index, &packet)
        : scalar_b16_packets
            ? loom_amdgpu_fragment_memory_select_low_subword_packet(
                  descriptor_set, plan, register_index, &packet)
        : load_packed_16bit_result
            ? loom_amdgpu_fragment_memory_select_packed_16bit_result_load_packet(
                  descriptor_set, plan, register_index, &packet)
        : store_narrow_f32_to_bf16
            ? loom_amdgpu_fragment_memory_select_narrowed_store_packet(
                  descriptor_set, layout, plan, register_index,
                  crosslane_packed_b16_store_flags, &packet)
            : loom_amdgpu_fragment_memory_select_packet(
                  descriptor_set, layout, plan, register_index, &packet);
    if (!selected ||
        !loom_amdgpu_fragment_memory_packet_addresses_fit_u32(
            layout, plan, &packet, diagnostic) ||
        !loom_amdgpu_fragment_memory_plan_push_packet(plan, &packet)) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.packet"));
    }
    register_index += packet.result_register_count;
  }
  if (plan->packet_count == 0) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.packet"));
  }
  plan->epilogue_strategy =
      loom_amdgpu_fragment_memory_plan_epilogue_strategy(plan);
  return true;
}

static void loom_amdgpu_fragment_memory_apply_fp8_load_strategy_flags(
    const loom_value_fact_table_t* fact_table,
    const loom_low_descriptor_set_t* descriptor_set, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* plan) {
  if (!loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
          plan->payload_form)) {
    return;
  }

  const loom_scalar_type_t result_element_type =
      loom_amdgpu_fragment_memory_load_fp8_result_element_type(
          plan->payload_form);
  const bool result_is_f16 = result_element_type == LOOM_SCALAR_TYPE_F16;
  loom_amdgpu_fp8_decode_plan_t decode_plan = {0};
  loom_amdgpu_initialize_fp8_decode_plan_from_descriptor_set(
      descriptor_set, plan->view_element_type, &decode_plan);
  loom_amdgpu_descriptor_ref_t scalef32_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  loom_amdgpu_fp8_16bit_capabilities_t capabilities =
      LOOM_AMDGPU_FP8_16BIT_CAPABILITY_NONE;
  const bool has_identity_scalef32_descriptor =
      descriptor_set != NULL &&
      loom_amdgpu_fp8_scalef32_descriptor_ref(plan->view_element_type,
                                              result_element_type,
                                              &scalef32_descriptor_ref) &&
      loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                         scalef32_descriptor_ref);
  if (has_identity_scalef32_descriptor) {
    capabilities |=
        result_is_f16
            ? LOOM_AMDGPU_FP8_16BIT_CAPABILITY_IDENTITY_SCALEF32_F16_PAIR
            : LOOM_AMDGPU_FP8_16BIT_CAPABILITY_IDENTITY_SCALEF32_BF16_PAIR;
  }
  loom_amdgpu_descriptor_ref_t e8m0_pk8_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  const bool has_identity_e8m0_pk8_descriptor =
      descriptor_set != NULL &&
      loom_amdgpu_fragment_memory_packets_support_fp8_e8m0_pk8(plan) &&
      loom_amdgpu_fp8_e8m0_pk8_descriptor_ref(plan->view_element_type,
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
        loom_amdgpu_fp8_native_descriptor_refs(
            plan->view_element_type, LOOM_SCALAR_TYPE_F16, &native_refs) &&
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

  const loom_amdgpu_fragment_memory_packet_flags_t packet_flags =
      result_is_f16
          ? loom_amdgpu_fragment_memory_fp8_to_f16_decode_packet_flags(
                &decode_plan, capabilities, decode_value_flags)
          : loom_amdgpu_fragment_memory_fp8_decode_packet_flags(
                &decode_plan, capabilities, decode_value_flags);
  if (packet_flags == 0) {
    return;
  }
  for (uint16_t i = 0; i < plan->packet_count; ++i) {
    plan->packets[i].flags |= packet_flags;
  }
  plan->packet_flags |= packet_flags;
}

static bool loom_amdgpu_analyze_vector_fragment_memory_plan_impl(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_fragment_memory_contract_candidate_list_t*
        contract_candidates,
    const loom_amdgpu_source_alloca_layout_t* alloca_layout,
    loom_symbol_ref_t target_ref, loom_func_like_t source_function,
    const loom_op_t* source_op,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_amdgpu_fragment_memory_plan_t* out_plan) {
  *out_plan = (loom_amdgpu_fragment_memory_plan_t){0};
  const loom_amdgpu_fragment_memory_environment_t environment = {
      .module = module,
      .fact_table = fact_table,
      .view_regions = view_regions,
      .bundle = bundle,
      .descriptor_set = descriptor_set,
      .contract_candidates = contract_candidates,
      .alloca_layout = alloca_layout,
      .feature_bits =
          contract_candidates != NULL
              ? contract_candidates->feature_bits
              : loom_amdgpu_fragment_memory_feature_bits_from_target_ref(
                    module, target_ref),
      .source_function = source_function,
  };
  loom_amdgpu_fragment_memory_source_t source = {0};
  loom_amdgpu_fragment_memory_source_from_op(source_op, operation_kind,
                                             &source);
  if (!loom_amdgpu_fragment_memory_analyze(&environment, &source,
                                           operation_kind, out_plan,
                                           /*diagnostic=*/NULL)) {
    return false;
  }
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(out_plan->layout_kind);
  if (layout == NULL ||
      !loom_amdgpu_fragment_memory_plan_packets(
          environment.descriptor_set, layout, out_plan, /*diagnostic=*/NULL)) {
    return false;
  }
  loom_amdgpu_fragment_memory_apply_fp8_load_strategy_flags(
      fact_table, descriptor_set, source_op, out_plan);
  return true;
}

iree_status_t loom_amdgpu_analyze_vector_fragment_memory_plan(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t target_ref, loom_func_like_t source_function,
    const loom_op_t* source_op,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected) {
  *out_selected = loom_amdgpu_analyze_vector_fragment_memory_plan_impl(
      module, fact_table, view_regions, bundle, descriptor_set,
      /*contract_candidates=*/NULL, loom_amdgpu_source_alloca_layout_empty(),
      target_ref, source_function, source_op, operation_kind, out_plan);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_fragment_memory_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  const loom_amdgpu_fragment_memory_contract_candidate_list_t*
      contract_candidates = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fragment_memory_contract_candidates(
      context, &contract_candidates));
  const loom_amdgpu_source_alloca_layout_t* alloca_layout = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_for_lower_context(
      context, &alloca_layout));
  *out_selected = loom_amdgpu_analyze_vector_fragment_memory_plan_impl(
      module, loom_low_lower_context_fact_table(context), view_regions,
      loom_low_lower_context_bundle(context),
      loom_low_lower_context_descriptor_set(context), contract_candidates,
      alloca_layout, loom_low_lower_context_target_ref(context),
      loom_low_lower_context_source_function(context), source_op,
      operation_kind, out_plan);
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_fragment_load_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected) {
  return loom_amdgpu_fragment_memory_select(context, source_op,
                                            LOOM_AMDGPU_MEMORY_OPERATION_LOAD,
                                            out_plan, out_selected);
}

iree_status_t loom_amdgpu_select_vector_fragment_store_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected) {
  return loom_amdgpu_fragment_memory_select(context, source_op,
                                            LOOM_AMDGPU_MEMORY_OPERATION_STORE,
                                            out_plan, out_selected);
}

iree_status_t loom_amdgpu_select_vector_fragment_repack_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_repack_plan_t* out_plan, bool* out_selected) {
  *out_selected = false;
  const loom_value_id_t source = loom_vector_fragment_repack_source(source_op);
  const loom_value_id_t result = loom_vector_fragment_repack_result(source_op);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t source_type = loom_module_value_type(module, source);
  const loom_type_t result_type = loom_module_value_type(module, result);
  const loom_vector_fragment_role_flags_t result_role_flags =
      loom_vector_fragment_fact_role_flags(
          loom_vector_fragment_repack_role(source_op));
  *out_plan = (loom_amdgpu_fragment_repack_plan_t){
      .source = source,
      .result = result,
      .strategy = LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_DIAGNOSTIC,
      .reason = LOOM_AMDGPU_FRAGMENT_REPACK_REASON_SOURCE_FACTS,
      .source_role_flags = 0,
      .result_role_flags = result_role_flags,
      .source_type = source_type,
      .result_type = result_type,
  };

  loom_vector_fragment_fact_t source_fact = {0};
  if (fact_table == NULL ||
      !loom_vector_fragment_fact_query_value_facts(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table, source), &source_fact) ||
      source_fact.role_flags == 0 || result_role_flags == 0) {
    *out_selected = true;
    return iree_ok_status();
  }
  out_plan->source_role_flags = source_fact.role_flags;

  if (!loom_amdgpu_fragment_repack_shape_matches(
          fact_table, source_fact, loom_vector_fragment_repack_rows(source_op),
          loom_vector_fragment_repack_columns(source_op))) {
    out_plan->reason = LOOM_AMDGPU_FRAGMENT_REPACK_REASON_SHAPE;
    *out_selected = true;
    return iree_ok_status();
  }

  const loom_amdgpu_fragment_repack_reason_t reason =
      loom_amdgpu_fragment_repack_transition_reason(
          source_fact.role_flags, result_role_flags, source_type, result_type);
  if (reason == LOOM_AMDGPU_FRAGMENT_REPACK_REASON_NONE) {
    out_plan->strategy = LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_ALIAS;
  } else {
    out_plan->reason = reason;
    const loom_amdgpu_fragment_memory_contract_candidate_list_t*
        contract_candidates = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_fragment_memory_contract_candidates(
        context, &contract_candidates));
    (void)loom_amdgpu_fragment_repack_select_target_strategy(
        module, loom_low_lower_context_bundle(context),
        loom_low_lower_context_descriptor_set(context), contract_candidates,
        loom_low_lower_context_target_ref(context), out_plan);
  }
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_diagnostic(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  loom_module_t* module = loom_low_lower_context_module(context);
  const loom_diagnostic_param_t params[] = {
      loom_param_string(loom_low_lower_context_target_key(context)),
      loom_param_string(loom_low_lower_context_export_name(context)),
      loom_param_string(loom_low_lower_context_config_key(context)),
      loom_param_string(loom_low_lower_context_function_name(context)),
      loom_param_string(loom_op_name(module, source_op)),
      loom_param_string(
          loom_amdgpu_fragment_repack_role_flags_key(plan->source_role_flags)),
      loom_param_string(
          loom_amdgpu_fragment_repack_role_flags_key(plan->result_role_flags)),
      loom_param_type(plan->source_type),
      loom_param_type(plan->result_type),
      loom_param_string(loom_amdgpu_fragment_repack_plan_key(plan)),
      loom_param_string(loom_amdgpu_fragment_repack_reason_key(plan->reason)),
  };
  return loom_low_lower_emit_error_ref(context, source_op,
                                       LOOM_ERR_AMDGPU_041_REF, params,
                                       IREE_ARRAYSIZE(params));
}

iree_status_t loom_amdgpu_low_legality_verify_vector_fragment_memory(
    const loom_target_low_legality_provider_t* provider,
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    bool* out_handled) {
  (void)provider;
  const loom_target_bundle_t* bundle = loom_target_low_legality_bundle(context);
  if (!loom_amdgpu_low_legality_bundle_is_amdgpu(bundle)) {
    return iree_ok_status();
  }
  *out_handled = true;

  loom_amdgpu_memory_operation_kind_t operation_kind =
      LOOM_AMDGPU_MEMORY_OPERATION_LOAD;
  if (op->kind == LOOM_OP_VECTOR_FRAGMENT_STORE) {
    operation_kind = LOOM_AMDGPU_MEMORY_OPERATION_STORE;
  } else if (op->kind != LOOM_OP_VECTOR_FRAGMENT_LOAD) {
    *out_handled = false;
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_target_low_legality_view_regions(context, &view_regions));
  const loom_amdgpu_source_alloca_layout_t* alloca_layout = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_for_low_legality(
      context, &alloca_layout));
  const loom_amdgpu_fragment_memory_environment_t environment = {
      .module = module,
      .fact_table = loom_target_low_legality_fact_table(context),
      .view_regions = view_regions,
      .bundle = bundle,
      .descriptor_set = loom_target_low_legality_descriptor_set(context),
      .alloca_layout = alloca_layout,
      .feature_bits = loom_amdgpu_fragment_memory_feature_bits_from_target_ref(
          module, loom_target_low_legality_target_ref(context)),
      .source_function = loom_target_low_legality_function(context),
  };
  loom_amdgpu_fragment_memory_source_t source = {0};
  loom_amdgpu_fragment_memory_source_from_op(op, operation_kind, &source);
  loom_amdgpu_fragment_memory_diagnostic_t diagnostic = {0};
  loom_amdgpu_fragment_memory_plan_t plan = {0};
  if (loom_amdgpu_fragment_memory_analyze(&environment, &source, operation_kind,
                                          &plan, &diagnostic)) {
    const loom_amdgpu_matrix_fragment_layout_t* layout =
        loom_amdgpu_matrix_fragment_layout_for_kind(plan.layout_kind);
    if (layout != NULL &&
        loom_amdgpu_fragment_memory_plan_packets(environment.descriptor_set,
                                                 layout, &plan, &diagnostic)) {
      return iree_ok_status();
    }
  }
  iree_string_view_t constraint_key = diagnostic.constraint_key;
  if (iree_string_view_is_empty(constraint_key)) {
    constraint_key = IREE_SV("fragment_memory");
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}

static iree_status_t loom_amdgpu_fragment_memory_packet_type(
    loom_low_lower_context_t* context, uint16_t packet_register_count,
    loom_type_t vgpr_type, loom_type_t* out_type) {
  if (packet_register_count == 1) {
    *out_type = vgpr_type;
    return iree_ok_status();
  }
  return loom_amdgpu_make_vgpr_range_type(context, packet_register_count,
                                          out_type);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_packed_16bit_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_payload, uint16_t payload_register_count,
    uint16_t lane_index, loom_type_t vgpr_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  const uint16_t source_register_index = lane_index / 2u;
  if (source_register_index >= payload_register_count) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment packed b16 lane");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_value_id_t source_register = low_payload;
  if (payload_register_count != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_payload, source_register_index, vgpr_type,
        &source_register));
  }
  if ((lane_index & 1u) == 0) {
    *out_lane = source_register;
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
      source_register, vgpr_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_f32_to_16bit_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors,
    loom_value_id_t low_source, uint16_t source_register_count,
    uint16_t register_index, loom_value_id_t low_scale, loom_type_t vgpr_type,
    loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  loom_value_id_t source_register = low_source;
  if (source_register_count != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_source, register_index, vgpr_type,
        &source_register));
  }
  if (low_scale != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32,
        source_register, low_scale, vgpr_type, &source_register));
  }
  return loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
      context, source_op, bf16_pack_descriptors, source_register, vgpr_type,
      out_lane);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_f16_to_f32_lane(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t source_register_count,
    uint16_t register_index, loom_type_t vgpr_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  loom_value_id_t source_register = low_source;
  if (source_register_count != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_source, register_index, vgpr_type,
        &source_register));
  }
  return loom_amdgpu_emit_vgpr_unary(context, source_op,
                                     LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_F16,
                                     source_register, vgpr_type, out_lane);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_f32_pair_to_packed_16bit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors,
    loom_value_id_t low_source, uint16_t register_index, loom_type_t vgpr_type,
    loom_value_id_t low_scale, loom_value_id_t* out_packed) {
  *out_packed = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_low_slice(context, source_op, low_source, register_index,
                                 vgpr_type, &low_source_register));
  loom_value_id_t high_source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_source, register_index + 1u, vgpr_type,
      &high_source_register));
  if (low_scale != LOOM_VALUE_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32,
        low_source_register, low_scale, vgpr_type, &low_source_register));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32,
        high_source_register, low_scale, vgpr_type, &high_source_register));
  }
  return loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
      context, source_op, bf16_pack_descriptors, low_source_register,
      high_source_register, vgpr_type, out_packed);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_packed_16bit_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_value_id_t low_payload,
    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors,
    uint16_t register_index, uint16_t result_register_count,
    uint16_t packet_register_count, loom_value_id_t low_scale,
    loom_type_t vgpr_type, loom_value_id_t* out_packet) {
  *out_packet = LOOM_VALUE_ID_INVALID;
  if (result_register_count == 1) {
    if (plan->narrowed_result_round_source != LOOM_VALUE_ID_INVALID) {
      return loom_amdgpu_emit_fragment_memory_f32_to_16bit_lane(
          context, source_op, bf16_pack_descriptors, low_payload,
          plan->register_count, register_index, low_scale, vgpr_type,
          out_packet);
    }
    return loom_amdgpu_emit_fragment_memory_packed_16bit_lane(
        context, source_op, low_payload, plan->payload_register_count,
        register_index, vgpr_type, out_packet);
  }

  if (plan->narrowed_result_round_source == LOOM_VALUE_ID_INVALID) {
    loom_type_t packet_type = vgpr_type;
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_type(
        context, packet_register_count, vgpr_type, &packet_type));
    if (register_index == 0 &&
        packet_register_count == plan->payload_register_count) {
      *out_packet = low_payload;
      return iree_ok_status();
    }
    return loom_amdgpu_emit_low_slice(context, source_op, low_payload,
                                      register_index / 2u, packet_type,
                                      out_packet);
  }

  loom_value_id_t packed_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {
      0};
  for (uint16_t i = 0; i < packet_register_count; ++i) {
    const uint16_t source_register_index = register_index + i * 2u;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_f32_pair_to_packed_16bit(
            context, source_op, bf16_pack_descriptors, low_payload,
            source_register_index, vgpr_type, low_scale, &packed_registers[i]));
  }
  if (packet_register_count == 1) {
    *out_packet = packed_registers[0];
    return iree_ok_status();
  }

  loom_type_t packet_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_type(
      context, packet_register_count, vgpr_type, &packet_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), packed_registers,
      packet_register_count, packet_type, source_op->location, &concat_op));
  *out_packet = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_f16_to_f32_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_value_id_t low_payload,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    loom_type_t vgpr_type, loom_value_id_t* out_packet) {
  *out_packet = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_EQ(packet->result_register_count, packet->packet_register_count);
  loom_value_id_t converted_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] =
      {0};
  for (uint16_t i = 0; i < packet->packet_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_f16_to_f32_lane(
        context, source_op, low_payload, plan->register_count,
        packet->register_index + i, vgpr_type, &converted_registers[i]));
  }
  if (packet->packet_register_count == 1) {
    *out_packet = converted_registers[0];
    return iree_ok_status();
  }

  loom_type_t packet_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_type(
      context, packet->packet_register_count, vgpr_type, &packet_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_concat_build(loom_low_lower_context_builder(context),
                            converted_registers, packet->packet_register_count,
                            packet_type, source_op->location, &concat_op));
  *out_packet = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static void loom_amdgpu_fragment_memory_split_static_offset(
    loom_low_lower_context_t* context,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    loom_value_fact_memory_space_t memory_space, uint64_t static_byte_offset,
    uint64_t* out_vaddr_static_byte_offset, int64_t* out_immediate_offset) {
  *out_vaddr_static_byte_offset = static_byte_offset;
  *out_immediate_offset = 0;
  loom_amdgpu_descriptor_offset_immediate_info_t offset_info = {0};
  if (!loom_amdgpu_fragment_memory_descriptor_offset_info(
          context, descriptor_ref, &offset_info) ||
      offset_info.unit_byte_count == 0) {
    return;
  }

  if (memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    if ((static_byte_offset % offset_info.unit_byte_count) != 0) {
      return;
    }
    const uint64_t encoded_offset =
        static_byte_offset / offset_info.unit_byte_count;
    if (encoded_offset > offset_info.unsigned_max ||
        encoded_offset > INT64_MAX) {
      return;
    }
    *out_vaddr_static_byte_offset = 0;
    *out_immediate_offset = (int64_t)encoded_offset;
    return;
  }

  if (offset_info.unsigned_max == UINT64_MAX) {
    return;
  }
  const uint64_t window_unit_count = offset_info.unsigned_max + 1;
  if (offset_info.unit_byte_count > UINT64_MAX / window_unit_count) {
    return;
  }
  const uint64_t window_byte_count =
      window_unit_count * offset_info.unit_byte_count;
  if (window_byte_count == 0) {
    return;
  }
  const uint64_t window_base =
      (static_byte_offset / window_byte_count) * window_byte_count;
  const uint64_t window_offset = static_byte_offset - window_base;
  const uint64_t encoded_offset = window_offset / offset_info.unit_byte_count;
  if (encoded_offset > offset_info.unsigned_max || encoded_offset > INT64_MAX) {
    return;
  }
  const uint64_t immediate_byte_offset =
      encoded_offset * offset_info.unit_byte_count;
  *out_vaddr_static_byte_offset =
      window_base + (window_offset - immediate_byte_offset);
  *out_immediate_offset = (int64_t)encoded_offset;
}

static iree_status_t loom_amdgpu_emit_fragment_memory_base_address_accumulator(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    loom_amdgpu_fragment_lane_ids_t* lane_ids, uint16_t lane_divisor,
    loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_accumulator_t* out_accumulator) {
  *out_accumulator = (loom_amdgpu_fragment_memory_address_accumulator_t){
      .value = LOOM_VALUE_ID_INVALID,
      .register_kind = LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE,
  };
  uint32_t lane_mod_stride = 0;
  uint32_t lane_div_stride = 0;
  uint64_t unused_static_byte_offset = 0;
  // Fragment maps keep register coordinates as static byte offsets. The lane
  // terms are shared by every packet in this fragment memory operation.
  if (!loom_amdgpu_fragment_memory_register_terms(
          layout, plan, /*register_index=*/0, &lane_mod_stride,
          &lane_div_stride, &unused_static_byte_offset)) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment register map");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_amdgpu_fragment_memory_address_base_key_t key;
  const bool cacheable = loom_amdgpu_fragment_memory_address_base_key_for_plan(
      context, plan, vgpr_type, lane_mod_stride, lane_div_stride, &key);
  loom_amdgpu_fragment_memory_cache_t* cache = NULL;
  if (cacheable) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_get_fragment_memory_cache(context, &cache));
    if (cache->address_base.accumulator.register_kind !=
            LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE &&
        loom_amdgpu_fragment_memory_address_base_key_is_equal(
            &cache->address_base.key, &key)) {
      *out_accumulator = cache->address_base.accumulator;
      return iree_ok_status();
    }
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_dynamic_source_terms(
      context, source_op, plan, sgpr_type, vgpr_type, out_accumulator));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_lane_terms(
      context, source_op, lane_ids, lane_mod_stride, lane_div_stride,
      lane_divisor, sgpr_type, vgpr_type, out_accumulator));
  if (cache != NULL && out_accumulator->register_kind !=
                           LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE) {
    cache->address_base.key = key;
    cache->address_base.accumulator = *out_accumulator;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_vaddr(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan, uint16_t register_index,
    uint16_t element_index, loom_amdgpu_descriptor_ref_t descriptor_ref,
    const loom_amdgpu_fragment_memory_address_accumulator_t* base_accumulator,
    loom_value_id_t low_resource, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_t* out_address) {
  *out_address = (loom_amdgpu_fragment_memory_address_t){
      .low_vaddr = LOOM_VALUE_ID_INVALID,
      .immediate_offset = 0,
  };
  uint64_t static_byte_offset = 0;
  if (!loom_amdgpu_fragment_memory_vaddr_static_offset_u32(
          layout, plan, register_index, element_index, &static_byte_offset)) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment memory address range");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  loom_amdgpu_fragment_memory_address_accumulator_t accumulator =
      *base_accumulator;

  loom_amdgpu_fragment_memory_split_static_offset(
      context, descriptor_ref, plan->source.memory_space, static_byte_offset,
      &static_byte_offset, &out_address->immediate_offset);

  if (static_byte_offset != 0) {
    if (accumulator.register_kind ==
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE) {
      return loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
          (uint32_t)static_byte_offset, vgpr_type, &out_address->low_vaddr);
    }
    if (accumulator.register_kind ==
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR) {
      loom_value_id_t low_static_offset = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
          (uint32_t)static_byte_offset, sgpr_type, &low_static_offset));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_sgpr_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ADD_U32,
          accumulator.value, low_static_offset, sgpr_type, &accumulator.value));
    } else {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32_LIT,
          accumulator.value, (uint32_t)static_byte_offset, vgpr_type,
          &accumulator.value));
    }
  }

  if (accumulator.register_kind ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE) {
    return loom_amdgpu_emit_const_u32(context, source_op,
                                      LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
                                      vgpr_type, &out_address->low_vaddr);
  }
  return loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, accumulator.value, &out_address->low_vaddr);
}

static loom_low_memory_space_t loom_amdgpu_fragment_memory_low_space(
    loom_value_fact_memory_space_t memory_space) {
  switch (memory_space) {
    case LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL:
    case LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT:
    case LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR:
      return LOOM_LOW_MEMORY_SPACE_GLOBAL;
    case LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP:
      return LOOM_LOW_MEMORY_SPACE_WORKGROUP;
    case LOOM_VALUE_FACT_MEMORY_SPACE_UNKNOWN:
    case LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE:
    case LOOM_VALUE_FACT_MEMORY_SPACE_HOST:
    case LOOM_VALUE_FACT_MEMORY_SPACE_GENERIC:
    default:
      return LOOM_LOW_MEMORY_SPACE_GENERIC;
  }
}

static bool loom_amdgpu_fragment_memory_uses_buffer_descriptor(
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  return plan->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR;
}

static iree_string_view_t loom_amdgpu_fragment_memory_report_address_form(
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  if (loom_amdgpu_fragment_memory_uses_buffer_descriptor(plan)) {
    return IREE_SV("buffer_vaddr");
  }
  if (plan->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL ||
      plan->source.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT) {
    return loom_amdgpu_memory_address_form_name(
        LOOM_AMDGPU_MEMORY_ADDRESS_FORM_GLOBAL_SADDR);
  }
  return loom_amdgpu_memory_address_form_name(
      LOOM_AMDGPU_MEMORY_ADDRESS_FORM_DEFAULT);
}

static bool loom_amdgpu_fragment_memory_packet_static_offset(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    uint16_t element_index, int64_t* out_static_byte_offset) {
  return loom_amdgpu_fragment_memory_static_offset_i64(
      layout, plan, packet->register_index, element_index,
      out_static_byte_offset);
}

static uint32_t loom_amdgpu_fragment_memory_packet_dynamic_stride_bytes(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet) {
  uint32_t lane_mod_stride = 0;
  uint32_t lane_div_stride = 0;
  uint64_t unused_static_byte_offset = 0;
  if (!loom_amdgpu_fragment_memory_register_terms(
          layout, plan, packet->register_index, &lane_mod_stride,
          &lane_div_stride, &unused_static_byte_offset)) {
    return 0;
  }
  return lane_mod_stride != 0 ? lane_mod_stride : lane_div_stride;
}

static loom_amdgpu_fp8_packed_u16_repairs_t
loom_amdgpu_fragment_memory_packet_fp8_repairs(
    loom_amdgpu_fragment_memory_packet_flags_t packet_flags) {
  const loom_amdgpu_fragment_memory_packet_flags_t repair_packet_flags =
      packet_flags & LOOM_AMDGPU_FRAGMENT_FP8_REPAIR_PACKET_FLAGS;
  const uint32_t repairs =
      repair_packet_flags >> LOOM_AMDGPU_FRAGMENT_FP8_REPAIR_PACKET_FLAG_SHIFT;
  return (loom_amdgpu_fp8_packed_u16_repairs_t)repairs;
}

static iree_string_view_t
loom_amdgpu_fragment_memory_fp8_packed_decode_strategy_key(
    loom_amdgpu_fragment_memory_packet_flags_t packet_flags) {
  const loom_amdgpu_fp8_packed_u16_repairs_t repairs =
      loom_amdgpu_fragment_memory_packet_fp8_repairs(packet_flags);
  return loom_amdgpu_fp8_packed_bf16_repair_reason_key(repairs);
}

static iree_string_view_t
loom_amdgpu_fragment_memory_fp8_packed_f16_decode_strategy_key(
    loom_amdgpu_fragment_memory_packet_flags_t packet_flags) {
  const loom_amdgpu_fp8_packed_u16_repairs_t repairs =
      loom_amdgpu_fragment_memory_packet_fp8_repairs(packet_flags);
  return loom_amdgpu_fp8_packed_f16_repair_reason_key(repairs);
}

typedef uint32_t loom_amdgpu_fragment_memory_fp8_strategy_properties_t;

enum loom_amdgpu_fragment_memory_fp8_strategy_property_bits_e {
  LOOM_AMDGPU_FRAGMENT_MEMORY_FP8_STRATEGY_PROPERTY_NONE = 0u,
  LOOM_AMDGPU_FRAGMENT_MEMORY_FP8_STRATEGY_PROPERTY_NO_FALLBACK = 1u << 0,
};

typedef struct loom_amdgpu_fragment_memory_fp8_strategy_row_t {
  // Packet flag pattern identifying the fixed report strategy.
  loom_amdgpu_fragment_memory_packet_flags_t packet_flags;
  // Stable compile-report strategy key.
  iree_string_view_t strategy_key;
  // Report classification properties.
  loom_amdgpu_fragment_memory_fp8_strategy_properties_t properties;
} loom_amdgpu_fragment_memory_fp8_strategy_row_t;

static const loom_amdgpu_fragment_memory_fp8_strategy_row_t
    kLoomAmdgpuFragmentMemoryFp8StrategyRows[] = {
        {
            .packet_flags =
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_IDENTITY_E8M0_PK8_F16,
            .strategy_key = IREE_SVL("fp8_identity_e8m0_pk8_f16"),
            .properties =
                LOOM_AMDGPU_FRAGMENT_MEMORY_FP8_STRATEGY_PROPERTY_NO_FALLBACK,
        },
        {
            .packet_flags =
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_IDENTITY_E8M0_PK8_BF16,
            .strategy_key = IREE_SVL("fp8_identity_e8m0_pk8_bf16"),
            .properties =
                LOOM_AMDGPU_FRAGMENT_MEMORY_FP8_STRATEGY_PROPERTY_NO_FALLBACK,
        },
        {
            .packet_flags =
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_IDENTITY_SCALEF32_F16_PAIR,
            .strategy_key = IREE_SVL("fp8_identity_scalef32_f16_pair"),
            .properties =
                LOOM_AMDGPU_FRAGMENT_MEMORY_FP8_STRATEGY_PROPERTY_NO_FALLBACK,
        },
        {
            .packet_flags =
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_IDENTITY_SCALEF32_BF16_PAIR,
            .strategy_key = IREE_SVL("fp8_identity_scalef32_bf16_pair"),
            .properties =
                LOOM_AMDGPU_FRAGMENT_MEMORY_FP8_STRATEGY_PROPERTY_NO_FALLBACK,
        },
        {
            .packet_flags =
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_NATIVE_F16_PAIR,
            .strategy_key = IREE_SVL("fp8_native_f16_pair"),
            .properties =
                LOOM_AMDGPU_FRAGMENT_MEMORY_FP8_STRATEGY_PROPERTY_NO_FALLBACK,
        },
        {
            .packet_flags =
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_NATIVE_F32_PAIR |
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_NATIVE_BF16_PACK,
            .strategy_key = IREE_SVL("fp8_native_f32_pair_native_bf16_pack"),
            .properties =
                LOOM_AMDGPU_FRAGMENT_MEMORY_FP8_STRATEGY_PROPERTY_NO_FALLBACK,
        },
        {
            .packet_flags =
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_NATIVE_F32_PAIR,
            .strategy_key = IREE_SVL("fp8_native_f32_pair_manual_bf16_pack"),
            .properties =
                LOOM_AMDGPU_FRAGMENT_MEMORY_FP8_STRATEGY_PROPERTY_NO_FALLBACK,
        },
        {
            .packet_flags =
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_FULL_BF16_DECODE,
            .strategy_key = IREE_SVL("fp8_full_bf16_decode"),
            .properties =
                LOOM_AMDGPU_FRAGMENT_MEMORY_FP8_STRATEGY_PROPERTY_NONE,
        },
};

typedef struct loom_amdgpu_fragment_memory_fp8_fallback_reason_row_t {
  // Packet flags required to select the fallback reason.
  loom_amdgpu_fragment_memory_packet_flags_t packet_flags;
  // Stable compile-report fallback reason key.
  iree_string_view_t reason_key;
} loom_amdgpu_fragment_memory_fp8_fallback_reason_row_t;

static const loom_amdgpu_fragment_memory_fp8_fallback_reason_row_t
    kLoomAmdgpuFragmentMemoryFp8FullDecodeFallbackReasonRows[] = {
        {
            .packet_flags =
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_MISSING_VALUE_FINITE |
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_MISSING_VALUE_NOT_SUBNORMAL,
            .reason_key = IREE_SVL("missing_finite_not_subnormal"),
        },
        {
            .packet_flags =
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_MISSING_VALUE_FINITE,
            .reason_key = IREE_SVL("missing_finite"),
        },
        {
            .packet_flags =
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_MISSING_VALUE_NOT_SUBNORMAL,
            .reason_key = IREE_SVL("missing_not_subnormal"),
        },
        {
            .packet_flags =
                LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_MISSING_TARGET_PACKETS,
            .reason_key = IREE_SVL("missing_target_packets"),
        },
};

static const loom_amdgpu_fragment_memory_fp8_strategy_row_t*
loom_amdgpu_fragment_memory_lookup_fp8_strategy_row(
    loom_amdgpu_fragment_memory_packet_flags_t packet_flags) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuFragmentMemoryFp8StrategyRows); ++i) {
    const loom_amdgpu_fragment_memory_fp8_strategy_row_t* row =
        &kLoomAmdgpuFragmentMemoryFp8StrategyRows[i];
    if (iree_all_bits_set(packet_flags, row->packet_flags)) {
      return row;
    }
  }
  return NULL;
}

static iree_string_view_t
loom_amdgpu_fragment_memory_fp8_full_decode_fallback_reason(
    loom_amdgpu_fragment_memory_packet_flags_t packet_flags) {
  for (iree_host_size_t i = 0;
       i <
       IREE_ARRAYSIZE(kLoomAmdgpuFragmentMemoryFp8FullDecodeFallbackReasonRows);
       ++i) {
    const loom_amdgpu_fragment_memory_fp8_fallback_reason_row_t* row =
        &kLoomAmdgpuFragmentMemoryFp8FullDecodeFallbackReasonRows[i];
    if (iree_all_bits_set(packet_flags, row->packet_flags)) {
      return row->reason_key;
    }
  }
  return iree_string_view_empty();
}

static iree_string_view_t loom_amdgpu_fragment_memory_packet_strategy_key(
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet) {
  if (loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
          plan->payload_form)) {
    const loom_amdgpu_fragment_memory_fp8_strategy_row_t* strategy_row =
        loom_amdgpu_fragment_memory_lookup_fp8_strategy_row(packet->flags);
    if (strategy_row != NULL) {
      return strategy_row->strategy_key;
    }
    if (iree_any_bit_set(
            packet->flags,
            LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_PACKED_BF16_DECODE)) {
      return loom_amdgpu_fragment_memory_fp8_packed_decode_strategy_key(
          packet->flags);
    }
    if (iree_any_bit_set(
            packet->flags,
            LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_PACKED_F16_DECODE)) {
      return loom_amdgpu_fragment_memory_fp8_packed_f16_decode_strategy_key(
          packet->flags);
    }
    if (iree_any_bit_set(
            packet->flags,
            LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_FULL_BF16_DECODE)) {
      return IREE_SV("fp8_full_bf16_decode");
    }
    return iree_string_view_empty();
  }

  if (plan->payload_form !=
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16) {
    return iree_string_view_empty();
  }
  if (iree_all_bits_set(
          packet->flags,
          LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE)) {
    if (loom_amdgpu_fragment_memory_epilogue_strategy_uses_dpp(
            plan->epilogue_strategy)) {
      return IREE_SV("dpp_packed_bf16_store");
    }
    return IREE_SV("ds_bpermute_packed_bf16_store");
  }
  if (packet->result_register_count > 1) {
    return IREE_SV("packed_bf16_store");
  }
  if (plan->epilogue_strategy ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_SCALAR_B16_STORE) {
    return IREE_SV("scalar_bf16_store");
  }
  return iree_string_view_empty();
}

iree_string_view_t loom_amdgpu_fragment_memory_plan_key(
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  if (plan->packet_count == 0) {
    return iree_string_view_empty();
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
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet) {
  if (loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
          plan->payload_form)) {
    const loom_amdgpu_fragment_memory_fp8_strategy_row_t* strategy_row =
        loom_amdgpu_fragment_memory_lookup_fp8_strategy_row(packet->flags);
    if (strategy_row != NULL &&
        iree_any_bit_set(
            strategy_row->properties,
            LOOM_AMDGPU_FRAGMENT_MEMORY_FP8_STRATEGY_PROPERTY_NO_FALLBACK)) {
      return iree_string_view_empty();
    }
    if (iree_any_bit_set(
            packet->flags,
            LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_PACKED_BF16_DECODE)) {
      return iree_string_view_empty();
    }
    if (iree_any_bit_set(
            packet->flags,
            LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_PACKED_F16_DECODE)) {
      return iree_string_view_empty();
    }
    if (iree_any_bit_set(
            packet->flags,
            LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_FULL_BF16_DECODE)) {
      return loom_amdgpu_fragment_memory_fp8_full_decode_fallback_reason(
          packet->flags);
    }
    return iree_string_view_empty();
  }

  if (plan->payload_form ==
          LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT &&
      packet->result_register_count == 1 && plan->register_count > 1) {
    const uint16_t remaining = plan->register_count - packet->register_index;
    if (remaining > 1 &&
        !loom_amdgpu_fragment_memory_register_group_is_contiguous(
            layout, plan, packet->register_index, 2,
            plan->element_byte_count)) {
      return IREE_SV("fragment_noncontiguous_registers");
    }
    return iree_string_view_empty();
  }

  if (plan->payload_form !=
          LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16 ||
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
    if (candidate <= 1 || candidate > remaining ||
        !loom_amdgpu_fragment_memory_can_use_packed_payload_slice(
            plan, packet->register_index, candidate)) {
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
            layout, plan, packet->register_index, candidate,
            plan->element_byte_count)) {
      return IREE_SV("fragment_noncontiguous_registers");
    }
  }
  return iree_string_view_empty();
}

static iree_status_t loom_amdgpu_fragment_memory_packet_resource(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_value_id_t low_binding,
    loom_value_id_t* out_low_packet_resource,
    loom_value_id_t* out_low_soffset) {
  *out_low_packet_resource = low_binding;
  *out_low_soffset = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_fragment_memory_uses_buffer_descriptor(plan)) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_hal_buffer_descriptor(
      context, source_op, low_binding, &plan->source, out_low_packet_resource));
  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  return loom_amdgpu_emit_const_u32(context, source_op,
                                    LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0,
                                    sgpr_type, out_low_soffset);
}

static iree_status_t loom_amdgpu_record_fragment_memory_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_op_t* low_op, const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    uint16_t element_index, uint32_t vector_lane_count) {
  loom_low_memory_access_summary_t summary = {
      .memory_space =
          loom_amdgpu_fragment_memory_low_space(plan->source.memory_space),
      .alias_root_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
      .alias_group_id = LOOM_LOW_MEMORY_ALIAS_ID_NONE,
  };
  if (summary.memory_space != LOOM_LOW_MEMORY_SPACE_GENERIC) {
    summary.precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_SPACE;
  }
  if (plan->source.alias_scope_id != LOOM_VALUE_FACT_ALIAS_SCOPE_ID_NONE) {
    summary.alias_root_id = plan->source.alias_scope_id;
    summary.precision_flags |= LOOM_LOW_MEMORY_ACCESS_PRECISION_ROOT;
  }
  IREE_RETURN_IF_ERROR(
      loom_low_lower_record_memory_access_summary(context, low_op, &summary));
  if (!loom_low_lower_context_wants_report_rows(context)) {
    return iree_ok_status();
  }

  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_lower_context_descriptor_set(context);
  const loom_low_descriptor_t* descriptor =
      loom_amdgpu_descriptor_ref_descriptor(descriptor_set,
                                            packet->descriptor_ref);
  const loom_low_descriptor_memory_effect_summary_t issued =
      loom_low_descriptor_memory_effect_summary(descriptor_set, descriptor);
  iree_string_view_t packet_key = iree_string_view_empty();
  if (descriptor != NULL) {
    packet_key = loom_low_descriptor_set_string(descriptor_set,
                                                descriptor->key_string_offset);
  }
  int64_t static_offset_bytes = plan->source.static_byte_offset;
  (void)loom_amdgpu_fragment_memory_packet_static_offset(
      layout, plan, packet, element_index, &static_offset_bytes);
  loom_low_source_memory_access_plan_t packet_source = plan->source;
  packet_source.static_byte_offset = static_offset_bytes;
  packet_source.element_byte_count = plan->element_byte_count;
  packet_source.vector_lane_count = vector_lane_count;
  packet_source.vector_lane_byte_stride = plan->element_byte_count;
  loom_low_lower_memory_report_row_t row = {
      .function_name = loom_low_lower_context_function_name(context),
      .source_op_name =
          loom_op_name(loom_low_lower_context_module(context), source_op),
      .source_op_kind = source_op->kind,
      .source_root_name = loom_module_value_name(
          loom_low_lower_context_module(context), plan->source.root_value_id),
      .source_root_argument_index =
          loom_low_lower_source_memory_root_argument_index(context,
                                                           &plan->source),
      .memory_space = loom_amdgpu_memory_space_name(plan->source.memory_space),
      .operation_kind = loom_amdgpu_memory_operation_name(plan->operation_kind),
      .packet_key = packet_key,
      .strategy_key =
          loom_amdgpu_fragment_memory_packet_strategy_key(plan, packet),
      .address_form = loom_amdgpu_fragment_memory_report_address_form(plan),
      .dynamic_term_kind = IREE_SV("vaddr"),
      .fallback_reason = loom_amdgpu_fragment_memory_packet_fallback_reason(
          descriptor_set, layout, plan, packet),
      .static_offset_bytes = static_offset_bytes,
      .element_byte_count = plan->element_byte_count,
      .vector_lane_count = vector_lane_count,
      .issued_read_byte_count = issued.read_byte_count,
      .issued_write_byte_count = issued.write_byte_count,
      .issued_read_unknown_width_count = issued.read_unknown_width_count,
      .issued_write_unknown_width_count = issued.write_unknown_width_count,
      .dynamic_stride_bytes =
          loom_amdgpu_fragment_memory_packet_dynamic_stride_bytes(layout, plan,
                                                                  packet),
      .vector_lane_stride_bytes = plan->element_byte_count,
      .bank_stride_words = 0,
      .bank_conflict_degree = 0,
      .bank_conflict_kind = iree_string_view_empty(),
  };
  loom_amdgpu_memory_report_row_populate_storage_schema(context, &plan->source,
                                                        &row);
  IREE_RETURN_IF_ERROR(
      loom_low_lower_memory_report_row_populate_source_interval(
          context, &packet_source, &row));
  return loom_low_lower_record_memory_report_row(context, source_op, &row);
}

static iree_status_t loom_amdgpu_make_fragment_memory_attrs(
    loom_low_lower_context_t* context, loom_named_attr_t* attrs,
    iree_host_size_t attr_capacity, int64_t immediate_offset,
    iree_host_size_t* out_attr_count) {
  *out_attr_count = 0;
  return loom_amdgpu_append_i64_attr(context, IREE_SV("offset"),
                                     immediate_offset, attrs, attr_capacity,
                                     out_attr_count);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_lane_ids(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint16_t lane_divisor, loom_type_t vgpr_type,
    loom_amdgpu_fragment_lane_ids_t* out_lane_ids) {
  IREE_ASSERT_GT(lane_divisor, 0u);
  IREE_ASSERT_EQ(lane_divisor & (lane_divisor - 1u), 0u);
  *out_lane_ids = (loom_amdgpu_fragment_lane_ids_t){
      .lane = LOOM_VALUE_ID_INVALID,
      .lane_mod = LOOM_VALUE_ID_INVALID,
      .lane_div = LOOM_VALUE_ID_INVALID,
  };
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_amdgpu_fragment_memory_cache_t* cache = NULL;
  if (builder->ip.block != NULL && builder->ip.before_op == NULL) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_get_fragment_memory_cache(context, &cache));
    if (loom_amdgpu_fragment_memory_lane_id_cache_matches(
            context, cache, lane_divisor, vgpr_type)) {
      *out_lane_ids = cache->lane_id_cache.lane_ids;
      return iree_ok_status();
    }
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, vgpr_type, &out_lane_ids->lane));
  if (cache != NULL) {
    cache->lane_id_cache.block = builder->ip.block;
    cache->lane_id_cache.vgpr_type = vgpr_type;
    cache->lane_id_cache.lane_divisor = lane_divisor;
    cache->lane_id_cache.lane_ids = *out_lane_ids;
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_load_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    uint16_t element_index, uint32_t vector_lane_count, loom_type_t result_type,
    const loom_amdgpu_fragment_memory_address_t* address,
    loom_value_id_t low_resource, loom_value_id_t low_soffset,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_fragment_memory_attrs(
      context, attrs, IREE_ARRAYSIZE(attrs), address->immediate_offset,
      &attr_count));

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, packet->descriptor_ref, &descriptor));
  loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_descriptor_has_implicit_resource_operand(
          loom_low_lower_context_descriptor_set(context),
          descriptor.descriptor)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_m0_u32(context, source_op, &descriptor, 0, &low_m0));
  }

  loom_value_id_t operands[4] = {0};
  iree_host_size_t operand_count = 0;
  if (loom_amdgpu_fragment_memory_uses_buffer_descriptor(plan)) {
    operands[operand_count++] = low_resource;
    operands[operand_count++] = address->low_vaddr;
    operands[operand_count++] = low_soffset;
  } else {
    operands[operand_count++] = address->low_vaddr;
    if (plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
      operands[operand_count++] = low_resource;
    }
  }
  if (low_m0 != LOOM_VALUE_ID_INVALID) {
    operands[operand_count++] = low_m0;
  }
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), &result_type, 1,
      /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
      &low_op));
  IREE_RETURN_IF_ERROR(loom_amdgpu_record_fragment_memory_packet(
      context, source_op, low_op, layout, plan, packet, element_index,
      vector_lane_count));
  *out_low_packet = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_low_subword_load_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_packet, loom_type_t vgpr_type,
    loom_value_id_t* out_full_packet) {
  *out_full_packet = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_low_value_defines_vgpr_low16(context, low_packet)) {
    return loom_amdgpu_emit_vgpr_unary(
        context, source_op,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_U32_OFFSET_0_WIDTH_16_LOW16,
        low_packet, vgpr_type, out_full_packet);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, low_packet, out_full_packet));
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      *out_full_packet, UINT32_C(0xFFFF), vgpr_type, out_full_packet);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_cmp_u32_lit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_descriptor_ref_t descriptor_ref, loom_value_id_t low_value,
    uint32_t immediate, loom_type_t vgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_mask) {
  *out_mask = LOOM_VALUE_ID_INVALID;
  loom_low_lower_resolved_descriptor_t src1_inline_descriptor = {0};
  if (immediate <= LOOM_AMDGPU_SOURCE_INLINE_U32_MAX) {
    const loom_amdgpu_descriptor_ref_t src1_inline_ref =
        loom_amdgpu_fragment_memory_compare_i32_src1_inline_ref(descriptor_ref);
    if (src1_inline_ref != LOOM_AMDGPU_DESCRIPTOR_REF_NONE) {
      bool present = false;
      IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref_if_present(
          context, src1_inline_ref, &src1_inline_descriptor, &present));
    }
  }
  if (src1_inline_descriptor.descriptor != NULL) {
    loom_named_attr_t attrs[1] = {0};
    iree_host_size_t attr_count = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_i64_attr(context, IREE_SV("rhs"), immediate, attrs,
                                    IREE_ARRAYSIZE(attrs), &attr_count));
    const loom_value_id_t operands[] = {low_value};
    loom_op_t* compare_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &src1_inline_descriptor, operands, IREE_ARRAYSIZE(operands),
        loom_make_named_attr_slice(attrs, attr_count), &mask_type, 1,
        /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
        &compare_op));
    *out_mask = loom_value_slice_get(loom_low_op_results(compare_op), 0);
    return iree_ok_status();
  }

  loom_value_id_t low_immediate = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, immediate,
      vgpr_type, &low_immediate));
  const loom_value_id_t operands[] = {low_value, low_immediate};
  loom_op_t* compare_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, descriptor_ref, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &mask_type, 1, &compare_op));
  *out_mask = loom_value_slice_get(loom_low_op_results(compare_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_select_b32(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_false_value, loom_value_id_t low_true_value,
    loom_value_id_t low_condition, loom_type_t vgpr_type,
    loom_value_id_t* out_value) {
  *out_value = LOOM_VALUE_ID_INVALID;
  const loom_value_id_t operands[] = {
      low_false_value,
      low_true_value,
      low_condition,
  };
  loom_op_t* select_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CNDMASK_B32, operands,
      IREE_ARRAYSIZE(operands), loom_named_attr_slice_empty(), &vgpr_type, 1,
      &select_op));
  *out_value = loom_value_slice_get(loom_low_op_results(select_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_source_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan, loom_value_id_t low_source,
    loom_type_t vgpr_type, loom_value_id_t* out_source_registers) {
  if (plan->source_register_count == 1) {
    out_source_registers[0] = low_source;
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < plan->source_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(context, source_op,
                                                    low_source, i, vgpr_type,
                                                    &out_source_registers[i]));
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fragment_repack_narrow_source_registers_to_bf16(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors,
    loom_type_t vgpr_type, loom_value_id_t* inout_source_registers) {
  for (uint16_t i = 0; i < plan->source_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
        context, source_op, bf16_pack_descriptors, inout_source_registers[i],
        vgpr_type, &inout_source_registers[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_source_register_selector(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    loom_amdgpu_fragment_lane_ids_t* lane_ids, loom_type_t vgpr_type,
    loom_value_id_t* out_low_source_selector) {
  *out_low_source_selector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fragment_memory_lane_mod(
      context, source_op, plan->lane_divisor, vgpr_type, lane_ids));
  switch (plan->source_map_kind) {
    case LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN:
      return loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
          loom_amdgpu_fragment_repack_log2_u16(plan->lane_group_count),
          lane_ids->lane_mod, vgpr_type, out_low_source_selector);
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN:
      return loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
          lane_ids->lane_mod, plan->source_register_count - 1u, vgpr_type,
          out_low_source_selector);
    default:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU fragment repack source map");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t
loom_amdgpu_emit_fragment_repack_exact_source_register_masks(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    loom_amdgpu_fragment_lane_ids_t* lane_ids, loom_type_t vgpr_type,
    loom_type_t mask_type, loom_value_id_t* out_source_register_masks) {
  if (plan->source_register_count <= 1) {
    return iree_ok_status();
  }

  loom_value_id_t low_source_selector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fragment_repack_source_register_selector(
          context, source_op, plan, lane_ids, vgpr_type, &low_source_selector));
  for (uint16_t i = 1; i < plan->source_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_cmp_u32_lit(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
        low_source_selector, i, vgpr_type, mask_type,
        &out_source_register_masks[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_source_register_bit_masks(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    loom_amdgpu_fragment_lane_ids_t* lane_ids, loom_type_t vgpr_type,
    loom_type_t mask_type, loom_value_id_t* out_source_register_bit_masks) {
  if (plan->source_register_count <= 1) {
    return iree_ok_status();
  }

  loom_value_id_t low_source_selector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fragment_repack_source_register_selector(
          context, source_op, plan, lane_ids, vgpr_type, &low_source_selector));
  const uint16_t bit_count =
      loom_amdgpu_fragment_repack_log2_u16(plan->source_register_count);
  for (uint16_t i = 0; i < bit_count; ++i) {
    loom_value_id_t low_selector_bit = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
        low_source_selector, UINT32_C(1) << i, vgpr_type, &low_selector_bit));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_cmp_u32_lit(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_NE_I32,
        low_selector_bit, 0, vgpr_type, mask_type,
        &out_source_register_bit_masks[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_lane_group_byte_base(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    loom_amdgpu_fragment_lane_ids_t* lane_ids, loom_type_t vgpr_type,
    loom_value_id_t* out_low_byte_base) {
  *out_low_byte_base = LOOM_VALUE_ID_INVALID;
  loom_value_id_t source_lane_group = LOOM_VALUE_ID_INVALID;
  switch (plan->source_map_kind) {
    case LOOM_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN: {
      if (!loom_amdgpu_fragment_repack_has_static_zero_source_byte_base(plan)) {
        IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fragment_memory_lane_mod(
            context, source_op, plan->lane_divisor, vgpr_type, lane_ids));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
            lane_ids->lane_mod, plan->lane_group_count - 1u, vgpr_type,
            &source_lane_group));
      } else {
        if (plan->result_lane_div_byte_shift == 0) {
          return loom_amdgpu_emit_const_u32(
              context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32, 0,
              vgpr_type, out_low_byte_base);
        }
        IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fragment_memory_lane_div(
            context, source_op, plan->lane_divisor, vgpr_type, lane_ids));
        return loom_amdgpu_emit_vgpr_shift(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
            plan->result_lane_div_byte_shift, lane_ids->lane_div, vgpr_type,
            out_low_byte_base);
      }
      break;
    }
    case LOOM_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN: {
      IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fragment_memory_lane_mod(
          context, source_op, plan->lane_divisor, vgpr_type, lane_ids));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
          loom_amdgpu_fragment_repack_log2_u16(plan->source_register_count),
          lane_ids->lane_mod, vgpr_type, &source_lane_group));
      break;
    }
    default:
      IREE_ASSERT_UNREACHABLE("unknown AMDGPU fragment repack source map");
      IREE_BUILTIN_UNREACHABLE();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      plan->source_lane_group_byte_shift, source_lane_group, vgpr_type,
      out_low_byte_base));
  if (plan->result_lane_div_byte_shift == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fragment_memory_lane_div(
      context, source_op, plan->lane_divisor, vgpr_type, lane_ids));
  loom_value_id_t low_lane_div_byte_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT,
      plan->result_lane_div_byte_shift, lane_ids->lane_div, vgpr_type,
      &low_lane_div_byte_base));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_ADD_U32,
      *out_low_byte_base, low_lane_div_byte_base, vgpr_type, out_low_byte_base);
}

static iree_status_t loom_amdgpu_emit_fragment_repack_bpermute_candidates(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    const loom_low_lower_resolved_descriptor_t* bpermute_descriptor,
    const loom_value_id_t* source_registers,
    loom_value_id_t low_source_byte_offset, uint32_t static_byte_offset,
    loom_type_t vgpr_type, loom_value_id_t* out_low_candidates) {
  for (uint16_t i = 0; i < plan->source_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_bpermute_register(
        context, source_op, bpermute_descriptor, low_source_byte_offset,
        static_byte_offset, source_registers[i], vgpr_type,
        &out_low_candidates[i]));
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fragment_repack_dpp_packed_source_registers(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    const loom_low_lower_resolved_descriptor_t* dpp_descriptor,
    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors,
    const loom_value_id_t* source_registers, bool pre_narrow_source_registers,
    loom_type_t vgpr_type, loom_value_id_t* out_packed_source_registers) {
  const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor =
      iree_any_bit_set(bf16_pack_descriptors->flags,
                       LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_PACK_U16)
          ? &bf16_pack_descriptors->pack_u16_descriptor
          : NULL;
  for (uint16_t i = 0; i < plan->source_register_count; ++i) {
    loom_value_id_t paired_source_register = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_dpp_register(
        context, source_op, dpp_descriptor, source_registers[i],
        LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_1, vgpr_type, &paired_source_register));
    if (pre_narrow_source_registers) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_bf16_lane_pair(
          context, source_op, pack_u16_descriptor, source_registers[i],
          paired_source_register, vgpr_type, &out_packed_source_registers[i]));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
              context, source_op, bf16_pack_descriptors, source_registers[i],
              paired_source_register, vgpr_type,
              &out_packed_source_registers[i]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_linear_bpermute_element(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    const loom_low_lower_resolved_descriptor_t* bpermute_descriptor,
    const loom_value_id_t* source_registers,
    const loom_value_id_t* source_register_masks,
    loom_value_id_t low_source_byte_offset, uint32_t static_byte_offset,
    loom_type_t vgpr_type, loom_value_id_t* out_low_element) {
  *out_low_element = LOOM_VALUE_ID_INVALID;
  loom_value_id_t candidates[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
      LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_repack_bpermute_candidates(
      context, source_op, plan, bpermute_descriptor, source_registers,
      low_source_byte_offset, static_byte_offset, vgpr_type, candidates));
  *out_low_element = candidates[0];
  for (uint16_t i = 1; i < plan->source_register_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_select_b32(
        context, source_op, *out_low_element, candidates[i],
        source_register_masks[i], vgpr_type, out_low_element));
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_repack_tree_bpermute_element(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan,
    const loom_low_lower_resolved_descriptor_t* bpermute_descriptor,
    const loom_value_id_t* source_registers,
    const loom_value_id_t* source_register_bit_masks,
    loom_value_id_t low_source_byte_offset, uint32_t static_byte_offset,
    loom_type_t vgpr_type, loom_value_id_t* out_low_element) {
  *out_low_element = LOOM_VALUE_ID_INVALID;
  loom_value_id_t selected[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
      LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_repack_bpermute_candidates(
      context, source_op, plan, bpermute_descriptor, source_registers,
      low_source_byte_offset, static_byte_offset, vgpr_type, selected));
  uint16_t selected_count = plan->source_register_count;
  for (uint16_t bit_index = 0; selected_count > 1; ++bit_index) {
    uint16_t next_count = 0;
    for (uint16_t i = 0; i < selected_count; i += 2) {
      if (i + 1 == selected_count) {
        selected[next_count++] = selected[i];
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_select_b32(
          context, source_op, selected[i], selected[i + 1],
          source_register_bit_masks[bit_index], vgpr_type,
          &selected[next_count++]));
    }
    selected_count = next_count;
  }
  *out_low_element = selected[0];
  return iree_ok_status();
}

static bool loom_amdgpu_fragment_repack_uses_dpp_packed_source_pairs(
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  return plan->strategy ==
         LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_DPP_BPERMUTE;
}

static iree_status_t
loom_amdgpu_emit_fragment_repack_result_to_lhs_bf16_bpermute(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  loom_value_id_t low_source = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->source, &low_source));
  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  loom_type_t mask_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));

  loom_value_id_t source_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
      LOOM_VALUE_ID_INVALID};
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_repack_source_registers(
      context, source_op, plan, low_source, vgpr_type, source_registers));

  const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_get_bf16_pack_descriptors(context, &bf16_pack_descriptors));
  const bool pre_narrow_source_registers =
      !iree_any_bit_set(bf16_pack_descriptors->flags,
                        LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE);
  if (pre_narrow_source_registers) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_repack_narrow_source_registers_to_bf16(
            context, source_op, plan, bf16_pack_descriptors, vgpr_type,
            source_registers));
  }

  loom_amdgpu_fragment_lane_ids_t lane_ids;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_lane_ids(
      context, source_op, plan->lane_divisor, vgpr_type, &lane_ids));
  loom_value_id_t
      source_register_masks[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
          LOOM_VALUE_ID_INVALID};
  const bool use_source_register_bit_tree =
      loom_amdgpu_fragment_repack_uses_source_register_bit_tree(plan);
  if (use_source_register_bit_tree) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_repack_source_register_bit_masks(
            context, source_op, plan, &lane_ids, vgpr_type, mask_type,
            source_register_masks));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_repack_exact_source_register_masks(
            context, source_op, plan, &lane_ids, vgpr_type, mask_type,
            source_register_masks));
  }

  loom_value_id_t low_lane_group_byte_base = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_repack_lane_group_byte_base(
      context, source_op, plan, &lane_ids, vgpr_type,
      &low_lane_group_byte_base));

  loom_low_lower_resolved_descriptor_t bpermute_descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
      &bpermute_descriptor));
  const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor =
      iree_any_bit_set(bf16_pack_descriptors->flags,
                       LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_PACK_U16)
          ? &bf16_pack_descriptors->pack_u16_descriptor
          : NULL;

  const bool use_dpp_packed_source_pairs =
      loom_amdgpu_fragment_repack_uses_dpp_packed_source_pairs(plan);
  loom_value_id_t
      packed_source_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
          LOOM_VALUE_ID_INVALID};
  if (use_dpp_packed_source_pairs) {
    const loom_low_descriptor_set_t* descriptor_set =
        loom_low_lower_context_descriptor_set(context);
    const loom_amdgpu_descriptor_ref_t dpp_descriptor_ref =
        loom_amdgpu_dpp_b32_descriptor_ref(descriptor_set);
    IREE_ASSERT_NE(dpp_descriptor_ref, LOOM_AMDGPU_DESCRIPTOR_REF_NONE);
    loom_low_lower_resolved_descriptor_t dpp_descriptor = {0};
    IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
        context, dpp_descriptor_ref, &dpp_descriptor));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_repack_dpp_packed_source_registers(
            context, source_op, plan, &dpp_descriptor, bf16_pack_descriptors,
            source_registers, pre_narrow_source_registers, vgpr_type,
            packed_source_registers));
  }

  loom_value_id_t result_registers[LOOM_AMDGPU_MAX_SCALARIZED_32BIT_LANES] = {
      LOOM_VALUE_ID_INVALID};
  for (uint16_t register_index = 0;
       register_index < plan->result_register_count; ++register_index) {
    if (use_dpp_packed_source_pairs) {
      const uint32_t static_byte_offset =
          (uint32_t)register_index * LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT *
          LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
      if (use_source_register_bit_tree) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_repack_tree_bpermute_element(
                context, source_op, plan, &bpermute_descriptor,
                packed_source_registers, source_register_masks,
                low_lane_group_byte_base, static_byte_offset, vgpr_type,
                &result_registers[register_index]));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_repack_linear_bpermute_element(
                context, source_op, plan, &bpermute_descriptor,
                packed_source_registers, source_register_masks,
                low_lane_group_byte_base, static_byte_offset, vgpr_type,
                &result_registers[register_index]));
      }
      continue;
    }

    loom_value_id_t elements[LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT] = {
        LOOM_VALUE_ID_INVALID};
    for (uint16_t element_index = 0;
         element_index < LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
         ++element_index) {
      const uint16_t reduction =
          (uint16_t)(register_index *
                         LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT +
                     element_index);
      const uint32_t static_byte_offset =
          (uint32_t)reduction * LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT;
      if (use_source_register_bit_tree) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_repack_tree_bpermute_element(
                context, source_op, plan, &bpermute_descriptor,
                source_registers, source_register_masks,
                low_lane_group_byte_base, static_byte_offset, vgpr_type,
                &elements[element_index]));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_repack_linear_bpermute_element(
                context, source_op, plan, &bpermute_descriptor,
                source_registers, source_register_masks,
                low_lane_group_byte_base, static_byte_offset, vgpr_type,
                &elements[element_index]));
      }
    }
    if (pre_narrow_source_registers) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_packed_bf16_lane_pair(
          context, source_op, pack_u16_descriptor, elements[0], elements[1],
          vgpr_type, &result_registers[register_index]));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
              context, source_op, bf16_pack_descriptors, elements[0],
              elements[1], vgpr_type, &result_registers[register_index]));
    }
  }

  return loom_amdgpu_bind_low_register_range(context, source_op, plan->result,
                                             result_registers,
                                             plan->result_register_count);
}

iree_status_t loom_amdgpu_lower_vector_fragment_repack(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_repack_plan_t* plan) {
  switch (plan->strategy) {
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_ALIAS:
      return loom_low_lower_bind_value_alias(context, plan->source,
                                             plan->result);
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_DPP_BPERMUTE:
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_RESULT_TO_LHS_BF16_BPERMUTE:
      return loom_amdgpu_emit_fragment_repack_result_to_lhs_bf16_bpermute(
          context, source_op, plan);
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_DIAGNOSTIC:
      return loom_amdgpu_emit_fragment_repack_diagnostic(context, source_op,
                                                         plan);
    case LOOM_AMDGPU_FRAGMENT_REPACK_STRATEGY_NONE:
    default:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU fragment repack strategy");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_fp8_source_byte(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fp8_decode_plan_t* decode_plan,
    loom_value_id_t low_source_packet, uint16_t packet_register_count,
    uint16_t byte_index, loom_type_t vgpr_type, loom_value_id_t* out_low_byte) {
  *out_low_byte = LOOM_VALUE_ID_INVALID;
  const uint16_t source_register_index = byte_index / 4u;
  if (source_register_index >= packet_register_count) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU FP8 fragment packet");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_value_id_t low_source_register = low_source_packet;
  if (packet_register_count != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_source_packet, source_register_index, vgpr_type,
        &low_source_register));
  }

  const uint16_t bit_offset = (byte_index & 3u) * 8u;
  if (bit_offset != 0 &&
      iree_any_bit_set(decode_plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_BFE_U32)) {
    loom_named_attr_t attrs[2] = {0};
    iree_host_size_t attr_count = 0;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_i64_attr(context, IREE_SV("offset"), bit_offset,
                                    attrs, IREE_ARRAYSIZE(attrs), &attr_count));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_append_i64_attr(context, IREE_SV("width"), 8, attrs,
                                    IREE_ARRAYSIZE(attrs), &attr_count));

    const loom_value_id_t operands[] = {low_source_register};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, &decode_plan->bfe_u32_descriptor, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(attrs, attr_count),
        &vgpr_type, 1, /*tied_results=*/NULL, /*tied_result_count=*/0,
        source_op->location, &low_op));
    *out_low_byte = loom_value_slice_get(loom_low_op_results(low_op), 0);
    return iree_ok_status();
  }

  loom_value_id_t shifted_byte = low_source_register;
  if (bit_offset != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        bit_offset, low_source_register, vgpr_type, &shifted_byte));
  }
  *out_low_byte = shifted_byte;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_fp8_source_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t packet_register_count,
    uint16_t source_register_index, loom_type_t vgpr_type,
    loom_value_id_t* out_source_register) {
  *out_source_register = LOOM_VALUE_ID_INVALID;
  if (source_register_index >= packet_register_count) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU FP8 fragment packet");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_value_id_t source_register = low_source;
  if (packet_register_count != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_source, source_register_index, vgpr_type,
        &source_register));
  }
  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
      context, source_op, source_register, &source_register));
  *out_source_register = source_register;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_shifted_fp8_pair_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t packet_register_count,
    uint16_t result_register_index, loom_type_t vgpr_type,
    loom_value_id_t* out_source_register) {
  *out_source_register = LOOM_VALUE_ID_INVALID;
  const uint16_t byte_index =
      result_register_index * LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
  const uint16_t source_register_index = byte_index / 4u;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_fp8_source_register(
      context, source_op, low_source, packet_register_count,
      source_register_index, vgpr_type, out_source_register));

  const uint16_t bit_offset = (byte_index & 3u) * 8u;
  if (bit_offset != 0) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT,
        bit_offset, *out_source_register, vgpr_type, out_source_register));
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_identity_scalef32_fp8_to_packed_bf16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t packet_register_count,
    uint16_t result_register_index,
    const loom_low_lower_resolved_descriptor_t* scalef32_bf16_descriptor,
    loom_value_id_t low_identity_scale, loom_type_t vgpr_type,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;

  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fragment_memory_shifted_fp8_pair_register(
          context, source_op, low_source, packet_register_count,
          result_register_index, vgpr_type, &source_register));

  const loom_value_id_t operands[] = {source_register, low_identity_scale};
  loom_op_t* convert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, scalef32_bf16_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &vgpr_type, 1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &convert_op));
  *out_low_packet = loom_value_slice_get(loom_low_op_results(convert_op), 0);
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_native_fp8_to_packed_bf16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t packet_register_count,
    uint16_t result_register_index,
    const loom_low_lower_resolved_descriptor_t* native_f32_pair_descriptor,
    loom_type_t native_f32_pair_type,
    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors,
    loom_type_t vgpr_type, loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;

  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fragment_memory_shifted_fp8_pair_register(
          context, source_op, low_source, packet_register_count,
          result_register_index, vgpr_type, &source_register));

  const loom_value_id_t operands[] = {source_register};
  loom_op_t* convert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, native_f32_pair_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &native_f32_pair_type, 1,
      /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &convert_op));
  const loom_value_id_t converted_pair =
      loom_value_slice_get(loom_low_op_results(convert_op), 0);
  loom_value_id_t low_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, converted_pair, 0, vgpr_type, &low_lane));
  loom_value_id_t high_lane = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, converted_pair, 1, vgpr_type, &high_lane));
  return loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
      context, source_op, bf16_pack_descriptors, low_lane, high_lane, vgpr_type,
      out_low_packet);
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_identity_scalef32_fp8_to_packed_f16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t packet_register_count,
    uint16_t result_register_index,
    const loom_low_lower_resolved_descriptor_t* scalef32_f16_descriptor,
    loom_value_id_t low_identity_scale, loom_type_t vgpr_type,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;

  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fragment_memory_shifted_fp8_pair_register(
          context, source_op, low_source, packet_register_count,
          result_register_index, vgpr_type, &source_register));

  const loom_value_id_t operands[] = {source_register, low_identity_scale};
  loom_op_t* convert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, scalef32_f16_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &vgpr_type, 1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &convert_op));
  *out_low_packet = loom_value_slice_get(loom_low_op_results(convert_op), 0);
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_identity_e8m0_pk8_fp8_to_16bit_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_packet, uint16_t packet_register_count,
    uint16_t result_register_count,
    const loom_low_lower_resolved_descriptor_t* e8m0_pk8_descriptor,
    loom_value_id_t low_identity_scale, loom_type_t vgpr_type,
    loom_value_id_t* out_low_result_registers) {
  if (result_register_count == 0 || result_register_count % 4u != 0 ||
      packet_register_count < result_register_count / 2u) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU FP8 E8M0 pk8 fragment packet");
    IREE_BUILTIN_UNREACHABLE();
  }

  loom_type_t source_pair_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 2, &source_pair_type));
  loom_type_t result_quad_type = loom_type_none();
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_make_vgpr_range_type(context, 4, &result_quad_type));

  for (uint16_t result_register_index = 0;
       result_register_index < result_register_count;
       result_register_index += 4u) {
    const uint16_t source_register_index = result_register_index / 2u;
    loom_value_id_t source_registers[2] = {LOOM_VALUE_ID_INVALID,
                                           LOOM_VALUE_ID_INVALID};
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_fp8_source_register(
        context, source_op, low_source_packet, packet_register_count,
        source_register_index, vgpr_type, &source_registers[0]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_fp8_source_register(
        context, source_op, low_source_packet, packet_register_count,
        source_register_index + 1u, vgpr_type, &source_registers[1]));

    loom_op_t* source_pair_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_concat_build(
        loom_low_lower_context_builder(context), source_registers,
        IREE_ARRAYSIZE(source_registers), source_pair_type, source_op->location,
        &source_pair_op));
    const loom_value_id_t source_pair =
        loom_value_slice_get(loom_low_op_results(source_pair_op), 0);

    const loom_value_id_t operands[] = {source_pair, low_identity_scale};
    loom_op_t* convert_op = NULL;
    IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
        context, e8m0_pk8_descriptor, operands, IREE_ARRAYSIZE(operands),
        loom_named_attr_slice_empty(), &result_quad_type, 1,
        /*tied_results=*/NULL, /*tied_result_count=*/0, source_op->location,
        &convert_op));
    const loom_value_id_t converted_quad =
        loom_value_slice_get(loom_low_op_results(convert_op), 0);
    for (uint16_t quad_index = 0; quad_index < 4u; ++quad_index) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, converted_quad, quad_index, vgpr_type,
          &out_low_result_registers[result_register_index + quad_index]));
    }
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_native_fp8_to_packed_f16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t packet_register_count,
    uint16_t result_register_index,
    const loom_low_lower_resolved_descriptor_t* native_f16_pair_descriptor,
    loom_type_t vgpr_type, loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;

  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fragment_memory_shifted_fp8_pair_register(
          context, source_op, low_source, packet_register_count,
          result_register_index, vgpr_type, &source_register));

  const loom_value_id_t operands[] = {source_register};
  loom_op_t* convert_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, native_f16_pair_descriptor, operands, IREE_ARRAYSIZE(operands),
      loom_named_attr_slice_empty(), &vgpr_type, 1, /*tied_results=*/NULL,
      /*tied_result_count=*/0, source_op->location, &convert_op));
  *out_low_packet = loom_value_slice_get(loom_low_op_results(convert_op), 0);
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_fp8_to_packed_f16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t packet_register_count,
    uint16_t result_register_index,
    const loom_amdgpu_fp8_decode_plan_t* decode_plan,
    loom_amdgpu_fp8_decode_value_flags_t decode_value_flags,
    const loom_low_lower_resolved_descriptor_t* scalef32_f16_descriptor,
    loom_value_id_t low_identity_scale,
    const loom_amdgpu_fp8_native_descriptors_t* native_f16_descriptors,
    loom_type_t vgpr_type, loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;
  if (scalef32_f16_descriptor != NULL) {
    return loom_amdgpu_emit_fragment_memory_identity_scalef32_fp8_to_packed_f16_register(
        context, source_op, low_source, packet_register_count,
        result_register_index, scalef32_f16_descriptor, low_identity_scale,
        vgpr_type, out_low_packet);
  }
  if (native_f16_descriptors != NULL &&
      iree_any_bit_set(native_f16_descriptors->flags,
                       LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR)) {
    return loom_amdgpu_emit_fragment_memory_native_fp8_to_packed_f16_register(
        context, source_op, low_source, packet_register_count,
        result_register_index, &native_f16_descriptors->pair_descriptor,
        vgpr_type, out_low_packet);
  }

  loom_value_id_t source_register = LOOM_VALUE_ID_INVALID;
  const uint16_t byte_index =
      result_register_index * LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
  const uint16_t source_register_index = byte_index / 4u;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_fp8_source_register(
      context, source_op, low_source, packet_register_count,
      source_register_index, vgpr_type, &source_register));
  const loom_amdgpu_fp8_packed_u16_pair_source_t pair_source = {
      .source_register = source_register,
      .byte_offset = byte_index & 3u,
      .live_lane_count = LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT,
  };
  return loom_amdgpu_emit_fp8_pairs_to_packed_f16_finite(
      context, source_op, decode_plan, &pair_source, /*pair_count=*/1,
      decode_value_flags, vgpr_type, sgpr_type, mask_type, out_low_packet);
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_fp8_to_packed_bf16_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t packet_register_count,
    uint16_t result_register_index,
    const loom_amdgpu_fp8_decode_plan_t* decode_plan,
    loom_amdgpu_fp8_decode_value_flags_t decode_value_flags,
    const loom_low_lower_resolved_descriptor_t* scalef32_bf16_descriptor,
    loom_value_id_t low_identity_scale, loom_type_t native_f32_pair_type,
    const loom_amdgpu_fp8_native_descriptors_t* native_f32_descriptors,
    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors,
    loom_type_t vgpr_type, loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;
  if (scalef32_bf16_descriptor != NULL) {
    return loom_amdgpu_emit_fragment_memory_identity_scalef32_fp8_to_packed_bf16_register(
        context, source_op, low_source, packet_register_count,
        result_register_index, scalef32_bf16_descriptor, low_identity_scale,
        vgpr_type, out_low_packet);
  }

  const bool prefer_packed_bf16 =
      loom_amdgpu_fp8_prefers_packed_bf16_pair_decode(decode_plan,
                                                      decode_value_flags);
  if (prefer_packed_bf16) {
    const uint16_t byte_index =
        result_register_index * LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
    const uint16_t source_register_index = byte_index / 4u;

    loom_value_id_t low_source_register = low_source;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_fp8_source_register(
        context, source_op, low_source, packet_register_count,
        source_register_index, vgpr_type, &low_source_register));

    const loom_amdgpu_fp8_packed_u16_pair_source_t pair_source = {
        .source_register = low_source_register,
        .byte_offset = byte_index & 3u,
        .live_lane_count = LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT,
    };
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_pairs_to_packed_bf16(
        context, source_op, decode_plan, &pair_source, /*pair_count=*/1,
        decode_value_flags, vgpr_type, sgpr_type, mask_type, out_low_packet));
    return iree_ok_status();
  }

  if (native_f32_descriptors != NULL &&
      iree_any_bit_set(native_f32_descriptors->flags,
                       LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR)) {
    return loom_amdgpu_emit_fragment_memory_native_fp8_to_packed_bf16_register(
        context, source_op, low_source, packet_register_count,
        result_register_index, &native_f32_descriptors->pair_descriptor,
        native_f32_pair_type, bf16_pack_descriptors, vgpr_type, out_low_packet);
  }

  loom_value_id_t low_elements[LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT] =
      {0};
  for (uint16_t element_index = 0;
       element_index < LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
       ++element_index) {
    const uint16_t byte_index =
        result_register_index * LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT +
        element_index;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_fp8_source_byte(
        context, source_op, decode_plan, low_source, packet_register_count,
        byte_index, vgpr_type, &low_elements[element_index]));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_to_bf16_lane(
        context, source_op, decode_plan, low_elements[element_index],
        decode_value_flags, vgpr_type, sgpr_type, mask_type,
        &low_elements[element_index]));
  }

  const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor =
      iree_any_bit_set(decode_plan->flags,
                       LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PACK_U16)
          ? &decode_plan->pack_u16_descriptor
          : NULL;
  return loom_amdgpu_emit_packed_bf16_lane_pair(
      context, source_op, pack_u16_descriptor, low_elements[0], low_elements[1],
      vgpr_type, out_low_packet);
}

static iree_status_t loom_amdgpu_fragment_memory_prepare_fp8_pair_sources(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_packet, uint16_t packet_register_count,
    uint16_t result_register_count, loom_type_t vgpr_type,
    loom_amdgpu_fp8_packed_u16_pair_source_t* out_pair_sources) {
  loom_value_id_t source_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {
      0};
  for (uint16_t source_register_index = 0;
       source_register_index < packet_register_count; ++source_register_index) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_fp8_source_register(
        context, source_op, low_source_packet, packet_register_count,
        source_register_index, vgpr_type,
        &source_registers[source_register_index]));
  }

  for (uint16_t result_register_index = 0;
       result_register_index < result_register_count; ++result_register_index) {
    const uint16_t byte_index =
        result_register_index * LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
    const uint16_t source_register_index = byte_index / 4u;
    if (source_register_index >= packet_register_count) {
      IREE_ASSERT_UNREACHABLE("selected AMDGPU FP8 fragment packet");
      IREE_BUILTIN_UNREACHABLE();
    }
    out_pair_sources[result_register_index] =
        (loom_amdgpu_fp8_packed_u16_pair_source_t){
            .source_register = source_registers[source_register_index],
            .byte_offset = byte_index & 3u,
            .live_lane_count = 2u,
        };
  }
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_try_emit_fragment_memory_fp8_to_packed_bf16_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_packet, uint16_t packet_register_count,
    uint16_t result_register_count,
    const loom_amdgpu_fp8_decode_plan_t* decode_plan,
    loom_amdgpu_fp8_decode_value_flags_t decode_value_flags,
    loom_type_t vgpr_type, loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_low_result_registers) {
  if (packet_register_count == 0 ||
      packet_register_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS ||
      result_register_count == 0 ||
      result_register_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return iree_ok_status();
  }

  loom_amdgpu_fp8_packed_u16_pair_source_t
      pair_sources[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_prepare_fp8_pair_sources(
      context, source_op, low_source_packet, packet_register_count,
      result_register_count, vgpr_type, pair_sources));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_pairs_to_packed_bf16(
      context, source_op, decode_plan, pair_sources, result_register_count,
      decode_value_flags, vgpr_type, sgpr_type, mask_type,
      out_low_result_registers));
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_try_emit_fragment_memory_fp8_to_packed_f16_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source_packet, uint16_t packet_register_count,
    uint16_t result_register_count,
    const loom_amdgpu_fp8_decode_plan_t* decode_plan,
    loom_amdgpu_fp8_decode_value_flags_t decode_value_flags,
    loom_type_t vgpr_type, loom_type_t sgpr_type, loom_type_t mask_type,
    loom_value_id_t* out_low_result_registers) {
  if (packet_register_count == 0 ||
      packet_register_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS ||
      result_register_count == 0 ||
      result_register_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS ||
      !loom_amdgpu_can_emit_fp8_pair_to_packed_f16_finite(decode_plan,
                                                          decode_value_flags)) {
    return iree_ok_status();
  }

  loom_amdgpu_fp8_packed_u16_pair_source_t
      pair_sources[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_prepare_fp8_pair_sources(
      context, source_op, low_source_packet, packet_register_count,
      result_register_count, vgpr_type, pair_sources));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fp8_pairs_to_packed_f16_finite(
      context, source_op, decode_plan, pair_sources, result_register_count,
      decode_value_flags, vgpr_type, sgpr_type, mask_type,
      out_low_result_registers));
  return iree_ok_status();
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_fp8_to_packed_16bit_load_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    const loom_amdgpu_fragment_memory_address_accumulator_t* base_accumulator,
    loom_value_id_t low_address_resource, loom_value_id_t low_packet_resource,
    loom_type_t vgpr_type, loom_type_t mask_type, loom_value_id_t low_soffset,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;
  loom_amdgpu_fragment_memory_packet_plan_t report_packet = *packet;
  loom_amdgpu_fragment_memory_address_t address;
  const loom_amdgpu_fp8_decode_plan_t* decode_plan = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_decode_plan(
      context, plan->view_element_type, &decode_plan));
  const loom_scalar_type_t result_element_type =
      loom_amdgpu_fragment_memory_load_fp8_result_element_type(
          plan->payload_form);
  const bool result_is_f16 = result_element_type == LOOM_SCALAR_TYPE_F16;
  const loom_low_lower_resolved_descriptor_t* scalef32_descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_scalef32_descriptor(
      context, plan->view_element_type, result_element_type,
      &scalef32_descriptor));
  const loom_low_lower_resolved_descriptor_t* e8m0_pk8_descriptor = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_e8m0_pk8_descriptor(
      context, plan->view_element_type, result_element_type,
      &e8m0_pk8_descriptor));
  const loom_amdgpu_fragment_memory_packet_flags_t e8m0_pk8_flag =
      result_is_f16
          ? LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_IDENTITY_E8M0_PK8_F16
          : LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_IDENTITY_E8M0_PK8_BF16;
  const loom_amdgpu_fragment_memory_packet_flags_t scalef32_flag =
      result_is_f16
          ? LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_IDENTITY_SCALEF32_F16_PAIR
          : LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_FP8_IDENTITY_SCALEF32_BF16_PAIR;
  const bool use_identity_e8m0_pk8_descriptor =
      e8m0_pk8_descriptor != NULL &&
      iree_any_bit_set(report_packet.flags, e8m0_pk8_flag);
  const bool use_identity_scalef32_descriptor =
      scalef32_descriptor != NULL &&
      iree_any_bit_set(report_packet.flags, scalef32_flag);
  const loom_amdgpu_fp8_native_descriptors_t* native_f16_descriptors = NULL;
  if (result_is_f16) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_native_descriptors(
        context, plan->view_element_type, LOOM_SCALAR_TYPE_F16,
        &native_f16_descriptors));
  }
  const bool has_native_f16_pair =
      native_f16_descriptors != NULL &&
      iree_any_bit_set(native_f16_descriptors->flags,
                       LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR);
  const loom_amdgpu_fp8_native_descriptors_t* native_f32_descriptors = NULL;
  if (!result_is_f16) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_get_fp8_native_descriptors(
        context, plan->view_element_type, LOOM_SCALAR_TYPE_F32,
        &native_f32_descriptors));
  }
  const bool has_native_f32_pair =
      native_f32_descriptors != NULL &&
      iree_any_bit_set(native_f32_descriptors->flags,
                       LOOM_AMDGPU_FP8_NATIVE_DESCRIPTOR_FLAG_HAS_PAIR);
  loom_amdgpu_fp8_decode_value_flags_t decode_value_flags =
      LOOM_AMDGPU_FP8_DECODE_VALUE_FLAG_NONE;
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  if (fact_table != NULL) {
    const loom_value_facts_t source_facts = loom_value_fact_table_lookup(
        fact_table, loom_vector_fragment_load_result(source_op));
    decode_value_flags =
        loom_amdgpu_fp8_decode_value_flags_from_facts(source_facts);
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
      context, source_op, layout, plan, report_packet.register_index,
      /*element_index=*/0, report_packet.descriptor_ref, base_accumulator,
      low_address_resource, vgpr_type, &address));

  loom_type_t packet_type = vgpr_type;
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_type(
      context, report_packet.packet_register_count, vgpr_type, &packet_type));
  loom_value_id_t low_source_packet = LOOM_VALUE_ID_INVALID;
  const uint32_t vector_lane_count =
      (uint32_t)report_packet.result_register_count *
      LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_load_packet(
      context, source_op, layout, plan, &report_packet, /*element_index=*/0,
      vector_lane_count, packet_type, &address, low_packet_resource,
      low_soffset, &low_source_packet));
  if (report_packet.result_register_count == 1) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_low_subword_load_packet(
            context, source_op, low_source_packet, vgpr_type,
            &low_source_packet));
  }

  loom_type_t sgpr_type = loom_type_none();
  loom_type_t native_f32_pair_type = loom_type_none();
  loom_amdgpu_bf16_pack_descriptors_t bf16_pack_descriptors = {0};
  const bool prefer_packed_bf16 =
      !result_is_f16 && loom_amdgpu_fp8_prefers_packed_bf16_pair_decode(
                            decode_plan, decode_value_flags);
  const bool prefer_packed_f16 =
      result_is_f16 && !has_native_f16_pair &&
      loom_amdgpu_can_emit_fp8_pair_to_packed_f16_finite(decode_plan,
                                                         decode_value_flags);
  if (!use_identity_scalef32_descriptor && !use_identity_e8m0_pk8_descriptor) {
    if (prefer_packed_bf16 || prefer_packed_f16 ||
        (result_is_f16 && !has_native_f16_pair)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
    } else if (!result_is_f16 && has_native_f32_pair) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_make_vgpr_range_type(context, 2, &native_f32_pair_type));
      bf16_pack_descriptors = (loom_amdgpu_bf16_pack_descriptors_t){
          .flags =
              (iree_any_bit_set(
                   decode_plan->flags,
                   LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_NATIVE_BF16_PACK)
                   ? LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE
                   : LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_NONE) |
              (iree_any_bit_set(decode_plan->flags,
                                LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_PACK_U16)
                   ? LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_PACK_U16
                   : LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_NONE) |
              (iree_any_bit_set(
                   decode_plan->flags,
                   LOOM_AMDGPU_FP8_DECODE_PLAN_FLAG_HAS_ADD3_SRC2_LITERAL)
                   ? LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_ADD3_SRC2_LITERAL
                   : LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_NONE),
          .native_descriptor = decode_plan->native_bf16_pack_descriptor,
          .pack_u16_descriptor = decode_plan->pack_u16_descriptor,
          .add3_src2_literal_descriptor =
              decode_plan->add3_src2_literal_descriptor,
      };
    } else if (!result_is_f16) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
    }
  }
  loom_value_id_t low_identity_scalef32_scale = LOOM_VALUE_ID_INVALID;
  if (use_identity_scalef32_descriptor) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        LOOM_AMDGPU_FP8_F32_IDENTITY_SCALE_BITS, vgpr_type,
        &low_identity_scalef32_scale));
  }
  loom_value_id_t low_identity_e8m0_scale = LOOM_VALUE_ID_INVALID;
  if (use_identity_e8m0_pk8_descriptor) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MOV_B32,
        LOOM_AMDGPU_FP8_E8M0FNU_PACKED_IDENTITY_SCALE_BITS, vgpr_type,
        &low_identity_e8m0_scale));
  }

  loom_value_id_t low_result_registers[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS];
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(low_result_registers); ++i) {
    low_result_registers[i] = LOOM_VALUE_ID_INVALID;
  }
  if (use_identity_e8m0_pk8_descriptor) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_identity_e8m0_pk8_fp8_to_16bit_packet(
            context, source_op, low_source_packet,
            report_packet.packet_register_count,
            report_packet.result_register_count, e8m0_pk8_descriptor,
            low_identity_e8m0_scale, vgpr_type, low_result_registers));
  }
  if (!use_identity_scalef32_descriptor && !use_identity_e8m0_pk8_descriptor &&
      prefer_packed_bf16) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_try_emit_fragment_memory_fp8_to_packed_bf16_packet(
            context, source_op, low_source_packet,
            report_packet.packet_register_count,
            report_packet.result_register_count, decode_plan,
            decode_value_flags, vgpr_type, sgpr_type, mask_type,
            low_result_registers));
  }
  if (!use_identity_scalef32_descriptor && !use_identity_e8m0_pk8_descriptor &&
      prefer_packed_f16) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_try_emit_fragment_memory_fp8_to_packed_f16_packet(
            context, source_op, low_source_packet,
            report_packet.packet_register_count,
            report_packet.result_register_count, decode_plan,
            decode_value_flags, vgpr_type, sgpr_type, mask_type,
            low_result_registers));
  }
  if (low_result_registers[0] == LOOM_VALUE_ID_INVALID) {
    for (uint16_t result_register_index = 0;
         result_register_index < report_packet.result_register_count;
         ++result_register_index) {
      if (result_is_f16) {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_memory_fp8_to_packed_f16_register(
                context, source_op, low_source_packet,
                report_packet.packet_register_count, result_register_index,
                decode_plan, decode_value_flags,
                use_identity_scalef32_descriptor ? scalef32_descriptor : NULL,
                low_identity_scalef32_scale, native_f16_descriptors, vgpr_type,
                sgpr_type, mask_type,
                &low_result_registers[result_register_index]));
      } else {
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_memory_fp8_to_packed_bf16_register(
                context, source_op, low_source_packet,
                report_packet.packet_register_count, result_register_index,
                decode_plan, decode_value_flags,
                use_identity_scalef32_descriptor ? scalef32_descriptor : NULL,
                low_identity_scalef32_scale, native_f32_pair_type,
                native_f32_descriptors, &bf16_pack_descriptors, vgpr_type,
                sgpr_type, mask_type,
                &low_result_registers[result_register_index]));
      }
    }
  }
  if (report_packet.result_register_count == 1) {
    *out_low_packet = low_result_registers[0];
    return iree_ok_status();
  }

  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_type(
      context, report_packet.result_register_count, vgpr_type, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_result_registers,
      report_packet.result_register_count, result_type, source_op->location,
      &concat_op));
  *out_low_packet = loom_low_concat_result(concat_op);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_packed_b16_load_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    const loom_amdgpu_fragment_memory_address_accumulator_t* base_accumulator,
    loom_value_id_t low_address_resource, loom_value_id_t low_packet_resource,
    loom_type_t vgpr_type, loom_value_id_t low_soffset,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_elements[LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT] =
      {0};
  for (uint16_t element_index = 0;
       element_index < LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
       ++element_index) {
    loom_amdgpu_fragment_memory_address_t address;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, layout, plan, packet->register_index, element_index,
        packet->descriptor_ref, base_accumulator, low_address_resource,
        vgpr_type, &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_load_packet(
        context, source_op, layout, plan, packet, element_index,
        /*vector_lane_count=*/1, vgpr_type, &address, low_packet_resource,
        low_soffset, &low_elements[element_index]));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_low_subword_load_packet(
            context, source_op, low_elements[element_index], vgpr_type,
            &low_elements[element_index]));
  }

  loom_value_id_t high_element = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
      low_elements[1], vgpr_type, &high_element));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_elements[0],
      high_element, vgpr_type, out_low_packet);
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_packed_16bit_result_load_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    const loom_amdgpu_fragment_memory_address_accumulator_t* base_accumulator,
    loom_value_id_t low_address_resource, loom_value_id_t low_packet_resource,
    loom_type_t vgpr_type, loom_value_id_t low_soffset,
    loom_value_id_t* out_low_packet) {
  *out_low_packet = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_elements[LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT] =
      {0};
  for (uint16_t i = 0; i < packet->result_register_count; ++i) {
    const loom_amdgpu_fragment_memory_packet_plan_t element_packet = {
        .register_index = (uint16_t)(packet->register_index + i),
        .result_register_count = 1,
        .packet_register_count = packet->packet_register_count,
        .descriptor_ref = packet->descriptor_ref,
    };
    loom_amdgpu_fragment_memory_address_t address;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, layout, plan, element_packet.register_index,
        /*element_index=*/0, packet->descriptor_ref, base_accumulator,
        low_address_resource, vgpr_type, &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_load_packet(
        context, source_op, layout, plan, &element_packet, /*element_index=*/0,
        /*vector_lane_count=*/1, vgpr_type, &address, low_packet_resource,
        low_soffset, &low_elements[i]));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_low_subword_load_packet(
            context, source_op, low_elements[i], vgpr_type, &low_elements[i]));
  }
  if (packet->result_register_count == 1) {
    *out_low_packet = low_elements[0];
    return iree_ok_status();
  }

  loom_value_id_t high_element = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHLREV_B32_LIT, 16,
      low_elements[1], vgpr_type, &high_element));
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_OR_B32, low_elements[0],
      high_element, vgpr_type, out_low_packet);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_packed_b16_store_element(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_payload_register, uint16_t element_index,
    loom_type_t vgpr_type, loom_value_id_t* out_low_element) {
  *out_low_element = LOOM_VALUE_ID_INVALID;
  if (element_index == 0) {
    return loom_amdgpu_materialize_low_vgpr_b32(
        context, source_op, low_payload_register, out_low_element);
  }
  if (element_index == 1) {
    return loom_amdgpu_emit_vgpr_shift(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 16,
        low_payload_register, vgpr_type, out_low_element);
  }
  IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment packed b16 element");
  IREE_BUILTIN_UNREACHABLE();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_saveexec(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_condition, loom_type_t mask_type,
    loom_value_id_t* out_saved_exec) {
  *out_saved_exec = LOOM_VALUE_ID_INVALID;
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  const loom_type_t result_types[] = {mask_type, scc_type};
  loom_op_t* saveexec_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
      &low_condition, 1, loom_named_attr_slice_empty(), result_types,
      IREE_ARRAYSIZE(result_types), &saveexec_op));
  *out_saved_exec = loom_value_slice_get(loom_low_op_results(saveexec_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_restore_exec(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_saved_exec) {
  loom_op_t* low_op = NULL;
  return loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &low_saved_exec, 1, loom_named_attr_slice_empty(),
      /*result_types=*/NULL, /*result_count=*/0, &low_op);
}

static iree_status_t loom_amdgpu_emit_fragment_store_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    uint16_t element_index, uint32_t vector_lane_count,
    const loom_amdgpu_fragment_memory_address_t* address,
    loom_value_id_t low_payload_register, loom_value_id_t low_resource,
    loom_value_id_t low_soffset) {
  loom_named_attr_t attrs[1] = {0};
  iree_host_size_t attr_count = 0;
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_fragment_memory_attrs(
      context, attrs, IREE_ARRAYSIZE(attrs), address->immediate_offset,
      &attr_count));

  loom_low_lower_resolved_descriptor_t descriptor = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
      context, packet->descriptor_ref, &descriptor));
  loom_value_id_t low_m0 = LOOM_VALUE_ID_INVALID;
  if (loom_amdgpu_descriptor_has_implicit_resource_operand(
          loom_low_lower_context_descriptor_set(context),
          descriptor.descriptor)) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_m0_u32(context, source_op, &descriptor, 0, &low_m0));
  }

  loom_value_id_t operands[5] = {0};
  iree_host_size_t operand_count = 0;
  if (loom_amdgpu_fragment_memory_uses_buffer_descriptor(plan)) {
    operands[operand_count++] = low_payload_register;
    operands[operand_count++] = low_resource;
    operands[operand_count++] = address->low_vaddr;
    operands[operand_count++] = low_soffset;
  } else {
    operands[operand_count++] = address->low_vaddr;
    operands[operand_count++] = low_payload_register;
    if (plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
      operands[operand_count++] = low_resource;
    }
  }
  if (low_m0 != LOOM_VALUE_ID_INVALID) {
    operands[operand_count++] = low_m0;
  }
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_emit_resolved_descriptor_op(
      context, &descriptor, operands, operand_count,
      loom_make_named_attr_slice(attrs, attr_count), /*result_types=*/NULL,
      /*result_count=*/0, /*tied_results=*/NULL, /*tied_result_count=*/0,
      source_op->location, &low_op));
  return loom_amdgpu_record_fragment_memory_packet(
      context, source_op, low_op, layout, plan, packet, element_index,
      vector_lane_count);
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_crosslane_packed_b16_pair_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    const loom_low_lower_resolved_descriptor_t* crosslane_descriptor,
    loom_value_id_t low_paired_lane_byte_offset, loom_value_id_t low_source,
    loom_type_t vgpr_type, loom_value_id_t* out_paired_source) {
  *out_paired_source = LOOM_VALUE_ID_INVALID;
  if (iree_all_bits_set(
          packet->flags,
          LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE_DPP)) {
    return loom_amdgpu_emit_subgroup_dpp_register(
        context, source_op, crosslane_descriptor, low_source,
        LOOM_AMDGPU_DPP_CTRL_QUAD_SWAP_1, vgpr_type, out_paired_source);
  }
  return loom_amdgpu_emit_subgroup_bpermute_register(
      context, source_op, crosslane_descriptor, low_paired_lane_byte_offset,
      /*static_byte_offset=*/0, low_source, vgpr_type, out_paired_source);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_apply_f32_scale(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, loom_value_id_t low_scale,
    loom_type_t vgpr_type, loom_value_id_t* out_scaled) {
  *out_scaled = low_source;
  if (low_scale == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  return loom_amdgpu_emit_vgpr_binary(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_MUL_F32, low_source,
      low_scale, vgpr_type, out_scaled);
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_crosslane_packed_b16_prepare_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    const loom_low_lower_resolved_descriptor_t* crosslane_descriptor,
    loom_value_id_t low_paired_lane_byte_offset, loom_value_id_t low_payload,
    const loom_amdgpu_bf16_pack_descriptors_t* pre_narrow_bf16_descriptors,
    loom_value_id_t low_scale, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_pending_store_t* out_pending_store) {
  *out_pending_store = (loom_amdgpu_fragment_memory_pending_store_t){
      .packet = *packet,
      .low_source_register = LOOM_VALUE_ID_INVALID,
      .low_paired_source_register = LOOM_VALUE_ID_INVALID,
  };
  loom_value_id_t low_source_register = LOOM_VALUE_ID_INVALID;
  const bool source_is_packed =
      plan->narrowed_result_packed_source != LOOM_VALUE_ID_INVALID;
  if (source_is_packed) {
    loom_value_id_t low_packed_register = low_payload;
    if (plan->payload_register_count != 1) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_payload, packet->register_index / 2u,
          vgpr_type, &low_packed_register));
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_packed_b16_store_element(
            context, source_op, low_packed_register,
            packet->register_index & 1u, vgpr_type, &low_source_register));
  } else {
    low_source_register = low_payload;
    if (plan->register_count != 1) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_payload, packet->register_index, vgpr_type,
          &low_source_register));
    }
  }
  if (pre_narrow_bf16_descriptors != NULL) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_apply_f32_scale(
        context, source_op, low_source_register, low_scale, vgpr_type,
        &low_source_register));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_f32_to_bf16_lane_with_descriptors(
        context, source_op, pre_narrow_bf16_descriptors, low_source_register,
        vgpr_type, &low_source_register));
  }
  out_pending_store->low_source_register = low_source_register;

  return loom_amdgpu_emit_fragment_memory_crosslane_packed_b16_pair_register(
      context, source_op, packet, crosslane_descriptor,
      low_paired_lane_byte_offset, low_source_register, vgpr_type,
      &out_pending_store->low_paired_source_register);
}

static iree_status_t
loom_amdgpu_emit_fragment_memory_flush_crosslane_packed_b16_stores(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors,
    const loom_amdgpu_fragment_memory_address_accumulator_t* base_accumulator,
    const loom_amdgpu_fragment_memory_pending_store_t* pending_stores,
    iree_host_size_t pending_store_count,
    loom_amdgpu_fragment_memory_pending_store_payload_form_t payload_form,
    loom_value_id_t low_even_lane_mask, loom_type_t vgpr_type,
    loom_type_t mask_type, loom_value_id_t low_address_resource,
    loom_value_id_t low_packet_resource, loom_value_id_t low_scale,
    loom_value_id_t low_paired_scale, loom_value_id_t low_soffset) {
  if (pending_store_count == 0) {
    return iree_ok_status();
  }
  const loom_low_lower_resolved_descriptor_t* pack_u16_descriptor =
      iree_any_bit_set(bf16_pack_descriptors->flags,
                       LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_PACK_U16)
          ? &bf16_pack_descriptors->pack_u16_descriptor
          : NULL;

  loom_value_id_t low_saved_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_saveexec(
      context, source_op, low_even_lane_mask, mask_type, &low_saved_exec));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < pending_store_count && iree_status_is_ok(status); ++i) {
    const loom_amdgpu_fragment_memory_pending_store_t* pending_store =
        &pending_stores[i];
    loom_value_id_t low_source_register = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_paired_source_register = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_payload_packet = LOOM_VALUE_ID_INVALID;
    switch (payload_form) {
      case LOOM_AMDGPU_FRAGMENT_MEMORY_PENDING_STORE_PAYLOAD_FORM_F32:
        status = loom_amdgpu_emit_fragment_memory_apply_f32_scale(
            context, source_op, pending_store->low_source_register, low_scale,
            vgpr_type, &low_source_register);
        if (!iree_status_is_ok(status)) {
          break;
        }
        status = loom_amdgpu_emit_fragment_memory_apply_f32_scale(
            context, source_op, pending_store->low_paired_source_register,
            low_paired_scale, vgpr_type, &low_paired_source_register);
        if (!iree_status_is_ok(status)) {
          break;
        }
        status = loom_amdgpu_emit_f32_pair_to_packed_bf16_with_descriptors(
            context, source_op, bf16_pack_descriptors, low_source_register,
            low_paired_source_register, vgpr_type, &low_payload_packet);
        break;
      case LOOM_AMDGPU_FRAGMENT_MEMORY_PENDING_STORE_PAYLOAD_FORM_BF16:
        status = loom_amdgpu_emit_packed_bf16_lane_pair(
            context, source_op, pack_u16_descriptor,
            pending_store->low_source_register,
            pending_store->low_paired_source_register, vgpr_type,
            &low_payload_packet);
        break;
      default:
        IREE_ASSERT_UNREACHABLE(
            "selected AMDGPU fragment pending store payload form");
        IREE_BUILTIN_UNREACHABLE();
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    loom_amdgpu_fragment_memory_address_t address;
    status = loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, layout, plan, pending_store->packet.register_index,
        /*element_index=*/0, pending_store->packet.descriptor_ref,
        base_accumulator, low_address_resource, vgpr_type, &address);
    if (!iree_status_is_ok(status)) {
      break;
    }
    status = loom_amdgpu_emit_fragment_store_packet(
        context, source_op, layout, plan, &pending_store->packet,
        /*element_index=*/0, LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT,
        &address, low_payload_packet, low_packet_resource, low_soffset);
  }
  return iree_status_join(status, loom_amdgpu_emit_fragment_memory_restore_exec(
                                      context, source_op, low_saved_exec));
}

static iree_status_t loom_amdgpu_emit_fragment_memory_packed_b16_store_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet,
    const loom_amdgpu_fragment_memory_address_accumulator_t* base_accumulator,
    loom_value_id_t low_payload, loom_value_id_t low_address_resource,
    loom_value_id_t low_packet_resource, loom_type_t vgpr_type,
    loom_value_id_t low_soffset) {
  loom_value_id_t low_payload_register = low_payload;
  if (plan->register_count != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_payload, packet->register_index, vgpr_type,
        &low_payload_register));
  }
  for (uint16_t element_index = 0;
       element_index < LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT;
       ++element_index) {
    loom_value_id_t low_element = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_packed_b16_store_element(
            context, source_op, low_payload_register, element_index, vgpr_type,
            &low_element));
    loom_amdgpu_fragment_memory_address_t address;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, layout, plan, packet->register_index, element_index,
        packet->descriptor_ref, base_accumulator, low_address_resource,
        vgpr_type, &address));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_store_packet(
        context, source_op, layout, plan, packet, element_index,
        /*vector_lane_count=*/1, &address, low_element, low_packet_resource,
        low_soffset));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_lower_vector_fragment_load(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(plan->layout_kind);
  if (layout == NULL) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment memory layout");
    IREE_BUILTIN_UNREACHABLE();
  }
  IREE_ASSERT_GT(plan->packet_count, 0u);

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  uint16_t lane_divisor = 0;
  if (!loom_amdgpu_fragment_memory_lane_divisor(layout, plan, &lane_divisor)) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment lane divisor");
    IREE_BUILTIN_UNREACHABLE();
  }
  loom_amdgpu_fragment_lane_ids_t lane_ids;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_lane_ids(
      context, source_op, lane_divisor, vgpr_type, &lane_ids));
  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, plan->role);
  const bool low_subword =
      loom_amdgpu_fragment_memory_role_uses_low_subword(role_layout);
  const bool packed_b16_elements =
      loom_amdgpu_fragment_memory_role_uses_packed_b16_elements(role_layout);
  const bool load_packed_16bit_result =
      plan->payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT;
  const bool load_fp8_to_16bit =
      loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
          plan->payload_form);
  loom_type_t mask_type = loom_type_none();
  if (load_fp8_to_16bit) {
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
  }

  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_packet_resource = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  if (plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context,
        loom_low_source_memory_access_base_view_value_id(&plan->source),
        &low_resource));
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_resource(
        context, source_op, plan, low_resource, &low_packet_resource,
        &low_soffset));
  }
  loom_amdgpu_fragment_memory_address_accumulator_t base_accumulator;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fragment_memory_base_address_accumulator(
          context, source_op, layout, plan, &lane_ids, lane_divisor, vgpr_type,
          &base_accumulator));

  loom_value_id_t low_packets[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  for (uint16_t packet_index = 0; packet_index < plan->packet_count;
       ++packet_index) {
    const loom_amdgpu_fragment_memory_packet_plan_t* packet =
        &plan->packets[packet_index];
    if (load_fp8_to_16bit) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_fp8_to_packed_16bit_load_packet(
              context, source_op, layout, plan, packet, &base_accumulator,
              low_resource, low_packet_resource, vgpr_type, mask_type,
              low_soffset, &low_packets[packet_index]));
      continue;
    }
    if (load_packed_16bit_result) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_packed_16bit_result_load_packet(
              context, source_op, layout, plan, packet, &base_accumulator,
              low_resource, low_packet_resource, vgpr_type, low_soffset,
              &low_packets[packet_index]));
      continue;
    }
    if (packed_b16_elements) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_packed_b16_load_packet(
              context, source_op, layout, plan, packet, &base_accumulator,
              low_resource, low_packet_resource, vgpr_type, low_soffset,
              &low_packets[packet_index]));
      continue;
    }
    loom_type_t packet_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_type(
        context, packet->packet_register_count, vgpr_type, &packet_type));
    loom_amdgpu_fragment_memory_address_t address;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, layout, plan, packet->register_index,
        /*element_index=*/0, packet->descriptor_ref, &base_accumulator,
        low_resource, vgpr_type, &address));
    const uint32_t vector_lane_count =
        low_subword ? 1
                    : (uint32_t)packet->result_register_count *
                          plan->elements_per_register;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_load_packet(
        context, source_op, layout, plan, packet, /*element_index=*/0,
        vector_lane_count, packet_type, &address, low_packet_resource,
        low_soffset, &low_packets[packet_index]));
    if (low_subword) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_low_subword_load_packet(
              context, source_op, low_packets[packet_index], vgpr_type,
              &low_packets[packet_index]));
    }
  }

  if (plan->packet_count == 1) {
    return loom_low_lower_bind_value(context, plan->payload, low_packets[0]);
  }
  loom_type_t result_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_range_type(
      context, plan->payload_register_count, &result_type));
  loom_op_t* concat_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_concat_build(
      loom_low_lower_context_builder(context), low_packets, plan->packet_count,
      result_type, source_op->location, &concat_op));
  return loom_low_lower_bind_value(context, plan->payload,
                                   loom_low_concat_result(concat_op));
}

iree_status_t loom_amdgpu_lower_vector_fragment_store(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(plan->layout_kind);
  if (layout == NULL) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment memory layout");
    IREE_BUILTIN_UNREACHABLE();
  }
  IREE_ASSERT_GT(plan->packet_count, 0u);

  loom_type_t vgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_vgpr_type(context, &vgpr_type));
  uint16_t lane_divisor = 0;
  if (!loom_amdgpu_fragment_memory_lane_divisor(layout, plan, &lane_divisor)) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment lane divisor");
    IREE_BUILTIN_UNREACHABLE();
  }
  loom_amdgpu_fragment_lane_ids_t lane_ids;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_lane_ids(
      context, source_op, lane_divisor, vgpr_type, &lane_ids));
  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, plan->role);
  const bool packed_b16_elements =
      loom_amdgpu_fragment_memory_role_uses_packed_b16_elements(role_layout);

  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_packet_resource = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  if (plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context,
        loom_low_source_memory_access_base_view_value_id(&plan->source),
        &low_resource));
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_resource(
        context, source_op, plan, low_resource, &low_packet_resource,
        &low_soffset));
  }
  loom_amdgpu_fragment_memory_address_accumulator_t base_accumulator;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fragment_memory_base_address_accumulator(
          context, source_op, layout, plan, &lane_ids, lane_divisor, vgpr_type,
          &base_accumulator));

  if (plan->payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16) {
    loom_type_t mask_type = loom_type_none();
    const loom_amdgpu_bf16_pack_descriptors_t* bf16_pack_descriptors = NULL;
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_get_bf16_pack_descriptors(context, &bf16_pack_descriptors));
    const bool has_crosslane_packed_b16_store =
        loom_amdgpu_fragment_memory_epilogue_strategy_is_crosslane_packed_b16(
            plan->epilogue_strategy);
    const bool has_dpp_crosslane_packed_b16_store =
        loom_amdgpu_fragment_memory_epilogue_strategy_uses_dpp(
            plan->epilogue_strategy);
    const bool pre_narrow_crosslane_sources =
        has_crosslane_packed_b16_store &&
        plan->narrowed_result_round_source != LOOM_VALUE_ID_INVALID &&
        !iree_any_bit_set(bf16_pack_descriptors->flags,
                          LOOM_AMDGPU_BF16_PACK_DESCRIPTOR_FLAG_HAS_NATIVE);
    const bool copy_packed_crosslane_sources =
        has_crosslane_packed_b16_store &&
        plan->narrowed_result_packed_source != LOOM_VALUE_ID_INVALID;
    const loom_amdgpu_bf16_pack_descriptors_t* pre_narrow_bf16_descriptors =
        pre_narrow_crosslane_sources ? bf16_pack_descriptors : NULL;
    const loom_amdgpu_fragment_memory_pending_store_payload_form_t
        pending_store_payload_form =
            pre_narrow_crosslane_sources || copy_packed_crosslane_sources
                ? LOOM_AMDGPU_FRAGMENT_MEMORY_PENDING_STORE_PAYLOAD_FORM_BF16
                : LOOM_AMDGPU_FRAGMENT_MEMORY_PENDING_STORE_PAYLOAD_FORM_F32;
    if (has_crosslane_packed_b16_store) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_make_sgpr_range_type(context, 2, &mask_type));
    }
    loom_value_id_t low_paired_lane_byte_offset = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_even_lane_mask = LOOM_VALUE_ID_INVALID;
    loom_low_lower_resolved_descriptor_t crosslane_descriptor = {0};
    if (has_crosslane_packed_b16_store) {
      if (has_dpp_crosslane_packed_b16_store) {
        const loom_low_descriptor_set_t* descriptor_set =
            loom_low_lower_context_descriptor_set(context);
        const loom_amdgpu_descriptor_ref_t dpp_ref =
            loom_amdgpu_fragment_memory_crosslane_packed_b16_store_dpp_ref(
                descriptor_set);
        IREE_ASSERT_NE(dpp_ref, LOOM_AMDGPU_DESCRIPTOR_REF_NONE);
        IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
            context, dpp_ref, &crosslane_descriptor));
      } else {
        IREE_RETURN_IF_ERROR(loom_amdgpu_resolve_descriptor_ref(
            context, LOOM_AMDGPU_DESCRIPTOR_REF_DS_BPERMUTE_B32,
            &crosslane_descriptor));
        loom_value_id_t low_paired_lane = LOOM_VALUE_ID_INVALID;
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_XOR_B32_LIT,
            lane_ids.lane, 1, vgpr_type, &low_paired_lane));
        IREE_RETURN_IF_ERROR(loom_amdgpu_emit_subgroup_lane_byte_offset(
            context, source_op, low_paired_lane, vgpr_type,
            &low_paired_lane_byte_offset));
      }
      loom_value_id_t low_lane_mod_low_bit = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_ensure_fragment_memory_lane_mod(
          context, source_op, lane_divisor, vgpr_type, &lane_ids));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
          lane_ids.lane_mod, 1, vgpr_type, &low_lane_mod_low_bit));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_cmp_u32_lit(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CMP_EQ_I32,
          low_lane_mod_low_bit, 0, vgpr_type, mask_type, &low_even_lane_mask));
    }

    loom_value_id_t low_payload = LOOM_VALUE_ID_INVALID;
    if (plan->narrowed_result_packed_source != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, plan->narrowed_result_packed_source, &low_payload));
    } else if (plan->narrowed_result_round_source != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, plan->narrowed_result_round_source, &low_payload));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->payload, &low_payload));
    }
    loom_value_id_t low_scale = LOOM_VALUE_ID_INVALID;
    loom_value_id_t low_paired_scale = LOOM_VALUE_ID_INVALID;
    if (plan->narrowed_result_scale_source != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, plan->narrowed_result_scale_source, &low_scale));
      IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_full_low_vgpr_b32(
          context, source_op, low_scale, &low_scale));
      low_paired_scale = low_scale;
    }

    loom_amdgpu_fragment_memory_pending_store_t
        pending_stores[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
    iree_host_size_t pending_store_count = 0;
    for (uint16_t packet_index = 0; packet_index < plan->packet_count;
         ++packet_index) {
      const loom_amdgpu_fragment_memory_packet_plan_t* packet =
          &plan->packets[packet_index];
      if (iree_all_bits_set(
              packet->flags,
              LOOM_AMDGPU_FRAGMENT_MEMORY_PACKET_FLAG_CROSSLANE_PACKED_B16_STORE)) {
        IREE_ASSERT_LT(pending_store_count, IREE_ARRAYSIZE(pending_stores));
        IREE_RETURN_IF_ERROR(
            loom_amdgpu_emit_fragment_memory_crosslane_packed_b16_prepare_store(
                context, source_op, plan, packet, &crosslane_descriptor,
                low_paired_lane_byte_offset, low_payload,
                pre_narrow_bf16_descriptors, low_scale, vgpr_type,
                &pending_stores[pending_store_count++]));
        continue;
      }

      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_flush_crosslane_packed_b16_stores(
              context, source_op, layout, plan, bf16_pack_descriptors,
              &base_accumulator, pending_stores, pending_store_count,
              pending_store_payload_form, low_even_lane_mask, vgpr_type,
              mask_type, low_resource, low_packet_resource, low_scale,
              low_paired_scale, low_soffset));
      pending_store_count = 0;

      loom_value_id_t low_payload_packet = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_packed_16bit_packet(
          context, source_op, plan, low_payload, bf16_pack_descriptors,
          packet->register_index, packet->result_register_count,
          packet->packet_register_count, low_scale, vgpr_type,
          &low_payload_packet));
      loom_amdgpu_fragment_memory_address_t address;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
          context, source_op, layout, plan, packet->register_index,
          /*element_index=*/0, packet->descriptor_ref, &base_accumulator,
          low_resource, vgpr_type, &address));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_store_packet(
          context, source_op, layout, plan, packet, /*element_index=*/0,
          packet->result_register_count, &address, low_payload_packet,
          low_packet_resource, low_soffset));
    }
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_fragment_memory_flush_crosslane_packed_b16_stores(
            context, source_op, layout, plan, bf16_pack_descriptors,
            &base_accumulator, pending_stores, pending_store_count,
            pending_store_payload_form, low_even_lane_mask, vgpr_type,
            mask_type, low_resource, low_packet_resource, low_scale,
            low_paired_scale, low_soffset));
    return iree_ok_status();
  }

  loom_value_id_t low_payload = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_lookup_value(context, plan->payload, &low_payload));

  if (packed_b16_elements) {
    for (uint16_t packet_index = 0; packet_index < plan->packet_count;
         ++packet_index) {
      const loom_amdgpu_fragment_memory_packet_plan_t* packet =
          &plan->packets[packet_index];
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_fragment_memory_packed_b16_store_packet(
              context, source_op, layout, plan, packet, &base_accumulator,
              low_payload, low_resource, low_packet_resource, vgpr_type,
              low_soffset));
    }
    return iree_ok_status();
  }

  for (uint16_t packet_index = 0; packet_index < plan->packet_count;
       ++packet_index) {
    const loom_amdgpu_fragment_memory_packet_plan_t* packet =
        &plan->packets[packet_index];
    loom_value_id_t low_payload_packet = low_payload;
    if (plan->payload_form ==
        LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_EXTEND_F16_TO_F32) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_f16_to_f32_packet(
          context, source_op, plan, low_payload, packet, vgpr_type,
          &low_payload_packet));
    } else if (packet->result_register_count != plan->register_count) {
      loom_type_t packet_type = loom_type_none();
      IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_type(
          context, packet->packet_register_count, vgpr_type, &packet_type));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
          context, source_op, low_payload, packet->register_index, packet_type,
          &low_payload_packet));
    }
    loom_amdgpu_fragment_memory_address_t address;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_vaddr(
        context, source_op, layout, plan, packet->register_index,
        /*element_index=*/0, packet->descriptor_ref, &base_accumulator,
        low_resource, vgpr_type, &address));
    const uint32_t vector_lane_count =
        (uint32_t)packet->result_register_count * plan->elements_per_register;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_store_packet(
        context, source_op, layout, plan, packet, /*element_index=*/0,
        vector_lane_count, &address, low_payload_packet, low_packet_resource,
        low_soffset));
  }
  return iree_ok_status();
}

void loom_amdgpu_mark_fragment_memory_plan_storage_demands(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan) {
  (void)source_op;
  if (plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    loom_low_lower_require_source_value_storage(
        context,
        loom_low_source_memory_access_base_view_value_id(&plan->source));
  }
  for (uint8_t i = 0; i < plan->source.dynamic_term_count; ++i) {
    if (loom_amdgpu_fragment_memory_uses_dynamic_view_base_value(plan, i)) {
      loom_low_lower_require_source_value_storage(
          context, plan->source.dynamic_view_base_value_id);
      continue;
    }
    const loom_low_source_memory_dynamic_term_t* term =
        &plan->source.dynamic_terms[i];
    loom_low_lower_require_source_value_storage(context, term->index);
    for (uint8_t j = 0; j < term->stride_value_count; ++j) {
      loom_low_lower_require_source_value_storage(context,
                                                  term->stride_values[j]);
    }
  }

  switch (plan->operation_kind) {
    case LOOM_AMDGPU_MEMORY_OPERATION_LOAD:
      return;
    case LOOM_AMDGPU_MEMORY_OPERATION_STORE:
      if (plan->narrowed_result_packed_source != LOOM_VALUE_ID_INVALID) {
        loom_low_lower_require_source_value_storage(
            context, plan->narrowed_result_packed_source);
      } else if (plan->narrowed_result_round_source != LOOM_VALUE_ID_INVALID) {
        loom_low_lower_require_source_value_storage(
            context, plan->narrowed_result_round_source);
      } else {
        loom_low_lower_require_source_value_storage(context, plan->payload);
      }
      return;
    case LOOM_AMDGPU_MEMORY_OPERATION_COUNT_:
      break;
  }
  IREE_ASSERT_UNREACHABLE("unknown AMDGPU fragment memory operation kind");
}
