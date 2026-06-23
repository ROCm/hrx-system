// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment.h"

#include <stdint.h>

#include "loom/ir/attribute.h"
#include "loom/ir/context.h"
#include "loom/ir/scalar_type.h"
#include "loom/ir/types.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/kernel/ops.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/vector/fragment.h"
#include "loom/ops/vector/memory.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_id/target_id.h"
#include "loom/util/fact_table.h"

enum {
  LOOM_AMDGPU_FRAGMENT_VIEW_RANK = 2,
  LOOM_AMDGPU_FRAGMENT_LANE_MODULUS = 16,
  LOOM_AMDGPU_FRAGMENT_REGISTER_BYTE_COUNT = 4,
  LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_PACKET_REGISTERS = 4,
  LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_COUNT = 2,
  LOOM_AMDGPU_FRAGMENT_PACKED_B16_ELEMENT_BIT_COUNT = 16,
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

typedef struct loom_amdgpu_fragment_memory_descriptor_table_t {
  // Descriptor refs for normal 32-bit-register packet payloads, indexed by
  // operation kind and packet register count.
  loom_amdgpu_descriptor_ref_t
      packet_refs[LOOM_AMDGPU_MEMORY_OPERATION_COUNT_]
                 [LOOM_AMDGPU_FRAGMENT_MEMORY_MAX_PACKET_REGISTERS + 1u];
  // Descriptor ref for a scalar 16-bit load packet.
  loom_amdgpu_descriptor_ref_t load_b16_ref;
  // Descriptor ref for a scalar 16-bit store packet.
  loom_amdgpu_descriptor_ref_t store_b16_ref;
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
                .load_b16_ref =
                    LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_LOAD_B16_D16_SADDR,
                .store_b16_ref =
                    LOOM_AMDGPU_DESCRIPTOR_REF_GLOBAL_STORE_B16_SADDR,
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
                .load_b16_ref = LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_LOAD_B16_D16,
                .store_b16_ref = LOOM_AMDGPU_DESCRIPTOR_REF_BUFFER_STORE_B16,
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
                .load_b16_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_READ_U16,
                .store_b16_ref = LOOM_AMDGPU_DESCRIPTOR_REF_DS_WRITE_B16,
            },
};

typedef struct loom_amdgpu_fragment_memory_environment_t {
  // Source module being checked or lowered.
  const loom_module_t* module;
  // Source facts available for shape, view, and address reasoning.
  const loom_value_fact_table_t* fact_table;
  // Target bundle selected for this source-to-low attempt.
  const loom_target_bundle_t* bundle;
  // Low descriptor set selected by the target bundle.
  const loom_low_descriptor_set_t* descriptor_set;
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

typedef struct loom_amdgpu_fragment_lane_ids_t {
  // Full subgroup lane id.
  loom_value_id_t lane;
  // Lane id modulo the fragment row/column tile.
  loom_value_id_t lane_mod;
  // Lane id divided by the fragment row/column tile.
  loom_value_id_t lane_div;
} loom_amdgpu_fragment_lane_ids_t;

typedef struct loom_amdgpu_fragment_memory_address_t {
  // Low packet address operand after static offset immediates are split out.
  loom_value_id_t low_vaddr;
  // Encoded descriptor offset immediate value.
  int64_t immediate_offset;
} loom_amdgpu_fragment_memory_address_t;

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
  switch (role) {
    case LOOM_VECTOR_ROLE_LHS:
      *out_role = LOOM_CONTRACT_OPERAND_ROLE_LHS;
      return true;
    case LOOM_VECTOR_ROLE_RHS:
      *out_role = LOOM_CONTRACT_OPERAND_ROLE_RHS;
      return true;
    case LOOM_VECTOR_ROLE_INIT:
      *out_role = LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR;
      return true;
    case LOOM_VECTOR_ROLE_RESULT:
      *out_role = LOOM_CONTRACT_OPERAND_ROLE_RESULT;
      return true;
    case LOOM_VECTOR_ROLE_COUNT_:
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_source_operation_kind(
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_low_source_memory_operation_kind_t* out_operation_kind) {
  *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  switch (operation_kind) {
    case LOOM_AMDGPU_MEMORY_OPERATION_LOAD:
      *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
      return true;
    case LOOM_AMDGPU_MEMORY_OPERATION_STORE:
      *out_operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE;
      return true;
    case LOOM_AMDGPU_MEMORY_OPERATION_COUNT_:
      return false;
  }
  return false;
}

static bool loom_amdgpu_fragment_memory_exact_nonnegative_i64(
    const loom_value_fact_table_t* fact_table, loom_value_id_t value_id,
    int64_t* out_value) {
  return loom_amdgpu_value_facts_as_exact_non_negative_i64(
      loom_value_fact_table_lookup(fact_table, value_id), out_value);
}

static bool loom_amdgpu_fragment_memory_shape_matches(
    const loom_value_fact_table_t* fact_table,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role, loom_value_id_t rows,
    loom_value_id_t columns) {
  int64_t row_count = 0;
  int64_t column_count = 0;
  if (!loom_amdgpu_fragment_memory_exact_nonnegative_i64(fact_table, rows,
                                                         &row_count) ||
      !loom_amdgpu_fragment_memory_exact_nonnegative_i64(fact_table, columns,
                                                         &column_count)) {
    return false;
  }

  const loom_amdgpu_matrix_tile_shape_t shape = layout->tile_shape;
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
  return element_type == LOOM_SCALAR_TYPE_F16 ||
         element_type == LOOM_SCALAR_TYPE_BF16;
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
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_scalar_type_t expected_element_type,
    loom_scalar_type_t payload_element_type,
    loom_scalar_type_t view_element_type,
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
  if (element_bit_count != 16) {
    return false;
  }
  uint32_t payload_bit_count = 0;
  uint32_t register_count = 0;
  return loom_amdgpu_type_packed_16bit_float_storage(
             payload_type, &payload_bit_count, &register_count) &&
         register_count == role_layout->register_count &&
         payload_bit_count == (uint32_t)role_layout->register_count *
                                  role_layout->elements_per_register *
                                  (uint32_t)role_layout->element_bit_count;
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
  if (loom_type_element_type(payload_type) == LOOM_SCALAR_TYPE_F32) {
    const uint32_t register_count =
        loom_amdgpu_vector_f32_lane_count(payload_type);
    if (register_count == 0 || register_count > UINT16_MAX) {
      return false;
    }
    *out_register_count = (uint16_t)register_count;
    return true;
  }
  uint32_t unused_payload_bit_count = 0;
  uint32_t register_count = 0;
  if (!loom_amdgpu_type_packed_16bit_float_storage(
          payload_type, &unused_payload_bit_count, &register_count) ||
      register_count == 0 || register_count > UINT16_MAX) {
    return false;
  }
  *out_register_count = (uint16_t)register_count;
  return true;
}

static bool loom_amdgpu_fragment_memory_payload_matches(
    loom_type_t payload_type,
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
          payload_type, expected_element_type, role_layout);
    }
    case LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR:
    case LOOM_CONTRACT_OPERAND_ROLE_RESULT: {
      if (loom_amdgpu_fragment_memory_payload_matches_role_storage(
              payload_type, expected_element_type, role_layout)) {
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
              payload_element_type)) {
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
    case LOOM_AMDGPU_MATRIX_NUMERIC_F32:
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

static bool loom_amdgpu_fragment_memory_descriptor_role_element_type(
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor,
    loom_contract_operand_role_t role, loom_scalar_type_t* out_element_type) {
  loom_amdgpu_matrix_payload_shape_t payload = {0};
  return loom_amdgpu_fragment_memory_descriptor_payload(descriptor, role,
                                                        &payload) &&
         loom_amdgpu_fragment_memory_scalar_type_from_numeric(
             payload.numeric_type, out_element_type);
}

typedef enum loom_amdgpu_fragment_memory_element_match_e {
  LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE = 0,
  LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_ADAPTED = 1,
  LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_EXACT = 2,
} loom_amdgpu_fragment_memory_element_match_t;

static loom_amdgpu_fragment_memory_element_match_t
loom_amdgpu_fragment_memory_payload_element_match(
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_type_t payload_type,
    loom_type_t view_type, loom_scalar_type_t expected_element_type,
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
  const loom_scalar_type_t payload_element_type =
      loom_type_element_type(payload_type);
  const loom_scalar_type_t view_element_type =
      loom_type_element_type(view_type);
  if (!loom_amdgpu_fragment_memory_payload_form_select(
          operation_kind, role, expected_element_type, payload_element_type,
          view_element_type, out_payload_form)) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE;
  }
  return *out_payload_form == LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE
             ? LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_EXACT
             : LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_ADAPTED;
}

static bool loom_amdgpu_fragment_memory_target_layout(
    const loom_amdgpu_fragment_memory_environment_t* environment,
    loom_contract_operand_role_t role,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_type_t payload_type, loom_type_t view_type,
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

  const iree_host_size_t descriptor_count =
      loom_amdgpu_matrix_contract_descriptor_count();
  const uint32_t wave_size = environment->bundle->snapshot->subgroup_size;
  const loom_amdgpu_matrix_fragment_layout_t* best_layout = NULL;
  loom_scalar_type_t best_element_type = LOOM_SCALAR_TYPE_COUNT_;
  loom_amdgpu_fragment_memory_payload_form_t best_payload_form =
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
  loom_amdgpu_fragment_memory_element_match_t best_match =
      LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE;
  bool rejected_payload_layout = false;
  bool rejected_payload_form = false;
  for (iree_host_size_t i = 0; i < descriptor_count; ++i) {
    const loom_amdgpu_matrix_contract_descriptor_t* descriptor =
        loom_amdgpu_matrix_contract_descriptor_at(i);
    const loom_amdgpu_matrix_fragment_layout_t* layout =
        loom_amdgpu_matrix_contract_descriptor_fragment_layout(descriptor);
    if (layout == NULL ||
        !loom_amdgpu_matrix_contract_is_available(
            descriptor, environment->feature_bits, wave_size) ||
        !loom_amdgpu_descriptor_set_has_ref(environment->descriptor_set,
                                            descriptor->low_descriptor_ref)) {
      continue;
    }
    loom_scalar_type_t expected_element_type = LOOM_SCALAR_TYPE_COUNT_;
    if (!loom_amdgpu_fragment_memory_descriptor_role_element_type(
            descriptor, role, &expected_element_type)) {
      continue;
    }
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
        loom_amdgpu_matrix_fragment_role_layout(layout, role);
    if (role_layout == NULL) {
      continue;
    }
    if (!loom_amdgpu_fragment_memory_payload_matches(
            payload_type, operation_kind, expected_element_type, role_layout)) {
      loom_amdgpu_fragment_memory_payload_form_t payload_form =
          LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
      if (loom_amdgpu_fragment_memory_payload_element_match(
              operation_kind, role, payload_type, view_type,
              expected_element_type, &payload_form) !=
          LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE) {
        rejected_payload_layout = true;
      }
      continue;
    }
    loom_amdgpu_fragment_memory_payload_form_t payload_form =
        LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
    const loom_amdgpu_fragment_memory_element_match_t match =
        loom_amdgpu_fragment_memory_payload_element_match(
            operation_kind, role, payload_type, view_type,
            expected_element_type, &payload_form);
    if (match == LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE) {
      rejected_payload_form = true;
      continue;
    }
    if (match <= best_match) {
      continue;
    }
    best_layout = layout;
    best_element_type = expected_element_type;
    best_payload_form = payload_form;
    best_match = match;
    if (match == LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_EXACT) {
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
      diagnostic, rejected_payload_form
                      ? (operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_STORE
                             ? IREE_SV("fragment_memory.store_conversion")
                             : IREE_SV("fragment_memory.payload_form"))
                      : (rejected_payload_layout
                             ? IREE_SV("fragment_memory.payload_layout")
                             : IREE_SV("fragment_memory.target_layout")));
}

static bool loom_amdgpu_fragment_memory_view_matches(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_type_t view_type, loom_type_t payload_type,
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
  if (loom_type_is_vector(payload_type) &&
      payload_element_type != view_element_type &&
      !(view_is_narrowed && payload_element_type == expected_element_type) &&
      !(view_is_extended && view_element_type == expected_element_type)) {
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
          !view_is_packed_16bit_result) {
        return false;
      }
      break;
    case LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN:
    default:
      return false;
  }
  if (view_element_type != expected_element_type && !view_is_narrowed &&
      !view_is_packed_16bit_result) {
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
      case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION:
      case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION:
        packed_axis = 1;
        break;
      case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION:
      case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION:
        packed_axis = 0;
        break;
      case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN_LOW_SUBWORD:
      case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN_LOW_SUBWORD:
      case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_GROUP_PACKED_ROW_COLUMN:
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
  loom_amdgpu_fragment_memory_domain_t domain =
      LOOM_AMDGPU_FRAGMENT_MEMORY_DOMAIN_COUNT_;
  if (!loom_amdgpu_fragment_memory_domain_from_space(memory_space, &domain)) {
    return false;
  }
  loom_amdgpu_descriptor_ref_t descriptor_ref = LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  switch (operation_kind) {
    case LOOM_AMDGPU_MEMORY_OPERATION_LOAD:
      descriptor_ref = kFragmentMemoryDescriptorTables[domain].load_b16_ref;
      break;
    case LOOM_AMDGPU_MEMORY_OPERATION_STORE:
      descriptor_ref = kFragmentMemoryDescriptorTables[domain].store_b16_ref;
      break;
    case LOOM_AMDGPU_MEMORY_OPERATION_COUNT_:
      return false;
  }
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

static bool loom_amdgpu_fragment_memory_space_supports_access(
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_value_fact_memory_space_t memory_space,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout,
    loom_amdgpu_fragment_memory_payload_form_t payload_form) {
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

static loom_value_id_t loom_amdgpu_fragment_memory_narrowed_result_round_source(
    const loom_module_t* module, loom_value_id_t payload,
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout) {
  if (payload >= module->values.count || role_layout == NULL) {
    return LOOM_VALUE_ID_INVALID;
  }
  const loom_type_t payload_type = loom_module_value_type(module, payload);
  if (loom_type_element_type(payload_type) == LOOM_SCALAR_TYPE_F32 &&
      loom_amdgpu_vector_f32_lane_count(payload_type) ==
          role_layout->register_count) {
    return payload;
  }

  const loom_value_t* payload_value = loom_module_value(module, payload);
  const loom_op_t* defining_op = loom_value_def_op(payload_value);
  if (defining_op == NULL || !loom_vector_fptrunc_isa(defining_op) ||
      loom_vector_fptrunc_result(defining_op) != payload) {
    return LOOM_VALUE_ID_INVALID;
  }

  const loom_value_id_t input = loom_vector_fptrunc_input(defining_op);
  if (input >= module->values.count) {
    return LOOM_VALUE_ID_INVALID;
  }
  const loom_type_t input_type = loom_module_value_type(module, input);
  if (loom_type_element_type(input_type) != LOOM_SCALAR_TYPE_F32 ||
      loom_amdgpu_vector_f32_lane_count(input_type) !=
          role_layout->register_count) {
    return LOOM_VALUE_ID_INVALID;
  }
  return input;
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
  const loom_amdgpu_matrix_fragment_layout_t* layout = NULL;
  loom_scalar_type_t expected_element_type = LOOM_SCALAR_TYPE_COUNT_;
  loom_amdgpu_fragment_memory_payload_form_t payload_form =
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
  if (!loom_amdgpu_fragment_memory_target_layout(
          environment, role, operation_kind, payload_type, view_type, &layout,
          &expected_element_type, &payload_form, diagnostic)) {
    return false;
  }
  if (!loom_amdgpu_fragment_memory_shape_matches(environment->fact_table,
                                                 layout, role, source->rows,
                                                 source->columns)) {
    return loom_amdgpu_fragment_memory_reject(diagnostic,
                                              IREE_SV("fragment_memory.shape"));
  }

  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, role);
  if (role_layout == NULL ||
      role_layout->register_count > LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.role_layout"));
  }
  if (!loom_amdgpu_fragment_memory_payload_matches(
          payload_type, operation_kind, expected_element_type, role_layout)) {
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
          operation_kind, expected_element_type, role_layout, payload_form,
          &access)) {
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
  if (!loom_low_source_memory_access_plan_build_indexed(
          environment->module, environment->fact_table, source_operation_kind,
          source->view, source->dynamic_indices, source->static_indices,
          scalar_vector_type, (loom_vector_memory_cache_policy_t){0},
          &source_access, &source_diagnostic)) {
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

  const loom_value_id_t narrowed_result_round_source =
      payload_form ==
              LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16
          ? loom_amdgpu_fragment_memory_narrowed_result_round_source(
                environment->module, source->payload, role_layout)
          : LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_fragment_memory_payload_form_has_descriptors(
          environment->descriptor_set, payload_form)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.store_conversion"));
  }

  if (source_access.memory_space == LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    uint64_t root_byte_offset = 0;
    if (!loom_amdgpu_source_lds_layout_lookup_root(
            environment->fact_table, environment->source_function,
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
        .payload_form = payload_form,
        .narrowed_result_round_source = narrowed_result_round_source,
    };
    for (uint8_t axis = 0; axis < access.view_rank; ++axis) {
      out_plan->axis_byte_strides[axis] = axis_byte_strides[axis];
    }
  }
  return true;
}

static iree_status_t loom_amdgpu_fragment_memory_low_register_kind(
    loom_low_lower_context_t* context, loom_value_id_t low_value,
    loom_amdgpu_fragment_memory_address_register_kind_t* out_register_kind) {
  *out_register_kind = LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE;
  if (low_value == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }

  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_type_t low_type = loom_module_value_type(module, low_value);
  bool is_sgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_SGPR, &is_sgpr));
  if (is_sgpr && loom_low_register_type_unit_count(low_type) == 1) {
    *out_register_kind = LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_SGPR;
    return iree_ok_status();
  }

  bool is_vgpr = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_low_type_register_class_is(
      context, low_type, LOOM_AMDGPU_REG_CLASS_ID_VGPR, &is_vgpr));
  if (is_vgpr && loom_low_register_type_unit_count(low_type) == 1) {
    *out_register_kind = LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_VGPR;
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_INTERNAL,
      "AMDGPU fragment memory address selected a non-scalar register");
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

static bool loom_amdgpu_fragment_memory_uses_dynamic_view_base_value(
    const loom_amdgpu_fragment_memory_plan_t* plan, uint8_t term_index) {
  // The source plan keeps normalized dynamic terms for legality, but the
  // original byte-domain view-base value may already be materialized and shared
  // by nearby fragment ops. When using that value, emission subtracts the
  // extracted static view-base delta from the immediate side of the address.
  return term_index == 0 && plan->source.dynamic_view_base_term_count == 1 &&
         plan->source.dynamic_view_base_value_id != LOOM_VALUE_ID_INVALID;
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
        LOOM_AMDGPU_FRAGMENT_MEMORY_ADDRESS_REGISTER_NONE;
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_low_register_kind(
        context, low_index, &register_kind));
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
    const loom_amdgpu_fragment_lane_ids_t* lane_ids, uint32_t lane_mod_stride,
    uint32_t lane_div_stride, loom_type_t sgpr_type, loom_type_t vgpr_type,
    loom_amdgpu_fragment_memory_address_accumulator_t* inout_accumulator) {
  if (lane_mod_stride != 0 && lane_div_stride != 0 &&
      (uint64_t)lane_mod_stride * LOOM_AMDGPU_FRAGMENT_LANE_MODULUS ==
          lane_div_stride) {
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
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_PACKED_REDUCTION:
      *out_lane_mod_stride = plan->axis_byte_strides[0];
      *out_static_byte_offset = (uint64_t)register_index *
                                plan->elements_per_register *
                                plan->axis_byte_strides[1];
      return true;
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_PACKED_REDUCTION:
      *out_lane_mod_stride = plan->axis_byte_strides[1];
      *out_static_byte_offset = (uint64_t)register_index *
                                plan->elements_per_register *
                                plan->axis_byte_strides[0];
      return true;
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN:
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_REGISTER_INTERLEAVED_ROW_COLUMN_LOW_SUBWORD:
      *out_lane_mod_stride = plan->axis_byte_strides[1];
      *out_lane_div_stride = plan->axis_byte_strides[0];
      *out_static_byte_offset =
          (uint64_t)register_index *
          (shape.result_row_count / plan->register_count) *
          plan->axis_byte_strides[0];
      return true;
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_ROW_LANE_GROUP_PACKED_REDUCTION:
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
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_MOD_COLUMN_LANE_GROUP_PACKED_REDUCTION:
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
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN:
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_GROUP_REGISTER_ROW_COLUMN_LOW_SUBWORD:
      *out_lane_mod_stride = plan->axis_byte_strides[1];
      if (!loom_amdgpu_fragment_memory_scale_stride_u32(
              plan->register_count, plan->axis_byte_strides[0],
              out_lane_div_stride)) {
        return false;
      }
      *out_static_byte_offset =
          (uint64_t)register_index * plan->axis_byte_strides[0];
      return true;
    case LOOM_AMDGPU_MATRIX_FRAGMENT_MAP_LANE_GROUP_PACKED_ROW_COLUMN:
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
    loom_amdgpu_fragment_memory_packet_plan_t* out_packet) {
  *out_packet = (loom_amdgpu_fragment_memory_packet_plan_t){0};
  const uint16_t remaining = plan->register_count - register_index;
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(kLoomAmdgpuFragmentMemoryNarrowedStoreCandidates);
       ++i) {
    const uint16_t candidate =
        kLoomAmdgpuFragmentMemoryNarrowedStoreCandidates[i];
    if (candidate > remaining ||
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
  return true;
}

static bool loom_amdgpu_fragment_memory_plan_packets(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_amdgpu_fragment_memory_plan_t* plan,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
  plan->packet_count = 0;
  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, plan->role);
  const bool scalar_b16_packets =
      loom_amdgpu_fragment_memory_role_uses_scalar_b16_packets(role_layout);
  const bool load_packed_16bit_result =
      plan->payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT;
  for (uint16_t register_index = 0; register_index < plan->register_count;) {
    loom_amdgpu_fragment_memory_packet_plan_t packet = {0};
    const bool selected =
        scalar_b16_packets
            ? loom_amdgpu_fragment_memory_select_low_subword_packet(
                  descriptor_set, plan, register_index, &packet)
        : load_packed_16bit_result
            ? loom_amdgpu_fragment_memory_select_packed_16bit_result_load_packet(
                  descriptor_set, plan, register_index, &packet)
        : plan->payload_form ==
                LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16
            ? loom_amdgpu_fragment_memory_select_narrowed_store_packet(
                  descriptor_set, layout, plan, register_index, &packet)
            : loom_amdgpu_fragment_memory_select_packet(
                  descriptor_set, layout, plan, register_index, &packet);
    if (!selected ||
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
  return true;
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

iree_status_t loom_amdgpu_analyze_vector_fragment_memory_plan(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    loom_symbol_ref_t target_ref, loom_func_like_t source_function,
    const loom_op_t* source_op,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected) {
  *out_plan = (loom_amdgpu_fragment_memory_plan_t){0};
  *out_selected = false;
  const loom_amdgpu_fragment_memory_environment_t environment = {
      .module = module,
      .fact_table = fact_table,
      .bundle = bundle,
      .descriptor_set = descriptor_set,
      .feature_bits = loom_amdgpu_fragment_memory_feature_bits_from_target_ref(
          module, target_ref),
      .source_function = source_function,
  };
  loom_amdgpu_fragment_memory_source_t source = {0};
  loom_amdgpu_fragment_memory_source_from_op(source_op, operation_kind,
                                             &source);
  if (!loom_amdgpu_fragment_memory_analyze(&environment, &source,
                                           operation_kind, out_plan,
                                           /*diagnostic=*/NULL)) {
    return iree_ok_status();
  }
  const loom_amdgpu_matrix_fragment_layout_t* layout =
      loom_amdgpu_matrix_fragment_layout_for_kind(out_plan->layout_kind);
  if (layout == NULL ||
      !loom_amdgpu_fragment_memory_plan_packets(
          environment.descriptor_set, layout, out_plan, /*diagnostic=*/NULL)) {
    return iree_ok_status();
  }
  *out_selected = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_fragment_memory_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected) {
  const loom_module_t* module = loom_low_lower_context_module(context);
  return loom_amdgpu_analyze_vector_fragment_memory_plan(
      module, loom_low_lower_context_fact_table(context),
      loom_low_lower_context_bundle(context),
      loom_low_lower_context_descriptor_set(context),
      loom_low_lower_context_target_ref(context),
      loom_low_lower_context_source_function(context), source_op,
      operation_kind, out_plan, out_selected);
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
  const loom_amdgpu_fragment_memory_environment_t environment = {
      .module = module,
      .fact_table = loom_target_low_legality_fact_table(context),
      .bundle = bundle,
      .descriptor_set = loom_target_low_legality_descriptor_set(context),
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
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU narrowed fragment store lane exceeds payload register count");
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
    loom_value_id_t low_source, uint16_t source_register_count,
    uint16_t register_index, loom_type_t vgpr_type, loom_value_id_t* out_lane) {
  *out_lane = LOOM_VALUE_ID_INVALID;
  loom_value_id_t source_register = low_source;
  if (source_register_count != 1) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
        context, source_op, low_source, register_index, vgpr_type,
        &source_register));
  }
  return loom_amdgpu_emit_f32_to_bf16_lane(context, source_op, source_register,
                                           vgpr_type, out_lane);
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
  const loom_value_id_t operands[] = {source_register};
  loom_op_t* low_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_F16, operands,
      IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0), &vgpr_type,
      1, &low_op));
  *out_lane = loom_value_slice_get(loom_low_op_results(low_op), 0);
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_emit_fragment_memory_f32_pair_to_packed_16bit(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t low_source, uint16_t register_index, loom_type_t vgpr_type,
    loom_value_id_t* out_packed) {
  *out_packed = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_low_slice(context, source_op, low_source, register_index,
                                 vgpr_type, &low_source_register));
  loom_value_id_t high_source_register = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_slice(
      context, source_op, low_source, register_index + 1u, vgpr_type,
      &high_source_register));
  return loom_amdgpu_emit_f32_pair_to_packed_bf16(
      context, source_op, low_source_register, high_source_register, vgpr_type,
      out_packed);
}

static iree_status_t loom_amdgpu_emit_fragment_memory_packed_16bit_packet(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_fragment_memory_plan_t* plan, loom_value_id_t low_payload,
    uint16_t register_index, uint16_t result_register_count,
    uint16_t packet_register_count, loom_type_t vgpr_type,
    loom_value_id_t* out_packet) {
  *out_packet = LOOM_VALUE_ID_INVALID;
  if (result_register_count == 1) {
    if (plan->narrowed_result_round_source != LOOM_VALUE_ID_INVALID) {
      return loom_amdgpu_emit_fragment_memory_f32_to_16bit_lane(
          context, source_op, low_payload, plan->register_count, register_index,
          vgpr_type, out_packet);
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
            context, source_op, low_payload, source_register_index, vgpr_type,
            &packed_registers[i]));
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
    const loom_amdgpu_fragment_lane_ids_t* lane_ids, loom_type_t vgpr_type,
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

  loom_type_t sgpr_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_sgpr_type(context, &sgpr_type));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_dynamic_source_terms(
      context, source_op, plan, sgpr_type, vgpr_type, out_accumulator));
  return loom_amdgpu_emit_fragment_memory_lane_terms(
      context, source_op, lane_ids, lane_mod_stride, lane_div_stride, sgpr_type,
      vgpr_type, out_accumulator);
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
  uint32_t lane_mod_stride = 0;
  uint32_t lane_div_stride = 0;
  if (plan->source.static_byte_offset < 0) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment memory base offset");
    IREE_BUILTIN_UNREACHABLE();
  }
  int64_t static_byte_offset_i64 = plan->source.static_byte_offset;
  if (loom_amdgpu_fragment_memory_uses_dynamic_view_base_value(
          plan, /*term_index=*/0) &&
      !loom_checked_sub_i64(static_byte_offset_i64,
                            plan->source.static_view_base_byte_offset,
                            &static_byte_offset_i64)) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment memory static offset");
    IREE_BUILTIN_UNREACHABLE();
  }
  uint64_t register_static_offset = 0;
  if (!loom_amdgpu_fragment_memory_register_terms(
          layout, plan, register_index, &lane_mod_stride, &lane_div_stride,
          &register_static_offset)) {
    IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment register map");
    IREE_BUILTIN_UNREACHABLE();
  }
  if (element_index != 0) {
    const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
        loom_amdgpu_matrix_fragment_role_layout(layout, plan->role);
    if (!loom_amdgpu_fragment_memory_role_uses_packed_b16_elements(
            role_layout) ||
        element_index >= role_layout->elements_per_register) {
      IREE_ASSERT_UNREACHABLE("selected AMDGPU fragment element map");
      IREE_BUILTIN_UNREACHABLE();
    }
    const uint64_t element_static_offset =
        (uint64_t)element_index * plan->axis_byte_strides[0];
    register_static_offset += element_static_offset;
  }
  if (register_static_offset > INT64_MAX ||
      !iree_checked_add_i64(static_byte_offset_i64,
                            (int64_t)register_static_offset,
                            &static_byte_offset_i64)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU fragment memory byte offset exceeds i64");
  }
  uint64_t static_byte_offset = (uint64_t)static_byte_offset_i64;
  if (static_byte_offset > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU fragment memory byte offset exceeds u32");
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
  *out_static_byte_offset = plan->source.static_byte_offset;
  if (plan->source.static_byte_offset < 0) {
    return false;
  }
  uint32_t unused_lane_mod_stride = 0;
  uint32_t unused_lane_div_stride = 0;
  uint64_t register_static_offset = 0;
  if (!loom_amdgpu_fragment_memory_register_terms(
          layout, plan, packet->register_index, &unused_lane_mod_stride,
          &unused_lane_div_stride, &register_static_offset)) {
    return false;
  }
  if (element_index != 0) {
    if (plan->axis_byte_strides[0] != 0 &&
        element_index > UINT64_MAX / plan->axis_byte_strides[0]) {
      return false;
    }
    const uint64_t element_static_offset =
        (uint64_t)element_index * plan->axis_byte_strides[0];
    if (register_static_offset > UINT64_MAX - element_static_offset) {
      return false;
    }
    register_static_offset += element_static_offset;
  }
  if (register_static_offset > INT64_MAX) {
    return false;
  }
  return iree_checked_add_i64(plan->source.static_byte_offset,
                              (int64_t)register_static_offset,
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

static iree_string_view_t loom_amdgpu_fragment_memory_packet_fallback_reason(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_amdgpu_fragment_memory_plan_t* plan,
    const loom_amdgpu_fragment_memory_packet_plan_t* packet) {
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
  iree_string_view_t packet_key = iree_string_view_empty();
  if (descriptor != NULL) {
    packet_key = loom_low_descriptor_set_string(descriptor_set,
                                                descriptor->key_string_offset);
  }
  int64_t static_offset_bytes = plan->source.static_byte_offset;
  (void)loom_amdgpu_fragment_memory_packet_static_offset(
      layout, plan, packet, element_index, &static_offset_bytes);
  const loom_low_lower_memory_report_row_t row = {
      .function_name = loom_low_lower_context_function_name(context),
      .source_op_name =
          loom_op_name(loom_low_lower_context_module(context), source_op),
      .source_op_kind = source_op->kind,
      .memory_space = loom_amdgpu_memory_space_name(plan->source.memory_space),
      .operation_kind = loom_amdgpu_memory_operation_name(plan->operation_kind),
      .packet_key = packet_key,
      .address_form = loom_amdgpu_fragment_memory_report_address_form(plan),
      .dynamic_term_kind = IREE_SV("vaddr"),
      .fallback_reason = loom_amdgpu_fragment_memory_packet_fallback_reason(
          descriptor_set, layout, plan, packet),
      .static_offset_bytes = static_offset_bytes,
      .element_byte_count = plan->element_byte_count,
      .vector_lane_count = vector_lane_count,
      .dynamic_stride_bytes =
          loom_amdgpu_fragment_memory_packet_dynamic_stride_bytes(layout, plan,
                                                                  packet),
      .vector_lane_stride_bytes = plan->element_byte_count,
      .bank_stride_words = 0,
      .bank_conflict_degree = 0,
      .bank_conflict_kind = iree_string_view_empty(),
  };
  return loom_low_lower_record_memory_report_row(context, &row);
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
    loom_type_t vgpr_type, loom_amdgpu_fragment_lane_ids_t* out_lane_ids) {
  *out_lane_ids = (loom_amdgpu_fragment_lane_ids_t){
      .lane = LOOM_VALUE_ID_INVALID,
      .lane_mod = LOOM_VALUE_ID_INVALID,
      .lane_div = LOOM_VALUE_ID_INVALID,
  };
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_current_subgroup_lane_id(
      context, source_op, vgpr_type, &out_lane_ids->lane));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      out_lane_ids->lane, LOOM_AMDGPU_FRAGMENT_LANE_MODULUS - 1, vgpr_type,
      &out_lane_ids->lane_mod));
  return loom_amdgpu_emit_vgpr_shift(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_LSHRREV_B32_LIT, 4,
      out_lane_ids->lane, vgpr_type, &out_lane_ids->lane_div);
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
    const loom_value_id_t operands[] = {low_packet};
    loom_op_t* low_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op,
        LOOM_AMDGPU_DESCRIPTOR_REF_V_BFE_U32_OFFSET_0_WIDTH_16_LOW16, operands,
        IREE_ARRAYSIZE(operands), loom_make_named_attr_slice(NULL, 0),
        &vgpr_type, 1, &low_op));
    *out_full_packet = loom_value_slice_get(loom_low_op_results(low_op), 0);
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(loom_amdgpu_materialize_low_vgpr_b32(
      context, source_op, low_packet, out_full_packet));
  return loom_amdgpu_emit_vgpr_binary_immediate(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_V_AND_B32_LIT,
      *out_full_packet, UINT32_C(0xFFFF), vgpr_type, out_full_packet);
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
  loom_amdgpu_fragment_lane_ids_t lane_ids;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_lane_ids(
      context, source_op, vgpr_type, &lane_ids));
  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, plan->role);
  const bool low_subword =
      loom_amdgpu_fragment_memory_role_uses_low_subword(role_layout);
  const bool packed_b16_elements =
      loom_amdgpu_fragment_memory_role_uses_packed_b16_elements(role_layout);
  const bool load_packed_16bit_result =
      plan->payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_PACKED_16BIT_RESULT;

  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_packet_resource = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  if (plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, plan->source.view_value_id, &low_resource));
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_resource(
        context, source_op, plan, low_resource, &low_packet_resource,
        &low_soffset));
  }
  loom_amdgpu_fragment_memory_address_accumulator_t base_accumulator;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fragment_memory_base_address_accumulator(
          context, source_op, layout, plan, &lane_ids, vgpr_type,
          &base_accumulator));

  loom_value_id_t low_packets[LOOM_AMDGPU_MAX_PACKED_32BIT_REGISTERS] = {0};
  for (uint16_t packet_index = 0; packet_index < plan->packet_count;
       ++packet_index) {
    const loom_amdgpu_fragment_memory_packet_plan_t* packet =
        &plan->packets[packet_index];
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
  loom_amdgpu_fragment_lane_ids_t lane_ids;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_lane_ids(
      context, source_op, vgpr_type, &lane_ids));
  const loom_amdgpu_matrix_fragment_role_layout_t* role_layout =
      loom_amdgpu_matrix_fragment_role_layout(layout, plan->role);
  const bool packed_b16_elements =
      loom_amdgpu_fragment_memory_role_uses_packed_b16_elements(role_layout);

  loom_value_id_t low_resource = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_packet_resource = LOOM_VALUE_ID_INVALID;
  loom_value_id_t low_soffset = LOOM_VALUE_ID_INVALID;
  if (plan->source.memory_space != LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
        context, plan->source.view_value_id, &low_resource));
    IREE_RETURN_IF_ERROR(loom_amdgpu_fragment_memory_packet_resource(
        context, source_op, plan, low_resource, &low_packet_resource,
        &low_soffset));
  }
  loom_amdgpu_fragment_memory_address_accumulator_t base_accumulator;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_emit_fragment_memory_base_address_accumulator(
          context, source_op, layout, plan, &lane_ids, vgpr_type,
          &base_accumulator));

  if (plan->payload_form ==
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16) {
    loom_value_id_t low_payload = LOOM_VALUE_ID_INVALID;
    if (plan->narrowed_result_round_source != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_low_lower_lookup_value(
          context, plan->narrowed_result_round_source, &low_payload));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_low_lower_lookup_value(context, plan->payload, &low_payload));
    }

    for (uint16_t packet_index = 0; packet_index < plan->packet_count;
         ++packet_index) {
      const loom_amdgpu_fragment_memory_packet_plan_t* packet =
          &plan->packets[packet_index];
      loom_value_id_t low_payload_packet = LOOM_VALUE_ID_INVALID;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_fragment_memory_packed_16bit_packet(
          context, source_op, plan, low_payload, packet->register_index,
          packet->result_register_count, packet->packet_register_count,
          vgpr_type, &low_payload_packet));
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
    loom_low_lower_require_source_value_storage(context,
                                                plan->source.view_value_id);
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
      if (plan->narrowed_result_round_source != LOOM_VALUE_ID_INVALID) {
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
