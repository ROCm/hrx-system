// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_plan.h"

#include <stdint.h>

#include "iree/base/internal/math.h"
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
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_address.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_packet.h"
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
  LOOM_AMDGPU_FRAGMENT_PACKED_B8_ELEMENT_BIT_COUNT = 8,
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
  const loom_amdgpu_matrix_fragment_contract_candidates_t* contract_candidates;
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

static bool loom_amdgpu_fragment_memory_can_narrow_result_store(
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_scalar_type_t expected_element_type,
    loom_scalar_type_t storage_element_type) {
  return operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_STORE &&
         loom_amdgpu_matrix_fragment_role_is_result_like(role) &&
         expected_element_type == LOOM_SCALAR_TYPE_F32 &&
         storage_element_type == LOOM_SCALAR_TYPE_BF16;
}

static bool loom_amdgpu_fragment_memory_can_extend_result_store(
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_scalar_type_t expected_element_type,
    loom_scalar_type_t payload_element_type) {
  return operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_STORE &&
         loom_amdgpu_matrix_fragment_role_is_result_like(role) &&
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
         loom_amdgpu_matrix_fragment_role_is_result_like(role) &&
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

static bool loom_amdgpu_fragment_memory_role_view_axis(
    loom_contract_operand_role_t role,
    loom_matrix_fragment_axis_t semantic_axis, uint8_t* out_axis) {
  static const uint8_t kViewAxes[][LOOM_MATRIX_FRAGMENT_AXIS_COUNT] = {
      [LOOM_CONTRACT_OPERAND_ROLE_UNKNOWN] = {UINT8_MAX, UINT8_MAX, UINT8_MAX,
                                              UINT8_MAX},
      [LOOM_CONTRACT_OPERAND_ROLE_LHS] = {UINT8_MAX, 0, UINT8_MAX, 1},
      [LOOM_CONTRACT_OPERAND_ROLE_RHS] = {UINT8_MAX, UINT8_MAX, 1, 0},
      [LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR] = {UINT8_MAX, 0, 1, UINT8_MAX},
      [LOOM_CONTRACT_OPERAND_ROLE_RESULT] = {UINT8_MAX, 0, 1, UINT8_MAX},
  };
  *out_axis = UINT8_MAX;
  if ((iree_host_size_t)role >= IREE_ARRAYSIZE(kViewAxes) ||
      semantic_axis >= LOOM_MATRIX_FRAGMENT_AXIS_COUNT) {
    return false;
  }
  *out_axis = kViewAxes[role][semantic_axis];
  return *out_axis != UINT8_MAX;
}

static bool loom_amdgpu_fragment_memory_role_packed_element_axis(
    const loom_matrix_fragment_role_layout_t* role_layout, uint8_t* out_axis) {
  *out_axis = UINT8_MAX;
  if (role_layout == NULL) {
    return false;
  }
  loom_matrix_fragment_axis_t semantic_axis = LOOM_MATRIX_FRAGMENT_AXIS_COUNT;
  if (!loom_amdgpu_matrix_fragment_role_layout_packed_element_axis(
          role_layout, &semantic_axis)) {
    return false;
  }
  return loom_amdgpu_fragment_memory_role_view_axis(role_layout->role,
                                                    semantic_axis, out_axis);
}

static bool loom_amdgpu_fragment_memory_requires_native_payload_storage(
    loom_amdgpu_memory_operation_kind_t operation_kind,
    const loom_matrix_fragment_role_layout_t* role_layout,
    loom_amdgpu_fragment_memory_payload_form_t payload_form) {
  return operation_kind == LOOM_AMDGPU_MEMORY_OPERATION_STORE &&
         (loom_amdgpu_matrix_fragment_role_layout_uses_low_subword(
              role_layout) ||
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
    const loom_matrix_fragment_role_layout_t* role_layout) {
  if (view_storage_schema == NULL || !loom_type_is_view(view_type)) {
    return false;
  }
  const loom_value_fact_encoded_operand_schema_t operand =
      view_storage_schema->encoded_operand;
  if (loom_value_fact_encoded_operand_schema_is_unknown(operand) ||
      !iree_any_bit_set(operand.payload_packing,
                        LOOM_VALUE_FACT_PAYLOAD_PACKING_TARGET_FRAGMENT) ||
      operand.payload_register_count != role_layout->register_count ||
      operand.payload_element_count != role_layout->payload_element_count) {
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
    const loom_matrix_fragment_role_layout_t* role_layout) {
  if (operation_kind != LOOM_AMDGPU_MEMORY_OPERATION_LOAD ||
      !loom_amdgpu_matrix_fragment_role_is_result_like(role_layout->role) ||
      expected_element_type != LOOM_SCALAR_TYPE_F32 ||
      role_layout->element_bit_count != 32) {
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
    const loom_matrix_fragment_role_layout_t* role_layout) {
  if (role_layout == NULL || !loom_type_is_vector(payload_type) ||
      loom_type_rank(payload_type) != 1 ||
      !loom_type_is_all_static(payload_type)) {
    return false;
  }

  switch (role_layout->role) {
    case LOOM_CONTRACT_OPERAND_ROLE_LHS:
    case LOOM_CONTRACT_OPERAND_ROLE_RHS: {
      return loom_amdgpu_matrix_fragment_payload_matches_role_storage(
                 payload_type, expected_element_type, role_layout) ||
             loom_amdgpu_fragment_memory_payload_matches_encoded_storage(
                 payload_type, view_type, view_storage_schema,
                 expected_element_type, role_layout);
    }
    case LOOM_CONTRACT_OPERAND_ROLE_ACCUMULATOR:
    case LOOM_CONTRACT_OPERAND_ROLE_RESULT: {
      if (loom_amdgpu_matrix_fragment_payload_matches_role_storage(
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
        return loom_amdgpu_matrix_fragment_payload_matches_role_storage(
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
    const loom_matrix_fragment_role_layout_t* role_layout) {
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
    const loom_matrix_fragment_role_layout_t* role_layout) {
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
    const loom_matrix_fragment_role_layout_t* role_layout,
    uint16_t* out_register_count) {
  *out_register_count = 0;
  if (source >= module->values.count || role_layout == NULL ||
      !loom_amdgpu_matrix_fragment_role_is_result_like(role_layout->role) ||
      role_layout->element_bit_count != 32) {
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
    const loom_matrix_fragment_role_layout_t* role_layout,
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
    const loom_matrix_fragment_role_layout_t* role_layout,
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
    const loom_matrix_fragment_role_layout_t* role_layout,
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

  const loom_amdgpu_matrix_fragment_contract_candidates_t* candidates =
      environment->contract_candidates;
  const iree_host_size_t descriptor_count =
      loom_amdgpu_matrix_fragment_contract_candidate_count(candidates);
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
        loom_amdgpu_matrix_fragment_contract_candidate_at(candidates, i);
    if (candidates == NULL &&
        !loom_amdgpu_matrix_fragment_contract_is_available(
            descriptor, environment->descriptor_set, environment->feature_bits,
            wave_size)) {
      continue;
    }
    loom_amdgpu_matrix_fragment_role_storage_t role_storage = {0};
    if (!loom_amdgpu_matrix_fragment_descriptor_role_storage(descriptor, role,
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
      if (loom_amdgpu_matrix_fragment_tile_shape_matches(
              environment->fact_table, descriptor->tile_shape, role, rows,
              columns) &&
          loom_amdgpu_fragment_memory_payload_matches_descriptor_storage(
              payload_type, expected_element_type, &role_storage.payload)) {
        rejected_target_layout = true;
      }
      continue;
    }
    const loom_matrix_fragment_role_layout_t* role_layout =
        loom_matrix_fragment_role_layout(layout, role);
    if (role_layout == NULL) {
      continue;
    }
    const bool shape_matches = loom_amdgpu_matrix_fragment_shape_matches(
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

static bool loom_amdgpu_fragment_memory_view_matches(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_type_t view_type, loom_type_t payload_type,
    const loom_value_fact_storage_schema_t* view_storage_schema,
    loom_amdgpu_memory_operation_kind_t operation_kind,
    loom_scalar_type_t expected_element_type,
    const loom_matrix_fragment_role_layout_t* role_layout,
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
    const loom_vector_memory_access_t* access, uint32_t* out_axis_byte_strides,
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
  return true;
}

static bool loom_amdgpu_fragment_memory_add_scaled_stride(
    uint32_t factor, uint32_t byte_stride, uint32_t* inout_byte_stride) {
  const uint64_t result =
      (uint64_t)*inout_byte_stride + (uint64_t)factor * byte_stride;
  if (result > UINT32_MAX) {
    return false;
  }
  *inout_byte_stride = (uint32_t)result;
  return true;
}

static bool loom_amdgpu_fragment_memory_compile_address_layout(
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    const loom_matrix_fragment_role_layout_t* role_layout, uint8_t view_rank,
    uint16_t element_byte_count, const uint32_t* axis_byte_strides,
    loom_amdgpu_fragment_memory_address_layout_t* out_address_layout,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
  *out_address_layout = (loom_amdgpu_fragment_memory_address_layout_t){0};
  const uint16_t payload_elements_per_register =
      loom_amdgpu_matrix_fragment_payload_elements_per_register(role_layout);
  if (layout->tile_shape.block_count != 1 ||
      iree_any_bit_set(role_layout->coordinate_flags,
                       LOOM_MATRIX_FRAGMENT_COORDINATE_BLOCK) ||
      payload_elements_per_register == 0 ||
      role_layout->coordinate_element_offset != 0 ||
      role_layout->coordinate_element_stride == 0 ||
      (payload_elements_per_register %
       role_layout->coordinate_element_stride) != 0) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.address_layout"));
  }
  IREE_ASSERT_EQ(role_layout->payload_element_count,
                 role_layout->register_count * payload_elements_per_register);

  out_address_layout->payload_elements_per_register =
      payload_elements_per_register;
  loom_matrix_fragment_axis_t register_axis = LOOM_MATRIX_FRAGMENT_AXIS_COUNT;
  uint8_t register_view_axis = UINT8_MAX;
  for (iree_host_size_t i = 0; i < LOOM_MATRIX_FRAGMENT_AXIS_COUNT; ++i) {
    if (!iree_any_bit_set(role_layout->coordinate_flags, 1u << i)) {
      continue;
    }
    const loom_matrix_fragment_axis_t axis = (loom_matrix_fragment_axis_t)i;
    uint8_t view_axis = UINT8_MAX;
    if (!loom_amdgpu_fragment_memory_role_view_axis(role_layout->role, axis,
                                                    &view_axis) ||
        view_axis >= view_rank) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.address_layout"));
    }
    const loom_matrix_fragment_axis_layout_t* axis_layout =
        &role_layout->axes[axis];
    if (axis_layout->element_count > 1 || axis_layout->outer_count > 1) {
      if (register_axis != LOOM_MATRIX_FRAGMENT_AXIS_COUNT) {
        return loom_amdgpu_fragment_memory_reject(
            diagnostic, IREE_SV("fragment_memory.address_layout"));
      }
      register_axis = axis;
      register_view_axis = view_axis;
    }
    if (axis_layout->thread_count <= 1) {
      continue;
    }
    uint32_t lane_byte_stride = 0;
    if (!loom_amdgpu_fragment_memory_add_scaled_stride(
            axis_layout->element_count, axis_byte_strides[view_axis],
            &lane_byte_stride)) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.address_layout"));
    }
    uint16_t lane_divisor = 0;
    uint32_t* lane_stride = NULL;
    if (axis_layout->thread_stride == 1) {
      lane_divisor = axis_layout->thread_count;
      lane_stride = &out_address_layout->lane_mod_byte_stride;
    } else if ((layout->wave_size % axis_layout->thread_stride) == 0 &&
               layout->wave_size / axis_layout->thread_stride ==
                   axis_layout->thread_count) {
      lane_divisor = axis_layout->thread_stride;
      lane_stride = &out_address_layout->lane_div_byte_stride;
    } else {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.address_layout"));
    }
    if (!loom_amdgpu_u32_is_power_of_two(lane_divisor) ||
        (out_address_layout->lane_divisor != 0 &&
         out_address_layout->lane_divisor != lane_divisor) ||
        !loom_amdgpu_fragment_memory_add_scaled_stride(1, lane_byte_stride,
                                                       lane_stride)) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.address_layout"));
    }
    out_address_layout->lane_divisor = lane_divisor;
  }
  if (out_address_layout->lane_divisor == 0) {
    out_address_layout->lane_divisor = 1;
  }

  if (role_layout->register_count > 1) {
    if (register_axis == LOOM_MATRIX_FRAGMENT_AXIS_COUNT) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.address_layout"));
    }
    const loom_matrix_fragment_axis_layout_t* axis_layout =
        &role_layout->axes[register_axis];
    const uint32_t coordinate_step =
        payload_elements_per_register / role_layout->coordinate_element_stride;
    uint32_t coordinate_stride = 0;
    if (axis_layout->outer_count == 1) {
      coordinate_stride = coordinate_step;
    } else if (axis_layout->element_count == 1 ||
               (coordinate_step % axis_layout->element_count) == 0) {
      coordinate_stride = coordinate_step * axis_layout->thread_count;
    } else {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.address_layout"));
    }
    if (!loom_amdgpu_fragment_memory_add_scaled_stride(
            coordinate_stride, axis_byte_strides[register_view_axis],
            &out_address_layout->register_byte_stride)) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.address_layout"));
    }
  }

  uint8_t packed_view_axis = UINT8_MAX;
  if (loom_amdgpu_fragment_memory_role_packed_element_axis(role_layout,
                                                           &packed_view_axis)) {
    if (packed_view_axis >= view_rank) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.address_layout"));
    }
    const uint32_t packed_element_byte_stride =
        axis_byte_strides[packed_view_axis];
    if (!loom_amdgpu_matrix_fragment_role_layout_uses_scalar_b16_packets(
            role_layout) &&
        packed_element_byte_stride != element_byte_count) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.packed_axis_stride"));
    }
    out_address_layout->packed_element_byte_stride = packed_element_byte_stride;
  }
  return true;
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
    const loom_matrix_fragment_role_layout_t* role_layout) {
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

  const loom_matrix_fragment_role_layout_t* role_layout =
      loom_matrix_fragment_role_layout(layout, role);
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
  if (!loom_amdgpu_fragment_memory_fill_view_strides(&access, axis_byte_strides,
                                                     diagnostic)) {
    return false;
  }
  loom_amdgpu_fragment_memory_address_layout_t address_layout = {0};
  if (!loom_amdgpu_fragment_memory_compile_address_layout(
          layout, role_layout, access.view_rank,
          (uint16_t)access.static_element_byte_count, axis_byte_strides,
          &address_layout, diagnostic)) {
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
        .element_byte_count = (uint16_t)source_access.element_byte_count,
        .view_element_type = loom_type_element_type(view_type),
        .address_layout = address_layout,
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

static bool loom_amdgpu_analyze_vector_fragment_memory_plan_impl(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    const loom_view_region_table_t* view_regions,
    const loom_target_bundle_t* bundle,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_amdgpu_matrix_fragment_contract_candidates_t*
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
              : loom_amdgpu_matrix_fragment_feature_bits_from_target_ref(
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
  if (layout == NULL || !loom_amdgpu_fragment_memory_plan_packets(
                            environment.descriptor_set, layout, out_plan,
                            /*out_constraint_key=*/NULL)) {
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
  const loom_amdgpu_matrix_fragment_contract_candidates_t* contract_candidates =
      NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_fragment_contract_candidates(
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
      .feature_bits = loom_amdgpu_matrix_fragment_feature_bits_from_target_ref(
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
    if (layout != NULL && loom_amdgpu_fragment_memory_plan_packets(
                              environment.descriptor_set, layout, &plan,
                              &diagnostic.constraint_key)) {
      return iree_ok_status();
    }
  }
  iree_string_view_t constraint_key = diagnostic.constraint_key;
  if (iree_string_view_is_empty(constraint_key)) {
    constraint_key = IREE_SV("fragment_memory");
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}
