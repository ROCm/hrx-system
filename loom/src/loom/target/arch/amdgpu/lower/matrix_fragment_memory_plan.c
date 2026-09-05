// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_plan.h"

#include <stdint.h>
#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/codegen/low/lower/representation_observer.h"
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
#include "loom/target/arch/amdgpu/lower/encoding/float16.h"
#include "loom/target/arch/amdgpu/lower/encoding/fp8.h"
#include "loom/target/arch/amdgpu/lower/legality.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_address.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_layout.h"
#include "loom/target/arch/amdgpu/lower/matrix_fragment_memory_packet.h"
#include "loom/target/arch/amdgpu/lower/memory.h"
#include "loom/target/arch/amdgpu/lower/subgroup.h"
#include "loom/target/arch/amdgpu/lower/topology.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
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
  // Optional source fragment block count value.
  loom_value_id_t blocks;
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

typedef struct loom_amdgpu_fragment_memory_view_numeric_t {
  // Exact numeric format declared by the source view storage.
  loom_value_fact_numeric_format_flags_t exact_format;
  // Format accepted by native descriptors after applying proven value facts.
  loom_value_fact_numeric_format_flags_t descriptor_format;
  // Numeric facts published by the source view storage schema.
  loom_value_facts_t content_facts;
} loom_amdgpu_fragment_memory_view_numeric_t;

// Representation-invariant facts prepared once for one fragment memory op.
typedef struct loom_amdgpu_fragment_memory_prepared_t {
  // Matrix role derived from the source fragment role.
  loom_contract_operand_role_t role;
  // Source fragment payload type.
  loom_type_t payload_type;
  // Source memory view type.
  loom_type_t view_type;
  // Scalar element type stored by the source memory view.
  loom_scalar_type_t view_element_type;
  // Exact storage schema attached to the source view, when present.
  loom_value_fact_storage_schema_t view_storage_schema;
  // Whether |view_storage_schema| contains an exact encoded operand schema.
  bool has_view_storage_schema;
  // Numeric interpretation derived from the view element and storage schema.
  loom_amdgpu_fragment_memory_view_numeric_t view_numeric;
  // Target-neutral description of the source memory view.
  loom_vector_memory_access_t access;
  // Source view byte strides, including dynamic stride expressions.
  loom_low_source_memory_axis_byte_stride_t
      axis_byte_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  // Exact byte strides for static source view axes.
  uint32_t static_axis_byte_strides[LOOM_ENCODING_ADDRESS_LAYOUT_MAX_RANK];
  // Target-neutral indexed source memory access plan.
  loom_low_source_memory_access_plan_t source_access;
  // Number of 32-bit registers occupied by the authored payload.
  uint16_t payload_register_count;
} loom_amdgpu_fragment_memory_prepared_t;

IREE_ATTRIBUTE_NOINLINE static bool loom_amdgpu_fragment_memory_reject(
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

static bool loom_amdgpu_fragment_memory_can_narrow_result_store(
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_scalar_type_t expected_element_type,
    loom_scalar_type_t storage_element_type) {
  return operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE &&
         loom_amdgpu_matrix_fragment_role_is_result_like(role) &&
         expected_element_type == LOOM_SCALAR_TYPE_F32 &&
         loom_scalar_type_set_contains(LOOM_SCALAR_TYPE_SET_16BIT_FLOAT,
                                       storage_element_type);
}

static bool loom_amdgpu_fragment_memory_can_extend_result_store(
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_scalar_type_t expected_element_type,
    loom_scalar_type_t storage_element_type) {
  return operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE &&
         loom_amdgpu_matrix_fragment_role_is_result_like(role) &&
         expected_element_type == LOOM_SCALAR_TYPE_F16 &&
         storage_element_type == LOOM_SCALAR_TYPE_F32;
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
    loom_value_fact_numeric_format_flags_t source_format,
    loom_scalar_type_t result_element_type) {
  if (descriptor_set == NULL) {
    return false;
  }
  loom_amdgpu_fp8_native_descriptor_refs_t native_refs = {0};
  if (loom_amdgpu_fp8_native_descriptor_refs(source_format, result_element_type,
                                             &native_refs) &&
      native_refs.pair != LOOM_AMDGPU_DESCRIPTOR_REF_NONE &&
      loom_amdgpu_descriptor_set_has_ref(descriptor_set, native_refs.pair)) {
    return true;
  }
  loom_amdgpu_descriptor_ref_t scale_descriptor_ref =
      LOOM_AMDGPU_DESCRIPTOR_REF_NONE;
  if (loom_amdgpu_fp8_scalef32_descriptor_ref(
          source_format, result_element_type, &scale_descriptor_ref) &&
      loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                         scale_descriptor_ref)) {
    return true;
  }
  return loom_amdgpu_fp8_e8m0_pk8_descriptor_ref(
             source_format, result_element_type, &scale_descriptor_ref) &&
         loom_amdgpu_descriptor_set_has_ref(descriptor_set,
                                            scale_descriptor_ref);
}

static loom_amdgpu_fragment_memory_view_numeric_t
loom_amdgpu_fragment_memory_view_numeric(
    const loom_value_fact_storage_schema_t* view_storage_schema,
    loom_scalar_type_t view_element_type) {
  const loom_value_fact_numeric_format_flags_t exact_format =
      view_storage_schema != NULL &&
              view_storage_schema->encoded_operand.element_format !=
                  LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE
          ? view_storage_schema->encoded_operand.element_format
          : loom_numeric_format_from_scalar_type(view_element_type);
  loom_amdgpu_fragment_memory_view_numeric_t numeric = {
      .exact_format = exact_format,
      .descriptor_format = exact_format,
      .content_facts = loom_value_facts_unknown(),
  };
  loom_scalar_type_fp8_format_t unused_format = {0};
  if (!loom_scalar_type_fp8_format(view_element_type, &unused_format)) {
    return numeric;
  }
  loom_encoding_query_storage_schema_content_facts(
      view_storage_schema, view_element_type, &numeric.content_facts);
  numeric.descriptor_format = loom_amdgpu_fp8_descriptor_source_format(
      numeric.exact_format, numeric.content_facts);
  return numeric;
}

static bool loom_amdgpu_fragment_memory_can_load_fp8_to_16bit(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_scalar_type_t expected_element_type,
    loom_scalar_type_t payload_element_type,
    loom_scalar_type_t view_element_type,
    const loom_amdgpu_fragment_memory_view_numeric_t* view_numeric,
    loom_amdgpu_fragment_memory_payload_form_t* out_payload_form) {
  *out_payload_form = LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
  loom_scalar_type_fp8_format_t unused_format = {0};
  if (operation_kind != LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD ||
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
    case LOOM_SCALAR_TYPE_F16: {
      if (!loom_amdgpu_fragment_memory_descriptor_set_has_fp8_to_16bit_native(
              descriptor_set, view_numeric->descriptor_format,
              expected_element_type) &&
          !loom_value_facts_is_finite(view_numeric->content_facts)) {
        return false;
      }
      *out_payload_form =
          LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_F16;
      return true;
    }
    default:
      return false;
  }
}

static bool loom_amdgpu_fragment_memory_can_load_packed_16bit_result(
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_scalar_type_t expected_element_type,
    loom_scalar_type_t payload_element_type,
    loom_scalar_type_t view_element_type) {
  return operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD &&
         loom_amdgpu_matrix_fragment_role_is_result_like(role) &&
         expected_element_type == LOOM_SCALAR_TYPE_F32 &&
         payload_element_type == view_element_type &&
         loom_amdgpu_fragment_memory_scalar_type_is_16bit_float(
             payload_element_type);
}

static bool loom_amdgpu_fragment_memory_payload_form_select(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_scalar_type_t expected_element_type,
    loom_scalar_type_t payload_element_type,
    loom_scalar_type_t view_element_type,
    const loom_amdgpu_fragment_memory_view_numeric_t* view_numeric,
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
          payload_element_type, view_element_type, view_numeric,
          out_payload_form)) {
    return true;
  }
  if ((payload_element_type == expected_element_type ||
       payload_element_type == view_element_type) &&
      loom_amdgpu_fragment_memory_can_narrow_result_store(
          operation_kind, role, expected_element_type, view_element_type)) {
    switch (view_element_type) {
      case LOOM_SCALAR_TYPE_BF16:
        *out_payload_form =
            LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_BF16;
        return true;
      case LOOM_SCALAR_TYPE_F16:
        *out_payload_form =
            LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_NARROW_F32_TO_F16;
        return true;
      default:
        IREE_ASSERT_UNREACHABLE("selected AMDGPU narrowed store element type");
        IREE_BUILTIN_UNREACHABLE();
    }
  }
  if (payload_element_type == expected_element_type &&
      loom_amdgpu_fragment_memory_can_extend_result_store(
          operation_kind, role, expected_element_type, view_element_type)) {
    *out_payload_form =
        LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_EXTEND_F16_TO_F32;
    return true;
  }
  return false;
}

static bool loom_amdgpu_fragment_memory_requires_native_payload_storage(
    loom_low_source_memory_operation_kind_t operation_kind,
    const loom_matrix_fragment_role_layout_t* role_layout,
    loom_amdgpu_fragment_memory_payload_form_t payload_form) {
  return operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE &&
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
    case LOOM_SCALAR_TYPE_I8:
      if ((info->kind != LOOM_NUMERIC_FORMAT_KIND_SIGNED_INTEGER &&
           info->kind != LOOM_NUMERIC_FORMAT_KIND_UNSIGNED_INTEGER) ||
          info->storage_bit_count != 8) {
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
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_scalar_type_t expected_element_type,
    const loom_matrix_fragment_role_layout_t* role_layout) {
  if (operation_kind != LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD ||
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
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_scalar_type_t expected_element_type,
    const loom_matrix_fragment_role_layout_t* role_layout) {
  if (role_layout == NULL || !loom_type_is_vector(payload_type) ||
      loom_type_rank(payload_type) != 1 ||
      !loom_type_is_all_static(payload_type)) {
    return false;
  }

  switch (role) {
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
              operation_kind, role, expected_element_type,
              loom_type_element_type(view_type))) {
        return false;
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

static bool loom_amdgpu_fragment_memory_sparsity_schema_matches(
    const loom_value_fact_storage_schema_t* storage_schema,
    const loom_matrix_fragment_role_layout_t* role_layout) {
  const bool role_is_compressed =
      role_layout->reduction_group.storage_element_count != 0;
  if (storage_schema == NULL) {
    return !role_is_compressed;
  }
  const loom_value_fact_encoded_operand_schema_t operand =
      storage_schema->encoded_operand;
  if (!role_is_compressed) {
    return operand.sparsity_policy == LOOM_VALUE_FACT_SPARSITY_POLICY_NONE;
  }
  return operand.sparsity_policy ==
             LOOM_VALUE_FACT_SPARSITY_POLICY_N_M_STRUCTURED &&
         operand.sparsity_group.nonzero_element_count ==
             role_layout->reduction_group.storage_element_count &&
         operand.sparsity_group.element_count ==
             role_layout->reduction_group.logical_element_count;
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

static bool loom_amdgpu_fragment_memory_is_packed_16bit_result_source(
    const loom_module_t* module, loom_value_id_t source,
    const loom_matrix_fragment_role_layout_t* role_layout,
    loom_scalar_type_t storage_element_type, uint16_t* out_register_count) {
  *out_register_count = 0;
  if (source >= module->values.count || role_layout == NULL ||
      role_layout->element_bit_count != 32) {
    return false;
  }
  const loom_type_t source_type = loom_module_value_type(module, source);
  loom_amdgpu_vector_storage_t storage = {0};
  if (!loom_amdgpu_type_vector_storage(source_type, &storage) ||
      storage.element_type != storage_element_type ||
      storage.element_bit_count != 16 ||
      storage.element_count != role_layout->register_count ||
      storage.register_count == 0 || storage.register_count > UINT16_MAX) {
    return false;
  }
  *out_register_count = (uint16_t)storage.register_count;
  return true;
}

static loom_value_id_t
loom_amdgpu_fragment_memory_same_lane_packed_16bit_source(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t payload,
    const loom_matrix_fragment_role_layout_t* role_layout,
    loom_scalar_type_t storage_element_type, uint16_t* out_register_count) {
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

  return loom_amdgpu_fragment_memory_is_packed_16bit_result_source(
             module, lane_origin.source_value_id, role_layout,
             storage_element_type, out_register_count)
             ? lane_origin.source_value_id
             : LOOM_VALUE_ID_INVALID;
}

static loom_amdgpu_fragment_memory_narrowed_result_sources_t
loom_amdgpu_fragment_memory_narrowed_result_sources(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_value_id_t payload,
    const loom_matrix_fragment_role_layout_t* role_layout,
    loom_scalar_type_t storage_element_type) {
  loom_amdgpu_fragment_memory_narrowed_result_sources_t sources = {
      .round_source = LOOM_VALUE_ID_INVALID,
      .scale_source = LOOM_VALUE_ID_INVALID,
      .packed_source = LOOM_VALUE_ID_INVALID,
  };
  sources.packed_source =
      loom_amdgpu_fragment_memory_same_lane_packed_16bit_source(
          module, fact_table, payload, role_layout, storage_element_type,
          &sources.packed_register_count);
  if (sources.packed_source != LOOM_VALUE_ID_INVALID) return sources;

  sources.round_source =
      loom_amdgpu_fragment_memory_is_f32_result_source(module, payload,
                                                       role_layout)
          ? payload
          : loom_amdgpu_fragment_memory_same_lane_round_source(
                module, fact_table, payload, role_layout);
  if (sources.round_source == LOOM_VALUE_ID_INVALID) return sources;

  loom_value_fact_uniform_scale_origin_t scale_origin = {0};
  if (loom_value_fact_table_query_uniform_scale_origin(
          fact_table, module, sources.round_source, &scale_origin) &&
      loom_amdgpu_fragment_memory_is_f32_result_source(
          module, scale_origin.source_value_id, role_layout)) {
    sources.round_source = scale_origin.source_value_id;
    sources.scale_source = scale_origin.scale_value_id;
  }
  return sources;
}

static loom_amdgpu_fragment_memory_layout_match_t
loom_amdgpu_fragment_memory_layout_match_rank(
    const loom_amdgpu_fragment_memory_environment_t* environment,
    loom_value_id_t payload,
    const loom_matrix_fragment_role_layout_t* role_layout,
    loom_amdgpu_fragment_memory_element_match_t element_match,
    loom_amdgpu_fragment_memory_payload_form_t payload_form,
    loom_amdgpu_fragment_memory_narrowed_result_sources_t*
        out_narrowed_result_sources) {
  if (element_match == LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NONE;
  }
  if (loom_amdgpu_fragment_memory_payload_form_is_store_narrow_f32_to_16bit(
          payload_form)) {
    const loom_scalar_type_t storage_element_type =
        loom_amdgpu_fragment_memory_store_narrow_result_element_type(
            payload_form);
    *out_narrowed_result_sources =
        loom_amdgpu_fragment_memory_narrowed_result_sources(
            environment->module, environment->fact_table, payload, role_layout,
            storage_element_type);
    if (out_narrowed_result_sources->round_source != LOOM_VALUE_ID_INVALID ||
        out_narrowed_result_sources->packed_source != LOOM_VALUE_ID_INVALID) {
      return LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NARROWED_F32_RESULT;
    }
    // A matching 16-bit payload width does not prove that the value carries
    // this f32 result layout. Only the result itself or an exact same-lane
    // derivative can select a narrowed f32 publication.
    return LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NONE;
  }
  return element_match == LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_EXACT
             ? LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_EXACT
             : LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_ADAPTED;
}

static loom_amdgpu_fragment_memory_element_match_t
loom_amdgpu_fragment_memory_payload_element_match(
    const loom_low_descriptor_set_t* descriptor_set,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_contract_operand_role_t role, loom_type_t payload_type,
    loom_type_t view_type,
    const loom_value_fact_storage_schema_t* view_storage_schema,
    const loom_amdgpu_fragment_memory_view_numeric_t* view_numeric,
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
          payload_element_type, view_element_type, view_numeric,
          out_payload_form)) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE;
  }
  return *out_payload_form == LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE
             ? LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_EXACT
             : LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_ADAPTED;
}

typedef enum loom_amdgpu_fragment_memory_layout_rejection_flag_bits_e {
  LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_REJECTION_FLAG_NONE = 0u,
  LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_REJECTION_FLAG_SHAPE = 1u << 0,
  LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_REJECTION_FLAG_PAYLOAD_LAYOUT = 1u << 1,
  LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_REJECTION_FLAG_SPARSITY_LAYOUT = 1u << 2,
  LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_REJECTION_FLAG_PAYLOAD_FORM = 1u << 3,
} loom_amdgpu_fragment_memory_layout_rejection_flag_bits_t;
typedef uint8_t loom_amdgpu_fragment_memory_layout_rejection_flags_t;

static bool loom_amdgpu_fragment_memory_shape_matches_representation(
    const loom_amdgpu_fragment_memory_environment_t* environment,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_contract_operand_role_t role,
    loom_amdgpu_matrix_result_representation_flags_t representation_flags,
    loom_value_id_t blocks, loom_value_id_t rows, loom_value_id_t columns) {
  return loom_amdgpu_matrix_fragment_tile_shape_matches(
      environment->fact_table,
      loom_amdgpu_matrix_fragment_source_tile_shape(layout, role,
                                                    representation_flags),
      role, blocks, rows, columns);
}

static loom_amdgpu_fragment_memory_layout_match_t
loom_amdgpu_fragment_memory_evaluate_layout(
    const loom_amdgpu_fragment_memory_environment_t* environment,
    loom_contract_operand_role_t role,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_value_id_t payload, loom_type_t payload_type, loom_type_t view_type,
    loom_value_id_t blocks, loom_value_id_t rows, loom_value_id_t columns,
    const loom_value_fact_storage_schema_t* view_storage_schema,
    const loom_amdgpu_fragment_memory_view_numeric_t* view_numeric,
    const loom_amdgpu_matrix_fragment_layout_t* layout,
    loom_amdgpu_matrix_numeric_type_t numeric_type,
    loom_scalar_type_t expected_element_type,
    loom_amdgpu_matrix_result_representation_flags_t representation_flags,
    loom_amdgpu_fragment_memory_payload_form_t* out_payload_form,
    loom_amdgpu_fragment_memory_layout_rejection_flags_t* rejection_flags,
    loom_amdgpu_fragment_memory_narrowed_result_sources_t*
        out_narrowed_result_sources) {
  *out_payload_form = LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
  *out_narrowed_result_sources =
      (loom_amdgpu_fragment_memory_narrowed_result_sources_t){
          .round_source = LOOM_VALUE_ID_INVALID,
          .scale_source = LOOM_VALUE_ID_INVALID,
          .packed_source = LOOM_VALUE_ID_INVALID,
      };
  const loom_matrix_fragment_role_layout_t* role_layout =
      loom_matrix_fragment_role_layout(layout, role);
  if (role_layout == NULL) return LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NONE;
  const bool source_is_sparse =
      view_storage_schema != NULL &&
      iree_any_bit_set(view_storage_schema->encoded_operand.sparsity_policy,
                       LOOM_VALUE_FACT_SPARSITY_POLICY_ALL);
  const bool role_is_sparse =
      role_layout->reduction_group.storage_element_count != 0;
  if (source_is_sparse != role_is_sparse) {
    return LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NONE;
  }
  if (!loom_amdgpu_fragment_memory_numeric_schema_matches(
          numeric_type, view_storage_schema)) {
    *rejection_flags |=
        LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_REJECTION_FLAG_PAYLOAD_LAYOUT;
    return LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NONE;
  }
  const bool shape_matches =
      loom_amdgpu_fragment_memory_shape_matches_representation(
          environment, layout, role, representation_flags, blocks, rows,
          columns);
  const bool payload_matches = loom_amdgpu_fragment_memory_payload_matches(
      payload_type, view_type, view_storage_schema, operation_kind, role,
      expected_element_type, role_layout);
  const loom_amdgpu_fragment_memory_element_match_t element_match =
      loom_amdgpu_fragment_memory_payload_element_match(
          environment->descriptor_set, operation_kind, role, payload_type,
          view_type, view_storage_schema, view_numeric, expected_element_type,
          role_layout, out_payload_form);
  if (!shape_matches) {
    if (payload_matches &&
        element_match != LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE) {
      *rejection_flags |=
          LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_REJECTION_FLAG_SHAPE;
    }
    return LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NONE;
  }
  if (!payload_matches) {
    if (element_match != LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE) {
      *rejection_flags |=
          LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_REJECTION_FLAG_PAYLOAD_LAYOUT;
    }
    return LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NONE;
  }
  if (element_match == LOOM_AMDGPU_FRAGMENT_MEMORY_ELEMENT_MATCH_NONE) {
    *rejection_flags |=
        LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_REJECTION_FLAG_PAYLOAD_FORM;
    return LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NONE;
  }
  if (!loom_amdgpu_fragment_memory_sparsity_schema_matches(view_storage_schema,
                                                           role_layout)) {
    *rejection_flags |=
        LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_REJECTION_FLAG_SPARSITY_LAYOUT;
    return LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NONE;
  }
  return loom_amdgpu_fragment_memory_layout_match_rank(
      environment, payload, role_layout, element_match, *out_payload_form,
      out_narrowed_result_sources);
}

static bool loom_amdgpu_fragment_memory_reject_layout(
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_amdgpu_fragment_memory_layout_rejection_flags_t rejection_flags,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
  iree_string_view_t constraint_key = IREE_SV("fragment_memory.target_layout");
  if (iree_any_bit_set(
          rejection_flags,
          LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_REJECTION_FLAG_PAYLOAD_FORM)) {
    constraint_key = operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE
                         ? IREE_SV("fragment_memory.store_conversion")
                         : IREE_SV("fragment_memory.payload_form");
  } else if (
      iree_any_bit_set(
          rejection_flags,
          LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_REJECTION_FLAG_SPARSITY_LAYOUT)) {
    constraint_key = IREE_SV("fragment_memory.sparsity_layout");
  } else if (
      iree_any_bit_set(
          rejection_flags,
          LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_REJECTION_FLAG_PAYLOAD_LAYOUT)) {
    constraint_key = IREE_SV("fragment_memory.payload_layout");
  } else if (iree_any_bit_set(
                 rejection_flags,
                 LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_REJECTION_FLAG_SHAPE)) {
    constraint_key = IREE_SV("fragment_memory.shape");
  }
  return loom_amdgpu_fragment_memory_reject(diagnostic, constraint_key);
}

static bool loom_amdgpu_fragment_memory_target_layout(
    const loom_amdgpu_fragment_memory_environment_t* environment,
    loom_contract_operand_role_t role,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_value_id_t payload, loom_type_t payload_type, loom_type_t view_type,
    loom_value_id_t blocks, loom_value_id_t rows, loom_value_id_t columns,
    const loom_value_fact_storage_schema_t* view_storage_schema,
    const loom_amdgpu_fragment_memory_view_numeric_t* view_numeric,
    loom_amdgpu_matrix_result_representation_id_t required_representation,
    const loom_amdgpu_matrix_fragment_layout_t** out_layout,
    loom_amdgpu_matrix_result_representation_flags_t* out_representation_flags,
    loom_amdgpu_fragment_memory_payload_form_t* out_payload_form,
    loom_amdgpu_fragment_memory_narrowed_result_sources_t*
        out_narrowed_result_sources,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
  *out_layout = NULL;
  *out_representation_flags = 0;
  *out_payload_form = LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
  *out_narrowed_result_sources =
      (loom_amdgpu_fragment_memory_narrowed_result_sources_t){
          .round_source = LOOM_VALUE_ID_INVALID,
          .scale_source = LOOM_VALUE_ID_INVALID,
          .packed_source = LOOM_VALUE_ID_INVALID,
      };
  if (environment->bundle == NULL || environment->bundle->snapshot == NULL ||
      environment->descriptor_set == NULL) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.target_layout"));
  }

  loom_amdgpu_fragment_memory_layout_rejection_flags_t rejection_flags =
      LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_REJECTION_FLAG_NONE;
  if (required_representation !=
      LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_NONE) {
    const loom_amdgpu_matrix_result_representation_t* representation =
        loom_amdgpu_matrix_result_representation_at(required_representation);
    IREE_ASSERT(representation != NULL,
                "selected matrix representation must name a generated row");
    const loom_amdgpu_matrix_fragment_layout_t* layout =
        loom_amdgpu_matrix_fragment_layout_for_kind(
            (loom_amdgpu_matrix_fragment_layout_kind_t)
                representation->fragment_layout_kind);
    IREE_ASSERT(layout != NULL,
                "selected matrix representation must name a fragment layout");
    IREE_ASSERT_TRUE(loom_amdgpu_matrix_fragment_role_is_result_like(role));
    loom_scalar_type_t expected_element_type = LOOM_SCALAR_TYPE_NONE;
    const bool has_expected_element_type =
        loom_amdgpu_matrix_fragment_scalar_type_from_numeric(
            (loom_amdgpu_matrix_numeric_type_t)representation->numeric_type,
            &expected_element_type);
    IREE_ASSERT_TRUE(has_expected_element_type);
    loom_amdgpu_fragment_memory_payload_form_t payload_form =
        LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
    const loom_amdgpu_fragment_memory_layout_match_t match =
        loom_amdgpu_fragment_memory_evaluate_layout(
            environment, role, operation_kind, payload, payload_type, view_type,
            blocks, rows, columns, view_storage_schema, view_numeric, layout,
            (loom_amdgpu_matrix_numeric_type_t)representation->numeric_type,
            expected_element_type, representation->flags, &payload_form,
            &rejection_flags, out_narrowed_result_sources);
    if (match == LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NONE) {
      return loom_amdgpu_fragment_memory_reject_layout(
          operation_kind, rejection_flags, diagnostic);
    }
    *out_layout = layout;
    *out_representation_flags = representation->flags;
    *out_payload_form = payload_form;
    return true;
  }

  const loom_amdgpu_matrix_fragment_contract_candidates_t* candidates =
      environment->contract_candidates;
  const iree_host_size_t descriptor_count =
      loom_amdgpu_matrix_fragment_contract_candidate_count(candidates);
  const uint32_t wave_size = candidates != NULL
                                 ? candidates->wave_size
                                 : environment->bundle->snapshot->subgroup_size;
  const loom_amdgpu_matrix_fragment_layout_t* best_layout = NULL;
  loom_amdgpu_fragment_memory_payload_form_t best_payload_form =
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
  loom_amdgpu_fragment_memory_layout_match_t best_match =
      LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NONE;
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
    const loom_scalar_type_t expected_element_type = role_storage.element_type;
    const loom_amdgpu_matrix_fragment_layout_t* layout =
        loom_amdgpu_matrix_contract_descriptor_fragment_layout(descriptor);
    if (layout == NULL) {
      continue;
    }
    loom_amdgpu_fragment_memory_payload_form_t payload_form =
        LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
    loom_amdgpu_fragment_memory_narrowed_result_sources_t
        narrowed_result_sources;
    const loom_amdgpu_fragment_memory_layout_match_t layout_match =
        loom_amdgpu_fragment_memory_evaluate_layout(
            environment, role, operation_kind, payload, payload_type, view_type,
            blocks, rows, columns, view_storage_schema, view_numeric, layout,
            role_storage.payload.numeric_type, expected_element_type,
            /*representation_flags=*/0, &payload_form, &rejection_flags,
            &narrowed_result_sources);
    if (layout_match <= best_match) {
      continue;
    }
    best_layout = layout;
    best_payload_form = payload_form;
    *out_narrowed_result_sources = narrowed_result_sources;
    best_match = layout_match;
    if (layout_match ==
        LOOM_AMDGPU_FRAGMENT_MEMORY_LAYOUT_MATCH_NARROWED_F32_RESULT) {
      break;
    }
  }
  if (best_layout != NULL) {
    *out_layout = best_layout;
    *out_payload_form = best_payload_form;
    return true;
  }

  return loom_amdgpu_fragment_memory_reject_layout(operation_kind,
                                                   rejection_flags, diagnostic);
}

static bool loom_amdgpu_fragment_memory_describe_view(
    const loom_module_t* module, const loom_value_fact_table_t* fact_table,
    loom_type_t view_type, uint8_t expected_view_rank,
    loom_vector_memory_access_t* out_access) {
  *out_access = (loom_vector_memory_access_t){0};
  if (!loom_type_is_view(view_type) ||
      loom_type_rank(view_type) != expected_view_rank) {
    return false;
  }
  const loom_scalar_type_t view_element_type =
      loom_type_element_type(view_type);
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

static bool loom_amdgpu_fragment_memory_query_view_strides(
    const loom_value_fact_table_t* fact_table,
    const loom_vector_memory_access_t* access,
    loom_low_source_memory_axis_byte_stride_t* out_axis_byte_strides,
    uint32_t* out_static_axis_byte_strides,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
  for (uint8_t axis = 0; axis < access->view_rank; ++axis) {
    loom_low_source_memory_axis_byte_stride_t* axis_stride =
        &out_axis_byte_strides[axis];
    loom_low_source_memory_query_axis_byte_stride(fact_table, access, axis,
                                                  axis_stride);
    if ((axis_stride->kind != LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_STATIC &&
         axis_stride->kind !=
             LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_DYNAMIC) ||
        axis_stride->static_byte_coefficient <= 0 ||
        axis_stride->static_byte_coefficient > UINT32_MAX ||
        !loom_value_facts_fit_unsigned_bit_count(axis_stride->byte_facts, 32) ||
        axis_stride->byte_facts.range_lo <= 0) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.view_stride"));
    }
    if (axis_stride->kind == LOOM_LOW_SOURCE_MEMORY_AXIS_BYTE_STRIDE_STATIC) {
      out_static_axis_byte_strides[axis] =
          (uint32_t)axis_stride->static_byte_coefficient;
    }
  }
  return true;
}

static void loom_amdgpu_fragment_memory_source_from_op(
    const loom_op_t* source_op,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_amdgpu_fragment_memory_source_t* out_source) {
  *out_source = (loom_amdgpu_fragment_memory_source_t){
      .vector_role = LOOM_VECTOR_ROLE_COUNT_,
      .view = LOOM_VALUE_ID_INVALID,
      .payload = LOOM_VALUE_ID_INVALID,
      .blocks = LOOM_VALUE_ID_INVALID,
      .rows = LOOM_VALUE_ID_INVALID,
      .columns = LOOM_VALUE_ID_INVALID,
      .static_indices = loom_attr_absent(),
      .dynamic_indices = {0},
      .cache_scope = loom_attr_absent(),
      .cache_temporal = loom_attr_absent(),
  };
  if (operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD) {
    out_source->vector_role = loom_vector_fragment_load_role(source_op);
    out_source->view = loom_vector_fragment_load_view(source_op);
    out_source->payload = loom_vector_fragment_load_result(source_op);
    out_source->blocks = loom_vector_fragment_load_blocks(source_op);
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
  out_source->blocks = loom_vector_fragment_store_blocks(source_op);
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

static bool loom_amdgpu_fragment_memory_fp8_load_scale_source(
    const loom_value_fact_table_t* fact_table, loom_value_id_t payload,
    loom_amdgpu_fragment_memory_payload_form_t payload_form,
    loom_scalar_type_t view_element_type,
    const loom_value_fact_storage_schema_t* view_storage_schema,
    const loom_matrix_fragment_role_layout_t* role_layout,
    loom_value_id_t* out_scale_source) {
  *out_scale_source = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_fragment_memory_payload_form_is_load_fp8_to_16bit(
          payload_form) ||
      view_storage_schema == NULL) {
    return true;
  }

  const loom_value_fact_encoded_operand_schema_t schema =
      view_storage_schema->encoded_operand;
  if (schema.scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE &&
      schema.secondary_scale_format == LOOM_VALUE_FACT_NUMERIC_FORMAT_NONE &&
      schema.affine_policy == LOOM_VALUE_FACT_AFFINE_POLICY_NONE) {
    return true;
  }
  if (payload_form !=
          LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_LOAD_FP8_TO_BF16 ||
      !loom_amdgpu_fp8_encoded_operand_schema_matches(
          schema, view_element_type, role_layout->payload_element_count,
          LOOM_AMDGPU_FP8_ENCODED_OPERAND_SCHEMA_KIND_SCALE_F32) ||
      fact_table == NULL) {
    return false;
  }

  loom_vector_fragment_fact_t fragment_fact;
  if (!loom_vector_fragment_fact_query_value_facts(
          &fact_table->context,
          loom_value_fact_table_lookup(fact_table, payload), &fragment_fact)) {
    return false;
  }
  if (!iree_any_bit_set(fragment_fact.auxiliary.present_keys,
                        loom_encoding_auxiliary_key_flag(
                            LOOM_ENCODING_AUXILIARY_KEY_SCALE))) {
    return false;
  }
  const loom_value_id_t scale_source =
      fragment_fact.auxiliary.values[LOOM_ENCODING_AUXILIARY_KEY_SCALE];
  *out_scale_source = scale_source;
  return true;
}

static bool loom_amdgpu_fragment_memory_dynamic_base_is_subgroup_uniform(
    const loom_low_source_memory_access_plan_t* source) {
  for (uint8_t i = 0; i < source->dynamic_term_count; ++i) {
    if (!loom_value_facts_is_subgroup_uniform(
            source->dynamic_terms[i].byte_facts)) {
      return false;
    }
  }
  return true;
}

static bool loom_amdgpu_fragment_memory_prepare(
    const loom_amdgpu_fragment_memory_environment_t* environment,
    const loom_amdgpu_fragment_memory_source_t* source,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_amdgpu_fragment_memory_prepared_t* out_prepared,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
  *out_prepared = (loom_amdgpu_fragment_memory_prepared_t){0};
  if (!loom_kernel_def_isa(environment->source_function.op)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.kernel_entry"));
  }
  if (!loom_attr_is_absent(source->cache_scope) ||
      !loom_attr_is_absent(source->cache_temporal)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.cache_policy"));
  }
  if (!loom_amdgpu_fragment_memory_role_from_vector_role(source->vector_role,
                                                         &out_prepared->role)) {
    return loom_amdgpu_fragment_memory_reject(diagnostic,
                                              IREE_SV("fragment_memory.role"));
  }

  out_prepared->payload_type =
      loom_module_value_type(environment->module, source->payload);
  out_prepared->view_type =
      loom_module_value_type(environment->module, source->view);
  const loom_fact_context_t* fact_context =
      environment->fact_table != NULL ? &environment->fact_table->context
                                      : NULL;
  out_prepared->has_view_storage_schema =
      loom_encoding_query_type_storage_schema(
          fact_context, environment->module, out_prepared->view_type,
          &out_prepared->view_storage_schema) &&
      !loom_value_fact_encoded_operand_schema_is_unknown(
          out_prepared->view_storage_schema.encoded_operand);
  out_prepared->view_element_type =
      loom_type_element_type(out_prepared->view_type);
  out_prepared->view_numeric = loom_amdgpu_fragment_memory_view_numeric(
      out_prepared->has_view_storage_schema ? &out_prepared->view_storage_schema
                                            : NULL,
      out_prepared->view_element_type);
  if (!loom_amdgpu_fragment_memory_payload_storage_register_count(
          out_prepared->payload_type, &out_prepared->payload_register_count)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.payload_storage"));
  }

  const iree_host_size_t view_rank = loom_type_rank(out_prepared->view_type);
  if (view_rank > LOOM_AMDGPU_FRAGMENT_MEMORY_VIEW_RANK_CAPACITY ||
      !loom_amdgpu_fragment_memory_describe_view(
          environment->module, environment->fact_table, out_prepared->view_type,
          (uint8_t)view_rank, &out_prepared->access)) {
    return loom_amdgpu_fragment_memory_reject(diagnostic,
                                              IREE_SV("fragment_memory.view"));
  }
  if (!loom_amdgpu_fragment_memory_query_view_strides(
          environment->fact_table, &out_prepared->access,
          out_prepared->axis_byte_strides,
          out_prepared->static_axis_byte_strides, diagnostic)) {
    return false;
  }

  const loom_type_t scalar_vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, out_prepared->view_element_type,
                          loom_dim_pack_static(1), /*encoding_id=*/0);
  loom_low_source_memory_access_diagnostic_t source_diagnostic = {0};
  if (!loom_low_source_memory_access_plan_build_indexed(
          environment->view_regions, operation_kind, source->view,
          source->dynamic_indices, source->static_indices, scalar_vector_type,
          (loom_vector_memory_cache_policy_t){0}, &out_prepared->source_access,
          &source_diagnostic)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, loom_low_source_memory_access_rejection_key(
                        source_diagnostic.rejection_bits));
  }
  if (out_prepared->source_access.element_byte_count > UINT16_MAX) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("source_memory.element_width"));
  }
  if (out_prepared->source_access.memory_space ==
      LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP) {
    uint64_t root_byte_offset = 0;
    if (!loom_amdgpu_source_alloca_layout_lookup_root(
            environment->alloca_layout, LOOM_VALUE_FACT_MEMORY_SPACE_WORKGROUP,
            out_prepared->source_access.root_value_id, &root_byte_offset) ||
        root_byte_offset > INT64_MAX) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.workgroup_root"));
    }
    if (!iree_checked_add_i64(
            out_prepared->source_access.static_byte_offset,
            (int64_t)root_byte_offset,
            &out_prepared->source_access.static_byte_offset)) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.base_offset"));
    }
  }
  return loom_amdgpu_fragment_memory_source_plan_supports_addressing(
      &out_prepared->source_access,
      diagnostic != NULL ? &diagnostic->constraint_key : NULL);
}

static bool loom_amdgpu_fragment_memory_evaluate_prepared(
    const loom_amdgpu_fragment_memory_environment_t* environment,
    const loom_amdgpu_fragment_memory_source_t* source,
    loom_low_source_memory_operation_kind_t operation_kind,
    const loom_amdgpu_fragment_memory_prepared_t* prepared,
    loom_amdgpu_matrix_result_representation_id_t required_representation,
    loom_amdgpu_fragment_memory_plan_t* out_plan,
    loom_amdgpu_fragment_memory_publication_choice_t* out_publication_choice,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
  if (out_plan != NULL) {
    *out_plan = (loom_amdgpu_fragment_memory_plan_t){0};
  }
  if (out_publication_choice != NULL) {
    out_publication_choice->strategy =
        LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_NONE;
  }
  const loom_amdgpu_matrix_fragment_layout_t* layout = NULL;
  loom_amdgpu_matrix_result_representation_flags_t representation_flags = 0;
  loom_amdgpu_fragment_memory_payload_form_t payload_form =
      LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_NATIVE;
  loom_amdgpu_fragment_memory_narrowed_result_sources_t narrowed_result_sources;
  if (!loom_amdgpu_fragment_memory_target_layout(
          environment, prepared->role, operation_kind, source->payload,
          prepared->payload_type, prepared->view_type, source->blocks,
          source->rows, source->columns,
          prepared->has_view_storage_schema ? &prepared->view_storage_schema
                                            : NULL,
          &prepared->view_numeric, required_representation, &layout,
          &representation_flags, &payload_form, &narrowed_result_sources,
          diagnostic)) {
    return false;
  }

  const loom_matrix_fragment_role_layout_t* role_layout =
      loom_matrix_fragment_role_layout(layout, prepared->role);
  const uint8_t expected_view_rank =
      layout->tile_shape.block_count > 1
          ? LOOM_AMDGPU_FRAGMENT_BLOCKED_VIEW_RANK
          : LOOM_AMDGPU_FRAGMENT_UNBLOCKED_VIEW_RANK;
  if (prepared->access.view_rank != expected_view_rank) {
    return loom_amdgpu_fragment_memory_reject(diagnostic,
                                              IREE_SV("fragment_memory.view"));
  }
  loom_value_id_t fp8_load_scale_source = LOOM_VALUE_ID_INVALID;
  if (!loom_amdgpu_fragment_memory_fp8_load_scale_source(
          environment->fact_table, source->payload, payload_form,
          prepared->view_element_type,
          prepared->has_view_storage_schema ? &prepared->view_storage_schema
                                            : NULL,
          role_layout, &fp8_load_scale_source)) {
    return loom_amdgpu_fragment_memory_reject(diagnostic,
                                              IREE_SV("fragment_memory.scale"));
  }
  uint16_t payload_register_count = prepared->payload_register_count;
  if (loom_amdgpu_fragment_memory_requires_native_payload_storage(
          operation_kind, role_layout, payload_form) &&
      !loom_amdgpu_fragment_memory_payload_has_native_storage(
          environment->fact_table, source->payload)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.payload_storage"));
  }

  loom_amdgpu_fragment_memory_address_layout_t address_layout = {0};
  loom_amdgpu_fragment_memory_runtime_axis_t
      runtime_axes[LOOM_AMDGPU_FRAGMENT_MEMORY_VIEW_RANK_CAPACITY] = {0};
  if (!loom_amdgpu_fragment_memory_compile_address_layout(
          prepared->role, role_layout, representation_flags,
          prepared->access.view_rank, prepared->axis_byte_strides,
          &address_layout, runtime_axes,
          diagnostic != NULL ? &diagnostic->constraint_key : NULL)) {
    return false;
  }
  loom_amdgpu_fragment_memory_packetization_t packetization =
      LOOM_AMDGPU_FRAGMENT_MEMORY_PACKETIZATION_NATIVE;
  if (!loom_amdgpu_fragment_memory_select_packetization(
          prepared->role, role_layout, payload_form,
          (uint16_t)prepared->access.static_element_byte_count, &address_layout,
          runtime_axes, prepared->access.view_rank, &packetization,
          diagnostic != NULL ? &diagnostic->constraint_key : NULL)) {
    return false;
  }
  const bool store_narrow_f32_to_16bit =
      loom_amdgpu_fragment_memory_payload_form_is_store_narrow_f32_to_16bit(
          payload_form);
  if (payload_form ==
          LOOM_AMDGPU_FRAGMENT_MEMORY_PAYLOAD_FORM_STORE_EXTEND_F16_TO_F32 &&
      !loom_amdgpu_descriptor_set_has_ref(
          environment->descriptor_set,
          LOOM_AMDGPU_DESCRIPTOR_REF_V_CVT_F32_F16)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.store_conversion"));
  }

  if (!loom_amdgpu_fragment_memory_address_range_fits_u32(
          &prepared->source_access, &address_layout, runtime_axes,
          prepared->access.view_rank, layout->wave_size,
          role_layout->register_count,
          diagnostic != NULL ? &diagnostic->constraint_key : NULL)) {
    return false;
  }

  if (!loom_amdgpu_fragment_memory_space_supports_access(
          operation_kind, prepared->source_access.memory_space, packetization,
          payload_form)) {
    return loom_amdgpu_fragment_memory_reject(
        diagnostic, IREE_SV("fragment_memory.memory_space"));
  }

  if (narrowed_result_sources.packed_source != LOOM_VALUE_ID_INVALID) {
    payload_register_count = narrowed_result_sources.packed_register_count;
  }

  loom_amdgpu_fragment_memory_publication_choice_t local_publication_choice;
  loom_amdgpu_fragment_memory_publication_choice_t* publication_choice =
      out_publication_choice != NULL ? out_publication_choice
                                     : &local_publication_choice;
  if (store_narrow_f32_to_16bit) {
    loom_amdgpu_fragment_publication_source_flags_t publication_source_flags =
        LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_NONE;
    if (narrowed_result_sources.round_source != LOOM_VALUE_ID_INVALID) {
      publication_source_flags |=
          LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_ROUNDED;
    }
    if (narrowed_result_sources.scale_source != LOOM_VALUE_ID_INVALID) {
      publication_source_flags |=
          LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_SCALED;
    }
    if (narrowed_result_sources.packed_source != LOOM_VALUE_ID_INVALID) {
      publication_source_flags |=
          LOOM_AMDGPU_FRAGMENT_PUBLICATION_SOURCE_FLAG_PACKED;
    }
    const loom_amdgpu_fragment_memory_publication_query_t publication_query = {
        .descriptor_set = environment->descriptor_set,
        .layout = layout,
        .address_layout = &address_layout,
        .runtime_axes = runtime_axes,
        .static_axis_byte_strides = prepared->static_axis_byte_strides,
        .memory_space = prepared->source_access.memory_space,
        .role = prepared->role,
        .representation_flags = representation_flags,
        .payload_form = payload_form,
        .register_count = role_layout->register_count,
        .element_byte_count =
            (uint16_t)prepared->source_access.element_byte_count,
        .view_rank = prepared->access.view_rank,
        .source_flags = publication_source_flags,
    };
    if (!loom_amdgpu_fragment_memory_select_publication(&publication_query,
                                                        publication_choice)) {
      return loom_amdgpu_fragment_memory_reject(
          diagnostic, IREE_SV("fragment_memory.packet"));
    }
  } else {
    publication_choice = NULL;
  }

  if (out_plan != NULL) {
    *out_plan = (loom_amdgpu_fragment_memory_plan_t){
        .operation_kind = operation_kind,
        .role = prepared->role,
        .layout_kind = layout->kind,
        .source = prepared->source_access,
        .dynamic_base_is_subgroup_uniform =
            loom_amdgpu_fragment_memory_dynamic_base_is_subgroup_uniform(
                &prepared->source_access),
        .payload = source->payload,
        .fp8_load_scale_source = fp8_load_scale_source,
        .view_rank = prepared->access.view_rank,
        .representation_flags = representation_flags,
        .register_count = role_layout->register_count,
        .payload_register_count = payload_register_count,
        .element_byte_count =
            (uint16_t)prepared->source_access.element_byte_count,
        .view_element_type = prepared->view_element_type,
        .view_element_format = prepared->view_numeric.exact_format,
        .descriptor_source_format = prepared->view_numeric.descriptor_format,
        .address_layout = address_layout,
        .payload_form = payload_form,
        .packetization = packetization,
        .narrowed_result_round_source = narrowed_result_sources.round_source,
        .narrowed_result_scale_source = narrowed_result_sources.scale_source,
        .narrowed_result_packed_source = narrowed_result_sources.packed_source,
    };
    for (uint8_t axis = 0; axis < prepared->access.view_rank; ++axis) {
      out_plan->static_axis_byte_strides[axis] =
          prepared->static_axis_byte_strides[axis];
    }
    memcpy(out_plan->runtime_axes, runtime_axes, sizeof(runtime_axes));
    if (!loom_amdgpu_fragment_memory_plan_packets(
            environment->descriptor_set, layout, publication_choice, out_plan,
            diagnostic != NULL ? &diagnostic->constraint_key : NULL)) {
      return false;
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
    const loom_amdgpu_target_facts_t* target_facts,
    loom_func_like_t source_function, const loom_op_t* source_op,
    loom_low_source_memory_operation_kind_t operation_kind,
    loom_amdgpu_matrix_result_representation_id_t required_representation,
    loom_amdgpu_fragment_memory_plan_t* out_plan,
    loom_amdgpu_fragment_memory_diagnostic_t* diagnostic) {
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
              : loom_amdgpu_matrix_fragment_feature_bits(target_facts),
      .source_function = source_function,
  };
  loom_amdgpu_fragment_memory_source_t source = {0};
  loom_amdgpu_fragment_memory_source_from_op(source_op, operation_kind,
                                             &source);
  loom_amdgpu_fragment_memory_prepared_t prepared = {0};
  if (!loom_amdgpu_fragment_memory_prepare(
          &environment, &source, operation_kind, &prepared, diagnostic) ||
      !loom_amdgpu_fragment_memory_evaluate_prepared(
          &environment, &source, operation_kind, &prepared,
          required_representation, out_plan,
          /*out_publication_choice=*/NULL, diagnostic)) {
    return false;
  }
  return loom_amdgpu_fragment_memory_select_fp8_load_decode_plan(
      fact_table, descriptor_set, source_op, out_plan);
}

iree_status_t loom_amdgpu_query_accumulator_fragment_store_representations(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_representation_candidate_t* out_candidates,
    iree_host_size_t* out_candidate_count) {
  *out_candidate_count = 0;
  const loom_module_t* module = loom_low_lower_context_module(context);
  const loom_value_fact_table_t* fact_table =
      loom_low_lower_context_fact_table(context);
  const loom_view_region_table_t* view_regions = NULL;
  IREE_RETURN_IF_ERROR(
      loom_low_lower_context_view_regions(context, &view_regions));
  const loom_amdgpu_matrix_fragment_contract_candidates_t* contract_candidates =
      NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_matrix_fragment_contract_candidates(
      context, &contract_candidates));
  if (contract_candidates == NULL) return iree_ok_status();
  const loom_amdgpu_source_alloca_layout_t* alloca_layout = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_source_alloca_layout_for_lower_context(
      context, &alloca_layout));
  const loom_amdgpu_fragment_memory_environment_t environment = {
      .module = module,
      .fact_table = fact_table,
      .view_regions = view_regions,
      .bundle = loom_low_lower_context_bundle(context),
      .descriptor_set = loom_low_lower_context_descriptor_set(context),
      .contract_candidates = contract_candidates,
      .alloca_layout = alloca_layout,
      .feature_bits = contract_candidates->feature_bits,
      .source_function = loom_low_lower_context_source_function(context),
  };
  loom_amdgpu_fragment_memory_source_t source = {0};
  loom_amdgpu_fragment_memory_source_from_op(
      source_op, LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE, &source);
  loom_amdgpu_fragment_memory_prepared_t prepared = {0};
  if (!loom_amdgpu_fragment_memory_prepare(
          &environment, &source, LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE,
          &prepared, /*diagnostic=*/NULL)) {
    return iree_ok_status();
  }
  uint64_t representation_bits =
      contract_candidates->exact_result_representation_bits & ~UINT64_C(1);
  while (representation_bits != 0) {
    const loom_amdgpu_matrix_result_representation_id_t representation_id =
        (loom_amdgpu_matrix_result_representation_id_t)
            iree_math_count_trailing_zeros_u64(representation_bits);
    representation_bits &= representation_bits - 1u;
    loom_amdgpu_fragment_memory_publication_choice_t publication_choice;
    if (!loom_amdgpu_fragment_memory_evaluate_prepared(
            &environment, &source, LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE,
            &prepared, representation_id, /*out_plan=*/NULL,
            &publication_choice, /*diagnostic=*/NULL) ||
        publication_choice.strategy ==
            LOOM_AMDGPU_FRAGMENT_MEMORY_EPILOGUE_STRATEGY_NONE) {
      continue;
    }
    out_candidates[(*out_candidate_count)++] =
        (loom_low_representation_candidate_t){
            .representation = representation_id,
            .cost = publication_choice.cost,
        };
  }
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_fragment_memory_select(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_source_memory_operation_kind_t operation_kind,
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
  loom_amdgpu_matrix_result_representation_id_t required_representation =
      LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_NONE;
  if (operation_kind == LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE &&
      loom_vector_fragment_store_role(source_op) == LOOM_VECTOR_ROLE_RESULT) {
    loom_low_representation_id_t selected_representation =
        LOOM_LOW_REPRESENTATION_ID_NONE;
    IREE_RETURN_IF_ERROR(loom_low_lower_representation_lookup(
        context, loom_vector_fragment_store_value(source_op),
        &selected_representation));
    if (selected_representation != LOOM_LOW_REPRESENTATION_ID_NONE) {
      IREE_ASSERT_LE(selected_representation,
                     LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_MAX_ID);
      required_representation = (loom_amdgpu_matrix_result_representation_id_t)
          selected_representation;
    }
  }
  loom_amdgpu_fragment_memory_diagnostic_t diagnostic = {0};
  *out_selected = loom_amdgpu_analyze_vector_fragment_memory_plan_impl(
      module, loom_low_lower_context_fact_table(context), view_regions,
      loom_low_lower_context_bundle(context),
      loom_low_lower_context_descriptor_set(context), contract_candidates,
      alloca_layout,
      loom_amdgpu_target_facts_cast(
          loom_low_lower_context_target_facts(context)),
      loom_low_lower_context_source_function(context), source_op,
      operation_kind, required_representation, out_plan, &diagnostic);
  if (!*out_selected && required_representation !=
                            LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_NONE) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "selected matrix result representation %u cannot lower fragment "
        "memory operation: %.*s",
        required_representation, (int)diagnostic.constraint_key.size,
        diagnostic.constraint_key.data);
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_select_vector_fragment_load_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected) {
  return loom_amdgpu_fragment_memory_select(
      context, source_op, LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD, out_plan,
      out_selected);
}

iree_status_t loom_amdgpu_select_vector_fragment_store_plan(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_fragment_memory_plan_t* out_plan, bool* out_selected) {
  return loom_amdgpu_fragment_memory_select(
      context, source_op, LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE, out_plan,
      out_selected);
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

  loom_low_source_memory_operation_kind_t operation_kind =
      LOOM_LOW_SOURCE_MEMORY_OPERATION_LOAD;
  if (op->kind == LOOM_OP_VECTOR_FRAGMENT_STORE) {
    operation_kind = LOOM_LOW_SOURCE_MEMORY_OPERATION_STORE;
  } else if (op->kind != LOOM_OP_VECTOR_FRAGMENT_LOAD) {
    *out_handled = false;
    return iree_ok_status();
  }

  const loom_module_t* module = loom_target_low_legality_module(context);
  const loom_view_region_table_t* view_regions =
      loom_target_low_legality_view_regions(context);
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
      .feature_bits = loom_amdgpu_matrix_fragment_feature_bits(
          loom_amdgpu_target_facts_cast(
              loom_target_low_legality_target_facts(context))),
      .source_function = loom_target_low_legality_function(context),
  };
  loom_amdgpu_fragment_memory_source_t source = {0};
  loom_amdgpu_fragment_memory_source_from_op(op, operation_kind, &source);
  loom_amdgpu_fragment_memory_diagnostic_t diagnostic = {0};
  loom_amdgpu_fragment_memory_plan_t plan = {0};
  loom_amdgpu_fragment_memory_prepared_t prepared = {0};
  if (loom_amdgpu_fragment_memory_prepare(&environment, &source, operation_kind,
                                          &prepared, &diagnostic) &&
      loom_amdgpu_fragment_memory_evaluate_prepared(
          &environment, &source, operation_kind, &prepared,
          LOOM_AMDGPU_MATRIX_RESULT_REPRESENTATION_NONE, &plan,
          /*out_publication_choice=*/NULL, &diagnostic)) {
    return iree_ok_status();
  }
  iree_string_view_t constraint_key = diagnostic.constraint_key;
  if (iree_string_view_is_empty(constraint_key)) {
    constraint_key = IREE_SV("fragment_memory");
  }
  return loom_amdgpu_low_legality_reject(context, op, constraint_key);
}
