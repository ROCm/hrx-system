// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/target/registers.h"
#include "loom/target/test/contracts/core.h"
#include "loom/target/test/contracts/core_lower_rules.h"
#include "loom/target/test/descriptors.h"
#include "loom/target/test/lower.h"

//===----------------------------------------------------------------------===//
// Type mapping
//===----------------------------------------------------------------------===//

static bool loom_test_low_is_i32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32;
}

static bool loom_test_low_is_i8(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I8;
}

static bool loom_test_low_is_f32(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32;
}

static bool loom_test_low_is_index_like(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         (loom_type_element_type(type) == LOOM_SCALAR_TYPE_INDEX ||
          loom_type_element_type(type) == LOOM_SCALAR_TYPE_OFFSET);
}

static bool loom_test_low_is_i1(loom_type_t type) {
  return loom_type_is_scalar(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1;
}

static bool loom_test_low_is_vector_4xi32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_test_low_is_vector_4xf32(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_F32 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_test_low_is_vector_4xi1(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I1 &&
         loom_type_dim_static_size_at(type, 0) == 4;
}

static bool loom_test_low_is_vector_16xi8(loom_type_t type) {
  return loom_type_is_vector(type) && loom_type_rank(type) == 1 &&
         loom_type_is_all_static(type) &&
         loom_type_element_type(type) == LOOM_SCALAR_TYPE_I8 &&
         loom_type_dim_static_size_at(type, 0) == 16;
}

static iree_status_t loom_test_low_make_register_type(
    loom_low_lower_context_t* context, uint16_t register_class_id,
    uint32_t unit_count, loom_type_t* out_type) {
  *out_type = loom_low_register_type(
      loom_low_lower_context_descriptor_set(context)->stable_id,
      register_class_id, unit_count);
  return iree_ok_status();
}

iree_status_t loom_test_low_lower_map_type(void* user_data,
                                           loom_low_lower_context_t* context,
                                           const loom_op_t* source_op,
                                           loom_type_t source_type,
                                           loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_test_low_is_i32(source_type) || loom_test_low_is_i1(source_type) ||
      loom_test_low_is_index_like(source_type)) {
    return loom_test_low_make_register_type(
        context, TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 1, out_low_type);
  }
  if (loom_test_low_is_i8(source_type)) {
    return loom_test_low_make_register_type(
        context, TEST_LOW_CORE_REG_CLASS_ID_TEST_I8, 1, out_low_type);
  }
  if (loom_test_low_is_f32(source_type)) {
    return loom_test_low_make_register_type(
        context, TEST_LOW_CORE_REG_CLASS_ID_TEST_F32, 1, out_low_type);
  }
  if (loom_test_low_is_vector_4xi32(source_type)) {
    return loom_test_low_make_register_type(
        context, TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 4, out_low_type);
  }
  if (loom_test_low_is_vector_4xf32(source_type)) {
    return loom_test_low_make_register_type(
        context, TEST_LOW_CORE_REG_CLASS_ID_TEST_F32, 4, out_low_type);
  }
  if (loom_test_low_is_vector_4xi1(source_type)) {
    return loom_test_low_make_register_type(
        context, TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 4, out_low_type);
  }
  if (loom_test_low_is_vector_16xi8(source_type)) {
    return loom_test_low_make_register_type(
        context, TEST_LOW_CORE_REG_CLASS_ID_TEST_I8, 16, out_low_type);
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("source"), source_type);
}

iree_status_t loom_test_low_lower_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  if (loom_type_is_buffer(source_type)) {
    loom_type_t resource_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_test_low_make_register_type(
        context, TEST_LOW_CORE_REG_CLASS_ID_TEST_PTR, 1, &resource_type));
    *out_argument = (loom_low_lower_abi_argument_t){
        .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_RESOURCE,
        .abi_type = resource_type,
        .resource_import_kind = LOOM_LOW_RESOURCE_IMPORT_KIND_NATIVE_POINTER,
        .resource_index = source_argument_index,
        .resource_source_type = source_type,
    };
    return iree_ok_status();
  }

  *out_argument = (loom_low_lower_abi_argument_t){
      .kind = LOOM_LOW_LOWER_ABI_ARGUMENT_DIRECT,
      .abi_type = loom_type_none(),
      .resource_source_type = loom_type_none(),
  };
  return loom_test_low_lower_map_type(user_data, context, source_function_op,
                                      source_type, &out_argument->abi_type);
}

iree_status_t loom_test_low_lower_rule_match_map_value(
    void* user_data, const loom_low_lower_rule_match_context_t* context,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)user_data;
  (void)source_op;
  const loom_target_contract_query_environment_t environment = {
      .module = context->module,
      .descriptor_set = context->descriptor_set,
  };
  return loom_test_low_lower_map_contract_value(
      NULL, &environment, source_op, source_value_id, out_mapped_value);
}

iree_status_t loom_test_low_lower_map_contract_value(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_op_t* source_op, loom_value_id_t source_value_id,
    loom_low_lower_rule_mapped_value_t* out_mapped_value) {
  (void)user_data;
  (void)source_op;
  *out_mapped_value = loom_low_lower_rule_mapped_value_none();
  loom_type_t source_type =
      loom_module_value_type(environment->module, source_value_id);
  if (loom_test_low_is_i32(source_type) || loom_test_low_is_i1(source_type) ||
      loom_test_low_is_index_like(source_type)) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 1);
  } else if (loom_test_low_is_i8(source_type)) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        TEST_LOW_CORE_REG_CLASS_ID_TEST_I8, 1);
  } else if (loom_test_low_is_f32(source_type)) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        TEST_LOW_CORE_REG_CLASS_ID_TEST_F32, 1);
  } else if (loom_test_low_is_vector_4xi32(source_type)) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 4);
  } else if (loom_test_low_is_vector_4xf32(source_type)) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        TEST_LOW_CORE_REG_CLASS_ID_TEST_F32, 4);
  } else if (loom_test_low_is_vector_4xi1(source_type)) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        TEST_LOW_CORE_REG_CLASS_ID_TEST_I32, 4);
  } else if (loom_test_low_is_vector_16xi8(source_type)) {
    *out_mapped_value = loom_low_lower_rule_mapped_value_register(
        TEST_LOW_CORE_REG_CLASS_ID_TEST_I8, 16);
  }
  return iree_ok_status();
}

static iree_status_t loom_test_low_matrix_options(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    loom_contract_vector_mma_options_t* out_options) {
  (void)user_data;
  (void)environment;
  (void)rule;
  *out_options = (loom_contract_vector_mma_options_t){
      .k_group_size = 1,
      .fragment =
          {
              .atom_bits = LOOM_CONTRACT_FRAGMENT_SUBGROUP_LANE,
              .subgroup_size = 2,
          },
      .capability_class = LOOM_CONTRACT_CAPABILITY_CLASS_GPU_MATRIX,
      .policy = LOOM_LOWERING_POLICY_TARGET_PRIMITIVE_REQUIRED,
  };
  return iree_ok_status();
}

static bool loom_test_low_matrix_request_is_tiny_i32(
    const loom_contract_request_t* request) {
  return request->shape.m == 2 && request->shape.n == 2 &&
         request->shape.k == 2 && request->lhs.payload_element_count == 4 &&
         request->rhs.payload_element_count == 4 &&
         request->accumulator.payload_element_count == 4 &&
         request->result.payload_element_count == 4 &&
         request->lhs.numeric_type == LOOM_CONTRACT_NUMERIC_I32 &&
         request->rhs.numeric_type == LOOM_CONTRACT_NUMERIC_I32 &&
         request->accumulator.numeric_type == LOOM_CONTRACT_NUMERIC_I32 &&
         request->result.numeric_type == LOOM_CONTRACT_NUMERIC_I32;
}

static bool loom_test_low_matrix_target_is_quirky(
    const loom_target_contract_query_environment_t* environment) {
  return environment->bundle != NULL && environment->bundle->snapshot != NULL &&
         environment->bundle->snapshot->subgroup_size == 7;
}

static iree_status_t loom_test_low_matrix_query(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, const loom_contract_request_t* request,
    loom_target_contract_query_result_t* out_result) {
  (void)user_data;
  (void)rule;
  (void)source_op;
  *out_result = loom_target_contract_query_result_empty();
  if (!loom_test_low_matrix_request_is_tiny_i32(request)) {
    return iree_ok_status();
  }
  if (!loom_test_low_matrix_target_is_quirky(environment)) {
    out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_UNSUPPORTED;
    return iree_ok_status();
  }

  const loom_low_descriptor_t* descriptor =
      loom_low_descriptor_set_descriptor_at(
          environment->descriptor_set,
          TEST_LOW_CORE_DESCRIPTOR_REF_TEST_MMA_I32_2X2X2);
  if (descriptor == NULL) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "test-low matrix descriptor ref missing");
  }
  out_result->outcome = LOOM_TARGET_CONTRACT_QUERY_LEGAL;
  out_result->selected_descriptor = descriptor;
  return iree_ok_status();
}

static const loom_low_lower_rule_set_t* const kTestLowRuleSets[] = {
    &loom_test_low_core_lower_rule_set,
};

static const loom_target_contract_binding_t kTestLowContractBindings[] = {
    {
        .fragment = &loom_test_low_core_contract_fragment,
        .rule_set_index = 0,
    },
};

static const loom_low_lower_policy_t kTestLowLowerPolicy = {
    .name = IREE_SVL("test-low-lower-policy"),
    .error_catalog = &loom_error_catalog_core,
    .map_type = {.fn = loom_test_low_lower_map_type, .user_data = NULL},
    .map_contract_value = {.fn = loom_test_low_lower_map_contract_value,
                           .user_data = NULL},
    .map_argument = {.fn = loom_test_low_lower_map_argument, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kTestLowRuleSets),
            .values = kTestLowRuleSets,
        },
    .contract_bindings = kTestLowContractBindings,
    .contract_binding_count = IREE_ARRAYSIZE(kTestLowContractBindings),
    .descriptor_matrix =
        {
            .options = loom_test_low_matrix_options,
            .query = loom_test_low_matrix_query,
        },
};

const loom_low_lower_policy_t* loom_test_low_lower_policy(void) {
  return &kTestLowLowerPolicy;
}

void loom_test_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("test.low.core"),
          .policy = &kTestLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
