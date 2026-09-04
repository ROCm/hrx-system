// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/error_catalog.h"
#include "loom/ir/module.h"
#include "loom/ir/scalar_type.h"
#include "loom/target/arch/amd/xdna/aie2p/contracts/core.h"
#include "loom/target/arch/amd/xdna/aie2p/contracts/core_lower_rules.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/lower.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/matrix.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/storage.h"

static bool loom_aie2p_source_type_supported(void* user_data,
                                             const loom_module_t* module,
                                             loom_type_t source_type) {
  (void)user_data;
  (void)module;
  if (!loom_type_is_scalar(source_type) && !loom_type_is_vector(source_type)) {
    return false;
  }
  const loom_scalar_type_t element_type = loom_type_element_type(source_type);
  return element_type == LOOM_SCALAR_TYPE_F8E4M3 ||
         element_type == LOOM_SCALAR_TYPE_F8E5M2;
}

static iree_status_t loom_aie2p_map_type(void* user_data,
                                         loom_low_lower_context_t* context,
                                         const loom_op_t* source_op,
                                         loom_type_t source_type,
                                         loom_type_t* out_low_type) {
  (void)user_data;
  if (loom_type_is_scalar(source_type)) {
    switch (loom_type_element_type(source_type)) {
      case LOOM_SCALAR_TYPE_INDEX:
      case LOOM_SCALAR_TYPE_OFFSET:
      case LOOM_SCALAR_TYPE_I1:
      case LOOM_SCALAR_TYPE_I8:
      case LOOM_SCALAR_TYPE_I16:
      case LOOM_SCALAR_TYPE_I32:
      case LOOM_SCALAR_TYPE_F8E4M3:
      case LOOM_SCALAR_TYPE_F8E5M2:
      case LOOM_SCALAR_TYPE_F16:
      case LOOM_SCALAR_TYPE_BF16:
      case LOOM_SCALAR_TYPE_F32:
        return loom_low_lower_make_register_type(
            context, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 1, out_low_type);
      case LOOM_SCALAR_TYPE_I64:
      case LOOM_SCALAR_TYPE_F64:
        return loom_low_lower_make_register_type(
            context, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 2, out_low_type);
      default:
        break;
    }
  }
  if (loom_type_is_vector(source_type) &&
      loom_type_is_all_static(source_type)) {
    uint64_t element_count = 0;
    if (!loom_type_static_element_count(source_type, &element_count) ||
        element_count == 0) {
      return loom_low_lower_emit_source_type_unsupported(
          context, source_op, IREE_SV("source"), source_type);
    }
    const loom_scalar_type_t element_type = loom_type_element_type(source_type);
    const bool is_rank_one = loom_type_rank(source_type) == 1;
    if (is_rank_one && element_count == 8 &&
        element_type == LOOM_SCALAR_TYPE_BF16) {
      return loom_low_lower_make_register_type(
          context, AIE2P_CORE_REG_CLASS_ID_AIE2P_EWL, 1, out_low_type);
    }
    if (is_rank_one && element_count == 64 &&
        element_type == LOOM_SCALAR_TYPE_BF16) {
      return loom_low_lower_make_register_type(
          context, AIE2P_CORE_REG_CLASS_ID_AIE2P_VEC256, 4, out_low_type);
    }
    if (is_rank_one && element_count == 64 &&
        element_type == LOOM_SCALAR_TYPE_I32) {
      return loom_low_lower_make_register_type(
          context, AIE2P_CORE_REG_CLASS_ID_AIE2P_MBMS, 4, out_low_type);
    }
    if (is_rank_one && element_count == 32 &&
        element_type == LOOM_SCALAR_TYPE_F32) {
      return loom_low_lower_make_register_type(
          context, AIE2P_CORE_REG_CLASS_ID_AIE2P_MBMS, 2, out_low_type);
    }
    if (is_rank_one && element_count == 64 &&
        element_type == LOOM_SCALAR_TYPE_F32) {
      return loom_low_lower_make_register_type(
          context, AIE2P_CORE_REG_CLASS_ID_AIE2P_MBMS, 4, out_low_type);
    }
    if (element_count <= 64 && element_type == LOOM_SCALAR_TYPE_I1) {
      return loom_low_lower_make_register_type(
          context, AIE2P_CORE_REG_CLASS_ID_AIE2P_ELPREDICATE, 1, out_low_type);
    }
    const int32_t element_bits = loom_scalar_type_bitwidth(element_type);
    if (element_bits > 0 && element_count <= 512 / (uint32_t)element_bits) {
      // Ordinary vectors retain a full X-register carrier. Narrow vector
      // memory forms address W subregisters of that carrier; choosing a W
      // carrier from the logical type alone makes the same SSA value unusable
      // by the 512-bit vector ALU.
      return loom_low_lower_make_register_type(
          context, AIE2P_CORE_REG_CLASS_ID_AIE2P_VEC256, 2, out_low_type);
    }
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("source"), source_type);
}

static iree_status_t loom_aie2p_map_argument(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_function_op, uint16_t source_argument_index,
    loom_value_id_t source_argument_id,
    loom_low_lower_abi_argument_t* out_argument) {
  loom_type_t source_type = loom_module_value_type(
      loom_low_lower_context_module(context), source_argument_id);
  if (loom_type_is_buffer(source_type)) {
    loom_type_t resource_type = loom_type_none();
    IREE_RETURN_IF_ERROR(loom_low_lower_make_register_type(
        context, AIE2P_CORE_REG_CLASS_ID_AIE2P_EP, 1, &resource_type));
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
  return loom_aie2p_map_type(user_data, context, source_function_op,
                             source_type, &out_argument->abi_type);
}

static const loom_low_lower_rule_set_t* const kAie2pCoreRuleSets[] = {
    &loom_amd_xdna_aie2p_core_lower_rule_set,
};

static const loom_target_contract_binding_t kAie2pCoreContractBindings[] = {
    {&loom_amd_xdna_aie2p_core_contract_fragment, 0},
};

static iree_status_t loom_aie2p_preselect_op(void* user_data,
                                             loom_low_lower_context_t* context,
                                             const loom_op_t* source_op,
                                             loom_low_lower_plan_t* out_plan) {
  (void)user_data;
  IREE_RETURN_IF_ERROR(
      loom_aie2p_select_matrix_plan(context, source_op, out_plan));
  if (!loom_low_lower_plan_is_empty(*out_plan)) {
    return iree_ok_status();
  }
  return loom_aie2p_select_storage_plan(context, source_op, out_plan);
}

static void loom_aie2p_mark_plan_storage_demands(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_low_lower_plan_t plan) {
  (void)user_data;
  if (loom_aie2p_matrix_plan_isa(plan)) {
    loom_aie2p_mark_matrix_plan_demands(context, source_op, plan);
  } else if (loom_aie2p_storage_plan_isa(plan)) {
    loom_aie2p_mark_storage_plan_demands(context, source_op, plan);
  } else {
    IREE_ASSERT_UNREACHABLE("AIE2P storage demand has unknown plan kind");
  }
}

static void loom_aie2p_describe_plan(void* user_data,
                                     loom_low_lower_context_t* context,
                                     const loom_op_t* source_op,
                                     loom_low_lower_plan_t plan,
                                     loom_low_lower_plan_report_t* out_report) {
  (void)user_data;
  if (loom_aie2p_matrix_plan_isa(plan)) {
    loom_aie2p_describe_matrix_plan(context, source_op, plan, out_report);
  } else if (loom_aie2p_storage_plan_isa(plan)) {
    loom_aie2p_describe_storage_plan(context, source_op, plan, out_report);
  } else {
    IREE_ASSERT_UNREACHABLE("AIE2P report has unknown plan kind");
  }
}

static iree_status_t loom_aie2p_emit_op(void* user_data,
                                        loom_low_lower_context_t* context,
                                        const loom_op_t* source_op,
                                        loom_low_lower_plan_t plan) {
  (void)user_data;
  if (loom_aie2p_matrix_plan_isa(plan)) {
    return loom_aie2p_emit_matrix_plan(context, source_op, plan);
  }
  if (loom_aie2p_storage_plan_isa(plan)) {
    return loom_aie2p_emit_storage_plan(context, source_op, plan);
  }
  IREE_ASSERT_UNREACHABLE("AIE2P emission has unknown plan kind");
  IREE_BUILTIN_UNREACHABLE();
}

static const loom_low_lower_policy_t kAie2pCoreLowLowerPolicy = {
    .name = IREE_SVL("amd-xdna-aie2p-core-low-lower"),
    .error_catalog = &loom_error_catalog_core,
    .source_type_supported =
        {
            .fn = loom_aie2p_source_type_supported,
            .user_data = NULL,
        },
    .map_type = {.fn = loom_aie2p_map_type, .user_data = NULL},
    .map_argument = {.fn = loom_aie2p_map_argument, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kAie2pCoreRuleSets),
            .values = kAie2pCoreRuleSets,
        },
    .contract_bindings = kAie2pCoreContractBindings,
    .contract_binding_count = IREE_ARRAYSIZE(kAie2pCoreContractBindings),
    .descriptor_matrix =
        {
            .options = loom_aie2p_descriptor_matrix_options,
            .query = loom_aie2p_descriptor_matrix_query,
            .attrs = NULL,
            .user_data = NULL,
        },
    .preselect_op = {.fn = loom_aie2p_preselect_op, .user_data = NULL},
    .mark_plan_storage_demands =
        {
            .fn = loom_aie2p_mark_plan_storage_demands,
            .user_data = NULL,
        },
    .describe_plan = {.fn = loom_aie2p_describe_plan, .user_data = NULL},
    .emit_op = {.fn = loom_aie2p_emit_op, .user_data = NULL},
};

const loom_low_lower_policy_t* loom_aie2p_core_low_lower_policy(void) {
  return &kAie2pCoreLowLowerPolicy;
}

void loom_aie2p_low_lower_policy_registry_initialize(
    loom_low_lower_policy_registry_t* out_registry) {
  static const loom_low_lower_policy_registry_entry_t kEntries[] = {
      {
          .contract_set_key = IREE_SVL("amd.xdna.aie2p.core"),
          .policy = &kAie2pCoreLowLowerPolicy,
      },
  };
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, kEntries, IREE_ARRAYSIZE(kEntries));
}
