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
        return loom_low_lower_make_register_type(
            context, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 1, out_low_type);
      case LOOM_SCALAR_TYPE_I64:
        return loom_low_lower_make_register_type(
            context, AIE2P_CORE_REG_CLASS_ID_AIE2P_ER, 2, out_low_type);
      default:
        break;
    }
  }
  if (loom_type_is_vector(source_type) && loom_type_rank(source_type) == 1 &&
      loom_type_is_all_static(source_type)) {
    const int64_t lane_count = loom_type_dim_static_size_at(source_type, 0);
    const loom_scalar_type_t element_type = loom_type_element_type(source_type);
    if (lane_count > 0 && lane_count <= 64 &&
        element_type == LOOM_SCALAR_TYPE_I1) {
      return loom_low_lower_make_register_type(
          context, AIE2P_CORE_REG_CLASS_ID_AIE2P_ELPREDICATE, 1, out_low_type);
    }
    const bool fits_vec512 =
        (element_type == LOOM_SCALAR_TYPE_I8 && lane_count <= 64) ||
        (element_type == LOOM_SCALAR_TYPE_I16 && lane_count <= 32) ||
        (element_type == LOOM_SCALAR_TYPE_I32 && lane_count <= 16);
    if (lane_count > 0 && fits_vec512) {
      return loom_low_lower_make_register_type(
          context, AIE2P_CORE_REG_CLASS_ID_AIE2P_VEC512, 1, out_low_type);
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

static const loom_low_lower_policy_t kAie2pCoreLowLowerPolicy = {
    .name = IREE_SVL("amd-xdna-aie2p-core-low-lower"),
    .error_catalog = &loom_error_catalog_core,
    .map_type = {.fn = loom_aie2p_map_type, .user_data = NULL},
    .map_argument = {.fn = loom_aie2p_map_argument, .user_data = NULL},
    .rule_sets =
        {
            .count = IREE_ARRAYSIZE(kAie2pCoreRuleSets),
            .values = kAie2pCoreRuleSets,
        },
    .contract_bindings = kAie2pCoreContractBindings,
    .contract_binding_count = IREE_ARRAYSIZE(kAie2pCoreContractBindings),
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
