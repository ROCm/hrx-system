// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/lower/configuration.h"

#include "loom/target/arch/amd/xdna/aie2p/contracts/configuration.h"
#include "loom/target/arch/amd/xdna/aie2p/contracts/configuration_lower_rules.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/configuration_descriptors.h"
#include "loom/target/arch/amd/xdna/error_catalog.h"

static iree_status_t loom_aie2p_configuration_map_type(
    void* user_data, loom_low_lower_context_t* context,
    const loom_op_t* source_op, loom_type_t source_type,
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
      case LOOM_SCALAR_TYPE_I64:
        return loom_low_lower_make_typed_register_type(
            context, AIE2P_CONFIGURATION_REG_CLASS_ID_AIE2P_CONFIG_SCALAR, 1,
            loom_type_scalar(LOOM_SCALAR_TYPE_I64), out_low_type);
      default:
        break;
    }
  }
  return loom_low_lower_emit_source_type_unsupported(
      context, source_op, IREE_SV("configuration integer"), source_type);
}

#include "loom/target/arch/amd/xdna/aie2p/contracts/configuration_index.inl"

const loom_low_lower_policy_t loom_aie2p_configuration_low_lower_policy = {
    .name = IREE_SVL("amd-xdna-aie2p-configuration-low-lower"),
    .error_catalog = &loom_xdna_error_catalog,
    .map_type = {.fn = loom_aie2p_configuration_map_type},
    .contract = LOOM_AIE2P_CONFIGURATION_CONTRACT,
};
