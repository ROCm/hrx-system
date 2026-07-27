// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/module_contract.h"

loom_spirv_module_contract_t loom_spirv_module_contract_from_target(
    const loom_low_resolved_target_t* target) {
  return (loom_spirv_module_contract_t){
      .target_name = target->target_name,
      .snapshot_name = target->bundle_storage.snapshot.name,
      .codegen_format = target->bundle_storage.snapshot.codegen_format,
      .artifact_format = target->bundle_storage.snapshot.artifact_format,
      .abi_kind = target->bundle_storage.export_plan.abi_kind,
      .descriptor_set_stable_id = target->descriptor_set->stable_id,
      .contract_set_key = target->descriptor_set_key,
      .contract_feature_bits = target->feature_bits,
  };
}

bool loom_spirv_module_contract_equal(const loom_spirv_module_contract_t* lhs,
                                      const loom_spirv_module_contract_t* rhs) {
  return iree_string_view_equal(lhs->snapshot_name, rhs->snapshot_name) &&
         lhs->codegen_format == rhs->codegen_format &&
         lhs->artifact_format == rhs->artifact_format &&
         lhs->abi_kind == rhs->abi_kind &&
         lhs->descriptor_set_stable_id == rhs->descriptor_set_stable_id &&
         iree_string_view_equal(lhs->contract_set_key, rhs->contract_set_key) &&
         lhs->contract_feature_bits == rhs->contract_feature_bits;
}
