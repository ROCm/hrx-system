// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Module-level SPIR-V target contract.
//
// SPIR-V binary emission produces one physical module at a time. Every function
// in that module must agree on the target snapshot, artifact format, ABI,
// descriptor set, and feature contract so module-level declarations and
// capabilities have one meaning.

#ifndef LOOM_TARGET_ARCH_SPIRV_MODULE_CONTRACT_H_
#define LOOM_TARGET_ARCH_SPIRV_MODULE_CONTRACT_H_

#include "iree/base/api.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_spirv_module_contract_t {
  // Borrowed target-record symbol name selected by the function.
  iree_string_view_t target_name;
  // Borrowed target snapshot name selected by the function.
  iree_string_view_t snapshot_name;
  // Target codegen format selected by the function.
  loom_target_codegen_format_t codegen_format;
  // Target artifact format selected by the function.
  loom_target_artifact_format_t artifact_format;
  // Export ABI kind selected by the function.
  loom_target_abi_kind_t abi_kind;
  // Descriptor set stable ID selected by the function.
  uint64_t descriptor_set_stable_id;
  // Borrowed target-contract descriptor-set key.
  iree_string_view_t contract_set_key;
  // Target-contract feature bits selected by the function.
  uint64_t contract_feature_bits;
} loom_spirv_module_contract_t;

// Captures the SPIR-V module contract implied by |target|.
loom_spirv_module_contract_t loom_spirv_module_contract_from_target(
    const loom_low_resolved_target_t* target);

// Returns true when both contracts can be emitted into one SPIR-V module.
bool loom_spirv_module_contract_equal(const loom_spirv_module_contract_t* lhs,
                                      const loom_spirv_module_contract_t* rhs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_MODULE_CONTRACT_H_
