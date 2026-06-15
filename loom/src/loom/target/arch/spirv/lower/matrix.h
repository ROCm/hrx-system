// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V lowering bridge for generic matrix contracts.

#ifndef LOOM_TARGET_ARCH_SPIRV_LOWER_MATRIX_H_
#define LOOM_TARGET_ARCH_SPIRV_LOWER_MATRIX_H_

#include "loom/analysis/contract_vector.h"
#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Supplies SPIR-V options for shared descriptor-matrix source adapters.
iree_status_t loom_spirv_descriptor_matrix_options(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    loom_contract_vector_mma_options_t* out_options);

// Projects generic matrix contracts to SPIR-V cooperative matrix properties.
iree_status_t loom_spirv_descriptor_matrix_query(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, const loom_contract_request_t* contract_request,
    loom_target_contract_query_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_LOWER_MATRIX_H_
