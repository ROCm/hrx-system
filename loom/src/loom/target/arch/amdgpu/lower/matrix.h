// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU lowering bridge for generic matrix contracts.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_H_

#include "loom/analysis/contract_vector.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

// Supplies AMDGPU options for shared descriptor-matrix source adapters.
iree_status_t loom_amdgpu_descriptor_matrix_options(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    loom_contract_vector_mma_options_t* out_options);

// Projects generic matrix contracts to AMDGPU matrix descriptors.
iree_status_t loom_amdgpu_descriptor_matrix_query(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, const loom_contract_request_t* contract_request,
    loom_target_contract_query_result_t* out_result);

// Materializes AMDGPU matrix descriptor immediate attributes.
iree_status_t loom_amdgpu_descriptor_matrix_attrs(
    void* user_data, loom_low_lower_context_t* context,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_contract_request_t* contract_request,
    const loom_low_descriptor_t* descriptor,
    loom_named_attr_slice_t* out_attrs);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_MATRIX_H_
