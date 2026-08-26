// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 packed-vector contraction selection.

#ifndef LOOM_TARGET_ARCH_X86_LOWER_CONTRACTION_H_
#define LOOM_TARGET_ARCH_X86_LOWER_CONTRACTION_H_

#include "loom/codegen/low/lower/lower.h"

#ifdef __cplusplus
extern "C" {
#endif

// Supplies the shared vector.mma adapter with packed-vector projection policy.
iree_status_t loom_x86_descriptor_matrix_options(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    loom_contract_vector_mma_options_t* out_options);

// Selects an x86 packed-dot descriptor for a projected vector.mma contract.
iree_status_t loom_x86_descriptor_matrix_query(
    void* user_data,
    const loom_target_contract_query_environment_t* environment,
    const loom_target_contract_descriptor_matrix_rule_t* rule,
    const loom_op_t* source_op, const loom_contract_request_t* request,
    loom_target_contract_query_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_LOWER_CONTRACTION_H_
