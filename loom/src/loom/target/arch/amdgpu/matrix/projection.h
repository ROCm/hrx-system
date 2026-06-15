// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Projection from generic Loom contract requests to AMDGPU matrix contracts.

#ifndef LOOM_TARGET_ARCH_AMDGPU_MATRIX_PROJECTION_H_
#define LOOM_TARGET_ARCH_AMDGPU_MATRIX_PROJECTION_H_

#include "loom/analysis/contract.h"
#include "loom/target/arch/amdgpu/matrix/contract.h"

#ifdef __cplusplus
extern "C" {
#endif

// Projects a generic contract request into the AMDGPU matrix selector shape.
//
// The generic request supplies target-independent algebra, numeric, fragment,
// capability, and policy facts. The caller supplies processor feature bits and
// selected wave size from the AMDGPU target record.
bool loom_amdgpu_matrix_contract_match_request_from_contract(
    const loom_contract_request_t* contract_request,
    loom_amdgpu_matrix_feature_bits_t feature_bits, uint32_t wave_size,
    loom_amdgpu_matrix_contract_match_request_t* out_request,
    loom_contract_diagnostic_t* out_diagnostic);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_MATRIX_PROJECTION_H_
