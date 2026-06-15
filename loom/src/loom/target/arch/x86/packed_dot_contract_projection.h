// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Projection from generic Loom contract requests to x86 packed-dot contracts.

#ifndef LOOM_TARGET_ARCH_X86_PACKED_DOT_CONTRACT_PROJECTION_H_
#define LOOM_TARGET_ARCH_X86_PACKED_DOT_CONTRACT_PROJECTION_H_

#include "loom/analysis/contract.h"
#include "loom/target/arch/x86/packed_dot_contract.h"

#ifdef __cplusplus
extern "C" {
#endif

// Projects a generic contract request into the x86 packed-dot selector shape.
//
// The generic request supplies target-independent algebra, numeric, fragment,
// and policy facts. The caller supplies the concrete x86 feature bits from the
// selected target record.
bool loom_x86_packed_dot_match_request_from_contract(
    const loom_contract_request_t* contract_request,
    loom_x86_packed_dot_feature_bits_t feature_bits,
    loom_x86_packed_dot_match_request_t* out_request,
    loom_contract_diagnostic_t* out_diagnostic);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_ARCH_X86_PACKED_DOT_CONTRACT_PROJECTION_H_
