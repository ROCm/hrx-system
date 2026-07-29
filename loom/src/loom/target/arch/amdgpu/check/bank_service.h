// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU bank-service proof emission for source-low Loom tests.

#ifndef LOOM_TARGET_ARCH_AMDGPU_CHECK_BANK_SERVICE_H_
#define LOOM_TARGET_ARCH_AMDGPU_CHECK_BANK_SERVICE_H_

#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits source-low bank-service proofs from structured compile-report rows.
extern const loom_check_emit_provider_t
    loom_amdgpu_bank_service_loom_check_emit_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_CHECK_BANK_SERVICE_H_
