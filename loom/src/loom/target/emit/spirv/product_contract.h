// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V binary kernel product contract.

#ifndef LOOM_TARGET_EMIT_SPIRV_PRODUCT_CONTRACT_H_
#define LOOM_TARGET_EMIT_SPIRV_PRODUCT_CONTRACT_H_

#include "loom/target/product_contract.h"

#ifdef __cplusplus
extern "C" {
#endif

// Product-owned lowering contract for HAL-loadable SPIR-V binary kernels.
extern const loom_target_product_contract_t
    loom_spirv_binary_kernel_product_contract;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_PRODUCT_CONTRACT_H_
