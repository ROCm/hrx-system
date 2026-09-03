// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V compiled-kernel product format and provider.

#ifndef LOOM_TOOLING_TARGET_SPIRV_PRODUCT_PROVIDER_H_
#define LOOM_TOOLING_TARGET_SPIRV_PRODUCT_PROVIDER_H_

#include "loom/product/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Public format name for SPIR-V binary kernel products.
#define LOOM_SPIRV_PRODUCT_FORMAT_BINARY "spirv-binary"

// SPIR-V binary single-file kernel-product format.
extern const loom_product_format_t loom_spirv_binary_product_format;

// Native SPIR-V binary kernel-product implementation.
extern const loom_product_format_provider_t loom_spirv_binary_product_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TARGET_SPIRV_PRODUCT_PROVIDER_H_
