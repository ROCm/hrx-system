// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU compiled-kernel product format and provider.

#ifndef LOOM_TOOLING_TARGET_AMDGPU_PRODUCT_PROVIDER_H_
#define LOOM_TOOLING_TARGET_AMDGPU_PRODUCT_PROVIDER_H_

#include "loom/product/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Public format name for AMDGPU HSACO kernel products.
#define LOOM_AMDGPU_PRODUCT_FORMAT_HSACO "amdgpu-hsaco"

// Public artifact format name for AMDGPU assembly listings.
#define LOOM_AMDGPU_PRODUCT_ARTIFACT_FORMAT_ASSEMBLY "amdgpu-assembly"

// AMDGPU HSACO single-file kernel-product format.
extern const loom_product_format_t loom_amdgpu_hsaco_product_format;

// Native AMDGPU HSACO kernel-product implementation.
extern const loom_product_format_provider_t loom_amdgpu_hsaco_product_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TARGET_AMDGPU_PRODUCT_PROVIDER_H_
