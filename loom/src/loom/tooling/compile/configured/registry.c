// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tooling/compile/configured/registry.h"

#include "loom/product/kernel.h"
#include "loom/target/arch/cmd/product.h"
#include "loom/tooling/target/cmd/product_provider.h"

#ifndef LOOM_CONFIG_PRODUCT_HAVE_AMDGPU_HSACO
#define LOOM_CONFIG_PRODUCT_HAVE_AMDGPU_HSACO 0
#endif  // LOOM_CONFIG_PRODUCT_HAVE_AMDGPU_HSACO
#ifndef LOOM_CONFIG_PRODUCT_HAVE_SPIRV_BINARY
#define LOOM_CONFIG_PRODUCT_HAVE_SPIRV_BINARY 0
#endif  // LOOM_CONFIG_PRODUCT_HAVE_SPIRV_BINARY

#if LOOM_CONFIG_PRODUCT_HAVE_AMDGPU_HSACO
#include "loom/tooling/target/amdgpu/product_provider.h"
#endif  // LOOM_CONFIG_PRODUCT_HAVE_AMDGPU_HSACO
#if LOOM_CONFIG_PRODUCT_HAVE_SPIRV_BINARY
#include "loom/tooling/target/spirv/product_provider.h"
#endif  // LOOM_CONFIG_PRODUCT_HAVE_SPIRV_BINARY

static const loom_product_operation_t* const kConfiguredProductOperations[] = {
    &loom_kernel_product_operation,
    &loom_cmd_product_operation,
};

static const loom_product_format_t* const kConfiguredProductFormats[] = {
#if LOOM_CONFIG_PRODUCT_HAVE_AMDGPU_HSACO
    &loom_amdgpu_hsaco_product_format,
#endif  // LOOM_CONFIG_PRODUCT_HAVE_AMDGPU_HSACO
#if LOOM_CONFIG_PRODUCT_HAVE_SPIRV_BINARY
    &loom_spirv_binary_product_format,
#endif  // LOOM_CONFIG_PRODUCT_HAVE_SPIRV_BINARY
    &loom_cmd_product_format,
};

static const loom_product_format_provider_t* const
    kConfiguredProductProviders[] = {
#if LOOM_CONFIG_PRODUCT_HAVE_AMDGPU_HSACO
        &loom_amdgpu_hsaco_product_provider,
#endif  // LOOM_CONFIG_PRODUCT_HAVE_AMDGPU_HSACO
#if LOOM_CONFIG_PRODUCT_HAVE_SPIRV_BINARY
        &loom_spirv_binary_product_provider,
#endif  // LOOM_CONFIG_PRODUCT_HAVE_SPIRV_BINARY
        &loom_cmd_product_provider,
};

static const loom_product_registry_t kConfiguredProductRegistry = {
    .operations =
        {
            .values = kConfiguredProductOperations,
            .count = IREE_ARRAYSIZE(kConfiguredProductOperations),
        },
    .formats =
        {
            .values = kConfiguredProductFormats,
            .count = IREE_ARRAYSIZE(kConfiguredProductFormats),
        },
    .providers =
        {
            .values = kConfiguredProductProviders,
            .count = IREE_ARRAYSIZE(kConfiguredProductProviders),
        },
};

const loom_product_registry_t* loom_configured_product_registry(void) {
  return &kConfiguredProductRegistry;
}
