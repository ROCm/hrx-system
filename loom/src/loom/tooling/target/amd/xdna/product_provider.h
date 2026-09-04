// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Canonical AMD XDNA kernel-product format and provider.

#ifndef LOOM_TOOLING_TARGET_AMD_XDNA_PRODUCT_PROVIDER_H_
#define LOOM_TOOLING_TARGET_AMD_XDNA_PRODUCT_PROVIDER_H_

#include "loom/product/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Public format name for canonical AMD XDNA kernel products.
#define LOOM_XDNA_PRODUCT_FORMAT "xdna"

// Canonical single-file XDNA kernel-product format.
extern const loom_product_format_t loom_xdna_product_format;

// Native AIE2P implementation of the canonical XDNA product format.
extern const loom_product_format_provider_t loom_xdna_product_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TARGET_AMD_XDNA_PRODUCT_PROVIDER_H_
