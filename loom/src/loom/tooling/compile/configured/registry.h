// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Configured product-format registry selected by //loom/config/target.

#ifndef LOOM_TOOLING_COMPILE_CONFIGURED_REGISTRY_H_
#define LOOM_TOOLING_COMPILE_CONFIGURED_REGISTRY_H_

#include "loom/product/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the static configured product operation, format, and provider
// registry. The registry is independent of HAL drivers and physical devices.
const loom_product_registry_t* loom_configured_product_registry(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_COMPILE_CONFIGURED_REGISTRY_H_
