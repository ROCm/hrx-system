// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Kernel-product adapter for target artifact emitters.

#ifndef LOOM_TOOLING_COMPILE_ARTIFACT_PRODUCT_H_
#define LOOM_TOOLING_COMPILE_ARTIFACT_PRODUCT_H_

#include "loom/tooling/compile/artifact.h"
#include "loom/tooling/compile/product.h"

#ifdef __cplusplus
extern "C" {
#endif

// Builds one immutable kernel product with |artifact_provider|.
//
// This adapter preserves the existing target emitters behind the common
// product-format provider boundary. The target-native and executable payloads
// must be the same byte sequence: an emitter with distinct payloads requires a
// product format that declares both roles rather than silently dropping one.
iree_status_t loom_artifact_provider_build_kernel_product(
    const loom_product_format_provider_t* product_provider,
    const loom_artifact_provider_t* artifact_provider,
    const loom_product_build_request_t* request, loom_product_t** out_product);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_COMPILE_ARTIFACT_PRODUCT_H_
