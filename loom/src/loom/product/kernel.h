// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable compiled-kernel products.

#ifndef LOOM_PRODUCT_KERNEL_H_
#define LOOM_PRODUCT_KERNEL_H_

#include "iree/base/api.h"
#include "loom/product/registry.h"
#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Process-local descriptor for compiled kernel products.
extern const loom_product_descriptor_t loom_kernel_product_descriptor;

// Semantic kernel product operation and its durable root operations.
extern const loom_product_operation_t loom_kernel_product_operation;

// Construction options for one immutable compiled-kernel product.
typedef struct loom_kernel_product_options_t {
  // Canonical family-owned target selector used to compile the product.
  iree_string_view_t target_key;

  // Complete resolved target bundle retained by value in the product.
  const loom_target_bundle_t* target_bundle;

  // Artifact table to clone. Byte sequences are retained, and all string
  // metadata is copied.
  const loom_product_artifact_t* artifacts;

  // Number of entries in |artifacts|.
  iree_host_size_t artifact_count;

  // Artifact ordinal containing the loadable kernel payload.
  iree_host_size_t loadable_artifact_ordinal;

  // Number of compiled kernel exports addressable by product-local ordinal.
  iree_host_size_t export_count;

  // Number of unresolved requirements addressable by product-local ordinal.
  iree_host_size_t requirement_count;
} loom_kernel_product_options_t;

// Creates an immutable kernel product by cloning |options| metadata and
// retaining every artifact byte sequence.
iree_status_t loom_kernel_product_create(
    const loom_kernel_product_options_t* options, iree_allocator_t allocator,
    loom_product_t** out_product);

// Returns the canonical selected target key, or empty for another product.
iree_string_view_t loom_kernel_product_target_key(
    const loom_product_t* product);

// Returns the durable resolved target bundle, or NULL for another product.
const loom_target_bundle_t* loom_kernel_product_target_bundle(
    const loom_product_t* product);

// Returns the loadable kernel artifact, or NULL for another product.
const loom_product_artifact_t* loom_kernel_product_loadable_artifact(
    const loom_product_t* product);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PRODUCT_KERNEL_H_
