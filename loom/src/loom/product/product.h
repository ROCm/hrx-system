// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable compiler products shared by native tooling and API bindings.

#ifndef LOOM_PRODUCT_PRODUCT_H_
#define LOOM_PRODUCT_PRODUCT_H_

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "loom/product/artifact.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_product_t loom_product_t;
typedef struct loom_product_descriptor_t loom_product_descriptor_t;

// Destroys a concrete product after its final reference is released.
typedef void (*loom_product_destroy_fn_t)(loom_product_t* product);

// Process-local identity and lifetime operations for one product
// representation.
//
// Descriptor addresses are stable for the lifetime of the process and are
// compared by identity for in-process request routing. The name is used at
// configuration and diagnostic boundaries; descriptor addresses are never
// serialized.
struct loom_product_descriptor_t {
  // Registered product representation name.
  iree_string_view_t name;

  // Destroys products carrying this descriptor.
  loom_product_destroy_fn_t destroy;
};

// Base structure embedded at offset zero by concrete immutable products.
//
// Product-specific APIs interpret exported roots and unresolved requirements.
// The generic base owns only their ordinal-space sizes and the common artifact
// table.
struct loom_product_t {
  // Atomic reference count controlling the product lifetime.
  iree_atomic_ref_count_t ref_count;

  // Process-local concrete representation descriptor.
  const loom_product_descriptor_t* descriptor;

  // Product-owned immutable artifact table.
  struct {
    // Borrowed artifact views owned by the concrete product.
    const loom_product_artifact_t* values;

    // Number of entries in |values|.
    iree_host_size_t count;
  } artifacts;

  // Number of exported roots addressable by product-local ordinal.
  iree_host_size_t export_count;

  // Number of unresolved requirements addressable by product-local ordinal.
  iree_host_size_t requirement_count;
};

// Initializes an implementation-provided immutable product base.
void loom_product_initialize(const loom_product_descriptor_t* descriptor,
                             const loom_product_artifact_t* artifacts,
                             iree_host_size_t artifact_count,
                             iree_host_size_t export_count,
                             iree_host_size_t requirement_count,
                             loom_product_t* out_product);

// Returns true when |product| has exactly |descriptor|.
bool loom_product_isa(const loom_product_t* product,
                      const loom_product_descriptor_t* descriptor);

// Retains |product| for another owner. NULL is allowed.
void loom_product_retain(loom_product_t* product);

// Releases |product|. NULL is allowed.
void loom_product_release(loom_product_t* product);

// Returns the process-local descriptor identifying |product|, or NULL.
const loom_product_descriptor_t* loom_product_descriptor(
    const loom_product_t* product);

// Returns the number of artifacts owned by |product|, or zero.
iree_host_size_t loom_product_artifact_count(const loom_product_t* product);

// Returns the artifact at |ordinal|, or NULL when out of range.
const loom_product_artifact_t* loom_product_artifact_at(
    const loom_product_t* product, iree_host_size_t ordinal);

// Returns the number of exported roots in |product|, or zero.
iree_host_size_t loom_product_export_count(const loom_product_t* product);

// Returns the number of unresolved requirements in |product|, or zero.
iree_host_size_t loom_product_requirement_count(const loom_product_t* product);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PRODUCT_PRODUCT_H_
