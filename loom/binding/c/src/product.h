// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PRODUCT_STORAGE_H_
#define LOOMC_PRODUCT_STORAGE_H_

#include "iree/base/internal/atomics.h"
#include "loomc/product.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Destroys a concrete product after its final reference is released.
typedef void(LOOMC_API_PTR* loomc_product_destroy_fn_t)(
    loomc_product_t* product);

// Process-local descriptor implemented by one concrete product type.
struct loomc_product_descriptor_t {
  // Destroys |product| and all of its owned storage.
  loomc_product_destroy_fn_t destroy;
};

// Base structure embedded at offset zero by concrete product types.
struct loomc_product_t {
  // Atomic reference count controlling the product lifetime.
  iree_atomic_ref_count_t ref_count;

  // Process-local concrete representation descriptor.
  const loomc_product_descriptor_t* descriptor;

  // Borrowed immutable artifact views owned by the concrete product.
  struct {
    // Product-owned artifact table.
    const loomc_artifact_t* values;

    // Number of entries in |values|.
    loomc_host_size_t count;
  } artifacts;

  // Number of exported roots addressable by child-root ordinal.
  loomc_host_size_t export_count;

  // Number of unresolved parent-local requirements.
  loomc_host_size_t requirement_count;
};

// Initializes an implementation-provided product base.
LOOMC_API_PRIVATE void loomc_product_initialize(
    const loomc_product_descriptor_t* descriptor,
    const loomc_artifact_t* artifacts, loomc_host_size_t artifact_count,
    loomc_host_size_t export_count, loomc_host_size_t requirement_count,
    loomc_product_t* out_product);

// Returns true when |product| has the given process-local descriptor.
LOOMC_API_PRIVATE bool loomc_product_isa(
    const loomc_product_t* product,
    const loomc_product_descriptor_t* descriptor);

// Allocates an immutable request and transfers |*inout_source| on success.
//
// |product_descriptor| is the required successful product representation and
// must have process lifetime. It is retained by identity rather than ownership.
//
// Root order defines product export order and may contain duplicate source
// addresses when linking coalesces distinct logical roots. Bindings must be
// sorted by requirement ordinal and refer only to supplied roots. These are
// trusted producer invariants rather than public validation obligations.
LOOMC_API_PRIVATE loomc_status_t loomc_request_create_take_source(
    const loomc_product_descriptor_t* product_descriptor,
    loomc_source_t** inout_source, const loomc_request_root_t* roots,
    loomc_host_size_t root_count, const loomc_request_binding_t* bindings,
    loomc_host_size_t binding_count, loomc_allocator_t allocator,
    loomc_request_t** out_request);

// Returns the borrowed root table owned by |request|.
LOOMC_API_PRIVATE const loomc_request_root_t* loomc_request_roots(
    const loomc_request_t* request);

// Returns the borrowed binding table owned by |request|.
LOOMC_API_PRIVATE const loomc_request_binding_t* loomc_request_bindings(
    const loomc_request_t* request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_PRODUCT_STORAGE_H_
