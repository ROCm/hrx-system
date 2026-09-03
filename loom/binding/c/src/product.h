// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PRODUCT_STORAGE_H_
#define LOOMC_PRODUCT_STORAGE_H_

#include "loom/product/product.h"
#include "loomc/product.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Reinterprets an opaque public product as its core representation.
static inline loom_product_t* loomc_product_to_product(
    loomc_product_t* product) {
  return (loom_product_t*)product;
}

// Reinterprets an opaque public product as its immutable core representation.
static inline const loom_product_t* loomc_product_to_const_product(
    const loomc_product_t* product) {
  return (const loom_product_t*)product;
}

// Reinterprets a core product as its public opaque handle.
static inline loomc_product_t* loomc_product_from_product(
    loom_product_t* product) {
  return (loomc_product_t*)product;
}

// Reinterprets a public product descriptor as its core identity.
static inline const loom_product_descriptor_t*
loomc_product_descriptor_to_product(
    const loomc_product_descriptor_t* descriptor) {
  return (const loom_product_descriptor_t*)descriptor;
}

// Reinterprets a core product descriptor as its public opaque identity.
static inline const loomc_product_descriptor_t*
loomc_product_descriptor_from_product(
    const loom_product_descriptor_t* descriptor) {
  return (const loomc_product_descriptor_t*)descriptor;
}

// Returns true when |product| has the given process-local descriptor.
LOOMC_API_PRIVATE bool loomc_product_isa(
    const loomc_product_t* product,
    const loom_product_descriptor_t* descriptor);

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
