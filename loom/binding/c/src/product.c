// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "product.h"

#include <string.h>

#include "iree/base/api.h"

void loomc_product_initialize(const loomc_product_descriptor_t* descriptor,
                              const loomc_artifact_t* artifacts,
                              loomc_host_size_t artifact_count,
                              loomc_host_size_t export_count,
                              loomc_host_size_t requirement_count,
                              loomc_product_t* out_product) {
  iree_atomic_ref_count_init(&out_product->ref_count);
  out_product->descriptor = descriptor;
  out_product->artifacts.values = artifacts;
  out_product->artifacts.count = artifact_count;
  out_product->export_count = export_count;
  out_product->requirement_count = requirement_count;
}

bool loomc_product_isa(const loomc_product_t* product,
                       const loomc_product_descriptor_t* descriptor) {
  return product != NULL && product->descriptor == descriptor;
}

void loomc_product_retain(loomc_product_t* product) {
  if (product == NULL) return;
  iree_atomic_ref_count_inc(&product->ref_count);
}

void loomc_product_release(loomc_product_t* product) {
  if (product == NULL) return;
  if (iree_atomic_ref_count_dec(&product->ref_count) == 1) {
    product->descriptor->destroy(product);
  }
}

const loomc_product_descriptor_t* loomc_product_descriptor(
    const loomc_product_t* product) {
  return product ? product->descriptor : NULL;
}

loomc_host_size_t loomc_product_artifact_count(const loomc_product_t* product) {
  return product ? product->artifacts.count : 0;
}

const loomc_artifact_t* loomc_product_artifact_at(
    const loomc_product_t* product, loomc_host_size_t ordinal) {
  if (product == NULL || ordinal >= product->artifacts.count) return NULL;
  return &product->artifacts.values[ordinal];
}

loomc_host_size_t loomc_product_export_count(const loomc_product_t* product) {
  return product ? product->export_count : 0;
}

loomc_host_size_t loomc_product_requirement_count(
    const loomc_product_t* product) {
  return product ? product->requirement_count : 0;
}

struct loomc_request_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for request and table storage.
  loomc_allocator_t allocator;

  // Owned immutable Loom bytecode source.
  loomc_source_t* source;

  // Required process-local successful product representation.
  const loomc_product_descriptor_t* product_descriptor;

  // Canonical roots in the request source.
  struct {
    // Request-owned root table.
    loomc_request_root_t* values;

    // Number of entries in |values|.
    loomc_host_size_t count;
  } roots;

  // Provisional parent requirement bindings.
  struct {
    // Request-owned binding table.
    loomc_request_binding_t* values;

    // Number of entries in |values|.
    loomc_host_size_t count;
  } bindings;
};

loomc_status_t loomc_request_create_take_source(
    const loomc_product_descriptor_t* product_descriptor,
    loomc_source_t** inout_source, const loomc_request_root_t* roots,
    loomc_host_size_t root_count, const loomc_request_binding_t* bindings,
    loomc_host_size_t binding_count, loomc_allocator_t allocator,
    loomc_request_t** out_request) {
  *out_request = NULL;

  iree_host_size_t root_storage_size = 0;
  iree_host_size_t binding_storage_size = 0;
  iree_host_size_t allocation_size = sizeof(loomc_request_t);
  if (!iree_host_size_checked_mul(root_count, sizeof(*roots),
                                  &root_storage_size) ||
      !iree_host_size_checked_mul(binding_count, sizeof(*bindings),
                                  &binding_storage_size) ||
      !iree_host_size_checked_add(allocation_size, root_storage_size,
                                  &allocation_size) ||
      !iree_host_size_checked_add(allocation_size, binding_storage_size,
                                  &allocation_size)) {
    return loomc_make_status(LOOMC_STATUS_RESOURCE_EXHAUSTED,
                             "request metadata is too large");
  }

  loomc_request_t* request = NULL;
  LOOMC_RETURN_IF_ERROR(loomc_allocator_malloc_uninitialized(
      allocator, allocation_size, (void**)&request));
  iree_atomic_ref_count_init(&request->ref_count);
  request->allocator = allocator;
  request->source = *inout_source;
  request->product_descriptor = product_descriptor;
  request->roots.count = root_count;
  request->bindings.count = binding_count;

  uint8_t* storage = (uint8_t*)(request + 1);
  request->roots.values = (loomc_request_root_t*)storage;
  if (root_storage_size != 0) {
    memcpy(request->roots.values, roots, root_storage_size);
    storage += root_storage_size;
  }
  request->bindings.values = (loomc_request_binding_t*)storage;
  if (binding_storage_size != 0) {
    memcpy(request->bindings.values, bindings, binding_storage_size);
  }

  *inout_source = NULL;
  *out_request = request;
  return loomc_ok_status();
}

loomc_status_t loomc_request_create(
    const loomc_product_descriptor_t* product_descriptor,
    loomc_source_t* source, const loomc_request_root_t* roots,
    loomc_host_size_t root_count, const loomc_request_binding_t* bindings,
    loomc_host_size_t binding_count, loomc_allocator_t allocator,
    loomc_request_t** out_request) {
  if (out_request == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "out_request must not be NULL");
  }
  *out_request = NULL;
  if (product_descriptor == NULL || source == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "product_descriptor and source must not be NULL");
  }
  if (!loomc_allocator_is_valid(allocator)) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "allocator must be valid");
  }
  if (loomc_source_format(source) != LOOMC_SOURCE_FORMAT_BYTECODE) {
    return loomc_make_status(LOOMC_STATUS_FAILED_PRECONDITION,
                             "request source is not Loom bytecode");
  }
  if (root_count == 0 || roots == NULL || root_count > UINT32_MAX) {
    return loomc_make_status(
        LOOMC_STATUS_INVALID_ARGUMENT,
        "request roots must contain between 1 and UINT32_MAX entries");
  }
  if (binding_count != 0 && bindings == NULL) {
    return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                             "binding_count is nonzero but bindings is NULL");
  }
  for (loomc_host_size_t i = 0; i < root_count; ++i) {
    if (roots[i].reserved != 0) {
      return loomc_make_status(LOOMC_STATUS_INVALID_ARGUMENT,
                               "request root reserved fields must be zero");
    }
  }
  for (loomc_host_size_t i = 0; i < binding_count; ++i) {
    if (bindings[i].root_ordinal >= root_count) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "request binding refers to an unavailable root ordinal");
    }
    if (i != 0 && bindings[i - 1].requirement_ordinal >=
                      bindings[i].requirement_ordinal) {
      return loomc_make_status(
          LOOMC_STATUS_INVALID_ARGUMENT,
          "request bindings are not strictly ordered by requirement ordinal");
    }
  }

  loomc_source_retain(source);
  loomc_source_t* transferred_source = source;
  loomc_status_t status = loomc_request_create_take_source(
      product_descriptor, &transferred_source, roots, root_count, bindings,
      binding_count, allocator, out_request);
  loomc_source_release(transferred_source);
  return status;
}

void loomc_request_retain(loomc_request_t* request) {
  if (request == NULL) return;
  iree_atomic_ref_count_inc(&request->ref_count);
}

void loomc_request_release(loomc_request_t* request) {
  if (request == NULL) return;
  if (iree_atomic_ref_count_dec(&request->ref_count) != 1) return;
  loomc_allocator_t allocator = request->allocator;
  loomc_source_release(request->source);
  loomc_allocator_free(allocator, request);
}

const loomc_product_descriptor_t* loomc_request_product_descriptor(
    const loomc_request_t* request) {
  return request ? request->product_descriptor : NULL;
}

loomc_source_t* loomc_request_source(const loomc_request_t* request) {
  return request ? request->source : NULL;
}

loomc_host_size_t loomc_request_root_count(const loomc_request_t* request) {
  return request ? request->roots.count : 0;
}

bool loomc_request_root_at(const loomc_request_t* request,
                           loomc_request_root_ordinal_t ordinal,
                           loomc_request_root_t* out_root) {
  if (request == NULL || out_root == NULL || ordinal >= request->roots.count) {
    return false;
  }
  *out_root = request->roots.values[ordinal];
  return true;
}

loomc_host_size_t loomc_request_binding_count(const loomc_request_t* request) {
  return request ? request->bindings.count : 0;
}

bool loomc_request_binding_at(const loomc_request_t* request,
                              loomc_host_size_t ordinal,
                              loomc_request_binding_t* out_binding) {
  if (request == NULL || out_binding == NULL ||
      ordinal >= request->bindings.count) {
    return false;
  }
  *out_binding = request->bindings.values[ordinal];
  return true;
}

const loomc_request_root_t* loomc_request_roots(
    const loomc_request_t* request) {
  return request ? request->roots.values : NULL;
}

const loomc_request_binding_t* loomc_request_bindings(
    const loomc_request_t* request) {
  return request ? request->bindings.values : NULL;
}
