// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "product.h"

#include <string.h>

#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

struct loomc_request_t {
  // Atomic reference count for shared immutable ownership.
  iree_atomic_ref_count_t ref_count;

  // Allocator used for request and table storage.
  loomc_allocator_t allocator;

  // Owned immutable Loom bytecode source.
  loomc_source_t* source;

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
