// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/link/provider_resolver.h"

#include <stdlib.h>

#include "loom/link/module_index.h"

static int loom_link_provider_binding_compare(const void* lhs_ptr,
                                              const void* rhs_ptr) {
  const loom_link_provider_binding_t* lhs =
      (const loom_link_provider_binding_t*)lhs_ptr;
  const loom_link_provider_binding_t* rhs =
      (const loom_link_provider_binding_t*)rhs_ptr;
  return iree_string_view_compare(lhs->key, rhs->key);
}

iree_status_t loom_link_provider_resolver_prepare(
    iree_host_size_t provider_count, loom_link_provider_binding_t* bindings,
    iree_host_size_t binding_count,
    loom_link_provider_resolver_t* out_resolver) {
  IREE_ASSERT_ARGUMENT(out_resolver);
  *out_resolver = (loom_link_provider_resolver_t){0};
  if (binding_count != 0 && !bindings) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "provider binding count is nonzero but bindings is NULL");
  }
  for (iree_host_size_t i = 0; i < binding_count; ++i) {
    const loom_link_provider_binding_t* binding = &bindings[i];
    if (iree_string_view_is_empty(binding->key) || !binding->key.data) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "provider binding key must not be empty");
    }
    if (binding->provider_ordinal >= provider_count) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "provider binding '%.*s' ordinal %" PRIhsz
                              " is out of range for %" PRIhsz " providers",
                              (int)binding->key.size, binding->key.data,
                              binding->provider_ordinal, provider_count);
    }
  }

  if (binding_count > 1) {
    qsort(bindings, binding_count, sizeof(*bindings),
          loom_link_provider_binding_compare);
  }
  for (iree_host_size_t i = 1; i < binding_count; ++i) {
    if (iree_string_view_equal(bindings[i - 1].key, bindings[i].key)) {
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "provider key '%.*s' is bound more than once",
                              (int)bindings[i].key.size, bindings[i].key.data);
    }
  }

  *out_resolver = (loom_link_provider_resolver_t){
      .bindings = bindings,
      .binding_count = binding_count,
  };
  return iree_ok_status();
}

iree_host_size_t loom_link_provider_resolver_lookup(
    const loom_link_provider_resolver_t* resolver, iree_string_view_t key) {
  if (!resolver || resolver->binding_count == 0) {
    return LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
  }
  iree_host_size_t low = 0;
  iree_host_size_t high = resolver->binding_count;
  while (low < high) {
    const iree_host_size_t middle = low + (high - low) / 2;
    const loom_link_provider_binding_t* binding = &resolver->bindings[middle];
    const int comparison = iree_string_view_compare(key, binding->key);
    if (comparison < 0) {
      high = middle;
    } else if (comparison > 0) {
      low = middle + 1;
    } else {
      return binding->provider_ordinal;
    }
  }
  return LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL;
}
