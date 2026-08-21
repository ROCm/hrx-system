// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable resolver for compile-time module import provider keys.

#ifndef LOOM_LINK_PROVIDER_RESOLVER_H_
#define LOOM_LINK_PROVIDER_RESOLVER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// One opaque provider key bound to an indexed source provider.
typedef struct loom_link_provider_binding_t {
  // Opaque module.import provider key.
  iree_string_view_t key;
  // Module-index provider ordinal supplying the key.
  iree_host_size_t provider_ordinal;
} loom_link_provider_binding_t;

// Immutable sorted provider-key resolver.
//
// The resolver borrows bindings and their key storage from its owner. Prepared
// bindings may be shared by concurrent link plans while the owner remains live.
typedef struct loom_link_provider_resolver_t {
  // Provider bindings in strictly increasing key order.
  const loom_link_provider_binding_t* bindings;
  // Number of entries in bindings.
  iree_host_size_t binding_count;
} loom_link_provider_resolver_t;

// Sorts and validates |bindings| in place and returns a borrowed resolver view.
//
// Keys must be nonempty and unique. Every provider ordinal must be less than
// |provider_count|. Input order has no semantic effect after preparation.
iree_status_t loom_link_provider_resolver_prepare(
    iree_host_size_t provider_count, loom_link_provider_binding_t* bindings,
    iree_host_size_t binding_count,
    loom_link_provider_resolver_t* out_resolver);

// Resolves |key| to its module-index provider ordinal or
// LOOM_LINK_MODULE_INDEX_INVALID_ORDINAL when no binding exists.
iree_host_size_t loom_link_provider_resolver_lookup(
    const loom_link_provider_resolver_t* resolver, iree_string_view_t key);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_PROVIDER_RESOLVER_H_
