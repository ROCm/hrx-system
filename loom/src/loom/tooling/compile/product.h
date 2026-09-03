// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Product-format compilation boundary shared by tools and embeddings.

#ifndef LOOM_TOOLING_COMPILE_PRODUCT_H_
#define LOOM_TOOLING_COMPILE_PRODUCT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/module.h"
#include "loom/product/registry.h"
#include "loom/target/low_descriptor_registry.h"
#include "loom/target/provider.h"
#include "loom/tooling/compile/options.h"

#ifdef __cplusplus
extern "C" {
#endif

// Fully planned inputs for one product-format provider invocation.
//
// Selection happens before this boundary. Providers consume the linked module,
// perform the product-specific preparation and emission they own, and return
// one immutable product without performing filesystem persistence.
struct loom_product_build_request_t {
  // Composed target capabilities linked into the compiler.
  const loom_target_environment_t* target_environment;

  // Target-low descriptor registry matching |target_environment|.
  const loom_target_low_descriptor_registry_t* low_descriptor_registry;

  // Mutable linked module in the representation expected by the provider.
  loom_module_t* module;

  // Selected structured target profile, or NULL for target-neutral products
  // and the transitional authored-target emission path.
  const loom_target_profile_t* target_profile;

  // Canonical family-owned target selector used for artifact metadata.
  iree_string_view_t target_key;

  // Stable product-local identifier for the provider's loadable artifact.
  iree_string_view_t artifact_identifier;

  // Number of selected roots represented in the returned product.
  iree_host_size_t export_count;

  // Common pipeline and emission options.
  const loom_compile_options_t* compile_options;

  // Reusable compiler arena block pool for provider-local scratch state.
  iree_arena_block_pool_t* block_pool;

  // Provider-specific option chain, or NULL.
  const void* option_chain;

  // Host allocator used for product storage that escapes the invocation.
  iree_allocator_t allocator;
};

// Invokes |provider| and validates its immutable product against the selected
// public format. An OK status with a NULL product means compiler diagnostics
// rejected the source and were emitted through the request options.
iree_status_t loom_product_format_provider_build(
    const loom_product_format_provider_t* provider,
    const loom_product_build_request_t* request, loom_product_t** out_product);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_COMPILE_PRODUCT_H_
