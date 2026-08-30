// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_REQUEST_INDEX_OVERLAY_H_
#define LOOMC_REQUEST_INDEX_OVERLAY_H_

#include "iree/base/internal/arena.h"
#include "loom/link/module_index.h"
#include "loomc/context.h"
#include "loomc/link_index.h"
#include "loomc/product.h"
#include "loomc/result.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Invocation-local projection of one bytecode request into a module index.
//
// The mutable index borrows an optional frozen library as its base and appends
// the request source as its only INPUT provider. Root ordinals address the
// combined index directly and preserve request-root order and duplicates.
typedef struct loomc_request_index_overlay_t {
  // Owned mutable index containing the request provider and optional library.
  loom_link_module_index_t* module_index;

  // Combined-index ordinal of the appended request provider.
  iree_host_size_t request_provider_ordinal;

  // Scratch-arena array of exact combined-index root symbol ordinals.
  const iree_host_size_t* root_symbol_ordinals;

  // Number of entries in |root_symbol_ordinals|.
  iree_host_size_t root_symbol_count;
} loomc_request_index_overlay_t;

// Indexes |request| over an optional frozen library and maps its exact roots.
//
// |scratch_arena| owns the root projection while |out_overlay| owns the mutable
// index. Source decoding diagnostics are appended to |result|. An OK return
// may accompany a failed result; callers skip dependent work and still
// deinitialize the overlay.
LOOMC_API_PRIVATE loomc_status_t loomc_request_index_overlay_initialize(
    loomc_context_t* context, iree_arena_block_pool_t* block_pool,
    const loomc_link_index_t* library_index, const loomc_request_t* request,
    loomc_result_t* result, iree_arena_allocator_t* scratch_arena,
    loomc_allocator_t allocator, loomc_request_index_overlay_t* out_overlay);

// Releases storage owned by |overlay|.
LOOMC_API_PRIVATE void loomc_request_index_overlay_deinitialize(
    loomc_request_index_overlay_t* overlay);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_REQUEST_INDEX_OVERLAY_H_
