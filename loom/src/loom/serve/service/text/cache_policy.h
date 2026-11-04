// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_CACHE_POLICY_H_
#define LOOM_SERVE_SERVICE_TEXT_CACHE_POLICY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Omission requests cold sizing; an explicit zero disables the cache.
#define LOOM_SERVE_TEXT_CACHE_AUTOMATIC IREE_HOST_SIZE_MAX

// Default soft target: 2 GiB, reduced to 1/16 of a nonzero shared physical
// budget. This consumes configured policy, never allocator availability.
uint64_t loom_serve_text_cache_default_target(uint64_t memory_limit);

typedef struct loom_serve_text_cache_options_t {
  // Soft target for additional initial storage, not a physical reservation or
  // a limit on growing active-session COW headroom. Zero disables auto sizing.
  uint64_t target_bytes;
  // Detached histories, or AUTOMATIC. Nonzero counts need one exchange slot.
  iree_host_size_t session_capacity;
  // Shared intermediate prefixes, or AUTOMATIC.
  iree_host_size_t prefix_capacity;
} loom_serve_text_cache_options_t;

typedef struct loom_serve_text_cache_selection_t {
  // Resolved detached-history capacity, excluding the admission exchange.
  iree_host_size_t session_capacity;
  // Resolved shared-prefix capacity.
  iree_host_size_t prefix_capacity;
  // Total source endpoints, including explicit pins and admission exchange.
  iree_host_size_t checkpoint_capacity;
  // Soft target used for automatic selection, not a lifetime memory ceiling.
  uint64_t target_bytes;
  // Additional initial declared state and checkpoint maps over explicit counts.
  uint64_t additional_bytes;
} loom_serve_text_cache_selection_t;

typedef struct loom_serve_text_cache_cost_t {
  // Exact declared state extents plus native checkpoint directory storage.
  uint64_t storage_bytes;
} loom_serve_text_cache_cost_t;

// Evaluates the source's exact storage for a candidate endpoint count. Success
// replaces the caller's previous candidate; failure retains partial ownership
// for its cleanup. No device compilation, allocation, or parameter I/O occurs.
typedef iree_status_t (*loom_serve_text_cache_evaluate_fn_t)(
    void* user_data, iree_host_size_t checkpoint_capacity,
    loom_serve_text_cache_cost_t* out_cost);

// Cold policy for validated explicit counts. Automatic sizing gives up to two
// shared prefixes priority, then searches detached counts across the complete
// signed endpoint space. Every selected count has been evaluated against the
// soft target; geometric search is a heuristic, not a promise of optimality for
// arbitrary source size functions. The final evaluation is the selected source
// declaration, ready for native compilation. Evaluation errors propagate
// without disabling caches.
iree_status_t loom_serve_text_cache_select(
    const loom_serve_text_cache_options_t* options,
    iree_host_size_t explicit_capacity,
    loom_serve_text_cache_evaluate_fn_t evaluate, void* user_data,
    loom_serve_text_cache_selection_t* out_selection);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_CACHE_POLICY_H_
