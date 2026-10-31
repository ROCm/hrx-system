// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_EXECUTABLE_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_EXECUTABLE_H_

#include "iree/base/byte_sequence.h"
#include "loom/serve/runtime/execution/dispatch.h"
#include "loomc/launch_config.h"

#ifdef __cplusplus
extern "C" {
#endif

// Exact emitted-image reuse within one immutable loader target/queue family.
// Each image has one HAL executable, independent of input specialization and
// entry ordering. This cache is used only during preparation, not execution.
typedef struct loom_serve_executable_cache_t loom_serve_executable_cache_t;

// One input's entry mapping and optional direct launch geometry. Views of an
// identical image share code, not launch configurations. A view retains the
// image independently of its originating cache.
typedef struct loom_serve_executable_t loom_serve_executable_t;

typedef struct loom_serve_executable_request_t {
  // Owned contiguous emitted bytes transferred to the cache. Native artifact
  // publication coalesces producer storage before independent device loads.
  iree_byte_sequence_t* image;
  // Root names borrowed through completion, in this input's binding order.
  const iree_string_view_t* roots;
  // Number of roots; a direct launch has exactly one.
  iree_host_size_t root_count;
  // Optional direct geometry borrowed through completion.
  const loomc_launch_config_t* launch_config;
  // Receives owned status and view (NULL on failure). May destroy this request
  // and the cache. Invoked without the cache lock, possibly before load
  // returns.
  void (*completion)(void* user_data, iree_status_t status,
                     loom_serve_executable_t* executable);
  // Borrowed completion context.
  void* user_data;
  // Cache-owned pending linkage, valid only until completion.
  struct loom_serve_executable_request_t* next;
} loom_serve_executable_request_t;

// Borrows the immutable loader domain through cache destruction. Accepted
// requests keep that domain and cache alive until their completions begin.
iree_status_t loom_serve_executable_cache_allocate(
    const iree_hal_queue_family_t* queue_family,
    const iree_hal_executable_target_t* target, iree_allocator_t host_allocator,
    loom_serve_executable_cache_t** out_cache);
void loom_serve_executable_cache_free(loom_serve_executable_cache_t* cache);

// Concurrent calls coalesce exact bytes, including a load already in flight.
// The first producer loads outside the lookup lock; other producers subscribe
// and return without occupying a worker waiting for that load. Every accepted
// request completes exactly once, including allocation and loader failures.
void loom_serve_executable_cache_load(loom_serve_executable_cache_t* cache,
                                      loom_serve_executable_request_t* request);

typedef struct loom_serve_executable_cache_storage_t {
  // Distinct images retained, not the number of specialized views.
  iree_host_size_t count;
  // Image bytes, records, and lookup storage, excluding driver allocations.
  iree_host_size_t host_bytes;
  // Retained lookup capacity included in host_bytes.
  iree_host_size_t index_bytes;
  // Exact image bytes supplied to the loader, counted once per image.
  iree_host_size_t code_bytes;
} loom_serve_executable_cache_storage_t;

// Owner-only after accepted loads complete. Trim releases cache-exclusive
// images; views and external HAL consumers preserve them.
loom_serve_executable_cache_storage_t loom_serve_executable_cache_storage(
    const loom_serve_executable_cache_t* cache);
void loom_serve_executable_cache_trim(loom_serve_executable_cache_t* cache);

void loom_serve_executable_retain(loom_serve_executable_t* executable);
void loom_serve_executable_release(loom_serve_executable_t* executable);
// Checks view, direct-descriptor, and external HAL ownership. The caller holds
// a reference and serializes new borrows; concurrent releases only defer trim.
bool loom_serve_executable_is_shared(const loom_serve_executable_t* executable);
iree_host_size_t loom_serve_executable_storage_size(
    const loom_serve_executable_t* executable);
iree_hal_executable_t* loom_serve_executable_handle(
    const loom_serve_executable_t* executable);
iree_hal_executable_function_t loom_serve_executable_function(
    const loom_serve_executable_t* executable, iree_host_size_t ordinal);
loom_serve_dispatch_t* loom_serve_executable_dispatch(
    const loom_serve_executable_t* executable);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_EXECUTABLE_H_
