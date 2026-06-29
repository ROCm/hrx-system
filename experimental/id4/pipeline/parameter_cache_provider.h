// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_ID4_PIPELINE_PARAMETER_CACHE_PROVIDER_H_
#define EXPERIMENTAL_ID4_PIPELINE_PARAMETER_CACHE_PROVIDER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_provider.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creation options for a source-resident parameter cache provider.
typedef struct id4_pipeline_parameter_cache_provider_options_t {
  // Size of this structure in bytes for ABI compatibility.
  iree_host_size_t structure_size;
  // Reserved extension chain pointer. Must be NULL.
  const void* next;
  // Upstream provider containing the authoritative parameter bytes.
  iree_io_parameter_provider_t* source_provider;
  // Buffer parameters used for device-local source cache entries.
  iree_hal_buffer_params_t cache_params;
  // Maximum live cached source bytes, or zero for unbounded cache growth.
  iree_device_size_t maximum_cached_byte_length;
} id4_pipeline_parameter_cache_provider_options_t;

// Snapshot of source-resident parameter cache provider state.
typedef struct id4_pipeline_parameter_cache_provider_statistics_t {
  // Number of exact source spans currently cached.
  iree_host_size_t entry_count;
  // Sum of live source-resident cache buffer byte lengths.
  iree_device_size_t cached_byte_length;
  // Largest observed value of |cached_byte_length|.
  iree_device_size_t peak_cached_byte_length;
  // Configured maximum live cached source bytes, or zero when unbounded.
  iree_device_size_t maximum_cached_byte_length;
  // Number of upstream gather calls issued to fill cache entries.
  iree_host_size_t source_gather_count;
  // Number of caller gather requests served from an existing cache entry.
  iree_host_size_t cache_reuse_count;
  // Number of cache entries evicted by budget pressure or notifications.
  iree_host_size_t evicted_entry_count;
} id4_pipeline_parameter_cache_provider_statistics_t;

// Wraps |source_provider| with an exact-span device cache for gather requests.
//
// Cached entries are keyed by device, queue affinity, source scope, source key,
// source offset, and byte length. Target buffer offsets are not part of the
// cache key, so repeated gathers of the same source range into different slabs
// reuse the same source-resident bytes. Gather calls preserve the caller's
// wait/signal timeline by inserting explicit semaphore edges from cache fills
// to the final target copy.
//
// The cache is read-only. Query and load requests are delegated to the upstream
// provider; scatter requests fail because writes would require cache
// invalidation. If a maximum cached byte length is configured, the provider
// evicts oldest entries before inserting a new entry that would exceed the
// budget. SUSPEND and LOW_MEMORY notifications drop all cached entries after
// forwarding the notification upstream.
iree_status_t id4_pipeline_parameter_cache_provider_create(
    const id4_pipeline_parameter_cache_provider_options_t* options,
    iree_allocator_t host_allocator,
    iree_io_parameter_provider_t** out_provider);

// Queries source-resident cache statistics from |provider|.
iree_status_t id4_pipeline_parameter_cache_provider_query_statistics(
    iree_io_parameter_provider_t* provider,
    id4_pipeline_parameter_cache_provider_statistics_t* out_statistics);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_ID4_PIPELINE_PARAMETER_CACHE_PROVIDER_H_
