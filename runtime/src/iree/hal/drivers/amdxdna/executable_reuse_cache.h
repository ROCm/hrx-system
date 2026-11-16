// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_EXECUTABLE_REUSE_CACHE_H_
#define IREE_HAL_DRIVERS_AMDXDNA_EXECUTABLE_REUSE_CACHE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Identifies a reusable single-entry-point, single-run executable.
typedef struct iree_hal_amdxdna_executable_reuse_key_t {
  // Caller-defined bytes that, together with |control_code|, determine the
  // executable completely: for example a producer-provided context image
  // identity plus the entry point name and ordinals. Compared exactly.
  iree_const_byte_span_t descriptor;
  // Control code of the executable's only run. Compared exactly against the
  // cached executable's own control code, so the key never duplicates it.
  iree_const_byte_span_t control_code;
} iree_hal_amdxdna_executable_reuse_key_t;

typedef struct iree_hal_amdxdna_executable_reuse_cache_limits_t {
  // Maximum number of cached executables.
  iree_host_size_t max_entries;
  // Maximum host bytes retained on behalf of cached executables. Each entry
  // is charged its exclusive bytes; each program image is charged once while
  // any cached entry references it.
  iree_host_size_t max_host_bytes;
} iree_hal_amdxdna_executable_reuse_cache_limits_t;

typedef struct iree_hal_amdxdna_executable_reuse_cache_stats_t {
  iree_host_size_t entry_count;
  // Total charged bytes, including |image_bytes|.
  iree_host_size_t host_bytes;
  // Bytes of the distinct program images referenced by cached entries.
  iree_host_size_t image_bytes;
  uint64_t hit_count;
  uint64_t miss_count;
  uint64_t eviction_count;
} iree_hal_amdxdna_executable_reuse_cache_stats_t;

// LRU cache of loaded executables keyed by reuse key. It retains each cached
// executable, so callers that repeatedly create the same executable get the
// existing one back without serializing or parsing it again. Thread-safe.
//
// Cached executables are released outside the cache lock: their destruction
// reaches device context and command caches, which must never nest under it.
typedef struct iree_hal_amdxdna_executable_reuse_cache_t
    iree_hal_amdxdna_executable_reuse_cache_t;

iree_status_t iree_hal_amdxdna_executable_reuse_cache_create(
    iree_hal_amdxdna_executable_reuse_cache_limits_t limits,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_executable_reuse_cache_t** out_cache);

// Releases every cached executable and frees |cache|.
void iree_hal_amdxdna_executable_reuse_cache_destroy(
    iree_hal_amdxdna_executable_reuse_cache_t* cache);

// Returns a retained executable matching |key|, or NULL on a miss.
iree_hal_executable_t* iree_hal_amdxdna_executable_reuse_cache_lookup(
    iree_hal_amdxdna_executable_reuse_cache_t* cache,
    const iree_hal_amdxdna_executable_reuse_key_t* key);

// Caches |executable| under |key|, evicting least-recently-used entries to stay
// within the limits. Executables that do not have exactly one entry point with
// one run whose control code equals |key->control_code|, that exceed the byte
// budget alone, or whose key is already cached are not inserted. Returns true
// when the cache retained |executable|.
bool iree_hal_amdxdna_executable_reuse_cache_insert(
    iree_hal_amdxdna_executable_reuse_cache_t* cache,
    const iree_hal_amdxdna_executable_reuse_key_t* key,
    iree_hal_executable_t* executable);

// Releases every cached executable. Statistics counters are preserved.
void iree_hal_amdxdna_executable_reuse_cache_clear(
    iree_hal_amdxdna_executable_reuse_cache_t* cache);

void iree_hal_amdxdna_executable_reuse_cache_query_stats(
    iree_hal_amdxdna_executable_reuse_cache_t* cache,
    iree_hal_amdxdna_executable_reuse_cache_stats_t* out_stats);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_EXECUTABLE_REUSE_CACHE_H_
