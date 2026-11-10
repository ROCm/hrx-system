// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBHRX_SRC_LIBHRX_AMDXDNA_EXECUTABLE_CACHE_H_
#define LIBHRX_SRC_LIBHRX_AMDXDNA_EXECUTABLE_CACHE_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "hrx_amdxdna.h"
#include "iree/hal/executable.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HRX_AMDXDNA_EXECUTABLE_CACHE_CAPACITY 4096
#define HRX_AMDXDNA_EXECUTABLE_CACHE_HASH_CAPACITY 8192

typedef struct hrx_amdxdna_executable_cache_descriptor_t {
  uint64_t key_hash_1;
  uint64_t key_hash_2;
  hrx_amdxdna_xclbin_identity_t xclbin_identity;
  size_t transaction_size;
  size_t entry_point_name_size;
  size_t host_bytes;
  hrx_amdxdna_context_mode_t context_mode;
  uint32_t xclbin_ordinal;
  uint32_t pdi_ordinal;
} hrx_amdxdna_executable_cache_descriptor_t;

typedef struct hrx_amdxdna_executable_cache_entry_t {
  hrx_amdxdna_executable_cache_descriptor_t descriptor;
  iree_hal_executable_t* hal_executable;
  uint64_t last_use;
} hrx_amdxdna_executable_cache_entry_t;

typedef struct hrx_amdxdna_executable_cache_s {
  size_t max_entries;
  size_t max_host_bytes;
  size_t host_bytes;
  size_t entry_count;
  size_t hash_tombstone_count;
  uint64_t use_clock;
  uint32_t hash_table[HRX_AMDXDNA_EXECUTABLE_CACHE_HASH_CAPACITY];
  hrx_amdxdna_executable_cache_entry_t
      entries[HRX_AMDXDNA_EXECUTABLE_CACHE_CAPACITY];
} hrx_amdxdna_executable_cache_s;

// Builds an immutable cache descriptor from validated direct-transaction
// executable parameters.
bool hrx_amdxdna_executable_cache_descriptor_initialize(
    const hrx_amdxdna_xclbin_identity_t* xclbin_identity,
    const hrx_amdxdna_executable_entry_point_t* entry_point,
    const hrx_amdxdna_executable_run_t* run,
    hrx_amdxdna_executable_cache_descriptor_t* out_descriptor);

void hrx_amdxdna_executable_cache_initialize(
    size_t max_entries, size_t max_host_bytes,
    hrx_amdxdna_executable_cache_s* out_cache);

// Returns a retained executable on a hit.
iree_hal_executable_t* hrx_amdxdna_executable_cache_lookup(
    hrx_amdxdna_executable_cache_s* cache,
    const hrx_amdxdna_executable_cache_descriptor_t* descriptor);

// Retains |hal_executable| when inserted. Returns false when the descriptor is
// larger than the cache budget.
bool hrx_amdxdna_executable_cache_insert(
    hrx_amdxdna_executable_cache_s* cache,
    const hrx_amdxdna_executable_cache_descriptor_t* descriptor,
    iree_hal_executable_t* hal_executable);

// Releases every executable retained by |cache|.
void hrx_amdxdna_executable_cache_deinitialize(
    hrx_amdxdna_executable_cache_s* cache);

#ifdef __cplusplus
}
#endif

#endif  // LIBHRX_SRC_LIBHRX_AMDXDNA_EXECUTABLE_CACHE_H_
