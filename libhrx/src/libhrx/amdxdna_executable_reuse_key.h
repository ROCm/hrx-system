// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef LIBHRX_SRC_LIBHRX_AMDXDNA_EXECUTABLE_REUSE_KEY_H_
#define LIBHRX_SRC_LIBHRX_AMDXDNA_EXECUTABLE_REUSE_KEY_H_

#include <stdint.h>

#include "hrx_amdxdna.h"
#include "iree/hal/drivers/amdxdna/executable_reuse_cache.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HRX_AMDXDNA_EXECUTABLE_REUSE_KEY_INLINE_CAPACITY 256

// Device reuse key of a single-run direct-transaction executable. The
// descriptor holds the producer's xclbin identity, the entry point context
// mode, PDI ordinal and name; the control code is the run transaction itself,
// so the key aliases |run->transaction| and must not outlive it. Not copyable:
// the descriptor may point into |inline_descriptor|.
typedef struct hrx_amdxdna_executable_reuse_key_t {
  iree_hal_amdxdna_executable_reuse_key_t hal_key;
  // Heap descriptor storage when the descriptor exceeds the inline capacity.
  uint8_t* heap_descriptor;
  uint8_t inline_descriptor[HRX_AMDXDNA_EXECUTABLE_REUSE_KEY_INLINE_CAPACITY];
} hrx_amdxdna_executable_reuse_key_t;

hrx_status_t hrx_amdxdna_executable_reuse_key_initialize(
    const hrx_amdxdna_xclbin_identity_t* xclbin_identity,
    const hrx_amdxdna_executable_entry_point_t* entry_point,
    const hrx_amdxdna_executable_run_t* run,
    hrx_amdxdna_executable_reuse_key_t* out_key);

void hrx_amdxdna_executable_reuse_key_deinitialize(
    hrx_amdxdna_executable_reuse_key_t* key);

#ifdef __cplusplus
}
#endif

#endif  // LIBHRX_SRC_LIBHRX_AMDXDNA_EXECUTABLE_REUSE_KEY_H_
