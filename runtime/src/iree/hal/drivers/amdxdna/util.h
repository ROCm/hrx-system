// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_UTIL_H_
#define IREE_HAL_DRIVERS_AMDXDNA_UTIL_H_

#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"

// Fast non-cryptographic hash for bucketing byte contents. Callers that need
// equality must still compare bytes exactly. Chain calls through |seed| to hash
// several spans.
static inline uint64_t iree_hal_amdxdna_hash_bytes(
    uint64_t seed, iree_const_byte_span_t bytes) {
  const uint64_t kMultiplier = 0x9E3779B97F4A7C15ull;
  uint64_t hash = seed ^ 0xCBF29CE484222325ull ^ (uint64_t)bytes.data_length;
  iree_host_size_t i = 0;
  for (; i + sizeof(uint64_t) <= bytes.data_length; i += sizeof(uint64_t)) {
    uint64_t word = 0;
    memcpy(&word, bytes.data + i, sizeof(word));
    hash = (hash ^ word) * kMultiplier;
    hash ^= hash >> 29;
  }
  for (; i < bytes.data_length; ++i) {
    hash = (hash ^ bytes.data[i]) * kMultiplier;
  }
  return hash ^ (hash >> 32);
}

static inline iree_status_t iree_hal_amdxdna_status_from_errno(
    int err, const char* message) {
  if (err == 0) return iree_ok_status();
  if (err < 0) err = -err;
  return iree_make_status(iree_status_code_from_errno(err), "%s: errno %d",
                          message, err);
}

#ifndef NDEBUG
#define IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_value, vtable, subvalue_t) \
  (IREE_HAL_ASSERT_TYPE(base_value, &vtable), (subvalue_t*)(base_value))
#else
#define IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_value, vtable, subvalue_t) \
  ((subvalue_t*)(base_value))
#endif

#endif  // IREE_HAL_DRIVERS_AMDXDNA_UTIL_H_
