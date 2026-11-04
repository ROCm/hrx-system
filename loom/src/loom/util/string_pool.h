// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_UTIL_STRING_POOL_H_
#define LOOM_UTIL_STRING_POOL_H_

#include "iree/base/api.h"

// A static string slice: low 23 bits are a byte offset, bit 23 selects the
// bank, and high 8 bits are its byte length. Bank zero holds primary bytes and
// bank one holds optional secondary bytes. Either bank may be shared by several
// table owners. Slices may overlap and need no terminator.
// Generation proves each slice fits its bank, each bank is at most 8 MiB, and
// every length is at most 255. The reference stays four bytes in descriptor
// rows.
typedef uint32_t loom_string_ref_t;

// An absent string, distinct from the empty slice at reference zero. This
// encoding cannot name a complete slice within the pool size limit.
#define LOOM_STRING_REF_NONE UINT32_MAX

// Constructs a reference for constant tables. The producer owns both bounds.
#define LOOM_STRING_REF(bank, offset, length)                                  \
  ((loom_string_ref_t)(((uint32_t)(length) << 24) | ((uint32_t)(bank) << 23) | \
                       (uint32_t)(offset)))

typedef struct loom_string_pool_t {
  // Borrowed primary and secondary bytes; returned views retain their lifetime.
  const char* data[2];
  // Byte count of each bank, excluding any trailing C literal terminator.
  uint32_t data_length[2];
} loom_string_pool_t;

// Checks whether a reference names a complete slice. Used by table verification
// and debug assertions; compiler consumers use producer-validated references.
static inline bool loom_string_pool_contains(const loom_string_pool_t* pool,
                                             loom_string_ref_t ref) {
  const uint32_t bank = (ref >> 23) & 1;
  const uint32_t offset = ref & UINT32_C(0x007FFFFF);
  const uint32_t length = ref >> 24;
  return ref != LOOM_STRING_REF_NONE && pool->data[bank] != NULL &&
         offset <= pool->data_length[bank] &&
         length <= pool->data_length[bank] - offset;
}

// Returns a view of a trusted slice without reading any length from the pool.
static inline iree_string_view_t loom_string_pool_get(
    const loom_string_pool_t* pool, loom_string_ref_t ref) {
  IREE_ASSERT(loom_string_pool_contains(pool, ref));
  return iree_make_string_view(
      pool->data[(ref >> 23) & 1] + (ref & UINT32_C(0x007FFFFF)), ref >> 24);
}

#endif  // LOOM_UTIL_STRING_POOL_H_
