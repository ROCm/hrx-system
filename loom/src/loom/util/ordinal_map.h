// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compact arena-owned maps from scoped source ordinals to output IDs.

#ifndef LOOM_UTIL_ORDINAL_MAP_H_
#define LOOM_UTIL_ORDINAL_MAP_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

// Transient map from a 24-bit source ordinal and caller-owned 3-bit domain
// to a 32-bit output ID. Domains distinguish independent uses of the same
// source identity without imposing a particular reader or analysis vocabulary.
//
// Leaves pack a 24-bit source ordinal, a 3-bit identity domain, and a 32-bit
// output ID into a tagged word. Each hash bucket stores a singleton directly
// or a compressed four-bit radix root on collision. Lookup touches one bucket
// and at most seven branches regardless of source identity distribution; there
// are no linear probe chains. Geometric bucket growth and bounded local branch
// growth retain linear arena storage, including abandoned arrays and branches.
// A projection with one mapping needs no allocation.
typedef struct loom_ordinal_map_t {
  // Caller-owned storage that remains valid throughout projection use.
  iree_arena_allocator_t* arena;
  // Hash-indexed roots with inline space for the first reached identity.
  struct {
    // Tagged leaves, aligned branch pointers, or zero for empty buckets.
    uint64_t* values;
    // Number of distinct projected source identities.
    uint32_t count;
    // Power-of-two bucket count with at most one-half identity occupancy.
    uint32_t capacity;
    // Initial bucket storage used before the first arena allocation.
    uint64_t initial_values[2];
  } buckets;
} loom_ordinal_map_t;

// Initializes an empty projection using |arena| for retained branch storage.
// Releasing the arena releases all projection storage; no teardown is required.
// The initialized projection remains at its original address during use.
void loom_ordinal_map_initialize(iree_arena_allocator_t* arena,
                                 loom_ordinal_map_t* out_projection);

// Looks up one projected identity. Returns false when it has not been reached.
bool loom_ordinal_map_lookup(const loom_ordinal_map_t* projection,
                             uint8_t domain, uint32_t source_ordinal,
                             uint32_t* out_target_id);

// Inserts one projected identity.
//
// Repeating an identical mapping is a no-op. A source identity mapping to two
// target IDs violates the owner's identity invariant and is asserted in
// debug builds. Allocation failure is the only runtime failure.
iree_status_t loom_ordinal_map_insert(loom_ordinal_map_t* projection,
                                      uint8_t domain, uint32_t source_ordinal,
                                      uint32_t target_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_UTIL_ORDINAL_MAP_H_
