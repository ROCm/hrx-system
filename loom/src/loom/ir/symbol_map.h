// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Maps interned string IDs (loom_string_id_t) to symbol IDs (uint16_t)
// for O(1) amortized lookup. A secondary index over the module's flat
// symbol table (loom_symbol_table_t), which remains the source of
// truth for symbol data.
//
// Implementation: open-addressed hash table with Fibonacci hashing,
// linear probing, ~70% max load factor, power-of-two capacity.
// Supports erasure via tombstones for use cases like symbol DCE.
//
// Arena-allocated: all storage comes from the caller's arena. No
// deinitialization needed — freeing the arena reclaims everything.
// A zero-initialized loom_symbol_map_t is valid (empty, lazy-allocs
// on first insert).
//
// Thread safety: single-owner, no synchronization.

#ifndef LOOM_IR_SYMBOL_MAP_H_
#define LOOM_IR_SYMBOL_MAP_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Sentinel value for tombstoned (erased) slots. Must not collide
// with LOOM_STRING_ID_INVALID (empty). Both reserved key values are
// rejected by insertion APIs and treated as not-found by lookup/erase.
#define LOOM_SYMBOL_MAP_TOMBSTONE \
  ((loom_string_id_t)(LOOM_STRING_ID_INVALID - 1))

typedef struct loom_symbol_map_entry_t {
  loom_string_id_t
      name_id;  // LOOM_STRING_ID_INVALID = empty, TOMBSTONE = erased.
  uint16_t symbol_id;
} loom_symbol_map_entry_t;

typedef struct loom_symbol_map_t {
  loom_symbol_map_entry_t* entries;
  iree_host_size_t capacity;         // Power of 2. 0 = unallocated.
  iree_host_size_t count;            // Live entries (excludes tombstones).
  iree_host_size_t tombstone_count;  // Erased entries still occupying slots.
} loom_symbol_map_t;

// Looks up a symbol by interned name ID.
// Returns the symbol_id, or LOOM_SYMBOL_ID_INVALID if not found.
uint16_t loom_symbol_map_find(const loom_symbol_map_t* map,
                              loom_string_id_t name_id);

// Inserts a name_id -> symbol_id mapping. The caller must ensure
// name_id is not already present (undefined behavior on duplicate) and must
// not pass LOOM_STRING_ID_INVALID or LOOM_SYMBOL_MAP_TOMBSTONE.
// Arena-allocates on first insert and when capacity growth is needed.
iree_status_t loom_symbol_map_insert(loom_symbol_map_t* map,
                                     iree_arena_allocator_t* arena,
                                     loom_string_id_t name_id,
                                     uint16_t symbol_id);

// Finds an existing entry or inserts a new one.
// If name_id is already present, sets *out_symbol_id to the existing
// symbol_id. If not present, inserts name_id -> new_symbol_id and
// sets *out_symbol_id to new_symbol_id. Reserved key values are rejected with
// INVALID_ARGUMENT.
iree_status_t loom_symbol_map_find_or_insert(loom_symbol_map_t* map,
                                             iree_arena_allocator_t* arena,
                                             loom_string_id_t name_id,
                                             uint16_t new_symbol_id,
                                             uint16_t* out_symbol_id);

// Erases name_id from the map by tombstoning its slot.
// Returns true if found and erased, false if not present.
// Tombstoned slots are reclaimed on the next capacity growth (rehash).
bool loom_symbol_map_erase(loom_symbol_map_t* map, loom_string_id_t name_id);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_SYMBOL_MAP_H_
