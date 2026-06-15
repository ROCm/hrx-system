// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/symbol_map.h"

#include <string.h>

// Fibonacci hashing for integer keys. Multiplying by the golden ratio
// constant gives excellent bit mixing for sequential integers (which
// is the common case: interned string IDs are assigned sequentially).
static uint32_t loom_symbol_map_hash(loom_string_id_t name_id) {
  return (uint32_t)name_id * 2654435769u;
}

static bool loom_symbol_map_slot_is_empty(loom_string_id_t name_id) {
  return name_id == LOOM_STRING_ID_INVALID;
}

static bool loom_symbol_map_slot_is_tombstone(loom_string_id_t name_id) {
  return name_id == LOOM_SYMBOL_MAP_TOMBSTONE;
}

static bool loom_symbol_map_slot_is_occupied(loom_string_id_t name_id) {
  return !loom_symbol_map_slot_is_empty(name_id) &&
         !loom_symbol_map_slot_is_tombstone(name_id);
}

static bool loom_symbol_map_name_id_is_reserved(loom_string_id_t name_id) {
  return loom_symbol_map_slot_is_empty(name_id) ||
         loom_symbol_map_slot_is_tombstone(name_id);
}

// Ensures the map has room for at least one more entry. Lazily
// allocates on first use, then grows by doubling with full rehash.
// Tombstones are dropped during rehash.
static iree_status_t loom_symbol_map_ensure_capacity(
    loom_symbol_map_t* map, iree_arena_allocator_t* arena) {
  // Occupied slots = live entries + tombstones. Grow when occupied
  // slots exceed ~70% of capacity.
  iree_host_size_t occupied = map->count + map->tombstone_count + 1;
  if (map->capacity > 0 && occupied * 10 < map->capacity * 7) {
    return iree_ok_status();
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t old_capacity = map->capacity;
  loom_symbol_map_entry_t* old_entries = map->entries;

  iree_host_size_t new_capacity = old_capacity == 0 ? 16 : old_capacity * 2;
  loom_symbol_map_entry_t* new_entries = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate_array(arena, new_capacity,
                                    sizeof(loom_symbol_map_entry_t),
                                    (void**)&new_entries));

  // Initialize all slots to empty.
  for (iree_host_size_t i = 0; i < new_capacity; ++i) {
    new_entries[i].name_id = LOOM_STRING_ID_INVALID;
    new_entries[i].symbol_id = LOOM_SYMBOL_ID_INVALID;
  }

  // Rehash live entries (skip tombstones — they are dropped).
  iree_host_size_t mask = new_capacity - 1;
  for (iree_host_size_t i = 0; i < old_capacity; ++i) {
    if (!loom_symbol_map_slot_is_occupied(old_entries[i].name_id)) continue;
    uint32_t hash = loom_symbol_map_hash(old_entries[i].name_id);
    iree_host_size_t slot = hash & mask;
    while (!loom_symbol_map_slot_is_empty(new_entries[slot].name_id)) {
      slot = (slot + 1) & mask;
    }
    new_entries[slot] = old_entries[i];
  }

  // Old entries are arena-allocated — no need to free.
  map->entries = new_entries;
  map->capacity = new_capacity;
  map->tombstone_count = 0;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

uint16_t loom_symbol_map_find(const loom_symbol_map_t* map,
                              loom_string_id_t name_id) {
  if (map->capacity == 0 || loom_symbol_map_name_id_is_reserved(name_id)) {
    return LOOM_SYMBOL_ID_INVALID;
  }
  iree_host_size_t mask = map->capacity - 1;
  iree_host_size_t slot = loom_symbol_map_hash(name_id) & mask;
  while (true) {
    loom_string_id_t slot_name = map->entries[slot].name_id;
    if (loom_symbol_map_slot_is_empty(slot_name)) {
      return LOOM_SYMBOL_ID_INVALID;
    }
    if (slot_name == name_id) {
      return map->entries[slot].symbol_id;
    }
    // Tombstones: continue probing past them.
    slot = (slot + 1) & mask;
  }
}

iree_status_t loom_symbol_map_insert(loom_symbol_map_t* map,
                                     iree_arena_allocator_t* arena,
                                     loom_string_id_t name_id,
                                     uint16_t symbol_id) {
  if (loom_symbol_map_name_id_is_reserved(name_id)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reserved symbol-map name id %u", name_id);
  }
  IREE_RETURN_IF_ERROR(loom_symbol_map_ensure_capacity(map, arena));
  iree_host_size_t mask = map->capacity - 1;
  iree_host_size_t slot = loom_symbol_map_hash(name_id) & mask;

  // Track the first tombstone we pass — we can reuse it for insertion.
  iree_host_size_t tombstone_slot = IREE_HOST_SIZE_MAX;
  while (true) {
    loom_string_id_t slot_name = map->entries[slot].name_id;
    if (loom_symbol_map_slot_is_empty(slot_name)) {
      if (tombstone_slot != IREE_HOST_SIZE_MAX) {
        slot = tombstone_slot;
        --map->tombstone_count;
      }
      map->entries[slot].name_id = name_id;
      map->entries[slot].symbol_id = symbol_id;
      ++map->count;
      return iree_ok_status();
    }
    if (loom_symbol_map_slot_is_tombstone(slot_name) &&
        tombstone_slot == IREE_HOST_SIZE_MAX) {
      tombstone_slot = slot;
    }
    slot = (slot + 1) & mask;
  }
}

iree_status_t loom_symbol_map_find_or_insert(loom_symbol_map_t* map,
                                             iree_arena_allocator_t* arena,
                                             loom_string_id_t name_id,
                                             uint16_t new_symbol_id,
                                             uint16_t* out_symbol_id) {
  if (loom_symbol_map_name_id_is_reserved(name_id)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "reserved symbol-map name id %u", name_id);
  }
  IREE_RETURN_IF_ERROR(loom_symbol_map_ensure_capacity(map, arena));
  iree_host_size_t mask = map->capacity - 1;
  iree_host_size_t slot = loom_symbol_map_hash(name_id) & mask;

  iree_host_size_t tombstone_slot = IREE_HOST_SIZE_MAX;
  while (true) {
    loom_string_id_t slot_name = map->entries[slot].name_id;
    if (loom_symbol_map_slot_is_empty(slot_name)) {
      // Not found: insert.
      if (tombstone_slot != IREE_HOST_SIZE_MAX) {
        slot = tombstone_slot;
        --map->tombstone_count;
      }
      map->entries[slot].name_id = name_id;
      map->entries[slot].symbol_id = new_symbol_id;
      ++map->count;
      *out_symbol_id = new_symbol_id;
      return iree_ok_status();
    }
    if (slot_name == name_id) {
      *out_symbol_id = map->entries[slot].symbol_id;
      return iree_ok_status();
    }
    if (loom_symbol_map_slot_is_tombstone(slot_name) &&
        tombstone_slot == IREE_HOST_SIZE_MAX) {
      tombstone_slot = slot;
    }
    slot = (slot + 1) & mask;
  }
}

bool loom_symbol_map_erase(loom_symbol_map_t* map, loom_string_id_t name_id) {
  if (map->capacity == 0 || loom_symbol_map_name_id_is_reserved(name_id)) {
    return false;
  }
  iree_host_size_t mask = map->capacity - 1;
  iree_host_size_t slot = loom_symbol_map_hash(name_id) & mask;
  while (true) {
    loom_string_id_t slot_name = map->entries[slot].name_id;
    if (loom_symbol_map_slot_is_empty(slot_name)) {
      return false;
    }
    if (slot_name == name_id) {
      map->entries[slot].name_id = LOOM_SYMBOL_MAP_TOMBSTONE;
      --map->count;
      ++map->tombstone_count;
      return true;
    }
    slot = (slot + 1) & mask;
  }
}
