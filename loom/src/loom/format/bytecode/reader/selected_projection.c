// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/selected_projection.h"

#include <string.h>

#define LOOM_BYTECODE_SELECTED_PROJECTION_ORDINAL_BITS 24
#define LOOM_BYTECODE_SELECTED_PROJECTION_ORDINAL_MASK \
  ((UINT64_C(1) << LOOM_BYTECODE_SELECTED_PROJECTION_ORDINAL_BITS) - 1)
#define LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SHIFT 24
#define LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_MASK \
  (UINT64_C(7) << LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SHIFT)
#define LOOM_BYTECODE_SELECTED_PROJECTION_TARGET_SHIFT 32
#define LOOM_BYTECODE_SELECTED_PROJECTION_TARGET_MASK \
  (LOOM_BYTECODE_SELECTED_PROJECTION_ORDINAL_MASK     \
   << LOOM_BYTECODE_SELECTED_PROJECTION_TARGET_SHIFT)
#define LOOM_BYTECODE_SELECTED_PROJECTION_KEY_MASK  \
  (LOOM_BYTECODE_SELECTED_PROJECTION_ORDINAL_MASK | \
   LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_MASK)
#define LOOM_BYTECODE_SELECTED_PROJECTION_EMPTY UINT64_MAX
#define LOOM_BYTECODE_SELECTED_PROJECTION_INITIAL_CAPACITY 16

static uint32_t loom_bytecode_selected_projection_key(
    loom_bytecode_selected_projection_domain_t domain,
    uint32_t source_ordinal) {
  IREE_ASSERT(domain >= LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SYMBOL_NAME &&
              domain < LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_COUNT);
  IREE_ASSERT(source_ordinal <= LOOM_BYTECODE_SELECTED_PROJECTION_ORDINAL_MASK);
  return source_ordinal |
         ((uint32_t)domain << LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SHIFT);
}

static uint32_t loom_bytecode_selected_projection_hash(uint32_t key) {
  return key * UINT32_C(2654435769);
}

static uint64_t loom_bytecode_selected_projection_pack(uint32_t key,
                                                       uint32_t target_id) {
  IREE_ASSERT(target_id <= LOOM_BYTECODE_SELECTED_PROJECTION_ORDINAL_MASK);
  return key | ((uint64_t)target_id
                << LOOM_BYTECODE_SELECTED_PROJECTION_TARGET_SHIFT);
}

static uint32_t loom_bytecode_selected_projection_unpack_target(
    uint64_t entry) {
  return (uint32_t)((entry & LOOM_BYTECODE_SELECTED_PROJECTION_TARGET_MASK) >>
                    LOOM_BYTECODE_SELECTED_PROJECTION_TARGET_SHIFT);
}

static void loom_bytecode_selected_projection_insert_packed(
    uint64_t* slots, iree_host_size_t capacity, uint64_t entry) {
  const uint32_t key =
      (uint32_t)(entry & LOOM_BYTECODE_SELECTED_PROJECTION_KEY_MASK);
  const iree_host_size_t mask = capacity - 1;
  iree_host_size_t slot = loom_bytecode_selected_projection_hash(key) & mask;
  while (slots[slot] != LOOM_BYTECODE_SELECTED_PROJECTION_EMPTY) {
    slot = (slot + 1) & mask;
  }
  slots[slot] = entry;
}

static iree_status_t loom_bytecode_selected_projection_ensure_capacity(
    loom_bytecode_selected_projection_t* projection) {
  if (projection->slots.capacity > 0 &&
      projection->slots.count + 1 <= projection->slots.capacity / 2) {
    return iree_ok_status();
  }

  iree_host_size_t new_capacity =
      LOOM_BYTECODE_SELECTED_PROJECTION_INITIAL_CAPACITY;
  if (projection->slots.capacity > 0 &&
      !iree_host_size_checked_mul(projection->slots.capacity, 2,
                                  &new_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "selected projection capacity overflow");
  }

  uint64_t* new_slots = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_array_uninitialized(
      projection->allocator, new_capacity, sizeof(*new_slots),
      (void**)&new_slots));
  memset(new_slots, 0xFF, new_capacity * sizeof(*new_slots));
  for (iree_host_size_t i = 0; i < projection->slots.capacity; ++i) {
    const uint64_t entry = projection->slots.values[i];
    if (entry != LOOM_BYTECODE_SELECTED_PROJECTION_EMPTY) {
      loom_bytecode_selected_projection_insert_packed(new_slots, new_capacity,
                                                      entry);
    }
  }
  iree_allocator_free(projection->allocator, projection->slots.values);
  projection->slots.values = new_slots;
  projection->slots.capacity = new_capacity;
  return iree_ok_status();
}

void loom_bytecode_selected_projection_initialize(
    iree_allocator_t allocator,
    loom_bytecode_selected_projection_t* out_projection) {
  *out_projection = (loom_bytecode_selected_projection_t){
      .allocator = allocator,
  };
}

void loom_bytecode_selected_projection_deinitialize(
    loom_bytecode_selected_projection_t* projection) {
  iree_allocator_free(projection->allocator, projection->slots.values);
  memset(projection, 0, sizeof(*projection));
}

bool loom_bytecode_selected_projection_lookup(
    const loom_bytecode_selected_projection_t* projection,
    loom_bytecode_selected_projection_domain_t domain, uint32_t source_ordinal,
    uint32_t* out_target_id) {
  if (projection->slots.capacity == 0) {
    return false;
  }
  const uint32_t key =
      loom_bytecode_selected_projection_key(domain, source_ordinal);
  const iree_host_size_t mask = projection->slots.capacity - 1;
  iree_host_size_t slot = loom_bytecode_selected_projection_hash(key) & mask;
  while (true) {
    const uint64_t entry = projection->slots.values[slot];
    if (entry == LOOM_BYTECODE_SELECTED_PROJECTION_EMPTY) {
      return false;
    }
    if ((entry & LOOM_BYTECODE_SELECTED_PROJECTION_KEY_MASK) == key) {
      *out_target_id = loom_bytecode_selected_projection_unpack_target(entry);
      return true;
    }
    slot = (slot + 1) & mask;
  }
}

iree_status_t loom_bytecode_selected_projection_insert(
    loom_bytecode_selected_projection_t* projection,
    loom_bytecode_selected_projection_domain_t domain, uint32_t source_ordinal,
    uint32_t target_id) {
  const uint32_t key =
      loom_bytecode_selected_projection_key(domain, source_ordinal);
  uint32_t existing_target_id = 0;
  if (loom_bytecode_selected_projection_lookup(
          projection, domain, source_ordinal, &existing_target_id)) {
    IREE_ASSERT(existing_target_id == target_id);
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_bytecode_selected_projection_ensure_capacity(projection));
  loom_bytecode_selected_projection_insert_packed(
      projection->slots.values, projection->slots.capacity,
      loom_bytecode_selected_projection_pack(key, target_id));
  ++projection->slots.count;
  return iree_ok_status();
}
