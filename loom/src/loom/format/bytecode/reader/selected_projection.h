// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Reached-only source-to-output identity projection for selective reads.

#ifndef LOOM_FORMAT_BYTECODE_READER_SELECTED_PROJECTION_H_
#define LOOM_FORMAT_BYTECODE_READER_SELECTED_PROJECTION_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Identity domains requiring memoization during selective materialization.
//
// Strings are projected directly through the output interner and registered
// operations resolve directly through the context. Every other source-table
// identity receives an entry only when reached.
typedef enum loom_bytecode_selected_projection_domain_e {
  LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SYMBOL_NAME = 0,
  LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_SOURCE = 1,
  LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_ENCODING = 2,
  LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_TYPE = 3,
  LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_LOCATION = 4,
  LOOM_BYTECODE_SELECTED_PROJECTION_DOMAIN_COUNT = 5,
} loom_bytecode_selected_projection_domain_t;

// Transient source-ordinal to compact output-ID map.
//
// The map owns reclaimable host allocations and must be deinitialized. Each
// occupied slot packs a 24-bit source ordinal, a 3-bit identity domain, and a
// 24-bit output ID into one 64-bit word. Capacity is always a power of two and
// occupancy never exceeds one half, bounding probe lengths for reached facts.
typedef struct loom_bytecode_selected_projection_t {
  // Allocator owning slot storage across rehashes and teardown.
  iree_allocator_t allocator;
  // Packed open-addressed slot storage.
  struct {
    // Allocator-owned packed entries; UINT64_MAX denotes an empty slot.
    uint64_t* values;
    // Number of occupied slots.
    iree_host_size_t count;
    // Power-of-two slot count, or zero before the first insertion.
    iree_host_size_t capacity;
  } slots;
} loom_bytecode_selected_projection_t;

// Initializes an empty projection using |allocator| for transient storage.
void loom_bytecode_selected_projection_initialize(
    iree_allocator_t allocator,
    loom_bytecode_selected_projection_t* out_projection);

// Releases all storage owned by |projection|.
void loom_bytecode_selected_projection_deinitialize(
    loom_bytecode_selected_projection_t* projection);

// Looks up one projected identity. Returns false when it has not been reached.
bool loom_bytecode_selected_projection_lookup(
    const loom_bytecode_selected_projection_t* projection,
    loom_bytecode_selected_projection_domain_t domain, uint32_t source_ordinal,
    uint32_t* out_target_id);

// Inserts one projected identity.
//
// Repeating an identical mapping is a no-op. A source identity mapping to two
// target IDs violates the selective materializer invariant and is asserted in
// debug builds. Allocation failure is the only runtime failure.
iree_status_t loom_bytecode_selected_projection_insert(
    loom_bytecode_selected_projection_t* projection,
    loom_bytecode_selected_projection_domain_t domain, uint32_t source_ordinal,
    uint32_t target_id);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_FORMAT_BYTECODE_READER_SELECTED_PROJECTION_H_
