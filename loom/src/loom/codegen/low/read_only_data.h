// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Ordered immutable-data requirements of verified Low functions and programs.

#ifndef LOOM_CODEGEN_LOW_READ_ONLY_DATA_H_
#define LOOM_CODEGEN_LOW_READ_ONLY_DATA_H_

#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_LOW_READ_ONLY_DATA_ORDINAL_INVALID UINT32_MAX

// One referenced definition in an immutable module snapshot.
typedef struct loom_low_read_only_data_requirement_t {
  // Module-local symbol retained by the descriptor immediate.
  loom_symbol_ref_t symbol;
  // Source definition supplying location and symbol ownership.
  const loom_op_t* definition;
  // Immutable definition bytes.
  iree_const_byte_span_t contents;
  // Required power-of-two byte alignment.
  uint64_t minimum_alignment;
  // Definitions that must occupy disjoint target-defined storage banks.
  loom_symbol_ref_array_t bank_conflicts;
} loom_low_read_only_data_requirement_t;

// Unique definitions in first-reference order. Records borrow their source
// definitions and bytes. The index belongs to the same immutable module
// snapshot; it does not discard conflicts with definitions outside this set.
typedef struct loom_low_read_only_data_set_t {
  // Arena-owned requirement records.
  const loom_low_read_only_data_requirement_t* values;
  // Number of records in |values|.
  iree_host_size_t count;
  // Open-addressed ordinals plus one; zero marks an empty slot. NULL for at
  // most four records, which use bounded linear lookup.
  const uint16_t* ordinal_slots;
  // Power-of-two number of entries in |ordinal_slots|, or zero.
  iree_host_size_t slot_count;
} loom_low_read_only_data_set_t;

// Zero-initialize before construction; finish after the last insertion. Every
// insertion uses the same arena and verified immutable module snapshot.
typedef struct loom_low_read_only_data_builder_t {
  // Mutable ordered records owned by the construction arena.
  loom_low_read_only_data_requirement_t* values;
  // Number of initialized records.
  iree_host_size_t count;
  // Allocated capacity of |values|.
  iree_host_size_t capacity;
  // Mutable open-addressed ordinal index.
  uint16_t* ordinal_slots;
  // Allocated power-of-two index capacity, or zero for a small set.
  iree_host_size_t slot_count;
} loom_low_read_only_data_builder_t;

// Returns the retained ordinal or LOOM_LOW_READ_ONLY_DATA_ORDINAL_INVALID.
uint32_t loom_low_read_only_data_ordinal(
    const loom_low_read_only_data_set_t* data, loom_symbol_ref_t symbol);

// Adds a verified global.rodata definition once, retaining its source facts.
iree_status_t loom_low_read_only_data_builder_insert(
    loom_low_read_only_data_builder_t* builder, const loom_module_t* module,
    loom_symbol_ref_t symbol, iree_arena_allocator_t* arena);

// Unions already inventoried definitions without revisiting their source IR.
// Existing definitions keep their first ordinal; new ones follow |data| order.
iree_status_t loom_low_read_only_data_builder_extend(
    loom_low_read_only_data_builder_t* builder,
    const loom_low_read_only_data_set_t* data, iree_arena_allocator_t* arena);

// Publishes the completed arena-owned index without copying its storage.
loom_low_read_only_data_set_t loom_low_read_only_data_builder_finish(
    const loom_low_read_only_data_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_READ_ONLY_DATA_H_
