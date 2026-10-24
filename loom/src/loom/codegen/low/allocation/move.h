// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Allocation-owned physical move rows.

#ifndef LOOM_CODEGEN_LOW_ALLOCATION_MOVE_H_
#define LOOM_CODEGEN_LOW_ALLOCATION_MOVE_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation/assignment.h"

#ifdef __cplusplus
extern "C" {
#endif

// One target-visible allocation unit within the owning plan's descriptor set.
// Semantic value identity and type do not belong to a physical move: the
// descriptor register class defines its storage and encoding.
typedef struct loom_low_move_location_t {
  // Target-visible storage kind, a loom_low_allocation_location_kind_t value.
  uint16_t location_kind;
  // Register class in the owning plan's descriptor set.
  uint16_t descriptor_reg_class_id;
  // Physical register, target ID, provisional spill slot, or move-cell ordinal.
  uint32_t location;
} loom_low_move_location_t;

static_assert(sizeof(loom_low_move_location_t) == 8,
              "physical unit locations must remain compact");

enum loom_low_move_kind_e {
  // Reads the source unit and overwrites the destination unit.
  LOOM_LOW_MOVE_KIND_COPY = 0,
  // Reads both units and exchanges their contents in one native operation.
  LOOM_LOW_MOVE_KIND_EXCHANGE = 1,
};
typedef uint8_t loom_low_move_kind_t;

// One synchronous physical transfer between allocation units.
typedef struct loom_low_move_t {
  // Unit overwritten by the transfer, and also read by an exchange.
  loom_low_move_location_t destination;
  // Unit read by the transfer, and also overwritten by an exchange.
  loom_low_move_location_t source;
  // Copy or full-unit exchange semantics retained by allocation.
  loom_low_move_kind_t kind;
} loom_low_move_t;

// Each transfer has the same number of reads and writes. An exchange's two
// reads precede both writes at one native issue position.
static inline uint32_t loom_low_move_access_count(const loom_low_move_t* move) {
  return move->kind == LOOM_LOW_MOVE_KIND_EXCHANGE ? 2 : 1;
}

// Returns read |index|, which is less than loom_low_move_access_count(move).
static inline const loom_low_move_location_t* loom_low_move_read(
    const loom_low_move_t* move, uint32_t index) {
  return index == 0 ? &move->source : &move->destination;
}

// Returns write |index|, which is less than loom_low_move_access_count(move).
static inline const loom_low_move_location_t* loom_low_move_write(
    const loom_low_move_t* move, uint32_t index) {
  return index == 0 ? &move->destination : &move->source;
}

// Invocation-owned cell for a final synchronous transport temporary. Cells
// are disjoint from each other, authored storage, and outgoing ABI packets.
// The target's frame layout assigns concrete offsets once before emission.
typedef struct loom_low_move_storage_t {
  // Register class defining value or ownership-preserving reference transfers.
  uint16_t register_class;
  // Function-local storage space, a loom_storage_space_t value.
  uint16_t space;
  // Bytes in one saved allocation unit.
  uint32_t byte_length;
  // Required power-of-two byte alignment.
  uint32_t byte_alignment;
} loom_low_move_storage_t;

// Contiguous range in an allocation-owned move row table.
typedef struct loom_low_move_range_t {
  // First move row in the table.
  iree_host_size_t start;
  // Number of move rows in the range.
  iree_host_size_t count;
} loom_low_move_range_t;

// Final move rows and cycle-scratch metadata for one parallel move group.
typedef struct loom_low_move_group_t {
  // Sequential physical moves emitted for the group.
  loom_low_move_range_t moves;
  // First entry in the allocation scratch-move index table.
  iree_host_size_t scratch_move_index_start;
  // Number of distinct cycle-scratch locations used by the group.
  iree_host_size_t scratch_move_index_count;
} loom_low_move_group_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_ALLOCATION_MOVE_H_
