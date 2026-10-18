// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/scf/scf_memory.h"

bool loom_scf_memory_space_is_read_ahead_source(
    loom_value_fact_memory_space_t space) {
  return space == LOOM_VALUE_FACT_MEMORY_SPACE_GLOBAL ||
         space == LOOM_VALUE_FACT_MEMORY_SPACE_CONSTANT ||
         space == LOOM_VALUE_FACT_MEMORY_SPACE_DESCRIPTOR;
}

static iree_host_size_t loom_scf_memory_slot(const loom_op_t* op,
                                             iree_host_size_t capacity) {
  uintptr_t bits = (uintptr_t)op;
  bits ^= bits >> 17;
  bits *= (uintptr_t)0xed5ad4bbU;
  bits ^= bits >> 11;
  return (iree_host_size_t)bits & (capacity - 1);
}

static void loom_scf_memory_insert_entry(loom_scf_memory_entry_t* entries,
                                         iree_host_size_t capacity,
                                         loom_scf_memory_entry_t entry) {
  iree_host_size_t slot = loom_scf_memory_slot(entry.op, capacity);
  while (entries[slot].op) {
    slot = (slot + 1) & (capacity - 1);
  }
  entries[slot] = entry;
}

iree_status_t loom_scf_memory_insert(loom_scf_memory_t* memory,
                                     const loom_op_t* op, uint8_t space,
                                     uint32_t scope) {
  if (memory->count + 1 > memory->capacity - memory->capacity / 4) {
    if (memory->capacity > IREE_HOST_SIZE_MAX / 2) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "SCF memory table capacity overflow");
    }
    const iree_host_size_t capacity =
        memory->capacity ? memory->capacity * 2 : 8;
    loom_scf_memory_entry_t* entries = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        memory->arena, capacity, sizeof(*entries), (void**)&entries));
    memset(entries, 0, capacity * sizeof(*entries));
    for (iree_host_size_t i = 0; i < memory->capacity; ++i) {
      if (memory->entries[i].op) {
        loom_scf_memory_insert_entry(entries, capacity, memory->entries[i]);
      }
    }
    memory->entries = entries;
    memory->capacity = capacity;
  }
  loom_scf_memory_insert_entry(
      memory->entries, memory->capacity,
      (loom_scf_memory_entry_t){.op = op, .space = space, .scope = scope});
  ++memory->count;
  return iree_ok_status();
}

uint8_t loom_scf_memory_lookup(const loom_scf_memory_t* memory,
                               const loom_op_t* op) {
  iree_host_size_t slot = loom_scf_memory_slot(op, memory->capacity);
  while (memory->entries[slot].op != op) {
    IREE_ASSERT(memory->entries[slot].op);
    slot = (slot + 1) & (memory->capacity - 1);
  }
  return memory->entries[slot].space;
}

iree_status_t loom_scf_memory_project(loom_scf_memory_t* memory,
                                      loom_ir_remap_t* remap) {
  IREE_ASSERT(remap->op_projection.cursor == remap->op_projection.count);
  for (iree_host_size_t i = 0; i < remap->op_projection.count; ++i) {
    const loom_ir_remap_op_projection_t* entry =
        &remap->op_projection.entries[i];
    const uint8_t space = loom_scf_memory_lookup(memory, entry->source_op);
    IREE_RETURN_IF_ERROR(
        loom_scf_memory_insert(memory, entry->target_op, space, UINT32_MAX));
  }
  remap->op_projection.entries = NULL;
  remap->op_projection.count = remap->op_projection.cursor = 0;
  return iree_ok_status();
}
