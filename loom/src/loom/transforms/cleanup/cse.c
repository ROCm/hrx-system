// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/cse.h"

#include <stdint.h>
#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/codegen/low/pipeline/pass_environment.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/target/selection.h"
#include "loom/util/dominance.h"

#define LOOM_CSE_STATISTICS(V, statistics_type)                        \
  V(statistics_type, expressions_eliminated, "expressions-eliminated", \
    "Number of redundant expressions removed.")

LOOM_PASS_STATISTICS_DEFINE(loom_cse_statistics, loom_cse_statistics_t,
                            LOOM_CSE_STATISTICS)

static const loom_pass_info_t loom_cse_pass_info_storage = {
    .name = IREE_SVL("cse"),
    .description = IREE_SVL("Eliminate common subexpressions."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_cse_statistics_layout,
};

const loom_pass_info_t* loom_cse_pass_info(void) {
  return &loom_cse_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Op hashing and equality
//===----------------------------------------------------------------------===//

// FNV-1a hash over a byte range, folded into a running hash.
static uint32_t loom_cse_hash_bytes(const void* data, iree_host_size_t length,
                                    uint32_t hash) {
  const uint8_t* bytes = (const uint8_t*)data;
  for (iree_host_size_t i = 0; i < length; ++i) {
    hash ^= bytes[i];
    hash *= 16777619u;
  }
  return hash;
}

// Computes a content-aware hash for an op based on its kind, operands,
// result types, attributes, and instance flags. Uses loom_attribute_hash
// for each attribute so pointer-valued attribute kinds (I64_ARRAY,
// PREDICATE_LIST, DICT) are hashed by content rather than pointer value.
// Result types are included because they are not always derivable from
// (kind, operands, attributes) — cast and conversion ops can produce
// different result types from the same operands.
static uint32_t loom_cse_hash_op(const loom_module_t* module,
                                 const loom_op_t* op) {
  uint32_t hash = 2166136261u;
  hash = loom_cse_hash_bytes(&op->kind, sizeof(op->kind), hash);
  const loom_value_id_t* operands = loom_op_operands((loom_op_t*)op);
  hash = loom_cse_hash_bytes(
      operands, (iree_host_size_t)op->operand_count * sizeof(loom_value_id_t),
      hash);
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  uint8_t operand_segment_count = loom_op_vtable_operand_segment_count(vtable);
  if (operand_segment_count > 0) {
    hash = loom_cse_hash_bytes(
        loom_op_const_operand_segment_counts(op),
        (iree_host_size_t)operand_segment_count * sizeof(uint16_t), hash);
  }
  const loom_value_id_t* results = loom_op_results((loom_op_t*)op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      loom_type_t type = loom_module_value_type(module, results[i]);
      hash = loom_cse_hash_bytes(&type, sizeof(type), hash);
    }
  }
  if (op->attribute_count > 0) {
    const loom_attribute_t* attrs = loom_op_attrs((loom_op_t*)op);
    for (uint8_t i = 0; i < op->attribute_count; ++i) {
      uint32_t attribute_hash = loom_attribute_hash(&attrs[i]);
      hash = loom_cse_hash_bytes(&attribute_hash, sizeof(attribute_hash), hash);
    }
  }
  hash = loom_cse_hash_bytes(&op->instance_flags, sizeof(op->instance_flags),
                             hash);
  return hash;
}

// Structural equality: two ops are CSE-equivalent if they have the
// same kind, operands (same value IDs in same order), result types
// (checked via the module's value table), attributes (structurally
// equal via loom_attribute_equal), instance flags, and no regions.
static bool loom_cse_ops_equal(const loom_module_t* module, const loom_op_t* a,
                               const loom_op_t* b) {
  if (a->kind != b->kind) return false;
  if (a->operand_count != b->operand_count) return false;
  if (a->result_count != b->result_count) return false;
  if (a->attribute_count != b->attribute_count) return false;
  if (a->instance_flags != b->instance_flags) return false;
  if (a->region_count > 0) return false;
  const loom_value_id_t* a_operands = loom_op_operands((loom_op_t*)a);
  const loom_value_id_t* b_operands = loom_op_operands((loom_op_t*)b);
  if (memcmp(a_operands, b_operands,
             (iree_host_size_t)a->operand_count * sizeof(loom_value_id_t)) !=
      0) {
    return false;
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, a);
  uint8_t operand_segment_count = loom_op_vtable_operand_segment_count(vtable);
  if (operand_segment_count > 0 &&
      memcmp(loom_op_const_operand_segment_counts(a),
             loom_op_const_operand_segment_counts(b),
             (iree_host_size_t)operand_segment_count * sizeof(uint16_t)) != 0) {
    return false;
  }
  const loom_value_id_t* a_results = loom_op_results((loom_op_t*)a);
  const loom_value_id_t* b_results = loom_op_results((loom_op_t*)b);
  for (uint16_t i = 0; i < a->result_count; ++i) {
    if (a_results[i] != LOOM_VALUE_ID_INVALID &&
        b_results[i] != LOOM_VALUE_ID_INVALID) {
      loom_type_t a_type = loom_module_value_type(module, a_results[i]);
      loom_type_t b_type = loom_module_value_type(module, b_results[i]);
      if (!loom_type_equal(a_type, b_type)) return false;
    }
  }
  if (a->attribute_count > 0) {
    const loom_attribute_t* a_attrs = loom_op_attrs((loom_op_t*)a);
    const loom_attribute_t* b_attrs = loom_op_attrs((loom_op_t*)b);
    for (uint8_t i = 0; i < a->attribute_count; ++i) {
      if (!loom_attribute_equal(&a_attrs[i], &b_attrs[i])) return false;
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Target-low state identity
//===----------------------------------------------------------------------===//

typedef struct loom_cse_low_state_t {
  uint64_t dependencies;
  uint64_t reads;
  uint64_t writes;
} loom_cse_low_state_t;

static uint64_t loom_cse_low_state_bit(uint16_t register_class_id) {
  return register_class_id < 64 ? ((uint64_t)1u << register_class_id)
                                : UINT64_MAX;
}

static uint16_t loom_cse_low_descriptor_state_register_class_id(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_operand_t* operand) {
  IREE_ASSERT_EQ(operand->reg_class_alt_count, 1);
  IREE_ASSERT_LT(operand->reg_class_alt_start,
                 descriptor_set->reg_class_alt_count);
  const loom_low_reg_class_alt_t* alt =
      &descriptor_set->reg_class_alts[operand->reg_class_alt_start];
  IREE_ASSERT_NE(alt->reg_class_id, LOOM_LOW_REG_CLASS_NONE);
  IREE_ASSERT_LT(alt->reg_class_id, descriptor_set->reg_class_count);
  return alt->reg_class_id;
}

static loom_cse_low_state_t loom_cse_low_descriptor_state(
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_descriptor_t* descriptor) {
  loom_cse_low_state_t state = {0};
  for (uint16_t i = 0; i < descriptor->operand_count; ++i) {
    const uint32_t operand_index = descriptor->operand_start + i;
    IREE_ASSERT_LT(operand_index, descriptor_set->operand_count);
    const loom_low_operand_t* operand =
        &descriptor_set->operands[operand_index];
    if (iree_any_bit_set(operand->flags,
                         LOOM_LOW_OPERAND_FLAG_SCHEDULE_ONLY_STATE)) {
      continue;
    }
    const loom_low_operand_flags_t state_flags =
        operand->flags &
        (LOOM_LOW_OPERAND_FLAG_STATE_READ | LOOM_LOW_OPERAND_FLAG_STATE_WRITE);
    if (state_flags == 0) {
      continue;
    }

    const uint64_t state_bit =
        loom_cse_low_state_bit(loom_cse_low_descriptor_state_register_class_id(
            descriptor_set, operand));
    if (iree_any_bit_set(state_flags, LOOM_LOW_OPERAND_FLAG_STATE_READ)) {
      state.reads |= state_bit;
      const bool has_explicit_packet_value =
          i >= descriptor->result_count &&
          loom_low_operand_role_is_packet_operand(operand->role);
      if (!has_explicit_packet_value) {
        state.dependencies |= state_bit;
      }
    }
    if (iree_any_bit_set(state_flags, LOOM_LOW_OPERAND_FLAG_STATE_WRITE)) {
      state.writes |= state_bit;
      if (i < descriptor->result_count) {
        state.dependencies |= state_bit;
      }
    }
  }
  return state;
}

static iree_status_t loom_cse_resolve_low_packet_state(
    const loom_module_t* module, const loom_low_resolved_target_t* target,
    const loom_op_t* op, loom_cse_low_state_t* out_state) {
  *out_state = (loom_cse_low_state_t){0};
  if (!target || !target->descriptor_set) {
    return iree_ok_status();
  }

  loom_low_resolved_descriptor_packet_t packet = {0};
  IREE_RETURN_IF_ERROR(
      loom_low_resolve_descriptor_packet(module, target, op, &packet));
  if (packet.kind == LOOM_LOW_DESCRIPTOR_PACKET_NONE || !packet.descriptor) {
    return iree_ok_status();
  }

  *out_state =
      loom_cse_low_descriptor_state(target->descriptor_set, packet.descriptor);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Hash table with tombstone support
//===----------------------------------------------------------------------===//
//
// Open-addressed hash table using linear probing with tombstone entries.
// Tombstones are left when write barriers invalidate non-PURE entries.
// Without tombstones, invalidation breaks probe chains: if entries A, B, C
// all hash to slot 5 and B is evicted (set to NULL), lookups for C would
// stop at the NULL in B's slot and fail. Tombstones keep the probe chain
// intact — find skips them, insert can reuse them.

// Sentinel pointer value for tombstone entries. Distinguishable from
// NULL (empty slot) and any valid op pointer (which will be at least
// 4-byte aligned and thus > 1).
#define LOOM_CSE_TOMBSTONE ((loom_op_t*)(uintptr_t)1)

typedef struct loom_cse_entry_t {
  // NULL = empty, TOMBSTONE = deleted, else = live.
  loom_op_t* op;
  // FNV-1a hash of the op's identity.
  uint32_t hash;
  // Trait flags at insert time.
  loom_trait_flags_t traits;
  // Target state registers this entry depends on.
  uint64_t state_dependency_bits;
} loom_cse_entry_t;

typedef struct loom_cse_table_t {
  loom_cse_entry_t* entries;
  // Slots containing live non-PURE entries, used for O(non-pure) write
  // barriers.
  iree_host_size_t* non_pure_slots;
  // Slots containing live entries whose identity depends on target state.
  iree_host_size_t* state_dependency_slots;
  // Always a power of 2.
  iree_host_size_t capacity;
  // Live entries only (excludes tombstones).
  iree_host_size_t count;
  // Count of live non-PURE slot entries.
  iree_host_size_t non_pure_slot_count;
  // Count of live target-state-dependent slot entries.
  iree_host_size_t state_dependency_slot_count;
  // Capacity of non_pure_slots.
  iree_host_size_t non_pure_slot_capacity;
  // Capacity of state_dependency_slots.
  iree_host_size_t state_dependency_slot_capacity;
} loom_cse_table_t;

static iree_status_t loom_cse_table_initialize(iree_arena_allocator_t* arena,
                                               iree_host_size_t capacity,
                                               iree_host_size_t max_entry_count,
                                               loom_cse_table_t* table) {
  table->entries = NULL;
  table->non_pure_slots = NULL;
  table->state_dependency_slots = NULL;
  table->capacity = capacity;
  table->count = 0;
  table->non_pure_slot_count = 0;
  table->state_dependency_slot_count = 0;
  table->non_pure_slot_capacity = max_entry_count;
  table->state_dependency_slot_capacity = max_entry_count;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(loom_cse_entry_t), (void**)&table->entries));
  memset(table->entries, 0, capacity * sizeof(loom_cse_entry_t));
  if (max_entry_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, max_entry_count, sizeof(iree_host_size_t),
        (void**)&table->non_pure_slots));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, max_entry_count, sizeof(iree_host_size_t),
        (void**)&table->state_dependency_slots));
  }
  return iree_ok_status();
}

// Finds an equivalent op in this table only (no chain walk).
// Skips tombstones during probing. Stops at empty (NULL) slots.
static loom_op_t* loom_cse_table_find(const loom_cse_table_t* table,
                                      const loom_module_t* module,
                                      const loom_op_t* op, uint32_t hash) {
  iree_host_size_t mask = table->capacity - 1;
  iree_host_size_t slot = hash & mask;
  while (true) {
    const loom_cse_entry_t* entry = &table->entries[slot];
    if (!entry->op) return NULL;  // Empty — end of probe chain.
    if (entry->op != LOOM_CSE_TOMBSTONE && entry->hash == hash &&
        loom_cse_ops_equal(module, entry->op, op)) {
      return entry->op;
    }
    slot = (slot + 1) & mask;
  }
}

// Inserts an op into this table. Caller must verify no duplicate exists.
// Reuses tombstone slots when available to reclaim space.
static void loom_cse_table_insert(loom_cse_table_t* table, loom_op_t* op,
                                  uint32_t hash, loom_trait_flags_t traits,
                                  uint64_t state_dependency_bits) {
  iree_host_size_t mask = table->capacity - 1;
  iree_host_size_t slot = hash & mask;
  while (table->entries[slot].op &&
         table->entries[slot].op != LOOM_CSE_TOMBSTONE) {
    slot = (slot + 1) & mask;
  }
  table->entries[slot] = (loom_cse_entry_t){
      .op = op,
      .hash = hash,
      .traits = traits,
      .state_dependency_bits = state_dependency_bits,
  };
  ++table->count;
  if (!(traits & LOOM_TRAIT_PURE)) {
    IREE_ASSERT(table->non_pure_slot_count < table->non_pure_slot_capacity);
    table->non_pure_slots[table->non_pure_slot_count++] = slot;
  }
  if (state_dependency_bits != 0) {
    IREE_ASSERT(table->state_dependency_slot_count <
                table->state_dependency_slot_capacity);
    table->state_dependency_slots[table->state_dependency_slot_count++] = slot;
  }
}

// Evicts all non-PURE entries from the table by replacing them with
// tombstones. PURE entries (no memory effects) survive write barriers
// because their results don't depend on mutable resource state.
// The non_pure_slots side list avoids scanning the full hash table capacity on
// every write in large blocks that alternate reads and writes.
static void loom_cse_table_invalidate_reads(loom_cse_table_t* table) {
  if (table->non_pure_slot_count == 0) return;
  for (iree_host_size_t i = 0; i < table->non_pure_slot_count; ++i) {
    iree_host_size_t slot = table->non_pure_slots[i];
    loom_op_t* op = table->entries[slot].op;
    if (op && op != LOOM_CSE_TOMBSTONE &&
        !(table->entries[slot].traits & LOOM_TRAIT_PURE)) {
      table->entries[slot].op = LOOM_CSE_TOMBSTONE;
      --table->count;
    }
  }
  table->non_pure_slot_count = 0;
}

// Evicts entries whose identity depends on target state written by the current
// packet. Stale slots from other invalidation paths are compacted while walking
// the side list.
static void loom_cse_table_invalidate_state_dependencies(
    loom_cse_table_t* table, uint64_t state_write_bits) {
  if (state_write_bits == 0 || table->state_dependency_slot_count == 0) return;
  iree_host_size_t live_slot_count = 0;
  for (iree_host_size_t i = 0; i < table->state_dependency_slot_count; ++i) {
    iree_host_size_t slot = table->state_dependency_slots[i];
    loom_cse_entry_t* entry = &table->entries[slot];
    if (!entry->op || entry->op == LOOM_CSE_TOMBSTONE) {
      continue;
    }
    if ((entry->state_dependency_bits & state_write_bits) != 0) {
      entry->op = LOOM_CSE_TOMBSTONE;
      --table->count;
      continue;
    }
    table->state_dependency_slots[live_slot_count++] = slot;
  }
  table->state_dependency_slot_count = live_slot_count;
}

// Evicts every entry from the table. This is used for execution-state barriers:
// a pure value materialized under one dynamic participant set may not be
// reusable after a later convergent operation changes that set.
static void loom_cse_table_invalidate_all(loom_cse_table_t* table) {
  if (table->count == 0) return;
  memset(table->entries, 0, table->capacity * sizeof(*table->entries));
  table->count = 0;
  table->non_pure_slot_count = 0;
  table->state_dependency_slot_count = 0;
}

//===----------------------------------------------------------------------===//
// Scope chain
//===----------------------------------------------------------------------===//
//
// CSE operates on a scope chain that mirrors the region nesting tree.
// Each scope owns a hash table of CSE candidates from its block.
// Lookup walks up the chain: an inner op can be replaced by an
// equivalent outer op if the outer scope is reachable (non-isolated).
// Write barriers propagate upward: a write inside a nested region
// invalidates read-only entries in all ancestor scopes.
//
// For isolated-from-above regions, the child scope's parent is NULL —
// the chain is broken and no outer values are considered.
//
// Multi-block regions use the computed CFG dominator tree even if no cleanup
// pass has stamped the region flag yet. Every reachable block scope is parented
// by its immediate dominator scope. That makes producers from any dominating
// block visible to dominated descendants without speculating work across
// control-flow predicates.
//
// Arena allocation: all scopes and tables are allocated from a
// dedicated scope arena that is reset between top-level blocks.
// This bounds peak memory to the largest single subtree rather than
// the sum of all subtrees across the function.

typedef struct loom_cse_scope_t {
  loom_cse_table_t table;
  struct loom_cse_scope_t* parent;
} loom_cse_scope_t;

// Allocates a new scope with a hash table sized for |block|.
static iree_status_t loom_cse_scope_allocate(iree_arena_allocator_t* arena,
                                             loom_cse_scope_t* parent,
                                             const loom_block_t* block,
                                             loom_cse_scope_t** out_scope) {
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(loom_cse_scope_t), (void**)out_scope));
  (*out_scope)->parent = parent;
  iree_host_size_t capacity = iree_host_size_next_power_of_two(
      iree_max((iree_host_size_t)block->op_count * 2, 16));
  return loom_cse_table_initialize(arena, capacity, block->op_count,
                                   &(*out_scope)->table);
}

// Walks the scope chain looking for an equivalent op.
static loom_op_t* loom_cse_scope_lookup(const loom_cse_scope_t* scope,
                                        const loom_module_t* module,
                                        const loom_op_t* op, uint32_t hash) {
  for (const loom_cse_scope_t* s = scope; s; s = s->parent) {
    loom_op_t* found = loom_cse_table_find(&s->table, module, op, hash);
    if (found) return found;
  }
  return NULL;
}

// Propagates write barrier up the entire scope chain.
static void loom_cse_scope_invalidate_reads(loom_cse_scope_t* scope) {
  for (loom_cse_scope_t* s = scope; s; s = s->parent) {
    loom_cse_table_invalidate_reads(&s->table);
  }
}

// Propagates target-state write barriers up the entire scope chain.
static void loom_cse_scope_invalidate_state_dependencies(
    loom_cse_scope_t* scope, uint64_t state_write_bits) {
  if (state_write_bits == 0) return;
  for (loom_cse_scope_t* s = scope; s; s = s->parent) {
    loom_cse_table_invalidate_state_dependencies(&s->table, state_write_bits);
  }
}

// Propagates an execution-state barrier up the entire scope chain.
static void loom_cse_scope_invalidate_all(loom_cse_scope_t* scope) {
  for (loom_cse_scope_t* s = scope; s; s = s->parent) {
    loom_cse_table_invalidate_all(&s->table);
  }
}

//===----------------------------------------------------------------------===//
// DFS stack
//===----------------------------------------------------------------------===//
//
// Iterative DFS over the region tree using an explicit stack of frames.
// Each frame represents a block being processed with a cursor into its
// ordered op list. When an op has nested regions, child frames are pushed
// onto the stack and processed before the parent frame resumes.
//
// The stack is allocated from the pass arena (pass lifetime) and grows
// by doubling via iree_arena_grow_array. Old arrays are abandoned in
// the arena and reclaimed when the pass completes.

typedef struct loom_cse_frame_t {
  loom_block_t* block;
  loom_op_t* next_op;
  loom_cse_scope_t* scope;
} loom_cse_frame_t;

typedef struct loom_cse_stack_t {
  loom_cse_frame_t* frames;
  iree_host_size_t count;
  iree_host_size_t capacity;
} loom_cse_stack_t;

#define LOOM_CSE_INITIAL_STACK_CAPACITY 32

static iree_status_t loom_cse_stack_initialize(iree_arena_allocator_t* arena,
                                               loom_cse_stack_t* stack) {
  stack->count = 0;
  stack->capacity = LOOM_CSE_INITIAL_STACK_CAPACITY;
  return iree_arena_allocate_array(
      arena, stack->capacity, sizeof(loom_cse_frame_t), (void**)&stack->frames);
}

// Ensures the stack has room for |additional| frames. Grows by doubling
// if needed. Must be called before pushing any frames.
static iree_status_t loom_cse_stack_reserve(loom_cse_stack_t* stack,
                                            iree_arena_allocator_t* arena,
                                            iree_host_size_t additional) {
  iree_host_size_t required = stack->count + additional;
  if (required <= stack->capacity) return iree_ok_status();
  return iree_arena_grow_array(arena, stack->count, required,
                               sizeof(loom_cse_frame_t), &stack->capacity,
                               (void**)&stack->frames);
}

static void loom_cse_stack_push(loom_cse_stack_t* stack, loom_block_t* block,
                                loom_cse_scope_t* scope) {
  IREE_ASSERT(stack->count < stack->capacity);
  stack->frames[stack->count++] = (loom_cse_frame_t){
      .block = block,
      .next_op = block->first_op,
      .scope = scope,
  };
}

//===----------------------------------------------------------------------===//
// Region child frame pushing
//===----------------------------------------------------------------------===//

static loom_cse_scope_t* loom_cse_cfg_scope_parent_for_block(
    const loom_dominance_info_t* dominance, loom_region_t* region,
    loom_cse_scope_t* parent_scope, loom_cse_scope_t** block_scopes,
    const loom_block_t* block) {
  const loom_block_t* immediate_dominator =
      loom_dominance_immediate_dominator_block(dominance, block);
  uint16_t immediate_dominator_index = 0;
  if (!loom_region_try_block_index(region, immediate_dominator,
                                   &immediate_dominator_index)) {
    return parent_scope;
  }
  return block_scopes[immediate_dominator_index]
             ? block_scopes[immediate_dominator_index]
             : parent_scope;
}

static iree_status_t loom_cse_compute_cfg_block_order(
    iree_arena_allocator_t* arena, const loom_dominance_info_t* dominance,
    loom_region_t* region, uint16_t** out_order) {
  *out_order = NULL;
  if (region->block_count == 0) return iree_ok_status();

  uint16_t* order = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, region->block_count, sizeof(*order), (void**)&order));
  bool* scheduled = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, region->block_count, sizeof(*scheduled), (void**)&scheduled));
  memset(scheduled, 0,
         (iree_host_size_t)region->block_count * sizeof(*scheduled));

  uint16_t scheduled_count = 0;
  while (scheduled_count < region->block_count) {
    bool made_progress = false;
    for (uint16_t block_index = 0; block_index < region->block_count;
         ++block_index) {
      if (scheduled[block_index]) continue;

      const loom_block_t* block = loom_region_const_block(region, block_index);
      const loom_block_t* immediate_dominator =
          loom_dominance_immediate_dominator_block(dominance, block);
      uint16_t immediate_dominator_index = 0;
      const bool has_same_region_dominator = loom_region_try_block_index(
          region, immediate_dominator, &immediate_dominator_index);
      if (has_same_region_dominator && !scheduled[immediate_dominator_index]) {
        continue;
      }

      scheduled[block_index] = true;
      order[scheduled_count++] = block_index;
      made_progress = true;
    }

    if (!made_progress) {
      // Malformed CFG dominance should already be rejected by verification.
      // Keep CSE conservative by scheduling the remaining blocks in region
      // order; their scopes fall back to the nearest known parent scope.
      for (uint16_t block_index = 0; block_index < region->block_count;
           ++block_index) {
        if (scheduled[block_index]) continue;
        scheduled[block_index] = true;
        order[scheduled_count++] = block_index;
      }
    }
  }

  *out_order = order;
  return iree_ok_status();
}

static iree_status_t loom_cse_push_cfg_region_block_frames(
    loom_cse_stack_t* stack, iree_arena_allocator_t* pass_arena,
    iree_arena_allocator_t* scope_arena, const loom_dominance_info_t* dominance,
    loom_region_t* region, loom_cse_scope_t* parent_scope) {
  IREE_RETURN_IF_ERROR(
      loom_cse_stack_reserve(stack, pass_arena, region->block_count));

  loom_cse_scope_t** block_scopes = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(scope_arena, region->block_count,
                                sizeof(*block_scopes), (void**)&block_scopes));
  memset(block_scopes, 0,
         (iree_host_size_t)region->block_count * sizeof(*block_scopes));

  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(region, block_index);
    IREE_RETURN_IF_ERROR(loom_cse_scope_allocate(
        scope_arena, /*parent=*/NULL, block, &block_scopes[block_index]));
  }
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    loom_block_t* block = loom_region_block(region, block_index);
    block_scopes[block_index]->parent = loom_cse_cfg_scope_parent_for_block(
        dominance, region, parent_scope, block_scopes, block);
  }

  uint16_t* order = NULL;
  IREE_RETURN_IF_ERROR(
      loom_cse_compute_cfg_block_order(pass_arena, dominance, region, &order));
  for (int32_t i = (int32_t)region->block_count - 1; i >= 0; --i) {
    uint16_t block_index = order[i];
    loom_cse_stack_push(stack, loom_region_block(region, block_index),
                        block_scopes[block_index]);
  }
  return iree_ok_status();
}

// Pushes frames for all blocks in a region.
//
// For single-block regions (the common case), pushes one frame with
// a child scope whose parent is |parent_scope|.
//
// For multi-block regions, block scopes follow the computed immediate-dominator
// tree and frames are processed in dominator-before-dominated order.
static iree_status_t loom_cse_push_region_block_frames(
    loom_cse_stack_t* stack, iree_arena_allocator_t* pass_arena,
    iree_arena_allocator_t* scope_arena, const loom_dominance_info_t* dominance,
    loom_region_t* region, loom_cse_scope_t* parent_scope) {
  if (!region || region->block_count == 0) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_cse_stack_reserve(stack, pass_arena, region->block_count));

  if (region->block_count == 1) {
    loom_block_t* entry_block = loom_region_entry_block(region);
    loom_cse_scope_t* child_scope = NULL;
    IREE_RETURN_IF_ERROR(loom_cse_scope_allocate(scope_arena, parent_scope,
                                                 entry_block, &child_scope));
    loom_cse_stack_push(stack, entry_block, child_scope);
    return iree_ok_status();
  }

  return loom_cse_push_cfg_region_block_frames(stack, pass_arena, scope_arena,
                                               dominance, region, parent_scope);
}

// Pushes child frames for all nested regions of an op.
static iree_status_t loom_cse_push_region_frames(
    loom_cse_stack_t* stack, iree_arena_allocator_t* pass_arena,
    iree_arena_allocator_t* scope_arena, const loom_dominance_info_t* dominance,
    const loom_op_t* op, loom_cse_scope_t* parent_scope) {
  loom_region_t** regions = loom_op_regions((loom_op_t*)op);
  // Push regions in reverse order so the first region's first block
  // is on top of the stack and processed first.
  for (int32_t r = (int32_t)op->region_count - 1; r >= 0; --r) {
    IREE_RETURN_IF_ERROR(loom_cse_push_region_block_frames(
        stack, pass_arena, scope_arena, dominance, regions[r], parent_scope));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// CSE eligibility
//===----------------------------------------------------------------------===//

// Returns true if the op must not be CSE'd. An op prevents CSE if it
// has observable side effects (writes, unknown effects, non-determinism),
// depends on a convergent participant set, or produces a unique identity per
// execution (allocations).
static inline bool loom_cse_prevents_cse(loom_trait_flags_t traits) {
  return (traits & (LOOM_TRAIT_WRITES_MEMORY | LOOM_TRAIT_UNKNOWN_EFFECTS |
                    LOOM_TRAIT_NON_DETERMINISTIC | LOOM_TRAIT_UNIQUE_IDENTITY |
                    LOOM_TRAIT_CONVERGENT)) != 0;
}

static bool loom_cse_use_consumes_operand(const loom_use_t use) {
  const loom_op_t* user_op = loom_use_user_op(use);
  const uint16_t operand_index = loom_use_operand_index(use);
  const loom_tied_result_t* tied_results = loom_op_tied_results(user_op);
  for (uint16_t i = 0; i < user_op->tied_result_count; ++i) {
    if (tied_results[i].operand_index == operand_index) {
      return true;
    }
  }
  return false;
}

static bool loom_cse_result_is_consumed(const loom_module_t* module,
                                        const loom_op_t* op) {
  const loom_value_id_t* results = loom_op_results((loom_op_t*)op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_value_id_t result = results[i];
    if (result == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const loom_value_t* value = loom_module_value(module, result);
    const loom_use_t* use = NULL;
    loom_value_for_each_use(value, use) {
      if (loom_cse_use_consumes_operand(*use)) {
        return true;
      }
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

iree_status_t loom_cse_run(loom_pass_t* pass, loom_module_t* module,
                           loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();
  loom_cse_statistics_t* statistics = loom_cse_statistics(pass);

  loom_low_resolved_target_t low_target = {0};
  const loom_low_resolved_target_t* low_target_ptr = NULL;
  if (loom_low_function_def_isa(function.op)) {
    const loom_low_pass_capability_t* low_capability =
        loom_low_pass_capability_from_pass(pass);
    const loom_low_descriptor_registry_t* descriptor_registry =
        loom_low_pass_capability_descriptor_registry(low_capability);
    if (descriptor_registry) {
      const loom_target_pass_capability_t* target_capability =
          loom_target_pass_capability_from_pass(pass);
      IREE_RETURN_IF_ERROR(loom_low_resolve_function_target(
          module, function.op, descriptor_registry,
          loom_target_pass_capability_target_selection(target_capability),
          pass->diagnostic_emitter, &low_target));
      if (low_target.descriptor_set) {
        low_target_ptr = &low_target;
      }
    }
  }

  loom_dominance_info_t dominance = {0};
  IREE_RETURN_IF_ERROR(
      loom_dominance_info_initialize(module, pass->arena, &dominance));

  // Scope arena: holds all scope structs and hash table arrays.
  // Reset between root regions to bound peak memory to the largest single
  // function-like region. Shares the pass arena's block pool.
  iree_arena_allocator_t scope_arena;
  iree_arena_initialize(pass->arena->block_pool, &scope_arena);

  // DFS stack: allocated from the pass arena (pass lifetime).
  loom_cse_stack_t stack;
  iree_status_t status = loom_cse_stack_initialize(pass->arena, &stack);

  for (uint8_t region_index = 0;
       region_index < loom_func_like_region_count(function) &&
       iree_status_is_ok(status);
       ++region_index) {
    loom_region_t* region = loom_func_like_region(function, region_index);
    if (!region) continue;

    iree_arena_reset(&scope_arena);
    stack.count = 0;
    status = loom_cse_push_region_block_frames(&stack, pass->arena,
                                               &scope_arena, &dominance, region,
                                               /*parent_scope=*/NULL);
    while (iree_status_is_ok(status) && stack.count > 0) {
      loom_cse_frame_t* frame = &stack.frames[stack.count - 1];

      // Block done — pop frame.
      if (!frame->next_op) {
        --stack.count;
        continue;
      }

      loom_op_t* op = frame->next_op;
      frame->next_op = op->next_op;
      if (op->flags & LOOM_OP_FLAG_DEAD) continue;

      const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
      if (!vtable) {
        // Unknown op kind — conservatively treat as a write barrier.
        // The verifier enforces that all ops have vtables, so this
        // should not occur in valid IR. But if it does, treating the
        // op as transparent would silently allow CSE across writes.
        loom_cse_scope_invalidate_reads(frame->scope);
        continue;
      }

      loom_trait_flags_t traits = loom_op_effective_traits(module, op);
      loom_cse_low_state_t low_state = {0};
      status = loom_cse_resolve_low_packet_state(module, low_target_ptr, op,
                                                 &low_state);
      if (!iree_status_is_ok(status)) break;
      const uint64_t state_write_bits = low_state.writes;

      // Execution-state barrier: convergent operations can change which
      // dynamic participants define later values. A pure materialization before
      // the barrier may only be valid for the old participant set, so no
      // candidate survives.
      if (loom_traits_are_convergent(traits) ||
          loom_op_regions_have_convergent_effects(op)) {
        loom_cse_scope_invalidate_all(frame->scope);
      } else if (loom_traits_may_write(traits) ||
                 loom_op_regions_have_write_effects(op)) {
        // Write barrier: a write or unknown-effect op invalidates all non-PURE
        // entries up the scope chain. This must happen before any other checks
        // because result-less writes still invalidate.
        loom_cse_scope_invalidate_reads(frame->scope);
      }

      // Push child frames for nested regions.
      if (op->region_count > 0) {
        loom_cse_scope_invalidate_state_dependencies(frame->scope,
                                                     state_write_bits);
        loom_cse_scope_t* parent_scope =
            loom_traits_is_isolated(traits) ? NULL : frame->scope;

        status = loom_cse_push_region_frames(&stack, pass->arena, &scope_arena,
                                             &dominance, op, parent_scope);
        if (!iree_status_is_ok(status)) break;
        continue;  // Ops with regions are never CSE candidates.
      }

      // CSE candidate check: must have results, no regions (handled
      // above), no writes, no unknown effects, and be deterministic.
      if (op->result_count == 0) {
        loom_cse_scope_invalidate_state_dependencies(frame->scope,
                                                     state_write_bits);
        continue;
      }
      if (op->tied_result_count != 0) {
        loom_cse_scope_invalidate_state_dependencies(frame->scope,
                                                     state_write_bits);
        continue;
      }
      if (loom_cse_result_is_consumed(module, op)) {
        loom_cse_scope_invalidate_state_dependencies(frame->scope,
                                                     state_write_bits);
        continue;
      }
      if (loom_cse_prevents_cse(traits)) {
        loom_cse_scope_invalidate_state_dependencies(frame->scope,
                                                     state_write_bits);
        continue;
      }

      // Look up the scope chain for an equivalent op.
      uint32_t hash = loom_cse_hash_op(module, op);
      loom_op_t* existing =
          loom_cse_scope_lookup(frame->scope, module, op, hash);
      if (existing) {
        // Replace all uses and erase.
        loom_value_id_t* op_results = loom_op_results(op);
        loom_value_id_t* existing_results = loom_op_results(existing);
        for (uint16_t r = 0; r < op->result_count; ++r) {
          if (op_results[r] != LOOM_VALUE_ID_INVALID &&
              existing_results[r] != LOOM_VALUE_ID_INVALID) {
            status = loom_value_replace_all_uses_with(module, op_results[r],
                                                      existing_results[r]);
            if (!iree_status_is_ok(status)) break;
          }
        }
        if (!iree_status_is_ok(status)) break;
        status = loom_op_erase(module, op);
        if (!iree_status_is_ok(status)) break;
        loom_pass_mark_changed(pass);
        ++statistics->expressions_eliminated;
      } else {
        loom_cse_scope_invalidate_state_dependencies(frame->scope,
                                                     state_write_bits);
        loom_cse_table_insert(&frame->scope->table, op, hash, traits,
                              low_state.dependencies);
      }
    }
  }

  iree_arena_deinitialize(&scope_arena);
  return status;
}
