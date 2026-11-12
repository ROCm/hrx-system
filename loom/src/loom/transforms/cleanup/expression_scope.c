// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/expression_scope.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/structural_hash.h"
#include "loom/ops/op_defs.h"
#include "loom/util/dominance.h"
#include "loom/util/segmented_storage.h"

//===----------------------------------------------------------------------===//
// Expression identity
//===----------------------------------------------------------------------===//

void loom_expression_hash_cache_initialize(
    loom_expression_hash_cache_t* cache) {
  *cache = (loom_expression_hash_cache_t){0};
  cache->type_hash = loom_type_hash(cache->type);
}

// Computes a content-aware hash for an op based on its kind, operands,
// result types, attributes, and instance flags. Uses loom_attribute_hash
// for each attribute so pointer-valued attribute kinds (I64_ARRAY,
// PREDICATE_LIST, DICT) are hashed by content rather than pointer value.
// Result types are included because they are not always derivable from
// (kind, operands, attributes) — cast and conversion ops can produce
// different result types from the same operands.
uint32_t loom_expression_hash_with_value_identity(
    const loom_module_t* module, const loom_op_t* op,
    loom_expression_value_identity_fn_t value_identity, void* user_data,
    loom_expression_hash_cache_t* cache) {
  uint32_t hash = loom_structural_hash_initialize();
  hash = loom_structural_hash_mix_u32(
      hash, (uint32_t)op->kind | ((uint32_t)op->operand_count << 16));
  hash = loom_structural_hash_mix_u32(
      hash, (uint32_t)op->result_count | ((uint32_t)op->attribute_count << 16) |
                ((uint32_t)op->instance_flags << 24));
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_value_id_t identity =
        value_identity ? value_identity(user_data, operands[i]) : operands[i];
    hash = loom_structural_hash_mix_u32(hash, identity);
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  uint8_t operand_segment_count = loom_op_vtable_operand_segment_count(vtable);
  hash = loom_structural_hash_mix_u8(hash, operand_segment_count);
  if (operand_segment_count > 0) {
    const uint16_t* segment_counts = loom_op_const_operand_segment_counts(op);
    for (uint8_t i = 0; i < operand_segment_count; ++i) {
      hash = loom_structural_hash_mix_u16(hash, segment_counts[i]);
    }
  }
  const loom_value_id_t* results = loom_op_results((loom_op_t*)op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      loom_type_t type = loom_module_value_type(module, results[i]);
      if (memcmp(&type, &cache->type, sizeof(type)) != 0) {
        cache->type = type;
        cache->type_hash = loom_type_hash(type);
      }
      hash = loom_structural_hash_mix_u32(hash, cache->type_hash);
    }
  }
  if (op->attribute_count > 0) {
    const loom_attribute_t* attrs = loom_op_attrs((loom_op_t*)op);
    for (uint8_t i = 0; i < op->attribute_count; ++i) {
      uint32_t attribute_hash = loom_attribute_hash(&attrs[i]);
      hash = loom_structural_hash_mix_u32(hash, attribute_hash);
    }
  }
  return loom_structural_hash_finalize(hash);
}

uint32_t loom_expression_hash(const loom_module_t* module, const loom_op_t* op,
                              loom_expression_hash_cache_t* cache) {
  return loom_expression_hash_with_value_identity(
      module, op, /*value_identity=*/NULL, /*user_data=*/NULL, cache);
}

// Structural equality: two ops are CSE-equivalent if they have the
// same kind, operands (same value IDs in same order), result types
// (checked via the module's value table), attributes (structurally
// equal via loom_attribute_equal), instance flags, and no regions.
bool loom_expression_equal_with_value_identity(
    const loom_module_t* module, const loom_op_t* a, const loom_op_t* b,
    loom_expression_value_identity_fn_t value_identity, void* user_data) {
  if (a->kind != b->kind) {
    return false;
  }
  if (a->operand_count != b->operand_count) {
    return false;
  }
  if (a->result_count != b->result_count) {
    return false;
  }
  if (a->attribute_count != b->attribute_count) {
    return false;
  }
  if (a->instance_flags != b->instance_flags) {
    return false;
  }
  if (a->region_count > 0) {
    return false;
  }
  const loom_value_id_t* a_operands = loom_op_const_operands(a);
  const loom_value_id_t* b_operands = loom_op_const_operands(b);
  for (uint16_t i = 0; i < a->operand_count; ++i) {
    const loom_value_id_t a_identity =
        value_identity ? value_identity(user_data, a_operands[i])
                       : a_operands[i];
    const loom_value_id_t b_identity =
        value_identity ? value_identity(user_data, b_operands[i])
                       : b_operands[i];
    if (a_identity != b_identity) {
      return false;
    }
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
      if (!loom_type_equal(a_type, b_type)) {
        return false;
      }
    }
  }
  if (a->attribute_count > 0) {
    const loom_attribute_t* a_attrs = loom_op_attrs((loom_op_t*)a);
    const loom_attribute_t* b_attrs = loom_op_attrs((loom_op_t*)b);
    for (uint8_t i = 0; i < a->attribute_count; ++i) {
      if (!loom_attribute_equal(&a_attrs[i], &b_attrs[i])) {
        return false;
      }
    }
  }
  return true;
}

bool loom_expression_equal(const loom_module_t* module, const loom_op_t* a,
                           const loom_op_t* b) {
  return loom_expression_equal_with_value_identity(
      module, a, b, /*value_identity=*/NULL, /*user_data=*/NULL);
}

//===----------------------------------------------------------------------===//
// Dominance scopes
//===----------------------------------------------------------------------===//

enum loom_expression_scope_flag_bits_e {
  // Incoming control flow can observe a different mutable state.
  LOOM_EXPRESSION_SCOPE_FLAG_STATE_BARRIER = 1u << 0,
  // Outer expressions are unavailable across lexical isolation or independent
  // execution, even when the latter permits explicitly captured values.
  LOOM_EXPRESSION_SCOPE_FLAG_ISOLATED = 1u << 1,
};
typedef uint8_t loom_expression_scope_flags_t;

struct loom_expression_scope_t {
  // Dominating block or enclosing block, including across isolation boundaries.
  loom_expression_scope_t* parent;
  // First index entry owned by this block; also its rollback checkpoint.
  uint32_t entry_start;
  // Earliest visible entry after the nearest isolation boundary.
  uint32_t visible_start;
  // Earliest visible entry after isolation or a mutable-state CFG boundary.
  uint32_t stateful_start;
  // Scope availability and traversal state.
  loom_expression_scope_flags_t flags;
};

// Entries double as the rollback log. Leaving a scope unlinks each insertion
// once and reuses its storage; there is no separate undo allocation.
typedef struct loom_expression_binding_t {
  // Available producer and caller metadata.
  loom_expression_entry_t entry;
  // Previous visible binding for this key, or UINT32_MAX for a newly added key.
  uint32_t previous;
  // Linear-probe distance from entry.hash to the slot restored on scope exit.
  // At most one slot per definition is occupied, bounding this by the u32
  // definition count even when the half-full index has more than 2^32 slots.
  uint32_t probe_distance;
} loom_expression_binding_t;

// One fixed-capacity generation of the open-addressed structural-key index.
// Growth publishes a new generation after rehashing active bindings; the pass
// arena reclaims prior pages together at reset.
typedef struct loom_expression_bucket_table_t {
  // Contiguous prefix of latest binding indexes per structural key.
  void* primary;
  // Number of slots in primary.
  iree_host_size_t primary_count;
  // Stable fixed tail pages of latest binding indexes per structural key.
  loom_segmented_storage_t pages;
  // Shift mapping a tail slot to its page index.
  uint8_t page_shift;
  // Whether slots encode binding index plus one in uint16_t entries.
  bool compact_heads;
  // Mask mapping a tail slot to its position within a page.
  uint32_t page_mask;
  // Power-of-two slot count minus one; the table stays at most half full.
  iree_host_size_t mask;
} loom_expression_bucket_table_t;

// Large expression indexes use fixed pages so every payload remains eligible
// for the reusable arena block pool. Page capacities are powers of two, making
// random access one shift, one mask, and one page-pointer load.
enum {
  LOOM_EXPRESSION_BINDING_PAGE_MAX_CAPACITY = 2048,
  LOOM_EXPRESSION_BUCKET_PAGE_MAX_BYTES = 64 * 1024,
  LOOM_EXPRESSION_SCOPE_PAGE_MAX_CAPACITY = 64,
};

// One value-keyed entry in the sparse fact-identity index. A slot for an
// original value tracks its active refinement; a slot for an identity result
// points back to the record that names its original value. Result slots remain
// allocated after scope exit so open-addressing lookup never needs tombstones.
typedef struct loom_expression_refinement_slot_t {
  loom_value_id_t value_id;
  uint32_t refinement;
} loom_expression_refinement_slot_t;

// One active dominance-scoped fact identity. Records form a stack, and each
// original value separately links its nested refinements through |previous|.
typedef struct loom_expression_refinement_t {
  loom_value_id_t root;
  loom_value_id_t result;
  uint32_t previous;
  loom_expression_scope_t* scope;
} loom_expression_refinement_t;

// Lazily allocated state used only by walks that encounter fact identities.
typedef struct loom_expression_refinement_state_t {
  loom_expression_refinement_slot_t* slots;
  iree_host_size_t slot_count;
  iree_host_size_t slot_capacity;
  loom_expression_refinement_t* refinements;
  iree_host_size_t refinement_count;
  iree_host_size_t refinement_capacity;
} loom_expression_refinement_state_t;

struct loom_expression_walk_t {
  // Module containing all visited operations.
  loom_module_t* module;
  // Semantic identity selected by the caller.
  loom_expression_identity_policy_t identity_policy;
  // Shared dominator-order operation and scope traversal.
  loom_dominance_walk_t* dominance_walk;
  // Arena owning the walk's adaptive index generations.
  iree_arena_allocator_t* arena;
  // Innermost active dominance scope, including completed dominating blocks.
  loom_expression_scope_t* active_scope;
  // Current stable page receiving newly entered dominance scopes.
  loom_expression_scope_t* scope_page;
  // Number of initialized records in scope_page.
  uint16_t scope_page_count;
  // Number of records allocated in every scope page.
  uint16_t scope_page_capacity;
  // Stable adaptive pages of entries shared by all dominance scopes.
  loom_segmented_storage_t binding_pages;
  // Shift mapping a binding index to its page index.
  uint8_t binding_page_shift;
  // Mask mapping a binding index to its position within a page.
  uint32_t binding_page_mask;
  // Latest binding per structural key.
  loom_expression_bucket_table_t buckets;
  // Whether block entry expands storage for the operations in that scope.
  bool adaptive_capacity;
  // Binding count that can be inserted without growing either index.
  uint32_t binding_capacity;
  // Number of active bindings; scopes retain checkpoints into this array.
  uint32_t binding_count;
  // Definition frontier: counts value-producing ops, not result-less barriers.
  // Each counted op owns a distinct value-table entry, so the u32 value-ID
  // domain bounds this counter even when arbitrarily many writes intervene.
  uint32_t epoch;
  // Sparse dominance-scoped FactIdentity representatives, allocated on first
  // use so ordinary expression walks retain their existing storage shape.
  loom_expression_refinement_state_t* refinement_state;
};

static uint8_t loom_expression_page_shift(const iree_arena_allocator_t* arena,
                                          iree_host_size_t element_size,
                                          iree_host_size_t maximum_capacity) {
  const iree_host_size_t maximum_allocation_size =
      iree_arena_block_pool_max_allocation_size(arena->block_pool);
  const iree_host_size_t capacity_limit = iree_min(
      maximum_capacity, iree_max(maximum_allocation_size / element_size, 1));
  uint8_t shift = 0;
  while (((iree_host_size_t)1 << (shift + 1)) <= capacity_limit) {
    ++shift;
  }
  return shift;
}

static loom_expression_binding_t* loom_expression_binding(
    loom_expression_walk_t* walk, uint32_t index) {
  loom_expression_binding_t* page =
      (loom_expression_binding_t*)loom_segmented_storage_segment(
          &walk->binding_pages, index >> walk->binding_page_shift);
  return &page[index & walk->binding_page_mask];
}

static const loom_expression_binding_t* loom_expression_const_binding(
    const loom_expression_walk_t* walk, uint32_t index) {
  const loom_expression_binding_t* page =
      (const loom_expression_binding_t*)loom_segmented_storage_const_segment(
          &walk->binding_pages, index >> walk->binding_page_shift);
  return &page[index & walk->binding_page_mask];
}

static uint32_t loom_expression_bucket_head(
    const loom_expression_bucket_table_t* table, iree_host_size_t slot) {
  if (slot < table->primary_count) {
    if (table->compact_heads) {
      const uint16_t encoded = ((const uint16_t*)table->primary)[slot];
      return encoded ? (uint32_t)encoded - 1 : UINT32_MAX;
    }
    return ((const uint32_t*)table->primary)[slot];
  }
  slot -= table->primary_count;
  const iree_host_size_t page_index = slot >> table->page_shift;
  const iree_host_size_t page_offset = slot & table->page_mask;
  const void* page =
      loom_segmented_storage_const_segment(&table->pages, (uint32_t)page_index);
  if (table->compact_heads) {
    const uint16_t encoded = ((const uint16_t*)page)[page_offset];
    return encoded ? (uint32_t)encoded - 1 : UINT32_MAX;
  }
  return ((const uint32_t*)page)[page_offset];
}

static void loom_expression_set_bucket_head(
    loom_expression_bucket_table_t* table, iree_host_size_t slot,
    uint32_t binding_index) {
  if (slot < table->primary_count) {
    if (table->compact_heads) {
      IREE_ASSERT(binding_index == UINT32_MAX || binding_index < UINT16_MAX);
      ((uint16_t*)table->primary)[slot] =
          binding_index == UINT32_MAX ? 0 : (uint16_t)(binding_index + 1);
    } else {
      ((uint32_t*)table->primary)[slot] = binding_index;
    }
    return;
  }
  slot -= table->primary_count;
  const iree_host_size_t page_index = slot >> table->page_shift;
  const iree_host_size_t page_offset = slot & table->page_mask;
  void* page =
      loom_segmented_storage_segment(&table->pages, (uint32_t)page_index);
  if (table->compact_heads) {
    IREE_ASSERT(binding_index == UINT32_MAX || binding_index < UINT16_MAX);
    ((uint16_t*)page)[page_offset] =
        binding_index == UINT32_MAX ? 0 : (uint16_t)(binding_index + 1);
    return;
  }
  ((uint32_t*)page)[page_offset] = binding_index;
}

static iree_status_t loom_expression_bucket_table_allocate(
    iree_arena_allocator_t* arena, uint32_t entry_capacity,
    loom_expression_bucket_table_t* out_table) {
  *out_table = (loom_expression_bucket_table_t){0};
  iree_host_size_t minimum_slot_count = 0;
  if (!iree_host_size_checked_mul((iree_host_size_t)entry_capacity, 2,
                                  &minimum_slot_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "expression index capacity overflow");
  }
  const iree_host_size_t slot_count =
      iree_host_size_next_power_of_two(iree_max(minimum_slot_count, 1));
  out_table->mask = slot_count - 1;
  out_table->compact_heads = entry_capacity <= UINT16_MAX;
  const iree_host_size_t element_size =
      out_table->compact_heads ? sizeof(uint16_t) : sizeof(uint32_t);
  out_table->primary_count = iree_min(
      slot_count, iree_arena_block_pool_max_allocation_size(arena->block_pool) /
                      element_size);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, out_table->primary_count, element_size, &out_table->primary));
  const uint8_t initial_byte = out_table->compact_heads ? 0 : 0xFF;
  memset(out_table->primary, initial_byte,
         out_table->primary_count * element_size);
  const iree_host_size_t tail_count = slot_count - out_table->primary_count;
  const iree_host_size_t maximum_page_capacity =
      LOOM_EXPRESSION_BUCKET_PAGE_MAX_BYTES / element_size;
  const iree_host_size_t desired_page_capacity =
      tail_count == 0 ? 1
                      : iree_min(iree_host_size_next_power_of_two(tail_count),
                                 maximum_page_capacity);
  out_table->page_shift =
      loom_expression_page_shift(arena, element_size, desired_page_capacity);
  out_table->page_mask = ((uint32_t)1 << out_table->page_shift) - 1;
  const iree_host_size_t page_capacity = (iree_host_size_t)1
                                         << out_table->page_shift;
  loom_segmented_storage_initialize(page_capacity * element_size, element_size,
                                    &out_table->pages);
  const iree_host_size_t page_count =
      tail_count == 0 ? 0 : ((tail_count - 1) >> out_table->page_shift) + 1;
  if (page_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "expression index exceeds page directory");
  }
  for (iree_host_size_t i = 0; i < page_count; ++i) {
    void* page = NULL;
    IREE_RETURN_IF_ERROR(
        loom_segmented_storage_append(&out_table->pages, arena, &page));
    memset(page, initial_byte, page_capacity * element_size);
  }
  return iree_ok_status();
}

static iree_status_t loom_expression_binding_pages_reserve(
    loom_expression_walk_t* walk, uint32_t entry_capacity) {
  const uint32_t required_page_count =
      entry_capacity == 0
          ? 0
          : ((entry_capacity - 1) >> walk->binding_page_shift) + 1;
  while (walk->binding_pages.segment_count < required_page_count) {
    void* page = NULL;
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(&walk->binding_pages,
                                                       walk->arena, &page));
  }
  return iree_ok_status();
}

static uint32_t loom_expression_bucket_entry_capacity(
    const loom_expression_bucket_table_t* table) {
  iree_host_size_t capacity = (table->mask + 1) / 2;
  if (table->compact_heads) {
    capacity = iree_min(capacity, (iree_host_size_t)UINT16_MAX);
  }
  return (uint32_t)iree_min(capacity, (iree_host_size_t)UINT32_MAX);
}

static uint32_t loom_expression_binding_page_capacity(
    const loom_expression_walk_t* walk) {
  const uint64_t capacity = (uint64_t)walk->binding_pages.segment_count
                            << walk->binding_page_shift;
  return (uint32_t)iree_min(capacity, (uint64_t)UINT32_MAX);
}

static void loom_expression_update_binding_capacity(
    loom_expression_walk_t* walk) {
  walk->binding_capacity =
      iree_min(loom_expression_bucket_entry_capacity(&walk->buckets),
               loom_expression_binding_page_capacity(walk));
}

static iree_status_t loom_expression_bucket_table_grow(
    loom_expression_walk_t* walk, uint32_t required_entry_capacity) {
  loom_expression_bucket_table_t grown = {0};
  IREE_RETURN_IF_ERROR(loom_expression_bucket_table_allocate(
      walk->arena, required_entry_capacity, &grown));

  const loom_expression_bucket_table_t old = walk->buckets;
  const iree_host_size_t old_slot_count = old.mask + 1;
  for (iree_host_size_t old_slot = 0; old_slot < old_slot_count; ++old_slot) {
    const uint32_t head = loom_expression_bucket_head(&old, old_slot);
    if (head == UINT32_MAX) {
      continue;
    }
    const loom_expression_binding_t* head_binding =
        loom_expression_const_binding(walk, head);
    iree_host_size_t new_slot = head_binding->entry.hash & grown.mask;
    while (loom_expression_bucket_head(&grown, new_slot) != UINT32_MAX) {
      new_slot = (new_slot + 1) & grown.mask;
    }
    loom_expression_set_bucket_head(&grown, new_slot, head);
    for (uint32_t binding_index = head; binding_index != UINT32_MAX;) {
      loom_expression_binding_t* binding =
          loom_expression_binding(walk, binding_index);
      binding->probe_distance =
          (uint32_t)((new_slot - binding->entry.hash) & grown.mask);
      binding_index = binding->previous;
    }
  }
  walk->buckets = grown;
  return iree_ok_status();
}

static iree_status_t loom_expression_reserve_binding_capacity(
    loom_expression_walk_t* walk, uint32_t required_entry_capacity) {
  if (required_entry_capacity <= walk->binding_capacity) {
    return iree_ok_status();
  }
  if (required_entry_capacity >
      loom_expression_bucket_entry_capacity(&walk->buckets)) {
    IREE_RETURN_IF_ERROR(
        loom_expression_bucket_table_grow(walk, required_entry_capacity));
  }
  IREE_RETURN_IF_ERROR(
      loom_expression_binding_pages_reserve(walk, required_entry_capacity));
  loom_expression_update_binding_capacity(walk);
  return iree_ok_status();
}

static loom_expression_probe_t loom_expression_scope_probe_slot(
    const loom_expression_cursor_t* cursor, uint32_t hash) {
  const loom_expression_walk_t* walk = cursor->walk;
  iree_host_size_t slot = hash & walk->buckets.mask;
  uint32_t binding_index = loom_expression_bucket_head(&walk->buckets, slot);
  while (binding_index != UINT32_MAX) {
    const loom_expression_entry_t* entry =
        &loom_expression_const_binding(walk, binding_index)->entry;
    if (entry->hash == hash) {
      const bool equal =
          walk->identity_policy.equal
              ? walk->identity_policy.equal(walk->identity_policy.user_data,
                                            cursor->module, entry->op,
                                            cursor->op)
              : loom_expression_equal(cursor->module, entry->op, cursor->op);
      if (equal) {
        return (loom_expression_probe_t){
            .slot = slot,
            .binding_index = binding_index,
        };
      }
    }
    slot = (slot + 1) & walk->buckets.mask;
    binding_index = loom_expression_bucket_head(&walk->buckets, slot);
  }
  return (loom_expression_probe_t){
      .slot = slot,
      .binding_index = UINT32_MAX,
  };
}

loom_expression_probe_t loom_expression_scope_probe(
    const loom_expression_cursor_t* cursor, uint32_t hash,
    loom_expression_lookup_flags_t flags) {
  loom_expression_walk_t* walk = cursor->walk;
  const loom_expression_scope_t* scope = cursor->scope;
  const uint32_t first =
      iree_any_bit_set(flags, LOOM_EXPRESSION_LOOKUP_FLAG_STATEFUL)
          ? scope->stateful_start
          : scope->visible_start;
  loom_expression_probe_t probe =
      loom_expression_scope_probe_slot(cursor, hash);
  if (probe.binding_index != UINT32_MAX && probe.binding_index >= first) {
    probe.entry = &loom_expression_binding(walk, probe.binding_index)->entry;
  }
  return probe;
}

void loom_expression_scope_insert(const loom_expression_cursor_t* cursor,
                                  loom_expression_probe_t probe,
                                  loom_expression_entry_t entry) {
  loom_expression_walk_t* walk = cursor->walk;
  iree_host_size_t slot = probe.slot;
  uint32_t previous = probe.binding_index;
  if (previous != UINT32_MAX && previous >= cursor->scope->entry_start) {
    loom_expression_binding(walk, previous)->entry = entry;
    return;
  }
  IREE_ASSERT(walk->binding_count < walk->binding_capacity);
  const uint32_t index = walk->binding_count++;
  *loom_expression_binding(walk, index) = (loom_expression_binding_t){
      .entry = entry,
      .previous = previous,
      .probe_distance = (uint32_t)((slot - entry.hash) & walk->buckets.mask),
  };
  loom_expression_set_bucket_head(&walk->buckets, slot, index);
}

//===----------------------------------------------------------------------===//
// Dominance-scoped fact identities
//===----------------------------------------------------------------------===//

static uint32_t loom_expression_refinement_hash(loom_value_id_t value_id) {
  uint32_t hash = loom_structural_hash_initialize();
  hash = loom_structural_hash_mix_u32(hash, value_id);
  return loom_structural_hash_finalize(hash);
}

static iree_host_size_t loom_expression_refinement_slot_index(
    const loom_expression_refinement_state_t* state, loom_value_id_t value_id) {
  const iree_host_size_t mask = state->slot_capacity - 1;
  iree_host_size_t slot = loom_expression_refinement_hash(value_id) & mask;
  while (state->slots[slot].value_id != LOOM_VALUE_ID_INVALID &&
         state->slots[slot].value_id != value_id) {
    slot = (slot + 1) & mask;
  }
  return slot;
}

static loom_expression_refinement_slot_t* loom_expression_refinement_find_slot(
    loom_expression_refinement_state_t* state, loom_value_id_t value_id) {
  if (!state || state->slot_capacity == 0) {
    return NULL;
  }
  loom_expression_refinement_slot_t* slot =
      &state->slots[loom_expression_refinement_slot_index(state, value_id)];
  return slot->value_id == value_id ? slot : NULL;
}

static iree_status_t loom_expression_refinement_reserve_slots(
    loom_expression_walk_t* walk, iree_host_size_t additional_count) {
  loom_expression_refinement_state_t* state = walk->refinement_state;
  iree_host_size_t required_count = 0;
  if (!iree_host_size_checked_add(state->slot_count, additional_count,
                                  &required_count)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "fact identity index capacity overflow");
  }
  if (required_count <= state->slot_capacity / 2) {
    return iree_ok_status();
  }

  iree_host_size_t new_capacity =
      state->slot_capacity ? state->slot_capacity : 8;
  while (required_count > new_capacity / 2) {
    if (!iree_host_size_checked_mul(new_capacity, 2, &new_capacity)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "fact identity index capacity overflow");
    }
  }
  loom_expression_refinement_slot_t* new_slots = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      loom_expression_walk_dominance_info(walk)->arena, new_capacity,
      sizeof(*new_slots), (void**)&new_slots));
  memset(new_slots, 0xFF, new_capacity * sizeof(*new_slots));

  loom_expression_refinement_slot_t* old_slots = state->slots;
  const iree_host_size_t old_capacity = state->slot_capacity;
  state->slots = new_slots;
  state->slot_capacity = new_capacity;
  for (iree_host_size_t i = 0; i < old_capacity; ++i) {
    if (old_slots[i].value_id == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const iree_host_size_t slot =
        loom_expression_refinement_slot_index(state, old_slots[i].value_id);
    new_slots[slot] = old_slots[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_expression_refinement_reserve_record(
    loom_expression_walk_t* walk) {
  loom_expression_refinement_state_t* state = walk->refinement_state;
  if (state->refinement_count < state->refinement_capacity) {
    return iree_ok_status();
  }
  const iree_host_size_t minimum_capacity =
      iree_max(state->refinement_count + 1, 8);
  return iree_arena_grow_array(
      loom_expression_walk_dominance_info(walk)->arena, state->refinement_count,
      minimum_capacity, sizeof(*state->refinements),
      &state->refinement_capacity, (void**)&state->refinements);
}

static loom_value_id_t loom_expression_refinement_root(
    loom_expression_refinement_state_t* state, loom_value_id_t value_id) {
  loom_expression_refinement_slot_t* slot =
      loom_expression_refinement_find_slot(state, value_id);
  if (!slot || slot->refinement == UINT32_MAX) {
    return value_id;
  }
  return state->refinements[slot->refinement].root;
}

iree_status_t loom_expression_scope_record_fact_identity(
    const loom_expression_cursor_t* cursor) {
  loom_expression_walk_t* walk = cursor->walk;
  if (!walk->refinement_state) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(
        loom_expression_walk_dominance_info(walk)->arena,
        sizeof(*walk->refinement_state), (void**)&walk->refinement_state));
    memset(walk->refinement_state, 0, sizeof(*walk->refinement_state));
  }
  loom_expression_refinement_state_t* state = walk->refinement_state;
  const loom_value_id_t* operands = loom_op_const_operands(cursor->op);
  const loom_value_id_t* results = loom_op_const_results(cursor->op);
  for (uint16_t i = 0; i < cursor->op->result_count; ++i) {
    const loom_value_id_t root =
        loom_expression_refinement_root(state, operands[i]);
    const loom_value_id_t result = results[i];
    loom_expression_refinement_slot_t* root_slot =
        loom_expression_refinement_find_slot(state, root);
    loom_expression_refinement_slot_t* result_slot =
        loom_expression_refinement_find_slot(state, result);
    const iree_host_size_t additional_count =
        (root_slot ? 0 : 1) + (result_slot ? 0 : 1);
    IREE_RETURN_IF_ERROR(
        loom_expression_refinement_reserve_slots(walk, additional_count));
    IREE_RETURN_IF_ERROR(loom_expression_refinement_reserve_record(walk));

    root_slot = loom_expression_refinement_find_slot(state, root);
    if (!root_slot) {
      const iree_host_size_t slot =
          loom_expression_refinement_slot_index(state, root);
      root_slot = &state->slots[slot];
      *root_slot = (loom_expression_refinement_slot_t){
          .value_id = root,
          .refinement = UINT32_MAX,
      };
      ++state->slot_count;
    }
    result_slot = loom_expression_refinement_find_slot(state, result);
    if (!result_slot) {
      const iree_host_size_t slot =
          loom_expression_refinement_slot_index(state, result);
      result_slot = &state->slots[slot];
      *result_slot = (loom_expression_refinement_slot_t){
          .value_id = result,
          .refinement = UINT32_MAX,
      };
      ++state->slot_count;
    }

    const uint32_t refinement = (uint32_t)state->refinement_count++;
    state->refinements[refinement] = (loom_expression_refinement_t){
        .root = root,
        .result = result,
        .previous = root_slot->refinement,
        .scope = cursor->scope,
    };
    root_slot->refinement = refinement;
    result_slot->refinement = refinement;
  }
  return iree_ok_status();
}

loom_value_id_t loom_expression_scope_preferred_value(
    const loom_expression_cursor_t* cursor, loom_value_id_t value_id) {
  loom_expression_refinement_state_t* state = cursor->walk->refinement_state;
  if (!state) {
    return value_id;
  }
  const loom_value_id_t root = loom_expression_refinement_root(state, value_id);
  loom_expression_refinement_slot_t* root_slot =
      loom_expression_refinement_find_slot(state, root);
  if (!root_slot || root_slot->refinement == UINT32_MAX) {
    return value_id;
  }
  const loom_value_id_t preferred =
      state->refinements[root_slot->refinement].result;
  if (!loom_type_equal(loom_module_value_type(cursor->module, value_id),
                       loom_module_value_type(cursor->module, preferred))) {
    return value_id;
  }
  return loom_value_ownership_use_count(
             loom_module_value(cursor->module, preferred)) == 0
             ? preferred
             : value_id;
}

static void loom_expression_scope_leave_refinements(
    loom_expression_walk_t* walk) {
  loom_expression_refinement_state_t* state = walk->refinement_state;
  if (!state) {
    return;
  }
  while (state->refinement_count > 0) {
    const loom_expression_refinement_t* refinement =
        &state->refinements[state->refinement_count - 1];
    if (refinement->scope != walk->active_scope) {
      break;
    }
    loom_expression_refinement_slot_t* root_slot =
        loom_expression_refinement_find_slot(state, refinement->root);
    loom_expression_refinement_slot_t* result_slot =
        loom_expression_refinement_find_slot(state, refinement->result);
    IREE_ASSERT(root_slot != NULL);
    IREE_ASSERT(result_slot != NULL);
    root_slot->refinement = refinement->previous;
    result_slot->refinement = UINT32_MAX;
    --state->refinement_count;
  }
}

// Establishes one rollback checkpoint when the shared dominance walk enters a
// block. Visibility cutoffs make isolation and mutable-state barriers constant
// time without changing structural-expression lookup policy.
static iree_status_t loom_expression_walk_enter_scope(
    void* user_data, uint32_t result_operation_count,
    loom_dominance_walk_scope_flags_t dominance_flags) {
  loom_expression_walk_t* walk = (loom_expression_walk_t*)user_data;
  if (walk->adaptive_capacity) {
    if (result_operation_count > UINT32_MAX - walk->binding_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "expression index exceeds u32 bindings");
    }
    IREE_RETURN_IF_ERROR(loom_expression_reserve_binding_capacity(
        walk, walk->binding_count + result_operation_count));
  }
  if (walk->scope_page_count == walk->scope_page_capacity) {
    loom_expression_scope_t* scope_page = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(walk->arena, walk->scope_page_capacity,
                                  sizeof(*scope_page), (void**)&scope_page));
    walk->scope_page = scope_page;
    walk->scope_page_count = 0;
  }
  loom_expression_scope_t* scope = &walk->scope_page[walk->scope_page_count++];
  loom_expression_scope_flags_t flags = 0;
  if (iree_any_bit_set(dominance_flags,
                       LOOM_DOMINANCE_WALK_SCOPE_FLAG_STATE_BARRIER)) {
    flags |= LOOM_EXPRESSION_SCOPE_FLAG_STATE_BARRIER;
  }
  if (iree_any_bit_set(
          dominance_flags,
          LOOM_DOMINANCE_WALK_SCOPE_FLAG_ISOLATED |
              LOOM_DOMINANCE_WALK_SCOPE_FLAG_INDEPENDENT_EXECUTION)) {
    flags |= LOOM_EXPRESSION_SCOPE_FLAG_ISOLATED;
  }
  loom_expression_scope_t* parent = walk->active_scope;
  *scope = (loom_expression_scope_t){
      .parent = parent,
      .entry_start = walk->binding_count,
      .visible_start =
          !parent ||
                  iree_any_bit_set(flags, LOOM_EXPRESSION_SCOPE_FLAG_ISOLATED)
              ? walk->binding_count
              : parent->visible_start,
      .stateful_start =
          !parent || iree_any_bit_set(
                         flags, LOOM_EXPRESSION_SCOPE_FLAG_ISOLATED |
                                    LOOM_EXPRESSION_SCOPE_FLAG_STATE_BARRIER)
              ? walk->binding_count
              : parent->stateful_start,
      .flags = flags,
  };
  walk->active_scope = scope;
  return iree_ok_status();
}

// Restores every binding introduced in the active scope. Strict LIFO removal
// preserves open-addressing probe chains without tombstones: keys displaced by
// a new key are removed before that key's slot is cleared.
static void loom_expression_walk_leave_scope(void* user_data) {
  loom_expression_walk_t* walk = (loom_expression_walk_t*)user_data;
  loom_expression_scope_leave_refinements(walk);
  const uint32_t checkpoint = walk->active_scope->entry_start;
  while (walk->binding_count > checkpoint) {
    const loom_expression_binding_t* binding =
        loom_expression_const_binding(walk, --walk->binding_count);
    const iree_host_size_t slot =
        ((iree_host_size_t)binding->entry.hash + binding->probe_distance) &
        walk->buckets.mask;
    loom_expression_set_bucket_head(&walk->buckets, slot, binding->previous);
  }
  walk->active_scope = walk->active_scope->parent;
}

iree_status_t loom_expression_walk_allocate(
    loom_module_t* module, loom_region_t* region,
    const loom_expression_identity_policy_t* identity_policy,
    uint32_t maximum_entry_count, iree_arena_allocator_t* arena,
    loom_expression_walk_t** out_walk) {
  *out_walk = NULL;
  loom_expression_walk_t* walk = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*walk), (void**)&walk));
  *walk = (loom_expression_walk_t){
      .module = module,
      .identity_policy = identity_policy
                             ? *identity_policy
                             : (loom_expression_identity_policy_t){0},
      .arena = arena,
      .adaptive_capacity = maximum_entry_count == 0,
  };
  uint32_t initial_entry_capacity = maximum_entry_count;
  if (initial_entry_capacity == 0 && region->block_count != 0) {
    initial_entry_capacity =
        loom_region_entry_block(region)->result_operation_count;
  }
  walk->binding_page_shift =
      loom_expression_page_shift(arena, sizeof(loom_expression_binding_t),
                                 LOOM_EXPRESSION_BINDING_PAGE_MAX_CAPACITY);
  walk->binding_page_mask = ((uint32_t)1 << walk->binding_page_shift) - 1;
  loom_segmented_storage_initialize(
      ((iree_host_size_t)walk->binding_page_mask + 1) *
          sizeof(loom_expression_binding_t),
      iree_alignof(loom_expression_binding_t), &walk->binding_pages);
  IREE_RETURN_IF_ERROR(
      loom_expression_binding_pages_reserve(walk, initial_entry_capacity));
  IREE_RETURN_IF_ERROR(loom_expression_bucket_table_allocate(
      arena, initial_entry_capacity, &walk->buckets));
  loom_expression_update_binding_capacity(walk);

  const uint8_t scope_page_shift =
      loom_expression_page_shift(arena, sizeof(loom_expression_scope_t),
                                 LOOM_EXPRESSION_SCOPE_PAGE_MAX_CAPACITY);
  walk->scope_page_capacity = (uint16_t)1 << scope_page_shift;
  walk->scope_page_count = walk->scope_page_capacity;
  IREE_RETURN_IF_ERROR(loom_dominance_walk_allocate(
      module, region,
      (loom_dominance_walk_callbacks_t){
          .user_data = walk,
          .enter_scope = loom_expression_walk_enter_scope,
          .leave_scope = loom_expression_walk_leave_scope,
      },
      arena, &walk->dominance_walk));
  *out_walk = walk;
  return iree_ok_status();
}

iree_status_t loom_expression_walk_next(loom_expression_walk_t* walk,
                                        loom_expression_cursor_t* out_cursor) {
  *out_cursor = (loom_expression_cursor_t){0};
  loom_dominance_walk_cursor_t dominance_cursor;
  IREE_RETURN_IF_ERROR(
      loom_dominance_walk_next(walk->dominance_walk, &dominance_cursor));
  if (!dominance_cursor.op) {
    return iree_ok_status();
  }
  *out_cursor = (loom_expression_cursor_t){
      .walk = walk,
      .module = walk->module,
      .scope = walk->active_scope,
      .op = dominance_cursor.op,
      .traits = dominance_cursor.traits,
      .epoch = walk->epoch,
  };
  walk->epoch += dominance_cursor.op->result_count != 0;
  return iree_ok_status();
}

const loom_dominance_info_t* loom_expression_walk_dominance_info(
    const loom_expression_walk_t* walk) {
  return loom_dominance_walk_info(walk->dominance_walk);
}

uint32_t loom_expression_observe_barriers(
    const loom_expression_cursor_t* cursor,
    loom_expression_barriers_t* barriers) {
  if (loom_traits_are_convergent(cursor->traits) ||
      loom_op_regions_have_convergent_effects(cursor->op)) {
    barriers->convergence = cursor->epoch;
  }
  if (loom_traits_may_write(cursor->traits) ||
      loom_op_regions_have_write_effects(cursor->op)) {
    barriers->memory = cursor->epoch;
  }
  return iree_any_bit_set(cursor->traits, LOOM_TRAIT_PURE)
             ? barriers->convergence
             : iree_max(barriers->convergence, barriers->memory);
}

//===----------------------------------------------------------------------===//
// Semantic eligibility and replacement
//===----------------------------------------------------------------------===//

static bool loom_expression_op_transfers_operand_ownership(
    const loom_module_t* module, const loom_op_t* op) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_use_index_t* indices = loom_op_operand_use_indices(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    const loom_value_t* value = loom_module_value(module, operands[i]);
    const loom_use_t use = loom_value_uses(value)[indices[i]];
    if (iree_any_bit_set(loom_use_flags(use), LOOM_USE_FLAG_CONSUMES)) {
      return true;
    }
  }
  return false;
}

bool loom_expression_results_are_shareable(const loom_module_t* module,
                                           const loom_op_t* op) {
  const loom_value_id_t* results = loom_op_results((loom_op_t*)op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    const loom_value_id_t result = results[i];
    if (result == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const loom_value_t* value = loom_module_value(module, result);
    if (loom_value_ownership_use_count(value) != 0) {
      return false;
    }
  }
  return true;
}

bool loom_expression_can_share_result_ownership(const loom_module_t* module,
                                                const loom_op_t* op) {
  return !loom_expression_op_transfers_operand_ownership(module, op) &&
         loom_expression_results_are_shareable(module, op);
}

bool loom_expression_op_is_reusable(const loom_module_t* module,
                                    const loom_op_t* op,
                                    loom_trait_flags_t traits) {
  return op->result_count != 0 && op->region_count == 0 &&
         !loom_traits_has_side_effects(traits) &&
         !loom_traits_has_unique_identity(traits) &&
         loom_expression_can_share_result_ownership(module, op);
}

static bool loom_expression_user_is_not_fact_identity(const loom_op_t* user_op,
                                                      void* user_data) {
  (void)user_data;
  return !loom_traits_are_fact_identity(user_op->traits);
}

iree_status_t loom_expression_replace(const loom_expression_cursor_t* cursor,
                                      loom_op_t* op, loom_op_t* existing) {
  loom_module_t* module = cursor->module;
  loom_value_id_t* results = loom_op_results(op);
  const loom_value_id_t* existing_results = loom_op_results(existing);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < op->result_count && iree_status_is_ok(status); ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID &&
        existing_results[i] != LOOM_VALUE_ID_INVALID) {
      const loom_value_id_t preferred =
          loom_expression_scope_preferred_value(cursor, existing_results[i]);
      if (preferred != existing_results[i]) {
        // Ordinary consumers benefit from the strongest dominating identity,
        // while a following FactIdentity must see the canonical producer. The
        // latter also leaves the old value's embedded type/attribute uses for
        // the full replacement below, allowing an equivalent identity's
        // predicates to become structurally identical and CSE away.
        status = loom_value_replace_uses_if(
            module, results[i], preferred,
            loom_expression_user_is_not_fact_identity, NULL);
      }
      if (iree_status_is_ok(status)) {
        status = loom_value_replace_all_uses_with(module, results[i],
                                                  existing_results[i]);
      }
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_op_erase(module, op);
  }
  return status;
}
