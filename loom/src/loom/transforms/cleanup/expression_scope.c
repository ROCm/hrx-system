// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/expression_scope.h"

#include <string.h>

#include "loom/analysis/ownership.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/structural_hash.h"
#include "loom/ops/op_defs.h"
#include "loom/util/dominance.h"

//===----------------------------------------------------------------------===//
// Expression identity
//===----------------------------------------------------------------------===//

// Computes a content-aware hash for an op based on its kind, operands,
// result types, attributes, and instance flags. Uses loom_attribute_hash
// for each attribute so pointer-valued attribute kinds (I64_ARRAY,
// PREDICATE_LIST, DICT) are hashed by content rather than pointer value.
// Result types are included because they are not always derivable from
// (kind, operands, attributes) — cast and conversion ops can produce
// different result types from the same operands.
uint32_t loom_expression_hash(const loom_module_t* module,
                              const loom_op_t* op) {
  uint32_t hash = loom_structural_hash_initialize();
  hash = loom_structural_hash_mix_u16(hash, op->kind);
  hash = loom_structural_hash_mix_u16(hash, op->operand_count);
  const loom_value_id_t* operands = loom_op_operands((loom_op_t*)op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    hash = loom_structural_hash_mix_u32(hash, operands[i]);
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
  hash = loom_structural_hash_mix_u16(hash, op->result_count);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      loom_type_t type = loom_module_value_type(module, results[i]);
      hash = loom_structural_hash_mix_u32(hash, loom_type_hash(type));
    }
  }
  hash = loom_structural_hash_mix_u8(hash, op->attribute_count);
  if (op->attribute_count > 0) {
    const loom_attribute_t* attrs = loom_op_attrs((loom_op_t*)op);
    for (uint8_t i = 0; i < op->attribute_count; ++i) {
      uint32_t attribute_hash = loom_attribute_hash(&attrs[i]);
      hash = loom_structural_hash_mix_u32(hash, attribute_hash);
    }
  }
  hash = loom_structural_hash_mix_u8(hash, op->instance_flags);
  return loom_structural_hash_finalize(hash);
}

// Structural equality: two ops are CSE-equivalent if they have the
// same kind, operands (same value IDs in same order), result types
// (checked via the module's value table), attributes (structurally
// equal via loom_attribute_equal), instance flags, and no regions.
static bool loom_expression_equal(const loom_module_t* module,
                                  const loom_op_t* a, const loom_op_t* b) {
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
// Dominance scopes
//===----------------------------------------------------------------------===//

struct loom_expression_scope_t {
  // Fixed-capacity structural identity table for this block.
  loom_expression_entry_t* entries;
  // Power-of-two table capacity, with at least twice the block's op count.
  iree_host_size_t capacity;
  // Dominating block or non-isolated enclosing block.
  loom_expression_scope_t* parent;
  // Next pending block in the region traversal.
  loom_expression_scope_t* previous_frame;
  // Next operation in this block; saved before the caller can erase an op.
  loom_op_t* next_op;
  // Whether incoming control flow can observe a different mutable state.
  bool blocks_stateful_parent_lookup;
};

struct loom_expression_walk_t {
  // Module containing all visited operations.
  loom_module_t* module;
  // Arena owning the walk and all root-region scopes.
  iree_arena_allocator_t* arena;
  // Cached CFG graphs and dominance order.
  loom_dominance_info_t dominance;
  // Top pending block frame.
  loom_expression_scope_t* top;
  // Definition frontier: counts value-producing ops, not result-less barriers.
  // Each counted op owns a distinct value-table entry, so the u32 value-ID
  // domain bounds this counter even when arbitrarily many writes intervene.
  uint32_t epoch;
};

static iree_host_size_t loom_expression_scope_slot(
    const loom_expression_scope_t* scope, const loom_module_t* module,
    const loom_op_t* op, uint32_t hash) {
  const iree_host_size_t mask = scope->capacity - 1;
  iree_host_size_t slot = hash & mask;
  while (scope->entries[slot].op) {
    const loom_expression_entry_t* entry = &scope->entries[slot];
    if (entry->hash == hash && loom_expression_equal(module, entry->op, op)) {
      break;
    }
    slot = (slot + 1) & mask;
  }
  return slot;
}

loom_expression_entry_t* loom_expression_scope_find(
    const loom_expression_cursor_t* cursor, uint32_t hash,
    loom_expression_lookup_flags_t flags) {
  for (loom_expression_scope_t* scope = cursor->scope; scope;
       scope = scope->parent) {
    const iree_host_size_t slot =
        loom_expression_scope_slot(scope, cursor->module, cursor->op, hash);
    loom_expression_entry_t* entry = &scope->entries[slot];
    if (entry->op) return entry;
    if (iree_any_bit_set(flags, LOOM_EXPRESSION_LOOKUP_FLAG_STATEFUL) &&
        scope->blocks_stateful_parent_lookup) {
      break;
    }
  }
  return NULL;
}

void loom_expression_scope_insert(const loom_expression_cursor_t* cursor,
                                  loom_expression_entry_t entry) {
  const iree_host_size_t slot = loom_expression_scope_slot(
      cursor->scope, cursor->module, entry.op, entry.hash);
  cursor->scope->entries[slot] = entry;
}

static bool loom_expression_cfg_state_barrier(
    const loom_dominance_region_traversal_t* traversal, uint16_t block_index) {
  const loom_cfg_graph_t* graph = traversal->graph;
  if (graph->malformed) return true;
  const loom_cfg_block_info_t* block = &graph->blocks[block_index];
  if (block_index == 0 && block->predecessor_count == 0) return false;
  if (block->predecessor_count != 1) return true;
  const uint16_t immediate_dominator =
      traversal->immediate_dominators[block_index];
  return immediate_dominator == UINT16_MAX ||
         immediate_dominator == block_index ||
         graph->predecessor_indices[block->predecessor_start] !=
             immediate_dominator;
}

static iree_status_t loom_expression_walk_push_region(
    loom_expression_walk_t* walk, loom_region_t* region,
    loom_expression_scope_t* parent) {
  if (!region || region->block_count == 0) return iree_ok_status();
  loom_expression_scope_t* scopes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      walk->arena, region->block_count, sizeof(*scopes), (void**)&scopes));
  const loom_dominance_region_traversal_t traversal =
      loom_dominance_region_traversal(&walk->dominance, region);
  for (uint16_t i = 0; i < region->block_count; ++i) {
    loom_block_t* block = loom_region_block(region, i);
    loom_expression_scope_t* scope = &scopes[i];
    *scope = (loom_expression_scope_t){
        .capacity = iree_host_size_next_power_of_two(
            iree_max((iree_host_size_t)block->op_count * 2, 16)),
        .parent = parent,
        .next_op = block->first_op,
    };
    if (traversal.graph) {
      const uint16_t immediate_dominator = traversal.immediate_dominators[i];
      if (immediate_dominator != UINT16_MAX && immediate_dominator != i) {
        scope->parent = &scopes[immediate_dominator];
      }
      scope->blocks_stateful_parent_lookup =
          loom_expression_cfg_state_barrier(&traversal, i);
    }
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(walk->arena, scope->capacity,
                                                   sizeof(*scope->entries),
                                                   (void**)&scope->entries));
    memset(scope->entries, 0, scope->capacity * sizeof(*scope->entries));
  }
  for (int32_t i = (int32_t)region->block_count - 1; i >= 0; --i) {
    const uint16_t block_index =
        traversal.graph ? traversal.block_order[i] : (uint16_t)i;
    scopes[block_index].previous_frame = walk->top;
    walk->top = &scopes[block_index];
  }
  return iree_ok_status();
}

iree_status_t loom_expression_walk_initialize(
    loom_module_t* module, loom_region_t* region, iree_arena_allocator_t* arena,
    loom_expression_walk_t** out_walk) {
  *out_walk = NULL;
  loom_expression_walk_t* walk = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, sizeof(*walk), (void**)&walk));
  *walk = (loom_expression_walk_t){.module = module, .arena = arena};
  IREE_RETURN_IF_ERROR(loom_dominance_info_initialize_region(
      module, region, arena, &walk->dominance));
  IREE_RETURN_IF_ERROR(loom_expression_walk_push_region(walk, region, NULL));
  *out_walk = walk;
  return iree_ok_status();
}

iree_status_t loom_expression_walk_next(loom_expression_walk_t* walk,
                                        loom_expression_cursor_t* out_cursor) {
  *out_cursor = (loom_expression_cursor_t){0};
  while (walk->top) {
    loom_expression_scope_t* scope = walk->top;
    loom_op_t* op = scope->next_op;
    if (!op) {
      walk->top = scope->previous_frame;
      continue;
    }
    scope->next_op = op->next_op;
    if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) continue;
    const loom_trait_flags_t traits =
        loom_op_effective_traits(walk->module, op);
    *out_cursor = (loom_expression_cursor_t){
        .module = walk->module,
        .scope = scope,
        .op = op,
        .traits = traits,
        .epoch = walk->epoch,
    };
    walk->epoch += op->result_count != 0;
    for (int32_t r = (int32_t)op->region_count - 1; r >= 0; --r) {
      IREE_RETURN_IF_ERROR(loom_expression_walk_push_region(
          walk, loom_op_regions(op)[r],
          loom_traits_is_isolated(traits) ? NULL : scope));
    }
    return iree_ok_status();
  }
  return iree_ok_status();
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

static bool loom_expression_result_transfers_operand_ownership(
    const loom_module_t* module, const loom_op_t* op, uint16_t result_index,
    uint16_t* out_operand_index) {
  loom_ownership_result_effect_t effect = {0};
  if (!loom_ownership_result_effect_at(module, op, result_index, &effect) ||
      (effect.effect != LOOM_RESULT_OWNERSHIP_TIED &&
       effect.effect != LOOM_RESULT_OWNERSHIP_MOVED)) {
    return false;
  }
  *out_operand_index = effect.source_operand_index;
  return true;
}

static bool loom_expression_op_transfers_operand_ownership(
    const loom_module_t* module, const loom_op_t* op) {
  for (uint16_t i = 0; i < op->result_count; ++i) {
    uint16_t operand_index = 0;
    if (loom_expression_result_transfers_operand_ownership(module, op, i,
                                                           &operand_index)) {
      return true;
    }
  }
  return false;
}

static bool loom_expression_use_consumes_operand(const loom_module_t* module,
                                                 const loom_use_t use) {
  const loom_op_t* user_op = loom_use_user_op(use);
  const uint16_t operand_index = loom_use_operand_index(use);
  for (uint16_t i = 0; i < user_op->result_count; ++i) {
    uint16_t source_operand_index = 0;
    if (loom_expression_result_transfers_operand_ownership(
            module, user_op, i, &source_operand_index) &&
        source_operand_index == operand_index) {
      return true;
    }
  }
  return false;
}

static bool loom_expression_result_is_consumed(const loom_module_t* module,
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
      if (loom_expression_use_consumes_operand(module, *use)) {
        return true;
      }
    }
  }
  return false;
}

bool loom_expression_is_reusable(const loom_expression_cursor_t* cursor) {
  return cursor->op->result_count != 0 && cursor->op->region_count == 0 &&
         !loom_traits_has_side_effects(cursor->traits) &&
         !iree_any_bit_set(cursor->traits, LOOM_TRAIT_UNIQUE_IDENTITY |
                                               LOOM_TRAIT_CONVERGENT) &&
         !loom_expression_op_transfers_operand_ownership(cursor->module,
                                                         cursor->op) &&
         !loom_expression_result_is_consumed(cursor->module, cursor->op);
}

iree_status_t loom_expression_replace(loom_module_t* module, loom_op_t* op,
                                      loom_op_t* existing) {
  loom_value_id_t* results = loom_op_results(op);
  const loom_value_id_t* existing_results = loom_op_results(existing);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < op->result_count && iree_status_is_ok(status); ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID &&
        existing_results[i] != LOOM_VALUE_ID_INVALID) {
      status = loom_value_replace_all_uses_with(module, results[i],
                                                existing_results[i]);
    }
  }
  if (iree_status_is_ok(status)) status = loom_op_erase(module, op);
  return status;
}
