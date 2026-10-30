// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/fact_propagation.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/intern_table.h"
#include "loom/ir/module.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/fact_cfg.h"
#include "loom/util/fact_loop.h"
#include "loom/util/fact_observation.h"

enum loom_rewriter_fact_entry_flag_bits_e {
  // The equation has a pending entry in one inference queue.
  LOOM_REWRITER_FACT_PENDING = 1u << 0,
  // The indexed direct-use closure has not yet published a semantic edit.
  LOOM_REWRITER_FACT_INVALIDATED = 1u << 1,
  // Direct inputs changed: blocks publish semantic invalidation, and structured
  // operations refresh nested facts before joining their results.
  LOOM_REWRITER_FACT_INPUTS_CHANGED = 1u << 2,
  // A completed full summary established this owner's body and scope facts.
  // Missing rows have no witness; structural edits withdraw it.
  LOOM_REWRITER_FACT_BODY_CURRENT = 1u << 3,
};

typedef struct loom_rewriter_fact_entry_t {
  // Arena-stable operation or block identity; the queue determines its kind.
  void* owner;
  // Pending membership owned exclusively by this rewriter.
  uint8_t flags;
} loom_rewriter_fact_entry_t;

typedef struct loom_rewriter_fact_queue_t {
  // Stable membership rows retained in the rewriter's pass arena.
  loom_rewriter_fact_entry_t** values;
  // Number of populated queue slots.
  iree_host_size_t count;
  // Reusable slot capacity retained until the pass arena is released.
  iree_host_size_t capacity;
} loom_rewriter_fact_queue_t;

// Membership is separate from IR and from the optional rewrite-policy queue.
// Existing use indexes own dependencies; these rows only remember pending work.
struct loom_rewriter_pending_facts_t {
  // Pointer-keyed index of stable membership rows.
  loom_intern_table_t index;
  // Fixed-size chunks of membership rows; queues never retain movable slots.
  loom_segmented_storage_t rows;
  // Ordinary equations consumed in publication order.
  loom_rewriter_fact_queue_t operations;
  // First unconsumed ordinary equation.
  iree_host_size_t operation_head;
  // Structured summaries consumed after their ordinary input equations.
  loom_rewriter_fact_queue_t summaries;
  // Semantic edits retained until their complete direct-use closure is queued.
  loom_rewriter_fact_queue_t invalidations;
  // Argument equations consumed before operation equations.
  loom_rewriter_fact_queue_t blocks;
};

#define LOOM_REWRITER_FACT_ROWS_PER_SEGMENT 128u

static loom_rewriter_fact_entry_t* loom_rewriter_fact_entry_at(
    loom_rewriter_pending_facts_t* pending, uint32_t index) {
  loom_rewriter_fact_entry_t* rows = loom_segmented_storage_segment(
      &pending->rows, index / LOOM_REWRITER_FACT_ROWS_PER_SEGMENT);
  return &rows[index % LOOM_REWRITER_FACT_ROWS_PER_SEGMENT];
}

typedef struct loom_rewriter_fact_probe_t {
  // Owner of the canonical row index.
  loom_rewriter_pending_facts_t* pending;
  // Operation or block identity being looked up.
  const void* owner;
} loom_rewriter_fact_probe_t;

static uint32_t loom_rewriter_fact_owner_hash(const void* owner) {
  uintptr_t bits = (uintptr_t)owner >> 4;
  bits ^= bits >> 16;
  return (uint32_t)(bits * UINT32_C(0x9e3779b9));
}

static bool loom_rewriter_fact_owner_equal(const void* context,
                                           uint32_t index) {
  const loom_rewriter_fact_probe_t* probe = context;
  return loom_rewriter_fact_entry_at(probe->pending, index)->owner ==
         probe->owner;
}

static loom_rewriter_fact_entry_t* loom_rewriter_find_fact_entry(
    loom_rewriter_t* rewriter, const void* owner) {
  if (!rewriter->pending_facts) {
    return NULL;
  }
  const loom_rewriter_fact_probe_t context = {rewriter->pending_facts, owner};
  const loom_intern_probe_t probe = loom_intern_table_probe(
      &rewriter->pending_facts->index, loom_rewriter_fact_owner_hash(owner),
      loom_rewriter_fact_owner_equal, &context);
  return probe.index == UINT32_MAX ? NULL
                                   : loom_rewriter_fact_entry_at(
                                         rewriter->pending_facts, probe.index);
}

void loom_rewriter_invalidate_enclosing_fact_bodies(
    loom_rewriter_t* rewriter, loom_op_t* owner) {
  if (!rewriter->fact_table) return;
  for (loom_op_t* current = owner; current; current = current->parent_op) {
    if (!current->region_count) continue;
    loom_rewriter_fact_entry_t* entry =
        loom_rewriter_find_fact_entry(rewriter, current);
    if (entry) entry->flags &= ~LOOM_REWRITER_FACT_BODY_CURRENT;
  }
}

static iree_status_t loom_rewriter_get_fact_entry(
    loom_rewriter_t* rewriter, void* owner,
    loom_rewriter_fact_entry_t** out_entry) {
  loom_rewriter_pending_facts_t* pending = rewriter->pending_facts;
  if (!pending) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate(rewriter->arena, sizeof(*pending),
                                             (void**)&pending));
    memset(pending, 0, sizeof(*pending));
    IREE_RETURN_IF_ERROR(
        loom_intern_table_initialize(rewriter->arena, 0, &pending->index));
    loom_segmented_storage_initialize(LOOM_REWRITER_FACT_ROWS_PER_SEGMENT *
                                          sizeof(loom_rewriter_fact_entry_t),
                                      iree_alignof(loom_rewriter_fact_entry_t),
                                      &pending->rows);
    rewriter->pending_facts = pending;
  }
  const uint32_t hash = loom_rewriter_fact_owner_hash(owner);
  const loom_rewriter_fact_probe_t context = {pending, owner};
  loom_intern_probe_t probe = loom_intern_table_probe(
      &pending->index, hash, loom_rewriter_fact_owner_equal, &context);
  if (probe.index != UINT32_MAX) {
    *out_entry = loom_rewriter_fact_entry_at(pending, probe.index);
    return iree_ok_status();
  }
  if (pending->index.count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "rewrite fact owner count exceeds index capacity");
  }
  IREE_RETURN_IF_ERROR(loom_intern_table_reserve_insert(
      rewriter->arena, &pending->index, hash, 1, &probe.slot));
  const uint32_t index = (uint32_t)pending->index.count;
  if (index / LOOM_REWRITER_FACT_ROWS_PER_SEGMENT ==
      pending->rows.segment_count) {
    void* segment = NULL;
    IREE_RETURN_IF_ERROR(loom_segmented_storage_append(
        &pending->rows, rewriter->arena, &segment));
  }
  loom_rewriter_fact_entry_t* entry =
      loom_rewriter_fact_entry_at(pending, index);
  *entry = (loom_rewriter_fact_entry_t){.owner = owner};
  loom_intern_table_insert(&pending->index, probe.slot, hash, index);
  *out_entry = entry;
  return iree_ok_status();
}

static iree_status_t loom_rewriter_append_fact_entry(
    loom_rewriter_t* rewriter, loom_rewriter_fact_queue_t* queue,
    loom_rewriter_fact_entry_t* entry) {
  if (queue->count == queue->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        rewriter->arena, queue->count, queue->count + 1, sizeof(*queue->values),
        &queue->capacity, (void**)&queue->values));
  }
  queue->values[queue->count++] = entry;
  return iree_ok_status();
}

static void loom_rewriter_clear_fact_invalidations(loom_rewriter_t* rewriter) {
  loom_rewriter_fact_queue_t* queue = &rewriter->pending_facts->invalidations;
  for (iree_host_size_t i = 0; i < queue->count; ++i) {
    queue->values[i]->flags &= ~LOOM_REWRITER_FACT_INVALIDATED;
  }
  queue->count = 0;
}

static void loom_rewriter_clear_pending_facts(loom_rewriter_t* rewriter) {
  loom_rewriter_pending_facts_t* pending = rewriter->pending_facts;
  if (!pending) {
    return;
  }
  loom_intern_table_clear(&pending->index);
  pending->operations.count = 0;
  pending->operation_head = 0;
  pending->summaries.count = 0;
  pending->invalidations.count = 0;
  pending->blocks.count = 0;
}

static iree_status_t loom_rewriter_publish_fact_invalidations(
    loom_rewriter_t* rewriter);

bool loom_rewriter_op_summarizes_nested_regions(
    const loom_op_vtable_t* vtable) {
  return vtable && (vtable->loop_like || vtable->region_branch);
}

static iree_status_t loom_rewriter_add_fact_equation(loom_rewriter_t* rewriter,
                                                     loom_op_t* op,
                                                     bool inputs_changed) {
  IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, op));
  if (!rewriter->fact_table || (op->flags & LOOM_OP_FLAG_DEAD)) {
    return iree_ok_status();
  }
#ifndef NDEBUG
  loom_value_fact_table_invalidate_observations(rewriter->fact_table);
#endif
  loom_rewriter_fact_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_rewriter_get_fact_entry(rewriter, op, &entry));
  if (op->region_count && inputs_changed) {
    entry->flags |= LOOM_REWRITER_FACT_INPUTS_CHANGED;
  }
  if (entry->flags & LOOM_REWRITER_FACT_PENDING) {
    return iree_ok_status();
  }
  const bool summary = loom_rewriter_op_summarizes_nested_regions(
      loom_op_vtable(rewriter->module, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_append_fact_entry(
      rewriter,
      summary ? &rewriter->pending_facts->summaries
              : &rewriter->pending_facts->operations,
      entry));
  entry->flags |= LOOM_REWRITER_FACT_PENDING;
  return iree_ok_status();
}

iree_status_t loom_rewriter_add_fact_user_to_worklist(loom_rewriter_t* rewriter,
                                                      loom_op_t* op) {
  return loom_rewriter_add_fact_equation(rewriter, op, true);
}

static iree_status_t loom_rewriter_add_changed_fact_equation(
    loom_rewriter_t* rewriter, loom_op_t* op,
    loom_rewriter_user_change_flags_t flags, bool inputs_changed) {
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_fact_equation(rewriter, op, inputs_changed));
  if (!rewriter->fact_table || (op->flags & LOOM_OP_FLAG_DEAD) ||
      !iree_any_bit_set(flags, LOOM_REWRITER_USER_CHANGE_FLAG_REPLACED |
                                   LOOM_REWRITER_USER_CHANGE_FLAG_SEMANTIC)) {
    return iree_ok_status();
  }
  loom_rewriter_fact_entry_t* entry =
      loom_rewriter_find_fact_entry(rewriter, op);
  if (!(entry->flags & LOOM_REWRITER_FACT_INVALIDATED)) {
    IREE_RETURN_IF_ERROR(loom_rewriter_append_fact_entry(
        rewriter, &rewriter->pending_facts->invalidations, entry));
    entry->flags |= LOOM_REWRITER_FACT_INVALIDATED;
  }
  return iree_ok_status();
}

iree_status_t loom_rewriter_add_changed_fact_user(
    loom_rewriter_t* rewriter, loom_op_t* op,
    loom_rewriter_user_change_flags_t flags) {
  return loom_rewriter_add_changed_fact_equation(rewriter, op, flags, true);
}

iree_status_t loom_rewriter_add_fact_block(loom_rewriter_t* rewriter,
                                           loom_block_t* block) {
#ifndef NDEBUG
  loom_value_fact_table_invalidate_observations(rewriter->fact_table);
#endif
  loom_rewriter_fact_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(loom_rewriter_get_fact_entry(rewriter, block, &entry));
  if (!(entry->flags & LOOM_REWRITER_FACT_PENDING)) {
    IREE_RETURN_IF_ERROR(loom_rewriter_append_fact_entry(
        rewriter, &rewriter->pending_facts->blocks, entry));
  }
  entry->flags |=
      LOOM_REWRITER_FACT_PENDING | LOOM_REWRITER_FACT_INPUTS_CHANGED;
  return iree_ok_status();
}

iree_status_t loom_rewriter_relocate_fact_block(loom_rewriter_t* rewriter,
                                                loom_block_t* source,
                                                loom_block_t* target) {
  loom_rewriter_fact_entry_t* entry =
      loom_rewriter_find_fact_entry(rewriter, source);
  if (entry && (entry->flags & LOOM_REWRITER_FACT_PENDING)) {
    IREE_RETURN_IF_ERROR(loom_rewriter_add_fact_block(rewriter, target));
    entry->flags = 0;
  }
  return iree_ok_status();
}

static bool loom_rewriter_summary_region_is_ready(
    const loom_op_t* op, const loom_op_vtable_t* vtable, uint8_t region_index) {
  if (region_index >= op->region_count) {
    return false;
  }
  const loom_region_descriptor_t* descriptor =
      loom_op_vtable_region_descriptor(vtable, region_index);
  loom_region_t* region = loom_op_regions(op)[region_index];
  if (!descriptor || !region || region->block_count != 1) {
    return false;
  }
  const loom_block_t* block = loom_region_const_entry_block(region);
  const loom_op_t* terminator = block->last_op;
  if (!terminator) {
    return false;
  }
  return loom_region_descriptor_matches_terminator(descriptor,
                                                   terminator->kind);
}

bool loom_rewriter_nested_region_summary_is_ready(
    const loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_op_vtable_t* vtable) {
  if (!vtable) {
    return true;
  }
  if (vtable->loop_like) {
    if (!loom_rewriter_summary_region_is_ready(
            op, vtable, vtable->loop_like->body_region_index)) {
      return false;
    }
    const uint8_t condition_region_index =
        vtable->loop_like->condition_region_index;
    if (condition_region_index != LOOM_REGION_INDEX_NONE &&
        !loom_rewriter_summary_region_is_ready(op, vtable,
                                               condition_region_index)) {
      return false;
    }
  }
  if (vtable->region_branch) {
    loom_region_branch_t branch = loom_region_branch_cast(rewriter->module, op);
    for (uint8_t region_index = 0; region_index < op->region_count;
         ++region_index) {
      if (!loom_region_branch_region(rewriter->module, branch, region_index)) {
        continue;
      }
      if (!loom_rewriter_summary_region_is_ready(op, vtable, region_index)) {
        return false;
      }
    }
  }
  return true;
}

// Recomputes direct result facts. Callers own result-user and parent-summary
// scheduling because semantic mutations can matter even when direct facts
// remain equal.
iree_status_t loom_rewriter_recompute_op_facts(loom_rewriter_t* rewriter,
                                               loom_op_t* op) {
  if (!rewriter->fact_table) {
    return iree_ok_status();
  }

  bool facts_changed = false;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_op_and_report(
      rewriter->fact_table, rewriter->module, op, &facts_changed));
  if (!facts_changed) {
    return iree_ok_status();
  }

  rewriter->flags |= LOOM_REWRITER_FLAG_FACTS_CHANGED;
  return iree_ok_status();
}

// Region owners publish inherited context before their children are built or
// revisited. Cyclic summaries refresh it again when control facts change.
iree_status_t loom_rewriter_seed_nested_execution_facts(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  if (!rewriter->fact_table || !op->region_count) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_nested_target_scopes(
      rewriter->fact_table, rewriter->module, op));
  const loom_value_facts_t scope = loom_value_fact_table_block_temporal_scope(
      rewriter->fact_table, op->parent_block);
  const bool may_repeat =
      loom_loop_like_isa(loom_loop_like_cast(rewriter->module, op)) ||
      loom_value_fact_table_block_may_repeat(rewriter->fact_table,
                                             op->parent_block);
  const loom_region_branch_t branch =
      loom_region_branch_cast(rewriter->module, op);
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (regions[i]) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_set_region_temporal_scope(
          rewriter->fact_table, regions[i], op, scope, may_repeat));
      IREE_RETURN_IF_ERROR(loom_value_fact_table_set_region_branch_truth(
          rewriter->fact_table, regions[i],
          loom_region_branch_region_truth(branch, i)));
      if (regions[i]->block_count) {
        IREE_RETURN_IF_ERROR(loom_value_fact_table_seed_loop_iv_arg(
            rewriter->fact_table, rewriter->module,
            loom_region_entry_block(regions[i]), op));
      }
    }
  }
  return iree_ok_status();
}

struct loom_rewriter_cfg_region_t {
  // Region represented by the published snapshot.
  const loom_region_t* region;
  // Owned resettable snapshot storage.
  iree_arena_allocator_t arena;
  // Next entry in the address hash bucket.
  loom_rewriter_cfg_region_t* next_bucket;
  // Next entry in the complete list for cleanup.
  loom_rewriter_cfg_region_t* next_entry;
};

static iree_host_size_t loom_rewriter_cfg_region_hash(
    const loom_region_t* region) {
  uintptr_t bits = (uintptr_t)region;
  bits ^= bits >> 17;
  bits *= (uintptr_t)0xed5ad4bbU;
  return bits ^ (bits >> 11);
}

void loom_rewriter_release_cfg_facts(loom_rewriter_t* rewriter) {
  for (loom_rewriter_cfg_region_t* entry = rewriter->cfg_facts.entries; entry;
       entry = entry->next_entry) {
    loom_value_fact_table_forget_cfg_region(rewriter->fact_table,
                                            entry->region);
    iree_arena_deinitialize(&entry->arena);
  }
  memset(&rewriter->cfg_facts, 0, sizeof(rewriter->cfg_facts));
}

static iree_status_t loom_rewriter_cfg_region_storage(
    loom_rewriter_t* rewriter, const loom_region_t* region,
    loom_rewriter_cfg_region_t** out_entry) {
  if (rewriter->cfg_facts.bucket_count) {
    iree_host_size_t bucket = loom_rewriter_cfg_region_hash(region) &
                              (rewriter->cfg_facts.bucket_count - 1);
    for (loom_rewriter_cfg_region_t* entry =
             rewriter->cfg_facts.buckets[bucket];
         entry; entry = entry->next_bucket) {
      if (entry->region == region) {
        *out_entry = entry;
        return iree_ok_status();
      }
    }
  }
  if (rewriter->cfg_facts.count >= rewriter->cfg_facts.bucket_count / 2) {
    iree_host_size_t bucket_count = rewriter->cfg_facts.bucket_count
                                        ? rewriter->cfg_facts.bucket_count * 2
                                        : 8;
    loom_rewriter_cfg_region_t** buckets = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        rewriter->arena, bucket_count, sizeof(*buckets), (void**)&buckets));
    memset(buckets, 0, bucket_count * sizeof(*buckets));
    for (loom_rewriter_cfg_region_t* entry = rewriter->cfg_facts.entries; entry;
         entry = entry->next_entry) {
      iree_host_size_t bucket =
          loom_rewriter_cfg_region_hash(entry->region) & (bucket_count - 1);
      entry->next_bucket = buckets[bucket];
      buckets[bucket] = entry;
    }
    rewriter->cfg_facts.buckets = buckets;
    rewriter->cfg_facts.bucket_count = bucket_count;
  }
  loom_rewriter_cfg_region_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(rewriter->arena, sizeof(*entry), (void**)&entry));
  memset(entry, 0, sizeof(*entry));
  entry->region = region;
  iree_arena_initialize(rewriter->arena->block_pool, &entry->arena);
  iree_host_size_t bucket = loom_rewriter_cfg_region_hash(region) &
                            (rewriter->cfg_facts.bucket_count - 1);
  entry->next_bucket = rewriter->cfg_facts.buckets[bucket];
  rewriter->cfg_facts.buckets[bucket] = entry;
  entry->next_entry = rewriter->cfg_facts.entries;
  rewriter->cfg_facts.entries = entry;
  ++rewriter->cfg_facts.count;
  *out_entry = entry;
  return iree_ok_status();
}

static void loom_rewriter_on_summary_computed(void* user_data,
                                              const loom_op_t* op) {
  loom_rewriter_t* rewriter = user_data;
  loom_rewriter_fact_entry_t* entry =
      loom_rewriter_find_fact_entry(rewriter, op);
  if (entry) {
    entry->flags &=
        ~(LOOM_REWRITER_FACT_PENDING | LOOM_REWRITER_FACT_INPUTS_CHANGED);
  }
}

void loom_rewriter_attach_value_facts(loom_rewriter_t* rewriter,
                                      loom_value_fact_table_t* facts) {
  IREE_ASSERT(!facts || !facts->has_conditioned_results);
  loom_rewriter_clear_pending_facts(rewriter);
  loom_rewriter_release_cfg_facts(rewriter);
  if (rewriter->fact_table) {
    rewriter->fact_table->summary_computed.fn = NULL;
    rewriter->fact_table->summary_computed.user_data = NULL;
  }
  rewriter->fact_table = facts;
  if (facts) {
    facts->summary_computed.fn = loom_rewriter_on_summary_computed;
    facts->summary_computed.user_data = rewriter;
  }
}

static iree_status_t loom_rewriter_cfg_argument_changed(
    void* user_data, loom_value_id_t value_id) {
  loom_rewriter_t* rewriter = user_data;
  rewriter->flags |= LOOM_REWRITER_FLAG_FACTS_CHANGED;
  return loom_rewriter_add_users_to_worklist(rewriter, value_id, /*flags=*/0);
}

static iree_status_t loom_rewriter_publish_cfg_control_changes(
    loom_rewriter_t* rewriter, const loom_value_fact_cfg_region_t* structure) {
  uint16_t block_index = 0;
  while (loom_value_fact_control_take_changed_block(structure->control,
                                                    &block_index)) {
    loom_op_t* terminator = structure->graph.blocks[block_index].block->last_op;
    if (!terminator || !terminator->successor_count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_fact_user_to_worklist(rewriter, terminator));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_summary_ops_to_worklist(rewriter, terminator));
  }
  return iree_ok_status();
}

static iree_status_t loom_rewriter_update_cfg_block_facts(
    loom_rewriter_t* rewriter, const loom_value_fact_cfg_region_t* structure,
    uint16_t block_index) {
  if (structure->control_flow.components.count != 0 &&
      loom_cfg_graph_block_is_reachable(&structure->graph, block_index)) {
    const loom_scc_t* component =
        &structure->control_flow.components
             .values[structure->graph.blocks[block_index].component];
    if (component->is_cycle) {
      iree_arena_allocator_t scratch_arena;
      iree_arena_initialize(rewriter->arena->block_pool, &scratch_arena);
      iree_status_t status = loom_value_fact_table_recompute_cfg_component(
          rewriter->fact_table, rewriter->module, structure, component,
          &scratch_arena, loom_rewriter_cfg_argument_changed, rewriter);
      iree_arena_deinitialize(&scratch_arena);
      if (iree_status_is_ok(status)) {
        status = loom_rewriter_publish_cfg_control_changes(rewriter, structure);
      }
      iree_bitmap_reset(structure->control_flow.dirty,
                        structure->graph.blocks[block_index].component);
      return status;
    }
  }
  if (loom_cfg_graph_block_is_reachable(&structure->graph, block_index)) {
    iree_bitmap_reset(structure->control_flow.dirty,
                      structure->graph.blocks[block_index].component);
  }
  return loom_value_fact_table_update_cfg_block_args(
      rewriter->fact_table, rewriter->module, structure, block_index,
      loom_rewriter_cfg_argument_changed, rewriter);
}

static iree_status_t loom_rewriter_update_block_facts(loom_rewriter_t* rewriter,
                                                      loom_block_t* block) {
  if (!block->parent_region) {
    return iree_ok_status();
  }
  loom_op_t* parent = loom_value_fact_table_region_parent(rewriter->fact_table,
                                                          block->parent_region);
  if (parent && (parent->flags & LOOM_OP_FLAG_DEAD)) {
    return iree_ok_status();
  }
  const loom_value_fact_cfg_region_t* structure =
      loom_value_fact_table_lookup_cfg_region(rewriter->fact_table,
                                              block->parent_region);
  if (structure && block->region_index != 0) {
    const iree_host_size_t component =
        structure->graph.blocks[block->region_index].component;
    if (iree_bitmap_test(structure->control_flow.dirty, component)) {
      return loom_rewriter_update_cfg_block_facts(rewriter, structure,
                                                  block->region_index);
    }
    return iree_ok_status();
  }
  if (parent && loom_rewriter_op_summarizes_nested_regions(
                    loom_op_vtable(rewriter->module, parent))) {
    return loom_rewriter_add_fact_user_to_worklist(rewriter, parent);
  }
  IREE_RETURN_IF_ERROR(loom_value_fact_table_refresh_entry_args(
      rewriter->fact_table, rewriter->module, block, parent));
  for (uint16_t i = 0; i < block->arg_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_add_users_to_worklist(
        rewriter, loom_block_arg_id(block, i), 0));
  }
  return iree_ok_status();
}

// Compare retained edge identities and reachability, not the mutable fields of
// their IR terminators. Operand edits are separately scheduled by the rewriter.
static bool loom_rewriter_cfg_predecessors_equal(
    const loom_value_fact_cfg_region_t* old_structure, uint16_t old_index,
    const loom_value_fact_cfg_region_t* new_structure, uint16_t new_index) {
  const loom_cfg_graph_t* old_graph = &old_structure->graph;
  const loom_cfg_graph_t* new_graph = &new_structure->graph;
  loom_cfg_edge_index_span_t old_edges =
      loom_cfg_graph_predecessor_edges(old_graph, old_index);
  loom_cfg_edge_index_span_t new_edges =
      loom_cfg_graph_predecessor_edges(new_graph, new_index);
  iree_host_size_t old_position = 0;
  iree_host_size_t new_position = 0;
  while (true) {
    while (old_position < old_edges.count &&
           !old_graph
                ->blocks[old_graph->edges[old_edges.values[old_position]]
                             .source_block_index]
                .reachable) {
      ++old_position;
    }
    while (new_position < new_edges.count &&
           !new_graph
                ->blocks[new_graph->edges[new_edges.values[new_position]]
                             .source_block_index]
                .reachable) {
      ++new_position;
    }
    if (old_position == old_edges.count || new_position == new_edges.count) {
      return old_position == old_edges.count && new_position == new_edges.count;
    }
    const loom_cfg_edge_info_t* old_edge =
        &old_graph->edges[old_edges.values[old_position++]];
    const loom_cfg_edge_info_t* new_edge =
        &new_graph->edges[new_edges.values[new_position++]];
    if (old_edge->terminator != new_edge->terminator ||
        old_edge->successor_index != new_edge->successor_index ||
        old_edge->selector_value_id != new_edge->selector_value_id ||
        old_graph->blocks[old_edge->source_block_index].block !=
            new_graph->blocks[new_edge->source_block_index].block ||
        loom_value_fact_control_execution(old_structure->control,
                                          old_edge->source_block_index)
                .flags !=
            loom_value_fact_control_execution(new_structure->control,
                                              new_edge->source_block_index)
                .flags) {
      return false;
    }
  }
}

static iree_status_t loom_rewriter_refresh_cfg_block_facts(
    loom_rewriter_t* rewriter,
    const loom_value_fact_cfg_region_t* old_structure,
    const loom_value_fact_cfg_region_t* structure,
    iree_arena_allocator_t* arena) {
  iree_host_size_t* old_indices = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(arena, structure->graph.block_count,
                                sizeof(*old_indices), (void**)&old_indices));
  memset(old_indices, 0xFF,
         structure->graph.block_count * sizeof(*old_indices));
  if (old_structure) {
    for (iree_host_size_t i = 0; i < old_structure->graph.block_count; ++i) {
      iree_host_size_t new_index = loom_cfg_graph_block_index(
          &structure->graph, old_structure->graph.blocks[i].block);
      if (new_index != IREE_HOST_SIZE_MAX) {
        old_indices[new_index] = i;
      }
    }
  }
  bool* updated_components = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, structure->control_flow.components.count,
      sizeof(*updated_components), (void**)&updated_components));
  memset(
      updated_components, 0,
      structure->control_flow.components.count * sizeof(*updated_components));
  for (uint16_t i = 0; i < structure->graph.block_count; ++i) {
    if (!structure->graph.blocks[i].reachable) {
      continue;
    }
    if (!structure->graph.blocks[i].component_is_cyclic &&
        old_indices[i] != IREE_HOST_SIZE_MAX &&
        old_structure->graph.blocks[old_indices[i]].component_is_cyclic) {
      // Breaking a cycle removes a temporal constraint without necessarily
      // changing any block argument. Refresh direct and nested observers.
      loom_op_t* op = NULL;
      loom_block_for_each_op((loom_block_t*)structure->graph.blocks[i].block,
                             op) {
        IREE_RETURN_IF_ERROR(loom_rewriter_record_subtree(rewriter, op));
      }
    }
    if (structure->control_flow.components.count) {
      iree_host_size_t component_index = structure->graph.blocks[i].component;
      const loom_scc_t* component =
          &structure->control_flow.components.values[component_index];
      if (component->is_cycle) {
        if (updated_components[component_index]) {
          continue;
        }
        bool was_dirty =
            iree_bitmap_test(structure->control_flow.dirty, component_index);
        if (old_indices[i] != IREE_HOST_SIZE_MAX &&
            old_structure->control_flow.components.count &&
            old_structure->graph.blocks[old_indices[i]].reachable) {
          was_dirty |= iree_bitmap_test(
              old_structure->control_flow.dirty,
              old_structure->graph.blocks[old_indices[i]].component);
        }
        if (!was_dirty && old_indices[i] != IREE_HOST_SIZE_MAX &&
            loom_rewriter_cfg_predecessors_equal(old_structure, old_indices[i],
                                                 structure, i)) {
          continue;
        }
        updated_components[component_index] = true;
      }
    }
    IREE_RETURN_IF_ERROR(
        loom_rewriter_update_cfg_block_facts(rewriter, structure, i));
  }
  return loom_rewriter_publish_cfg_control_changes(rewriter, structure);
}

iree_status_t loom_rewriter_refresh_cfg_facts(loom_rewriter_t* rewriter,
                                              loom_region_t* region) {
  if (!rewriter->fact_table) {
    return iree_ok_status();
  }
  loom_rewriter_cfg_region_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_cfg_region_storage(rewriter, region, &storage));
  const loom_value_fact_cfg_region_t* old_structure =
      loom_value_fact_table_lookup_cfg_region(rewriter->fact_table, region);
  iree_arena_allocator_t next_arena;
  iree_arena_initialize(rewriter->arena->block_pool, &next_arena);
  loom_value_fact_cfg_region_t* structure = NULL;
  iree_status_t status =
      iree_arena_allocate(&next_arena, sizeof(*structure), (void**)&structure);
  if (iree_status_is_ok(status)) {
    status = loom_value_fact_cfg_region_initialize(rewriter->module, region,
                                                   &next_arena, structure);
  }
  if (iree_status_is_ok(status)) {
    status = loom_value_fact_table_set_cfg_region(
        rewriter->fact_table, rewriter->module, region, structure);
  }
  if (iree_status_is_ok(status)) {
    iree_arena_checkpoint_t checkpoint =
        iree_arena_checkpoint_save(&next_arena);
    // The fresh snapshot starts with optimistic control distributions. Its
    // initial selector seed is consumed by the old/new predecessor comparison
    // below, not an incremental change to the retained facts. Only changes
    // during the refresh need subsequent worklist publication.
    uint16_t initialized_block = 0;
    while (loom_value_fact_control_take_changed_block(structure->control,
                                                      &initialized_block)) {
    }
    // Publish semantic edits against the replacement graph before solving
    // it. Pending work for other owners remains queued for observation.
    status = loom_rewriter_publish_fact_invalidations(rewriter);
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_refresh_cfg_block_facts(rewriter, old_structure,
                                                     structure, &next_arena);
    }
    iree_arena_checkpoint_restore(&checkpoint);
    iree_arena_deinitialize(&storage->arena);
    storage->arena = next_arena;
  } else {
    iree_arena_deinitialize(&next_arena);
  }
  return status;
}

iree_status_t loom_rewriter_update_successor_facts(loom_rewriter_t* rewriter,
                                                   loom_op_t* op) {
  if (!op->successor_count) {
    return iree_ok_status();
  }
  const loom_value_fact_cfg_region_t* structure = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_get_or_build_cfg_region(
      rewriter->fact_table, rewriter->module, op->parent_block->parent_region,
      &structure));
  loom_value_fact_cfg_update_control(rewriter->fact_table, structure,
                                     op->parent_block->region_index);
  IREE_RETURN_IF_ERROR(
      loom_rewriter_publish_cfg_control_changes(rewriter, structure));
  const uint16_t block_index = op->parent_block->region_index;
  if (structure->graph.blocks[block_index].component_is_cyclic) {
    const iree_host_size_t component_index =
        structure->graph.blocks[block_index].component;
    if (structure->control_flow.anchors[component_index] == op &&
        iree_bitmap_test(structure->control_flow.dirty, component_index)) {
      IREE_RETURN_IF_ERROR(loom_rewriter_update_cfg_block_facts(
          rewriter, structure, block_index));
    }
  }
  if (op->successor_count != 1) {
    return iree_ok_status();
  }
  loom_block_t* successor = loom_op_successors(op)[0];
  if (!successor->arg_count) {
    return iree_ok_status();
  }
  const loom_cfg_block_info_t* successor_info =
      &structure->graph.blocks[successor->region_index];
  if (!successor_info->reachable ||
      !iree_bitmap_test(structure->control_flow.dirty,
                        successor_info->component)) {
    return iree_ok_status();
  }
  return loom_rewriter_update_cfg_block_facts(rewriter, structure,
                                              successor->region_index);
}

iree_status_t loom_rewriter_enable_region_analysis(
    loom_rewriter_t* rewriter, loom_func_like_t function, loom_region_t* region,
    loom_op_t* parent_op, loom_value_fact_table_t* facts) {
  loom_rewriter_attach_value_facts(rewriter, facts);
  return loom_value_fact_table_compute_region(
      rewriter->fact_table, rewriter->module, function, region, parent_op);
}

iree_status_t loom_rewriter_enable_analysis(loom_rewriter_t* rewriter,
                                            loom_func_like_t function,
                                            loom_value_fact_table_t* facts) {
  return loom_rewriter_enable_region_analysis(
      rewriter, function, loom_func_like_body(function), function.op, facts);
}

static iree_status_t loom_rewriter_add_cfg_summary_to_worklist(
    loom_rewriter_t* rewriter, const loom_block_t* block,
    bool arguments_changed) {
  if (!rewriter->fact_table || !block || !block->parent_region) {
    return iree_ok_status();
  }
  const loom_value_fact_cfg_region_t* structure =
      loom_value_fact_table_lookup_cfg_region(rewriter->fact_table,
                                              block->parent_region);
  if (!structure) {
    return iree_ok_status();
  }
  iree_host_size_t block_index =
      loom_cfg_graph_block_index(&structure->graph, block);
  if (block_index == IREE_HOST_SIZE_MAX ||
      !structure->graph.blocks[block_index].reachable) {
    return iree_ok_status();
  }
  iree_host_size_t component_index =
      structure->graph.blocks[block_index].component;
  if (!structure->graph.blocks[block_index].component_is_cyclic) {
    // Incoming terminators are already queued by the mutation or control
    // publication. Their first visit rejoins all inputs; the rest reuse it.
    if (arguments_changed && block->arg_count) {
      iree_bitmap_set(structure->control_flow.dirty, component_index);
    }
    return iree_ok_status();
  }
  loom_op_t* anchor = structure->control_flow.anchors[component_index];
  if (anchor) {
    iree_bitmap_set(structure->control_flow.dirty, component_index);
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_fact_user_to_worklist(rewriter, anchor));
  }
  return iree_ok_status();
}

// Branch operand identities define the forwarding graph independently of the
// facts carried by those operands. Structural CFG edits publish a new snapshot;
// payload edits only invalidate this successor's retained partition.
void loom_rewriter_invalidate_cfg_forwarding(loom_rewriter_t* rewriter,
                                             const loom_op_t* op) {
  if (!rewriter->fact_table || op->successor_count != 1) {
    return;
  }
  const loom_block_t* successor = loom_op_successors(op)[0];
  if (!successor->arg_count) {
    return;
  }
  const loom_value_fact_cfg_region_t* structure =
      loom_value_fact_table_lookup_cfg_region(rewriter->fact_table,
                                              successor->parent_region);
  if (!structure || !structure->control_flow.components.count) {
    return;
  }
  iree_host_size_t block_index =
      loom_cfg_graph_block_index(&structure->graph, successor);
  if (block_index == IREE_HOST_SIZE_MAX ||
      !structure->graph.blocks[block_index].component_is_cyclic) {
    return;
  }
  structure->control_flow
      .forwarding[structure->graph.blocks[block_index].component]
      .dirty = true;
}

// Body edits can enable a parent rewrite without changing its value equations.
// A terminator changes its owning summary. Further fact dependencies flow
// through that summary's indexed result users, not lexical containment.
iree_status_t loom_rewriter_add_parent_summary_ops_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  if (!rewriter->worklist_chunk && !rewriter->fact_table) {
    return iree_ok_status();
  }
  bool changes_summary =
      op && iree_any_bit_set(op->traits, LOOM_TRAIT_TERMINATOR);
  bool has_cfg_facts =
      rewriter->fact_table && rewriter->fact_table->regions.cfg_count;
  for (loom_op_t* parent = op ? op->parent_op : NULL; parent;
       parent = parent->parent_op) {
    if (changes_summary && has_cfg_facts) {
      IREE_RETURN_IF_ERROR(loom_rewriter_add_cfg_summary_to_worklist(
          rewriter, parent->parent_block, /*arguments_changed=*/false));
    }
    const loom_op_vtable_t* vtable = loom_op_vtable(rewriter->module, parent);
    if (loom_rewriter_op_summarizes_nested_regions(vtable)) {
      const loom_loop_like_t loop =
          loom_loop_like_cast(rewriter->module, parent);
      // A counted body without carried state has no feedback/result summary.
      // Direct range or execution-scope updates still refresh its whole body.
      const bool independent_counted_body =
          loom_loop_like_isa(loop) && loom_loop_like_has_counted_range(loop) &&
          !loom_loop_like_condition_region(loop) &&
          loom_loop_like_iter_args(loop).count == 0;
      if (!independent_counted_body) {
        // Resultless branches have no outward value equation. Direct selector
        // edits still queue the owner and refresh its nested execution facts.
        if (changes_summary &&
            (!vtable->region_branch || parent->result_count)) {
          // Close semantic dependencies before solving, including feedback
          // owners whose current numeric facts stay equal. A changed yield does
          // not change its branch owner's inputs or nested execution context.
          IREE_RETURN_IF_ERROR(loom_rewriter_add_changed_fact_equation(
              rewriter, parent, LOOM_REWRITER_USER_CHANGE_FLAG_SEMANTIC,
              /*inputs_changed=*/false));
        } else {
          IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, parent));
        }
      }
      // The nearest owner consumes the terminator's equation. Ancestors still
      // receive rewrite-policy notifications for structural patterns.
      changes_summary = false;
    }
  }
  return iree_ok_status();
}

iree_status_t loom_rewriter_add_summary_ops_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  if (op && rewriter->fact_table && rewriter->fact_table->regions.cfg_count) {
    // Terminators own local forwarding and control equations. Ordinary
    // operations publish through their value users and enclosing summaries.
    if (iree_any_bit_set(op->traits, LOOM_TRAIT_TERMINATOR)) {
      IREE_RETURN_IF_ERROR(loom_rewriter_add_cfg_summary_to_worklist(
          rewriter, op->parent_block, /*arguments_changed=*/false));
    }
    for (uint8_t i = 0; i < op->successor_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_rewriter_add_cfg_summary_to_worklist(
          rewriter, loom_op_successors(op)[i], /*arguments_changed=*/true));
    }
  }
  return loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, op);
}

// Attribute membership includes TYPE and predicate dependencies even when the
// provider is absent from the owner's ordinary operands.
static iree_status_t loom_rewriter_add_attribute_users_to_worklist(
    loom_rewriter_t* rewriter, loom_value_id_t value_id,
    loom_rewriter_user_change_flags_t flags) {
  loom_type_use_iterator_t users;
  loom_attribute_users_begin(&rewriter->module->type_uses, value_id, &users);
  for (loom_attribute_user_t user = loom_attribute_users_next(&users); user.op;
       user = loom_attribute_users_next(&users)) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_changed_fact_user(rewriter, user.op, flags));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_summary_ops_to_worklist(rewriter, user.op));
  }
  return iree_ok_status();
}

// Adds operand, attribute, and value-type users to the worklist.
iree_status_t loom_rewriter_add_users_to_worklist(
    loom_rewriter_t* rewriter, loom_value_id_t value_id,
    loom_rewriter_user_change_flags_t flags) {
  if (!rewriter->worklist_chunk && !rewriter->fact_table) {
    return iree_ok_status();
  }
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    loom_op_t* user_op = loom_use_user_op(uses[i]);
    if (iree_any_bit_set(flags, LOOM_REWRITER_USER_CHANGE_FLAG_REPLACED)) {
      loom_rewriter_invalidate_cfg_forwarding(rewriter, user_op);
    }
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_changed_fact_user(rewriter, user_op, flags));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_summary_ops_to_worklist(rewriter, user_op));
  }
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_attribute_users_to_worklist(rewriter, value_id, flags));
  loom_type_use_iterator_t type_users;
  loom_module_value_type_users(rewriter->module, value_id, &type_users);
  for (loom_value_id_t user_value_id = loom_type_users_next(&type_users);
       user_value_id != LOOM_VALUE_ID_INVALID;
       user_value_id = loom_type_users_next(&type_users)) {
    loom_value_t* user_value =
        loom_module_value(rewriter->module, user_value_id);
    if (!loom_value_is_block_arg(user_value)) {
      loom_op_t* def = loom_value_def_op(user_value);
      if (def) {
        IREE_RETURN_IF_ERROR(
            loom_rewriter_add_changed_fact_user(rewriter, def, flags));
        IREE_RETURN_IF_ERROR(
            loom_rewriter_add_summary_ops_to_worklist(rewriter, def));
      }
    }
    const loom_use_t* user_value_uses = loom_value_uses(user_value);
    for (uint32_t i = 0; i < user_value->use_count; ++i) {
      loom_op_t* user_op = loom_use_user_op(user_value_uses[i]);
      IREE_RETURN_IF_ERROR(
          loom_rewriter_add_changed_fact_user(rewriter, user_op, flags));
      IREE_RETURN_IF_ERROR(
          loom_rewriter_add_summary_ops_to_worklist(rewriter, user_op));
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_add_attribute_users_to_worklist(
        rewriter, user_value_id, flags));
  }
  return iree_ok_status();
}

// Adds users of all results of |op| to the worklist.
iree_status_t loom_rewriter_add_result_users_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op,
    loom_rewriter_user_change_flags_t flags) {
  loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(
          loom_rewriter_add_users_to_worklist(rewriter, results[i], flags));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_rewriter_update_op_facts(loom_rewriter_t* rewriter,
                                                   loom_op_t* op,
                                                   bool inputs_changed) {
  IREE_RETURN_IF_ERROR(loom_rewriter_update_successor_facts(rewriter, op));
  // Constant-like ops are already the canonical representation of their
  // compile-time value. Their source contract forbids operands and regions,
  // while semantic mutation refreshes their facts at the mutation boundary.
  // Recomputing here can neither expose downstream facts nor fold the op to a
  // different representation.
  if (iree_any_bit_set(op->traits, LOOM_TRAIT_CONSTANT_LIKE)) {
    return iree_ok_status();
  }
  // A changed yield only updates its owner's result join. Direct inputs can
  // change nested execution facts; loops always retain their feedback solve.
  loom_rewriter_fact_entry_t* entry =
      loom_rewriter_find_fact_entry(rewriter, op);
  const bool has_body_witness =
      entry && (entry->flags & LOOM_REWRITER_FACT_BODY_CURRENT) != 0;
  const bool owns_regions = op->region_count != 0;
  const loom_loop_like_t loop = loom_loop_like_cast(rewriter->module, op);
  const bool counted_loop =
      loom_loop_like_isa(loop) && loom_loop_like_has_counted_range(loop);
  const bool needs_full_body =
      inputs_changed || (counted_loop && !has_body_witness);
  const loom_op_t* previous_sparse_first_owner =
      rewriter->fact_table->sparse_first_owner;
  rewriter->fact_table->sparse_first_owner =
      counted_loop && !needs_full_body ? op : NULL;
  bool facts_changed = false;
  iree_status_t status = needs_full_body
                             ? loom_value_fact_table_compute_op_and_report(
                                   rewriter->fact_table, rewriter->module, op,
                                   &facts_changed)
                             : loom_value_fact_table_compute_op_summary(
                                   rewriter->fact_table, rewriter->module, op,
                                   &facts_changed);
  rewriter->fact_table->sparse_first_owner =
      previous_sparse_first_owner;
  IREE_RETURN_IF_ERROR(status);
  const bool established_body_witness = needs_full_body && owns_regions && entry;
  if (established_body_witness) {
    entry->flags |= LOOM_REWRITER_FACT_BODY_CURRENT;
  }
  if (facts_changed) {
    rewriter->flags |= LOOM_REWRITER_FLAG_FACTS_CHANGED;
    // Ensure downstream ops are revisited with updated facts. Type-use edges
    // can point back to another result of the same op, so only reschedule when
    // facts actually changed; otherwise dynamic shaped multi-result ops can
    // keep themselves on the worklist forever.
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_result_users_to_worklist(rewriter, op, /*flags=*/0));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_summary_ops_to_worklist(rewriter, op));
  }

  return iree_ok_status();
}

// Publish semantic edits before evaluating equations: an unchanged current
// row does not prove that downstream cyclic equations retain the same meaning.
// Membership remains set through the closure because type/attribute edges can
// cycle independently of operand SSA.
static iree_status_t loom_rewriter_publish_fact_invalidations(
    loom_rewriter_t* rewriter) {
  loom_rewriter_pending_facts_t* pending = rewriter->pending_facts;
  if (!pending) {
    return iree_ok_status();
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < pending->blocks.count && iree_status_is_ok(status); ++i) {
    loom_rewriter_fact_entry_t* entry = pending->blocks.values[i];
    loom_block_t* block = entry->owner;
    if (!(entry->flags & LOOM_REWRITER_FACT_INPUTS_CHANGED) ||
        !block->parent_region) {
      continue;
    }
    loom_op_t* parent = loom_value_fact_table_region_parent(
        rewriter->fact_table, block->parent_region);
    if (parent && (parent->flags & LOOM_OP_FLAG_DEAD)) {
      continue;
    }
    for (uint16_t j = 0; j < block->arg_count && iree_status_is_ok(status);
         ++j) {
      status = loom_rewriter_add_users_to_worklist(
          rewriter, loom_block_arg_id(block, j),
          LOOM_REWRITER_USER_CHANGE_FLAG_SEMANTIC);
    }
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_add_cfg_summary_to_worklist(rewriter, block, true);
    }
    if (iree_status_is_ok(status)) {
      entry->flags &= ~LOOM_REWRITER_FACT_INPUTS_CHANGED;
    }
  }
  for (iree_host_size_t i = 0;
       i < pending->invalidations.count && iree_status_is_ok(status); ++i) {
    loom_op_t* op = pending->invalidations.values[i]->owner;
    if (op->flags & LOOM_OP_FLAG_DEAD) {
      continue;
    }
    status = loom_rewriter_add_result_users_to_worklist(
        rewriter, op, LOOM_REWRITER_USER_CHANGE_FLAG_SEMANTIC);
    if (iree_status_is_ok(status)) {
      status = loom_rewriter_add_summary_ops_to_worklist(rewriter, op);
    }
  }
  loom_rewriter_clear_fact_invalidations(rewriter);
  return status;
}

// The summary queue is LIFO. A changed outer yield can therefore reach its
// loop before pending branch joins inside the body. Publish the deepest live
// child first so every parent consumes maintained descendant results.
static iree_host_size_t loom_rewriter_deepest_pending_descendant(
    const loom_rewriter_pending_facts_t* pending, const loom_op_t* owner) {
  iree_host_size_t selected = IREE_HOST_SIZE_MAX;
  uint32_t selected_depth = 0;
  for (iree_host_size_t i = 0; i < pending->summaries.count; ++i) {
    const loom_rewriter_fact_entry_t* entry = pending->summaries.values[i];
    const bool queued = (entry->flags & LOOM_REWRITER_FACT_PENDING) != 0;
    const loom_op_t* child = entry->owner;
    const bool live = (child->flags & LOOM_OP_FLAG_DEAD) == 0;
    const bool eligible_child = queued && live;
    if (!eligible_child) continue;
    uint32_t depth = 0;
    for (const loom_op_t* ancestor = child->parent_op; ancestor;
         ancestor = ancestor->parent_op) {
      ++depth;
      if (ancestor != owner) continue;
      if (depth > selected_depth) {
        selected = i;
        selected_depth = depth;
      }
      break;
    }
  }
  return selected;
}

iree_status_t loom_rewriter_propagate_value_facts(loom_rewriter_t* rewriter) {
  loom_rewriter_pending_facts_t* pending = rewriter->pending_facts;
  if (!pending) {
    return iree_ok_status();
  }
  iree_status_t status = loom_rewriter_publish_fact_invalidations(rewriter);
  while (iree_status_is_ok(status) &&
         (pending->operation_head < pending->operations.count ||
          pending->summaries.count || pending->blocks.count)) {
    if (pending->invalidations.count) {
      // Publishing changed equations can expose another summary dependency.
      // Finish its semantic closure before evaluating the next queued owner.
      status = loom_rewriter_publish_fact_invalidations(rewriter);
      if (!iree_status_is_ok(status)) {
        break;
      }
    }
    if (pending->blocks.count) {
      loom_rewriter_fact_entry_t* entry =
          pending->blocks.values[--pending->blocks.count];
      const bool queued = (entry->flags & LOOM_REWRITER_FACT_PENDING) != 0;
      entry->flags &=
          ~(LOOM_REWRITER_FACT_PENDING | LOOM_REWRITER_FACT_INPUTS_CHANGED);
      if (queued) {
        status = loom_rewriter_update_block_facts(rewriter, entry->owner);
      }
      continue;
    }
    loom_rewriter_fact_entry_t* entry = NULL;
    bool summary_entry = false;
    if (pending->operation_head < pending->operations.count) {
      entry = pending->operations.values[pending->operation_head++];
    } else {
      pending->operations.count = 0;
      pending->operation_head = 0;
      entry = pending->summaries.values[--pending->summaries.count];
      summary_entry = true;
    }
    const bool queued = (entry->flags & LOOM_REWRITER_FACT_PENDING) != 0;
    const bool inputs_changed =
        (entry->flags & LOOM_REWRITER_FACT_INPUTS_CHANGED) != 0;
    loom_op_t* op = entry->owner;
    const bool live = (op->flags & LOOM_OP_FLAG_DEAD) == 0;
    const bool has_body_witness =
        (entry->flags & LOOM_REWRITER_FACT_BODY_CURRENT) != 0;
    const loom_loop_like_t loop =
        summary_entry && live
            ? loom_loop_like_cast(rewriter->module, op)
            : (loom_loop_like_t){0};
    const bool counted_loop =
        loom_loop_like_isa(loop) && loom_loop_like_has_counted_range(loop);
    const bool can_publish_children = summary_entry && queued && live &&
                                      !inputs_changed && has_body_witness &&
                                      counted_loop;
    if (can_publish_children) {
      const iree_host_size_t child_index =
          loom_rewriter_deepest_pending_descendant(pending, op);
      if (child_index != IREE_HOST_SIZE_MAX) {
        // Reuse the slot just popped from this queue. The child retains its
        // direct-input reason and ordinary invalidations still drain first.
        const iree_host_size_t last = pending->summaries.count++;
        pending->summaries.values[last] =
            pending->summaries.values[child_index];
        pending->summaries.values[child_index] = entry;
        continue;
      }
    }
    entry->flags &=
        ~(LOOM_REWRITER_FACT_PENDING | LOOM_REWRITER_FACT_INPUTS_CHANGED);
    const bool should_update = live && queued;
    if (should_update) {
      status = loom_rewriter_update_op_facts(rewriter, op, inputs_changed);
    }
  }
  if (iree_status_is_ok(status)) {
    pending->operations.count = 0;
    pending->operation_head = 0;
#ifndef NDEBUG
    loom_value_fact_table_publish_observations(rewriter->fact_table);
#endif
  }
  return status;
}
