// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/util/fact_loop_replay.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_snapshot.h"
#include "loom/util/walk.h"

typedef struct loom_value_fact_loop_input_t {
  // External value read by the solve or one of its nested solves.
  loom_value_id_t value;
  // Complete numeric publication; zero divisor preserves an absent row.
  loom_value_facts_t facts;
  // Immutable auxiliary publications borrowed from the populated fact scope.
  loom_value_fact_auxiliary_snapshot_t auxiliary;
} loom_value_fact_loop_input_t;

struct loom_value_fact_loop_reads_t {
  // Immutable ownership index shared by every nested solve.
  const loom_value_fact_loop_replay_index_t* index;
  // Loop using this depth's buffers, or NULL while the buffers are idle.
  const loom_value_fact_loop_replay_record_t* record;
  // Value-ID membership bits, allocated before entering an infallible query.
  uint64_t* bits;
  // Nonzero words, so clearing and publication visit only observed inputs.
  uint32_t* words;
  // Number of nonzero words in words.
  iree_host_size_t word_count;
  // Reusable buffers at the next nesting depth, shared by sibling loops.
  loom_value_fact_loop_reads_t* child;
};

struct loom_value_fact_loop_replay_record_t {
  // Body bound to this record through the table's existing region index.
  const loom_region_t* body;
  // Lexically enclosing loop, used only while constructing ownership.
  loom_value_fact_loop_replay_record_t* parent;
  // Next root-owned record, including partially constructed records on failure.
  loom_value_fact_loop_replay_record_t* next;
  // One-based preorder owner ordinal; zero denotes an external definition.
  uint32_t ordinal;
  // Exclusive end of this loop's descendant-owner interval.
  uint32_t subtree_end;
  // Structural region depth used only during the single ownership walk.
  uint32_t depth;
  // Complete solves published after this root's withdrawal serial.
  uint64_t withdrawal;
  // Target environment used by the completed solve, or NULL when generic.
  const loom_target_facts_t* target;
  // Execution distribution inherited by the completed solve.
  loom_value_fact_flags_t temporal;
  // Whether inputs and the complete body/result publication are established.
  bool completed;
  // Inputs accumulated across the immutable solve, including indirect reads.
  loom_value_fact_loop_input_t* inputs;
  // Number of established input snapshots.
  iree_host_size_t input_count;
  // Allocated input slots; growth is bounded by this root's arena lifetime.
  iree_host_size_t input_capacity;
  // Observation scratch shared with siblings at this depth.
  loom_value_fact_loop_reads_t* reads;
};

struct loom_value_fact_loop_replay_index_t {
  // Arena owned by the outer solve, never the persistent fact table arena.
  iree_arena_allocator_t* arena;
  // Table whose region entries borrow records until the root returns.
  loom_value_fact_table_t* table;
  // Module whose immutable definitions establish ownership.
  const loom_module_t* module;
  // Nearest loop owner for each definition in the root's value range.
  uint32_t* owners;
  // Root definitions only; values outside the root cannot change in this solve.
  loom_value_fact_range_t range;
  // Next one-based loop ordinal assigned by the ownership walk.
  uint32_t next_ordinal;
  // Active ownership-walk ancestor, consumed only during index construction.
  loom_value_fact_loop_replay_record_t* walking;
  // All allocated records, used to remove borrowed bindings before release.
  loom_value_fact_loop_replay_record_t* records;
  // Monotonic serial for resets that can erase a completed nested publication.
  uint64_t withdrawal;
};

static iree_status_t loom_value_fact_loop_index_define(
    loom_value_fact_loop_replay_index_t* index, loom_value_id_t value,
    uint32_t owner) {
  IREE_RETURN_IF_ERROR(
      loom_value_fact_range_grow(index->arena, value, sizeof(*index->owners),
                                 NULL, &index->range, (void**)&index->owners));
  index->owners[loom_value_fact_range_offset(index->range, value)] = owner;
  return iree_ok_status();
}

// The ownership walk assigns each definition once. A subtree is a contiguous
// interval of loop ordinals even when its SSA IDs are sparse or interleaved.
static iree_status_t loom_value_fact_loop_index_op(
    loom_value_fact_loop_replay_index_t* index, loom_op_t* op, uint32_t depth) {
  while (index->walking && index->walking->depth >= depth) {
    index->walking->subtree_end = index->next_ordinal;
    index->walking = index->walking->parent;
  }
  const loom_loop_like_t loop = loom_loop_like_cast(index->module, op);
  if (loom_loop_like_isa(loop) && (loom_loop_like_condition_region(loop) ||
                                   loom_loop_like_has_counted_range(loop))) {
    loom_value_fact_loop_replay_record_t* record = NULL;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate(index->arena, sizeof(*record), (void**)&record));
    *record = (loom_value_fact_loop_replay_record_t){
        .body = loom_loop_like_body(loop),
        .parent = index->walking,
        .next = index->records,
        .ordinal = index->next_ordinal++,
        .depth = depth,
    };
    index->records = record;
    loom_value_fact_loop_reads_t** reads =
        record->parent ? &record->parent->reads->child : &record->reads;
    if (!*reads) {
      IREE_RETURN_IF_ERROR(
          iree_arena_allocate(index->arena, sizeof(**reads), (void**)reads));
      **reads = (loom_value_fact_loop_reads_t){.index = index};
    }
    record->reads = *reads;
    IREE_RETURN_IF_ERROR(loom_value_fact_table_set_loop_replay(
        index->table, record->body, record));
    index->walking = record;
  }
  const uint32_t owner = index->walking->ordinal;
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_loop_index_define(index, results[i], owner));
  }
  loom_region_t* const* regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (!regions[i]) {
      continue;
    }
    const loom_block_t* block = NULL;
    loom_region_for_each_block(regions[i], block) {
      for (uint16_t j = 0; j < block->arg_count; ++j) {
        IREE_RETURN_IF_ERROR(loom_value_fact_loop_index_define(
            index, loom_block_arg_id(block, j), owner));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_value_fact_loop_index_visit(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  *out_result = LOOM_WALK_CONTINUE;
  return loom_value_fact_loop_index_op(
      (loom_value_fact_loop_replay_index_t*)user_data, op,
      (uint32_t)context->depth + 1);
}

static iree_status_t loom_value_fact_loop_index_initialize(
    loom_value_fact_table_t* table, const loom_module_t* module,
    loom_value_fact_loop_replay_t* root) {
  IREE_RETURN_IF_ERROR(iree_arena_allocate(root->arena, sizeof(*root->index),
                                           (void**)&root->index));
  loom_value_fact_loop_replay_index_t* index = root->index;
  *index = (loom_value_fact_loop_replay_index_t){
      .arena = root->arena,
      .table = table,
      .module = module,
      .next_ordinal = 1,
  };
  IREE_RETURN_IF_ERROR(loom_value_fact_loop_index_op(index, root->op, 0));
  loom_region_t* const* regions = loom_op_regions(root->op);
  iree_status_t status = iree_ok_status();
  for (uint8_t i = 0; i < root->op->region_count && iree_status_is_ok(status);
       ++i) {
    if (!regions[i]) {
      continue;
    }
    loom_walk_result_t result;
    status = loom_walk_region(
        module, regions[i], LOOM_WALK_PRE_ORDER,
        (loom_walk_callback_t){loom_value_fact_loop_index_visit, index},
        &result);
  }
  for (loom_value_fact_loop_replay_record_t* record = index->walking; record;
       record = record->parent) {
    record->subtree_end = index->next_ordinal;
  }
  index->walking = NULL;
  return status;
}

void loom_value_fact_loop_record_read(loom_value_fact_loop_reads_t* reads,
                                      loom_value_id_t value_id) {
  if (!loom_value_fact_range_contains(reads->index->range, value_id)) {
    return;
  }
  const iree_host_size_t offset =
      loom_value_fact_range_offset(reads->index->range, value_id);
  const uint32_t owner = reads->index->owners[offset];
  if (!owner ||
      (owner >= reads->record->ordinal && owner < reads->record->subtree_end)) {
    return;
  }
  const uint32_t word = offset / 64;
  const uint64_t bit = UINT64_C(1) << (offset % 64);
  if (!reads->bits[word]) {
    reads->words[reads->word_count++] = word;
  }
  reads->bits[word] |= bit;
}

static void loom_value_fact_loop_reads_reset(
    loom_value_fact_loop_reads_t* reads) {
  for (iree_host_size_t i = 0; i < reads->word_count; ++i) {
    reads->bits[reads->words[i]] = 0;
  }
  reads->word_count = 0;
  reads->record = NULL;
}

static void loom_value_fact_loop_input_snapshot(
    const loom_value_fact_table_t* table, loom_value_id_t value,
    loom_value_fact_loop_input_t* input) {
  *input = (loom_value_fact_loop_input_t){.value = value};
  (void)loom_value_fact_table_try_lookup(table, value, &input->facts);
  loom_value_fact_table_snapshot_auxiliary(table, value, &input->auxiliary);
}

static bool loom_value_fact_loop_inputs_equal(
    const loom_value_fact_table_t* table,
    const loom_value_fact_loop_replay_record_t* record, loom_op_t* op) {
  if (!table->replaying || !record->completed ||
      record->withdrawal != record->reads->index->withdrawal ||
      record->target != table->context.target_facts ||
      record->temporal !=
          loom_value_fact_table_block_temporal_scope(table, op->parent_block)
              .flags) {
    return false;
  }
  for (iree_host_size_t i = 0; i < record->input_count; ++i) {
    loom_value_fact_loop_input_t current;
    loom_value_fact_loop_input_snapshot(table, record->inputs[i].value,
                                        &current);
    if (!loom_value_facts_equal(record->inputs[i].facts, current.facts) ||
        !loom_value_fact_auxiliary_snapshot_equal(&record->inputs[i].auxiliary,
                                                  &current.auxiliary)) {
      return false;
    }
  }
  return true;
}

static void loom_value_fact_loop_forward_reads(
    loom_value_fact_loop_reads_t* reads,
    const loom_value_fact_loop_replay_record_t* record) {
  if (!reads) {
    return;
  }
  for (iree_host_size_t i = 0; i < record->input_count; ++i) {
    loom_value_fact_loop_record_read(reads, record->inputs[i].value);
  }
}

iree_status_t loom_value_fact_loop_replay_begin(
    loom_value_fact_table_t* table, const loom_module_t* module, loom_op_t* op,
    iree_arena_allocator_t* arena, loom_value_fact_loop_replay_t* replay,
    bool* out_reused) {
  *out_reused = false;
  *replay = (loom_value_fact_loop_replay_t){
      .parent = table->replay.loop,
      .op = op,
      .arena = arena,
      .parent_reads = table->replay.reads,
  };
  replay->root = replay->parent ? replay->parent->root : replay;
  // Operand refinements can consume cached constraints whose table reads
  // occurred before this nested solve. Their owning analysis must recompute
  // the body; observations here cannot establish that scope's dependencies.
  if (!replay->parent || table->context.refine_operands.fn) {
    table->replay.loop = replay;
    return iree_ok_status();
  }
  if (!replay->root->index) {
    IREE_RETURN_IF_ERROR(
        loom_value_fact_loop_index_initialize(table, module, replay->root));
  }
  loom_value_fact_loop_replay_record_t* record =
      loom_value_fact_table_lookup_loop_replay(
          table, loom_loop_like_body(loom_loop_like_cast(module, op)));
  // Comparison is not a solve dependency. Its retained inputs are forwarded
  // explicitly, whether this call reuses the publication or computes it again.
  table->replay.reads = NULL;
  *out_reused = loom_value_fact_loop_inputs_equal(table, record, op);
  table->replay.reads = replay->parent_reads;
  if (*out_reused) {
    loom_value_fact_loop_forward_reads(replay->parent_reads, record);
    table->replay.loop = replay;
    return iree_ok_status();
  }
  loom_value_fact_loop_reads_t* reads = record->reads;
  if (!reads->bits) {
    const iree_host_size_t word_count =
        (reads->index->range.capacity + 63) / 64;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        replay->root->arena, word_count, sizeof(*reads->words),
        (void**)&reads->words));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(replay->root->arena, word_count,
                                  sizeof(*reads->bits), (void**)&reads->bits));
    memset(reads->bits, 0, word_count * sizeof(*reads->bits));
  }
  reads->record = record;
  // Preserve dependencies of equations that can themselves reuse prior facts.
  loom_value_fact_loop_forward_reads(reads, record);
  replay->record = record;
  table->replay.loop = replay;
  table->replay.reads = reads;
  return iree_ok_status();
}

static iree_status_t loom_value_fact_loop_publish_inputs(
    loom_value_fact_table_t* table, loom_value_fact_loop_replay_t* replay) {
  loom_value_fact_loop_replay_record_t* record = replay->record;
  loom_value_fact_loop_reads_t* reads = record->reads;
  iree_host_size_t input_count = 0;
  for (iree_host_size_t i = 0; i < reads->word_count; ++i) {
    input_count += iree_math_count_ones_u64(reads->bits[reads->words[i]]);
  }
  if (input_count > record->input_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        replay->root->arena, 0, input_count, sizeof(*record->inputs),
        &record->input_capacity, (void**)&record->inputs));
  }
  record->input_count = 0;
  for (iree_host_size_t i = 0; i < reads->word_count; ++i) {
    const uint32_t word = reads->words[i];
    uint64_t bits = reads->bits[word];
    while (bits) {
      const loom_value_id_t value = reads->index->range.first_value_id +
                                    word * 64 +
                                    iree_math_count_trailing_zeros_u64(bits);
      bits &= bits - 1;
      loom_value_fact_loop_input_snapshot(
          table, value, &record->inputs[record->input_count++]);
    }
  }
  record->target = table->context.target_facts;
  record->temporal = loom_value_fact_table_block_temporal_scope(
                         table, replay->op->parent_block)
                         .flags;
  record->withdrawal = replay->root->index->withdrawal;
  record->completed = true;
  return iree_ok_status();
}

iree_status_t loom_value_fact_loop_replay_end(
    loom_value_fact_table_t* table, loom_value_fact_loop_replay_t* replay,
    bool completed) {
  table->replay.reads = NULL;
  iree_status_t status = iree_ok_status();
  if (replay->record) {
    replay->record->completed = false;
    if (completed) {
      status = loom_value_fact_loop_publish_inputs(table, replay);
    }
    loom_value_fact_loop_reads_reset(replay->record->reads);
    if (iree_status_is_ok(status) && completed) {
      loom_value_fact_loop_forward_reads(replay->parent_reads, replay->record);
    }
  }
  if (replay->index) {
    for (loom_value_fact_loop_replay_record_t* record = replay->index->records;
         record; record = record->next) {
      loom_value_fact_table_clear_loop_replay(table, record->body);
    }
  }
  table->replay.loop = replay->parent;
  table->replay.reads = replay->parent_reads;
  return status;
}

void loom_value_fact_loop_replay_invalidate(loom_value_fact_table_t* table) {
  if (table->replay.loop && table->replay.loop->root->index) {
    ++table->replay.loop->root->index->withdrawal;
  }
}
