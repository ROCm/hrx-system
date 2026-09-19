// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/rewriter.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/facts.h"
#include "loom/ir/module.h"
#include "loom/ir/value_refs.h"
#include "loom/ops/op_defs.h"
#include "loom/util/fact_cfg.h"

#define LOOM_REWRITER_INITIAL_WORKLIST_CAPACITY 64
#define LOOM_REWRITER_INITIAL_REGION_STACK_CAPACITY 8

typedef enum loom_rewriter_user_change_flag_bits_e {
  // Operand identities are changing, not just the facts of the same value.
  LOOM_REWRITER_USER_CHANGE_FLAG_REPLACED = 1u << 0,
} loom_rewriter_user_change_flag_bits_t;
typedef uint32_t loom_rewriter_user_change_flags_t;

static iree_status_t loom_rewriter_add_users_to_worklist(
    loom_rewriter_t* rewriter, loom_value_id_t value_id,
    loom_rewriter_user_change_flags_t flags);
static iree_status_t loom_rewriter_add_result_users_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op);
static iree_status_t loom_rewriter_add_summary_ops_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op);
static iree_status_t loom_rewriter_add_parent_summary_ops_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op);

static bool loom_rewriter_op_summarizes_nested_regions(
    const loom_op_vtable_t* vtable) {
  return vtable && (vtable->loop_like || vtable->region_branch);
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
  return descriptor->terminator == LOOM_OP_KIND_UNKNOWN ||
         terminator->kind == descriptor->terminator;
}

static bool loom_rewriter_nested_region_summary_is_ready(
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
static iree_status_t loom_rewriter_recompute_op_facts(loom_rewriter_t* rewriter,
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

//===----------------------------------------------------------------------===//
// Builder callback
//===----------------------------------------------------------------------===//

// Region owners publish inherited context before their children are built or
// revisited. Cyclic summaries refresh it again when control facts change.
static iree_status_t loom_rewriter_seed_nested_temporal_scope(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  if (!rewriter->fact_table || !op->region_count) {
    return iree_ok_status();
  }
  const loom_value_facts_t scope = loom_value_fact_table_block_temporal_scope(
      rewriter->fact_table, op->parent_block);
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    if (regions[i]) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_set_region_temporal_scope(
          rewriter->fact_table, regions[i], scope));
    }
  }
  return iree_ok_status();
}

// Callback installed on the builder. Fired by finalize_op after a new op's
// direct fields are fully wired. Adds the op to the rewriter's worklist so the
// driver can attempt patterns on it. When analysis is enabled, also computes
// facts for ordinary op results and complete structured ops. Structured op
// shells remain on the worklist until callers have populated the regions
// created by their builders.
static iree_status_t loom_rewriter_on_op_finalized(void* user_data,
                                                   loom_op_t* op) {
  loom_rewriter_t* rewriter = (loom_rewriter_t*)user_data;
  ++rewriter->created_op_count;
  IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_seed_nested_temporal_scope(rewriter, op));
  // Newly defined results have no users yet. Only a terminator changes an
  // existing control or payload equation before its results are connected.
  if (iree_any_bit_set(op->traits, LOOM_TRAIT_TERMINATOR)) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_summary_ops_to_worklist(rewriter, op));
  } else {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, op));
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(rewriter->module, op);
  if (!loom_rewriter_nested_region_summary_is_ready(rewriter, op, vtable)) {
    return iree_ok_status();
  }
  return loom_rewriter_recompute_op_facts(rewriter, op);
}

// Reinserted subtrees retain their IR effect summaries and receive execution
// context from their new parent. Requeue children even when the owner has no
// result-fact summary: their observations can change independently of it.
static iree_status_t loom_rewriter_record_subtree(loom_rewriter_t* rewriter,
                                                  loom_op_t* op) {
  loom_module_record_op_summaries(rewriter->module, op);
  IREE_RETURN_IF_ERROR(loom_rewriter_seed_nested_temporal_scope(rewriter, op));
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_region_t* region = regions[region_index];
    if (!region) {
      continue;
    }
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        IREE_RETURN_IF_ERROR(loom_rewriter_record_subtree(rewriter, child_op));
      }
    }
  }
  return rewriter->fact_table ? loom_rewriter_add_to_worklist(rewriter, op)
                              : iree_ok_status();
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

static void loom_rewriter_release_cfg_facts(loom_rewriter_t* rewriter) {
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

//===----------------------------------------------------------------------===//
// Rewriter lifecycle
//===----------------------------------------------------------------------===//

iree_status_t loom_rewriter_initialize(loom_rewriter_t* rewriter,
                                       loom_module_t* module,
                                       iree_arena_allocator_t* arena) {
  memset(rewriter, 0, sizeof(*rewriter));
  rewriter->module = module;
  rewriter->arena = arena;
  rewriter->name_policy = LOOM_REWRITER_NAME_POLICY_DEFAULT;

  // Initialize the builder with the module's arena (new ops live in
  // the module, not the pass scratch arena).
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &rewriter->builder);

  // Install the finalize callback so new ops enter the worklist.
  rewriter->builder.on_op_finalized.fn = loom_rewriter_on_op_finalized;
  rewriter->builder.on_op_finalized.user_data = rewriter;

  // Allocate initial worklist from the pass arena.
  iree_host_size_t capacity = LOOM_REWRITER_INITIAL_WORKLIST_CAPACITY;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, capacity, sizeof(loom_op_t*), (void**)&rewriter->worklist));
  rewriter->worklist_capacity = capacity;

  return iree_ok_status();
}

void loom_rewriter_deinitialize(loom_rewriter_t* rewriter) {
  loom_rewriter_release_cfg_facts(rewriter);
  // Clear ON_WORKLIST bits on any remaining ops.
  for (iree_host_size_t i = 0; i < rewriter->worklist_count; ++i) {
    rewriter->worklist[i]->flags &= ~LOOM_OP_FLAG_ON_WORKLIST;
  }
  // The worklist array itself is arena-allocated — freed when the
  // pass arena is deinitialized.
  memset(rewriter, 0, sizeof(*rewriter));
}

iree_status_t loom_rewriter_seed_region(loom_rewriter_t* rewriter,
                                        loom_region_t* region) {
  if (!region) {
    return iree_ok_status();
  }

  // Iterative DFS over the explicit region tree. We maintain a stack of regions
  // to visit so that ops in nested regions are added to the worklist alongside
  // top-level ops.
  iree_host_size_t stack_capacity = LOOM_REWRITER_INITIAL_REGION_STACK_CAPACITY;
  loom_region_t** region_stack = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(rewriter->arena, stack_capacity,
                                sizeof(loom_region_t*), (void**)&region_stack));
  iree_host_size_t stack_count = 0;
  region_stack[stack_count++] = region;

  while (stack_count > 0) {
    loom_region_t* region = region_stack[--stack_count];
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, op));
        if (op->region_count == 0) {
          continue;
        }
        // Ensure space for all nested regions of this op.
        iree_host_size_t needed = stack_count + op->region_count;
        if (needed > stack_capacity) {
          IREE_RETURN_IF_ERROR(iree_arena_grow_array(
              rewriter->arena, stack_count, needed, sizeof(loom_region_t*),
              &stack_capacity, (void**)&region_stack));
        }
        loom_region_t** regions = loom_op_regions(op);
        for (uint8_t r = 0; r < op->region_count; ++r) {
          if (regions[r]) {
            region_stack[stack_count++] = regions[r];
          }
        }
      }
    }
  }

  return iree_ok_status();
}

iree_status_t loom_rewriter_seed_function(loom_rewriter_t* rewriter,
                                          loom_func_like_t function) {
  return loom_rewriter_seed_region(rewriter, loom_func_like_body(function));
}

void loom_rewriter_attach_value_facts(loom_rewriter_t* rewriter,
                                      loom_value_fact_table_t* facts) {
  loom_rewriter_release_cfg_facts(rewriter);
  rewriter->fact_table = facts;
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
    IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, terminator));
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
      structure->control_flow
          .dirty[structure->graph.blocks[block_index].component] = false;
      return status;
    }
  }
  return loom_value_fact_table_update_cfg_block_args(
      rewriter->fact_table, rewriter->module, structure, block_index,
      loom_rewriter_cfg_argument_changed, rewriter);
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
        bool was_dirty = false;
        if (old_indices[i] != IREE_HOST_SIZE_MAX &&
            old_structure->control_flow.components.count &&
            old_structure->graph.blocks[old_indices[i]].reachable) {
          was_dirty =
              old_structure->control_flow
                  .dirty[old_structure->graph.blocks[old_indices[i]].component];
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
    status = loom_rewriter_refresh_cfg_block_facts(rewriter, old_structure,
                                                   structure, &next_arena);
    iree_arena_checkpoint_restore(&checkpoint);
    iree_arena_deinitialize(&storage->arena);
    storage->arena = next_arena;
  } else {
    iree_arena_deinitialize(&next_arena);
  }
  return status;
}

static iree_status_t loom_rewriter_update_successor_facts(
    loom_rewriter_t* rewriter, loom_op_t* op) {
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
        structure->control_flow.dirty[component_index]) {
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
  if (structure->control_flow.components.count &&
      structure->graph.blocks[successor->region_index].reachable) {
    iree_host_size_t component_index =
        structure->graph.blocks[successor->region_index].component;
    if (structure->control_flow.components.values[component_index].is_cycle &&
        !structure->control_flow.dirty[component_index]) {
      return iree_ok_status();
    }
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

iree_status_t loom_rewriter_build_constant(loom_rewriter_t* rewriter,
                                           loom_value_facts_t facts,
                                           loom_type_t result_type,
                                           loom_location_id_t location,
                                           loom_value_id_t* out_value_id) {
  if (!rewriter->materialize_constant) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "no materialize_constant callback set");
  }
  return rewriter->materialize_constant(&rewriter->builder, facts, result_type,
                                        location, out_value_id);
}

iree_status_t loom_rewriter_try_fold(loom_rewriter_t* rewriter, loom_op_t* op,
                                     bool* out_folded) {
  *out_folded = false;
  if (!rewriter->fact_table) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_update_successor_facts(rewriter, op));
  // Constant-like ops are already the canonical representation of their
  // compile-time value. Their source contract forbids operands and regions,
  // while semantic mutation refreshes their facts at the mutation boundary.
  // Recomputing here can neither expose downstream facts nor fold the op to a
  // different representation.
  if (iree_any_bit_set(op->traits, LOOM_TRAIT_CONSTANT_LIKE)) {
    return iree_ok_status();
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(rewriter->module, op);
  if (!vtable || (!vtable->infer_facts &&
                  !loom_rewriter_op_summarizes_nested_regions(vtable))) {
    return iree_ok_status();
  }

  // Compute facts for this op (updates the table, reuses table scratch).
  bool facts_changed = false;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_op_and_report(
      rewriter->fact_table, rewriter->module, op, &facts_changed));
  if (facts_changed) {
    rewriter->flags |= LOOM_REWRITER_FLAG_FACTS_CHANGED;
    // Ensure downstream ops are revisited with updated facts. Type-use edges
    // can point back to another result of the same op, so only reschedule when
    // facts actually changed; otherwise dynamic shaped multi-result ops can
    // keep themselves on the worklist forever.
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_result_users_to_worklist(rewriter, op));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_summary_ops_to_worklist(rewriter, op));
  }

  // Cannot materialize without a callback, and don't replace
  // constant-like ops with themselves.
  if (!rewriter->materialize_constant) {
    return iree_ok_status();
  }
  if (!vtable->infer_facts) {
    return iree_ok_status();
  }
  if (loom_traits_has_side_effects(
          loom_op_effective_traits(rewriter->module, op))) {
    return iree_ok_status();
  }

  // Check if all results are now exact constants.
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) {
      return iree_ok_status();
    }
    loom_value_facts_t facts =
        loom_value_fact_table_lookup(rewriter->fact_table, results[i]);
    if (!loom_value_facts_is_exact(facts)) {
      return iree_ok_status();
    }
  }
  if (op->result_count == 0) {
    return iree_ok_status();
  }

  // All results are exact. Materialize constants in the same block
  // as the op being folded, then RAUW+erase. Dead operands are
  // handled by cascading DCE (erase adds providers to worklist,
  // worklist pop checks is_trivially_dead).
  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);

  loom_value_id_t* replacement_ids = NULL;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_value_id_scratch(
      rewriter->fact_table, op->result_count, &replacement_ids));

  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_facts_t facts =
        loom_value_fact_table_lookup(rewriter->fact_table, results[i]);
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, results[i]);
    IREE_RETURN_IF_ERROR(loom_rewriter_build_constant(
        rewriter, facts, result_type, op->location, &replacement_ids[i]));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacement_ids, op->result_count, value_checkpoint));

  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacement_ids, op->result_count));

  *out_folded = true;
  return iree_ok_status();
}

loom_value_id_t loom_rewriter_value_checkpoint(
    const loom_rewriter_t* rewriter) {
  iree_host_size_t value_count = rewriter->module->values.count;
  return value_count <= UINT32_MAX ? (loom_value_id_t)value_count
                                   : LOOM_VALUE_ID_INVALID;
}

iree_status_t loom_rewriter_preserve_result_names_on_new_values(
    loom_rewriter_t* rewriter, const loom_op_t* op,
    const loom_value_id_t* replacements, uint16_t count,
    loom_value_id_t value_checkpoint) {
  if (count > op->result_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replacement count %u exceeds op result count %u",
                            (unsigned)count, (unsigned)op->result_count);
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < count; ++i) {
    loom_value_id_t old_result = results[i];
    loom_value_id_t replacement = replacements[i];
    if (old_result == LOOM_VALUE_ID_INVALID ||
        replacement == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    if (replacement < value_checkpoint) {
      continue;
    }
    if ((iree_host_size_t)replacement >= rewriter->module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "replacement value %%%u out of range",
                              (unsigned)replacement);
    }
    IREE_RETURN_IF_ERROR(
        loom_rewriter_copy_value_name(rewriter, old_result, replacement));
  }
  return iree_ok_status();
}

iree_status_t loom_rewriter_copy_value_name(loom_rewriter_t* rewriter,
                                            loom_value_id_t source_value,
                                            loom_value_id_t target_value) {
  if (!iree_any_bit_set(rewriter->name_policy,
                        LOOM_REWRITER_NAME_POLICY_PRESERVE_NAMES)) {
    return iree_ok_status();
  }
  return loom_module_copy_value_name(rewriter->module, source_value,
                                     target_value);
}

iree_status_t loom_rewriter_move_value_name(loom_rewriter_t* rewriter,
                                            loom_value_id_t source_value,
                                            loom_value_id_t target_value) {
  if (!iree_any_bit_set(rewriter->name_policy,
                        LOOM_REWRITER_NAME_POLICY_PRESERVE_NAMES)) {
    return iree_ok_status();
  }
  return loom_module_move_value_name(rewriter->module, source_value,
                                     target_value);
}

iree_status_t loom_rewriter_clear_value_name(loom_rewriter_t* rewriter,
                                             loom_value_id_t value) {
  return loom_module_clear_value_name(rewriter->module, value);
}

iree_status_t loom_rewriter_try_set_derived_value_name(
    loom_rewriter_t* rewriter, loom_value_id_t source_value,
    loom_value_id_t target_value, iree_string_view_t suffix) {
  if (!iree_any_bit_set(rewriter->name_policy,
                        LOOM_REWRITER_NAME_POLICY_DERIVE_DEBUG_NAMES)) {
    return iree_ok_status();
  }
  return loom_module_try_set_derived_value_name(
      rewriter->module, source_value, target_value, suffix, rewriter->arena);
}

//===----------------------------------------------------------------------===//
// Mini-DCE
//===----------------------------------------------------------------------===//

bool loom_rewriter_is_trivially_dead(const loom_rewriter_t* rewriter,
                                     const loom_op_t* op) {
  return loom_op_is_trivially_dead(rewriter->module, op);
}

iree_status_t loom_rewriter_erase_if_dead(loom_rewriter_t* rewriter,
                                          loom_op_t* op, bool* out_erased) {
  *out_erased = false;
  if (!loom_rewriter_is_trivially_dead(rewriter, op)) {
    return iree_ok_status();
  }
  // loom_rewriter_erase adds operand providers to the worklist for
  // cascading DCE before erasing.
  IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, op));
  *out_erased = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Worklist
//===----------------------------------------------------------------------===//

iree_status_t loom_rewriter_add_to_worklist(loom_rewriter_t* rewriter,
                                            loom_op_t* op) {
  if (op->flags & (LOOM_OP_FLAG_DEAD | LOOM_OP_FLAG_ON_WORKLIST)) {
    return iree_ok_status();
  }
  op->flags |= LOOM_OP_FLAG_ON_WORKLIST;

  // Grow worklist if needed (arena allocation — old array is abandoned).
  if (rewriter->worklist_count >= rewriter->worklist_capacity) {
    iree_status_t status = iree_arena_grow_array(
        rewriter->arena, rewriter->worklist_count, rewriter->worklist_count + 1,
        sizeof(loom_op_t*), &rewriter->worklist_capacity,
        (void**)&rewriter->worklist);
    if (!iree_status_is_ok(status)) {
      op->flags &= ~LOOM_OP_FLAG_ON_WORKLIST;
      return status;
    }
  }

  rewriter->worklist[rewriter->worklist_count++] = op;
  return iree_ok_status();
}

static iree_status_t loom_rewriter_add_cfg_summary_to_worklist(
    loom_rewriter_t* rewriter, const loom_block_t* block) {
  if (!rewriter->fact_table || !block || !block->parent_region) {
    return iree_ok_status();
  }
  const loom_value_fact_cfg_region_t* structure =
      loom_value_fact_table_lookup_cfg_region(rewriter->fact_table,
                                              block->parent_region);
  if (!structure || !structure->control_flow.components.count) {
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
  loom_op_t* anchor = structure->control_flow.anchors[component_index];
  if (anchor) {
    structure->control_flow.dirty[component_index] = true;
    IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, anchor));
  }
  return iree_ok_status();
}

// Branch operand identities define the forwarding graph independently of the
// facts carried by those operands. Structural CFG edits publish a new snapshot;
// payload edits only invalidate this successor's retained partition.
static void loom_rewriter_invalidate_cfg_forwarding(loom_rewriter_t* rewriter,
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

// Structured parents own the summaries of their regions. Inserting or erasing
// an unused definition does not change an equation in its own CFG component.
static iree_status_t loom_rewriter_add_parent_summary_ops_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  bool has_cfg_facts =
      rewriter->fact_table && rewriter->fact_table->regions.cfg_count;
  for (loom_op_t* parent = op ? op->parent_op : NULL; parent;
       parent = parent->parent_op) {
    if (has_cfg_facts) {
      IREE_RETURN_IF_ERROR(loom_rewriter_add_cfg_summary_to_worklist(
          rewriter, parent->parent_block));
    }
    const loom_op_vtable_t* vtable = loom_op_vtable(rewriter->module, parent);
    if (loom_rewriter_op_summarizes_nested_regions(vtable)) {
      IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, parent));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_rewriter_add_summary_ops_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  if (op && rewriter->fact_table && rewriter->fact_table->regions.cfg_count) {
    const loom_op_vtable_t* vtable = loom_op_vtable(rewriter->module, op);
    // Opaque operations define facts from their result types, independently of
    // their operands. Operand rewrites still revisit the operation, but cannot
    // change a cyclic fact equation through it. Terminators feed block
    // arguments or structured summaries even without an inference callback.
    if ((vtable && vtable->infer_facts) ||
        loom_rewriter_op_summarizes_nested_regions(vtable) ||
        iree_any_bit_set(op->traits, LOOM_TRAIT_TERMINATOR)) {
      IREE_RETURN_IF_ERROR(loom_rewriter_add_cfg_summary_to_worklist(
          rewriter, op->parent_block));
    }
    for (uint8_t i = 0; i < op->successor_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_rewriter_add_cfg_summary_to_worklist(
          rewriter, loom_op_successors(op)[i]));
    }
  }
  return loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, op);
}

loom_op_t* loom_rewriter_pop(loom_rewriter_t* rewriter) {
  while (rewriter->worklist_count > 0) {
    loom_op_t* op = rewriter->worklist[--rewriter->worklist_count];
    op->flags &= ~LOOM_OP_FLAG_ON_WORKLIST;
    // Skip ops that were erased while on the worklist.
    if (op->flags & LOOM_OP_FLAG_DEAD) {
      continue;
    }
    return op;
  }
  return NULL;
}

//===----------------------------------------------------------------------===//
// Safe IR mutation
//===----------------------------------------------------------------------===//

static iree_status_t loom_rewriter_add_value_ref_provider_to_worklist(
    loom_value_id_t value_id, void* user_data) {
  loom_rewriter_t* rewriter = (loom_rewriter_t*)user_data;
  if (value_id >= rewriter->module->values.count) {
    return iree_ok_status();
  }
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  if (loom_value_is_block_arg(value)) {
    return iree_ok_status();
  }
  loom_op_t* def = loom_value_def_op(value);
  if (!def) {
    return iree_ok_status();
  }
  return loom_rewriter_add_to_worklist(rewriter, def);
}

static iree_status_t loom_rewriter_add_type_ref_providers_to_worklist(
    loom_rewriter_t* rewriter, loom_type_t type) {
  return loom_type_walk_value_refs(
      rewriter->module, type, loom_rewriter_add_value_ref_provider_to_worklist,
      rewriter);
}

static iree_status_t loom_rewriter_add_subtree_providers_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  return loom_op_walk_subtree_value_refs(
      rewriter->module, op, loom_rewriter_add_value_ref_provider_to_worklist,
      rewriter);
}

// Attribute membership includes TYPE and predicate dependencies even when the
// provider is absent from the owner's ordinary operands.
static iree_status_t loom_rewriter_add_attribute_users_to_worklist(
    loom_rewriter_t* rewriter, loom_value_id_t value_id) {
  loom_type_use_iterator_t users;
  loom_attribute_users_begin(&rewriter->module->type_uses, value_id, &users);
  for (loom_attribute_user_t user = loom_attribute_users_next(&users); user.op;
       user = loom_attribute_users_next(&users)) {
    IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, user.op));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_summary_ops_to_worklist(rewriter, user.op));
  }
  return iree_ok_status();
}

// Adds operand, attribute, and value-type users to the worklist.
static iree_status_t loom_rewriter_add_users_to_worklist(
    loom_rewriter_t* rewriter, loom_value_id_t value_id,
    loom_rewriter_user_change_flags_t flags) {
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    loom_op_t* user_op = loom_use_user_op(uses[i]);
    if (iree_any_bit_set(flags, LOOM_REWRITER_USER_CHANGE_FLAG_REPLACED)) {
      loom_rewriter_invalidate_cfg_forwarding(rewriter, user_op);
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, user_op));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_summary_ops_to_worklist(rewriter, user_op));
  }
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_attribute_users_to_worklist(rewriter, value_id));
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
        IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, def));
        IREE_RETURN_IF_ERROR(
            loom_rewriter_add_summary_ops_to_worklist(rewriter, def));
      }
    }
    const loom_use_t* user_value_uses = loom_value_uses(user_value);
    for (uint32_t i = 0; i < user_value->use_count; ++i) {
      loom_op_t* user_op = loom_use_user_op(user_value_uses[i]);
      IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, user_op));
      IREE_RETURN_IF_ERROR(
          loom_rewriter_add_summary_ops_to_worklist(rewriter, user_op));
    }
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_attribute_users_to_worklist(rewriter, user_value_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_rewriter_add_operand_users_except_to_worklist(
    loom_rewriter_t* rewriter, loom_value_id_t value_id,
    const loom_op_t* except_op) {
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    loom_op_t* user_op = loom_use_user_op(uses[i]);
    if (user_op == except_op) {
      continue;
    }
    loom_rewriter_invalidate_cfg_forwarding(rewriter, user_op);
    IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, user_op));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_summary_ops_to_worklist(rewriter, user_op));
  }
  return iree_ok_status();
}

iree_status_t loom_rewriter_replace_all_uses_and_erase(
    loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_value_id_t* replacements, uint16_t count) {
  loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        rewriter, results[i], replacements[i]));
  }
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_subtree_providers_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_op_erase(rewriter->module, op));
  ++rewriter->erased_op_count;
  rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

iree_status_t loom_rewriter_replace_all_uses_with(loom_rewriter_t* rewriter,
                                                  loom_value_id_t old_value,
                                                  loom_value_id_t new_value) {
  if (old_value == new_value) {
    return iree_ok_status();
  }
  if (old_value == LOOM_VALUE_ID_INVALID ||
      new_value == LOOM_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replacement values must be valid");
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_add_users_to_worklist(
      rewriter, old_value, LOOM_REWRITER_USER_CHANGE_FLAG_REPLACED));
  IREE_RETURN_IF_ERROR(
      loom_value_replace_all_uses_with(rewriter->module, old_value, new_value));
  rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

iree_status_t loom_rewriter_replace_all_uses_except(
    loom_rewriter_t* rewriter, loom_value_id_t old_value,
    loom_value_id_t new_value, const loom_op_t* except_op) {
  if (old_value == new_value) {
    return iree_ok_status();
  }
  if (old_value == LOOM_VALUE_ID_INVALID ||
      new_value == LOOM_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replacement values must be valid");
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_add_operand_users_except_to_worklist(
      rewriter, old_value, except_op));
  IREE_RETURN_IF_ERROR(loom_value_replace_all_uses_except(
      rewriter->module, old_value, new_value, except_op));
  rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

iree_status_t loom_rewriter_replace_results_with_materialized_values_and_erase(
    loom_rewriter_t* rewriter, loom_op_t* op,
    loom_materialize_value_fn_t materialize_value) {
  if (op->result_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "op has no results to replace");
  }

  loom_builder_set_before(&rewriter->builder, op);
  loom_value_id_t value_checkpoint = loom_rewriter_value_checkpoint(rewriter);
  const loom_value_id_t* results = loom_op_const_results(op);
  loom_value_id_t* replacement_ids = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      rewriter->arena, op->result_count, sizeof(loom_value_id_t),
      (void**)&replacement_ids));
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "op result %u is invalid", (unsigned)i);
    }
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, results[i]);
    IREE_RETURN_IF_ERROR(materialize_value(&rewriter->builder, result_type,
                                           op->location, &replacement_ids[i]));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacement_ids, op->result_count, value_checkpoint));

  return loom_rewriter_replace_all_uses_and_erase(rewriter, op, replacement_ids,
                                                  op->result_count);
}

iree_status_t loom_rewriter_erase(loom_rewriter_t* rewriter, loom_op_t* op) {
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_subtree_providers_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_op_erase(rewriter->module, op));
  ++rewriter->erased_op_count;
  rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

static bool loom_rewriter_op_is_ancestor_of(const loom_op_t* ancestor,
                                            const loom_op_t* op) {
  for (const loom_op_t* current = op; current; current = current->parent_op) {
    if (current == ancestor) {
      return true;
    }
  }
  return false;
}

static bool loom_rewriter_op_belongs_to_module(const loom_module_t* module,
                                               const loom_op_t* op) {
  if (!op) {
    return false;
  }
  const loom_op_t* root_op = op;
  while (root_op->parent_op != NULL) {
    root_op = root_op->parent_op;
  }
  return root_op->parent_block != NULL &&
         root_op->parent_block->parent_region == module->body;
}

static bool loom_rewriter_parent_owns_block(const loom_module_t* module,
                                            const loom_op_t* parent_op,
                                            const loom_block_t* block) {
  if (!block || !block->parent_region) {
    return false;
  }
  uint16_t block_index = 0;
  if (!loom_region_try_block_index(block->parent_region, block, &block_index)) {
    return false;
  }
  if (parent_op == NULL) {
    return block->parent_region == module->body;
  }
  if (!loom_rewriter_op_belongs_to_module(module, parent_op)) {
    return false;
  }
  loom_region_t* const* regions = loom_op_regions(parent_op);
  for (uint8_t i = 0; i < parent_op->region_count; ++i) {
    if (regions[i] == block->parent_region) {
      return true;
    }
  }
  return false;
}

iree_status_t loom_rewriter_move_region_blocks(
    loom_rewriter_t* rewriter, loom_region_t* source_region,
    loom_op_t* source_parent_op, loom_region_t* target_region,
    uint16_t target_block_index, loom_op_t* target_parent_op,
    loom_block_t** out_moved_entry_block) {
  *out_moved_entry_block = NULL;
  if (source_region == target_region) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "source and target regions must be distinct");
  }
  if (source_region->block_count == 0 || target_region->block_count == 0 ||
      source_region->blocks[0] != &source_region->entry_block) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "region move requires source and target entry blocks and an embedded "
        "source entry");
  }
  if (target_block_index > target_region->block_count ||
      (target_region->block_count > 0 && target_block_index == 0)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target block insertion index %u is outside the movable range for %u "
        "blocks",
        (unsigned)target_block_index, (unsigned)target_region->block_count);
  }

  iree_host_size_t final_block_count = 0;
  if (!iree_host_size_checked_add(target_region->block_count,
                                  source_region->block_count,
                                  &final_block_count) ||
      final_block_count > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "moved region block count exceeds UINT16_MAX");
  }
  if (!loom_rewriter_parent_owns_block(rewriter->module, source_parent_op,
                                       source_region->blocks[0]) ||
      !loom_rewriter_parent_owns_block(rewriter->module, target_parent_op,
                                       target_region->blocks[0])) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source and target regions must belong to their declared parents");
  }
  if (source_parent_op != target_parent_op &&
      loom_rewriter_op_is_ancestor_of(source_parent_op, target_parent_op)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot move a region into one of its owning op's descendants");
  }
  for (uint16_t block_index = 0; block_index < source_region->block_count;
       ++block_index) {
    loom_block_t* block = source_region->blocks[block_index];
    if (!block || block->parent_region != source_region ||
        block->region_index != block_index) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "source region block ownership is malformed");
    }
    for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
      const loom_value_id_t value_id = loom_block_arg_id(block, arg_index);
      if (value_id >= rewriter->module->values.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "source block argument is out of range");
      }
      const loom_value_t* value = loom_module_value(rewriter->module, value_id);
      if (!loom_value_is_block_arg(value) ||
          loom_value_def_block(value) != block) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "source block argument definition does not match its block");
      }
    }
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (op->parent_block != block || op->parent_op != source_parent_op ||
          iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "source block operation ownership is malformed");
      }
    }
  }

  loom_block_t* moved_entry_block = NULL;
  IREE_RETURN_IF_ERROR(
      loom_module_allocate_block(rewriter->module, &moved_entry_block));
  iree_host_size_t entry_comment_count = 0;
  const iree_string_view_t* entry_comments = loom_module_block_comments(
      rewriter->module, &source_region->entry_block, &entry_comment_count);
  if (entry_comment_count > 0) {
    IREE_RETURN_IF_ERROR(
        loom_module_attach_block_comments(rewriter->module, moved_entry_block,
                                          entry_comments, entry_comment_count));
  }

  const bool use_source_storage =
      source_region->block_capacity > target_region->block_capacity;
  loom_region_t* storage_region =
      use_source_storage ? source_region : target_region;
  IREE_RETURN_IF_ERROR(loom_region_reserve_block_capacity(
      rewriter->module, storage_region, final_block_count));

  for (uint16_t block_index = 0; block_index < source_region->block_count;
       ++block_index) {
    loom_op_t* op = NULL;
    loom_block_for_each_op(source_region->blocks[block_index], op) {
      IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, op));
    }
  }
  if (source_parent_op != NULL) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_to_worklist(rewriter, source_parent_op));
  }
  if (target_parent_op != NULL && target_parent_op != source_parent_op) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_to_worklist(rewriter, target_parent_op));
  }
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_summary_ops_to_worklist(rewriter, source_parent_op));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_summary_ops_to_worklist(rewriter, target_parent_op));

  const uint16_t source_block_count = source_region->block_count;
  const uint16_t target_block_count = target_region->block_count;
  const loom_region_instance_flags_t source_flags = source_region->flags;
  loom_block_t* old_entry_block = &source_region->entry_block;
  *moved_entry_block = *old_entry_block;
  moved_entry_block->label_id = LOOM_STRING_ID_INVALID;
  source_region->blocks[0] = moved_entry_block;

  for (uint16_t block_index = 0; block_index < source_block_count;
       ++block_index) {
    loom_block_t* block = source_region->blocks[block_index];
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      loom_module_drop_op_summaries(rewriter->module, op);
    }
  }

  loom_block_t** moved_blocks = source_region->blocks;
  loom_block_t** target_blocks = target_region->blocks;
  loom_block_t** combined_blocks = storage_region->blocks;
  const uint16_t combined_capacity = storage_region->block_capacity;
  if (storage_region == source_region) {
    memmove(combined_blocks + target_block_index, moved_blocks,
            (iree_host_size_t)source_block_count * sizeof(*combined_blocks));
    if (target_block_index > 0) {
      memcpy(combined_blocks, target_blocks,
             (iree_host_size_t)target_block_index * sizeof(*combined_blocks));
    }
    const uint16_t target_tail_count = target_block_count - target_block_index;
    if (target_tail_count > 0) {
      memcpy(combined_blocks + target_block_index + source_block_count,
             target_blocks + target_block_index,
             (iree_host_size_t)target_tail_count * sizeof(*combined_blocks));
    }
  } else {
    const uint16_t target_tail_count = target_block_count - target_block_index;
    if (target_tail_count > 0) {
      memmove(combined_blocks + target_block_index + source_block_count,
              combined_blocks + target_block_index,
              (iree_host_size_t)target_tail_count * sizeof(*combined_blocks));
    }
    memcpy(combined_blocks + target_block_index, moved_blocks,
           (iree_host_size_t)source_block_count * sizeof(*combined_blocks));
  }

  memset(old_entry_block, 0, sizeof(*old_entry_block));
  old_entry_block->label_id = LOOM_STRING_ID_INVALID;
  old_entry_block->region_index = 0;
  old_entry_block->parent_region = source_region;
  source_region->block_count = 1;
  source_region->block_capacity = 1;
  source_region->flags = 0;
  source_region->blocks = source_region->inline_blocks;
  source_region->inline_blocks[0] = old_entry_block;

  target_region->block_count = (uint16_t)final_block_count;
  target_region->block_capacity = combined_capacity;
  target_region->blocks = combined_blocks;
  target_region->flags |= source_flags;
  for (uint16_t block_index = target_block_index;
       block_index < target_region->block_count; ++block_index) {
    loom_block_t* block = target_region->blocks[block_index];
    block->parent_region = target_region;
    block->region_index = block_index;
  }
  for (uint16_t block_index = target_block_index;
       block_index < target_block_index + source_block_count; ++block_index) {
    loom_block_t* block = target_region->blocks[block_index];
    block->label_id = LOOM_STRING_ID_INVALID;
    for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
      loom_value_t* value = loom_module_value(
          rewriter->module, loom_block_arg_id(block, arg_index));
      value->def = loom_value_def_make_block(block, arg_index);
    }
    loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      op->parent_block = block;
      op->parent_op = target_parent_op;
      loom_block_t** successors = loom_op_successors(op);
      for (uint8_t successor_index = 0; successor_index < op->successor_count;
           ++successor_index) {
        if (successors[successor_index] == old_entry_block) {
          successors[successor_index] = moved_entry_block;
        }
      }
      IREE_RETURN_IF_ERROR(loom_rewriter_record_subtree(rewriter, op));
    }
  }

  IREE_ASSERT_EQ(source_region->read_effect_count, 0u);
  IREE_ASSERT_EQ(source_region->write_effect_count, 0u);
  IREE_ASSERT_EQ(source_region->convergent_effect_count, 0u);
  rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  *out_moved_entry_block = moved_entry_block;
  return iree_ok_status();
}

iree_status_t loom_rewriter_move_before(loom_rewriter_t* rewriter,
                                        loom_op_t* op, loom_op_t* before_op) {
  if (!op || !before_op) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "move requires live op and insertion target");
  }
  if (op == before_op || op->next_op == before_op) {
    return iree_ok_status();
  }
  if (iree_any_bit_set(op->flags | before_op->flags, LOOM_OP_FLAG_DEAD) ||
      !op->parent_block || !before_op->parent_block) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot move dead or unlinked operation");
  }
  // Same-module detached regions are valid move sources (materialization uses
  // them for staged IR), so source ownership cannot be inferred from ancestry.
  // The insertion point must be rooted in the rewrite module: it is the side
  // whose ancestry and region summaries this rewriter will update.
  if (!loom_rewriter_op_belongs_to_module(rewriter->module, before_op)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "move insertion target must belong to the rewrite "
                            "module");
  }
  if (loom_rewriter_op_is_ancestor_of(op, before_op)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot move an operation before one of its descendants");
  }

  loom_module_t* module = rewriter->module;
  loom_block_t* original_block = op->parent_block;
  loom_op_t* original_next_op = op->next_op;
  loom_op_t* original_parent_op = op->parent_op;
  loom_block_t* target_block = before_op->parent_block;
  loom_op_t* target_parent_op = before_op->parent_op;

  loom_block_unlink_op(module, op);
  op->parent_block = NULL;
  op->parent_op = target_parent_op;
  iree_status_t status =
      loom_block_insert_before_op(module, target_block, before_op, op);
  if (!iree_status_is_ok(status)) {
    op->parent_block = NULL;
    op->parent_op = original_parent_op;
    iree_status_t restore_status = loom_block_insert_before_op(
        module, original_block, original_next_op, op);
    if (iree_status_is_ok(restore_status)) {
      restore_status = loom_rewriter_record_subtree(rewriter, op);
    }
    return iree_status_join(status, restore_status);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_record_subtree(rewriter, op));

  IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_add_summary_ops_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_summary_ops_to_worklist(rewriter, before_op));
  rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

iree_status_t loom_rewriter_move_to_block_end(loom_rewriter_t* rewriter,
                                              loom_op_t* op,
                                              loom_block_t* target_block,
                                              loom_op_t* target_parent_op) {
  if (!op || !target_block) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "move requires live op and target block");
  }
  if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD) || !op->parent_block ||
      !target_block->parent_region) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot move dead or unlinked operation");
  }
  if (!loom_rewriter_op_belongs_to_module(rewriter->module, op) ||
      !loom_rewriter_parent_owns_block(rewriter->module, target_parent_op,
                                       target_block)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "move source and target must belong to the rewrite module");
  }
  if (target_parent_op != NULL &&
      loom_rewriter_op_is_ancestor_of(op, target_parent_op)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "cannot move an operation into one of its descendants");
  }
  if (op->parent_block == target_block && op->next_op == NULL &&
      op->parent_op == target_parent_op) {
    return iree_ok_status();
  }

  loom_module_t* module = rewriter->module;
  loom_block_t* original_block = op->parent_block;
  loom_op_t* original_next_op = op->next_op;
  loom_op_t* original_parent_op = op->parent_op;

  loom_block_unlink_op(module, op);
  op->parent_block = NULL;
  op->parent_op = target_parent_op;
  iree_status_t status = loom_block_append_op(module, target_block, op);
  if (!iree_status_is_ok(status)) {
    op->parent_block = NULL;
    op->parent_op = original_parent_op;
    iree_status_t restore_status = loom_block_insert_before_op(
        module, original_block, original_next_op, op);
    if (iree_status_is_ok(restore_status)) {
      restore_status = loom_rewriter_record_subtree(rewriter, op);
    }
    return iree_status_join(status, restore_status);
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_record_subtree(rewriter, op));

  IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_add_summary_ops_to_worklist(rewriter, op));
  rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

iree_status_t loom_rewriter_set_operand(loom_rewriter_t* rewriter,
                                        loom_op_t* op, uint16_t operand_index,
                                        loom_value_id_t new_value) {
  IREE_RETURN_IF_ERROR(
      loom_op_set_operand(rewriter->module, op, operand_index, new_value));
  loom_rewriter_invalidate_cfg_forwarding(rewriter, op);
  IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_add_summary_ops_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_recompute_op_facts(rewriter, op));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_result_users_to_worklist(rewriter, op));
  rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

iree_status_t loom_rewriter_set_value_type(loom_rewriter_t* rewriter,
                                           loom_value_id_t value_id,
                                           loom_type_t new_type) {
  loom_type_t old_type = loom_module_value_type(rewriter->module, value_id);
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_type_ref_providers_to_worklist(rewriter, old_type));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_type_ref_providers_to_worklist(rewriter, new_type));
  IREE_RETURN_IF_ERROR(
      loom_module_set_value_type(rewriter->module, value_id, new_type));
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  if (!loom_value_is_block_arg(value)) {
    loom_op_t* defining_op = loom_value_def_op(value);
    if (defining_op &&
        iree_any_bit_set(defining_op->traits, LOOM_TRAIT_CONSTANT_LIKE)) {
      IREE_RETURN_IF_ERROR(
          loom_rewriter_recompute_op_facts(rewriter, defining_op));
      IREE_RETURN_IF_ERROR(
          loom_rewriter_add_summary_ops_to_worklist(rewriter, defining_op));
    }
  }
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_users_to_worklist(rewriter, value_id, /*flags=*/0));
  rewriter->flags |=
      LOOM_REWRITER_FLAG_CHANGED | LOOM_REWRITER_FLAG_TYPE_CHANGED;
  return iree_ok_status();
}

// Adds users of all results of |op| to the worklist.
static iree_status_t loom_rewriter_add_result_users_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] != LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_rewriter_add_users_to_worklist(
          rewriter, results[i], /*flags=*/0));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_rewriter_validate_attr_write(
    loom_rewriter_t* rewriter, const loom_op_t* op, uint16_t attr_index,
    loom_attribute_t value) {
  if (attr_index >= op->attribute_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "attribute index %u is out of range for op with %u attributes",
        (unsigned)attr_index, (unsigned)op->attribute_count);
  }

  const loom_op_vtable_t* vtable = loom_op_vtable(rewriter->module, op);
  if (!vtable || !vtable->attr_descriptors ||
      attr_index >= vtable->attribute_count) {
    if (value.kind == LOOM_ATTR_DICT) {
      IREE_RETURN_IF_ERROR(
          loom_module_verify_canonical_attr_dict(rewriter->module, value));
    }
    return iree_ok_status();
  }

  const loom_attr_descriptor_t* descriptor =
      &vtable->attr_descriptors[attr_index];
  if ((descriptor->flags & LOOM_ATTR_OPTIONAL) && loom_attr_is_absent(value)) {
    return iree_ok_status();
  }
  if (!loom_attr_descriptor_accepts_kind(descriptor,
                                         (loom_attr_kind_t)value.kind)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "attribute index %u on op '%.*s' expects kind %u, got %u",
        (unsigned)attr_index, (int)loom_op_vtable_name(vtable).size,
        loom_op_vtable_name(vtable).data, (unsigned)descriptor->attr_kind,
        (unsigned)value.kind);
  }
  if (value.kind == LOOM_ATTR_DICT) {
    IREE_RETURN_IF_ERROR(
        loom_module_verify_canonical_attr_dict(rewriter->module, value));
  }
  return iree_ok_status();
}

iree_status_t loom_rewriter_set_attr(loom_rewriter_t* rewriter, loom_op_t* op,
                                     uint16_t attr_index,
                                     loom_attribute_t value) {
  IREE_RETURN_IF_ERROR(
      loom_rewriter_validate_attr_write(rewriter, op, attr_index, value));
  IREE_RETURN_IF_ERROR(
      loom_op_set_attr(rewriter->module, op, (uint8_t)attr_index, value));
  IREE_RETURN_IF_ERROR(loom_rewriter_recompute_op_facts(rewriter, op));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_result_users_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_add_summary_ops_to_worklist(rewriter, op));
  rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

iree_status_t loom_rewriter_replace_attr_dict(
    loom_rewriter_t* rewriter, loom_op_t* op, uint16_t attr_index,
    loom_named_attr_update_slice_t updates) {
  if (attr_index >= op->attribute_count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "attribute index %u is out of range for op with %u attributes",
        (unsigned)attr_index, (unsigned)op->attribute_count);
  }

  const loom_op_vtable_t* vtable = loom_op_vtable(rewriter->module, op);
  if (!vtable || !vtable->attr_descriptors ||
      attr_index >= vtable->attribute_count ||
      vtable->attr_descriptors[attr_index].attr_kind != LOOM_ATTR_DICT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "attribute index %u is not a DICT attribute on this op",
        (unsigned)attr_index);
  }

  loom_attribute_t replacement = {0};
  IREE_RETURN_IF_ERROR(loom_module_replace_canonical_attr_dict(
      rewriter->module, loom_attr_as_dict(loom_op_attrs(op)[attr_index]),
      updates, &replacement));
  return loom_rewriter_set_attr(rewriter, op, attr_index, replacement);
}

iree_status_t loom_rewriter_set_instance_flags(loom_rewriter_t* rewriter,
                                               loom_op_t* op, uint8_t flags) {
  if (op->instance_flags == flags) {
    return iree_ok_status();
  }
  loom_trait_flags_t old_traits = op->traits;
  op->instance_flags = flags;
  loom_op_refresh_effective_traits(rewriter->module, op);
  loom_module_update_op_direct_summaries(rewriter->module, op, old_traits,
                                         op->traits);
  IREE_RETURN_IF_ERROR(loom_rewriter_recompute_op_facts(rewriter, op));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_result_users_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_add_summary_ops_to_worklist(rewriter, op));
  rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}
