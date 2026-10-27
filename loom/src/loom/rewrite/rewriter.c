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
#include "loom/ir/type_refs.h"
#include "loom/ir/value_refs.h"
#include "loom/ops/op_defs.h"
#include "loom/target/math_policy.h"
#include "loom/util/fact_cfg.h"
#include "loom/util/fact_loop.h"
#include "loom/util/fact_observation.h"
#include "loom/util/fact_vector_partition.h"

#define LOOM_REWRITER_INITIAL_WORKLIST_CAPACITY 64
#define LOOM_REWRITER_MAX_WORKLIST_CHUNK_SIZE (16 * 1024)
#define LOOM_REWRITER_INITIAL_REGION_STACK_CAPACITY 8

// Share the paged-value lookup across rewrite callbacks instead of duplicating
// it in every producer-matching pattern.
IREE_ATTRIBUTE_NOINLINE loom_op_t* loom_rewriter_value_defining_op(
    const loom_rewriter_t* rewriter, loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(rewriter->module, value_id);
  if (loom_value_is_block_arg(value)) {
    return NULL;
  }
  loom_op_t* definition = loom_value_def_op(value);
  return definition && !iree_any_bit_set(definition->flags, LOOM_OP_FLAG_DEAD)
             ? definition
             : NULL;
}

bool loom_rewriter_prefers_fma(const loom_rewriter_t* rewriter,
                               loom_type_t value_type, uint8_t fastmath_flags) {
  return loom_target_math_policy_prefers_fma(rewriter->math_policy,
                                             rewriter->math_target_bundle,
                                             value_type, fastmath_flags);
}

//===----------------------------------------------------------------------===//
// Builder callback
//===----------------------------------------------------------------------===//

static loom_block_t* loom_rewriter_definition_block(const loom_value_t* value) {
  if (loom_value_is_block_arg(value)) {
    return loom_value_def_block(value);
  }
  loom_op_t* op = loom_value_def_op(value);
  return op ? op->parent_block : NULL;
}

static iree_status_t loom_rewriter_notify_extent_provider(
    loom_rewriter_t* rewriter, loom_value_id_t provider,
    const loom_block_t* carrier_block) {
  if (!rewriter->fact_table || !carrier_block) {
    return iree_ok_status();
  }
  const loom_value_t* value = loom_module_value(rewriter->module, provider);
  if (loom_rewriter_definition_block(value) != carrier_block) {
    return iree_ok_status();
  }
  if (loom_value_is_block_arg(value)) {
    return loom_rewriter_add_fact_block(rewriter, loom_value_def_block(value));
  }
  return loom_rewriter_add_changed_fact_user(
      rewriter, loom_value_def_op(value),
      LOOM_REWRITER_USER_CHANGE_FLAG_SEMANTIC);
}

static iree_status_t loom_rewriter_notify_value_type_providers(
    loom_rewriter_t* rewriter, loom_value_id_t carrier) {
  if (!rewriter->fact_table || carrier == LOOM_VALUE_ID_INVALID ||
      !loom_type_is_shaped(loom_module_value_type(rewriter->module, carrier))) {
    return iree_ok_status();
  }
  const loom_block_t* block = loom_rewriter_definition_block(
      loom_module_value(rewriter->module, carrier));
  loom_type_use_iterator_t dependencies;
  loom_module_value_type_dependencies(rewriter->module, carrier, &dependencies);
  for (loom_value_id_t provider = loom_type_dependencies_next(&dependencies);
       provider != LOOM_VALUE_ID_INVALID;
       provider = loom_type_dependencies_next(&dependencies)) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_notify_extent_provider(rewriter, provider, block));
  }
  return iree_ok_status();
}

static iree_status_t loom_rewriter_notify_result_type_providers(
    loom_rewriter_t* rewriter, const loom_op_t* op) {
  if (!rewriter->fact_table) {
    return iree_ok_status();
  }
  for (uint16_t i = 0; i < op->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_notify_value_type_providers(
        rewriter, loom_op_const_results(op)[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_rewriter_on_argument_defined(
    void* user_data, loom_value_id_t argument) {
  loom_rewriter_t* rewriter = user_data;
  if (!rewriter->fact_table) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_rewriter_notify_value_type_providers(rewriter, argument));
  return loom_rewriter_add_fact_block(
      rewriter,
      loom_value_def_block(loom_module_value(rewriter->module, argument)));
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
  IREE_RETURN_IF_ERROR(
      loom_rewriter_notify_result_type_providers(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_seed_nested_execution_facts(rewriter, op));
  // Newly defined results have no users yet. Only a terminator changes an
  // existing control or payload equation before its results are connected.
  if (iree_any_bit_set(op->traits, LOOM_TRAIT_TERMINATOR)) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_summary_ops_to_worklist(rewriter, op));
  }
  if (rewriter->operation_creation.source != NULL &&
      rewriter->operation_creation.callback.fn != NULL) {
    IREE_RETURN_IF_ERROR(rewriter->operation_creation.callback.fn(
        rewriter->operation_creation.callback.user_data,
        rewriter->operation_creation.source, op));
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
iree_status_t loom_rewriter_record_subtree(loom_rewriter_t* rewriter,
                                           loom_op_t* op) {
  loom_module_record_op_summaries(rewriter->module, op);
  IREE_RETURN_IF_ERROR(loom_rewriter_seed_nested_execution_facts(rewriter, op));
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
  return rewriter->fact_table
             ? loom_rewriter_add_fact_user_to_worklist(rewriter, op)
             : iree_ok_status();
}

// Arena-backed worklist storage that grows without copying or abandoning prior
// entries. Chunks double from the historical 64-entry initial capacity up to a
// pool-sized 16 KiB ceiling, keeping both small and large rewrite runs dense.
struct loom_rewriter_worklist_chunk_t {
  // Older chunk containing the next entries after this chunk drains.
  loom_rewriter_worklist_chunk_t* previous;
  // Newer empty chunk retained for reuse after the worklist shrinks.
  loom_rewriter_worklist_chunk_t* next;
  // Number of pending operation pointers in entries.
  iree_host_size_t entry_count;
  // Number of operation pointers allocated in entries.
  iree_host_size_t entry_capacity;
  // LIFO operation pointers owned by this chunk.
  loom_op_t* entries[];
};

IREE_ATTRIBUTE_NOINLINE static iree_status_t
loom_rewriter_worklist_chunk_allocate(
    loom_rewriter_t* rewriter, iree_host_size_t desired_capacity,
    loom_rewriter_worklist_chunk_t* previous,
    loom_rewriter_worklist_chunk_t** out_chunk) {
  *out_chunk = NULL;
  const iree_host_size_t pooled_size =
      iree_arena_block_pool_max_allocation_size(rewriter->arena->block_pool);
  const iree_host_size_t chunk_size = iree_min(
      pooled_size, (iree_host_size_t)LOOM_REWRITER_MAX_WORKLIST_CHUNK_SIZE);
  const iree_host_size_t maximum_capacity =
      chunk_size > sizeof(loom_rewriter_worklist_chunk_t)
          ? (chunk_size - sizeof(loom_rewriter_worklist_chunk_t)) /
                sizeof(loom_op_t*)
          : 1;
  const iree_host_size_t entry_capacity =
      iree_max(iree_min(desired_capacity, maximum_capacity), 1);
  const iree_host_size_t allocation_size =
      sizeof(loom_rewriter_worklist_chunk_t) +
      entry_capacity * sizeof(loom_op_t*);
  loom_rewriter_worklist_chunk_t* chunk = NULL;
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(rewriter->arena, allocation_size, (void**)&chunk));
  *chunk = (loom_rewriter_worklist_chunk_t){
      .previous = previous,
      .entry_capacity = entry_capacity,
  };
  if (previous) {
    previous->next = chunk;
  }
  *out_chunk = chunk;
  return iree_ok_status();
}

// Advances from a full chunk without inflating the inlined queueing path with
// geometric capacity selection and allocation failure handling.
IREE_ATTRIBUTE_NOINLINE static iree_status_t loom_rewriter_worklist_advance(
    loom_rewriter_t* rewriter) {
  loom_rewriter_worklist_chunk_t* chunk = rewriter->worklist_chunk;
  if (chunk->next) {
    rewriter->worklist_chunk = chunk->next;
    return iree_ok_status();
  }
  const iree_host_size_t desired_capacity =
      iree_max(chunk->entry_capacity * 2,
               (iree_host_size_t)LOOM_REWRITER_INITIAL_WORKLIST_CAPACITY);
  return loom_rewriter_worklist_chunk_allocate(
      rewriter, desired_capacity, chunk, &rewriter->worklist_chunk);
}

//===----------------------------------------------------------------------===//
// Rewriter lifecycle
//===----------------------------------------------------------------------===//

void loom_rewriter_initialize(loom_rewriter_t* rewriter, loom_module_t* module,
                              iree_arena_allocator_t* arena) {
  memset(rewriter, 0, sizeof(*rewriter));
  rewriter->module = module;
  rewriter->arena = arena;
  rewriter->name_policy = LOOM_REWRITER_NAME_POLICY_DEFAULT;

  // Initialize the builder with the module's arena (new ops live in
  // the module, not the pass scratch arena).
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &rewriter->builder);

  // Record effects and facts independently of optional worklist scheduling.
  rewriter->builder.on_op_finalized.fn = loom_rewriter_on_op_finalized;
  rewriter->builder.on_op_finalized.user_data = rewriter;
  rewriter->builder.on_argument_defined = (loom_builder_argument_callback_t){
      .fn = loom_rewriter_on_argument_defined, .user_data = rewriter};
}

void loom_rewriter_deinitialize(loom_rewriter_t* rewriter) {
  loom_rewriter_attach_value_facts(rewriter, NULL);
  // Clear ON_WORKLIST bits on any remaining ops. Start at the oldest chunk so
  // emptied newer chunks retained for reuse are harmless.
  loom_rewriter_worklist_chunk_t* chunk = rewriter->worklist_chunk;
  while (chunk && chunk->previous) {
    chunk = chunk->previous;
  }
  for (; chunk; chunk = chunk->next) {
    for (iree_host_size_t i = 0; i < chunk->entry_count; ++i) {
      chunk->entries[i]->flags &= ~LOOM_OP_FLAG_ON_WORKLIST;
    }
  }
  // Worklist chunks are arena-allocated and released with the pass arena.
  memset(rewriter, 0, sizeof(*rewriter));
}

void loom_rewriter_set_operation_creation_callback(
    loom_rewriter_t* rewriter,
    loom_rewriter_operation_creation_callback_t callback) {
  rewriter->operation_creation.callback = callback;
}

void loom_rewriter_set_operation_creation_source(loom_rewriter_t* rewriter,
                                                 const loom_op_t* source_op) {
  rewriter->operation_creation.source = source_op;
}

iree_status_t loom_rewriter_enable_worklist(loom_rewriter_t* rewriter) {
  if (rewriter->worklist_chunk) {
    return iree_ok_status();
  }
  return loom_rewriter_worklist_chunk_allocate(
      rewriter, LOOM_REWRITER_INITIAL_WORKLIST_CAPACITY, /*previous=*/NULL,
      &rewriter->worklist_chunk);
}

iree_status_t loom_rewriter_seed_region(loom_rewriter_t* rewriter,
                                        loom_region_t* region) {
  if (!region) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_enable_worklist(rewriter));

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

iree_status_t loom_rewriter_build_constant(loom_rewriter_t* rewriter,
                                           loom_value_facts_t facts,
                                           loom_type_t result_type,
                                           loom_location_id_t location,
                                           loom_value_id_t* out_value_id) {
  if (!loom_rewriter_has_constant_materializer(rewriter)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "no constant materializer policy set");
  }
  return rewriter->constant_materializer.materialize(
      &rewriter->builder, facts, result_type, location, out_value_id);
}

static bool loom_rewriter_can_fold(const loom_rewriter_t* rewriter,
                                   const loom_op_t* op) {
  if (!rewriter->fact_table) {
    return false;
  }
#ifndef NDEBUG
  const loom_value_fact_observation_t observation =
      loom_value_fact_observation_capture(rewriter->fact_table);
  loom_value_fact_observation_assert_valid(observation);
#endif
  const loom_op_vtable_t* vtable = loom_op_vtable(rewriter->module, op);
  if (!vtable || iree_any_bit_set(op->traits, LOOM_TRAIT_CONSTANT_LIKE)) {
    return false;
  }

  // Cannot materialize without a complete policy, and don't replace
  // constant-like ops with themselves.
  if (!loom_rewriter_has_constant_materializer(rewriter)) {
    return false;
  }
  if (!vtable->infer_facts) {
    return false;
  }
  const loom_trait_flags_t traits =
      loom_op_effective_traits(rewriter->module, op);
  if (loom_traits_has_side_effects(traits)) {
    return false;
  }

  // Preflight the complete replacement family. Exact facts do not imply that
  // the selected dialect policy can represent the value as a constant.
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) {
      return false;
    }
    loom_value_facts_t facts =
        loom_value_fact_table_lookup(rewriter->fact_table, results[i]);
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, results[i]);
    if (!loom_value_facts_is_exact(facts) ||
        !loom_rewriter_can_materialize_constant(rewriter, facts, result_type)) {
      return false;
    }
  }
  if (op->result_count == 0) {
    return false;
  }

  return true;
}

// A prepared batch supplies retained exact facts. The immediate fold reads
// its coherent results directly, avoiding batch storage for a single rewrite.
static iree_status_t loom_rewriter_fold_results(
    loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_value_facts_t* exact_facts) {
#ifndef NDEBUG
  const loom_value_fact_observation_t observation =
      loom_value_fact_observation_capture(rewriter->fact_table);
#endif
  const loom_value_id_t* results = loom_op_const_results(op);
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
#ifndef NDEBUG
    if (!exact_facts) {
      loom_value_fact_observation_assert_valid(observation);
    }
#endif
    loom_value_facts_t facts =
        exact_facts
            ? exact_facts[i]
            : loom_value_fact_table_lookup(rewriter->fact_table, results[i]);
    loom_type_t result_type =
        loom_module_value_type(rewriter->module, results[i]);
    IREE_RETURN_IF_ERROR(loom_rewriter_build_constant(
        rewriter, facts, result_type, op->location, &replacement_ids[i]));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, op, replacement_ids, op->result_count, value_checkpoint));

  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, op, replacement_ids, op->result_count));

  return iree_ok_status();
}

iree_status_t loom_rewriter_try_fold(loom_rewriter_t* rewriter, loom_op_t* op,
                                     bool* out_folded) {
  *out_folded = false;
  IREE_RETURN_IF_ERROR(loom_rewriter_propagate_value_facts(rewriter));
  if (!loom_rewriter_can_fold(rewriter, op)) {
    return iree_ok_status();
  }
#ifndef NDEBUG
  const loom_value_fact_observation_t observation =
      loom_value_fact_observation_capture(rewriter->fact_table);
#endif
  if (loom_traits_are_fact_identity(op->traits) &&
      rewriter->pending_exact_relations_callback.fn &&
      loom_value_fact_table_has_pending_exact_relations(rewriter->fact_table)) {
    IREE_RETURN_IF_ERROR(rewriter->pending_exact_relations_callback.fn(
        rewriter->pending_exact_relations_callback.user_data, rewriter));
  }
#ifndef NDEBUG
  loom_value_fact_observation_assert_valid(observation);
#endif
  IREE_RETURN_IF_ERROR(loom_rewriter_fold_results(rewriter, op, NULL));
  *out_folded = true;
  return iree_ok_status();
}

typedef struct loom_rewrite_fold_action_t {
  // Root-local application owned by the shared batch.
  loom_rewrite_action_t action;
  // Exact constant payload for every result; no captured SSA references.
  loom_value_facts_t facts[];
} loom_rewrite_fold_action_t;

static iree_status_t loom_rewriter_apply_fold(
    const loom_rewrite_action_t* action, loom_rewriter_t* rewriter) {
  // Folding a consumer can make another planned root dead. Erase that root
  // directly instead of allocating constants that the worklist would discard.
  if (loom_rewriter_is_trivially_dead(rewriter, action->root)) {
    return loom_rewriter_erase(rewriter, action->root);
  }
  const loom_rewrite_fold_action_t* fold =
      (const loom_rewrite_fold_action_t*)action;
  return loom_rewriter_fold_results(rewriter, action->root, fold->facts);
}

iree_status_t loom_rewriter_prepare_fold(const loom_rewriter_t* rewriter,
                                         const loom_op_t* op,
                                         iree_arena_allocator_t* arena,
                                         loom_rewrite_action_t** out_action) {
  *out_action = NULL;
  if (op->region_count || !loom_rewriter_can_fold(rewriter, op)) {
    return iree_ok_status();
  }
  if (loom_traits_are_fact_identity(op->traits) &&
      loom_value_fact_table_has_pending_exact_relations(rewriter->fact_table)) {
    return iree_ok_status();
  }
  loom_rewrite_fold_action_t* fold = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate(
      arena, sizeof(*fold) + op->result_count * sizeof(fold->facts[0]),
      (void**)&fold));
  fold->action.apply = loom_rewriter_apply_fold;
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    fold->facts[i] =
        loom_value_fact_table_lookup(rewriter->fact_table, results[i]);
  }
  *out_action = &fold->action;
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

// SSA identities and materializations borrow live definitions, unlike facts
// that remain meaningful after their producer is removed. The erasing mutation
// owns withdrawal for the complete removed subtree.
static void loom_rewriter_erase_value_facts(loom_rewriter_t* rewriter,
                                            loom_op_t* op) {
  if (!rewriter->fact_table || (!rewriter->fact_table->identities.nodes &&
                                !rewriter->fact_table->vector_partitions)) {
    return;
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_value_fact_table_erase_identity_value(rewriter->fact_table,
                                               results[i]);
    loom_value_fact_table_invalidate_vector_partition_value(
        rewriter->fact_table, results[i]);
  }
  for (uint8_t i = 0; i < op->region_count; ++i) {
    loom_region_t* region = loom_op_regions(op)[i];
    if (!region) {
      continue;
    }
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      for (uint16_t j = 0; j < block->arg_count; ++j) {
        loom_value_fact_table_erase_identity_value(rewriter->fact_table,
                                                   block->arg_ids[j]);
        loom_value_fact_table_invalidate_vector_partition_value(
            rewriter->fact_table, block->arg_ids[j]);
      }
      for (loom_op_t* child = block->first_op; child; child = child->next_op) {
        loom_rewriter_erase_value_facts(rewriter, child);
      }
    }
  }
}

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
  if (!rewriter->worklist_chunk ||
      (op->flags & (LOOM_OP_FLAG_DEAD | LOOM_OP_FLAG_ON_WORKLIST))) {
    return iree_ok_status();
  }
  op->flags |= LOOM_OP_FLAG_ON_WORKLIST;

  loom_rewriter_worklist_chunk_t* chunk = rewriter->worklist_chunk;
  if (IREE_UNLIKELY(chunk->entry_count >= chunk->entry_capacity)) {
    iree_status_t status = loom_rewriter_worklist_advance(rewriter);
    if (!iree_status_is_ok(status)) {
      op->flags &= ~LOOM_OP_FLAG_ON_WORKLIST;
      return status;
    }
    chunk = rewriter->worklist_chunk;
  }

  chunk->entries[chunk->entry_count++] = op;
  return iree_ok_status();
}

loom_op_t* loom_rewriter_pop(loom_rewriter_t* rewriter) {
  loom_rewriter_worklist_chunk_t* chunk = rewriter->worklist_chunk;
  while (chunk) {
    if (chunk->entry_count == 0) {
      if (!chunk->previous) {
        return NULL;
      }
      chunk = chunk->previous;
      rewriter->worklist_chunk = chunk;
      continue;
    }
    loom_op_t* op = chunk->entries[--chunk->entry_count];
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
    loom_value_id_t value_id, loom_value_id_t carrier, void* user_data) {
  loom_rewriter_t* rewriter = user_data;
  const loom_value_t* value = loom_module_value(rewriter->module, value_id);
  if (!loom_value_is_block_arg(value) && loom_value_def_op(value)) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_to_worklist(rewriter, loom_value_def_op(value)));
  }
  if (carrier != LOOM_VALUE_ID_INVALID &&
      loom_type_is_shaped(loom_module_value_type(rewriter->module, carrier))) {
    return loom_rewriter_notify_extent_provider(
        rewriter, value_id,
        loom_rewriter_definition_block(
            loom_module_value(rewriter->module, carrier)));
  }
  return iree_ok_status();
}

typedef struct loom_rewriter_type_provider_context_t {
  // Borrowed active mutation owner.
  loom_rewriter_t* rewriter;
  // Same-block extent rule applies only to a shaped type carrier.
  const loom_block_t* carrier_block;
} loom_rewriter_type_provider_context_t;

static iree_status_t loom_rewriter_notify_type_provider(
    loom_value_id_t value_id, void* user_data) {
  loom_rewriter_type_provider_context_t* context = user_data;
  IREE_RETURN_IF_ERROR(loom_rewriter_add_value_ref_provider_to_worklist(
      value_id, LOOM_VALUE_ID_INVALID, context->rewriter));
  return loom_rewriter_notify_extent_provider(context->rewriter, value_id,
                                              context->carrier_block);
}

static iree_status_t loom_rewriter_add_type_ref_providers_to_worklist(
    loom_rewriter_t* rewriter, loom_value_id_t carrier, loom_type_t type) {
  if (!rewriter->worklist_chunk && !rewriter->fact_table) {
    return iree_ok_status();
  }
  loom_rewriter_type_provider_context_t context = {
      .rewriter = rewriter,
      .carrier_block = loom_type_is_shaped(type)
                           ? loom_rewriter_definition_block(
                                 loom_module_value(rewriter->module, carrier))
                           : NULL,
  };
  return loom_type_walk_value_refs(
      rewriter->module, type, loom_rewriter_notify_type_provider, &context);
}

static iree_status_t loom_rewriter_add_subtree_providers_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  if (!rewriter->worklist_chunk && !rewriter->fact_table) {
    return iree_ok_status();
  }
  return loom_op_walk_subtree_value_refs(
      rewriter->module, op, loom_rewriter_add_value_ref_provider_to_worklist,
      rewriter);
}

static iree_status_t loom_rewriter_add_operand_users_except_to_worklist(
    loom_rewriter_t* rewriter, loom_value_id_t value_id,
    const loom_op_t* except_op) {
  const loom_rewriter_user_change_flags_t flags =
      LOOM_REWRITER_USER_CHANGE_FLAG_REPLACED;
  if (!rewriter->worklist_chunk && !rewriter->fact_table) {
    return iree_ok_status();
  }
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    loom_op_t* user_op = loom_use_user_op(uses[i]);
    if (user_op == except_op) {
      continue;
    }
    loom_rewriter_invalidate_cfg_forwarding(rewriter, user_op);
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_changed_fact_user(rewriter, user_op, flags));
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
  if (!loom_rewriter_is_trivially_dead(rewriter, op)) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, op));
  }
  loom_rewriter_erase_value_facts(rewriter, op);
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
  if (rewriter->fact_table) {
    loom_type_use_iterator_t carriers;
    loom_module_value_type_users(rewriter->module, old_value, &carriers);
    for (loom_value_id_t carrier = loom_type_users_next(&carriers);
         carrier != LOOM_VALUE_ID_INVALID;
         carrier = loom_type_users_next(&carriers)) {
      if (!loom_type_is_shaped(
              loom_module_value_type(rewriter->module, carrier))) {
        continue;
      }
      const loom_block_t* block = loom_rewriter_definition_block(
          loom_module_value(rewriter->module, carrier));
      IREE_RETURN_IF_ERROR(
          loom_rewriter_notify_extent_provider(rewriter, old_value, block));
      IREE_RETURN_IF_ERROR(
          loom_rewriter_notify_extent_provider(rewriter, new_value, block));
    }
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_add_users_to_worklist(
      rewriter, old_value, LOOM_REWRITER_USER_CHANGE_FLAG_REPLACED));
  IREE_RETURN_IF_ERROR(
      loom_value_replace_all_uses_with(rewriter->module, old_value, new_value));
  if (rewriter->fact_table) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_replace_identity_value(
        rewriter->fact_table, old_value, new_value));
  }
  if (rewriter->fact_table && rewriter->fact_table->vector_partitions) {
    if (loom_type_equal(loom_module_value_type(rewriter->module, old_value),
                        loom_module_value_type(rewriter->module, new_value))) {
      IREE_RETURN_IF_ERROR(loom_value_fact_table_replace_vector_partition_value(
          rewriter->fact_table, old_value, new_value));
    } else {
      loom_value_fact_table_invalidate_vector_partition_value(
          rewriter->fact_table, old_value);
    }
  }
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
  if (!loom_rewriter_is_trivially_dead(rewriter, op)) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, op));
  }
  loom_rewriter_erase_value_facts(rewriter, op);
  IREE_RETURN_IF_ERROR(loom_op_erase(rewriter->module, op));
  ++rewriter->erased_op_count;
  rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

iree_status_t loom_rewriter_erase_closed_set(loom_rewriter_t* rewriter,
                                             loom_op_t* const* ops,
                                             iree_host_size_t count) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_subtree_providers_to_worklist(rewriter, ops[i]));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, ops[i]));
    loom_rewriter_erase_value_facts(rewriter, ops[i]);
  }
  loom_op_erase_closed_set(rewriter->module, ops, count);
  rewriter->erased_op_count += count;
  if (count) {
    rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  }
  return iree_ok_status();
}

iree_status_t loom_rewriter_remove_blocks(loom_rewriter_t* rewriter,
                                          loom_region_t* region,
                                          const bool* remove_blocks,
                                          uint16_t remove_block_count,
                                          iree_arena_allocator_t* scratch_arena,
                                          uint16_t* out_removed_count) {
  if (rewriter->fact_table) {
    for (uint16_t i = 0; i < remove_block_count; ++i) {
      if (!remove_blocks[i]) {
        continue;
      }
      loom_block_t* block = region->blocks[i];
      for (uint16_t j = 0; j < block->arg_count; ++j) {
        loom_value_fact_table_erase_identity_value(rewriter->fact_table,
                                                   block->arg_ids[j]);
        loom_value_fact_table_invalidate_vector_partition_value(
            rewriter->fact_table, block->arg_ids[j]);
      }
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        loom_rewriter_erase_value_facts(rewriter, op);
      }
    }
  }
  IREE_RETURN_IF_ERROR(loom_region_remove_blocks(
      rewriter->module, region, remove_blocks, remove_block_count,
      scratch_arena, out_removed_count));
  if (*out_removed_count) {
    rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  }
  return iree_ok_status();
}

iree_status_t loom_rewriter_remove_block_args(loom_rewriter_t* rewriter,
                                              loom_block_t* block,
                                              const bool* remove_args,
                                              uint16_t remove_arg_count,
                                              uint16_t* out_removed_count) {
  *out_removed_count = 0;
  for (uint16_t i = 0; i < remove_arg_count; ++i) {
    if (remove_args[i]) {
      const loom_value_id_t value = loom_block_arg_id(block, i);
      IREE_RETURN_IF_ERROR(loom_rewriter_add_type_ref_providers_to_worklist(
          rewriter, value, loom_module_value_type(rewriter->module, value)));
    }
  }
  if (rewriter->fact_table) {
    for (uint16_t i = 0; i < remove_arg_count; ++i) {
      if (remove_args[i]) {
        loom_value_fact_table_erase_identity_value(rewriter->fact_table,
                                                   block->arg_ids[i]);
        loom_value_fact_table_invalidate_vector_partition_value(
            rewriter->fact_table, block->arg_ids[i]);
      }
    }
  }
  const uint16_t removed = loom_block_remove_args(
      rewriter->module, block, remove_args, remove_arg_count);
  if (removed) {
    rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  }
  *out_removed_count = removed;
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

  if (rewriter->worklist_chunk) {
    for (uint16_t block_index = 0; block_index < source_region->block_count;
         ++block_index) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(source_region->blocks[block_index], op) {
        IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, op));
      }
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
  IREE_RETURN_IF_ERROR(loom_rewriter_relocate_fact_block(
      rewriter, old_entry_block, moved_entry_block));
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

  IREE_RETURN_IF_ERROR(
      loom_rewriter_notify_result_type_providers(rewriter, op));
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
  IREE_RETURN_IF_ERROR(
      loom_rewriter_notify_result_type_providers(rewriter, op));

  IREE_RETURN_IF_ERROR(loom_rewriter_add_changed_fact_user(
      rewriter, op, LOOM_REWRITER_USER_CHANGE_FLAG_SEMANTIC));
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

  IREE_RETURN_IF_ERROR(
      loom_rewriter_notify_result_type_providers(rewriter, op));
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
  IREE_RETURN_IF_ERROR(
      loom_rewriter_notify_result_type_providers(rewriter, op));

  IREE_RETURN_IF_ERROR(loom_rewriter_add_changed_fact_user(
      rewriter, op, LOOM_REWRITER_USER_CHANGE_FLAG_SEMANTIC));
  IREE_RETURN_IF_ERROR(loom_rewriter_add_summary_ops_to_worklist(rewriter, op));
  rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

// Publishes one complete operand-tuple mutation to the analyses and worklist.
static iree_status_t loom_rewriter_finish_operand_change(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  loom_rewriter_invalidate_cfg_forwarding(rewriter, op);
  IREE_RETURN_IF_ERROR(loom_rewriter_add_fact_user_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_add_summary_ops_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_recompute_op_facts(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_add_result_users_to_worklist(
      rewriter, op, LOOM_REWRITER_USER_CHANGE_FLAG_SEMANTIC));
  rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

iree_status_t loom_rewriter_set_operand(loom_rewriter_t* rewriter,
                                        loom_op_t* op, uint16_t operand_index,
                                        loom_value_id_t new_value) {
  if (loom_op_operands(op)[operand_index] == new_value) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(
      loom_op_set_operand(rewriter->module, op, operand_index, new_value));
  return loom_rewriter_finish_operand_change(rewriter, op);
}

iree_status_t loom_rewriter_set_operands(loom_rewriter_t* rewriter,
                                         loom_op_t* op,
                                         const loom_value_id_t* new_values) {
  loom_value_id_t* operands = loom_op_operands(op);
  bool changed = false;
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (operands[i] == new_values[i]) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_op_set_operand(rewriter->module, op, i, new_values[i]));
    changed = true;
  }
  return changed ? loom_rewriter_finish_operand_change(rewriter, op)
                 : iree_ok_status();
}

iree_status_t loom_rewriter_set_value_type(loom_rewriter_t* rewriter,
                                           loom_value_id_t value_id,
                                           loom_type_t new_type) {
  loom_type_t old_type = loom_module_value_type(rewriter->module, value_id);
  IREE_RETURN_IF_ERROR(loom_rewriter_add_type_ref_providers_to_worklist(
      rewriter, value_id, old_type));
  IREE_RETURN_IF_ERROR(loom_rewriter_add_type_ref_providers_to_worklist(
      rewriter, value_id, new_type));
  IREE_RETURN_IF_ERROR(
      loom_module_set_value_type(rewriter->module, value_id, new_type));
  if (rewriter->fact_table && !loom_type_equal(old_type, new_type)) {
    loom_value_fact_table_invalidate_vector_partition_value(
        rewriter->fact_table, value_id);
  }
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  if (!loom_value_is_block_arg(value)) {
    loom_op_t* defining_op = loom_value_def_op(value);
    if (defining_op) {
      IREE_RETURN_IF_ERROR(
          loom_rewriter_add_fact_user_to_worklist(rewriter, defining_op));
      IREE_RETURN_IF_ERROR(
          loom_rewriter_add_summary_ops_to_worklist(rewriter, defining_op));
      if (iree_any_bit_set(defining_op->traits, LOOM_TRAIT_CONSTANT_LIKE)) {
        IREE_RETURN_IF_ERROR(
            loom_rewriter_recompute_op_facts(rewriter, defining_op));
      }
    }
  } else if (rewriter->fact_table) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_fact_block(rewriter, loom_value_def_block(value)));
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_add_users_to_worklist(
      rewriter, value_id, LOOM_REWRITER_USER_CHANGE_FLAG_SEMANTIC));
  rewriter->flags |=
      LOOM_REWRITER_FLAG_CHANGED | LOOM_REWRITER_FLAG_TYPE_CHANGED;
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
  IREE_RETURN_IF_ERROR(loom_rewriter_add_fact_user_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_recompute_op_facts(rewriter, op));
  if (rewriter->fact_table && op->region_count) {
    const loom_op_vtable_t* vtable = loom_op_vtable(rewriter->module, op);
    for (uint8_t i = 0; i < op->region_count; ++i) {
      const loom_region_descriptor_t* descriptor =
          loom_op_vtable_region_descriptor(vtable, i);
      if (descriptor &&
          descriptor->execution_target_attr_index_plus_one == attr_index + 1) {
        // Target binding can change every fact in the region. Publish the new
        // facts before any queued user can fold against the old environment.
        IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_region_tree(
            rewriter->fact_table, rewriter->module, loom_op_regions(op)[i],
            op));
        IREE_RETURN_IF_ERROR(loom_rewriter_record_subtree(rewriter, op));
      }
    }
  }
  IREE_RETURN_IF_ERROR(loom_rewriter_add_result_users_to_worklist(
      rewriter, op, LOOM_REWRITER_USER_CHANGE_FLAG_SEMANTIC));
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
  IREE_RETURN_IF_ERROR(loom_rewriter_add_fact_user_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_recompute_op_facts(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_add_result_users_to_worklist(
      rewriter, op, LOOM_REWRITER_USER_CHANGE_FLAG_SEMANTIC));
  IREE_RETURN_IF_ERROR(loom_rewriter_add_summary_ops_to_worklist(rewriter, op));
  rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}
