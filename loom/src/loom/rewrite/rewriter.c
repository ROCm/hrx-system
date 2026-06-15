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
#include "loom/ops/op_defs.h"

#define LOOM_REWRITER_INITIAL_WORKLIST_CAPACITY 64
#define LOOM_REWRITER_INITIAL_REGION_STACK_CAPACITY 8

static iree_status_t loom_rewriter_add_result_users_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op);
static iree_status_t loom_rewriter_add_parent_summary_ops_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op);

typedef enum loom_rewriter_fact_recompute_flag_bits_e {
  LOOM_REWRITER_FACT_RECOMPUTE_FLAG_ENQUEUE_RESULT_USERS = 1u << 0,
} loom_rewriter_fact_recompute_flag_bits_t;
typedef uint32_t loom_rewriter_fact_recompute_flags_t;

static iree_status_t loom_rewriter_recompute_op_facts(
    loom_rewriter_t* rewriter, loom_op_t* op,
    loom_rewriter_fact_recompute_flags_t flags) {
  if (!rewriter->fact_table) return iree_ok_status();

  bool facts_changed = false;
  IREE_RETURN_IF_ERROR(loom_value_fact_table_compute_op_and_report(
      rewriter->fact_table, rewriter->module, op, &facts_changed));
  if (!facts_changed) return iree_ok_status();

  rewriter->flags |= LOOM_REWRITER_FLAG_FACTS_CHANGED;
  if (iree_any_bit_set(
          flags, LOOM_REWRITER_FACT_RECOMPUTE_FLAG_ENQUEUE_RESULT_USERS)) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_result_users_to_worklist(rewriter, op));
  }
  return loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, op);
}

//===----------------------------------------------------------------------===//
// Builder callback
//===----------------------------------------------------------------------===//

// Callback installed on the builder. Fired by finalize_op after a new
// op is fully wired. Adds the op to the rewriter's worklist so the
// driver can attempt patterns on it. When analysis is enabled, also
// computes facts for the new op's results.
static iree_status_t loom_rewriter_on_op_finalized(void* user_data,
                                                   loom_op_t* op) {
  loom_rewriter_t* rewriter = (loom_rewriter_t*)user_data;
  ++rewriter->created_op_count;
  IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, op));
  return loom_rewriter_recompute_op_facts(rewriter, op, /*flags=*/0);
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
  if (!region) return iree_ok_status();

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
        if (op->region_count == 0) continue;
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

iree_status_t loom_rewriter_enable_region_analysis(
    loom_rewriter_t* rewriter, loom_func_like_t function, loom_region_t* region,
    loom_op_t* parent_op, loom_value_fact_table_t* facts) {
  return loom_rewriter_enable_region_analysis_with_seed_facts(
      rewriter, function, region, parent_op, facts, NULL);
}

iree_status_t loom_rewriter_enable_region_analysis_with_seed_facts(
    loom_rewriter_t* rewriter, loom_func_like_t function, loom_region_t* region,
    loom_op_t* parent_op, loom_value_fact_table_t* facts,
    const loom_value_fact_table_t* seed_facts) {
  rewriter->fact_table = facts;
  if (seed_facts) {
    IREE_RETURN_IF_ERROR(loom_value_fact_table_clone_defined_facts(
        rewriter->fact_table, seed_facts, rewriter->module));
  }
  return loom_value_fact_table_compute_region(
      rewriter->fact_table, rewriter->module, function, region, parent_op);
}

iree_status_t loom_rewriter_enable_analysis(loom_rewriter_t* rewriter,
                                            loom_func_like_t function,
                                            loom_value_fact_table_t* facts) {
  return loom_rewriter_enable_analysis_with_seed_facts(rewriter, function,
                                                       facts, NULL);
}

iree_status_t loom_rewriter_enable_analysis_with_seed_facts(
    loom_rewriter_t* rewriter, loom_func_like_t function,
    loom_value_fact_table_t* facts, const loom_value_fact_table_t* seed_facts) {
  return loom_rewriter_enable_region_analysis_with_seed_facts(
      rewriter, function, loom_func_like_body(function), function.op, facts,
      seed_facts);
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
  if (!rewriter->fact_table) return iree_ok_status();
  const loom_op_vtable_t* vtable = loom_op_vtable(rewriter->module, op);
  if (!vtable || (!vtable->infer_facts && !vtable->loop_like)) {
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
        loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, op));
  }

  // Cannot materialize without a callback, and don't replace
  // constant-like ops with themselves.
  if (!rewriter->materialize_constant) return iree_ok_status();
  if (!vtable->infer_facts) return iree_ok_status();
  if (vtable->traits & LOOM_TRAIT_CONSTANT_LIKE) return iree_ok_status();

  // Check if all results are now exact constants.
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) return iree_ok_status();
    loom_value_facts_t facts =
        loom_value_fact_table_lookup(rewriter->fact_table, results[i]);
    if (!loom_value_facts_is_exact(facts)) return iree_ok_status();
  }
  if (op->result_count == 0) return iree_ok_status();

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
    if (replacement < value_checkpoint) continue;
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
  if (!loom_rewriter_is_trivially_dead(rewriter, op)) return iree_ok_status();
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

static iree_status_t loom_rewriter_add_parent_summary_ops_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  for (loom_op_t* parent = op ? op->parent_op : NULL; parent;
       parent = parent->parent_op) {
    const loom_op_vtable_t* vtable = loom_op_vtable(rewriter->module, parent);
    if (vtable && vtable->loop_like) {
      IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, parent));
    }
  }
  return iree_ok_status();
}

loom_op_t* loom_rewriter_pop(loom_rewriter_t* rewriter) {
  while (rewriter->worklist_count > 0) {
    loom_op_t* op = rewriter->worklist[--rewriter->worklist_count];
    op->flags &= ~LOOM_OP_FLAG_ON_WORKLIST;
    // Skip ops that were erased while on the worklist.
    if (op->flags & LOOM_OP_FLAG_DEAD) continue;
    return op;
  }
  return NULL;
}

//===----------------------------------------------------------------------===//
// Safe IR mutation
//===----------------------------------------------------------------------===//

// Adds operand defining ops to the worklist so cascading DCE can
// catch them if they become dead after the op using them is erased.
// Must be called BEFORE the op is erased (while operands are valid).
static iree_status_t loom_rewriter_add_operand_providers_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (operands[i] == LOOM_VALUE_ID_INVALID) continue;
    loom_value_t* value = loom_module_value(rewriter->module, operands[i]);
    if (loom_value_is_block_arg(value)) continue;
    loom_op_t* def = loom_value_def_op(value);
    if (def) {
      IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, def));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_rewriter_add_type_ref_provider_to_worklist(
    loom_value_id_t value_id, void* user_data) {
  loom_rewriter_t* rewriter = (loom_rewriter_t*)user_data;
  if (value_id >= rewriter->module->values.count) return iree_ok_status();
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  if (loom_value_is_block_arg(value)) return iree_ok_status();
  loom_op_t* def = loom_value_def_op(value);
  if (!def) return iree_ok_status();
  return loom_rewriter_add_to_worklist(rewriter, def);
}

static iree_status_t loom_rewriter_add_type_ref_providers_to_worklist(
    loom_rewriter_t* rewriter, loom_type_t type) {
  return loom_type_walk_value_refs(
      type, loom_rewriter_add_type_ref_provider_to_worklist, rewriter);
}

static iree_status_t loom_rewriter_add_subtree_operand_providers_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_operand_providers_to_worklist(rewriter, op));

  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    loom_region_t* region = regions[i];
    if (!region) continue;
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        IREE_RETURN_IF_ERROR(
            loom_rewriter_add_subtree_operand_providers_to_worklist(rewriter,
                                                                    child_op));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_rewriter_add_subtree_providers_to_worklist(
    loom_rewriter_t* rewriter, loom_op_t* op) {
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_subtree_operand_providers_to_worklist(rewriter, op));
  return loom_op_walk_subtree_type_refs(
      rewriter->module, op, loom_rewriter_add_type_ref_provider_to_worklist,
      rewriter);
}

// Adds all users of a value to the worklist.
static iree_status_t loom_rewriter_add_users_to_worklist(
    loom_rewriter_t* rewriter, loom_value_id_t value_id) {
  loom_value_t* value = loom_module_value(rewriter->module, value_id);
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    loom_op_t* user_op = loom_use_user_op(uses[i]);
    IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, user_op));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, user_op));
  }
  if (value_id >= rewriter->module->type_uses.value_capacity) {
    return iree_ok_status();
  }
  loom_type_use_id_t use_id =
      rewriter->module->type_uses.value_heads[value_id].first_incoming_use_id;
  while (use_id != LOOM_TYPE_USE_ID_INVALID) {
    const loom_type_use_t* type_use =
        &rewriter->module->type_uses.records[use_id];
    loom_value_t* user_value =
        loom_module_value(rewriter->module, type_use->user_value_id);
    if (!loom_value_is_block_arg(user_value)) {
      loom_op_t* def = loom_value_def_op(user_value);
      if (def) {
        IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, def));
        IREE_RETURN_IF_ERROR(
            loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, def));
      }
    }
    const loom_use_t* user_value_uses = loom_value_uses(user_value);
    for (uint32_t i = 0; i < user_value->use_count; ++i) {
      loom_op_t* user_op = loom_use_user_op(user_value_uses[i]);
      IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, user_op));
      IREE_RETURN_IF_ERROR(
          loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, user_op));
    }
    use_id = type_use->next_incoming_use_id;
  }
  return iree_ok_status();
}

iree_status_t loom_rewriter_replace_all_uses_and_erase(
    loom_rewriter_t* rewriter, loom_op_t* op,
    const loom_value_id_t* replacements, uint16_t count) {
  loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) continue;
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
  if (old_value == new_value) return iree_ok_status();
  if (old_value == LOOM_VALUE_ID_INVALID ||
      new_value == LOOM_VALUE_ID_INVALID) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replacement values must be valid");
  }
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_users_to_worklist(rewriter, old_value));
  IREE_RETURN_IF_ERROR(
      loom_value_replace_all_uses_with(rewriter->module, old_value, new_value));
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
    if (current == ancestor) return true;
  }
  return false;
}

static void loom_rewriter_record_subtree_effects(loom_module_t* module,
                                                 loom_op_t* op) {
  loom_module_record_op_effects(module, op);
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_region_t* region = regions[region_index];
    if (!region) continue;
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        loom_rewriter_record_subtree_effects(module, child_op);
      }
    }
  }
}

iree_status_t loom_rewriter_move_before(loom_rewriter_t* rewriter,
                                        loom_op_t* op, loom_op_t* before_op) {
  if (!op || !before_op) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "move requires live op and insertion target");
  }
  if (op == before_op || op->next_op == before_op) return iree_ok_status();
  if (iree_any_bit_set(op->flags | before_op->flags, LOOM_OP_FLAG_DEAD) ||
      !op->parent_block || !before_op->parent_block) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot move dead or unlinked operation");
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
      loom_rewriter_record_subtree_effects(module, op);
    }
    return iree_status_join(status, restore_status);
  }
  loom_rewriter_record_subtree_effects(module, op);

  IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, before_op));
  rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}

iree_status_t loom_rewriter_set_operand(loom_rewriter_t* rewriter,
                                        loom_op_t* op, uint16_t operand_index,
                                        loom_value_id_t new_value) {
  IREE_RETURN_IF_ERROR(
      loom_op_set_operand(rewriter->module, op, operand_index, new_value));
  IREE_RETURN_IF_ERROR(loom_rewriter_add_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(loom_rewriter_recompute_op_facts(
      rewriter, op, LOOM_REWRITER_FACT_RECOMPUTE_FLAG_ENQUEUE_RESULT_USERS));
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
  IREE_RETURN_IF_ERROR(loom_rewriter_add_users_to_worklist(rewriter, value_id));
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
      IREE_RETURN_IF_ERROR(
          loom_rewriter_add_users_to_worklist(rewriter, results[i]));
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
  if (value.kind != descriptor->attr_kind) {
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
  loom_trait_flags_t old_traits = op->traits;
  loom_op_attrs(op)[attr_index] = value;
  IREE_RETURN_IF_ERROR(
      loom_module_note_op_attribute_value_refs(rewriter->module, op));
  loom_op_refresh_effective_traits(rewriter->module, op);
  loom_module_update_op_direct_effects(op, old_traits, op->traits);
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_result_users_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, op));
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
  if (op->instance_flags == flags) return iree_ok_status();
  loom_trait_flags_t old_traits = op->traits;
  op->instance_flags = flags;
  loom_op_refresh_effective_traits(rewriter->module, op);
  loom_module_update_op_direct_effects(op, old_traits, op->traits);
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_result_users_to_worklist(rewriter, op));
  IREE_RETURN_IF_ERROR(
      loom_rewriter_add_parent_summary_ops_to_worklist(rewriter, op));
  rewriter->flags |= LOOM_REWRITER_FLAG_CHANGED;
  return iree_ok_status();
}
