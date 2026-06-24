// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/loop/loop_fusion.h"

#include "loom/analysis/loop_domain.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

#define LOOM_LOOP_FUSION_STATISTICS(V, statistics_type)          \
  V(statistics_type, loops_visited, "loops-visited",             \
    "Number of scf.for operations inspected.")                   \
  V(statistics_type, loops_fused, "loops-fused",                 \
    "Number of adjacent loop pairs fused.")                      \
  V(statistics_type, candidates_rejected, "candidates-rejected", \
    "Number of adjacent loop pairs rejected by preflight.")

LOOM_PASS_STATISTICS_DEFINE(loom_loop_fusion_statistics,
                            loom_loop_fusion_statistics_t,
                            LOOM_LOOP_FUSION_STATISTICS)

static const loom_pass_info_t loom_loop_fusion_pass_info_storage = {
    .name = IREE_SVL("loop-fusion"),
    .description = IREE_SVL("Fuse compatible adjacent scf.for loops."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_loop_fusion_statistics_layout,
};

const loom_pass_info_t* loom_loop_fusion_pass_info(void) {
  return &loom_loop_fusion_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Scratch stacks
//===----------------------------------------------------------------------===//

#define LOOM_LOOP_FUSION_INITIAL_REGION_STACK_CAPACITY 16

typedef struct loom_loop_fusion_region_stack_t {
  // Region pointers waiting to be processed.
  loom_region_t** regions;
  // Number of queued region pointers.
  iree_host_size_t count;
  // Allocated region pointer capacity.
  iree_host_size_t capacity;
} loom_loop_fusion_region_stack_t;

static iree_status_t loom_loop_fusion_region_stack_initialize(
    iree_arena_allocator_t* arena, loom_loop_fusion_region_stack_t* stack) {
  stack->count = 0;
  stack->capacity = LOOM_LOOP_FUSION_INITIAL_REGION_STACK_CAPACITY;
  return iree_arena_allocate_array(
      arena, stack->capacity, sizeof(loom_region_t*), (void**)&stack->regions);
}

static iree_status_t loom_loop_fusion_region_stack_push(
    iree_arena_allocator_t* arena, loom_loop_fusion_region_stack_t* stack,
    loom_region_t* region) {
  if (!region || region->block_count == 0) return iree_ok_status();
  if (stack->count >= stack->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, stack->count, stack->count + 1, sizeof(loom_region_t*),
        &stack->capacity, (void**)&stack->regions));
  }
  stack->regions[stack->count++] = region;
  return iree_ok_status();
}

static loom_region_t* loom_loop_fusion_region_stack_pop(
    loom_loop_fusion_region_stack_t* stack) {
  return stack->count > 0 ? stack->regions[--stack->count] : NULL;
}

//===----------------------------------------------------------------------===//
// Candidate state
//===----------------------------------------------------------------------===//

typedef struct loom_loop_fusion_for_info_t {
  // scf.for operation.
  loom_op_t* op;
  // Body region of the loop.
  loom_region_t* body;
  // Single entry/body block.
  loom_block_t* block;
  // Body terminator. Must be scf.yield.
  loom_op_t* yield_op;
  // Variadic loop initial values.
  loom_value_slice_t iter_args;
  // Variadic loop result values.
  loom_value_slice_t results;
  // Counted loop domain.
  loom_loop_domain_t domain;
  // Induction variable block argument.
  loom_value_id_t induction_variable;
} loom_loop_fusion_for_info_t;

typedef struct loom_loop_fusion_context_t {
  // Pass instance owning statistics and scratch memory.
  loom_pass_t* pass;
  // Typed statistics storage for the current pass invocation.
  loom_loop_fusion_statistics_t* statistics;
  // Module being transformed.
  loom_module_t* module;
  // Rewriter used for all IR mutation.
  loom_rewriter_t* rewriter;
  // Per-candidate transient arena for remaps and temporary value arrays.
  iree_arena_allocator_t* fusion_arena;
  // Region DFS stack for function traversal.
  loom_loop_fusion_region_stack_t region_stack;
} loom_loop_fusion_context_t;

//===----------------------------------------------------------------------===//
// Structural queries
//===----------------------------------------------------------------------===//

static bool loom_loop_fusion_op_is_under_op(const loom_op_t* root,
                                            const loom_op_t* op) {
  for (const loom_op_t* current = op; current; current = current->parent_op) {
    if (current == root) return true;
  }
  return false;
}

static bool loom_loop_fusion_block_is_under_op(const loom_op_t* root,
                                               const loom_block_t* block) {
  if (!block || !block->first_op) return false;
  return loom_loop_fusion_op_is_under_op(root, block->first_op->parent_op);
}

static bool loom_loop_fusion_value_is_type_used_under_op(
    const loom_module_t* module, loom_value_id_t value_id,
    const loom_op_t* root) {
  if (value_id >= module->values.count ||
      value_id >= module->type_uses.value_capacity) {
    return false;
  }
  loom_type_use_id_t use_id =
      module->type_uses.value_heads[value_id].first_incoming_use_id;
  while (use_id != LOOM_TYPE_USE_ID_INVALID) {
    const loom_type_use_t* type_use = &module->type_uses.records[use_id];
    if (type_use->user_value_id >= module->values.count) return true;
    const loom_value_t* user_value =
        loom_module_value(module, type_use->user_value_id);
    if (loom_value_is_block_arg(user_value)) {
      if (loom_loop_fusion_block_is_under_op(
              root, loom_value_def_block(user_value))) {
        return true;
      }
    } else if (loom_loop_fusion_op_is_under_op(root,
                                               loom_value_def_op(user_value))) {
      return true;
    }
    use_id = type_use->next_incoming_use_id;
  }
  return false;
}

static bool loom_loop_fusion_op_subtree_effects_are_allowed(
    const loom_module_t* module, const loom_op_t* op, bool allow_reads) {
  loom_trait_flags_t traits = loom_op_effective_traits(module, op);
  if (iree_any_bit_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_WRITES_MEMORY |
                                   LOOM_TRAIT_UNKNOWN_EFFECTS |
                                   LOOM_TRAIT_NON_DETERMINISTIC |
                                   LOOM_TRAIT_CONVERGENT)) {
    return false;
  }
  if (!allow_reads && iree_any_bit_set(traits, LOOM_TRAIT_READS_MEMORY)) {
    return false;
  }

  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_region_t* region = regions[region_index];
    if (!region) continue;
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        if (!loom_loop_fusion_op_subtree_effects_are_allowed(module, child_op,
                                                             allow_reads)) {
          return false;
        }
      }
    }
  }
  return true;
}

static bool loom_loop_fusion_block_effects_are_allowed(
    const loom_module_t* module, const loom_block_t* block, bool allow_reads) {
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (!loom_loop_fusion_op_subtree_effects_are_allowed(module, op,
                                                         allow_reads)) {
      return false;
    }
  }
  return true;
}

// Reads the counted-loop subset accepted by this pass. scf.while remains a
// LoopLike op for generic analyses such as LICM, but it has no counted domain
// until a separate analysis proves one.
static bool loom_loop_fusion_read_for_info(loom_op_t* op,
                                           loom_loop_fusion_for_info_t* info) {
  if (!loom_scf_for_isa(op)) return false;
  if (op->tied_result_count != 0) return false;

  loom_region_t* body = loom_scf_for_body(op);
  if (!body || body->block_count != 1) return false;
  loom_block_t* block = loom_region_entry_block(body);
  if (!block || block->op_count == 0) return false;

  loom_op_t* yield_op = block->last_op;
  if (!yield_op || !loom_scf_yield_isa(yield_op)) return false;

  loom_value_slice_t iter_args = loom_scf_for_iter_args(op);
  loom_value_slice_t results = loom_scf_for_results(op);
  loom_value_slice_t yielded = loom_scf_yield_values(yield_op);
  if (iter_args.count != results.count || yielded.count != results.count) {
    return false;
  }
  if (block->arg_count != (uint16_t)(1 + iter_args.count)) return false;

  *info = (loom_loop_fusion_for_info_t){
      .op = op,
      .body = body,
      .block = block,
      .yield_op = yield_op,
      .iter_args = iter_args,
      .results = results,
      .domain =
          {
              .lower_bound = loom_scf_for_lower_bound(op),
              .upper_bound = loom_scf_for_upper_bound(op),
              .step = loom_scf_for_step(op),
          },
      .induction_variable = loom_region_entry_arg_id(body, 0),
  };
  return true;
}

//===----------------------------------------------------------------------===//
// Lane-local forwarding legality
//===----------------------------------------------------------------------===//

static bool loom_loop_fusion_index_list_is_single_dynamic_iv(
    loom_value_slice_t dynamic_indices, loom_attribute_t static_indices,
    loom_value_id_t induction_variable) {
  if (dynamic_indices.count != 1) return false;
  if (dynamic_indices.values[0] != induction_variable) return false;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1) {
    return false;
  }
  return static_indices.i64_array[0] == INT64_MIN;
}

static bool loom_loop_fusion_first_yield_is_lane_insert(
    const loom_module_t* module, const loom_loop_fusion_for_info_t* first,
    uint16_t result_ordinal) {
  loom_value_slice_t yielded = loom_scf_yield_values(first->yield_op);
  loom_value_id_t yielded_value = yielded.values[result_ordinal];
  if (yielded_value == LOOM_VALUE_ID_INVALID) return false;

  const loom_value_t* yielded_value_def =
      loom_module_value(module, yielded_value);
  if (loom_value_is_block_arg(yielded_value_def)) return false;
  loom_op_t* insert_op = loom_value_def_op(yielded_value_def);
  if (!insert_op || !loom_vector_insert_isa(insert_op)) return false;
  if (loom_vector_insert_result(insert_op) != yielded_value) return false;
  if (loom_vector_insert_dest(insert_op) !=
      loom_region_entry_arg_id(first->body, (uint16_t)(1 + result_ordinal))) {
    return false;
  }
  return loom_loop_fusion_index_list_is_single_dynamic_iv(
      loom_vector_insert_indices(insert_op),
      loom_vector_insert_static_indices(insert_op), first->induction_variable);
}

static bool loom_loop_fusion_second_use_is_lane_extract(
    const loom_loop_fusion_for_info_t* second, loom_value_id_t first_result,
    loom_use_t use) {
  loom_op_t* user_op = loom_use_user_op(use);
  if (!loom_vector_extract_isa(user_op)) return false;
  if (loom_use_operand_index(use) != 0) return false;
  if (loom_vector_extract_source(user_op) != first_result) return false;
  return loom_loop_fusion_index_list_is_single_dynamic_iv(
      loom_vector_extract_indices(user_op),
      loom_vector_extract_static_indices(user_op), second->induction_variable);
}

static bool loom_loop_fusion_first_result_uses_are_lane_local(
    const loom_module_t* module, const loom_loop_fusion_for_info_t* first,
    const loom_loop_fusion_for_info_t* second, uint16_t result_ordinal) {
  loom_value_id_t first_result = first->results.values[result_ordinal];
  if (first_result == LOOM_VALUE_ID_INVALID ||
      first_result >= module->values.count) {
    return false;
  }
  if (loom_loop_fusion_value_is_type_used_under_op(module, first_result,
                                                   second->op)) {
    return false;
  }

  bool needs_lane_forwarding = false;
  const loom_value_t* value = loom_module_value(module, first_result);
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    loom_op_t* user_op = loom_use_user_op(uses[i]);
    if (!loom_loop_fusion_op_is_under_op(second->op, user_op)) continue;
    needs_lane_forwarding = true;
    if (!loom_loop_fusion_second_use_is_lane_extract(second, first_result,
                                                     uses[i])) {
      return false;
    }
  }
  if (!needs_lane_forwarding) return true;
  return loom_loop_fusion_first_yield_is_lane_insert(module, first,
                                                     result_ordinal);
}

static bool loom_loop_fusion_first_results_are_legal_for_second_body(
    const loom_module_t* module, const loom_loop_fusion_for_info_t* first,
    const loom_loop_fusion_for_info_t* second) {
  for (uint16_t i = 0; i < first->results.count; ++i) {
    if (!loom_loop_fusion_first_result_uses_are_lane_local(module, first,
                                                           second, i)) {
      return false;
    }
  }
  return true;
}

static bool loom_loop_fusion_second_iter_args_do_not_use_first_results(
    const loom_loop_fusion_for_info_t* first,
    const loom_loop_fusion_for_info_t* second) {
  for (uint16_t i = 0; i < second->iter_args.count; ++i) {
    for (uint16_t j = 0; j < first->results.count; ++j) {
      if (second->iter_args.values[i] == first->results.values[j]) {
        return false;
      }
    }
  }
  return true;
}

static bool loom_loop_fusion_candidate_is_legal(
    loom_loop_fusion_context_t* context,
    const loom_loop_fusion_for_info_t* first,
    const loom_loop_fusion_for_info_t* second) {
  uint32_t combined_iter_arg_count =
      (uint32_t)first->iter_args.count + (uint32_t)second->iter_args.count;
  if (combined_iter_arg_count > UINT16_MAX) {
    return false;
  }
  uint32_t combined_result_count =
      (uint32_t)first->results.count + (uint32_t)second->results.count;
  if (combined_result_count > UINT16_MAX) {
    return false;
  }
  if (!loom_loop_domain_equal(context->rewriter->fact_table, first->domain,
                              second->domain)) {
    return false;
  }
  if (!loom_loop_fusion_block_effects_are_allowed(context->module, first->block,
                                                  /*allow_reads=*/true)) {
    return false;
  }
  if (!loom_loop_fusion_block_effects_are_allowed(context->module,
                                                  second->block,
                                                  /*allow_reads=*/false)) {
    return false;
  }
  if (!loom_loop_fusion_second_iter_args_do_not_use_first_results(first,
                                                                  second)) {
    return false;
  }
  return loom_loop_fusion_first_results_are_legal_for_second_body(
      context->module, first, second);
}

//===----------------------------------------------------------------------===//
// Fusion
//===----------------------------------------------------------------------===//

static iree_status_t loom_loop_fusion_concat_iter_args(
    iree_arena_allocator_t* arena, const loom_loop_fusion_for_info_t* first,
    const loom_loop_fusion_for_info_t* second, loom_value_id_t** out_iter_args,
    uint16_t* out_count) {
  uint32_t combined_count =
      (uint32_t)first->iter_args.count + (uint32_t)second->iter_args.count;
  *out_count = (uint16_t)combined_count;
  *out_iter_args = NULL;
  if (combined_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, combined_count, sizeof(loom_value_id_t), (void**)out_iter_args));
  for (uint16_t i = 0; i < first->iter_args.count; ++i) {
    (*out_iter_args)[i] = first->iter_args.values[i];
  }
  for (uint16_t i = 0; i < second->iter_args.count; ++i) {
    (*out_iter_args)[first->iter_args.count + i] = second->iter_args.values[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_loop_fusion_make_placeholder_result_types(
    iree_arena_allocator_t* arena, const loom_loop_fusion_for_info_t* first,
    const loom_loop_fusion_for_info_t* second, loom_type_t** out_result_types,
    uint16_t* out_count) {
  uint32_t combined_count =
      (uint32_t)first->results.count + (uint32_t)second->results.count;
  *out_count = (uint16_t)combined_count;
  *out_result_types = NULL;
  if (combined_count == 0) return iree_ok_status();

  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, combined_count, sizeof(loom_type_t), (void**)out_result_types));
  for (uint16_t i = 0; i < combined_count; ++i) {
    (*out_result_types)[i] = loom_type_none();
  }
  return iree_ok_status();
}

static iree_status_t loom_loop_fusion_map_old_results_to_fused(
    loom_ir_remap_t* remap, const loom_loop_fusion_for_info_t* first,
    const loom_loop_fusion_for_info_t* second, loom_op_t* fused_loop) {
  loom_value_slice_t fused_results = loom_scf_for_results(fused_loop);
  if (first->results.count > 0) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(remap, first->results.values,
                                                  fused_results.values,
                                                  first->results.count));
  }
  if (second->results.count > 0) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_values(
        remap, second->results.values,
        fused_results.values + first->results.count, second->results.count));
  }
  return iree_ok_status();
}

static iree_status_t loom_loop_fusion_set_fused_result_types(
    loom_loop_fusion_context_t* context,
    const loom_loop_fusion_for_info_t* first,
    const loom_loop_fusion_for_info_t* second, loom_op_t* fused_loop) {
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      context->module, context->module, context->fusion_arena,
      &(loom_ir_remap_options_t){
          .allow_unmapped_values = true,
      },
      &remap));
  IREE_RETURN_IF_ERROR(loom_loop_fusion_map_old_results_to_fused(
      &remap, first, second, fused_loop));

  loom_value_slice_t fused_results = loom_scf_for_results(fused_loop);
  loom_type_t* first_result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_value_types(&remap, first->results.values,
                                                 first->results.count,
                                                 &first_result_types));
  for (uint16_t i = 0; i < first->results.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_set_value_type(
        context->rewriter, fused_results.values[i], first_result_types[i]));
  }

  loom_type_t* second_result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_ir_remap_value_types(&remap, second->results.values,
                                                 second->results.count,
                                                 &second_result_types));
  for (uint16_t i = 0; i < second->results.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_set_value_type(
        context->rewriter, fused_results.values[first->results.count + i],
        second_result_types[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_loop_fusion_copy_result_names(
    loom_module_t* module, const loom_loop_fusion_for_info_t* first,
    const loom_loop_fusion_for_info_t* second, loom_op_t* fused_loop) {
  loom_value_slice_t fused_results = loom_scf_for_results(fused_loop);
  for (uint16_t i = 0; i < first->results.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_module_copy_value_name(
        module, first->results.values[i], fused_results.values[i]));
  }
  for (uint16_t i = 0; i < second->results.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_module_copy_value_name(
        module, second->results.values[i],
        fused_results.values[first->results.count + i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_loop_fusion_bind_remap_for_loop_body(
    const loom_loop_fusion_for_info_t* source_loop, loom_op_t* fused_loop,
    uint16_t fused_iter_arg_offset, loom_ir_remap_t* remap) {
  loom_region_t* fused_body = loom_scf_for_body(fused_loop);
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_map_value(remap, source_loop->induction_variable,
                              loom_region_entry_arg_id(fused_body, 0)));
  for (uint16_t i = 0; i < source_loop->iter_args.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        remap, loom_region_entry_arg_id(source_loop->body, (uint16_t)(1 + i)),
        loom_region_entry_arg_id(fused_body,
                                 (uint16_t)(1 + fused_iter_arg_offset + i))));
  }
  return iree_ok_status();
}

static iree_status_t loom_loop_fusion_resolve_yield_values(
    loom_ir_remap_t* remap, const loom_loop_fusion_for_info_t* loop,
    loom_value_id_t* values) {
  loom_value_slice_t yielded = loom_scf_yield_values(loop->yield_op);
  for (uint16_t i = 0; i < yielded.count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_resolve_value(remap, yielded.values[i], &values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_loop_fusion_move_body_before_yield(
    loom_loop_fusion_context_t* context,
    const loom_loop_fusion_for_info_t* loop, loom_op_t* yield_op,
    loom_ir_remap_t* remap) {
  loom_ir_move_block_options_t options = {
      .omit_terminators = true,
  };
  return loom_ir_move_block_ops_before(context->rewriter, loop->block, yield_op,
                                       remap, &options);
}

static iree_status_t loom_loop_fusion_clear_result_names(loom_module_t* module,
                                                         loom_op_t* op) {
  loom_value_id_t* results = loom_op_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID) continue;
    IREE_RETURN_IF_ERROR(loom_module_clear_value_name(module, results[i]));
  }

  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    loom_region_t* region = regions[region_index];
    if (!region) continue;
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        IREE_RETURN_IF_ERROR(
            loom_loop_fusion_clear_result_names(module, child_op));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_loop_fusion_clear_block_result_names(
    loom_module_t* module, loom_block_t* block) {
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_scf_yield_isa(op)) continue;
    IREE_RETURN_IF_ERROR(loom_loop_fusion_clear_result_names(module, op));
  }
  return iree_ok_status();
}

static iree_status_t loom_loop_fusion_replace_yield_values(
    loom_loop_fusion_context_t* context, loom_op_t* yield_op,
    const loom_value_id_t* values, uint16_t count) {
  for (uint16_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_set_operand(context->rewriter, yield_op, i, values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_loop_fusion_replace_old_loop_results(
    loom_loop_fusion_context_t* context,
    const loom_loop_fusion_for_info_t* first,
    const loom_loop_fusion_for_info_t* second, loom_op_t* fused_loop) {
  loom_value_slice_t fused_results = loom_scf_for_results(fused_loop);
  for (uint16_t i = 0; i < first->results.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        context->rewriter, first->results.values[i], fused_results.values[i]));
  }
  for (uint16_t i = 0; i < second->results.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        context->rewriter, second->results.values[i],
        fused_results.values[first->results.count + i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_loop_fusion_build_provisional_yield_values(
    loom_op_t* fused_loop, loom_value_id_t* values, uint16_t count) {
  loom_region_t* body = loom_scf_for_body(fused_loop);
  for (uint16_t i = 0; i < count; ++i) {
    values[i] = loom_region_entry_arg_id(body, (uint16_t)(1 + i));
  }
  return iree_ok_status();
}

static iree_status_t loom_loop_fusion_fuse_pair(
    loom_loop_fusion_context_t* context,
    const loom_loop_fusion_for_info_t* first,
    const loom_loop_fusion_for_info_t* second, loom_op_t** out_fused_loop) {
  *out_fused_loop = NULL;
  iree_arena_allocator_t* scratch_arena = context->fusion_arena;
  iree_arena_reset(scratch_arena);

  loom_value_id_t* iter_args = NULL;
  uint16_t iter_arg_count = 0;
  IREE_RETURN_IF_ERROR(loom_loop_fusion_concat_iter_args(
      scratch_arena, first, second, &iter_args, &iter_arg_count));

  loom_type_t* result_types = NULL;
  uint16_t result_count = 0;
  IREE_RETURN_IF_ERROR(loom_loop_fusion_make_placeholder_result_types(
      scratch_arena, first, second, &result_types, &result_count));

  loom_value_id_t* provisional_yield_values = NULL;
  if (result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, result_count, sizeof(loom_value_id_t),
        (void**)&provisional_yield_values));
  }

  loom_value_id_t* first_yield_values = NULL;
  if (first->results.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, first->results.count, sizeof(loom_value_id_t),
        (void**)&first_yield_values));
  }

  loom_value_id_t* final_yield_values = NULL;
  if (result_count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        scratch_arena, result_count, sizeof(loom_value_id_t),
        (void**)&final_yield_values));
  }

  loom_ir_remap_t first_remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(context->module, context->module, scratch_arena,
                               &(loom_ir_remap_options_t){
                                   .allow_unmapped_values = true,
                               },
                               &first_remap));

  loom_ir_remap_t second_remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_initialize(context->module, context->module, scratch_arena,
                               &(loom_ir_remap_options_t){
                                   .allow_unmapped_values = true,
                               },
                               &second_remap));

  loom_builder_t* builder = &context->rewriter->builder;
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_before(builder, first->op);

  loom_op_t* fused_loop = NULL;
  iree_status_t status = loom_scf_for_build(
      builder, /*build_flags=*/0, first->domain.lower_bound,
      first->domain.upper_bound, first->domain.step, iter_args, iter_arg_count,
      result_types, result_count, NULL, 0, LOOM_VALUE_ID_INVALID,
      /*unroll_policy=*/0, /*unroll_schedule=*/0, first->op->location,
      &fused_loop);
  if (iree_status_is_ok(status)) {
    status = loom_loop_fusion_copy_result_names(context->module, first, second,
                                                fused_loop);
  }
  if (iree_status_is_ok(status)) {
    status = loom_loop_fusion_set_fused_result_types(context, first, second,
                                                     fused_loop);
  }

  if (iree_status_is_ok(status)) {
    status = loom_loop_fusion_build_provisional_yield_values(
        fused_loop, provisional_yield_values, result_count);
  }

  loom_op_t* fused_yield = NULL;
  if (iree_status_is_ok(status)) {
    loom_builder_ip_t body_ip = loom_builder_enter_region(
        builder, fused_loop, loom_scf_for_body(fused_loop));
    status =
        loom_scf_yield_build(builder, provisional_yield_values, result_count,
                             first->op->location, &fused_yield);
    loom_builder_restore(builder, body_ip);
  }

  if (iree_status_is_ok(status)) {
    status = loom_loop_fusion_bind_remap_for_loop_body(first, fused_loop, 0,
                                                       &first_remap);
  }
  if (iree_status_is_ok(status)) {
    status = loom_loop_fusion_move_body_before_yield(context, first,
                                                     fused_yield, &first_remap);
  }

  if (iree_status_is_ok(status)) {
    status = loom_loop_fusion_resolve_yield_values(&first_remap, first,
                                                   first_yield_values);
  }

  if (iree_status_is_ok(status)) {
    status = loom_loop_fusion_bind_remap_for_loop_body(
        second, fused_loop, first->iter_args.count, &second_remap);
  }
  if (iree_status_is_ok(status)) {
    for (uint16_t i = 0; i < first->results.count; ++i) {
      status = loom_ir_remap_map_value(&second_remap, first->results.values[i],
                                       first_yield_values[i]);
      if (!iree_status_is_ok(status)) break;
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_loop_fusion_clear_block_result_names(context->module,
                                                       second->block);
  }
  if (iree_status_is_ok(status)) {
    status = loom_loop_fusion_move_body_before_yield(
        context, second, fused_yield, &second_remap);
  }

  if (iree_status_is_ok(status)) {
    for (uint16_t i = 0; i < first->results.count; ++i) {
      final_yield_values[i] = first_yield_values[i];
    }
    if (second->results.count > 0) {
      status = loom_loop_fusion_resolve_yield_values(
          &second_remap, second, final_yield_values + first->results.count);
    }
  }
  if (iree_status_is_ok(status)) {
    status = loom_loop_fusion_replace_yield_values(
        context, fused_yield, final_yield_values, result_count);
  }
  if (iree_status_is_ok(status)) {
    status = loom_loop_fusion_replace_old_loop_results(context, first, second,
                                                       fused_loop);
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_erase(context->rewriter, second->op);
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_erase(context->rewriter, first->op);
  }

  loom_builder_restore(builder, saved_ip);
  if (iree_status_is_ok(status)) {
    *out_fused_loop = fused_loop;
  }
  iree_arena_reset(scratch_arena);
  return status;
}

//===----------------------------------------------------------------------===//
// Traversal
//===----------------------------------------------------------------------===//

static iree_status_t loom_loop_fusion_push_child_regions(
    loom_loop_fusion_context_t* context, loom_op_t* op) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_loop_fusion_region_stack_push(
        context->pass->arena, &context->region_stack, regions[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_loop_fusion_process_block(
    loom_loop_fusion_context_t* context, loom_block_t* block,
    bool* out_changed) {
  loom_op_t* op = block->first_op;
  while (op) {
    loom_op_t* next_op = op->next_op;
    if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
      op = next_op;
      continue;
    }

    loom_loop_fusion_for_info_t first = {0};
    loom_loop_fusion_for_info_t second = {0};
    if (next_op && loom_loop_fusion_read_for_info(op, &first)) {
      ++context->statistics->loops_visited;
      if (loom_loop_fusion_read_for_info(next_op, &second)) {
        if (loom_loop_fusion_candidate_is_legal(context, &first, &second)) {
          loom_op_t* fused_loop = NULL;
          IREE_RETURN_IF_ERROR(loom_loop_fusion_fuse_pair(
              context, &first, &second, &fused_loop));
          ++context->statistics->loops_fused;
          *out_changed = true;
          op = fused_loop;
          continue;
        }
        ++context->statistics->candidates_rejected;
      }
    }

    IREE_RETURN_IF_ERROR(loom_loop_fusion_push_child_regions(context, op));
    op = next_op;
  }
  return iree_ok_status();
}

static iree_status_t loom_loop_fusion_process_function_once(
    loom_loop_fusion_context_t* context, loom_func_like_t function,
    bool* out_changed) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();

  context->region_stack.count = 0;
  IREE_RETURN_IF_ERROR(loom_loop_fusion_region_stack_push(
      context->pass->arena, &context->region_stack, body));

  while (true) {
    loom_region_t* region =
        loom_loop_fusion_region_stack_pop(&context->region_stack);
    if (!region) break;

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      IREE_RETURN_IF_ERROR(
          loom_loop_fusion_process_block(context, block, out_changed));
    }
  }

  return iree_ok_status();
}

iree_status_t loom_loop_fusion_run(loom_pass_t* pass, loom_module_t* module,
                                   loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  loom_value_fact_table_t* facts = NULL;
  iree_status_t status = loom_pass_value_facts_prepare(
      pass, module, loom_pass_value_fact_scope_function(function), &facts);
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_enable_analysis(&rewriter, function, facts);
  }

  iree_arena_allocator_t fusion_arena = {0};
  bool fusion_arena_initialized = false;
  if (iree_status_is_ok(status)) {
    iree_arena_initialize(pass->arena->block_pool, &fusion_arena);
    fusion_arena_initialized = true;
  }

  loom_loop_fusion_context_t context = {
      .pass = pass,
      .statistics = loom_loop_fusion_statistics(pass),
      .module = module,
      .rewriter = &rewriter,
      .fusion_arena = &fusion_arena,
  };
  if (iree_status_is_ok(status)) {
    status = loom_loop_fusion_region_stack_initialize(pass->arena,
                                                      &context.region_stack);
  }

  bool changed = true;
  bool any_changed = false;
  while (iree_status_is_ok(status) && changed) {
    changed = false;
    status =
        loom_loop_fusion_process_function_once(&context, function, &changed);
    any_changed |= changed;
  }

  if (iree_status_is_ok(status) && any_changed) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  if (!iree_status_is_ok(status)) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
  }
  if (fusion_arena_initialized) {
    iree_arena_deinitialize(&fusion_arena);
  }
  return status;
}
