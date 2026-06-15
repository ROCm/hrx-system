// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cfg/branch_fusion.h"

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/scf/ops.h"
#include "loom/rewrite/materialize.h"
#include "loom/rewrite/remap.h"
#include "loom/rewrite/rewriter.h"

#define LOOM_BRANCH_FUSION_STATISTICS(V, statistics_type)        \
  V(statistics_type, branches_visited, "branches-visited",       \
    "Number of region-branch operations inspected.")             \
  V(statistics_type, branches_fused, "branches-fused",           \
    "Number of adjacent branch pairs fused.")                    \
  V(statistics_type, candidates_rejected, "candidates-rejected", \
    "Number of adjacent branch pairs rejected by preflight.")

LOOM_PASS_STATISTICS_DEFINE(loom_branch_fusion_statistics,
                            loom_branch_fusion_statistics_t,
                            LOOM_BRANCH_FUSION_STATISTICS)

static const loom_pass_info_t loom_branch_fusion_pass_info_storage = {
    .name = IREE_SVL("branch-fusion"),
    .description = IREE_SVL("Fuse compatible adjacent branch regions."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_branch_fusion_statistics_layout,
};

const loom_pass_info_t* loom_branch_fusion_pass_info(void) {
  return &loom_branch_fusion_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Scratch stacks
//===----------------------------------------------------------------------===//

#define LOOM_BRANCH_FUSION_INITIAL_REGION_STACK_CAPACITY 16
#define LOOM_BRANCH_FUSION_IF_REGION_COUNT 2

typedef struct loom_branch_fusion_region_stack_t {
  // Region pointers waiting to be processed.
  loom_region_t** regions;
  // Number of queued region pointers.
  iree_host_size_t count;
  // Allocated region pointer capacity.
  iree_host_size_t capacity;
} loom_branch_fusion_region_stack_t;

static iree_status_t loom_branch_fusion_region_stack_initialize(
    iree_arena_allocator_t* arena, loom_branch_fusion_region_stack_t* stack) {
  stack->count = 0;
  stack->capacity = LOOM_BRANCH_FUSION_INITIAL_REGION_STACK_CAPACITY;
  return iree_arena_allocate_array(
      arena, stack->capacity, sizeof(loom_region_t*), (void**)&stack->regions);
}

static iree_status_t loom_branch_fusion_region_stack_push(
    iree_arena_allocator_t* arena, loom_branch_fusion_region_stack_t* stack,
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

static loom_region_t* loom_branch_fusion_region_stack_pop(
    loom_branch_fusion_region_stack_t* stack) {
  return stack->count > 0 ? stack->regions[--stack->count] : NULL;
}

//===----------------------------------------------------------------------===//
// Candidate state
//===----------------------------------------------------------------------===//

typedef struct loom_branch_fusion_info_t {
  // Branch operation.
  loom_op_t* op;
  // RegionBranch interface for op.
  loom_region_branch_t branch;
  // Branch selector value.
  loom_value_id_t selector;
  // Variadic branch results.
  loom_value_slice_t results;
  // Entry blocks for the then/else regions.
  loom_block_t* blocks[LOOM_BRANCH_FUSION_IF_REGION_COUNT];
  // Terminators for the then/else regions.
  loom_op_t* terminators[LOOM_BRANCH_FUSION_IF_REGION_COUNT];
} loom_branch_fusion_info_t;

typedef struct loom_branch_fusion_context_t {
  // Pass instance owning persistent scratch memory.
  loom_pass_t* pass;
  // Typed statistics storage for the current pass invocation.
  loom_branch_fusion_statistics_t* statistics;
  // Module being transformed.
  loom_module_t* module;
  // Rewriter used for all IR mutation.
  loom_rewriter_t* rewriter;
  // Per-candidate transient arena for remaps and temporary value arrays.
  iree_arena_allocator_t* fusion_arena;
  // Region DFS stack for function traversal.
  loom_branch_fusion_region_stack_t region_stack;
} loom_branch_fusion_context_t;

//===----------------------------------------------------------------------===//
// Structural queries
//===----------------------------------------------------------------------===//

static bool loom_branch_fusion_read_if_info(
    const loom_module_t* module, loom_op_t* op,
    loom_branch_fusion_info_t* out_info) {
  if (!loom_scf_if_isa(op)) return false;
  if (op->tied_result_count != 0) return false;
  if (op->region_count != LOOM_BRANCH_FUSION_IF_REGION_COUNT) return false;

  loom_region_branch_t branch = loom_region_branch_cast(module, op);
  if (!loom_region_branch_isa(branch)) return false;

  loom_value_id_t selector = loom_region_branch_selector(branch);
  if (selector == LOOM_VALUE_ID_INVALID || selector >= module->values.count) {
    return false;
  }

  loom_value_slice_t results = loom_scf_if_results(op);
  loom_block_t* blocks[LOOM_BRANCH_FUSION_IF_REGION_COUNT] = {0};
  loom_op_t* terminators[LOOM_BRANCH_FUSION_IF_REGION_COUNT] = {0};
  for (uint8_t i = 0; i < LOOM_BRANCH_FUSION_IF_REGION_COUNT; ++i) {
    loom_region_t* region = loom_region_branch_region(module, branch, i);
    if (!region || region->block_count != 1) return false;
    blocks[i] = loom_region_entry_block(region);
    if (!blocks[i]) return false;
    terminators[i] = loom_region_branch_region_terminator(module, branch, i);
    if (!terminators[i]) return false;
    if (terminators[i]->operand_count != results.count) return false;
  }

  *out_info = (loom_branch_fusion_info_t){
      .op = op,
      .branch = branch,
      .selector = selector,
      .results = results,
      .blocks = {blocks[0], blocks[1]},
      .terminators = {terminators[0], terminators[1]},
  };
  return true;
}

static bool loom_branch_fusion_candidate_is_legal(
    const loom_branch_fusion_info_t* first,
    const loom_branch_fusion_info_t* second) {
  if (first->op->kind != second->op->kind) return false;
  if (first->selector != second->selector) return false;
  uint32_t combined_count =
      (uint32_t)first->results.count + (uint32_t)second->results.count;
  return combined_count <= UINT16_MAX;
}

//===----------------------------------------------------------------------===//
// Fusion
//===----------------------------------------------------------------------===//

static iree_status_t loom_branch_fusion_concat_placeholder_types(
    iree_arena_allocator_t* arena, const loom_branch_fusion_info_t* first,
    const loom_branch_fusion_info_t* second, loom_type_t** out_result_types,
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

static iree_status_t loom_branch_fusion_map_old_results_to_fused(
    const loom_branch_fusion_info_t* first,
    const loom_branch_fusion_info_t* second, loom_op_t* fused_if,
    loom_ir_remap_t* remap) {
  loom_value_slice_t fused_results = loom_scf_if_results(fused_if);
  for (uint16_t i = 0; i < first->results.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        remap, first->results.values[i], fused_results.values[i]));
  }
  for (uint16_t i = 0; i < second->results.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        remap, second->results.values[i],
        fused_results.values[first->results.count + i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_branch_fusion_set_fused_result_types(
    loom_branch_fusion_context_t* context,
    const loom_branch_fusion_info_t* first,
    const loom_branch_fusion_info_t* second, loom_op_t* fused_if) {
  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(
      context->module, context->module, context->fusion_arena,
      &(loom_ir_remap_options_t){
          .allow_unmapped_values = true,
      },
      &remap));
  IREE_RETURN_IF_ERROR(loom_branch_fusion_map_old_results_to_fused(
      first, second, fused_if, &remap));

  loom_value_slice_t fused_results = loom_scf_if_results(fused_if);
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

static iree_status_t loom_branch_fusion_copy_result_names(
    loom_module_t* module, const loom_branch_fusion_info_t* first,
    const loom_branch_fusion_info_t* second, loom_op_t* fused_if) {
  loom_value_slice_t fused_results = loom_scf_if_results(fused_if);
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

static iree_status_t loom_branch_fusion_build_placeholder_values(
    iree_arena_allocator_t* arena, loom_value_id_t selector,
    uint16_t result_count, loom_value_id_t** out_values) {
  *out_values = NULL;
  if (result_count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, result_count, sizeof(loom_value_id_t), (void**)out_values));
  for (uint16_t i = 0; i < result_count; ++i) {
    (*out_values)[i] = selector;
  }
  return iree_ok_status();
}

static iree_status_t loom_branch_fusion_build_region_terminators(
    loom_branch_fusion_context_t* context, loom_op_t* fused_if,
    const loom_value_id_t* placeholder_values, uint16_t result_count,
    loom_op_t** out_terminators) {
  loom_builder_t* builder = &context->rewriter->builder;
  loom_region_branch_t fused_branch =
      loom_region_branch_cast(context->module, fused_if);

  iree_status_t status = iree_ok_status();
  for (uint8_t i = 0;
       i < LOOM_BRANCH_FUSION_IF_REGION_COUNT && iree_status_is_ok(status);
       ++i) {
    loom_region_t* fused_region =
        loom_region_branch_region(context->module, fused_branch, i);
    loom_builder_ip_t region_ip =
        loom_builder_enter_region(builder, fused_if, fused_region);
    status = loom_region_branch_build_region_terminator(
        builder, context->module, fused_branch, i, placeholder_values,
        result_count, fused_if->location, &out_terminators[i]);
    loom_builder_restore(builder, region_ip);
  }
  return status;
}

static bool loom_branch_fusion_op_has_result_name(const loom_module_t* module,
                                                  const loom_op_t* op,
                                                  loom_string_id_t name_id) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID ||
        results[i] >= module->values.count) {
      continue;
    }
    if (loom_module_value(module, results[i])->name_id == name_id) return true;
  }
  return false;
}

static bool loom_branch_fusion_block_prefix_has_result_name(
    const loom_module_t* module, const loom_block_t* block,
    const loom_op_t* before_op, loom_string_id_t name_id) {
  if (name_id == LOOM_STRING_ID_INVALID) return false;
  const loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (op == before_op) break;
    if (loom_branch_fusion_op_has_result_name(module, op, name_id)) {
      return true;
    }
  }
  return false;
}

static iree_status_t loom_branch_fusion_clear_conflicting_result_names(
    loom_module_t* module, loom_block_t* source_block,
    loom_block_t* target_block, loom_op_t* before_op) {
  loom_op_t* op = NULL;
  loom_block_for_each_op(source_block, op) {
    if (loom_scf_yield_isa(op)) continue;
    loom_value_id_t* results = loom_op_results(op);
    for (uint16_t i = 0; i < op->result_count; ++i) {
      if (results[i] == LOOM_VALUE_ID_INVALID ||
          results[i] >= module->values.count) {
        continue;
      }
      loom_value_t* value = loom_module_value(module, results[i]);
      if (!loom_branch_fusion_block_prefix_has_result_name(
              module, target_block, before_op, value->name_id)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_module_clear_value_name(module, results[i]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_branch_fusion_move_body_before_terminator(
    loom_branch_fusion_context_t* context, loom_block_t* source_block,
    loom_op_t* fused_terminator, loom_ir_remap_t* remap) {
  loom_ir_move_block_options_t options = {
      .omit_terminators = true,
  };
  return loom_ir_move_block_ops_before(context->rewriter, source_block,
                                       fused_terminator, remap, &options);
}

static iree_status_t loom_branch_fusion_resolve_terminator_values(
    loom_ir_remap_t* remap, const loom_op_t* terminator,
    loom_value_id_t* values) {
  loom_value_slice_t yielded = {
      .values = (loom_value_id_t*)loom_op_const_operands(terminator),
      .count = terminator->operand_count,
  };
  for (uint16_t i = 0; i < yielded.count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_resolve_value(remap, yielded.values[i], &values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_branch_fusion_replace_terminator_values(
    loom_branch_fusion_context_t* context, loom_op_t* terminator,
    const loom_value_id_t* values, uint16_t count) {
  for (uint16_t i = 0; i < count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_rewriter_set_operand(context->rewriter, terminator, i, values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_branch_fusion_replace_old_results(
    loom_branch_fusion_context_t* context,
    const loom_branch_fusion_info_t* first,
    const loom_branch_fusion_info_t* second, loom_op_t* fused_if) {
  loom_value_slice_t fused_results = loom_scf_if_results(fused_if);
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

static iree_status_t loom_branch_fusion_initialize_region_remap(
    loom_branch_fusion_context_t* context, loom_ir_remap_t* remap) {
  return loom_ir_remap_initialize(context->module, context->module,
                                  context->fusion_arena,
                                  &(loom_ir_remap_options_t){
                                      .allow_unmapped_values = true,
                                  },
                                  remap);
}

static iree_status_t loom_branch_fusion_fuse_region(
    loom_branch_fusion_context_t* context,
    const loom_branch_fusion_info_t* first,
    const loom_branch_fusion_info_t* second, uint8_t region_index,
    loom_op_t* fused_terminator, loom_value_id_t* first_yield_values,
    loom_value_id_t* final_yield_values) {
  loom_ir_remap_t first_remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_branch_fusion_initialize_region_remap(context, &first_remap));
  IREE_RETURN_IF_ERROR(loom_branch_fusion_move_body_before_terminator(
      context, first->blocks[region_index], fused_terminator, &first_remap));
  IREE_RETURN_IF_ERROR(loom_branch_fusion_resolve_terminator_values(
      &first_remap, first->terminators[region_index], first_yield_values));

  loom_ir_remap_t second_remap = {0};
  IREE_RETURN_IF_ERROR(
      loom_branch_fusion_initialize_region_remap(context, &second_remap));
  for (uint16_t i = 0; i < first->results.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        &second_remap, first->results.values[i], first_yield_values[i]));
  }
  IREE_RETURN_IF_ERROR(loom_branch_fusion_clear_conflicting_result_names(
      context->module, second->blocks[region_index],
      fused_terminator->parent_block, fused_terminator));
  IREE_RETURN_IF_ERROR(loom_branch_fusion_move_body_before_terminator(
      context, second->blocks[region_index], fused_terminator, &second_remap));

  for (uint16_t i = 0; i < first->results.count; ++i) {
    final_yield_values[i] = first_yield_values[i];
  }
  IREE_RETURN_IF_ERROR(loom_branch_fusion_resolve_terminator_values(
      &second_remap, second->terminators[region_index],
      final_yield_values + first->results.count));
  return loom_branch_fusion_replace_terminator_values(
      context, fused_terminator, final_yield_values,
      (uint16_t)(first->results.count + second->results.count));
}

static iree_status_t loom_branch_fusion_fuse_pair(
    loom_branch_fusion_context_t* context,
    const loom_branch_fusion_info_t* first,
    const loom_branch_fusion_info_t* second, loom_op_t** out_fused_if) {
  *out_fused_if = NULL;
  iree_arena_allocator_t* scratch_arena = context->fusion_arena;
  iree_arena_reset(scratch_arena);

  loom_type_t* placeholder_types = NULL;
  uint16_t result_count = 0;
  IREE_RETURN_IF_ERROR(loom_branch_fusion_concat_placeholder_types(
      scratch_arena, first, second, &placeholder_types, &result_count));

  loom_value_id_t* placeholder_values = NULL;
  IREE_RETURN_IF_ERROR(loom_branch_fusion_build_placeholder_values(
      scratch_arena, first->selector, result_count, &placeholder_values));

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

  loom_op_t* fused_if = NULL;
  loom_builder_t* builder = &context->rewriter->builder;
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_before(builder, first->op);

  iree_status_t status = loom_scf_if_build(
      builder, LOOM_SCF_IF_BUILD_FLAG_HAS_ELSE_REGION, first->selector,
      placeholder_types, result_count, NULL, 0, first->op->location, &fused_if);
  if (iree_status_is_ok(status)) {
    status = loom_branch_fusion_copy_result_names(context->module, first,
                                                  second, fused_if);
  }
  if (iree_status_is_ok(status)) {
    status = loom_branch_fusion_set_fused_result_types(context, first, second,
                                                       fused_if);
  }

  loom_op_t* fused_terminators[LOOM_BRANCH_FUSION_IF_REGION_COUNT] = {0};
  if (iree_status_is_ok(status)) {
    status = loom_branch_fusion_build_region_terminators(
        context, fused_if, placeholder_values, result_count, fused_terminators);
  }

  for (uint8_t i = 0;
       i < LOOM_BRANCH_FUSION_IF_REGION_COUNT && iree_status_is_ok(status);
       ++i) {
    status = loom_branch_fusion_fuse_region(
        context, first, second, i, fused_terminators[i], first_yield_values,
        final_yield_values);
  }

  if (iree_status_is_ok(status)) {
    status = loom_branch_fusion_replace_old_results(context, first, second,
                                                    fused_if);
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_erase(context->rewriter, second->op);
  }
  if (iree_status_is_ok(status)) {
    status = loom_rewriter_erase(context->rewriter, first->op);
  }

  loom_builder_restore(builder, saved_ip);
  if (iree_status_is_ok(status)) {
    *out_fused_if = fused_if;
  }
  iree_arena_reset(scratch_arena);
  return status;
}

//===----------------------------------------------------------------------===//
// Traversal
//===----------------------------------------------------------------------===//

static iree_status_t loom_branch_fusion_push_child_regions(
    loom_branch_fusion_context_t* context, loom_op_t* op) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_branch_fusion_region_stack_push(
        context->pass->arena, &context->region_stack, regions[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_branch_fusion_process_block(
    loom_branch_fusion_context_t* context, loom_block_t* block,
    bool* out_changed) {
  loom_op_t* op = block->first_op;
  while (op) {
    loom_op_t* next_op = op->next_op;
    if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
      op = next_op;
      continue;
    }

    loom_branch_fusion_info_t first = {0};
    loom_branch_fusion_info_t second = {0};
    if (next_op &&
        loom_branch_fusion_read_if_info(context->module, op, &first)) {
      ++context->statistics->branches_visited;
      if (loom_branch_fusion_read_if_info(context->module, next_op, &second)) {
        if (loom_branch_fusion_candidate_is_legal(&first, &second)) {
          loom_op_t* fused_if = NULL;
          IREE_RETURN_IF_ERROR(loom_branch_fusion_fuse_pair(
              context, &first, &second, &fused_if));
          ++context->statistics->branches_fused;
          *out_changed = true;
          op = fused_if;
          continue;
        }
        ++context->statistics->candidates_rejected;
      }
    }

    IREE_RETURN_IF_ERROR(loom_branch_fusion_push_child_regions(context, op));
    op = next_op;
  }
  return iree_ok_status();
}

static iree_status_t loom_branch_fusion_process_function_once(
    loom_branch_fusion_context_t* context, loom_func_like_t function,
    bool* out_changed) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) return iree_ok_status();

  context->region_stack.count = 0;
  IREE_RETURN_IF_ERROR(loom_branch_fusion_region_stack_push(
      context->pass->arena, &context->region_stack, body));

  while (true) {
    loom_region_t* region =
        loom_branch_fusion_region_stack_pop(&context->region_stack);
    if (!region) break;

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      IREE_RETURN_IF_ERROR(
          loom_branch_fusion_process_block(context, block, out_changed));
    }
  }

  return iree_ok_status();
}

iree_status_t loom_branch_fusion_run(loom_pass_t* pass, loom_module_t* module,
                                     loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));

  iree_arena_allocator_t fusion_arena = {0};
  iree_arena_initialize(pass->arena->block_pool, &fusion_arena);

  loom_branch_fusion_context_t context = {
      .pass = pass,
      .statistics = loom_branch_fusion_statistics(pass),
      .module = module,
      .rewriter = &rewriter,
      .fusion_arena = &fusion_arena,
  };
  iree_status_t status = loom_branch_fusion_region_stack_initialize(
      pass->arena, &context.region_stack);

  bool changed = true;
  bool any_changed = false;
  while (iree_status_is_ok(status) && changed) {
    changed = false;
    status =
        loom_branch_fusion_process_function_once(&context, function, &changed);
    any_changed |= changed;
  }

  if (iree_status_is_ok(status) && any_changed) {
    loom_pass_mark_changed(pass);
  }
  loom_rewriter_deinitialize(&rewriter);
  iree_arena_deinitialize(&fusion_arena);
  return status;
}
