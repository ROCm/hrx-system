// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/vector/sink_single_use_reads.h"

#include "loom/analysis/motion.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/rewriter.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

#define LOOM_SINK_SINGLE_USE_READS_STATISTICS(V, statistics_type)     \
  V(statistics_type, read_ops_checked, "read-ops-checked",            \
    "Number of read-only producer ops checked for sinking.")          \
  V(statistics_type, read_ops_sunk, "read-ops-sunk",                  \
    "Number of read-only producer ops sunk toward their single use.") \
  V(statistics_type, blocks_checked, "blocks-checked",                \
    "Number of blocks checked for sinkable read ops.")

LOOM_PASS_STATISTICS_DEFINE(loom_sink_single_use_reads_statistics,
                            loom_sink_single_use_reads_statistics_t,
                            LOOM_SINK_SINGLE_USE_READS_STATISTICS)

static const loom_pass_info_t loom_sink_single_use_reads_pass_info_storage = {
    .name = IREE_SVL("sink-single-use-reads"),
    .description = IREE_SVL(
        "Sink single-use multi-lane vector reads toward their sole same-block "
        "user."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_sink_single_use_reads_statistics_layout,
};

const loom_pass_info_t* loom_sink_single_use_reads_pass_info(void) {
  return &loom_sink_single_use_reads_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Sinking
//===----------------------------------------------------------------------===//

typedef struct loom_sink_single_use_reads_context_t {
  // Pass instance owning diagnostics, statistics, and scratch storage.
  loom_pass_t* pass;
  // Module being transformed.
  loom_module_t* module;
  // Rewriter used for IR movement.
  loom_rewriter_t* rewriter;
  // Typed statistics storage for this pass invocation.
  loom_sink_single_use_reads_statistics_t* statistics;
  // Reusable block operation list.
  loom_op_t** ops;
  // Barrier segment for each operation in ops.
  uint32_t* segments;
  // Read operation selected for movement.
  loom_op_t** move_ops;
  // User operation that each selected read should move before.
  loom_op_t** move_before_ops;
  // Allocated capacity of the per-block arrays above.
  iree_host_size_t op_capacity;
} loom_sink_single_use_reads_context_t;

typedef struct loom_sink_single_use_reads_region_stack_t {
  // Region pointers waiting to be processed.
  loom_region_t** regions;
  // Number of queued regions.
  iree_host_size_t count;
  // Allocated region pointer capacity.
  iree_host_size_t capacity;
} loom_sink_single_use_reads_region_stack_t;

#define LOOM_SINK_SINGLE_USE_READS_INITIAL_REGION_CAPACITY 16

static iree_status_t loom_sink_single_use_reads_region_stack_initialize(
    iree_arena_allocator_t* arena,
    loom_sink_single_use_reads_region_stack_t* stack) {
  stack->count = 0;
  stack->capacity = LOOM_SINK_SINGLE_USE_READS_INITIAL_REGION_CAPACITY;
  return iree_arena_allocate_array(
      arena, stack->capacity, sizeof(loom_region_t*), (void**)&stack->regions);
}

static iree_status_t loom_sink_single_use_reads_region_stack_push(
    iree_arena_allocator_t* arena,
    loom_sink_single_use_reads_region_stack_t* stack, loom_region_t* region) {
  if (!region || region->block_count == 0) return iree_ok_status();
  if (stack->count >= stack->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, stack->count, stack->count + 1, sizeof(loom_region_t*),
        &stack->capacity, (void**)&stack->regions));
  }
  stack->regions[stack->count++] = region;
  return iree_ok_status();
}

static loom_region_t* loom_sink_single_use_reads_region_stack_pop(
    loom_sink_single_use_reads_region_stack_t* stack) {
  return stack->count > 0 ? stack->regions[--stack->count] : NULL;
}

static iree_status_t loom_sink_single_use_reads_reserve_ops(
    loom_sink_single_use_reads_context_t* context, iree_host_size_t count) {
  if (count <= context->op_capacity) return iree_ok_status();
  if (context->op_capacity == 0) {
    context->op_capacity = count < 16 ? 16 : count;
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(context->pass->arena, context->op_capacity,
                                  sizeof(loom_op_t*), (void**)&context->ops));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, context->op_capacity, sizeof(uint32_t),
        (void**)&context->segments));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        context->pass->arena, context->op_capacity, sizeof(loom_op_t*),
        (void**)&context->move_ops));
    return iree_arena_allocate_array(context->pass->arena, context->op_capacity,
                                     sizeof(loom_op_t*),
                                     (void**)&context->move_before_ops);
  }

  iree_host_size_t old_capacity = context->op_capacity;
  iree_host_size_t grown_capacity = old_capacity;
  iree_host_size_t array_capacity = old_capacity;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      context->pass->arena, old_capacity, count, sizeof(loom_op_t*),
      &array_capacity, (void**)&context->ops));
  grown_capacity = array_capacity;
  array_capacity = old_capacity;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      context->pass->arena, old_capacity, count, sizeof(uint32_t),
      &array_capacity, (void**)&context->segments));
  if (array_capacity < grown_capacity) grown_capacity = array_capacity;
  array_capacity = old_capacity;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      context->pass->arena, old_capacity, count, sizeof(loom_op_t*),
      &array_capacity, (void**)&context->move_ops));
  if (array_capacity < grown_capacity) grown_capacity = array_capacity;
  array_capacity = old_capacity;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      context->pass->arena, old_capacity, count, sizeof(loom_op_t*),
      &array_capacity, (void**)&context->move_before_ops));
  if (array_capacity < grown_capacity) grown_capacity = array_capacity;
  context->op_capacity = grown_capacity;
  return iree_ok_status();
}

static bool loom_sink_single_use_reads_has_multi_lane_vector_result(
    const loom_sink_single_use_reads_context_t* context, loom_op_t* op) {
  loom_value_id_t result = loom_op_results(op)[0];
  loom_type_t result_type = loom_module_value_type(context->module, result);
  uint64_t element_count = 0;
  return loom_type_is_vector(result_type) &&
         loom_type_static_element_count(result_type, &element_count) &&
         element_count > 1;
}

static bool loom_sink_single_use_reads_is_read_candidate(
    const loom_sink_single_use_reads_context_t* context, loom_op_t* op) {
  if ((op->flags & LOOM_OP_FLAG_DEAD) || op->result_count != 1 ||
      op->region_count != 0) {
    return false;
  }
  if (!loom_sink_single_use_reads_has_multi_lane_vector_result(context, op)) {
    return false;
  }
  loom_memory_access_t access = loom_memory_access_cast(context->module, op);
  if (!loom_memory_access_isa(access)) return false;
  const loom_trait_flags_t traits =
      loom_op_effective_traits(context->module, op);
  if (!loom_traits_may_read(traits) || loom_traits_may_write(traits)) {
    return false;
  }
  if (iree_any_bit_set(traits, LOOM_TRAIT_HINT | LOOM_TRAIT_POISON_BOUNDARY |
                                   LOOM_TRAIT_CONVERGENT |
                                   LOOM_TRAIT_UNIQUE_IDENTITY)) {
    return false;
  }
  loom_value_id_t result = loom_op_results(op)[0];
  if (loom_module_value_has_predicate_attribute_uses(context->module, result) ||
      loom_module_value_has_type_uses(context->module, result)) {
    return false;
  }
  return loom_value_has_single_use(loom_module_value(context->module, result));
}

static bool loom_sink_single_use_reads_find_same_block_user(
    const loom_sink_single_use_reads_context_t* context, loom_op_t* op,
    loom_op_t** out_user_op) {
  *out_user_op = NULL;
  loom_value_id_t result = loom_op_results(op)[0];
  const loom_use_t* use =
      loom_value_single_use(loom_module_value(context->module, result));
  if (!use) return false;
  loom_op_t* user_op = loom_use_user_op(*use);
  if (!user_op || (user_op->flags & LOOM_OP_FLAG_DEAD)) return false;
  if (user_op->parent_block != op->parent_block || user_op == op ||
      user_op->block_ordinal <= op->block_ordinal) {
    return false;
  }
  const loom_trait_flags_t user_traits =
      loom_op_effective_traits(context->module, user_op);
  if (!iree_any_bit_set(user_traits, LOOM_TRAIT_PURE) ||
      loom_traits_may_read(user_traits) || loom_traits_may_write(user_traits) ||
      iree_any_bit_set(user_traits, LOOM_TRAIT_HINT |
                                        LOOM_TRAIT_POISON_BOUNDARY |
                                        LOOM_TRAIT_CONVERGENT |
                                        LOOM_TRAIT_UNIQUE_IDENTITY)) {
    return false;
  }
  *out_user_op = user_op;
  return true;
}

static iree_host_size_t loom_sink_single_use_reads_find_ordinal(
    loom_op_t** ops, iree_host_size_t count, uint64_t block_ordinal) {
  iree_host_size_t low = 0;
  iree_host_size_t high = count;
  while (low < high) {
    iree_host_size_t mid = low + ((high - low) >> 1);
    uint64_t mid_ordinal = ops[mid]->block_ordinal;
    if (mid_ordinal < block_ordinal) {
      low = mid + 1;
    } else {
      high = mid;
    }
  }
  return low < count && ops[low]->block_ordinal == block_ordinal
             ? low
             : IREE_HOST_SIZE_MAX;
}

static bool loom_sink_single_use_reads_user_has_unique_read_producer(
    loom_sink_single_use_reads_context_t* context, iree_host_size_t count,
    iree_host_size_t candidate_index, iree_host_size_t user_index) {
  loom_op_t* candidate_op = context->ops[candidate_index];
  loom_op_t* user_op = context->ops[user_index];
  const loom_value_id_t* operands = loom_op_const_operands(user_op);
  iree_host_size_t read_producer_count = 0;
  bool candidate_found = false;
  for (uint16_t i = 0; i < user_op->operand_count; ++i) {
    loom_value_t* operand = loom_module_value(context->module, operands[i]);
    if (loom_value_is_block_arg(operand)) continue;
    loom_op_t* def_op = loom_value_def_op(operand);
    if (!def_op || def_op->parent_block != user_op->parent_block ||
        def_op->block_ordinal >= user_op->block_ordinal) {
      continue;
    }
    iree_host_size_t def_index = loom_sink_single_use_reads_find_ordinal(
        context->ops, count, def_op->block_ordinal);
    if (def_index == IREE_HOST_SIZE_MAX ||
        context->segments[def_index] != context->segments[user_index]) {
      continue;
    }
    if (!loom_sink_single_use_reads_is_read_candidate(context, def_op)) {
      continue;
    }
    ++read_producer_count;
    candidate_found |= def_op == candidate_op;
    if (read_producer_count > 1) return false;
  }
  return candidate_found && read_producer_count == 1;
}

static iree_status_t loom_sink_single_use_reads_collect_block(
    loom_sink_single_use_reads_context_t* context, loom_block_t* block,
    iree_host_size_t* out_count) {
  *out_count = 0;
  iree_host_size_t count = 0;
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (op->flags & LOOM_OP_FLAG_DEAD) continue;
    ++count;
  }
  if (count == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(loom_sink_single_use_reads_reserve_ops(context, count));

  uint32_t segment = 0;
  iree_host_size_t index = 0;
  loom_block_for_each_op(block, op) {
    if (op->flags & LOOM_OP_FLAG_DEAD) continue;
    context->ops[index] = op;
    context->segments[index] = segment;
    ++index;
    if (!loom_motion_read_can_cross_op(context->module, op)) {
      ++segment;
    }
  }
  *out_count = count;
  return iree_ok_status();
}

static iree_status_t loom_sink_single_use_reads_process_block(
    loom_sink_single_use_reads_context_t* context, loom_block_t* block) {
  ++context->statistics->blocks_checked;
  iree_host_size_t count = 0;
  IREE_RETURN_IF_ERROR(
      loom_sink_single_use_reads_collect_block(context, block, &count));
  if (count == 0) return iree_ok_status();

  iree_host_size_t move_count = 0;
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_op_t* op = context->ops[i];
    if (!loom_sink_single_use_reads_is_read_candidate(context, op)) {
      continue;
    }
    ++context->statistics->read_ops_checked;

    loom_op_t* user_op = NULL;
    if (!loom_sink_single_use_reads_find_same_block_user(context, op,
                                                         &user_op)) {
      continue;
    }
    iree_host_size_t user_index = loom_sink_single_use_reads_find_ordinal(
        context->ops, count, user_op->block_ordinal);
    if (user_index == IREE_HOST_SIZE_MAX || user_index == i + 1) {
      continue;
    }
    if (context->segments[i] != context->segments[user_index]) {
      continue;
    }
    if (!loom_sink_single_use_reads_user_has_unique_read_producer(
            context, count, i, user_index)) {
      continue;
    }

    context->move_ops[move_count] = op;
    context->move_before_ops[move_count] = user_op;
    ++move_count;
  }

  for (iree_host_size_t i = 0; i < move_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_move_before(
        context->rewriter, context->move_ops[i], context->move_before_ops[i]));
    ++context->statistics->read_ops_sunk;
  }
  if (move_count != 0) loom_pass_mark_changed(context->pass);
  return iree_ok_status();
}

static iree_status_t loom_sink_single_use_reads_push_child_regions(
    loom_sink_single_use_reads_context_t* context,
    loom_sink_single_use_reads_region_stack_t* stack, loom_op_t* op) {
  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_sink_single_use_reads_region_stack_push(
        context->pass->arena, stack, regions[i]));
  }
  return iree_ok_status();
}

iree_status_t loom_sink_single_use_reads_run(loom_pass_t* pass,
                                             loom_module_t* module,
                                             loom_func_like_t function) {
  if (!loom_func_like_body(function)) return iree_ok_status();

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  iree_status_t status = iree_ok_status();
  loom_sink_single_use_reads_context_t context = {
      .pass = pass,
      .module = module,
      .rewriter = &rewriter,
      .statistics = loom_sink_single_use_reads_statistics(pass),
  };

  loom_sink_single_use_reads_region_stack_t stack;
  status =
      loom_sink_single_use_reads_region_stack_initialize(pass->arena, &stack);
  if (iree_status_is_ok(status)) {
    status = loom_sink_single_use_reads_region_stack_push(
        pass->arena, &stack, loom_func_like_body(function));
  }

  while (iree_status_is_ok(status)) {
    loom_region_t* region = loom_sink_single_use_reads_region_stack_pop(&stack);
    if (!region) break;

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      status = loom_sink_single_use_reads_process_block(&context, block);
      if (!iree_status_is_ok(status)) break;
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        if (op->flags & LOOM_OP_FLAG_DEAD) continue;
        status =
            loom_sink_single_use_reads_push_child_regions(&context, &stack, op);
        if (!iree_status_is_ok(status)) break;
      }
      if (!iree_status_is_ok(status)) break;
    }
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
