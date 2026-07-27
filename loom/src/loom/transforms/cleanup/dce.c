// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/dce.h"

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

static const loom_pass_statistic_field_t kDCEStatisticFields[] = {
    LOOM_DCE_STATISTICS(LOOM_PASS_STATISTIC_LAYOUT_FIELD_,
                        loom_dce_statistics_t)};

const loom_pass_statistic_layout_t loom_dce_statistic_layout =
    LOOM_PASS_STATISTIC_LAYOUT(loom_dce_statistics_t, kDCEStatisticFields);

static const loom_pass_info_t loom_dce_pass_info_storage = {
    .name = IREE_SVL("dce"),
    .description = IREE_SVL("Remove operations with unused results."),
    .kind = LOOM_PASS_FUNCTION,
    .statistic_layout = &loom_dce_statistic_layout,
};

const loom_pass_info_t* loom_dce_pass_info(void) {
  return &loom_dce_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Worklist
//===----------------------------------------------------------------------===//
//
// DCE is driven by a deduplicated worklist so cascading deadness follows only
// the values whose use counts changed. A fixed-point whole-function scan can
// require one full scan per dependency layer when producers and consumers live
// in different blocks or regions.

#define LOOM_DCE_INITIAL_CAPACITY 32

typedef struct loom_dce_worklist_t {
  // Deduplicated operations to re-check for deadness.
  loom_op_t** entries;
  // Number of queued operations.
  iree_host_size_t count;
  // Capacity of entries.
  iree_host_size_t capacity;
} loom_dce_worklist_t;

static iree_status_t loom_dce_worklist_initialize(
    iree_arena_allocator_t* arena, loom_dce_worklist_t* worklist) {
  worklist->count = 0;
  worklist->capacity = LOOM_DCE_INITIAL_CAPACITY;
  return iree_arena_allocate_array(arena, worklist->capacity,
                                   sizeof(loom_op_t*),
                                   (void**)&worklist->entries);
}

static void loom_dce_worklist_deinitialize(loom_dce_worklist_t* worklist) {
  for (iree_host_size_t i = 0; i < worklist->count; ++i) {
    worklist->entries[i]->flags &= ~LOOM_OP_FLAG_ON_WORKLIST;
  }
}

static iree_status_t loom_dce_worklist_push(iree_arena_allocator_t* arena,
                                            loom_dce_worklist_t* worklist,
                                            loom_op_t* op) {
  if (!op || iree_any_bit_set(op->flags,
                              LOOM_OP_FLAG_DEAD | LOOM_OP_FLAG_ON_WORKLIST)) {
    return iree_ok_status();
  }
  if (worklist->count >= worklist->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, worklist->count, worklist->count + 1, sizeof(loom_op_t*),
        &worklist->capacity, (void**)&worklist->entries));
  }
  op->flags |= LOOM_OP_FLAG_ON_WORKLIST;
  worklist->entries[worklist->count++] = op;
  return iree_ok_status();
}

static loom_op_t* loom_dce_worklist_pop(loom_dce_worklist_t* worklist) {
  while (worklist->count > 0) {
    loom_op_t* op = worklist->entries[--worklist->count];
    op->flags &= ~LOOM_OP_FLAG_ON_WORKLIST;
    if (iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
      continue;
    }
    return op;
  }
  return NULL;
}

//===----------------------------------------------------------------------===//
// Worklist seeding
//===----------------------------------------------------------------------===//

static iree_status_t loom_dce_seed_worklist(
    iree_arena_allocator_t* arena, loom_module_t* module, loom_region_t* body,
    const loom_dce_deadness_query_callback_t* deadness_query,
    loom_dce_worklist_t* worklist) {
  loom_region_t** region_stack = NULL;
  iree_host_size_t stack_count = 0;
  iree_host_size_t stack_capacity = LOOM_DCE_INITIAL_CAPACITY;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, stack_capacity, sizeof(loom_region_t*), (void**)&region_stack));

  region_stack[stack_count++] = body;

  while (stack_count > 0) {
    loom_region_t* region = region_stack[--stack_count];

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        op->flags &= ~LOOM_OP_FLAG_ON_WORKLIST;
        bool is_dead = false;
        IREE_RETURN_IF_ERROR(deadness_query->fn(deadness_query->user_data,
                                                module, op, &is_dead));
        if (is_dead) {
          IREE_RETURN_IF_ERROR(loom_dce_worklist_push(arena, worklist, op));
        }

        loom_region_t** regions = loom_op_regions(op);
        for (uint8_t r = 0; r < op->region_count; ++r) {
          if (!regions[r] || regions[r]->block_count == 0) {
            continue;
          }
          if (stack_count >= stack_capacity) {
            IREE_RETURN_IF_ERROR(iree_arena_grow_array(
                arena, stack_count, stack_count + 1, sizeof(loom_region_t*),
                &stack_capacity, (void**)&region_stack));
          }
          region_stack[stack_count++] = regions[r];
        }
      }
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Cascading enqueue helpers
//===----------------------------------------------------------------------===//

static iree_status_t loom_dce_add_operand_providers_to_worklist(
    iree_arena_allocator_t* arena, loom_module_t* module,
    loom_dce_worklist_t* worklist, loom_op_t* op) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    if (operands[i] == LOOM_VALUE_ID_INVALID) {
      continue;
    }
    loom_value_t* value = loom_module_value(module, operands[i]);
    if (loom_value_is_block_arg(value)) {
      continue;
    }
    loom_op_t* def = loom_value_def_op(value);
    IREE_RETURN_IF_ERROR(loom_dce_worklist_push(arena, worklist, def));
  }
  return iree_ok_status();
}

typedef struct loom_dce_type_ref_provider_context_t {
  // Module containing the value table and use-def links.
  loom_module_t* module;
  // Pass arena used for worklist growth.
  iree_arena_allocator_t* arena;
  // Candidate worklist to receive defining ops.
  loom_dce_worklist_t* worklist;
} loom_dce_type_ref_provider_context_t;

static iree_status_t loom_dce_add_type_ref_provider_to_worklist(
    loom_value_id_t value_id, void* user_data) {
  loom_dce_type_ref_provider_context_t* context =
      (loom_dce_type_ref_provider_context_t*)user_data;
  if (value_id >= context->module->values.count) {
    return iree_ok_status();
  }
  loom_value_t* value = loom_module_value(context->module, value_id);
  if (loom_value_is_block_arg(value)) {
    return iree_ok_status();
  }
  return loom_dce_worklist_push(context->arena, context->worklist,
                                loom_value_def_op(value));
}

// Queues providers that may become dead after erasing |op| and its nested
// regions. Region operands are ordinary uses, so erasing only the parent op's
// operands would leave captured values live through unreachable child ops.
// Type references on result types and block arguments are also liveness edges,
// so their providers must be rechecked when the carrier subtree disappears.
static iree_status_t loom_dce_add_subtree_operand_providers_to_worklist(
    iree_arena_allocator_t* arena, loom_module_t* module,
    loom_dce_worklist_t* worklist, loom_op_t* op) {
  IREE_RETURN_IF_ERROR(
      loom_dce_add_operand_providers_to_worklist(arena, module, worklist, op));

  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t i = 0; i < op->region_count; ++i) {
    loom_region_t* region = regions[i];
    if (!region) {
      continue;
    }
    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        IREE_RETURN_IF_ERROR(loom_dce_add_subtree_operand_providers_to_worklist(
            arena, module, worklist, child_op));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_dce_add_subtree_providers_to_worklist(
    iree_arena_allocator_t* arena, loom_module_t* module,
    loom_dce_worklist_t* worklist, loom_op_t* op) {
  IREE_RETURN_IF_ERROR(loom_dce_add_subtree_operand_providers_to_worklist(
      arena, module, worklist, op));
  loom_dce_type_ref_provider_context_t context = {
      .module = module,
      .arena = arena,
      .worklist = worklist,
  };
  return loom_op_walk_subtree_type_refs(
      module, op, loom_dce_add_type_ref_provider_to_worklist, &context);
}

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

static iree_status_t loom_dce_default_deadness_query(
    void* user_data, const loom_module_t* module, const loom_op_t* op,
    bool* out_is_dead) {
  (void)user_data;
  *out_is_dead = loom_op_is_trivially_dead(module, op);
  return iree_ok_status();
}

iree_status_t loom_dce_run(loom_pass_t* pass, loom_module_t* module,
                           loom_func_like_t function) {
  const loom_dce_deadness_query_callback_t deadness_query = {
      .fn = loom_dce_default_deadness_query,
  };
  return loom_dce_run_with_deadness_query(pass, module, function,
                                          deadness_query);
}

iree_status_t loom_dce_run_with_deadness_query(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function,
    loom_dce_deadness_query_callback_t deadness_query) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }
  loom_dce_statistics_t* statistics = loom_dce_statistics(pass);

  loom_dce_worklist_t worklist = {0};
  iree_status_t status = loom_dce_worklist_initialize(pass->arena, &worklist);
  for (uint8_t region_index = 0;
       region_index < loom_func_like_region_count(function) &&
       iree_status_is_ok(status);
       ++region_index) {
    loom_region_t* region = loom_func_like_region(function, region_index);
    if (!region) continue;
    status = loom_dce_seed_worklist(pass->arena, module, region,
                                    &deadness_query, &worklist);
  }

  while (iree_status_is_ok(status)) {
    loom_op_t* op = loom_dce_worklist_pop(&worklist);
    if (!op) {
      break;
    }
    bool is_dead = false;
    status = deadness_query.fn(deadness_query.user_data, module, op, &is_dead);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (!is_dead) {
      continue;
    }

    status = loom_dce_add_subtree_providers_to_worklist(pass->arena, module,
                                                        &worklist, op);
    if (!iree_status_is_ok(status)) {
      break;
    }
    status = loom_op_erase(module, op);
    if (!iree_status_is_ok(status)) {
      break;
    }
    loom_pass_mark_changed(pass);
    ++statistics->ops_eliminated;
  }

  loom_dce_worklist_deinitialize(&worklist);
  return status;
}
