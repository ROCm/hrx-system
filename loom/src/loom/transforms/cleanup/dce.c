// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/cleanup/dce.h"

#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/location_compaction.h"
#include "loom/ir/module.h"
#include "loom/ir/value_refs.h"
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

#define LOOM_LOCATION_DCE_STATISTICS(V, statistics_type)           \
  V(statistics_type, locations_eliminated, "locations-eliminated", \
    "Number of unreachable source-location rows removed from the module.")

LOOM_PASS_STATISTICS_DEFINE(loom_location_dce_statistics,
                            loom_location_dce_statistics_t,
                            LOOM_LOCATION_DCE_STATISTICS)

static const loom_pass_info_t loom_location_dce_pass_info_storage = {
    .name = IREE_SVL("location-dce"),
    .description = IREE_SVL("Remove unreachable source locations."),
    .kind = LOOM_PASS_MODULE,
    .statistic_layout = &loom_location_dce_statistics_layout,
};

const loom_pass_info_t* loom_location_dce_pass_info(void) {
  return &loom_location_dce_pass_info_storage;
}

iree_status_t loom_location_dce_run(loom_pass_t* pass, loom_module_t* module) {
  iree_host_size_t removed_count = 0;
  IREE_RETURN_IF_ERROR(
      loom_module_compact_locations(module, pass->arena, &removed_count));
  if (removed_count == 0) {
    return iree_ok_status();
  }
  loom_pass_mark_changed(pass);
  loom_location_dce_statistics(pass)->locations_eliminated += removed_count;
  return iree_ok_status();
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

typedef struct loom_dce_lifetime_region_t {
  // Region selected for ownership-aware cleanup during worklist seeding.
  loom_region_t* region;
  // Enclosing operation whose erasure retires the selected region.
  loom_op_t* owner;
} loom_dce_lifetime_region_t;

typedef struct loom_dce_worklist_t {
  // Deduplicated operations to re-check for deadness.
  loom_op_t** entries;
  // Number of queued operations.
  iree_host_size_t count;
  // Capacity of entries.
  iree_host_size_t capacity;
  // Regions containing operations whose deadness needs a lifetime closure.
  loom_dce_lifetime_region_t* lifetime_regions;
  // Number of regions selected during the initial structural walk.
  iree_host_size_t lifetime_region_count;
  // Allocated capacity of lifetime_regions.
  iree_host_size_t lifetime_region_capacity;
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
    loom_op_t* lifetime_owner = NULL;

    loom_block_t* block = NULL;
    loom_region_for_each_block(region, block) {
      loom_op_t* op = NULL;
      loom_block_for_each_op(block, op) {
        op->flags &= ~LOOM_OP_FLAG_ON_WORKLIST;
        const loom_dead_state_op_kind_t kind =
            deadness_query->fn(deadness_query->user_data, module, op);
        if (kind == LOOM_DEAD_STATE_OP_LIFETIME ||
            kind == LOOM_DEAD_STATE_OP_CONSUMING) {
          lifetime_owner = op->parent_op;
        }
        if (kind == LOOM_DEAD_STATE_OP_VALUE &&
            loom_op_results_unused(module, op)) {
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
    if (lifetime_owner) {
      IREE_RETURN_IF_ERROR(
          iree_arena_grow_array(arena, worklist->lifetime_region_count,
                                worklist->lifetime_region_count + 1,
                                sizeof(*worklist->lifetime_regions),
                                &worklist->lifetime_region_capacity,
                                (void**)&worklist->lifetime_regions));
      worklist->lifetime_regions[worklist->lifetime_region_count++] =
          (loom_dce_lifetime_region_t){region, lifetime_owner};
    }
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Cascading enqueue helpers
//===----------------------------------------------------------------------===//

typedef struct loom_dce_value_ref_provider_context_t {
  // Module containing the value table and use-def links.
  loom_module_t* module;
  // Pass arena used for worklist growth.
  iree_arena_allocator_t* arena;
  // Candidate worklist to receive defining ops.
  loom_dce_worklist_t* worklist;
} loom_dce_value_ref_provider_context_t;

static iree_status_t loom_dce_add_value_ref_provider_to_worklist(
    loom_value_id_t value_id, loom_value_id_t type_carrier, void* user_data) {
  (void)type_carrier;
  loom_dce_value_ref_provider_context_t* context =
      (loom_dce_value_ref_provider_context_t*)user_data;
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

static iree_status_t loom_dce_add_subtree_providers_to_worklist(
    iree_arena_allocator_t* arena, loom_module_t* module,
    loom_dce_worklist_t* worklist, loom_op_t* op) {
  loom_dce_value_ref_provider_context_t context = {
      .module = module,
      .arena = arena,
      .worklist = worklist,
  };
  return loom_op_walk_subtree_value_refs(
      module, op, loom_dce_add_value_ref_provider_to_worklist, &context);
}

//===----------------------------------------------------------------------===//
// Implementation
//===----------------------------------------------------------------------===//

iree_status_t loom_dce_run(loom_pass_t* pass, loom_module_t* module,
                           loom_func_like_t function) {
  const loom_dce_deadness_query_callback_t deadness_query = {
      .fn = loom_dead_state_classify_op,
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
    if (!region) {
      continue;
    }
    status = loom_dce_seed_worklist(pass->arena, module, region,
                                    &deadness_query, &worklist);
  }

  while (iree_status_is_ok(status)) {
    loom_op_t* op = loom_dce_worklist_pop(&worklist);
    if (!op) {
      break;
    }
    if (deadness_query.fn(deadness_query.user_data, module, op) !=
            LOOM_DEAD_STATE_OP_VALUE ||
        !loom_op_results_unused(module, op)) {
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
  iree_arena_allocator_t arena;
  iree_arena_initialize(pass->arena->block_pool, &arena);
  // Nested regions are independent scopes. Their captures remain live in the
  // enclosing scope until structured control is lowered to ordinary CFG.
  for (iree_host_size_t i = 0;
       i < worklist.lifetime_region_count && iree_status_is_ok(status); ++i) {
    const loom_dce_lifetime_region_t scope = worklist.lifetime_regions[i];
    // The worklist may have erased an enclosing operation and its subtree.
    if (scope.owner->flags & LOOM_OP_FLAG_DEAD) {
      continue;
    }
    loom_region_t* region = scope.region;
    iree_arena_reset(&arena);
    loom_local_value_domain_t domain = {0};
    status = loom_local_value_domain_acquire_for_region(module, region, &arena,
                                                        &domain);
    loom_cfg_graph_t graph = {0};
    if (iree_status_is_ok(status)) {
      status = loom_cfg_graph_build(module, region, &arena, &graph);
    }
    loom_rewriter_t rewriter;
    loom_rewriter_initialize(&rewriter, module, &arena);
    iree_host_size_t removed_arguments = 0;
    iree_host_size_t removed_operations = 0;
    if (iree_status_is_ok(status)) {
      status = loom_cfg_eliminate_dead_state(
          &rewriter, &graph, &domain, deadness_query.fn,
          deadness_query.user_data, &removed_arguments, &removed_operations);
    }
    loom_rewriter_deinitialize(&rewriter);
    loom_local_value_domain_release(&domain);
    if (removed_arguments || removed_operations) {
      loom_pass_mark_changed(pass);
      statistics->ops_eliminated += removed_operations;
    }
  }
  iree_arena_deinitialize(&arena);
  return status;
}
