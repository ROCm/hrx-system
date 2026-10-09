// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/kernel/promote_private_fragments.h"

#include <stdint.h>
#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/buffer/ops.h"
#include "loom/ops/cfg/ops.h"
#include "loom/ops/index/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/ops/vector/ops.h"
#include "loom/ops/view/ops.h"
#include "loom/pass/value_facts.h"
#include "loom/rewrite/rewriter.h"
#include "loom/transforms/kernel/private_storage.h"
#include "loom/util/dominance.h"
#include "loom/util/walk.h"

//===----------------------------------------------------------------------===//
// Statistics
//===----------------------------------------------------------------------===//

#define LOOM_PROMOTE_PRIVATE_FRAGMENTS_STATISTICS(V, statistics_type)  \
  V(statistics_type, fragments_promoted, "fragments-promoted",         \
    "Number of private fragment views promoted to SSA vectors.")       \
  V(statistics_type, loads_promoted, "loads-promoted",                 \
    "Number of private scalar loads replaced with vector.extract.")    \
  V(statistics_type, loads_forwarded, "loads-forwarded",               \
    "Number of private scalar loads replaced by reaching SSA values.") \
  V(statistics_type, noop_stores_erased, "noop-stores-erased",         \
    "Number of private load-store no-op pairs erased.")

LOOM_PASS_STATISTICS_DEFINE(loom_promote_private_fragments_statistics,
                            loom_promote_private_fragments_statistics_t,
                            LOOM_PROMOTE_PRIVATE_FRAGMENTS_STATISTICS)

static const loom_pass_info_t loom_promote_private_fragments_pass_info_storage =
    {
        .name = IREE_SVL("promote-private-fragments"),
        .description =
            IREE_SVL("Promote private scalar cells and fragments to SSA."),
        .kind = LOOM_PASS_FUNCTION,
        .statistic_layout = &loom_promote_private_fragments_statistics_layout,
};

const loom_pass_info_t* loom_promote_private_fragments_pass_info(void) {
  return &loom_promote_private_fragments_pass_info_storage;
}

//===----------------------------------------------------------------------===//
// Candidate collection
//===----------------------------------------------------------------------===//

#define LOOM_PROMOTE_PRIVATE_FRAGMENTS_INITIAL_VIEW_CAPACITY 16

typedef struct loom_promote_private_fragments_view_list_t {
  // Collected buffer.view operations in dominance order.
  loom_op_t** ops;
  // Number of collected view operations.
  iree_host_size_t count;
  // Allocated view operation pointer capacity.
  iree_host_size_t capacity;
} loom_promote_private_fragments_view_list_t;

typedef struct loom_promote_private_fragments_collect_context_t {
  // Pass scratch arena used for the collected view pointer list.
  iree_arena_allocator_t* arena;
  // Collected buffer.view operations.
  loom_promote_private_fragments_view_list_t* views;
} loom_promote_private_fragments_collect_context_t;

static iree_status_t loom_promote_private_fragments_view_list_initialize(
    iree_arena_allocator_t* arena,
    loom_promote_private_fragments_view_list_t* list) {
  list->count = 0;
  list->capacity = LOOM_PROMOTE_PRIVATE_FRAGMENTS_INITIAL_VIEW_CAPACITY;
  return iree_arena_allocate_array(arena, list->capacity, sizeof(loom_op_t*),
                                   (void**)&list->ops);
}

static iree_status_t loom_promote_private_fragments_view_list_push(
    iree_arena_allocator_t* arena,
    loom_promote_private_fragments_view_list_t* list, loom_op_t* op) {
  if (list->count >= list->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, list->count, list->count + 1, sizeof(loom_op_t*),
        &list->capacity, (void**)&list->ops));
  }
  list->ops[list->count++] = op;
  return iree_ok_status();
}

static iree_status_t loom_promote_private_fragments_collect_view(
    void* user_data, loom_op_t* op, const loom_walk_context_t* context,
    loom_walk_result_t* out_result) {
  loom_promote_private_fragments_collect_context_t* collect_context =
      (loom_promote_private_fragments_collect_context_t*)user_data;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_buffer_view_isa(op)) {
    return iree_ok_status();
  }
  return loom_promote_private_fragments_view_list_push(
      collect_context->arena, collect_context->views, op);
}

//===----------------------------------------------------------------------===//
// Pattern model
//===----------------------------------------------------------------------===//

typedef struct loom_promote_private_fragments_copy_loop_t {
  // Copy loop to erase after materializing the vector load.
  loom_op_t* loop_op;
  // Scalar source load inside the copy loop.
  loom_op_t* source_load_op;
  // Scalar store to the private fragment inside the copy loop.
  loom_op_t* private_store_op;
  // Source view loaded by the copy loop.
  loom_value_id_t source_view;
  // Loop induction variable.
  loom_value_id_t induction_variable;
  // Static vector.load origin indices.
  int64_t static_indices[LOOM_TYPE_MAX_RANK];
  // Number of static vector.load origin indices.
  uint16_t static_index_count;
  // Dynamic vector.load origin operands.
  loom_value_id_t indices[LOOM_TYPE_MAX_RANK];
  // Number of dynamic vector.load origin operands.
  uint16_t index_count;
  // Result type of the replacement vector.load.
  loom_type_t vector_type;
} loom_promote_private_fragments_copy_loop_t;

typedef struct loom_promote_private_fragments_op_list_t {
  // Collected private memory operations.
  loom_op_t** ops;
  // Number of collected operations.
  iree_host_size_t count;
  // Allocated operation pointer capacity.
  iree_host_size_t capacity;
} loom_promote_private_fragments_op_list_t;

static iree_status_t loom_promote_private_fragments_op_list_initialize(
    iree_arena_allocator_t* arena,
    loom_promote_private_fragments_op_list_t* list) {
  list->count = 0;
  list->capacity = 4;
  return iree_arena_allocate_array(arena, list->capacity, sizeof(loom_op_t*),
                                   (void**)&list->ops);
}

static iree_status_t loom_promote_private_fragments_op_list_push(
    iree_arena_allocator_t* arena,
    loom_promote_private_fragments_op_list_t* list, loom_op_t* op) {
  if (list->count >= list->capacity) {
    iree_host_size_t new_count = list->count + 1;
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(arena, list->count, new_count, sizeof(loom_op_t*),
                              &list->capacity, (void**)&list->ops));
  }
  list->ops[list->count++] = op;
  return iree_ok_status();
}

static bool loom_promote_private_fragments_exact_index_constant(
    const loom_module_t* module, loom_value_id_t value_id, int64_t* out_value) {
  if (value_id == LOOM_VALUE_ID_INVALID) {
    return false;
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return false;
  }
  const loom_op_t* op = loom_value_def_op(value);
  if (!loom_index_constant_isa(op)) {
    return false;
  }
  loom_attribute_t attr = loom_index_constant_value(op);
  if (attr.kind != LOOM_ATTR_I64) {
    return false;
  }
  *out_value = attr.i64;
  return true;
}

static bool loom_promote_private_fragments_is_private_rank1_view(
    const loom_module_t* module, const loom_op_t* view_op, int64_t* out_length,
    loom_op_t** out_alloca_op) {
  if (!loom_buffer_view_isa(view_op)) {
    return false;
  }
  loom_value_id_t view = loom_buffer_view_result(view_op);
  loom_type_t view_type = loom_module_value_type(module, view);
  if (!loom_type_is_view(view_type) || loom_type_rank(view_type) != 1 ||
      loom_type_dim_is_dynamic_at(view_type, 0)) {
    return false;
  }
  int64_t length = loom_type_dim_static_size_at(view_type, 0);
  if (length <= 0) {
    return false;
  }

  const loom_value_t* buffer_value =
      loom_module_value(module, loom_buffer_view_buffer(view_op));
  if (loom_value_is_block_arg(buffer_value)) {
    return false;
  }
  // Promotion reasons about this view's users only. Another storage user can
  // observe erased writes or clobber values forwarded through the view.
  if (!loom_value_has_single_use(buffer_value)) {
    return false;
  }
  loom_op_t* alloca_op = loom_value_def_op(buffer_value);
  if (!loom_buffer_alloca_isa(alloca_op)) {
    return false;
  }
  if (loom_buffer_alloca_memory_space(alloca_op) !=
      LOOM_VALUE_FACT_MEMORY_SPACE_PRIVATE) {
    return false;
  }

  *out_length = length;
  *out_alloca_op = alloca_op;
  return true;
}

static bool loom_promote_private_fragments_is_rank1_dynamic_index(
    loom_attribute_t static_indices, loom_value_slice_t indices,
    loom_value_id_t index) {
  return static_indices.kind == LOOM_ATTR_I64_ARRAY &&
         static_indices.count == 1 &&
         static_indices.i64_array[0] == INT64_MIN && indices.count == 1 &&
         indices.values[0] == index;
}

static bool loom_promote_private_fragments_is_observable(
    const loom_module_t* module, const loom_op_t* op) {
  return iree_any_bit_set(loom_op_effective_traits(module, op),
                          LOOM_TRAIT_OBSERVABLE_EFFECT);
}

static bool loom_promote_private_fragments_source_origin_from_copy_load(
    const loom_module_t* module, const loom_op_t* source_load_op,
    loom_value_id_t induction_variable, int64_t fragment_length,
    loom_promote_private_fragments_copy_loop_t* out_copy_loop) {
  loom_value_id_t source_view = loom_view_load_view(source_load_op);
  loom_type_t source_view_type = loom_module_value_type(module, source_view);
  if (!loom_type_is_view(source_view_type)) {
    return false;
  }
  uint8_t source_rank = loom_type_rank(source_view_type);
  if (source_rank == 0 || source_rank > LOOM_TYPE_MAX_RANK) {
    return false;
  }
  uint8_t vector_axis = (uint8_t)(source_rank - 1);
  if (loom_type_dim_is_dynamic_at(source_view_type, vector_axis) ||
      loom_type_dim_static_size_at(source_view_type, vector_axis) !=
          fragment_length) {
    return false;
  }

  loom_attribute_t source_static_indices =
      loom_view_load_static_indices(source_load_op);
  loom_value_slice_t source_indices = loom_view_load_indices(source_load_op);
  if (source_static_indices.kind != LOOM_ATTR_I64_ARRAY ||
      source_static_indices.count != source_rank) {
    return false;
  }

  uint16_t dynamic_index_ordinal = 0;
  bool found_induction_variable = false;
  for (uint8_t axis = 0; axis < source_rank; ++axis) {
    int64_t static_index = source_static_indices.i64_array[axis];
    if (static_index != INT64_MIN) {
      out_copy_loop->static_indices[axis] = static_index;
      continue;
    }
    if (dynamic_index_ordinal >= source_indices.count) {
      return false;
    }
    loom_value_id_t dynamic_index =
        source_indices.values[dynamic_index_ordinal];
    if (dynamic_index == induction_variable) {
      if (axis != vector_axis || found_induction_variable) {
        return false;
      }
      out_copy_loop->static_indices[axis] = 0;
      found_induction_variable = true;
    } else {
      out_copy_loop->static_indices[axis] = INT64_MIN;
      out_copy_loop->indices[out_copy_loop->index_count++] = dynamic_index;
    }
    ++dynamic_index_ordinal;
  }
  if (!found_induction_variable ||
      dynamic_index_ordinal != source_indices.count) {
    return false;
  }

  loom_type_t source_result_type =
      loom_module_value_type(module, loom_view_load_result(source_load_op));
  if (!loom_type_is_scalar(source_result_type)) {
    return false;
  }
  out_copy_loop->source_view = source_view;
  out_copy_loop->static_index_count = source_rank;
  out_copy_loop->vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, loom_type_element_type(source_result_type),
      loom_dim_pack_static(fragment_length), 0);
  return true;
}

static bool loom_promote_private_fragments_loop_body_matches(
    const loom_op_t* loop_op, const loom_op_t* source_load_op,
    const loom_op_t* private_store_op) {
  loom_region_t* body = loom_scf_for_body(loop_op);
  if (!body || body->block_count != 1) {
    return false;
  }
  loom_block_t* block = loom_region_entry_block(body);
  iree_host_size_t non_terminator_count = 0;
  bool saw_source_load = false;
  bool saw_private_store = false;
  loom_op_t* op = NULL;
  loom_block_for_each_op(block, op) {
    if (loom_scf_yield_isa(op)) {
      continue;
    }
    ++non_terminator_count;
    if (op == source_load_op) {
      saw_source_load = true;
    } else if (op == private_store_op) {
      saw_private_store = true;
    } else {
      return false;
    }
  }
  return non_terminator_count == 2 && saw_source_load && saw_private_store &&
         source_load_op->next_op == private_store_op;
}

static bool loom_promote_private_fragments_read_copy_loop(
    const loom_module_t* module, loom_value_id_t private_view,
    loom_op_t* private_store_op, int64_t fragment_length,
    loom_promote_private_fragments_copy_loop_t* out_copy_loop) {
  memset(out_copy_loop, 0, sizeof(*out_copy_loop));

  loom_op_t* loop_op = private_store_op->parent_op;
  if (!loom_scf_for_isa(loop_op)) {
    return false;
  }
  loom_region_t* body = loom_scf_for_body(loop_op);
  if (!body || body->block_count != 1 ||
      loom_region_entry_arg_count(body) == 0) {
    return false;
  }
  if (loom_scf_for_iter_args(loop_op).count != 0 ||
      loom_scf_for_results(loop_op).count != 0) {
    return false;
  }

  int64_t lower_bound = 0;
  int64_t upper_bound = 0;
  int64_t step = 0;
  if (!loom_promote_private_fragments_exact_index_constant(
          module, loom_scf_for_lower_bound(loop_op), &lower_bound) ||
      !loom_promote_private_fragments_exact_index_constant(
          module, loom_scf_for_upper_bound(loop_op), &upper_bound) ||
      !loom_promote_private_fragments_exact_index_constant(
          module, loom_scf_for_step(loop_op), &step)) {
    return false;
  }
  if (lower_bound != 0 || upper_bound != fragment_length || step != 1) {
    return false;
  }

  loom_value_id_t induction_variable = loom_region_entry_arg_id(body, 0);
  if (!loom_promote_private_fragments_is_rank1_dynamic_index(
          loom_view_store_static_indices(private_store_op),
          loom_view_store_indices(private_store_op), induction_variable)) {
    return false;
  }

  loom_value_id_t stored_value = loom_view_store_value(private_store_op);
  const loom_value_t* stored = loom_module_value(module, stored_value);
  if (loom_value_is_block_arg(stored)) {
    return false;
  }
  loom_op_t* source_load_op = loom_value_def_op(stored);
  if (!loom_view_load_isa(source_load_op)) {
    return false;
  }
  if (loom_promote_private_fragments_is_observable(module, source_load_op) ||
      loom_promote_private_fragments_is_observable(module, private_store_op)) {
    return false;
  }
  if (source_load_op->parent_op != loop_op ||
      source_load_op->parent_block != private_store_op->parent_block) {
    return false;
  }
  if (!loom_value_has_single_use(stored)) {
    return false;
  }
  if (!loom_promote_private_fragments_loop_body_matches(loop_op, source_load_op,
                                                        private_store_op)) {
    return false;
  }

  out_copy_loop->loop_op = loop_op;
  out_copy_loop->source_load_op = source_load_op;
  out_copy_loop->private_store_op = private_store_op;
  out_copy_loop->induction_variable = induction_variable;
  if (!loom_promote_private_fragments_source_origin_from_copy_load(
          module, source_load_op, induction_variable, fragment_length,
          out_copy_loop)) {
    return false;
  }
  return loom_view_store_view(private_store_op) == private_view;
}

static iree_status_t loom_promote_private_fragments_collect_uses(
    loom_pass_t* pass, loom_module_t* module, loom_value_id_t private_view,
    int64_t fragment_length,
    loom_promote_private_fragments_copy_loop_t* out_copy_loop,
    loom_promote_private_fragments_op_list_t* out_loads, bool* out_promotable) {
  *out_promotable = false;
  IREE_RETURN_IF_ERROR(loom_promote_private_fragments_op_list_initialize(
      pass->arena, out_loads));

  const loom_value_t* view_value = loom_module_value(module, private_view);
  bool found_copy_loop = false;
  const loom_use_t* use = NULL;
  loom_value_for_each_use(view_value, use) {
    loom_op_t* user_op = loom_use_user_op(*use);
    if (loom_view_load_isa(user_op) && loom_use_operand_index(*use) == 0) {
      if (loom_promote_private_fragments_is_observable(module, user_op)) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_promote_private_fragments_op_list_push(
          pass->arena, out_loads, user_op));
      continue;
    }
    if (loom_view_store_isa(user_op) && loom_use_operand_index(*use) == 1) {
      if (found_copy_loop) {
        return iree_ok_status();
      }
      if (!loom_promote_private_fragments_read_copy_loop(
              module, private_view, user_op, fragment_length, out_copy_loop)) {
        return iree_ok_status();
      }
      found_copy_loop = true;
      continue;
    }
    return iree_ok_status();
  }

  *out_promotable = found_copy_loop && out_loads->count > 0;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Rewrite
//===----------------------------------------------------------------------===//

typedef struct loom_promote_private_fragments_context_t {
  // Pass instance owning statistics and scratch allocations.
  loom_pass_t* pass;
  // Typed statistics storage for the current pass invocation.
  loom_promote_private_fragments_statistics_t* statistics;
  // Module being transformed.
  loom_module_t* module;
  // Dominance state for store-to-load forwarding queries.
  const loom_dominance_info_t* dominance;
  // Shared rewriter for all promotions.
  loom_rewriter_t* rewriter;
} loom_promote_private_fragments_context_t;

static iree_status_t loom_promote_private_fragments_replace_load(
    loom_promote_private_fragments_context_t* context, loom_op_t* load_op,
    loom_value_id_t vector_value) {
  loom_builder_set_before(&context->rewriter->builder, load_op);
  loom_value_id_t checkpoint =
      loom_rewriter_value_checkpoint(context->rewriter);
  loom_value_id_t result = loom_view_load_result(load_op);
  loom_type_t result_type = loom_module_value_type(context->module, result);
  loom_value_slice_t indices = loom_view_load_indices(load_op);
  loom_attribute_t static_indices = loom_view_load_static_indices(load_op);

  loom_op_t* extract_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_extract_build(
      &context->rewriter->builder, vector_value, indices.values, indices.count,
      static_indices.i64_array, static_indices.count, result_type,
      load_op->location, &extract_op));
  loom_value_id_t replacement = loom_vector_extract_result(extract_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      context->rewriter, load_op, &replacement, 1, checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      context->rewriter, load_op, &replacement, 1));
  ++context->statistics->loads_promoted;
  return iree_ok_status();
}

static iree_status_t loom_promote_private_fragments_try_erase_dead(
    loom_promote_private_fragments_context_t* context, loom_op_t* op) {
  if (!op || iree_any_bit_set(op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_ok_status();
  }
  bool erased = false;
  return loom_rewriter_erase_if_dead(context->rewriter, op, &erased);
}

static bool loom_promote_private_fragments_i64_array_equal(
    loom_attribute_t lhs, loom_attribute_t rhs) {
  if (lhs.kind != LOOM_ATTR_I64_ARRAY || rhs.kind != LOOM_ATTR_I64_ARRAY ||
      lhs.count != rhs.count) {
    return false;
  }
  for (uint16_t i = 0; i < lhs.count; ++i) {
    if (lhs.i64_array[i] != rhs.i64_array[i]) {
      return false;
    }
  }
  return true;
}

static bool loom_promote_private_fragments_value_slice_equal(
    loom_value_slice_t lhs, loom_value_slice_t rhs) {
  if (lhs.count != rhs.count) {
    return false;
  }
  for (uint16_t i = 0; i < lhs.count; ++i) {
    if (lhs.values[i] != rhs.values[i]) {
      return false;
    }
  }
  return true;
}

static bool loom_promote_private_fragments_load_matches_store_indices(
    const loom_op_t* store_op, const loom_op_t* load_op) {
  return loom_promote_private_fragments_i64_array_equal(
             loom_view_store_static_indices(store_op),
             loom_view_load_static_indices(load_op)) &&
         loom_promote_private_fragments_value_slice_equal(
             loom_view_store_indices(store_op),
             loom_view_load_indices(load_op));
}

static iree_status_t loom_promote_private_fragments_collect_single_store_uses(
    loom_pass_t* pass, loom_module_t* module, loom_value_id_t private_view,
    loom_op_t** out_store_op,
    loom_promote_private_fragments_op_list_t* out_loads, bool* out_promotable) {
  *out_store_op = NULL;
  *out_promotable = false;
  IREE_RETURN_IF_ERROR(loom_promote_private_fragments_op_list_initialize(
      pass->arena, out_loads));

  const loom_value_t* view_value = loom_module_value(module, private_view);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(view_value, use) {
    loom_op_t* user_op = loom_use_user_op(*use);
    if (loom_view_load_isa(user_op) && loom_use_operand_index(*use) == 0) {
      if (loom_promote_private_fragments_is_observable(module, user_op)) {
        return iree_ok_status();
      }
      IREE_RETURN_IF_ERROR(loom_promote_private_fragments_op_list_push(
          pass->arena, out_loads, user_op));
      continue;
    }
    if (loom_view_store_isa(user_op) && loom_use_operand_index(*use) == 1) {
      if (loom_promote_private_fragments_is_observable(module, user_op)) {
        return iree_ok_status();
      }
      if (*out_store_op) {
        return iree_ok_status();
      }
      *out_store_op = user_op;
      continue;
    }
    return iree_ok_status();
  }

  *out_promotable = *out_store_op && out_loads->count > 0;
  return iree_ok_status();
}

static bool loom_promote_private_fragments_can_forward_single_store(
    const loom_promote_private_fragments_context_t* context,
    const loom_op_t* store_op,
    const loom_promote_private_fragments_op_list_t* loads) {
  loom_value_id_t stored_value = loom_view_store_value(store_op);
  for (iree_host_size_t i = 0; i < loads->count; ++i) {
    loom_op_t* load_op = loads->ops[i];
    if (!loom_promote_private_fragments_load_matches_store_indices(store_op,
                                                                   load_op)) {
      return false;
    }
    if (!loom_dominates_op(context->dominance, store_op, load_op)) {
      return false;
    }
    if (!loom_dominates_value(context->dominance, stored_value, load_op)) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_promote_private_fragments_forward_single_store(
    loom_promote_private_fragments_context_t* context, loom_op_t* view_op,
    loom_op_t* alloca_op, loom_op_t* store_op,
    const loom_promote_private_fragments_op_list_t* loads) {
  loom_value_id_t stored_value = loom_view_store_value(store_op);
  for (iree_host_size_t i = 0; i < loads->count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
        context->rewriter, loads->ops[i], &stored_value, 1));
  }

  IREE_RETURN_IF_ERROR(loom_rewriter_erase(context->rewriter, store_op));
  IREE_RETURN_IF_ERROR(
      loom_promote_private_fragments_try_erase_dead(context, view_op));
  IREE_RETURN_IF_ERROR(
      loom_promote_private_fragments_try_erase_dead(context, alloca_op));

  context->statistics->loads_forwarded += loads->count;
  return iree_ok_status();
}

static iree_status_t loom_promote_private_fragments_name_vector_value(
    loom_promote_private_fragments_context_t* context,
    loom_value_id_t private_view, loom_value_id_t vector_value) {
  return loom_rewriter_try_set_derived_value_name(
      context->rewriter, private_view, vector_value, IREE_SV("vector"));
}

static iree_status_t loom_promote_private_fragments_promote(
    loom_promote_private_fragments_context_t* context, loom_op_t* view_op,
    loom_op_t* alloca_op,
    const loom_promote_private_fragments_copy_loop_t* copy_loop,
    const loom_promote_private_fragments_op_list_t* loads) {
  loom_builder_set_before(&context->rewriter->builder, copy_loop->loop_op);
  int64_t* static_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->rewriter->builder.arena, copy_loop->static_index_count,
      sizeof(int64_t), (void**)&static_indices));
  memcpy(static_indices, copy_loop->static_indices,
         copy_loop->static_index_count * sizeof(int64_t));

  loom_op_t* vector_load_op = NULL;
  IREE_RETURN_IF_ERROR(loom_vector_load_build(
      &context->rewriter->builder, 0, /*instance_flags=*/0,
      copy_loop->source_view, copy_loop->indices, copy_loop->index_count,
      static_indices, copy_loop->static_index_count, 0, 0,
      copy_loop->vector_type, copy_loop->source_load_op->location,
      &vector_load_op));
  loom_value_id_t vector_value = loom_vector_load_result(vector_load_op);
  IREE_RETURN_IF_ERROR(loom_promote_private_fragments_name_vector_value(
      context, loom_buffer_view_result(view_op), vector_value));

  for (iree_host_size_t i = 0; i < loads->count; ++i) {
    IREE_RETURN_IF_ERROR(loom_promote_private_fragments_replace_load(
        context, loads->ops[i], vector_value));
  }

  IREE_RETURN_IF_ERROR(
      loom_rewriter_erase(context->rewriter, copy_loop->loop_op));
  IREE_RETURN_IF_ERROR(
      loom_promote_private_fragments_try_erase_dead(context, view_op));
  IREE_RETURN_IF_ERROR(
      loom_promote_private_fragments_try_erase_dead(context, alloca_op));

  ++context->statistics->fragments_promoted;
  return iree_ok_status();
}

static iree_status_t loom_promote_private_fragments_try_single_store_forward(
    loom_promote_private_fragments_context_t* context, loom_op_t* view_op,
    loom_op_t* alloca_op, loom_value_id_t private_view, bool* out_changed) {
  loom_op_t* store_op = NULL;
  loom_promote_private_fragments_op_list_t loads = {0};
  bool promotable = false;
  IREE_RETURN_IF_ERROR(loom_promote_private_fragments_collect_single_store_uses(
      context->pass, context->module, private_view, &store_op, &loads,
      &promotable));
  if (!promotable || !loom_promote_private_fragments_can_forward_single_store(
                         context, store_op, &loads)) {
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_promote_private_fragments_forward_single_store(
      context, view_op, alloca_op, store_op, &loads));
  *out_changed = true;
  return iree_ok_status();
}

static bool loom_promote_private_fragments_rank1_static_slot_index(
    const loom_module_t* module, loom_attribute_t static_indices,
    loom_value_slice_t indices, int64_t fragment_length,
    int64_t* out_slot_index) {
  *out_slot_index = 0;
  if (static_indices.kind != LOOM_ATTR_I64_ARRAY || static_indices.count != 1) {
    return false;
  }
  int64_t slot_index = static_indices.i64_array[0];
  if (slot_index == INT64_MIN) {
    if (indices.count != 1) {
      return false;
    }
    if (!loom_promote_private_fragments_exact_index_constant(
            module, indices.values[0], &slot_index)) {
      return false;
    }
  } else if (indices.count != 0) {
    return false;
  }
  if (slot_index < 0 || slot_index >= fragment_length) {
    return false;
  }
  *out_slot_index = slot_index;
  return true;
}

static bool loom_promote_private_fragments_is_static_slot_load(
    const loom_module_t* module, const loom_op_t* op,
    loom_value_id_t private_view, int64_t fragment_length,
    int64_t* out_slot_index) {
  return loom_view_load_isa(op) && loom_view_load_view(op) == private_view &&
         loom_promote_private_fragments_rank1_static_slot_index(
             module, loom_view_load_static_indices(op),
             loom_view_load_indices(op), fragment_length, out_slot_index);
}

static bool loom_promote_private_fragments_is_static_slot_store(
    const loom_module_t* module, const loom_op_t* op,
    loom_value_id_t private_view, int64_t fragment_length,
    int64_t* out_slot_index) {
  return loom_view_store_isa(op) && loom_view_store_view(op) == private_view &&
         loom_promote_private_fragments_rank1_static_slot_index(
             module, loom_view_store_static_indices(op),
             loom_view_store_indices(op), fragment_length, out_slot_index);
}

static iree_status_t
loom_promote_private_fragments_try_erase_static_slot_noop_stores(
    loom_promote_private_fragments_context_t* context,
    loom_value_id_t private_view, int64_t fragment_length, bool* out_changed) {
  *out_changed = false;

  loom_promote_private_fragments_op_list_t loads = {0};
  loom_promote_private_fragments_op_list_t stores = {0};
  IREE_RETURN_IF_ERROR(loom_promote_private_fragments_op_list_initialize(
      context->pass->arena, &loads));
  IREE_RETURN_IF_ERROR(loom_promote_private_fragments_op_list_initialize(
      context->pass->arena, &stores));

  const loom_value_t* view_value =
      loom_module_value(context->module, private_view);
  const loom_use_t* use = NULL;
  loom_value_for_each_use(view_value, use) {
    loom_op_t* store_op = loom_use_user_op(*use);
    if (!loom_view_store_isa(store_op) || loom_use_operand_index(*use) != 1) {
      continue;
    }
    if (loom_promote_private_fragments_is_observable(context->module,
                                                     store_op)) {
      continue;
    }
    int64_t store_slot_index = 0;
    if (!loom_promote_private_fragments_is_static_slot_store(
            context->module, store_op, private_view, fragment_length,
            &store_slot_index)) {
      continue;
    }

    loom_value_id_t stored_value = loom_view_store_value(store_op);
    const loom_value_t* stored =
        loom_module_value(context->module, stored_value);
    if (loom_value_is_block_arg(stored) || !loom_value_has_single_use(stored)) {
      continue;
    }
    loom_op_t* load_op = loom_value_def_op(stored);
    int64_t load_slot_index = 0;
    if (!loom_promote_private_fragments_is_static_slot_load(
            context->module, load_op, private_view, fragment_length,
            &load_slot_index) ||
        load_slot_index != store_slot_index ||
        loom_promote_private_fragments_is_observable(context->module,
                                                     load_op)) {
      continue;
    }
    if (load_op->parent_block != store_op->parent_block ||
        load_op->next_op != store_op) {
      continue;
    }

    IREE_RETURN_IF_ERROR(loom_promote_private_fragments_op_list_push(
        context->pass->arena, &loads, load_op));
    IREE_RETURN_IF_ERROR(loom_promote_private_fragments_op_list_push(
        context->pass->arena, &stores, store_op));
  }

  for (iree_host_size_t i = 0; i < stores.count; ++i) {
    loom_op_t* store_op = stores.ops[i];
    loom_op_t* load_op = loads.ops[i];
    if (iree_any_bit_set(store_op->flags, LOOM_OP_FLAG_DEAD) ||
        iree_any_bit_set(load_op->flags, LOOM_OP_FLAG_DEAD)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(context->rewriter, store_op));
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(context->rewriter, load_op));
    ++context->statistics->noop_stores_erased;
    *out_changed = true;
  }
  return iree_ok_status();
}

static iree_status_t loom_promote_private_fragments_process_view(
    loom_promote_private_fragments_context_t* context, loom_op_t* view_op,
    bool* out_changed) {
  if (iree_any_bit_set(view_op->flags, LOOM_OP_FLAG_DEAD)) {
    return iree_ok_status();
  }

  int64_t fragment_length = 0;
  loom_op_t* alloca_op = NULL;
  if (!loom_promote_private_fragments_is_private_rank1_view(
          context->module, view_op, &fragment_length, &alloca_op)) {
    return iree_ok_status();
  }

  loom_value_id_t private_view = loom_buffer_view_result(view_op);
  IREE_RETURN_IF_ERROR(
      loom_promote_private_fragments_try_erase_static_slot_noop_stores(
          context, private_view, fragment_length, out_changed));
  if (*out_changed) {
    return iree_ok_status();
  }

  loom_promote_private_fragments_copy_loop_t copy_loop = {0};
  loom_promote_private_fragments_op_list_t loads = {0};
  bool promotable = false;
  IREE_RETURN_IF_ERROR(loom_promote_private_fragments_collect_uses(
      context->pass, context->module, private_view, fragment_length, &copy_loop,
      &loads, &promotable));
  if (!promotable) {
    return loom_promote_private_fragments_try_single_store_forward(
        context, view_op, alloca_op, private_view, out_changed);
  }

  IREE_RETURN_IF_ERROR(loom_promote_private_fragments_promote(
      context, view_op, alloca_op, &copy_loop, &loads));
  *out_changed = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Allocation-wide SSA transport
//===----------------------------------------------------------------------===//

static bool loom_private_storage_transport_selected(
    const loom_private_storage_transport_t* transport) {
  return transport->live && transport->cell->allocation->selected;
}

static iree_host_size_t loom_private_storage_signature_count(
    const loom_private_storage_signature_t* signature) {
  iree_host_size_t count = 0;
  for (loom_private_storage_transport_t* t = signature->transports; t;
       t = t->next) {
    count += loom_private_storage_transport_selected(t);
  }
  return count;
}

// Source leaves can name loads or old control values. Resolve the retained
// correspondence rather than following erased producers in the IR.
static loom_value_id_t loom_private_storage_replacement(
    loom_private_storage_plan_t* plan, loom_private_storage_value_t* value) {
  for (;;) {
    value = loom_private_storage_value_resolve(value);
    if (value->replacement != LOOM_VALUE_ID_INVALID) {
      return value->replacement;
    }
    const loom_value_ordinal_t ordinal =
        loom_local_value_domain_ordinal(&plan->domain, value->source);
    loom_private_storage_access_t* load = plan->loads[ordinal];
    if (!load || !load->cell->allocation->selected) {
      return value->source;
    }
    value = load->value;
  }
}

static iree_status_t loom_private_storage_replace_identity(
    loom_private_storage_plan_t* plan, loom_rewriter_t* rewriter,
    loom_value_id_t source, loom_value_id_t replacement) {
  const loom_value_ordinal_t ordinal =
      loom_local_value_domain_ordinal(&plan->domain, source);
  if (plan->values[ordinal]) {
    plan->values[ordinal]->replacement = replacement;
  }
  IREE_RETURN_IF_ERROR(
      loom_rewriter_copy_value_name(rewriter, source, replacement));
  return loom_rewriter_replace_all_uses_with(rewriter, source, replacement);
}

// Extends a verified control operation's operand/result tuple. Prefix types
// retain their identities until the caller substitutes the replacement results;
// that substitution also updates dependent types and attribute references.
static iree_status_t loom_private_storage_extend_op(
    loom_rewriter_t* rewriter, loom_op_t* source, uint16_t operand_position,
    const loom_value_id_t* appended_operands, uint16_t appended_operand_count,
    const loom_value_id_t* appended_results, uint16_t appended_result_count,
    uint8_t region_count, loom_op_t** out_op) {
  loom_builder_t* builder = &rewriter->builder;
  loom_builder_set_before(builder, source);
  IREE_RETURN_IF_ERROR(loom_builder_check_count_range(
      (iree_host_size_t)source->operand_count + appended_operand_count,
      UINT16_MAX, IREE_SV("promoted control operands")));
  IREE_RETURN_IF_ERROR(loom_builder_check_count_range(
      (iree_host_size_t)source->result_count + appended_result_count,
      UINT16_MAX, IREE_SV("promoted control results")));
  const loom_op_vtable_t* vtable = loom_op_vtable(rewriter->module, source);
  const uint8_t segment_count = loom_op_vtable_operand_segment_count(vtable);
  uint16_t segments[UINT8_MAX];
  if (segment_count) {
    memcpy(segments, loom_op_const_operand_segment_counts(source),
           segment_count * sizeof(*segments));
    loom_loop_like_t loop = loom_loop_like_cast(rewriter->module, source);
    segments[loop.vtable->iter_args_operand_field_index] +=
        appended_operand_count;
  }
  IREE_RETURN_IF_ERROR(loom_builder_allocate_segmented_op_with_successors(
      builder, source->kind, source->operand_count + appended_operand_count,
      segment_count ? segments : NULL, segment_count,
      source->result_count + appended_result_count, source->successor_count,
      region_count, source->tied_result_count, source->attribute_count,
      source->location, out_op));
  loom_op_t* op = *out_op;
  op->instance_flags = source->instance_flags;
  op->traits = source->traits;
  op->flags |= source->flags & LOOM_OP_SOURCE_PRESENTATION_FLAG_MASK;
  memcpy(loom_op_attrs(op), loom_op_const_attrs(source),
         source->attribute_count * sizeof(loom_attribute_t));
  memcpy(loom_op_successors(op), loom_op_const_successors(source),
         source->successor_count * sizeof(loom_block_t*));
  IREE_RETURN_IF_ERROR(loom_builder_copy_tied_results(
      loom_op_tied_results(source), source->tied_result_count, op));
  loom_value_id_t* operands = loom_op_operands(op);
  memcpy(operands, loom_op_const_operands(source),
         operand_position * sizeof(*operands));
  if (appended_operand_count) {
    memcpy(operands + operand_position, appended_operands,
           appended_operand_count * sizeof(*operands));
  }
  memcpy(operands + operand_position + appended_operand_count,
         loom_op_const_operands(source) + operand_position,
         (source->operand_count - operand_position) * sizeof(*operands));
  for (uint16_t i = 0; i < source->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_builder_define_result(
        builder,
        loom_module_value_type(rewriter->module,
                               loom_op_const_results(source)[i]),
        &loom_op_results(op)[i]));
  }
  if (appended_result_count) {
    memcpy(loom_op_results(op) + source->result_count, appended_results,
           appended_result_count * sizeof(*appended_results));
  }
  return iree_ok_status();
}

static iree_status_t loom_private_storage_extend_terminator(
    loom_private_storage_plan_t* plan, loom_rewriter_t* rewriter,
    loom_private_storage_signature_t* signature, uint8_t region_index,
    loom_region_t* region, loom_op_t* parent, loom_value_id_t* values,
    uint16_t count) {
  uint16_t index = 0;
  for (loom_private_storage_transport_t* t = signature->transports; t;
       t = t->next) {
    if (!loom_private_storage_transport_selected(t)) {
      continue;
    }
    values[index++] =
        loom_private_storage_replacement(plan, t->outgoing[region_index]);
  }
  loom_block_t* block = loom_region_entry_block(region);
  loom_op_t* source = block->last_op;
  loom_op_t* terminator = NULL;
  if (source) {
    IREE_RETURN_IF_ERROR(
        loom_private_storage_extend_op(rewriter, source, source->operand_count,
                                       values, count, NULL, 0, 0, &terminator));
    IREE_RETURN_IF_ERROR(
        loom_builder_finalize_op(&rewriter->builder, terminator));
    return loom_rewriter_erase(rewriter, source);
  }
  loom_builder_enter_region(&rewriter->builder, parent, region);
  return loom_region_branch_build_region_terminator(
      &rewriter->builder, plan->module,
      loom_region_branch_cast(plan->module, parent), region_index, values,
      count, parent->location, &terminator);
}

static iree_status_t loom_private_storage_rewrite_structured(
    loom_private_storage_plan_t* plan, loom_rewriter_t* rewriter,
    loom_private_storage_signature_t* signature, uint16_t count) {
  loom_op_t* source = signature->op;
  const loom_loop_like_t loop = loom_loop_like_cast(plan->module, source);
  const bool is_loop = loom_loop_like_isa(loop);
  loom_value_id_t* values = NULL;
  loom_value_id_t* results = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, count, sizeof(*values), (void**)&values));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, count, sizeof(*results), (void**)&results));
  uint16_t index = 0;
  for (loom_private_storage_transport_t* t = signature->transports; t;
       t = t->next) {
    if (!loom_private_storage_transport_selected(t)) {
      continue;
    }
    if (is_loop) {
      values[index] = loom_private_storage_replacement(plan, t->initial);
    }
    results[index++] = t->result->replacement;
  }
  const loom_value_slice_t initial =
      is_loop ? loom_loop_like_iter_args(loop) : (loom_value_slice_t){0};
  uint16_t operand_position = source->operand_count;
  if (is_loop) {
    operand_position =
        loop.vtable->iter_args_operand_field_index + initial.count;
    if (loop.vtable->segmented_operands) {
      operand_position = initial.count;
      const uint16_t* segments = loom_op_const_operand_segment_counts(source);
      for (uint8_t i = 0; i < loop.vtable->iter_args_operand_field_index; ++i) {
        operand_position += segments[i];
      }
    }
  }
  loom_op_t* target = NULL;
  IREE_RETURN_IF_ERROR(loom_private_storage_extend_op(
      rewriter, source, operand_position, values, is_loop ? count : 0, results,
      count, signature->region_count, &target));
  for (uint8_t r = 0; r < signature->region_count; ++r) {
    loom_block_t* block = NULL;
    IREE_RETURN_IF_ERROR(
        loom_builder_create_region(&rewriter->builder, target, r, &block));
    loom_region_t* old_region =
        r < source->region_count ? loom_op_regions(source)[r] : NULL;
    loom_block_t* old_block =
        old_region ? loom_region_entry_block(old_region) : NULL;
    if (old_block) {
      loom_op_regions(target)[r]->flags = old_region->flags;
      loom_op_regions(target)[r]->source_flags = old_region->source_flags;
      block->flags = old_block->flags;
      block->label_id = old_block->label_id;
      for (uint16_t a = 0; a < old_block->arg_count; ++a) {
        const loom_value_id_t old_argument = loom_block_arg_id(old_block, a);
        loom_value_id_t argument;
        IREE_RETURN_IF_ERROR(loom_builder_define_block_arg(
            &rewriter->builder, block,
            loom_module_value_type(plan->module, old_argument), &argument));
        IREE_RETURN_IF_ERROR(loom_private_storage_replace_identity(
            plan, rewriter, old_argument, argument));
      }
    }
    if (is_loop) {
      const bool condition = r == loop.vtable->condition_region_index;
      for (loom_private_storage_transport_t* t = signature->transports; t;
           t = t->next) {
        if (!loom_private_storage_transport_selected(t)) {
          continue;
        }
        loom_private_storage_value_t* argument =
            condition || !t->body ? t->entry : t->body;
        IREE_RETURN_IF_ERROR(
            loom_block_add_arg(plan->module, block, argument->replacement));
      }
    }
    if (old_block) {
      while (old_block->first_op) {
        IREE_RETURN_IF_ERROR(loom_rewriter_move_to_block_end(
            rewriter, old_block->first_op, block, target));
      }
    }
    IREE_RETURN_IF_ERROR(loom_private_storage_extend_terminator(
        plan, rewriter, signature, r, loom_op_regions(target)[r], target,
        values, count));
  }
  IREE_RETURN_IF_ERROR(loom_builder_finalize_op(&rewriter->builder, target));
  for (uint16_t i = 0; i < source->result_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_private_storage_replace_identity(
        plan, rewriter, loom_op_const_results(source)[i],
        loom_op_const_results(target)[i]));
  }
  return loom_rewriter_erase(rewriter, source);
}

static iree_status_t loom_private_storage_rewrite_cfg(
    loom_private_storage_plan_t* plan, loom_rewriter_t* rewriter,
    loom_private_storage_signature_t* signature, uint16_t count) {
  loom_value_id_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      plan->arena, count, sizeof(*values), (void**)&values));
  for (loom_private_storage_transport_t* t = signature->transports; t;
       t = t->next) {
    if (!loom_private_storage_transport_selected(t)) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_block_add_arg(plan->module, signature->block,
                                            t->entry->replacement));
  }
  for (iree_host_size_t p = 0; p < signature->predecessor_count; ++p) {
    uint16_t index = 0;
    for (loom_private_storage_transport_t* t = signature->transports; t;
         t = t->next) {
      if (!loom_private_storage_transport_selected(t)) {
        continue;
      }
      values[index++] = loom_private_storage_replacement(plan, t->outgoing[p]);
    }
    loom_op_t* source = signature->predecessors[p].terminator;
    loom_op_t* branch = NULL;
    if (loom_cfg_br_isa(source)) {
      IREE_RETURN_IF_ERROR(loom_private_storage_extend_op(
          rewriter, source, source->operand_count, values, count, NULL, 0, 0,
          &branch));
      IREE_RETURN_IF_ERROR(
          loom_builder_finalize_op(&rewriter->builder, branch));
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, source));
    } else {
      // Conditional edges have no payload. A dedicated edge block carries the
      // tuple without executing work or changing the original branch decision.
      loom_block_t* edge = NULL;
      IREE_RETURN_IF_ERROR(loom_region_append_block(
          plan->module, signature->block->parent_region, &edge));
      loom_op_successors(source)[signature->predecessors[p].successor_index] =
          edge;
      rewriter->builder.ip =
          (loom_builder_ip_t){.block = edge, .parent_op = source->parent_op};
      IREE_RETURN_IF_ERROR(loom_cfg_br_build(&rewriter->builder,
                                             signature->block, values, count,
                                             source->location, &branch));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_private_storage_rewrite(
    loom_private_storage_plan_t* plan, loom_rewriter_t* rewriter,
    loom_promote_private_fragments_statistics_t* statistics) {
  // Define every appended identity first so cyclic and out-of-order CFG edges
  // can reference their destinations before those signatures are rebuilt.
  for (loom_private_storage_signature_t* s = plan->signatures; s; s = s->next) {
    for (loom_private_storage_transport_t* t = s->transports; t; t = t->next) {
      if (!loom_private_storage_transport_selected(t)) {
        continue;
      }
      loom_private_storage_value_t* definitions[] = {t->entry, t->body,
                                                     t->result};
      for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(definitions); ++i) {
        if (!definitions[i]) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_builder_define_value(
            &rewriter->builder, t->cell->type, &definitions[i]->replacement));
      }
    }
  }
  for (loom_private_storage_signature_t* s = plan->signatures; s; s = s->next) {
    const iree_host_size_t count = loom_private_storage_signature_count(s);
    if (!count) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_builder_check_count_range(
        count, UINT16_MAX, IREE_SV("promoted control tuple")));
    if (s->op) {
      IREE_RETURN_IF_ERROR(
          loom_private_storage_rewrite_structured(plan, rewriter, s, count));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_private_storage_rewrite_cfg(plan, rewriter, s, count));
    }
  }
  for (loom_private_storage_access_t* a = plan->accesses; a; a = a->next) {
    if (!a->cell || !a->cell->allocation->selected ||
        !loom_view_load_isa(a->op)) {
      continue;
    }
    const loom_value_id_t replacement =
        loom_private_storage_replacement(plan, a->value);
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        rewriter, loom_view_load_result(a->op), replacement));
    ++statistics->loads_forwarded;
  }
  // Stores may retain load results until all forwarding is complete.
  for (loom_private_storage_access_t* a = plan->accesses; a; a = a->next) {
    if (a->cell && a->cell->allocation->selected &&
        loom_view_store_isa(a->op)) {
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, a->op));
    }
  }
  for (loom_private_storage_access_t* a = plan->accesses; a; a = a->next) {
    if (a->cell && a->cell->allocation->selected && loom_view_load_isa(a->op)) {
      IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, a->op));
    }
  }
  for (iree_host_size_t i = plan->projection_count; i > 0; --i) {
    bool erased;
    IREE_RETURN_IF_ERROR(loom_rewriter_erase_if_dead(
        rewriter, plan->projections[i - 1], &erased));
  }
  for (loom_private_storage_allocation_t* a = plan->allocations; a;
       a = a->next) {
    bool erased;
    IREE_RETURN_IF_ERROR(loom_rewriter_erase_if_dead(rewriter, a->op, &erased));
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Pass entry point
//===----------------------------------------------------------------------===//

iree_status_t loom_promote_private_fragments_run(loom_pass_t* pass,
                                                 loom_module_t* module,
                                                 loom_func_like_t function) {
  loom_region_t* body = loom_func_like_body(function);
  if (!body) {
    return iree_ok_status();
  }

  loom_rewriter_t rewriter;
  loom_rewriter_initialize(&rewriter, module, pass->arena);
  iree_status_t status = iree_ok_status();
  bool changed = false;
  bool iteration_changed = true;
  while (iree_status_is_ok(status) && iteration_changed) {
    iteration_changed = false;

    loom_dominance_info_t dominance = {0};
    status = loom_dominance_info_initialize_region(module, body, pass->arena,
                                                   &dominance);
    loom_promote_private_fragments_view_list_t views = {0};
    if (iree_status_is_ok(status)) {
      status = loom_promote_private_fragments_view_list_initialize(pass->arena,
                                                                   &views);
    }

    loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
    loom_promote_private_fragments_collect_context_t collect_context = {
        .arena = pass->arena,
        .views = &views,
    };
    if (iree_status_is_ok(status)) {
      status = loom_walk_function(
          module, function, LOOM_WALK_PRE_ORDER,
          (loom_walk_callback_t){
              .fn = loom_promote_private_fragments_collect_view,
              .user_data = &collect_context,
          },
          &walk_result);
    }

    loom_promote_private_fragments_context_t context = {
        .pass = pass,
        .statistics = loom_promote_private_fragments_statistics(pass),
        .module = module,
        .dominance = &dominance,
        .rewriter = &rewriter,
    };
    for (iree_host_size_t i = 0; i < views.count && iree_status_is_ok(status);
         ++i) {
      status = loom_promote_private_fragments_process_view(
          &context, views.ops[i], &iteration_changed);
      if (iteration_changed) {
        changed = true;
        break;
      }
    }
  }
  loom_private_storage_plan_t plan = {0};
  if (iree_status_is_ok(status)) {
    if (changed) {
      loom_pass_value_fact_owner_invalidate(pass->value_facts);
    }
    status = loom_private_storage_plan_build(pass, module, function, &plan);
  }
  if (iree_status_is_ok(status) && plan.cell_count) {
    status = loom_private_storage_rewrite(
        &plan, &rewriter, loom_promote_private_fragments_statistics(pass));
  }
  loom_private_storage_plan_release(&plan);
  if (iree_any_bit_set(rewriter.flags, LOOM_REWRITER_FLAG_CHANGED)) {
    loom_pass_value_fact_owner_invalidate(pass->value_facts);
    if (iree_status_is_ok(status)) {
      loom_pass_mark_changed(pass);
    }
  }

  loom_rewriter_deinitialize(&rewriter);
  return status;
}
