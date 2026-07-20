// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/transforms/residency_recipes.h"

#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/dominance.h"
#include "loom/util/walk.h"

#define LOOM_LOW_SEAL_RESIDENCY_RECIPES_STATISTICS(V, statistics_type) \
  V(statistics_type, candidates_sealed, "candidates-sealed",           \
    "Number of target-low residency candidates sealed.")               \
  V(statistics_type, producers_recorded, "producers-recorded",         \
    "Number of target-low recipe producers recorded.")

LOOM_PASS_STATISTICS_DEFINE(loom_low_seal_residency_recipes_statistics,
                            loom_low_seal_residency_recipes_statistics_t,
                            LOOM_LOW_SEAL_RESIDENCY_RECIPES_STATISTICS)

static const loom_pass_info_t
    loom_low_seal_residency_recipes_pass_info_storage = {
        .name = IREE_SVL("low-seal-residency-recipes"),
        .description =
            IREE_SVL("Seal target-low residency rematerialization recipes."),
        .kind = LOOM_PASS_FUNCTION,
        .statistic_layout = &loom_low_seal_residency_recipes_statistics_layout,
};

const loom_pass_info_t* loom_low_seal_residency_recipes_pass_info(void) {
  return &loom_low_seal_residency_recipes_pass_info_storage;
}

typedef struct loom_low_residency_value_list_t {
  // Arena owning the dynamically grown value array.
  iree_arena_allocator_t* arena;
  // Unique values in insertion order.
  loom_value_id_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
  // Allocated entry capacity of |values|.
  iree_host_size_t capacity;
} loom_low_residency_value_list_t;

typedef struct loom_low_residency_recipe_entry_t {
  // Producer represented by this recipe entry.
  loom_op_t* op;
  // One result tracking the producer through SSA replacement.
  loom_value_id_t value_id;
} loom_low_residency_recipe_entry_t;

typedef struct loom_low_residency_recipe_list_t {
  // Arena owning the dynamically grown recipe array.
  iree_arena_allocator_t* arena;
  // Unique producers in discovery order.
  loom_low_residency_recipe_entry_t* entries;
  // Number of entries in |entries|.
  iree_host_size_t count;
  // Allocated entry capacity of |entries|.
  iree_host_size_t capacity;
} loom_low_residency_recipe_list_t;

typedef struct loom_low_residency_marker_list_t {
  // Arena owning the dynamically grown marker array.
  iree_arena_allocator_t* arena;
  // Candidate markers in source dominance order.
  loom_op_t** ops;
  // Number of entries in |ops|.
  iree_host_size_t count;
  // Allocated entry capacity of |ops|.
  iree_host_size_t capacity;
} loom_low_residency_marker_list_t;

static bool loom_low_residency_value_list_contains(
    const loom_low_residency_value_list_t* list, loom_value_id_t value_id) {
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    if (list->values[i] == value_id) return true;
  }
  return false;
}

static iree_status_t loom_low_residency_value_list_append_unique(
    loom_low_residency_value_list_t* list, loom_value_id_t value_id) {
  if (loom_low_residency_value_list_contains(list, value_id)) {
    return iree_ok_status();
  }
  if (list->count == list->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        list->arena, list->count, list->count + 1, sizeof(*list->values),
        &list->capacity, (void**)&list->values));
  }
  list->values[list->count++] = value_id;
  return iree_ok_status();
}

static iree_host_size_t loom_low_residency_recipe_list_find(
    const loom_low_residency_recipe_list_t* list, const loom_op_t* op) {
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    if (list->entries[i].op == op) return i;
  }
  return IREE_HOST_SIZE_MAX;
}

static iree_status_t loom_low_residency_recipe_list_append_unique(
    loom_low_residency_recipe_list_t* list, loom_op_t* op,
    loom_value_id_t value_id) {
  if (loom_low_residency_recipe_list_find(list, op) != IREE_HOST_SIZE_MAX) {
    return iree_ok_status();
  }
  if (list->count == list->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        list->arena, list->count, list->count + 1, sizeof(*list->entries),
        &list->capacity, (void**)&list->entries));
  }
  list->entries[list->count++] = (loom_low_residency_recipe_entry_t){
      .op = op,
      .value_id = value_id,
  };
  return iree_ok_status();
}

static iree_status_t loom_low_residency_marker_list_append(
    loom_low_residency_marker_list_t* list, loom_op_t* op) {
  if (list->count == list->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        list->arena, list->count, list->count + 1, sizeof(*list->ops),
        &list->capacity, (void**)&list->ops));
  }
  list->ops[list->count++] = op;
  return iree_ok_status();
}

static iree_status_t loom_low_residency_collect_marker(
    void* user_data, loom_op_t* op, const loom_walk_context_t* walk_context,
    loom_walk_result_t* out_result) {
  (void)walk_context;
  *out_result = LOOM_WALK_CONTINUE;
  if (!loom_low_residency_candidate_isa(op)) return iree_ok_status();
  return loom_low_residency_marker_list_append(
      (loom_low_residency_marker_list_t*)user_data, op);
}

static iree_status_t loom_low_residency_validate_value(
    const loom_module_t* module, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "residency recipe references an invalid value");
  }
  return iree_ok_status();
}

// Resolves transparent candidate chains and merges their proof boundaries.
static iree_status_t loom_low_residency_resolve_candidate_source(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_low_residency_value_list_t* captures, loom_value_id_t* out_value_id) {
  *out_value_id = LOOM_VALUE_ID_INVALID;
  for (iree_host_size_t depth = 0; depth <= module->values.count; ++depth) {
    IREE_RETURN_IF_ERROR(loom_low_residency_validate_value(module, value_id));
    if (loom_low_residency_value_list_contains(captures, value_id)) {
      *out_value_id = value_id;
      return iree_ok_status();
    }
    const loom_value_t* value = loom_module_value(module, value_id);
    if (loom_value_is_block_arg(value)) {
      *out_value_id = value_id;
      return iree_ok_status();
    }
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (!loom_low_residency_candidate_isa(defining_op)) {
      *out_value_id = value_id;
      return iree_ok_status();
    }
    const loom_value_slice_t nested_captures =
        loom_low_residency_candidate_captures(defining_op);
    for (iree_host_size_t i = 0; i < nested_captures.count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_residency_value_list_append_unique(
          captures, nested_captures.values[i]));
    }
    value_id = loom_low_residency_candidate_source(defining_op);
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "residency candidate marker graph contains a cycle");
}

static iree_status_t loom_low_residency_discover_recipe(
    const loom_module_t* module, const loom_op_t* marker_op,
    const loom_dominance_info_t* dominance,
    loom_low_residency_value_list_t* captures,
    loom_low_residency_value_list_t* pending,
    loom_low_residency_recipe_list_t* recipe) {
  // Newly discovered block arguments or nested proof captures can shorten a
  // slice already visited in this iteration. Repeat until the boundary is
  // stable, then retain only the final reachable producer closure.
  for (iree_host_size_t iteration = 0; iteration <= module->values.count;
       ++iteration) {
    const iree_host_size_t capture_count_before = captures->count;
    pending->count = 0;
    recipe->count = 0;
    IREE_RETURN_IF_ERROR(loom_low_residency_value_list_append_unique(
        pending, loom_low_residency_candidate_source(marker_op)));

    for (iree_host_size_t pending_index = 0; pending_index < pending->count;
         ++pending_index) {
      loom_value_id_t value_id = pending->values[pending_index];
      if (loom_low_residency_value_list_contains(captures, value_id)) {
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_low_residency_resolve_candidate_source(
          module, value_id, captures, &value_id));
      if (loom_low_residency_value_list_contains(captures, value_id)) {
        continue;
      }

      const loom_value_t* value = loom_module_value(module, value_id);
      if (loom_value_is_block_arg(value)) {
        IREE_RETURN_IF_ERROR(
            loom_low_residency_value_list_append_unique(captures, value_id));
        continue;
      }
      loom_op_t* defining_op = loom_value_def_op(value);
      if (defining_op == NULL ||
          iree_any_bit_set(defining_op->flags, LOOM_OP_FLAG_DEAD)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "residency recipe references an erased producer");
      }
      if (loom_low_live_in_isa(defining_op) ||
          loom_low_resource_isa(defining_op)) {
        IREE_RETURN_IF_ERROR(
            loom_low_residency_value_list_append_unique(captures, value_id));
        continue;
      }
      if (!loom_dominates_op(dominance, defining_op, marker_op) ||
          defining_op->region_count != 0 || defining_op->successor_count != 0 ||
          defining_op->tied_result_count != 0) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "residency recipe producer is not a finite dominating packet");
      }

      const iree_host_size_t previous_count = recipe->count;
      IREE_RETURN_IF_ERROR(loom_low_residency_recipe_list_append_unique(
          recipe, defining_op, value_id));
      if (recipe->count == previous_count) continue;
      const loom_value_id_t* operands = loom_op_const_operands(defining_op);
      for (uint16_t i = 0; i < defining_op->operand_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_low_residency_value_list_append_unique(pending, operands[i]));
      }
    }

    if (captures->count == capture_count_before) return iree_ok_status();
  }
  return iree_make_status(
      IREE_STATUS_FAILED_PRECONDITION,
      "residency recipe boundary did not converge within the value graph");
}

static iree_status_t loom_low_residency_dependency_ready(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_low_residency_value_list_t* captures,
    const loom_low_residency_recipe_list_t* recipe, const bool* emitted,
    bool* out_ready) {
  *out_ready = false;
  if (loom_low_residency_value_list_contains(captures, value_id)) {
    *out_ready = true;
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_low_residency_resolve_candidate_source(
      module, value_id, captures, &value_id));
  if (loom_low_residency_value_list_contains(captures, value_id)) {
    *out_ready = true;
    return iree_ok_status();
  }
  const loom_value_t* value = loom_module_value(module, value_id);
  if (loom_value_is_block_arg(value)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "sealed residency recipe omitted a block-argument capture");
  }
  const loom_op_t* defining_op = loom_value_def_op(value);
  const iree_host_size_t producer_index =
      loom_low_residency_recipe_list_find(recipe, defining_op);
  if (producer_index == IREE_HOST_SIZE_MAX) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "sealed residency recipe omitted a producer dependency");
  }
  *out_ready = emitted[producer_index];
  return iree_ok_status();
}

static iree_status_t loom_low_residency_order_recipe(
    const loom_module_t* module, loom_low_residency_value_list_t* captures,
    const loom_low_residency_recipe_list_t* recipe,
    iree_arena_allocator_t* arena, loom_value_id_t** out_values) {
  *out_values = NULL;
  if (recipe->count == 0) return iree_ok_status();

  bool* emitted = NULL;
  loom_value_id_t* values = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, recipe->count, sizeof(*emitted), (void**)&emitted));
  memset(emitted, 0, recipe->count * sizeof(*emitted));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, recipe->count, sizeof(*values), (void**)&values));

  iree_host_size_t emitted_count = 0;
  while (emitted_count < recipe->count) {
    bool made_progress = false;
    for (iree_host_size_t i = 0; i < recipe->count; ++i) {
      if (emitted[i]) continue;
      const loom_op_t* op = recipe->entries[i].op;
      bool ready = true;
      const loom_value_id_t* operands = loom_op_const_operands(op);
      for (uint16_t operand_index = 0; operand_index < op->operand_count;
           ++operand_index) {
        bool dependency_ready = false;
        IREE_RETURN_IF_ERROR(loom_low_residency_dependency_ready(
            module, operands[operand_index], captures, recipe, emitted,
            &dependency_ready));
        if (!dependency_ready) {
          ready = false;
          break;
        }
      }
      if (!ready) continue;
      emitted[i] = true;
      values[emitted_count++] = recipe->entries[i].value_id;
      made_progress = true;
    }
    if (!made_progress) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "residency recipe producer graph is cyclic or incomplete");
    }
  }
  *out_values = values;
  return iree_ok_status();
}

static bool loom_low_residency_value_slice_equal(loom_value_slice_t slice,
                                                 const loom_value_id_t* values,
                                                 iree_host_size_t value_count) {
  return slice.count == value_count &&
         (value_count == 0 ||
          memcmp(slice.values, values, value_count * sizeof(*values)) == 0);
}

static iree_status_t loom_low_residency_seal_marker(
    loom_module_t* module, loom_op_t* marker_op,
    const loom_dominance_info_t* dominance, loom_rewriter_t* rewriter,
    bool* out_changed, iree_host_size_t* out_producer_count) {
  *out_changed = false;
  *out_producer_count = 0;

  loom_low_residency_value_list_t captures = {
      .arena = rewriter->arena,
  };
  const loom_value_slice_t old_captures =
      loom_low_residency_candidate_captures(marker_op);
  for (iree_host_size_t i = 0; i < old_captures.count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_low_residency_validate_value(module, old_captures.values[i]));
    IREE_RETURN_IF_ERROR(loom_low_residency_value_list_append_unique(
        &captures, old_captures.values[i]));
  }
  loom_low_residency_value_list_t pending = {
      .arena = rewriter->arena,
  };
  loom_low_residency_recipe_list_t recipe = {
      .arena = rewriter->arena,
  };
  IREE_RETURN_IF_ERROR(loom_low_residency_discover_recipe(
      module, marker_op, dominance, &captures, &pending, &recipe));
  loom_value_id_t* ordered_recipe = NULL;
  IREE_RETURN_IF_ERROR(loom_low_residency_order_recipe(
      module, &captures, &recipe, rewriter->arena, &ordered_recipe));

  const loom_value_slice_t old_recipe =
      loom_low_residency_candidate_recipe(marker_op);
  if (loom_low_residency_candidate_sealed(marker_op) &&
      loom_low_residency_value_slice_equal(old_captures, captures.values,
                                           captures.count) &&
      loom_low_residency_value_slice_equal(old_recipe, ordered_recipe,
                                           recipe.count)) {
    return iree_ok_status();
  }

  const loom_value_id_t source = loom_low_residency_candidate_source(marker_op);
  const loom_type_t result_type = loom_module_value_type(
      module, loom_low_residency_candidate_result(marker_op));
  loom_builder_set_before(&rewriter->builder, marker_op);
  const loom_value_id_t value_checkpoint =
      loom_rewriter_value_checkpoint(rewriter);
  loom_op_t* sealed_op = NULL;
  IREE_RETURN_IF_ERROR(loom_low_residency_candidate_build(
      &rewriter->builder, loom_low_residency_candidate_candidate_id(marker_op),
      loom_low_residency_candidate_recompute_cost(marker_op), source,
      result_type, captures.values, captures.count, ordered_recipe,
      recipe.count, loom_low_residency_candidate_preserves_baseline(marker_op),
      /*sealed=*/true, marker_op->location, &sealed_op));
  const loom_value_id_t sealed_result =
      loom_low_residency_candidate_result(sealed_op);
  IREE_RETURN_IF_ERROR(loom_rewriter_preserve_result_names_on_new_values(
      rewriter, marker_op, &sealed_result, 1, value_checkpoint));
  IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_and_erase(
      rewriter, marker_op, &sealed_result, 1));
  *out_changed = true;
  *out_producer_count = recipe.count;
  return iree_ok_status();
}

iree_status_t loom_low_seal_residency_recipes_run(loom_pass_t* pass,
                                                  loom_module_t* module,
                                                  loom_func_like_t function) {
  if (!loom_low_function_def_isa(function.op) ||
      !loom_func_like_body(function)) {
    return iree_ok_status();
  }

  loom_low_residency_marker_list_t markers = {
      .arena = pass->arena,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_region(
      module, loom_low_function_body(function.op), LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_low_residency_collect_marker, &markers},
      pass->arena, &walk_result));
  if (markers.count == 0) return iree_ok_status();

  loom_dominance_info_t dominance = {0};
  IREE_RETURN_IF_ERROR(
      loom_dominance_info_initialize(module, pass->arena, &dominance));
  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(
      loom_rewriter_initialize(&rewriter, module, pass->arena));
  loom_low_seal_residency_recipes_statistics_t* statistics =
      loom_low_seal_residency_recipes_statistics(pass);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < markers.count && iree_status_is_ok(status);
       ++i) {
    bool changed = false;
    iree_host_size_t producer_count = 0;
    status =
        loom_low_residency_seal_marker(module, markers.ops[i], &dominance,
                                       &rewriter, &changed, &producer_count);
    if (changed) {
      loom_pass_mark_changed(pass);
      ++statistics->candidates_sealed;
      statistics->producers_recorded += producer_count;
    }
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}
