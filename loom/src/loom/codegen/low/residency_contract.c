// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/residency_contract.h"

#include <string.h>

#include "loom/codegen/low/function.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/rewrite/rewriter.h"
#include "loom/util/dominance.h"
#include "loom/util/walk.h"

typedef struct loom_low_residency_marker_list_t {
  // Arena owning the dynamically grown marker array.
  iree_arena_allocator_t* arena;
  // Residency marker ops in source dominance order.
  loom_op_t** ops;
  // Number of entries in |ops|.
  iree_host_size_t count;
  // Allocated entry capacity of |ops|.
  iree_host_size_t capacity;
} loom_low_residency_marker_list_t;

typedef struct loom_low_residency_use_list_t {
  // Arena owning the dynamically grown use array.
  iree_arena_allocator_t* arena;
  // Leaf operand uses reached through candidate marker chains.
  loom_use_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
  // Allocated entry capacity of |values|.
  iree_host_size_t capacity;
} loom_low_residency_use_list_t;

typedef struct loom_low_residency_value_list_t {
  // Arena owning the dynamically grown value array.
  iree_arena_allocator_t* arena;
  // Unique target-low values in insertion order.
  loom_value_id_t* values;
  // Number of entries in |values|.
  iree_host_size_t count;
  // Allocated entry capacity of |values|.
  iree_host_size_t capacity;
} loom_low_residency_value_list_t;

typedef struct loom_low_residency_recipe_t {
  // Producer operations in dependency order.
  loom_op_t** ops;
  // Number of entries in |ops|.
  iree_host_size_t op_count;
  // Values captured outside the producer slice.
  loom_value_id_t* inputs;
  // Number of entries in |inputs|.
  iree_host_size_t input_count;
} loom_low_residency_recipe_t;

typedef struct loom_low_residency_marker_info_t {
  // Candidate marker operation to consume.
  loom_op_t* op;
  // Root non-marker value feeding the candidate chain.
  loom_value_id_t root_value_id;
} loom_low_residency_marker_info_t;

// A source residency contract lowers at the start of the function packet
// stream, after any ABI live-in and resource declarations. Contract-free
// functions take this bounded preamble probe instead of walking the low IR.
static bool loom_low_residency_function_has_contract(loom_op_t* low_func_op) {
  const loom_region_t* body = loom_low_function_body(low_func_op);
  if (body == NULL || body->block_count == 0) return false;
  const loom_block_t* entry_block = loom_region_const_entry_block(body);
  const loom_op_t* op = NULL;
  loom_block_for_each_op(entry_block, op) {
    if (loom_low_live_in_isa(op) || loom_low_resource_isa(op)) continue;
    return loom_low_residency_require_isa(op);
  }
  return false;
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
  if (!loom_low_residency_require_isa(op) &&
      !loom_low_residency_candidate_isa(op)) {
    return iree_ok_status();
  }
  return loom_low_residency_marker_list_append(
      (loom_low_residency_marker_list_t*)user_data, op);
}

static iree_status_t loom_low_residency_use_list_append(
    loom_low_residency_use_list_t* list, loom_use_t use) {
  if (list->count == list->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        list->arena, list->count, list->count + 1, sizeof(*list->values),
        &list->capacity, (void**)&list->values));
  }
  list->values[list->count++] = use;
  return iree_ok_status();
}

static bool loom_low_residency_value_list_contains(
    const loom_low_residency_value_list_t* list, loom_value_id_t value_id) {
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    if (list->values[i] == value_id) return true;
  }
  return false;
}

static iree_status_t loom_low_residency_value_list_append_unique(
    loom_low_residency_value_list_t* list, loom_value_id_t value_id,
    bool* out_inserted) {
  *out_inserted = false;
  if (loom_low_residency_value_list_contains(list, value_id)) {
    return iree_ok_status();
  }
  if (list->count == list->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        list->arena, list->count, list->count + 1, sizeof(*list->values),
        &list->capacity, (void**)&list->values));
  }
  list->values[list->count++] = value_id;
  *out_inserted = true;
  return iree_ok_status();
}

static iree_host_size_t loom_low_residency_op_list_find(
    const loom_low_residency_marker_list_t* list, const loom_op_t* op) {
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    if (list->ops[i] == op) return i;
  }
  return IREE_HOST_SIZE_MAX;
}

static iree_status_t loom_low_residency_resolve_root_value(
    const loom_module_t* module, loom_value_id_t value_id,
    iree_host_size_t marker_count, loom_value_id_t* out_root_value_id);

// Validates the sealed target-low producer slice between recorded captures and
// the candidate result. Recipe membership is represented with SSA operands so
// canonical replacement carries it forward without relying on block locality.
static iree_status_t loom_low_residency_collect_recipe(
    const loom_module_t* module, const loom_op_t* marker_op,
    loom_value_id_t root_value_id, const loom_value_id_t* captures,
    iree_host_size_t capture_count, const loom_value_id_t* recipe_members,
    iree_host_size_t recipe_member_count, iree_host_size_t marker_count,
    const loom_dominance_info_t* dominance, iree_arena_allocator_t* arena,
    loom_low_residency_recipe_t* out_recipe) {
  *out_recipe = (loom_low_residency_recipe_t){0};
  if (root_value_id == LOOM_VALUE_ID_INVALID ||
      root_value_id >= module->values.count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "residency recipe has an invalid root value");
  }
  const loom_value_t* root_value = loom_module_value(module, root_value_id);
  if (loom_value_is_block_arg(root_value)) {
    bool root_is_captured = false;
    for (iree_host_size_t i = 0; i < capture_count; ++i) {
      if (captures[i] == root_value_id) {
        root_is_captured = true;
        break;
      }
    }
    if (!root_is_captured || recipe_member_count != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "sealed residency recipe has an unbounded block-argument root");
    }
    out_recipe->inputs = (loom_value_id_t*)captures;
    out_recipe->input_count = capture_count;
    return iree_ok_status();
  }
  const loom_op_t* root_op = loom_value_def_op(root_value);
  if (root_op == NULL || root_op->parent_block == NULL ||
      !loom_dominates_op(dominance, root_op, marker_op)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "residency recipe root no longer dominates its candidate marker");
  }

  loom_low_residency_value_list_t inputs = {.arena = arena};
  for (iree_host_size_t i = 0; i < capture_count; ++i) {
    if (captures[i] == LOOM_VALUE_ID_INVALID ||
        captures[i] >= module->values.count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "residency recipe capture is invalid");
    }
    bool inserted = false;
    IREE_RETURN_IF_ERROR(loom_low_residency_value_list_append_unique(
        &inputs, captures[i], &inserted));
  }

  loom_low_residency_marker_list_t recipe_ops = {.arena = arena};
  for (iree_host_size_t i = 0; i < recipe_member_count; ++i) {
    loom_value_id_t value_id = recipe_members[i];
    IREE_RETURN_IF_ERROR(loom_low_residency_resolve_root_value(
        module, value_id, marker_count, &value_id));
    if (loom_low_residency_value_list_contains(&inputs, value_id)) continue;
    const loom_value_t* value = loom_module_value(module, value_id);
    if (loom_value_is_block_arg(value)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "sealed residency recipe member is an uncaptured block argument");
    }
    loom_op_t* defining_op = loom_value_def_op(value);
    if (defining_op == NULL ||
        iree_any_bit_set(defining_op->flags, LOOM_OP_FLAG_DEAD)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "sealed residency recipe member was erased");
    }
    if (!loom_dominates_op(dominance, defining_op, marker_op) ||
        defining_op->region_count != 0 || defining_op->successor_count != 0 ||
        defining_op->tied_result_count != 0) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "sealed residency recipe member is not a finite dominating packet");
    }
    if (loom_low_residency_op_list_find(&recipe_ops, defining_op) ==
        IREE_HOST_SIZE_MAX) {
      IREE_RETURN_IF_ERROR(
          loom_low_residency_marker_list_append(&recipe_ops, defining_op));
    }
  }
  if (recipe_ops.count == 0) {
    if (loom_low_residency_value_list_contains(&inputs, root_value_id)) {
      out_recipe->inputs = inputs.values;
      out_recipe->input_count = inputs.count;
      return iree_ok_status();
    }
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "sealed residency recipe is empty");
  }

  bool* reachable_ops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, recipe_ops.count, sizeof(*reachable_ops), (void**)&reachable_ops));
  memset(reachable_ops, 0, recipe_ops.count * sizeof(*reachable_ops));
  loom_low_residency_value_list_t pending = {.arena = arena};
  bool inserted = false;
  IREE_RETURN_IF_ERROR(loom_low_residency_value_list_append_unique(
      &pending, root_value_id, &inserted));
  for (iree_host_size_t pending_index = 0; pending_index < pending.count;
       ++pending_index) {
    loom_value_id_t value_id = pending.values[pending_index];
    IREE_RETURN_IF_ERROR(loom_low_residency_resolve_root_value(
        module, value_id, marker_count, &value_id));
    if (loom_low_residency_value_list_contains(&inputs, value_id)) continue;
    const loom_value_t* value = loom_module_value(module, value_id);
    if (loom_value_is_block_arg(value)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "residency recipe depends on an unrecorded block argument");
    }
    loom_op_t* defining_op = loom_value_def_op(value);
    if (defining_op == NULL ||
        iree_any_bit_set(defining_op->flags, LOOM_OP_FLAG_DEAD)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "residency recipe references an erased producer");
    }
    const iree_host_size_t recipe_index =
        loom_low_residency_op_list_find(&recipe_ops, defining_op);
    if (recipe_index == IREE_HOST_SIZE_MAX) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "sealed residency recipe omitted a producer dependency");
    }
    if (reachable_ops[recipe_index]) continue;
    reachable_ops[recipe_index] = true;
    const loom_value_id_t* operands = loom_op_const_operands(defining_op);
    for (uint16_t i = 0; i < defining_op->operand_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_low_residency_value_list_append_unique(
          &pending, operands[i], &inserted));
    }
  }
  for (iree_host_size_t i = 0; i < recipe_ops.count; ++i) {
    if (!reachable_ops[i]) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "sealed residency recipe contains an unreachable producer");
    }
    const loom_value_id_t* operands = loom_op_const_operands(recipe_ops.ops[i]);
    for (uint16_t operand_index = 0;
         operand_index < recipe_ops.ops[i]->operand_count; ++operand_index) {
      loom_value_id_t operand_id = operands[operand_index];
      IREE_RETURN_IF_ERROR(loom_low_residency_resolve_root_value(
          module, operand_id, marker_count, &operand_id));
      if (loom_low_residency_value_list_contains(&inputs, operand_id)) {
        continue;
      }
      const loom_value_t* operand = loom_module_value(module, operand_id);
      if (loom_value_is_block_arg(operand)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "sealed residency recipe omitted a block-argument input");
      }
      const iree_host_size_t dependency_index = loom_low_residency_op_list_find(
          &recipe_ops, loom_value_def_op(operand));
      if (dependency_index == IREE_HOST_SIZE_MAX || dependency_index >= i) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "sealed residency recipe is not in dependency order");
      }
    }
  }
  if (recipe_ops.count > UINT32_MAX || inputs.count > UINT32_MAX) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "residency recipe exceeds its bounded representation");
  }
  out_recipe->ops = recipe_ops.ops;
  out_recipe->op_count = recipe_ops.count;
  out_recipe->inputs = inputs.values;
  out_recipe->input_count = inputs.count;
  return iree_ok_status();
}

static iree_status_t loom_low_residency_resolve_root_value(
    const loom_module_t* module, loom_value_id_t value_id,
    iree_host_size_t marker_count, loom_value_id_t* out_root_value_id) {
  *out_root_value_id = LOOM_VALUE_ID_INVALID;
  for (iree_host_size_t depth = 0; depth <= marker_count; ++depth) {
    if (value_id == LOOM_VALUE_ID_INVALID || value_id >= module->values.count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "residency candidate references invalid value");
    }
    const loom_value_t* value = loom_module_value(module, value_id);
    if (loom_value_is_block_arg(value)) {
      *out_root_value_id = value_id;
      return iree_ok_status();
    }
    const loom_op_t* defining_op = loom_value_def_op(value);
    if (!loom_low_residency_candidate_isa(defining_op)) {
      *out_root_value_id = value_id;
      return iree_ok_status();
    }
    value_id = loom_low_residency_candidate_source(defining_op);
  }
  return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                          "residency candidate marker graph contains a cycle");
}

static iree_status_t loom_low_residency_collect_leaf_uses(
    const loom_module_t* module, loom_value_id_t marker_result,
    iree_host_size_t marker_count, iree_arena_allocator_t* arena,
    loom_low_residency_use_list_t* out_uses) {
  *out_uses = (loom_low_residency_use_list_t){
      .arena = arena,
  };
  loom_value_id_t* value_stack = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, marker_count + 1, sizeof(*value_stack), (void**)&value_stack));
  iree_host_size_t stack_count = 1;
  value_stack[0] = marker_result;
  iree_host_size_t expanded_marker_count = 0;
  while (stack_count != 0) {
    const loom_value_id_t value_id = value_stack[--stack_count];
    const loom_value_t* value = loom_module_value(module, value_id);
    const loom_use_t* uses = loom_value_uses(value);
    for (uint32_t i = 0; i < value->use_count; ++i) {
      const loom_use_t use = uses[i];
      loom_op_t* user_op = loom_use_user_op(use);
      if (user_op == NULL ||
          iree_any_bit_set(user_op->flags, LOOM_OP_FLAG_DEAD)) {
        continue;
      }
      if (loom_low_residency_candidate_isa(user_op)) {
        if (loom_use_operand_index(use) == 0) {
          if (stack_count >= marker_count ||
              expanded_marker_count >= marker_count) {
            return iree_make_status(
                IREE_STATUS_FAILED_PRECONDITION,
                "residency candidate marker graph is cyclic or malformed");
          }
          value_stack[stack_count++] =
              loom_low_residency_candidate_result(user_op);
          ++expanded_marker_count;
        }
        // Capture and recipe operands are proof metadata. They keep the
        // boundary alive until contract consumption but are not dynamic uses
        // at which a materialization can be restored.
        continue;
      }
      IREE_RETURN_IF_ERROR(loom_low_residency_use_list_append(out_uses, use));
    }
  }
  return iree_ok_status();
}

iree_status_t loom_low_residency_contract_consume(
    loom_module_t* module, loom_op_t* low_func_op,
    iree_arena_allocator_t* arena,
    loom_low_residency_contract_t* out_contract) {
  if (module == NULL || low_func_op == NULL || arena == NULL ||
      out_contract == NULL || !loom_low_function_def_isa(low_func_op)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "residency contract requires a module, low "
                            "function, arena, and output");
  }
  *out_contract = (loom_low_residency_contract_t){0};
  if (!loom_low_residency_function_has_contract(low_func_op)) {
    return iree_ok_status();
  }

  loom_low_residency_marker_list_t markers = {
      .arena = arena,
  };
  loom_walk_result_t walk_result = LOOM_WALK_CONTINUE;
  IREE_RETURN_IF_ERROR(loom_walk_region(
      module, loom_low_function_body(low_func_op), LOOM_WALK_PRE_ORDER,
      (loom_walk_callback_t){loom_low_residency_collect_marker, &markers},
      arena, &walk_result));
  if (markers.count == 0) return iree_ok_status();

  loom_dominance_info_t dominance = {0};
  IREE_RETURN_IF_ERROR(
      loom_dominance_info_initialize(module, arena, &dominance));

  loom_low_residency_marker_info_t* candidate_markers = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, markers.count,
                                                 sizeof(*candidate_markers),
                                                 (void**)&candidate_markers));
  loom_low_residency_contract_candidate_t* candidates = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, markers.count, sizeof(*candidates), (void**)&candidates));
  iree_host_size_t candidate_count = 0;
  iree_host_size_t candidate_marker_count = 0;
  for (iree_host_size_t i = 0; i < markers.count; ++i) {
    loom_op_t* marker_op = markers.ops[i];
    if (loom_low_residency_require_isa(marker_op)) {
      const int64_t minimum = loom_low_residency_require_minimum(marker_op);
      if (minimum < -1 || (minimum >= 0 && (uint64_t)minimum > UINT32_MAX)) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "low residency minimum is outside its "
                                "validated representation");
      }
      out_contract->has_requirement = true;
      if (minimum >= 0) {
        out_contract->has_minimum_requirement = true;
        out_contract->minimum_required_tier =
            iree_max(out_contract->minimum_required_tier, (uint32_t)minimum);
      }
      if (loom_low_residency_require_preserve(marker_op)) {
        const int64_t projected_baseline =
            loom_low_residency_require_projected_baseline(marker_op);
        if (projected_baseline <= 0 ||
            (uint64_t)projected_baseline > UINT32_MAX) {
          return iree_make_status(
              IREE_STATUS_OUT_OF_RANGE,
              "low residency projected baseline is outside uint32");
        }
        out_contract->preserves_baseline = true;
        out_contract->projected_baseline_tier =
            iree_max(out_contract->projected_baseline_tier,
                     (uint32_t)projected_baseline);
      }
      out_contract->required_tier =
          iree_max(out_contract->minimum_required_tier,
                   out_contract->projected_baseline_tier);
      continue;
    }

    loom_value_id_t root_value_id = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_low_residency_resolve_root_value(
        module, loom_low_residency_candidate_source(marker_op), markers.count,
        &root_value_id));
    loom_low_residency_use_list_t uses = {0};
    IREE_RETURN_IF_ERROR(loom_low_residency_collect_leaf_uses(
        module, loom_low_residency_candidate_result(marker_op), markers.count,
        arena, &uses));
    if (uses.count > UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "residency candidate has too many recorded uses");
    }
    const int64_t candidate_id =
        loom_low_residency_candidate_candidate_id(marker_op);
    const int64_t recompute_cost =
        loom_low_residency_candidate_recompute_cost(marker_op);
    if (candidate_id < 0 || (uint64_t)candidate_id > UINT32_MAX ||
        recompute_cost < 0 || (uint64_t)recompute_cost > UINT32_MAX) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low residency candidate metadata is outside uint32");
    }
    if (!loom_low_residency_candidate_sealed(marker_op)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "low residency candidate recipe was not sealed by final "
          "target-low preparation");
    }
    candidate_markers[candidate_marker_count++] =
        (loom_low_residency_marker_info_t){
            .op = marker_op,
            .root_value_id = root_value_id,
        };
    if (uses.count == 0) continue;

    const loom_value_slice_t marker_captures =
        loom_low_residency_candidate_captures(marker_op);
    loom_value_id_t* captures = NULL;
    if (marker_captures.count != 0) {
      IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
          arena, marker_captures.count, sizeof(*captures), (void**)&captures));
      for (iree_host_size_t j = 0; j < marker_captures.count; ++j) {
        IREE_RETURN_IF_ERROR(loom_low_residency_resolve_root_value(
            module, marker_captures.values[j], markers.count, &captures[j]));
      }
    }
    loom_low_residency_recipe_t recipe = {0};
    const loom_value_slice_t marker_recipe =
        loom_low_residency_candidate_recipe(marker_op);
    IREE_RETURN_IF_ERROR(loom_low_residency_collect_recipe(
        module, marker_op, root_value_id, captures, marker_captures.count,
        marker_recipe.values, marker_recipe.count, markers.count, &dominance,
        arena, &recipe));
    candidates[candidate_count] = (loom_low_residency_contract_candidate_t){
        .candidate_id = (uint32_t)candidate_id,
        .recompute_cost = (uint32_t)recompute_cost,
        .value_id = root_value_id,
        .uses = uses.values,
        .use_count = (uint32_t)uses.count,
        .materialization_ops = recipe.ops,
        .materialization_op_count = (uint32_t)recipe.op_count,
        .materialization_inputs = recipe.inputs,
        .materialization_input_count = (uint32_t)recipe.input_count,
        .preserves_baseline =
            loom_low_residency_candidate_preserves_baseline(marker_op),
    };
    ++candidate_count;
  }
  out_contract->candidates = candidates;
  out_contract->candidate_count = candidate_count;

  loom_rewriter_t rewriter;
  IREE_RETURN_IF_ERROR(loom_rewriter_initialize(&rewriter, module, arena));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = candidate_marker_count;
       i > 0 && iree_status_is_ok(status); --i) {
    const loom_low_residency_marker_info_t* marker = &candidate_markers[i - 1];
    status = loom_rewriter_replace_all_uses_and_erase(
        &rewriter, marker->op, &marker->root_value_id, 1);
  }
  for (iree_host_size_t i = markers.count; i > 0 && iree_status_is_ok(status);
       --i) {
    loom_op_t* marker_op = markers.ops[i - 1];
    if (loom_low_residency_require_isa(marker_op)) {
      status = loom_rewriter_erase(&rewriter, marker_op);
    }
  }
  loom_rewriter_deinitialize(&rewriter);
  return status;
}

iree_status_t loom_low_residency_contract_evaluate(
    const loom_low_residency_contract_t* contract,
    const loom_target_residency_model_t* residency_model,
    const loom_low_allocation_table_t* allocation,
    iree_arena_allocator_t* arena, uint32_t* out_tier, bool* out_satisfied) {
  if (contract == NULL || allocation == NULL || arena == NULL ||
      out_tier == NULL || out_satisfied == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "exact residency evaluation requires inputs and "
                            "tier and satisfaction outputs");
  }
  *out_tier = 0;
  *out_satisfied = !contract->has_requirement;
  if (!contract->has_requirement) return iree_ok_status();
  if (loom_target_residency_model_is_empty(residency_model)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "exact residency contract requires a target residency model");
  }
  if (contract->required_tier > residency_model->best_tier) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "required residency tier exceeds target best tier");
  }
  const iree_host_size_t resource_count =
      residency_model->direct_resources.resource_count;
  if (allocation->assigned_extents.count != resource_count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "allocation register classes do not match residency resources");
  }
  uint64_t* direct_resource_units = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, resource_count, sizeof(*direct_resource_units),
      (void**)&direct_resource_units));
  for (iree_host_size_t i = 0; i < resource_count; ++i) {
    direct_resource_units[i] =
        allocation->assigned_extents.ends_by_reg_class[i];
  }
  loom_target_residency_evaluator_t evaluator;
  IREE_RETURN_IF_ERROR(
      loom_target_residency_evaluator_initialize(residency_model, &evaluator));
  IREE_RETURN_IF_ERROR(loom_target_residency_evaluator_evaluate_tier(
      &evaluator, direct_resource_units, resource_count, out_tier));
  *out_satisfied = *out_tier >= contract->required_tier;
  return iree_ok_status();
}

static bool loom_low_residency_candidate_placement_restored(
    const loom_module_t* module,
    const loom_low_residency_contract_candidate_t* candidate) {
  const loom_value_t* value = loom_module_value(module, candidate->value_id);
  const loom_op_t* defining_op =
      loom_value_is_block_arg(value) ? NULL : loom_value_def_op(value);
  for (uint32_t i = 0; i < candidate->use_count; ++i) {
    loom_op_t* user_op = loom_use_user_op(candidate->uses[i]);
    const uint16_t operand_index = loom_use_operand_index(candidate->uses[i]);
    if (user_op != NULL &&
        !iree_any_bit_set(user_op->flags, LOOM_OP_FLAG_DEAD) &&
        operand_index < user_op->operand_count &&
        loom_op_operands(user_op)[operand_index] == candidate->value_id) {
      // An adjacent retained use is already at the materialization boundary;
      // cloning the producer there cannot shorten its exact live range.
      if (defining_op == NULL ||
          user_op->parent_block != defining_op->parent_block ||
          user_op->prev_op != defining_op) {
        return false;
      }
    }
  }
  return true;
}

static bool loom_low_residency_repair_scope_is_valid(
    loom_low_residency_repair_scope_t scope) {
  return scope == LOOM_LOW_RESIDENCY_REPAIR_SCOPE_ALL_CANDIDATES ||
         scope == LOOM_LOW_RESIDENCY_REPAIR_SCOPE_PRESERVED_BASELINE;
}

static bool loom_low_residency_candidate_is_in_repair_scope(
    const loom_low_residency_contract_candidate_t* candidate,
    loom_low_residency_repair_scope_t scope) {
  IREE_ASSERT(loom_low_residency_repair_scope_is_valid(scope));
  return scope == LOOM_LOW_RESIDENCY_REPAIR_SCOPE_ALL_CANDIDATES ||
         candidate->preserves_baseline;
}

iree_status_t loom_low_residency_contract_try_repair(
    loom_module_t* module, const loom_low_allocation_table_t* allocation,
    loom_low_residency_contract_t* contract,
    loom_low_residency_repair_scope_t scope, iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* out_result) {
  if (module == NULL || allocation == NULL || contract == NULL ||
      !loom_low_residency_repair_scope_is_valid(scope) || arena == NULL ||
      out_result == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "residency repair requires contract inputs and "
                            "a result output");
  }
  *out_result = (loom_low_allocation_rematerialization_result_t){
      .value_id = LOOM_VALUE_ID_INVALID,
      .assignment_index = UINT32_MAX,
  };

  for (;;) {
    iree_host_size_t selected_index = IREE_HOST_SIZE_MAX;
    uint32_t selected_recompute_cost = UINT32_MAX;
    uint32_t selected_use_count = UINT32_MAX;
    uint32_t selected_candidate_id = UINT32_MAX;
    for (iree_host_size_t i = 0; i < contract->candidate_count; ++i) {
      const loom_low_residency_contract_candidate_t* candidate =
          &contract->candidates[i];
      if (!candidate->attempted &&
          loom_low_residency_candidate_is_in_repair_scope(candidate, scope) &&
          (candidate->recompute_cost < selected_recompute_cost ||
           (candidate->recompute_cost == selected_recompute_cost &&
            (candidate->use_count < selected_use_count ||
             (candidate->use_count == selected_use_count &&
              candidate->candidate_id < selected_candidate_id))))) {
        selected_index = i;
        selected_recompute_cost = candidate->recompute_cost;
        selected_use_count = candidate->use_count;
        selected_candidate_id = candidate->candidate_id;
      }
    }
    if (selected_index == IREE_HOST_SIZE_MAX) return iree_ok_status();

    loom_low_residency_contract_candidate_t* candidate =
        &contract->candidates[selected_index];
    candidate->attempted = true;
    IREE_RETURN_IF_ERROR(loom_low_allocation_rematerialize_candidate_uses(
        module, allocation, candidate->value_id, candidate->materialization_ops,
        candidate->materialization_op_count, candidate->materialization_inputs,
        candidate->materialization_input_count, candidate->uses,
        candidate->use_count, arena, out_result));
    candidate->cloned_packet_count = out_result->cloned_packet_count;
    candidate->rewritten_operand_count = out_result->rewritten_operand_count;
    candidate->restored =
        loom_low_residency_candidate_placement_restored(module, candidate);
    if (out_result->rewritten_operand_count != 0) return iree_ok_status();
  }
}

iree_status_t loom_low_residency_contract_try_repair_remaining(
    loom_module_t* module, const loom_low_allocation_table_t* allocation,
    loom_low_residency_contract_t* contract,
    loom_low_residency_repair_scope_t scope, iree_arena_allocator_t* arena,
    loom_low_allocation_rematerialization_result_t* out_result,
    uint32_t* out_repaired_candidate_count) {
  if (module == NULL || allocation == NULL || contract == NULL ||
      !loom_low_residency_repair_scope_is_valid(scope) || arena == NULL ||
      out_result == NULL || out_repaired_candidate_count == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "terminal residency repair requires a repaired-candidate output");
  }
  *out_result = (loom_low_allocation_rematerialization_result_t){
      .value_id = LOOM_VALUE_ID_INVALID,
      .assignment_index = UINT32_MAX,
  };
  *out_repaired_candidate_count = 0;
  for (iree_host_size_t i = 0; i < contract->candidate_count; ++i) {
    loom_low_residency_contract_candidate_t* candidate =
        &contract->candidates[i];
    if (candidate->attempted ||
        !loom_low_residency_candidate_is_in_repair_scope(candidate, scope)) {
      continue;
    }
    candidate->attempted = true;
    loom_low_allocation_rematerialization_result_t candidate_result = {0};
    IREE_RETURN_IF_ERROR(loom_low_allocation_rematerialize_candidate_uses(
        module, allocation, candidate->value_id, candidate->materialization_ops,
        candidate->materialization_op_count, candidate->materialization_inputs,
        candidate->materialization_input_count, candidate->uses,
        candidate->use_count, arena, &candidate_result));
    candidate->cloned_packet_count = candidate_result.cloned_packet_count;
    candidate->rewritten_operand_count =
        candidate_result.rewritten_operand_count;
    candidate->restored =
        loom_low_residency_candidate_placement_restored(module, candidate);
    if (candidate_result.rewritten_operand_count == 0) continue;
    if (*out_repaired_candidate_count == 0) {
      out_result->value_id = candidate_result.value_id;
      out_result->assignment_index = candidate_result.assignment_index;
    }
    if (UINT32_MAX - out_result->cloned_packet_count <
            candidate_result.cloned_packet_count ||
        UINT32_MAX - out_result->rewritten_operand_count <
            candidate_result.rewritten_operand_count ||
        *out_repaired_candidate_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "terminal residency repair count exceeds "
                              "uint32");
    }
    out_result->cloned_packet_count += candidate_result.cloned_packet_count;
    out_result->rewritten_operand_count +=
        candidate_result.rewritten_operand_count;
    ++*out_repaired_candidate_count;
  }
  return iree_ok_status();
}

bool loom_low_residency_contract_candidates_exhausted(
    const loom_low_residency_contract_t* contract,
    loom_low_residency_repair_scope_t scope) {
  IREE_ASSERT(loom_low_residency_repair_scope_is_valid(scope));
  for (iree_host_size_t i = 0; i < contract->candidate_count; ++i) {
    const loom_low_residency_contract_candidate_t* candidate =
        &contract->candidates[i];
    if (loom_low_residency_candidate_is_in_repair_scope(candidate, scope) &&
        !candidate->attempted) {
      return false;
    }
  }
  return true;
}

bool loom_low_residency_contract_candidates_restored(
    const loom_low_residency_contract_t* contract,
    loom_low_residency_repair_scope_t scope) {
  IREE_ASSERT(loom_low_residency_repair_scope_is_valid(scope));
  for (iree_host_size_t i = 0; i < contract->candidate_count; ++i) {
    const loom_low_residency_contract_candidate_t* candidate =
        &contract->candidates[i];
    if (loom_low_residency_candidate_is_in_repair_scope(candidate, scope) &&
        !candidate->restored) {
      return false;
    }
  }
  return true;
}

void loom_low_residency_contract_resolve_preserved_baseline(
    loom_low_residency_contract_t* contract, uint32_t exact_baseline_tier) {
  IREE_ASSERT(contract->preserves_baseline);
  IREE_ASSERT(loom_low_residency_contract_candidates_exhausted(
      contract, LOOM_LOW_RESIDENCY_REPAIR_SCOPE_PRESERVED_BASELINE));
  IREE_ASSERT(loom_low_residency_contract_candidates_restored(
      contract, LOOM_LOW_RESIDENCY_REPAIR_SCOPE_PRESERVED_BASELINE));
  contract->required_tier = iree_max(
      contract->has_minimum_requirement ? contract->minimum_required_tier : 0,
      exact_baseline_tier);
  contract->preserved_baseline_resolved = true;
}
