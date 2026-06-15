// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"
#include "loom/ops/func/ops.h"
#include "loom/rewrite/callable.h"
#include "loom/rewrite/materialize.h"

typedef enum loom_callable_outline_value_mark_e {
  LOOM_CALLABLE_OUTLINE_VALUE_MARK_UNSEEN = 0,
  LOOM_CALLABLE_OUTLINE_VALUE_MARK_VISITING = 1,
  LOOM_CALLABLE_OUTLINE_VALUE_MARK_ADDED = 2,
} loom_callable_outline_value_mark_t;

typedef struct loom_callable_outline_range_state_t {
  // Module containing both the source range and outlined callable.
  loom_module_t* module;
  // First root op selected for outlining.
  loom_op_t* first_op;
  // Excluded root op after the selected range, or NULL for end-of-block.
  loom_op_t* after_last_op;
  // Parent block shared by all selected root ops.
  loom_block_t* block;
  // Scratch arena for transient lists, marks, and remap tables.
  iree_arena_allocator_t* arena;
  // Local ordinal domain covering the containing region.
  loom_local_value_domain_t* value_domain;
} loom_callable_outline_range_state_t;

typedef struct loom_callable_outline_value_list_t {
  // Ordered value IDs in first-use/dependency-before-user order.
  loom_value_id_t* values;
  // Number of populated value IDs.
  iree_host_size_t count;
  // Allocated value ID slots.
  iree_host_size_t capacity;
  // Dense local-ordinal mark state.
  uint8_t* marks;
  // Allocated local-ordinal mark slots.
  iree_host_size_t mark_capacity;
} loom_callable_outline_value_list_t;

static bool loom_callable_outline_root_is_selected(
    const loom_callable_outline_range_state_t* state, const loom_op_t* op) {
  if (!op || op->parent_block != state->block) return false;
  for (const loom_op_t* current = state->first_op; current;
       current = current->next_op) {
    if (current == state->after_last_op) break;
    if (current == op) return true;
  }
  return false;
}

static bool loom_callable_outline_op_is_inside_range(
    const loom_callable_outline_range_state_t* state, const loom_op_t* op) {
  for (const loom_op_t* current = op; current; current = current->parent_op) {
    if (loom_callable_outline_root_is_selected(state, current)) return true;
  }
  return false;
}

static bool loom_callable_outline_region_contains_block(
    const loom_region_t* region, const loom_block_t* target_block) {
  if (!region || !target_block) return false;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(region, block_index);
    if (block == target_block) return true;
    const loom_op_t* op = block->first_op;
    while (op) {
      loom_region_t** regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (loom_callable_outline_region_contains_block(regions[i],
                                                        target_block)) {
          return true;
        }
      }
      op = op->next_op;
    }
  }
  return false;
}

static bool loom_callable_outline_block_is_inside_range(
    const loom_callable_outline_range_state_t* state,
    const loom_block_t* block) {
  if (!block) return false;
  if (block == state->block) return false;
  if (block->first_op) {
    return loom_callable_outline_op_is_inside_range(state,
                                                    block->first_op->parent_op);
  }
  for (const loom_op_t* root_op = state->first_op; root_op;
       root_op = root_op->next_op) {
    if (root_op == state->after_last_op) break;
    loom_region_t** regions = loom_op_regions(root_op);
    for (uint8_t i = 0; i < root_op->region_count; ++i) {
      if (loom_callable_outline_region_contains_block(regions[i], block)) {
        return true;
      }
    }
  }
  return false;
}

static bool loom_callable_outline_value_is_defined_inside_range(
    const loom_callable_outline_range_state_t* state,
    loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= state->module->values.count) {
    return false;
  }
  const loom_value_t* value = loom_module_value(state->module, value_id);
  if (loom_value_is_block_arg(value)) {
    return loom_callable_outline_block_is_inside_range(
        state, loom_value_def_block(value));
  }
  return loom_callable_outline_op_is_inside_range(state,
                                                  loom_value_def_op(value));
}

static iree_status_t loom_callable_outline_value_list_initialize(
    const loom_callable_outline_range_state_t* state,
    loom_callable_outline_value_list_t* out_list) {
  memset(out_list, 0, sizeof(*out_list));
  out_list->mark_capacity = state->value_domain->value_count;
  if (out_list->mark_capacity == 0) return iree_ok_status();
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate_array(state->arena, out_list->mark_capacity,
                                sizeof(uint8_t), (void**)&out_list->marks));
  memset(out_list->marks, 0, out_list->mark_capacity * sizeof(uint8_t));
  return iree_ok_status();
}

static iree_status_t loom_callable_outline_value_list_ensure_mark(
    const loom_callable_outline_range_state_t* state,
    loom_callable_outline_value_list_t* list, loom_value_id_t value_id,
    loom_value_ordinal_t* out_value_ordinal) {
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(loom_local_value_domain_register_value(
      state->value_domain, state->arena, value_id, &value_ordinal));
  const iree_host_size_t required_capacity =
      (iree_host_size_t)value_ordinal + 1;
  if (required_capacity > list->mark_capacity) {
    const iree_host_size_t old_capacity = list->mark_capacity;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, old_capacity, required_capacity, sizeof(*list->marks),
        &list->mark_capacity, (void**)&list->marks));
    memset(list->marks + old_capacity, 0,
           (list->mark_capacity - old_capacity) * sizeof(*list->marks));
  }
  *out_value_ordinal = value_ordinal;
  return iree_ok_status();
}

static iree_status_t loom_callable_outline_value_list_append(
    iree_arena_allocator_t* arena, loom_callable_outline_value_list_t* list,
    loom_value_id_t value_id) {
  if (list->count >= list->capacity) {
    iree_host_size_t minimum_capacity = list->count + 1;
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        arena, list->count, minimum_capacity, sizeof(loom_value_id_t),
        &list->capacity, (void**)&list->values));
  }
  list->values[list->count++] = value_id;
  return iree_ok_status();
}

typedef struct loom_callable_outline_pending_refs_t {
  // Scratch arena backing the value list.
  iree_arena_allocator_t* arena;
  // Value refs found by a type walk.
  loom_value_id_t* values;
  // Number of populated value refs.
  iree_host_size_t count;
  // Allocated value ref slots.
  iree_host_size_t capacity;
} loom_callable_outline_pending_refs_t;

static iree_status_t loom_callable_outline_pending_refs_append(
    loom_value_id_t value_id, void* user_data) {
  loom_callable_outline_pending_refs_t* refs =
      (loom_callable_outline_pending_refs_t*)user_data;
  if (refs->count >= refs->capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        refs->arena, refs->count, refs->count + 1, sizeof(loom_value_id_t),
        &refs->capacity, (void**)&refs->values));
  }
  refs->values[refs->count++] = value_id;
  return iree_ok_status();
}

static iree_status_t loom_callable_outline_add_capture(
    const loom_callable_outline_range_state_t* state,
    loom_callable_outline_value_list_t* captures, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= state->module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "capture value %%%u is invalid",
                            (unsigned)value_id);
  }
  if (loom_callable_outline_value_is_defined_inside_range(state, value_id)) {
    return iree_ok_status();
  }
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(loom_callable_outline_value_list_ensure_mark(
      state, captures, value_id, &value_ordinal));
  if (captures->marks[value_ordinal] ==
      LOOM_CALLABLE_OUTLINE_VALUE_MARK_ADDED) {
    return iree_ok_status();
  }
  if (captures->marks[value_ordinal] ==
      LOOM_CALLABLE_OUTLINE_VALUE_MARK_VISITING) {
    return iree_ok_status();
  }
  captures->marks[value_ordinal] = LOOM_CALLABLE_OUTLINE_VALUE_MARK_VISITING;

  loom_callable_outline_pending_refs_t refs = {
      .arena = state->arena,
  };
  IREE_RETURN_IF_ERROR(loom_type_walk_value_refs(
      loom_module_value_type(state->module, value_id),
      loom_callable_outline_pending_refs_append, &refs));
  for (iree_host_size_t i = 0; i < refs.count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_callable_outline_add_capture(state, captures, refs.values[i]));
  }

  IREE_RETURN_IF_ERROR(loom_callable_outline_value_list_append(
      state->arena, captures, value_id));
  captures->marks[value_ordinal] = LOOM_CALLABLE_OUTLINE_VALUE_MARK_ADDED;
  return iree_ok_status();
}

static iree_status_t loom_callable_outline_collect_type_captures(
    const loom_callable_outline_range_state_t* state,
    loom_callable_outline_value_list_t* captures, loom_type_t type) {
  loom_callable_outline_pending_refs_t refs = {
      .arena = state->arena,
  };
  IREE_RETURN_IF_ERROR(loom_type_walk_value_refs(
      type, loom_callable_outline_pending_refs_append, &refs));
  for (iree_host_size_t i = 0; i < refs.count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_callable_outline_add_capture(state, captures, refs.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_callable_outline_collect_attr_captures(
    const loom_callable_outline_range_state_t* state,
    loom_callable_outline_value_list_t* captures, const loom_attribute_t* attr,
    uint8_t depth) {
  if (!attr) return iree_ok_status();
  if (depth > LOOM_ATTR_DICT_MAX_NESTING_DEPTH) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "attribute nesting exceeds maximum depth");
  }
  switch ((loom_attr_kind_t)attr->kind) {
    case LOOM_ATTR_TYPE:
      if (attr->type_id == LOOM_TYPE_ID_INVALID ||
          attr->type_id >= state->module->types.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "type attribute id %u is invalid",
                                (unsigned)attr->type_id);
      }
      return loom_callable_outline_collect_type_captures(
          state, captures, state->module->types.entries[attr->type_id]);
    case LOOM_ATTR_PREDICATE_LIST:
      if (attr->count > 0 && !attr->predicate_list) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "predicate-list attribute payload is NULL");
      }
      for (uint16_t i = 0; i < attr->count; ++i) {
        const loom_predicate_t* predicate = &attr->predicate_list[i];
        for (uint8_t j = 0; j < IREE_ARRAYSIZE(predicate->arg_tags); ++j) {
          if (predicate->arg_tags[j] != LOOM_PRED_ARG_VALUE) continue;
          if (predicate->args[j] < 0 ||
              (uint64_t)predicate->args[j] >= state->module->values.count) {
            return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "predicate value argument is invalid");
          }
          IREE_RETURN_IF_ERROR(loom_callable_outline_add_capture(
              state, captures, (loom_value_id_t)predicate->args[j]));
        }
      }
      return iree_ok_status();
    case LOOM_ATTR_DICT:
      if (attr->count > 0 && !attr->dict_entries) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "dict attribute payload is NULL");
      }
      for (uint16_t i = 0; i < attr->count; ++i) {
        IREE_RETURN_IF_ERROR(loom_callable_outline_collect_attr_captures(
            state, captures, &attr->dict_entries[i].value,
            (uint8_t)(depth + 1)));
      }
      return iree_ok_status();
    case LOOM_ATTR_ENCODING: {
      const loom_encoding_t* encoding =
          loom_module_encoding(state->module, attr->encoding_id);
      if (!encoding) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "encoding attribute id %u is invalid",
                                (unsigned)attr->encoding_id);
      }
      if (encoding->attribute_count > 0 && !encoding->attributes) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "encoding attribute payload is NULL");
      }
      for (uint8_t i = 0; i < encoding->attribute_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_callable_outline_collect_attr_captures(
            state, captures, &encoding->attributes[i].value,
            (uint8_t)(depth + 1)));
      }
      return iree_ok_status();
    }
    default:
      return iree_ok_status();
  }
}

static iree_status_t loom_callable_outline_collect_op_captures(
    const loom_callable_outline_range_state_t* state,
    loom_callable_outline_value_list_t* captures, const loom_op_t* op) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_callable_outline_add_capture(state, captures, operands[i]));
  }

  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == LOOM_VALUE_ID_INVALID ||
        results[i] >= state->module->values.count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "result value on outlined op is invalid");
    }
    IREE_RETURN_IF_ERROR(loom_callable_outline_collect_type_captures(
        state, captures, loom_module_value_type(state->module, results[i])));
  }

  const loom_attribute_t* attrs = loom_op_const_attrs(op);
  for (uint8_t i = 0; i < op->attribute_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_callable_outline_collect_attr_captures(
        state, captures, &attrs[i], 0));
  }

  loom_region_t** regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    const loom_region_t* region = regions[region_index];
    if (!region) continue;
    for (uint16_t block_index = 0; block_index < region->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(region, block_index);
      for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
        loom_value_id_t arg_id = loom_block_arg_id(block, arg_index);
        if (arg_id == LOOM_VALUE_ID_INVALID ||
            arg_id >= state->module->values.count) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "block argument value is invalid");
        }
        IREE_RETURN_IF_ERROR(loom_callable_outline_collect_type_captures(
            state, captures, loom_module_value_type(state->module, arg_id)));
      }
      const loom_op_t* child_op = block->first_op;
      while (child_op) {
        IREE_RETURN_IF_ERROR(loom_callable_outline_collect_op_captures(
            state, captures, child_op));
        child_op = child_op->next_op;
      }
    }
  }
  return iree_ok_status();
}

static bool loom_callable_outline_value_has_use_outside_range(
    const loom_callable_outline_range_state_t* state,
    loom_value_id_t value_id) {
  const loom_value_t* value = loom_module_value(state->module, value_id);
  const loom_use_t* uses = loom_value_uses(value);
  for (uint32_t i = 0; i < value->use_count; ++i) {
    if (!loom_callable_outline_op_is_inside_range(state,
                                                  loom_use_user_op(uses[i]))) {
      return true;
    }
  }

  if (value_id >= state->module->type_uses.value_capacity) return false;
  loom_type_use_id_t use_id =
      state->module->type_uses.value_heads[value_id].first_incoming_use_id;
  while (use_id != LOOM_TYPE_USE_ID_INVALID) {
    const loom_type_use_t* type_use = &state->module->type_uses.records[use_id];
    const loom_value_t* user_value =
        loom_module_value(state->module, type_use->user_value_id);
    bool inside = loom_value_is_block_arg(user_value)
                      ? loom_callable_outline_block_is_inside_range(
                            state, loom_value_def_block(user_value))
                      : loom_callable_outline_op_is_inside_range(
                            state, loom_value_def_op(user_value));
    if (!inside) return true;
    use_id = type_use->next_incoming_use_id;
  }
  return false;
}

static iree_status_t loom_callable_outline_add_live_out(
    const loom_callable_outline_range_state_t* state,
    loom_callable_outline_value_list_t* captures,
    loom_callable_outline_value_list_t* live_outs, loom_value_id_t value_id) {
  if (value_id == LOOM_VALUE_ID_INVALID ||
      value_id >= state->module->values.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "live-out value %%%u is invalid",
                            (unsigned)value_id);
  }
  if (!loom_callable_outline_value_is_defined_inside_range(state, value_id)) {
    return loom_callable_outline_add_capture(state, captures, value_id);
  }
  loom_value_ordinal_t value_ordinal = LOOM_VALUE_ORDINAL_INVALID;
  IREE_RETURN_IF_ERROR(loom_callable_outline_value_list_ensure_mark(
      state, live_outs, value_id, &value_ordinal));
  if (live_outs->marks[value_ordinal] ==
      LOOM_CALLABLE_OUTLINE_VALUE_MARK_ADDED) {
    return iree_ok_status();
  }
  if (live_outs->marks[value_ordinal] ==
      LOOM_CALLABLE_OUTLINE_VALUE_MARK_VISITING) {
    return iree_ok_status();
  }
  live_outs->marks[value_ordinal] = LOOM_CALLABLE_OUTLINE_VALUE_MARK_VISITING;

  loom_callable_outline_pending_refs_t refs = {
      .arena = state->arena,
  };
  IREE_RETURN_IF_ERROR(loom_type_walk_value_refs(
      loom_module_value_type(state->module, value_id),
      loom_callable_outline_pending_refs_append, &refs));
  for (iree_host_size_t i = 0; i < refs.count; ++i) {
    if (loom_callable_outline_value_is_defined_inside_range(state,
                                                            refs.values[i])) {
      IREE_RETURN_IF_ERROR(loom_callable_outline_add_live_out(
          state, captures, live_outs, refs.values[i]));
    } else {
      IREE_RETURN_IF_ERROR(
          loom_callable_outline_add_capture(state, captures, refs.values[i]));
    }
  }

  IREE_RETURN_IF_ERROR(loom_callable_outline_value_list_append(
      state->arena, live_outs, value_id));
  live_outs->marks[value_ordinal] = LOOM_CALLABLE_OUTLINE_VALUE_MARK_ADDED;
  return iree_ok_status();
}

static iree_status_t loom_callable_outline_collect_live_outs(
    const loom_callable_outline_range_state_t* state,
    loom_callable_outline_value_list_t* captures,
    loom_callable_outline_value_list_t* live_outs) {
  for (loom_op_t* op = state->first_op; op && op != state->after_last_op;
       op = op->next_op) {
    const loom_value_id_t* results = loom_op_const_results(op);
    for (uint16_t i = 0; i < op->result_count; ++i) {
      if (results[i] == LOOM_VALUE_ID_INVALID ||
          results[i] >= state->module->values.count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "outlined op result value is invalid");
      }
      if (loom_callable_outline_value_has_use_outside_range(state,
                                                            results[i])) {
        IREE_RETURN_IF_ERROR(loom_callable_outline_add_live_out(
            state, captures, live_outs, results[i]));
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_callable_outline_validate_range(
    loom_rewriter_t* rewriter, loom_op_t* first_op, loom_op_t* after_last_op,
    loom_callable_outline_range_state_t* out_state,
    iree_host_size_t* out_root_count) {
  *out_root_count = 0;
  if (iree_any_bit_set(first_op->flags, LOOM_OP_FLAG_DEAD) ||
      !first_op->parent_block) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "first outlined op must be live and linked");
  }
  if (after_last_op &&
      (iree_any_bit_set(after_last_op->flags, LOOM_OP_FLAG_DEAD) ||
       after_last_op->parent_block != first_op->parent_block)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "outline range end must be live and in the same block as the start");
  }
  if (after_last_op == first_op) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "cannot outline an empty range");
  }
  if (!first_op->parent_op) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "cannot outline module-scope operations");
  }

  loom_callable_outline_range_state_t state = {
      .module = rewriter->module,
      .first_op = first_op,
      .after_last_op = after_last_op,
      .block = first_op->parent_block,
      .arena = rewriter->arena,
  };
  bool found_range_end = after_last_op == NULL;
  for (loom_op_t* op = first_op; op; op = op->next_op) {
    if (op == after_last_op) {
      found_range_end = true;
      break;
    }
    const loom_op_vtable_t* vtable = loom_op_vtable(rewriter->module, op);
    if (!vtable) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "cannot outline an op with no registered vtable");
    }
    if (iree_any_bit_set(vtable->traits, LOOM_TRAIT_TERMINATOR)) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "cannot outline a terminator operation");
    }
    ++*out_root_count;
  }
  if (!found_range_end) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "outline range end is not reachable from start");
  }
  *out_state = state;
  return iree_ok_status();
}

static iree_status_t loom_callable_outline_validate_symbol(
    loom_module_t* module, loom_symbol_ref_t outlined_ref) {
  if (!loom_symbol_ref_is_valid(outlined_ref) || outlined_ref.module_id != 0 ||
      outlined_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "outlined symbol ref {module=%u, symbol=%u} is invalid",
        (unsigned)outlined_ref.module_id, (unsigned)outlined_ref.symbol_id);
  }
  const loom_symbol_t* symbol =
      &module->symbols.entries[outlined_ref.symbol_id];
  if (symbol->defining_op) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "outlined symbol already has a defining op");
  }
  if (symbol->name_id == LOOM_STRING_ID_INVALID ||
      symbol->name_id >= module->strings.count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "outlined symbol name id %u is invalid",
                            (unsigned)symbol->name_id);
  }
  return iree_ok_status();
}

static iree_status_t loom_callable_outline_collect_root_ops(
    const loom_callable_outline_range_state_t* state, iree_host_size_t count,
    loom_op_t*** out_ops) {
  *out_ops = NULL;
  if (count == 0) return iree_ok_status();
  loom_op_t** ops = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      state->arena, count, sizeof(loom_op_t*), (void**)&ops));
  iree_host_size_t index = 0;
  for (loom_op_t* op = state->first_op; op && op != state->after_last_op;
       op = op->next_op) {
    ops[index++] = op;
  }
  *out_ops = ops;
  return iree_ok_status();
}

static iree_status_t loom_callable_outline_make_none_types(
    iree_arena_allocator_t* arena, iree_host_size_t count,
    loom_type_t** out_types) {
  *out_types = NULL;
  if (count == 0) return iree_ok_status();
  loom_type_t* types = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, count, sizeof(loom_type_t), (void**)&types));
  for (iree_host_size_t i = 0; i < count; ++i) {
    types[i] = loom_type_none();
  }
  *out_types = types;
  return iree_ok_status();
}

static iree_status_t loom_callable_outline_build_function(
    loom_rewriter_t* rewriter, const loom_callable_outline_range_state_t* state,
    const loom_callable_outline_value_list_t* captures,
    const loom_callable_outline_value_list_t* live_outs,
    loom_symbol_ref_t outlined_ref, loom_ir_remap_t* body_remap,
    loom_func_like_t* out_func) {
  loom_type_t* arg_types = NULL;
  IREE_RETURN_IF_ERROR(loom_callable_outline_make_none_types(
      state->arena, captures->count, &arg_types));
  loom_type_t* result_types = NULL;
  IREE_RETURN_IF_ERROR(loom_callable_outline_make_none_types(
      state->arena, live_outs->count, &result_types));

  loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  loom_builder_set_block(&rewriter->builder, loom_module_block(state->module));
  loom_op_t* func_op = NULL;
  iree_status_t status = loom_func_def_build(
      &rewriter->builder, 0, 0, 0, 0, 0, 0, loom_symbol_ref_null(), 0,
      loom_named_attr_slice_empty(), LOOM_STRING_ID_INVALID,
      loom_named_attr_slice_empty(), outlined_ref, arg_types, captures->count,
      result_types, live_outs->count, NULL, 0, NULL, 0,
      state->first_op->location, &func_op);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  loom_func_like_t func = loom_func_like_cast(state->module, func_op);
  if (!loom_func_like_isa(func)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "outlined op is not function-like");
  }
  uint16_t arg_count = 0;
  const loom_value_id_t* args = loom_func_like_arg_ids(func, &arg_count);
  if (arg_count != captures->count) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "outlined function argument count does not match captures");
  }

  loom_ir_remap_t remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(state->module, state->module,
                                                state->arena, NULL, &remap));
  for (iree_host_size_t i = 0; i < captures->count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_ir_remap_map_value(&remap, captures->values[i], args[i]));
  }
  for (iree_host_size_t i = 0; i < captures->count; ++i) {
    loom_type_t arg_type = {0};
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        &remap, loom_module_value_type(state->module, captures->values[i]),
        &arg_type));
    IREE_RETURN_IF_ERROR(
        loom_rewriter_set_value_type(rewriter, args[i], arg_type));
  }

  *body_remap = remap;
  *out_func = func;
  return iree_ok_status();
}

static iree_status_t loom_callable_outline_clone_body(
    loom_rewriter_t* rewriter, const loom_callable_outline_range_state_t* state,
    loom_func_like_t func, loom_ir_remap_t* body_remap) {
  loom_region_t* body = loom_func_like_body(func);
  if (!body) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "outlined function has no body");
  }
  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(&rewriter->builder, func.op, body);
  iree_status_t status = iree_ok_status();
  for (loom_op_t* op = state->first_op;
       op && op != state->after_last_op && iree_status_is_ok(status);
       op = op->next_op) {
    loom_op_t* cloned_op = NULL;
    status = loom_ir_clone_op(&rewriter->builder, op, body_remap, &cloned_op);
  }
  loom_builder_restore(&rewriter->builder, saved_ip);
  return status;
}

static iree_status_t loom_callable_outline_build_return(
    loom_rewriter_t* rewriter, const loom_callable_outline_range_state_t* state,
    loom_func_like_t func, const loom_callable_outline_value_list_t* live_outs,
    loom_ir_remap_t* body_remap) {
  loom_value_id_t* returned_values = NULL;
  if (live_outs->count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->arena, live_outs->count, sizeof(loom_value_id_t),
        (void**)&returned_values));
  }
  loom_value_slice_t func_results = loom_func_def_results(func.op);
  if (func_results.count != live_outs->count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "outlined function result count mismatch");
  }
  for (iree_host_size_t i = 0; i < live_outs->count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_resolve_value(
        body_remap, live_outs->values[i], &returned_values[i]));
    loom_type_t result_type = {0};
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        body_remap, loom_module_value_type(state->module, live_outs->values[i]),
        &result_type));
    IREE_RETURN_IF_ERROR(loom_rewriter_set_value_type(
        rewriter, func_results.values[i], result_type));
  }

  loom_region_t* body = loom_func_like_body(func);
  loom_builder_ip_t saved_ip =
      loom_builder_enter_region(&rewriter->builder, func.op, body);
  loom_op_t* return_op = NULL;
  iree_status_t status =
      loom_func_return_build(&rewriter->builder, returned_values,
                             live_outs->count, func.op->location, &return_op);
  loom_builder_restore(&rewriter->builder, saved_ip);
  return status;
}

typedef struct loom_callable_outline_selected_type_ref_check_t {
  // Original source range being replaced.
  const loom_callable_outline_range_state_t* state;
  // Cleared when a remapped caller-side type still references erased values.
  bool valid;
} loom_callable_outline_selected_type_ref_check_t;

static iree_status_t loom_callable_outline_check_no_selected_type_ref(
    loom_value_id_t value_id, void* user_data) {
  loom_callable_outline_selected_type_ref_check_t* check =
      (loom_callable_outline_selected_type_ref_check_t*)user_data;
  if (loom_callable_outline_value_is_defined_inside_range(check->state,
                                                          value_id)) {
    check->valid = false;
  }
  return iree_ok_status();
}

static iree_status_t loom_callable_outline_build_call(
    loom_rewriter_t* rewriter, const loom_callable_outline_range_state_t* state,
    const loom_callable_outline_value_list_t* captures,
    const loom_callable_outline_value_list_t* live_outs,
    loom_symbol_ref_t outlined_ref, loom_op_t** out_call_op) {
  loom_type_t* result_placeholders = NULL;
  IREE_RETURN_IF_ERROR(loom_callable_outline_make_none_types(
      state->arena, live_outs->count, &result_placeholders));

  loom_builder_ip_t saved_ip = loom_builder_save(&rewriter->builder);
  loom_builder_set_before(&rewriter->builder, state->first_op);
  loom_op_t* call_op = NULL;
  iree_status_t status = loom_func_call_build(
      &rewriter->builder, 0, 0, 0, 0, outlined_ref, captures->values,
      captures->count, result_placeholders, live_outs->count, NULL, 0,
      state->first_op->location, &call_op);
  loom_builder_restore(&rewriter->builder, saved_ip);
  IREE_RETURN_IF_ERROR(status);

  loom_value_slice_t call_results = loom_func_call_results(call_op);
  if (call_results.count != live_outs->count) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "outline call result count mismatch");
  }

  loom_ir_remap_options_t remap_options = {
      .allow_unmapped_values = true,
  };
  loom_ir_remap_t call_type_remap = {0};
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(state->module, state->module,
                                                state->arena, &remap_options,
                                                &call_type_remap));
  for (iree_host_size_t i = 0; i < live_outs->count; ++i) {
    IREE_RETURN_IF_ERROR(loom_ir_remap_map_value(
        &call_type_remap, live_outs->values[i], call_results.values[i]));
  }
  for (iree_host_size_t i = 0; i < live_outs->count; ++i) {
    loom_type_t result_type = {0};
    IREE_RETURN_IF_ERROR(loom_ir_remap_type(
        &call_type_remap,
        loom_module_value_type(state->module, live_outs->values[i]),
        &result_type));
    loom_callable_outline_selected_type_ref_check_t check = {
        .state = state,
        .valid = true,
    };
    IREE_RETURN_IF_ERROR(loom_type_walk_value_refs(
        result_type, loom_callable_outline_check_no_selected_type_ref, &check));
    if (!check.valid) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "outlined call result type still references an erased value");
    }
    IREE_RETURN_IF_ERROR(loom_rewriter_set_value_type(
        rewriter, call_results.values[i], result_type));
  }

  *out_call_op = call_op;
  return iree_ok_status();
}

static iree_status_t loom_callable_outline_replace_live_outs(
    loom_rewriter_t* rewriter,
    const loom_callable_outline_value_list_t* live_outs, loom_op_t* call_op) {
  loom_value_slice_t call_results = loom_func_call_results(call_op);
  for (iree_host_size_t i = 0; i < live_outs->count; ++i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_replace_all_uses_with(
        rewriter, live_outs->values[i], call_results.values[i]));
  }
  return iree_ok_status();
}

static iree_status_t loom_callable_outline_erase_roots(
    loom_rewriter_t* rewriter, loom_op_t** roots, iree_host_size_t root_count) {
  for (iree_host_size_t i = root_count; i > 0; --i) {
    IREE_RETURN_IF_ERROR(loom_rewriter_erase(rewriter, roots[i - 1]));
  }
  return iree_ok_status();
}

iree_status_t loom_callable_outline_range(
    loom_rewriter_t* rewriter, loom_op_t* first_op, loom_op_t* after_last_op,
    loom_symbol_ref_t outlined_ref,
    loom_callable_outline_result_t* out_result) {
  *out_result = (loom_callable_outline_result_t){0};

  loom_callable_outline_range_state_t state = {0};
  iree_host_size_t root_count = 0;
  IREE_RETURN_IF_ERROR(loom_callable_outline_validate_range(
      rewriter, first_op, after_last_op, &state, &root_count));
  IREE_RETURN_IF_ERROR(
      loom_callable_outline_validate_symbol(state.module, outlined_ref));

  loom_op_t** roots = NULL;
  IREE_RETURN_IF_ERROR(
      loom_callable_outline_collect_root_ops(&state, root_count, &roots));

  loom_local_value_domain_t value_domain = {0};
  IREE_RETURN_IF_ERROR(loom_local_value_domain_acquire_for_region(
      state.module, state.block->parent_region, state.arena, &value_domain));
  state.value_domain = &value_domain;

  iree_status_t status = iree_ok_status();
  loom_callable_outline_value_list_t captures = {0};
  status = loom_callable_outline_value_list_initialize(&state, &captures);
  for (iree_host_size_t i = 0; i < root_count && iree_status_is_ok(status);
       ++i) {
    status =
        loom_callable_outline_collect_op_captures(&state, &captures, roots[i]);
  }

  loom_callable_outline_value_list_t live_outs = {0};
  if (iree_status_is_ok(status)) {
    status = loom_callable_outline_value_list_initialize(&state, &live_outs);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_callable_outline_collect_live_outs(&state, &captures, &live_outs);
  }

  loom_ir_remap_t body_remap = {0};
  loom_func_like_t outlined = {0};
  if (iree_status_is_ok(status)) {
    status = loom_callable_outline_build_function(rewriter, &state, &captures,
                                                  &live_outs, outlined_ref,
                                                  &body_remap, &outlined);
  }
  if (iree_status_is_ok(status)) {
    status = loom_callable_outline_clone_body(rewriter, &state, outlined,
                                              &body_remap);
  }
  if (iree_status_is_ok(status)) {
    status = loom_callable_outline_build_return(rewriter, &state, outlined,
                                                &live_outs, &body_remap);
  }

  loom_op_t* call_op = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_callable_outline_build_call(
        rewriter, &state, &captures, &live_outs, outlined_ref, &call_op);
  }
  if (iree_status_is_ok(status)) {
    status =
        loom_callable_outline_replace_live_outs(rewriter, &live_outs, call_op);
  }
  if (iree_status_is_ok(status)) {
    status = loom_callable_outline_erase_roots(rewriter, roots, root_count);
  }

  if (iree_status_is_ok(status)) {
    out_result->outlined = outlined;
    out_result->call_op = call_op;
  }
  loom_local_value_domain_release(&value_domain);
  return status;
}
