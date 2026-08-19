// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/symbol_equivalence.h"

#include <string.h>

#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/link/symbol_policy.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/remap.h"

typedef enum loom_symbol_equivalence_pair_state_e {
  LOOM_SYMBOL_EQUIVALENCE_PAIR_COMPARING = 0,
  LOOM_SYMBOL_EQUIVALENCE_PAIR_EQUAL = 1,
  LOOM_SYMBOL_EQUIVALENCE_PAIR_DIFFERENT = 2,
} loom_symbol_equivalence_pair_state_t;

typedef struct loom_symbol_equivalence_symbol_pair_t {
  // Symbol in the left definition closure.
  const loom_symbol_t* lhs;
  // Corresponding symbol in the right definition closure.
  const loom_symbol_t* rhs;
  // Recursive comparison state for this pair.
  loom_symbol_equivalence_pair_state_t state;
} loom_symbol_equivalence_symbol_pair_t;

typedef struct loom_symbol_equivalence_block_pair_t {
  // Block in the left definition closure.
  const loom_block_t* lhs;
  // Corresponding block in the right definition closure.
  const loom_block_t* rhs;
} loom_symbol_equivalence_block_pair_t;

typedef struct loom_symbol_equivalence_state_t {
  // Verified module containing both definitions.
  const loom_module_t* module;
  // Arena for all transient maps and pair tables.
  iree_arena_allocator_t* arena;

  // Alpha-renaming state for SSA values across both definition closures.
  struct {
    // Sparse left-to-right value lookup.
    loom_ir_remap_t lhs_to_rhs;
    // Sparse right-to-left value lookup enforcing a bijection.
    loom_ir_remap_t rhs_to_lhs;
    // Left values in insertion order for type remapping.
    loom_value_id_t* lhs;
    // Right values parallel to |lhs|.
    loom_value_id_t* rhs;
    // Number of mapped value pairs.
    iree_host_size_t count;
    // Allocated entries in |lhs| and |rhs|.
    iree_host_size_t capacity;
  } values;

  // Alpha-renaming state for blocks across both definition closures.
  struct {
    // Mapped block pairs.
    loom_symbol_equivalence_block_pair_t* pairs;
    // Number of mapped block pairs.
    iree_host_size_t count;
    // Allocated entries in |pairs|.
    iree_host_size_t capacity;
  } blocks;

  // Recursive private-symbol closure comparison state.
  struct {
    // Mapped symbol pairs.
    loom_symbol_equivalence_symbol_pair_t* pairs;
    // Number of mapped symbol pairs.
    iree_host_size_t count;
    // Allocated entries in |pairs|.
    iree_host_size_t capacity;
  } symbols;
} loom_symbol_equivalence_state_t;

static iree_status_t loom_symbol_equivalence_compare_symbol_refs(
    loom_symbol_equivalence_state_t* state, loom_symbol_ref_t lhs_ref,
    loom_symbol_ref_t rhs_ref, bool* out_equivalent);

static iree_status_t loom_symbol_equivalence_reserve_values(
    loom_symbol_equivalence_state_t* state,
    iree_host_size_t required_capacity) {
  if (required_capacity <= state->values.capacity) return iree_ok_status();

  iree_host_size_t lhs_capacity = state->values.capacity;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      state->arena, state->values.count, required_capacity,
      sizeof(*state->values.lhs), &lhs_capacity, (void**)&state->values.lhs));
  iree_host_size_t rhs_capacity = state->values.capacity;
  IREE_RETURN_IF_ERROR(iree_arena_grow_array(
      state->arena, state->values.count, required_capacity,
      sizeof(*state->values.rhs), &rhs_capacity, (void**)&state->values.rhs));
  state->values.capacity =
      lhs_capacity < rhs_capacity ? lhs_capacity : rhs_capacity;
  return iree_ok_status();
}

static iree_status_t loom_symbol_equivalence_map_value(
    loom_symbol_equivalence_state_t* state, loom_value_id_t lhs_value,
    loom_value_id_t rhs_value, bool* out_equivalent) {
  *out_equivalent = false;
  if (lhs_value == LOOM_VALUE_ID_INVALID ||
      rhs_value == LOOM_VALUE_ID_INVALID) {
    *out_equivalent = lhs_value == rhs_value;
    return iree_ok_status();
  }

  loom_value_id_t mapped_rhs = LOOM_VALUE_ID_INVALID;
  loom_value_id_t mapped_lhs = LOOM_VALUE_ID_INVALID;
  const bool has_lhs = loom_ir_remap_try_lookup_value(&state->values.lhs_to_rhs,
                                                      lhs_value, &mapped_rhs);
  const bool has_rhs = loom_ir_remap_try_lookup_value(&state->values.rhs_to_lhs,
                                                      rhs_value, &mapped_lhs);
  if (has_lhs || has_rhs) {
    *out_equivalent = has_lhs && has_rhs && mapped_rhs == rhs_value &&
                      mapped_lhs == lhs_value;
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(
      loom_ir_remap_map_value(&state->values.lhs_to_rhs, lhs_value, rhs_value));
  IREE_RETURN_IF_ERROR(
      loom_ir_remap_map_value(&state->values.rhs_to_lhs, rhs_value, lhs_value));
  IREE_RETURN_IF_ERROR(
      loom_symbol_equivalence_reserve_values(state, state->values.count + 1));
  state->values.lhs[state->values.count] = lhs_value;
  state->values.rhs[state->values.count] = rhs_value;
  ++state->values.count;
  *out_equivalent = true;
  return iree_ok_status();
}

static bool loom_symbol_equivalence_values_equal(
    const loom_symbol_equivalence_state_t* state, loom_value_id_t lhs_value,
    loom_value_id_t rhs_value) {
  if (lhs_value == LOOM_VALUE_ID_INVALID ||
      rhs_value == LOOM_VALUE_ID_INVALID) {
    return lhs_value == rhs_value;
  }
  loom_value_id_t mapped_rhs = LOOM_VALUE_ID_INVALID;
  return loom_ir_remap_try_lookup_value(&state->values.lhs_to_rhs, lhs_value,
                                        &mapped_rhs) &&
         mapped_rhs == rhs_value;
}

static bool loom_symbol_equivalence_types_equal(
    const loom_symbol_equivalence_state_t* state, loom_type_t lhs_type,
    loom_type_t rhs_type) {
  loom_type_value_remap_t remap = {
      .source_values = state->values.lhs,
      .target_values = state->values.rhs,
      .count = state->values.count,
  };
  return loom_type_equal_after_value_remap(state->module, lhs_type, rhs_type,
                                           &remap);
}

static iree_status_t loom_symbol_equivalence_map_block(
    loom_symbol_equivalence_state_t* state, const loom_block_t* lhs_block,
    const loom_block_t* rhs_block, bool* out_equivalent) {
  *out_equivalent = false;
  for (iree_host_size_t i = 0; i < state->blocks.count; ++i) {
    const loom_symbol_equivalence_block_pair_t* pair = &state->blocks.pairs[i];
    if (pair->lhs == lhs_block || pair->rhs == rhs_block) {
      *out_equivalent = pair->lhs == lhs_block && pair->rhs == rhs_block;
      return iree_ok_status();
    }
  }
  if (state->blocks.count >= state->blocks.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, state->blocks.count, state->blocks.count + 1,
        sizeof(*state->blocks.pairs), &state->blocks.capacity,
        (void**)&state->blocks.pairs));
  }
  state->blocks.pairs[state->blocks.count++] =
      (loom_symbol_equivalence_block_pair_t){
          .lhs = lhs_block,
          .rhs = rhs_block,
      };
  *out_equivalent = true;
  return iree_ok_status();
}

static bool loom_symbol_equivalence_blocks_equal(
    const loom_symbol_equivalence_state_t* state, const loom_block_t* lhs_block,
    const loom_block_t* rhs_block) {
  for (iree_host_size_t i = 0; i < state->blocks.count; ++i) {
    const loom_symbol_equivalence_block_pair_t* pair = &state->blocks.pairs[i];
    if (pair->lhs == lhs_block) return pair->rhs == rhs_block;
  }
  return false;
}

static iree_status_t loom_symbol_equivalence_compare_predicates(
    loom_symbol_equivalence_state_t* state, const loom_attribute_t* lhs_attr,
    const loom_attribute_t* rhs_attr, bool* out_equivalent) {
  *out_equivalent = false;
  if (lhs_attr->count != rhs_attr->count) return iree_ok_status();
  for (uint16_t i = 0; i < lhs_attr->count; ++i) {
    const loom_predicate_t* lhs = &lhs_attr->predicate_list[i];
    const loom_predicate_t* rhs = &rhs_attr->predicate_list[i];
    if (lhs->kind != rhs->kind || lhs->arg_count != rhs->arg_count) {
      return iree_ok_status();
    }
    for (uint8_t j = 0; j < lhs->arg_count; ++j) {
      if (lhs->arg_tags[j] != rhs->arg_tags[j]) return iree_ok_status();
      switch ((loom_predicate_arg_tag_t)lhs->arg_tags[j]) {
        case LOOM_PRED_ARG_NONE:
          break;
        case LOOM_PRED_ARG_CONST:
          if (lhs->args[j] != rhs->args[j]) return iree_ok_status();
          break;
        case LOOM_PRED_ARG_VALUE:
          if (!loom_symbol_equivalence_values_equal(
                  state, (loom_value_id_t)lhs->args[j],
                  (loom_value_id_t)rhs->args[j])) {
            return iree_ok_status();
          }
          break;
        default:
          IREE_ASSERT_UNREACHABLE("verified predicate has an invalid arg tag");
          IREE_BUILTIN_UNREACHABLE();
      }
    }
  }
  *out_equivalent = true;
  return iree_ok_status();
}

static iree_status_t loom_symbol_equivalence_compare_attributes(
    loom_symbol_equivalence_state_t* state, const loom_attribute_t* lhs_attr,
    const loom_attribute_t* rhs_attr, iree_host_size_t depth,
    bool* out_equivalent) {
  *out_equivalent = false;
  if (lhs_attr->kind != rhs_attr->kind) return iree_ok_status();
  switch ((loom_attr_kind_t)lhs_attr->kind) {
    case LOOM_ATTR_SYMBOL:
      return loom_symbol_equivalence_compare_symbol_refs(
          state, lhs_attr->symbol, rhs_attr->symbol, out_equivalent);
    case LOOM_ATTR_TYPE:
      *out_equivalent = loom_symbol_equivalence_types_equal(
          state, state->module->types.entries[lhs_attr->type_id],
          state->module->types.entries[rhs_attr->type_id]);
      return iree_ok_status();
    case LOOM_ATTR_PREDICATE_LIST:
      return loom_symbol_equivalence_compare_predicates(
          state, lhs_attr, rhs_attr, out_equivalent);
    case LOOM_ATTR_DICT:
      if (lhs_attr->count != rhs_attr->count ||
          depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_ok_status();
      }
      for (uint16_t i = 0; i < lhs_attr->count; ++i) {
        const loom_named_attr_t* lhs_entry = &lhs_attr->dict_entries[i];
        const loom_named_attr_t* rhs_entry = &rhs_attr->dict_entries[i];
        if (lhs_entry->name_id != rhs_entry->name_id) return iree_ok_status();
        bool entry_equivalent = false;
        IREE_RETURN_IF_ERROR(loom_symbol_equivalence_compare_attributes(
            state, &lhs_entry->value, &rhs_entry->value, depth + 1,
            &entry_equivalent));
        if (!entry_equivalent) return iree_ok_status();
      }
      *out_equivalent = true;
      return iree_ok_status();
    case LOOM_ATTR_PARAMETERIZED:
      if (lhs_attr->reserved_1 != rhs_attr->reserved_1 ||
          lhs_attr->count != rhs_attr->count ||
          depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_ok_status();
      }
      for (uint16_t i = 0; i < lhs_attr->count; ++i) {
        bool slot_equivalent = false;
        IREE_RETURN_IF_ERROR(loom_symbol_equivalence_compare_attributes(
            state, &lhs_attr->parameterized_slots[i],
            &rhs_attr->parameterized_slots[i], depth + 1, &slot_equivalent));
        if (!slot_equivalent) return iree_ok_status();
      }
      *out_equivalent = true;
      return iree_ok_status();
    case LOOM_ATTR_PARAMETERIZED_ARRAY:
      if (lhs_attr->count != rhs_attr->count ||
          depth >= LOOM_ATTR_AGGREGATE_MAX_NESTING_DEPTH) {
        return iree_ok_status();
      }
      for (uint16_t i = 0; i < lhs_attr->count; ++i) {
        bool element_equivalent = false;
        IREE_RETURN_IF_ERROR(loom_symbol_equivalence_compare_attributes(
            state, &lhs_attr->parameterized_array[i],
            &rhs_attr->parameterized_array[i], depth + 1, &element_equivalent));
        if (!element_equivalent) return iree_ok_status();
      }
      *out_equivalent = true;
      return iree_ok_status();
    default:
      *out_equivalent = loom_attribute_equal(lhs_attr, rhs_attr);
      return iree_ok_status();
  }
}

static iree_status_t loom_symbol_equivalence_compare_op(
    loom_symbol_equivalence_state_t* state, const loom_op_t* lhs_op,
    const loom_op_t* rhs_op, bool* out_equivalent);

static iree_status_t loom_symbol_equivalence_compare_block(
    loom_symbol_equivalence_state_t* state, const loom_block_t* lhs_block,
    const loom_block_t* rhs_block, bool* out_equivalent) {
  *out_equivalent = false;
  if (lhs_block->arg_count != rhs_block->arg_count ||
      lhs_block->op_count != rhs_block->op_count ||
      ((lhs_block->flags ^ rhs_block->flags) &
       (loom_block_flags_t)~LOOM_BLOCK_SOURCE_PRESENTATION_FLAG_MASK) != 0) {
    return iree_ok_status();
  }

  for (uint16_t i = 0; i < lhs_block->arg_count; ++i) {
    bool arg_equivalent = false;
    IREE_RETURN_IF_ERROR(loom_symbol_equivalence_map_value(
        state, loom_block_arg_id(lhs_block, i), loom_block_arg_id(rhs_block, i),
        &arg_equivalent));
    if (!arg_equivalent) return iree_ok_status();
  }
  for (uint16_t i = 0; i < lhs_block->arg_count; ++i) {
    if (!loom_symbol_equivalence_types_equal(
            state,
            loom_module_value_type(state->module,
                                   loom_block_arg_id(lhs_block, i)),
            loom_module_value_type(state->module,
                                   loom_block_arg_id(rhs_block, i)))) {
      return iree_ok_status();
    }
  }

  const loom_op_t* lhs_op = lhs_block->first_op;
  const loom_op_t* rhs_op = rhs_block->first_op;
  while (lhs_op && rhs_op) {
    bool op_equivalent = false;
    IREE_RETURN_IF_ERROR(loom_symbol_equivalence_compare_op(
        state, lhs_op, rhs_op, &op_equivalent));
    if (!op_equivalent) return iree_ok_status();
    lhs_op = lhs_op->next_op;
    rhs_op = rhs_op->next_op;
  }
  *out_equivalent = lhs_op == NULL && rhs_op == NULL;
  return iree_ok_status();
}

static iree_status_t loom_symbol_equivalence_compare_region(
    loom_symbol_equivalence_state_t* state, const loom_region_t* lhs_region,
    const loom_region_t* rhs_region, bool* out_equivalent) {
  *out_equivalent = false;
  if (lhs_region->block_count != rhs_region->block_count ||
      lhs_region->flags != rhs_region->flags) {
    return iree_ok_status();
  }

  for (uint16_t i = 0; i < lhs_region->block_count; ++i) {
    bool block_equivalent = false;
    IREE_RETURN_IF_ERROR(loom_symbol_equivalence_map_block(
        state, loom_region_const_block(lhs_region, i),
        loom_region_const_block(rhs_region, i), &block_equivalent));
    if (!block_equivalent) return iree_ok_status();
  }
  for (uint16_t i = 0; i < lhs_region->block_count; ++i) {
    bool block_equivalent = false;
    IREE_RETURN_IF_ERROR(loom_symbol_equivalence_compare_block(
        state, loom_region_const_block(lhs_region, i),
        loom_region_const_block(rhs_region, i), &block_equivalent));
    if (!block_equivalent) return iree_ok_status();
  }
  *out_equivalent = true;
  return iree_ok_status();
}

static iree_status_t loom_symbol_equivalence_compare_op(
    loom_symbol_equivalence_state_t* state, const loom_op_t* lhs_op,
    const loom_op_t* rhs_op, bool* out_equivalent) {
  *out_equivalent = false;
  if (lhs_op->kind != rhs_op->kind ||
      lhs_op->operand_count != rhs_op->operand_count ||
      lhs_op->result_count != rhs_op->result_count ||
      lhs_op->tied_result_count != rhs_op->tied_result_count ||
      lhs_op->region_count != rhs_op->region_count ||
      lhs_op->successor_count != rhs_op->successor_count ||
      lhs_op->attribute_count != rhs_op->attribute_count ||
      lhs_op->instance_flags != rhs_op->instance_flags ||
      lhs_op->traits != rhs_op->traits) {
    return iree_ok_status();
  }

  const loom_value_id_t* lhs_operands = loom_op_const_operands(lhs_op);
  const loom_value_id_t* rhs_operands = loom_op_const_operands(rhs_op);
  for (uint16_t i = 0; i < lhs_op->operand_count; ++i) {
    if (!loom_symbol_equivalence_values_equal(state, lhs_operands[i],
                                              rhs_operands[i])) {
      return iree_ok_status();
    }
  }

  const loom_value_id_t* lhs_results = loom_op_const_results(lhs_op);
  const loom_value_id_t* rhs_results = loom_op_const_results(rhs_op);
  for (uint16_t i = 0; i < lhs_op->result_count; ++i) {
    bool result_equivalent = false;
    IREE_RETURN_IF_ERROR(loom_symbol_equivalence_map_value(
        state, lhs_results[i], rhs_results[i], &result_equivalent));
    if (!result_equivalent) return iree_ok_status();
  }
  for (uint16_t i = 0; i < lhs_op->result_count; ++i) {
    if (!loom_symbol_equivalence_types_equal(
            state, loom_module_value_type(state->module, lhs_results[i]),
            loom_module_value_type(state->module, rhs_results[i]))) {
      return iree_ok_status();
    }
  }

  if (lhs_op->tied_result_count > 0 &&
      memcmp(loom_op_tied_results(lhs_op), loom_op_tied_results(rhs_op),
             (iree_host_size_t)lhs_op->tied_result_count *
                 sizeof(loom_tied_result_t)) != 0) {
    return iree_ok_status();
  }
  const loom_op_vtable_t* vtable = loom_op_vtable(state->module, lhs_op);
  const uint8_t segment_count = loom_op_vtable_operand_segment_count(vtable);
  if (segment_count > 0 &&
      memcmp(loom_op_const_operand_segment_counts(lhs_op),
             loom_op_const_operand_segment_counts(rhs_op),
             (iree_host_size_t)segment_count * sizeof(uint16_t)) != 0) {
    return iree_ok_status();
  }

  const loom_attribute_t* lhs_attrs = loom_op_const_attrs(lhs_op);
  const loom_attribute_t* rhs_attrs = loom_op_const_attrs(rhs_op);
  for (uint8_t i = 0; i < lhs_op->attribute_count; ++i) {
    bool attr_equivalent = false;
    IREE_RETURN_IF_ERROR(loom_symbol_equivalence_compare_attributes(
        state, &lhs_attrs[i], &rhs_attrs[i], /*depth=*/0, &attr_equivalent));
    if (!attr_equivalent) return iree_ok_status();
  }

  loom_block_t* const* lhs_successors = loom_op_const_successors(lhs_op);
  loom_block_t* const* rhs_successors = loom_op_const_successors(rhs_op);
  for (uint8_t i = 0; i < lhs_op->successor_count; ++i) {
    if (!loom_symbol_equivalence_blocks_equal(state, lhs_successors[i],
                                              rhs_successors[i])) {
      return iree_ok_status();
    }
  }
  for (uint8_t i = 0; i < lhs_op->region_count; ++i) {
    bool region_equivalent = false;
    IREE_RETURN_IF_ERROR(loom_symbol_equivalence_compare_region(
        state, loom_op_regions(lhs_op)[i], loom_op_regions(rhs_op)[i],
        &region_equivalent));
    if (!region_equivalent) return iree_ok_status();
  }

  *out_equivalent = true;
  return iree_ok_status();
}

static iree_status_t loom_symbol_equivalence_map_function_arguments(
    loom_symbol_equivalence_state_t* state, const loom_symbol_t* lhs_symbol,
    const loom_symbol_t* rhs_symbol, bool* out_equivalent) {
  *out_equivalent = false;
  loom_func_like_t lhs =
      loom_func_like_cast(state->module, lhs_symbol->defining_op);
  loom_func_like_t rhs =
      loom_func_like_cast(state->module, rhs_symbol->defining_op);
  if (loom_func_like_isa(lhs) != loom_func_like_isa(rhs)) {
    return iree_ok_status();
  }
  if (!loom_func_like_isa(lhs)) {
    *out_equivalent = true;
    return iree_ok_status();
  }

  uint16_t lhs_count = 0;
  uint16_t rhs_count = 0;
  const loom_value_id_t* lhs_args = loom_func_like_arg_ids(lhs, &lhs_count);
  const loom_value_id_t* rhs_args = loom_func_like_arg_ids(rhs, &rhs_count);
  if (lhs_count != rhs_count) return iree_ok_status();
  for (uint16_t i = 0; i < lhs_count; ++i) {
    bool arg_equivalent = false;
    IREE_RETURN_IF_ERROR(loom_symbol_equivalence_map_value(
        state, lhs_args[i], rhs_args[i], &arg_equivalent));
    if (!arg_equivalent) return iree_ok_status();
  }
  *out_equivalent = true;
  return iree_ok_status();
}

static iree_status_t loom_symbol_equivalence_compare_symbol_refs(
    loom_symbol_equivalence_state_t* state, loom_symbol_ref_t lhs_ref,
    loom_symbol_ref_t rhs_ref, bool* out_equivalent) {
  *out_equivalent = false;
  if (!loom_symbol_ref_is_valid(lhs_ref) ||
      !loom_symbol_ref_is_valid(rhs_ref)) {
    *out_equivalent = !loom_symbol_ref_is_valid(lhs_ref) &&
                      !loom_symbol_ref_is_valid(rhs_ref);
    return iree_ok_status();
  }
  if (lhs_ref.module_id == rhs_ref.module_id &&
      lhs_ref.symbol_id == rhs_ref.symbol_id) {
    *out_equivalent = true;
    return iree_ok_status();
  }

  const loom_symbol_t* lhs = &state->module->symbols.entries[lhs_ref.symbol_id];
  const loom_symbol_t* rhs = &state->module->symbols.entries[rhs_ref.symbol_id];
  if (loom_link_symbol_has_global_identity(state->module, lhs) ||
      loom_link_symbol_has_global_identity(state->module, rhs)) {
    return iree_ok_status();
  }

  for (iree_host_size_t i = 0; i < state->symbols.count; ++i) {
    const loom_symbol_equivalence_symbol_pair_t* pair =
        &state->symbols.pairs[i];
    if (pair->lhs == lhs || pair->rhs == rhs) {
      *out_equivalent = pair->lhs == lhs && pair->rhs == rhs &&
                        pair->state != LOOM_SYMBOL_EQUIVALENCE_PAIR_DIFFERENT;
      return iree_ok_status();
    }
  }

  if (state->symbols.count >= state->symbols.capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, state->symbols.count, state->symbols.count + 1,
        sizeof(*state->symbols.pairs), &state->symbols.capacity,
        (void**)&state->symbols.pairs));
  }
  const iree_host_size_t pair_index = state->symbols.count++;
  state->symbols.pairs[pair_index] = (loom_symbol_equivalence_symbol_pair_t){
      .lhs = lhs,
      .rhs = rhs,
      .state = LOOM_SYMBOL_EQUIVALENCE_PAIR_COMPARING,
  };

  bool equivalent = lhs->kind == rhs->kind && lhs->flags == rhs->flags &&
                    lhs->defining_op != NULL && rhs->defining_op != NULL;
  if (equivalent) {
    IREE_RETURN_IF_ERROR(loom_symbol_equivalence_map_function_arguments(
        state, lhs, rhs, &equivalent));
  }
  if (equivalent) {
    IREE_RETURN_IF_ERROR(loom_symbol_equivalence_compare_op(
        state, lhs->defining_op, rhs->defining_op, &equivalent));
  }
  state->symbols.pairs[pair_index].state =
      equivalent ? LOOM_SYMBOL_EQUIVALENCE_PAIR_EQUAL
                 : LOOM_SYMBOL_EQUIVALENCE_PAIR_DIFFERENT;
  *out_equivalent = equivalent;
  return iree_ok_status();
}

iree_status_t loom_symbol_definitions_equivalent(
    const loom_module_t* module, loom_symbol_ref_t lhs_ref,
    loom_symbol_ref_t rhs_ref, iree_arena_allocator_t* scratch_arena,
    bool* out_equivalent) {
  *out_equivalent = false;
  loom_symbol_equivalence_state_t state = {
      .module = module,
      .arena = scratch_arena,
  };
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(module, (loom_module_t*)module,
                                                scratch_arena, /*options=*/NULL,
                                                &state.values.lhs_to_rhs));
  IREE_RETURN_IF_ERROR(loom_ir_remap_initialize(module, (loom_module_t*)module,
                                                scratch_arena, /*options=*/NULL,
                                                &state.values.rhs_to_lhs));
  return loom_symbol_equivalence_compare_symbol_refs(&state, lhs_ref, rhs_ref,
                                                     out_equivalent);
}
