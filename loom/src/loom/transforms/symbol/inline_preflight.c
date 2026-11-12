// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/transforms/symbol/inline_preflight.h"

#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/op_defs.h"
#include "loom/rewrite/callable.h"

static_assert(LOOM_INLINE_BLOCKER_COUNT <= UINT8_MAX,
              "cached preflight blockers must fit in one byte");

static bool loom_inline_op_is_inside_region(const loom_op_t* op,
                                            const loom_region_t* region) {
  for (const loom_op_t* current = op; current; current = current->parent_op) {
    const loom_region_t* parent_region =
        current->parent_block ? current->parent_block->parent_region : NULL;
    if (parent_region == region) {
      return true;
    }
  }
  return false;
}

static loom_inline_blocker_t loom_inline_validate_call_kind(
    const loom_inline_plan_entry_t* entry) {
  switch (loom_call_like_kind(entry->call)) {
    case LOOM_CALL_LIKE_KIND_SEMANTIC:
    case LOOM_CALL_LIKE_KIND_TEMPLATE:
    case LOOM_CALL_LIKE_KIND_COMPOSITION:
    case LOOM_CALL_LIKE_KIND_COMMAND_PROGRAM:
      if (loom_func_like_isa(entry->callee)) {
        return LOOM_INLINE_BLOCKER_NONE;
      }
      return LOOM_INLINE_BLOCKER_FUNC_CALL_TARGET_NOT_FUNCTION_LIKE;
    case LOOM_CALL_LIKE_KIND_LOW_INTERNAL:
      if (loom_low_func_def_isa(entry->callee.op)) {
        return LOOM_INLINE_BLOCKER_NONE;
      }
      return LOOM_INLINE_BLOCKER_LOW_CALLEE_KIND;
    default:
      return LOOM_INLINE_BLOCKER_UNSUPPORTED_CALL_KIND;
  }
}

static void loom_inline_inspect_low_body_op(const loom_module_t* module,
                                            const loom_op_t* op,
                                            bool schedule_locked,
                                            loom_inline_symbol_info_t* symbol,
                                            loom_inline_blocker_t* blocker) {
  loom_inline_blocker_t local_blocker = LOOM_INLINE_BLOCKER_NONE;
  if (loom_low_resource_isa(op) || loom_low_live_in_isa(op)) {
    local_blocker = LOOM_INLINE_BLOCKER_LOW_ENTRY_RESOURCE;
    symbol->body_has_resources |= loom_low_resource_isa(op);
  } else if (schedule_locked && op->region_count != 0) {
    local_blocker = LOOM_INLINE_BLOCKER_LOW_LOCKED_NESTED_REGION;
  } else if (loom_low_storage_reserve_isa(op)) {
    symbol->body_has_storage = true;
    const loom_type_t storage_type =
        loom_module_value_type(module, loom_low_storage_reserve_storage(op));
    symbol->body_has_workgroup_storage |=
        loom_type_storage_space(storage_type) == LOOM_STORAGE_SPACE_WORKGROUP;
  }
  if (*blocker == LOOM_INLINE_BLOCKER_NONE) {
    *blocker = local_blocker;
  }
  loom_region_t* const* regions = loom_op_regions(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    const loom_region_t* region = regions[region_index];
    if (!region) {
      continue;
    }
    for (uint16_t block_index = 0; block_index < region->block_count;
         ++block_index) {
      const loom_block_t* block = loom_region_const_block(region, block_index);
      const loom_op_t* child_op = NULL;
      loom_block_for_each_op(block, child_op) {
        loom_inline_inspect_low_body_op(module, child_op, schedule_locked,
                                        symbol, blocker);
      }
    }
  }
}

loom_inline_blocker_t loom_inline_preflight_site(
    const loom_inline_callables_plan_t* state,
    const loom_inline_plan_entry_t* entry) {
  const loom_module_t* module = state->module;
  if (!loom_call_like_isa(entry->call)) {
    return LOOM_INLINE_BLOCKER_CALL_NOT_CALL_LIKE;
  }
  if (entry->source_symbol_id >= module->symbols.count) {
    return LOOM_INLINE_BLOCKER_CALL_NOT_OWNED_BY_SYMBOL;
  }
  if (entry->target_symbol_id >= module->symbols.count) {
    return LOOM_INLINE_BLOCKER_INVALID_CALLEE_SYMBOL;
  }
  if (!loom_func_like_isa(entry->callee)) {
    return LOOM_INLINE_BLOCKER_CALLEE_NOT_FUNCTION_LIKE;
  }
  loom_inline_blocker_t call_kind_blocker =
      loom_inline_validate_call_kind(entry);
  if (call_kind_blocker != LOOM_INLINE_BLOCKER_NONE) {
    return call_kind_blocker;
  }

  if (loom_call_like_operand_offset(entry->call) != 0 ||
      loom_call_like_result_offset(entry->call) != 0 ||
      entry->call_op->region_count != 0 ||
      entry->call_op->successor_count != 0) {
    return LOOM_INLINE_BLOCKER_NON_CALL_SHAPE;
  }

  loom_region_t* body = loom_func_like_body(entry->callee);
  if (!body) {
    return LOOM_INLINE_BLOCKER_CALLEE_MISSING_BODY;
  }
  if (body->block_count == 0) {
    return LOOM_INLINE_BLOCKER_CALLEE_BODY_EMPTY;
  }
  if (loom_inline_op_is_inside_region(entry->call_op, body)) {
    return LOOM_INLINE_BLOCKER_RECURSIVE_BODY;
  }

  const loom_region_descriptor_t* body_descriptor =
      loom_func_like_body_region_descriptor(module, entry->callee);
  if (!body_descriptor || body_descriptor->terminator == LOOM_OP_KIND_UNKNOWN) {
    return LOOM_INLINE_BLOCKER_CALLEE_BODY_INVALID_TERMINATOR;
  }

  // Input verification establishes the callee's signature and CFG. The call
  // site must still be able to represent the cloned control flow.
  if (!state->symbols[entry->target_symbol_id].body_will_be_linear &&
      !loom_callable_call_site_allows_cfg_splice(module, entry->call_op)) {
    return LOOM_INLINE_BLOCKER_CALLER_REGION_NOT_CFG_CAPABLE;
  }

  return LOOM_INLINE_BLOCKER_NONE;
}

static loom_inline_blocker_t loom_inline_inspect_low_body(
    const loom_module_t* module, loom_inline_symbol_info_t* symbol) {
  if (!loom_low_func_def_isa(symbol->function.op)) {
    return LOOM_INLINE_BLOCKER_NONE;
  }
  const loom_region_t* body = loom_func_like_body(symbol->function);
  if (!body || body->block_count == 0) {
    return LOOM_INLINE_BLOCKER_NONE;
  }
  loom_inline_blocker_t blocker = LOOM_INLINE_BLOCKER_NONE;
  const uint8_t allocation = loom_low_func_def_allocation(symbol->function.op);
  if (allocation != 0 && allocation != LOOM_LOW_ALLOCATION_VIRTUAL) {
    blocker = LOOM_INLINE_BLOCKER_LOW_ALLOCATION;
  }
  const bool schedule_locked =
      loom_low_func_def_schedule(symbol->function.op) ==
      LOOM_LOW_SCHEDULE_LOCKED;
  for (uint16_t block_index = 0; block_index < body->block_count;
       ++block_index) {
    const loom_block_t* block = loom_region_const_block(body, block_index);
    const loom_op_t* body_op = NULL;
    loom_block_for_each_op(block, body_op) {
      loom_inline_inspect_low_body_op(module, body_op, schedule_locked, symbol,
                                      &blocker);
    }
  }
  return blocker;
}

loom_inline_blocker_t loom_inline_preflight_body(
    loom_inline_callables_plan_t* state, loom_symbol_id_t symbol_id) {
  loom_inline_symbol_info_t* symbol = &state->symbols[symbol_id];
  if (symbol->body_preflight == 0) {
    symbol->body_preflight =
        (uint8_t)loom_inline_inspect_low_body(state->module, symbol) + 1;
  }
  return (loom_inline_blocker_t)(symbol->body_preflight - 1);
}

loom_inline_blocker_t loom_inline_preflight_entry(
    loom_inline_callables_plan_t* state,
    const loom_inline_plan_entry_t* entry) {
  const loom_inline_blocker_t site_blocker =
      loom_inline_preflight_site(state, entry);
  return site_blocker != LOOM_INLINE_BLOCKER_NONE
             ? site_blocker
             : loom_inline_preflight_body(state, entry->target_symbol_id);
}
