// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/liveness_uses.h"

#include "loom/ir/ancestry.h"
#include "loom/ir/module.h"
#include "loom/ir/types.h"

typedef struct loom_liveness_type_ref_callback_state_t {
  // Receives each semantic read that survives the current filter.
  loom_liveness_value_callback_t visitor;
} loom_liveness_type_ref_callback_state_t;

static iree_status_t loom_liveness_type_ref_callback(loom_value_id_t value_id,
                                                     void* user_data) {
  loom_liveness_type_ref_callback_state_t* state =
      (loom_liveness_type_ref_callback_state_t*)user_data;
  return state->visitor.fn(state->visitor.user_data, value_id);
}

iree_status_t loom_liveness_for_each_type_ref(
    const loom_module_t* module, loom_type_t type,
    loom_liveness_value_callback_t visitor) {
  loom_liveness_type_ref_callback_state_t state = {
      .visitor = visitor,
  };
  return loom_type_walk_value_refs(module, type,
                                   loom_liveness_type_ref_callback, &state);
}

static bool loom_liveness_op_defines_value(const loom_op_t* op,
                                           loom_value_id_t value_id) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    if (results[i] == value_id) {
      return true;
    }
  }
  return false;
}

typedef struct loom_liveness_external_use_state_t {
  // Module owning the captured values and their types.
  const loom_module_t* module;
  // Defines the subtree whose local values are excluded.
  const loom_op_t* owner_op;
  // Receives each semantic read that survives the current filter.
  loom_liveness_value_callback_t visitor;
} loom_liveness_external_use_state_t;

static iree_status_t loom_liveness_external_value_callback(
    void* user_data, loom_value_id_t value_id) {
  loom_liveness_external_use_state_t* state =
      (loom_liveness_external_use_state_t*)user_data;
  if (loom_op_subtree_defines_value(state->module, state->owner_op, value_id)) {
    return iree_ok_status();
  }
  return state->visitor.fn(state->visitor.user_data, value_id);
}

static iree_status_t loom_liveness_for_each_region_external_use(
    loom_liveness_external_use_state_t* state, const loom_region_t* region) {
  const loom_block_t* block = NULL;
  loom_region_for_each_block(region, block) {
    for (uint16_t i = 0; i < block->arg_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_liveness_for_each_type_ref(
          state->module, loom_block_arg_type(state->module, block, i),
          loom_liveness_value_callback_make(
              loom_liveness_external_value_callback, state)));
    }
    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      const loom_value_id_t* operands = loom_op_const_operands(op);
      for (uint16_t i = 0; i < op->operand_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_liveness_external_value_callback(state, operands[i]));
        IREE_RETURN_IF_ERROR(loom_liveness_for_each_type_ref(
            state->module, loom_module_value_type(state->module, operands[i]),
            loom_liveness_value_callback_make(
                loom_liveness_external_value_callback, state)));
      }
      const loom_value_id_t* results = loom_op_const_results(op);
      for (uint16_t i = 0; i < op->result_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_liveness_for_each_type_ref(
            state->module, loom_module_value_type(state->module, results[i]),
            loom_liveness_value_callback_make(
                loom_liveness_external_value_callback, state)));
      }
      loom_region_t* const* regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        IREE_RETURN_IF_ERROR(
            loom_liveness_for_each_region_external_use(state, regions[i]));
      }
    }
  }
  return iree_ok_status();
}

iree_status_t loom_liveness_for_each_nested_external_use(
    const loom_module_t* module, const loom_op_t* owner_op,
    loom_liveness_value_callback_t visitor) {
  loom_liveness_external_use_state_t state = {
      .module = module,
      .owner_op = owner_op,
      .visitor = visitor,
  };
  loom_region_t* const* regions = loom_op_regions(owner_op);
  for (uint8_t i = 0; i < owner_op->region_count; ++i) {
    IREE_RETURN_IF_ERROR(
        loom_liveness_for_each_region_external_use(&state, regions[i]));
  }
  return iree_ok_status();
}

typedef struct loom_liveness_result_type_ref_state_t {
  // Operation whose results must not count as type-reference reads.
  const loom_op_t* op;
  // Receives each semantic read that survives the current filter.
  loom_liveness_value_callback_t visitor;
} loom_liveness_result_type_ref_state_t;

static iree_status_t loom_liveness_result_type_ref_callback(
    loom_value_id_t value_id, void* user_data) {
  loom_liveness_result_type_ref_state_t* state =
      (loom_liveness_result_type_ref_state_t*)user_data;
  if (loom_liveness_op_defines_value(state->op, value_id)) {
    return iree_ok_status();
  }
  return state->visitor.fn(state->visitor.user_data, value_id);
}

iree_status_t loom_liveness_for_each_op_direct_use(
    const loom_module_t* module, const loom_op_t* op,
    loom_liveness_value_callback_t visitor) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  for (uint16_t i = 0; i < op->operand_count; ++i) {
    IREE_RETURN_IF_ERROR(visitor.fn(visitor.user_data, operands[i]));
    IREE_RETURN_IF_ERROR(loom_liveness_for_each_type_ref(
        module, loom_module_value_type(module, operands[i]), visitor));
  }
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint16_t i = 0; i < op->result_count; ++i) {
    loom_type_t result_type = loom_module_value_type(module, results[i]);
    loom_liveness_result_type_ref_state_t state = {
        .op = op,
        .visitor = visitor,
    };
    IREE_RETURN_IF_ERROR(loom_type_walk_value_refs(
        module, result_type, loom_liveness_result_type_ref_callback, &state));
  }
  return iree_ok_status();
}

iree_status_t loom_liveness_for_each_op_use(
    const loom_module_t* module, const loom_op_t* op,
    loom_liveness_value_callback_t visitor) {
  IREE_RETURN_IF_ERROR(
      loom_liveness_for_each_op_direct_use(module, op, visitor));
  return loom_liveness_for_each_nested_external_use(module, op, visitor);
}
