// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/value_relation.h"

#include "loom/ir/context.h"
#include "loom/util/cfg_graph.h"

typedef enum loom_value_relation_iterator_phase_e {
  LOOM_VALUE_RELATION_PHASE_TIED_RESULT = 0,
  LOOM_VALUE_RELATION_PHASE_FACT_IDENTITY = 1,
  LOOM_VALUE_RELATION_PHASE_VALUE_ALIAS = 2,
  LOOM_VALUE_RELATION_PHASE_SELECT_PAYLOAD = 3,
  LOOM_VALUE_RELATION_PHASE_ELEMENTWISE = 4,
  LOOM_VALUE_RELATION_PHASE_CFG_ARGUMENT = 5,
  LOOM_VALUE_RELATION_PHASE_LOOP_ENTRY = 6,
  LOOM_VALUE_RELATION_PHASE_LOOP_BYPASS = 7,
  LOOM_VALUE_RELATION_PHASE_LOOP_TERMINATOR = 8,
  LOOM_VALUE_RELATION_PHASE_REGION_TERMINATOR = 9,
  LOOM_VALUE_RELATION_PHASE_END = 10,
} loom_value_relation_iterator_phase_t;

static bool loom_value_relation_kind_enabled(
    const loom_value_relation_iterator_t* iterator,
    loom_value_relation_kind_t kind) {
  return iree_any_bit_set(iterator->relation_mask,
                          LOOM_VALUE_RELATION_MASK(kind));
}

static void loom_value_relation_advance_phase(
    loom_value_relation_iterator_t* iterator) {
  ++iterator->phase;
  iterator->outer_index = 0;
  iterator->inner_index = 0;
  iterator->select_payload_ordinal = 0;
}

static bool loom_value_relation_emit(loom_value_id_t source_value_id,
                                     loom_value_id_t destination_value_id,
                                     uint16_t source_operand_index,
                                     loom_value_relation_kind_t kind,
                                     loom_value_relation_t* out_relation) {
  *out_relation = (loom_value_relation_t){
      .source_value_id = source_value_id,
      .destination_value_id = destination_value_id,
      .source_operand_index = source_operand_index,
      .kind = kind,
  };
  return true;
}

static bool loom_value_relation_next_tied_result(
    loom_value_relation_iterator_t* iterator,
    loom_value_relation_t* out_relation) {
  if (!loom_value_relation_kind_enabled(iterator,
                                        LOOM_VALUE_RELATION_TIED_RESULT)) {
    return false;
  }
  const loom_op_t* op = iterator->op;
  if (iterator->outer_index >= op->tied_result_count) return false;
  const loom_tied_result_t tied =
      loom_op_tied_results(op)[iterator->outer_index++];
  IREE_ASSERT(tied.operand_index < op->operand_count &&
                  tied.result_index < op->result_count,
              "verified tied result must reference existing values");
  const bool emitted = loom_value_relation_emit(
      loom_op_const_operands(op)[tied.operand_index],
      loom_op_const_results(op)[tied.result_index], tied.operand_index,
      LOOM_VALUE_RELATION_TIED_RESULT, out_relation);
  if (tied.has_type_change) {
    out_relation->flags |= LOOM_VALUE_RELATION_FLAG_TYPE_CHANGE;
  }
  return emitted;
}

static bool loom_value_relation_next_fact_identity(
    loom_value_relation_iterator_t* iterator,
    loom_value_relation_t* out_relation) {
  const loom_op_t* op = iterator->op;
  if (!loom_value_relation_kind_enabled(iterator,
                                        LOOM_VALUE_RELATION_FACT_IDENTITY) ||
      !loom_traits_are_fact_identity(op->traits)) {
    return false;
  }
  IREE_ASSERT_EQ(op->operand_count, op->result_count,
                 "verified fact identity fields must have equal arity");
  if (iterator->outer_index >= op->result_count) return false;
  const uint16_t index = iterator->outer_index++;
  return loom_value_relation_emit(
      loom_op_const_operands(op)[index], loom_op_const_results(op)[index],
      index, LOOM_VALUE_RELATION_FACT_IDENTITY, out_relation);
}

static bool loom_value_relation_next_value_alias(
    loom_value_relation_iterator_t* iterator,
    loom_value_relation_t* out_relation) {
  const loom_op_t* op = iterator->op;
  if (!loom_value_relation_kind_enabled(iterator,
                                        LOOM_VALUE_RELATION_VALUE_ALIAS) ||
      !loom_traits_are_value_alias(op->traits) || iterator->outer_index != 0) {
    return false;
  }
  IREE_ASSERT(op->operand_count >= 1 && op->result_count == 1,
              "verified value alias must have operand zero and one result");
  iterator->outer_index = 1;
  return loom_value_relation_emit(
      loom_op_const_operands(op)[0], loom_op_const_results(op)[0], 0,
      LOOM_VALUE_RELATION_VALUE_ALIAS, out_relation);
}

static bool loom_value_relation_next_select_payload(
    loom_value_relation_iterator_t* iterator,
    loom_value_relation_t* out_relation) {
  const loom_op_t* op = iterator->op;
  if (!loom_value_relation_kind_enabled(iterator,
                                        LOOM_VALUE_RELATION_SELECT_PAYLOAD) ||
      iterator->vtable == NULL || op->result_count == 0 ||
      !iree_any_bit_set(iterator->vtable->operand_role_mask,
                        LOOM_OPERAND_ROLE_MASK_SELECT_PAYLOAD)) {
    return false;
  }
  while (iterator->outer_index < op->operand_count) {
    const uint16_t operand_index = iterator->outer_index++;
    if (loom_op_operand_role_at(iterator->vtable, op, operand_index) !=
        LOOM_OPERAND_ROLE_SELECT_PAYLOAD) {
      continue;
    }
    const uint16_t result_index =
        iterator->select_payload_ordinal++ % op->result_count;
    return loom_value_relation_emit(
        loom_op_const_operands(op)[operand_index],
        loom_op_const_results(op)[result_index], operand_index,
        LOOM_VALUE_RELATION_SELECT_PAYLOAD, out_relation);
  }
  return false;
}

static bool loom_value_relation_elementwise_operand_is_data(
    const loom_value_relation_iterator_t* iterator, uint16_t operand_index) {
  const loom_operand_role_t role =
      loom_op_operand_role_at(iterator->vtable, iterator->op, operand_index);
  return role != LOOM_OPERAND_ROLE_CONTROL_CONDITION &&
         role != LOOM_OPERAND_ROLE_SELECT_CONDITION &&
         role != LOOM_OPERAND_ROLE_SELECT_PAYLOAD;
}

static bool loom_value_relation_next_elementwise(
    loom_value_relation_iterator_t* iterator,
    loom_value_relation_t* out_relation) {
  const loom_op_t* op = iterator->op;
  if (!loom_value_relation_kind_enabled(iterator,
                                        LOOM_VALUE_RELATION_ELEMENTWISE) ||
      !iree_any_bit_set(op->traits, LOOM_TRAIT_ELEMENTWISE) ||
      op->result_count == 0) {
    return false;
  }
  while (iterator->outer_index < op->operand_count) {
    if (!loom_value_relation_elementwise_operand_is_data(
            iterator, iterator->outer_index)) {
      ++iterator->outer_index;
      iterator->inner_index = 0;
      continue;
    }
    if (iterator->inner_index < op->result_count) {
      const uint16_t operand_index = iterator->outer_index;
      const uint16_t result_index = iterator->inner_index++;
      return loom_value_relation_emit(
          loom_op_const_operands(op)[operand_index],
          loom_op_const_results(op)[result_index], operand_index,
          LOOM_VALUE_RELATION_ELEMENTWISE, out_relation);
    }
    ++iterator->outer_index;
    iterator->inner_index = 0;
  }
  return false;
}

static bool loom_value_relation_next_cfg_argument(
    loom_value_relation_iterator_t* iterator,
    loom_value_relation_t* out_relation) {
  const loom_op_t* op = iterator->op;
  if (!loom_value_relation_kind_enabled(iterator,
                                        LOOM_VALUE_RELATION_CFG_ARGUMENT)) {
    return false;
  }
  while (iterator->outer_index < op->successor_count) {
    const loom_block_t* successor =
        loom_op_const_successors(op)[iterator->outer_index];
    const loom_value_id_t* payload = NULL;
    uint16_t payload_count = 0;
    if (!loom_cfg_terminator_payload_for_successor(op, successor, &payload,
                                                   &payload_count)) {
      ++iterator->outer_index;
      iterator->inner_index = 0;
      continue;
    }
    IREE_ASSERT_EQ(payload_count, successor->arg_count,
                   "verified CFG payload must match destination arguments");
    if (iterator->inner_index < payload_count) {
      const uint16_t index = iterator->inner_index++;
      const iree_host_size_t operand_index =
          (iree_host_size_t)(payload - loom_op_const_operands(op)) + index;
      IREE_ASSERT_LT(operand_index, op->operand_count);
      return loom_value_relation_emit(
          payload[index], loom_block_arg_id(successor, index),
          (uint16_t)operand_index, LOOM_VALUE_RELATION_CFG_ARGUMENT,
          out_relation);
    }
    ++iterator->outer_index;
    iterator->inner_index = 0;
  }
  return false;
}

static bool loom_value_relation_next_loop_entry(
    loom_value_relation_iterator_t* iterator,
    loom_value_relation_t* out_relation) {
  if (!loom_value_relation_kind_enabled(iterator,
                                        LOOM_VALUE_RELATION_LOOP_CARRIED) ||
      iterator->vtable == NULL || iterator->vtable->loop_like == NULL) {
    return false;
  }
  const loom_op_t* op = iterator->op;
  const loom_loop_like_t loop = {
      .op = (loom_op_t*)op,
      .vtable = iterator->vtable->loop_like,
  };
  const loom_value_slice_t iter_args = loom_loop_like_iter_args(loop);
  if (iterator->outer_index >= iter_args.count) return false;
  IREE_ASSERT_EQ(iter_args.count, op->result_count,
                 "verified loop state must match result arity");
  loom_region_t* entry_region = loom_loop_like_condition_region(loop);
  uint16_t block_arg_offset = 0;
  if (entry_region == NULL) {
    entry_region = loom_loop_like_body(loop);
    block_arg_offset =
        loop.vtable->iv_block_arg_index == LOOM_BLOCK_ARG_INDEX_NONE
            ? 0
            : (uint16_t)(loop.vtable->iv_block_arg_index + 1);
  }
  IREE_ASSERT(entry_region != NULL && entry_region->block_count == 1,
              "verified loop entry region must have one block");
  const loom_block_t* entry_block = loom_region_const_entry_block(entry_region);
  IREE_ASSERT_EQ(entry_block->arg_count,
                 (uint16_t)(iter_args.count + block_arg_offset),
                 "verified loop entry arguments must match carried state");
  const uint16_t index = iterator->outer_index++;
  const iree_host_size_t operand_index =
      (iree_host_size_t)(iter_args.values - loom_op_const_operands(op)) + index;
  IREE_ASSERT_LT(operand_index, op->operand_count);
  return loom_value_relation_emit(
      iter_args.values[index],
      loom_block_arg_id(entry_block, (uint16_t)(block_arg_offset + index)),
      (uint16_t)operand_index, LOOM_VALUE_RELATION_LOOP_CARRIED, out_relation);
}

static bool loom_value_relation_next_loop_bypass(
    loom_value_relation_iterator_t* iterator,
    loom_value_relation_t* out_relation) {
  if (!loom_value_relation_kind_enabled(iterator,
                                        LOOM_VALUE_RELATION_LOOP_BYPASS) ||
      iterator->vtable == NULL || iterator->vtable->loop_like == NULL) {
    return false;
  }
  const loom_op_t* op = iterator->op;
  const loom_loop_like_t loop = {
      .op = (loom_op_t*)op,
      .vtable = iterator->vtable->loop_like,
  };
  if (loom_loop_like_condition_region(loop) != NULL) return false;
  const loom_value_slice_t iter_args = loom_loop_like_iter_args(loop);
  IREE_ASSERT_EQ(iter_args.count, op->result_count,
                 "verified counted-loop state must match result arity");
  if (iterator->outer_index >= iter_args.count) return false;
  const uint16_t index = iterator->outer_index++;
  const iree_host_size_t operand_index =
      (iree_host_size_t)(iter_args.values - loom_op_const_operands(op)) + index;
  IREE_ASSERT_LT(operand_index, op->operand_count);
  return loom_value_relation_emit(
      iter_args.values[index], loom_op_const_results(op)[index],
      (uint16_t)operand_index, LOOM_VALUE_RELATION_LOOP_BYPASS, out_relation);
}

static bool loom_value_relation_is_direct_region_terminator(
    const loom_op_t* op, const loom_region_t* region) {
  if (op->parent_block == NULL || region == NULL || region->block_count != 1 ||
      op->parent_block != loom_region_const_entry_block(region)) {
    return false;
  }
  return op->parent_block->last_op == op;
}

static bool loom_value_relation_next_loop_terminator(
    loom_value_relation_iterator_t* iterator,
    loom_value_relation_t* out_relation) {
  if (!loom_value_relation_kind_enabled(iterator,
                                        LOOM_VALUE_RELATION_LOOP_CARRIED)) {
    return false;
  }
  const loom_op_t* op = iterator->op;
  loom_op_t* parent_op = op->parent_op;
  if (parent_op == NULL) return false;
  const loom_op_vtable_t* parent_vtable =
      loom_op_vtable(iterator->module, parent_op);
  if (parent_vtable == NULL || parent_vtable->loop_like == NULL) return false;
  const loom_loop_like_t loop = {
      .op = parent_op,
      .vtable = parent_vtable->loop_like,
  };
  loom_region_t* body = loom_loop_like_body(loop);
  loom_region_t* condition = loom_loop_like_condition_region(loop);
  const bool is_body =
      loom_value_relation_is_direct_region_terminator(op, body);
  const bool is_condition =
      loom_value_relation_is_direct_region_terminator(op, condition);
  if (!is_body && !is_condition) return false;

  const uint16_t state_count = parent_op->result_count;
  IREE_ASSERT_EQ(loom_loop_like_iter_args(loop).count, state_count,
                 "verified loop state must match result arity");
  const loom_block_t* body_block = loom_region_const_entry_block(body);
  const uint16_t body_arg_offset =
      loop.vtable->iv_block_arg_index == LOOM_BLOCK_ARG_INDEX_NONE
          ? 0
          : (uint16_t)(loop.vtable->iv_block_arg_index + 1);
  IREE_ASSERT_EQ(body_block->arg_count,
                 (uint16_t)(state_count + body_arg_offset),
                 "verified loop body arguments must match carried state");

  uint16_t source_operand_index = iterator->outer_index;
  uint8_t destination_count = 0;
  loom_value_id_t destinations[2] = {LOOM_VALUE_ID_INVALID,
                                     LOOM_VALUE_ID_INVALID};
  if (is_condition) {
    IREE_ASSERT_EQ(body_arg_offset, 0,
                   "condition loops must not have an induction variable");
    IREE_ASSERT(condition != NULL && condition->block_count == 1,
                "verified loop condition region must have one block");
    IREE_ASSERT_EQ(op->operand_count, (uint16_t)(state_count + 1),
                   "verified loop condition must forward every state value");
    if (iterator->outer_index >= state_count) return false;
    source_operand_index = (uint16_t)(iterator->outer_index + 1);
    destinations[0] = loom_block_arg_id(body_block, iterator->outer_index);
    destinations[1] = loom_op_const_results(parent_op)[iterator->outer_index];
    destination_count = 2;
  } else if (condition != NULL) {
    IREE_ASSERT_EQ(op->operand_count, state_count,
                   "verified loop body yield must match carried state");
    if (iterator->outer_index >= state_count) return false;
    const loom_block_t* condition_block =
        loom_region_const_entry_block(condition);
    IREE_ASSERT_EQ(condition_block->arg_count, state_count,
                   "verified loop condition arguments must match state");
    destinations[0] = loom_block_arg_id(condition_block, iterator->outer_index);
    destination_count = 1;
  } else {
    IREE_ASSERT_EQ(op->operand_count, state_count,
                   "verified counted-loop yield must match carried state");
    if (iterator->outer_index >= state_count) return false;
    destinations[0] = loom_block_arg_id(
        body_block, (uint16_t)(body_arg_offset + iterator->outer_index));
    destinations[1] = loom_op_const_results(parent_op)[iterator->outer_index];
    destination_count = 2;
  }

  IREE_ASSERT_LT(iterator->inner_index, destination_count);
  const loom_value_id_t destination = destinations[iterator->inner_index++];
  if (iterator->inner_index == destination_count) {
    ++iterator->outer_index;
    iterator->inner_index = 0;
  }
  return loom_value_relation_emit(
      loom_op_const_operands(op)[source_operand_index], destination,
      source_operand_index, LOOM_VALUE_RELATION_LOOP_CARRIED, out_relation);
}

static bool loom_value_relation_next_region_terminator(
    loom_value_relation_iterator_t* iterator,
    loom_value_relation_t* out_relation) {
  if (!loom_value_relation_kind_enabled(iterator,
                                        LOOM_VALUE_RELATION_REGION_RESULT)) {
    return false;
  }
  const loom_op_t* op = iterator->op;
  loom_op_t* parent_op = op->parent_op;
  if (parent_op == NULL) return false;
  const loom_op_vtable_t* parent_vtable =
      loom_op_vtable(iterator->module, parent_op);
  if (parent_vtable == NULL || parent_vtable->region_branch == NULL ||
      op->parent_block == NULL || op->parent_block->parent_region == NULL ||
      op->parent_block->last_op != op) {
    return false;
  }
  IREE_ASSERT_EQ(op->operand_count, parent_op->result_count,
                 "verified region yield must match parent results");
  if (iterator->outer_index >= op->operand_count) return false;
  const uint16_t index = iterator->outer_index++;
  return loom_value_relation_emit(loom_op_const_operands(op)[index],
                                  loom_op_const_results(parent_op)[index],
                                  index, LOOM_VALUE_RELATION_REGION_RESULT,
                                  out_relation);
}

void loom_value_relation_iterator_initialize(
    const loom_module_t* module, const loom_op_t* op,
    loom_value_relation_mask_t relation_mask,
    loom_value_relation_iterator_t* out_iterator) {
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(op);
  IREE_ASSERT_ARGUMENT(out_iterator);
  const loom_op_vtable_t* vtable = loom_op_vtable(module, op);
  loom_value_relation_mask_t possible_mask = 0;
  if (op->tied_result_count != 0) {
    possible_mask |= LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_TIED_RESULT);
  }
  if (loom_traits_are_fact_identity(op->traits)) {
    possible_mask |=
        LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_FACT_IDENTITY);
  }
  if (loom_traits_are_value_alias(op->traits)) {
    possible_mask |= LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_VALUE_ALIAS);
  }
  if (vtable != NULL &&
      iree_any_bit_set(vtable->operand_role_mask,
                       LOOM_OPERAND_ROLE_MASK_SELECT_PAYLOAD)) {
    possible_mask |=
        LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_SELECT_PAYLOAD);
  }
  if (iree_any_bit_set(op->traits, LOOM_TRAIT_ELEMENTWISE)) {
    possible_mask |= LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_ELEMENTWISE);
  }
  if (op->successor_count != 0) {
    possible_mask |= LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_CFG_ARGUMENT);
  }
  if (vtable != NULL && vtable->loop_like != NULL) {
    possible_mask |= LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_LOOP_CARRIED);
    if (vtable->loop_like->condition_region_index == LOOM_REGION_INDEX_NONE) {
      possible_mask |=
          LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_LOOP_BYPASS);
    }
  }
  if (op->parent_op != NULL && op->parent_block != NULL &&
      op->parent_block->last_op == op) {
    const loom_op_vtable_t* parent_vtable =
        loom_op_vtable(module, op->parent_op);
    if (parent_vtable != NULL && parent_vtable->loop_like != NULL) {
      possible_mask |=
          LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_LOOP_CARRIED);
    }
    if (parent_vtable != NULL && parent_vtable->region_branch != NULL) {
      possible_mask |=
          LOOM_VALUE_RELATION_MASK(LOOM_VALUE_RELATION_REGION_RESULT);
    }
  }
  *out_iterator = (loom_value_relation_iterator_t){
      .module = module,
      .op = op,
      .vtable = vtable,
      .relation_mask =
          relation_mask & possible_mask & LOOM_VALUE_RELATION_MASK_ALL,
  };
}

bool loom_value_relation_iterator_next(loom_value_relation_iterator_t* iterator,
                                       loom_value_relation_t* out_relation) {
  IREE_ASSERT_ARGUMENT(iterator);
  IREE_ASSERT_ARGUMENT(iterator->module);
  IREE_ASSERT_ARGUMENT(iterator->op);
  IREE_ASSERT_ARGUMENT(out_relation);
  if (iterator->relation_mask == 0) return false;
  while (iterator->phase < LOOM_VALUE_RELATION_PHASE_END) {
    bool found = false;
    switch ((loom_value_relation_iterator_phase_t)iterator->phase) {
      case LOOM_VALUE_RELATION_PHASE_TIED_RESULT:
        found = loom_value_relation_next_tied_result(iterator, out_relation);
        break;
      case LOOM_VALUE_RELATION_PHASE_FACT_IDENTITY:
        found = loom_value_relation_next_fact_identity(iterator, out_relation);
        break;
      case LOOM_VALUE_RELATION_PHASE_VALUE_ALIAS:
        found = loom_value_relation_next_value_alias(iterator, out_relation);
        break;
      case LOOM_VALUE_RELATION_PHASE_SELECT_PAYLOAD:
        found = loom_value_relation_next_select_payload(iterator, out_relation);
        break;
      case LOOM_VALUE_RELATION_PHASE_ELEMENTWISE:
        found = loom_value_relation_next_elementwise(iterator, out_relation);
        break;
      case LOOM_VALUE_RELATION_PHASE_CFG_ARGUMENT:
        found = loom_value_relation_next_cfg_argument(iterator, out_relation);
        break;
      case LOOM_VALUE_RELATION_PHASE_LOOP_ENTRY:
        found = loom_value_relation_next_loop_entry(iterator, out_relation);
        break;
      case LOOM_VALUE_RELATION_PHASE_LOOP_BYPASS:
        found = loom_value_relation_next_loop_bypass(iterator, out_relation);
        break;
      case LOOM_VALUE_RELATION_PHASE_LOOP_TERMINATOR:
        found =
            loom_value_relation_next_loop_terminator(iterator, out_relation);
        break;
      case LOOM_VALUE_RELATION_PHASE_REGION_TERMINATOR:
        found =
            loom_value_relation_next_region_terminator(iterator, out_relation);
        break;
      case LOOM_VALUE_RELATION_PHASE_END:
        break;
    }
    if (found) return true;
    loom_value_relation_advance_phase(iterator);
  }
  return false;
}
