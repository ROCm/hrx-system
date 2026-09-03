// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/source_program.h"

#include <stdlib.h>
#include <string.h>

#include "loom/ops/op_defs.h"
#include "loom/util/cfg_graph.h"

typedef struct loom_source_program_build_state_t {
  // Program receiving indexed events.
  loom_source_program_t* program;
  // Arena owning the growing event array.
  iree_arena_allocator_t* arena;
} loom_source_program_build_state_t;

static int loom_source_program_compare_value_flows(const void* lhs_ptr,
                                                   const void* rhs_ptr) {
  const loom_source_program_value_flow_t* lhs =
      (const loom_source_program_value_flow_t*)lhs_ptr;
  const loom_source_program_value_flow_t* rhs =
      (const loom_source_program_value_flow_t*)rhs_ptr;
  if (lhs->source != rhs->source) return lhs->source < rhs->source ? -1 : 1;
  if (lhs->target != rhs->target) return lhs->target < rhs->target ? -1 : 1;
  return 0;
}

static void loom_source_program_canonicalize_value_flows(
    loom_source_program_t* program) {
  if (program->value_flow_count < 2) return;
  qsort(program->value_flows, program->value_flow_count,
        sizeof(*program->value_flows), loom_source_program_compare_value_flows);
  uint32_t write_index = 1;
  for (uint32_t read_index = 1; read_index < program->value_flow_count;
       ++read_index) {
    loom_source_program_value_flow_t* previous =
        &program->value_flows[write_index - 1];
    const loom_source_program_value_flow_t current =
        program->value_flows[read_index];
    if (previous->source == current.source &&
        previous->target == current.target) {
      previous->kinds |= current.kinds;
      continue;
    }
    program->value_flows[write_index++] = current;
  }
  program->value_flow_count = write_index;
}

static iree_status_t loom_source_program_append_value_flow(
    loom_source_program_build_state_t* state, loom_value_id_t source_value_id,
    loom_value_id_t target_value_id,
    loom_source_program_value_flow_kinds_t kinds) {
  if (source_value_id == target_value_id) return iree_ok_status();
  const loom_local_value_domain_t* value_domain = state->program->value_domain;
  const loom_value_ordinal_t source =
      loom_local_value_domain_try_ordinal(value_domain, source_value_id);
  const loom_value_ordinal_t target =
      loom_local_value_domain_try_ordinal(value_domain, target_value_id);
  if (source == LOOM_VALUE_ORDINAL_INVALID ||
      target == LOOM_VALUE_ORDINAL_INVALID) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source value flow escapes the indexed value domain");
  }
  loom_source_program_t* program = state->program;
  if (program->value_flow_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source program exceeds value flow range");
  }
  const iree_host_size_t minimum_capacity =
      (iree_host_size_t)program->value_flow_count + 1;
  if (minimum_capacity > program->value_flow_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, program->value_flow_count, minimum_capacity,
        sizeof(*program->value_flows), &program->value_flow_capacity,
        (void**)&program->value_flows));
  }
  program->value_flows[program->value_flow_count++] =
      (loom_source_program_value_flow_t){
          .source = source,
          .target = target,
          .kinds = kinds,
      };
  return iree_ok_status();
}

static loom_source_program_value_flags_t loom_source_program_use_role_flags(
    loom_operand_role_t role) {
  switch (role) {
    case LOOM_OPERAND_ROLE_CONTROL_CONDITION:
      return LOOM_SOURCE_PROGRAM_VALUE_HAS_CONTROL_CONDITION_USE;
    case LOOM_OPERAND_ROLE_SELECT_CONDITION:
      return LOOM_SOURCE_PROGRAM_VALUE_HAS_SELECT_CONDITION_USE;
    case LOOM_OPERAND_ROLE_SELECT_PAYLOAD:
      return LOOM_SOURCE_PROGRAM_VALUE_HAS_SELECT_PAYLOAD_USE;
    case LOOM_OPERAND_ROLE_BROADCAST_SOURCE:
      return LOOM_SOURCE_PROGRAM_VALUE_HAS_BROADCAST_SOURCE_USE;
    case LOOM_OPERAND_ROLE_COMPOSITE_ELEMENT:
      return LOOM_SOURCE_PROGRAM_VALUE_HAS_COMPOSITE_ELEMENT_USE;
    default:
      return 0;
  }
}

static iree_status_t loom_source_program_append_use(
    loom_source_program_build_state_t* state, const loom_op_t* user_op,
    loom_source_program_node_ordinal_t user_node, uint16_t operand_index) {
  const loom_value_id_t value_id =
      loom_op_const_operands(user_op)[operand_index];
  const loom_value_ordinal_t value = loom_local_value_domain_try_ordinal(
      state->program->value_domain, value_id);
  if (value == LOOM_VALUE_ORDINAL_INVALID) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source operand use escapes the indexed value domain");
  }
  loom_source_program_t* program = state->program;
  if (program->use_count == LOOM_SOURCE_PROGRAM_USE_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source program exceeds use ordinal range");
  }
  const iree_host_size_t minimum_capacity =
      (iree_host_size_t)program->use_count + 1;
  if (minimum_capacity > program->use_capacity) {
    IREE_RETURN_IF_ERROR(
        iree_arena_grow_array(state->arena, program->use_count,
                              minimum_capacity, sizeof(*program->uses),
                              &program->use_capacity, (void**)&program->uses));
  }
  const loom_operand_role_t role =
      loom_op_operand_role(state->program->module, user_op, operand_index);
  program->uses[program->use_count++] = (loom_source_program_use_t){
      .value = value,
      .user_node = user_node,
      .operand_index = operand_index,
      .operand_role = role,
  };
  program->values[value].flags |= loom_source_program_use_role_flags(role);
  return iree_ok_status();
}

static iree_status_t loom_source_program_record_definition(
    loom_source_program_build_state_t* state, loom_value_id_t value_id,
    loom_source_program_node_ordinal_t definition_node,
    loom_source_program_node_ordinal_t definition_block_node,
    loom_source_program_value_flags_t flags) {
  const loom_value_ordinal_t value = loom_local_value_domain_try_ordinal(
      state->program->value_domain, value_id);
  if (value == LOOM_VALUE_ORDINAL_INVALID) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "source value definition escapes the indexed value domain");
  }
  loom_source_program_value_t* record = &state->program->values[value];
  if (iree_any_bit_set(record->flags, LOOM_SOURCE_PROGRAM_VALUE_DEFINED)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "source value has multiple indexed definitions");
  }
  record->definition_node = definition_node;
  record->definition_block_node = definition_block_node;
  record->flags |= LOOM_SOURCE_PROGRAM_VALUE_DEFINED | flags;
  return iree_ok_status();
}

static int loom_source_program_compare_uses(const void* lhs_ptr,
                                            const void* rhs_ptr) {
  const loom_source_program_use_t* lhs =
      (const loom_source_program_use_t*)lhs_ptr;
  const loom_source_program_use_t* rhs =
      (const loom_source_program_use_t*)rhs_ptr;
  if (lhs->value != rhs->value) return lhs->value < rhs->value ? -1 : 1;
  if (lhs->user_node != rhs->user_node) {
    return lhs->user_node < rhs->user_node ? -1 : 1;
  }
  if (lhs->operand_index != rhs->operand_index) {
    return lhs->operand_index < rhs->operand_index ? -1 : 1;
  }
  return 0;
}

static void loom_source_program_finalize_uses(loom_source_program_t* program) {
  if (program->use_count != 0) {
    qsort(program->uses, program->use_count, sizeof(*program->uses),
          loom_source_program_compare_uses);
  }
  for (loom_source_program_use_ordinal_t i = 0; i < program->use_count; ++i) {
    const loom_source_program_use_t* use = &program->uses[i];
    loom_source_program_value_t* value = &program->values[use->value];
    if (value->use_count == 0) value->use_start = i;
    ++value->use_count;
    const loom_op_t* user_op =
        loom_source_program_node_operation(&program->nodes[use->user_node]);
    const loom_op_t* defining_op =
        value->definition_node != LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID &&
                !iree_any_bit_set(value->flags,
                                  LOOM_SOURCE_PROGRAM_VALUE_BLOCK_ARGUMENT)
            ? loom_source_program_node_operation(
                  &program->nodes[value->definition_node])
            : NULL;
    if (defining_op != NULL &&
        defining_op->parent_block != user_op->parent_block) {
      value->flags |= LOOM_SOURCE_PROGRAM_VALUE_HAS_CROSS_BLOCK_USE;
    }
  }
}

static iree_status_t loom_source_program_index_cfg_relations(
    loom_source_program_build_state_t* state, const loom_op_t* op) {
  loom_block_t* const* successors = loom_op_const_successors(op);
  for (uint8_t successor_index = 0; successor_index < op->successor_count;
       ++successor_index) {
    const loom_block_t* destination = successors[successor_index];
    const loom_value_id_t* payload = NULL;
    uint16_t payload_count = 0;
    if (!loom_cfg_terminator_payload_for_successor(op, destination, &payload,
                                                   &payload_count)) {
      continue;
    }
    const uint16_t relation_count =
        iree_min(payload_count, destination->arg_count);
    for (uint16_t i = 0; i < relation_count; ++i) {
      const loom_value_id_t block_arg = loom_block_arg_id(destination, i);
      IREE_RETURN_IF_ERROR(loom_source_program_append_value_flow(
          state, payload[i], block_arg,
          LOOM_SOURCE_PROGRAM_VALUE_FLOW_CFG_PAYLOAD));
      const loom_value_ordinal_t block_arg_ordinal =
          loom_local_value_domain_ordinal(state->program->value_domain,
                                          block_arg);
      state->program->values[block_arg_ordinal].flags |=
          LOOM_SOURCE_PROGRAM_VALUE_HAS_CFG_PREDECESSOR;
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_source_program_index_local_relations(
    loom_source_program_build_state_t* state, const loom_op_t* op) {
  const loom_value_id_t* operands = loom_op_const_operands(op);
  const loom_value_id_t* results = loom_op_const_results(op);
  const loom_tied_result_t* tied_results = loom_op_tied_results(op);
  for (uint16_t i = 0; i < op->tied_result_count; ++i) {
    const loom_tied_result_t tied = tied_results[i];
    if (tied.operand_index >= op->operand_count ||
        tied.result_index >= op->result_count) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "source op carries an invalid tied result");
    }
    IREE_RETURN_IF_ERROR(loom_source_program_append_value_flow(
        state, operands[tied.operand_index], results[tied.result_index],
        LOOM_SOURCE_PROGRAM_VALUE_FLOW_TIED_RESULT));
  }

  const loom_trait_flags_t traits =
      loom_op_effective_traits(state->program->module, op);
  if (loom_traits_are_fact_identity(traits)) {
    const uint16_t relation_count =
        iree_min(op->operand_count, op->result_count);
    for (uint16_t i = 0; i < relation_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_source_program_append_value_flow(
          state, operands[i], results[i],
          LOOM_SOURCE_PROGRAM_VALUE_FLOW_FACT_IDENTITY));
    }
  }
  if (loom_traits_are_value_alias(traits) && op->operand_count != 0 &&
      op->result_count != 0) {
    IREE_RETURN_IF_ERROR(loom_source_program_append_value_flow(
        state, operands[0], results[0],
        LOOM_SOURCE_PROGRAM_VALUE_FLOW_VALUE_ALIAS));
  }
  return iree_ok_status();
}

static iree_status_t loom_source_program_index_control_merge_flows(
    loom_source_program_build_state_t* state, const loom_op_t* op) {
  if (op->successor_count != 2) return iree_ok_status();
  loom_value_id_t condition = LOOM_VALUE_ID_INVALID;
  if (!loom_op_first_operand_with_role(state->program->module, op,
                                       LOOM_OPERAND_ROLE_CONTROL_CONDITION,
                                       &condition)) {
    return iree_ok_status();
  }

  loom_block_t* const* arms = loom_op_const_successors(op);
  if (arms[0] == NULL || arms[1] == NULL || arms[0]->op_count == 0 ||
      arms[1]->op_count == 0) {
    return iree_ok_status();
  }
  const loom_op_t* arm_terminators[2] = {
      loom_block_const_last_op(arms[0]),
      loom_block_const_last_op(arms[1]),
  };
  loom_block_t* const* lhs_successors =
      loom_op_const_successors(arm_terminators[0]);
  loom_block_t* const* rhs_successors =
      loom_op_const_successors(arm_terminators[1]);
  for (uint8_t lhs_index = 0; lhs_index < arm_terminators[0]->successor_count;
       ++lhs_index) {
    const loom_block_t* destination = lhs_successors[lhs_index];
    if (destination == NULL) continue;
    for (uint8_t rhs_index = 0; rhs_index < arm_terminators[1]->successor_count;
         ++rhs_index) {
      if (rhs_successors[rhs_index] != destination) continue;
      const loom_value_id_t* lhs_payload = NULL;
      const loom_value_id_t* rhs_payload = NULL;
      uint16_t lhs_payload_count = 0;
      uint16_t rhs_payload_count = 0;
      if (!loom_cfg_terminator_payload_for_successor(arm_terminators[0],
                                                     destination, &lhs_payload,
                                                     &lhs_payload_count) ||
          !loom_cfg_terminator_payload_for_successor(arm_terminators[1],
                                                     destination, &rhs_payload,
                                                     &rhs_payload_count)) {
        continue;
      }
      const uint16_t merge_count =
          iree_min(destination->arg_count,
                   iree_min(lhs_payload_count, rhs_payload_count));
      for (uint16_t i = 0; i < merge_count; ++i) {
        IREE_RETURN_IF_ERROR(loom_source_program_append_value_flow(
            state, condition, loom_block_arg_id(destination, i),
            LOOM_SOURCE_PROGRAM_VALUE_FLOW_CONTROL_MERGE));
        IREE_RETURN_IF_ERROR(loom_source_program_append_value_flow(
            state, condition, lhs_payload[i],
            LOOM_SOURCE_PROGRAM_VALUE_FLOW_CONTROL_MERGE));
        IREE_RETURN_IF_ERROR(loom_source_program_append_value_flow(
            state, condition, rhs_payload[i],
            LOOM_SOURCE_PROGRAM_VALUE_FLOW_CONTROL_MERGE));
      }
    }
  }
  return iree_ok_status();
}

static const loom_op_t* loom_source_program_region_terminator(
    const loom_region_t* region) {
  if (region == NULL || region->block_count != 1) return NULL;
  const loom_block_t* block = loom_region_const_entry_block(region);
  return block != NULL ? block->last_op : NULL;
}

static iree_status_t loom_source_program_join_value(
    loom_source_program_build_state_t* state, loom_value_id_t anchor,
    loom_value_id_t value) {
  if (anchor == LOOM_VALUE_ID_INVALID || value == LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  return loom_source_program_append_value_flow(
      state, anchor, value, LOOM_SOURCE_PROGRAM_VALUE_FLOW_LOOP_CARRY);
}

static iree_status_t loom_source_program_index_loop_relations(
    loom_source_program_build_state_t* state, const loom_op_t* op,
    loom_loop_like_t loop) {
  const loom_value_slice_t iter_args = loom_loop_like_iter_args(loop);
  const loom_value_id_t* results = loom_op_const_results(op);
  const uint16_t state_count = iree_min(iter_args.count, op->result_count);
  loom_region_t* body = loom_loop_like_body(loop);
  const loom_block_t* body_entry = body != NULL && body->block_count != 0
                                       ? loom_region_const_entry_block(body)
                                       : NULL;
  const uint16_t body_arg_offset =
      loop.vtable->iv_block_arg_index == LOOM_BLOCK_ARG_INDEX_NONE
          ? 0
          : (uint16_t)(loop.vtable->iv_block_arg_index + 1);
  const loom_op_t* body_terminator =
      loom_source_program_region_terminator(body);
  const loom_value_id_t* body_yields =
      body_terminator != NULL ? loom_op_const_operands(body_terminator) : NULL;

  loom_region_t* condition_region = loom_loop_like_condition_region(loop);
  const loom_block_t* condition_entry =
      condition_region != NULL && condition_region->block_count != 0
          ? loom_region_const_entry_block(condition_region)
          : NULL;
  const loom_op_t* condition_terminator =
      loom_source_program_region_terminator(condition_region);
  const loom_value_id_t* condition_forwarded =
      condition_terminator != NULL && condition_terminator->operand_count != 0
          ? loom_op_const_operands(condition_terminator) + 1
          : NULL;
  const uint16_t condition_forwarded_count =
      condition_terminator != NULL && condition_terminator->operand_count != 0
          ? (uint16_t)(condition_terminator->operand_count - 1)
          : 0;

  for (uint16_t i = 0; i < state_count; ++i) {
    const loom_value_id_t anchor = iter_args.values[i];
    IREE_RETURN_IF_ERROR(
        loom_source_program_join_value(state, anchor, results[i]));
    if (body_entry != NULL && body_arg_offset + i < body_entry->arg_count) {
      IREE_RETURN_IF_ERROR(loom_source_program_join_value(
          state, anchor, loom_block_arg_id(body_entry, body_arg_offset + i)));
    }
    if (body_terminator != NULL && i < body_terminator->operand_count) {
      IREE_RETURN_IF_ERROR(
          loom_source_program_join_value(state, anchor, body_yields[i]));
    }
    if (condition_entry != NULL && i < condition_entry->arg_count) {
      IREE_RETURN_IF_ERROR(loom_source_program_join_value(
          state, anchor, loom_block_arg_id(condition_entry, i)));
    }
    if (condition_forwarded != NULL && i < condition_forwarded_count) {
      IREE_RETURN_IF_ERROR(loom_source_program_join_value(
          state, anchor, condition_forwarded[i]));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_source_program_index_region_branch_relations(
    loom_source_program_build_state_t* state, const loom_op_t* op,
    loom_region_branch_t branch) {
  const loom_value_id_t* results = loom_op_const_results(op);
  for (uint8_t region_index = 0; region_index < op->region_count;
       ++region_index) {
    const loom_op_t* terminator = loom_region_branch_region_terminator(
        state->program->module, branch, region_index);
    if (terminator == NULL) continue;
    const uint16_t relation_count =
        iree_min(terminator->operand_count, op->result_count);
    const loom_value_id_t* yielded = loom_op_const_operands(terminator);
    for (uint16_t i = 0; i < relation_count; ++i) {
      IREE_RETURN_IF_ERROR(loom_source_program_append_value_flow(
          state, yielded[i], results[i],
          LOOM_SOURCE_PROGRAM_VALUE_FLOW_REGION_YIELD));
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_source_program_index_value_relations(
    loom_source_program_build_state_t* state, const loom_op_t* op) {
  IREE_RETURN_IF_ERROR(loom_source_program_index_cfg_relations(state, op));
  IREE_RETURN_IF_ERROR(loom_source_program_index_local_relations(state, op));
  IREE_RETURN_IF_ERROR(
      loom_source_program_index_control_merge_flows(state, op));
  loom_loop_like_t loop =
      loom_loop_like_cast(state->program->module, (loom_op_t*)op);
  if (loom_loop_like_isa(loop)) {
    IREE_RETURN_IF_ERROR(
        loom_source_program_index_loop_relations(state, op, loop));
  }
  loom_region_branch_t branch =
      loom_region_branch_cast(state->program->module, (loom_op_t*)op);
  if (loom_region_branch_isa(branch)) {
    IREE_RETURN_IF_ERROR(
        loom_source_program_index_region_branch_relations(state, op, branch));
  }
  return iree_ok_status();
}

static bool loom_source_program_role_is_condition(uint8_t role) {
  return role == LOOM_OPERAND_ROLE_CONTROL_CONDITION ||
         role == LOOM_OPERAND_ROLE_SELECT_CONDITION;
}

static iree_status_t loom_source_program_append_later_condition_flow(
    loom_source_program_build_state_t* state,
    loom_value_ordinal_t earlier_condition,
    loom_value_id_t later_condition_id) {
  const loom_value_ordinal_t later_condition =
      loom_local_value_domain_try_ordinal(state->program->value_domain,
                                          later_condition_id);
  if (later_condition == LOOM_VALUE_ORDINAL_INVALID) return iree_ok_status();
  const loom_source_program_value_t* earlier =
      &state->program->values[earlier_condition];
  const loom_source_program_value_t* later =
      &state->program->values[later_condition];
  if (!iree_any_bit_set(later->flags, LOOM_SOURCE_PROGRAM_VALUE_DEFINED) ||
      iree_any_bit_set(later->flags,
                       LOOM_SOURCE_PROGRAM_VALUE_BLOCK_ARGUMENT) ||
      later->definition_block_node != earlier->definition_block_node ||
      later->definition_node <= earlier->definition_node) {
    return iree_ok_status();
  }
  return loom_source_program_append_value_flow(
      state, later_condition_id,
      state->program->value_domain->value_ids[earlier_condition],
      LOOM_SOURCE_PROGRAM_VALUE_FLOW_CONDITION_ORDER);
}

static iree_status_t loom_source_program_index_condition_order_flows(
    loom_source_program_build_state_t* state) {
  loom_source_program_t* program = state->program;
  const loom_module_t* module = program->module;
  for (loom_value_ordinal_t value_ordinal = 0;
       value_ordinal < program->value_domain->value_count; ++value_ordinal) {
    const loom_source_program_value_t* value = &program->values[value_ordinal];
    const loom_source_program_value_flags_t condition_use_flags =
        LOOM_SOURCE_PROGRAM_VALUE_HAS_CONTROL_CONDITION_USE |
        LOOM_SOURCE_PROGRAM_VALUE_HAS_SELECT_CONDITION_USE;
    if (!iree_all_bits_set(value->flags, LOOM_SOURCE_PROGRAM_VALUE_DEFINED) ||
        iree_any_bit_set(value->flags,
                         LOOM_SOURCE_PROGRAM_VALUE_BLOCK_ARGUMENT) ||
        !iree_any_bit_set(value->flags, condition_use_flags)) {
      continue;
    }
    const loom_source_program_use_t* uses =
        loom_source_program_value_uses(program, value);
    for (uint32_t use_index = 0; use_index < value->use_count; ++use_index) {
      const loom_source_program_use_t* use = &uses[use_index];
      const loom_op_t* user_op =
          loom_source_program_node_operation(&program->nodes[use->user_node]);
      const loom_block_t* definition_block = loom_source_program_node_block(
          &program->nodes[value->definition_block_node]);
      if (user_op->parent_block != definition_block ||
          !loom_source_program_role_is_condition(use->operand_role)) {
        continue;
      }
      const loom_value_id_t* operands = loom_op_const_operands(user_op);
      for (uint16_t operand_index = 0; operand_index < user_op->operand_count;
           ++operand_index) {
        if (operand_index == use->operand_index) continue;
        const loom_value_id_t operand_id = operands[operand_index];
        IREE_RETURN_IF_ERROR(loom_source_program_append_later_condition_flow(
            state, value_ordinal, operand_id));

        const loom_value_ordinal_t operand_ordinal =
            loom_local_value_domain_try_ordinal(program->value_domain,
                                                operand_id);
        if (operand_ordinal == LOOM_VALUE_ORDINAL_INVALID) continue;
        const loom_source_program_value_t* operand =
            &program->values[operand_ordinal];
        if (!iree_all_bits_set(operand->flags,
                               LOOM_SOURCE_PROGRAM_VALUE_DEFINED) ||
            iree_any_bit_set(operand->flags,
                             LOOM_SOURCE_PROGRAM_VALUE_BLOCK_ARGUMENT) ||
            operand->definition_block_node != value->definition_block_node) {
          continue;
        }
        const loom_value_t* operand_value =
            loom_module_value(module, operand_id);
        const loom_op_t* operand_defining_op = loom_value_def_op(operand_value);
        const loom_value_id_t* defining_operands =
            loom_op_const_operands(operand_defining_op);
        for (uint16_t defining_operand_index = 0;
             defining_operand_index < operand_defining_op->operand_count;
             ++defining_operand_index) {
          const loom_operand_role_t role = loom_op_operand_role(
              module, operand_defining_op, defining_operand_index);
          if (!loom_source_program_role_is_condition(role)) continue;
          IREE_RETURN_IF_ERROR(loom_source_program_append_later_condition_flow(
              state, value_ordinal, defining_operands[defining_operand_index]));
        }
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_source_program_append_node(
    loom_source_program_build_state_t* state, loom_source_program_node_t node,
    loom_source_program_node_ordinal_t* out_ordinal) {
  loom_source_program_t* program = state->program;
  if (program->node_count == LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source program exceeds node ordinal range");
  }
  const iree_host_size_t minimum_capacity =
      (iree_host_size_t)program->node_count + 1;
  if (minimum_capacity > program->node_capacity) {
    IREE_RETURN_IF_ERROR(iree_arena_grow_array(
        state->arena, program->node_count, minimum_capacity,
        sizeof(*program->nodes), &program->node_capacity,
        (void**)&program->nodes));
  }
  const loom_source_program_node_ordinal_t ordinal = program->node_count++;
  program->nodes[ordinal] = node;
  *out_ordinal = ordinal;
  return iree_ok_status();
}

static iree_status_t loom_source_program_index_region(
    loom_source_program_build_state_t* state, const loom_region_t* region,
    const loom_op_t* context_op, uint16_t region_depth, bool is_root_region) {
  if (state->program->region_count == UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "source program exceeds region count range");
  }
  ++state->program->region_count;
  for (uint16_t block_index = 0; block_index < region->block_count;
       ++block_index) {
    if (state->program->block_count == UINT32_MAX) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "source program exceeds block count range");
    }
    const loom_block_t* block = loom_region_const_block(region, block_index);
    loom_source_program_node_ordinal_t block_ordinal =
        LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID;
    IREE_RETURN_IF_ERROR(loom_source_program_append_node(
        state,
        (loom_source_program_node_t){
            .object = block,
            .context_op = context_op,
            .region_depth = region_depth,
            .kind = LOOM_SOURCE_PROGRAM_NODE_BLOCK,
            .flags = is_root_region && block_index == 0
                         ? LOOM_SOURCE_PROGRAM_NODE_ROOT_ENTRY_BLOCK
                         : 0,
        },
        &block_ordinal));
    ++state->program->block_count;
    for (uint16_t arg_index = 0; arg_index < block->arg_count; ++arg_index) {
      loom_source_program_value_flags_t value_flags =
          LOOM_SOURCE_PROGRAM_VALUE_BLOCK_ARGUMENT;
      if (is_root_region && block_index == 0) {
        value_flags |= LOOM_SOURCE_PROGRAM_VALUE_ROOT_ENTRY_ARGUMENT;
      }
      IREE_RETURN_IF_ERROR(loom_source_program_record_definition(
          state, loom_block_arg_id(block, arg_index), block_ordinal,
          block_ordinal, value_flags));
    }

    const loom_op_t* op = NULL;
    loom_block_for_each_op(block, op) {
      if (state->program->operation_count == UINT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "source program exceeds operation count range");
      }
      loom_source_program_node_ordinal_t op_ordinal =
          LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID;
      IREE_RETURN_IF_ERROR(loom_source_program_append_node(
          state,
          (loom_source_program_node_t){
              .object = op,
              .region_depth = region_depth,
              .kind = LOOM_SOURCE_PROGRAM_NODE_OPERATION,
          },
          &op_ordinal));
      ++state->program->operation_count;
      const loom_value_id_t* results = loom_op_const_results(op);
      for (uint16_t result_index = 0; result_index < op->result_count;
           ++result_index) {
        IREE_RETURN_IF_ERROR(loom_source_program_record_definition(
            state, results[result_index], op_ordinal, block_ordinal,
            /*flags=*/0));
      }
      for (uint16_t operand_index = 0; operand_index < op->operand_count;
           ++operand_index) {
        IREE_RETURN_IF_ERROR(loom_source_program_append_use(
            state, op, op_ordinal, operand_index));
      }

      if (op->region_count != 0 && region_depth == UINT16_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "source program exceeds region nesting range");
      }
      loom_region_t* const* child_regions = loom_op_regions(op);
      for (uint8_t i = 0; i < op->region_count; ++i) {
        if (child_regions[i] == NULL) {
          continue;
        }
        IREE_RETURN_IF_ERROR(loom_source_program_index_region(
            state, child_regions[i], op, (uint16_t)(region_depth + 1),
            /*is_root_region=*/false));
      }
      IREE_RETURN_IF_ERROR(
          loom_source_program_index_value_relations(state, op));
      state->program->nodes[op_ordinal].subtree_limit =
          state->program->node_count;
    }
    state->program->nodes[block_ordinal].subtree_limit =
        state->program->node_count;
  }
  return iree_ok_status();
}

iree_status_t loom_source_program_build(
    const loom_module_t* module, const loom_op_t* root_context_op,
    const loom_region_t* root_region,
    const loom_local_value_domain_t* value_domain,
    iree_arena_allocator_t* arena, loom_source_program_t* out_program) {
  IREE_ASSERT(module != NULL);
  IREE_ASSERT(root_region != NULL);
  IREE_ASSERT(value_domain != NULL);
  IREE_ASSERT(loom_local_value_domain_is_acquired(value_domain));
  IREE_ASSERT(value_domain->module == module);
  IREE_ASSERT(value_domain->region == root_region);
  IREE_ASSERT(iree_any_bit_set(value_domain->flags,
                               LOOM_LOCAL_VALUE_DOMAIN_FLAG_REGION_TREE));
  IREE_ASSERT(arena != NULL);
  IREE_ASSERT(out_program != NULL);
  *out_program = (loom_source_program_t){
      .module = module,
      .root_region = root_region,
      .value_domain = value_domain,
  };
  iree_status_t status = iree_ok_status();
  if (value_domain->value_count != 0) {
    status = iree_arena_allocate_array(arena, value_domain->value_count,
                                       sizeof(*out_program->values),
                                       (void**)&out_program->values);
    if (iree_status_is_ok(status)) {
      for (loom_value_ordinal_t i = 0; i < value_domain->value_count; ++i) {
        out_program->values[i] = (loom_source_program_value_t){
            .definition_node = LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID,
            .definition_block_node = LOOM_SOURCE_PROGRAM_NODE_ORDINAL_INVALID,
            .use_start = LOOM_SOURCE_PROGRAM_USE_ORDINAL_INVALID,
        };
      }
    }
  }
  loom_source_program_build_state_t state = {
      .program = out_program,
      .arena = arena,
  };
  if (iree_status_is_ok(status)) {
    status = loom_source_program_index_region(
        &state, root_region, root_context_op, /*region_depth=*/0,
        /*is_root_region=*/true);
  }
  if (iree_status_is_ok(status)) {
    loom_source_program_finalize_uses(out_program);
    status = loom_source_program_index_condition_order_flows(&state);
  }
  if (iree_status_is_ok(status)) {
    loom_source_program_canonicalize_value_flows(out_program);
  }
  if (!iree_status_is_ok(status)) {
    *out_program = (loom_source_program_t){0};
  }
  return status;
}
