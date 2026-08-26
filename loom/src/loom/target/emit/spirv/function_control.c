// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/spirv/function_control.h"

#include "loom/ops/low/ops.h"
#include "loom/target/arch/spirv/descriptors/descriptors.h"
#include "loom/target/emit/spirv/binary_format.h"
#include "loom/target/emit/spirv/function_emitter.h"

typedef struct loom_spirv_emit_region_exit_t {
  // Region terminator op when present.
  const loom_op_t* op;
  // Label ID of the block that reaches the parent merge/continue edge.
  uint32_t predecessor_label_id;
} loom_spirv_emit_region_exit_t;

static iree_status_t loom_spirv_emit_branch_label(
    loom_spirv_emit_state_t* state, uint32_t label_id) {
  const uint32_t operands[] = {label_id};
  return loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_BRANCH, operands, IREE_ARRAYSIZE(operands));
}

static bool loom_spirv_emit_loop_counter_is_unsigned(
    loom_spirv_value_type_t value_type) {
  if (value_type.value_class == LOOM_SPIRV_VALUE_CLASS_OFFSET64) {
    return true;
  }
  if (value_type.value_class != LOOM_SPIRV_VALUE_CLASS_SCALAR) {
    IREE_CHECK_UNREACHABLE("verified SPIR-V low.scf.for counter type");
    return false;
  }
  const loom_spirv_scalar_type_descriptor_t* descriptor =
      loom_spirv_scalar_type_descriptor(value_type.scalar_type);
  if (descriptor == NULL ||
      descriptor->kind == LOOM_SPIRV_SCALAR_TYPE_KIND_FLOAT) {
    IREE_CHECK_UNREACHABLE("verified SPIR-V low.scf.for counter type");
    return false;
  }
  return descriptor->kind == LOOM_SPIRV_SCALAR_TYPE_KIND_UNSIGNED_INT;
}

static uint32_t loom_spirv_emit_for_loop_control(const loom_op_t* op) {
  if (loom_low_scf_for_unroll_factor_is_present(op) ||
      !loom_attr_is_absent(
          loom_op_attrs(op)[loom_low_scf_for_unroll_policy_ATTR_INDEX])) {
    return LOOM_SPIRV_LOOP_CONTROL_UNROLL_MASK;
  }
  return LOOM_SPIRV_LOOP_CONTROL_NONE;
}

static const loom_op_t* loom_spirv_emit_region_terminator(
    const loom_region_t* region) {
  const loom_block_t* block = loom_region_const_entry_block(region);
  return block != NULL ? block->last_op : NULL;
}

static iree_status_t loom_spirv_emit_region_ops(
    loom_spirv_emit_state_t* state, const loom_region_t* region,
    loom_spirv_emit_region_exit_t* out_exit) {
  *out_exit = (loom_spirv_emit_region_exit_t){
      .predecessor_label_id = state->current_label_id,
  };
  const loom_block_t* block = loom_region_const_entry_block(region);
  if (block == NULL) {
    return iree_ok_status();
  }
  for (const loom_op_t* op = block->first_op; op != NULL; op = op->next_op) {
    if (loom_low_scf_yield_isa(op) || loom_low_scf_condition_isa(op)) {
      *out_exit = (loom_spirv_emit_region_exit_t){
          .op = op,
          .predecessor_label_id = state->current_label_id,
      };
      return iree_ok_status();
    }
    IREE_RETURN_IF_ERROR(loom_spirv_emit_low_op(state, op));
  }
  out_exit->predecessor_label_id = state->current_label_id;
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_scf_result_phi(
    loom_spirv_emit_state_t* state, loom_value_id_t result_value_id,
    loom_value_id_t then_value_id, uint32_t then_predecessor_label_id,
    loom_value_id_t else_value_id, uint32_t else_predecessor_label_id) {
  loom_spirv_module_value_ref_t then_ref = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, then_value_id, &then_ref));
  loom_spirv_module_value_ref_t else_ref = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, else_value_id, &else_ref));
  IREE_ASSERT_EQ(then_ref.type_id, else_ref.type_id);
  IREE_ASSERT(
      loom_spirv_value_type_equal(then_ref.value_type, else_ref.value_type));

  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_reserve_value_ref(
      state, result_value_id, then_ref.type_id, then_ref.value_type,
      &result_id));
  const uint32_t operands[] = {
      then_ref.type_id,          result_id,   then_ref.id,
      then_predecessor_label_id, else_ref.id, else_predecessor_label_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_PHI, operands, IREE_ARRAYSIZE(operands)));
  const loom_spirv_module_value_ref_t result_ref = {
      .id = result_id,
      .type_id = then_ref.type_id,
      .value_type = then_ref.value_type,
  };
  return loom_spirv_emit_define_value(state, result_value_id, result_ref, true);
}

iree_status_t loom_spirv_emit_scf_if(loom_spirv_emit_state_t* state,
                                     const loom_op_t* op) {
  loom_spirv_module_value_ref_t condition = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
      state, loom_low_scf_if_condition(op), &condition));

  const loom_value_slice_t results = loom_low_scf_if_results(op);
  const loom_region_t* then_region = loom_low_scf_if_then_region(op);
  const loom_region_t* else_region = loom_low_scf_if_else_region(op);
  const uint32_t condition_label_id = state->current_label_id;
  const uint32_t then_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t merge_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t else_label_id =
      else_region != NULL ? loom_spirv_emit_allocate_id(state) : merge_label_id;

  const uint32_t selection_merge_operands[] = {
      merge_label_id,
      LOOM_SPIRV_SELECTION_CONTROL_NONE,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_SELECTION_MERGE, selection_merge_operands,
      IREE_ARRAYSIZE(selection_merge_operands)));
  const uint32_t branch_operands[] = {
      condition.id,
      then_label_id,
      else_label_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_BRANCH_CONDITIONAL, branch_operands,
      IREE_ARRAYSIZE(branch_operands)));

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, then_label_id));
  loom_spirv_emit_region_exit_t then_exit = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_region_ops(state, then_region, &then_exit));
  const uint32_t then_predecessor_label_id = state->current_label_id;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_branch_label(state, merge_label_id));

  loom_spirv_emit_region_exit_t else_exit = {
      .predecessor_label_id = condition_label_id,
  };
  if (else_region != NULL) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, else_label_id));
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_region_ops(state, else_region, &else_exit));
    else_exit.predecessor_label_id = state->current_label_id;
    IREE_RETURN_IF_ERROR(loom_spirv_emit_branch_label(state, merge_label_id));
  }

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, merge_label_id));
  if (results.count == 0) {
    return iree_ok_status();
  }
  IREE_ASSERT(then_exit.op != NULL);
  IREE_ASSERT(else_exit.op != NULL);
  const loom_value_slice_t then_values =
      loom_low_scf_yield_values(then_exit.op);
  const loom_value_slice_t else_values =
      loom_low_scf_yield_values(else_exit.op);
  IREE_ASSERT_EQ(then_values.count, results.count);
  IREE_ASSERT_EQ(else_values.count, results.count);
  for (uint16_t i = 0; i < results.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_scf_result_phi(
        state, results.values[i], then_values.values[i],
        then_predecessor_label_id, else_values.values[i],
        else_exit.predecessor_label_id));
  }
  return iree_ok_status();
}

static iree_status_t loom_spirv_emit_lookup_or_reserve_typed_value(
    loom_spirv_emit_state_t* state, loom_value_id_t value_id, uint32_t type_id,
    loom_spirv_value_type_t value_type,
    loom_spirv_module_value_ref_t* out_value_ref) {
  if (loom_spirv_emit_value_ref_exists(state, value_id)) {
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_lookup_value(state, value_id, out_value_ref));
    IREE_ASSERT_EQ(out_value_ref->type_id, type_id);
    IREE_ASSERT(
        loom_spirv_value_type_equal(out_value_ref->value_type, value_type));
    return iree_ok_status();
  }
  uint32_t result_id = 0;
  IREE_RETURN_IF_ERROR(loom_spirv_emit_reserve_value_ref(
      state, value_id, type_id, value_type, &result_id));
  *out_value_ref = (loom_spirv_module_value_ref_t){
      .id = result_id,
      .type_id = type_id,
      .value_type = value_type,
  };
  return iree_ok_status();
}

iree_status_t loom_spirv_emit_scf_for(loom_spirv_emit_state_t* state,
                                      const loom_op_t* op) {
  loom_spirv_module_value_ref_t lower_bound = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
      state, loom_low_scf_for_lower_bound(op), &lower_bound));
  loom_spirv_module_value_ref_t upper_bound = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
      state, loom_low_scf_for_upper_bound(op), &upper_bound));
  loom_spirv_module_value_ref_t step = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_lookup_value(state, loom_low_scf_for_step(op), &step));
  IREE_ASSERT_EQ(lower_bound.type_id, upper_bound.type_id);
  IREE_ASSERT_EQ(lower_bound.type_id, step.type_id);
  IREE_ASSERT(loom_spirv_value_type_equal(lower_bound.value_type,
                                          upper_bound.value_type));
  IREE_ASSERT(
      loom_spirv_value_type_equal(lower_bound.value_type, step.value_type));

  const uint32_t preheader_label_id = state->current_label_id;
  const uint32_t header_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t body_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t merge_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t continue_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t next_iv_id = loom_spirv_emit_allocate_id(state);
  IREE_RETURN_IF_ERROR(loom_spirv_emit_branch_label(state, header_label_id));

  const loom_region_t* body_region = loom_low_scf_for_body(op);
  const loom_block_t* body_block = loom_region_const_entry_block(body_region);
  const loom_value_slice_t iter_args = loom_low_scf_for_iter_args(op);
  const loom_value_slice_t results = loom_low_scf_for_results(op);
  const loom_op_t* yield_op = loom_spirv_emit_region_terminator(body_region);
  loom_value_slice_t yielded_values = {0};
  if (yield_op != NULL && loom_low_scf_yield_isa(yield_op)) {
    yielded_values = loom_low_scf_yield_values(yield_op);
  }
  IREE_ASSERT(body_block != NULL);
  IREE_ASSERT_EQ(body_block->arg_count, (uint16_t)(iter_args.count + 1));
  IREE_ASSERT_EQ(results.count, iter_args.count);
  IREE_ASSERT_EQ(yielded_values.count, iter_args.count);

  const loom_value_id_t iv_value_id = loom_block_arg_id(body_block, 0);
  const uint32_t iv_id = loom_spirv_emit_allocate_id(state);
  const loom_spirv_module_value_ref_t iv_ref = {
      .id = iv_id,
      .type_id = lower_bound.type_id,
      .value_type = lower_bound.value_type,
  };
  loom_spirv_module_value_ref_t* initial_refs = NULL;
  loom_spirv_module_value_ref_t* next_refs = NULL;
  uint32_t* carried_phi_ids = NULL;
  if (iter_args.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, iter_args.count, sizeof(*initial_refs),
        (void**)&initial_refs));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->scratch_arena, iter_args.count,
                                  sizeof(*next_refs), (void**)&next_refs));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, iter_args.count, sizeof(*carried_phi_ids),
        (void**)&carried_phi_ids));
  }
  for (uint16_t i = 0; i < iter_args.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
        state, iter_args.values[i], &initial_refs[i]));
    carried_phi_ids[i] = loom_spirv_emit_allocate_id(state);
  }
  for (uint16_t i = 0; i < iter_args.count; ++i) {
    const loom_value_id_t yielded_value_id = yielded_values.values[i];
    if (yielded_value_id == iv_value_id) {
      IREE_ASSERT_EQ(initial_refs[i].type_id, iv_ref.type_id);
      IREE_ASSERT(loom_spirv_value_type_equal(initial_refs[i].value_type,
                                              iv_ref.value_type));
      next_refs[i] = iv_ref;
      continue;
    }
    bool yielded_block_arg = false;
    for (uint16_t j = 0; j < iter_args.count; ++j) {
      if (yielded_value_id != loom_block_arg_id(body_block, j + 1)) {
        continue;
      }
      IREE_ASSERT_EQ(initial_refs[i].type_id, initial_refs[j].type_id);
      IREE_ASSERT(loom_spirv_value_type_equal(initial_refs[i].value_type,
                                              initial_refs[j].value_type));
      next_refs[i] = (loom_spirv_module_value_ref_t){
          .id = carried_phi_ids[j],
          .type_id = initial_refs[j].type_id,
          .value_type = initial_refs[j].value_type,
      };
      yielded_block_arg = true;
      break;
    }
    if (yielded_block_arg) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_or_reserve_typed_value(
        state, yielded_value_id, initial_refs[i].type_id,
        initial_refs[i].value_type, &next_refs[i]));
  }

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, header_label_id));
  const uint32_t iv_phi_operands[] = {
      lower_bound.type_id, iv_id,      lower_bound.id,
      preheader_label_id,  next_iv_id, continue_label_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_PHI, iv_phi_operands, IREE_ARRAYSIZE(iv_phi_operands)));
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_define_value(state, iv_value_id, iv_ref, true));

  for (uint16_t i = 0; i < iter_args.count; ++i) {
    const loom_value_id_t body_arg_id = loom_block_arg_id(body_block, i + 1);
    const uint32_t body_arg_id_spirv = carried_phi_ids[i];
    const uint32_t operands[] = {
        initial_refs[i].type_id, body_arg_id_spirv, initial_refs[i].id,
        preheader_label_id,      next_refs[i].id,   continue_label_id,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_PHI, operands, IREE_ARRAYSIZE(operands)));
    const loom_spirv_module_value_ref_t body_arg_ref = {
        .id = body_arg_id_spirv,
        .type_id = initial_refs[i].type_id,
        .value_type = initial_refs[i].value_type,
    };
    IREE_RETURN_IF_ERROR(
        loom_spirv_emit_define_value(state, body_arg_id, body_arg_ref, true));
  }

  uint32_t bool_type_id = 0;
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_type_bool(state->type_context, &bool_type_id));
  const uint32_t condition_id = loom_spirv_emit_allocate_id(state);
  const uint32_t compare_opcode =
      loom_spirv_emit_loop_counter_is_unsigned(lower_bound.value_type)
          ? LOOM_SPIRV_OP_U_LESS_THAN
          : LOOM_SPIRV_OP_S_LESS_THAN;
  const uint32_t compare_operands[] = {
      bool_type_id,
      condition_id,
      iv_id,
      upper_bound.id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      compare_opcode, compare_operands, IREE_ARRAYSIZE(compare_operands)));
  const uint32_t loop_merge_operands[] = {
      merge_label_id,
      continue_label_id,
      loom_spirv_emit_for_loop_control(op),
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_LOOP_MERGE, loop_merge_operands,
      IREE_ARRAYSIZE(loop_merge_operands)));
  const uint32_t branch_operands[] = {
      condition_id,
      body_label_id,
      merge_label_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_BRANCH_CONDITIONAL, branch_operands,
      IREE_ARRAYSIZE(branch_operands)));

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, body_label_id));
  loom_spirv_emit_region_exit_t body_exit = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_region_ops(state, body_region, &body_exit));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_branch_label(state, continue_label_id));

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, continue_label_id));
  const uint32_t add_operands[] = {
      lower_bound.type_id,
      next_iv_id,
      iv_id,
      step.id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_I_ADD, add_operands, IREE_ARRAYSIZE(add_operands)));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_branch_label(state, header_label_id));

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, merge_label_id));
  for (uint16_t i = 0; i < results.count; ++i) {
    loom_spirv_module_value_ref_t body_arg_ref = {0};
    IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
        state, loom_block_arg_id(body_block, i + 1), &body_arg_ref));
    uint32_t result_id = 0;
    IREE_RETURN_IF_ERROR(loom_spirv_emit_reserve_value_ref(
        state, results.values[i], body_arg_ref.type_id, body_arg_ref.value_type,
        &result_id));
    const uint32_t operands[] = {
        body_arg_ref.type_id,
        result_id,
        body_arg_ref.id,
        header_label_id,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_PHI, operands, IREE_ARRAYSIZE(operands)));
    const loom_spirv_module_value_ref_t result_ref = {
        .id = result_id,
        .type_id = body_arg_ref.type_id,
        .value_type = body_arg_ref.value_type,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_emit_define_value(state, results.values[i],
                                                      result_ref, true));
  }
  return iree_ok_status();
}

iree_status_t loom_spirv_emit_scf_while(loom_spirv_emit_state_t* state,
                                        const loom_op_t* op) {
  const loom_region_t* before_region = loom_low_scf_while_before(op);
  const loom_region_t* after_region = loom_low_scf_while_after(op);
  const loom_block_t* before_block =
      loom_region_const_entry_block(before_region);
  const loom_block_t* after_block = loom_region_const_entry_block(after_region);
  const loom_op_t* condition_op =
      loom_spirv_emit_region_terminator(before_region);
  const loom_op_t* yield_op = loom_spirv_emit_region_terminator(after_region);
  const loom_value_slice_t iter_args = loom_low_scf_while_iter_args(op);
  const loom_value_slice_t results = loom_low_scf_while_results(op);
  IREE_ASSERT(before_block != NULL);
  IREE_ASSERT(after_block != NULL);
  IREE_ASSERT(condition_op != NULL && loom_low_scf_condition_isa(condition_op));
  IREE_ASSERT(yield_op != NULL && loom_low_scf_yield_isa(yield_op));
  const loom_value_slice_t forwarded =
      loom_low_scf_condition_forwarded(condition_op);
  const loom_value_slice_t yielded = loom_low_scf_yield_values(yield_op);
  IREE_ASSERT_EQ(before_block->arg_count, iter_args.count);
  IREE_ASSERT_EQ(after_block->arg_count, iter_args.count);
  IREE_ASSERT_EQ(forwarded.count, iter_args.count);
  IREE_ASSERT_EQ(yielded.count, iter_args.count);
  IREE_ASSERT_EQ(results.count, iter_args.count);

  const uint32_t preheader_label_id = state->current_label_id;
  const uint32_t header_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t condition_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t body_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t continue_label_id = loom_spirv_emit_allocate_id(state);
  const uint32_t merge_label_id = loom_spirv_emit_allocate_id(state);

  loom_spirv_module_value_ref_t* initial_refs = NULL;
  loom_spirv_module_value_ref_t* next_refs = NULL;
  loom_spirv_module_value_ref_t* forwarded_refs = NULL;
  uint32_t* before_phi_ids = NULL;
  uint32_t* after_phi_ids = NULL;
  if (iter_args.count > 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, iter_args.count, sizeof(*initial_refs),
        (void**)&initial_refs));
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(state->scratch_arena, iter_args.count,
                                  sizeof(*next_refs), (void**)&next_refs));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, iter_args.count, sizeof(*forwarded_refs),
        (void**)&forwarded_refs));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, iter_args.count, sizeof(*before_phi_ids),
        (void**)&before_phi_ids));
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        state->scratch_arena, iter_args.count, sizeof(*after_phi_ids),
        (void**)&after_phi_ids));
  }
  for (uint16_t i = 0; i < iter_args.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
        state, iter_args.values[i], &initial_refs[i]));
    before_phi_ids[i] = loom_spirv_emit_allocate_id(state);
    after_phi_ids[i] = loom_spirv_emit_allocate_id(state);
  }
  for (uint16_t i = 0; i < yielded.count; ++i) {
    const loom_value_id_t yielded_value_id = yielded.values[i];
    bool yielded_after_arg = false;
    for (uint16_t j = 0; j < after_block->arg_count; ++j) {
      if (yielded_value_id != loom_block_arg_id(after_block, j)) {
        continue;
      }
      IREE_ASSERT_EQ(initial_refs[i].type_id, initial_refs[j].type_id);
      IREE_ASSERT(loom_spirv_value_type_equal(initial_refs[i].value_type,
                                              initial_refs[j].value_type));
      next_refs[i] = (loom_spirv_module_value_ref_t){
          .id = after_phi_ids[j],
          .type_id = initial_refs[j].type_id,
          .value_type = initial_refs[j].value_type,
      };
      yielded_after_arg = true;
      break;
    }
    if (yielded_after_arg) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_or_reserve_typed_value(
        state, yielded_value_id, initial_refs[i].type_id,
        initial_refs[i].value_type, &next_refs[i]));
  }

  IREE_RETURN_IF_ERROR(loom_spirv_emit_branch_label(state, header_label_id));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, header_label_id));
  for (uint16_t i = 0; i < iter_args.count; ++i) {
    const loom_value_id_t before_arg_id = loom_block_arg_id(before_block, i);
    const uint32_t operands[] = {
        initial_refs[i].type_id, before_phi_ids[i], initial_refs[i].id,
        preheader_label_id,      next_refs[i].id,   continue_label_id,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_PHI, operands, IREE_ARRAYSIZE(operands)));
    const loom_spirv_module_value_ref_t before_arg_ref = {
        .id = before_phi_ids[i],
        .type_id = initial_refs[i].type_id,
        .value_type = initial_refs[i].value_type,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_emit_define_value(state, before_arg_id,
                                                      before_arg_ref, true));
  }
  const uint32_t loop_merge_operands[] = {
      merge_label_id,
      continue_label_id,
      LOOM_SPIRV_LOOP_CONTROL_NONE,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_LOOP_MERGE, loop_merge_operands,
      IREE_ARRAYSIZE(loop_merge_operands)));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_branch_label(state, condition_label_id));

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, condition_label_id));
  loom_spirv_emit_region_exit_t condition_exit = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_region_ops(state, before_region, &condition_exit));
  IREE_ASSERT(condition_exit.op == condition_op);
  loom_spirv_module_value_ref_t condition_ref = {0};
  IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
      state, loom_low_scf_condition_condition(condition_op), &condition_ref));
  for (uint16_t i = 0; i < forwarded.count; ++i) {
    IREE_RETURN_IF_ERROR(loom_spirv_emit_lookup_value(
        state, forwarded.values[i], &forwarded_refs[i]));
  }
  const uint32_t branch_operands[] = {
      condition_ref.id,
      body_label_id,
      merge_label_id,
  };
  IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
      loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
      LOOM_SPIRV_OP_BRANCH_CONDITIONAL, branch_operands,
      IREE_ARRAYSIZE(branch_operands)));

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, body_label_id));
  for (uint16_t i = 0; i < forwarded.count; ++i) {
    IREE_ASSERT_EQ(initial_refs[i].type_id, forwarded_refs[i].type_id);
    IREE_ASSERT(loom_spirv_value_type_equal(initial_refs[i].value_type,
                                            forwarded_refs[i].value_type));
    const uint32_t operands[] = {
        forwarded_refs[i].type_id,
        after_phi_ids[i],
        forwarded_refs[i].id,
        condition_exit.predecessor_label_id,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_PHI, operands, IREE_ARRAYSIZE(operands)));
    const loom_spirv_module_value_ref_t after_arg_ref = {
        .id = after_phi_ids[i],
        .type_id = forwarded_refs[i].type_id,
        .value_type = forwarded_refs[i].value_type,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_emit_define_value(
        state, loom_block_arg_id(after_block, i), after_arg_ref, true));
  }
  loom_spirv_emit_region_exit_t body_exit = {0};
  IREE_RETURN_IF_ERROR(
      loom_spirv_emit_region_ops(state, after_region, &body_exit));
  IREE_ASSERT(body_exit.op == yield_op);
  IREE_RETURN_IF_ERROR(loom_spirv_emit_branch_label(state, continue_label_id));

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, continue_label_id));
  IREE_RETURN_IF_ERROR(loom_spirv_emit_branch_label(state, header_label_id));

  IREE_RETURN_IF_ERROR(loom_spirv_emit_label_id(state, merge_label_id));
  for (uint16_t i = 0; i < results.count; ++i) {
    uint32_t result_id = 0;
    IREE_RETURN_IF_ERROR(loom_spirv_emit_reserve_value_ref(
        state, results.values[i], forwarded_refs[i].type_id,
        forwarded_refs[i].value_type, &result_id));
    const uint32_t operands[] = {
        forwarded_refs[i].type_id,
        result_id,
        forwarded_refs[i].id,
        condition_exit.predecessor_label_id,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_binary_write_instruction(
        loom_spirv_emit_section(state, LOOM_SPIRV_MODULE_SECTION_FUNCTION),
        LOOM_SPIRV_OP_PHI, operands, IREE_ARRAYSIZE(operands)));
    const loom_spirv_module_value_ref_t result_ref = {
        .id = result_id,
        .type_id = forwarded_refs[i].type_id,
        .value_type = forwarded_refs[i].value_type,
    };
    IREE_RETURN_IF_ERROR(loom_spirv_emit_define_value(state, results.values[i],
                                                      result_ref, true));
  }
  return iree_ok_status();
}
