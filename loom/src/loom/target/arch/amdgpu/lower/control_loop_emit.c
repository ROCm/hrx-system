// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string.h>

#include "iree/base/bitmap.h"
#include "loom/codegen/low/lower/control_carriers.h"
#include "loom/ir/module.h"
#include "loom/ops/cfg/ops.h"
#include "loom/target/arch/amdgpu/lower/constants.h"
#include "loom/target/arch/amdgpu/lower/control_loop.h"
#include "loom/target/arch/amdgpu/lower/control_loop_plan.h"
#include "loom/target/arch/amdgpu/lower/control_mask.h"
#include "loom/target/arch/amdgpu/lower/control_plan.h"
#include "loom/target/arch/amdgpu/lower/emit.h"
#include "loom/target/arch/amdgpu/lower/types.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

typedef struct loom_amdgpu_zero_placeholder_t {
  loom_type_t type;
  loom_value_id_t value;
} loom_amdgpu_zero_placeholder_t;

static iree_status_t loom_amdgpu_loop_emit_exec_restore_block(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t saved_exec, loom_block_t* restore_block,
    loom_block_t* restore_dest) {
  IREE_ASSERT_EQ(restore_block->op_count, 0);
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, restore_block);
  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, restore_dest, /*args=*/NULL, 0,
                               source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_capture_branch(
    loom_low_lower_context_t* context, const loom_amdgpu_loop_exit_plan_t* exit,
    const loom_op_t* source_op, const loom_value_id_t* payload_args,
    uint16_t payload_count, loom_value_id_t payload_update_mask,
    loom_value_id_t repeat_mask, const loom_value_id_t* resume_args,
    uint16_t resume_arg_count, const loom_value_id_t* local_args,
    uint16_t local_arg_count, loom_value_id_t local_mask,
    bool identity_payload) {
  const loom_amdgpu_loop_plan_t* plan = exit->loop;
  loom_block_t* capture_block =
      loom_low_lower_control_block(context, plan->capture_block);
  IREE_ASSERT(capture_block != NULL);
  IREE_ASSERT_EQ(payload_count, plan->capture_value_count);
  loom_module_t* module = loom_low_lower_context_module(context);

  loom_value_id_t* direct_resume_args = NULL;
  if (exit->is_divergent && exit->region == NULL &&
      exit->resume_arg_count != 0 && resume_args == NULL) {
    const loom_low_lower_control_carrier_slice_t carriers =
        loom_low_lower_control_block_carriers(context, exit->repeat_dest);
    IREE_ASSERT_EQ(carriers.count, exit->resume_arg_count);
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, carriers.count, sizeof(*direct_resume_args),
        (void**)&direct_resume_args));
    for (uint16_t i = 0; i < carriers.count; ++i) {
      direct_resume_args[i] =
          loom_low_lower_lookup_value(context, carriers.values[i]);
    }
    resume_args = direct_resume_args;
    resume_arg_count = carriers.count;
  }
  IREE_ASSERT_EQ(resume_arg_count, exit->resume_arg_count);

  const uint16_t capture_arg_count = capture_block->arg_count;
  loom_value_id_t* capture_args = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, capture_arg_count, sizeof(*capture_args),
      (void**)&capture_args));
  for (uint16_t i = 0; i < payload_count; ++i) {
    const loom_value_id_t retained_value = loom_block_arg_id(capture_block, i);
    if (payload_update_mask == LOOM_VALUE_ID_INVALID) {
      capture_args[i] = retained_value;
      continue;
    }
    const loom_type_t value_type =
        loom_module_value_type(module, retained_value);
    if (payload_args[i] == retained_value ||
        !loom_amdgpu_low_type_is_native_i1_mask(context, value_type)) {
      capture_args[i] = payload_args[i];
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_masked_merge_value(
        context, source_op, plan->saved_exec,
        (loom_amdgpu_mask_merge_constants_t){0}, retained_value,
        payload_args[i], payload_update_mask, value_type, &capture_args[i]));
  }
  capture_args[plan->capture_value_count] = repeat_mask;
  uint16_t private_index = plan->capture_value_count + 1;
  const uint16_t dispatch_count =
      (uint16_t)(plan->divergent_exit_count + (plan->entry_seeded ? 1 : 0));
  if (dispatch_count > 1) {
    loom_value_id_t selector = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        exit->selector, plan->selector_type, &selector));
    capture_args[private_index++] = selector;
  }
  if (plan->entry_seeded && plan->entry_arg_count != 0) {
    memcpy(&capture_args[plan->entry_arg_offset],
           &capture_block->arg_ids[plan->entry_arg_offset],
           plan->entry_arg_count * sizeof(*capture_args));
    private_index += plan->entry_arg_count;
  }
  IREE_ASSERT_EQ(private_index, plan->resume_arg_offset);

  loom_amdgpu_zero_placeholder_t* placeholders = NULL;
  if (plan->resume_arg_count != 0) {
    IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
        context, plan->resume_arg_count, sizeof(*placeholders),
        (void**)&placeholders));
  }
  uint16_t placeholder_count = 0;
  const uint16_t resume_limit =
      plan->resume_arg_offset + plan->resume_arg_count;
  for (uint16_t i = plan->resume_arg_offset; i < resume_limit; ++i) {
    if (i >= exit->resume_arg_offset &&
        i < exit->resume_arg_offset + resume_arg_count) {
      capture_args[i] = resume_args[i - exit->resume_arg_offset];
      continue;
    }
    if (plan->entry_seeded) {
      capture_args[i] = loom_block_arg_id(capture_block, i);
      continue;
    }
    const loom_type_t value_type =
        loom_module_value_type(module, loom_block_arg_id(capture_block, i));
    capture_args[i] = LOOM_VALUE_ID_INVALID;
    for (uint16_t j = 0; j < placeholder_count; ++j) {
      if (loom_type_equal(placeholders[j].type, value_type)) {
        capture_args[i] = placeholders[j].value;
        break;
      }
    }
    if (capture_args[i] != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_zero_merge_value(
        context, source_op, value_type, &capture_args[i]));
    placeholders[placeholder_count++] = (loom_amdgpu_zero_placeholder_t){
        .type = value_type,
        .value = capture_args[i],
    };
  }

  if (plan->forwards_to_parent) {
    IREE_ASSERT(local_args == NULL || local_arg_count == plan->local_arg_count);
    IREE_ASSERT(local_args != NULL || local_arg_count == 0);
    for (uint16_t i = 0; i < plan->local_arg_count; ++i) {
      capture_args[plan->local_arg_offset + i] =
          local_args != NULL
              ? local_args[i]
              : loom_block_arg_id(capture_block, plan->local_arg_offset + i);
    }
    capture_args[plan->local_mask_offset] =
        local_mask != LOOM_VALUE_ID_INVALID
            ? local_mask
            : loom_block_arg_id(capture_block, plan->local_mask_offset);
  } else {
    IREE_ASSERT(local_args == NULL);
    IREE_ASSERT_EQ(local_arg_count, 0);
    IREE_ASSERT_EQ(local_mask, LOOM_VALUE_ID_INVALID);
  }

  loom_op_t* branch_op = NULL;
  bool can_identity_payload = identity_payload && payload_count != 0;
  for (uint16_t i = 0; i < payload_count && can_identity_payload; ++i) {
    can_identity_payload = !loom_amdgpu_low_type_is_native_i1_mask(
        context, loom_module_value_type(module, payload_args[i]));
  }
  uint16_t identity_start = 0;
  uint16_t identity_count = can_identity_payload ? payload_count : 0;
  if (!can_identity_payload) {
    identity_start = exit->resume_arg_offset;
    identity_count = exit->resume_arg_count;
  }
  if (!can_identity_payload && exit->region != NULL) {
    IREE_ASSERT_LE(exit->region->source_merge_arg_count,
                   exit->region->merge_arg_count);
    identity_start += exit->region->source_merge_arg_count;
    identity_count =
        exit->region->merge_arg_count - exit->region->source_merge_arg_count;
  }
  if (identity_count != 0) {
    return loom_low_br_identity_build(
        loom_low_lower_context_builder(context), identity_start, identity_count,
        capture_block, capture_args, capture_arg_count, source_op->location,
        &branch_op);
  }
  return loom_low_br_build(loom_low_lower_context_builder(context),
                           capture_block, capture_args, capture_arg_count,
                           source_op->location, &branch_op);
}

static iree_status_t loom_amdgpu_emit_loop_exit_region_clean_adapters(
    loom_low_lower_context_t* context,
    const loom_amdgpu_loop_exit_region_plan_t* plan,
    loom_type_t exec_mask_type) {
  loom_block_t* dispatch =
      loom_low_lower_control_block(context, plan->arm_dispatch_block);
  IREE_ASSERT(dispatch != NULL);
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0;
       i < plan->clean_adapter_count && iree_status_is_ok(status); ++i) {
    loom_block_t* adapter =
        loom_low_lower_control_block(context, plan->clean_adapters[i].block);
    IREE_ASSERT(adapter != NULL);
    IREE_ASSERT_EQ(adapter->op_count, 0);
    IREE_ASSERT_EQ(adapter->arg_count, plan->merge_arg_count);
    loom_builder_set_block(builder, adapter);

    loom_op_t* read_exec_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, plan->source_op,
        LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
        /*operands=*/NULL, 0, loom_named_attr_slice_empty(), &exec_mask_type, 1,
        &read_exec_op);
    loom_value_id_t* args = NULL;
    if (iree_status_is_ok(status)) {
      status = loom_low_lower_allocate_emission_array(
          context, dispatch->arg_count, sizeof(*args), (void**)&args);
    }
    if (iree_status_is_ok(status) && plan->merge_arg_count != 0) {
      memcpy(args, adapter->arg_ids, plan->merge_arg_count * sizeof(*args));
    }
    if (iree_status_is_ok(status)) {
      args[plan->merge_arg_count] = loom_op_const_results(read_exec_op)[0];
      loom_op_t* branch_op = NULL;
      status = loom_low_br_build(builder, dispatch, args, dispatch->arg_count,
                                 plan->source_op->location, &branch_op);
    }
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_exit_region_no_clean_block(
    loom_low_lower_context_t* context,
    const loom_amdgpu_loop_exit_region_plan_t* plan,
    loom_type_t exec_mask_type) {
  loom_block_t* block =
      loom_low_lower_control_block(context, plan->no_clean_block);
  loom_block_t* dispatch =
      loom_low_lower_control_block(context, plan->arm_dispatch_block);
  IREE_ASSERT(block != NULL);
  IREE_ASSERT(dispatch != NULL);
  IREE_ASSERT_EQ(block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, block);
  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &plan->saved_exec, 1, loom_named_attr_slice_empty(),
      /*result_types=*/NULL, 0, &restore_op);

  loom_value_id_t* args = NULL;
  if (iree_status_is_ok(status)) {
    status = loom_low_lower_allocate_emission_array(
        context, dispatch->arg_count, sizeof(*args), (void**)&args);
  }
  loom_amdgpu_zero_placeholder_t* placeholders = NULL;
  if (iree_status_is_ok(status) && plan->source_merge_arg_count != 0) {
    status = loom_low_lower_allocate_emission_array(
        context, plan->source_merge_arg_count, sizeof(*placeholders),
        (void**)&placeholders);
  }
  uint16_t placeholder_count = 0;
  loom_module_t* module = loom_low_lower_context_module(context);
  for (uint16_t i = 0;
       i < plan->source_merge_arg_count && iree_status_is_ok(status); ++i) {
    const loom_type_t type =
        loom_module_value_type(module, loom_block_arg_id(dispatch, i));
    args[i] = LOOM_VALUE_ID_INVALID;
    for (uint16_t j = 0; j < placeholder_count; ++j) {
      if (loom_type_equal(placeholders[j].type, type)) {
        args[i] = placeholders[j].value;
        break;
      }
    }
    if (args[i] == LOOM_VALUE_ID_INVALID) {
      status = loom_amdgpu_emit_zero_merge_value(context, plan->source_op, type,
                                                 &args[i]);
      if (iree_status_is_ok(status)) {
        placeholders[placeholder_count++] = (loom_amdgpu_zero_placeholder_t){
            .type = type,
            .value = args[i],
        };
      }
    }
  }
  const loom_low_lower_control_carrier_slice_t carriers =
      loom_low_lower_control_block_carriers(context, plan->continuation);
  IREE_ASSERT_EQ(
      (uint16_t)(plan->merge_arg_count - plan->source_merge_arg_count),
      carriers.count);
  for (uint16_t i = 0; i < carriers.count && iree_status_is_ok(status); ++i) {
    args[plan->source_merge_arg_count + i] =
        loom_low_lower_lookup_value(context, carriers.values[i]);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_emit_zero_native_i1_mask(
        context, plan->source_op, exec_mask_type, &args[plan->merge_arg_count]);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, dispatch, args, dispatch->arg_count,
                               plan->source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_exit_region_dispatch_blocks(
    loom_low_lower_context_t* context,
    const loom_amdgpu_loop_exit_region_plan_t* plan, loom_type_t active_type) {
  loom_block_t* dispatch =
      loom_low_lower_control_block(context, plan->arm_dispatch_block);
  loom_block_t* arm_body =
      loom_low_lower_control_block(context, plan->arm_body_block);
  loom_block_t* clean_only =
      loom_low_lower_control_block(context, plan->clean_only_block);
  loom_block_t* continuation =
      loom_low_lower_control_block(context, plan->continuation);
  IREE_ASSERT(dispatch != NULL);
  IREE_ASSERT(arm_body != NULL);
  IREE_ASSERT(clean_only != NULL);
  IREE_ASSERT(continuation != NULL);
  IREE_ASSERT_EQ(dispatch->op_count, 0);
  IREE_ASSERT_EQ(clean_only->op_count, 0);

  const loom_value_id_t clean_mask =
      loom_block_arg_id(dispatch, plan->merge_arg_count);
  const loom_value_id_t arm_mask_operands[] = {
      plan->saved_exec,
      clean_mask,
  };
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, dispatch);
  loom_op_t* arm_active_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_ANDN2_B64_EXEC,
      arm_mask_operands, IREE_ARRAYSIZE(arm_mask_operands),
      loom_named_attr_slice_empty(), &active_type, 1, &arm_active_op);
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_cond_br_build(
        builder, loom_op_const_results(arm_active_op)[0], arm_body, clean_only,
        plan->source_op->location, &branch_op);
  }

  if (iree_status_is_ok(status)) {
    loom_builder_set_block(builder, clean_only);
    loom_op_t* restore_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        &clean_mask, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
        &restore_op);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, continuation, dispatch->arg_ids,
                               plan->merge_arg_count, plan->source_op->location,
                               &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_exit_region_merge_blocks(
    loom_low_lower_context_t* context,
    const loom_amdgpu_loop_exit_region_plan_t* plan, loom_type_t exec_mask_type,
    loom_type_t active_type) {
  const loom_amdgpu_loop_exit_plan_t* exit = plan->exit;
  const loom_amdgpu_loop_plan_t* loop = exit->loop;
  loom_block_t* dispatch =
      loom_low_lower_control_block(context, plan->arm_dispatch_block);
  loom_block_t* merge =
      loom_low_lower_control_block(context, plan->merge_block);
  loom_block_t* capture_forward =
      loom_low_lower_control_block(context, plan->capture_forward_block);
  loom_block_t* direct_forward =
      loom_low_lower_control_block(context, plan->direct_forward_block);
  loom_block_t* no_repeat =
      loom_low_lower_control_block(context, plan->no_repeat_block);
  loom_block_t* local_capture =
      loom_low_lower_control_block(context, exit->local_capture_block);
  loom_block_t* continuation =
      loom_low_lower_control_block(context, exit->repeat_dest);
  IREE_ASSERT(dispatch != NULL);
  IREE_ASSERT(merge != NULL);
  IREE_ASSERT(capture_forward != NULL);
  IREE_ASSERT(direct_forward != NULL);
  IREE_ASSERT(no_repeat != NULL);
  IREE_ASSERT(local_capture != NULL);
  IREE_ASSERT(continuation != NULL);
  IREE_ASSERT_EQ(merge->op_count, 0);
  IREE_ASSERT_EQ(capture_forward->op_count, 0);
  IREE_ASSERT_EQ(direct_forward->op_count, 0);
  IREE_ASSERT_EQ(no_repeat->op_count, 0);

  const loom_value_id_t clean_mask =
      loom_block_arg_id(dispatch, plan->merge_arg_count);
  const loom_value_id_t exit_mask =
      loom_block_arg_id(local_capture, loop->capture_value_count);
  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, merge);

  loom_op_t* read_exec_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      /*operands=*/NULL, 0, loom_named_attr_slice_empty(), &exec_mask_type, 1,
      &read_exec_op);
  loom_value_id_t survivors = LOOM_VALUE_ID_INVALID;
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_emit_native_i1_mask_binary(
        context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64,
        loom_op_const_results(read_exec_op)[0], clean_mask, exec_mask_type,
        &survivors);
  }

  loom_value_id_t* merged_args = NULL;
  if (iree_status_is_ok(status) && plan->merge_arg_count != 0) {
    status = loom_low_lower_allocate_emission_array(
        context, plan->merge_arg_count, sizeof(*merged_args),
        (void**)&merged_args);
  }
  loom_module_t* module = loom_low_lower_context_module(context);
  for (uint16_t i = 0;
       i < plan->source_merge_arg_count && iree_status_is_ok(status); ++i) {
    const loom_value_id_t clean_value = loom_block_arg_id(dispatch, i);
    const loom_value_id_t exit_value = loom_block_arg_id(merge, i);
    const loom_value_id_t false_value =
        plan->clean_successor_index == 0 ? exit_value : clean_value;
    const loom_value_id_t true_value =
        plan->clean_successor_index == 0 ? clean_value : exit_value;
    status = loom_amdgpu_emit_masked_merge_value(
        context, plan->source_op, plan->saved_exec,
        plan->merge_constants ? plan->merge_constants[i]
                              : (loom_amdgpu_mask_merge_constants_t){0},
        false_value, true_value, plan->source_condition,
        loom_module_value_type(module, clean_value), &merged_args[i]);
  }
  for (uint16_t i = plan->source_merge_arg_count; i < plan->merge_arg_count;
       ++i) {
    merged_args[i] = loom_block_arg_id(merge, i);
  }

  loom_op_t* returned_op = NULL;
  if (iree_status_is_ok(status)) {
    const loom_value_id_t returned_operands[] = {exit_mask, exit_mask};
    const loom_type_t returned_types[] = {exec_mask_type, active_type};
    status = loom_amdgpu_emit_low_op(
        context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64_SCC,
        returned_operands, IREE_ARRAYSIZE(returned_operands),
        loom_named_attr_slice_empty(), returned_types,
        IREE_ARRAYSIZE(returned_types), &returned_op);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_cond_br_build(
        builder, loom_op_const_results(returned_op)[1], capture_forward,
        direct_forward, plan->source_op->location, &branch_op);
  }

  if (iree_status_is_ok(status)) {
    loom_builder_set_block(builder, direct_forward);
    loom_op_t* set_exec_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        &survivors, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
        &set_exec_op);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, continuation, merged_args,
                               plan->merge_arg_count, plan->source_op->location,
                               &branch_op);
  }

  if (iree_status_is_ok(status)) {
    loom_builder_set_block(builder, capture_forward);
    loom_op_t* set_exec_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        &exit_mask, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
        &set_exec_op);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_emit_loop_capture_branch(
        context, exit, plan->source_op, local_capture->arg_ids,
        loop->capture_value_count, exit_mask, survivors, merged_args,
        plan->merge_arg_count, /*local_args=*/NULL,
        /*local_arg_count=*/0, LOOM_VALUE_ID_INVALID,
        /*identity_payload=*/false);
  }

  if (iree_status_is_ok(status)) {
    loom_builder_set_block(builder, no_repeat);
    loom_op_t* set_exec_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, plan->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        &exit_mask, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
        &set_exec_op);
  }
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_emit_loop_capture_branch(
        context, exit, plan->source_op, local_capture->arg_ids,
        loop->capture_value_count, exit_mask, clean_mask, dispatch->arg_ids,
        plan->merge_arg_count, /*local_args=*/NULL,
        /*local_arg_count=*/0, LOOM_VALUE_ID_INVALID,
        /*identity_payload=*/false);
  }

  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_exit_region_blocks(
    loom_low_lower_context_t* context,
    loom_amdgpu_loop_exit_region_plan_t* plan, loom_value_id_t saved_exec,
    loom_value_id_t source_condition, loom_type_t exec_mask_type,
    loom_type_t active_type) {
  IREE_ASSERT(plan->exit != NULL);
  plan->saved_exec = saved_exec;
  plan->source_condition = source_condition;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_loop_exit_region_clean_adapters(
      context, plan, exec_mask_type));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_loop_exit_region_no_clean_block(
      context, plan, exec_mask_type));
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_loop_exit_region_dispatch_blocks(
      context, plan, active_type));
  return loom_amdgpu_emit_loop_exit_region_merge_blocks(
      context, plan, exec_mask_type, active_type);
}

static iree_status_t loom_amdgpu_emit_loop_exit_adapters(
    loom_low_lower_context_t* context, const loom_amdgpu_loop_exit_plan_t* exit,
    loom_value_id_t saved_active) {
  const loom_amdgpu_loop_plan_t* plan = exit->loop;
  loom_block_t* capture_block = loom_low_lower_control_block(
      context,
      exit->region != NULL ? exit->local_capture_block : plan->capture_block);
  IREE_ASSERT(capture_block != NULL);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  iree_status_t status = iree_ok_status();
  for (uint16_t i = 0; i < exit->adapter_count && iree_status_is_ok(status);
       ++i) {
    loom_block_t* adapter =
        loom_low_lower_control_block(context, exit->adapters[i].block);
    IREE_ASSERT(adapter != NULL);
    IREE_ASSERT_EQ(adapter->op_count, 0);
    const uint16_t adapter_value_count =
        exit->kind == LOOM_AMDGPU_LOOP_EXIT_LOCAL ? plan->local_arg_count
                                                  : plan->capture_value_count;
    IREE_ASSERT_EQ(adapter->arg_count, adapter_value_count);
    loom_builder_set_block(builder, adapter);

    loom_op_t* read_exec_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, exit->source_op,
        LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
        /*operands=*/NULL, 0, loom_named_attr_slice_empty(),
        &plan->exec_mask_type, 1, &read_exec_op);
    loom_value_id_t exit_mask = LOOM_VALUE_ID_INVALID;
    loom_value_id_t repeat_mask = LOOM_VALUE_ID_INVALID;
    if (iree_status_is_ok(status)) {
      exit_mask = loom_value_slice_get(loom_low_op_results(read_exec_op), 0);
    }
    if (iree_status_is_ok(status) && exit->is_divergent) {
      IREE_ASSERT_NE(saved_active, LOOM_VALUE_ID_INVALID);
      status = loom_amdgpu_emit_native_i1_mask_binary(
          context, exit->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
          saved_active, exit_mask, plan->exec_mask_type, &repeat_mask);
    } else if (iree_status_is_ok(status)) {
      status = loom_amdgpu_emit_zero_native_i1_mask(
          context, exit->source_op, plan->exec_mask_type, &repeat_mask);
    }

    if (iree_status_is_ok(status) && exit->region != NULL) {
      IREE_ASSERT_NE(exit_mask, LOOM_VALUE_ID_INVALID);
      const uint16_t local_arg_count = capture_block->arg_count;
      loom_value_id_t* local_args = NULL;
      status = loom_low_lower_allocate_emission_array(
          context, local_arg_count, sizeof(*local_args), (void**)&local_args);
      if (iree_status_is_ok(status) && plan->capture_value_count != 0) {
        memcpy(local_args, adapter->arg_ids,
               plan->capture_value_count * sizeof(*local_args));
      }
      if (iree_status_is_ok(status)) {
        local_args[plan->capture_value_count] = exit_mask;
        local_args[plan->capture_value_count + 1] = repeat_mask;
        loom_op_t* branch_op = NULL;
        status = loom_low_br_build(builder, capture_block, local_args,
                                   local_arg_count, exit->source_op->location,
                                   &branch_op);
      }
    } else if (iree_status_is_ok(status) &&
               exit->kind == LOOM_AMDGPU_LOOP_EXIT_LOCAL) {
      const loom_value_id_t local_mask =
          loom_block_arg_id(capture_block, plan->local_mask_offset);
      loom_value_id_t accumulated_local_mask = LOOM_VALUE_ID_INVALID;
      status = loom_amdgpu_emit_native_i1_mask_binary(
          context, exit->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_OR_B64,
          local_mask, exit_mask, plan->exec_mask_type, &accumulated_local_mask);
      if (iree_status_is_ok(status)) {
        status = loom_amdgpu_emit_loop_capture_branch(
            context, exit, exit->source_op, capture_block->arg_ids,
            plan->capture_value_count,
            /*payload_update_mask=*/LOOM_VALUE_ID_INVALID, repeat_mask,
            /*resume_args=*/NULL, /*resume_arg_count=*/0, adapter->arg_ids,
            adapter->arg_count, accumulated_local_mask,
            /*identity_payload=*/false);
      }
    } else if (iree_status_is_ok(status)) {
      status = loom_amdgpu_emit_loop_capture_branch(
          context, exit, exit->source_op, adapter->arg_ids,
          plan->capture_value_count, exit_mask, repeat_mask,
          /*resume_args=*/NULL, /*resume_arg_count=*/0,
          /*local_args=*/NULL, /*local_arg_count=*/0, LOOM_VALUE_ID_INVALID,
          /*identity_payload=*/false);
    }
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_no_exit_block(
    loom_low_lower_context_t* context, const loom_amdgpu_loop_exit_plan_t* exit,
    loom_value_id_t saved_active) {
  loom_block_t* no_exit_block =
      loom_low_lower_control_block(context, exit->no_exit_block);
  loom_block_t* repeat_dest = loom_low_lower_control_block(
      context,
      exit->region != NULL ? exit->local_capture_block : exit->repeat_dest);
  IREE_ASSERT(no_exit_block != NULL);
  IREE_ASSERT(repeat_dest != NULL);
  IREE_ASSERT_EQ(no_exit_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, no_exit_block);
  loom_op_t* restore_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, exit->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
      &saved_active, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
      &restore_op);
  if (iree_status_is_ok(status)) {
    if (exit->region == NULL) {
      loom_value_slice_t repeat_args = {0};
      status = loom_low_lower_control_append_carrier_arguments(
          context, exit->repeat_dest, /*prefix_values=*/NULL,
          /*prefix_count=*/0, &repeat_args);
      loom_op_t* branch_op = NULL;
      if (iree_status_is_ok(status)) {
        status = loom_low_br_build(builder, repeat_dest, repeat_args.values,
                                   repeat_args.count, exit->source_op->location,
                                   &branch_op);
      }
    } else {
      const loom_amdgpu_loop_plan_t* plan = exit->loop;
      const uint16_t local_arg_count = repeat_dest->arg_count;
      loom_value_id_t* local_args = NULL;
      status = loom_low_lower_allocate_emission_array(
          context, local_arg_count, sizeof(*local_args), (void**)&local_args);
      loom_amdgpu_zero_placeholder_t* placeholders = NULL;
      if (iree_status_is_ok(status) && plan->capture_value_count != 0) {
        status = loom_low_lower_allocate_emission_array(
            context, plan->capture_value_count, sizeof(*placeholders),
            (void**)&placeholders);
      }
      uint16_t placeholder_count = 0;
      loom_module_t* module = loom_low_lower_context_module(context);
      for (uint16_t i = 0;
           i < plan->capture_value_count && iree_status_is_ok(status); ++i) {
        const loom_type_t type =
            loom_module_value_type(module, loom_block_arg_id(repeat_dest, i));
        local_args[i] = LOOM_VALUE_ID_INVALID;
        for (uint16_t j = 0; j < placeholder_count; ++j) {
          if (loom_type_equal(placeholders[j].type, type)) {
            local_args[i] = placeholders[j].value;
            break;
          }
        }
        if (local_args[i] == LOOM_VALUE_ID_INVALID) {
          status = loom_amdgpu_emit_zero_merge_value(context, exit->source_op,
                                                     type, &local_args[i]);
          if (iree_status_is_ok(status)) {
            placeholders[placeholder_count++] =
                (loom_amdgpu_zero_placeholder_t){
                    .type = type,
                    .value = local_args[i],
                };
          }
        }
      }
      if (iree_status_is_ok(status)) {
        status = loom_amdgpu_emit_zero_native_i1_mask(
            context, exit->source_op, plan->exec_mask_type,
            &local_args[plan->capture_value_count]);
      }
      if (iree_status_is_ok(status)) {
        local_args[plan->capture_value_count + 1] = saved_active;
        loom_op_t* branch_op = NULL;
        status =
            loom_low_br_build(builder, repeat_dest, local_args, local_arg_count,
                              exit->source_op->location, &branch_op);
      }
    }
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_local_capture_block(
    loom_low_lower_context_t* context,
    const loom_amdgpu_loop_exit_plan_t* exit) {
  IREE_ASSERT(exit->region != NULL);
  const loom_amdgpu_loop_plan_t* plan = exit->loop;
  loom_block_t* local_capture =
      loom_low_lower_control_block(context, exit->local_capture_block);
  loom_block_t* repeat_dest =
      loom_low_lower_control_block(context, exit->source_repeat_dest);
  loom_block_t* no_repeat =
      loom_low_lower_control_block(context, exit->region->no_repeat_block);
  IREE_ASSERT(local_capture != NULL);
  IREE_ASSERT(repeat_dest != NULL);
  IREE_ASSERT(no_repeat != NULL);
  IREE_ASSERT_EQ(local_capture->op_count, 0);

  const loom_value_id_t repeat_mask =
      loom_block_arg_id(local_capture, plan->capture_value_count + 1);
  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  const loom_value_id_t active_operands[] = {repeat_mask, repeat_mask};
  const loom_type_t active_result_types[] = {plan->exec_mask_type, scc_type};

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, local_capture);
  loom_op_t* active_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, exit->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64_SCC,
      active_operands, IREE_ARRAYSIZE(active_operands),
      loom_named_attr_slice_empty(), active_result_types,
      IREE_ARRAYSIZE(active_result_types), &active_op);
  if (iree_status_is_ok(status)) {
    loom_op_t* set_exec_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, exit->source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        loom_op_const_results(active_op), 1, loom_named_attr_slice_empty(),
        /*result_types=*/NULL, 0, &set_exec_op);
  }
  if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_cond_br_build(
        builder, loom_op_const_results(active_op)[1], repeat_dest, no_repeat,
        exit->source_op->location, &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_capture_blocks(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_amdgpu_loop_plan_t* plan, loom_value_id_t saved_exec) {
  loom_block_t* capture_block =
      loom_low_lower_control_block(context, plan->capture_block);
  loom_block_t* restore_block =
      loom_low_lower_control_block(context, plan->restore_block);
  loom_block_t* restore_dest =
      loom_low_lower_control_block(context, plan->restore_dest);
  IREE_ASSERT(capture_block != NULL);
  IREE_ASSERT(restore_block != NULL);
  IREE_ASSERT(plan->forwards_to_parent || restore_dest != NULL);
  IREE_ASSERT_EQ(capture_block->op_count, 0);
  IREE_ASSERT_EQ(restore_block->op_count, 0);

  loom_type_t scc_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &scc_type));
  const loom_value_id_t repeat_mask =
      loom_block_arg_id(capture_block, plan->capture_value_count);
  const loom_value_id_t active_operands[] = {repeat_mask, repeat_mask};
  const loom_type_t active_result_types[] = {plan->exec_mask_type, scc_type};

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  const loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, capture_block);
  loom_op_t* active_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_B64_SCC,
      active_operands, IREE_ARRAYSIZE(active_operands),
      loom_named_attr_slice_empty(), active_result_types,
      IREE_ARRAYSIZE(active_result_types), &active_op);
  if (iree_status_is_ok(status)) {
    const loom_value_id_t active_mask = loom_op_const_results(active_op)[0];
    loom_op_t* set_exec_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        &active_mask, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL,
        0, &set_exec_op);
  }
  if (iree_status_is_ok(status)) {
    const loom_amdgpu_loop_exit_plan_t* first = NULL;
    for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
      if (plan->exit_plans[i].is_divergent) {
        first = &plan->exit_plans[i];
        break;
      }
    }
    IREE_ASSERT(first != NULL);
    loom_op_t* branch_op = NULL;
    status = loom_low_cond_br_build(
        builder, loom_op_const_results(active_op)[1],
        loom_low_lower_control_block(context, plan->entry_seeded
                                                  ? plan->entry_dispatch_block
                                                  : first->dispatch_block),
        restore_block, source_op->location, &branch_op);
  }

  const uint16_t dispatch_count =
      (uint16_t)(plan->divergent_exit_count + (plan->entry_seeded ? 1 : 0));
  if (iree_status_is_ok(status) && plan->entry_seeded) {
    loom_block_t* dispatch =
        loom_low_lower_control_block(context, plan->entry_dispatch_block);
    loom_block_t* resume_dest =
        loom_low_lower_control_block(context, plan->header_dest);
    IREE_ASSERT(dispatch != NULL);
    IREE_ASSERT(resume_dest != NULL);
    IREE_ASSERT_EQ(dispatch->op_count, 0);
    if (plan->entry_resume_block != 0) {
      loom_block_t* resume =
          loom_low_lower_control_block(context, plan->entry_resume_block);
      IREE_ASSERT(resume != NULL);
      IREE_ASSERT_EQ(resume->op_count, 0);
      loom_builder_set_block(builder, resume);
      loom_op_t* branch_op = NULL;
      status = loom_low_br_build(
          builder, resume_dest, capture_block->arg_ids + plan->entry_arg_offset,
          plan->entry_arg_count, source_op->location, &branch_op);
      resume_dest = resume;
    }
    if (iree_status_is_ok(status)) {
      loom_value_id_t entry_selector = LOOM_VALUE_ID_INVALID;
      loom_builder_set_block(builder, dispatch);
      status = loom_amdgpu_emit_const_u32(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0,
          plan->selector_type, &entry_selector);
      const loom_value_id_t selector =
          loom_block_arg_id(capture_block, plan->capture_value_count + 1);
      const loom_value_id_t compare_operands[] = {selector, entry_selector};
      loom_op_t* compare_op = NULL;
      if (iree_status_is_ok(status)) {
        status = loom_amdgpu_emit_low_op(
            context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_I32,
            compare_operands, IREE_ARRAYSIZE(compare_operands),
            loom_named_attr_slice_empty(), &scc_type, 1, &compare_op);
      }
      if (iree_status_is_ok(status)) {
        const loom_amdgpu_loop_exit_plan_t* first = NULL;
        for (uint32_t i = 0; i < plan->exit_plan_count; ++i) {
          if (plan->exit_plans[i].is_divergent) {
            first = &plan->exit_plans[i];
            break;
          }
        }
        IREE_ASSERT(first != NULL);
        loom_op_t* branch_op = NULL;
        status = loom_low_cond_br_build(
            builder, loom_op_const_results(compare_op)[0], resume_dest,
            loom_low_lower_control_block(context, first->dispatch_block),
            source_op->location, &branch_op);
      }
    }
  }

  for (uint32_t i = 0; i < plan->exit_plan_count && iree_status_is_ok(status);
       ++i) {
    const loom_amdgpu_loop_exit_plan_t* exit = &plan->exit_plans[i];
    if (!exit->is_divergent) {
      continue;
    }
    loom_block_t* dispatch =
        loom_low_lower_control_block(context, exit->dispatch_block);
    IREE_ASSERT(dispatch != NULL);
    IREE_ASSERT_EQ(dispatch->op_count, 0);
    loom_builder_set_block(builder, dispatch);

    loom_block_t* resume_dest =
        loom_low_lower_control_block(context, exit->repeat_dest);
    if (exit->resume_block != 0) {
      loom_block_t* resume =
          loom_low_lower_control_block(context, exit->resume_block);
      IREE_ASSERT(resume != NULL);
      IREE_ASSERT_EQ(resume->op_count, 0);
      loom_builder_set_block(builder, resume);
      loom_op_t* branch_op = NULL;
      status = loom_low_br_build(
          builder, resume_dest,
          capture_block->arg_ids + exit->resume_arg_offset,
          exit->resume_arg_count, source_op->location, &branch_op);
      if (!iree_status_is_ok(status)) {
        break;
      }
      resume_dest = resume;
      loom_builder_set_block(builder, dispatch);
    }

    const bool is_last = exit->selector + 1 == dispatch_count;
    if (is_last) {
      loom_op_t* branch_op = NULL;
      status = loom_low_br_build(builder, resume_dest, /*args=*/NULL, 0,
                                 source_op->location, &branch_op);
      continue;
    }

    loom_value_id_t selector_constant = LOOM_VALUE_ID_INVALID;
    status = loom_amdgpu_emit_const_u32(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32,
        exit->selector, plan->selector_type, &selector_constant);
    if (!iree_status_is_ok(status)) {
      break;
    }
    const loom_value_id_t selector =
        loom_block_arg_id(capture_block, plan->capture_value_count + 1);
    const loom_value_id_t compare_operands[] = {selector, selector_constant};
    loom_op_t* compare_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_CMP_EQ_I32,
        compare_operands, IREE_ARRAYSIZE(compare_operands),
        loom_named_attr_slice_empty(), &scc_type, 1, &compare_op);
    if (iree_status_is_ok(status)) {
      const loom_amdgpu_loop_exit_plan_t* next = NULL;
      for (uint32_t j = i + 1; j < plan->exit_plan_count; ++j) {
        if (plan->exit_plans[j].is_divergent) {
          next = &plan->exit_plans[j];
          break;
        }
      }
      IREE_ASSERT(next != NULL);
      loom_op_t* branch_op = NULL;
      status = loom_low_cond_br_build(
          builder, loom_op_const_results(compare_op)[0], resume_dest,
          loom_low_lower_control_block(context, next->dispatch_block),
          source_op->location, &branch_op);
    }
  }

  if (iree_status_is_ok(status)) {
    loom_builder_set_block(builder, restore_block);
    loom_op_t* restore_op = NULL;
    status = loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC,
        &saved_exec, 1, loom_named_attr_slice_empty(), /*result_types=*/NULL, 0,
        &restore_op);
  }
  if (iree_status_is_ok(status) && plan->forwards_to_parent) {
    IREE_ASSERT(plan->parent_exit != NULL);
    const loom_value_id_t local_mask =
        loom_block_arg_id(capture_block, plan->local_mask_offset);
    loom_value_id_t terminal_mask = LOOM_VALUE_ID_INVALID;
    status = loom_amdgpu_emit_native_i1_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64, saved_exec,
        local_mask, plan->exec_mask_type, &terminal_mask);
    if (iree_status_is_ok(status)) {
      status = loom_amdgpu_emit_loop_capture_branch(
          context, plan->parent_exit, source_op, capture_block->arg_ids,
          plan->capture_value_count, terminal_mask, local_mask,
          capture_block->arg_ids + plan->local_arg_offset,
          plan->local_arg_count, /*local_args=*/NULL,
          /*local_arg_count=*/0, LOOM_VALUE_ID_INVALID,
          /*identity_payload=*/true);
    }
  } else if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, restore_dest, capture_block->arg_ids,
                               plan->capture_value_count, source_op->location,
                               &branch_op);
  }
  loom_builder_restore(builder, saved_ip);
  return status;
}

static iree_status_t loom_amdgpu_emit_loop_entry_capture_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_loop_plan_t* plan, const loom_block_t* entry_block,
    loom_value_id_t saved_exec) {
  IREE_ASSERT(plan->entry_seeded);
  loom_block_t* capture_block =
      loom_low_lower_control_block(context, plan->capture_block);
  IREE_ASSERT(capture_block != NULL);
  loom_value_id_t* args = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, capture_block->arg_count, sizeof(*args), (void**)&args));
  loom_value_id_t selector = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_const_u32(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B32, 0,
      plan->selector_type, &selector));
  loom_value_id_t local_mask = LOOM_VALUE_ID_INVALID;
  if (plan->forwards_to_parent) {
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_zero_native_i1_mask(
        context, source_op, plan->exec_mask_type, &local_mask));
  }
  loom_amdgpu_zero_placeholder_t* placeholders = NULL;
  IREE_RETURN_IF_ERROR(loom_low_lower_allocate_emission_array(
      context, capture_block->arg_count, sizeof(*placeholders),
      (void**)&placeholders));
  uint16_t placeholder_count = 0;
  loom_module_t* module = loom_low_lower_context_module(context);
  loom_block_t* parent_capture_block = NULL;
  if (plan->forwards_to_parent) {
    IREE_ASSERT(plan->parent_exit != NULL);
    parent_capture_block = loom_low_lower_control_block(
        context, plan->parent_exit->loop->capture_block);
    IREE_ASSERT(parent_capture_block != NULL);
    IREE_ASSERT_EQ(plan->capture_value_count,
                   plan->parent_exit->loop->capture_value_count);
  }
  for (uint16_t i = 0; i < capture_block->arg_count; ++i) {
    const loom_type_t type =
        loom_module_value_type(module, loom_block_arg_id(capture_block, i));
    args[i] = LOOM_VALUE_ID_INVALID;
    if (parent_capture_block != NULL && i < plan->capture_value_count) {
      args[i] = loom_block_arg_id(parent_capture_block, i);
    } else if (i == plan->capture_value_count) {
      args[i] = saved_exec;
    } else if (i == plan->capture_value_count + 1) {
      args[i] = selector;
    } else if (i >= plan->entry_arg_offset &&
               i < plan->entry_arg_offset + plan->entry_arg_count) {
      args[i] = entry_block->arg_ids[i - plan->entry_arg_offset];
    } else if (plan->forwards_to_parent && i == plan->local_mask_offset) {
      args[i] = local_mask;
    }
    if (args[i] == LOOM_VALUE_ID_INVALID && i < plan->capture_value_count &&
        loom_amdgpu_low_type_is_native_i1_mask(context, type)) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_zero_native_i1_mask(
          context, source_op, type, &args[i]));
    }
    if (args[i] != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    const loom_value_id_t seed_values[] = {saved_exec, selector};
    for (iree_host_size_t j = 0; j < IREE_ARRAYSIZE(seed_values); ++j) {
      if (loom_type_equal(type,
                          loom_module_value_type(module, seed_values[j]))) {
        args[i] = seed_values[j];
        break;
      }
    }
    for (uint16_t j = 0;
         j < entry_block->arg_count && args[i] == LOOM_VALUE_ID_INVALID; ++j) {
      if (loom_type_equal(
              type, loom_module_value_type(module, entry_block->arg_ids[j]))) {
        args[i] = entry_block->arg_ids[j];
      }
    }
    if (args[i] != LOOM_VALUE_ID_INVALID) {
      continue;
    }
    for (uint16_t j = 0; j < placeholder_count; ++j) {
      if (loom_type_equal(placeholders[j].type, type)) {
        args[i] = placeholders[j].value;
        break;
      }
    }
    if (args[i] == LOOM_VALUE_ID_INVALID) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_zero_merge_value(context, source_op,
                                                             type, &args[i]));
      placeholders[placeholder_count++] = (loom_amdgpu_zero_placeholder_t){
          .type = type,
          .value = args[i],
      };
    }
  }
  loom_op_t* branch_op = NULL;
  return loom_low_br_build(loom_low_lower_context_builder(context),
                           capture_block, args, capture_block->arg_count,
                           source_op->location, &branch_op);
}

static iree_status_t loom_amdgpu_emit_divergent_loop_blocks(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_type_t exec_mask_type, loom_amdgpu_loop_plan_t* plan,
    loom_value_id_t* out_saved_exec) {
  loom_block_t* entry_block =
      loom_low_lower_control_block(context, plan->entry_block);
  loom_block_t* header_dest =
      loom_low_lower_control_block(context, plan->header_dest);
  *out_saved_exec = plan->saved_exec;
  if (plan->saved_exec != LOOM_VALUE_ID_INVALID) {
    return iree_ok_status();
  }
  if (plan->parent_exit != NULL) {
    loom_value_id_t parent_saved_exec = LOOM_VALUE_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_divergent_loop_blocks(
        context, source_op, exec_mask_type, plan->parent_exit->loop,
        &parent_saved_exec));
  }
  IREE_ASSERT(entry_block != NULL);
  IREE_ASSERT(header_dest != NULL);
  IREE_ASSERT_EQ(entry_block->op_count, 0);

  loom_builder_t* builder = loom_low_lower_context_builder(context);
  loom_builder_ip_t saved_ip = loom_builder_save(builder);
  loom_builder_set_block(builder, entry_block);

  loom_op_t* read_exec_op = NULL;
  iree_status_t status = loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
      /*operands=*/NULL, 0, loom_named_attr_slice_empty(), &exec_mask_type, 1,
      &read_exec_op);
  if (iree_status_is_ok(status)) {
    *out_saved_exec =
        loom_value_slice_get(loom_low_op_results(read_exec_op), 0);
    plan->saved_exec = *out_saved_exec;
  }
  if (iree_status_is_ok(status) && plan->entry_seeded) {
    status = loom_amdgpu_emit_loop_entry_capture_branch(
        context, source_op, plan, entry_block, *out_saved_exec);
  } else if (iree_status_is_ok(status)) {
    loom_op_t* branch_op = NULL;
    status = loom_low_br_build(builder, header_dest, entry_block->arg_ids,
                               entry_block->arg_count, source_op->location,
                               &branch_op);
  }

  loom_builder_restore(builder, saved_ip);
  if (iree_status_is_ok(status) && plan->capture_exits) {
    status = loom_amdgpu_emit_loop_capture_blocks(context, source_op, plan,
                                                  *out_saved_exec);
  }
  if (iree_status_is_ok(status) && plan->capture_exits) {
    for (uint32_t i = 0; i < plan->exit_plan_count && iree_status_is_ok(status);
         ++i) {
      if (!plan->exit_plans[i].is_divergent) {
        status = loom_amdgpu_emit_loop_exit_adapters(
            context, &plan->exit_plans[i], LOOM_VALUE_ID_INVALID);
      }
    }
  }
  if (iree_status_is_ok(status) && !plan->capture_exits) {
    status = loom_amdgpu_loop_emit_exec_restore_block(
        context, source_op, *out_saved_exec,
        loom_low_lower_control_block(context, plan->restore_block),
        loom_low_lower_control_block(context, plan->restore_dest));
  }
  return status;
}

bool loom_amdgpu_loop_plan_isa(loom_low_lower_plan_t plan) {
  return plan.id == LOOM_AMDGPU_BRANCH_PLAN_DIVERGENT_LOOP ||
         plan.id == LOOM_AMDGPU_BRANCH_PLAN_LOOP_EXIT_REGION;
}

iree_status_t loom_amdgpu_loop_emit_cond_branch(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_low_lower_plan_t plan, loom_value_id_t low_condition,
    loom_block_t* low_true_dest, loom_block_t* low_false_dest,
    loom_type_t condition_type) {
  IREE_ASSERT(loom_amdgpu_loop_plan_isa(plan));
  const loom_amdgpu_loop_exit_plan_t* loop_exit = NULL;
  loom_amdgpu_loop_exit_region_plan_t* exit_region = NULL;
  if (plan.id == LOOM_AMDGPU_BRANCH_PLAN_DIVERGENT_LOOP) {
    loop_exit = (const loom_amdgpu_loop_exit_plan_t*)plan.target_data;
    IREE_ASSERT(loop_exit != NULL);
  } else {
    exit_region = (loom_amdgpu_loop_exit_region_plan_t*)plan.target_data;
    IREE_ASSERT(exit_region != NULL);
  }

  loom_type_t active_type = loom_type_none();
  IREE_RETURN_IF_ERROR(loom_amdgpu_make_scc_type(context, &active_type));
  const loom_type_t result_types[] = {condition_type, active_type};

  if (exit_region != NULL) {
    const loom_value_id_t source_condition = low_condition;
    if (exit_region->clean_successor_index != 0) {
      loom_op_t* read_exec_op = NULL;
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
          /*operands=*/NULL, 0, loom_named_attr_slice_empty(), &condition_type,
          1, &read_exec_op));
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
          loom_op_const_results(read_exec_op)[0], low_condition, condition_type,
          &low_condition));
    }

    loom_op_t* saveexec_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
        &low_condition, 1, loom_named_attr_slice_empty(), result_types,
        IREE_ARRAYSIZE(result_types), &saveexec_op));
    const loom_value_id_t saved_exec = loom_op_const_results(saveexec_op)[0];
    const loom_value_id_t active = loom_op_const_results(saveexec_op)[1];
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_loop_exit_region_blocks(
        context, exit_region, saved_exec, source_condition, condition_type,
        active_type));

    loom_block_t* clean_dest = exit_region->clean_successor_index == 0
                                   ? low_true_dest
                                   : low_false_dest;
    loom_block_t* no_clean_dest = exit_region->clean_successor_index == 0
                                      ? low_false_dest
                                      : low_true_dest;
    return loom_low_lower_control_emit_cond_branch(context, source_op, active,
                                                   clean_dest, no_clean_dest);
  }

  loom_value_id_t loop_saved_exec = LOOM_VALUE_ID_INVALID;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_divergent_loop_blocks(
      context, source_op, condition_type, loop_exit->loop, &loop_saved_exec));
  IREE_ASSERT_NE(loop_saved_exec, LOOM_VALUE_ID_INVALID);

  if (!loop_exit->loop->capture_exits) {
    if (loop_exit->exit_successor_mask == 3) {
      IREE_ASSERT_EQ(low_true_dest, low_false_dest);
      loom_op_t* branch_op = NULL;
      return loom_low_br_build(loom_low_lower_context_builder(context),
                               low_true_dest, /*args=*/NULL, 0,
                               source_op->location, &branch_op);
    }
    if (loop_exit->exit_successor_mask == 1) {
      IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
          context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
          loop_saved_exec, low_condition, condition_type, &low_condition));
      loom_block_t* repeat_dest = low_false_dest;
      low_false_dest = low_true_dest;
      low_true_dest = repeat_dest;
    }
  } else if (loop_exit->exit_successor_mask == 2) {
    // Select the false lanes from the currently active loop lanes. Lanes
    // retired on earlier iterations remain absent from both masks.
    loom_op_t* read_exec_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_MOV_B64_EXEC_READ,
        /*operands=*/NULL, 0, loom_named_attr_slice_empty(), &condition_type, 1,
        &read_exec_op));
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_native_i1_mask_binary(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_XOR_B64,
        loom_op_const_results(read_exec_op)[0], low_condition, condition_type,
        &low_condition));
  } else {
    IREE_ASSERT_NE(loop_exit->exit_successor_mask, 3);
  }

  if (loop_exit->loop->capture_exits) {
    loom_op_t* saveexec_op = NULL;
    IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
        context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
        &low_condition, 1, loom_named_attr_slice_empty(), result_types,
        IREE_ARRAYSIZE(result_types), &saveexec_op));
    const loom_value_id_t saved_active = loom_op_const_results(saveexec_op)[0];
    const loom_value_id_t exit_active = loom_op_const_results(saveexec_op)[1];
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_loop_exit_adapters(context, loop_exit, saved_active));
    IREE_RETURN_IF_ERROR(
        loom_amdgpu_emit_loop_no_exit_block(context, loop_exit, saved_active));
    if (loop_exit->region != NULL) {
      IREE_RETURN_IF_ERROR(
          loom_amdgpu_emit_loop_local_capture_block(context, loop_exit));
    }

    loom_block_t* exit_dest =
        loop_exit->exit_successor_mask == 1 ? low_true_dest : low_false_dest;
    loom_block_t* no_exit_dest =
        loop_exit->exit_successor_mask == 1 ? low_false_dest : low_true_dest;
    return loom_low_lower_control_emit_cond_branch(
        context, source_op, exit_active, exit_dest, no_exit_dest);
  }

  loom_op_t* saveexec_op = NULL;
  IREE_RETURN_IF_ERROR(loom_amdgpu_emit_low_op(
      context, source_op, LOOM_AMDGPU_DESCRIPTOR_REF_S_AND_SAVEEXEC_B64,
      &low_condition, 1, loom_named_attr_slice_empty(), result_types,
      IREE_ARRAYSIZE(result_types), &saveexec_op));
  return loom_low_lower_control_emit_cond_branch(
      context, source_op, loom_op_const_results(saveexec_op)[1], low_true_dest,
      low_false_dest);
}
