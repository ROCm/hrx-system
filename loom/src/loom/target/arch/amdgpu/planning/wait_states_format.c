// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/wait_states_format.h"

#include <inttypes.h>

#include "loom/codegen/low/packet_hazard_plan_json.h"
#include "loom/util/json.h"
#include "loom/util/stream.h"

static iree_status_t loom_amdgpu_wait_state_write_states_json(
    const loom_amdgpu_wait_state_plan_t* plan, loom_output_stream_t* stream) {
  loom_json_array_writer_t states;
  IREE_RETURN_IF_ERROR(loom_json_array_begin(stream, &states));
  for (iree_host_size_t i = 0; i < plan->state_count; ++i) {
    IREE_RETURN_IF_ERROR(loom_json_array_begin_element(&states));
    const loom_amdgpu_wait_state_t* state = &plan->states[i];
    loom_json_object_writer_t state_object;
    IREE_RETURN_IF_ERROR(loom_json_object_begin(stream, &state_object));
    IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
        &state_object, IREE_SV("index"), i));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("reason"), (uint32_t)state->reason));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &state_object, IREE_SV("reason_name"),
        loom_amdgpu_wait_state_reason_name(state->reason)));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("action"), (uint32_t)state->action));
    IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
        &state_object, IREE_SV("action_name"),
        loom_amdgpu_wait_state_action_name(state->action)));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("block"), state->block_index));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("node"), state->node_index));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("scheduled_ordinal"), state->scheduled_ordinal));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("producer_node"), state->producer_node));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("consumer_node"), state->consumer_node));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("required"), state->required_cycle_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("observed"), state->observed_cycle_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("residual"), state->cycle_count));
    IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
        &state_object, IREE_SV("delay_alu_immediate"),
        state->action == LOOM_AMDGPU_WAIT_STATE_ACTION_S_DELAY_ALU
            ? state->immediate
            : 0));
    if (state->instruction_offset != 0) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &state_object, IREE_SV("instruction_offset"),
          state->instruction_offset));
    }
    if (state->action == LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAITCNT_DEPCTR ||
        state->action == LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAIT_ALU) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &state_object, IREE_SV("depctr_immediate"), state->immediate));
    }
    if (state->matrix_result_use !=
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN) {
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &state_object, IREE_SV("matrix_wait_profile"),
          (uint32_t)state->matrix_wait_profile));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &state_object, IREE_SV("matrix_wait_profile_name"),
          loom_amdgpu_matrix_wait_profile_name(state->matrix_wait_profile)));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &state_object, IREE_SV("matrix_result_use"),
          (uint32_t)state->matrix_result_use));
      IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
          &state_object, IREE_SV("matrix_result_use_name"),
          loom_amdgpu_matrix_wait_result_use_name(state->matrix_result_use)));
      IREE_RETURN_IF_ERROR(loom_json_object_write_uint32_field(
          &state_object, IREE_SV("matrix_pass_count"),
          state->matrix_pass_count));
    }
    IREE_RETURN_IF_ERROR(loom_json_object_end(&state_object));
  }
  return loom_json_array_end(&states);
}

iree_status_t loom_amdgpu_wait_state_plan_format_text(
    const loom_amdgpu_wait_state_plan_t* plan, iree_string_builder_t* builder) {
  if (plan == NULL || builder == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU wait-state plan and builder are required");
  }
  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "amdgpu.wait_state_plan states=%" PRIhsz " progress=%" PRIhsz
      " hazards=%" PRIhsz "\n",
      plan->state_count, plan->progress.record_count,
      plan->hazard_plan.record_count));
  for (iree_host_size_t i = 0; i < plan->state_count; ++i) {
    const loom_amdgpu_wait_state_t* state = &plan->states[i];
    const iree_string_view_t reason_name =
        loom_amdgpu_wait_state_reason_name(state->reason);
    const iree_string_view_t action_name =
        loom_amdgpu_wait_state_action_name(state->action);
    IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
        builder,
        "state[%" PRIhsz "] reason=%.*s action=%.*s at=b%" PRIu32 ":n%" PRIu32
        "/o%" PRIu32 " producer=n%" PRIu32 " consumer=n%" PRIu32
        " required=%" PRIu16 " observed=%" PRIu16 " residual=%" PRIu16,
        i, (int)reason_name.size, reason_name.data, (int)action_name.size,
        action_name.data, state->block_index, state->node_index,
        state->scheduled_ordinal, state->producer_node, state->consumer_node,
        state->required_cycle_count, state->observed_cycle_count,
        state->cycle_count));
    if (state->instruction_offset != 0) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " after=%" PRIu32, state->instruction_offset));
    }
    if (state->action == LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAITCNT_DEPCTR ||
        state->action == LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAIT_ALU) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " depctr=0x%04" PRIx16, state->immediate));
    }
    if (state->action == LOOM_AMDGPU_WAIT_STATE_ACTION_S_DELAY_ALU) {
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " delay_alu=0x%04" PRIx16, state->immediate));
    }
    if (state->matrix_result_use !=
        LOOM_AMDGPU_MATRIX_WAIT_RESULT_USE_UNKNOWN) {
      const iree_string_view_t profile_name =
          loom_amdgpu_matrix_wait_profile_name(state->matrix_wait_profile);
      const iree_string_view_t result_use_name =
          loom_amdgpu_matrix_wait_result_use_name(state->matrix_result_use);
      IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
          builder, " matrix=%.*s/%.*s/pass%" PRIu16, (int)profile_name.size,
          profile_name.data, (int)result_use_name.size, result_use_name.data,
          state->matrix_pass_count));
    }
    IREE_RETURN_IF_ERROR(iree_string_builder_append_cstring(builder, "\n"));
  }
  return iree_ok_status();
}

iree_status_t loom_amdgpu_wait_state_plan_format_json(
    const loom_amdgpu_wait_state_plan_t* plan, iree_string_builder_t* builder) {
  if (plan == NULL || builder == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU wait-state plan and builder are required");
  }
  loom_output_stream_t stream;
  loom_output_stream_for_builder(builder, &stream);
  loom_json_object_writer_t object;
  IREE_RETURN_IF_ERROR(loom_json_object_begin(&stream, &object));
  IREE_RETURN_IF_ERROR(loom_json_object_write_string_field(
      &object, IREE_SV("format"), IREE_SV("loom.amdgpu.wait_state_plan.v1")));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("state_count"), plan->state_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("progress_count"), plan->progress.record_count));
  IREE_RETURN_IF_ERROR(loom_json_object_write_host_size_field(
      &object, IREE_SV("hazard_count"), plan->hazard_plan.record_count));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("states")));
  IREE_RETURN_IF_ERROR(loom_amdgpu_wait_state_write_states_json(plan, &stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("progress")));
  IREE_RETURN_IF_ERROR(
      loom_low_packet_progress_write_json_array(&plan->progress, &stream));
  IREE_RETURN_IF_ERROR(
      loom_json_object_begin_field(&object, IREE_SV("hazards")));
  IREE_RETURN_IF_ERROR(loom_low_packet_hazard_plan_write_json_array(
      &plan->hazard_plan, &stream));
  return loom_json_object_end(&object);
}
