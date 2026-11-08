// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/scalar/ops.h"
#include "loom/ops/scf/ops.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/input.h"

static iree_status_t loom_aie2p_native_input_begin(
    loom_aie2p_native_context_t* context, loom_aie2p_native_input_t* input) {
  loom_aie2p_worker_builder_t* code = &context->code;
  loom_builder_t builder;
  loom_op_t* function;
  const loom_type_t arguments[] = {code->scalar_type};
  IREE_RETURN_IF_ERROR(
      loom_aie2p_worker_helper(code, arguments, input->continuation ? 1 : 0,
                               NULL, 0, &builder, &input->begin, &function));
  loom_region_t* body = loom_low_func_def_body(function);
  const loom_value_id_t* values = loom_region_entry_block(body)->arg_ids;
  loom_block_t* payload = NULL;
  if (input->continuation) {
    loom_block_t* prefix;
    IREE_RETURN_IF_ERROR(loom_region_append_block(code->module, body, &prefix));
    IREE_RETURN_IF_ERROR(
        loom_region_append_block(code->module, body, &payload));
    loom_value_id_t zero, first;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(code, &builder, 0, &zero));
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_binary(
        code, &builder, AIE2P_CORE_DESCRIPTOR_REF_CMP_EQ_I32, values[0], zero,
        &first));
    loom_op_t* op;
    IREE_RETURN_IF_ERROR(loom_low_cond_br_build(
        &builder, first, prefix, payload, LOOM_LOCATION_UNKNOWN, &op));
    loom_builder_set_block(&builder, prefix);
  }
  for (iree_host_size_t i = 0; i + 1 < input->source->phase_count; ++i) {
    loom_value_id_t task;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(
        code, &builder, input->phases[i].descriptor, &task));
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
        code, &builder, input->control_packet, input->input_queue, &task, 1));
  }
  if (payload) {
    loom_op_t* op;
    IREE_RETURN_IF_ERROR(loom_low_br_build(&builder, payload, NULL, 0,
                                           LOOM_LOCATION_UNKNOWN, &op));
    loom_builder_set_block(&builder, payload);
  }
  const loom_aie2p_native_input_phase_t* phase =
      &input->phases[input->source->phase_count - 1];
  const uint32_t word =
      phase->descriptor |
      loom_xdna_register_field_encode(
          LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_CHANNEL_MM2S_TASK_QUEUE_REPEAT_COUNT,
          phase->repetitions - 1);
  loom_value_id_t task;
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_constant(code, &builder, word, &task));
  IREE_RETURN_IF_ERROR(loom_aie2p_worker_control_write(
      code, &builder, input->control_packet, input->input_queue, &task, 1));
  return loom_aie2p_worker_return(&builder, NULL, 0);
}

iree_status_t loom_aie2p_native_prepare_input(
    loom_aie2p_native_context_t* context, loom_aie2p_native_input_t* input) {
  const loom_aie2p_native_worker_t* owner =
      &context->workers[context->execution
                            .workers[context->core_engines[input->controller]]];
  if (input->invocation->ordinal) {
    loom_builder_t builder;
    loom_op_t* function;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
        &context->code, NULL, 0, NULL, 0, &builder, &input->start, &function));
    for (iree_host_size_t i = 0; i < input->source->phase_count; ++i) {
      const loom_aie2p_native_distribution_endpoint_t* receiver =
          &input->phases[i].receiver;
      IREE_RETURN_IF_ERROR(loom_aie2p_native_endpoint_start(
          context, &builder, receiver, LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
          owner->control_packets[receiver->tile - context->tiles]));
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&builder, NULL, 0));
  }
  IREE_RETURN_IF_ERROR(loom_aie2p_native_input_begin(context, input));
  if (input->invocation->ordinal) {
    loom_builder_t builder;
    loom_op_t* function;
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_helper(
        &context->code, NULL, 0, NULL, 0, &builder, &input->retire, &function));
    const iree_host_size_t shim = input->shim - context->tiles;
    IREE_RETURN_IF_ERROR(loom_aie2p_native_queue_retire(
        context, &builder, loom_low_func_def_body(function), input->shim,
        input->input_engine, LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM,
        owner->control_packets[shim], owner->control_replies[shim]));
    const iree_host_size_t count =
        input->shared_receiver ? 1 : input->source->phase_count;
    for (iree_host_size_t i = 0; i < count; ++i) {
      const loom_aie2p_native_distribution_endpoint_t* receiver =
          &input->phases[i].receiver;
      const iree_host_size_t tile = receiver->tile - context->tiles;
      IREE_RETURN_IF_ERROR(loom_aie2p_native_queue_retire(
          context, &builder, loom_low_func_def_body(function), receiver->tile,
          receiver->engine, LOOM_XDNA_DMA_DIRECTION_STREAM_TO_MEMORY,
          owner->control_packets[tile], owner->control_replies[tile]));
    }
    IREE_RETURN_IF_ERROR(loom_aie2p_worker_return(&builder, NULL, 0));
  }
  return iree_ok_status();
}

static iree_status_t loom_aie2p_native_input_constant(loom_builder_t* builder,
                                                      uint32_t value,
                                                      loom_value_id_t* result) {
  loom_op_t* op;
  IREE_RETURN_IF_ERROR(loom_scalar_constant_build(
      builder, loom_attr_i64((int32_t)value),
      loom_type_scalar(LOOM_SCALAR_TYPE_I32), LOOM_LOCATION_UNKNOWN, &op));
  *result = loom_scalar_constant_result(op);
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_initialize_input(
    loom_builder_t* builder, const loom_aie2p_native_input_t* input,
    loom_value_id_t* state) {
  loom_op_t* op;
  if (input->invocation->ordinal) {
    IREE_RETURN_IF_ERROR(
        loom_low_invoke_build(builder, 0, 0, 0, 0, input->start, NULL, 0, NULL,
                              0, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
  }
  if (!input->continuation) {
    return loom_low_invoke_build(builder, 0, 0, 0, 0, input->begin, NULL, 0,
                                 NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &op);
  }
  return loom_aie2p_native_input_constant(builder, 0, state);
}

iree_status_t loom_aie2p_native_emit_input(
    loom_builder_t* builder, const loom_aie2p_native_input_t* input,
    iree_host_size_t action, bool after, loom_value_id_t* state) {
  if (!input->continuation) {
    return iree_ok_status();
  }
  const iree_host_size_t tail = input->source->phase_count - 1;
  const bool bulk = action == input->continuation->phase_actions[tail];
  const bool first =
      action ==
      input->continuation->phase_actions[input->continuation->first_phase];
  if ((!bulk && !first) || (after && !bulk)) {
    return iree_ok_status();
  }
  loom_value_id_t* values = state + input->state_index;
  const loom_value_id_t position = values[0];
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t zero, one;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(builder, 0, &zero));
  IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(builder, 1, &one));
  loom_op_t *op, *branch;
  if (!after) {
    loom_value_id_t predicate = LOOM_VALUE_ID_INVALID;
    if (bulk) {
      loom_value_id_t period;
      IREE_RETURN_IF_ERROR(loom_aie2p_native_input_constant(
          builder, input->source->phases[tail].period, &period));
      IREE_RETURN_IF_ERROR(loom_scalar_remui_build(
          builder, position, period, type, LOOM_LOCATION_UNKNOWN, &op));
      IREE_RETURN_IF_ERROR(loom_scalar_cmpi_build(
          builder, LOOM_SCALAR_CMPI_PREDICATE_EQ, loom_scalar_remui_result(op),
          zero, LOOM_LOCATION_UNKNOWN, &op));
      predicate = loom_scalar_cmpi_result(op);
      if (!first) {
        IREE_RETURN_IF_ERROR(
            loom_scalar_cmpi_build(builder, LOOM_SCALAR_CMPI_PREDICATE_NE,
                                   position, zero, LOOM_LOCATION_UNKNOWN, &op));
        IREE_RETURN_IF_ERROR(loom_scalar_andi_build(
            builder, predicate, loom_scalar_cmpi_result(op),
            loom_type_scalar(LOOM_SCALAR_TYPE_I1), LOOM_LOCATION_UNKNOWN, &op));
        predicate = loom_scalar_andi_result(op);
      }
    }
    loom_builder_ip_t saved = {0};
    if (bulk) {
      IREE_RETURN_IF_ERROR(loom_scf_if_build(builder, 0, predicate, NULL, 0,
                                             NULL, 0, LOOM_LOCATION_UNKNOWN,
                                             &branch));
      saved = loom_builder_enter_region(builder, branch,
                                        loom_scf_if_then_region(branch));
    }
    const loom_value_id_t argument = bulk ? position : zero;
    IREE_RETURN_IF_ERROR(
        loom_low_invoke_build(builder, 0, 0, 0, 0, input->begin, &argument, 1,
                              NULL, 0, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
    if (bulk) {
      IREE_RETURN_IF_ERROR(
          loom_scf_yield_build(builder, NULL, 0, LOOM_LOCATION_UNKNOWN, &op));
      loom_builder_restore(builder, saved);
    }
    return iree_ok_status();
  }
  IREE_RETURN_IF_ERROR(loom_scalar_addi_build(builder, 0, position, one, type,
                                              LOOM_LOCATION_UNKNOWN, &op));
  values[0] = loom_scalar_addi_result(op);
  return iree_ok_status();
}
