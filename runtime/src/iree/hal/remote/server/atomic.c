// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/atomic.h"

#include "iree/hal/remote/protocol/queue.h"
#include "iree/hal/remote/server/session.h"

static_assert(IREE_HAL_REMOTE_ATOMIC_WIDTH_32 == IREE_HAL_ATOMIC_WIDTH_32,
              "remote 32-bit atomic width must match the HAL");
static_assert(IREE_HAL_REMOTE_ATOMIC_WIDTH_64 == IREE_HAL_ATOMIC_WIDTH_64,
              "remote 64-bit atomic width must match the HAL");
static_assert(IREE_HAL_REMOTE_ATOMIC_WAIT_CONDITION_EQUAL ==
                  IREE_HAL_ATOMIC_WAIT_CONDITION_EQUAL,
              "remote atomic wait conditions must match the HAL");
static_assert(IREE_HAL_REMOTE_ATOMIC_WAIT_CONDITION_NOT_EQUAL ==
                  IREE_HAL_ATOMIC_WAIT_CONDITION_NOT_EQUAL,
              "remote atomic wait conditions must match the HAL");
static_assert(IREE_HAL_REMOTE_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL ==
                  IREE_HAL_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL,
              "remote atomic wait conditions must match the HAL");
static_assert(IREE_HAL_REMOTE_ATOMIC_RMW_OPERATION_ADD ==
                  IREE_HAL_ATOMIC_RMW_OPERATION_ADD,
              "remote atomic RMW operations must match the HAL");
static_assert(IREE_HAL_REMOTE_ATOMIC_RMW_OPERATION_SUBTRACT ==
                  IREE_HAL_ATOMIC_RMW_OPERATION_SUBTRACT,
              "remote atomic RMW operations must match the HAL");
static_assert(IREE_HAL_REMOTE_ATOMIC_RMW_OPERATION_AND ==
                  IREE_HAL_ATOMIC_RMW_OPERATION_AND,
              "remote atomic RMW operations must match the HAL");
static_assert(IREE_HAL_REMOTE_ATOMIC_RMW_OPERATION_OR ==
                  IREE_HAL_ATOMIC_RMW_OPERATION_OR,
              "remote atomic RMW operations must match the HAL");
static_assert(IREE_HAL_REMOTE_ATOMIC_RMW_OPERATION_XOR ==
                  IREE_HAL_ATOMIC_RMW_OPERATION_XOR,
              "remote atomic RMW operations must match the HAL");
static_assert(IREE_HAL_REMOTE_ATOMIC_FLAGS_KNOWN == IREE_HAL_ATOMIC_FLAGS_KNOWN,
              "remote atomic flags must match the HAL");

static iree_hal_atomic_wait_params_t
iree_hal_remote_server_atomic_wait_params_from_wire(
    const iree_hal_remote_atomic_wait_params_t* wire_params) {
  return (iree_hal_atomic_wait_params_t){
      .value = wire_params->value,
      .mask = wire_params->mask,
      .flags = (iree_hal_atomic_flags_t)wire_params->flags,
      .width = (iree_hal_atomic_width_t)wire_params->width,
      .condition = (iree_hal_atomic_wait_condition_t)wire_params->condition,
      .reserved = wire_params->reserved,
  };
}

static iree_hal_atomic_store_params_t
iree_hal_remote_server_atomic_store_params_from_wire(
    const iree_hal_remote_atomic_store_params_t* wire_params) {
  iree_hal_atomic_store_params_t params = {
      .value = wire_params->value,
      .flags = (iree_hal_atomic_flags_t)wire_params->flags,
      .width = (iree_hal_atomic_width_t)wire_params->width,
  };
  memcpy(params.reserved, wire_params->reserved, sizeof(params.reserved));
  return params;
}

static iree_hal_atomic_rmw_params_t
iree_hal_remote_server_atomic_rmw_params_from_wire(
    const iree_hal_remote_atomic_rmw_params_t* wire_params) {
  return (iree_hal_atomic_rmw_params_t){
      .operand = wire_params->operand,
      .flags = (iree_hal_atomic_flags_t)wire_params->flags,
      .width = (iree_hal_atomic_width_t)wire_params->width,
      .operation = (iree_hal_atomic_rmw_operation_t)wire_params->operation,
      .reserved = wire_params->reserved,
  };
}

static iree_status_t iree_hal_remote_server_validate_fixed_queue_op(
    iree_const_byte_span_t command_data, const char* op_name,
    iree_hal_remote_queue_op_type_t expected_type,
    iree_host_size_t expected_length) {
  if (command_data.data_length != expected_length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s command length %" PRIhsz
                            " does not match canonical length %" PRIhsz,
                            op_name, command_data.data_length, expected_length);
  }
  iree_hal_remote_queue_op_header_t header;
  memcpy(&header, command_data.data, sizeof(header));
  if (header.type != expected_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s command has unexpected type 0x%04x", op_name,
                            header.type);
  }
  if (header.flags != 0 || header.reserved != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s command header reserved fields are nonzero",
                            op_name);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_server_resolve_queue_atomic_target(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_queue_atomic_target_t* target,
    iree_hal_atomic_width_t width, const char* op_name,
    iree_hal_buffer_ref_t* out_target_ref) {
  if (target->buffer_id == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s requires a direct target buffer", op_name);
  }
  return iree_hal_remote_server_resolve_command_buffer_ref(
      session_slot, target->buffer_id, /*buffer_slot=*/0, target->offset,
      iree_hal_atomic_width_byte_count(width), op_name, out_target_ref);
}

iree_status_t iree_hal_remote_server_queue_atomic_wait(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list,
    iree_const_byte_span_t command_data) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_server_validate_fixed_queue_op(
      command_data, "ATOMIC_WAIT", IREE_HAL_REMOTE_QUEUE_OP_ATOMIC_WAIT,
      sizeof(iree_hal_remote_queue_atomic_wait_op_t)));
  const iree_hal_remote_queue_atomic_wait_op_t* op =
      (const iree_hal_remote_queue_atomic_wait_op_t*)command_data.data;
  const iree_hal_atomic_wait_params_t params =
      iree_hal_remote_server_atomic_wait_params_from_wire(&op->params);
  IREE_RETURN_IF_ERROR(iree_hal_atomic_wait_params_validate(params));
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_remote_server_resolve_queue_atomic_target(
      session_slot, &op->target, params.width, "ATOMIC_WAIT", &target_ref));
  return iree_hal_device_queue_atomic_wait(
      local_device, (iree_hal_queue_affinity_t)op->target.queue_affinity,
      wait_list, signal_list, target_ref.buffer, target_ref.offset, params);
}

iree_status_t iree_hal_remote_server_queue_atomic_store(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list,
    iree_const_byte_span_t command_data) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_server_validate_fixed_queue_op(
      command_data, "ATOMIC_STORE", IREE_HAL_REMOTE_QUEUE_OP_ATOMIC_STORE,
      sizeof(iree_hal_remote_queue_atomic_store_op_t)));
  const iree_hal_remote_queue_atomic_store_op_t* op =
      (const iree_hal_remote_queue_atomic_store_op_t*)command_data.data;
  const iree_hal_atomic_store_params_t params =
      iree_hal_remote_server_atomic_store_params_from_wire(&op->params);
  IREE_RETURN_IF_ERROR(iree_hal_atomic_store_params_validate(params));
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_remote_server_resolve_queue_atomic_target(
      session_slot, &op->target, params.width, "ATOMIC_STORE", &target_ref));
  return iree_hal_device_queue_atomic_store(
      local_device, (iree_hal_queue_affinity_t)op->target.queue_affinity,
      wait_list, signal_list, target_ref.buffer, target_ref.offset, params);
}

iree_status_t iree_hal_remote_server_queue_atomic_rmw(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_device_t* local_device, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list,
    iree_const_byte_span_t command_data) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_server_validate_fixed_queue_op(
      command_data, "ATOMIC_RMW", IREE_HAL_REMOTE_QUEUE_OP_ATOMIC_RMW,
      sizeof(iree_hal_remote_queue_atomic_rmw_op_t)));
  const iree_hal_remote_queue_atomic_rmw_op_t* op =
      (const iree_hal_remote_queue_atomic_rmw_op_t*)command_data.data;
  const iree_hal_atomic_rmw_params_t params =
      iree_hal_remote_server_atomic_rmw_params_from_wire(&op->params);
  IREE_RETURN_IF_ERROR(iree_hal_atomic_rmw_params_validate(params));
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_remote_server_resolve_queue_atomic_target(
      session_slot, &op->target, params.width, "ATOMIC_RMW", &target_ref));
  return iree_hal_device_queue_atomic_rmw(
      local_device, (iree_hal_queue_affinity_t)op->target.queue_affinity,
      wait_list, signal_list, target_ref.buffer, target_ref.offset, params);
}

iree_status_t iree_hal_remote_server_command_buffer_atomic_wait(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_command_buffer_t* local_command_buffer,
    const iree_hal_remote_atomic_wait_cmd_t* command) {
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_remote_server_resolve_command_buffer_ref(
      session_slot, command->target.buffer_id, command->target.buffer_slot,
      command->target.offset, command->target.length, "ATOMIC_WAIT",
      &target_ref));
  return iree_hal_command_buffer_atomic_wait(
      local_command_buffer,
      (iree_hal_execution_stage_t)command->source_stage_mask,
      (iree_hal_execution_stage_t)command->target_stage_mask, target_ref,
      iree_hal_remote_server_atomic_wait_params_from_wire(&command->params));
}

iree_status_t iree_hal_remote_server_command_buffer_atomic_store(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_command_buffer_t* local_command_buffer,
    const iree_hal_remote_atomic_store_cmd_t* command) {
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_remote_server_resolve_command_buffer_ref(
      session_slot, command->target.buffer_id, command->target.buffer_slot,
      command->target.offset, command->target.length, "ATOMIC_STORE",
      &target_ref));
  return iree_hal_command_buffer_atomic_store(
      local_command_buffer,
      (iree_hal_execution_stage_t)command->source_stage_mask,
      (iree_hal_execution_stage_t)command->target_stage_mask, target_ref,
      iree_hal_remote_server_atomic_store_params_from_wire(&command->params));
}

iree_status_t iree_hal_remote_server_command_buffer_atomic_rmw(
    iree_hal_remote_server_session_t* session_slot,
    iree_hal_command_buffer_t* local_command_buffer,
    const iree_hal_remote_atomic_rmw_cmd_t* command) {
  iree_hal_buffer_ref_t target_ref;
  IREE_RETURN_IF_ERROR(iree_hal_remote_server_resolve_command_buffer_ref(
      session_slot, command->target.buffer_id, command->target.buffer_slot,
      command->target.offset, command->target.length, "ATOMIC_RMW",
      &target_ref));
  return iree_hal_command_buffer_atomic_rmw(
      local_command_buffer,
      (iree_hal_execution_stage_t)command->source_stage_mask,
      (iree_hal_execution_stage_t)command->target_stage_mask, target_ref,
      iree_hal_remote_server_atomic_rmw_params_from_wire(&command->params));
}
