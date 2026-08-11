// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/protocol/commands.h"

static iree_status_t iree_hal_remote_command_validate_length(
    const char* command_name, uint32_t actual_length,
    iree_host_size_t required_length) {
  if (actual_length != required_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%s command length %u does not match canonical length %" PRIhsz,
        command_name, actual_length, required_length);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_command_validate_minimum_length(
    const char* command_name, uint32_t actual_length,
    iree_host_size_t minimum_length) {
  if (actual_length < minimum_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%s command length %u is smaller than minimum length %" PRIhsz,
        command_name, actual_length, minimum_length);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_command_validate_buffer_barriers(
    const char* command_name, const uint8_t* command_data,
    iree_host_size_t buffer_barriers_offset, uint16_t buffer_barrier_count) {
  for (uint16_t i = 0; i < buffer_barrier_count; ++i) {
    iree_hal_remote_buffer_barrier_t barrier;
    memcpy(&barrier,
           command_data + buffer_barriers_offset +
               i * sizeof(iree_hal_remote_buffer_barrier_t),
           sizeof(barrier));
    if (barrier.reserved != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%s buffer barrier %u reserved field is nonzero",
                              command_name, i);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_command_validate_execution_barrier(
    const uint8_t* command_data, uint32_t command_length) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_command_validate_minimum_length(
      "EXECUTION_BARRIER", command_length,
      sizeof(iree_hal_remote_execution_barrier_cmd_t)));

  iree_hal_remote_execution_barrier_cmd_t command;
  memcpy(&command, command_data, sizeof(command));
  if (command.barrier_flags != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "EXECUTION_BARRIER reserved flags field is nonzero");
  }

  iree_host_size_t memory_barriers_offset = 0;
  iree_host_size_t buffer_barriers_offset = 0;
  iree_host_size_t required_length = 0;
  IREE_RETURN_IF_ERROR(
      IREE_STRUCT_LAYOUT(sizeof(command), &required_length,
                         IREE_STRUCT_FIELD(command.memory_barrier_count,
                                           iree_hal_remote_memory_barrier_t,
                                           &memory_barriers_offset),
                         IREE_STRUCT_FIELD(command.buffer_barrier_count,
                                           iree_hal_remote_buffer_barrier_t,
                                           &buffer_barriers_offset),
                         IREE_STRUCT_FIELD_ALIGNED(0, uint8_t, 8, NULL)));
  IREE_RETURN_IF_ERROR(iree_hal_remote_command_validate_length(
      "EXECUTION_BARRIER", command_length, required_length));
  return iree_hal_remote_command_validate_buffer_barriers(
      "EXECUTION_BARRIER", command_data, buffer_barriers_offset,
      command.buffer_barrier_count);
}

static iree_status_t iree_hal_remote_command_validate_event_signal(
    const uint8_t* command_data, uint32_t command_length) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_command_validate_length(
      "EVENT_SIGNAL", command_length,
      sizeof(iree_hal_remote_event_signal_cmd_t)));
  iree_hal_remote_event_signal_cmd_t command;
  memcpy(&command, command_data, sizeof(command));
  if (command.reserved != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "EVENT_SIGNAL reserved field is nonzero");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_command_validate_event_reset(
    const uint8_t* command_data, uint32_t command_length) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_command_validate_length(
      "EVENT_RESET", command_length,
      sizeof(iree_hal_remote_event_reset_cmd_t)));
  iree_hal_remote_event_reset_cmd_t command;
  memcpy(&command, command_data, sizeof(command));
  if (command.reserved != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "EVENT_RESET reserved field is nonzero");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_command_validate_event_wait(
    const uint8_t* command_data, uint32_t command_length) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_command_validate_minimum_length(
      "EVENT_WAIT", command_length, sizeof(iree_hal_remote_event_wait_cmd_t)));

  iree_hal_remote_event_wait_cmd_t command;
  memcpy(&command, command_data, sizeof(command));
  if (command.reserved != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "EVENT_WAIT reserved field is nonzero");
  }

  iree_host_size_t event_ids_offset = 0;
  iree_host_size_t memory_barriers_offset = 0;
  iree_host_size_t buffer_barriers_offset = 0;
  iree_host_size_t required_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(command), &required_length,
      IREE_STRUCT_FIELD(command.event_count, iree_hal_remote_resource_id_t,
                        &event_ids_offset),
      IREE_STRUCT_FIELD(command.memory_barrier_count,
                        iree_hal_remote_memory_barrier_t,
                        &memory_barriers_offset),
      IREE_STRUCT_FIELD(command.buffer_barrier_count,
                        iree_hal_remote_buffer_barrier_t,
                        &buffer_barriers_offset),
      IREE_STRUCT_FIELD_ALIGNED(0, uint8_t, 8, NULL)));
  IREE_RETURN_IF_ERROR(iree_hal_remote_command_validate_length(
      "EVENT_WAIT", command_length, required_length));
  return iree_hal_remote_command_validate_buffer_barriers(
      "EVENT_WAIT", command_data, buffer_barriers_offset,
      command.buffer_barrier_count);
}

static iree_status_t iree_hal_remote_command_validate_buffer_fill(
    const uint8_t* command_data, uint32_t command_length) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_command_validate_length(
      "BUFFER_FILL", command_length,
      sizeof(iree_hal_remote_buffer_fill_cmd_t)));
  iree_hal_remote_buffer_fill_cmd_t command;
  memcpy(&command, command_data, sizeof(command));
  if (command.reserved0[0] != 0 || command.reserved0[1] != 0 ||
      command.reserved0[2] != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "BUFFER_FILL reserved field is nonzero");
  }
  if (command.pattern_length != 1 && command.pattern_length != 2 &&
      command.pattern_length != 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "BUFFER_FILL pattern length %u is invalid",
                            command.pattern_length);
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_command_validate_buffer_update(
    const uint8_t* command_data, uint32_t command_length) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_command_validate_minimum_length(
      "BUFFER_UPDATE", command_length,
      sizeof(iree_hal_remote_buffer_update_cmd_t)));
  iree_hal_remote_buffer_update_cmd_t command;
  memcpy(&command, command_data, sizeof(command));
  if (command.target_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "BUFFER_UPDATE length %" PRIu64
                            " exceeds host capacity",
                            command.target_length);
  }

  iree_host_size_t data_offset = 0;
  iree_host_size_t required_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(command), &required_length,
      IREE_STRUCT_FIELD((iree_host_size_t)command.target_length, uint8_t,
                        &data_offset),
      IREE_STRUCT_FIELD_ALIGNED(0, uint8_t, 8, NULL)));
  return iree_hal_remote_command_validate_length(
      "BUFFER_UPDATE", command_length, required_length);
}

static iree_status_t iree_hal_remote_command_validate_buffer_copy(
    const uint8_t* command_data, uint32_t command_length) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_command_validate_length(
      "BUFFER_COPY", command_length,
      sizeof(iree_hal_remote_buffer_copy_cmd_t)));
  iree_hal_remote_buffer_copy_cmd_t command;
  memcpy(&command, command_data, sizeof(command));
  if (command.reserved != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "BUFFER_COPY reserved field is nonzero");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_command_validate_dispatch(
    const uint8_t* command_data, uint32_t command_length) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_command_validate_minimum_length(
      "DISPATCH", command_length, sizeof(iree_hal_remote_dispatch_cmd_t)));
  iree_hal_remote_dispatch_cmd_t command;
  memcpy(&command, command_data, sizeof(command));
  if (command.reserved1 != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "DISPATCH reserved field is nonzero");
  }

  iree_host_size_t constants_offset = 0;
  iree_host_size_t bindings_offset = 0;
  iree_host_size_t required_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(command), &required_length,
      IREE_STRUCT_FIELD(command.constant_count, uint32_t, &constants_offset),
      IREE_STRUCT_FIELD_ALIGNED(command.binding_count,
                                iree_hal_remote_binding_t, 8, &bindings_offset),
      IREE_STRUCT_FIELD_ALIGNED(0, uint8_t, 8, NULL)));
  IREE_RETURN_IF_ERROR(iree_hal_remote_command_validate_length(
      "DISPATCH", command_length, required_length));

  for (uint16_t i = 0; i < command.binding_count; ++i) {
    iree_hal_remote_binding_t binding;
    memcpy(
        &binding,
        command_data + bindings_offset + i * sizeof(iree_hal_remote_binding_t),
        sizeof(binding));
    if (binding.reserved != 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "DISPATCH binding %u reserved field is nonzero",
                              i);
    }
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_command_validate_debug_group_begin(
    const uint8_t* command_data, uint32_t command_length) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_command_validate_minimum_length(
      "DEBUG_GROUP_BEGIN", command_length,
      sizeof(iree_hal_remote_debug_group_begin_cmd_t)));
  iree_hal_remote_debug_group_begin_cmd_t command;
  memcpy(&command, command_data, sizeof(command));
  if (command.reserved != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "DEBUG_GROUP_BEGIN reserved field is nonzero");
  }

  iree_host_size_t label_offset = 0;
  iree_host_size_t required_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(command), &required_length,
      IREE_STRUCT_FIELD(command.label_length, uint8_t, &label_offset),
      IREE_STRUCT_FIELD_ALIGNED(0, uint8_t, 8, NULL)));
  return iree_hal_remote_command_validate_length(
      "DEBUG_GROUP_BEGIN", command_length, required_length);
}

static iree_status_t iree_hal_remote_command_validate_extension(
    const uint8_t* command_data, uint32_t command_length) {
  IREE_RETURN_IF_ERROR(iree_hal_remote_command_validate_minimum_length(
      "COMMAND_EXTENSION", command_length,
      sizeof(iree_hal_remote_command_extension_cmd_t)));
  iree_hal_remote_command_extension_cmd_t command;
  memcpy(&command, command_data, sizeof(command));
  if (command.reserved != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "COMMAND_EXTENSION reserved field is nonzero");
  }

  iree_host_size_t payload_offset = 0;
  iree_host_size_t required_length = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(command), &required_length,
      IREE_STRUCT_FIELD(command.payload_length, uint8_t, &payload_offset),
      IREE_STRUCT_FIELD_ALIGNED(0, uint8_t, 8, NULL)));
  return iree_hal_remote_command_validate_length(
      "COMMAND_EXTENSION", command_length, required_length);
}

iree_status_t iree_hal_remote_command_parse(
    iree_const_byte_span_t stream_bytes,
    iree_hal_remote_command_view_t* out_command) {
  IREE_ASSERT_ARGUMENT(out_command);
  memset(out_command, 0, sizeof(*out_command));

  if (stream_bytes.data_length < sizeof(iree_hal_remote_cmd_header_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command stream has a truncated header");
  }

  iree_hal_remote_cmd_header_t header;
  memcpy(&header, stream_bytes.data, sizeof(header));
  if (header.reserved != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command header reserved field is nonzero");
  }
  if (header.length < sizeof(header)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command length %u is smaller than its header",
                            header.length);
  }
  if (!iree_host_size_has_alignment(header.length, 8)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "command length %u is not 8-byte aligned",
                            header.length);
  }
  if (header.length > stream_bytes.data_length) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "command length %u exceeds remaining stream length %" PRIhsz,
        header.length, stream_bytes.data_length);
  }

  iree_status_t status = iree_ok_status();
  switch (header.type) {
    case IREE_HAL_REMOTE_CMD_EXECUTION_BARRIER:
      status = iree_hal_remote_command_validate_execution_barrier(
          stream_bytes.data, header.length);
      break;
    case IREE_HAL_REMOTE_CMD_EVENT_SIGNAL:
      status = iree_hal_remote_command_validate_event_signal(stream_bytes.data,
                                                             header.length);
      break;
    case IREE_HAL_REMOTE_CMD_EVENT_RESET:
      status = iree_hal_remote_command_validate_event_reset(stream_bytes.data,
                                                            header.length);
      break;
    case IREE_HAL_REMOTE_CMD_EVENT_WAIT:
      status = iree_hal_remote_command_validate_event_wait(stream_bytes.data,
                                                           header.length);
      break;
    case IREE_HAL_REMOTE_CMD_BUFFER_ADVISE:
      status = iree_hal_remote_command_validate_length(
          "BUFFER_ADVISE", header.length,
          sizeof(iree_hal_remote_buffer_advise_cmd_t));
      break;
    case IREE_HAL_REMOTE_CMD_BUFFER_FILL:
      status = iree_hal_remote_command_validate_buffer_fill(stream_bytes.data,
                                                            header.length);
      break;
    case IREE_HAL_REMOTE_CMD_BUFFER_UPDATE:
      status = iree_hal_remote_command_validate_buffer_update(stream_bytes.data,
                                                              header.length);
      break;
    case IREE_HAL_REMOTE_CMD_BUFFER_COPY:
      status = iree_hal_remote_command_validate_buffer_copy(stream_bytes.data,
                                                            header.length);
      break;
    case IREE_HAL_REMOTE_CMD_DISPATCH:
      status = iree_hal_remote_command_validate_dispatch(stream_bytes.data,
                                                         header.length);
      break;
    case IREE_HAL_REMOTE_CMD_DEBUG_GROUP_BEGIN:
      status = iree_hal_remote_command_validate_debug_group_begin(
          stream_bytes.data, header.length);
      break;
    case IREE_HAL_REMOTE_CMD_DEBUG_GROUP_END:
      status = iree_hal_remote_command_validate_length(
          "DEBUG_GROUP_END", header.length,
          sizeof(iree_hal_remote_debug_group_end_cmd_t));
      break;
    case IREE_HAL_REMOTE_CMD_COMMAND_EXTENSION:
      status = iree_hal_remote_command_validate_extension(stream_bytes.data,
                                                          header.length);
      break;
    default:
      status =
          iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                           "command type 0x%04x is not supported", header.type);
      break;
  }

  if (iree_status_is_ok(status)) {
    out_command->header = header;
    out_command->bytes =
        iree_make_const_byte_span(stream_bytes.data, header.length);
  }
  return status;
}
