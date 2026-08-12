// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/command_buffer.h"

#include "iree/hal/remote/client/buffer.h"
#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/client/event.h"
#include "iree/hal/remote/client/executable.h"
#include "iree/hal/remote/protocol/commands.h"
#include "iree/hal/remote/protocol/control.h"
#include "iree/hal/utils/resource_set.h"

//===----------------------------------------------------------------------===//
// iree_hal_remote_client_command_buffer_t
//===----------------------------------------------------------------------===//

// Initial stream buffer capacity. Covers small recordings (a few dispatches
// with barriers) without reallocation.
#define IREE_HAL_REMOTE_CB_INITIAL_CAPACITY 4096

static const iree_hal_command_buffer_vtable_t
    iree_hal_remote_client_command_buffer_vtable;

typedef struct iree_hal_remote_client_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  iree_hal_remote_client_device_t* device;

  // Serialized command stream (iree_hal_remote_cmd_header_t sequence).
  uint8_t* stream_data;
  iree_host_size_t stream_length;
  iree_host_size_t stream_capacity;

  // Server resource ID (reusable only). Initially provisional (from end()),
  // resolved to canonical when upload response arrives.
  iree_hal_remote_resource_id_t resource_id;

  // Retained direct resources referenced by the serialized command stream.
  iree_hal_resource_set_t* resource_set;
} iree_hal_remote_client_command_buffer_t;

static iree_hal_remote_client_command_buffer_t*
iree_hal_remote_client_command_buffer_cast(
    iree_hal_command_buffer_t* base_command_buffer) {
  IREE_HAL_ASSERT_TYPE(base_command_buffer,
                       &iree_hal_remote_client_command_buffer_vtable);
  return (iree_hal_remote_client_command_buffer_t*)base_command_buffer;
}

//===----------------------------------------------------------------------===//
// Buffer reference helpers
//===----------------------------------------------------------------------===//

static void iree_hal_remote_client_command_buffer_release_resources(
    iree_hal_remote_client_command_buffer_t* command_buffer) {
  iree_hal_resource_set_free(command_buffer->resource_set);
  command_buffer->resource_set = NULL;
}

static iree_status_t iree_hal_remote_client_command_buffer_retain_resource(
    iree_hal_remote_client_command_buffer_t* command_buffer,
    iree_hal_resource_t* resource) {
  return iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                      &resource);
}

// Resolves a buffer reference for wire serialization. Returns the resource_id
// of the root allocation and adjusts the range to be absolute within that
// allocation (adding the subspan byte_offset if the buffer is a subspan).
static iree_hal_remote_resource_id_t
iree_hal_remote_client_cb_resolve_buffer_ref(
    iree_hal_remote_client_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t ref, uint64_t* out_offset, uint64_t* out_length,
    iree_status_t* out_status) {
  if (!ref.buffer) {
    *out_offset = ref.offset;
    *out_length = ref.length;
    return 0;
  }
  iree_hal_remote_resource_id_t resource_id = 0;
  uint64_t byte_offset = 0;
  uint64_t byte_length = 0;
  iree_status_t status = iree_hal_remote_client_buffer_resolve_wire_range(
      ref.buffer, ref.offset, ref.length, &resource_id, &byte_offset,
      &byte_length);
  if (!iree_status_is_ok(status)) {
    *out_status = status;
    return 0;
  }
  status = iree_hal_remote_client_command_buffer_retain_resource(
      command_buffer, (iree_hal_resource_t*)ref.buffer);
  if (!iree_status_is_ok(status)) {
    *out_status = status;
    return 0;
  }
  *out_offset = byte_offset;
  *out_length = byte_length;
  return resource_id;
}

// Encodes a direct or indirect HAL buffer reference for command stream use.
// Direct refs carry a server resource ID; indirect refs carry the binding table
// slot that will be resolved when the command buffer executes.
static void iree_hal_remote_client_cb_encode_buffer_ref(
    iree_hal_remote_client_command_buffer_t* command_buffer,
    iree_hal_buffer_ref_t ref, iree_hal_remote_resource_id_t* out_buffer_id,
    uint32_t* out_buffer_slot, uint64_t* out_offset, uint64_t* out_length,
    iree_status_t* out_status) {
  if (!ref.buffer) {
    *out_buffer_id = 0;
    *out_buffer_slot = ref.buffer_slot;
    *out_offset = ref.offset;
    *out_length = ref.length;
    return;
  }
  *out_buffer_id = iree_hal_remote_client_cb_resolve_buffer_ref(
      command_buffer, ref, out_offset, out_length, out_status);
  *out_buffer_slot = 0;
}

//===----------------------------------------------------------------------===//
// Stream writing helpers
//===----------------------------------------------------------------------===//

// Ensures the stream buffer has room for |additional_bytes| more bytes.
static iree_status_t iree_hal_remote_client_cb_ensure_capacity(
    iree_hal_remote_client_command_buffer_t* command_buffer,
    iree_host_size_t additional_bytes) {
  iree_host_size_t required = 0;
  if (!iree_host_size_checked_add(command_buffer->stream_length,
                                  additional_bytes, &required)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "command stream size overflow");
  }
  if (required <= command_buffer->stream_capacity) return iree_ok_status();

  return iree_allocator_grow_array(
      command_buffer->host_allocator, required, /*element_size=*/1,
      &command_buffer->stream_capacity, (void**)&command_buffer->stream_data);
}

// Appends one complete zeroed command and initializes its common header.
static iree_status_t iree_hal_remote_client_cb_append_command(
    iree_hal_remote_client_command_buffer_t* command_buffer,
    iree_hal_remote_cmd_type_t command_type, iree_host_size_t command_length,
    void** out_command) {
  IREE_ASSERT(command_length >= sizeof(iree_hal_remote_cmd_header_t));
  IREE_ASSERT(iree_host_size_has_alignment(command_length, 8));
  if (command_length > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "remote command length %" PRIhsz
                            " exceeds wire limit %u",
                            command_length, (unsigned)UINT32_MAX);
  }

  IREE_RETURN_IF_ERROR(iree_hal_remote_client_cb_ensure_capacity(
      command_buffer, command_length));
  uint8_t* ptr = command_buffer->stream_data + command_buffer->stream_length;
  memset(ptr, 0, command_length);
  iree_hal_remote_cmd_header_t* header = (iree_hal_remote_cmd_header_t*)ptr;
  header->type = (uint16_t)command_type;
  header->length = (uint32_t)command_length;
  command_buffer->stream_length += command_length;
  *out_command = ptr;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

static void iree_hal_remote_client_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;

  // Release server-side resource for reusable command buffers.
  if (command_buffer->resource_id != 0) {
    iree_status_ignore(iree_hal_remote_client_device_release_resource(
        command_buffer->device, command_buffer->resource_id));
  }

  iree_hal_remote_client_command_buffer_release_resources(command_buffer);
  iree_allocator_free(host_allocator, command_buffer->stream_data);
  iree_allocator_free(host_allocator, command_buffer);
}

static iree_status_t iree_hal_remote_client_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  // Re-recording: release old server resource if present.
  if (command_buffer->resource_id != 0) {
    iree_status_ignore(iree_hal_remote_client_device_release_resource(
        command_buffer->device, command_buffer->resource_id));
    command_buffer->resource_id = 0;
  }

  // Reset recording position (keeps the allocation for reuse).
  iree_hal_remote_client_command_buffer_release_resources(command_buffer);
  iree_status_t status = iree_hal_resource_set_allocate(
      &command_buffer->device->resource_set_block_pool,
      &command_buffer->resource_set);
  if (iree_status_is_ok(status)) {
    command_buffer->stream_length = 0;
  }
  return status;
}

static iree_status_t iree_hal_remote_client_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  // For reusable command buffers, upload the recorded stream to the server.
  // The upload is async via control RPC. We assign a provisional resource ID
  // that can be referenced immediately in queue_execute.
  if (!iree_all_bits_set(base_command_buffer->mode,
                         IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    // Build COMMAND_BUFFER_UPLOAD request.
    const iree_host_size_t header_size =
        sizeof(iree_hal_remote_control_envelope_t) +
        sizeof(iree_hal_remote_command_buffer_upload_request_t);
    iree_host_size_t message_size = 0;
    iree_host_size_t stream_offset = 0;
    IREE_RETURN_IF_ERROR(
        IREE_STRUCT_LAYOUT(header_size, &message_size,
                           IREE_STRUCT_FIELD(command_buffer->stream_length,
                                             uint8_t, &stream_offset)));

    uint8_t* message_buffer = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        command_buffer->host_allocator, message_size, (void**)&message_buffer));
    memset(message_buffer, 0, header_size);

    iree_hal_remote_control_envelope_t* envelope =
        (iree_hal_remote_control_envelope_t*)message_buffer;
    envelope->message_type = IREE_HAL_REMOTE_CONTROL_COMMAND_BUFFER_UPLOAD;

    iree_hal_remote_command_buffer_upload_request_t* request =
        (iree_hal_remote_command_buffer_upload_request_t*)(envelope + 1);
    request->provisional_id = IREE_HAL_REMOTE_RESOURCE_ID_PROVISIONAL(
        IREE_HAL_REMOTE_RESOURCE_TYPE_COMMAND_BUFFER, 0);
    request->mode = (uint32_t)base_command_buffer->mode;
    request->categories = (uint32_t)base_command_buffer->allowed_categories;
    request->binding_capacity = (uint16_t)base_command_buffer->binding_capacity;
    request->upload_flags = IREE_HAL_REMOTE_UPLOAD_FLAG_INLINE_DATA;
    request->data_length = command_buffer->stream_length;

    // Copy stream data after the header.
    if (command_buffer->stream_length > 0) {
      memcpy(message_buffer + stream_offset, command_buffer->stream_data,
             command_buffer->stream_length);
    }

    // Send RPC and get resolved resource ID.
    iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
    iree_async_buffer_lease_t response_lease;
    memset(&response_lease, 0, sizeof(response_lease));
    iree_status_t status = iree_hal_remote_client_device_control_rpc(
        command_buffer->device,
        iree_make_const_byte_span(message_buffer, message_size),
        &response_payload, &response_lease);

    iree_allocator_free(command_buffer->host_allocator, message_buffer);

    if (iree_status_is_ok(status)) {
      if (response_payload.data_length <
          sizeof(iree_hal_remote_command_buffer_upload_response_t)) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "COMMAND_BUFFER_UPLOAD response too short: %" PRIhsz " bytes",
            response_payload.data_length);
      }
    }
    if (iree_status_is_ok(status)) {
      const iree_hal_remote_command_buffer_upload_response_t* response =
          (const iree_hal_remote_command_buffer_upload_response_t*)
              response_payload.data;
      command_buffer->resource_id = response->resolved_id;
    }

    iree_async_buffer_lease_release(&response_lease);
    IREE_RETURN_IF_ERROR(status);
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Debug groups
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_remote_client_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);
  (void)location;

  if (label.size > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "debug label length %" PRIhsz
                            " exceeds wire limit %u",
                            label.size, (unsigned)UINT16_MAX);
  }

  iree_host_size_t label_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_debug_group_begin_cmd_t), &total_size,
      IREE_STRUCT_FIELD(label.size, uint8_t, &label_offset),
      IREE_STRUCT_FIELD_ALIGNED(0, uint8_t, 8, NULL)));

  void* ptr = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_remote_client_cb_append_command(
      command_buffer, IREE_HAL_REMOTE_CMD_DEBUG_GROUP_BEGIN, total_size, &ptr));

  iree_hal_remote_debug_group_begin_cmd_t* cmd =
      (iree_hal_remote_debug_group_begin_cmd_t*)ptr;
  cmd->label_color = label_color.value;
  cmd->label_length = (uint16_t)label.size;

  if (label.size > 0) {
    memcpy((uint8_t*)cmd + label_offset, label.data, label.size);
  }

  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  void* ptr = NULL;
  return iree_hal_remote_client_cb_append_command(
      command_buffer, IREE_HAL_REMOTE_CMD_DEBUG_GROUP_END,
      sizeof(iree_hal_remote_debug_group_end_cmd_t), &ptr);
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_remote_client_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  iree_status_t status = iree_ok_status();
  if (memory_barrier_count > UINT16_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "memory barrier count %" PRIhsz
                              " exceeds wire limit %u",
                              memory_barrier_count, (unsigned)UINT16_MAX);
  }
  if (iree_status_is_ok(status) && buffer_barrier_count > UINT16_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "buffer barrier count %" PRIhsz
                              " exceeds wire limit %u",
                              buffer_barrier_count, (unsigned)UINT16_MAX);
  }
  if (iree_status_is_ok(status) && memory_barrier_count > 0 &&
      !memory_barriers) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "memory barrier list is NULL");
  }
  if (iree_status_is_ok(status) && buffer_barrier_count > 0 &&
      !buffer_barriers) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "buffer barrier list is NULL");
  }
  iree_host_size_t memory_barriers_offset = 0;
  iree_host_size_t buffer_barriers_offset = 0;
  iree_host_size_t total_size = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_remote_execution_barrier_cmd_t), &total_size,
        IREE_STRUCT_FIELD(memory_barrier_count,
                          iree_hal_remote_memory_barrier_t,
                          &memory_barriers_offset),
        IREE_STRUCT_FIELD(buffer_barrier_count,
                          iree_hal_remote_buffer_barrier_t,
                          &buffer_barriers_offset),
        IREE_STRUCT_FIELD_ALIGNED(0, uint8_t, 8, NULL));
  }
  const iree_host_size_t stream_start = command_buffer->stream_length;
  void* ptr = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_cb_append_command(
        command_buffer, IREE_HAL_REMOTE_CMD_EXECUTION_BARRIER, total_size,
        &ptr);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_remote_execution_barrier_cmd_t* cmd =
        (iree_hal_remote_execution_barrier_cmd_t*)ptr;
    cmd->source_stage_mask = (uint32_t)source_stage_mask;
    cmd->target_stage_mask = (uint32_t)target_stage_mask;
    cmd->memory_barrier_count = (uint16_t)memory_barrier_count;
    cmd->buffer_barrier_count = (uint16_t)buffer_barrier_count;

    // Serialize memory barriers.
    iree_hal_remote_memory_barrier_t* wire_memory_barriers =
        (iree_hal_remote_memory_barrier_t*)((uint8_t*)cmd +
                                            memory_barriers_offset);
    for (iree_host_size_t i = 0; i < memory_barrier_count; ++i) {
      iree_hal_remote_memory_barrier_t* wire = &wire_memory_barriers[i];
      wire->source_scope = (uint32_t)memory_barriers[i].source_scope;
      wire->target_scope = (uint32_t)memory_barriers[i].target_scope;
    }

    // Serialize buffer barriers (resolve buffer IDs + subspan offsets).
    iree_hal_remote_buffer_barrier_t* wire_buffer_barriers =
        (iree_hal_remote_buffer_barrier_t*)((uint8_t*)cmd +
                                            buffer_barriers_offset);
    for (iree_host_size_t i = 0;
         i < buffer_barrier_count && iree_status_is_ok(status); ++i) {
      iree_hal_remote_buffer_barrier_t* wire = &wire_buffer_barriers[i];
      wire->source_scope = (uint32_t)buffer_barriers[i].source_scope;
      wire->target_scope = (uint32_t)buffer_barriers[i].target_scope;
      uint64_t barrier_offset = 0;
      uint64_t barrier_length = 0;
      iree_hal_remote_client_cb_encode_buffer_ref(
          command_buffer, buffer_barriers[i].buffer_ref, &wire->buffer_id,
          &wire->buffer_slot, &barrier_offset, &barrier_length, &status);
      wire->offset = barrier_offset;
      wire->length = barrier_length;
    }
  }

  if (!iree_status_is_ok(status)) {
    command_buffer->stream_length = stream_start;
  }
  return status;
}

static iree_status_t iree_hal_remote_client_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  const iree_host_size_t stream_start = command_buffer->stream_length;
  void* ptr = NULL;
  iree_status_t status = iree_hal_remote_client_cb_append_command(
      command_buffer, IREE_HAL_REMOTE_CMD_EVENT_SIGNAL,
      sizeof(iree_hal_remote_event_signal_cmd_t), &ptr);
  if (iree_status_is_ok(status)) {
    if (!iree_hal_remote_client_event_isa(event)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "cannot record remote command buffer signal_event with a non-remote "
          "event");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_command_buffer_retain_resource(
        command_buffer, (iree_hal_resource_t*)event);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_event_signal_cmd_t* cmd =
        (iree_hal_remote_event_signal_cmd_t*)ptr;
    cmd->event_id = iree_hal_remote_client_event_resource_id(event);
    cmd->source_stage_mask = (uint32_t)source_stage_mask;
  }

  if (!iree_status_is_ok(status)) {
    command_buffer->stream_length = stream_start;
  }
  return status;
}

static iree_status_t iree_hal_remote_client_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  const iree_host_size_t stream_start = command_buffer->stream_length;
  void* ptr = NULL;
  iree_status_t status = iree_hal_remote_client_cb_append_command(
      command_buffer, IREE_HAL_REMOTE_CMD_EVENT_RESET,
      sizeof(iree_hal_remote_event_reset_cmd_t), &ptr);
  if (iree_status_is_ok(status)) {
    if (!iree_hal_remote_client_event_isa(event)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "cannot record remote command buffer reset_event with a non-remote "
          "event");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_command_buffer_retain_resource(
        command_buffer, (iree_hal_resource_t*)event);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_remote_event_reset_cmd_t* cmd =
        (iree_hal_remote_event_reset_cmd_t*)ptr;
    cmd->event_id = iree_hal_remote_client_event_resource_id(event);
    cmd->source_stage_mask = (uint32_t)source_stage_mask;
  }

  if (!iree_status_is_ok(status)) {
    command_buffer->stream_length = stream_start;
  }
  return status;
}

static iree_status_t iree_hal_remote_client_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  iree_status_t status = iree_ok_status();
  if (event_count > UINT16_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "event count %" PRIhsz " exceeds wire limit %u",
                              event_count, (unsigned)UINT16_MAX);
  }
  if (iree_status_is_ok(status) && memory_barrier_count > UINT16_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "memory barrier count %" PRIhsz
                              " exceeds wire limit %u",
                              memory_barrier_count, (unsigned)UINT16_MAX);
  }
  if (iree_status_is_ok(status) && buffer_barrier_count > UINT16_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "buffer barrier count %" PRIhsz
                              " exceeds wire limit %u",
                              buffer_barrier_count, (unsigned)UINT16_MAX);
  }
  if (iree_status_is_ok(status) && event_count > 0 && !events) {
    status =
        iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "event list is NULL");
  }
  if (iree_status_is_ok(status) && memory_barrier_count > 0 &&
      !memory_barriers) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "memory barrier list is NULL");
  }
  if (iree_status_is_ok(status) && buffer_barrier_count > 0 &&
      !buffer_barriers) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "buffer barrier list is NULL");
  }

  iree_host_size_t event_ids_offset = 0;
  iree_host_size_t memory_barriers_offset = 0;
  iree_host_size_t buffer_barriers_offset = 0;
  iree_host_size_t total_size = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        sizeof(iree_hal_remote_event_wait_cmd_t), &total_size,
        IREE_STRUCT_FIELD(event_count, iree_hal_remote_resource_id_t,
                          &event_ids_offset),
        IREE_STRUCT_FIELD(memory_barrier_count,
                          iree_hal_remote_memory_barrier_t,
                          &memory_barriers_offset),
        IREE_STRUCT_FIELD(buffer_barrier_count,
                          iree_hal_remote_buffer_barrier_t,
                          &buffer_barriers_offset),
        IREE_STRUCT_FIELD_ALIGNED(0, uint8_t, 8, NULL));
  }
  const iree_host_size_t stream_start = command_buffer->stream_length;
  void* ptr = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_cb_append_command(
        command_buffer, IREE_HAL_REMOTE_CMD_EVENT_WAIT, total_size, &ptr);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_remote_event_wait_cmd_t* cmd =
        (iree_hal_remote_event_wait_cmd_t*)ptr;
    cmd->source_stage_mask = (uint32_t)source_stage_mask;
    cmd->target_stage_mask = (uint32_t)target_stage_mask;
    cmd->event_count = (uint16_t)event_count;
    cmd->memory_barrier_count = (uint16_t)memory_barrier_count;
    cmd->buffer_barrier_count = (uint16_t)buffer_barrier_count;

    iree_hal_remote_resource_id_t* event_ids =
        (iree_hal_remote_resource_id_t*)((uint8_t*)cmd + event_ids_offset);
    for (iree_host_size_t i = 0; i < event_count && iree_status_is_ok(status);
         ++i) {
      if (!iree_hal_remote_client_event_isa(events[i])) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "cannot record remote command buffer wait_events with a "
            "non-remote event");
        break;
      }
      status = iree_hal_remote_client_command_buffer_retain_resource(
          command_buffer, (iree_hal_resource_t*)events[i]);
      event_ids[i] = iree_hal_remote_client_event_resource_id(events[i]);
    }

    if (iree_status_is_ok(status)) {
      iree_hal_remote_memory_barrier_t* wire_memory_barriers =
          (iree_hal_remote_memory_barrier_t*)((uint8_t*)cmd +
                                              memory_barriers_offset);
      for (iree_host_size_t i = 0; i < memory_barrier_count; ++i) {
        iree_hal_remote_memory_barrier_t* wire = &wire_memory_barriers[i];
        wire->source_scope = (uint32_t)memory_barriers[i].source_scope;
        wire->target_scope = (uint32_t)memory_barriers[i].target_scope;
      }
    }

    if (iree_status_is_ok(status)) {
      iree_hal_remote_buffer_barrier_t* wire_buffer_barriers =
          (iree_hal_remote_buffer_barrier_t*)((uint8_t*)cmd +
                                              buffer_barriers_offset);
      for (iree_host_size_t i = 0;
           i < buffer_barrier_count && iree_status_is_ok(status); ++i) {
        iree_hal_remote_buffer_barrier_t* wire = &wire_buffer_barriers[i];
        wire->source_scope = (uint32_t)buffer_barriers[i].source_scope;
        wire->target_scope = (uint32_t)buffer_barriers[i].target_scope;
        uint64_t barrier_offset = 0;
        uint64_t barrier_length = 0;
        iree_hal_remote_client_cb_encode_buffer_ref(
            command_buffer, buffer_barriers[i].buffer_ref, &wire->buffer_id,
            &wire->buffer_slot, &barrier_offset, &barrier_length, &status);
        wire->offset = barrier_offset;
        wire->length = barrier_length;
      }
    }
  }

  if (!iree_status_is_ok(status)) {
    command_buffer->stream_length = stream_start;
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Transfer operations
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_remote_client_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  const iree_host_size_t stream_start = command_buffer->stream_length;
  void* ptr = NULL;
  iree_status_t status = iree_hal_remote_client_cb_append_command(
      command_buffer, IREE_HAL_REMOTE_CMD_BUFFER_FILL,
      sizeof(iree_hal_remote_buffer_fill_cmd_t), &ptr);

  if (iree_status_is_ok(status)) {
    iree_hal_remote_buffer_fill_cmd_t* cmd =
        (iree_hal_remote_buffer_fill_cmd_t*)ptr;
    uint64_t target_offset = 0;
    uint64_t target_length = 0;
    iree_hal_remote_client_cb_encode_buffer_ref(
        command_buffer, target_ref, &cmd->target_buffer_id,
        &cmd->target_buffer_slot, &target_offset, &target_length, &status);
    cmd->target_offset = target_offset;
    cmd->target_length = target_length;
    cmd->pattern_length = (uint8_t)pattern_length;
    cmd->fill_flags = (uint32_t)flags;

    if (iree_status_is_ok(status)) {
      // Zero-extend pattern into uint32_t.
      memcpy(&cmd->pattern, pattern,
             iree_min(pattern_length, sizeof(cmd->pattern)));
    }
  }

  if (!iree_status_is_ok(status)) {
    command_buffer->stream_length = stream_start;
  }
  return status;
}

static iree_status_t iree_hal_remote_client_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  if (target_ref.length > IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "update length %" PRIdsz " exceeds HAL limit %u",
                            target_ref.length,
                            (unsigned)IREE_HAL_COMMAND_BUFFER_MAX_UPDATE_SIZE);
  }
  iree_host_size_t data_length = (iree_host_size_t)target_ref.length;
  iree_host_size_t data_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_buffer_update_cmd_t), &total_size,
      IREE_STRUCT_FIELD(data_length, uint8_t, &data_offset),
      IREE_STRUCT_FIELD_ALIGNED(0, uint8_t, 8, NULL)));

  const iree_host_size_t stream_start = command_buffer->stream_length;
  void* ptr = NULL;
  iree_status_t status = iree_hal_remote_client_cb_append_command(
      command_buffer, IREE_HAL_REMOTE_CMD_BUFFER_UPDATE, total_size, &ptr);

  if (iree_status_is_ok(status)) {
    iree_hal_remote_buffer_update_cmd_t* cmd =
        (iree_hal_remote_buffer_update_cmd_t*)ptr;
    uint64_t update_target_offset = 0;
    uint64_t update_target_length = 0;
    iree_hal_remote_client_cb_encode_buffer_ref(
        command_buffer, target_ref, &cmd->target_buffer_id,
        &cmd->target_buffer_slot, &update_target_offset, &update_target_length,
        &status);
    cmd->target_offset = update_target_offset;
    cmd->target_length = update_target_length;
    cmd->update_flags = (uint32_t)flags;

    if (iree_status_is_ok(status)) {
      // Deep-copy the source data into the stream.
      memcpy((uint8_t*)cmd + data_offset,
             (const uint8_t*)source_buffer + source_offset, data_length);
    }
  }

  if (!iree_status_is_ok(status)) {
    command_buffer->stream_length = stream_start;
  }
  return status;
}

static iree_status_t iree_hal_remote_client_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  const iree_host_size_t stream_start = command_buffer->stream_length;
  void* ptr = NULL;
  iree_status_t status = iree_hal_remote_client_cb_append_command(
      command_buffer, IREE_HAL_REMOTE_CMD_BUFFER_COPY,
      sizeof(iree_hal_remote_buffer_copy_cmd_t), &ptr);

  if (iree_status_is_ok(status)) {
    iree_hal_remote_buffer_copy_cmd_t* cmd =
        (iree_hal_remote_buffer_copy_cmd_t*)ptr;
    uint64_t copy_source_offset = 0;
    uint64_t copy_source_length = 0;
    uint64_t copy_target_offset = 0;
    uint64_t copy_target_length = 0;
    iree_hal_remote_client_cb_encode_buffer_ref(
        command_buffer, source_ref, &cmd->source_buffer_id,
        &cmd->source_buffer_slot, &copy_source_offset, &copy_source_length,
        &status);
    cmd->source_offset = copy_source_offset;
    if (iree_status_is_ok(status)) {
      iree_hal_remote_client_cb_encode_buffer_ref(
          command_buffer, target_ref, &cmd->target_buffer_id,
          &cmd->target_buffer_slot, &copy_target_offset, &copy_target_length,
          &status);
    }
    cmd->target_offset = copy_target_offset;
    cmd->length = source_ref.length == IREE_HAL_WHOLE_BUFFER
                      ? iree_min(copy_source_length, copy_target_length)
                      : copy_source_length;
    cmd->copy_flags = (uint32_t)flags;
  }

  if (!iree_status_is_ok(status)) {
    command_buffer->stream_length = stream_start;
  }
  return status;
}

static iree_status_t iree_hal_remote_client_command_buffer_advise_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref,
    iree_hal_memory_advise_flags_t advise_flags, uint64_t arg0, uint64_t arg1) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  const iree_host_size_t stream_start = command_buffer->stream_length;
  void* ptr = NULL;
  iree_status_t status = iree_hal_remote_client_cb_append_command(
      command_buffer, IREE_HAL_REMOTE_CMD_BUFFER_ADVISE,
      sizeof(iree_hal_remote_buffer_advise_cmd_t), &ptr);

  if (iree_status_is_ok(status)) {
    iree_hal_remote_buffer_advise_cmd_t* cmd =
        (iree_hal_remote_buffer_advise_cmd_t*)ptr;
    uint64_t advise_offset = 0;
    uint64_t advise_length = 0;
    iree_hal_remote_client_cb_encode_buffer_ref(
        command_buffer, buffer_ref, &cmd->buffer_id, &cmd->buffer_slot,
        &advise_offset, &advise_length, &status);
    cmd->offset = advise_offset;
    cmd->length = advise_length;
    cmd->advise_flags = (uint32_t)advise_flags;
    cmd->argument0 = arg0;
    cmd->argument1 = arg1;
  }

  if (!iree_status_is_ok(status)) {
    command_buffer->stream_length = stream_start;
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Collective operations
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_remote_client_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not supported on remote device");
}

//===----------------------------------------------------------------------===//
// Dispatch
//===----------------------------------------------------------------------===//

static iree_status_t iree_hal_remote_client_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);

  iree_status_t status = iree_ok_status();
  if ((constants.data_length % sizeof(uint32_t)) != 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dispatch constants length %" PRIhsz
                              " is not a multiple of %zu bytes",
                              constants.data_length, sizeof(uint32_t));
  }
  if (iree_status_is_ok(status) &&
      constants.data_length / sizeof(uint32_t) > UINT16_MAX) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "dispatch constant count %" PRIhsz " exceeds wire limit %u",
        constants.data_length / sizeof(uint32_t), (unsigned)UINT16_MAX);
  }
  if (iree_status_is_ok(status) && bindings.count > UINT16_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "dispatch binding count %" PRIhsz
                              " exceeds wire limit %u",
                              bindings.count, (unsigned)UINT16_MAX);
  }
  if (!iree_status_is_ok(status)) return status;

  uint16_t constant_count =
      (uint16_t)(constants.data_length / sizeof(uint32_t));
  uint16_t binding_count = (uint16_t)bindings.count;

  const iree_host_size_t constants_size = constants.data_length;
  iree_host_size_t constants_offset = 0;
  iree_host_size_t bindings_offset = 0;
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_dispatch_cmd_t), &total_size,
      IREE_STRUCT_FIELD(constant_count, uint32_t, &constants_offset),
      IREE_STRUCT_FIELD_ALIGNED(binding_count, iree_hal_remote_binding_t, 8,
                                &bindings_offset),
      IREE_STRUCT_FIELD_ALIGNED(0, uint8_t, 8, NULL)));

  const iree_host_size_t stream_start = command_buffer->stream_length;
  void* ptr = NULL;
  status = iree_hal_remote_client_cb_append_command(
      command_buffer, IREE_HAL_REMOTE_CMD_DISPATCH, total_size, &ptr);

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_command_buffer_retain_resource(
        command_buffer, (iree_hal_resource_t*)executable);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_remote_dispatch_cmd_t* cmd = (iree_hal_remote_dispatch_cmd_t*)ptr;
    cmd->executable_id =
        iree_hal_remote_client_executable_resource_id(executable);
    cmd->function_value = function.value;
    memcpy(cmd->config.workgroup_size, config.workgroup_size,
           sizeof(config.workgroup_size));
    memcpy(cmd->config.workgroup_count, config.workgroup_count,
           sizeof(config.workgroup_count));
    cmd->config.dynamic_workgroup_local_memory =
        config.dynamic_workgroup_local_memory;
    uint64_t workgroup_count_offset = 0;
    uint64_t workgroup_count_length = 0;
    cmd->config.workgroup_count_buffer_id =
        iree_hal_remote_client_cb_resolve_buffer_ref(
            command_buffer, config.workgroup_count_ref, &workgroup_count_offset,
            &workgroup_count_length, &status);
    cmd->config.workgroup_count_offset = workgroup_count_offset;
    cmd->config.workgroup_count_length = workgroup_count_length;
    cmd->config.workgroup_count_buffer_slot =
        config.workgroup_count_ref.buffer_slot;
    cmd->constant_count = constant_count;
    cmd->binding_count = binding_count;
    cmd->dispatch_flags = flags;

    // Constants (padded to 8-byte alignment).
    uint8_t* cursor = (uint8_t*)cmd + constants_offset;
    if (constants_size > 0) {
      memcpy(cursor, constants.data, constants_size);
    }

    // Bindings.
    iree_hal_remote_binding_t* wire_bindings =
        (iree_hal_remote_binding_t*)((uint8_t*)cmd + bindings_offset);
    for (uint16_t i = 0; i < binding_count && iree_status_is_ok(status); ++i) {
      const iree_hal_buffer_ref_t* ref = &bindings.values[i];
      uint64_t binding_offset = 0;
      uint64_t binding_length = 0;
      wire_bindings[i].buffer_id = iree_hal_remote_client_cb_resolve_buffer_ref(
          command_buffer, *ref, &binding_offset, &binding_length, &status);
      wire_bindings[i].offset = binding_offset;
      wire_bindings[i].length = binding_length;
      wire_bindings[i].buffer_slot = ref->buffer_slot;
    }
  }

  if (!iree_status_is_ok(status)) {
    command_buffer->stream_length = stream_start;
  }
  return status;
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_hal_command_buffer_vtable_t
    iree_hal_remote_client_command_buffer_vtable = {
        .destroy = iree_hal_remote_client_command_buffer_destroy,
        .begin = iree_hal_remote_client_command_buffer_begin,
        .end = iree_hal_remote_client_command_buffer_end,
        .begin_debug_group =
            iree_hal_remote_client_command_buffer_begin_debug_group,
        .end_debug_group =
            iree_hal_remote_client_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_remote_client_command_buffer_execution_barrier,
        .signal_event = iree_hal_remote_client_command_buffer_signal_event,
        .reset_event = iree_hal_remote_client_command_buffer_reset_event,
        .wait_events = iree_hal_remote_client_command_buffer_wait_events,
        .fill_buffer = iree_hal_remote_client_command_buffer_fill_buffer,
        .update_buffer = iree_hal_remote_client_command_buffer_update_buffer,
        .copy_buffer = iree_hal_remote_client_command_buffer_copy_buffer,
        .advise_buffer = iree_hal_remote_client_command_buffer_advise_buffer,
        .collective = iree_hal_remote_client_command_buffer_collective,
        .dispatch = iree_hal_remote_client_command_buffer_dispatch,
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_hal_remote_client_command_buffer_create(
    iree_hal_remote_client_device_t* device,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_command_buffer = NULL;

  if (binding_capacity > UINT16_MAX) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "remote command buffer binding capacity %" PRIhsz
                            " exceeds wire limit %u",
                            binding_capacity, (unsigned)UINT16_MAX);
  }

  iree_host_size_t validation_size =
      iree_hal_command_buffer_validation_state_size(mode, binding_capacity);
  iree_host_size_t total_size = 0;
  iree_host_size_t validation_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_client_command_buffer_t), &total_size,
      IREE_STRUCT_FIELD(validation_size, uint8_t, &validation_offset));

  iree_hal_remote_client_command_buffer_t* command_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, total_size,
                                   (void**)&command_buffer);
  }

  if (iree_status_is_ok(status)) {
    memset(command_buffer, 0, total_size);

    // Allocate initial stream buffer.
    status = iree_allocator_malloc(host_allocator,
                                   IREE_HAL_REMOTE_CB_INITIAL_CAPACITY,
                                   (void**)&command_buffer->stream_data);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_command_buffer_initialize(
        iree_hal_device_allocator((iree_hal_device_t*)device), mode,
        command_categories, queue_affinity, binding_capacity,
        (uint8_t*)command_buffer + validation_offset,
        &iree_hal_remote_client_command_buffer_vtable, &command_buffer->base);
    command_buffer->host_allocator = host_allocator;
    command_buffer->device = device;
    command_buffer->stream_length = 0;
    command_buffer->stream_capacity = IREE_HAL_REMOTE_CB_INITIAL_CAPACITY;
    command_buffer->resource_id = 0;
    status = iree_hal_resource_set_allocate(&device->resource_set_block_pool,
                                            &command_buffer->resource_set);
    if (iree_status_is_ok(status)) {
      *out_command_buffer = &command_buffer->base;
    }
  }

  if (!iree_status_is_ok(status) && command_buffer) {
    iree_allocator_free(host_allocator, command_buffer->stream_data);
    iree_allocator_free(host_allocator, command_buffer);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

bool iree_hal_remote_client_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(command_buffer,
                              &iree_hal_remote_client_command_buffer_vtable);
}

iree_const_byte_span_t iree_hal_remote_client_command_buffer_stream(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);
  return iree_make_const_byte_span(command_buffer->stream_data,
                                   command_buffer->stream_length);
}

iree_hal_remote_resource_id_t iree_hal_remote_client_command_buffer_resource_id(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_remote_client_command_buffer_t* command_buffer =
      iree_hal_remote_client_command_buffer_cast(base_command_buffer);
  return command_buffer->resource_id;
}
