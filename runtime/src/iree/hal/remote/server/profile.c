// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/server/profile.h"

#include "iree/hal/remote/protocol/control.h"
#include "iree/hal/remote/protocol/profile.h"
#include "iree/hal/remote/server/bulk_profile_sender.h"
#include "iree/hal/remote/server/profile_relay.h"
#include "iree/hal/remote/server/server.h"

typedef struct iree_hal_remote_server_profile_parse_storage_t {
  // Backing allocation for parsed counter set and counter name arrays.
  void* storage;

  // Parsed counter set selections in |storage|.
  iree_hal_profile_counter_set_selection_t* counter_sets;

  // Parsed counter name string views in |storage|.
  iree_string_view_t* counter_names;
} iree_hal_remote_server_profile_parse_storage_t;

typedef struct iree_hal_remote_server_profile_sink_t {
  // Resource header for iree_hal_profile_sink_t lifetime management.
  iree_hal_resource_t resource;

  // Host allocator used for sink-owned allocations.
  iree_allocator_t host_allocator;

  // Server retained while callbacks may reference |session_slot|.
  iree_hal_remote_server_t* server;

  // Session slot associated with this relay sink.
  iree_hal_remote_server_session_t* session_slot;

  // Session ID expected in |session_slot| while callbacks are active.
  uint64_t session_id;

  // Last profile callback sequence assigned by this sink.
  iree_atomic_int64_t last_sequence;
} iree_hal_remote_server_profile_sink_t;

static iree_hal_remote_server_profile_sink_t*
iree_hal_remote_server_profile_sink_cast(iree_hal_profile_sink_t* base_sink) {
  return (iree_hal_remote_server_profile_sink_t*)base_sink;
}

static uint64_t iree_hal_remote_server_profile_sink_last_sequence(
    iree_hal_profile_sink_t* base_sink) {
  if (!base_sink) return 0;
  iree_hal_remote_server_profile_sink_t* sink =
      iree_hal_remote_server_profile_sink_cast(base_sink);
  return (uint64_t)iree_atomic_load(&sink->last_sequence,
                                    iree_memory_order_relaxed);
}

//===----------------------------------------------------------------------===//
// Profile sink relay
//===----------------------------------------------------------------------===//

static void iree_hal_remote_server_profile_sink_destroy(
    iree_hal_profile_sink_t* base_sink) {
  iree_hal_remote_server_profile_sink_t* sink =
      iree_hal_remote_server_profile_sink_cast(base_sink);
  iree_allocator_t host_allocator = sink->host_allocator;
  iree_hal_remote_server_release(sink->server);
  iree_allocator_free(host_allocator, sink);
}

static iree_status_t iree_hal_remote_server_profile_string_length(
    iree_string_view_t value, const char* field_name, uint16_t* out_length) {
  *out_length = 0;
  if (value.size > UINT16_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "profile %s length %" PRIhsz
                            " exceeds wire limit %u",
                            field_name, value.size, (unsigned)UINT16_MAX);
  }
  if (value.size > 0 && !value.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "profile %s has NULL storage", field_name);
  }
  *out_length = (uint16_t)value.size;
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_server_profile_sum_payloads(
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs,
    iree_host_size_t* out_payload_length) {
  *out_payload_length = 0;
  if (iovec_count > 0 && !iovecs) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "profile payload iovecs must be provided when "
                            "iovec_count is nonzero");
  }
  iree_host_size_t payload_length = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < iovec_count && iree_status_is_ok(status);
       ++i) {
    if (iovecs[i].data_length > 0 && !iovecs[i].data) {
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "profile payload iovec %" PRIhsz " has NULL storage", i);
    }
    if (iree_status_is_ok(status) &&
        !iree_host_size_checked_add(payload_length, iovecs[i].data_length,
                                    &payload_length)) {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "profile payload length overflow");
    }
  }
  if (iree_status_is_ok(status)) *out_payload_length = payload_length;
  return status;
}

static uint64_t iree_hal_remote_server_profile_sink_allocate_sequence(
    iree_hal_remote_server_profile_sink_t* sink) {
  return (uint64_t)(iree_atomic_fetch_add(&sink->last_sequence, 1,
                                          iree_memory_order_relaxed) +
                    1);
}

static iree_status_t iree_hal_remote_server_profile_allocate_payload(
    uint64_t sequence, iree_hal_remote_profile_callback_type_t callback_type,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code, iree_host_size_t iovec_count,
    const iree_const_byte_span_t* iovecs, iree_allocator_t host_allocator,
    iree_byte_span_t* out_payload) {
  *out_payload = iree_byte_span_empty();

  uint16_t content_type_length = 0;
  uint16_t name_length = 0;
  iree_host_size_t profile_payload_length = 0;
  iree_status_t status = iree_hal_remote_server_profile_string_length(
      metadata->content_type, "content type", &content_type_length);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_profile_string_length(metadata->name,
                                                          "name", &name_length);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_profile_sum_payloads(
        iovec_count, iovecs, &profile_payload_length);
  }

  iree_host_size_t total_size = 0;
  iree_host_size_t content_type_offset = 0;
  iree_host_size_t name_offset = 0;
  iree_host_size_t profile_payload_offset = 0;
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_profile_transfer_layout(
        content_type_length, name_length, profile_payload_length, &total_size,
        &content_type_offset, &name_offset, &profile_payload_offset);
  }

  uint8_t* payload_data = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, total_size,
                                   (void**)&payload_data);
  }
  if (iree_status_is_ok(status)) {
    memset(payload_data, 0, total_size);
    iree_hal_remote_profile_transfer_header_t* header =
        (iree_hal_remote_profile_transfer_header_t*)payload_data;
    header->sequence = sequence;
    header->session_id = metadata->session_id;
    header->stream_id = metadata->stream_id;
    header->event_id = metadata->event_id;
    header->executable_id = metadata->executable_id;
    header->command_buffer_id = metadata->command_buffer_id;
    header->chunk_flags = metadata->flags;
    header->dropped_record_count = metadata->dropped_record_count;
    header->payload_length = (uint64_t)profile_payload_length;
    header->session_status_code = session_status_code;
    header->physical_device_ordinal = metadata->physical_device_ordinal;
    header->queue_ordinal = metadata->queue_ordinal;
    header->content_type_length = content_type_length;
    header->name_length = name_length;
    header->callback_type = callback_type;
    if (content_type_length > 0) {
      memcpy(payload_data + content_type_offset, metadata->content_type.data,
             content_type_length);
    }
    if (name_length > 0) {
      memcpy(payload_data + name_offset, metadata->name.data, name_length);
    }

    uint8_t* profile_payload_data = payload_data + profile_payload_offset;
    for (iree_host_size_t i = 0; i < iovec_count; ++i) {
      if (iovecs[i].data_length > 0) {
        memcpy(profile_payload_data, iovecs[i].data, iovecs[i].data_length);
      }
      profile_payload_data += iovecs[i].data_length;
    }
    *out_payload = iree_make_byte_span(payload_data, total_size);
  } else {
    iree_allocator_free(host_allocator, payload_data);
  }
  return status;
}

static iree_status_t iree_hal_remote_server_profile_sink_submit(
    iree_hal_profile_sink_t* base_sink,
    iree_hal_remote_profile_callback_type_t callback_type,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code, iree_host_size_t iovec_count,
    const iree_const_byte_span_t* iovecs) {
  iree_hal_remote_server_profile_sink_t* sink =
      iree_hal_remote_server_profile_sink_cast(base_sink);
  const uint64_t sequence =
      iree_hal_remote_server_profile_sink_allocate_sequence(sink);

  iree_byte_span_t payload = iree_byte_span_empty();
  iree_status_t status = iree_hal_remote_server_profile_allocate_payload(
      sequence, callback_type, metadata, session_status_code, iovec_count,
      iovecs, sink->host_allocator, &payload);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_profile_submit_transfer(
        sink->session_slot, sink->session_id, base_sink, payload);
    payload = iree_byte_span_empty();
  }
  if (!iree_status_is_ok(status)) {
    iree_status_t observe_status =
        iree_hal_remote_server_profile_relay_observe_transfer(
            sink->session_slot, sequence, iree_status_clone(status));
    status = iree_status_join(status, observe_status);
  }
  return status;
}

static iree_status_t iree_hal_remote_server_profile_sink_begin(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata) {
  return iree_hal_remote_server_profile_sink_submit(
      base_sink, IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_BEGIN_SESSION, metadata,
      IREE_STATUS_OK, 0, NULL);
}

static iree_status_t iree_hal_remote_server_profile_sink_write(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs) {
  return iree_hal_remote_server_profile_sink_submit(
      base_sink, IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_WRITE_CHUNK, metadata,
      IREE_STATUS_OK, iovec_count, iovecs);
}

static iree_status_t iree_hal_remote_server_profile_sink_end(
    iree_hal_profile_sink_t* base_sink,
    const iree_hal_profile_chunk_metadata_t* metadata,
    iree_status_code_t session_status_code) {
  return iree_hal_remote_server_profile_sink_submit(
      base_sink, IREE_HAL_REMOTE_PROFILE_CALLBACK_TYPE_END_SESSION, metadata,
      session_status_code, 0, NULL);
}

static const iree_hal_profile_sink_vtable_t
    iree_hal_remote_server_profile_sink_vtable = {
        .destroy = iree_hal_remote_server_profile_sink_destroy,
        .begin_session = iree_hal_remote_server_profile_sink_begin,
        .write = iree_hal_remote_server_profile_sink_write,
        .end_session = iree_hal_remote_server_profile_sink_end,
};

static iree_status_t iree_hal_remote_server_profile_sink_create(
    iree_hal_remote_server_session_t* session_slot,
    iree_allocator_t host_allocator, iree_hal_profile_sink_t** out_sink) {
  *out_sink = NULL;

  iree_hal_remote_server_profile_sink_t* sink = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*sink), (void**)&sink);
  if (iree_status_is_ok(status)) {
    memset(sink, 0, sizeof(*sink));
    iree_hal_resource_initialize(&iree_hal_remote_server_profile_sink_vtable,
                                 &sink->resource);
    sink->host_allocator = host_allocator;
    sink->server = session_slot->server;
    iree_hal_remote_server_retain(sink->server);
    sink->session_slot = session_slot;
    sink->session_id = session_slot->session_id;
    iree_atomic_store(&sink->last_sequence, 0, iree_memory_order_relaxed);
    *out_sink = (iree_hal_profile_sink_t*)sink;
  }
  return status;
}

//===----------------------------------------------------------------------===//
// PROFILING_BEGIN parsing
//===----------------------------------------------------------------------===//

static void iree_hal_remote_server_profile_parse_storage_deinitialize(
    iree_hal_remote_server_profile_parse_storage_t* parse_storage,
    iree_allocator_t host_allocator) {
  iree_allocator_free(host_allocator, parse_storage->storage);
  memset(parse_storage, 0, sizeof(*parse_storage));
}

static iree_status_t iree_hal_remote_server_profile_allocate_parse_storage(
    uint32_t counter_set_count, uint32_t counter_name_count,
    iree_allocator_t host_allocator,
    iree_hal_remote_server_profile_parse_storage_t* parse_storage) {
  iree_host_size_t total_size = 0;
  iree_host_size_t counter_sets_offset = 0;
  iree_host_size_t counter_names_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      0, &total_size,
      IREE_STRUCT_FIELD(counter_set_count,
                        iree_hal_profile_counter_set_selection_t,
                        &counter_sets_offset),
      IREE_STRUCT_FIELD(counter_name_count, iree_string_view_t,
                        &counter_names_offset));
  if (iree_status_is_ok(status) && total_size > 0) {
    status = iree_allocator_malloc(host_allocator, total_size,
                                   &parse_storage->storage);
  }
  if (iree_status_is_ok(status) && total_size > 0) {
    memset(parse_storage->storage, 0, total_size);
    uint8_t* storage_data = (uint8_t*)parse_storage->storage;
    parse_storage->counter_sets =
        (iree_hal_profile_counter_set_selection_t*)(storage_data +
                                                    counter_sets_offset);
    parse_storage->counter_names =
        (iree_string_view_t*)(storage_data + counter_names_offset);
  }
  return status;
}

static iree_status_t iree_hal_remote_server_profile_parse_string(
    const uint8_t* body, iree_host_size_t body_length,
    iree_host_size_t* inout_offset, uint32_t string_length,
    iree_string_view_t* out_string) {
  iree_host_size_t string_offset = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      *inout_offset, inout_offset,
      IREE_STRUCT_FIELD_ALIGNED(string_length, char, 8, &string_offset));
  if (iree_status_is_ok(status) && *inout_offset > body_length) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "PROFILING_BEGIN string tail truncated");
  }
  if (iree_status_is_ok(status)) {
    *out_string =
        iree_make_string_view((const char*)body + string_offset, string_length);
  }
  return status;
}

static iree_status_t iree_hal_remote_server_profile_parse_begin_request(
    const uint8_t* body, iree_host_size_t body_length,
    iree_allocator_t host_allocator,
    iree_hal_device_profiling_options_t* out_options,
    iree_hal_remote_server_profile_parse_storage_t* parse_storage) {
  memset(out_options, 0, sizeof(*out_options));
  memset(parse_storage, 0, sizeof(*parse_storage));

  const iree_hal_remote_profiling_begin_request_t* request = NULL;
  iree_status_t status = iree_ok_status();
  if (body_length < sizeof(iree_hal_remote_profiling_begin_request_t)) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PROFILING_BEGIN body too small: %" PRIhsz " bytes", body_length);
  }
  if (iree_status_is_ok(status)) {
    request = (const iree_hal_remote_profiling_begin_request_t*)body;
    if (request->reserved[0] != 0 || request->reserved[1] != 0) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "PROFILING_BEGIN reserved fields are nonzero");
    }
  }
  if (iree_status_is_ok(status) &&
      !iree_any_bit_set(
          request->capture_filter_flags,
          IREE_HAL_PROFILE_CAPTURE_FILTER_FLAG_EXECUTABLE_FUNCTION_PATTERN) &&
      request->executable_function_pattern_length != 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "PROFILING_BEGIN executable function pattern "
                              "provided without filter bit");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_profile_allocate_parse_storage(
        request->counter_set_count, request->counter_name_count, host_allocator,
        parse_storage);
  }

  iree_string_view_t executable_function_pattern = iree_string_view_empty();
  iree_host_size_t read_offset =
      sizeof(iree_hal_remote_profiling_begin_request_t);
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_server_profile_parse_string(
        body, body_length, &read_offset,
        request->executable_function_pattern_length,
        &executable_function_pattern);
  }

  uint32_t parsed_counter_name_count = 0;
  for (uint32_t i = 0;
       iree_status_is_ok(status) && i < request->counter_set_count; ++i) {
    iree_host_size_t counter_set_offset = 0;
    iree_host_size_t counter_set_name_offset = 0;
    const iree_hal_remote_profile_counter_set_selection_t* wire_counter_set =
        NULL;
    status = IREE_STRUCT_LAYOUT(
        read_offset, &read_offset,
        IREE_STRUCT_FIELD_ALIGNED(
            1, iree_hal_remote_profile_counter_set_selection_t, 8,
            &counter_set_offset));
    if (iree_status_is_ok(status) && read_offset > body_length) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "PROFILING_BEGIN counter set tail truncated");
    }
    if (iree_status_is_ok(status)) {
      wire_counter_set =
          (const iree_hal_remote_profile_counter_set_selection_t*)(body +
                                                                   counter_set_offset);
      if (wire_counter_set->reserved != 0) {
        status = iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "PROFILING_BEGIN counter set reserved field is nonzero");
      }
    }
    if (iree_status_is_ok(status)) {
      status = IREE_STRUCT_LAYOUT(
          read_offset, &read_offset,
          IREE_STRUCT_FIELD_ALIGNED(wire_counter_set->name_length, char, 8,
                                    &counter_set_name_offset));
    }
    if (iree_status_is_ok(status) && read_offset > body_length) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "PROFILING_BEGIN counter set name truncated");
    }
    if (iree_status_is_ok(status) &&
        wire_counter_set->counter_name_count >
            request->counter_name_count - parsed_counter_name_count) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "PROFILING_BEGIN counter name count mismatch");
    }
    if (iree_status_is_ok(status)) {
      iree_hal_profile_counter_set_selection_t* counter_set =
          &parse_storage->counter_sets[i];
      counter_set->flags = wire_counter_set->flags;
      counter_set->name =
          iree_make_string_view((const char*)body + counter_set_name_offset,
                                wire_counter_set->name_length);
      counter_set->counter_name_count = wire_counter_set->counter_name_count;
      counter_set->counter_names =
          wire_counter_set->counter_name_count == 0
              ? NULL
              : &parse_storage->counter_names[parsed_counter_name_count];
    }

    for (uint32_t j = 0;
         iree_status_is_ok(status) && j < wire_counter_set->counter_name_count;
         ++j) {
      iree_host_size_t counter_name_offset = 0;
      iree_host_size_t counter_name_data_offset = 0;
      const iree_hal_remote_profile_counter_name_t* wire_counter_name = NULL;
      status = IREE_STRUCT_LAYOUT(
          read_offset, &read_offset,
          IREE_STRUCT_FIELD_ALIGNED(1, iree_hal_remote_profile_counter_name_t,
                                    8, &counter_name_offset));
      if (iree_status_is_ok(status) && read_offset > body_length) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "PROFILING_BEGIN counter name tail truncated");
      }
      if (iree_status_is_ok(status)) {
        wire_counter_name =
            (const iree_hal_remote_profile_counter_name_t*)(body +
                                                            counter_name_offset);
        if (wire_counter_name->reserved != 0) {
          status = iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "PROFILING_BEGIN counter name reserved field is nonzero");
        }
      }
      if (iree_status_is_ok(status)) {
        status = IREE_STRUCT_LAYOUT(
            read_offset, &read_offset,
            IREE_STRUCT_FIELD_ALIGNED(wire_counter_name->name_length, char, 8,
                                      &counter_name_data_offset));
      }
      if (iree_status_is_ok(status) && read_offset > body_length) {
        status =
            iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "PROFILING_BEGIN counter name string truncated");
      }
      if (iree_status_is_ok(status)) {
        parse_storage->counter_names[parsed_counter_name_count++] =
            iree_make_string_view((const char*)body + counter_name_data_offset,
                                  wire_counter_name->name_length);
      }
    }
  }
  if (iree_status_is_ok(status) &&
      parsed_counter_name_count != request->counter_name_count) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "PROFILING_BEGIN expected %u counter names but parsed %u",
        request->counter_name_count, parsed_counter_name_count);
  }
  if (iree_status_is_ok(status) && read_offset != body_length) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "PROFILING_BEGIN has %" PRIhsz " trailing bytes",
                              body_length - read_offset);
  }

  if (iree_status_is_ok(status)) {
    out_options->flags = (iree_hal_device_profiling_flags_t)request->flags;
    out_options->data_families =
        (iree_hal_device_profiling_data_families_t)request->data_families;
    out_options->capture_filter.flags =
        (iree_hal_profile_capture_filter_flags_t)request->capture_filter_flags;
    out_options->capture_filter.executable_function_pattern =
        executable_function_pattern;
    out_options->capture_filter.command_buffer_id = request->command_buffer_id;
    out_options->capture_filter.command_index = request->command_index;
    out_options->capture_filter.physical_device_ordinal =
        request->physical_device_ordinal;
    out_options->capture_filter.queue_ordinal = request->queue_ordinal;
    out_options->counter_set_count = request->counter_set_count;
    out_options->counter_sets = parse_storage->counter_sets;
  } else {
    iree_hal_remote_server_profile_parse_storage_deinitialize(parse_storage,
                                                              host_allocator);
  }
  return status;
}

static iree_status_t iree_hal_remote_server_profile_parse_empty_request(
    const char* message_name, const uint8_t* body,
    iree_host_size_t body_length) {
  if (body_length != sizeof(iree_hal_remote_profiling_flush_request_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%s body size mismatch: expected %" PRIhsz " bytes, got %" PRIhsz,
        message_name, sizeof(iree_hal_remote_profiling_flush_request_t),
        body_length);
  }
  const iree_hal_remote_profiling_flush_request_t* request =
      (const iree_hal_remote_profiling_flush_request_t*)body;
  if (request->flags != 0 || request->reserved != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s reserved fields are nonzero", message_name);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Control handlers
//===----------------------------------------------------------------------===//

static bool iree_hal_remote_server_profile_options_request_data(
    const iree_hal_device_profiling_options_t* options) {
  return options->data_families != IREE_HAL_DEVICE_PROFILING_DATA_NONE ||
         iree_hal_device_profiling_options_requests_lightweight_statistics(
             options);
}

iree_status_t iree_hal_remote_server_handle_profiling_begin(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  iree_allocator_t host_allocator = session_slot->server->host_allocator;
  iree_hal_device_profiling_options_t options;
  iree_hal_remote_server_profile_parse_storage_t parse_storage;
  iree_status_t status = iree_hal_remote_server_profile_parse_begin_request(
      body, body_length, host_allocator, &options, &parse_storage);
  if (!iree_status_is_ok(status)) {
    return iree_hal_remote_server_session_send_error_response(session_slot,
                                                              envelope, status);
  }

  const bool data_requested =
      iree_hal_remote_server_profile_options_request_data(&options);
  iree_hal_profile_sink_t* profile_sink = NULL;
  if (data_requested) {
    status = iree_hal_remote_server_profile_sink_create(
        session_slot, host_allocator, &profile_sink);
  }
  if (iree_status_is_ok(status) && data_requested) {
    options.sink = profile_sink;
    status = iree_hal_remote_server_profile_relay_prepare_begin(session_slot,
                                                                profile_sink);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_profiling_begin(session_slot->server->devices[0],
                                             &options);
  }

  uint64_t target_sequence =
      iree_hal_remote_server_profile_sink_last_sequence(profile_sink);
  if (!iree_status_is_ok(status) && data_requested) {
    iree_hal_profile_sink_t* detached_sink =
        iree_hal_remote_server_profile_relay_detach_active_sink(session_slot,
                                                                profile_sink);
    iree_hal_profile_sink_release(detached_sink);
  }

  iree_status_t send_status =
      iree_hal_remote_server_profile_relay_defer_response(
          session_slot, envelope, target_sequence, status);
  iree_hal_remote_server_profile_parse_storage_deinitialize(&parse_storage,
                                                            host_allocator);
  iree_hal_profile_sink_release(profile_sink);
  return send_status;
}

iree_status_t iree_hal_remote_server_handle_profiling_flush(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  iree_status_t status = iree_hal_remote_server_profile_parse_empty_request(
      "PROFILING_FLUSH", body, body_length);
  iree_hal_profile_sink_t* profile_sink = NULL;
  if (iree_status_is_ok(status)) {
    profile_sink =
        iree_hal_remote_server_profile_relay_retain_active_sink(session_slot);
    if (!profile_sink) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "remote profiling session is not active");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_profiling_flush(session_slot->server->devices[0]);
  }

  uint64_t target_sequence =
      iree_hal_remote_server_profile_sink_last_sequence(profile_sink);
  iree_status_t send_status =
      iree_hal_remote_server_profile_relay_defer_response(
          session_slot, envelope, target_sequence, status);
  iree_hal_profile_sink_release(profile_sink);
  return send_status;
}

iree_status_t iree_hal_remote_server_handle_profiling_end(
    iree_hal_remote_server_session_t* session_slot,
    const iree_hal_remote_control_envelope_t* envelope, const uint8_t* body,
    iree_host_size_t body_length) {
  iree_status_t status = iree_hal_remote_server_profile_parse_empty_request(
      "PROFILING_END", body, body_length);
  iree_hal_profile_sink_t* profile_sink = NULL;
  if (iree_status_is_ok(status)) {
    profile_sink =
        iree_hal_remote_server_profile_relay_retain_active_sink(session_slot);
    if (!profile_sink) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "remote profiling session is not active");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_profiling_end(session_slot->server->devices[0]);
  }

  const uint64_t target_sequence =
      iree_hal_remote_server_profile_sink_last_sequence(profile_sink);
  if (profile_sink) {
    iree_hal_profile_sink_t* detached_sink =
        iree_hal_remote_server_profile_relay_detach_active_sink(session_slot,
                                                                profile_sink);
    iree_hal_profile_sink_release(detached_sink);
  }

  iree_status_t send_status =
      iree_hal_remote_server_profile_relay_defer_response(
          session_slot, envelope, target_sequence, status);
  iree_hal_profile_sink_release(profile_sink);
  return send_status;
}
