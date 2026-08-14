// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/remote/client/executable.h"

#include "iree/base/threading/mutex.h"
#include "iree/hal/remote/client/buffer.h"
#include "iree/hal/remote/client/device.h"
#include "iree/hal/remote/protocol/control.h"

static const iree_hal_executable_vtable_t
    iree_hal_remote_client_executable_vtable;

typedef struct iree_hal_remote_client_executable_global_buffer_t {
  // Next affinity-specific buffer cached for the executable global.
  struct iree_hal_remote_client_executable_global_buffer_t* next;

  // Queue affinity used to request |buffer| from the server executable.
  iree_hal_queue_affinity_t queue_affinity;

  // Remote buffer alias borrowed by callers of executable_global_buffer.
  iree_hal_buffer_t* buffer;
} iree_hal_remote_client_executable_global_buffer_t;

typedef struct iree_hal_remote_client_executable_global_t {
  // Next global entry in the executable-owned cache.
  struct iree_hal_remote_client_executable_global_t* next;

  // Executable-local global name storage.
  iree_string_view_t name;

  // Total byte length of every buffer materialized for the global.
  iree_device_size_t byte_length;

  // Affinity-specific remote buffers owned by the executable.
  iree_hal_remote_client_executable_global_buffer_t* buffer_list;
} iree_hal_remote_client_executable_global_t;

typedef struct iree_hal_remote_client_executable_t {
  // Base HAL resource header.
  iree_hal_resource_t resource;

  // Host allocator used for executable-owned metadata storage.
  iree_allocator_t host_allocator;

  // Remote device used for control RPCs and resource release.
  iree_hal_remote_client_device_t* device;

  // Server-side executable resource ID.
  iree_hal_remote_resource_id_t resource_id;

  // Guards |global_list|.
  iree_slim_mutex_t global_mutex;

  // Cached globals resolved by name and their affinity-specific buffer aliases.
  iree_hal_remote_client_executable_global_t* global_list;

  // Number of dispatchable functions reported by EXECUTABLE_UPLOAD.
  iree_host_size_t function_count;

  // Cached immutable function metadata indexed by dense function index.
  iree_hal_executable_function_info_t* function_infos;

  // Per-function cached parameter arrays indexed by dense function index.
  iree_hal_executable_function_parameter_t** function_parameters;

  // Base allocations for each cached parameter array and its name storage.
  void** function_parameter_storage;
} iree_hal_remote_client_executable_t;

static iree_status_t iree_hal_remote_client_executable_query_function_info(
    iree_hal_remote_client_executable_t* executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  memset(out_info, 0, sizeof(*out_info));

  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_executable_query_function_request_t request;
  } request_message;
  memset(&request_message, 0, sizeof(request_message));
  request_message.envelope.message_type =
      IREE_HAL_REMOTE_CONTROL_EXECUTABLE_QUERY_FUNCTION;
  request_message.request.executable_id = executable->resource_id;
  request_message.request.function_value = function.value;

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  iree_status_t status = iree_hal_remote_client_device_control_rpc(
      executable->device,
      iree_make_const_byte_span(&request_message, sizeof(request_message)),
      &response_payload, &response_lease);

  const iree_hal_remote_executable_query_function_response_t* response = NULL;
  iree_host_size_t name_offset = 0;
  if (iree_status_is_ok(status)) {
    if (response_payload.data_length <
        sizeof(iree_hal_remote_executable_query_function_response_t)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "EXECUTABLE_QUERY_FUNCTION response too short: "
          "%" PRIhsz " < %" PRIhsz,
          response_payload.data_length,
          sizeof(iree_hal_remote_executable_query_function_response_t));
    } else {
      response = (const iree_hal_remote_executable_query_function_response_t*)
                     response_payload.data;
      const uint64_t allowed_flags =
          IREE_HAL_EXECUTABLE_FUNCTION_FLAG_SEQUENTIAL |
          IREE_HAL_EXECUTABLE_FUNCTION_FLAG_WORKGROUP_SIZE_DYNAMIC;
      const uint32_t allowed_resource_usage_flags =
          IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_ALL;
      if (response->reserved0 != 0 || response->reserved1 != 0 ||
          response->occupancy_reserved != 0) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "EXECUTABLE_QUERY_FUNCTION response reserved field is nonzero");
      } else if ((response->flags & ~allowed_flags) != 0) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "EXECUTABLE_QUERY_FUNCTION response has "
                                  "unknown flags 0x%016" PRIx64,
                                  response->flags);
      } else if ((response->resource_usage_provided_flags &
                  ~allowed_resource_usage_flags) != 0) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "EXECUTABLE_QUERY_FUNCTION response has unknown resource usage "
            "flags 0x%08" PRIx32,
            response->resource_usage_provided_flags &
                ~allowed_resource_usage_flags);
      } else if (
          (!iree_all_bits_set(
               response->resource_usage_provided_flags,
               IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_WORKGROUP_LOCAL_MEMORY) &&
           response->fixed_workgroup_local_memory_size != 0) ||
          (!iree_all_bits_set(
               response->resource_usage_provided_flags,
               IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_PRIVATE_MEMORY) &&
           response->fixed_private_memory_size != 0) ||
          (!iree_all_bits_set(
               response->resource_usage_provided_flags,
               IREE_HAL_EXECUTABLE_FUNCTION_RESOURCE_FLAG_INVOCATION_REGISTERS) &&
           response->invocation_register_count != 0)) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "EXECUTABLE_QUERY_FUNCTION response populates unavailable "
            "resource usage fields");
      } else {
        iree_host_size_t expected_length = 0;
        status = IREE_STRUCT_LAYOUT(
            sizeof(iree_hal_remote_executable_query_function_response_t),
            &expected_length,
            IREE_STRUCT_FIELD(response->name_length, char, &name_offset));
        if (iree_status_is_ok(status) &&
            response_payload.data_length < expected_length) {
          status =
              iree_make_status(IREE_STATUS_INTERNAL,
                               "EXECUTABLE_QUERY_FUNCTION response truncated: "
                               "%" PRIhsz " < %" PRIhsz,
                               response_payload.data_length, expected_length);
        }
      }
    }
  }

  char* name_storage = NULL;
  if (iree_status_is_ok(status) && response->name_length > 0) {
    status = iree_allocator_clone(
        executable->host_allocator,
        iree_make_const_byte_span(response_payload.data + name_offset,
                                  response->name_length),
        (void**)&name_storage);
  }

  if (iree_status_is_ok(status)) {
    out_info->name = iree_make_string_view(
        name_storage, (iree_host_size_t)response->name_length);
    out_info->flags = (iree_hal_executable_function_flags_t)response->flags;
    out_info->constant_byte_length = response->constant_byte_length;
    out_info->binding_count = response->binding_count;
    out_info->parameter_count = response->parameter_count;
    out_info->maximum_workgroup_invocations =
        response->maximum_workgroup_invocations;
    memcpy(out_info->workgroup_size, response->workgroup_size,
           sizeof(out_info->workgroup_size));
    out_info->resource_usage.provided_flags =
        (iree_hal_executable_function_resource_flags_t)
            response->resource_usage_provided_flags;
    out_info->resource_usage.fixed_workgroup_local_memory_size =
        response->fixed_workgroup_local_memory_size;
    out_info->resource_usage.fixed_private_memory_size =
        response->fixed_private_memory_size;
    out_info->resource_usage.invocation_register_count =
        response->invocation_register_count;
    out_info->occupancy_info.reserved = response->occupancy_reserved;
  }

  iree_async_buffer_lease_release(&response_lease);
  return status;
}

static iree_status_t iree_hal_remote_client_executable_query_parameters(
    iree_hal_remote_client_executable_t* executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    void** out_parameter_storage,
    iree_hal_executable_function_parameter_t** out_parameters) {
  *out_parameter_storage = NULL;
  *out_parameters = NULL;

  iree_status_t status = iree_ok_status();
  if (capacity > UINT16_MAX) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "function parameter count %" PRIhsz
                              " exceeds wire limit %u",
                              capacity, (unsigned)UINT16_MAX);
  }

  struct {
    iree_hal_remote_control_envelope_t envelope;
    iree_hal_remote_executable_query_parameters_request_t request;
  } request_message;
  if (iree_status_is_ok(status)) {
    memset(&request_message, 0, sizeof(request_message));
    request_message.envelope.message_type =
        IREE_HAL_REMOTE_CONTROL_EXECUTABLE_QUERY_PARAMETERS;
    request_message.request.executable_id = executable->resource_id;
    request_message.request.function_value = function.value;
    request_message.request.capacity = (uint16_t)capacity;
  }

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_control_rpc(
        executable->device,
        iree_make_const_byte_span(&request_message, sizeof(request_message)),
        &response_payload, &response_lease);
  }

  const iree_hal_remote_executable_query_parameters_response_t* response = NULL;
  iree_host_size_t parameters_offset = 0;
  iree_host_size_t names_offset = 0;
  if (iree_status_is_ok(status)) {
    if (response_payload.data_length <
        sizeof(iree_hal_remote_executable_query_parameters_response_t)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "EXECUTABLE_QUERY_PARAMETERS response too short: "
          "%" PRIhsz " < %" PRIhsz,
          response_payload.data_length,
          sizeof(iree_hal_remote_executable_query_parameters_response_t));
    } else {
      response = (const iree_hal_remote_executable_query_parameters_response_t*)
                     response_payload.data;
      if (response->reserved != 0) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "EXECUTABLE_QUERY_PARAMETERS response reserved field is nonzero");
      } else {
        iree_host_size_t expected_length = 0;
        status = IREE_STRUCT_LAYOUT(
            sizeof(iree_hal_remote_executable_query_parameters_response_t),
            &expected_length,
            IREE_STRUCT_FIELD(response->parameter_count,
                              iree_hal_remote_executable_function_parameter_t,
                              &parameters_offset),
            IREE_STRUCT_FIELD(response->name_data_length, char, &names_offset));
        if (iree_status_is_ok(status) &&
            response_payload.data_length < expected_length) {
          status = iree_make_status(
              IREE_STATUS_INTERNAL,
              "EXECUTABLE_QUERY_PARAMETERS response truncated: "
              "%" PRIhsz " < %" PRIhsz,
              response_payload.data_length, expected_length);
        }
        if (iree_status_is_ok(status) &&
            response->parameter_count != capacity) {
          status = iree_make_status(
              IREE_STATUS_INTERNAL,
              "EXECUTABLE_QUERY_PARAMETERS returned %" PRIu16
              " parameters but function metadata declared %" PRIhsz,
              response->parameter_count, capacity);
        }
      }
    }
  }

  void* parameter_storage = NULL;
  iree_hal_executable_function_parameter_t* parameters = NULL;
  iree_host_size_t parameter_count = 0;
  iree_host_size_t names_storage_offset = 0;
  if (iree_status_is_ok(status)) {
    parameter_count = response->parameter_count;
    if (parameter_count > 0) {
      iree_host_size_t allocation_size = 0;
      iree_host_size_t parameter_storage_offset = 0;
      status = IREE_STRUCT_LAYOUT(
          0, &allocation_size,
          IREE_STRUCT_FIELD(parameter_count,
                            iree_hal_executable_function_parameter_t,
                            &parameter_storage_offset),
          IREE_STRUCT_FIELD(response->name_data_length, char,
                            &names_storage_offset));
      if (iree_status_is_ok(status)) {
        status = iree_allocator_malloc(executable->host_allocator,
                                       allocation_size, &parameter_storage);
      }
      if (iree_status_is_ok(status)) {
        memset(parameter_storage, 0, allocation_size);
        parameters =
            (iree_hal_executable_function_parameter_t*)((char*)
                                                            parameter_storage +
                                                        parameter_storage_offset);
      }
    }
  }

  if (iree_status_is_ok(status) && parameter_count > 0) {
    const iree_hal_remote_executable_function_parameter_t* wire_parameters =
        (const iree_hal_remote_executable_function_parameter_t*)(response_payload
                                                                     .data +
                                                                 parameters_offset);
    const char* response_names =
        (const char*)response_payload.data + names_offset;
    char* name_storage = (char*)parameter_storage + names_storage_offset;
    iree_host_size_t name_offset = 0;
    const iree_hal_executable_function_parameter_flags_t known_parameter_flags =
        IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET;
    for (iree_host_size_t i = 0;
         i < parameter_count && iree_status_is_ok(status); ++i) {
      iree_host_size_t next_name_offset = 0;
      if (wire_parameters[i].reserved[0] != 0 ||
          wire_parameters[i].reserved[1] != 0 ||
          wire_parameters[i].reserved[2] != 0 ||
          wire_parameters[i].reserved[3] != 0 ||
          wire_parameters[i].reserved[4] != 0) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "EXECUTABLE_QUERY_PARAMETERS parameter reserved field is nonzero");
      } else if (wire_parameters[i].type >
                 IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_TYPE_BUFFER_PTR) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "EXECUTABLE_QUERY_PARAMETERS parameter has invalid type %" PRIu8,
            wire_parameters[i].type);
      } else if (iree_any_bit_set(wire_parameters[i].flags,
                                  ~known_parameter_flags)) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "EXECUTABLE_QUERY_PARAMETERS parameter has "
                                  "unknown flags 0x%04" PRIx16,
                                  wire_parameters[i].flags);
      } else if (!iree_host_size_checked_add(name_offset,
                                             wire_parameters[i].name_length,
                                             &next_name_offset) ||
                 next_name_offset > response->name_data_length) {
        status =
            iree_make_status(IREE_STATUS_INTERNAL,
                             "EXECUTABLE_QUERY_PARAMETERS name data truncated");
      } else {
        parameters[i].type =
            (iree_hal_executable_function_parameter_type_t)wire_parameters[i]
                .type;
        parameters[i].size = wire_parameters[i].size;
        parameters[i].flags =
            (iree_hal_executable_function_parameter_flags_t)wire_parameters[i]
                .flags;
        parameters[i].offset = wire_parameters[i].offset;
        parameters[i].native_abi_offset = wire_parameters[i].native_abi_offset;
        if (wire_parameters[i].name_length > 0) {
          memcpy(name_storage + name_offset, response_names + name_offset,
                 wire_parameters[i].name_length);
        }
        parameters[i].name = iree_make_string_view(
            name_storage + name_offset, wire_parameters[i].name_length);
        name_offset = next_name_offset;
      }
    }
    if (iree_status_is_ok(status) &&
        name_offset != response->name_data_length) {
      status =
          iree_make_status(IREE_STATUS_INTERNAL,
                           "EXECUTABLE_QUERY_PARAMETERS response has %" PRIu32
                           " name bytes but consumed %" PRIhsz,
                           response->name_data_length, name_offset);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_parameter_storage = parameter_storage;
    *out_parameters = parameters;
  } else {
    iree_allocator_free(executable->host_allocator, parameter_storage);
  }

  iree_async_buffer_lease_release(&response_lease);
  return status;
}

static iree_status_t iree_hal_remote_client_executable_initialize_metadata(
    iree_hal_remote_client_executable_t* executable) {
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < executable->function_count && iree_status_is_ok(status); ++i) {
    iree_hal_executable_function_info_t info;
    status = iree_hal_remote_client_executable_query_function_info(
        executable, iree_hal_executable_function_from_index((uint32_t)i),
        &info);
    if (iree_status_is_ok(status)) {
      executable->function_infos[i] = info;
    }
    if (iree_status_is_ok(status) && info.parameter_count > 0) {
      void* parameter_storage = NULL;
      iree_hal_executable_function_parameter_t* parameters = NULL;
      status = iree_hal_remote_client_executable_query_parameters(
          executable, iree_hal_executable_function_from_index((uint32_t)i),
          info.parameter_count, &parameter_storage, &parameters);
      if (iree_status_is_ok(status)) {
        executable->function_parameter_storage[i] = parameter_storage;
        executable->function_parameters[i] = parameters;
      }
    }
  }
  return status;
}

static void iree_hal_remote_client_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (executable->resource_id != 0) {
    iree_status_ignore(iree_hal_remote_client_device_release_resource(
        executable->device, executable->resource_id));
  }

  iree_allocator_t host_allocator = executable->host_allocator;
  for (iree_host_size_t i = 0; i < executable->function_count; ++i) {
    iree_allocator_free(host_allocator,
                        (void*)executable->function_infos[i].name.data);
    iree_allocator_free(host_allocator,
                        executable->function_parameter_storage[i]);
  }
  iree_hal_remote_client_executable_global_t* global = executable->global_list;
  while (global) {
    iree_hal_remote_client_executable_global_t* next_global = global->next;
    iree_hal_remote_client_executable_global_buffer_t* global_buffer =
        global->buffer_list;
    while (global_buffer) {
      iree_hal_remote_client_executable_global_buffer_t* next_global_buffer =
          global_buffer->next;
      iree_hal_buffer_release(global_buffer->buffer);
      iree_allocator_free(host_allocator, global_buffer);
      global_buffer = next_global_buffer;
    }
    iree_allocator_free(host_allocator, global);
    global = next_global;
  }
  iree_slim_mutex_deinitialize(&executable->global_mutex);
  iree_allocator_free(host_allocator, executable);
  IREE_TRACE_ZONE_END(z0);
}

static iree_host_size_t iree_hal_remote_client_executable_function_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  return executable->function_count;
}

static iree_status_t iree_hal_remote_client_executable_function_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "function value %" PRIu64 " is outside dense function count %" PRIhsz,
        (uint64_t)function.value, executable->function_count);
  }
  const uint32_t function_index = iree_hal_executable_function_index(function);
  *out_info = executable->function_infos[function_index];
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_executable_function_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  if (!iree_hal_executable_function_is_index_in_range(
          function, executable->function_count)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "function value %" PRIu64 " is outside dense function count %" PRIhsz,
        (uint64_t)function.value, executable->function_count);
  }
  const uint32_t function_index = iree_hal_executable_function_index(function);
  iree_host_size_t parameter_count =
      executable->function_infos[function_index].parameter_count;
  iree_host_size_t copy_count = iree_min(capacity, parameter_count);
  if (copy_count > 0) {
    memcpy(out_parameters, executable->function_parameters[function_index],
           copy_count * sizeof(out_parameters[0]));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_executable_lookup_function_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  *out_function = iree_hal_executable_function_invalid();
  for (iree_host_size_t i = 0; i < executable->function_count; ++i) {
    if (iree_string_view_equal(executable->function_infos[i].name, name)) {
      *out_function = iree_hal_executable_function_from_index((uint32_t)i);
      return iree_ok_status();
    }
  }
  return iree_make_status(IREE_STATUS_NOT_FOUND,
                          "function '%.*s' not found in executable",
                          (int)name.size, name.data);
}

static iree_hal_remote_client_executable_global_t*
iree_hal_remote_client_executable_find_global_locked(
    iree_hal_remote_client_executable_t* executable, iree_string_view_t name) {
  for (iree_hal_remote_client_executable_global_t* global =
           executable->global_list;
       global != NULL; global = global->next) {
    if (iree_string_view_equal(global->name, name)) return global;
  }
  return NULL;
}

static iree_hal_remote_client_executable_global_t*
iree_hal_remote_client_executable_global_from_handle_locked(
    iree_hal_remote_client_executable_t* executable,
    iree_hal_executable_global_t global) {
  iree_hal_remote_client_executable_global_t* expected_global =
      (iree_hal_remote_client_executable_global_t*)(uintptr_t)global.value;
  for (iree_hal_remote_client_executable_global_t* current_global =
           executable->global_list;
       current_global != NULL; current_global = current_global->next) {
    if (current_global == expected_global) return current_global;
  }
  return NULL;
}

static iree_hal_remote_client_executable_global_buffer_t*
iree_hal_remote_client_executable_find_global_buffer_locked(
    iree_hal_remote_client_executable_global_t* global,
    iree_hal_queue_affinity_t queue_affinity) {
  for (iree_hal_remote_client_executable_global_buffer_t* global_buffer =
           global->buffer_list;
       global_buffer != NULL; global_buffer = global_buffer->next) {
    if (global_buffer->queue_affinity == queue_affinity) return global_buffer;
  }
  return NULL;
}

static iree_status_t iree_hal_remote_client_executable_global_allocate(
    iree_hal_remote_client_executable_t* executable, iree_string_view_t name,
    iree_device_size_t byte_length,
    iree_hal_remote_client_executable_global_t** out_global) {
  *out_global = NULL;
  iree_host_size_t name_offset = 0;
  iree_host_size_t total_size = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_remote_client_executable_global_t), &total_size,
      IREE_STRUCT_FIELD(name.size, char, &name_offset));
  iree_hal_remote_client_executable_global_t* global = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(executable->host_allocator, total_size,
                                   (void**)&global);
  }
  if (iree_status_is_ok(status)) {
    memset(global, 0, total_size);
    char* name_storage = (char*)global + name_offset;
    if (name.size > 0) memcpy(name_storage, name.data, name.size);
    global->name = iree_make_string_view(name_storage, name.size);
    global->byte_length = byte_length;
    *out_global = global;
  }
  return status;
}

static void iree_hal_remote_client_executable_global_free(
    iree_hal_remote_client_executable_t* executable,
    iree_hal_remote_client_executable_global_t* global) {
  if (!global) return;
  iree_hal_remote_client_executable_global_buffer_t* global_buffer =
      global->buffer_list;
  while (global_buffer) {
    iree_hal_remote_client_executable_global_buffer_t* next_global_buffer =
        global_buffer->next;
    iree_hal_buffer_release(global_buffer->buffer);
    iree_allocator_free(executable->host_allocator, global_buffer);
    global_buffer = next_global_buffer;
  }
  iree_allocator_free(executable->host_allocator, global);
}

static iree_status_t iree_hal_remote_client_executable_query_global_by_name(
    iree_hal_remote_client_executable_t* executable, iree_string_view_t name,
    bool* out_found, iree_device_size_t* out_byte_length) {
  *out_found = false;
  *out_byte_length = 0;

  iree_status_t status = iree_ok_status();
  if (name.size > UINT16_MAX) {
    status =
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "global name length %" PRIhsz " exceeds wire limit %u",
                         name.size, (unsigned)UINT16_MAX);
  }

  iree_host_size_t message_length = 0;
  iree_host_size_t name_offset = 0;
  const iree_host_size_t header_size =
      sizeof(iree_hal_remote_control_envelope_t) +
      sizeof(iree_hal_remote_executable_lookup_global_request_t);
  if (iree_status_is_ok(status)) {
    status =
        IREE_STRUCT_LAYOUT(header_size, &message_length,
                           IREE_STRUCT_FIELD(name.size, char, &name_offset));
  }

  uint8_t* message = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(executable->host_allocator, message_length,
                                   (void**)&message);
  }
  if (iree_status_is_ok(status)) {
    memset(message, 0, message_length);
    iree_hal_remote_control_envelope_t* envelope =
        (iree_hal_remote_control_envelope_t*)message;
    envelope->message_type = IREE_HAL_REMOTE_CONTROL_EXECUTABLE_LOOKUP_GLOBAL;
    iree_hal_remote_executable_lookup_global_request_t* request =
        (iree_hal_remote_executable_lookup_global_request_t*)(envelope + 1);
    request->executable_id = executable->resource_id;
    request->name_length = (uint16_t)name.size;
    if (name.size > 0) {
      memcpy(message + name_offset, name.data, name.size);
    }
  }

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_control_rpc(
        executable->device, iree_make_const_byte_span(message, message_length),
        &response_payload, &response_lease);
  }

  iree_device_size_t byte_length = 0;
  bool found = false;
  if (iree_status_is_ok(status)) {
    if (response_payload.data_length <
        sizeof(iree_hal_remote_executable_lookup_global_response_t)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "EXECUTABLE_LOOKUP_GLOBAL response too short: "
          "%" PRIhsz " < %" PRIhsz,
          response_payload.data_length,
          sizeof(iree_hal_remote_executable_lookup_global_response_t));
    } else {
      const iree_hal_remote_executable_lookup_global_response_t* response =
          (const iree_hal_remote_executable_lookup_global_response_t*)
              response_payload.data;
      if (response->reserved != 0) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "EXECUTABLE_LOOKUP_GLOBAL response reserved field is nonzero");
      } else if (response->found > 1) {
        status = iree_make_status(IREE_STATUS_INTERNAL,
                                  "EXECUTABLE_LOOKUP_GLOBAL response has "
                                  "invalid found value %" PRIu32,
                                  response->found);
      } else if (!response->found && response->byte_length != 0) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "EXECUTABLE_LOOKUP_GLOBAL absent response contains global data");
      } else if (response->byte_length > IREE_DEVICE_SIZE_MAX) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "EXECUTABLE_LOOKUP_GLOBAL byte length exceeds device capacity");
      } else {
        found = response->found != 0;
        byte_length = (iree_device_size_t)response->byte_length;
      }
    }
  }

  iree_async_buffer_lease_release(&response_lease);
  iree_allocator_free(executable->host_allocator, message);

  if (iree_status_is_ok(status)) {
    *out_found = found;
    *out_byte_length = byte_length;
  }
  return status;
}

static iree_status_t iree_hal_remote_client_executable_query_global_buffer(
    iree_hal_remote_client_executable_t* executable, iree_string_view_t name,
    iree_hal_queue_affinity_t queue_affinity,
    iree_device_size_t expected_byte_length, iree_hal_buffer_t** out_buffer) {
  *out_buffer = NULL;

  iree_status_t status = iree_ok_status();
  if (name.size > UINT16_MAX) {
    status =
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "global name length %" PRIhsz " exceeds wire limit %u",
                         name.size, (unsigned)UINT16_MAX);
  }

  const iree_host_size_t header_size =
      sizeof(iree_hal_remote_control_envelope_t) +
      sizeof(iree_hal_remote_executable_global_buffer_request_t);
  iree_host_size_t name_offset = 0;
  iree_host_size_t message_length = 0;
  if (iree_status_is_ok(status)) {
    status =
        IREE_STRUCT_LAYOUT(header_size, &message_length,
                           IREE_STRUCT_FIELD(name.size, char, &name_offset));
  }

  uint8_t* message = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(executable->host_allocator, message_length,
                                   (void**)&message);
  }
  if (iree_status_is_ok(status)) {
    memset(message, 0, message_length);
    iree_hal_remote_control_envelope_t* envelope =
        (iree_hal_remote_control_envelope_t*)message;
    envelope->message_type = IREE_HAL_REMOTE_CONTROL_EXECUTABLE_GLOBAL_BUFFER;
    iree_hal_remote_executable_global_buffer_request_t* request =
        (iree_hal_remote_executable_global_buffer_request_t*)(envelope + 1);
    request->executable_id = executable->resource_id;
    request->queue_affinity = (uint64_t)queue_affinity;
    request->name_length = (uint16_t)name.size;
    if (name.size > 0) {
      memcpy(message + name_offset, name.data, name.size);
    }
  }

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_control_rpc(
        executable->device, iree_make_const_byte_span(message, message_length),
        &response_payload, &response_lease);
  }

  iree_hal_remote_resource_id_t resolved_id = 0;
  iree_hal_buffer_params_t params = {0};
  iree_device_size_t byte_length = 0;
  iree_hal_buffer_placement_flags_t placement_flags =
      IREE_HAL_BUFFER_PLACEMENT_FLAG_NONE;
  if (iree_status_is_ok(status)) {
    if (response_payload.data_length <
        sizeof(iree_hal_remote_executable_global_buffer_response_t)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "EXECUTABLE_GLOBAL_BUFFER response too short: "
          "%" PRIhsz " < %" PRIhsz,
          response_payload.data_length,
          sizeof(iree_hal_remote_executable_global_buffer_response_t));
    } else {
      const iree_hal_remote_executable_global_buffer_response_t* response =
          (const iree_hal_remote_executable_global_buffer_response_t*)
              response_payload.data;
      if (response->reserved != 0 || response->params.reserved0 != 0 ||
          response->params.reserved1 != 0) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "EXECUTABLE_GLOBAL_BUFFER response reserved field is nonzero");
      } else if (IREE_HAL_REMOTE_RESOURCE_ID_TYPE(response->resolved_id) !=
                     IREE_HAL_REMOTE_RESOURCE_TYPE_BUFFER ||
                 IREE_HAL_REMOTE_RESOURCE_ID_IS_PROVISIONAL(
                     response->resolved_id)) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "EXECUTABLE_GLOBAL_BUFFER response has invalid buffer resource ID");
      } else if (response->byte_length > IREE_DEVICE_SIZE_MAX) {
        status = iree_make_status(
            IREE_STATUS_OUT_OF_RANGE,
            "EXECUTABLE_GLOBAL_BUFFER byte length exceeds device capacity");
      } else if ((iree_device_size_t)response->byte_length !=
                 expected_byte_length) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "EXECUTABLE_GLOBAL_BUFFER byte length changed from %" PRIu64
            " to %" PRIu64,
            (uint64_t)expected_byte_length, response->byte_length);
      } else {
        resolved_id = response->resolved_id;
        params.usage = (iree_hal_buffer_usage_t)response->params.usage;
        params.access = (iree_hal_memory_access_t)response->params.access;
        params.type = (iree_hal_memory_type_t)response->params.type;
        params.queue_affinity =
            (iree_hal_queue_affinity_t)response->params.queue_affinity;
        params.min_alignment =
            (iree_device_size_t)response->params.min_alignment;
        byte_length = (iree_device_size_t)response->byte_length;
        placement_flags =
            (iree_hal_buffer_placement_flags_t)response->placement_flags;
      }
    }
  }

  iree_async_buffer_lease_release(&response_lease);
  iree_allocator_free(executable->host_allocator, message);

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_buffer_create(
        executable->device, resolved_id, &params,
        /*allocation_size=*/byte_length, byte_length, placement_flags,
        executable->host_allocator, out_buffer);
    if (!iree_status_is_ok(status)) {
      status = iree_status_join(status,
                                iree_hal_remote_client_device_release_resource(
                                    executable->device, resolved_id));
    }
  }
  return status;
}

static iree_status_t
iree_hal_remote_client_executable_try_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();

  iree_slim_mutex_lock(&executable->global_mutex);
  iree_hal_remote_client_executable_global_t* global =
      iree_hal_remote_client_executable_find_global_locked(executable, name);
  if (global) {
    *out_found = true;
    *out_global =
        iree_hal_executable_global_from_value((uint64_t)(uintptr_t)global);
    iree_slim_mutex_unlock(&executable->global_mutex);
    return iree_ok_status();
  }
  iree_slim_mutex_unlock(&executable->global_mutex);

  iree_device_size_t byte_length = 0;
  bool found = false;
  iree_status_t status = iree_hal_remote_client_executable_query_global_by_name(
      executable, name, &found, &byte_length);
  IREE_RETURN_IF_ERROR(status);
  if (!found) {
    return iree_ok_status();
  }

  iree_hal_remote_client_executable_global_t* new_global = NULL;
  status = iree_hal_remote_client_executable_global_allocate(
      executable, name, byte_length, &new_global);
  IREE_RETURN_IF_ERROR(status);

  iree_slim_mutex_lock(&executable->global_mutex);
  global =
      iree_hal_remote_client_executable_find_global_locked(executable, name);
  if (global) {
    *out_found = true;
    *out_global =
        iree_hal_executable_global_from_value((uint64_t)(uintptr_t)global);
  } else {
    new_global->next = executable->global_list;
    executable->global_list = new_global;
    *out_found = true;
    *out_global =
        iree_hal_executable_global_from_value((uint64_t)(uintptr_t)new_global);
    new_global = NULL;
  }
  iree_slim_mutex_unlock(&executable->global_mutex);

  iree_hal_remote_client_executable_global_free(executable, new_global);
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_executable_global_info(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  memset(out_info, 0, sizeof(*out_info));

  iree_slim_mutex_lock(&executable->global_mutex);
  iree_hal_remote_client_executable_global_t* global_entry =
      iree_hal_remote_client_executable_global_from_handle_locked(executable,
                                                                  global);
  if (!global_entry) {
    iree_slim_mutex_unlock(&executable->global_mutex);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid remote executable global handle");
  }
  out_info->name = global_entry->name;
  out_info->byte_length = global_entry->byte_length;
  iree_slim_mutex_unlock(&executable->global_mutex);
  return iree_ok_status();
}

static iree_status_t iree_hal_remote_client_executable_global_buffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  *out_buffer = NULL;

  iree_slim_mutex_lock(&executable->global_mutex);
  iree_hal_remote_client_executable_global_t* global_entry =
      iree_hal_remote_client_executable_global_from_handle_locked(executable,
                                                                  global);
  if (!global_entry) {
    iree_slim_mutex_unlock(&executable->global_mutex);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid remote executable global handle");
  }
  iree_hal_remote_client_executable_global_buffer_t* cached_global_buffer =
      iree_hal_remote_client_executable_find_global_buffer_locked(
          global_entry, queue_affinity);
  if (cached_global_buffer) {
    *out_buffer = cached_global_buffer->buffer;
    iree_slim_mutex_unlock(&executable->global_mutex);
    return iree_ok_status();
  }
  iree_string_view_t name = global_entry->name;
  iree_device_size_t expected_byte_length = global_entry->byte_length;
  iree_slim_mutex_unlock(&executable->global_mutex);

  iree_hal_remote_client_executable_global_buffer_t* new_global_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(executable->host_allocator,
                                             sizeof(*new_global_buffer),
                                             (void**)&new_global_buffer));
  memset(new_global_buffer, 0, sizeof(*new_global_buffer));
  new_global_buffer->queue_affinity = queue_affinity;

  iree_status_t status = iree_hal_remote_client_executable_query_global_buffer(
      executable, name, queue_affinity, expected_byte_length,
      &new_global_buffer->buffer);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(executable->host_allocator, new_global_buffer);
    return status;
  }

  iree_slim_mutex_lock(&executable->global_mutex);
  global_entry = iree_hal_remote_client_executable_global_from_handle_locked(
      executable, global);
  if (!global_entry) {
    iree_slim_mutex_unlock(&executable->global_mutex);
    iree_hal_buffer_release(new_global_buffer->buffer);
    iree_allocator_free(executable->host_allocator, new_global_buffer);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid remote executable global handle");
  }
  cached_global_buffer =
      iree_hal_remote_client_executable_find_global_buffer_locked(
          global_entry, queue_affinity);
  if (cached_global_buffer) {
    *out_buffer = cached_global_buffer->buffer;
    iree_slim_mutex_unlock(&executable->global_mutex);
    iree_hal_buffer_release(new_global_buffer->buffer);
    iree_allocator_free(executable->host_allocator, new_global_buffer);
  } else {
    new_global_buffer->next = global_entry->buffer_list;
    global_entry->buffer_list = new_global_buffer;
    *out_buffer = new_global_buffer->buffer;
    iree_slim_mutex_unlock(&executable->global_mutex);
  }
  return iree_ok_status();
}

iree_status_t iree_hal_remote_client_executable_create(
    iree_hal_remote_client_device_t* device,
    iree_hal_remote_resource_id_t resource_id, iree_host_size_t function_count,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

  iree_hal_remote_client_executable_t* executable = NULL;
  iree_host_size_t function_infos_offset = 0;
  iree_host_size_t function_parameters_offset = 0;
  iree_host_size_t function_parameter_storage_offset = 0;
  iree_host_size_t total_size = 0;
  iree_status_t status = IREE_STRUCT_LAYOUT(
      sizeof(*executable), &total_size,
      IREE_STRUCT_FIELD(function_count, iree_hal_executable_function_info_t,
                        &function_infos_offset),
      IREE_STRUCT_FIELD(function_count,
                        iree_hal_executable_function_parameter_t*,
                        &function_parameters_offset),
      IREE_STRUCT_FIELD(function_count, void*,
                        &function_parameter_storage_offset));
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, total_size, (void**)&executable);
  }
  if (iree_status_is_ok(status)) {
    memset(executable, 0, total_size);
    iree_hal_resource_initialize(&iree_hal_remote_client_executable_vtable,
                                 &executable->resource);
    iree_slim_mutex_initialize(&executable->global_mutex);
    char* executable_storage = (char*)executable;
    executable->host_allocator = host_allocator;
    executable->device = device;
    executable->resource_id = resource_id;
    executable->function_count = function_count;
    executable->function_infos =
        (iree_hal_executable_function_info_t*)(executable_storage +
                                               function_infos_offset);
    executable->function_parameters =
        (iree_hal_executable_function_parameter_t**)(executable_storage +
                                                     function_parameters_offset);
    executable->function_parameter_storage =
        (void**)(executable_storage + function_parameter_storage_offset);

    status = iree_hal_remote_client_executable_initialize_metadata(executable);
  }
  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    if (resource_id != 0) {
      status = iree_status_join(
          status,
          iree_hal_remote_client_device_release_resource(device, resource_id));
    }
    if (executable) {
      executable->resource_id = 0;
      iree_hal_remote_client_executable_destroy(
          (iree_hal_executable_t*)executable);
    }
  }
  return status;
}

iree_status_t iree_hal_remote_client_executable_load(
    iree_hal_remote_client_device_t* device,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* load_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(load_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_host_size_t target_ordinal =
      iree_hal_device_spec_executable_target_ordinal(
          iree_hal_device_spec((iree_hal_device_t*)device), target);
  iree_status_t status = iree_ok_status();
  if (target_ordinal == IREE_HOST_SIZE_MAX) {
    status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "remote executable target must be borrowed from the device spec");
  }
  if (iree_status_is_ok(status) && load_params->constant_count > UINT16_MAX) {
    status = iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "executable constant count %" PRIhsz " exceeds wire limit %u",
        load_params->constant_count, (unsigned)UINT16_MAX);
  }

  const iree_host_size_t header_size =
      sizeof(iree_hal_remote_control_envelope_t) +
      sizeof(iree_hal_remote_executable_upload_request_t);
  iree_host_size_t constants_offset = 0;
  iree_host_size_t data_offset = 0;
  iree_host_size_t message_length = 0;
  if (iree_status_is_ok(status)) {
    status = IREE_STRUCT_LAYOUT(
        header_size, &message_length,
        IREE_STRUCT_FIELD_ALIGNED(load_params->constant_count, uint32_t, 8,
                                  &constants_offset),
        IREE_STRUCT_FIELD_ALIGNED(load_params->executable_data.data_length,
                                  uint8_t, 8, &data_offset));
  }

  uint8_t* message_buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, message_length,
                                   (void**)&message_buffer);
  }
  if (iree_status_is_ok(status)) {
    memset(message_buffer, 0, message_length);
    iree_hal_remote_control_envelope_t* envelope =
        (iree_hal_remote_control_envelope_t*)message_buffer;
    envelope->message_type = IREE_HAL_REMOTE_CONTROL_EXECUTABLE_UPLOAD;

    iree_hal_remote_executable_upload_request_t* request =
        (iree_hal_remote_executable_upload_request_t*)(envelope + 1);
    request->provisional_id = IREE_HAL_REMOTE_RESOURCE_ID_PROVISIONAL(
        IREE_HAL_REMOTE_RESOURCE_TYPE_EXECUTABLE, 0);
    request->target_ordinal = (uint64_t)target_ordinal;
    request->queue_affinity = (uint64_t)queue_affinity;
    request->data_length = (uint64_t)load_params->executable_data.data_length;
    request->load_flags = (uint32_t)load_params->flags;
    request->constant_count = (uint16_t)load_params->constant_count;
    request->upload_flags = IREE_HAL_REMOTE_UPLOAD_FLAG_INLINE_DATA;

    if (load_params->constant_count > 0) {
      memcpy(message_buffer + constants_offset, load_params->constants,
             load_params->constant_count * sizeof(uint32_t));
    }
    if (load_params->executable_data.data_length > 0) {
      memcpy(message_buffer + data_offset, load_params->executable_data.data,
             load_params->executable_data.data_length);
    }
  }

  iree_const_byte_span_t response_payload = iree_const_byte_span_empty();
  iree_async_buffer_lease_t response_lease;
  memset(&response_lease, 0, sizeof(response_lease));
  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_device_control_rpc(
        device, iree_make_const_byte_span(message_buffer, message_length),
        &response_payload, &response_lease);
  }
  iree_allocator_free(host_allocator, message_buffer);

  const iree_hal_remote_executable_upload_response_t* response = NULL;
  if (iree_status_is_ok(status)) {
    if (response_payload.data_length <
        sizeof(iree_hal_remote_executable_upload_response_t)) {
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "EXECUTABLE_UPLOAD response too short: "
          "%" PRIhsz " < %" PRIhsz,
          response_payload.data_length,
          sizeof(iree_hal_remote_executable_upload_response_t));
    } else {
      response = (const iree_hal_remote_executable_upload_response_t*)
                     response_payload.data;
      if (response->reserved != 0 ||
          IREE_HAL_REMOTE_RESOURCE_ID_TYPE(response->resolved_id) !=
              IREE_HAL_REMOTE_RESOURCE_TYPE_EXECUTABLE ||
          IREE_HAL_REMOTE_RESOURCE_ID_IS_PROVISIONAL(response->resolved_id)) {
        status = iree_make_status(
            IREE_STATUS_INTERNAL,
            "EXECUTABLE_UPLOAD response contains invalid resource metadata");
      }
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_remote_client_executable_create(
        device, response->resolved_id,
        (iree_host_size_t)response->function_count, host_allocator,
        out_executable);
  }

  iree_async_buffer_lease_release(&response_lease);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_hal_remote_resource_id_t iree_hal_remote_client_executable_resource_id(
    iree_hal_executable_t* base_executable) {
  iree_hal_remote_client_executable_t* executable =
      (iree_hal_remote_client_executable_t*)base_executable;
  return executable->resource_id;
}

static const iree_hal_executable_vtable_t
    iree_hal_remote_client_executable_vtable = {
        .destroy = iree_hal_remote_client_executable_destroy,
        .function_count = iree_hal_remote_client_executable_function_count,
        .function_info = iree_hal_remote_client_executable_function_info,
        .function_parameters =
            iree_hal_remote_client_executable_function_parameters,
        .lookup_function_by_name =
            iree_hal_remote_client_executable_lookup_function_by_name,
        .try_lookup_global_by_name =
            iree_hal_remote_client_executable_try_lookup_global_by_name,
        .global_info = iree_hal_remote_client_executable_global_info,
        .global_buffer = iree_hal_remote_client_executable_global_buffer,
};
