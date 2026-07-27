// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder_executable.h"

#include <inttypes.h>
#include <string.h>

#include "iree/hal/replay/recorder_record.h"

#define IREE_HAL_REPLAY_VTABLE_DISPATCH(resource, type_prefix, method_name) \
  ((const type_prefix##_vtable_t*)((const iree_hal_resource_t*)(resource))  \
       ->vtable)                                                            \
      ->method_name

//===----------------------------------------------------------------------===//
// iree_hal_replay_recorder_executable_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_replay_recorder_executable_function_map_entry_t {
  // Live function token returned by the wrapped executable.
  iree_hal_executable_function_t function;
  // Replay file ordinal assigned to the captured function slot.
  uint32_t ordinal;
} iree_hal_replay_recorder_executable_function_map_entry_t;

typedef struct iree_hal_replay_recorder_executable_t {
  // HAL resource header for the recording wrapper executable.
  iree_hal_resource_t resource;
  // Host allocator used for wrapper lifetime.
  iree_allocator_t host_allocator;
  // Shared recorder receiving all captured operations.
  iree_hal_replay_recorder_t* recorder;
  // Underlying executable receiving forwarded HAL calls.
  iree_hal_executable_t* base_executable;
  // Session-local device object id associated with this executable.
  iree_hal_replay_object_id_t device_id;
  // Session-local object id assigned to this executable.
  iree_hal_replay_object_id_t executable_id;
  // Number of entries in |function_map|.
  iree_host_size_t function_map_count;
  // Live function token to replay file ordinal mapping.
  iree_hal_replay_recorder_executable_function_map_entry_t* function_map;
} iree_hal_replay_recorder_executable_t;

static const iree_hal_executable_vtable_t
    iree_hal_replay_recorder_executable_vtable;

static bool iree_hal_replay_recorder_executable_isa(
    iree_hal_executable_t* base_executable) {
  return iree_hal_resource_is(base_executable,
                              &iree_hal_replay_recorder_executable_vtable);
}

static iree_hal_replay_recorder_executable_t*
iree_hal_replay_recorder_executable_cast(
    iree_hal_executable_t* base_executable) {
  IREE_HAL_ASSERT_TYPE(base_executable,
                       &iree_hal_replay_recorder_executable_vtable);
  return (iree_hal_replay_recorder_executable_t*)base_executable;
}

iree_hal_executable_t* iree_hal_replay_recorder_executable_base_or_self(
    iree_hal_executable_t* executable) {
  return iree_hal_replay_recorder_executable_isa(executable)
             ? iree_hal_replay_recorder_executable_cast(executable)
                   ->base_executable
             : executable;
}

iree_hal_replay_object_id_t iree_hal_replay_recorder_executable_id_or_none(
    iree_hal_executable_t* executable) {
  return iree_hal_replay_recorder_executable_isa(executable)
             ? iree_hal_replay_recorder_executable_cast(executable)
                   ->executable_id
             : IREE_HAL_REPLAY_OBJECT_ID_NONE;
}

static iree_status_t iree_hal_replay_recorder_executable_build_function_map(
    iree_hal_executable_t* executable, iree_allocator_t host_allocator,
    iree_host_size_t* out_function_map_count,
    iree_hal_replay_recorder_executable_function_map_entry_t**
        out_function_map) {
  *out_function_map_count = 0;
  *out_function_map = NULL;

  const iree_host_size_t function_count =
      iree_hal_executable_function_count(executable);
  if (IREE_UNLIKELY(function_count > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "replay recording function count exceeds dispatch ordinal range");
  }
  if (function_count == 0) return iree_ok_status();

  iree_host_size_t function_map_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          function_count,
          sizeof(iree_hal_replay_recorder_executable_function_map_entry_t),
          &function_map_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay function map size overflow");
  }
  iree_hal_replay_recorder_executable_function_map_entry_t* function_map = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, function_map_size,
                                             (void**)&function_map));

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < function_count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_executable_function_t indexed_function =
        iree_hal_executable_function_from_index((uint32_t)i);
    iree_hal_executable_function_info_t info;
    status =
        iree_hal_executable_function_info(executable, indexed_function, &info);
    iree_hal_executable_function_t dispatch_function = indexed_function;
    if (iree_status_is_ok(status) && iree_string_view_is_empty(info.name)) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "replay recording requires a name for executable function %" PRIhsz,
          i);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_executable_lookup_function_by_name(
          executable, info.name, &dispatch_function);
      if (!iree_status_is_ok(status)) {
        status = iree_status_annotate_f(
            status,
            "looking up recorded executable function '%.*s' for function "
            "%" PRIhsz,
            (int)info.name.size, info.name.data, i);
      }
    }
    for (iree_host_size_t j = 0; j < i && iree_status_is_ok(status); ++j) {
      if (IREE_UNLIKELY(function_map[j].function.value ==
                        dispatch_function.value)) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "replay recording requires unique executable function names but "
            "'%.*s' resolves to the same function token as function %" PRIhsz,
            (int)info.name.size, info.name.data, j);
      }
    }
    if (iree_status_is_ok(status)) {
      function_map[i].function = dispatch_function;
      function_map[i].ordinal = (uint32_t)i;
    }
  }
  if (iree_status_is_ok(status)) {
    *out_function_map_count = function_count;
    *out_function_map = function_map;
  } else {
    iree_allocator_free(host_allocator, function_map);
  }
  return status;
}

iree_status_t iree_hal_replay_recorder_executable_recorded_ordinal(
    iree_hal_executable_t* executable, iree_hal_executable_function_t function,
    uint32_t* out_ordinal) {
  IREE_ASSERT_ARGUMENT(executable);
  IREE_ASSERT_ARGUMENT(out_ordinal);
  *out_ordinal = 0;

  if (!iree_hal_replay_recorder_executable_isa(executable)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "replay recording requires an executable created by the replay "
        "recorder device load path");
  }

  iree_hal_replay_recorder_executable_t* recorder_executable =
      iree_hal_replay_recorder_executable_cast(executable);
  for (iree_host_size_t i = 0; i < recorder_executable->function_map_count;
       ++i) {
    const iree_hal_replay_recorder_executable_function_map_entry_t* entry =
        &recorder_executable->function_map[i];
    if (entry->function.value == function.value) {
      *out_ordinal = entry->ordinal;
      return iree_ok_status();
    }
  }
  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "replay recording could not map executable function token %" PRIu64
      " to a captured function ordinal",
      function.value);
}

static iree_status_t iree_hal_replay_recorder_executable_create_proxy(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t executable_id,
    iree_hal_executable_t* base_executable, iree_allocator_t host_allocator,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_executable);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

  iree_host_size_t function_map_count = 0;
  iree_hal_replay_recorder_executable_function_map_entry_t* function_map = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_executable_build_function_map(
      base_executable, host_allocator, &function_map_count, &function_map));

  iree_hal_replay_recorder_executable_t* executable = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*executable), (void**)&executable);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, function_map);
    return status;
  }
  memset(executable, 0, sizeof(*executable));
  iree_hal_resource_initialize(&iree_hal_replay_recorder_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->recorder = recorder;
  iree_hal_replay_recorder_retain(executable->recorder);
  executable->base_executable = base_executable;
  iree_hal_executable_retain(executable->base_executable);
  executable->device_id = device_id;
  executable->executable_id = executable_id;
  executable->function_map_count = function_map_count;
  executable->function_map = function_map;

  *out_executable = (iree_hal_executable_t*)executable;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_recorder_executable_begin_operation(
    iree_hal_replay_recorder_executable_t* executable,
    iree_hal_replay_operation_code_t operation_code,
    iree_hal_replay_payload_type_t payload_type,
    iree_hal_replay_pending_record_t* out_pending_record) {
  return iree_hal_replay_recorder_begin_operation(
      executable->recorder, executable->device_id, executable->executable_id,
      IREE_HAL_REPLAY_OBJECT_ID_NONE, IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE,
      operation_code, payload_type, out_pending_record);
}

static void iree_hal_replay_recorder_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_replay_recorder_executable_t* executable =
      iree_hal_replay_recorder_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_executable_release(executable->base_executable);
  iree_hal_replay_recorder_release(executable->recorder);
  iree_allocator_free(host_allocator, executable->function_map);
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

static iree_host_size_t iree_hal_replay_recorder_executable_function_count(
    iree_hal_executable_t* base_executable) {
  iree_hal_replay_recorder_executable_t* executable =
      iree_hal_replay_recorder_executable_cast(base_executable);
  iree_hal_replay_pending_record_t pending_record;
  iree_status_t status = iree_hal_replay_recorder_executable_begin_operation(
      executable, IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_FUNCTION_COUNT,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record);
  iree_host_size_t count =
      iree_hal_executable_function_count(executable->base_executable);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_end_operation(&pending_record,
                                                    iree_ok_status());
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_replay_recorder_fail(executable->recorder,
                                  iree_status_code(status));
    iree_status_ignore(status);
  }
  return count;
}

static iree_status_t iree_hal_replay_recorder_executable_function_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  iree_hal_replay_recorder_executable_t* executable =
      iree_hal_replay_recorder_executable_cast(base_executable);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_executable_begin_operation(
      executable, IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_FUNCTION_INFO,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record, iree_hal_executable_function_info(
                           executable->base_executable, function, out_info));
}

static iree_status_t iree_hal_replay_recorder_executable_function_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  iree_hal_replay_recorder_executable_t* executable =
      iree_hal_replay_recorder_executable_cast(base_executable);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_executable_begin_operation(
      executable, IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_FUNCTION_PARAMETERS,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record,
      iree_hal_executable_function_parameters(
          executable->base_executable, function, capacity, out_parameters));
}

static iree_status_t
iree_hal_replay_recorder_executable_lookup_function_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_executable_function_t* out_function) {
  iree_hal_replay_recorder_executable_t* executable =
      iree_hal_replay_recorder_executable_cast(base_executable);
  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_executable_begin_operation(
      executable,
      IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_LOOKUP_FUNCTION_BY_NAME,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, &pending_record));
  return iree_hal_replay_recorder_end_operation(
      &pending_record, iree_hal_executable_lookup_function_by_name(
                           executable->base_executable, name, out_function));
}

static iree_status_t
iree_hal_replay_recorder_executable_try_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    bool* out_found, iree_hal_executable_global_t* out_global) {
  iree_hal_replay_recorder_executable_t* executable =
      iree_hal_replay_recorder_executable_cast(base_executable);
  (void)executable;
  (void)name;
  *out_found = false;
  *out_global = iree_hal_executable_global_invalid();
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_recorder_executable_global_info(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info) {
  iree_hal_replay_recorder_executable_t* executable =
      iree_hal_replay_recorder_executable_cast(base_executable);
  (void)executable;
  (void)global;
  memset(out_info, 0, sizeof(*out_info));
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "invalid replay recorder executable global");
}

static iree_status_t iree_hal_replay_recorder_executable_global_buffer(
    iree_hal_executable_t* base_executable, iree_hal_executable_global_t global,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  iree_hal_replay_recorder_executable_t* executable =
      iree_hal_replay_recorder_executable_cast(base_executable);
  (void)executable;
  (void)global;
  (void)queue_affinity;
  *out_buffer = NULL;
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "invalid replay recorder executable global");
}

static iree_status_t iree_hal_replay_recorder_load_payload_iovecs(
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* params,
    iree_const_byte_span_t executable_metadata,
    iree_hal_replay_executable_load_payload_t* out_payload,
    iree_const_byte_span_t out_iovecs[6]) {
  memset(out_payload, 0, sizeof(*out_payload));
  out_payload->queue_affinity = queue_affinity;
  out_payload->target_physical_device_affinity =
      target->physical_device_affinity;
  out_payload->executable_data_length = params->executable_data.data_length;
  out_payload->constant_count = params->constant_count;
  out_payload->load_flags = params->flags;
  out_payload->target_kind = target->kind;
  out_payload->target_flags = target->flags;
  if (IREE_UNLIKELY(target->family.size > UINT32_MAX ||
                    target->target_key.size > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable target identity length overflow");
  }
  out_payload->target_family_length = (uint32_t)target->family.size;
  out_payload->target_key_length = (uint32_t)target->target_key.size;
  if (IREE_UNLIKELY(executable_metadata.data_length > UINT32_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable metadata byte length overflow");
  }
  out_payload->executable_metadata_length =
      (uint32_t)executable_metadata.data_length;

  if (IREE_UNLIKELY(params->constant_count > 0 && !params->constants)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable constants are required");
  }
  iree_host_size_t constant_bytes = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          params->constant_count, sizeof(uint32_t), &constant_bytes))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable constant byte length overflow");
  }

  out_iovecs[0] = iree_make_const_byte_span(out_payload, sizeof(*out_payload));
  out_iovecs[1] =
      iree_make_const_byte_span(target->family.data, target->family.size);
  out_iovecs[2] = iree_make_const_byte_span(target->target_key.data,
                                            target->target_key.size);
  out_iovecs[3] = params->executable_data;
  out_iovecs[4] = iree_make_const_byte_span(params->constants, constant_bytes);
  out_iovecs[5] = executable_metadata;
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_recorder_capture_executable_metadata(
    iree_hal_executable_t* executable, iree_allocator_t host_allocator,
    iree_byte_span_t* out_storage, iree_const_byte_span_t* out_metadata) {
  IREE_ASSERT_ARGUMENT(out_storage);
  IREE_ASSERT_ARGUMENT(out_metadata);
  *out_storage = iree_byte_span_empty();
  *out_metadata = iree_const_byte_span_empty();

  const iree_host_size_t function_count =
      iree_hal_executable_function_count(executable);
  if (IREE_UNLIKELY(function_count > UINT32_MAX)) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "replay executable function count exceeds dispatch ordinal range");
  }
  iree_host_size_t function_info_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          function_count, sizeof(iree_hal_executable_function_info_t),
          &function_info_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable function metadata size overflow");
  }

  iree_hal_executable_function_info_t* function_infos = NULL;
  if (function_info_size > 0) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        host_allocator, function_info_size, (void**)&function_infos));
  }
  iree_status_t status = iree_ok_status();
  iree_host_size_t parameter_count = 0;
  iree_host_size_t function_name_storage_size = 0;
  for (iree_host_size_t i = 0; i < function_count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_executable_function_info(
        executable, iree_hal_executable_function_from_index((uint32_t)i),
        &function_infos[i]);
    if (iree_status_is_ok(status)) {
      if (IREE_UNLIKELY(iree_string_view_is_empty(function_infos[i].name))) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "replay recording requires a name for executable function %" PRIhsz,
            i);
      }
    }
    if (iree_status_is_ok(status)) {
      if (IREE_UNLIKELY(function_infos[i].name.size > UINT16_MAX)) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "executable function name length overflow");
      }
    }
    for (iree_host_size_t j = 0; j < i && iree_status_is_ok(status); ++j) {
      if (IREE_UNLIKELY(iree_string_view_equal(function_infos[i].name,
                                               function_infos[j].name))) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "replay recording requires unique executable function names but "
            "function %" PRIhsz " and %" PRIhsz " are both named '%.*s'",
            j, i, (int)function_infos[i].name.size,
            function_infos[i].name.data);
      }
    }
    if (iree_status_is_ok(status)) {
      if (IREE_UNLIKELY(!iree_host_size_checked_add(
              parameter_count, function_infos[i].parameter_count,
              &parameter_count))) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "executable parameter metadata count overflow");
      }
    }
    if (iree_status_is_ok(status)) {
      if (IREE_UNLIKELY(!iree_host_size_checked_add(
              function_name_storage_size, function_infos[i].name.size,
              &function_name_storage_size))) {
        status =
            iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                             "executable function name storage size overflow");
      }
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, function_infos);
    return status;
  }

  iree_hal_executable_function_parameter_t* parameters = NULL;
  iree_host_size_t parameter_info_size = 0;
  if (parameter_count > 0) {
    if (IREE_UNLIKELY(!iree_host_size_checked_mul(
            parameter_count, sizeof(iree_hal_executable_function_parameter_t),
            &parameter_info_size))) {
      iree_allocator_free(host_allocator, function_infos);
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "executable parameter metadata size overflow");
    }
    status = iree_allocator_malloc(host_allocator, parameter_info_size,
                                   (void**)&parameters);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(host_allocator, function_infos);
      return status;
    }
  }

  iree_host_size_t parameter_index = 0;
  for (iree_host_size_t i = 0; i < function_count && iree_status_is_ok(status);
       ++i) {
    const iree_host_size_t function_parameter_count =
        function_infos[i].parameter_count;
    if (function_parameter_count == 0) continue;
    status = iree_hal_executable_function_parameters(
        executable, iree_hal_executable_function_from_index((uint32_t)i),
        function_parameter_count, parameters + parameter_index);
    parameter_index += function_parameter_count;
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(host_allocator, parameters);
    iree_allocator_free(host_allocator, function_infos);
    return status;
  }

  iree_host_size_t metadata_size = 0;
  iree_host_size_t function_metadata_size = 0;
  iree_host_size_t parameter_metadata_size = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_mul(
              function_count,
              sizeof(iree_hal_replay_executable_function_metadata_t),
              &function_metadata_size) ||
          !iree_host_size_checked_mul(
              parameter_count,
              sizeof(iree_hal_replay_executable_parameter_metadata_t),
              &parameter_metadata_size) ||
          !iree_host_size_checked_add(
              sizeof(iree_hal_replay_executable_metadata_header_t),
              function_metadata_size, &metadata_size) ||
          !iree_host_size_checked_add(metadata_size, parameter_metadata_size,
                                      &metadata_size) ||
          !iree_host_size_checked_add(metadata_size, function_name_storage_size,
                                      &metadata_size))) {
    iree_allocator_free(host_allocator, parameters);
    iree_allocator_free(host_allocator, function_infos);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable replay metadata size overflow");
  }

  uint8_t* metadata_storage = NULL;
  status = iree_allocator_malloc(host_allocator, metadata_size,
                                 (void**)&metadata_storage);
  if (iree_status_is_ok(status)) {
    memset(metadata_storage, 0, metadata_size);
    iree_hal_replay_executable_metadata_header_t* header =
        (iree_hal_replay_executable_metadata_header_t*)metadata_storage;
    header->function_count = function_count;
    header->parameter_count = parameter_count;
    header->function_name_storage_length = function_name_storage_size;
    iree_hal_replay_executable_function_metadata_t* function_metadata =
        (iree_hal_replay_executable_function_metadata_t*)(metadata_storage +
                                                          sizeof(*header));
    iree_hal_replay_executable_parameter_metadata_t* parameter_metadata =
        (iree_hal_replay_executable_parameter_metadata_t*)(metadata_storage +
                                                           sizeof(*header) +
                                                           function_metadata_size);
    uint8_t* function_name_storage = metadata_storage + sizeof(*header) +
                                     function_metadata_size +
                                     parameter_metadata_size;

    parameter_index = 0;
    iree_host_size_t function_name_offset = 0;
    for (iree_host_size_t i = 0; i < function_count; ++i) {
      function_metadata[i].flags = function_infos[i].flags;
      function_metadata[i].workgroup_size[0] =
          function_infos[i].workgroup_size[0];
      function_metadata[i].workgroup_size[1] =
          function_infos[i].workgroup_size[1];
      function_metadata[i].workgroup_size[2] =
          function_infos[i].workgroup_size[2];
      function_metadata[i].constant_byte_length =
          function_infos[i].constant_byte_length;
      function_metadata[i].binding_count = function_infos[i].binding_count;
      function_metadata[i].parameter_count = function_infos[i].parameter_count;
      function_metadata[i].name_length = (uint16_t)function_infos[i].name.size;
      if (!iree_string_view_is_empty(function_infos[i].name)) {
        memcpy(function_name_storage + function_name_offset,
               function_infos[i].name.data, function_infos[i].name.size);
        function_name_offset += function_infos[i].name.size;
      }
      for (iree_host_size_t j = 0; j < function_infos[i].parameter_count; ++j) {
        const iree_hal_executable_function_parameter_t* parameter =
            &parameters[parameter_index++];
        parameter_metadata->offset = parameter->offset;
        parameter_metadata->native_abi_offset =
            iree_any_bit_set(
                parameter->flags,
                IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET)
                ? parameter->native_abi_offset
                : 0;
        parameter_metadata->flags = parameter->flags;
        parameter_metadata->type = parameter->type;
        parameter_metadata->size = parameter->size;
        ++parameter_metadata;
      }
    }
    *out_storage = iree_make_byte_span(metadata_storage, metadata_size);
    *out_metadata = iree_make_const_byte_span(metadata_storage, metadata_size);
  }
  iree_allocator_free(host_allocator, parameters);
  iree_allocator_free(host_allocator, function_infos);
  return status;
}

iree_status_t iree_hal_replay_recorder_device_load_executable(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_device_t* base_device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_executable_target_t* target,
    const iree_hal_executable_load_params_t* params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(base_device);
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;

  iree_hal_replay_object_id_t executable_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_recorder_reserve_object_id(recorder, &executable_id));

  iree_hal_replay_executable_load_payload_t operation_payload;
  iree_const_byte_span_t operation_iovecs[6];
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_load_payload_iovecs(
      queue_affinity, target, params, iree_const_byte_span_empty(),
      &operation_payload, operation_iovecs));

  iree_hal_replay_pending_record_t pending_record;
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_begin_operation(
      recorder, device_id, device_id, executable_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_LOAD_EXECUTABLE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_EXECUTABLE_LOAD, &pending_record));

  iree_hal_executable_t* base_executable = NULL;
  iree_hal_executable_t* replay_executable = NULL;
  iree_status_t status = iree_hal_device_load_executable(
      base_device, queue_affinity, target, params, &base_executable);
  iree_byte_span_t executable_metadata_storage = iree_byte_span_empty();
  iree_const_byte_span_t executable_metadata = iree_const_byte_span_empty();
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_capture_executable_metadata(
        base_executable, host_allocator, &executable_metadata_storage,
        &executable_metadata);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_executable_create_proxy(
        recorder, device_id, executable_id, base_executable, host_allocator,
        &replay_executable);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_recorder_load_payload_iovecs(
        queue_affinity, target, params, executable_metadata, &operation_payload,
        operation_iovecs);
  }
  status = iree_hal_replay_recorder_end_creation_operation(
      &pending_record, status, IREE_ARRAYSIZE(operation_iovecs),
      operation_iovecs, IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE, executable_id,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE, 0, NULL);

  if (iree_status_is_ok(status)) {
    *out_executable = replay_executable;
  } else {
    iree_hal_executable_release(replay_executable);
  }
  iree_allocator_free(host_allocator, executable_metadata_storage.data);
  iree_hal_executable_release(base_executable);
  return status;
}

static const iree_hal_executable_vtable_t
    iree_hal_replay_recorder_executable_vtable = {
        .destroy = iree_hal_replay_recorder_executable_destroy,
        .function_count = iree_hal_replay_recorder_executable_function_count,
        .function_info = iree_hal_replay_recorder_executable_function_info,
        .function_parameters =
            iree_hal_replay_recorder_executable_function_parameters,
        .lookup_function_by_name =
            iree_hal_replay_recorder_executable_lookup_function_by_name,
        .try_lookup_global_by_name =
            iree_hal_replay_recorder_executable_try_lookup_global_by_name,
        .global_info = iree_hal_replay_recorder_executable_global_info,
        .global_buffer = iree_hal_replay_recorder_executable_global_buffer,
};
