// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/execute_object.h"

#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <string.h>

#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
#include <sys/stat.h>
#include <unistd.h>
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)

#include "iree/hal/replay/digest.h"
#include "iree/io/file_handle.h"

static iree_status_t iree_hal_replay_executor_store_device(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  if (executor->next_device_index >=
      iree_hal_device_group_device_count(executor->device_group)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay references more devices than provided");
  }
  iree_hal_device_t* device = iree_hal_device_group_device_at(
      executor->device_group, executor->next_device_index++);
  iree_hal_device_retain(device);
  iree_hal_replay_object_entry_t entry = {.value.device = device};
  return iree_hal_replay_executor_store(executor, record->header.object_id,
                                        IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
                                        entry);
}

static iree_status_t iree_hal_replay_executor_store_allocator(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.device_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_allocator_t* allocator =
      iree_hal_device_allocator(device_entry->value.device);
  iree_hal_allocator_retain(allocator);
  iree_hal_replay_object_entry_t entry = {.value.allocator = allocator};
  return iree_hal_replay_executor_store(executor, record->header.object_id,
                                        IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR,
                                        entry);
}

iree_status_t iree_hal_replay_executor_replay_object(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  switch (record->header.object_type) {
    case IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE:
      return iree_hal_replay_executor_store_device(executor, record);
    case IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR:
      return iree_hal_replay_executor_store_allocator(executor, record);
    default:
      return iree_ok_status();
  }
}

static iree_status_t iree_hal_replay_executor_make_function_map_from_metadata(
    iree_hal_replay_executor_t* executor,
    iree_hal_replay_object_id_t executable_id,
    iree_const_byte_span_t executable_metadata,
    iree_hal_executable_t* executable, iree_host_size_t* out_function_map_count,
    iree_hal_executable_function_t** out_function_map) {
  *out_function_map_count = 0;
  *out_function_map = NULL;

  if (IREE_UNLIKELY(executable_metadata.data_length <
                    sizeof(iree_hal_replay_executable_metadata_header_t))) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay executable metadata is too short");
  }
  iree_hal_replay_executable_metadata_header_t header;
  memcpy(&header, executable_metadata.data, sizeof(header));
  if (IREE_UNLIKELY(header.reserved1 != 0)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay executable metadata reserved fields must "
                            "be zero");
  }
  if (IREE_UNLIKELY(header.function_count > IREE_HOST_SIZE_MAX ||
                    header.function_count > UINT32_MAX ||
                    header.parameter_count > IREE_HOST_SIZE_MAX ||
                    header.function_name_storage_length > IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay executable metadata count overflow");
  }
  const iree_host_size_t captured_function_count =
      (iree_host_size_t)header.function_count;
  const iree_host_size_t captured_parameter_count =
      (iree_host_size_t)header.parameter_count;
  const iree_host_size_t captured_name_storage_length =
      (iree_host_size_t)header.function_name_storage_length;

  iree_host_size_t function_metadata_size = 0;
  iree_host_size_t parameter_metadata_size = 0;
  iree_host_size_t expected_length = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
                        captured_function_count,
                        sizeof(iree_hal_replay_executable_function_metadata_t),
                        &function_metadata_size) ||
                    !iree_host_size_checked_mul(
                        captured_parameter_count,
                        sizeof(iree_hal_replay_executable_parameter_metadata_t),
                        &parameter_metadata_size) ||
                    !iree_host_size_checked_add(
                        sizeof(iree_hal_replay_executable_metadata_header_t),
                        function_metadata_size, &expected_length) ||
                    !iree_host_size_checked_add(expected_length,
                                                parameter_metadata_size,
                                                &expected_length) ||
                    !iree_host_size_checked_add(expected_length,
                                                captured_name_storage_length,
                                                &expected_length) ||
                    expected_length != executable_metadata.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay executable metadata length mismatch");
  }

  const uint8_t* function_metadata_data =
      executable_metadata.data +
      sizeof(iree_hal_replay_executable_metadata_header_t);
  const uint8_t* parameter_metadata_data =
      function_metadata_data + function_metadata_size;
  const iree_hal_replay_executable_function_metadata_t* function_metadata =
      (const iree_hal_replay_executable_function_metadata_t*)
          function_metadata_data;
  const iree_hal_replay_executable_parameter_metadata_t* parameter_metadata =
      (const iree_hal_replay_executable_parameter_metadata_t*)
          parameter_metadata_data;
  const char* name_storage =
      (const char*)(parameter_metadata_data + parameter_metadata_size);

  iree_hal_executable_function_t* function_map = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_allocate_function_map(
      executor, captured_function_count, &function_map));

  iree_host_size_t parameter_index = 0;
  iree_host_size_t name_offset = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < captured_function_count && iree_status_is_ok(status); ++i) {
    const iree_hal_replay_executable_function_metadata_t* captured =
        &function_metadata[i];
    if (IREE_UNLIKELY(captured->parameter_count >
                          captured_parameter_count - parameter_index ||
                      captured->name_length >
                          captured_name_storage_length - name_offset)) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "replay executable metadata count mismatch");
    }

    iree_string_view_t captured_name = iree_string_view_empty();
    if (iree_status_is_ok(status)) {
      captured_name = iree_make_string_view(
          name_storage + name_offset, (iree_host_size_t)captured->name_length);
      name_offset += captured->name_length;
    }

    iree_hal_executable_function_t function =
        iree_hal_executable_function_invalid();
    if (iree_status_is_ok(status)) {
      if (iree_string_view_is_empty(captured_name)) {
        status =
            iree_make_status(IREE_STATUS_DATA_LOSS,
                             "replay executable %" PRIu64
                             " captured function %" PRIhsz " is missing a name",
                             executable_id, i);
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_executable_lookup_function_by_name(
          executable, captured_name, &function);
      if (!iree_status_is_ok(status)) {
        status = iree_status_annotate_f(
            status,
            "looking up replay function '%.*s' for executable %" PRIu64
            " captured function %" PRIhsz,
            (int)captured_name.size, captured_name.data, executable_id, i);
      }
    }
    for (iree_host_size_t j = 0; j < i && iree_status_is_ok(status); ++j) {
      if (IREE_UNLIKELY(function_map[j].value == function.value)) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "replay executable %" PRIu64
            " captured function '%.*s' resolves to the same loaded function "
            "token as captured function %" PRIhsz,
            executable_id, (int)captured_name.size, captured_name.data, j);
      }
    }

    iree_hal_executable_function_info_t loaded_info;
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_executable_function_info(executable, function, &loaded_info);
    }
    if (iree_status_is_ok(status) &&
        (captured->flags != loaded_info.flags ||
         captured->constant_byte_length != loaded_info.constant_byte_length ||
         captured->binding_count != loaded_info.binding_count ||
         captured->parameter_count != loaded_info.parameter_count ||
         captured->workgroup_size[0] != loaded_info.workgroup_size[0] ||
         captured->workgroup_size[1] != loaded_info.workgroup_size[1] ||
         captured->workgroup_size[2] != loaded_info.workgroup_size[2])) {
      status = iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "replay executable %" PRIu64 " function %" PRIhsz
          " ABI mismatch: captured=(flags=0x%016" PRIx64
          " constant_bytes=%u bindings=%u parameters=%u "
          "workgroup_size=[%u,%u,%u]) "
          "loaded=(flags=0x%016" PRIx64
          " constant_bytes=%u bindings=%u parameters=%u "
          "workgroup_size=[%u,%u,%u])",
          executable_id, i, captured->flags,
          (uint32_t)captured->constant_byte_length,
          (uint32_t)captured->binding_count,
          (uint32_t)captured->parameter_count, captured->workgroup_size[0],
          captured->workgroup_size[1], captured->workgroup_size[2],
          loaded_info.flags, (uint32_t)loaded_info.constant_byte_length,
          (uint32_t)loaded_info.binding_count,
          (uint32_t)loaded_info.parameter_count, loaded_info.workgroup_size[0],
          loaded_info.workgroup_size[1], loaded_info.workgroup_size[2]);
    }

    iree_hal_executable_function_parameter_t* loaded_parameters = NULL;
    if (iree_status_is_ok(status) && captured->parameter_count != 0) {
      iree_host_size_t parameter_size = 0;
      if (IREE_UNLIKELY(!iree_host_size_checked_mul(
              captured->parameter_count,
              sizeof(iree_hal_executable_function_parameter_t),
              &parameter_size))) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "executable parameter metadata is too large");
      }
      if (iree_status_is_ok(status)) {
        status = iree_allocator_malloc(executor->host_allocator, parameter_size,
                                       (void**)&loaded_parameters);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_executable_function_parameters(
            executable, function, captured->parameter_count, loaded_parameters);
      }
    }
    for (iree_host_size_t j = 0;
         j < captured->parameter_count && iree_status_is_ok(status); ++j) {
      const iree_hal_replay_executable_parameter_metadata_t*
          captured_parameter = &parameter_metadata[parameter_index + j];
      if (IREE_UNLIKELY(captured_parameter->reserved0 != 0)) {
        status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                  "replay executable parameter metadata "
                                  "reserved fields must be zero");
        break;
      }
      const iree_hal_executable_function_parameter_t* loaded_parameter =
          &loaded_parameters[j];
      if (captured_parameter->type != loaded_parameter->type ||
          captured_parameter->size != loaded_parameter->size ||
          captured_parameter->flags != loaded_parameter->flags ||
          captured_parameter->offset != loaded_parameter->offset ||
          (iree_any_bit_set(
               captured_parameter->flags,
               IREE_HAL_EXECUTABLE_FUNCTION_PARAMETER_FLAG_NATIVE_ABI_OFFSET) &&
           captured_parameter->native_abi_offset !=
               loaded_parameter->native_abi_offset)) {
        status = iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "replay executable %" PRIu64 " function %" PRIhsz
            " parameter %" PRIhsz
            " ABI mismatch: captured=(type=%u size=%u flags=0x%04x "
            "offset=%u native_offset=%u) loaded=(type=%u size=%u flags=0x%04x "
            "offset=%u native_offset=%u)",
            executable_id, i, j, (uint32_t)captured_parameter->type,
            (uint32_t)captured_parameter->size,
            (uint32_t)captured_parameter->flags,
            (uint32_t)captured_parameter->offset,
            (uint32_t)captured_parameter->native_abi_offset,
            (uint32_t)loaded_parameter->type, (uint32_t)loaded_parameter->size,
            (uint32_t)loaded_parameter->flags,
            (uint32_t)loaded_parameter->offset,
            (uint32_t)loaded_parameter->native_abi_offset);
      }
    }
    parameter_index += captured->parameter_count;
    iree_allocator_free(executor->host_allocator, loaded_parameters);
    if (iree_status_is_ok(status)) {
      function_map[i] = function;
    }
  }
  if (iree_status_is_ok(status) &&
      IREE_UNLIKELY(parameter_index != captured_parameter_count ||
                    name_offset != captured_name_storage_length)) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "replay executable metadata count mismatch");
  }
  if (iree_status_is_ok(status)) {
    *out_function_map_count = captured_function_count;
    *out_function_map = function_map;
  } else {
    iree_allocator_free(executor->host_allocator, function_map);
  }
  return status;
}

static iree_status_t iree_hal_replay_executor_select_executable_target(
    iree_hal_device_t* device,
    const iree_hal_executable_target_selection_t* selection,
    iree_hal_executable_target_flags_t required_flags,
    const iree_hal_executable_target_t** out_target) {
  *out_target = NULL;
  const iree_hal_device_spec_t* device_spec = iree_hal_device_spec(device);
  if (IREE_UNLIKELY(!device_spec)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "replay device does not expose executable target facts");
  }
  const iree_hal_executable_target_selection_result_t result =
      iree_hal_device_spec_select_executable_target(device_spec, selection);
  if (result.outcome == IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_NO_MATCH) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "replay device does not support executable target '%.*s:%.*s'",
        (int)selection->family.size, selection->family.data,
        (int)selection->target_key.size, selection->target_key.data);
  }
  if (result.outcome ==
      IREE_HAL_EXECUTABLE_TARGET_SELECTION_OUTCOME_AMBIGUOUS) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "replay executable target '%.*s:%.*s' is ambiguous",
                            (int)selection->family.size, selection->family.data,
                            (int)selection->target_key.size,
                            selection->target_key.data);
  }
  if (IREE_UNLIKELY(!iree_all_bits_set(result.target->flags, required_flags))) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "replay executable target '%.*s:%.*s' is missing captured flags "
        "0x%08" PRIx32,
        (int)selection->family.size, selection->family.data,
        (int)selection->target_key.size, selection->target_key.data,
        required_flags & ~result.target->flags);
  }
  *out_target = result.target;
  return iree_ok_status();
}

static bool iree_hal_replay_executable_target_selection_is_empty(
    const iree_hal_executable_target_selection_t* selection) {
  return iree_string_view_is_empty(selection->family) &&
         iree_string_view_is_empty(selection->target_key) &&
         selection->kind_flags == IREE_HAL_EXECUTABLE_TARGET_KIND_FLAG_NONE &&
         selection->physical_device_affinity == 0;
}

static iree_status_t iree_hal_replay_executor_load_executable(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_EXECUTABLE_LOAD,
      sizeof(iree_hal_replay_executable_load_payload_t)));
  iree_hal_replay_executable_load_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));

  iree_host_size_t constant_bytes = 0;
  iree_host_size_t target_key_offset = 0;
  iree_host_size_t data_offset = 0;
  iree_host_size_t constants_offset = 0;
  iree_host_size_t metadata_offset = 0;
  iree_host_size_t expected_length = 0;
  if (IREE_UNLIKELY(
          payload.reserved0 != 0 || payload.reserved1 != 0 ||
          payload.target_kind > IREE_HAL_EXECUTABLE_TARGET_KIND_COMPOSITE ||
          payload.target_family_length == 0 || payload.target_key_length == 0 ||
          payload.executable_data_length > IREE_HOST_SIZE_MAX ||
          payload.constant_count > IREE_HOST_SIZE_MAX ||
          !iree_host_size_checked_mul((iree_host_size_t)payload.constant_count,
                                      sizeof(uint32_t), &constant_bytes) ||
          !iree_host_size_checked_add(sizeof(payload),
                                      payload.target_family_length,
                                      &target_key_offset) ||
          !iree_host_size_checked_add(
              target_key_offset, payload.target_key_length, &data_offset) ||
          !iree_host_size_checked_add(
              data_offset, (iree_host_size_t)payload.executable_data_length,
              &constants_offset) ||
          !iree_host_size_checked_add(constants_offset, constant_bytes,
                                      &metadata_offset) ||
          !iree_host_size_checked_add(
              metadata_offset,
              (iree_host_size_t)payload.executable_metadata_length,
              &expected_length) ||
          expected_length != record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay executable load payload is malformed");
  }

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_executable_target_selection_t target_selection = {
      .family = iree_make_string_view(
          (const char*)record->payload.data + sizeof(payload),
          payload.target_family_length),
      .target_key = iree_make_string_view(
          (const char*)record->payload.data + target_key_offset,
          payload.target_key_length),
      .kind_flags = 1u << payload.target_kind,
      .physical_device_affinity = payload.target_physical_device_affinity,
  };
  iree_hal_executable_load_params_t params;
  iree_hal_executable_load_params_initialize(&params);
  params.flags = payload.load_flags;
  params.executable_data = iree_make_const_byte_span(
      record->payload.data + data_offset,
      (iree_host_size_t)payload.executable_data_length);
  params.constant_count = (iree_host_size_t)payload.constant_count;
  iree_const_byte_span_t executable_metadata = iree_make_const_byte_span(
      record->payload.data + metadata_offset,
      (iree_host_size_t)payload.executable_metadata_length);
  if (IREE_UNLIKELY(executable_metadata.data_length == 0)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "replay executable load records require function metadata");
  }

  uint32_t* constants_storage = NULL;
  iree_hal_executable_t* executable = NULL;
  iree_host_size_t function_map_count = 0;
  iree_hal_executable_function_t* function_map = NULL;
  bool substituted = false;
  bool target_substituted = false;
  iree_string_view_t substitution_source = iree_string_view_empty();
  iree_status_t status = iree_ok_status();
  if (constant_bytes != 0) {
    status = iree_allocator_malloc(executor->host_allocator, constant_bytes,
                                   (void**)&constants_storage);
    if (iree_status_is_ok(status)) {
      memcpy(constants_storage, record->payload.data + constants_offset,
             constant_bytes);
      params.constants = constants_storage;
    }
  }
  if (iree_status_is_ok(status) &&
      executor->options->executable_substitution_callback.fn) {
    iree_hal_replay_executable_substitution_request_t request = {
        .sequence_ordinal = record->header.sequence_ordinal,
        .device_id = record->header.device_id,
        .executable_id = record->header.related_object_id,
        .captured_target = &target_selection,
        .captured_params = &params,
    };
    iree_hal_replay_executable_substitution_t substitution;
    memset(&substitution, 0, sizeof(substitution));
    status = executor->options->executable_substitution_callback.fn(
        executor->options->executable_substitution_callback.user_data, &request,
        &substitution);
    if (iree_status_is_ok(status) && substitution.substitute) {
      substituted = true;
      substitution_source = substitution.source;
      if (IREE_UNLIKELY(
              iree_const_byte_span_is_empty(substitution.executable_data))) {
        status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "substituted executable data is empty");
      } else {
        params.executable_data = substitution.executable_data;
      }
      if (iree_status_is_ok(status) &&
          !iree_hal_replay_executable_target_selection_is_empty(
              &substitution.target)) {
        target_selection = substitution.target;
        target_substituted = true;
      }
    }
  }
  const iree_hal_executable_target_t* target = NULL;
  if (iree_status_is_ok(status)) {
    const iree_hal_executable_target_flags_t required_flags =
        target_substituted ? IREE_HAL_EXECUTABLE_TARGET_FLAG_NONE
                           : payload.target_flags;
    status = iree_hal_replay_executor_select_executable_target(
        device_entry->value.device, &target_selection, required_flags, &target);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_device_load_executable(device_entry->value.device,
                                             payload.queue_affinity, target,
                                             &params, &executable);
    if (!iree_status_is_ok(status) && substituted) {
      status = iree_status_annotate_f(
          status,
          "loading substitute for captured executable %" PRIu64 " from '%.*s'",
          record->header.related_object_id, (int)substitution_source.size,
          substitution_source.data);
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_make_function_map_from_metadata(
        executor, record->header.related_object_id, executable_metadata,
        executable, &function_map_count, &function_map);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_replay_object_entry_t entry = {
        .value.executable = {
            .handle = executable,
            .function_map_count = function_map_count,
            .function_map = function_map,
        }};
    status = iree_hal_replay_executor_store(
        executor, record->header.related_object_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE, entry);
    executable = NULL;
    function_map = NULL;
    function_map_count = 0;
  } else {
    iree_hal_executable_release(executable);
    iree_allocator_free(executor->host_allocator, function_map);
  }
  iree_allocator_free(executor->host_allocator, constants_storage);
  return status;
}

static iree_status_t iree_hal_replay_executor_create_command_buffer(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_OBJECT,
      sizeof(iree_hal_replay_command_buffer_object_payload_t)));
  iree_hal_replay_command_buffer_object_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_create(
      device_entry->value.device, payload.mode, payload.command_categories,
      payload.queue_affinity, (iree_host_size_t)payload.binding_capacity,
      &command_buffer));
  iree_hal_replay_object_entry_t entry = {.value.command_buffer =
                                              command_buffer};
  return iree_hal_replay_executor_store(
      executor, record->header.related_object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER, entry);
}

static iree_status_t iree_hal_replay_executor_create_semaphore(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT,
      sizeof(iree_hal_replay_semaphore_object_payload_t)));
  iree_hal_replay_semaphore_object_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_semaphore_t* semaphore = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_semaphore_create(
      device_entry->value.device, payload.queue_affinity, payload.initial_value,
      payload.flags, &semaphore));
  iree_hal_replay_object_entry_t entry = {.value.semaphore = semaphore};
  return iree_hal_replay_executor_store(
      executor, record->header.related_object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_SEMAPHORE, entry);
}

static iree_status_t iree_hal_replay_executor_create_event(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_EVENT_OBJECT,
      sizeof(iree_hal_replay_event_object_payload_t)));
  iree_hal_replay_event_object_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));
  iree_hal_event_t* event = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_event_create(device_entry->value.device,
                                             payload.queue_affinity,
                                             payload.flags, &event));
  iree_hal_replay_object_entry_t entry = {.value.event = event};
  return iree_hal_replay_executor_store(
      executor, record->header.related_object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_EVENT, entry);
}

static iree_status_t iree_hal_replay_executor_validate_file_reference(
    const iree_hal_replay_file_object_payload_t* payload,
    iree_string_view_t captured_path, iree_string_view_t resolved_path,
    iree_io_file_handle_t* handle, iree_hal_file_t* file) {
  const uint64_t file_length = iree_hal_file_length(file);
  if (IREE_UNLIKELY(payload->file_length != file_length)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external replay file length mismatch for '%.*s' captured as '%.*s': "
        "captured=%" PRIu64 " current=%" PRIu64
        "; restore the matching file or fix --replay_file_remap",
        (int)resolved_path.size, resolved_path.data, (int)captured_path.size,
        captured_path.data, payload->file_length, file_length);
  }

  switch (payload->validation_type) {
    case IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_NONE:
      return iree_ok_status();
    case IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_IDENTITY:
      break;
    case IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_CONTENT_DIGEST:
      break;
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "external replay file validation type %" PRIu32
                              " is not executable",
                              payload->validation_type);
  }

#if IREE_FILE_IO_ENABLE && \
    (defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_LINUX))
  iree_io_file_handle_primitive_t primitive =
      iree_io_file_handle_primitive(handle);
  if (IREE_UNLIKELY(primitive.type != IREE_IO_FILE_HANDLE_TYPE_FD)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external replay file validation requires an fd-backed file");
  }
  if (payload->validation_type ==
      IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_CONTENT_DIGEST) {
    if (IREE_UNLIKELY(payload->digest_type !=
                      IREE_HAL_REPLAY_DIGEST_TYPE_FNV1A_64)) {
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "external replay file digest type %" PRIu32
                              " is not executable",
                              (uint32_t)payload->digest_type);
    }
    uint64_t state = iree_hal_replay_digest_fnv1a64_initialize();
    uint64_t offset = 0;
    uint8_t buffer[64 * 1024];
    while (offset < payload->file_length) {
      uint64_t chunk_length = payload->file_length - offset;
      if (chunk_length > sizeof(buffer)) chunk_length = sizeof(buffer);
      ssize_t read_length = pread(primitive.value.fd, buffer,
                                  (size_t)chunk_length, (off_t)offset);
      if (read_length < 0 && errno == EINTR) continue;
      if (read_length <= 0) {
        return iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "unable to read external replay file '%.*s' for digest validation",
            (int)resolved_path.size, resolved_path.data);
      }
      state = iree_hal_replay_digest_fnv1a64_update(
          state,
          iree_make_const_byte_span(buffer, (iree_host_size_t)read_length));
      offset += (uint64_t)read_length;
    }
    const uint64_t expected_digest =
        iree_hal_replay_digest_load_fnv1a64(payload->digest);
    if (IREE_UNLIKELY(state != expected_digest)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "external replay file digest mismatch for '%.*s' captured as "
          "'%.*s': expected=0x%016" PRIx64 " actual=0x%016" PRIx64
          "; restore the matching file or fix --replay_file_remap",
          (int)resolved_path.size, resolved_path.data, (int)captured_path.size,
          captured_path.data, expected_digest, state);
    }
    return iree_ok_status();
  }

  if (payload->file_device == 0 && payload->file_inode == 0 &&
      payload->file_mtime_ns == 0) {
    return iree_ok_status();
  }
  struct stat file_stat;
  if (IREE_UNLIKELY(fstat(primitive.value.fd, &file_stat) != 0)) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "unable to stat external replay file '%.*s'",
                            (int)resolved_path.size, resolved_path.data);
  }
  const uint64_t file_device = (uint64_t)file_stat.st_dev;
  const uint64_t file_inode = (uint64_t)file_stat.st_ino;
  const uint64_t file_mtime_ns =
      ((uint64_t)file_stat.st_mtim.tv_sec * 1000000000ull) +
      (uint64_t)file_stat.st_mtim.tv_nsec;
  if (IREE_UNLIKELY(payload->file_device != file_device ||
                    payload->file_inode != file_inode ||
                    payload->file_mtime_ns != file_mtime_ns)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "external replay file identity mismatch for '%.*s' captured as '%.*s': "
        "captured=(dev=%" PRIu64 ", inode=%" PRIu64 ", mtime_ns=%" PRIu64
        ") current=(dev=%" PRIu64 ", inode=%" PRIu64 ", mtime_ns=%" PRIu64
        "); identity validation is the default and does not read file "
        "contents; restore the original file identity, fix "
        "--replay_file_remap, or recapture with "
        "--device_replay_file_validation=digest when copied/staged files are "
        "intentional",
        (int)resolved_path.size, resolved_path.data, (int)captured_path.size,
        captured_path.data, payload->file_device, payload->file_inode,
        payload->file_mtime_ns, file_device, file_inode, file_mtime_ns);
  }
#else
  (void)handle;
  (void)captured_path;
  (void)resolved_path;
  if (payload->validation_type ==
      IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_CONTENT_DIGEST) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "external replay file content-digest validation requires POSIX file "
        "IO");
  }
#endif  // IREE_FILE_IO_ENABLE && (IREE_PLATFORM_ANDROID ||
        // IREE_PLATFORM_LINUX)
  return iree_ok_status();
}

static iree_status_t iree_hal_replay_executor_resolve_file_path(
    iree_hal_replay_executor_t* executor, iree_string_view_t captured_path,
    iree_string_view_t* out_resolved_path, char** out_resolved_path_storage) {
  IREE_ASSERT_ARGUMENT(out_resolved_path);
  IREE_ASSERT_ARGUMENT(out_resolved_path_storage);
  *out_resolved_path = captured_path;
  *out_resolved_path_storage = NULL;

  const iree_hal_replay_file_path_remap_t* selected_remap = NULL;
  for (iree_host_size_t i = 0; i < executor->options->file_path_remap_count;
       ++i) {
    const iree_hal_replay_file_path_remap_t* remap =
        &executor->options->file_path_remaps[i];
    if (iree_string_view_is_empty(remap->captured_prefix)) continue;
    if (!iree_string_view_starts_with(captured_path, remap->captured_prefix)) {
      continue;
    }
    if (!selected_remap ||
        remap->captured_prefix.size > selected_remap->captured_prefix.size) {
      selected_remap = remap;
    }
  }
  if (!selected_remap) return iree_ok_status();

  iree_string_view_t captured_suffix = iree_string_view_remove_prefix(
      captured_path, selected_remap->captured_prefix.size);
  iree_host_size_t resolved_length = 0;
  if (IREE_UNLIKELY(
          !iree_host_size_checked_add(selected_remap->replay_prefix.size,
                                      captured_suffix.size, &resolved_length) ||
          resolved_length == IREE_HOST_SIZE_MAX)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "remapped replay file path is too long");
  }
  char* resolved_path_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(executor->host_allocator,
                                             resolved_length + 1,
                                             (void**)&resolved_path_storage));
  memcpy(resolved_path_storage, selected_remap->replay_prefix.data,
         selected_remap->replay_prefix.size);
  memcpy(resolved_path_storage + selected_remap->replay_prefix.size,
         captured_suffix.data, captured_suffix.size);
  resolved_path_storage[resolved_length] = 0;
  *out_resolved_path =
      iree_make_string_view(resolved_path_storage, resolved_length);
  *out_resolved_path_storage = resolved_path_storage;
  return iree_ok_status();
}

typedef struct iree_hal_replay_executor_inline_file_release_t {
  // Host allocator used for the inline file copy and this release record.
  iree_allocator_t host_allocator;
} iree_hal_replay_executor_inline_file_release_t;

static void iree_hal_replay_executor_inline_file_release(
    void* user_data, iree_io_file_handle_primitive_t handle_primitive) {
  iree_hal_replay_executor_inline_file_release_t* release =
      (iree_hal_replay_executor_inline_file_release_t*)user_data;
  iree_allocator_t host_allocator = release->host_allocator;
  iree_allocator_free(host_allocator,
                      handle_primitive.value.host_allocation.data);
  iree_allocator_free(host_allocator, release);
}

static iree_io_file_access_t iree_hal_replay_executor_make_file_access(
    iree_hal_memory_access_t access) {
  iree_io_file_access_t file_access = 0;
  if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_READ)) {
    file_access |= IREE_IO_FILE_ACCESS_READ;
  }
  if (iree_any_bit_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
    file_access |= IREE_IO_FILE_ACCESS_WRITE;
  }
  return file_access ? file_access : IREE_IO_FILE_ACCESS_READ;
}

static iree_status_t iree_hal_replay_executor_wrap_inline_file(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_object_payload_t* payload,
    iree_const_byte_span_t reference, iree_io_file_handle_t** out_handle) {
  IREE_ASSERT_ARGUMENT(out_handle);
  *out_handle = NULL;

  if (IREE_UNLIKELY(payload->file_length != reference.data_length)) {
    return iree_make_status(
        IREE_STATUS_DATA_LOSS,
        "inline replay file length mismatch: file_length=%" PRIu64
        " reference_length=%" PRIhsz,
        payload->file_length, reference.data_length);
  }

  iree_hal_replay_executor_inline_file_release_t* release = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      executor->host_allocator, sizeof(*release), (void**)&release));
  release->host_allocator = executor->host_allocator;

  uint8_t* file_bytes = NULL;
  iree_status_t status = iree_ok_status();
  if (reference.data_length != 0) {
    status = iree_allocator_malloc(executor->host_allocator,
                                   reference.data_length, (void**)&file_bytes);
    if (iree_status_is_ok(status)) {
      memcpy(file_bytes, reference.data, reference.data_length);
    }
  }
  iree_io_file_handle_release_callback_t release_callback = {
      .fn = iree_hal_replay_executor_inline_file_release,
      .user_data = release,
  };
  if (iree_status_is_ok(status)) {
    status = iree_io_file_handle_wrap_host_allocation(
        iree_hal_replay_executor_make_file_access(payload->access),
        iree_make_byte_span(file_bytes, reference.data_length),
        release_callback, executor->host_allocator, out_handle);
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(executor->host_allocator, file_bytes);
    iree_allocator_free(executor->host_allocator, release);
  }
  return status;
}

static iree_status_t iree_hal_replay_executor_import_file(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_FILE_OBJECT,
      sizeof(iree_hal_replay_file_object_payload_t)));
  iree_hal_replay_file_object_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.reference_length > IREE_HOST_SIZE_MAX ||
                    sizeof(payload) +
                            (iree_host_size_t)payload.reference_length !=
                        record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay file object payload length mismatch");
  }

  iree_hal_replay_object_entry_t* device_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE,
      &device_entry));

  iree_const_byte_span_t reference =
      iree_make_const_byte_span(record->payload.data + sizeof(payload),
                                (iree_host_size_t)payload.reference_length);
  iree_io_file_handle_t* handle = NULL;
  char* resolved_path_storage = NULL;
  iree_string_view_t captured_external_path = iree_string_view_empty();
  iree_string_view_t resolved_external_path = iree_string_view_empty();
  bool import_file_handle = true;
  iree_status_t status = iree_ok_status();
  switch (payload.reference_type) {
    case IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_EXTERNAL_PATH: {
      iree_string_view_t path = iree_make_string_view(
          (const char*)reference.data, reference.data_length);
      captured_external_path = path;
      status = iree_hal_replay_executor_resolve_file_path(
          executor, path, &path, &resolved_path_storage);
      resolved_external_path = path;
      iree_io_file_mode_t mode = 0;
      if (iree_any_bit_set(payload.access, IREE_HAL_MEMORY_ACCESS_READ)) {
        mode |= IREE_IO_FILE_MODE_READ;
      }
      if (iree_any_bit_set(payload.access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
        mode |= IREE_IO_FILE_MODE_WRITE;
      }
      if (mode == 0) mode = IREE_IO_FILE_MODE_READ;
      if (iree_status_is_ok(status)) {
        status = iree_io_file_handle_open(mode, path, executor->host_allocator,
                                          &handle);
        if (!iree_status_is_ok(status)) {
          status = iree_status_annotate_f(
              status,
              "opening external replay file '%.*s' captured as '%.*s'; use "
              "--replay_file_remap=CAPTURED_PREFIX=REPLAY_PREFIX if the "
              "parameter root moved",
              (int)path.size, path.data, (int)captured_external_path.size,
              captured_external_path.data);
        }
      }
      break;
    }
    case IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_INLINE_BYTES:
      status = iree_hal_replay_executor_wrap_inline_file(executor, &payload,
                                                         reference, &handle);
      break;
    case IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_CAPTURED_RANGES:
      captured_external_path = iree_make_string_view(
          (const char*)reference.data, reference.data_length);
      import_file_handle = false;
      break;
    default:
      status = iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                "replay file reference type %" PRIu32
                                " is not executable",
                                payload.reference_type);
      break;
  }

  iree_hal_file_t* file = NULL;
  if (iree_status_is_ok(status) && import_file_handle) {
    status =
        iree_hal_file_import(device_entry->value.device, payload.queue_affinity,
                             payload.access, handle, payload.flags, &file);
  }
  if (iree_status_is_ok(status) &&
      payload.reference_type ==
          IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_EXTERNAL_PATH) {
    status = iree_hal_replay_executor_validate_file_reference(
        &payload, captured_external_path, resolved_external_path, handle, file);
  }
  iree_io_file_handle_release(handle);
  iree_allocator_free(executor->host_allocator, resolved_path_storage);

  if (iree_status_is_ok(status)) {
    iree_hal_replay_object_entry_t entry = {.value.file = file};
    status = iree_hal_replay_executor_store(
        executor, record->header.related_object_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_FILE, entry);
  } else {
    iree_hal_file_release(file);
  }
  return status;
}

static iree_status_t iree_hal_replay_executor_allocate_buffer(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_ALLOCATE_BUFFER,
      sizeof(iree_hal_replay_allocator_allocate_buffer_payload_t)));
  iree_hal_replay_allocator_allocate_buffer_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  iree_hal_replay_object_entry_t* allocator_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR,
      &allocator_entry));
  iree_hal_buffer_params_t params;
  IREE_RETURN_IF_ERROR(
      iree_hal_replay_executor_make_buffer_params(&payload, &params));
  iree_hal_buffer_t* buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      allocator_entry->value.allocator, params, payload.allocation_size,
      &buffer));
  iree_hal_replay_object_entry_t entry = {.value.buffer = buffer};
  return iree_hal_replay_executor_store(
      executor, record->header.related_object_id,
      IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, entry);
}

static iree_status_t iree_hal_replay_executor_import_buffer(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_IMPORT_BUFFER,
      sizeof(iree_hal_replay_allocator_import_buffer_payload_t)));
  iree_hal_replay_allocator_import_buffer_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.external_type !=
                    IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "replay import buffer external type %" PRIu32
                            " is not supported",
                            payload.external_type);
  }
  if (IREE_UNLIKELY(payload.data_length > IREE_HOST_SIZE_MAX ||
                    sizeof(payload) + (iree_host_size_t)payload.data_length !=
                        record->payload.data_length)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay import buffer payload length mismatch");
  }
  if (IREE_UNLIKELY(payload.data_length > payload.allocation.allocation_size)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay import buffer data overflows allocation");
  }
  iree_hal_replay_object_entry_t* allocator_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR,
      &allocator_entry));

  iree_hal_buffer_params_t params;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_make_buffer_params(
      &payload.allocation, &params));
  params.type |= IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params.access |= IREE_HAL_MEMORY_ACCESS_WRITE;
  params.usage |= IREE_HAL_BUFFER_USAGE_MAPPING;

  iree_hal_buffer_t* buffer = NULL;
  iree_status_t status = iree_hal_allocator_allocate_buffer(
      allocator_entry->value.allocator, params,
      payload.allocation.allocation_size, &buffer);
  iree_const_byte_span_t data =
      iree_make_const_byte_span(record->payload.data + sizeof(payload),
                                (iree_host_size_t)payload.data_length);
  if (iree_status_is_ok(status)) {
    status = iree_hal_replay_executor_write_buffer_data(
        buffer, /*byte_offset=*/0, payload.allocation.allocation_size,
        IREE_HAL_MEMORY_ACCESS_WRITE, data);
  }
  if (iree_status_is_ok(status)) {
    iree_hal_replay_object_entry_t entry = {.value.buffer = buffer};
    status = iree_hal_replay_executor_store(
        executor, record->header.related_object_id,
        IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER, entry);
  } else {
    iree_hal_buffer_release(buffer);
  }
  return status;
}

static iree_status_t iree_hal_replay_executor_replay_buffer_range_data(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_require_payload(
      record, IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA,
      sizeof(iree_hal_replay_buffer_range_data_payload_t)));
  iree_hal_replay_buffer_range_data_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (IREE_UNLIKELY(payload.data_length >
                    record->payload.data_length - sizeof(payload))) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay buffer range data overflows payload");
  }
  iree_hal_replay_object_entry_t* buffer_entry = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_replay_executor_lookup(
      executor, record->header.object_id, IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
      &buffer_entry));
  iree_const_byte_span_t data =
      iree_make_const_byte_span(record->payload.data + sizeof(payload),
                                (iree_host_size_t)payload.data_length);
  return iree_hal_replay_executor_write_buffer_data(
      buffer_entry->value.buffer, payload.byte_offset, payload.byte_length,
      payload.memory_access, data);
}

iree_status_t iree_hal_replay_executor_replay_object_operation(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record) {
  switch (record->header.operation_code) {
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_LOAD_EXECUTABLE:
      return iree_hal_replay_executor_load_executable(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_COMMAND_BUFFER:
      return iree_hal_replay_executor_create_command_buffer(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_EVENT:
      return iree_hal_replay_executor_create_event(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_IMPORT_FILE:
      return iree_hal_replay_executor_import_file(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_SEMAPHORE:
      return iree_hal_replay_executor_create_semaphore(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_ALLOCATE_BUFFER:
      return iree_hal_replay_executor_allocate_buffer(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_IMPORT_BUFFER:
      return iree_hal_replay_executor_import_buffer(executor, record);
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_FLUSH_RANGE:
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_UNMAP_RANGE:
      if (record->header.payload_type ==
          IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA) {
        return iree_hal_replay_executor_replay_buffer_range_data(executor,
                                                                 record);
      }
      return iree_ok_status();
    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "replay object operation %s is not implemented",
          iree_hal_replay_operation_code_string(record->header.operation_code));
  }
}
