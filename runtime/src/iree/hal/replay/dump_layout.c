// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/dump_layout.h"

#include <inttypes.h>
#include <string.h>

iree_status_t iree_hal_replay_dump_payload_length_check(
    const iree_hal_replay_file_record_t* record,
    iree_host_size_t expected_payload_length) {
  if (IREE_LIKELY(record->payload.data_length == expected_payload_length)) {
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_DATA_LOSS,
                          "replay payload type %u has %" PRIhsz
                          " bytes; expected %" PRIhsz,
                          record->header.payload_type,
                          record->payload.data_length, expected_payload_length);
}

iree_hal_replay_file_range_t iree_hal_replay_dump_record_payload_range(
    const iree_hal_replay_file_record_t* record,
    iree_host_size_t record_offset) {
  iree_hal_replay_file_range_t range = iree_hal_replay_file_range_empty();
  range.offset = (uint64_t)record_offset + record->header.header_length;
  range.length = record->header.payload_length;
  range.uncompressed_length = record->header.payload_length;
  range.compression_type = IREE_HAL_REPLAY_COMPRESSION_TYPE_NONE;
  range.digest_type = IREE_HAL_REPLAY_DIGEST_TYPE_NONE;
  return range;
}

const char* iree_hal_replay_dump_file_reference_type_string(
    iree_hal_replay_file_reference_type_t reference_type) {
  switch (reference_type) {
    case IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_NONE:
      return "none";
    case IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_EXTERNAL_PATH:
      return "external_path";
    case IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_INLINE_BYTES:
      return "inline_bytes";
    case IREE_HAL_REPLAY_FILE_REFERENCE_TYPE_CAPTURED_RANGES:
      return "captured_ranges";
    default:
      return "unknown";
  }
}

const char* iree_hal_replay_dump_file_validation_type_string(
    iree_hal_replay_file_validation_type_t validation_type) {
  switch (validation_type) {
    case IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_NONE:
      return "none";
    case IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_IDENTITY:
      return "identity";
    case IREE_HAL_REPLAY_FILE_VALIDATION_TYPE_CONTENT_DIGEST:
      return "digest";
    default:
      return "unknown";
  }
}

iree_hal_replay_file_range_t iree_hal_replay_dump_payload_subrange(
    const iree_hal_replay_file_range_t* payload_range,
    iree_host_size_t payload_offset, iree_host_size_t payload_length) {
  iree_hal_replay_file_range_t range = iree_hal_replay_file_range_empty();
  range.offset = payload_range->offset + payload_offset;
  range.length = payload_length;
  range.uncompressed_length = payload_length;
  range.compression_type = IREE_HAL_REPLAY_COMPRESSION_TYPE_NONE;
  range.digest_type = IREE_HAL_REPLAY_DIGEST_TYPE_NONE;
  return range;
}

iree_status_t iree_hal_replay_dump_compute_executable_load_ranges(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_executable_load_payload_t* payload,
    iree_hal_replay_dump_executable_load_ranges_t* out_ranges) {
  memset(out_ranges, 0, sizeof(*out_ranges));
  if (payload->executable_data_length > IREE_HOST_SIZE_MAX ||
      payload->constant_count > IREE_HOST_SIZE_MAX ||
      !iree_host_size_checked_mul((iree_host_size_t)payload->constant_count,
                                  sizeof(uint32_t),
                                  &out_ranges->constant_bytes)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay executable load payload overflow");
  }
  out_ranges->target_family_offset = sizeof(*payload);
  iree_host_size_t expected_length = 0;
  if (!iree_host_size_checked_add(
          out_ranges->target_family_offset,
          (iree_host_size_t)payload->target_family_length,
          &out_ranges->target_key_offset) ||
      !iree_host_size_checked_add(out_ranges->target_key_offset,
                                  (iree_host_size_t)payload->target_key_length,
                                  &out_ranges->data_offset) ||
      !iree_host_size_checked_add(
          out_ranges->data_offset,
          (iree_host_size_t)payload->executable_data_length,
          &out_ranges->constants_offset) ||
      !iree_host_size_checked_add(out_ranges->constants_offset,
                                  out_ranges->constant_bytes,
                                  &out_ranges->metadata_offset) ||
      !iree_host_size_checked_add(
          out_ranges->metadata_offset,
          (iree_host_size_t)payload->executable_metadata_length,
          &expected_length) ||
      expected_length != record->payload.data_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay executable load payload length "
                            "mismatch");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_replay_dump_read_executable_metadata_header(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_executable_load_payload_t* payload,
    const iree_hal_replay_dump_executable_load_ranges_t* ranges,
    bool* out_has_metadata,
    iree_hal_replay_executable_metadata_header_t* out_header) {
  *out_has_metadata = false;
  if (payload->executable_metadata_length <
      sizeof(iree_hal_replay_executable_metadata_header_t)) {
    if (payload->executable_metadata_length == 0) return iree_ok_status();
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay executable metadata is too short");
  }
  memcpy(out_header, record->payload.data + ranges->metadata_offset,
         sizeof(*out_header));
  if (out_header->reserved1 != 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay executable metadata reserved fields must "
                            "be zero");
  }
  if (out_header->function_count > IREE_HOST_SIZE_MAX ||
      out_header->parameter_count > IREE_HOST_SIZE_MAX ||
      out_header->function_name_storage_length > IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay executable metadata count overflow");
  }
  iree_host_size_t function_metadata_size = 0;
  iree_host_size_t parameter_metadata_size = 0;
  iree_host_size_t expected_length = 0;
  if (!iree_host_size_checked_mul(
          (iree_host_size_t)out_header->function_count,
          sizeof(iree_hal_replay_executable_function_metadata_t),
          &function_metadata_size) ||
      !iree_host_size_checked_mul(
          (iree_host_size_t)out_header->parameter_count,
          sizeof(iree_hal_replay_executable_parameter_metadata_t),
          &parameter_metadata_size) ||
      !iree_host_size_checked_add(
          sizeof(iree_hal_replay_executable_metadata_header_t),
          function_metadata_size, &expected_length) ||
      !iree_host_size_checked_add(expected_length, parameter_metadata_size,
                                  &expected_length) ||
      !iree_host_size_checked_add(
          expected_length,
          (iree_host_size_t)out_header->function_name_storage_length,
          &expected_length) ||
      expected_length != payload->executable_metadata_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay executable metadata length mismatch");
  }
  *out_has_metadata = true;
  return iree_ok_status();
}

iree_status_t iree_hal_replay_dump_dispatch_layout(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_dispatch_payload_t* payload,
    iree_host_size_t* out_wait_payloads_offset,
    iree_host_size_t* out_wait_payloads_size,
    iree_host_size_t* out_signal_payloads_offset,
    iree_host_size_t* out_signal_payloads_size,
    iree_host_size_t* out_constants_offset,
    iree_host_size_t* out_binding_payloads_offset,
    iree_host_size_t* out_binding_payloads_size) {
  iree_host_size_t wait_payloads_size = 0;
  iree_host_size_t signal_payloads_size = 0;
  iree_host_size_t binding_payloads_size = 0;
  if (payload->wait_semaphore_count > IREE_HOST_SIZE_MAX ||
      payload->signal_semaphore_count > IREE_HOST_SIZE_MAX ||
      payload->binding_count > IREE_HOST_SIZE_MAX ||
      payload->constants_length > IREE_HOST_SIZE_MAX ||
      !iree_host_size_checked_mul(
          (iree_host_size_t)payload->wait_semaphore_count,
          sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
          &wait_payloads_size) ||
      !iree_host_size_checked_mul(
          (iree_host_size_t)payload->signal_semaphore_count,
          sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
          &signal_payloads_size) ||
      !iree_host_size_checked_mul((iree_host_size_t)payload->binding_count,
                                  sizeof(iree_hal_replay_buffer_ref_payload_t),
                                  &binding_payloads_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay dispatch payload count overflow");
  }

  iree_host_size_t offset = sizeof(*payload);
  *out_wait_payloads_offset = offset;
  *out_wait_payloads_size = wait_payloads_size;
  if (!iree_host_size_checked_add(offset, wait_payloads_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay dispatch payload length overflow");
  }
  *out_signal_payloads_offset = offset;
  *out_signal_payloads_size = signal_payloads_size;
  if (!iree_host_size_checked_add(offset, signal_payloads_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay dispatch payload length overflow");
  }
  *out_constants_offset = offset;
  if (!iree_host_size_checked_add(
          offset, (iree_host_size_t)payload->constants_length, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay dispatch payload length overflow");
  }
  *out_binding_payloads_offset = offset;
  *out_binding_payloads_size = binding_payloads_size;
  if (!iree_host_size_checked_add(offset, binding_payloads_size, &offset) ||
      offset != record->payload.data_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay dispatch payload length mismatch");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_replay_dump_queue_execute_layout(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_device_queue_execute_payload_t* payload,
    iree_host_size_t* out_wait_payloads_offset,
    iree_host_size_t* out_wait_payloads_size,
    iree_host_size_t* out_signal_payloads_offset,
    iree_host_size_t* out_signal_payloads_size,
    iree_host_size_t* out_binding_payloads_offset,
    iree_host_size_t* out_binding_payloads_size) {
  iree_host_size_t wait_payloads_size = 0;
  iree_host_size_t signal_payloads_size = 0;
  iree_host_size_t binding_payloads_size = 0;
  if (payload->wait_semaphore_count > IREE_HOST_SIZE_MAX ||
      payload->signal_semaphore_count > IREE_HOST_SIZE_MAX ||
      payload->binding_count > IREE_HOST_SIZE_MAX ||
      !iree_host_size_checked_mul(
          (iree_host_size_t)payload->wait_semaphore_count,
          sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
          &wait_payloads_size) ||
      !iree_host_size_checked_mul(
          (iree_host_size_t)payload->signal_semaphore_count,
          sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
          &signal_payloads_size) ||
      !iree_host_size_checked_mul((iree_host_size_t)payload->binding_count,
                                  sizeof(iree_hal_replay_buffer_ref_payload_t),
                                  &binding_payloads_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue execute payload count overflow");
  }

  iree_host_size_t offset = sizeof(*payload);
  *out_wait_payloads_offset = offset;
  *out_wait_payloads_size = wait_payloads_size;
  if (!iree_host_size_checked_add(offset, wait_payloads_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue execute payload length overflow");
  }
  *out_signal_payloads_offset = offset;
  *out_signal_payloads_size = signal_payloads_size;
  if (!iree_host_size_checked_add(offset, signal_payloads_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue execute payload length overflow");
  }
  *out_binding_payloads_offset = offset;
  *out_binding_payloads_size = binding_payloads_size;
  if (!iree_host_size_checked_add(offset, binding_payloads_size, &offset) ||
      offset != record->payload.data_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay queue execute payload length mismatch");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_replay_dump_queue_alloca_layout(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_device_queue_alloca_payload_t* payload,
    iree_host_size_t* out_wait_payloads_offset,
    iree_host_size_t* out_wait_payloads_size,
    iree_host_size_t* out_signal_payloads_offset,
    iree_host_size_t* out_signal_payloads_size) {
  iree_host_size_t wait_payloads_size = 0;
  iree_host_size_t signal_payloads_size = 0;
  if (payload->wait_semaphore_count > IREE_HOST_SIZE_MAX ||
      payload->signal_semaphore_count > IREE_HOST_SIZE_MAX ||
      !iree_host_size_checked_mul(
          (iree_host_size_t)payload->wait_semaphore_count,
          sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
          &wait_payloads_size) ||
      !iree_host_size_checked_mul(
          (iree_host_size_t)payload->signal_semaphore_count,
          sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
          &signal_payloads_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue alloca payload count overflow");
  }

  iree_host_size_t offset = sizeof(*payload);
  *out_wait_payloads_offset = offset;
  *out_wait_payloads_size = wait_payloads_size;
  if (!iree_host_size_checked_add(offset, wait_payloads_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue alloca payload length overflow");
  }
  *out_signal_payloads_offset = offset;
  *out_signal_payloads_size = signal_payloads_size;
  if (!iree_host_size_checked_add(offset, signal_payloads_size, &offset) ||
      offset != record->payload.data_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay queue alloca payload length mismatch");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_replay_dump_queue_payload_layout(
    const iree_hal_replay_file_record_t* record, iree_host_size_t header_size,
    uint64_t wait_semaphore_count, uint64_t signal_semaphore_count,
    uint64_t trailing_payload_length,
    iree_host_size_t* out_wait_payloads_offset,
    iree_host_size_t* out_wait_payloads_size,
    iree_host_size_t* out_signal_payloads_offset,
    iree_host_size_t* out_signal_payloads_size,
    iree_host_size_t* out_trailing_payload_offset,
    iree_host_size_t* out_trailing_payload_size) {
  iree_host_size_t wait_payloads_size = 0;
  iree_host_size_t signal_payloads_size = 0;
  if (wait_semaphore_count > IREE_HOST_SIZE_MAX ||
      signal_semaphore_count > IREE_HOST_SIZE_MAX ||
      trailing_payload_length > IREE_HOST_SIZE_MAX ||
      !iree_host_size_checked_mul(
          (iree_host_size_t)wait_semaphore_count,
          sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
          &wait_payloads_size) ||
      !iree_host_size_checked_mul(
          (iree_host_size_t)signal_semaphore_count,
          sizeof(iree_hal_replay_semaphore_timepoint_payload_t),
          &signal_payloads_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue payload count overflow");
  }

  iree_host_size_t offset = header_size;
  *out_wait_payloads_offset = offset;
  *out_wait_payloads_size = wait_payloads_size;
  if (!iree_host_size_checked_add(offset, wait_payloads_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue payload length overflow");
  }
  *out_signal_payloads_offset = offset;
  *out_signal_payloads_size = signal_payloads_size;
  if (!iree_host_size_checked_add(offset, signal_payloads_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay queue payload length overflow");
  }
  *out_trailing_payload_offset = offset;
  *out_trailing_payload_size = (iree_host_size_t)trailing_payload_length;
  if (!iree_host_size_checked_add(
          offset, (iree_host_size_t)trailing_payload_length, &offset) ||
      offset != record->payload.data_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay queue payload length mismatch");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_replay_dump_execution_barrier_layout(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_command_buffer_execution_barrier_payload_t* payload,
    iree_host_size_t* out_memory_barriers_offset,
    iree_host_size_t* out_memory_barriers_size,
    iree_host_size_t* out_buffer_barriers_offset,
    iree_host_size_t* out_buffer_barriers_size) {
  iree_host_size_t memory_barriers_size = 0;
  iree_host_size_t buffer_barriers_size = 0;
  if (payload->memory_barrier_count > IREE_HOST_SIZE_MAX ||
      payload->buffer_barrier_count > IREE_HOST_SIZE_MAX ||
      !iree_host_size_checked_mul(
          (iree_host_size_t)payload->memory_barrier_count,
          sizeof(iree_hal_replay_memory_barrier_payload_t),
          &memory_barriers_size) ||
      !iree_host_size_checked_mul(
          (iree_host_size_t)payload->buffer_barrier_count,
          sizeof(iree_hal_replay_buffer_barrier_payload_t),
          &buffer_barriers_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay execution barrier payload count overflow");
  }

  iree_host_size_t offset = sizeof(*payload);
  *out_memory_barriers_offset = offset;
  *out_memory_barriers_size = memory_barriers_size;
  if (!iree_host_size_checked_add(offset, memory_barriers_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay execution barrier payload length overflow");
  }
  *out_buffer_barriers_offset = offset;
  *out_buffer_barriers_size = buffer_barriers_size;
  if (!iree_host_size_checked_add(offset, buffer_barriers_size, &offset) ||
      offset != record->payload.data_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay execution barrier payload length mismatch");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_replay_dump_wait_events_layout(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_command_buffer_wait_events_payload_t* payload,
    iree_host_size_t* out_events_offset, iree_host_size_t* out_events_size,
    iree_host_size_t* out_memory_barriers_offset,
    iree_host_size_t* out_memory_barriers_size,
    iree_host_size_t* out_buffer_barriers_offset,
    iree_host_size_t* out_buffer_barriers_size) {
  iree_host_size_t events_size = 0;
  iree_host_size_t memory_barriers_size = 0;
  iree_host_size_t buffer_barriers_size = 0;
  if (IREE_UNLIKELY(payload->event_count > IREE_HOST_SIZE_MAX ||
                    payload->memory_barrier_count > IREE_HOST_SIZE_MAX ||
                    payload->buffer_barrier_count > IREE_HOST_SIZE_MAX ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload->event_count,
                        sizeof(iree_hal_replay_object_id_t), &events_size) ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload->memory_barrier_count,
                        sizeof(iree_hal_replay_memory_barrier_payload_t),
                        &memory_barriers_size) ||
                    !iree_host_size_checked_mul(
                        (iree_host_size_t)payload->buffer_barrier_count,
                        sizeof(iree_hal_replay_buffer_barrier_payload_t),
                        &buffer_barriers_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay wait events payload count overflow");
  }

  iree_host_size_t offset = sizeof(*payload);
  *out_events_offset = offset;
  *out_events_size = events_size;
  if (!iree_host_size_checked_add(offset, events_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay wait events payload length overflow");
  }
  *out_memory_barriers_offset = offset;
  *out_memory_barriers_size = memory_barriers_size;
  if (!iree_host_size_checked_add(offset, memory_barriers_size, &offset)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay wait events payload length overflow");
  }
  *out_buffer_barriers_offset = offset;
  *out_buffer_barriers_size = buffer_barriers_size;
  if (!iree_host_size_checked_add(offset, buffer_barriers_size, &offset) ||
      offset != record->payload.data_length) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay wait events payload length mismatch");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_replay_dump_scope_name(
    const iree_hal_replay_file_record_t* record,
    iree_string_view_t* out_scope_name) {
  *out_scope_name = iree_string_view_empty();
  if (record->payload.data_length < sizeof(iree_hal_replay_scope_payload_t)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay scope payload is short");
  }
  iree_hal_replay_scope_payload_t payload;
  memcpy(&payload, record->payload.data, sizeof(payload));
  if (payload.flags != IREE_HAL_REPLAY_SCOPE_FLAG_NONE ||
      payload.reserved0 != 0 || payload.reserved1 != 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay scope payload reserved fields must be "
                            "zero");
  }
  if (payload.name_length > IREE_HOST_SIZE_MAX ||
      sizeof(payload) + (iree_host_size_t)payload.name_length !=
          record->payload.data_length ||
      payload.name_length == 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "replay scope payload name length mismatch");
  }
  *out_scope_name =
      iree_make_string_view((const char*)record->payload.data + sizeof(payload),
                            (iree_host_size_t)payload.name_length);
  return iree_ok_status();
}
