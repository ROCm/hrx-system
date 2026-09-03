// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_DUMP_LAYOUT_H_
#define IREE_HAL_REPLAY_DUMP_LAYOUT_H_

#include "iree/hal/atomic.h"
#include "iree/hal/replay/file_reader.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Byte ranges within a validated executable-load payload.
typedef struct iree_hal_replay_dump_executable_load_ranges_t {
  // Byte offset of the target family string within the record payload.
  iree_host_size_t target_family_offset;
  // Byte offset of the target key string within the record payload.
  iree_host_size_t target_key_offset;
  // Byte offset of the executable data blob within the record payload.
  iree_host_size_t data_offset;
  // Byte offset of the specialization constants within the record payload.
  iree_host_size_t constants_offset;
  // Byte offset of the executable ABI metadata within the record payload.
  iree_host_size_t metadata_offset;
  // Byte length of the specialization constants.
  iree_host_size_t constant_bytes;
} iree_hal_replay_dump_executable_load_ranges_t;

// Byte ranges within a validated queue operation payload.
typedef struct iree_hal_replay_dump_queue_payload_layout_t {
  // Byte offset of the wait semaphore payloads.
  iree_host_size_t wait_payloads_offset;
  // Byte length of the wait semaphore payloads.
  iree_host_size_t wait_payloads_size;
  // Byte offset of the signal semaphore payloads.
  iree_host_size_t signal_payloads_offset;
  // Byte length of the signal semaphore payloads.
  iree_host_size_t signal_payloads_size;
  // Byte offset of the operation-specific trailing payload.
  iree_host_size_t trailing_payload_offset;
  // Byte length of the operation-specific trailing payload.
  iree_host_size_t trailing_payload_size;
} iree_hal_replay_dump_queue_payload_layout_t;

// Byte ranges within a validated exact-queue transfer payload.
typedef struct iree_hal_replay_dump_queue_transfer_layout_t {
  // Common semaphore and trailing payload ranges.
  iree_hal_replay_dump_queue_payload_layout_t queue;
  // Byte offset of the transfer operation descriptors.
  iree_host_size_t operation_payloads_offset;
  // Byte length of the transfer operation descriptors.
  iree_host_size_t operation_payloads_size;
  // Byte offset of the captured operation data.
  iree_host_size_t data_offset;
  // Byte length of the captured operation data.
  iree_host_size_t data_size;
} iree_hal_replay_dump_queue_transfer_layout_t;

iree_status_t iree_hal_replay_dump_payload_length_check(
    const iree_hal_replay_file_record_t* record,
    iree_host_size_t expected_payload_length);

iree_hal_replay_file_range_t iree_hal_replay_dump_record_payload_range(
    const iree_hal_replay_file_record_t* record,
    iree_host_size_t record_offset);

const char* iree_hal_replay_dump_file_reference_type_string(
    iree_hal_replay_file_reference_type_t reference_type);

const char* iree_hal_replay_dump_file_validation_type_string(
    iree_hal_replay_file_validation_type_t validation_type);

const char* iree_hal_replay_dump_atomic_wait_condition_string(
    iree_hal_atomic_wait_condition_t condition);

const char* iree_hal_replay_dump_atomic_rmw_operation_string(
    iree_hal_atomic_rmw_operation_t operation);

const char* iree_hal_replay_dump_queue_transfer_operation_type_string(
    iree_hal_replay_queue_transfer_operation_type_t operation_type);

iree_hal_replay_file_range_t iree_hal_replay_dump_payload_subrange(
    const iree_hal_replay_file_range_t* payload_range,
    iree_host_size_t payload_offset, iree_host_size_t payload_length);

iree_status_t iree_hal_replay_dump_compute_executable_load_ranges(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_executable_load_payload_t* payload,
    iree_hal_replay_dump_executable_load_ranges_t* out_ranges);

iree_status_t iree_hal_replay_dump_read_executable_metadata_header(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_executable_load_payload_t* payload,
    const iree_hal_replay_dump_executable_load_ranges_t* ranges,
    bool* out_has_metadata,
    iree_hal_replay_executable_metadata_header_t* out_header);

iree_status_t iree_hal_replay_dump_dispatch_layout(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_dispatch_payload_t* payload,
    iree_host_size_t* out_wait_payloads_offset,
    iree_host_size_t* out_wait_payloads_size,
    iree_host_size_t* out_signal_payloads_offset,
    iree_host_size_t* out_signal_payloads_size,
    iree_host_size_t* out_constants_offset,
    iree_host_size_t* out_binding_payloads_offset,
    iree_host_size_t* out_binding_payloads_size);

iree_status_t iree_hal_replay_dump_queue_execute_layout(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_device_queue_execute_payload_t* payload,
    iree_host_size_t* out_wait_payloads_offset,
    iree_host_size_t* out_wait_payloads_size,
    iree_host_size_t* out_signal_payloads_offset,
    iree_host_size_t* out_signal_payloads_size,
    iree_host_size_t* out_binding_payloads_offset,
    iree_host_size_t* out_binding_payloads_size);

iree_status_t iree_hal_replay_dump_queue_payload_layout(
    const iree_hal_replay_file_record_t* record, iree_host_size_t header_size,
    uint64_t wait_semaphore_count, uint64_t signal_semaphore_count,
    uint64_t trailing_payload_length,
    iree_hal_replay_dump_queue_payload_layout_t* out_layout);

iree_status_t iree_hal_replay_dump_queue_transfer_layout(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_queue_transfer_payload_t* payload,
    iree_hal_replay_dump_queue_transfer_layout_t* out_layout);

// Reads and validates descriptor |operation_ordinal| from |record|.
iree_status_t iree_hal_replay_dump_read_queue_transfer_operation(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_dump_queue_transfer_layout_t* layout,
    iree_host_size_t operation_ordinal,
    iree_hal_replay_queue_transfer_operation_payload_t* out_operation);

iree_status_t iree_hal_replay_dump_execution_barrier_layout(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_command_buffer_execution_barrier_payload_t* payload,
    iree_host_size_t* out_memory_barriers_offset,
    iree_host_size_t* out_memory_barriers_size,
    iree_host_size_t* out_buffer_barriers_offset,
    iree_host_size_t* out_buffer_barriers_size);

iree_status_t iree_hal_replay_dump_scope_name(
    const iree_hal_replay_file_record_t* record,
    iree_string_view_t* out_scope_name);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_DUMP_LAYOUT_H_
