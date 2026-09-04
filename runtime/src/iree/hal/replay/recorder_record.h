// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_RECORDER_RECORD_H_
#define IREE_HAL_REPLAY_RECORDER_RECORD_H_

#include "iree/base/api.h"
#include "iree/hal/replay/file_writer.h"
#include "iree/hal/replay/format.h"
#include "iree/hal/replay/recorder.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Pending operation metadata owned by a begin/end recording pair.
typedef struct iree_hal_replay_pending_record_t {
  // Recorder whose mutex is held until the pending record is completed.
  iree_hal_replay_recorder_t* recorder;
  // Metadata to write once the intercepted operation has completed.
  iree_hal_replay_file_record_metadata_t metadata;
} iree_hal_replay_pending_record_t;

// One object record appended after a successful creation operation.
typedef struct iree_hal_replay_created_object_record_t {
  // Session-local object id assigned before the operation began.
  iree_hal_replay_object_id_t object_id;
  // HAL object type stored under |object_id|.
  iree_hal_replay_object_type_t object_type;
  // Schema identifying the serialized object payload.
  iree_hal_replay_payload_type_t payload_type;
  // Number of byte spans composing the object payload.
  iree_host_size_t iovec_count;
  // Borrowed byte spans valid until the pending operation is completed.
  const iree_const_byte_span_t* iovecs;
} iree_hal_replay_created_object_record_t;

// Returns the immutable options captured by |recorder| at creation.
const iree_hal_replay_recorder_options_t* iree_hal_replay_recorder_options(
    const iree_hal_replay_recorder_t* recorder);

// Records |status_code| as the recorder's terminal failure.
void iree_hal_replay_recorder_fail(iree_hal_replay_recorder_t* recorder,
                                   iree_status_code_t status_code);

iree_status_t iree_hal_replay_recorder_reserve_object_id(
    iree_hal_replay_recorder_t* recorder,
    iree_hal_replay_object_id_t* out_object_id);

iree_status_t iree_hal_replay_recorder_record_object(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_type_t object_type,
    iree_hal_replay_payload_type_t payload_type, iree_host_size_t iovec_count,
    const iree_const_byte_span_t* iovecs,
    iree_hal_replay_object_id_t* out_object_id);

// Allocates and encodes the timepoints in |semaphore_list|.
//
// The caller owns the returned storage and must free it with |host_allocator|.
iree_status_t iree_hal_replay_recorder_allocate_semaphore_payloads(
    iree_hal_replay_recorder_t* recorder,
    const iree_hal_semaphore_list_t semaphore_list,
    iree_allocator_t host_allocator,
    iree_hal_replay_semaphore_timepoint_payload_t** out_payloads,
    iree_host_size_t* out_payloads_size);

iree_status_t iree_hal_replay_recorder_begin_operation(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t object_id,
    iree_hal_replay_object_id_t related_object_id,
    iree_hal_replay_object_type_t object_type,
    iree_hal_replay_operation_code_t operation_code,
    iree_hal_replay_payload_type_t payload_type,
    iree_hal_replay_pending_record_t* out_pending_record);

// Registers |semaphore| while |pending_record| owns the recorder mutex.
//
// The recorder retains |semaphore| on success and releases it with the
// recorder. Registration must complete before ending |pending_record| so a
// successfully recorded semaphore object is always available to later
// operation records.
iree_status_t iree_hal_replay_recorder_register_semaphore(
    iree_hal_replay_pending_record_t* pending_record,
    iree_hal_semaphore_t* semaphore, iree_hal_replay_object_id_t semaphore_id);

// Returns the captured object id for |semaphore|, or NONE when the semaphore
// was not created through |recorder|.
iree_hal_replay_object_id_t iree_hal_replay_recorder_semaphore_id_or_none(
    iree_hal_replay_recorder_t* recorder, iree_hal_semaphore_t* semaphore);

// Marks |pending_record| as a captured operation that cannot be replayed.
void iree_hal_replay_recorder_mark_unsupported(
    iree_hal_replay_pending_record_t* pending_record);

iree_status_t iree_hal_replay_recorder_end_operation(
    iree_hal_replay_pending_record_t* pending_record,
    iree_status_t operation_status);

iree_status_t iree_hal_replay_recorder_end_operation_with_payload(
    iree_hal_replay_pending_record_t* pending_record,
    iree_status_t operation_status, iree_host_size_t iovec_count,
    const iree_const_byte_span_t* iovecs);

iree_status_t iree_hal_replay_recorder_end_creation_operation(
    iree_hal_replay_pending_record_t* pending_record,
    iree_status_t operation_status, iree_host_size_t operation_iovec_count,
    const iree_const_byte_span_t* operation_iovecs,
    iree_hal_replay_object_type_t created_object_type,
    iree_hal_replay_object_id_t created_object_id,
    iree_hal_replay_payload_type_t object_payload_type,
    iree_host_size_t object_iovec_count,
    const iree_const_byte_span_t* object_iovecs);

// Completes a creation operation and atomically appends all successfully
// created object records before releasing the recorder mutex.
iree_status_t iree_hal_replay_recorder_end_creation_operation_list(
    iree_hal_replay_pending_record_t* pending_record,
    iree_status_t operation_status, iree_host_size_t operation_iovec_count,
    const iree_const_byte_span_t* operation_iovecs,
    iree_host_size_t created_object_count,
    const iree_hal_replay_created_object_record_t* created_objects);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_RECORDER_RECORD_H_
