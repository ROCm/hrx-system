// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_RECORDER_FILE_H_
#define IREE_HAL_REPLAY_RECORDER_FILE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/replay/format.h"
#include "iree/hal/replay/recorder.h"
#include "iree/io/file_handle.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_status_t iree_hal_replay_recorder_file_make_object_payload(
    iree_io_file_handle_t* handle,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_memory_access_t access, iree_hal_external_file_flags_t flags,
    iree_hal_file_t* base_file,
    iree_hal_replay_recorder_external_file_policy_t external_file_policy,
    iree_hal_replay_recorder_external_file_validation_t
        external_file_validation,
    iree_allocator_t host_allocator, iree_byte_span_t path_reference_storage,
    iree_byte_span_t* out_allocated_reference_storage,
    iree_hal_replay_file_object_payload_t* out_payload,
    iree_string_view_t* out_reference);

iree_status_t iree_hal_replay_recorder_file_capture_read_data(
    iree_hal_file_t* file, uint64_t source_offset, iree_device_size_t length,
    iree_allocator_t host_allocator, iree_byte_span_t* out_storage);

iree_status_t iree_hal_replay_recorder_file_create_proxy(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t file_id, iree_io_file_handle_t* source_handle,
    iree_hal_file_t* base_file, iree_allocator_t host_allocator,
    iree_hal_file_t** out_file);

iree_hal_file_t* iree_hal_replay_recorder_file_base_or_self(
    iree_hal_file_t* file);

iree_hal_replay_object_id_t iree_hal_replay_recorder_file_id_or_none(
    iree_hal_file_t* file);

// Returns the captured object id for |file| when it belongs to |recorder|, or
// NONE for files outside that recording session.
iree_hal_replay_object_id_t iree_hal_replay_recorder_find_file_id(
    iree_hal_replay_recorder_t* recorder, iree_hal_file_t* file);

// Returns true when |file| is an fd-backed file whose reads must be embedded as
// captured ranges under the immutable |recorder| policy.
bool iree_hal_replay_recorder_file_uses_captured_ranges(
    iree_hal_replay_recorder_t* recorder, iree_hal_file_t* file);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_RECORDER_FILE_H_
