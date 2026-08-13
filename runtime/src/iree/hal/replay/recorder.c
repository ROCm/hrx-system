// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/recorder.h"

#include <string.h>

#if defined(IREE_PLATFORM_ANDROID)
#include <unistd.h>
#elif defined(IREE_PLATFORM_APPLE)
#include <pthread.h>
#elif defined(IREE_PLATFORM_LINUX)
#include <sys/syscall.h>
#include <unistd.h>
#endif  // IREE_PLATFORM_*

#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/replay/file_writer.h"
#include "iree/hal/replay/recorder_record.h"

//===----------------------------------------------------------------------===//
// iree_hal_replay_recorder_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_replay_recorder_semaphore_entry_t {
  // Base HAL semaphore retained by the recorder registry.
  iree_hal_semaphore_t* semaphore;
  // Session-local object id assigned to the semaphore.
  iree_hal_replay_object_id_t semaphore_id;
  // Next retained semaphore registry entry.
  struct iree_hal_replay_recorder_semaphore_entry_t* next;
} iree_hal_replay_recorder_semaphore_entry_t;

struct iree_hal_replay_recorder_t {
  // Reference count used to manage shared recorder lifetime.
  iree_atomic_ref_count_t ref_count;
  // Host allocator used for recorder lifetime.
  iree_allocator_t host_allocator;
  // Immutable recorder options captured at creation.
  iree_hal_replay_recorder_options_t options;
  // Mutex serializing writer access and assigning capture-order ordinals.
  iree_slim_mutex_t mutex;
  // Append-only replay file writer owned by the recorder.
  iree_hal_replay_file_writer_t* writer;
  // Next sequence ordinal assigned to a replay record.
  uint64_t next_sequence_ordinal;
  // Next session-local object id assigned to a captured HAL object.
  iree_hal_replay_object_id_t next_object_id;
  // Terminal recorder failure code, or OK while recording may continue.
  iree_status_code_t terminal_status_code;
  // Retained raw semaphores assigned replay object ids.
  iree_hal_replay_recorder_semaphore_entry_t* semaphore_list;
  // True once the writer has been closed.
  bool closed;
};

static uint64_t iree_hal_replay_current_thread_id(void) {
#if defined(IREE_SYNCHRONIZATION_DISABLE_UNSAFE)
  return 0;
#elif defined(IREE_PLATFORM_ANDROID)
  return (uint64_t)gettid();
#elif defined(IREE_PLATFORM_APPLE)
  return (uint64_t)pthread_mach_thread_np(pthread_self());
#elif defined(IREE_PLATFORM_LINUX)
  return (uint64_t)syscall(__NR_gettid);
#elif defined(IREE_PLATFORM_WINDOWS)
  return (uint64_t)GetCurrentThreadId();
#else
  return 0;
#endif  // IREE_PLATFORM_*
}

static iree_status_t iree_hal_replay_recorder_make_terminal_status(
    iree_hal_replay_recorder_t* recorder) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT(recorder->terminal_status_code != IREE_STATUS_OK);
  return iree_make_status(
      recorder->terminal_status_code,
      "HAL replay recorder is already in a failed terminal state");
}

static iree_status_t iree_hal_replay_recorder_check_open_locked(
    iree_hal_replay_recorder_t* recorder) {
  if (IREE_UNLIKELY(recorder->terminal_status_code != IREE_STATUS_OK)) {
    return iree_hal_replay_recorder_make_terminal_status(recorder);
  }
  if (IREE_UNLIKELY(recorder->closed)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "HAL replay recorder is already closed");
  }
  return iree_ok_status();
}

static void iree_hal_replay_recorder_fail_locked(
    iree_hal_replay_recorder_t* recorder, iree_status_code_t status_code) {
  IREE_ASSERT_ARGUMENT(recorder);
  if (IREE_UNLIKELY(status_code != IREE_STATUS_OK &&
                    recorder->terminal_status_code == IREE_STATUS_OK)) {
    recorder->terminal_status_code = status_code;
  }
}

const iree_hal_replay_recorder_options_t* iree_hal_replay_recorder_options(
    const iree_hal_replay_recorder_t* recorder) {
  return &recorder->options;
}

void iree_hal_replay_recorder_fail(iree_hal_replay_recorder_t* recorder,
                                   iree_status_code_t status_code) {
  IREE_ASSERT_ARGUMENT(recorder);
  if (status_code == IREE_STATUS_OK) return;
  iree_slim_mutex_lock(&recorder->mutex);
  if (recorder->terminal_status_code == IREE_STATUS_OK) {
    recorder->terminal_status_code = status_code;
  }
  iree_slim_mutex_unlock(&recorder->mutex);
}

static iree_status_t iree_hal_replay_recorder_append_record_locked(
    iree_hal_replay_recorder_t* recorder,
    iree_hal_replay_file_record_metadata_t metadata,
    iree_host_size_t iovec_count, const iree_const_byte_span_t* iovecs,
    iree_hal_replay_file_range_t* out_payload_range) {
  IREE_RETURN_IF_ERROR(iree_hal_replay_recorder_check_open_locked(recorder));
  metadata.sequence_ordinal = recorder->next_sequence_ordinal++;
  metadata.thread_id = iree_hal_replay_current_thread_id();
  iree_status_t status = iree_hal_replay_file_writer_append_record(
      recorder->writer, &metadata, iovec_count, iovecs, out_payload_range);
  iree_hal_replay_recorder_fail_locked(recorder, iree_status_code(status));
  return status;
}

static iree_status_t iree_hal_replay_recorder_record_session(
    iree_hal_replay_recorder_t* recorder) {
  iree_slim_mutex_lock(&recorder->mutex);
  iree_hal_replay_file_record_metadata_t metadata = {
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION,
  };
  iree_status_t status = iree_hal_replay_recorder_append_record_locked(
      recorder, metadata, 0, NULL, NULL);
  iree_slim_mutex_unlock(&recorder->mutex);
  return status;
}

iree_status_t iree_hal_replay_recorder_reserve_object_id(
    iree_hal_replay_recorder_t* recorder,
    iree_hal_replay_object_id_t* out_object_id) {
  IREE_ASSERT_ARGUMENT(recorder);
  IREE_ASSERT_ARGUMENT(out_object_id);
  *out_object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;

  iree_slim_mutex_lock(&recorder->mutex);
  iree_status_t status = iree_hal_replay_recorder_check_open_locked(recorder);
  iree_hal_replay_object_id_t object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  if (iree_status_is_ok(status)) {
    object_id = recorder->next_object_id++;
  }
  iree_slim_mutex_unlock(&recorder->mutex);

  if (iree_status_is_ok(status)) *out_object_id = object_id;
  return status;
}

static iree_status_t iree_hal_replay_recorder_append_object_locked(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t object_id,
    iree_hal_replay_object_type_t object_type,
    iree_hal_replay_payload_type_t payload_type, iree_host_size_t iovec_count,
    const iree_const_byte_span_t* iovecs) {
  iree_hal_replay_file_record_metadata_t metadata = {
      .device_id = device_id,
      .object_id = object_id,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT,
      .payload_type = payload_type,
      .object_type = object_type,
  };
  return iree_hal_replay_recorder_append_record_locked(
      recorder, metadata, iovec_count, iovecs, NULL);
}

iree_status_t iree_hal_replay_recorder_record_object(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_type_t object_type,
    iree_hal_replay_payload_type_t payload_type, iree_host_size_t iovec_count,
    const iree_const_byte_span_t* iovecs,
    iree_hal_replay_object_id_t* out_object_id) {
  IREE_ASSERT_ARGUMENT(out_object_id);
  *out_object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;

  iree_slim_mutex_lock(&recorder->mutex);
  iree_status_t status = iree_hal_replay_recorder_check_open_locked(recorder);
  iree_hal_replay_object_id_t object_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  if (iree_status_is_ok(status)) {
    object_id = recorder->next_object_id++;
    status = iree_hal_replay_recorder_append_object_locked(
        recorder, device_id, object_id, object_type, payload_type, iovec_count,
        iovecs);
  }
  iree_slim_mutex_unlock(&recorder->mutex);

  if (iree_status_is_ok(status)) *out_object_id = object_id;
  return status;
}

iree_status_t iree_hal_replay_recorder_register_semaphore(
    iree_hal_replay_pending_record_t* pending_record,
    iree_hal_semaphore_t* semaphore, iree_hal_replay_object_id_t semaphore_id) {
  iree_hal_replay_recorder_t* recorder = pending_record->recorder;
  iree_hal_replay_recorder_semaphore_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(recorder->host_allocator,
                                             sizeof(*entry), (void**)&entry));
  entry->semaphore = semaphore;
  iree_hal_semaphore_retain(entry->semaphore);
  entry->semaphore_id = semaphore_id;
  entry->next = recorder->semaphore_list;
  recorder->semaphore_list = entry;
  return iree_ok_status();
}

static iree_hal_replay_object_id_t iree_hal_replay_recorder_lookup_semaphore_id(
    iree_hal_replay_recorder_t* recorder, iree_hal_semaphore_t* semaphore) {
  if (!semaphore) return IREE_HAL_REPLAY_OBJECT_ID_NONE;

  iree_slim_mutex_lock(&recorder->mutex);
  iree_hal_replay_object_id_t semaphore_id = IREE_HAL_REPLAY_OBJECT_ID_NONE;
  for (iree_hal_replay_recorder_semaphore_entry_t* entry =
           recorder->semaphore_list;
       entry; entry = entry->next) {
    if (entry->semaphore == semaphore) {
      semaphore_id = entry->semaphore_id;
      break;
    }
  }
  iree_slim_mutex_unlock(&recorder->mutex);
  return semaphore_id;
}

static iree_status_t iree_hal_replay_recorder_encode_semaphore_list(
    iree_hal_replay_recorder_t* recorder,
    const iree_hal_semaphore_list_t semaphore_list,
    iree_hal_replay_semaphore_timepoint_payload_t* out_payloads) {
  if (semaphore_list.count == 0) return iree_ok_status();
  if (IREE_UNLIKELY(!semaphore_list.semaphores ||
                    !semaphore_list.payload_values || !out_payloads)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "semaphore list storage is required");
  }
  for (iree_host_size_t i = 0; i < semaphore_list.count; ++i) {
    iree_hal_replay_object_id_t semaphore_id =
        iree_hal_replay_recorder_lookup_semaphore_id(
            recorder, semaphore_list.semaphores[i]);
    if (IREE_UNLIKELY(semaphore_id == IREE_HAL_REPLAY_OBJECT_ID_NONE)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "semaphore list contains a semaphore not created by the replay "
          "recorder");
    }
    out_payloads[i] = (iree_hal_replay_semaphore_timepoint_payload_t){
        .semaphore_id = semaphore_id,
        .value = semaphore_list.payload_values[i],
    };
  }
  return iree_ok_status();
}

iree_status_t iree_hal_replay_recorder_allocate_semaphore_payloads(
    iree_hal_replay_recorder_t* recorder,
    const iree_hal_semaphore_list_t semaphore_list,
    iree_allocator_t host_allocator,
    iree_hal_replay_semaphore_timepoint_payload_t** out_payloads,
    iree_host_size_t* out_payloads_size) {
  IREE_ASSERT_ARGUMENT(out_payloads);
  IREE_ASSERT_ARGUMENT(out_payloads_size);
  *out_payloads = NULL;
  *out_payloads_size = 0;
  if (semaphore_list.count == 0) return iree_ok_status();

  iree_host_size_t payloads_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          semaphore_list.count, sizeof(**out_payloads), &payloads_size))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "replay semaphore list count overflow");
  }
  iree_hal_replay_semaphore_timepoint_payload_t* payloads = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, payloads_size, (void**)&payloads));
  iree_status_t status = iree_hal_replay_recorder_encode_semaphore_list(
      recorder, semaphore_list, payloads);
  if (iree_status_is_ok(status)) {
    *out_payloads = payloads;
    *out_payloads_size = payloads_size;
  } else {
    iree_allocator_free(host_allocator, payloads);
  }
  return status;
}

iree_status_t iree_hal_replay_recorder_begin_operation(
    iree_hal_replay_recorder_t* recorder, iree_hal_replay_object_id_t device_id,
    iree_hal_replay_object_id_t object_id,
    iree_hal_replay_object_id_t related_object_id,
    iree_hal_replay_object_type_t object_type,
    iree_hal_replay_operation_code_t operation_code,
    iree_hal_replay_payload_type_t payload_type,
    iree_hal_replay_pending_record_t* out_pending_record) {
  IREE_ASSERT_ARGUMENT(out_pending_record);
  memset(out_pending_record, 0, sizeof(*out_pending_record));

  iree_slim_mutex_lock(&recorder->mutex);
  iree_status_t status = iree_hal_replay_recorder_check_open_locked(recorder);
  if (!iree_status_is_ok(status)) {
    iree_slim_mutex_unlock(&recorder->mutex);
    return status;
  }

  out_pending_record->recorder = recorder;
  out_pending_record->metadata = (iree_hal_replay_file_record_metadata_t){
      .sequence_ordinal = recorder->next_sequence_ordinal++,
      .thread_id = iree_hal_replay_current_thread_id(),
      .device_id = device_id,
      .object_id = object_id,
      .related_object_id = related_object_id,
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .payload_type = payload_type,
      .object_type = object_type,
      .operation_code = operation_code,
  };
  return iree_ok_status();
}

void iree_hal_replay_recorder_mark_unsupported(
    iree_hal_replay_pending_record_t* pending_record) {
  pending_record->metadata.record_type =
      IREE_HAL_REPLAY_FILE_RECORD_TYPE_UNSUPPORTED;
}

iree_status_t iree_hal_replay_recorder_end_operation_with_payload(
    iree_hal_replay_pending_record_t* pending_record,
    iree_status_t operation_status, iree_host_size_t iovec_count,
    const iree_const_byte_span_t* iovecs) {
  iree_hal_replay_recorder_t* recorder = pending_record->recorder;
  pending_record->metadata.status_code =
      (uint32_t)iree_status_code(operation_status);
  iree_status_t record_status = iree_hal_replay_file_writer_append_record(
      recorder->writer, &pending_record->metadata, iovec_count, iovecs, NULL);
  iree_hal_replay_recorder_fail_locked(recorder,
                                       iree_status_code(record_status));
  iree_slim_mutex_unlock(&recorder->mutex);
  return iree_status_join(record_status, operation_status);
}

iree_status_t iree_hal_replay_recorder_end_operation(
    iree_hal_replay_pending_record_t* pending_record,
    iree_status_t operation_status) {
  return iree_hal_replay_recorder_end_operation_with_payload(
      pending_record, operation_status, 0, NULL);
}

iree_status_t iree_hal_replay_recorder_end_creation_operation(
    iree_hal_replay_pending_record_t* pending_record,
    iree_status_t operation_status, iree_host_size_t operation_iovec_count,
    const iree_const_byte_span_t* operation_iovecs,
    iree_hal_replay_object_type_t created_object_type,
    iree_hal_replay_object_id_t created_object_id,
    iree_hal_replay_payload_type_t object_payload_type,
    iree_host_size_t object_iovec_count,
    const iree_const_byte_span_t* object_iovecs) {
  iree_hal_replay_recorder_t* recorder = pending_record->recorder;
  pending_record->metadata.status_code =
      (uint32_t)iree_status_code(operation_status);
  iree_status_t record_status = iree_hal_replay_file_writer_append_record(
      recorder->writer, &pending_record->metadata, operation_iovec_count,
      operation_iovecs, NULL);
  if (iree_status_is_ok(record_status) && iree_status_is_ok(operation_status)) {
    record_status = iree_hal_replay_recorder_append_object_locked(
        recorder, pending_record->metadata.device_id, created_object_id,
        created_object_type, object_payload_type, object_iovec_count,
        object_iovecs);
  }
  iree_hal_replay_recorder_fail_locked(recorder,
                                       iree_status_code(record_status));
  iree_slim_mutex_unlock(&recorder->mutex);
  return iree_status_join(record_status, operation_status);
}

IREE_API_EXPORT iree_status_t iree_hal_replay_recorder_create(
    iree_io_file_handle_t* file_handle,
    const iree_hal_replay_recorder_options_t* options,
    iree_allocator_t host_allocator,
    iree_hal_replay_recorder_t** out_recorder) {
  IREE_ASSERT_ARGUMENT(file_handle);
  IREE_ASSERT_ARGUMENT(out_recorder);
  *out_recorder = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_replay_recorder_options_t default_options =
      iree_hal_replay_recorder_options_default();
  if (!options) options = &default_options;
  if (IREE_UNLIKELY(options->flags != IREE_HAL_REPLAY_RECORDER_FLAG_NONE)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replay recorder flags are unknown");
  }
  if (IREE_UNLIKELY(
          options->external_file_policy !=
              IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_REFERENCE &&
          options->external_file_policy !=
              IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_CAPTURE_RANGES &&
          options->external_file_policy !=
              IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_CAPTURE_ALL &&
          options->external_file_policy !=
              IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_POLICY_FAIL)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replay recorder external file policy is unknown");
  }
  if (IREE_UNLIKELY(
          options->external_file_validation !=
              IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_VALIDATION_IDENTITY &&
          options->external_file_validation !=
              IREE_HAL_REPLAY_RECORDER_EXTERNAL_FILE_VALIDATION_CONTENT_DIGEST)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "replay recorder external file validation is unknown");
  }

  iree_hal_replay_file_writer_t* writer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_replay_file_writer_create(file_handle, host_allocator, &writer));

  iree_hal_replay_recorder_t* recorder = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*recorder), (void**)&recorder);
  if (iree_status_is_ok(status)) {
    memset(recorder, 0, sizeof(*recorder));
    iree_atomic_ref_count_init(&recorder->ref_count);
    recorder->host_allocator = host_allocator;
    recorder->options = *options;
    iree_slim_mutex_initialize(&recorder->mutex);
    recorder->writer = writer;
    recorder->next_object_id = 1;
    recorder->terminal_status_code = IREE_STATUS_OK;
    writer = NULL;
    status = iree_hal_replay_recorder_record_session(recorder);
  }

  if (iree_status_is_ok(status)) {
    *out_recorder = recorder;
  } else {
    iree_hal_replay_recorder_release(recorder);
  }
  iree_hal_replay_file_writer_free(writer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_replay_recorder_retain(
    iree_hal_replay_recorder_t* recorder) {
  if (IREE_LIKELY(recorder)) {
    iree_atomic_ref_count_inc(&recorder->ref_count);
  }
}

IREE_API_EXPORT void iree_hal_replay_recorder_release(
    iree_hal_replay_recorder_t* recorder) {
  if (IREE_LIKELY(recorder) &&
      iree_atomic_ref_count_dec(&recorder->ref_count) == 1) {
    iree_allocator_t host_allocator = recorder->host_allocator;
    IREE_TRACE_ZONE_BEGIN(z0);
    iree_hal_replay_recorder_semaphore_entry_t* semaphore_entry =
        recorder->semaphore_list;
    while (semaphore_entry) {
      iree_hal_replay_recorder_semaphore_entry_t* next_entry =
          semaphore_entry->next;
      iree_hal_semaphore_release(semaphore_entry->semaphore);
      iree_allocator_free(host_allocator, semaphore_entry);
      semaphore_entry = next_entry;
    }
    iree_hal_replay_file_writer_free(recorder->writer);
    iree_slim_mutex_deinitialize(&recorder->mutex);
    iree_allocator_free(host_allocator, recorder);
    IREE_TRACE_ZONE_END(z0);
  }
}

IREE_API_EXPORT iree_status_t
iree_hal_replay_recorder_close(iree_hal_replay_recorder_t* recorder) {
  IREE_ASSERT_ARGUMENT(recorder);

  iree_slim_mutex_lock(&recorder->mutex);
  iree_status_t status = iree_ok_status();
  if (IREE_UNLIKELY(recorder->terminal_status_code != IREE_STATUS_OK)) {
    status = iree_hal_replay_recorder_make_terminal_status(recorder);
  } else if (!recorder->closed) {
    status = iree_hal_replay_file_writer_close(recorder->writer);
    if (iree_status_is_ok(status)) {
      recorder->closed = true;
    } else {
      iree_hal_replay_recorder_fail_locked(recorder, iree_status_code(status));
    }
  }
  iree_slim_mutex_unlock(&recorder->mutex);
  return status;
}

static iree_status_t iree_hal_replay_recorder_record_scope(
    iree_hal_replay_recorder_t* recorder,
    iree_hal_replay_operation_code_t operation_code,
    iree_string_view_t scope_name) {
  IREE_ASSERT_ARGUMENT(recorder);
  if (IREE_UNLIKELY(iree_string_view_is_empty(scope_name))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "replay scope name must be non-empty");
  }

  iree_hal_replay_scope_payload_t payload = {
      .name_length = scope_name.size,
      .flags = IREE_HAL_REPLAY_SCOPE_FLAG_NONE,
  };
  const iree_const_byte_span_t iovecs[] = {
      iree_make_const_byte_span(&payload, sizeof(payload)),
      iree_make_const_byte_span(scope_name.data, scope_name.size),
  };
  iree_hal_replay_file_record_metadata_t metadata = {
      .record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION,
      .payload_type = IREE_HAL_REPLAY_PAYLOAD_TYPE_REPLAY_SCOPE,
      .operation_code = operation_code,
  };

  iree_slim_mutex_lock(&recorder->mutex);
  iree_status_t status = iree_hal_replay_recorder_append_record_locked(
      recorder, metadata, IREE_ARRAYSIZE(iovecs), iovecs, NULL);
  iree_slim_mutex_unlock(&recorder->mutex);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_replay_recorder_scope_begin(
    iree_hal_replay_recorder_t* recorder, iree_string_view_t scope_name) {
  return iree_hal_replay_recorder_record_scope(
      recorder, IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_BEGIN, scope_name);
}

IREE_API_EXPORT iree_status_t iree_hal_replay_recorder_scope_end(
    iree_hal_replay_recorder_t* recorder, iree_string_view_t scope_name) {
  return iree_hal_replay_recorder_record_scope(
      recorder, IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_END, scope_name);
}
