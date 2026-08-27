// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_DUMP_STATE_H_
#define IREE_HAL_REPLAY_DUMP_STATE_H_

#include "iree/hal/replay/dump.h"
#include "iree/hal/replay/file_reader.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Streaming state shared by the file coordinator and output renderer.
typedef struct iree_hal_replay_dump_context_t {
  // Caller-provided streaming sink.
  iree_hal_replay_dump_write_callback_t write_callback;
  // Host allocator used for temporary line construction.
  iree_allocator_t host_allocator;
} iree_hal_replay_dump_context_t;

// Aggregate properties discovered by the file summary scan.
typedef struct iree_hal_replay_dump_file_summary_t {
  // Total top-level records in the replay file.
  uint64_t record_count;
  // Number of HAL object records.
  uint64_t object_count;
  // Number of replayable HAL operation records.
  uint64_t operation_count;
  // Number of replay scope begin records.
  uint64_t scope_begin_count;
  // Number of replay scope end records.
  uint64_t scope_end_count;
  // Number of explicitly unsupported HAL operation records.
  uint64_t unsupported_count;
  // Number of HAL file object records.
  uint64_t file_object_count;
  // Number of file objects referencing environment files by path.
  uint64_t external_file_count;
  // Number of file objects embedded inline in the replay file.
  uint64_t inline_file_count;
  // Number of file objects represented by captured queue_read ranges.
  uint64_t range_file_count;
  // Number of file objects using unknown future reference types.
  uint64_t unknown_file_reference_count;
  // Number of external file references validated by platform identity.
  uint64_t identity_file_validation_count;
  // Number of external file references validated by content digest.
  uint64_t digest_file_validation_count;
  // Number of file references with no validation beyond length.
  uint64_t no_file_validation_count;
  // Number of file references using unknown future validation modes.
  uint64_t unknown_file_validation_count;
  // Total captured length of externally referenced files.
  uint64_t external_file_total_length;
  // Total embedded inline file bytes.
  uint64_t inline_file_total_length;
  // Total length of files represented by captured queue_read ranges.
  uint64_t range_file_total_length;
  // Total bytes embedded on captured queue_read operation records.
  uint64_t captured_read_total_length;
} iree_hal_replay_dump_file_summary_t;

// Flushes the current builder contents to the streaming sink.
static inline iree_status_t iree_hal_replay_dump_emit(
    iree_hal_replay_dump_context_t* context, iree_string_builder_t* builder) {
  if (iree_string_builder_size(builder) == 0) return iree_ok_status();
  iree_status_t status = context->write_callback.fn(
      context->write_callback.user_data, iree_string_builder_view(builder));
  iree_string_builder_reset(builder);
  return status;
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_DUMP_STATE_H_
