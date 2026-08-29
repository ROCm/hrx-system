// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_VULKAN_PROFILE_H_
#define IREE_HAL_VULKAN_PROFILE_H_

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_vulkan_profile_recorder_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_vulkan_profile_recorder_t
    iree_hal_vulkan_profile_recorder_t;

// Returns the HAL-native profiling data families produced by Vulkan recorders.
static inline iree_hal_device_profiling_data_families_t
iree_hal_vulkan_profile_recorder_supported_data_families(void) {
  return IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
         IREE_HAL_DEVICE_PROFILING_DATA_DEVICE_QUEUE_EVENTS |
         IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS |
         IREE_HAL_DEVICE_PROFILING_DATA_EXECUTABLE_METADATA |
         IREE_HAL_DEVICE_PROFILING_DATA_MEMORY_EVENTS;
}

// Metadata needed to begin a Vulkan driver profiling session.
//
// The recorder writes the supplied device and queue records during creation and
// does not retain the record arrays. |name| is copied into recorder-owned
// storage because flush/end can happen after profiling_begin returns.
typedef struct iree_hal_vulkan_profile_recorder_options_t {
  // Human-readable producer name used on session and metadata chunks.
  iree_string_view_t name;

  // Process-local profiling session identifier assigned by the caller.
  uint64_t session_id;

  // Number of physical device metadata records in |device_records|.
  iree_host_size_t device_record_count;

  // Borrowed physical device metadata records emitted at session begin.
  const iree_hal_profile_device_record_t* device_records;

  // Number of queue metadata records in |queue_records|.
  iree_host_size_t queue_record_count;

  // Borrowed queue metadata records emitted at session begin.
  const iree_hal_profile_queue_record_t* queue_records;

  // Maximum dispatch events retained between flushes; 0 selects the default.
  iree_host_size_t dispatch_event_capacity;

  // Maximum queue events retained between flushes; 0 selects the default.
  iree_host_size_t queue_event_capacity;

  // Maximum queue device events retained between flushes; 0 selects the
  // default.
  iree_host_size_t queue_device_event_capacity;

  // Maximum memory events retained between flushes; 0 selects the default.
  iree_host_size_t memory_event_capacity;
} iree_hal_vulkan_profile_recorder_options_t;

// Queue identity shared by Vulkan profiling records.
typedef struct iree_hal_vulkan_profile_queue_scope_t {
  // Session-local physical device ordinal associated with the record.
  uint32_t physical_device_ordinal;

  // Session-local queue ordinal associated with the record.
  uint32_t queue_ordinal;

  // Producer-defined stream identifier matching queue metadata.
  uint64_t stream_id;
} iree_hal_vulkan_profile_queue_scope_t;

// Returns a queue scope with absent ordinals and no stream id.
static inline iree_hal_vulkan_profile_queue_scope_t
iree_hal_vulkan_profile_queue_scope_default(void) {
  iree_hal_vulkan_profile_queue_scope_t scope;
  memset(&scope, 0, sizeof(scope));
  scope.physical_device_ordinal = UINT32_MAX;
  scope.queue_ordinal = UINT32_MAX;
  return scope;
}

// Device-timestamped dispatch data used to append one dispatch event.
typedef struct iree_hal_vulkan_profile_dispatch_event_info_t {
  // Flags describing how the dispatch was produced.
  iree_hal_profile_dispatch_event_flags_t flags;

  // Queue metadata identity associated with the dispatch event chunk.
  iree_hal_vulkan_profile_queue_scope_t scope;

  // Queue submission epoch containing this dispatch.
  uint64_t submission_id;

  // Session-local command-buffer identifier, or 0 for direct dispatch.
  uint64_t command_buffer_id;

  // Session-local executable identifier.
  uint64_t executable_id;

  // Command ordinal within a command buffer, or UINT32_MAX for direct dispatch.
  uint32_t command_index;

  // Executable function ordinal dispatched.
  uint32_t function_ordinal;

  // Workgroup counts submitted for each dimension.
  uint32_t workgroup_count[3];

  // Workgroup sizes submitted for each dimension.
  uint32_t workgroup_size[3];

  // Device timestamp captured when dispatch execution started.
  uint64_t start_tick;

  // Device timestamp captured when dispatch execution completed.
  uint64_t end_tick;
} iree_hal_vulkan_profile_dispatch_event_info_t;

// Returns default dispatch event append data.
static inline iree_hal_vulkan_profile_dispatch_event_info_t
iree_hal_vulkan_profile_dispatch_event_info_default(void) {
  iree_hal_vulkan_profile_dispatch_event_info_t info;
  memset(&info, 0, sizeof(info));
  info.scope = iree_hal_vulkan_profile_queue_scope_default();
  info.command_index = UINT32_MAX;
  info.function_ordinal = UINT32_MAX;
  return info;
}

// Queue operation data used to append one queue event record.
typedef struct iree_hal_vulkan_profile_queue_event_info_t {
  // Kind of queue operation represented by the event.
  iree_hal_profile_queue_event_type_t type;

  // Flags describing queue operation properties.
  iree_hal_profile_queue_event_flags_t flags;

  // Strategy used for wait dependencies on this operation.
  iree_hal_profile_queue_dependency_strategy_t dependency_strategy;

  // Queue metadata identity shared by the appended record.
  iree_hal_vulkan_profile_queue_scope_t scope;

  // IREE monotonic host timestamp when submitted, or 0 to sample now.
  iree_time_t host_time_ns;

  // IREE monotonic host timestamp when ready to execute, or 0 when unknown.
  iree_time_t ready_host_time_ns;

  // Queue submission epoch associated with this operation, or 0 when absent.
  uint64_t submission_id;

  // Session-local command-buffer identifier, or 0 when not applicable.
  uint64_t command_buffer_id;

  // Producer-defined allocation identifier, or 0 when not applicable.
  uint64_t allocation_id;

  // Number of wait semaphores supplied to the queue operation.
  uint32_t wait_count;

  // Number of signal semaphores supplied to the queue operation.
  uint32_t signal_count;

  // Number of dedicated dependency barrier packets emitted for this operation.
  uint32_t barrier_count;

  // Number of encoded payload operations represented by this queue operation.
  uint32_t operation_count;

  // Type-specific payload byte length, or 0 when not applicable.
  uint64_t payload_length;
} iree_hal_vulkan_profile_queue_event_info_t;

// Returns default queue event append data.
static inline iree_hal_vulkan_profile_queue_event_info_t
iree_hal_vulkan_profile_queue_event_info_default(void) {
  iree_hal_vulkan_profile_queue_event_info_t info;
  memset(&info, 0, sizeof(info));
  info.scope = iree_hal_vulkan_profile_queue_scope_default();
  return info;
}

// Queue operation data used to append one device-timestamped queue event.
typedef struct iree_hal_vulkan_profile_queue_device_event_info_t {
  // Kind of queue operation represented by the event.
  iree_hal_profile_queue_event_type_t type;

  // Flags describing queue operation properties.
  iree_hal_profile_queue_event_flags_t flags;

  // Queue metadata identity shared by the appended record.
  iree_hal_vulkan_profile_queue_scope_t scope;

  // Queue submission epoch associated with this operation, or 0 when absent.
  uint64_t submission_id;

  // Session-local command-buffer identifier, or 0 when not applicable.
  uint64_t command_buffer_id;

  // Producer-defined allocation identifier, or 0 when not applicable.
  uint64_t allocation_id;

  // Number of encoded payload operations represented by this queue operation.
  uint32_t operation_count;

  // Type-specific payload byte length, or 0 when not applicable.
  uint64_t payload_length;

  // Device timestamp captured when queue-visible work started.
  uint64_t start_tick;

  // Device timestamp captured when queue-visible work completed.
  uint64_t end_tick;
} iree_hal_vulkan_profile_queue_device_event_info_t;

// Returns default queue device event append data.
static inline iree_hal_vulkan_profile_queue_device_event_info_t
iree_hal_vulkan_profile_queue_device_event_info_default(void) {
  iree_hal_vulkan_profile_queue_device_event_info_t info;
  memset(&info, 0, sizeof(info));
  info.scope = iree_hal_vulkan_profile_queue_scope_default();
  return info;
}

// Begins a Vulkan profiling session and returns its recorder.
//
// Returns OK with |out_recorder| set to NULL when
// |profiling_options->data_families| is NONE. Otherwise the requested families
// must be a subset of
// iree_hal_vulkan_profile_recorder_supported_data_families() and
// |profiling_options->sink| must be non-NULL.
iree_status_t iree_hal_vulkan_profile_recorder_create(
    const iree_hal_vulkan_profile_recorder_options_t* recorder_options,
    const iree_hal_device_profiling_options_t* profiling_options,
    iree_allocator_t host_allocator,
    iree_hal_vulkan_profile_recorder_t** out_recorder);

// Destroys |recorder| and releases retained session resources.
//
// Callers should end active sessions with iree_hal_vulkan_profile_recorder_end
// before destroying the recorder so sink end-session failures can be observed.
void iree_hal_vulkan_profile_recorder_destroy(
    iree_hal_vulkan_profile_recorder_t* recorder);

// Returns true when |recorder| is active and any of |data_families| is enabled.
bool iree_hal_vulkan_profile_recorder_is_enabled(
    const iree_hal_vulkan_profile_recorder_t* recorder,
    iree_hal_device_profiling_data_families_t data_families);

// Returns the resolved profiling options held by |recorder|, or NULL.
const iree_hal_device_profiling_options_t*
iree_hal_vulkan_profile_recorder_options(
    const iree_hal_vulkan_profile_recorder_t* recorder);

// Emits executable function metadata for |executable_id| once per recorder
// session.
//
// |executable_id| must be a producer-defined nonzero identifier that remains
// stable for the lifetime of |executable|. The caller supplies the stable
// Vulkan executable id used by dispatch records in this session.
iree_status_t iree_hal_vulkan_profile_recorder_record_executable(
    iree_hal_vulkan_profile_recorder_t* recorder,
    iree_hal_executable_t* executable, uint64_t executable_id);

// Emits command-buffer and operation metadata once per recorder session.
//
// Returns OK without work when metadata is not enabled. The caller owns
// |command_buffer| and |operations|; the recorder does not retain them.
iree_status_t iree_hal_vulkan_profile_recorder_record_command_buffer(
    iree_hal_vulkan_profile_recorder_t* recorder,
    const iree_hal_profile_command_buffer_record_t* command_buffer,
    iree_host_size_t operation_count,
    const iree_hal_profile_command_operation_record_t* operations);

// Appends one producer-owned dispatch event to |recorder|.
//
// |out_event_id| may be NULL. When provided it receives the assigned event id,
// or 0 if dispatch events were not requested. Dispatch events are lossless
// profiling records. A full ring is flushed synchronously before appending; if
// that flush fails, the pending records are preserved and the failure is
// returned to the producer instead of silently truncating device timelines.
iree_status_t iree_hal_vulkan_profile_recorder_append_dispatch_event(
    iree_hal_vulkan_profile_recorder_t* recorder,
    const iree_hal_vulkan_profile_dispatch_event_info_t* event_info,
    uint64_t* out_event_id);

// Appends one host-timestamped queue event to |recorder|.
//
// |out_event_id| may be NULL. When provided it receives the assigned event id,
// or 0 if queue events were not requested or the event ring was full. Ring
// capacity pressure is reported later as truncated chunks during flush.
void iree_hal_vulkan_profile_recorder_append_queue_event(
    iree_hal_vulkan_profile_recorder_t* recorder,
    const iree_hal_vulkan_profile_queue_event_info_t* event_info,
    uint64_t* out_event_id);

// Appends one device-timestamped queue event to |recorder|.
//
// |out_event_id| may be NULL. When provided it receives the assigned event id,
// or 0 if queue device events were not requested. Unlike host queue events,
// queue device events are lossless profiling records. A full ring is flushed
// synchronously before appending; if that flush fails, the pending records are
// preserved and the failure is returned to the producer instead of silently
// truncating device timelines.
iree_status_t iree_hal_vulkan_profile_recorder_append_queue_device_event(
    iree_hal_vulkan_profile_recorder_t* recorder,
    const iree_hal_vulkan_profile_queue_device_event_info_t* event_info,
    uint64_t* out_event_id);

// Appends one host-timestamped memory lifecycle event to |recorder|.
//
// |event| is copied into the recorder. The recorder overwrites record_length
// and event_id and samples host_time_ns when it is zero. Queue-operation memory
// events must provide a valid physical device and queue ordinal. |out_event_id|
// may be NULL. When provided it receives the assigned event id, or 0 if memory
// events were not requested or the event ring was full. Ring capacity pressure
// is reported later as truncated chunks during flush.
void iree_hal_vulkan_profile_recorder_append_memory_event(
    iree_hal_vulkan_profile_recorder_t* recorder,
    const iree_hal_profile_memory_event_t* event, uint64_t* out_event_id);

// Writes clock-correlation records to the active session sink.
//
// The caller must provide real producer-obtained clock samples. This helper is
// intentionally not buffered: correlations are cold-path session calibration
// records, and failures must be reported directly to the profiling operation.
iree_status_t iree_hal_vulkan_profile_recorder_write_clock_correlations(
    iree_hal_vulkan_profile_recorder_t* recorder, iree_host_size_t record_count,
    const iree_hal_profile_clock_correlation_record_t* records);

// Writes all buffered profile records to the session sink.
iree_status_t iree_hal_vulkan_profile_recorder_flush(
    iree_hal_vulkan_profile_recorder_t* recorder);

// Flushes and ends the profiling session.
//
// The sink receives the terminal status code derived from the first failing
// flush or end-session operation. A second call after a successful end is a
// no-op.
iree_status_t iree_hal_vulkan_profile_recorder_end(
    iree_hal_vulkan_profile_recorder_t* recorder);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_VULKAN_PROFILE_H_
