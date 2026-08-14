// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REPLAY_EXECUTE_STATE_H_
#define IREE_HAL_REPLAY_EXECUTE_STATE_H_

#include "iree/hal/replay/execute.h"
#include "iree/hal/replay/file_reader.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define IREE_HAL_REPLAY_INLINE_SEMAPHORE_LIST_CAPACITY 8
#define IREE_HAL_REPLAY_INLINE_BUFFER_REF_LIST_CAPACITY 8
#define IREE_HAL_REPLAY_INLINE_BUFFER_BINDING_TABLE_CAPACITY 8

// Retained executable and captured-to-live function mapping.
typedef struct iree_hal_replay_executable_entry_t {
  // Retained HAL executable handle used for replay dispatches.
  iree_hal_executable_t* handle;
  // Number of captured function ordinals in |function_map|.
  iree_host_size_t function_map_count;
  // Captured function ordinal to live function token mapping.
  iree_hal_executable_function_t* function_map;
} iree_hal_replay_executable_entry_t;

// One retained object in the dense replay object table.
typedef struct iree_hal_replay_object_entry_t {
  // Captured object type stored in this entry.
  iree_hal_replay_object_type_t type;
  // Retained HAL resource or executable state selected by |type|.
  union {
    // Retained HAL device.
    iree_hal_device_t* device;
    // Retained HAL allocator.
    iree_hal_allocator_t* allocator;
    // Retained HAL buffer.
    iree_hal_buffer_t* buffer;
    // Retained HAL command buffer.
    iree_hal_command_buffer_t* command_buffer;
    // Retained executable and function mapping.
    iree_hal_replay_executable_entry_t executable;
    // Retained HAL semaphore.
    iree_hal_semaphore_t* semaphore;
    // Retained HAL file.
    iree_hal_file_t* file;
  } value;
} iree_hal_replay_object_entry_t;

// Mutable state owned by one prepared-plan execution.
typedef struct iree_hal_replay_executor_t {
  // Original replay file bytes.
  iree_const_byte_span_t file_contents;
  // Retained topology supplied by the caller.
  iree_hal_device_group_t* device_group;
  // Host allocator used for temporary replay state.
  iree_allocator_t host_allocator;
  // Execution options supplied by the caller.
  const iree_hal_replay_execute_options_t* options;
  // Dense session-local object table indexed by replay object id.
  iree_hal_replay_object_entry_t* objects;
  // Number of entries in |objects|.
  iree_host_size_t object_capacity;
  // Next caller-provided device consumed by a device object record.
  iree_host_size_t next_device_index;
} iree_hal_replay_executor_t;

// Temporary HAL semaphore list with inline storage for common cases.
typedef struct iree_hal_replay_semaphore_list_storage_t {
  // HAL semaphore list referencing arrays below.
  iree_hal_semaphore_list_t list;
  // Mutable semaphore pointer array used by |list|.
  iree_hal_semaphore_t** semaphores;
  // Mutable semaphore payload value array used by |list|.
  uint64_t* payload_values;
  // Inline storage used for common small semaphore lists.
  struct {
    // Inline semaphore pointer storage.
    iree_hal_semaphore_t*
        semaphores[IREE_HAL_REPLAY_INLINE_SEMAPHORE_LIST_CAPACITY];
    // Inline payload value storage.
    uint64_t payload_values[IREE_HAL_REPLAY_INLINE_SEMAPHORE_LIST_CAPACITY];
  } inline_storage;
  // Heap storage used when a captured list exceeds inline capacity.
  struct {
    // Heap-allocated semaphore pointer storage, or NULL when inline.
    iree_hal_semaphore_t** semaphores;
    // Heap-allocated payload value storage, or NULL when inline.
    uint64_t* payload_values;
  } allocated;
} iree_hal_replay_semaphore_list_storage_t;

// Temporary HAL buffer reference list with inline storage.
typedef struct iree_hal_replay_buffer_ref_list_storage_t {
  // HAL buffer ref list referencing |values|.
  iree_hal_buffer_ref_list_t list;
  // Mutable buffer ref array used by |list|.
  iree_hal_buffer_ref_t* values;
  // Inline storage used for common small binding lists.
  struct {
    // Inline buffer ref storage.
    iree_hal_buffer_ref_t
        values[IREE_HAL_REPLAY_INLINE_BUFFER_REF_LIST_CAPACITY];
  } inline_storage;
  // Heap storage used when a captured list exceeds inline capacity.
  struct {
    // Heap-allocated buffer ref storage, or NULL when inline.
    iree_hal_buffer_ref_t* values;
  } allocated;
} iree_hal_replay_buffer_ref_list_storage_t;

// Temporary HAL buffer binding table with inline storage.
typedef struct iree_hal_replay_buffer_binding_table_storage_t {
  // HAL buffer binding table referencing |bindings|.
  iree_hal_buffer_binding_table_t table;
  // Mutable buffer binding array used by |table|.
  iree_hal_buffer_binding_t* bindings;
  // Inline storage used for common small binding tables.
  struct {
    // Inline buffer binding storage.
    iree_hal_buffer_binding_t
        bindings[IREE_HAL_REPLAY_INLINE_BUFFER_BINDING_TABLE_CAPACITY];
  } inline_storage;
  // Heap storage used when a captured table exceeds inline capacity.
  struct {
    // Heap-allocated buffer binding storage, or NULL when inline.
    iree_hal_buffer_binding_t* bindings;
  } allocated;
} iree_hal_replay_buffer_binding_table_storage_t;

iree_status_t iree_hal_replay_executor_initialize(
    iree_hal_replay_executor_t* executor, iree_const_byte_span_t file_contents,
    iree_host_size_t object_capacity, iree_hal_device_group_t* device_group,
    const iree_hal_replay_execute_options_t* options,
    iree_allocator_t host_allocator);

void iree_hal_replay_executor_deinitialize(
    iree_hal_replay_executor_t* executor);

iree_status_t iree_hal_replay_executor_lookup(
    iree_hal_replay_executor_t* executor, iree_hal_replay_object_id_t object_id,
    iree_hal_replay_object_type_t expected_type,
    iree_hal_replay_object_entry_t** out_entry);

iree_status_t iree_hal_replay_executor_store(
    iree_hal_replay_executor_t* executor, iree_hal_replay_object_id_t object_id,
    iree_hal_replay_object_type_t object_type,
    iree_hal_replay_object_entry_t entry);

iree_status_t iree_hal_replay_executor_allocate_function_map(
    iree_hal_replay_executor_t* executor, iree_host_size_t function_count,
    iree_hal_executable_function_t** out_function_map);

iree_status_t iree_hal_replay_executor_resolve_function(
    iree_hal_replay_object_id_t executable_id,
    const iree_hal_replay_executable_entry_t* executable,
    uint32_t function_ordinal, iree_hal_executable_function_t* out_function);

iree_status_t iree_hal_replay_executor_require_payload(
    const iree_hal_replay_file_record_t* record,
    iree_hal_replay_payload_type_t payload_type,
    iree_host_size_t minimum_length);

iree_status_t iree_hal_replay_executor_make_buffer_params(
    const iree_hal_replay_allocator_allocate_buffer_payload_t* payload,
    iree_hal_buffer_params_t* out_params);

iree_status_t iree_hal_replay_executor_write_buffer_data(
    iree_hal_buffer_t* buffer, iree_device_size_t byte_offset,
    iree_device_size_t byte_length, iree_hal_memory_access_t memory_access,
    iree_const_byte_span_t data);

iree_status_t iree_hal_replay_executor_make_buffer_ref(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_buffer_ref_payload_t* payload,
    iree_hal_buffer_ref_t* out_ref);

void iree_hal_replay_semaphore_list_storage_deinitialize(
    iree_hal_replay_semaphore_list_storage_t* storage,
    iree_allocator_t host_allocator);

iree_status_t iree_hal_replay_executor_make_semaphore_list(
    iree_hal_replay_executor_t* executor, iree_const_byte_span_t payloads,
    iree_host_size_t count,
    iree_hal_replay_semaphore_list_storage_t* out_storage);

iree_status_t iree_hal_replay_executor_make_queue_semaphore_lists(
    iree_hal_replay_executor_t* executor,
    const iree_hal_replay_file_record_t* record, iree_host_size_t header_length,
    uint64_t wait_semaphore_count, uint64_t signal_semaphore_count,
    uint64_t trailing_payload_length,
    iree_hal_replay_semaphore_list_storage_t* out_wait_storage,
    iree_hal_replay_semaphore_list_storage_t* out_signal_storage,
    iree_const_byte_span_t* out_trailing_payload);

void iree_hal_replay_buffer_ref_list_storage_deinitialize(
    iree_hal_replay_buffer_ref_list_storage_t* storage,
    iree_allocator_t host_allocator);

iree_status_t iree_hal_replay_buffer_ref_list_storage_initialize(
    iree_hal_replay_executor_t* executor, iree_host_size_t count,
    iree_hal_replay_buffer_ref_list_storage_t* out_storage);

void iree_hal_replay_buffer_binding_table_storage_deinitialize(
    iree_hal_replay_buffer_binding_table_storage_t* storage,
    iree_allocator_t host_allocator);

iree_status_t iree_hal_replay_buffer_binding_table_storage_initialize(
    iree_hal_replay_executor_t* executor, iree_host_size_t count,
    iree_hal_replay_buffer_binding_table_storage_t* out_storage);

iree_status_t iree_hal_replay_executor_flush_and_wait(
    iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_semaphore_list_t signal_list);

iree_status_t iree_hal_replay_executor_dispatch_layout(
    const iree_hal_replay_file_record_t* record,
    const iree_hal_replay_dispatch_payload_t* payload,
    iree_host_size_t* out_wait_payloads_offset,
    iree_host_size_t* out_wait_payloads_size,
    iree_host_size_t* out_signal_payloads_offset,
    iree_host_size_t* out_signal_payloads_size,
    iree_host_size_t* out_constants_offset,
    iree_host_size_t* out_binding_payloads_offset,
    iree_host_size_t* out_binding_payloads_size);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REPLAY_EXECUTE_STATE_H_
