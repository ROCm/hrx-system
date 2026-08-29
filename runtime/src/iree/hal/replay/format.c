// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/replay/format.h"

IREE_API_EXPORT const char* iree_hal_replay_file_record_type_string(
    iree_hal_replay_file_record_type_t record_type) {
  switch (record_type) {
    case IREE_HAL_REPLAY_FILE_RECORD_TYPE_NONE:
      return "none";
    case IREE_HAL_REPLAY_FILE_RECORD_TYPE_SESSION:
      return "session";
    case IREE_HAL_REPLAY_FILE_RECORD_TYPE_OBJECT:
      return "object";
    case IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION:
      return "operation";
    case IREE_HAL_REPLAY_FILE_RECORD_TYPE_BLOB:
      return "blob";
    case IREE_HAL_REPLAY_FILE_RECORD_TYPE_UNSUPPORTED:
      return "unsupported";
    default:
      return "unknown";
  }
}

IREE_API_EXPORT const char* iree_hal_replay_object_type_string(
    iree_hal_replay_object_type_t object_type) {
  switch (object_type) {
    case IREE_HAL_REPLAY_OBJECT_TYPE_NONE:
      return "none";
    case IREE_HAL_REPLAY_OBJECT_TYPE_DEVICE:
      return "device";
    case IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR:
      return "allocator";
    case IREE_HAL_REPLAY_OBJECT_TYPE_POOL:
      return "pool";
    case IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER:
      return "buffer";
    case IREE_HAL_REPLAY_OBJECT_TYPE_COMMAND_BUFFER:
      return "command_buffer";
    case IREE_HAL_REPLAY_OBJECT_TYPE_EXECUTABLE:
      return "executable";
    case IREE_HAL_REPLAY_OBJECT_TYPE_SEMAPHORE:
      return "semaphore";
    case IREE_HAL_REPLAY_OBJECT_TYPE_FILE:
      return "file";
    case IREE_HAL_REPLAY_OBJECT_TYPE_CHANNEL:
      return "channel";
    case IREE_HAL_REPLAY_OBJECT_TYPE_HOST_CALL:
      return "host_call";
    default:
      return "unknown";
  }
}

IREE_API_EXPORT const char* iree_hal_replay_operation_code_string(
    iree_hal_replay_operation_code_t operation_code) {
  switch (operation_code) {
    case IREE_HAL_REPLAY_OPERATION_CODE_NONE:
      return "none";
    case IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_BEGIN:
      return "replay.scope_begin";
    case IREE_HAL_REPLAY_OPERATION_CODE_REPLAY_SCOPE_END:
      return "replay.scope_end";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_TRIM:
      return "device.trim";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_REFINE_TOPOLOGY_EDGE:
      return "device.refine_topology_edge";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_ASSIGN_TOPOLOGY_INFO:
      return "device.assign_topology_info";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_CHANNEL:
      return "device.create_channel";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_COMMAND_BUFFER:
      return "device.create_command_buffer";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_LOAD_EXECUTABLE:
      return "device.load_executable";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_IMPORT_FILE:
      return "device.import_file";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_CREATE_SEMAPHORE:
      return "device.create_semaphore";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUERY_QUEUE_POOL_BACKEND:
      return "device.query_queue_pool_backend";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ALLOCA:
      return "device.queue_alloca";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DEALLOCA:
      return "device.queue_dealloca";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_FILL:
      return "device.queue_fill";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_UPDATE:
      return "device.queue_update";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_COPY:
      return "device.queue_copy";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_READ:
      return "device.queue_read";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_WRITE:
      return "device.queue_write";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_HOST_CALL:
      return "device.queue_host_call";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_DISPATCH:
      return "device.queue_dispatch";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_EXECUTE:
      return "device.queue_execute";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_TIMESTAMP:
      return "device.queue_timestamp";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_WAIT:
      return "device.queue_atomic_wait";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_STORE:
      return "device.queue_atomic_store";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_ATOMIC_RMW:
      return "device.queue_atomic_rmw";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_QUEUE_FLUSH:
      return "device.queue_flush";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_BEGIN:
      return "device.profiling_begin";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_FLUSH:
      return "device.profiling_flush";
    case IREE_HAL_REPLAY_OPERATION_CODE_DEVICE_PROFILING_END:
      return "device.profiling_end";
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_TRIM:
      return "allocator.trim";
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_QUERY_MEMORY_HEAPS:
      return "allocator.query_memory_heaps";
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_ALLOCATE_BUFFER:
      return "allocator.allocate_buffer";
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_IMPORT_BUFFER:
      return "allocator.import_buffer";
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_EXPORT_BUFFER:
      return "allocator.export_buffer";
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_QUERY_GRANULARITY:
      return "allocator.virtual_memory_query_granularity";
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RESERVE:
      return "allocator.virtual_memory_reserve";
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE:
      return "allocator.virtual_memory_release";
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE:
      return "allocator.physical_memory_allocate";
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_FREE:
      return "allocator.physical_memory_free";
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_MAP:
      return "allocator.virtual_memory_map";
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP:
      return "allocator.virtual_memory_unmap";
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT:
      return "allocator.virtual_memory_protect";
    case IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_ADVISE:
      return "allocator.virtual_memory_advise";
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_MAP_RANGE:
      return "buffer.map_range";
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_UNMAP_RANGE:
      return "buffer.unmap_range";
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_INVALIDATE_RANGE:
      return "buffer.invalidate_range";
    case IREE_HAL_REPLAY_OPERATION_CODE_BUFFER_FLUSH_RANGE:
      return "buffer.flush_range";
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_BEGIN:
      return "command_buffer.begin";
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_END:
      return "command_buffer.end";
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_BEGIN_DEBUG_GROUP:
      return "command_buffer.begin_debug_group";
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_END_DEBUG_GROUP:
      return "command_buffer.end_debug_group";
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_EXECUTION_BARRIER:
      return "command_buffer.execution_barrier";
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ADVISE_BUFFER:
      return "command_buffer.advise_buffer";
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_FILL_BUFFER:
      return "command_buffer.fill_buffer";
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_UPDATE_BUFFER:
      return "command_buffer.update_buffer";
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_COPY_BUFFER:
      return "command_buffer.copy_buffer";
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_COLLECTIVE:
      return "command_buffer.collective";
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_DISPATCH:
      return "command_buffer.dispatch";
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_WAIT:
      return "command_buffer.atomic_wait";
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_STORE:
      return "command_buffer.atomic_store";
    case IREE_HAL_REPLAY_OPERATION_CODE_COMMAND_BUFFER_ATOMIC_RMW:
      return "command_buffer.atomic_rmw";
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_FUNCTION_COUNT:
      return "executable.function_count";
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_FUNCTION_INFO:
      return "executable.function_info";
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_FUNCTION_PARAMETERS:
      return "executable.function_parameters";
    case IREE_HAL_REPLAY_OPERATION_CODE_EXECUTABLE_LOOKUP_FUNCTION_BY_NAME:
      return "executable.lookup_function_by_name";
    default:
      return "unknown";
  }
}

IREE_API_EXPORT const char* iree_hal_replay_payload_type_string(
    iree_hal_replay_payload_type_t payload_type) {
  switch (payload_type) {
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_NONE:
      return "none";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_OBJECT:
      return "buffer_object";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_ALLOCATE_BUFFER:
      return "allocator_allocate_buffer";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_IMPORT_BUFFER:
      return "allocator_import_buffer";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_REPLAY_SCOPE:
      return "replay_scope";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE:
      return "buffer_range";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_BUFFER_RANGE_DATA:
      return "buffer_range_data";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_OBJECT:
      return "command_buffer_object";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_EXECUTABLE_LOAD:
      return "executable_load";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DISPATCH:
      return "dispatch";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_EXECUTE:
      return "device_queue_execute";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_SEMAPHORE_OBJECT:
      return "semaphore_object";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_COPY_BUFFER:
      return "command_buffer_copy_buffer";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ALLOCA:
      return "device_queue_alloca";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_EXECUTION_BARRIER:
      return "command_buffer_execution_barrier";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_DEALLOCA:
      return "device_queue_dealloca";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_FILL:
      return "device_queue_fill";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_UPDATE:
      return "device_queue_update";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_COPY:
      return "device_queue_copy";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_FILL_BUFFER:
      return "command_buffer_fill_buffer";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_UPDATE_BUFFER:
      return "command_buffer_update_buffer";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_FILE_OBJECT:
      return "file_object";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_READ:
      return "device_queue_read";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_WRITE:
      return "device_queue_write";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_WAIT:
      return "command_buffer_atomic_wait";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_STORE:
      return "command_buffer_atomic_store";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_COMMAND_BUFFER_ATOMIC_RMW:
      return "command_buffer_atomic_rmw";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_WAIT:
      return "device_queue_atomic_wait";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_STORE:
      return "device_queue_atomic_store";
    case IREE_HAL_REPLAY_PAYLOAD_TYPE_DEVICE_QUEUE_ATOMIC_RMW:
      return "device_queue_atomic_rmw";
    default:
      return "unknown";
  }
}
