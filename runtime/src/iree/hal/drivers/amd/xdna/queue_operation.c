// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/queue_operation.h"

#include "iree/hal/drivers/amd/fabric/buffer.h"
#include "iree/hal/drivers/amd/xdna/barrier.h"
#include "iree/hal/drivers/amd/xdna/transient_buffer.h"

static void iree_hal_amd_xdna_transfer_buffers(
    const iree_hal_amd_xdna_transfer_t* transfer,
    iree_hal_buffer_t** out_source, iree_hal_buffer_t** out_target) {
  const iree_hal_transfer_operation_t* operation = &transfer->operation;
  *out_source = NULL;
  *out_target = NULL;
  switch (operation->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
      if (!operation->fill.length) {
        break;
      }
      *out_target = operation->fill.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      if (!operation->update.length) {
        break;
      }
      *out_target = operation->update.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      if (!operation->copy.length) {
        break;
      }
      *out_source = operation->copy.source_buffer;
      *out_target = operation->copy.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      if (!operation->upload.length) {
        break;
      }
      *out_target = operation->upload.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      if (!operation->download.length) {
        break;
      }
      *out_source = operation->download.source_buffer;
      break;
  }
}

void iree_hal_amd_xdna_operation_release_resources(
    iree_hal_amd_xdna_operation_t* operation) {
  switch (operation->kind) {
    case IREE_HAL_AMD_XDNA_OPERATION_ALLOCA:
      if (operation->alloca.materialized_buffers &&
          operation->alloca.transient_buffers) {
        for (iree_host_size_t i = 0; i < operation->alloca.request_count; ++i) {
          iree_hal_buffer_release(operation->alloca.materialized_buffers[i]);
          iree_hal_buffer_release(operation->alloca.transient_buffers[i]);
        }
      }
      iree_hal_amd_xdna_operation_release_alloca_reservations(operation);
      if (operation->alloca.memory_wait) {
        IREE_ASSERT_TRUE(operation->alloca.memory_wait->kind ==
                         IREE_HAL_AMD_XDNA_MEMORY_WAIT_NONE);
        if (operation->alloca.memory_wait->capacity_wait) {
          iree_hal_pool_wait_free(operation->alloca.memory_wait->capacity_wait);
        }
      }
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_DEALLOCA:
      if (operation->dealloca.transient_buffers) {
        for (iree_host_size_t i = 0; i < operation->dealloca.buffer_count;
             ++i) {
          if (operation->dealloca.marks_owned) {
            iree_hal_buffer_allocation_abort_dealloca(
                operation->dealloca.transient_buffers[i]);
          }
          iree_hal_buffer_release(operation->dealloca.transient_buffers[i]);
        }
      }
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_DISPATCH:
      for (iree_host_size_t i = 0; i < operation->dispatch.binding_count; ++i) {
        iree_hal_buffer_release(
            operation->dispatch.bindings[i].buffer_ref.buffer);
      }
      iree_hal_amd_xdna_invocation_release(operation->native.invocation);
      iree_hal_executable_release(operation->dispatch.executable);
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_EXECUTE:
      iree_hal_amd_xdna_execution_release(operation->execution);
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_TRANSFER:
      for (iree_hal_amd_xdna_transfer_t* transfer = operation->transfer.head;
           transfer; transfer = transfer->next) {
        iree_hal_buffer_t* source = NULL;
        iree_hal_buffer_t* target = NULL;
        iree_hal_amd_xdna_transfer_buffers(transfer, &source, &target);
        iree_hal_buffer_release(source);
        iree_hal_buffer_release(target);
      }
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_FILE:
      if (operation->file.transfer) {
        iree_hal_amd_fabric_file_transfer_deinitialize(
            operation->file.transfer);
      }
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_HOST_CALL:
      iree_hal_resource_release(operation->host_call.call.resource);
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_BARRIER:
      break;
  }
  iree_hal_semaphore_list_release(operation->waits);
}

// Queue transfers use the slab's persistent host execution binding. This
// binding exists independently of public mapping grants, which describe host
// API access and may intentionally be empty for device-only pool scopes.
static iree_status_t iree_hal_amd_xdna_buffer_execution_read(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, iree_byte_span_t* out_span) {
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_native_host_span(buffer, offset, length, out_span));
  return iree_hal_amd_fabric_buffer_invalidate_host_range(
      buffer, offset, out_span->data_length);
}

static iree_status_t iree_hal_amd_xdna_buffer_execution_write(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length, iree_byte_span_t* out_span) {
  return iree_hal_buffer_native_host_span(buffer, offset, length, out_span);
}

static iree_status_t iree_hal_amd_xdna_buffer_execution_publish(
    iree_hal_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t length) {
  return iree_hal_amd_fabric_buffer_flush_host_range(buffer, offset, length);
}

static iree_status_t iree_hal_amd_xdna_update_execute(
    const iree_hal_amd_xdna_transfer_t* transfer) {
  const iree_hal_transfer_operation_t* operation = &transfer->operation;
  iree_byte_span_t target;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_write(
      operation->update.target_buffer, operation->update.target_offset,
      operation->update.length, &target));
  iree_hal_amd_xdna_queue_payload_copy(&transfer->update_payload, target.data);
  return iree_hal_amd_xdna_buffer_execution_publish(
      operation->update.target_buffer, operation->update.target_offset,
      target.data_length);
}

static iree_status_t iree_hal_amd_xdna_transfer_execute(
    const iree_hal_amd_xdna_transfer_t* transfer) {
  const iree_hal_transfer_operation_t* operation = &transfer->operation;
  switch (operation->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
      if (!operation->fill.length) {
        return iree_ok_status();
      }
      iree_byte_span_t fill_target;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_write(
          operation->fill.target_buffer, operation->fill.target_offset,
          operation->fill.length, &fill_target));
      if (operation->fill.pattern_length == 1) {
        memset(fill_target.data, *(const uint8_t*)operation->fill.pattern,
               fill_target.data_length);
      } else {
        for (iree_host_size_t offset = 0; offset < fill_target.data_length;
             offset += operation->fill.pattern_length) {
          memcpy(fill_target.data + offset, operation->fill.pattern,
                 operation->fill.pattern_length);
        }
      }
      return iree_hal_amd_xdna_buffer_execution_publish(
          operation->fill.target_buffer, operation->fill.target_offset,
          fill_target.data_length);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      if (!operation->update.length) {
        return iree_ok_status();
      }
      return iree_hal_amd_xdna_update_execute(transfer);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      if (!operation->copy.length) {
        return iree_ok_status();
      }
      iree_byte_span_t copy_source;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_read(
          operation->copy.source_buffer, operation->copy.source_offset,
          operation->copy.length, &copy_source));
      iree_byte_span_t copy_target;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_write(
          operation->copy.target_buffer, operation->copy.target_offset,
          operation->copy.length, &copy_target));
      const iree_host_size_t copy_length =
          iree_min(copy_source.data_length, copy_target.data_length);
      memcpy(copy_target.data, copy_source.data, copy_length);
      return iree_hal_amd_xdna_buffer_execution_publish(
          operation->copy.target_buffer, operation->copy.target_offset,
          copy_length);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      if (!operation->upload.length) {
        return iree_ok_status();
      }
      iree_byte_span_t upload_target;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_write(
          operation->upload.target_buffer, operation->upload.target_offset,
          operation->upload.length, &upload_target));
      memcpy(upload_target.data, operation->upload.source,
             upload_target.data_length);
      return iree_hal_amd_xdna_buffer_execution_publish(
          operation->upload.target_buffer, operation->upload.target_offset,
          upload_target.data_length);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
      if (!operation->download.length) {
        return iree_ok_status();
      }
      iree_byte_span_t download_source;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_read(
          operation->download.source_buffer, operation->download.source_offset,
          operation->download.length, &download_source));
      memcpy(operation->download.target, download_source.data,
             download_source.data_length);
      return iree_ok_status();
    default:
      IREE_BUILTIN_UNREACHABLE();
  }
}

// Executes a contiguous host phase after the preceding native prefix retires.
// Cache ownership follows the same recipes as direct queue transfers.
static iree_status_t iree_hal_amd_xdna_execution_transfer(
    iree_hal_amd_xdna_execution_t* execution) {
  const iree_hal_amd_xdna_record_t* record = execution->cursor;
  iree_hal_buffer_ref_t target;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
      execution->table, record->transfer.target, &target));
  iree_hal_amd_xdna_transfer_t transfer = {0};
  switch (record->kind) {
    case IREE_HAL_AMD_XDNA_RECORD_FILL:
      transfer.operation.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
      transfer.operation.fill.target_buffer = target.buffer;
      transfer.operation.fill.target_offset = target.offset;
      transfer.operation.fill.length = target.length;
      transfer.operation.fill.pattern = record->transfer.pattern;
      transfer.operation.fill.pattern_length = record->transfer.pattern_length;
      break;
    case IREE_HAL_AMD_XDNA_RECORD_UPDATE:
      transfer.operation.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE;
      transfer.operation.update.target_buffer = target.buffer;
      transfer.operation.update.target_offset = target.offset;
      transfer.operation.update.length = target.length;
      transfer.update_payload = record->transfer.payload;
      break;
    case IREE_HAL_AMD_XDNA_RECORD_COPY: {
      iree_hal_buffer_ref_t source;
      IREE_RETURN_IF_ERROR(iree_hal_buffer_binding_table_resolve_ref(
          execution->table, record->transfer.source, &source));
      transfer.operation.type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY;
      transfer.operation.copy.source_buffer = source.buffer;
      transfer.operation.copy.source_offset = source.offset;
      transfer.operation.copy.target_buffer = target.buffer;
      transfer.operation.copy.target_offset = target.offset;
      transfer.operation.copy.length = iree_min(source.length, target.length);
      break;
    }
    case IREE_HAL_AMD_XDNA_RECORD_DISPATCH:
      IREE_BUILTIN_UNREACHABLE();
  }
  return iree_hal_amd_xdna_transfer_execute(&transfer);
}

uint32_t iree_hal_amd_xdna_operation_execute_host(
    iree_hal_amd_xdna_operation_t* operation) {
  uint32_t count = 0;
  switch (operation->kind) {
    case IREE_HAL_AMD_XDNA_OPERATION_HOST_CALL: {
      IREE_TRACE_ZONE_BEGIN_NAMED(z0, "iree_hal_amd_xdna_host_call");
      iree_hal_host_call_context_t context = {
          .queue = (iree_hal_queue_t*)operation->queue,
          .signal_semaphore_list = operation->signals,
      };
      operation->status =
          operation->host_call.call.fn(operation->host_call.call.user_data,
                                       operation->host_call.args, &context);
      count = 1;
      IREE_TRACE_ZONE_END(z0);
      if (iree_status_is_deferred(operation->status)) {
        // The callback owns terminal publication through its cloned list.
        // Retire only our captured references and callback-state resource.
        iree_status_free(operation->status);
        operation->status = iree_ok_status();
        iree_hal_semaphore_list_release(operation->signals);
        operation->signals = iree_hal_semaphore_list_empty();
      }
      break;
    }
    case IREE_HAL_AMD_XDNA_OPERATION_EXECUTE:
      while (operation->execution->cursor &&
             operation->execution->cursor->kind !=
                 IREE_HAL_AMD_XDNA_RECORD_DISPATCH &&
             iree_status_is_ok(operation->status)) {
        operation->status =
            iree_hal_amd_xdna_execution_transfer(operation->execution);
        if (count < UINT32_MAX) {
          ++count;
        }
        if (iree_status_is_ok(operation->status)) {
          operation->execution->cursor = operation->execution->cursor->next;
        }
      }
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_TRANSFER:
      for (iree_hal_amd_xdna_transfer_t* transfer = operation->transfer.head;
           transfer && iree_status_is_ok(operation->status);
           transfer = transfer->next) {
        operation->status = iree_hal_amd_xdna_transfer_execute(transfer);
        if (count < UINT32_MAX) {
          ++count;
        }
      }
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_FILE:
      operation->status = operation->file.io_completed
                              ? iree_hal_amd_fabric_file_transfer_finish(
                                    operation->file.transfer)
                              : iree_hal_amd_fabric_file_transfer_prepare(
                                    operation->file.transfer);
      count =
          operation->file.io_completed || !operation->file.transfer->async.file;
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_BARRIER:
    case IREE_HAL_AMD_XDNA_OPERATION_ALLOCA:
    case IREE_HAL_AMD_XDNA_OPERATION_DEALLOCA:
    case IREE_HAL_AMD_XDNA_OPERATION_DISPATCH:
      IREE_BUILTIN_UNREACHABLE();
  }
  return count;
}

iree_status_t iree_hal_amd_xdna_queue_host_call(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_host_call_t call,
    const uint64_t args[4], iree_hal_host_call_flags_t flags) {
  iree_hal_amd_xdna_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_allocate(
      (iree_hal_amd_xdna_queue_t*)base, waits, signals,
      IREE_HAL_AMD_XDNA_OPERATION_HOST_CALL, &operation));
  operation->host_call.call = call;
  iree_hal_resource_retain(call.resource);
  memcpy(operation->host_call.args, args, sizeof(operation->host_call.args));
  operation->host_call.flags = flags;
  return iree_hal_amd_xdna_operation_submit(operation);
}

iree_status_t iree_hal_amd_xdna_queue_transfer(
    iree_hal_queue_t* base, const iree_hal_semaphore_list_t waits,
    const iree_hal_semaphore_list_t signals, iree_host_size_t count,
    const iree_hal_transfer_operation_t* operations,
    const iree_hal_queue_barriers_t* barriers) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_queue_barriers_validate(barriers));
  iree_hal_amd_xdna_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_allocate(
      (iree_hal_amd_xdna_queue_t*)base, waits, signals,
      IREE_HAL_AMD_XDNA_OPERATION_TRANSFER, &operation));
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < count && iree_status_is_ok(status); ++i) {
    iree_hal_amd_xdna_transfer_t* captured = NULL;
    status = iree_hal_amd_xdna_queue_capture_allocate_metadata(
        &operation->capture, sizeof(*captured), (void**)&captured);
    if (!iree_status_is_ok(status)) {
      break;
    }
    *captured = (iree_hal_amd_xdna_transfer_t){0};
    captured->operation = operations[i];
    if (captured->operation.type == IREE_HAL_TRANSFER_OPERATION_TYPE_FILL &&
        captured->operation.fill.length) {
      memcpy(captured->fill_pattern, captured->operation.fill.pattern,
             captured->operation.fill.pattern_length);
      captured->operation.fill.pattern = captured->fill_pattern;
    } else if (captured->operation.type ==
                   IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE &&
               captured->operation.update.length) {
      const uint8_t* source =
          (const uint8_t*)captured->operation.update.source_buffer +
          captured->operation.update.source_offset;
      status = iree_hal_amd_xdna_queue_capture_payload(
          &operation->capture,
          iree_make_const_byte_span(
              source, (iree_host_size_t)captured->operation.update.length),
          &captured->update_payload);
      captured->operation.update.source_buffer = NULL;
      captured->operation.update.source_offset = 0;
    }
    if (iree_status_is_ok(status)) {
      iree_hal_buffer_t* source = NULL;
      iree_hal_buffer_t* target = NULL;
      iree_hal_amd_xdna_transfer_buffers(captured, &source, &target);
      iree_hal_buffer_retain(source);
      iree_hal_buffer_retain(target);
      if (operation->transfer.tail) {
        operation->transfer.tail->next = captured;
      } else {
        operation->transfer.head = captured;
      }
      operation->transfer.tail = captured;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_operation_submit(operation);
  } else {
    iree_hal_amd_xdna_operation_discard(operation);
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_queue_file_transfer(
    iree_hal_queue_t* base, const iree_hal_semaphore_list_t waits,
    const iree_hal_semaphore_list_t signals, iree_hal_file_t* file,
    iree_hal_memory_access_t file_access, uint64_t file_offset,
    iree_hal_buffer_t* buffer, iree_device_size_t buffer_offset,
    iree_device_size_t length, const iree_hal_queue_barriers_t* barriers) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_queue_barriers_validate(barriers));
  iree_hal_amd_xdna_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_allocate(
      (iree_hal_amd_xdna_queue_t*)base, waits, signals,
      IREE_HAL_AMD_XDNA_OPERATION_FILE, &operation));
  iree_status_t status = iree_hal_amd_xdna_queue_capture_allocate_metadata(
      &operation->capture, sizeof(*operation->file.transfer),
      (void**)&operation->file.transfer);
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_fabric_file_transfer_initialize(
        file, file_access, file_offset, buffer, buffer_offset, length,
        operation->file.transfer);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_amd_xdna_operation_submit(operation);
  } else {
    iree_hal_amd_xdna_operation_discard(operation);
  }
  return status;
}

iree_status_t iree_hal_amd_xdna_queue_read(
    iree_hal_queue_t* base, const iree_hal_semaphore_list_t waits,
    const iree_hal_semaphore_list_t signals, iree_hal_file_t* source_file,
    uint64_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const iree_hal_queue_barriers_t* barriers, iree_hal_read_flags_t flags) {
  (void)flags;
  return iree_hal_amd_xdna_queue_file_transfer(
      base, waits, signals, source_file, IREE_HAL_MEMORY_ACCESS_READ,
      source_offset, target_buffer, target_offset, length, barriers);
}

iree_status_t iree_hal_amd_xdna_queue_write(
    iree_hal_queue_t* base, const iree_hal_semaphore_list_t waits,
    const iree_hal_semaphore_list_t signals, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_file_t* target_file,
    uint64_t target_offset, iree_device_size_t length,
    const iree_hal_queue_barriers_t* barriers, iree_hal_write_flags_t flags) {
  (void)flags;
  return iree_hal_amd_xdna_queue_file_transfer(
      base, waits, signals, target_file, IREE_HAL_MEMORY_ACCESS_WRITE,
      target_offset, source_buffer, source_offset, length, barriers);
}
