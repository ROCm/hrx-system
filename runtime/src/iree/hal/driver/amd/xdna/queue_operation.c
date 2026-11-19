// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/xdna/queue_operation.h"

#include "iree/hal/driver/amd/fabric/buffer.h"
#include "iree/hal/driver/amd/xdna/barrier.h"

static void iree_hal_amd_xdna_transfer_buffers(
    const iree_hal_amd_xdna_transfer_t* transfer,
    iree_hal_buffer_t** out_source, iree_hal_buffer_t** out_target) {
  const iree_hal_transfer_operation_t* operation = &transfer->operation;
  *out_source = NULL;
  *out_target = NULL;
  switch (operation->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL:
      *out_target = operation->fill.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      *out_target = operation->update.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY:
      *out_source = operation->copy.source_buffer;
      *out_target = operation->copy.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD:
      *out_target = operation->upload.target_buffer;
      break;
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD:
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
      iree_hal_amd_xdna_file_release(operation);
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
    const iree_hal_amd_xdna_transfer_t* transfer,
    const iree_hal_transfer_region_t* region) {
  const iree_hal_transfer_operation_t* operation = &transfer->operation;
  iree_byte_span_t target;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_write(
      operation->update.target_buffer, operation->update.target_offset,
      iree_hal_transfer_range_length(region->row_length, region->row_count,
                                     region->slice_count,
                                     &region->target_strides),
      &target));
  iree_hal_amd_xdna_queue_payload_copy(&transfer->update_payload, target.data,
                                       region);
  return iree_hal_amd_xdna_buffer_execution_publish(
      operation->update.target_buffer, operation->update.target_offset,
      target.data_length);
}

static iree_status_t iree_hal_amd_xdna_transfer_execute(
    const iree_hal_amd_xdna_transfer_t* transfer,
    const iree_hal_transfer_region_t* region) {
  const iree_hal_transfer_operation_t* operation = &transfer->operation;
  const uint64_t source_length = iree_hal_transfer_range_length(
      region->row_length, region->row_count, region->slice_count,
      &region->source_strides);
  const uint64_t target_length = iree_hal_transfer_range_length(
      region->row_length, region->row_count, region->slice_count,
      &region->target_strides);
  if (!region->row_length || !region->row_count || !region->slice_count) {
    return iree_ok_status();
  }
  switch (operation->type) {
    case IREE_HAL_TRANSFER_OPERATION_TYPE_FILL: {
      iree_byte_span_t fill_target;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_write(
          operation->fill.target_buffer, operation->fill.target_offset,
          target_length, &fill_target));
      const uint64_t row_length = region->row_length;
      const uint64_t row_count = region->row_count;
      const uint64_t slice_count = region->slice_count;
      const uint64_t row_stride =
          row_count > 1 ? region->target_strides.row_stride : 0;
      const uint64_t slice_stride =
          slice_count > 1 ? region->target_strides.slice_stride : 0;
      for (uint64_t z = 0; z < slice_count; ++z) {
        for (uint64_t y = 0; y < row_count; ++y) {
          uint8_t* row = fill_target.data + z * slice_stride + y * row_stride;
          if (operation->fill.pattern_length == 1) {
            memset(row, *(const uint8_t*)operation->fill.pattern, row_length);
          } else {
            for (uint64_t offset = 0; offset < row_length;
                 offset += operation->fill.pattern_length) {
              memcpy(row + offset, operation->fill.pattern,
                     operation->fill.pattern_length);
            }
          }
        }
      }
      return iree_hal_amd_xdna_buffer_execution_publish(
          operation->fill.target_buffer, operation->fill.target_offset,
          fill_target.data_length);
    }
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE:
      return iree_hal_amd_xdna_update_execute(transfer, region);
    case IREE_HAL_TRANSFER_OPERATION_TYPE_COPY: {
      iree_byte_span_t copy_source;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_read(
          operation->copy.source_buffer, operation->copy.source_offset,
          source_length, &copy_source));
      iree_byte_span_t copy_target;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_write(
          operation->copy.target_buffer, operation->copy.target_offset,
          target_length, &copy_target));
      iree_hal_transfer_copy_host(copy_source.data, copy_target.data, region);
      return iree_hal_amd_xdna_buffer_execution_publish(
          operation->copy.target_buffer, operation->copy.target_offset,
          copy_target.data_length);
    }
    case IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD: {
      iree_byte_span_t upload_target;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_write(
          operation->upload.target_buffer, operation->upload.target_offset,
          target_length, &upload_target));
      iree_hal_transfer_copy_host(operation->upload.source, upload_target.data,
                                  region);
      return iree_hal_amd_xdna_buffer_execution_publish(
          operation->upload.target_buffer, operation->upload.target_offset,
          upload_target.data_length);
    }
    case IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD: {
      iree_byte_span_t download_source;
      IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_buffer_execution_read(
          operation->download.source_buffer, operation->download.source_offset,
          source_length, &download_source));
      iree_hal_transfer_copy_host(download_source.data,
                                  operation->download.target, region);
      return iree_ok_status();
    }
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
  iree_hal_transfer_region_t region = {
      .row_length = target.length,
      .row_count = 1,
      .slice_count = 1,
  };
  if (record->rectangular) {
    region = *(const iree_hal_transfer_region_t*)(record + 1);
  }
  switch (record->kind) {
    case IREE_HAL_AMD_XDNA_RECORD_FILL:
      transfer.operation.type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
      transfer.operation.fill.target_buffer = target.buffer;
      transfer.operation.fill.target_offset = target.offset;
      transfer.operation.fill.pattern = record->transfer.pattern;
      transfer.operation.fill.pattern_length = record->transfer.pattern_length;
      break;
    case IREE_HAL_AMD_XDNA_RECORD_UPDATE:
      transfer.operation.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE;
      transfer.operation.update.target_buffer = target.buffer;
      transfer.operation.update.target_offset = target.offset;
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
      if (!record->rectangular) {
        region.row_length = iree_min(source.length, target.length);
      }
      break;
    }
    case IREE_HAL_AMD_XDNA_RECORD_DISPATCH:
      IREE_BUILTIN_UNREACHABLE();
  }
  return iree_hal_amd_xdna_transfer_execute(&transfer, &region);
}

uint32_t iree_hal_amd_xdna_operation_execute_host(
    iree_hal_amd_xdna_operation_t* operation) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, operation->kind);
  uint32_t count = 0;
  switch (operation->kind) {
    case IREE_HAL_AMD_XDNA_OPERATION_HOST_CALL: {
      IREE_TRACE_ZONE_BEGIN_NAMED(z1, "iree_hal_amd_xdna_host_call");
      iree_hal_host_call_context_t context = {
          .queue = (iree_hal_queue_t*)operation->queue,
          .signal_semaphore_list = operation->signals,
      };
      operation->status =
          operation->host_call.call.fn(operation->host_call.call.user_data,
                                       operation->host_call.args, &context);
      count = 1;
      IREE_TRACE_ZONE_END(z1);
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
        const iree_hal_transfer_region_t region =
            iree_hal_transfer_operation_region(&transfer->operation);
        operation->status =
            iree_hal_amd_xdna_transfer_execute(transfer, &region);
        if (count < UINT32_MAX) {
          ++count;
        }
      }
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_FILE:
      iree_hal_amd_xdna_file_execute(operation);
      break;
    case IREE_HAL_AMD_XDNA_OPERATION_BARRIER:
    case IREE_HAL_AMD_XDNA_OPERATION_ALLOCA:
    case IREE_HAL_AMD_XDNA_OPERATION_DEALLOCA:
    case IREE_HAL_AMD_XDNA_OPERATION_DISPATCH:
      IREE_BUILTIN_UNREACHABLE();
  }
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, count);
  IREE_TRACE_ZONE_END(z0);
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
    const uint64_t length = iree_hal_transfer_operation_length(&operations[i]);
    if (!length) {
      continue;
    }
    iree_hal_amd_xdna_transfer_t* captured = NULL;
    status = iree_hal_amd_xdna_queue_capture_allocate_metadata(
        &operation->capture,
        sizeof(*captured) + sizeof(iree_hal_transfer_region_t),
        (void**)&captured);
    if (!iree_status_is_ok(status)) {
      break;
    }
    *captured = (iree_hal_amd_xdna_transfer_t){0};
    captured->operation = operations[i];
    iree_hal_transfer_operation_capture_region(&captured->operation,
                                               captured + 1);
    if (captured->operation.type == IREE_HAL_TRANSFER_OPERATION_TYPE_FILL) {
      memcpy(captured->fill_pattern, captured->operation.fill.pattern,
             captured->operation.fill.pattern_length);
      captured->operation.fill.pattern = captured->fill_pattern;
    } else if (captured->operation.type ==
               IREE_HAL_TRANSFER_OPERATION_TYPE_UPDATE) {
      const uint8_t* source =
          (const uint8_t*)captured->operation.update.source_buffer +
          captured->operation.update.source_offset;
      status = iree_hal_amd_xdna_queue_capture_payload(
          &operation->capture,
          iree_make_const_byte_span(source, (iree_host_size_t)length),
          captured->operation.update.region, &captured->update_payload);
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
