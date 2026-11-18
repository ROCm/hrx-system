// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/math.h"
#include "iree/hal/driver/amd/fabric/buffer.h"
#include "iree/hal/driver/amd/fabric/file_transfer.h"
#include "iree/hal/driver/amd/status.h"
#include "iree/hal/driver/amd/xdna/barrier.h"
#include "iree/hal/driver/amd/xdna/queue_admission.h"
#include "iree/hal/driver/amd/xdna/queue_operation.h"
#include "iree/hal/driver/amd/xdna/queue_state.h"
#include "iree/hal/utils/file_request.h"
#include "iree/hal/utils/memory_file.h"

#define IREE_HAL_AMD_XDNA_FILE_CAPACITY 32
#define IREE_HAL_AMD_XDNA_FILE_DRAINING (INT64_C(1) << 32)
#define IREE_HAL_AMD_XDNA_FILE_HOST_WINDOW (UINT32_C(1) << 20)

typedef struct iree_hal_amd_xdna_file_binding_t {
  // Borrowed native file, or NULL for captured host memory.
  iree_async_file_t* handle;
  // Borrowed memory-file contents, already adjusted to the binding origin.
  iree_byte_span_t contents;
} iree_hal_amd_xdna_file_binding_t;

typedef struct iree_hal_amd_xdna_file_buffer_t {
  // Qualified native I/O view and exact cache recipe, borrowed from backing.
  iree_hal_amd_fabric_file_source_t source;
  // Private host execution binding for memory files or staged native I/O.
  iree_byte_span_t host;
  // True after native source qualification, even when no direct view exists.
  bool source_prepared;
} iree_hal_amd_xdna_file_buffer_t;

typedef struct iree_hal_amd_xdna_file_slot_t {
  // Enclosing finite owner, retained until this slot retires.
  iree_hal_amd_xdna_file_t* transfer;
  // Native request and its independent short-I/O continuation.
  iree_hal_file_request_t request;
  // Exact destination of the admitted span, including any staged read copy.
  iree_hal_file_transfer_range_t range;
  // Native callback result published by the corresponding ready bit.
  iree_status_t status;
} iree_hal_amd_xdna_file_slot_t;

struct iree_hal_amd_xdna_file_t {
  // Captured operation dominating metadata and logical operand references.
  iree_hal_amd_xdna_operation_t* operation;
  // Immutable file representations qualified during capture.
  iree_hal_amd_xdna_file_binding_t* files;
  // Per-binding facts resolved on first use after producer dependencies.
  iree_hal_amd_xdna_file_buffer_t* buffers;
  // Constant-space cursor over captured or acquired descriptions.
  iree_hal_file_transfer_iterator_t iterator;
  // Low bits publish completed slots; bit 32 owns progress or its worker
  // handoff.
  iree_atomic_int64_t ready;
  // Slots with accepted I/O, excluding completed staged bytes awaiting a copy.
  uint32_t active;
  // Prepared requests waiting for the host service to release its borrow.
  uint32_t pending;
  // Completed reads awaiting exact-range host publication or a staged copy.
  uint32_t finishing;
  // Mask of placed request slots; scalar operations place only one.
  uint32_t capacity;
  // Slot borrowing ordinary staging, or UINT32_MAX when available.
  uint32_t staging_slot;
  // True after the worker has acquired the live request metadata.
  bool begun;
  // First failure stops admission; accepted requests still drain.
  iree_status_t status;
  // Ordinary host staging, allocated lazily only for unpinnable DIRECT views.
  iree_byte_span_t staging;
  // Alignment of the current staging allocation.
  iree_host_size_t staging_alignment;
  // Bounded reusable native metadata, independent of logical row count.
  iree_hal_amd_xdna_file_slot_t slots[];
};

static bool iree_hal_amd_xdna_file_needs_cache(
    const iree_hal_amd_xdna_file_t* transfer, uint32_t index, bool release) {
  const iree_hal_amd_fabric_file_source_t* source =
      &transfer->buffers[index].source;
  const uint32_t effect = release ? IREE_HAL_MEMORY_EFFECT_HOST_FLUSH
                                  : IREE_HAL_MEMORY_EFFECT_HOST_INVALIDATE;
  return source->contents.data
             ? (source->effects.bits & effect) != 0
             : !iree_all_bits_set(
                   iree_hal_buffer_memory_type(
                       transfer->operation->file.batch.buffers.bindings[index]
                           .buffer),
                   IREE_HAL_MEMORY_TYPE_HOST_COHERENT);
}

static iree_status_t iree_hal_amd_xdna_file_cache(
    iree_hal_amd_xdna_file_t* transfer, iree_hal_file_transfer_range_t range,
    bool release) {
  if (!iree_hal_amd_xdna_file_needs_cache(transfer, range.buffer_binding,
                                          release)) {
    return iree_ok_status();
  }
  const iree_hal_amd_fabric_file_source_t* source =
      &transfer->buffers[range.buffer_binding].source;
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, range.length);
  iree_status_t status = iree_ok_status();
  if (source->contents.data) {
    status = IREE_HAL_AMD_STATUS_FROM_AMDF(
        source->api->host_mapping_cache_control(
            source->storage->mapping,
            release ? AMDF_HOST_CACHE_OPERATION_FLUSH
                    : AMDF_HOST_CACHE_OPERATION_INVALIDATE,
            source->source_offset + range.buffer_offset, range.length),
        "host_mapping_cache_control");
  } else {
    const iree_hal_buffer_binding_t binding =
        transfer->operation->file.batch.buffers.bindings[range.buffer_binding];
    status = release ? iree_hal_amd_fabric_buffer_flush_host_range(
                           binding.buffer, binding.offset + range.buffer_offset,
                           range.length)
                     : iree_hal_amd_fabric_buffer_invalidate_host_range(
                           binding.buffer, binding.offset + range.buffer_offset,
                           range.length);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amd_xdna_file_prepare_buffer(
    iree_hal_amd_xdna_file_t* transfer, uint32_t index, bool native) {
  iree_hal_amd_xdna_operation_t* operation = transfer->operation;
  const iree_hal_buffer_binding_t binding =
      operation->file.batch.buffers.bindings[index];
  iree_hal_amd_xdna_file_buffer_t* buffer = &transfer->buffers[index];
  if (!buffer->source_prepared) {
    iree_hal_amd_xdna_queue_t* queue = operation->queue;
    iree_hal_memory_effects_t before = {0}, after = {0};
    IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_file_source_initialize(
        queue->context->api, queue->base.queue_family,
        queue->context->queue_family_ordinal, operation->file.access, binding,
        &buffer->source, &before, &after));
    if (buffer->source.contents.data) {
      // Native XDNA command boundaries provide global visibility. No packet
      // encodes a range-specific recipe; a host view cannot discard one.
      if (iree_hal_memory_effects_requires_resources(before) ||
          iree_hal_memory_effects_requires_resources(after)) {
        return iree_make_status(
            IREE_STATUS_UNIMPLEMENTED,
            "XDNA file view requires ranged queue transitions");
      }
    }
    buffer->source_prepared = true;
  }
  if ((!native || !buffer->source.contents.data) && !buffer->host.data) {
    IREE_RETURN_IF_ERROR(iree_hal_buffer_native_host_span(
        binding.buffer, binding.offset, binding.length, &buffer->host));
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_file_begin(
    iree_hal_amd_xdna_file_t* transfer) {
  const iree_hal_file_transfer_batch_t* batch =
      &transfer->operation->file.batch;
  const iree_hal_file_transfer_list_t* list = &batch->transfers;
  uint64_t count = 0;
  if (list->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT) {
    iree_byte_span_t span;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_native_host_span(
        list->indirect.count_buffer, list->indirect.count_offset, 8, &span));
    IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_buffer_invalidate_host_range(
        list->indirect.count_buffer, list->indirect.count_offset, 8));
    count = iree_unaligned_load_le_u64(span.data);
  }
  IREE_RETURN_IF_ERROR(iree_hal_file_transfer_iterator_initialize(
      batch, count, NULL, &transfer->iterator));
  if (count) {
    const iree_hal_buffer_binding_t binding = list->indirect.records;
    const uint64_t length = count * sizeof(iree_hal_file_transfer_record_t);
    iree_byte_span_t span;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_native_host_span(
        binding.buffer, binding.offset, length, &span));
    IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_buffer_invalidate_host_range(
        binding.buffer, binding.offset, length));
    iree_hal_file_transfer_iterator_acquire(
        &transfer->iterator, (const void*)span.data, (iree_host_size_t)count);
  }
  transfer->begun = true;
  return iree_ok_status();
}

static void iree_hal_amd_xdna_file_drain(iree_hal_amd_xdna_file_t* transfer,
                                         iree_async_proactor_t* proactor);

static void iree_hal_amd_xdna_file_complete(void* user_data,
                                            iree_status_t status) {
  iree_hal_amd_xdna_file_slot_t* slot = user_data;
  iree_hal_amd_xdna_file_t* transfer = slot->transfer;
  iree_async_proactor_t* proactor =
      transfer->files[slot->range.file_binding].handle->proactor;
  slot->status = status;
  const uint32_t bit = UINT32_C(1) << (slot - transfer->slots);
  const int64_t previous = iree_atomic_fetch_or(
      &transfer->ready, IREE_HAL_AMD_XDNA_FILE_DRAINING | bit,
      iree_memory_order_acq_rel);
  if (!(previous & IREE_HAL_AMD_XDNA_FILE_DRAINING)) {
    iree_hal_amd_xdna_file_drain(transfer, proactor);
  }
  // This callback has released its borrow; another drainer can retire storage.
}

static iree_status_t iree_hal_amd_xdna_file_prepare_request(
    iree_hal_amd_xdna_file_t* transfer, iree_hal_file_transfer_range_t range,
    iree_async_span_t span, uint32_t index) {
  const iree_hal_file_binding_t file =
      transfer->operation->file.batch.files.bindings[range.file_binding];
  iree_hal_amd_xdna_file_slot_t* slot = &transfer->slots[index];
  IREE_RETURN_IF_ERROR(iree_hal_file_request_prepare(
      file.file, transfer->operation->file.access,
      file.offset + range.file_offset, range.length, span, &slot->request));
  iree_hal_file_request_bind(&slot->request,
                             (iree_hal_file_transfer_callback_t){
                                 iree_hal_amd_xdna_file_complete, slot});
  range.length = iree_hal_file_request_span(&slot->request).length;
  slot->range = range;
  iree_hal_file_transfer_iterator_advance(&transfer->iterator, range.length);
  return iree_ok_status();
}

static iree_status_t iree_hal_amd_xdna_file_prepare_direct(
    iree_hal_amd_xdna_file_t* transfer, iree_hal_file_transfer_range_t range,
    iree_host_size_t capacity, uint32_t index) {
  const iree_hal_amd_xdna_file_binding_t* file =
      &transfer->files[range.file_binding];
  const iree_hal_amd_xdna_file_buffer_t* buffer =
      &transfer->buffers[range.buffer_binding];
  const iree_hal_amd_fabric_file_source_t* source = &buffer->source;
  const iree_host_size_t length =
      (iree_host_size_t)iree_min(range.length, capacity);
  iree_async_span_t span;
  if (source->region) {
    if (source->region->proactor != file->handle->proactor) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "registered buffer and file use different I/O proactors");
    }
    span = iree_async_span_make(
        source->region, source->source_offset + range.buffer_offset, length);
  } else {
    uint8_t* data =
        source->contents.data ? source->contents.data : buffer->host.data;
    span = iree_async_span_from_ptr(data + range.buffer_offset, length);
  }
  return iree_hal_amd_xdna_file_prepare_request(transfer, range, span, index);
}

// Copies and allocation stay on the existing service. The only staging slab
// is at most the former scalar capacity; direct I/O uses no payload allocation.
static iree_status_t iree_hal_amd_xdna_file_prepare_staging(
    iree_hal_amd_xdna_file_t* transfer, iree_hal_file_transfer_range_t range) {
  const iree_hal_file_binding_t file =
      transfer->operation->file.batch.files.bindings[range.file_binding];
  const iree_async_file_requirements_t* requirements =
      iree_hal_file_requirements(file.file);
  const iree_host_size_t alignment =
      iree_max(transfer->operation->file.access == IREE_HAL_MEMORY_ACCESS_READ
                   ? requirements->read.buffer_alignment
                   : requirements->write.buffer_alignment,
               sizeof(void*));
  const iree_host_size_t capacity = (iree_host_size_t)iree_min(
      range.length, IREE_HAL_AMD_XDNA_FILE_HOST_WINDOW);
  if (transfer->staging.data_length < capacity ||
      transfer->staging_alignment < alignment) {
    const iree_allocator_t allocator =
        transfer->operation->queue->host_allocator;
    iree_allocator_free_aligned(allocator, transfer->staging.data);
    transfer->staging = iree_byte_span_empty();
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
        allocator, capacity, alignment, 0, (void**)&transfer->staging.data));
    transfer->staging.data_length = capacity;
    transfer->staging_alignment = alignment;
  }
  const uint32_t index = iree_math_count_trailing_zeros_u32(
      transfer->capacity & ~(transfer->active | transfer->pending));
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_file_prepare_request(
      transfer, range,
      iree_async_span_from_ptr(transfer->staging.data,
                               transfer->staging.data_length),
      index));
  if (transfer->operation->file.access == IREE_HAL_MEMORY_ACCESS_WRITE) {
    IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_file_cache(
        transfer, transfer->slots[index].range, false));
    memcpy(
        transfer->staging.data,
        transfer->buffers[range.buffer_binding].host.data + range.buffer_offset,
        transfer->slots[index].range.length);
  }
  transfer->staging_slot = index;
  transfer->pending |= UINT32_C(1) << index;
  return iree_ok_status();
}

void iree_hal_amd_xdna_file_execute(iree_hal_amd_xdna_operation_t* operation) {
  iree_hal_amd_xdna_file_t* transfer = operation->file.state;
  if (!transfer->begun) {
    transfer->status = iree_hal_amd_xdna_file_begin(transfer);
  }
  while (transfer->finishing && iree_status_is_ok(transfer->status)) {
    const uint32_t index =
        iree_math_count_trailing_zeros_u32(transfer->finishing);
    transfer->finishing &= transfer->finishing - 1;
    const iree_hal_file_transfer_range_t range = transfer->slots[index].range;
    if (index == transfer->staging_slot) {
      transfer->status = iree_hal_amd_xdna_file_cache(transfer, range, false);
      if (iree_status_is_ok(transfer->status)) {
        memcpy(transfer->buffers[range.buffer_binding].host.data +
                   range.buffer_offset,
               transfer->staging.data, range.length);
      }
      transfer->staging_slot = UINT32_MAX;
    }
    if (iree_status_is_ok(transfer->status)) {
      transfer->status = iree_hal_amd_xdna_file_cache(transfer, range, true);
    }
  }
  uint64_t budget = IREE_HAL_AMD_XDNA_FILE_HOST_WINDOW;
  uint32_t available = transfer->capacity & ~transfer->active;
  while (budget && available && iree_status_is_ok(transfer->status)) {
    iree_hal_file_transfer_range_t range;
    transfer->status =
        iree_hal_file_transfer_iterator_peek(&transfer->iterator, &range);
    if (!iree_status_is_ok(transfer->status) || !range.length) {
      break;
    }
    iree_hal_amd_xdna_file_binding_t* file =
        &transfer->files[range.file_binding];
    transfer->status = iree_hal_amd_xdna_file_prepare_buffer(
        transfer, range.buffer_binding, file->handle != NULL);
    if (!iree_status_is_ok(transfer->status)) {
      break;
    }
    iree_hal_amd_xdna_file_buffer_t* buffer =
        &transfer->buffers[range.buffer_binding];
    if (file->handle) {
      if (!buffer->source.contents.data &&
          iree_any_bit_set(file->handle->flags, IREE_ASYNC_FILE_FLAG_DIRECT)) {
        if (transfer->staging_slot == UINT32_MAX) {
          transfer->status =
              iree_hal_amd_xdna_file_prepare_staging(transfer, range);
        }
        break;
      }
      if (!iree_hal_amd_xdna_file_needs_cache(transfer, range.buffer_binding,
                                              false)) {
        break;
      }
      const uint32_t index = iree_math_count_trailing_zeros_u32(available);
      const uint32_t bit = UINT32_C(1) << index;
      available &= ~bit;
      // Cache work is bounded by useful bytes, never the whole binding. In
      // particular a tiny strided request cannot flush a model-sized arena.
      const iree_async_file_requirements_t* requirements =
          iree_hal_file_requirements(
              operation->file.batch.files.bindings[range.file_binding].file);
      const iree_async_file_transfer_requirements_t* geometry =
          operation->file.access == IREE_HAL_MEMORY_ACCESS_READ
              ? &requirements->read
              : &requirements->write;
      // Exceed the budget only to preserve one native alignment unit. Request
      // preparation still validates the complete logical range.
      const iree_host_size_t capacity =
          iree_max(budget, iree_max(geometry->buffer_alignment,
                                    iree_max(geometry->offset_alignment,
                                             geometry->length_alignment)));
      transfer->status = iree_hal_amd_xdna_file_prepare_direct(transfer, range,
                                                               capacity, index);
      if (iree_status_is_ok(transfer->status)) {
        range = transfer->slots[index].range;
        transfer->status = iree_hal_amd_xdna_file_cache(transfer, range, false);
      }
      if (iree_status_is_ok(transfer->status)) {
        transfer->pending |= bit;
        budget -= iree_min(range.length, budget);
      }
      continue;
    }
    range.length = iree_min(range.length, budget);
    transfer->status = iree_hal_amd_xdna_file_cache(transfer, range, false);
    if (!iree_status_is_ok(transfer->status)) {
      break;
    }
    uint8_t* file_data = file->contents.data + range.file_offset;
    uint8_t* buffer_data = buffer->host.data + range.buffer_offset;
    if (operation->file.access == IREE_HAL_MEMORY_ACCESS_READ) {
      memmove(buffer_data, file_data, range.length);
    } else {
      memmove(file_data, buffer_data, range.length);
    }
    if (operation->file.access == IREE_HAL_MEMORY_ACCESS_READ) {
      transfer->status = iree_hal_amd_xdna_file_cache(transfer, range, true);
    }
    iree_hal_file_transfer_iterator_advance(&transfer->iterator, range.length);
    budget -= range.length;
  }
}

static void iree_hal_amd_xdna_file_publish(
    void* user_data, iree_async_operation_t* base_operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  (void)base_operation;
  (void)flags;
  iree_hal_amd_xdna_operation_t* operation = user_data;
  operation->status = iree_status_join(operation->file.state->status, status);
  operation->file.state->status = iree_ok_status();
  iree_hal_amd_xdna_operation_complete(operation);
}

static void iree_hal_amd_xdna_file_finish(iree_hal_amd_xdna_file_t* transfer,
                                          iree_async_proactor_t* proactor) {
  iree_hal_amd_xdna_operation_t* operation = transfer->operation;
  if (proactor == operation->queue->proactor) {
    iree_hal_amd_xdna_file_publish(operation, NULL, iree_ok_status(), 0);
    return;
  }
  iree_async_operation_initialize(&operation->continuation,
                                  IREE_ASYNC_OPERATION_TYPE_NOP,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  iree_hal_amd_xdna_file_publish, operation);
  iree_status_t status = iree_async_proactor_submit_one(
      operation->queue->proactor, &operation->continuation);
  if (!iree_status_is_ok(status)) {
    status = iree_status_join(transfer->status, status);
    transfer->status = iree_ok_status();
    // Failed handoff cannot serialize operation reclamation. Preserve its owner
    // graph exactly as other unreturnable queue-service results do.
    iree_slim_mutex_lock(&operation->queue->state_mutex);
    operation->queue->operations_retained = true;
    iree_slim_mutex_unlock(&operation->queue->state_mutex);
    iree_hal_amd_xdna_queue_report(operation->queue, iree_status_code(status),
                                   iree_status_message(status));
    iree_hal_semaphore_list_fail(operation->signals, status);
  }
}

static void iree_hal_amd_xdna_file_submit(iree_hal_amd_xdna_file_t* transfer,
                                          iree_async_proactor_t* proactor,
                                          iree_async_operation_t** requests,
                                          iree_host_size_t count,
                                          uint32_t mask) {
  if (!count) {
    return;
  }
  if (iree_status_is_ok(transfer->status)) {
    transfer->status = iree_async_proactor_submit(
        proactor, (iree_async_operation_list_t){requests, count});
  }
  if (!iree_status_is_ok(transfer->status)) {
    transfer->active &= ~mask;
  }
}

static void iree_hal_amd_xdna_file_drain(
    iree_hal_amd_xdna_file_t* transfer,
    iree_async_proactor_t* completion_proactor) {
  for (;;) {
    uint32_t completed = (uint32_t)iree_atomic_exchange(
        &transfer->ready, IREE_HAL_AMD_XDNA_FILE_DRAINING,
        iree_memory_order_acquire);
    transfer->active &= ~completed;
    while (completed) {
      const uint32_t index = iree_math_count_trailing_zeros_u32(completed);
      completed &= completed - 1;
      iree_hal_amd_xdna_file_slot_t* slot = &transfer->slots[index];
      if (iree_status_is_ok(slot->status) &&
          transfer->operation->file.access == IREE_HAL_MEMORY_ACCESS_READ &&
          (index == transfer->staging_slot ||
           iree_hal_amd_xdna_file_needs_cache(
               transfer, slot->range.buffer_binding, true))) {
        transfer->finishing |= UINT32_C(1) << index;
      } else if (index == transfer->staging_slot) {
        transfer->staging_slot = UINT32_MAX;
      }
      transfer->status = iree_status_join(transfer->status, slot->status);
      slot->status = iree_ok_status();
    }
    bool needs_worker = transfer->finishing != 0;
    iree_async_operation_t* requests[IREE_HAL_AMD_XDNA_FILE_CAPACITY];
    iree_async_proactor_t* proactor = NULL;
    iree_host_size_t count = 0;
    uint32_t mask = 0;
    while (transfer->pending) {
      const uint32_t index =
          iree_math_count_trailing_zeros_u32(transfer->pending);
      const uint32_t bit = UINT32_C(1) << index;
      transfer->pending &= ~bit;
      iree_hal_amd_xdna_file_slot_t* slot = &transfer->slots[index];
      iree_async_proactor_t* next =
          transfer->files[slot->range.file_binding].handle->proactor;
      if (count && proactor != next) {
        iree_hal_amd_xdna_file_submit(transfer, proactor, requests, count,
                                      mask);
        count = mask = 0;
      }
      proactor = next;
      requests[count++] = &slot->request.read.base;
      mask |= bit;
      transfer->active |= bit;
    }
    uint32_t available =
        transfer->capacity & ~(transfer->active | transfer->finishing);
    while (available && !needs_worker && iree_status_is_ok(transfer->status)) {
      iree_hal_file_transfer_range_t range;
      transfer->status =
          iree_hal_file_transfer_iterator_peek(&transfer->iterator, &range);
      if (!iree_status_is_ok(transfer->status) || !range.length) {
        break;
      }
      iree_hal_amd_xdna_file_binding_t* file =
          &transfer->files[range.file_binding];
      iree_hal_amd_xdna_file_buffer_t* buffer =
          &transfer->buffers[range.buffer_binding];
      if (!file->handle || !buffer->source_prepared) {
        needs_worker = true;
        break;
      }
      if (!buffer->source.contents.data &&
          iree_any_bit_set(file->handle->flags, IREE_ASYNC_FILE_FLAG_DIRECT)) {
        needs_worker = transfer->staging_slot == UINT32_MAX;
        break;
      }
      if (iree_hal_amd_xdna_file_needs_cache(transfer, range.buffer_binding,
                                             false)) {
        needs_worker = true;
        break;
      }
      const uint32_t index = iree_math_count_trailing_zeros_u32(available);
      const uint32_t bit = UINT32_C(1) << index;
      available &= ~bit;
      transfer->status = iree_hal_amd_xdna_file_prepare_direct(
          transfer, range, UINT32_C(1) << 30, index);
      if (!iree_status_is_ok(transfer->status)) {
        break;
      }
      if (count && proactor != file->handle->proactor) {
        iree_hal_amd_xdna_file_submit(transfer, proactor, requests, count,
                                      mask);
        count = mask = 0;
      }
      proactor = file->handle->proactor;
      requests[count++] = &transfer->slots[index].request.read.base;
      mask |= bit;
      transfer->active |= bit;
    }
    iree_hal_amd_xdna_file_submit(transfer, proactor, requests, count, mask);
    if (iree_status_is_ok(transfer->status) && needs_worker) {
      iree_hal_amd_xdna_queue_service_enqueue(
          &transfer->operation->queue->host_worker,
          &transfer->operation->service_item);
      return;
    }
    if (!transfer->active) {
      iree_hal_amd_xdna_file_finish(transfer, completion_proactor);
      return;
    }
    int64_t expected = IREE_HAL_AMD_XDNA_FILE_DRAINING;
    if (iree_atomic_compare_exchange_strong(&transfer->ready, &expected, 0,
                                            iree_memory_order_release,
                                            iree_memory_order_relaxed)) {
      return;
    }
  }
}

void iree_hal_amd_xdna_file_continue(iree_hal_amd_xdna_operation_t* operation) {
  iree_hal_amd_xdna_file_t* transfer = operation->file.state;
  transfer->status = iree_status_join(transfer->status, operation->status);
  operation->status = iree_ok_status();
  iree_hal_amd_xdna_file_drain(transfer, operation->queue->proactor);
}

void iree_hal_amd_xdna_file_release(iree_hal_amd_xdna_operation_t* operation) {
  iree_hal_file_transfer_batch_t* batch = &operation->file.batch;
  for (iree_host_size_t i = 0; i < batch->files.count; ++i) {
    iree_hal_file_release(batch->files.bindings[i].file);
  }
  for (iree_host_size_t i = 0; i < batch->buffers.count; ++i) {
    iree_hal_buffer_release(batch->buffers.bindings[i].buffer);
  }
  if (batch->transfers.type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT) {
    iree_hal_buffer_release(batch->transfers.indirect.records.buffer);
    iree_hal_buffer_release(batch->transfers.indirect.count_buffer);
  }
  if (operation->file.state) {
    iree_allocator_free_aligned(operation->queue->host_allocator,
                                operation->file.state->staging.data);
  }
}

iree_status_t iree_hal_amd_xdna_queue_file_transfer(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_file_binding_table_t files,
    iree_hal_buffer_binding_table_t buffers,
    const iree_hal_file_transfer_list_t* transfers,
    iree_hal_memory_access_t file_access,
    const iree_hal_queue_barriers_t* barriers) {
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_queue_barriers_validate(barriers));
  iree_host_size_t batch_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_file_transfer_batch_storage_size(
      files, buffers, transfers, &batch_size));
  const uint32_t capacity =
      transfers->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST &&
              transfers->host.count == 1 &&
              transfers->host.transfers[0].row_count == 1 &&
              transfers->host.transfers[0].slice_count == 1
          ? 1
          : IREE_HAL_AMD_XDNA_FILE_CAPACITY;
  iree_host_size_t size = 0, batch_offset = 0, files_offset = 0,
                   buffers_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_amd_xdna_file_t) +
          capacity * sizeof(iree_hal_amd_xdna_file_slot_t),
      &size,
      IREE_STRUCT_FIELD_ALIGNED(batch_size, uint8_t, iree_max_align_t,
                                &batch_offset),
      IREE_STRUCT_FIELD(files.count, iree_hal_amd_xdna_file_binding_t,
                        &files_offset),
      IREE_STRUCT_FIELD(buffers.count, iree_hal_amd_xdna_file_buffer_t,
                        &buffers_offset)));
  iree_hal_amd_xdna_operation_t* operation = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amd_xdna_operation_allocate(
      (iree_hal_amd_xdna_queue_t*)base, waits, signals,
      IREE_HAL_AMD_XDNA_OPERATION_FILE, &operation));
  uint8_t* storage = NULL;
  iree_status_t status = iree_hal_amd_xdna_queue_capture_allocate_metadata(
      &operation->capture, size, (void**)&storage);
  iree_hal_file_transfer_batch_t batch = {0};
  if (iree_status_is_ok(status)) {
    memset(storage, 0, size);
    status = iree_hal_file_transfer_batch_capture(
        base, file_access, files, buffers, transfers, storage + batch_offset,
        &batch);
  }
  if (iree_status_is_ok(status)) {
    operation->file.state = (void*)storage;
    iree_hal_amd_xdna_file_t* transfer = operation->file.state;
    transfer->operation = operation;
    transfer->files = (void*)(storage + files_offset);
    transfer->buffers = (void*)(storage + buffers_offset);
    transfer->capacity = capacity == 32 ? UINT32_MAX : 1;
    transfer->staging_slot = UINT32_MAX;
    iree_atomic_store(&transfer->ready, IREE_HAL_AMD_XDNA_FILE_DRAINING,
                      iree_memory_order_relaxed);
    for (uint32_t i = 0; i < capacity; ++i) {
      transfer->slots[i].transfer = transfer;
    }
    for (iree_host_size_t i = 0; i < files.count && iree_status_is_ok(status);
         ++i) {
      const iree_hal_file_binding_t binding = batch.files.bindings[i];
      iree_hal_amd_xdna_file_binding_t* file = &transfer->files[i];
      file->handle = iree_hal_file_async_handle(binding.file);
      if (!file->handle) {
        if (iree_hal_memory_file_isa(binding.file)) {
          status = iree_hal_memory_file_contents(binding.file, &file->contents);
          if (iree_status_is_ok(status)) {
            file->contents = iree_make_byte_span(
                file->contents.data + binding.offset, binding.length);
          }
        } else {
          status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                    "XDNA file transfers require a memory file "
                                    "or an async file handle");
        }
      }
    }
  }
  if (iree_status_is_ok(status)) {
    operation->file.batch = batch;
    operation->file.access = file_access;
    for (iree_host_size_t i = 0; i < files.count; ++i) {
      iree_hal_file_retain(batch.files.bindings[i].file);
    }
    for (iree_host_size_t i = 0; i < buffers.count; ++i) {
      iree_hal_buffer_retain(batch.buffers.bindings[i].buffer);
    }
    if (transfers->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT) {
      iree_hal_buffer_retain(transfers->indirect.records.buffer);
      iree_hal_buffer_retain(transfers->indirect.count_buffer);
    }
    status = iree_hal_amd_xdna_operation_submit(operation);
  } else {
    iree_hal_amd_xdna_operation_discard(operation);
  }
  return status;
}

static iree_status_t iree_hal_amd_xdna_queue_file_linear(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_file_t* file,
    iree_hal_memory_access_t access, uint64_t file_offset,
    iree_hal_buffer_t* buffer, iree_device_size_t buffer_offset,
    iree_device_size_t length, const iree_hal_queue_barriers_t* barriers) {
  const iree_hal_file_binding_t file_binding = {file, file_offset, length};
  const iree_hal_buffer_binding_t buffer_binding = {buffer, buffer_offset,
                                                    length};
  const iree_hal_file_transfer_t transfer = {
      .row_length = length, .row_count = 1, .slice_count = 1};
  const iree_hal_file_transfer_list_t list = {.host = {1, &transfer}};
  return iree_hal_amd_xdna_queue_file_transfer(
      base, waits, signals, (iree_hal_file_binding_table_t){1, &file_binding},
      (iree_hal_buffer_binding_table_t){1, &buffer_binding}, &list, access,
      barriers);
}

iree_status_t iree_hal_amd_xdna_queue_read(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_file_t* source_file,
    uint64_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const iree_hal_queue_barriers_t* barriers, iree_hal_read_flags_t flags) {
  (void)flags;
  return iree_hal_amd_xdna_queue_file_linear(
      base, waits, signals, source_file, IREE_HAL_MEMORY_ACCESS_READ,
      source_offset, target_buffer, target_offset, length, barriers);
}

iree_status_t iree_hal_amd_xdna_queue_write(
    iree_hal_queue_t* base, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_file_t* target_file,
    uint64_t target_offset, iree_device_size_t length,
    const iree_hal_queue_barriers_t* barriers, iree_hal_write_flags_t flags) {
  (void)flags;
  return iree_hal_amd_xdna_queue_file_linear(
      base, waits, signals, target_file, IREE_HAL_MEMORY_ACCESS_WRITE,
      target_offset, source_buffer, source_offset, length, barriers);
}
