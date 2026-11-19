// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/vulkan/file_transfer.h"

#include "iree/base/internal/math.h"
#include "iree/hal/driver/vulkan/queue_staging.h"
#include "iree/hal/driver/vulkan/transient_buffer.h"
#include "iree/hal/utils/file_request.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/memory_file.h"

// Two native batches overlap I/O with device copies without reserving a full
// request array for every physical staging slot.
#define IREE_HAL_VULKAN_FILE_PIECE_CAPACITY 64
#define IREE_HAL_VULKAN_FILE_BATCH_CAPACITY 32
#define IREE_HAL_VULKAN_FILE_METADATA_READY (INT64_C(1) << 32)
#define IREE_HAL_VULKAN_FILE_STAGING_READY (INT64_C(1) << 33)
#define IREE_HAL_VULKAN_FILE_DRAINING (INT64_C(1) << 34)

typedef enum iree_hal_vulkan_file_kind_e {
  IREE_HAL_VULKAN_FILE_ASYNC,
  IREE_HAL_VULKAN_FILE_HOST,
  IREE_HAL_VULKAN_FILE_DEVICE,
} iree_hal_vulkan_file_kind_t;

typedef struct iree_hal_vulkan_file_binding_t {
  // Representation qualified once, independently of row traversal.
  iree_hal_vulkan_file_kind_t kind;
  union {
    // Borrowed native handle including the exact owning proactor.
    iree_async_file_t* handle;
    // Host-only memory file contents at the file binding origin.
    iree_byte_span_t host;
    // Native memory-file storage retained by the file, not per copy.
    iree_hal_buffer_t* buffer;
  };
} iree_hal_vulkan_file_binding_t;

typedef struct iree_hal_vulkan_file_buffer_t {
  // Original native host import, resolved after producer dependencies.
  iree_hal_vulkan_buffer_host_span_t span;
  // True after resolving this binding's direct-file route.
  bool prepared;
} iree_hal_vulkan_file_buffer_t;

typedef struct iree_hal_vulkan_file_piece_t {
  union {
    // Placed native request owning its partial-I/O progress.
    iree_hal_file_request_t request;
    // Borrowed host memory-file bytes for one staged copy.
    iree_byte_span_t host;
  };
  // Buffer table index for exact direct-host cache operations.
  uint32_t buffer_binding;
  // Byte offset within the captured buffer binding.
  iree_device_size_t buffer_offset;
  // Native copy geometry; row_length is this piece's useful byte count.
  iree_hal_transfer_region_t region;
} iree_hal_vulkan_file_piece_t;

typedef struct iree_hal_vulkan_file_transfer_t iree_hal_vulkan_file_transfer_t;

typedef enum iree_hal_vulkan_file_phase_e {
  IREE_HAL_VULKAN_FILE_IO,
  IREE_HAL_VULKAN_FILE_COPY,
} iree_hal_vulkan_file_phase_t;

typedef struct iree_hal_vulkan_file_window_t {
  // Enclosing capture, live until every accepted request/copy retires.
  iree_hal_vulkan_file_transfer_t* transfer;
  // Shared physical staging loan, or NULL for direct I/O and native files.
  iree_hal_vulkan_queue_staging_slot_t* slot;
  // Exact proactor for all native requests in this window.
  iree_async_proactor_t* proactor;
  // Placed bounded native request records or host spans.
  iree_hal_vulkan_file_piece_t* pieces;
  // Contiguous native copy array borrowed until its completion action.
  iree_hal_transfer_operation_t* copies;
  // Number of prepared pieces and corresponding copies.
  uint32_t count;
  // Bytes used in the staging slot, including alignment between pieces.
  iree_host_size_t staging_length;
  // File representation shared by this window's pieces.
  iree_hal_vulkan_file_kind_t kind;
  // Submitted phase, advanced only by the drainer.
  iree_hal_vulkan_file_phase_t phase;
  // Native I/O callbacks still using placed requests.
  iree_atomic_int32_t pending;
  // First owned failure, published before the terminal ready bit.
  iree_atomic_intptr_t failure;
} iree_hal_vulkan_file_window_t;

struct iree_hal_vulkan_file_transfer_t {
  // Retained by entry/copy actions, staging callbacks, and the active group.
  iree_hal_resource_t resource;
  // Queue supplying native execution and physical staging. Borrowed.
  iree_hal_vulkan_queue_t* queue;
  // Immutable tables and descriptions captured once for the entire operation.
  iree_hal_file_transfer_batch_t batch;
  // Per-file native facts in the same allocation as the capture.
  iree_hal_vulkan_file_binding_t* files;
  // Per-buffer direct-file facts, resolved lazily after producer waits.
  iree_hal_vulkan_file_buffer_t* buffers;
  // Captured terminal timepoints, retained through publication.
  iree_hal_semaphore_list_t signals;
  // Caller-requested global boundary after all file and native accesses.
  iree_hal_vulkan_queue_barriers_t publication;
  // READ for file-to-buffer, WRITE for buffer-to-file.
  iree_hal_memory_access_t access;
  // Constant-space traversal over captured or acquired metadata.
  iree_hal_file_transfer_iterator_t iterator;
  // Completed window bits, metadata/capacity wakeups, and drainer ownership.
  iree_atomic_int64_t ready;
  // Windows whose current phase has not returned ownership.
  uint32_t active;
  // Number of bounded windows in this capture.
  uint32_t window_count;
  // Placed request storage shared by all independently progressing windows.
  iree_hal_vulkan_file_piece_t* pieces;
  // Matching contiguous native copy storage.
  iree_hal_transfer_operation_t* copies;
  // Available placed records. Only the drainer claims and returns these bits.
  uint64_t free_pieces;
  // Owned aggregate failure, modified only by the drainer.
  iree_status_t failure;
  // True after initializing the iterator with the sampled live count.
  bool count_ready;
  // True after reaching the live prefix's end.
  bool exhausted;
  // True after terminal ownership has been claimed.
  bool finishing;
  // Shared ring currently holding the placed capacity waiter, or NULL.
  iree_hal_vulkan_queue_staging_ring_t* waiting_ring;
  // At most one pending capacity admission for the serial iterator.
  iree_hal_vulkan_queue_staging_waiter_t waiter;
  struct {
    // Borrowed download-ring slot while a device metadata copy is in flight.
    iree_hal_vulkan_queue_staging_slot_t* slot;
    // Native metadata copy, borrowed through its completion callback.
    iree_hal_transfer_operation_t copy;
    // Reusable page containing only live acquired records.
    iree_hal_file_transfer_record_t* records;
    // Number of records acquired by this page, or zero for the count word.
    iree_host_size_t record_count;
    // Native metadata copy geometry; row_length is the acquired byte count.
    iree_hal_transfer_region_t region;
    // Owned callback result, transferred by the metadata-ready bit.
    iree_status_t status;
    // True while the native metadata copy owns its slot.
    bool pending;
  } metadata;
  // Independent windows, each borrowing at most one physical staging slot.
  iree_hal_vulkan_file_window_t windows[];
};

static void iree_hal_vulkan_file_notify(
    iree_hal_vulkan_file_transfer_t* transfer, int64_t bits);

static void iree_hal_vulkan_file_release_captures(
    iree_hal_vulkan_file_transfer_t* transfer) {
  for (iree_host_size_t i = 0; i < transfer->batch.files.count; ++i) {
    iree_hal_file_release(transfer->batch.files.bindings[i].file);
  }
  for (iree_host_size_t i = 0; i < transfer->batch.buffers.count; ++i) {
    iree_hal_buffer_release(transfer->batch.buffers.bindings[i].buffer);
  }
  if (transfer->batch.transfers.type ==
      IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT) {
    iree_hal_buffer_release(transfer->batch.transfers.indirect.records.buffer);
    iree_hal_buffer_release(transfer->batch.transfers.indirect.count_buffer);
  }
  transfer->batch = (iree_hal_file_transfer_batch_t){0};
}

static void iree_hal_vulkan_file_destroy(iree_hal_resource_t* resource) {
  iree_hal_vulkan_file_transfer_t* transfer = (void*)resource;
  iree_hal_vulkan_file_release_captures(transfer);
  iree_hal_semaphore_list_release(transfer->signals);
  iree_status_free(transfer->failure);
  iree_allocator_free(transfer->queue->host_allocator, transfer);
}

static const iree_hal_resource_vtable_t iree_hal_vulkan_file_vtable = {
    .destroy = iree_hal_vulkan_file_destroy,
};

static void iree_hal_vulkan_file_finish(
    iree_hal_vulkan_file_transfer_t* transfer) {
  transfer->finishing = true;
  if (transfer->waiting_ring) {
    iree_hal_vulkan_queue_staging_ring_cancel_waiter(transfer->waiting_ring,
                                                     &transfer->waiter);
    transfer->waiting_ring = NULL;
  }
  iree_status_t status = transfer->failure;
  transfer->failure = iree_ok_status();
  iree_hal_vulkan_file_release_captures(transfer);
  if (iree_status_is_ok(status)) {
    // Native submission after host writes supplies the host-to-device domain
    // transition, including for consumers queued before those writes occurred.
    status = iree_hal_vulkan_queue_submit_barrier_with_action(
        transfer->queue, iree_hal_semaphore_list_empty(), transfer->signals,
        &transfer->publication,
        (iree_hal_vulkan_queue_completion_action_t){.resource =
                                                        &transfer->resource});
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(transfer->signals, status);
  }
  iree_hal_resource_release(&transfer->resource);
}

static void iree_hal_vulkan_file_staging_available(void* user_data) {
  iree_hal_vulkan_file_notify(user_data, IREE_HAL_VULKAN_FILE_STAGING_READY);
}

static bool iree_hal_vulkan_file_acquire_staging(
    iree_hal_vulkan_file_transfer_t* transfer,
    iree_hal_vulkan_queue_staging_ring_t* ring,
    iree_hal_vulkan_queue_staging_slot_t** out_slot) {
  while (!iree_hal_vulkan_queue_staging_ring_try_acquire(ring, out_slot)) {
    transfer->waiting_ring = ring;
    if (iree_hal_vulkan_queue_staging_ring_queue_waiter(
            ring, &transfer->waiter, iree_hal_vulkan_file_staging_available,
            transfer, &transfer->resource)) {
      return false;
    }
    transfer->waiting_ring = NULL;
  }
  return true;
}

static void iree_hal_vulkan_file_window_complete(
    iree_hal_vulkan_file_window_t* window, iree_status_t status) {
  if (!iree_status_is_ok(status)) {
    intptr_t expected = 0;
    if (!iree_atomic_compare_exchange_strong(
            &window->failure, &expected, (intptr_t)status,
            iree_memory_order_release, iree_memory_order_relaxed)) {
      iree_status_free(status);
    }
  }
  iree_hal_vulkan_file_transfer_t* transfer = window->transfer;
  iree_hal_vulkan_file_notify(transfer, INT64_C(1)
                                            << (window - transfer->windows));
}

static void iree_hal_vulkan_file_copy_complete(void* user_data,
                                               const iree_status_t status) {
  iree_hal_vulkan_file_window_complete(user_data, iree_status_clone(status));
}

static void iree_hal_vulkan_file_io_complete(void* user_data,
                                             iree_status_t status) {
  iree_hal_vulkan_file_window_t* window = user_data;
  if (!iree_status_is_ok(status)) {
    intptr_t expected = 0;
    if (!iree_atomic_compare_exchange_strong(
            &window->failure, &expected, (intptr_t)status,
            iree_memory_order_release, iree_memory_order_relaxed)) {
      iree_status_free(status);
    }
  }
  if (iree_atomic_fetch_sub(&window->pending, 1, iree_memory_order_acq_rel) ==
      1) {
    iree_hal_vulkan_file_window_complete(window, iree_ok_status());
  }
}

static iree_status_t iree_hal_vulkan_file_window_cache(
    iree_hal_vulkan_file_window_t* window, bool flush) {
  iree_hal_vulkan_file_transfer_t* transfer = window->transfer;
  iree_status_t status = iree_ok_status();
  for (uint32_t i = 0; i < window->count && iree_status_is_ok(status); ++i) {
    const iree_hal_vulkan_file_piece_t* piece = &window->pieces[i];
    iree_hal_vulkan_buffer_host_span_t span =
        transfer->buffers[piece->buffer_binding].span;
    if (span.coherent) {
      continue;
    }
    span.contents =
        iree_make_byte_span(span.contents.data + piece->buffer_offset,
                            (iree_host_size_t)piece->region.row_length);
    span.allocation_offset += piece->buffer_offset;
    status = flush ? iree_hal_vulkan_buffer_flush_host_span(&span)
                   : iree_hal_vulkan_buffer_invalidate_host_span(&span);
  }
  return status;
}

static void iree_hal_vulkan_file_window_io(
    iree_hal_vulkan_file_window_t* window) {
  window->phase = IREE_HAL_VULKAN_FILE_IO;
  iree_status_t status = window->slot
                             ? iree_ok_status()
                             : iree_hal_vulkan_file_window_cache(window, false);
  if (iree_status_is_ok(status)) {
    iree_async_operation_t* requests[IREE_HAL_VULKAN_FILE_BATCH_CAPACITY];
    for (uint32_t i = 0; i < window->count; ++i) {
      requests[i] = iree_hal_file_request_bind(
          &window->pieces[i].request,
          (iree_hal_file_transfer_callback_t){iree_hal_vulkan_file_io_complete,
                                              window});
    }
    iree_atomic_store(&window->pending, window->count,
                      iree_memory_order_relaxed);
    status = iree_async_proactor_submit(
        window->proactor,
        (iree_async_operation_list_t){requests, window->count});
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_file_window_complete(window, status);
  }
}

static void iree_hal_vulkan_file_window_copy(
    iree_hal_vulkan_file_window_t* window) {
  iree_hal_vulkan_file_transfer_t* transfer = window->transfer;
  const bool reading = transfer->access == IREE_HAL_MEMORY_ACCESS_READ;
  window->phase = IREE_HAL_VULKAN_FILE_COPY;
  iree_status_t status = iree_ok_status();
  iree_hal_vulkan_queue_barriers_t barriers = {0};
  if (window->slot && reading) {
    if (window->kind == IREE_HAL_VULKAN_FILE_HOST) {
      for (uint32_t i = 0; i < window->count; ++i) {
        const iree_hal_vulkan_file_piece_t* piece = &window->pieces[i];
        const iree_device_size_t offset =
            window->copies[i].copy.source_offset - window->slot->buffer_offset;
        memcpy(window->slot->host_span.data + offset, piece->host.data,
               piece->host.data_length);
      }
    }
    status = iree_hal_buffer_mapping_flush_range(
        &transfer->queue->upload_staging_ring->mapping,
        window->slot->buffer_offset, window->staging_length);
  } else if (window->slot) {
    barriers.after = (iree_hal_vulkan_barrier_t){
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST,
        .source_access_mask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .target_access_mask = VK_ACCESS_2_HOST_READ_BIT};
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_vulkan_queue_submit_native_transfer(
        transfer->queue, window->count, window->copies, 0, &barriers,
        (iree_hal_vulkan_queue_completion_action_t){
            .fn = iree_hal_vulkan_file_copy_complete,
            .user_data = window,
            .resource = &transfer->resource});
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_vulkan_file_window_complete(window, status);
  }
}

static iree_status_t iree_hal_vulkan_file_buffer_prepare(
    iree_hal_vulkan_file_transfer_t* transfer, uint32_t index) {
  iree_hal_vulkan_file_buffer_t* buffer = &transfer->buffers[index];
  if (buffer->prepared) {
    return iree_ok_status();
  }
  const iree_hal_buffer_binding_t binding =
      transfer->batch.buffers.bindings[index];
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_host_span(
      binding.buffer, binding.offset, binding.length, &buffer->span));
  buffer->prepared = true;
  return iree_ok_status();
}

static bool iree_hal_vulkan_file_range_is_staged(
    iree_hal_vulkan_file_transfer_t* transfer,
    const iree_hal_file_transfer_range_t* range) {
  const iree_hal_vulkan_file_kind_t kind =
      transfer->files[range->file_binding].kind;
  return kind == IREE_HAL_VULKAN_FILE_HOST ||
         (kind == IREE_HAL_VULKAN_FILE_ASYNC &&
          !transfer->buffers[range->buffer_binding].span.allocation);
}

static iree_status_t iree_hal_vulkan_file_window_fill(
    iree_hal_vulkan_file_window_t* window, uint32_t capacity) {
  iree_hal_vulkan_file_transfer_t* transfer = window->transfer;
  const bool reading = transfer->access == IREE_HAL_MEMORY_ACCESS_READ;
  iree_hal_vulkan_queue_staging_ring_t* ring =
      reading ? transfer->queue->upload_staging_ring
              : transfer->queue->download_staging_ring;
  window->count = 0;
  window->staging_length = 0;
  iree_status_t status = iree_ok_status();
  while (window->count < capacity && iree_status_is_ok(status)) {
    iree_hal_file_transfer_range_t range;
    status = iree_hal_file_transfer_iterator_peek(&transfer->iterator, &range);
    if (!iree_status_is_ok(status) || !range.length) {
      break;
    }
    iree_hal_vulkan_file_binding_t* file = &transfer->files[range.file_binding];
    const iree_hal_file_binding_t file_binding =
        transfer->batch.files.bindings[range.file_binding];
    const iree_hal_buffer_binding_t buffer_binding =
        transfer->batch.buffers.bindings[range.buffer_binding];
    if (file->kind == IREE_HAL_VULKAN_FILE_ASYNC) {
      status =
          iree_hal_vulkan_file_buffer_prepare(transfer, range.buffer_binding);
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (file->kind != window->kind ||
        iree_hal_vulkan_file_range_is_staged(transfer, &range) !=
            (window->slot != NULL) ||
        (file->kind == IREE_HAL_VULKAN_FILE_ASYNC &&
         file->handle->proactor != window->proactor)) {
      break;
    }
    iree_hal_vulkan_file_piece_t* piece = &window->pieces[window->count];
    iree_device_size_t position = window->staging_length;
    iree_device_size_t length = range.length;
    if (window->slot && file->kind == IREE_HAL_VULKAN_FILE_ASYNC) {
      const iree_async_file_requirements_t* requirements =
          iree_hal_file_requirements(file_binding.file);
      const uint64_t alignment = reading ? requirements->read.buffer_alignment
                                         : requirements->write.buffer_alignment;
      const uint64_t remainder = position % alignment;
      if (remainder) {
        position += alignment - remainder;
      }
      if (position >= window->slot->host_span.data_length) {
        break;
      }
    }
    if (file->kind == IREE_HAL_VULKAN_FILE_ASYNC) {
      const iree_hal_vulkan_buffer_host_span_t* direct =
          &transfer->buffers[range.buffer_binding].span;
      iree_async_span_t span =
          window->slot
              ? iree_async_span_from_ptr(
                    window->slot->host_span.data + position,
                    window->slot->host_span.data_length - position)
              : iree_async_span_from_ptr(
                    direct->contents.data + range.buffer_offset,
                    iree_min(length, IREE_HAL_VULKAN_QUEUE_STAGING_SLOT_SIZE));
      status =
          iree_hal_file_request_prepare(file_binding.file, transfer->access,
                                        file_binding.offset + range.file_offset,
                                        length, span, &piece->request);
      if (iree_status_is_ok(status)) {
        length = iree_hal_file_request_span(&piece->request).length;
      }
    } else if (file->kind == IREE_HAL_VULKAN_FILE_HOST) {
      if (position == window->slot->host_span.data_length) {
        break;
      }
      length = iree_min(length, window->slot->host_span.data_length - position);
      piece->host = iree_make_byte_span(file->host.data + range.file_offset,
                                        (iree_host_size_t)length);
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    piece->buffer_binding = range.buffer_binding;
    piece->buffer_offset = range.buffer_offset;
    piece->region = (iree_hal_transfer_region_t){
        .row_length = length,
        .row_count = 1,
        .slice_count = 1,
    };
    if (window->slot || file->kind == IREE_HAL_VULKAN_FILE_DEVICE) {
      iree_hal_buffer_t* other = window->slot ? ring->buffer : file->buffer;
      const iree_device_size_t other_offset =
          window->slot ? window->slot->buffer_offset + position
                       : file_binding.offset + range.file_offset;
      const iree_device_size_t payload_offset =
          buffer_binding.offset + range.buffer_offset;
      window->copies[window->count] = (iree_hal_transfer_operation_t){
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
          .copy = {.source_buffer = reading ? other : buffer_binding.buffer,
                   .source_offset = reading ? other_offset : payload_offset,
                   .target_buffer = reading ? buffer_binding.buffer : other,
                   .target_offset = reading ? payload_offset : other_offset,
                   .region = &piece->region}};
    }
    if (window->slot) {
      window->staging_length = position + length;
    }
    ++window->count;
    iree_hal_file_transfer_iterator_advance(&transfer->iterator, length);
  }
  return status;
}

static iree_status_t iree_hal_vulkan_file_metadata_apply(
    iree_hal_vulkan_file_transfer_t* transfer, const uint8_t* data) {
  if (!transfer->count_ready) {
    IREE_RETURN_IF_ERROR(iree_hal_file_transfer_iterator_initialize(
        &transfer->batch, iree_unaligned_load_le_u64(data), NULL,
        &transfer->iterator));
    transfer->count_ready = true;
  } else {
    memcpy(transfer->metadata.records, data,
           transfer->metadata.region.row_length);
    iree_hal_file_transfer_iterator_acquire(&transfer->iterator,
                                            transfer->metadata.records,
                                            transfer->metadata.record_count);
  }
  return iree_ok_status();
}

static void iree_hal_vulkan_file_metadata_complete(void* user_data,
                                                   const iree_status_t status) {
  iree_hal_vulkan_file_transfer_t* transfer = user_data;
  transfer->metadata.status = iree_status_clone(status);
  iree_hal_vulkan_file_notify(transfer, IREE_HAL_VULKAN_FILE_METADATA_READY);
}

static iree_status_t iree_hal_vulkan_file_metadata_acquire(
    iree_hal_vulkan_file_transfer_t* transfer) {
  const iree_hal_file_transfer_list_t* list = &transfer->batch.transfers;
  iree_hal_buffer_t* buffer = list->indirect.count_buffer;
  iree_device_size_t offset = list->indirect.count_offset;
  transfer->metadata.record_count = 0;
  transfer->metadata.region.row_length = 8;
  if (transfer->count_ready) {
    buffer = list->indirect.records.buffer;
    offset = list->indirect.records.offset +
             transfer->iterator.index * sizeof(iree_hal_file_transfer_record_t);
    transfer->metadata.record_count =
        iree_min(transfer->iterator.count - transfer->iterator.index,
                 IREE_HAL_VULKAN_FILE_BATCH_CAPACITY);
    transfer->metadata.region.row_length =
        transfer->metadata.record_count *
        sizeof(iree_hal_file_transfer_record_t);
  }
  iree_hal_vulkan_buffer_host_span_t span;
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_resolve_host_span(
      buffer, offset, transfer->metadata.region.row_length, &span));
  if (span.allocation) {
    IREE_RETURN_IF_ERROR(iree_hal_vulkan_buffer_invalidate_host_span(&span));
    return iree_hal_vulkan_file_metadata_apply(transfer, span.contents.data);
  }
  iree_hal_vulkan_queue_staging_ring_t* ring =
      transfer->queue->download_staging_ring;
  if (!iree_hal_vulkan_file_acquire_staging(transfer, ring,
                                            &transfer->metadata.slot)) {
    return iree_ok_status();
  }
  transfer->metadata.region.row_count = 1;
  transfer->metadata.region.slice_count = 1;
  transfer->metadata.copy = (iree_hal_transfer_operation_t){
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
      .copy = {.source_buffer = buffer,
               .source_offset = offset,
               .target_buffer = ring->buffer,
               .target_offset = transfer->metadata.slot->buffer_offset,
               .region = &transfer->metadata.region}};
  // The entry dependency acquires the complete published metadata once. Each
  // page only publishes its download to the host; it does not flush producers
  // again as the finite iterator advances.
  const iree_hal_vulkan_queue_barriers_t barriers = {
      .after = {.source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
                .target_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST,
                .source_access_mask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                .target_access_mask = VK_ACCESS_2_HOST_READ_BIT}};
  transfer->metadata.pending = true;
  iree_status_t status = iree_hal_vulkan_queue_submit_native_transfer(
      transfer->queue, 1, &transfer->metadata.copy, 0, &barriers,
      (iree_hal_vulkan_queue_completion_action_t){
          .fn = iree_hal_vulkan_file_metadata_complete,
          .user_data = transfer,
          .resource = &transfer->resource});
  if (!iree_status_is_ok(status)) {
    transfer->metadata.status = status;
    iree_hal_vulkan_file_notify(transfer, IREE_HAL_VULKAN_FILE_METADATA_READY);
  }
  return iree_ok_status();
}

static void iree_hal_vulkan_file_drain(
    iree_hal_vulkan_file_transfer_t* transfer) {
  const bool reading = transfer->access == IREE_HAL_MEMORY_ACCESS_READ;
  iree_hal_vulkan_queue_staging_ring_t* ring =
      reading ? transfer->queue->upload_staging_ring
              : transfer->queue->download_staging_ring;
  for (;;) {
    const int64_t ready =
        iree_atomic_exchange(&transfer->ready, IREE_HAL_VULKAN_FILE_DRAINING,
                             iree_memory_order_acquire);
    if (transfer->finishing) {
      return;
    }
    if (ready & IREE_HAL_VULKAN_FILE_STAGING_READY) {
      transfer->waiting_ring = NULL;
    }
    if (ready & IREE_HAL_VULKAN_FILE_METADATA_READY) {
      transfer->metadata.pending = false;
      iree_status_t status = transfer->metadata.status;
      transfer->metadata.status = iree_ok_status();
      iree_hal_vulkan_queue_staging_ring_t* download =
          transfer->queue->download_staging_ring;
      if (iree_status_is_ok(status)) {
        status = iree_hal_buffer_mapping_invalidate_range(
            &download->mapping, transfer->metadata.slot->buffer_offset,
            transfer->metadata.region.row_length);
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_vulkan_file_metadata_apply(
            transfer, transfer->metadata.slot->host_span.data);
      }
      iree_hal_vulkan_queue_staging_ring_release(download,
                                                 transfer->metadata.slot);
      transfer->metadata.slot = NULL;
      transfer->failure = iree_status_join(transfer->failure, status);
    }
    uint32_t completed = (uint32_t)ready;
    transfer->active &= ~completed;
    while (completed) {
      const uint32_t index = iree_math_count_trailing_zeros_u32(completed);
      const uint32_t bit = UINT32_C(1) << index;
      completed &= completed - 1;
      iree_hal_vulkan_file_window_t* window = &transfer->windows[index];
      iree_status_t status = (iree_status_t)iree_atomic_exchange(
          &window->failure, 0, iree_memory_order_acquire);
      transfer->failure = iree_status_join(transfer->failure, status);
      if (iree_status_is_ok(transfer->failure)) {
        if (window->phase == IREE_HAL_VULKAN_FILE_IO && reading) {
          if (window->slot) {
            transfer->active |= bit;
            iree_hal_vulkan_file_window_copy(window);
            continue;
          }
          transfer->failure = iree_hal_vulkan_file_window_cache(window, true);
        } else if (window->phase == IREE_HAL_VULKAN_FILE_COPY && window->slot &&
                   !reading) {
          transfer->failure = iree_hal_buffer_mapping_invalidate_range(
              &ring->mapping, window->slot->buffer_offset,
              window->staging_length);
          if (iree_status_is_ok(transfer->failure)) {
            if (window->kind == IREE_HAL_VULKAN_FILE_ASYNC) {
              transfer->active |= bit;
              iree_hal_vulkan_file_window_io(window);
              continue;
            }
            for (uint32_t i = 0; i < window->count; ++i) {
              const iree_device_size_t offset =
                  window->copies[i].copy.target_offset -
                  window->slot->buffer_offset;
              memcpy(window->pieces[i].host.data,
                     window->slot->host_span.data + offset,
                     window->pieces[i].host.data_length);
            }
          }
        }
      }
      if (window->slot) {
        iree_hal_vulkan_queue_staging_ring_release(ring, window->slot);
        window->slot = NULL;
      }
      transfer->free_pieces |=
          iree_math_mask_low_bits_u64(UINT64_MAX, window->count)
          << (window->pieces - transfer->pieces);
    }
    if (!transfer->count_ready && !transfer->metadata.pending &&
        !transfer->waiting_ring && iree_status_is_ok(transfer->failure)) {
      transfer->failure = iree_hal_vulkan_file_metadata_acquire(transfer);
    }
    uint32_t available =
        ((UINT32_C(1) << transfer->window_count) - 1) & ~transfer->active;
    while (available && transfer->free_pieces && transfer->count_ready &&
           !transfer->metadata.pending && !transfer->waiting_ring &&
           !transfer->exhausted && iree_status_is_ok(transfer->failure)) {
      iree_hal_file_transfer_range_t range;
      transfer->failure =
          iree_hal_file_transfer_iterator_peek(&transfer->iterator, &range);
      if (!iree_status_is_ok(transfer->failure)) {
        break;
      }
      if (!range.length) {
        if (transfer->iterator.index < transfer->iterator.count) {
          transfer->failure = iree_hal_vulkan_file_metadata_acquire(transfer);
          continue;
        }
        transfer->exhausted = true;
        break;
      }
      iree_hal_vulkan_file_binding_t* file =
          &transfer->files[range.file_binding];
      if (file->kind == IREE_HAL_VULKAN_FILE_ASYNC) {
        transfer->failure =
            iree_hal_vulkan_file_buffer_prepare(transfer, range.buffer_binding);
        if (!iree_status_is_ok(transfer->failure)) {
          break;
        }
      }
      const uint32_t index = iree_math_count_trailing_zeros_u32(available);
      const uint32_t bit = UINT32_C(1) << index;
      available &= available - 1;
      iree_hal_vulkan_file_window_t* window = &transfer->windows[index];
      window->kind = file->kind;
      window->proactor = file->kind == IREE_HAL_VULKAN_FILE_ASYNC
                             ? file->handle->proactor
                             : NULL;
      if (iree_hal_vulkan_file_range_is_staged(transfer, &range) &&
          !iree_hal_vulkan_file_acquire_staging(transfer, ring,
                                                &window->slot)) {
        break;
      }
      const uint32_t first =
          iree_math_count_trailing_zeros_u64(transfer->free_pieces);
      // An entirely free mask has no unavailable bit. Limit the scan to one
      // native batch so that this case yields its full capacity.
      const uint32_t capacity = iree_math_count_trailing_zeros_u64_width(
          ~(transfer->free_pieces >> first),
          IREE_HAL_VULKAN_FILE_BATCH_CAPACITY);
      window->pieces = transfer->pieces + first;
      window->copies = transfer->copies + first;
      transfer->failure = iree_hal_vulkan_file_window_fill(window, capacity);
      if (!iree_status_is_ok(transfer->failure)) {
        if (window->slot) {
          iree_hal_vulkan_queue_staging_ring_release(ring, window->slot);
          window->slot = NULL;
        }
        break;
      }
      transfer->free_pieces &=
          ~(iree_math_mask_low_bits_u64(UINT64_MAX, window->count) << first);
      transfer->active |= bit;
      if (window->kind == IREE_HAL_VULKAN_FILE_ASYNC &&
          (reading || !window->slot)) {
        iree_hal_vulkan_file_window_io(window);
      } else {
        iree_hal_vulkan_file_window_copy(window);
      }
    }
    if (!transfer->active && !transfer->metadata.pending &&
        (transfer->exhausted || !iree_status_is_ok(transfer->failure))) {
      iree_hal_vulkan_file_finish(transfer);
      return;
    }
    int64_t expected = IREE_HAL_VULKAN_FILE_DRAINING;
    if (iree_atomic_compare_exchange_strong(&transfer->ready, &expected, 0,
                                            iree_memory_order_release,
                                            iree_memory_order_relaxed)) {
      return;
    }
  }
}

static void iree_hal_vulkan_file_notify(
    iree_hal_vulkan_file_transfer_t* transfer, int64_t bits) {
  if (!(iree_atomic_fetch_or(&transfer->ready,
                             bits | IREE_HAL_VULKAN_FILE_DRAINING,
                             iree_memory_order_acq_rel) &
        IREE_HAL_VULKAN_FILE_DRAINING)) {
    iree_hal_vulkan_file_drain(transfer);
  }
}

static void iree_hal_vulkan_file_start(void* user_data,
                                       const iree_status_t status) {
  iree_hal_vulkan_file_transfer_t* transfer = user_data;
  transfer->failure = iree_status_clone(status);
  if (iree_status_is_ok(transfer->failure) &&
      transfer->batch.transfers.type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST) {
    transfer->failure = iree_hal_file_transfer_iterator_initialize(
        &transfer->batch, 0, NULL, &transfer->iterator);
    transfer->count_ready = true;
  }
  // One reference covers all accepted I/O. Completion actions and capacity
  // waiters independently retain their own callback borrows.
  iree_hal_resource_retain(&transfer->resource);
  iree_hal_vulkan_file_notify(transfer, 0);
}

static iree_status_t iree_hal_vulkan_file_validate_buffer(
    iree_hal_vulkan_queue_t* queue, iree_hal_buffer_t* buffer) {
  const iree_hal_buffer_placement_t placement =
      iree_hal_buffer_allocation_placement(buffer);
  if (!iree_hal_buffer_placement_is_undefined(placement) &&
      placement.device != (iree_hal_device_t*)queue->device) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "file transfer buffer belongs to another device");
  }
  iree_hal_buffer_t* allocation = iree_hal_buffer_allocated_buffer(buffer);
  if (!iree_hal_vulkan_buffer_is_native(allocation)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "file transfer buffer is not Vulkan storage");
  }
  if (iree_hal_vulkan_transient_buffer_isa(allocation) &&
      iree_hal_vulkan_transient_buffer_is_dealloca_queued(allocation)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "file transfer buffer is already deallocated");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_vulkan_queue_submit_file_transfer(
    iree_hal_vulkan_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_file_binding_table_t files,
    iree_hal_buffer_binding_table_t buffers,
    const iree_hal_file_transfer_list_t* transfers,
    iree_hal_memory_access_t file_access,
    const iree_hal_vulkan_queue_barriers_t* barriers) {
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_validate_semaphore_list(
      queue, waits, IREE_SV("wait")));
  IREE_RETURN_IF_ERROR(iree_hal_vulkan_queue_validate_semaphore_list(
      queue, signals, IREE_SV("signal")));
  iree_host_size_t batch_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_file_transfer_batch_storage_size(
      files, buffers, transfers, &batch_size));
  const bool linear =
      transfers->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST &&
      transfers->host.count == 1 &&
      transfers->host.transfers[0].row_count == 1 &&
      transfers->host.transfers[0].slice_count == 1;
  if (linear && files.count == 1 && buffers.count == 1 &&
      files.bindings[0].file) {
    iree_hal_buffer_t* storage =
        iree_hal_file_storage_buffer(files.bindings[0].file);
    if (storage && iree_hal_vulkan_buffer_is_native(
                       iree_hal_buffer_allocated_buffer(storage))) {
      // A memory-backed linear file is an ordinary native copy: no file owner,
      // staging, host completion boundary, or additional native submission.
      iree_alignas(iree_max_align_t)
          uint8_t capture[sizeof(iree_hal_file_binding_t) +
                          sizeof(iree_hal_buffer_binding_t) +
                          sizeof(iree_hal_file_transfer_t) +
                          3 * (iree_max_align_t - 1)];
      iree_hal_file_transfer_batch_t batch;
      IREE_RETURN_IF_ERROR(iree_hal_file_transfer_batch_capture(
          &queue->base, file_access, files, buffers, transfers, capture,
          &batch));
      const iree_hal_buffer_binding_t buffer = batch.buffers.bindings[0];
      const iree_hal_file_transfer_t* row = batch.transfers.host.transfers;
      IREE_RETURN_IF_ERROR(
          iree_hal_vulkan_file_validate_buffer(queue, storage));
      IREE_RETURN_IF_ERROR(
          iree_hal_vulkan_file_validate_buffer(queue, buffer.buffer));
      if (!row->row_length) {
        return iree_hal_vulkan_queue_submit_barrier(queue, waits, signals,
                                                    barriers);
      }
      const iree_device_size_t file_offset =
          batch.files.bindings[0].offset + row->file_offset;
      const iree_device_size_t buffer_offset =
          buffer.offset + row->buffer_offset;
      const bool reading = file_access == IREE_HAL_MEMORY_ACCESS_READ;
      return iree_hal_vulkan_queue_submit_copy(
          queue, waits, signals, reading ? storage : buffer.buffer,
          reading ? file_offset : buffer_offset,
          reading ? buffer.buffer : storage,
          reading ? buffer_offset : file_offset, row->row_length, barriers,
          IREE_HAL_COPY_FLAG_NONE);
    }
  }
  // HOST geometry bounds the useful in-flight records without expanding rows.
  // INDIRECT reserves only two bounded pages, independent of record capacity.
  uint32_t piece_capacity = IREE_HAL_VULKAN_FILE_PIECE_CAPACITY;
  if (transfers->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_HOST) {
    piece_capacity = 0;
    for (iree_host_size_t i = 0;
         i < transfers->host.count &&
         piece_capacity < IREE_HAL_VULKAN_FILE_PIECE_CAPACITY;
         ++i) {
      const iree_hal_file_transfer_t* row = &transfers->host.transfers[i];
      uint64_t pieces =
          row->row_length / IREE_HAL_VULKAN_QUEUE_STAGING_SLOT_SIZE +
          (row->row_length % IREE_HAL_VULKAN_QUEUE_STAGING_SLOT_SIZE != 0);
      pieces = iree_min(pieces, IREE_HAL_VULKAN_FILE_PIECE_CAPACITY) *
               iree_min(row->row_count, IREE_HAL_VULKAN_FILE_PIECE_CAPACITY) *
               iree_min(row->slice_count, IREE_HAL_VULKAN_FILE_PIECE_CAPACITY);
      piece_capacity = (uint32_t)iree_min(piece_capacity + pieces,
                                          IREE_HAL_VULKAN_FILE_PIECE_CAPACITY);
    }
    piece_capacity = iree_max(piece_capacity, 1);
  }
  const uint32_t window_count =
      iree_min(piece_capacity, IREE_HAL_VULKAN_QUEUE_STAGING_SLOT_COUNT);
  const bool indirect =
      transfers->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT;
  iree_host_size_t size = 0, batch_offset = 0, files_offset = 0,
                   buffers_offset = 0, pieces_offset = 0, copies_offset = 0,
                   records_offset = 0, semaphores_offset = 0,
                   payloads_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_vulkan_file_transfer_t) +
          window_count * sizeof(iree_hal_vulkan_file_window_t),
      &size,
      IREE_STRUCT_FIELD_ALIGNED(batch_size, uint8_t, iree_max_align_t,
                                &batch_offset),
      IREE_STRUCT_FIELD(files.count, iree_hal_vulkan_file_binding_t,
                        &files_offset),
      IREE_STRUCT_FIELD_ALIGNED(buffers.count, iree_hal_vulkan_file_buffer_t,
                                iree_alignof(iree_hal_vulkan_file_buffer_t),
                                &buffers_offset),
      IREE_STRUCT_FIELD(piece_capacity, iree_hal_vulkan_file_piece_t,
                        &pieces_offset),
      IREE_STRUCT_FIELD(piece_capacity, iree_hal_transfer_operation_t,
                        &copies_offset),
      IREE_STRUCT_FIELD(indirect ? IREE_HAL_VULKAN_FILE_BATCH_CAPACITY : 0,
                        iree_hal_file_transfer_record_t, &records_offset),
      IREE_STRUCT_FIELD(signals.count, iree_hal_semaphore_t*,
                        &semaphores_offset),
      IREE_STRUCT_FIELD_ALIGNED(signals.count, uint64_t, iree_alignof(uint64_t),
                                &payloads_offset)));
  iree_hal_vulkan_file_transfer_t* transfer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(queue->host_allocator, size, (void**)&transfer));
  iree_hal_resource_initialize(&iree_hal_vulkan_file_vtable,
                               &transfer->resource);
  transfer->queue = queue;
  transfer->access = file_access;
  transfer->window_count = window_count;
  transfer->pieces = (void*)((uint8_t*)transfer + pieces_offset);
  transfer->copies = (void*)((uint8_t*)transfer + copies_offset);
  transfer->free_pieces =
      iree_math_mask_low_bits_u64(UINT64_MAX, piece_capacity);
  transfer->files = (void*)((uint8_t*)transfer + files_offset);
  transfer->buffers = (void*)((uint8_t*)transfer + buffers_offset);
  transfer->metadata.records = (void*)((uint8_t*)transfer + records_offset);
  if (barriers) {
    transfer->publication.after = barriers->after;
  }
  iree_hal_file_transfer_batch_t batch = {0};
  iree_status_t status = iree_hal_file_transfer_batch_capture(
      &queue->base, file_access, files, buffers, transfers,
      (uint8_t*)transfer + batch_offset, &batch);
  for (iree_host_size_t i = 0; i < files.count && iree_status_is_ok(status);
       ++i) {
    const iree_hal_file_binding_t binding = batch.files.bindings[i];
    iree_hal_vulkan_file_binding_t* file = &transfer->files[i];
    iree_hal_buffer_t* storage = iree_hal_file_storage_buffer(binding.file);
    if (storage && iree_hal_vulkan_buffer_is_native(
                       iree_hal_buffer_allocated_buffer(storage))) {
      file->kind = IREE_HAL_VULKAN_FILE_DEVICE;
      file->buffer = storage;
      status = iree_hal_vulkan_file_validate_buffer(queue, storage);
    } else if (iree_hal_file_async_handle(binding.file)) {
      file->kind = IREE_HAL_VULKAN_FILE_ASYNC;
      file->handle = iree_hal_file_async_handle(binding.file);
    } else if (iree_hal_memory_file_isa(binding.file)) {
      file->kind = IREE_HAL_VULKAN_FILE_HOST;
      status = iree_hal_memory_file_contents(binding.file, &file->host);
      if (iree_status_is_ok(status)) {
        file->host = iree_make_byte_span(file->host.data + binding.offset,
                                         (iree_host_size_t)binding.length);
      }
    } else {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "Vulkan file transfer requires native file, "
                                "host memory, or device storage");
    }
  }
  for (iree_host_size_t i = 0; i < buffers.count && iree_status_is_ok(status);
       ++i) {
    status = iree_hal_vulkan_file_validate_buffer(
        queue, batch.buffers.bindings[i].buffer);
  }
  if (iree_status_is_ok(status) && indirect) {
    status = iree_hal_vulkan_file_validate_buffer(
        queue, transfers->indirect.count_buffer);
    if (iree_status_is_ok(status)) {
      status = iree_hal_vulkan_file_validate_buffer(
          queue, transfers->indirect.records.buffer);
    }
  }
  if (iree_status_is_ok(status)) {
    transfer->batch = batch;
    for (iree_host_size_t i = 0; i < files.count; ++i) {
      iree_hal_file_retain(batch.files.bindings[i].file);
    }
    for (iree_host_size_t i = 0; i < buffers.count; ++i) {
      iree_hal_buffer_retain(batch.buffers.bindings[i].buffer);
    }
    if (indirect) {
      iree_hal_buffer_retain(batch.transfers.indirect.records.buffer);
      iree_hal_buffer_retain(batch.transfers.indirect.count_buffer);
    }
    transfer->signals = (iree_hal_semaphore_list_t){
        signals.count, (void*)((uint8_t*)transfer + semaphores_offset),
        (void*)((uint8_t*)transfer + payloads_offset)};
    for (iree_host_size_t i = 0; i < signals.count; ++i) {
      transfer->signals.semaphores[i] = signals.semaphores[i];
      transfer->signals.payload_values[i] = signals.payload_values[i];
    }
    iree_hal_semaphore_list_retain(transfer->signals);
    for (uint32_t i = 0; i < window_count; ++i) {
      transfer->windows[i].transfer = transfer;
    }
    iree_hal_vulkan_queue_barriers_t entry = {
        .before = barriers ? barriers->before : (iree_hal_vulkan_barrier_t){0}};
    if (file_access == IREE_HAL_MEMORY_ACCESS_WRITE || indirect) {
      entry.before.source_stage_mask |=
          IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS;
      entry.before.source_access_mask |= VK_ACCESS_2_MEMORY_WRITE_BIT;
      entry.before.target_stage_mask |=
          IREE_HAL_EXECUTION_STAGE_HOST | IREE_HAL_EXECUTION_STAGE_TRANSFER;
      entry.before.target_access_mask |=
          VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
    }
    status = iree_hal_vulkan_queue_submit_barrier_with_action(
        queue, waits, iree_hal_semaphore_list_empty(), &entry,
        (iree_hal_vulkan_queue_completion_action_t){
            .fn = iree_hal_vulkan_file_start,
            .user_data = transfer,
            .resource = &transfer->resource});
  }
  iree_hal_resource_release(&transfer->resource);
  return status;
}
