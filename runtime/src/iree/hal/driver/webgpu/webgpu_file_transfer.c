// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/webgpu/webgpu_file_transfer.h"

#include "iree/async/operations/semaphore.h"
#include "iree/async/platform/js/proactor.h"
#include "iree/hal/driver/webgpu/webgpu_buffer.h"
#include "iree/hal/driver/webgpu/webgpu_fd_file.h"
#include "iree/hal/driver/webgpu/webgpu_imports.h"
#include "iree/hal/driver/webgpu/webgpu_queue.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/hal/utils/memory_file.h"

// Two windows overlap native work without storage proportional to file length.
#define IREE_HAL_WEBGPU_FILE_WINDOW_CAPACITY (1024 * 1024)
#define IREE_HAL_WEBGPU_FILE_PIECE_CAPACITY 32

// Wire layout consumed by FileTransferWindow in webgpu_imports.mjs.
typedef struct iree_hal_webgpu_file_piece_t {
  // Borrowed native payload buffer handle.
  uint32_t buffer;
  // Zero for wasm host memory, one for a JS file object.
  uint32_t file_kind;
  // Borrowed wasm base address or retained JS file-object index.
  uint32_t file;
  // Exact useful bytes, excluding native copy alignment padding.
  uint32_t length;
  // Absolute native payload byte offset, including HAL subspans.
  uint64_t buffer_offset;
  // Absolute byte offset within the file or host allocation.
  uint64_t file_offset;
  // Offset into bounded staging storage; always four-byte aligned.
  uint32_t staging_offset;
  // Zero padding keeping successive wire records eight-byte aligned.
  uint32_t reserved;
} iree_hal_webgpu_file_piece_t;
static_assert(sizeof(iree_hal_webgpu_file_piece_t) == 40,
              "file bridge piece wire size");

typedef struct iree_hal_webgpu_file_binding_t {
  // Zero for wasm host memory, one for a JS file object.
  uint32_t kind;
  // Borrowed wasm allocation address or retained JS file-object index.
  uint32_t handle;
} iree_hal_webgpu_file_binding_t;

typedef enum iree_hal_webgpu_file_phase_e {
  IREE_HAL_WEBGPU_FILE_PAYLOAD,
  IREE_HAL_WEBGPU_FILE_COUNT,
  IREE_HAL_WEBGPU_FILE_RECORDS,
} iree_hal_webgpu_file_phase_t;

typedef struct iree_hal_webgpu_file_transfer_t iree_hal_webgpu_file_transfer_t;

typedef struct iree_hal_webgpu_file_window_t {
  // Placed external completion token, reused only after its callback.
  iree_async_operation_t operation;
  // Enclosing group, live through this window's completion.
  iree_hal_webgpu_file_transfer_t* transfer;
  // JS owner of lazily allocated native staging resources.
  iree_hal_webgpu_handle_t handle;
  // Operation whose returned bytes the callback consumes.
  iree_hal_webgpu_file_phase_t phase;
  // Active piece count, or acquired record count for metadata.
  uint32_t count;
  // Native piece storage borrowed by JS until its completion.
  iree_hal_webgpu_file_piece_t pieces[IREE_HAL_WEBGPU_FILE_PIECE_CAPACITY];
} iree_hal_webgpu_file_window_t;

struct iree_hal_webgpu_file_transfer_t {
  // Placed producer wait; never occupies the native GPU FIFO.
  iree_async_semaphore_wait_operation_t wait_operation;
  // Queue supplying native handles and the event-loop proactor. Borrowed.
  iree_hal_webgpu_queue_t* queue;
  // Immutable retained bindings and captured HOST descriptions.
  iree_hal_file_transfer_batch_t batch;
  // Native file representations qualified once at capture.
  iree_hal_webgpu_file_binding_t* files;
  // Retained producer timepoints until the entry callback.
  iree_hal_semaphore_list_t waits;
  // Retained terminal timepoints until captures have retired.
  iree_hal_semaphore_list_t signals;
  // READ for file-to-buffer; WRITE for buffer-to-file.
  iree_hal_memory_access_t access;
  // Single event-loop owner traversing rows and slices without expansion.
  iree_hal_file_transfer_iterator_t iterator;
  // First terminal failure, owned until final publication.
  iree_status_t failure;
  // In-flight windows; callbacks clear their own bit before progressing.
  uint32_t active;
  // True after the live count has initialized the iterator.
  bool count_ready;
  // True after exhausting the live transfer prefix.
  bool exhausted;
  // Count and live metadata pages share this reusable, aligned storage.
  iree_hal_file_transfer_record_t records[IREE_HAL_WEBGPU_FILE_PIECE_CAPACITY];
  // Bounded independent native windows, not queues or workers.
  iree_hal_webgpu_file_window_t windows[2];
};

static void iree_hal_webgpu_file_release_storage(
    iree_hal_webgpu_file_transfer_t* transfer) {
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
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(transfer->windows); ++i) {
    if (transfer->windows[i].handle) {
      iree_hal_webgpu_import_file_transfer_window_destroy(
          transfer->windows[i].handle);
    }
  }
  iree_hal_semaphore_list_release(transfer->waits);
}

static void iree_hal_webgpu_file_finish(
    iree_hal_webgpu_file_transfer_t* transfer) {
  const iree_allocator_t allocator = transfer->queue->host_allocator;
  iree_hal_webgpu_file_release_storage(transfer);
  iree_status_t status = transfer->failure;
  if (iree_status_is_ok(status)) {
    // A host-mediated group spans multiple native submissions. Publishing an
    // earlier native queue epoch would let independent work falsely retire it.
    status = iree_hal_semaphore_list_signal(transfer->signals, NULL);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(transfer->signals, status);
  }
  iree_hal_semaphore_list_release(transfer->signals);
  iree_allocator_free(allocator, transfer);
}

// Completion and progression form a cycle through the event-loop proactor.
static void iree_hal_webgpu_file_progress(
    iree_hal_webgpu_file_transfer_t* transfer);

static void iree_hal_webgpu_file_window_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_hal_webgpu_file_window_t* window = user_data;
  iree_hal_webgpu_file_transfer_t* transfer = window->transfer;
  transfer->active &= ~(1u << (window - transfer->windows));
  if (iree_status_is_ok(status) && iree_status_is_ok(transfer->failure)) {
    if (window->phase == IREE_HAL_WEBGPU_FILE_COUNT) {
      const uint64_t count = iree_unaligned_load_le_u64(transfer->records);
      status = iree_hal_file_transfer_iterator_initialize(
          &transfer->batch, count, NULL, &transfer->iterator);
      transfer->count_ready = iree_status_is_ok(status);
    } else if (window->phase == IREE_HAL_WEBGPU_FILE_RECORDS) {
      iree_hal_file_transfer_iterator_acquire(&transfer->iterator,
                                              transfer->records, window->count);
    }
  }
  transfer->failure = iree_status_join(transfer->failure, status);
  iree_hal_webgpu_file_progress(transfer);
}

static iree_status_t iree_hal_webgpu_file_begin_window(
    iree_hal_webgpu_file_window_t* window, uint32_t* out_token) {
  iree_hal_webgpu_queue_t* queue = window->transfer->queue;
  if (!window->handle) {
    iree_hal_webgpu_isa_builtins_descriptor_t builtins;
    iree_hal_webgpu_builtins_get_descriptor(queue->builtins, &builtins);
    window->handle = iree_hal_webgpu_import_file_transfer_window_create(
        queue->device_handle, queue->queue_handle,
        (uint32_t)(uintptr_t)&builtins);
    if (!window->handle) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "WebGPU file window allocation failed");
    }
  }
  iree_async_operation_initialize(&window->operation,
                                  IREE_ASYNC_OPERATION_TYPE_NOP,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  iree_hal_webgpu_file_window_complete, window);
  IREE_RETURN_IF_ERROR(iree_async_proactor_js_submit_external(
      queue->proactor, &window->operation, out_token));
  window->transfer->active |= 1u << (window - window->transfer->windows);
  return iree_ok_status();
}

static iree_status_t iree_hal_webgpu_file_acquire(
    iree_hal_webgpu_file_transfer_t* transfer,
    iree_hal_webgpu_file_phase_t phase) {
  iree_hal_webgpu_file_window_t* window = &transfer->windows[0];
  window->phase = phase;
  iree_hal_buffer_t* buffer = transfer->batch.transfers.indirect.count_buffer;
  uint64_t offset = transfer->batch.transfers.indirect.count_offset;
  uint32_t length = 8;
  if (phase == IREE_HAL_WEBGPU_FILE_RECORDS) {
    buffer = transfer->batch.transfers.indirect.records.buffer;
    offset = transfer->batch.transfers.indirect.records.offset +
             transfer->iterator.index * sizeof(*transfer->records);
    window->count =
        (uint32_t)iree_min(transfer->iterator.count - transfer->iterator.index,
                           IREE_ARRAYSIZE(transfer->records));
    length = window->count * sizeof(*transfer->records);
  }
  uint32_t token = 0;
  IREE_RETURN_IF_ERROR(iree_hal_webgpu_file_begin_window(window, &token));
  iree_hal_webgpu_import_file_transfer_acquire(
      window->handle, iree_hal_webgpu_buffer_handle(buffer),
      iree_hal_buffer_byte_offset(buffer) + offset, length,
      (uint32_t)(uintptr_t)transfer->records, token);
  return iree_ok_status();
}

static void iree_hal_webgpu_file_progress(
    iree_hal_webgpu_file_transfer_t* transfer) {
  if (iree_status_is_ok(transfer->failure) && !transfer->count_ready) {
    if (!(transfer->active & 1)) {
      transfer->failure =
          iree_hal_webgpu_file_acquire(transfer, IREE_HAL_WEBGPU_FILE_COUNT);
    }
  } else {
    for (uint32_t i = 0;
         i < IREE_ARRAYSIZE(transfer->windows) &&
         iree_status_is_ok(transfer->failure) && !transfer->exhausted;
         ++i) {
      if (transfer->active & (1u << i)) {
        continue;
      }
      iree_hal_webgpu_file_window_t* window = &transfer->windows[i];
      window->phase = IREE_HAL_WEBGPU_FILE_PAYLOAD;
      window->count = 0;
      uint32_t used = 0;
      iree_hal_file_transfer_range_t range = {0};
      while (window->count < IREE_ARRAYSIZE(window->pieces) &&
             used + 4 <= IREE_HAL_WEBGPU_FILE_WINDOW_CAPACITY &&
             iree_status_is_ok(transfer->failure)) {
        transfer->failure =
            iree_hal_file_transfer_iterator_peek(&transfer->iterator, &range);
        if (!iree_status_is_ok(transfer->failure) || !range.length) {
          break;
        }
        const iree_hal_buffer_binding_t buffer =
            transfer->batch.buffers.bindings[range.buffer_binding];
        const iree_hal_webgpu_file_binding_t file =
            transfer->files[range.file_binding];
        const uint64_t buffer_offset =
            iree_hal_buffer_byte_offset(buffer.buffer) + buffer.offset +
            range.buffer_offset;
        const uint32_t bias = transfer->access == IREE_HAL_MEMORY_ACCESS_WRITE
                                  ? (uint32_t)(buffer_offset % 4)
                                  : 0;
        uint32_t length = (uint32_t)iree_min(
            range.length, IREE_HAL_WEBGPU_FILE_WINDOW_CAPACITY - used - bias);
        // Full windows end on words; a short logical tail stays byte-exact.
        if (length < range.length) {
          length &= ~3u;
          if (!length) {
            break;
          }
        }
        window->pieces[window->count++] = (iree_hal_webgpu_file_piece_t){
            .buffer = iree_hal_webgpu_buffer_handle(buffer.buffer),
            .file_kind = file.kind,
            .file = file.handle,
            .length = length,
            .buffer_offset = buffer_offset,
            .file_offset =
                transfer->batch.files.bindings[range.file_binding].offset +
                range.file_offset,
            .staging_offset = used,
        };
        used += (uint32_t)iree_host_align(length + bias, 4);
        iree_hal_file_transfer_iterator_advance(&transfer->iterator, length);
      }
      if (window->count) {
        uint32_t token = 0;
        iree_status_t status =
            iree_hal_webgpu_file_begin_window(window, &token);
        if (iree_status_is_ok(status)) {
          iree_hal_webgpu_import_file_transfer_submit(
              window->handle, (uint32_t)(uintptr_t)window->pieces,
              window->count, transfer->access, token);
        }
        transfer->failure = iree_status_join(transfer->failure, status);
      }
      if (iree_status_is_ok(transfer->failure) && !range.length) {
        if (transfer->iterator.index == transfer->iterator.count) {
          transfer->exhausted = true;
        } else if (!(transfer->active & 1)) {
          transfer->failure = iree_hal_webgpu_file_acquire(
              transfer, IREE_HAL_WEBGPU_FILE_RECORDS);
        }
        break;
      }
    }
  }
  if ((!iree_status_is_ok(transfer->failure) || transfer->exhausted) &&
      !transfer->active) {
    iree_hal_webgpu_file_finish(transfer);
  }
}

static void iree_hal_webgpu_file_wait_complete(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_hal_webgpu_file_transfer_t* transfer = user_data;
  iree_hal_semaphore_list_release(transfer->waits);
  transfer->waits = iree_hal_semaphore_list_empty();
  transfer->failure = status;
  iree_hal_webgpu_file_progress(transfer);
}

iree_status_t iree_hal_webgpu_queue_submit_file_transfer(
    iree_hal_webgpu_queue_t* queue, iree_hal_semaphore_list_t waits,
    iree_hal_semaphore_list_t signals, iree_hal_file_binding_table_t files,
    iree_hal_buffer_binding_table_t buffers,
    const iree_hal_file_transfer_list_t* transfers,
    iree_hal_memory_access_t file_access) {
  iree_host_size_t batch_size = 0;
  IREE_RETURN_IF_ERROR(iree_hal_file_transfer_batch_storage_size(
      files, buffers, transfers, &batch_size));
  iree_host_size_t size = 0, batch_offset = 0, files_offset = 0,
                   wait_semaphores_offset = 0, wait_values_offset = 0,
                   signal_semaphores_offset = 0, signal_values_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_hal_webgpu_file_transfer_t), &size,
      IREE_STRUCT_FIELD_ALIGNED(batch_size, uint8_t, iree_max_align_t,
                                &batch_offset),
      IREE_STRUCT_FIELD(files.count, iree_hal_webgpu_file_binding_t,
                        &files_offset),
      IREE_STRUCT_FIELD(waits.count, iree_hal_semaphore_t*,
                        &wait_semaphores_offset),
      IREE_STRUCT_FIELD(waits.count, uint64_t, &wait_values_offset),
      IREE_STRUCT_FIELD(signals.count, iree_hal_semaphore_t*,
                        &signal_semaphores_offset),
      IREE_STRUCT_FIELD(signals.count, uint64_t, &signal_values_offset)));
  iree_hal_webgpu_file_transfer_t* transfer = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(queue->host_allocator, size, (void**)&transfer));
  transfer->queue = queue;
  transfer->access = file_access;
  transfer->files = (void*)((uint8_t*)transfer + files_offset);
  iree_hal_file_transfer_batch_t batch = {0};
  iree_status_t status = iree_hal_file_transfer_batch_capture(
      &queue->base, file_access, files, buffers, transfers,
      (uint8_t*)transfer + batch_offset, &batch);
  for (iree_host_size_t i = 0; i < files.count && iree_status_is_ok(status);
       ++i) {
    iree_hal_file_t* file = batch.files.bindings[i].file;
    if (iree_hal_memory_file_isa(file)) {
      iree_byte_span_t contents = iree_byte_span_empty();
      status = iree_hal_memory_file_contents(file, &contents);
      transfer->files[i].handle = (uint32_t)(uintptr_t)contents.data;
    } else if (iree_hal_webgpu_fd_file_isa(file)) {
      transfer->files[i].kind = 1;
      transfer->files[i].handle = (uint32_t)iree_hal_webgpu_fd_file_fd(file);
    } else {
      status =
          iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                           "WebGPU requires host memory or JS file objects");
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(queue->host_allocator, transfer);
    return status;
  }
  transfer->batch = batch;
  for (iree_host_size_t i = 0; i < files.count; ++i) {
    iree_hal_file_retain(batch.files.bindings[i].file);
  }
  for (iree_host_size_t i = 0; i < buffers.count; ++i) {
    iree_hal_buffer_retain(batch.buffers.bindings[i].buffer);
  }
  if (transfers->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT) {
    iree_hal_buffer_retain(transfers->indirect.records.buffer);
    iree_hal_buffer_retain(transfers->indirect.count_buffer);
  } else {
    transfer->failure = iree_hal_file_transfer_iterator_initialize(
        &transfer->batch, 0, NULL, &transfer->iterator);
    transfer->count_ready = true;
  }
  transfer->waits = (iree_hal_semaphore_list_t){
      waits.count, (void*)((uint8_t*)transfer + wait_semaphores_offset),
      (void*)((uint8_t*)transfer + wait_values_offset)};
  transfer->signals = (iree_hal_semaphore_list_t){
      signals.count, (void*)((uint8_t*)transfer + signal_semaphores_offset),
      (void*)((uint8_t*)transfer + signal_values_offset)};
  for (iree_host_size_t i = 0; i < waits.count; ++i) {
    transfer->waits.semaphores[i] = waits.semaphores[i];
    transfer->waits.payload_values[i] = waits.payload_values[i];
    iree_hal_semaphore_retain(waits.semaphores[i]);
  }
  for (iree_host_size_t i = 0; i < signals.count; ++i) {
    transfer->signals.semaphores[i] = signals.semaphores[i];
    transfer->signals.payload_values[i] = signals.payload_values[i];
    iree_hal_semaphore_retain(signals.semaphores[i]);
  }
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(transfer->windows); ++i) {
    transfer->windows[i].transfer = transfer;
  }
  if (iree_hal_semaphore_list_poll(waits)) {
    iree_hal_semaphore_list_release(transfer->waits);
    transfer->waits = iree_hal_semaphore_list_empty();
    iree_hal_webgpu_file_progress(transfer);
    return iree_ok_status();
  }
  transfer->wait_operation.semaphores =
      (iree_async_semaphore_t**)transfer->waits.semaphores;
  transfer->wait_operation.values = transfer->waits.payload_values;
  transfer->wait_operation.count = waits.count;
  transfer->wait_operation.mode = IREE_ASYNC_WAIT_MODE_ALL;
  iree_async_operation_initialize(&transfer->wait_operation.base,
                                  IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT,
                                  IREE_ASYNC_OPERATION_FLAG_NONE,
                                  iree_hal_webgpu_file_wait_complete, transfer);
  status = iree_async_proactor_submit_one(queue->proactor,
                                          &transfer->wait_operation.base);
  if (!iree_status_is_ok(status)) {
    // Synchronous rejection owns no submitted work and leaves signal
    // semaphores untouched. Only accepted groups publish terminal failure.
    iree_hal_webgpu_file_release_storage(transfer);
    iree_hal_semaphore_list_release(transfer->signals);
    iree_allocator_free(queue->host_allocator, transfer);
  }
  return status;
}
