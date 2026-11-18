// Copyright 2026 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/proactor.h"
#include "iree/base/internal/math.h"
#include "iree/hal/driver/task/queue/queue.h"
#include "iree/hal/utils/file_request.h"

// A window bounds native request storage independently of the logical file
// size. A slot is refilled as soon as it completes, without waiting for peers.
#define IREE_HAL_TASK_FILE_REQUEST_CAPACITY 32
#define IREE_HAL_TASK_FILE_DRAINING (INT64_C(1) << 32)

typedef struct iree_hal_task_file_transfer_t iree_hal_task_file_transfer_t;

typedef struct iree_hal_task_file_host_t {
  // One finite worker visit for a group of synchronous file ranges.
  iree_task_process_t process;
  // Borrowed operation storage, live until the worker placement retires.
  iree_hal_task_file_transfer_t* transfer;
  // Slots owned by this worker; cleared by the drainer after release_fn.
  uint32_t slots;
} iree_hal_task_file_host_t;

typedef struct iree_hal_task_file_slot_t {
  // Borrowed enclosing operation state, live until this slot retires.
  iree_hal_task_file_transfer_t* transfer;
  // Completion result published by the ready bit, consumed by the drainer.
  iree_status_t status;
  union {
    // Native request with independent partial-I/O continuation.
    iree_hal_file_request_t request;
    // Blocking host access runs independently of the queue's control process.
    struct {
      // Borrowed file retained by the enclosing operation.
      iree_hal_file_t* file;
      // Absolute file byte offset of this admitted extent.
      uint64_t offset;
      // Borrowed payload span retained by the enclosing operation.
      iree_byte_span_t span;
    } host;
  };
} iree_hal_task_file_slot_t;

struct iree_hal_task_file_transfer_t {
  // Existing queue operation, which owns this state and every resource.
  iree_hal_task_queue_op_t* operation;
  // Per-binding host bases, resolved lazily on first actual access.
  uint8_t** buffers;
  // Single-owner progress over captured or acquired descriptions.
  iree_hal_file_transfer_iterator_t iterator;
  // Low bits publish completed slots; bit 32 assigns the sole drainer.
  // Publishing both in one RMW joins callbacks with native submit return.
  iree_atomic_int64_t ready;
  // Slots whose accepted native I/O or worker access has not retired.
  uint32_t active;
  // Mask of the placed request slots, possibly smaller than the maximum window.
  uint32_t capacity;
  // First failure; further admission stops while accepted slots drain.
  iree_status_t status;
  // Lazily placed worker state, absent for native asynchronous files.
  iree_hal_task_file_host_t* host;
  // Bounded reusable request state, never one allocation per logical row.
  iree_hal_task_file_slot_t slots[];
};

static void iree_hal_task_file_drain(iree_hal_task_file_transfer_t* transfer,
                                     bool on_worker);

static void iree_hal_task_file_retire(iree_hal_task_file_transfer_t* transfer,
                                      uint32_t slots, bool on_worker) {
  const int64_t previous = iree_atomic_fetch_or(
      &transfer->ready, IREE_HAL_TASK_FILE_DRAINING | slots,
      iree_memory_order_acq_rel);
  if (!(previous & IREE_HAL_TASK_FILE_DRAINING)) {
    iree_hal_task_file_drain(transfer, on_worker);
  }
  // Publishing the bit ends this callback's borrow. Another drainer may have
  // reused the slot or retired the whole operation before this call returns.
}

static void iree_hal_task_file_complete(void* user_data, iree_status_t status) {
  iree_hal_task_file_slot_t* slot = user_data;
  slot->status = status;
  iree_hal_task_file_retire(
      slot->transfer, UINT32_C(1) << (slot - slot->transfer->slots), false);
}

static iree_status_t iree_hal_task_file_host_execute(
    iree_task_process_t* process,
    const iree_task_worker_context_t* worker_context,
    iree_task_process_drain_result_t* out_result) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_task_file_host_t* host = process->user_data;
  uint32_t pending = host->slots;
  while (pending) {
    const uint32_t index = iree_math_count_trailing_zeros_u32(pending);
    pending &= pending - 1;
    iree_hal_task_file_slot_t* slot = &host->transfer->slots[index];
    if (host->transfer->operation->type == IREE_HAL_TASK_QUEUE_OP_READ) {
      slot->status = iree_hal_file_read(slot->host.file, slot->host.offset,
                                        slot->host.span);
    } else {
      slot->status =
          iree_hal_file_write(slot->host.file, slot->host.offset,
                              iree_const_cast_byte_span(slot->host.span));
    }
  }
  out_result->did_work = true;
  out_result->completed = true;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_task_file_host_release(iree_task_process_t* process) {
  iree_hal_task_file_host_t* host = process->user_data;
  iree_hal_task_file_retire(host->transfer, host->slots, true);
}

static iree_status_t iree_hal_task_file_prepare(
    iree_hal_task_file_transfer_t* transfer,
    iree_hal_file_transfer_range_t range, iree_async_file_t* native_file,
    iree_hal_task_file_slot_t* slot) {
  const iree_hal_file_binding_t file =
      transfer->operation->file.files.bindings[range.file_binding];
  if (!transfer->buffers[range.buffer_binding]) {
    const iree_hal_buffer_binding_t binding =
        transfer->operation->file.buffers.bindings[range.buffer_binding];
    iree_byte_span_t span;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_native_host_span(
        binding.buffer, binding.offset, binding.length, &span));
    transfer->buffers[range.buffer_binding] = span.data;
  }
  const iree_byte_span_t span = iree_make_byte_span(
      transfer->buffers[range.buffer_binding] + range.buffer_offset,
      (iree_host_size_t)iree_min(range.length, UINT32_C(1) << 30));
  if (native_file) {
    IREE_RETURN_IF_ERROR(iree_hal_file_request_prepare(
        file.file,
        transfer->operation->type == IREE_HAL_TASK_QUEUE_OP_READ
            ? IREE_HAL_MEMORY_ACCESS_READ
            : IREE_HAL_MEMORY_ACCESS_WRITE,
        file.offset + range.file_offset, range.length,
        iree_async_span_from_ptr(span.data, span.data_length), &slot->request));
    iree_hal_file_request_bind(
        &slot->request,
        (iree_hal_file_transfer_callback_t){iree_hal_task_file_complete, slot});
    iree_hal_file_transfer_iterator_advance(
        &transfer->iterator, iree_hal_file_request_span(&slot->request).length);
  } else {
    slot->host.file = file.file;
    slot->host.offset = file.offset + range.file_offset;
    slot->host.span = span;
    iree_hal_file_transfer_iterator_advance(&transfer->iterator,
                                            span.data_length);
  }
  return iree_ok_status();
}

static void iree_hal_task_file_submit(iree_hal_task_file_transfer_t* transfer,
                                      iree_async_proactor_t* proactor,
                                      iree_async_operation_t** requests,
                                      iree_host_size_t count, uint32_t mask) {
  if (!count) {
    return;
  }
  if (iree_status_is_ok(transfer->status)) {
    transfer->status = iree_async_proactor_submit(
        proactor, (iree_async_operation_list_t){requests, count});
  }
  if (!iree_status_is_ok(transfer->status)) {
    // Rejected/unsubmitted requests have no callbacks. Earlier accepted
    // requests still own their slots and must retire before terminal failure.
    transfer->active &= ~mask;
  }
}

static void iree_hal_task_file_submit_host(
    iree_hal_task_file_transfer_t* transfer, uint32_t slots) {
  if (!slots) {
    return;
  }
  if (iree_status_is_ok(transfer->status) && !transfer->host) {
    transfer->status = iree_arena_allocate_aligned(
        &transfer->operation->arena, sizeof(*transfer->host),
        iree_alignof(iree_hal_task_file_host_t), (void**)&transfer->host);
  }
  if (iree_status_is_ok(transfer->status)) {
    iree_hal_task_file_host_t* host = transfer->host;
    host->transfer = transfer;
    host->slots = slots;
    iree_task_process_initialize(iree_hal_task_file_host_execute, 0, 1,
                                 &host->process);
    host->process.user_data = host;
    host->process.release_fn = iree_hal_task_file_host_release;
    iree_task_executor_schedule_process(transfer->operation->queue->executor,
                                        &host->process);
  } else {
    transfer->active &= ~slots;
  }
}

static void iree_hal_task_file_finish(iree_hal_task_file_transfer_t* transfer,
                                      bool on_worker) {
  iree_hal_task_queue_op_t* operation = transfer->operation;
  if (!iree_status_is_ok(transfer->status)) {
    iree_hal_task_queue_op_fail(operation, transfer->status);
  } else if (!on_worker &&
             iree_hal_task_queue_memory_barriers_file_needs_worker(operation)) {
    operation->flags |= IREE_HAL_TASK_QUEUE_OP_FLAG_COMPLETING_IO;
    iree_hal_task_queue_schedule_ready(operation);
  } else {
    iree_hal_task_queue_op_complete(operation);
  }
}

static void iree_hal_task_file_drain(iree_hal_task_file_transfer_t* transfer,
                                     bool on_worker) {
  for (;;) {
    uint32_t completed = (uint32_t)iree_atomic_exchange(
        &transfer->ready, IREE_HAL_TASK_FILE_DRAINING,
        iree_memory_order_acquire);
    transfer->active &= ~completed;
    if (transfer->host && (completed & transfer->host->slots)) {
      transfer->host->slots = 0;
    }
    while (completed) {
      const uint32_t index = iree_math_count_trailing_zeros_u32(completed);
      completed &= completed - 1;
      if (iree_status_is_ok(transfer->status)) {
        transfer->status = transfer->slots[index].status;
      } else {
        iree_status_ignore(transfer->slots[index].status);
      }
    }

    iree_async_operation_t* requests[IREE_HAL_TASK_FILE_REQUEST_CAPACITY];
    iree_host_size_t count = 0;
    uint32_t pending = 0;
    uint32_t host_pending = 0;
    uint64_t host_capacity = UINT32_C(1) << 30;
    iree_async_proactor_t* proactor = NULL;
    uint32_t available = transfer->capacity & ~transfer->active;
    while (available && iree_status_is_ok(transfer->status)) {
      iree_hal_file_transfer_range_t range;
      transfer->status =
          iree_hal_file_transfer_iterator_peek(&transfer->iterator, &range);
      if (!iree_status_is_ok(transfer->status) || !range.length) {
        break;
      }
      iree_hal_file_t* file =
          transfer->operation->file.files.bindings[range.file_binding].file;
      iree_async_file_t* native_file = iree_hal_file_async_handle(file);
      if (!native_file) {
        if ((transfer->host && transfer->host->slots) || !host_capacity) {
          break;
        }
        range.length = iree_min(range.length, host_capacity);
      }
      const uint32_t index = iree_math_count_trailing_zeros_u32(available);
      const uint32_t bit = UINT32_C(1) << index;
      available &= ~bit;
      iree_hal_task_file_slot_t* slot = &transfer->slots[index];
      transfer->status =
          iree_hal_task_file_prepare(transfer, range, native_file, slot);
      if (!iree_status_is_ok(transfer->status)) {
        break;
      }
      transfer->active |= bit;
      if (native_file) {
        if (count && proactor != native_file->proactor) {
          iree_hal_task_file_submit(transfer, proactor, requests, count,
                                    pending);
          count = pending = 0;
        }
        proactor = native_file->proactor;
        requests[count++] = &slot->request.read.base;
        pending |= bit;
      } else {
        host_pending |= bit;
        host_capacity -= slot->host.span.data_length;
      }
    }
    iree_hal_task_file_submit(transfer, proactor, requests, count, pending);
    iree_hal_task_file_submit_host(transfer, host_pending);
    if (!transfer->active) {
      iree_hal_task_file_finish(transfer, on_worker);
      return;
    }
    int64_t expected = IREE_HAL_TASK_FILE_DRAINING;
    if (iree_atomic_compare_exchange_strong(&transfer->ready, &expected, 0,
                                            iree_memory_order_release,
                                            iree_memory_order_relaxed)) {
      return;
    }
    // A completion raced the release. Consume the newly published bits without
    // spinning for I/O, taking a lock, or moving progress to another thread.
  }
}

static iree_status_t iree_hal_task_file_begin(
    iree_hal_task_queue_op_t* operation,
    iree_hal_task_file_transfer_t** out_transfer) {
  *out_transfer = NULL;
  const iree_hal_file_transfer_list_t* list = &operation->file.transfers;
  uint64_t count = 0;
  if (list->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT) {
    iree_byte_span_t count_span;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_native_host_span(
        list->indirect.count_buffer, list->indirect.count_offset, 8,
        &count_span));
    count = iree_unaligned_load_le_u64(count_span.data);
  } else {
    count = list->host.count;
  }
  iree_hal_file_transfer_iterator_t iterator;
  IREE_RETURN_IF_ERROR(iree_hal_file_transfer_iterator_initialize(
      &operation->file, count, NULL, &iterator));
  if (count && list->type == IREE_HAL_FILE_TRANSFER_LIST_TYPE_INDIRECT) {
    const iree_hal_buffer_binding_t binding = list->indirect.records;
    const iree_device_size_t length =
        (iree_device_size_t)(count * sizeof(iree_hal_file_transfer_record_t));
    iree_byte_span_t record_span;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_native_host_span(
        binding.buffer, binding.offset, length, &record_span));
    if ((operation->flags & IREE_HAL_TASK_QUEUE_OP_FLAG_BEFORE_DEFAULT) &&
        !iree_any_bit_set(iree_hal_buffer_memory_type(binding.buffer),
                          IREE_HAL_MEMORY_TYPE_HOST_COHERENT)) {
      IREE_RETURN_IF_ERROR(iree_hal_buffer_invalidate_host_range(
          binding.buffer, binding.offset, length));
    }
    iterator.records = (const iree_hal_file_transfer_record_t*)record_span.data;
  }
  iree_hal_file_transfer_range_t first_range;
  IREE_RETURN_IF_ERROR(
      iree_hal_file_transfer_iterator_peek(&iterator, &first_range));
  if (!first_range.length) {
    return iree_ok_status();
  }

  // A single linear extent needs only one request slot. Repetition can fill a
  // native window even when the whole rectangle occupies just one descriptor.
  const uint32_t capacity = iterator.index == iterator.count &&
                                    iterator.current.row_count == 1 &&
                                    iterator.current.slice_count == 1
                                ? 1
                                : IREE_HAL_TASK_FILE_REQUEST_CAPACITY;
  iree_hal_task_file_transfer_t* transfer = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_aligned(
      &operation->arena,
      sizeof(*transfer) + capacity * sizeof(transfer->slots[0]),
      iree_alignof(iree_hal_task_file_transfer_t), (void**)&transfer));
  memset(transfer, 0, sizeof(*transfer));
  transfer->operation = operation;
  transfer->capacity =
      capacity == 32 ? UINT32_MAX : (UINT32_C(1) << capacity) - 1;
  iree_atomic_store(&transfer->ready, IREE_HAL_TASK_FILE_DRAINING,
                    iree_memory_order_relaxed);
  for (uint32_t i = 0; i < capacity; ++i) {
    transfer->slots[i].transfer = transfer;
  }
  transfer->iterator = iterator;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      &operation->arena, operation->file.buffers.count,
      sizeof(*transfer->buffers), (void**)&transfer->buffers));
  memset(transfer->buffers, 0,
         operation->file.buffers.count * sizeof(*transfer->buffers));
  *out_transfer = transfer;
  return iree_ok_status();
}

void iree_hal_task_queue_issue_file(iree_hal_task_queue_op_t* operation) {
  iree_hal_task_file_transfer_t* transfer = NULL;
  iree_status_t status = iree_hal_task_file_begin(operation, &transfer);
  if (!iree_status_is_ok(status)) {
    iree_hal_task_queue_op_fail(operation, status);
  } else if (transfer) {
    iree_hal_task_file_drain(transfer, true);
  } else {
    iree_hal_task_queue_op_complete(operation);
  }
}
