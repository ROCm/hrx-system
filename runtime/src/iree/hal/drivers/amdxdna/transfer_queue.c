// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdxdna/transfer_queue.h"

#include <string.h>

#include "iree/base/alignment.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"
#include "iree/hal/drivers/amdxdna/async_queue.h"

#define IREE_HAL_AMDXDNA_TRANSFER_CHUNK_SIZE (1 * 1024 * 1024)

// Keep this intentionally serialized. iree_hal_file_read/write map, sync, and
// unmap amdxdna buffers, and the native buffer mapping path does not currently
// provide a per-buffer serialization contract. More workers can be enabled once
// concurrent scoped maps/syncs are either serialized in the buffer layer or
// proven safe for the Linux KMQ and Windows MCDM shims.
#define IREE_HAL_AMDXDNA_TRANSFER_WORKER_COUNT 1

typedef enum iree_hal_amdxdna_transfer_kind_e {
  IREE_HAL_AMDXDNA_TRANSFER_READ_FILE_TO_BUFFER = 0,
  IREE_HAL_AMDXDNA_TRANSFER_WRITE_BUFFER_TO_FILE = 1,
} iree_hal_amdxdna_transfer_kind_t;

typedef struct iree_hal_amdxdna_transfer_op_t iree_hal_amdxdna_transfer_op_t;

typedef struct iree_hal_amdxdna_transfer_work_item_t {
  iree_hal_amdxdna_transfer_op_t* op;
  struct iree_hal_amdxdna_transfer_work_item_t* next;
} iree_hal_amdxdna_transfer_work_item_t;

struct iree_hal_amdxdna_transfer_queue_t {
  iree_allocator_t host_allocator;
  iree_device_size_t chunk_size;
  iree_hal_amdxdna_async_queue_t* wait_queue;

  iree_thread_t* worker_threads[IREE_HAL_AMDXDNA_TRANSFER_WORKER_COUNT];
  iree_host_size_t worker_count;
  iree_slim_mutex_t work_mutex;
  iree_notification_t work_notification;
  iree_atomic_int32_t shutdown_requested;
  iree_hal_amdxdna_transfer_work_item_t* work_head;
  iree_hal_amdxdna_transfer_work_item_t** work_tail_link;
};

struct iree_hal_amdxdna_transfer_op_t {
  iree_atomic_ref_count_t ref_count;
  iree_hal_amdxdna_transfer_queue_t* queue;
  iree_allocator_t host_allocator;
  iree_hal_amdxdna_transfer_kind_t kind;
  iree_hal_file_t* file;
  uint64_t file_offset;
  iree_hal_buffer_t* buffer;
  iree_device_size_t buffer_offset;
  iree_device_size_t length;
  iree_device_size_t chunk_size;
  iree_hal_semaphore_list_t signal_list;
  iree_atomic_uint64_t next_offset;
  iree_atomic_int32_t active_worker_count;
  iree_atomic_intptr_t error_status;
};

static void iree_hal_amdxdna_transfer_op_retain(
    iree_hal_amdxdna_transfer_op_t* op) {
  iree_atomic_ref_count_inc(&op->ref_count);
}

static void iree_hal_amdxdna_transfer_op_destroy(
    iree_hal_amdxdna_transfer_op_t* op) {
  iree_hal_semaphore_list_free(op->signal_list, op->host_allocator);
  iree_hal_file_release(op->file);
  iree_hal_buffer_release(op->buffer);
  iree_allocator_free(op->host_allocator, op);
}

static void iree_hal_amdxdna_transfer_op_release(
    iree_hal_amdxdna_transfer_op_t* op) {
  if (!op) return;
  if (iree_atomic_ref_count_dec(&op->ref_count) == 1) {
    iree_hal_amdxdna_transfer_op_destroy(op);
  }
}

static void iree_hal_amdxdna_transfer_op_set_error(
    iree_hal_amdxdna_transfer_op_t* op, iree_status_t status) {
  if (iree_status_is_ok(status)) return;
  intptr_t expected = 0;
  if (!iree_atomic_compare_exchange_strong(
          &op->error_status, &expected, (intptr_t)status,
          iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
    iree_status_free(status);
  }
}

static void iree_hal_amdxdna_transfer_op_finish(
    iree_hal_amdxdna_transfer_op_t* op) {
  iree_status_t status = (iree_status_t)iree_atomic_load(
      &op->error_status, iree_memory_order_acquire);
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(op->signal_list, /*frontier=*/NULL);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(op->signal_list, status);
  }
}

static iree_status_t iree_hal_amdxdna_transfer_op_create(
    iree_hal_amdxdna_transfer_queue_t* queue,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdxdna_transfer_kind_t kind, iree_hal_file_t* file,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length,
    iree_hal_amdxdna_transfer_op_t** out_op) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(file);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_ASSERT_ARGUMENT(out_op);
  *out_op = NULL;

  iree_hal_amdxdna_transfer_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(queue->host_allocator, sizeof(*op), (void**)&op));
  memset(op, 0, sizeof(*op));
  iree_atomic_ref_count_init(&op->ref_count);
  op->queue = queue;
  op->host_allocator = queue->host_allocator;
  op->kind = kind;
  op->file = file;
  op->file_offset = file_offset;
  op->buffer = buffer;
  op->buffer_offset = buffer_offset;
  op->length = length;
  op->chunk_size = queue->chunk_size;
  op->signal_list = iree_hal_semaphore_list_empty();
  iree_atomic_store(&op->next_offset, (uint64_t)0, iree_memory_order_relaxed);
  iree_atomic_store(&op->active_worker_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&op->error_status, (intptr_t)0, iree_memory_order_relaxed);
  iree_hal_file_retain(file);
  iree_hal_buffer_retain(buffer);

  iree_status_t status = iree_hal_semaphore_list_clone(
      &signal_semaphore_list, queue->host_allocator, &op->signal_list);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_transfer_op_release(op);
    return status;
  }

  *out_op = op;
  return iree_ok_status();
}

static void iree_hal_amdxdna_transfer_op_cleanup(void* user_data) {
  iree_hal_amdxdna_transfer_op_release(
      (iree_hal_amdxdna_transfer_op_t*)user_data);
}

static bool iree_hal_amdxdna_transfer_queue_pop_locked(
    iree_hal_amdxdna_transfer_queue_t* queue,
    iree_hal_amdxdna_transfer_work_item_t** out_item) {
  *out_item = queue->work_head;
  if (!*out_item) return false;
  queue->work_head = (*out_item)->next;
  if (!queue->work_head) {
    queue->work_tail_link = &queue->work_head;
  }
  (*out_item)->next = NULL;
  return true;
}

static iree_status_t iree_hal_amdxdna_transfer_queue_push_work(
    iree_hal_amdxdna_transfer_queue_t* queue,
    iree_hal_amdxdna_transfer_work_item_t* item) {
  item->next = NULL;
  iree_slim_mutex_lock(&queue->work_mutex);
  const bool shutdown = iree_atomic_load(&queue->shutdown_requested,
                                         iree_memory_order_acquire) != 0;
  if (!shutdown) {
    *queue->work_tail_link = item;
    queue->work_tail_link = &item->next;
  }
  iree_slim_mutex_unlock(&queue->work_mutex);
  if (IREE_UNLIKELY(shutdown)) {
    return iree_make_status(IREE_STATUS_CANCELLED, "transfer queue shut down");
  }
  iree_notification_post(&queue->work_notification, IREE_ALL_WAITERS);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_transfer_op_process_chunk(
    iree_hal_amdxdna_transfer_op_t* op, iree_device_size_t transfer_offset,
    iree_device_size_t chunk_length) {
  switch (op->kind) {
    case IREE_HAL_AMDXDNA_TRANSFER_READ_FILE_TO_BUFFER:
      return iree_hal_file_read(op->file, op->file_offset + transfer_offset,
                                op->buffer, op->buffer_offset + transfer_offset,
                                chunk_length);
    case IREE_HAL_AMDXDNA_TRANSFER_WRITE_BUFFER_TO_FILE:
      return iree_hal_file_write(
          op->file, op->file_offset + transfer_offset, op->buffer,
          op->buffer_offset + transfer_offset, chunk_length);
  }
  return iree_make_status(IREE_STATUS_INTERNAL, "unknown transfer kind");
}

static void iree_hal_amdxdna_transfer_work_item_run(
    iree_hal_amdxdna_transfer_work_item_t* item) {
  iree_hal_amdxdna_transfer_op_t* op = item->op;
  iree_allocator_t host_allocator = op->host_allocator;
  iree_allocator_free(host_allocator, item);

  for (;;) {
    if (!iree_status_is_ok((iree_status_t)iree_atomic_load(
            &op->error_status, iree_memory_order_acquire))) {
      break;
    }
    uint64_t transfer_offset = iree_atomic_fetch_add(
        &op->next_offset, op->chunk_size, iree_memory_order_acq_rel);
    if (transfer_offset >= op->length) break;
    iree_device_size_t chunk_length =
        iree_min(op->chunk_size, op->length - transfer_offset);
    iree_status_t status = iree_hal_amdxdna_transfer_op_process_chunk(
        op, transfer_offset, chunk_length);
    if (!iree_status_is_ok(status)) {
      iree_hal_amdxdna_transfer_op_set_error(op, status);
      break;
    }
  }

  if (iree_atomic_fetch_sub(&op->active_worker_count, 1,
                            iree_memory_order_acq_rel) == 1) {
    iree_hal_amdxdna_transfer_op_finish(op);
  }
  iree_hal_amdxdna_transfer_op_release(op);
}

static int iree_hal_amdxdna_transfer_queue_worker_main(void* arg) {
  iree_hal_amdxdna_transfer_queue_t* queue =
      (iree_hal_amdxdna_transfer_queue_t*)arg;
  for (;;) {
    iree_hal_amdxdna_transfer_work_item_t* item = NULL;
    iree_slim_mutex_lock(&queue->work_mutex);
    iree_hal_amdxdna_transfer_queue_pop_locked(queue, &item);
    const bool shutdown = iree_atomic_load(&queue->shutdown_requested,
                                           iree_memory_order_acquire) != 0;
    iree_slim_mutex_unlock(&queue->work_mutex);

    if (item) {
      iree_hal_amdxdna_transfer_work_item_run(item);
      continue;
    }
    if (shutdown) return 0;

    iree_wait_token_t token =
        iree_notification_prepare_wait(&queue->work_notification);
    iree_slim_mutex_lock(&queue->work_mutex);
    const bool has_work = queue->work_head != NULL;
    const bool shutdown_after_prepare =
        iree_atomic_load(&queue->shutdown_requested,
                         iree_memory_order_acquire) != 0;
    iree_slim_mutex_unlock(&queue->work_mutex);
    if (!has_work && !shutdown_after_prepare) {
      iree_notification_commit_wait(&queue->work_notification, token,
                                    /*spin_ns=*/0, IREE_TIME_INFINITE_FUTURE);
    } else {
      iree_notification_cancel_wait(&queue->work_notification);
    }
  }
}

static iree_status_t iree_hal_amdxdna_transfer_start_op_fn(void* user_data) {
  iree_hal_amdxdna_transfer_op_t* op =
      (iree_hal_amdxdna_transfer_op_t*)user_data;
  iree_hal_amdxdna_transfer_queue_t* queue = op->queue;
  uint64_t chunk_count = (op->length + op->chunk_size - 1) / op->chunk_size;
  iree_host_size_t lane_count =
      (iree_host_size_t)iree_min((uint64_t)queue->worker_count, chunk_count);

  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < lane_count; ++i) {
    iree_hal_amdxdna_transfer_work_item_t* item = NULL;
    status = iree_allocator_malloc(queue->host_allocator, sizeof(*item),
                                   (void**)&item);
    if (!iree_status_is_ok(status)) {
      iree_hal_amdxdna_transfer_op_set_error(op, status);
      break;
    }
    item->op = op;
    item->next = NULL;
    iree_hal_amdxdna_transfer_op_retain(op);
    iree_atomic_fetch_add(&op->active_worker_count, 1,
                          iree_memory_order_acq_rel);
    status = iree_hal_amdxdna_transfer_queue_push_work(queue, item);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(queue->host_allocator, item);
      iree_hal_amdxdna_transfer_op_set_error(op, status);
      if (iree_atomic_fetch_sub(&op->active_worker_count, 1,
                                iree_memory_order_acq_rel) == 1) {
        iree_hal_amdxdna_transfer_op_finish(op);
      }
      iree_hal_amdxdna_transfer_op_release(op);
      break;
    }
  }

  if (iree_atomic_load(&op->active_worker_count, iree_memory_order_acquire) ==
      0) {
    iree_hal_amdxdna_transfer_op_finish(op);
  }

  return iree_status_from_code(IREE_STATUS_DEFERRED);
}

static iree_status_t iree_hal_amdxdna_transfer_queue_enqueue(
    iree_hal_amdxdna_transfer_queue_t* queue,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_amdxdna_transfer_kind_t kind, iree_hal_file_t* file,
    uint64_t file_offset, iree_hal_buffer_t* buffer,
    iree_device_size_t buffer_offset, iree_device_size_t length) {
  iree_hal_amdxdna_transfer_op_t* op = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_transfer_op_create(
      queue, signal_semaphore_list, kind, file, file_offset, buffer,
      buffer_offset, length, &op));

  iree_status_t status = iree_hal_amdxdna_async_queue_enqueue(
      queue->wait_queue, wait_semaphore_list, signal_semaphore_list,
      iree_hal_amdxdna_transfer_start_op_fn,
      iree_hal_amdxdna_transfer_op_cleanup, op,
      /*retained_resources=*/NULL, /*retained_resource_count=*/0);
  if (!iree_status_is_ok(status)) {
    iree_hal_amdxdna_transfer_op_release(op);
    iree_hal_semaphore_list_fail(signal_semaphore_list,
                                 iree_status_clone(status));
  }
  return status;
}

iree_status_t iree_hal_amdxdna_transfer_queue_create(
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_amdxdna_transfer_queue_t** out_queue) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_queue);
  *out_queue = NULL;

  iree_hal_amdxdna_transfer_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*queue), (void**)&queue));
  memset(queue, 0, sizeof(*queue));
  queue->host_allocator = host_allocator;
  queue->chunk_size = IREE_HAL_AMDXDNA_TRANSFER_CHUNK_SIZE;
  queue->work_tail_link = &queue->work_head;
  iree_slim_mutex_initialize(&queue->work_mutex);
  iree_notification_initialize(&queue->work_notification);
  iree_atomic_store(&queue->shutdown_requested, 0, iree_memory_order_relaxed);

  iree_status_t status = iree_hal_amdxdna_async_queue_create(
      block_pool, host_allocator, &queue->wait_queue);
  const iree_host_size_t target_worker_count =
      iree_min((iree_host_size_t)IREE_HAL_AMDXDNA_TRANSFER_WORKER_COUNT,
               (iree_host_size_t)IREE_ARRAYSIZE(queue->worker_threads));
  for (iree_host_size_t i = 0;
       i < target_worker_count && iree_status_is_ok(status); ++i) {
    iree_thread_create_params_t thread_params = {0};
    thread_params.name = iree_make_cstring_view("amdxdna-transfer");
    status = iree_thread_create(iree_hal_amdxdna_transfer_queue_worker_main,
                                queue, thread_params, host_allocator,
                                &queue->worker_threads[i]);
    if (iree_status_is_ok(status)) {
      queue->worker_count = i + 1;
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_atomic_store(&queue->shutdown_requested, 1, iree_memory_order_release);
    iree_notification_post(&queue->work_notification, IREE_ALL_WAITERS);
    for (iree_host_size_t i = 0; i < queue->worker_count; ++i) {
      iree_thread_release(queue->worker_threads[i]);
    }
    iree_hal_amdxdna_async_queue_destroy(queue->wait_queue);
    iree_notification_deinitialize(&queue->work_notification);
    iree_slim_mutex_deinitialize(&queue->work_mutex);
    iree_allocator_free(host_allocator, queue);
    return status;
  }

  *out_queue = queue;
  return iree_ok_status();
}

void iree_hal_amdxdna_transfer_queue_set_frontier(
    iree_hal_amdxdna_transfer_queue_t* queue,
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis,
    iree_atomic_uint64_t* epoch_counter) {
  if (!queue) return;
  iree_hal_amdxdna_async_queue_set_frontier(queue->wait_queue, tracker, axis,
                                            epoch_counter);
}

void iree_hal_amdxdna_transfer_queue_destroy(
    iree_hal_amdxdna_transfer_queue_t* queue) {
  if (!queue) return;
  iree_hal_amdxdna_async_queue_destroy(queue->wait_queue);
  iree_atomic_store(&queue->shutdown_requested, 1, iree_memory_order_release);
  iree_notification_post(&queue->work_notification, IREE_ALL_WAITERS);
  for (iree_host_size_t i = 0; i < queue->worker_count; ++i) {
    iree_thread_release(queue->worker_threads[i]);
  }
  iree_notification_deinitialize(&queue->work_notification);
  iree_slim_mutex_deinitialize(&queue->work_mutex);
  iree_allocator_free(queue->host_allocator, queue);
}

iree_status_t iree_hal_amdxdna_transfer_queue_read(
    iree_hal_amdxdna_transfer_queue_t* queue,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_file_t* source_file, uint64_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  if (!iree_hal_file_supports_synchronous_io(source_file)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "queue_read requires a storage-buffer-backed or synchronous file");
  }
  return iree_hal_amdxdna_transfer_queue_enqueue(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDXDNA_TRANSFER_READ_FILE_TO_BUFFER, source_file, source_offset,
      target_buffer, target_offset, length);
}

iree_status_t iree_hal_amdxdna_transfer_queue_write(
    iree_hal_amdxdna_transfer_queue_t* queue,
    iree_hal_semaphore_list_t wait_semaphore_list,
    iree_hal_semaphore_list_t signal_semaphore_list,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_file_t* target_file, uint64_t target_offset,
    iree_device_size_t length) {
  if (!iree_hal_file_supports_synchronous_io(target_file)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "queue_write requires a storage-buffer-backed or synchronous file");
  }
  return iree_hal_amdxdna_transfer_queue_enqueue(
      queue, wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_AMDXDNA_TRANSFER_WRITE_BUFFER_TO_FILE, target_file,
      target_offset, source_buffer, source_offset, length);
}
