// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/posix/worker.h"

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <unistd.h>

#include "iree/async/file.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/net.h"
#include "iree/async/platform/posix/close.h"
#include "iree/async/platform/posix/proactor.h"
#include "iree/async/util/completion_pool.h"
#include "iree/async/util/operation_completion.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/base/internal/file_io_posix.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"

//===----------------------------------------------------------------------===//
// File operation handlers (blocking I/O)
//===----------------------------------------------------------------------===//

// Executes a FILE_OPEN operation using the open() syscall.
static iree_status_t iree_async_posix_execute_file_open(
    iree_async_proactor_posix_t* proactor,
    iree_async_file_open_operation_t* op) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Translate open flags to POSIX flags.
  int flags = 0;
  if (iree_any_bit_set(op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_READ)) {
    flags |= iree_any_bit_set(op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_WRITE)
                 ? O_RDWR
                 : O_RDONLY;
  } else if (iree_any_bit_set(op->open_flags,
                              IREE_ASYNC_FILE_OPEN_FLAG_WRITE)) {
    flags |= O_WRONLY;
  }
  if (iree_any_bit_set(op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_CREATE)) {
    flags |= O_CREAT;
  }
  if (iree_any_bit_set(op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_TRUNCATE)) {
    flags |= O_TRUNC;
  }
  if (iree_any_bit_set(op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_APPEND)) {
    flags |= O_APPEND;
  }
  flags |= O_CLOEXEC;  // Always set close-on-exec for safety.

  // Default mode for newly created files (rw-rw-r--).
  mode_t mode = 0664;

  // Execute the open syscall.
  int fd = open(op->path, flags, mode);
  if (fd < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "failed to open file '%s': %s", op->path,
                            strerror(errno));
  }

  iree_async_file_flags_t file_flags = IREE_ASYNC_FILE_FLAG_NONE;
  iree_status_t status = iree_ok_status();
  if (iree_any_bit_set(op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_DIRECT)) {
    file_flags |= IREE_ASYNC_FILE_FLAG_DIRECT;
    status = iree_file_enable_direct_io(fd);
  }

  // Publish only after native mode and wrapper creation both succeed.
  iree_async_primitive_t primitive = {.type = IREE_ASYNC_PRIMITIVE_TYPE_FD,
                                      .value = {.fd = fd}};
  if (iree_status_is_ok(status)) {
    status = iree_async_file_import(&proactor->base, primitive, file_flags,
                                    &op->opened_file);
  }
  if (!iree_status_is_ok(status)) {
    close(fd);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Executes a FILE_READ operation using the pread() syscall.
static iree_status_t iree_async_posix_execute_file_read(
    iree_async_file_read_operation_t* op) {
  IREE_TRACE_ZONE_BEGIN(z0);

  int fd = op->file->primitive.value.fd;
  void* buffer = iree_async_span_ptr(op->buffer);
  size_t length = op->buffer.length;
  off_t offset = (off_t)op->offset;

  // Execute the pread syscall (blocking).
  ssize_t result = pread(fd, buffer, length, offset);
  if (result < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "pread failed at offset %" PRIu64 ": %s",
                            op->offset, strerror(errno));
  }

  // Success - record bytes read.
  op->bytes_read = (iree_host_size_t)result;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Executes a FILE_WRITE operation using the pwrite() syscall.
static iree_status_t iree_async_posix_execute_file_write(
    iree_async_file_write_operation_t* op) {
  IREE_TRACE_ZONE_BEGIN(z0);

  int fd = op->file->primitive.value.fd;
  const void* buffer = iree_async_span_ptr(op->buffer);
  size_t length = op->buffer.length;
  off_t offset = (off_t)op->offset;

  // Execute the pwrite syscall (blocking).
  ssize_t result = pwrite(fd, buffer, length, offset);
  if (result < 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(iree_status_code_from_errno(errno),
                            "pwrite failed at offset %" PRIu64 ": %s",
                            op->offset, strerror(errno));
  }

  // Success - record bytes written.
  op->bytes_written = (iree_host_size_t)result;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Executes a FILE_CLOSE operation using the close() syscall.
static iree_status_t iree_async_posix_execute_file_close(
    iree_async_file_close_operation_t* op) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status =
      iree_async_posix_close_fd(&op->file->primitive.value.fd);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// File work and completion handoff
//===----------------------------------------------------------------------===//

void iree_async_posix_file_queue_push(
    iree_async_proactor_posix_t* proactor,
    const iree_async_posix_file_batch_t* batch) {
  iree_async_posix_file_queue_t* queue = &proactor->file_queue;
  iree_slim_mutex_lock(&queue->mutex);
  if (queue->tail) {
    queue->tail->next = batch->head;
  } else {
    queue->head = batch->head;
  }
  queue->tail = batch->tail;
  iree_slim_mutex_unlock(&queue->mutex);
  iree_notification_post(
      &queue->notification,
      (int32_t)iree_min(batch->count, proactor->worker_count));
}

static iree_async_operation_t* iree_async_posix_file_queue_pop(
    iree_async_posix_file_queue_t* queue) {
  iree_slim_mutex_lock(&queue->mutex);
  iree_async_operation_t* operation = queue->head;
  if (operation) {
    queue->head = operation->next;
    if (!queue->head) {
      queue->tail = NULL;
    }
    operation->next = NULL;
  }
  iree_slim_mutex_unlock(&queue->mutex);
  return operation;
}

static iree_status_t iree_async_posix_execute_file_operation(
    iree_async_proactor_posix_t* proactor, iree_async_operation_t* operation) {
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_FILE_OPEN:
      return iree_async_posix_execute_file_open(
          proactor, (iree_async_file_open_operation_t*)operation);
    case IREE_ASYNC_OPERATION_TYPE_FILE_READ:
      return iree_async_posix_execute_file_read(
          (iree_async_file_read_operation_t*)operation);
    case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE:
      return iree_async_posix_execute_file_write(
          (iree_async_file_write_operation_t*)operation);
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE: {
      iree_async_socket_t* socket =
          ((iree_async_socket_close_operation_t*)operation)->socket;
      iree_status_t status =
          iree_async_posix_close_fd(&socket->primitive.value.fd);
      socket->state = IREE_ASYNC_SOCKET_STATE_CLOSED;
      return status;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE:
      return iree_async_posix_execute_file_close(
          (iree_async_file_close_operation_t*)operation);
    default:
      IREE_ASSERT_UNREACHABLE(
          "only blocking file operations enter the worker queue");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static bool iree_async_posix_worker_completion_consumed(void* user_data) {
  iree_async_posix_worker_t* worker = (iree_async_posix_worker_t*)user_data;
  return !iree_atomic_load(&worker->reserved_completion_pending,
                           iree_memory_order_acquire);
}

void iree_async_posix_worker_release_completion(
    iree_async_posix_completion_t* completion) {
  iree_async_posix_worker_t* worker = iree_containerof(
      completion, iree_async_posix_worker_t, reserved_completion);
  iree_atomic_store(&worker->reserved_completion_pending, 0,
                    iree_memory_order_release);
  iree_notification_post(&worker->completion_consumed, 1);
}

static void iree_async_posix_worker_publish(
    iree_async_posix_worker_t* worker, iree_async_operation_t* operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  iree_async_proactor_posix_t* proactor = worker->proactor;
  iree_async_posix_completion_t* completion =
      iree_async_posix_completion_pool_acquire(&proactor->completion_pool);
  const bool use_reserved = !completion;
  if (use_reserved) {
    completion = &worker->reserved_completion;
    iree_atomic_store(&worker->reserved_completion_pending, 1,
                      iree_memory_order_relaxed);
  }
  completion->operation = operation;
  completion->status = status;
  completion->flags = flags;
  iree_atomic_slist_push(&proactor->completion_queue, &completion->slist_entry);
  iree_async_proactor_posix_wake_poll_thread(proactor);

  // Once the shared buffer fills, each worker publishes one last result and
  // sleeps until the poll owner takes it. The owner returns the slot before
  // invoking the callback, which may submit more file work.
  if (use_reserved) {
    iree_notification_await(&worker->completion_consumed,
                            iree_async_posix_worker_completion_consumed, worker,
                            iree_infinite_timeout());
  }
}

//===----------------------------------------------------------------------===//
// Worker thread entry point
//===----------------------------------------------------------------------===//

static int iree_async_posix_worker_main(void* user_data) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_posix_worker_t* worker = (iree_async_posix_worker_t*)user_data;
  iree_async_proactor_posix_t* proactor = worker->proactor;
  iree_async_posix_file_queue_t* queue = &proactor->file_queue;
  for (;;) {
    // Take the wait token before checking either condition. A racing enqueue
    // or shutdown then prevents commit_wait from sleeping through its wake.
    iree_wait_token_t token =
        iree_notification_prepare_wait(&queue->notification);
    if (iree_atomic_load(&proactor->shutdown_requested,
                         iree_memory_order_acquire)) {
      iree_notification_cancel_wait(&queue->notification);
      break;
    }
    iree_async_operation_t* operation = iree_async_posix_file_queue_pop(queue);
    if (!operation) {
      iree_notification_commit_wait(&queue->notification, token, /*spin_ns=*/0,
                                    IREE_TIME_INFINITE_FUTURE);
      continue;
    }
    iree_notification_cancel_wait(&queue->notification);
    IREE_TRACE_ZONE_BEGIN_NAMED(z1, "worker_execute_operation");
    const bool cancelled =
        iree_any_bit_set(iree_async_operation_load_internal_flags(operation),
                         IREE_ASYNC_POSIX_INTERNAL_FLAG_CANCELLED);
    iree_status_t status =
        cancelled
            ? iree_status_from_code(IREE_STATUS_CANCELLED)
            : iree_async_posix_execute_file_operation(proactor, operation);
    IREE_TRACE_ZONE_END(z1);
    iree_async_posix_worker_publish(
        worker, operation, status,
        cancelled ? IREE_ASYNC_COMPLETION_FLAG_NONE
                  : IREE_ASYNC_COMPLETION_FLAG_NATIVE_RESULT);
  }
  IREE_TRACE_ZONE_END(z0);
  return 0;
}

//===----------------------------------------------------------------------===//
// Worker lifecycle
//===----------------------------------------------------------------------===//

iree_status_t iree_async_posix_worker_initialize(
    iree_async_proactor_posix_t* proactor, iree_thread_affinity_t affinity,
    const iree_thread_affinity_set_t* processor_set, iree_allocator_t allocator,
    iree_async_posix_worker_t* out_worker) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Zero-init the worker struct. This ensures cleanup in deinitialize works
  // even if we fail partway through initialization.
  memset(out_worker, 0, sizeof(*out_worker));

  out_worker->proactor = proactor;
  out_worker->reserved_completion.worker_reserved = true;
  iree_notification_initialize(&out_worker->completion_consumed);

  // Create thread parameters.
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));
  params.name = iree_make_cstring_view("async_worker");
  params.create_suspended = false;

  params.initial_affinity = affinity;
  params.initial_processor_set = processor_set;

  // Create and start the worker thread.
  iree_status_t status =
      iree_thread_create(iree_async_posix_worker_main, out_worker, params,
                         allocator, &out_worker->thread);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_async_posix_worker_deinitialize(iree_async_posix_worker_t* worker) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_thread_release(worker->thread);
  iree_notification_deinitialize(&worker->completion_consumed);
  IREE_TRACE_ZONE_END(z0);
}
