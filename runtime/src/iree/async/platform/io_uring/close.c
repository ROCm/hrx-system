// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "iree/async/platform/io_uring/close.h"

#include <errno.h>
#include <sched.h>

#include "iree/async/operations/net.h"
#include "iree/async/platform/io_uring/socket.h"
#include "iree/async/platform/posix/descriptor.h"
#include "iree/async/platform/posix/file_queue.h"
#include "iree/async/util/continuation.h"
#include "iree/async/util/operation_completion.h"
#include "iree/base/threading/thread.h"

// A second unrelated file can complete while a filesystem holds one close
// inside flush. Only explicit FILE_CLOSE admission constructs this service.
typedef struct iree_async_io_uring_close_workers_t {
  // Native wake and completion destination. The owner joins before teardown.
  iree_async_proactor_io_uring_t* proactor;
  // Operations ready for a checked close or last-reference retirement.
  iree_async_posix_file_queue_t queue;
  // Owned worker handles. A null handle is valid during partial startup.
  iree_thread_t* threads[2];
} iree_async_io_uring_close_workers_t;

// The close-specific status storage leaves linked_next intact. The poll owner
// dispatches the suffix only after the close has retired.
static void iree_async_io_uring_publish_close(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_operation_t* operation) {
  iree_atomic_slist_push(&proactor->pending_software_operations,
                         (iree_atomic_slist_entry_t*)operation);
  iree_async_proactor_wake(&proactor->base);
}

static int iree_async_io_uring_close_worker_main(void* user_data) {
  iree_async_io_uring_close_workers_t* workers = user_data;
  iree_async_operation_t* base = NULL;
  while ((base = iree_async_posix_file_queue_take(&workers->queue))) {
    iree_async_file_close_operation_t* close_op =
        (iree_async_file_close_operation_t*)base;
    if (iree_status_is_ok(close_op->result) &&
        iree_any_bit_set(iree_async_operation_load_internal_flags(base),
                         IREE_ASYNC_IO_URING_CLOSE_CANCELLED)) {
      close_op->result = iree_status_from_code(IREE_STATUS_CANCELLED);
    }
    if (iree_status_is_ok(close_op->result)) {
      // A ring CQE cannot distinguish cancellation before issue from a
      // filesystem flush returning ECANCELED after descriptor detachment.
      iree_async_operation_set_internal_flags(base,
                                              IREE_ASYNC_IO_URING_CLOSE_ISSUED);
      close_op->result =
          iree_async_posix_close_fd(&close_op->file->primitive.value.fd);
    } else {
      // Final release can enter a blocking filesystem even on a skipped close.
      iree_async_operation_release_resources(base, NULL);
    }
    iree_async_io_uring_publish_close(workers->proactor, base);
  }
  return 0;
}

static void iree_async_io_uring_close_workers_destroy(
    iree_async_io_uring_close_workers_t* workers) {
  iree_async_posix_file_queue_stop(&workers->queue);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(workers->threads); ++i) {
    iree_thread_release(workers->threads[i]);
  }
  iree_async_posix_file_queue_deinitialize(&workers->queue);
  iree_allocator_free(workers->proactor->base.allocator, workers);
}

static iree_status_t iree_async_io_uring_create_close_workers(
    iree_async_proactor_io_uring_t* proactor) {
  iree_async_io_uring_close_workers_t* workers = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      proactor->base.allocator, sizeof(*workers), (void**)&workers));
  workers->proactor = proactor;
  iree_async_posix_file_queue_initialize(&workers->queue);
  iree_thread_create_params_t params = {
      .name = iree_make_cstring_view("async_close")};
  iree_thread_affinity_set_t processor_set;
  if (proactor->worker_affinity) {
    const size_t mask_size =
        CPU_ALLOC_SIZE(IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT);
    memset(&processor_set, 0, sizeof(processor_set));
    iree_bitmap_t set = iree_thread_affinity_set_bitmap(&processor_set);
    for (uint32_t cpu = 0; cpu < IREE_THREAD_AFFINITY_MAX_PROCESSOR_COUNT;
         ++cpu) {
      if (CPU_ISSET_S(cpu, mask_size,
                      (const cpu_set_t*)proactor->worker_affinity)) {
        iree_bitmap_set(set, cpu);
      }
    }
    params.initial_processor_set = &processor_set;
  }
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(workers->threads) && iree_status_is_ok(status); ++i) {
    status = iree_thread_create(iree_async_io_uring_close_worker_main, workers,
                                params, proactor->base.allocator,
                                &workers->threads[i]);
  }
  if (iree_status_is_ok(status)) {
    proactor->close_workers = workers;
  } else {
    iree_async_io_uring_close_workers_destroy(workers);
  }
  return status;
}

iree_status_t iree_async_io_uring_ensure_close_workers(
    iree_async_proactor_io_uring_t* proactor) {
  iree_slim_mutex_lock(&proactor->close_workers_mutex);
  iree_status_t status =
      proactor->close_workers
          ? iree_ok_status()
          : iree_async_io_uring_create_close_workers(proactor);
  iree_slim_mutex_unlock(&proactor->close_workers_mutex);
  return status;
}

void iree_async_io_uring_close_workers_free(
    iree_async_proactor_io_uring_t* proactor) {
  if (proactor->close_workers) {
    iree_async_io_uring_close_workers_destroy(proactor->close_workers);
  }
}

void iree_async_io_uring_enqueue_file_close(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_file_close_operation_t* operation, iree_status_t status) {
  operation->result = status;
  operation->base.next = NULL;
  iree_async_posix_file_batch_t batch = {
      .head = &operation->base, .tail = &operation->base, .count = 1};
  iree_async_posix_file_queue_push(
      &proactor->close_workers->queue, &batch,
      IREE_ARRAYSIZE(proactor->close_workers->threads));
}

void iree_async_io_uring_fail_chain(iree_async_proactor_io_uring_t* proactor,
                                    iree_async_operation_t* operation,
                                    iree_status_t status) {
  while (operation) {
    if (operation->type == IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE) {
      iree_async_io_uring_enqueue_file_close(
          proactor, (iree_async_file_close_operation_t*)operation, status);
      return;
    }
    if (operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE) {
      ((iree_async_socket_close_operation_t*)operation)->retirement_status =
          status;
      iree_async_io_uring_publish_close(proactor, operation);
      return;
    }
    iree_async_operation_t* next =
        iree_async_continuation_take_cancelled(operation);
    if (operation->completion_fn) {
      iree_async_proactor_io_uring_push_software_operation(proactor, operation,
                                                           status);
    } else {
      // A suppressed message tail can be freed by the predecessor callback.
      iree_async_operation_complete(operation, status,
                                    IREE_ASYNC_COMPLETION_FLAG_NONE);
    }
    operation = next;
    status = iree_status_from_code(IREE_STATUS_CANCELLED);
  }
}

static iree_host_size_t iree_async_io_uring_finish_close(
    iree_async_proactor_io_uring_t* proactor, iree_async_operation_t* operation,
    iree_status_t status, iree_async_completion_flags_t flags) {
  iree_async_operation_t* next = iree_async_continuation_take(operation);
  if (next) {
    if (iree_status_is_ok(status)) {
      iree_async_proactor_io_uring_dispatch_continuation_chain(proactor, next);
    } else {
      iree_async_proactor_io_uring_cancel_continuation_chain_to_mpsc(proactor,
                                                                     next);
    }
  }
  return iree_async_operation_complete(operation, status, flags);
}

iree_host_size_t iree_async_io_uring_dispatch_close(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_operation_t* operation) {
  if (operation->type == IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE) {
    iree_async_file_close_operation_t* close_op =
        (iree_async_file_close_operation_t*)operation;
    const bool issued =
        iree_any_bit_set(iree_async_operation_load_internal_flags(operation),
                         IREE_ASYNC_IO_URING_CLOSE_ISSUED);
    iree_status_t status = close_op->result;
    close_op->result = iree_ok_status();
    return iree_async_io_uring_finish_close(
        proactor, operation, status,
        issued ? IREE_ASYNC_COMPLETION_FLAG_NATIVE_RESULT
               : IREE_ASYNC_COMPLETION_FLAG_NONE);
  }

  iree_async_socket_close_operation_t* close_op =
      (iree_async_socket_close_operation_t*)operation;
  // Consume exactly the transferred reference. Only the final decrement owns
  // native destruction; all other owners keep a usable socket.
  operation->resources_acquired = false;
  if (iree_atomic_ref_count_dec(&close_op->socket->ref_count) == 1) {
    operation->next = proactor->retiring_sockets;
    proactor->retiring_sockets = operation;
    if (iree_async_io_uring_submit_socket_retirements(proactor)) {
      iree_async_proactor_wake(&proactor->base);
    }
    return 0;
  }
  iree_status_t status = close_op->retirement_status;
  close_op->retirement_status = iree_ok_status();
  return iree_async_io_uring_finish_close(proactor, operation, status,
                                          IREE_ASYNC_COMPLETION_FLAG_NONE);
}

bool iree_async_io_uring_submit_socket_retirements(
    iree_async_proactor_io_uring_t* proactor) {
  if (!proactor->retiring_sockets) {
    return false;
  }
  iree_io_uring_ring_sq_lock(&proactor->ring);
  while (proactor->retiring_sockets) {
    iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
    if (!sqe) {
      break;
    }
    iree_async_socket_close_operation_t* close_op =
        (iree_async_socket_close_operation_t*)proactor->retiring_sockets;
    proactor->retiring_sockets = close_op->base.next;
    close_op->base.next = NULL;
    sqe->opcode = IREE_IORING_OP_CLOSE;
    sqe->flags = IREE_IOSQE_ASYNC;
    sqe->fd = close_op->socket->primitive.value.fd;
    // User cancellation targets the public operation and cannot cancel its
    // private last-reference cleanup.
    sqe->user_data = iree_io_uring_internal_encode(
        IREE_IO_URING_TAG_SOCKET_RETIRE, close_op);
  }
  iree_io_uring_ring_sq_unlock(&proactor->ring);
  return proactor->retiring_sockets != NULL;
}

iree_host_size_t iree_async_io_uring_complete_socket_retirement(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe,
    iree_status_t* inout_poll_status) {
  iree_async_socket_close_operation_t* close_op =
      (iree_async_socket_close_operation_t*)(uintptr_t)
          iree_io_uring_internal_payload(cqe->user_data);
  if (cqe->res < 0 && cqe->res != -EBADF) {
    // A native socket has no filesystem flush. A refused cleanup still owns
    // the descriptor. Keep the operation for the poll caller's next attempt.
    close_op->base.next = proactor->retiring_sockets;
    proactor->retiring_sockets = &close_op->base;
    *inout_poll_status = iree_status_join(
        *inout_poll_status,
        iree_make_status(iree_status_code_from_errno(-cqe->res),
                         "retiring socket close: %s", strerror(-cqe->res)));
    return 0;
  }
  close_op->socket->primitive.value.fd = -1;
  iree_async_io_uring_socket_destroy(proactor, close_op->socket);
  iree_status_t status = close_op->retirement_status;
  close_op->retirement_status = iree_ok_status();
  return iree_async_io_uring_finish_close(proactor, &close_op->base, status,
                                          IREE_ASYNC_COMPLETION_FLAG_NONE);
}
