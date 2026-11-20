// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <errno.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "iree/async/operations/scheduling.h"
#include "iree/async/platform/io_uring/close.h"
#include "iree/async/platform/io_uring/notification.h"
#include "iree/async/platform/io_uring/proactor.h"
#include "iree/async/util/semaphore_wait.h"
#include "iree/async/util/sequence_emulation.h"

iree_status_t iree_async_proactor_io_uring_cancel(
    iree_async_proactor_t* base_proactor, iree_async_operation_t* operation) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);

  if (operation->type == IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE) {
    iree_async_operation_set_internal_flags(
        operation, IREE_ASYNC_IO_URING_CLOSE_CANCELLED);
    return iree_ok_status();
  }
  if (operation->type == IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT) {
    iree_async_io_uring_notification_cancel_wait(
        (iree_async_notification_wait_operation_t*)operation);
    return iree_ok_status();
  }

  // The poll owner joins software timepoints before retiring their trackers.
  if (operation->type == IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT) {
    iree_async_semaphore_wait_context_request_cancellation(
        &proactor->semaphore_wait_context,
        (iree_async_semaphore_wait_operation_t*)operation);
    return iree_ok_status();
  }

  // Sequences propagate cancellation to their current in-flight child.
  if (operation->type == IREE_ASYNC_OPERATION_TYPE_SEQUENCE) {
    return iree_async_sequence_cancel(
        base_proactor, (iree_async_sequence_operation_t*)operation);
  }

  iree_io_uring_ring_sq_lock(&proactor->ring);
  iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
  if (!sqe && iree_atomic_load(&proactor->polling.owner_tid,
                               iree_memory_order_relaxed) ==
                  (int32_t)syscall(__NR_gettid)) {
    // Cancellation from an owner callback must not depend on producers leaving
    // room in the SQ. Submit the prepared batch without waiting for completion,
    // retaining the lock until the cancellation has claimed a freed slot.
    iree_status_t status =
        iree_io_uring_ring_submit_pending_locked(&proactor->ring);
    if (!iree_status_is_ok(status)) {
      iree_io_uring_ring_sq_unlock(&proactor->ring);
      return status;
    }
    sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
  }
  if (!sqe) {
    iree_io_uring_ring_sq_unlock(&proactor->ring);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "SQ full, cannot submit cancel");
  }

  // ASYNC_CANCEL targets the operation's user_data. Its own CQE carries no
  // operation pointer; only the target's terminal CQE returns ownership.
  sqe->opcode = IREE_IORING_OP_ASYNC_CANCEL;
  sqe->fd = -1;
  if (operation->type == IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT) {
    // Cancel the POLL_ADD head of the linked POLL_ADD+READ pair. The READ CQE
    // reports cancellation and owns resource release and the callback; the
    // tagged POLL_ADD completion never dereferences the operation.
    sqe->addr = iree_io_uring_internal_encode(IREE_IO_URING_TAG_LINKED_POLL,
                                              (uintptr_t)operation);
  } else {
    sqe->addr = (uint64_t)(uintptr_t)operation;
  }
  sqe->user_data = iree_io_uring_internal_encode(IREE_IO_URING_TAG_CANCEL, 0);
  iree_io_uring_ring_sq_unlock(&proactor->ring);

  // Only the poll owner may enter a SINGLE_ISSUER ring. Other callers leave
  // submission to the next poll; the SQE is already fully prepared.
  iree_async_proactor_wake(&proactor->base);
  return iree_ok_status();
}

iree_status_t iree_async_proactor_io_uring_submit_cancel_requests(
    iree_async_proactor_io_uring_t* proactor) {
  if (!proactor->base.cancellations.list.head) {
    return iree_ok_status();
  }

  iree_status_t status = iree_ok_status();
  iree_io_uring_ring_sq_lock(&proactor->ring);
  for (iree_host_size_t i = 0;
       i < IREE_ASYNC_CANCEL_REQUEST_SERVICE_BUDGET &&
       proactor->base.cancellations.list.head && iree_status_is_ok(status);
       ++i) {
    iree_async_cancel_request_t* request =
        (iree_async_cancel_request_t*)proactor->base.cancellations.list.head;
    iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
    if (!sqe) {
      status = iree_io_uring_ring_submit_pending_locked(&proactor->ring);
      if (iree_status_is_ok(status)) {
        sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
        if (!sqe) {
          status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                    "SQ remained full after submission");
        }
      }
    }
    if (iree_status_is_ok(status)) {
      sqe->opcode = IREE_IORING_OP_ASYNC_CANCEL;
      sqe->fd = -1;
      sqe->addr = (uint64_t)(uintptr_t)request->target;
      sqe->user_data = iree_io_uring_internal_encode(IREE_IO_URING_TAG_CANCEL,
                                                     (uintptr_t)request);
      // A prepared SQE owns the key even if the subsequent enter fails.
      iree_async_proactor_issue_cancel_request(&proactor->base, request);
    }
  }
  iree_io_uring_ring_sq_unlock(&proactor->ring);
  if (proactor->base.cancellations.list.head) {
    iree_async_proactor_wake(&proactor->base);
  }
  return status;
}

iree_status_t iree_async_proactor_io_uring_complete_cancel_request(
    const iree_io_uring_cqe_t* cqe, iree_host_size_t* out_completed_count) {
  iree_async_cancel_request_t* request =
      (iree_async_cancel_request_t*)(uintptr_t)iree_io_uring_internal_payload(
          cqe->user_data);
  if (!request) {
    return iree_ok_status();
  }

  iree_status_t status = iree_ok_status();
  if (cqe->res < 0 && cqe->res != -ENOENT && cqe->res != -EALREADY) {
    status = iree_make_status(iree_status_code_from_errno(-cqe->res),
                              "io_uring cancellation failed: %d", -cqe->res);
  }
  // Even an error CQE retires the cancellation key. The target remains owned
  // until its independent terminal callback, which may still be outstanding.
  ++*out_completed_count;
  iree_async_cancel_request_complete(request);
  return status;
}
