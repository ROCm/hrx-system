// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_PLATFORM_IO_URING_CLOSE_H_
#define IREE_ASYNC_PLATFORM_IO_URING_CLOSE_H_

#include "iree/async/operations/file.h"
#include "iree/async/platform/io_uring/proactor.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

enum iree_async_io_uring_close_flags_e {
  // A cancellation request can win until the worker commits to the syscall.
  IREE_ASYNC_IO_URING_CLOSE_CANCELLED = 1u << 0,
  // The worker issued the syscall and recorded its actual filesystem result.
  IREE_ASYNC_IO_URING_CLOSE_ISSUED = 1u << 1,
};

// Acquires blocking capacity before an incoming batch transfers any reference.
iree_status_t iree_async_io_uring_ensure_close_workers(
    iree_async_proactor_io_uring_t* proactor);

// Joins any workers while the proactor and its wake descriptor are still live.
void iree_async_io_uring_close_workers_free(
    iree_async_proactor_io_uring_t* proactor);

// Transfers the operation, its suffix and |status| to a checked file worker.
// An error skips the syscall and retires only the transferred file reference.
void iree_async_io_uring_enqueue_file_close(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_file_close_operation_t* operation, iree_status_t status);

// Retires the head with |status| and cancels its unissued successors in order.
void iree_async_io_uring_fail_chain(iree_async_proactor_io_uring_t* proactor,
                                    iree_async_operation_t* head,
                                    iree_status_t status);

// Poll-owner delivery for a checked file or a skipped socket close.
iree_host_size_t iree_async_io_uring_dispatch_close(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_operation_t* operation);

// Prepares private native closes for final socket references. Returns true
// when more sockets remain than fit in the current submission extent.
bool iree_async_io_uring_submit_socket_retirements(
    iree_async_proactor_io_uring_t* proactor);

// Retires the wrapper and reports the original skipped operation result.
iree_host_size_t iree_async_io_uring_complete_socket_retirement(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe,
    iree_status_t* inout_poll_status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
#endif  // IREE_ASYNC_PLATFORM_IO_URING_CLOSE_H_
