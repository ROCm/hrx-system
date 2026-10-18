// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Owner-task dispatch for io_uring_register operations.

#ifndef IREE_ASYNC_PLATFORM_IO_URING_URING_REGISTRATION_H_
#define IREE_ASYNC_PLATFORM_IO_URING_URING_REGISTRATION_H_

#include <sys/uio.h>

#include "iree/async/platform/io_uring/defs.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Wakes the task that owns io_uring registration after a request is queued.
typedef struct iree_io_uring_registration_wake_callback_t {
  // Callback function invoked without the registration mutex held.
  void (*fn)(void* user_data);

  // Opaque value passed to |fn|.
  void* user_data;
} iree_io_uring_registration_wake_callback_t;

// Returns an empty registration wake callback.
static inline iree_io_uring_registration_wake_callback_t
iree_io_uring_registration_wake_callback_null(void) {
  iree_io_uring_registration_wake_callback_t callback = {NULL, NULL};
  return callback;
}

// Kernel ownership state for io_uring_register operations.
typedef enum iree_io_uring_registration_state_e {
  // The ring has no SINGLE_ISSUER restriction.
  IREE_IO_URING_REGISTRATION_STATE_DIRECT = 0,

  // The ring uses SINGLE_ISSUER but remains disabled and ownerless.
  IREE_IO_URING_REGISTRATION_STATE_UNBOUND = 1,

  // The ring uses SINGLE_ISSUER and |owner_tid| identifies its owner task.
  IREE_IO_URING_REGISTRATION_STATE_BOUND = 2,

  // Polling ended permanently and no further requests are accepted.
  IREE_IO_URING_REGISTRATION_STATE_CLOSED = 3,
} iree_io_uring_registration_state_t;

typedef struct iree_io_uring_registration_request_t
    iree_io_uring_registration_request_t;

// Serializes io_uring_register operations and dispatches them to the kernel's
// SINGLE_ISSUER owner task when required.
typedef struct iree_io_uring_registration_t {
  // Ring descriptor passed to io_uring_register.
  int ring_fd;

  // Protects ownership state and the pending request queue.
  iree_slim_mutex_t mutex;

  // Current kernel ownership state. Guarded by |mutex|.
  iree_io_uring_registration_state_t state;

  // Linux task ID recorded when a SINGLE_ISSUER ring becomes bound.
  // Guarded by |mutex| and meaningful only in BOUND state.
  int32_t owner_tid;

  // First queued registration request. Guarded by |mutex|.
  iree_io_uring_registration_request_t* pending_head;

  // Last queued registration request. Guarded by |mutex|.
  iree_io_uring_registration_request_t* pending_tail;

  // Non-zero when the pending queue may contain requests.
  iree_atomic_int32_t has_pending;

  // Callback used to wake the owner after a request is queued.
  // Guarded by |mutex|.
  iree_io_uring_registration_wake_callback_t wake_callback;
} iree_io_uring_registration_t;

// Initializes registration dispatch for a ring created with
// |actual_setup_flags|. The flags must be those accepted by io_uring_setup,
// not merely the requested flags before compatibility fallbacks.
void iree_io_uring_registration_initialize(
    int ring_fd, uint32_t actual_setup_flags,
    iree_io_uring_registration_t* out_registration);

// Permanently closes registration dispatch and deinitializes synchronization
// state. Completes any pending requests with -ESHUTDOWN.
void iree_io_uring_registration_deinitialize(
    iree_io_uring_registration_t* registration);

// Sets the callback used to wake a bound owner task. The callback must remain
// valid until registration dispatch is deinitialized.
void iree_io_uring_registration_set_wake_callback(
    iree_io_uring_registration_t* registration,
    iree_io_uring_registration_wake_callback_t callback);

// Enables an R_DISABLED ring and binds the calling task as its registration
// owner. Returns the raw syscall result or a negated errno value.
int iree_io_uring_registration_enable(
    iree_io_uring_registration_t* registration);

// Executes an io_uring_register operation synchronously.
//
// Operations execute directly before an R_DISABLED ring has an owner and on
// rings without SINGLE_ISSUER. After ownership is bound, non-owner callers
// queue stack-backed requests, wake the owner, and wait without a deadline.
// The argument memory must remain valid until this function returns.
//
// Returns the raw syscall result or a negated errno value.
int iree_io_uring_registration_execute(
    iree_io_uring_registration_t* registration, uint32_t opcode, void* arg,
    uint32_t argument_count);

// Pins |buffers| on the caller and installs them at |offset| in the existing
// fixed-buffer table. The caller owns these empty destination slots and keeps
// every backing range alive until unregistration and accepted I/O complete.
//
// When the caller is not the bound SINGLE_ISSUER owner, a temporary ring pins
// the pages before the owner attaches them with CLONE_BUFFERS. Only that
// attachment is dispatched to the owner. Source-ring teardown leaves the
// destination's references intact. No additional thread is created.
//
// Returns the installed count (possibly a prefix on the direct update path)
// or a negated errno. -EOPNOTSUPP means the kernel cannot attach existing pins;
// it never falls back to pinning on another task's progress loop.
int iree_io_uring_registration_update_buffers(
    iree_io_uring_registration_t* registration, uint32_t offset,
    const struct iovec* buffers, uint32_t count);

// Returns true when requests are waiting for the owner task.
static inline bool iree_io_uring_registration_has_pending(
    iree_io_uring_registration_t* registration) {
  return iree_atomic_load(&registration->has_pending,
                          iree_memory_order_acquire) != 0;
}

// Executes all pending requests on the owner task in FIFO order.
void iree_io_uring_registration_drain(
    iree_io_uring_registration_t* registration);

// Permanently retires the polling owner and completes queued requests with
// -ESHUTDOWN. Idempotent and callable from owner teardown paths.
void iree_io_uring_registration_end_polling(
    iree_io_uring_registration_t* registration);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_PLATFORM_IO_URING_URING_REGISTRATION_H_
