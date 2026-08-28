// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_OPERATIONS_MESSAGE_H_
#define IREE_ASYNC_OPERATIONS_MESSAGE_H_

#include "iree/async/operation.h"
#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_async_proactor_t iree_async_proactor_t;

//===----------------------------------------------------------------------===//
// Cross-proactor messaging
//===----------------------------------------------------------------------===//

// Flags for message operations.
enum iree_async_message_flag_bits_e {
  IREE_ASYNC_MESSAGE_FLAG_NONE = 0u,

  // Skips the source-side completion and transfers the payload to the target
  // as fire-and-forget work. The base completion_fn and pool must be NULL.
  // The MESSAGE must be a LINKED chain tail: a predecessor may link to it, but
  // the MESSAGE itself cannot have IREE_ASYNC_OPERATION_FLAG_LINKED set.
  IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION = 1u << 0,
};
typedef uint32_t iree_async_message_flags_t;

// Sends a message to another proactor.
//
// Cross-proactor messaging enables communication between proactors running on
// different threads. io_uring may post directly between rings; IOCP and POSIX
// use a bounded message pool and wake the target poll thread.
//
// Use cases:
//   - Worker pool coordination: main proactor dispatches work to worker
//     proactors, workers send completion notifications back
//   - LINK chain termination: RECV -> MESSAGE to notify a coordinator that
//     data arrived without returning to userspace between operations
//   - Load balancing: distribute incoming connections across worker proactors
//
// Native io_uring MSG_RING requires
// IREE_ASYNC_PROACTOR_CAPABILITY_PROACTOR_MESSAGING. The io_uring fallback,
// IOCP, and POSIX backends provide the same operation contract with a bounded
// software message pool.
//
// Implementation (io_uring):
//   Uses IORING_OP_MSG_RING with IORING_MSG_DATA. The message_data is encoded
//   into the target CQE's res field (lower 32 bits) and user_data field (for
//   callback dispatch). The target proactor's poll() receives the message as a
//   completion and invokes the registered message callback.
//
// Threading model:
//   Submission is thread-safe. The target proactor's message callback fires
//   from within its poll() on the polling thread. Heavy work should be
//   deferred to avoid stalling the target's completion dispatch.
//
// LINK chain usage:
//   MESSAGE operations can participate in LINK chains on the source side.
//   Example: RECV -> LINK -> MESSAGE sends a notification to another proactor
//   immediately after data arrives, with no userspace round-trip between them.
//
// Lifetime:
//   A normal operation remains proactor-owned until its source completion
//   callback fires. A directly submitted SKIP_SOURCE_COMPLETION operation may
//   be released after submit returns. A linked skip operation remains
//   proactor-owned until its predecessor callback begins; the backend consumes
//   all of its fields before that callback. The target proactor remains valid
//   until its message callback fires.
typedef struct iree_async_message_operation_t {
  iree_async_operation_t base;

  // Target proactor to receive the message.
  // Must be valid and not destroyed until the message is delivered.
  // The target receives a CQE that triggers its message callback.
  iree_async_proactor_t* target;

  // Arbitrary 64-bit payload delivered to the target.
  // On io_uring, the lower 32 bits are placed in cqe->res, and the full 64
  // bits are recoverable via the message callback's data parameter.
  uint64_t message_data;

  // Flags controlling message behavior (IREE_ASYNC_MESSAGE_FLAG_*).
  iree_async_message_flags_t message_flags;

  // Platform-specific storage. Backends may use this for handles, fds, or
  // other transient state needed between submission and completion.
  union {
    struct {
      // Ring fd of the target proactor (for MSG_RING).
      int target_ring_fd;
    } io_uring;
    struct {
      // Target proactor's wake eventfd for fallback path.
      int target_wake_fd;
      // Storage for eventfd WRITE value (must remain valid until CQE fires).
      uint64_t write_value;
    } fallback;
  } platform;
} iree_async_message_operation_t;

// Validates the backend-independent MESSAGE operation contract before any
// payload delivery or source-side submission occurs.
iree_status_t iree_async_message_operation_validate(
    const iree_async_message_operation_t* message);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_ASYNC_OPERATIONS_MESSAGE_H_
