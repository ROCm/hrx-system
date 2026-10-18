// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/io_uring/uring_registration.h"

#include <errno.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "iree/base/threading/notification.h"

// A synchronous request owned by the waiting caller's stack. The argument
// memory has the same lifetime and remains valid until |completed| is set.
struct iree_io_uring_registration_request_t {
  // Next request in the registration FIFO.
  iree_io_uring_registration_request_t* next;

  // io_uring_register opcode to execute.
  uint32_t opcode;

  // Caller-owned opcode argument memory.
  void* arg;

  // Opcode-specific argument count or size.
  uint32_t argument_count;

  // Raw syscall result or negated errno published by the owner.
  int result;

  // Non-zero after |result| has been published.
  iree_atomic_int32_t completed;

  // Wakes the waiting caller after completion.
  iree_notification_t notification;
};

// Executes an io_uring_register syscall, retrying interrupted calls and
// encoding failures as negated errno values.
static int iree_io_uring_registration_syscall(int ring_fd, uint32_t opcode,
                                              void* arg,
                                              uint32_t argument_count) {
  long result = 0;
  do {
    result = syscall(IREE_IO_URING_SYSCALL_REGISTER, ring_fd, opcode, arg,
                     argument_count);
  } while (result < 0 && errno == EINTR);
  return result < 0 ? -errno : (int)result;
}

// Returns true after a request result has been published.
static bool iree_io_uring_registration_request_is_complete(void* user_data) {
  iree_io_uring_registration_request_t* request =
      (iree_io_uring_registration_request_t*)user_data;
  return iree_atomic_load(&request->completed, iree_memory_order_acquire) != 0;
}

// Publishes |result| and wakes the waiting request owner.
static void iree_io_uring_registration_request_complete(
    iree_io_uring_registration_request_t* request, int result) {
  request->result = result;
  iree_atomic_store(&request->completed, 1, iree_memory_order_release);
  iree_notification_post(&request->notification, IREE_ALL_WAITERS);
}

// Drains all requests while the caller holds |registration->mutex|.
static void iree_io_uring_registration_drain_locked(
    iree_io_uring_registration_t* registration) {
  while (registration->pending_head) {
    iree_io_uring_registration_request_t* request = registration->pending_head;
    registration->pending_head = request->next;
    if (!registration->pending_head) {
      registration->pending_tail = NULL;
    }

    int result = iree_io_uring_registration_syscall(
        registration->ring_fd, request->opcode, request->arg,
        request->argument_count);
    iree_io_uring_registration_request_complete(request, result);
  }
  iree_atomic_store(&registration->has_pending, 0, iree_memory_order_release);
}

void iree_io_uring_registration_initialize(
    int ring_fd, uint32_t actual_setup_flags,
    iree_io_uring_registration_t* out_registration) {
  memset(out_registration, 0, sizeof(*out_registration));
  out_registration->ring_fd = ring_fd;
  iree_slim_mutex_initialize(&out_registration->mutex);
  iree_atomic_store(&out_registration->has_pending, 0,
                    iree_memory_order_relaxed);
  out_registration->wake_callback =
      iree_io_uring_registration_wake_callback_null();

  if (!iree_any_bit_set(actual_setup_flags, IREE_IORING_SETUP_SINGLE_ISSUER)) {
    out_registration->state = IREE_IO_URING_REGISTRATION_STATE_DIRECT;
  } else if (iree_any_bit_set(actual_setup_flags,
                              IREE_IORING_SETUP_R_DISABLED)) {
    out_registration->state = IREE_IO_URING_REGISTRATION_STATE_UNBOUND;
  } else {
    out_registration->state = IREE_IO_URING_REGISTRATION_STATE_BOUND;
    out_registration->owner_tid = (int32_t)syscall(__NR_gettid);
  }
}

void iree_io_uring_registration_deinitialize(
    iree_io_uring_registration_t* registration) {
  iree_io_uring_registration_end_polling(registration);
  iree_slim_mutex_deinitialize(&registration->mutex);
  registration->ring_fd = -1;
}

void iree_io_uring_registration_set_wake_callback(
    iree_io_uring_registration_t* registration,
    iree_io_uring_registration_wake_callback_t callback) {
  iree_slim_mutex_lock(&registration->mutex);
  registration->wake_callback = callback;
  iree_slim_mutex_unlock(&registration->mutex);
}

int iree_io_uring_registration_enable(
    iree_io_uring_registration_t* registration) {
  iree_slim_mutex_lock(&registration->mutex);
  if (registration->state == IREE_IO_URING_REGISTRATION_STATE_CLOSED) {
    iree_slim_mutex_unlock(&registration->mutex);
    return -ESHUTDOWN;
  }
  if (registration->state != IREE_IO_URING_REGISTRATION_STATE_UNBOUND) {
    iree_slim_mutex_unlock(&registration->mutex);
    return 0;
  }

  int result = iree_io_uring_registration_syscall(
      registration->ring_fd, IREE_IORING_REGISTER_ENABLE_RINGS, NULL, 0);
  if (result >= 0) {
    registration->owner_tid = (int32_t)syscall(__NR_gettid);
    registration->state = IREE_IO_URING_REGISTRATION_STATE_BOUND;
  }
  iree_slim_mutex_unlock(&registration->mutex);
  return result;
}

int iree_io_uring_registration_execute(
    iree_io_uring_registration_t* registration, uint32_t opcode, void* arg,
    uint32_t argument_count) {
  iree_io_uring_registration_request_t request = {
      .next = NULL,
      .opcode = opcode,
      .arg = arg,
      .argument_count = argument_count,
      .result = 0,
  };
  iree_atomic_store(&request.completed, 0, iree_memory_order_relaxed);
  iree_notification_initialize(&request.notification);

  const int32_t current_tid = (int32_t)syscall(__NR_gettid);
  iree_slim_mutex_lock(&registration->mutex);
  int result = 0;
  switch (registration->state) {
    case IREE_IO_URING_REGISTRATION_STATE_DIRECT:
    case IREE_IO_URING_REGISTRATION_STATE_UNBOUND:
      result = iree_io_uring_registration_syscall(registration->ring_fd, opcode,
                                                  arg, argument_count);
      iree_slim_mutex_unlock(&registration->mutex);
      break;
    case IREE_IO_URING_REGISTRATION_STATE_BOUND:
      if (registration->owner_tid == current_tid) {
        // Preserve FIFO ordering when the owner enters registration directly
        // while requests from other tasks are already queued.
        iree_io_uring_registration_drain_locked(registration);
        result = iree_io_uring_registration_syscall(
            registration->ring_fd, opcode, arg, argument_count);
        iree_slim_mutex_unlock(&registration->mutex);
        break;
      }
      if (!registration->wake_callback.fn) {
        iree_slim_mutex_unlock(&registration->mutex);
        result = -ESHUTDOWN;
        break;
      }

      if (registration->pending_tail) {
        registration->pending_tail->next = &request;
      } else {
        registration->pending_head = &request;
      }
      registration->pending_tail = &request;
      iree_atomic_store(&registration->has_pending, 1,
                        iree_memory_order_release);
      iree_io_uring_registration_wake_callback_t wake_callback =
          registration->wake_callback;
      iree_slim_mutex_unlock(&registration->mutex);

      wake_callback.fn(wake_callback.user_data);
      iree_notification_await(&request.notification,
                              iree_io_uring_registration_request_is_complete,
                              &request, iree_infinite_timeout());

      // The completion predicate may become visible before notification_post
      // returns. Synchronize with the owner, which posts while holding this
      // mutex, before destroying the stack-backed notification and request.
      iree_slim_mutex_lock(&registration->mutex);
      result = request.result;
      iree_slim_mutex_unlock(&registration->mutex);
      break;
    case IREE_IO_URING_REGISTRATION_STATE_CLOSED:
      iree_slim_mutex_unlock(&registration->mutex);
      result = -ESHUTDOWN;
      break;
  }

  iree_notification_deinitialize(&request.notification);
  return result;
}

int iree_io_uring_registration_update_buffers(
    iree_io_uring_registration_t* registration, uint32_t offset,
    const struct iovec* buffers, uint32_t count) {
  const int32_t current_tid = (int32_t)syscall(__NR_gettid);
  iree_slim_mutex_lock(&registration->mutex);
  if (registration->state == IREE_IO_URING_REGISTRATION_STATE_CLOSED ||
      (registration->state == IREE_IO_URING_REGISTRATION_STATE_BOUND &&
       registration->owner_tid != current_tid &&
       !registration->wake_callback.fn)) {
    iree_slim_mutex_unlock(&registration->mutex);
    return -ESHUTDOWN;
  }
  if (registration->state != IREE_IO_URING_REGISTRATION_STATE_BOUND ||
      registration->owner_tid == current_tid) {
    // Keep owner binding serialized with this decision: an UNBOUND ring can
    // become owned by another task as soon as the mutex is released.
    iree_io_uring_registration_drain_locked(registration);
    iree_io_uring_rsrc_update2_t update = {
        .offset = offset,
        .data = (uint64_t)(uintptr_t)buffers,
        .nr = count,
    };
    int result = iree_io_uring_registration_syscall(
        registration->ring_fd, IREE_IORING_REGISTER_BUFFERS_UPDATE, &update,
        sizeof(update));
    iree_slim_mutex_unlock(&registration->mutex);
    return result;
  }
  iree_slim_mutex_unlock(&registration->mutex);

  // No SQ/CQ mapping or submissions are needed: this descriptor only owns
  // pins until the destination takes its references. It must not use
  // SINGLE_ISSUER, as the destination owner imports it from a different task.
  iree_io_uring_params_t parameters = {0};
  long source_fd = 0;
  do {
    source_fd = syscall(IREE_IO_URING_SYSCALL_SETUP, 1, &parameters);
  } while (source_fd < 0 && errno == EINTR);
  if (source_fd < 0) {
    return -errno;
  }

  int result = iree_io_uring_registration_syscall(
      (int)source_fd, IREE_IORING_REGISTER_BUFFERS, (void*)buffers, count);
  if (result >= 0) {
    iree_io_uring_clone_buffers_t clone = {
        .src_fd = (uint32_t)source_fd,
        .flags = IREE_IORING_REGISTER_DST_REPLACE,
        .dst_off = offset,
        .nr = count,
    };
    result = iree_io_uring_registration_execute(
        registration, IREE_IORING_REGISTER_CLONE_BUFFERS, &clone, 1);
    if (result >= 0) {
      result = (int)count;
    } else if (result == -EINVAL) {
      // The ranges and flags are internally established. Older kernels reject
      // the opcode or lack range replacement; neither permits owner-side pins.
      result = -EOPNOTSUPP;
    }
  }
  close((int)source_fd);
  return result;
}

void iree_io_uring_registration_drain(
    iree_io_uring_registration_t* registration) {
  if (!iree_io_uring_registration_has_pending(registration)) {
    return;
  }

  const int32_t current_tid = (int32_t)syscall(__NR_gettid);
  iree_slim_mutex_lock(&registration->mutex);
  if (registration->state == IREE_IO_URING_REGISTRATION_STATE_BOUND &&
      registration->owner_tid == current_tid) {
    iree_io_uring_registration_drain_locked(registration);
  }
  iree_slim_mutex_unlock(&registration->mutex);
}

void iree_io_uring_registration_end_polling(
    iree_io_uring_registration_t* registration) {
  iree_slim_mutex_lock(&registration->mutex);
  if (registration->state != IREE_IO_URING_REGISTRATION_STATE_CLOSED) {
    registration->state = IREE_IO_URING_REGISTRATION_STATE_CLOSED;
    while (registration->pending_head) {
      iree_io_uring_registration_request_t* request =
          registration->pending_head;
      registration->pending_head = request->next;
      iree_io_uring_registration_request_complete(request, -ESHUTDOWN);
    }
    registration->pending_tail = NULL;
    iree_atomic_store(&registration->has_pending, 0, iree_memory_order_release);
  }
  iree_slim_mutex_unlock(&registration->mutex);
}
