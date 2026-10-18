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

// Stack-owned completion for the explicitly synchronous adapter. The mutex
// joins notification_post before the waiting caller destroys its notification.
typedef struct iree_io_uring_registration_wait_t {
  // Serializes terminal notification delivery with stack teardown.
  iree_slim_mutex_t mutex;
  // Raw syscall result or negated errno published by terminal delivery.
  int result;
  // Non-zero after |result| has been published.
  iree_atomic_int32_t completed;
  // Wakes the waiting caller after completion.
  iree_notification_t notification;
} iree_io_uring_registration_wait_t;

// Executes an io_uring_register syscall, retrying interrupted calls and
// encoding failures as negated errno values.
static int iree_io_uring_registration_syscall(int ring_fd, uint32_t opcode,
                                              void* arg,
                                              uint32_t argument_count) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_TRACE_ZONE_APPEND_VALUE_I64(z0, opcode);
  long result = 0;
  do {
    result = syscall(IREE_IO_URING_SYSCALL_REGISTER, ring_fd, opcode, arg,
                     argument_count);
  } while (result < 0 && errno == EINTR);
  const int register_result = result < 0 ? -errno : (int)result;
  IREE_TRACE_ZONE_END(z0);
  return register_result;
}

// Returns true after a request result has been published.
static bool iree_io_uring_registration_wait_is_complete(void* user_data) {
  iree_io_uring_registration_wait_t* wait = user_data;
  return iree_atomic_load(&wait->completed, iree_memory_order_acquire) != 0;
}

// Publishes |result| and wakes the waiting request owner.
static void iree_io_uring_registration_wait_complete(void* user_data,
                                                     int result) {
  iree_io_uring_registration_wait_t* wait = user_data;
  iree_slim_mutex_lock(&wait->mutex);
  wait->result = result;
  iree_atomic_store(&wait->completed, 1, iree_memory_order_release);
  iree_notification_post(&wait->notification, IREE_ALL_WAITERS);
  iree_slim_mutex_unlock(&wait->mutex);
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

static void iree_io_uring_registration_submit_request(
    iree_io_uring_registration_t* registration,
    iree_io_uring_registration_request_t* request, bool synchronous) {
  const int32_t current_tid = (int32_t)syscall(__NR_gettid);
  iree_io_uring_registration_callback_t completion = request->completion;
  iree_io_uring_registration_wake_callback_t wake_callback =
      iree_io_uring_registration_wake_callback_null();
  bool pending = false;
  bool drain = false;
  iree_slim_mutex_lock(&registration->mutex);
  int result = 0;
  switch (registration->state) {
    case IREE_IO_URING_REGISTRATION_STATE_DIRECT:
    case IREE_IO_URING_REGISTRATION_STATE_UNBOUND:
      result = iree_io_uring_registration_syscall(registration->ring_fd,
                                                  request->opcode, request->arg,
                                                  request->argument_count);
      break;
    case IREE_IO_URING_REGISTRATION_STATE_BOUND: {
      const bool owner = registration->owner_tid == current_tid;
      if (!owner && !registration->wake_callback.fn) {
        result = -ESHUTDOWN;
        break;
      }
      // Owner calls join the FIFO too, preserving requests accepted before
      // this call. Completion can reenter or reuse its request after unlinking.
      request->next = NULL;
      if (registration->pending_tail) {
        registration->pending_tail->next = request;
      } else {
        registration->pending_head = request;
        if (!owner) {
          wake_callback = registration->wake_callback;
        }
      }
      registration->pending_tail = request;
      iree_atomic_store(&registration->has_pending, 1,
                        iree_memory_order_release);
      pending = true;
      // Asynchronous completion chains retain bounded stack depth. Explicit
      // synchronous calls from a callback must drive their own completion.
      drain = owner && (!registration->draining || synchronous);
      break;
    }
    case IREE_IO_URING_REGISTRATION_STATE_CLOSED:
      result = -ESHUTDOWN;
      break;
  }
  iree_slim_mutex_unlock(&registration->mutex);

  if (!pending) {
    completion.fn(completion.user_data, result);
  } else if (drain) {
    iree_io_uring_registration_drain(registration);
  } else if (wake_callback.fn) {
    wake_callback.fn(wake_callback.user_data);
  }
}

void iree_io_uring_registration_submit(
    iree_io_uring_registration_t* registration,
    iree_io_uring_registration_request_t* request) {
  iree_io_uring_registration_submit_request(registration, request,
                                            /*synchronous=*/false);
}

int iree_io_uring_registration_execute(
    iree_io_uring_registration_t* registration, uint32_t opcode, void* arg,
    uint32_t argument_count) {
  iree_io_uring_registration_wait_t wait = {0};
  iree_slim_mutex_initialize(&wait.mutex);
  iree_atomic_store(&wait.completed, 0, iree_memory_order_relaxed);
  iree_notification_initialize(&wait.notification);
  iree_io_uring_registration_request_t request = {
      .opcode = opcode,
      .arg = arg,
      .argument_count = argument_count,
      .completion = {iree_io_uring_registration_wait_complete, &wait},
  };
  iree_io_uring_registration_submit_request(registration, &request,
                                            /*synchronous=*/true);
  iree_notification_await(&wait.notification,
                          iree_io_uring_registration_wait_is_complete, &wait,
                          iree_infinite_timeout());
  iree_slim_mutex_lock(&wait.mutex);
  const int result = wait.result;
  iree_slim_mutex_unlock(&wait.mutex);
  iree_notification_deinitialize(&wait.notification);
  iree_slim_mutex_deinitialize(&wait.mutex);
  return result;
}

int iree_io_uring_registration_update_buffers(
    iree_io_uring_registration_t* registration, uint32_t offset,
    const struct iovec* buffers, uint32_t count) {
  const int32_t current_tid = (int32_t)syscall(__NR_gettid);
  iree_slim_mutex_lock(&registration->mutex);
  while (registration->state == IREE_IO_URING_REGISTRATION_STATE_BOUND &&
         registration->owner_tid == current_tid && registration->pending_head) {
    iree_slim_mutex_unlock(&registration->mutex);
    iree_io_uring_registration_drain(registration);
    iree_slim_mutex_lock(&registration->mutex);
  }
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
  IREE_TRACE_ZONE_BEGIN_NAMED(z_setup, "io_uring_setup");
  do {
    source_fd = syscall(IREE_IO_URING_SYSCALL_SETUP, 1, &parameters);
  } while (source_fd < 0 && errno == EINTR);
  const int setup_error = source_fd < 0 ? errno : 0;
  IREE_TRACE_ZONE_END(z_setup);
  if (source_fd < 0) {
    return -setup_error;
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
  IREE_TRACE_ZONE_BEGIN_NAMED(z_close, "io_uring_close");
  close((int)source_fd);
  IREE_TRACE_ZONE_END(z_close);
  return result;
}

void iree_io_uring_registration_drain(
    iree_io_uring_registration_t* registration) {
  if (!iree_io_uring_registration_has_pending(registration)) {
    return;
  }

  const int32_t current_tid = (int32_t)syscall(__NR_gettid);
  iree_slim_mutex_lock(&registration->mutex);
  if (registration->state != IREE_IO_URING_REGISTRATION_STATE_BOUND ||
      registration->owner_tid != current_tid) {
    iree_slim_mutex_unlock(&registration->mutex);
    return;
  }
  const bool was_draining = registration->draining;
  registration->draining = true;
  while (registration->state == IREE_IO_URING_REGISTRATION_STATE_BOUND &&
         registration->pending_head) {
    iree_io_uring_registration_request_t* request = registration->pending_head;
    registration->pending_head = request->next;
    if (!registration->pending_head) {
      registration->pending_tail = NULL;
      iree_atomic_store(&registration->has_pending, 0,
                        iree_memory_order_release);
    }
    iree_io_uring_registration_callback_t completion = request->completion;
    int result = iree_io_uring_registration_syscall(
        registration->ring_fd, request->opcode, request->arg,
        request->argument_count);
    iree_slim_mutex_unlock(&registration->mutex);
    completion.fn(completion.user_data, result);
    iree_slim_mutex_lock(&registration->mutex);
  }
  registration->draining = was_draining;
  iree_slim_mutex_unlock(&registration->mutex);
}

void iree_io_uring_registration_end_polling(
    iree_io_uring_registration_t* registration) {
  iree_slim_mutex_lock(&registration->mutex);
  registration->state = IREE_IO_URING_REGISTRATION_STATE_CLOSED;
  iree_io_uring_registration_request_t* request = registration->pending_head;
  registration->pending_head = NULL;
  registration->pending_tail = NULL;
  iree_atomic_store(&registration->has_pending, 0, iree_memory_order_release);
  iree_slim_mutex_unlock(&registration->mutex);
  while (request) {
    iree_io_uring_registration_request_t* next = request->next;
    iree_io_uring_registration_callback_t completion = request->completion;
    completion.fn(completion.user_data, -ESHUTDOWN);
    request = next;
  }
}
