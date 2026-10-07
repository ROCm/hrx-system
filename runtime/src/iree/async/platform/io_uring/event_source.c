// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "iree/async/platform/io_uring/event_source.h"

#include <errno.h>
#include <poll.h>
#include <string.h>

#include "iree/async/platform/io_uring/proactor.h"

static void iree_async_io_uring_event_source_destroy(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_event_source_t* source) {
  if (source->previous) {
    source->previous->next = source->next;
  } else {
    proactor->event_sources = source->next;
  }
  if (source->next) {
    source->next->previous = source->previous;
  }
  iree_async_event_source_unregistered_callback_t callback =
      source->unregistered_callback;
  iree_allocator_free(proactor->base.allocator, source);
  if (callback.fn) {
    callback.fn(callback.user_data, iree_ok_status());
  }
}

// Reports terminal cancellation failure without returning borrowed ownership.
// The retained flag keeps both the native source key and owning proactor live.
static void iree_async_io_uring_event_source_fail(
    iree_async_event_source_t* source, iree_status_t status) {
  source->flags |= IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_RETAINED;
  iree_async_event_source_unregistered_callback_t callback =
      source->unregistered_callback;
  source->unregistered_callback =
      iree_async_event_source_unregistered_callback_none();
  if (callback.fn) {
    callback.fn(callback.user_data, status);
  } else {
    iree_status_free(status);
  }
}

iree_status_t iree_async_io_uring_event_source_register(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t handle,
    iree_async_event_source_callback_t callback,
    iree_async_event_source_t** out_event_source) {
  *out_event_source = NULL;
  if (handle.type != IREE_ASYNC_PRIMITIVE_TYPE_FD || handle.value.fd < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "register_event_source requires a valid FD");
  }
  if (!callback.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "register_event_source requires a callback");
  }
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_async_event_source_t* source = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(base_proactor->allocator,
                                             sizeof(*source), (void**)&source));
  source->fd = handle.value.fd;
  source->flags = IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_ARM_PENDING;
  source->callback = callback;
  source->next = proactor->event_sources;
  if (source->next) {
    source->next->previous = source;
  }
  proactor->event_sources = source;
  // The poll owner submits the initial arm, preserving SINGLE_ISSUER and
  // admitting registration batches larger than the submission queue.
  iree_async_proactor_wake(base_proactor);
  *out_event_source = source;
  return iree_ok_status();
}

void iree_async_io_uring_event_source_unregister(
    iree_async_proactor_t* base_proactor, iree_async_event_source_t* source,
    iree_async_event_source_unregistered_callback_t callback) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  source->callback.fn = NULL;
  source->unregistered_callback = callback;
  source->flags &= ~IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_ARM_PENDING;
  if (!source->flags) {
    iree_async_io_uring_event_source_destroy(proactor, source);
  } else {
    source->flags |= IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_CANCEL_PENDING;
    iree_async_proactor_wake(base_proactor);
  }
}

bool iree_async_io_uring_event_source_submit_pending(
    iree_async_proactor_io_uring_t* proactor) {
  bool has_pending = false;
  iree_io_uring_ring_sq_lock(&proactor->ring);
  for (iree_async_event_source_t* source = proactor->event_sources; source;
       source = source->next) {
    if (!iree_any_bit_set(
            source->flags,
            IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_ARM_PENDING |
                IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_CANCEL_PENDING)) {
      continue;
    }
    iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
    if (!sqe) {
      has_pending = true;
      break;
    }
    memset(sqe, 0, sizeof(*sqe));
    uint64_t poll_key = iree_io_uring_internal_encode(
        IREE_IO_URING_TAG_EVENT_SOURCE, (uintptr_t)source);
    if (iree_any_bit_set(source->flags,
                         IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_ARM_PENDING)) {
      sqe->opcode = IREE_IORING_OP_POLL_ADD;
      sqe->fd = source->fd;
      sqe->poll32_events = POLLIN;
      sqe->len = IREE_IORING_POLL_ADD_MULTI;
      sqe->user_data = poll_key;
      source->flags &= ~IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_ARM_PENDING;
      source->flags |= IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_POLL_IN_FLIGHT;
    } else {
      // ASYNC_CANCEL marks a persistent poll cancelled even while readiness
      // task work owns it. POLL_REMOVE may instead return EALREADY and leave
      // the poll armed, with no terminal receipt to join.
      sqe->opcode = IREE_IORING_OP_ASYNC_CANCEL;
      sqe->fd = -1;
      sqe->addr = poll_key;
      sqe->user_data = iree_io_uring_internal_encode(
          IREE_IO_URING_TAG_EVENT_SOURCE_CANCEL, (uintptr_t)source);
      source->flags &= ~IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_CANCEL_PENDING;
      source->flags |= IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_CANCEL_IN_FLIGHT;
    }
  }
  iree_io_uring_ring_sq_unlock(&proactor->ring);
  return has_pending;
}

void iree_async_io_uring_event_source_dispatch(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe) {
  iree_async_event_source_t* source =
      (iree_async_event_source_t*)(uintptr_t)iree_io_uring_internal_payload(
          cqe->user_data);
  if (!(cqe->flags & IREE_IORING_CQE_F_MORE)) {
    // The final poll retires native monitoring, but a submitted cancellation
    // still borrows its identity until the independent cancellation receipt.
    source->flags &= ~(IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_POLL_IN_FLIGHT |
                       IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_CANCEL_PENDING);
  }
  if (!source->callback.fn) {
    if (!source->flags) {
      iree_async_io_uring_event_source_destroy(proactor, source);
    }
    return;
  }
  iree_async_poll_events_t events = IREE_ASYNC_POLL_EVENT_NONE;
  if (cqe->res < 0) {
    events = IREE_ASYNC_POLL_EVENT_ERR;
  } else {
    if (cqe->res & POLLIN) {
      events |= IREE_ASYNC_POLL_EVENT_IN;
    }
    if (cqe->res & POLLOUT) {
      events |= IREE_ASYNC_POLL_EVENT_OUT;
    }
    if (cqe->res & (POLLERR | POLLNVAL)) {
      events |= IREE_ASYNC_POLL_EVENT_ERR;
    }
    if (cqe->res & (POLLHUP | POLLRDHUP)) {
      events |= IREE_ASYNC_POLL_EVENT_HUP;
    }
  }
  // Explicit unregistration remains required after a terminal native event.
  source->callback.fn(source->callback.user_data, source, events);
}

iree_status_t iree_async_io_uring_event_source_complete_cancel(
    iree_async_proactor_io_uring_t* proactor, const iree_io_uring_cqe_t* cqe) {
  iree_async_event_source_t* source =
      (iree_async_event_source_t*)(uintptr_t)iree_io_uring_internal_payload(
          cqe->user_data);
  source->flags &= ~IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_CANCEL_IN_FLIGHT;
  iree_status_t status = iree_ok_status();
  if (cqe->res < 0 && cqe->res != -ENOENT && cqe->res != -EALREADY) {
    status = iree_make_status(iree_status_code_from_errno(-cqe->res),
                              "io_uring event source cancellation failed: %d",
                              -cqe->res);
    // The standard proactor thread stops on poll failure, so retry cannot be
    // an ownership proof. Publish a terminal failure and retain the native key
    // plus every owner it may still reach.
    iree_async_io_uring_event_source_fail(source, iree_status_clone(status));
  }
  if (!source->flags) {
    iree_async_io_uring_event_source_destroy(proactor, source);
  }
  return status;
}

void iree_async_io_uring_event_source_unregister_all(
    iree_async_proactor_io_uring_t* proactor) {
  iree_async_event_source_t* source = proactor->event_sources;
  while (source) {
    iree_async_event_source_t* next = source->next;
    // A cleared callback already owns an admitted terminal unregistration.
    if (source->callback.fn) {
      iree_async_io_uring_event_source_unregister(
          &proactor->base, source,
          iree_async_event_source_unregistered_callback_none());
    }
    source = next;
  }
}
