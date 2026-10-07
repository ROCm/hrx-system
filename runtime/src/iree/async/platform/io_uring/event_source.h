// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_ASYNC_PLATFORM_IO_URING_EVENT_SOURCE_H_
#define IREE_ASYNC_PLATFORM_IO_URING_EVENT_SOURCE_H_

#include "iree/async/platform/io_uring/defs.h"
#include "iree/async/proactor.h"

#ifdef __cplusplus
extern "C" {
#endif

struct iree_async_proactor_io_uring_t;

enum iree_async_io_uring_event_source_flag_bits_e {
  // The poll owner has not yet prepared the initial multishot poll.
  IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_ARM_PENDING = 1u << 0,
  // A prepared or submitted poll owns the source until its final CQE.
  IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_POLL_IN_FLIGHT = 1u << 1,
  // Unregistration still needs an SQ slot for its cancellation.
  IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_CANCEL_PENDING = 1u << 2,
  // A prepared or submitted cancellation owns its key until its receipt.
  IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_CANCEL_IN_FLIGHT = 1u << 3,
  // Native cleanup failed and the source/proactor graph must be retained.
  IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_RETAINED = 1u << 4,
};
typedef uint32_t iree_async_io_uring_event_source_flags_t;

// Persistent poll owner. A cleared callback closes admission; native poll and
// cancellation ownership are independent and both must retire before release.
struct iree_async_event_source_t {
  // Next source in the owning proactor's intrusive list.
  struct iree_async_event_source_t* next;
  // Previous source in the owning proactor's intrusive list.
  struct iree_async_event_source_t* previous;
  // Borrowed descriptor retained through terminal unregistration completion.
  int fd;
  // Pending owner work and admitted native obligations.
  iree_async_io_uring_event_source_flags_t flags;
  // Readiness callback; cleared when terminal unregistration starts.
  iree_async_event_source_callback_t callback;
  // Final return of borrowed handle and callback-context ownership.
  iree_async_event_source_unregistered_callback_t unregistered_callback;
};

iree_status_t iree_async_io_uring_event_source_register(
    iree_async_proactor_t* base_proactor, iree_async_primitive_t handle,
    iree_async_event_source_callback_t callback,
    iree_async_event_source_t** out_event_source);

void iree_async_io_uring_event_source_unregister(
    iree_async_proactor_t* base_proactor, iree_async_event_source_t* source,
    iree_async_event_source_unregistered_callback_t callback);

// Queues pending arm/cancel work. Returns true if SQ pressure left work
// pending.
bool iree_async_io_uring_event_source_submit_pending(
    struct iree_async_proactor_io_uring_t* proactor);

// Dispatches a multishot poll result or terminal retirement.
void iree_async_io_uring_event_source_dispatch(
    struct iree_async_proactor_io_uring_t* proactor,
    const iree_io_uring_cqe_t* cqe);

// Joins a cancellation receipt and reports native failure through poll().
iree_status_t iree_async_io_uring_event_source_complete_cancel(
    struct iree_async_proactor_io_uring_t* proactor,
    const iree_io_uring_cqe_t* cqe);

// Stops remaining callback admission while preserving owned unregistrations.
// Native receipts must still be driven before the ring is closed.
void iree_async_io_uring_event_source_unregister_all(
    struct iree_async_proactor_io_uring_t* proactor);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_ASYNC_PLATFORM_IO_URING_EVENT_SOURCE_H_
