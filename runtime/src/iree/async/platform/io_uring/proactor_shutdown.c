// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>
#include <unistd.h>

#include "iree/async/platform/io_uring/notification.h"
#include "iree/async/platform/io_uring/proactor.h"
#include "iree/async/platform/io_uring/relay.h"

// User operations and owner-bound resources are drained before the polling
// task exits. Remaining observer unregistrations retire on that same owner;
// a proactor with no borrowed observers needs no native progress to destroy.
static iree_status_t iree_async_proactor_io_uring_drain_observers(
    iree_async_proactor_io_uring_t* proactor) {
  iree_async_io_uring_event_source_unregister_all(proactor);
  iree_async_io_uring_unregister_all_relays(proactor);
  for (iree_async_event_source_t* source = proactor->event_sources; source;
       source = source->next) {
    if (iree_any_bit_set(source->flags,
                         IREE_ASYNC_IO_URING_EVENT_SOURCE_FLAG_RETAINED)) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "io_uring event source cleanup retained native-reachable state");
    }
  }
  iree_status_t status = iree_ok_status();
  while ((proactor->event_sources || proactor->relays) &&
         iree_status_is_ok(status)) {
    // Never-issued notification monitors can retire entirely in software.
    iree_async_io_uring_notification_drain_pending(proactor);
    if (!proactor->event_sources && !proactor->relays) {
      break;
    }
    status = iree_io_uring_ring_enable(&proactor->ring);
    if (iree_status_is_ok(status)) {
      status =
          iree_async_proactor_io_uring_submit_pending_event_monitors(proactor);
    }
    if (iree_status_is_ok(status)) {
      status = iree_async_proactor_io_uring_submit_cancel_requests(proactor);
    }
    if (iree_status_is_ok(status)) {
      status = iree_io_uring_ring_wait_cqe(&proactor->ring, /*min_complete=*/1,
                                           /*flush_pending=*/true,
                                           IREE_DURATION_INFINITE);
    }
    if (iree_status_is_ok(status)) {
      uint32_t tail =
          iree_atomic_load((iree_atomic_int32_t*)proactor->ring.cq_tail,
                           iree_memory_order_acquire);
      while (*proactor->ring.cq_head != tail) {
        const iree_io_uring_cqe_t* cqe =
            &proactor->ring
                 .cqes[*proactor->ring.cq_head & proactor->ring.cq_mask];
        iree_async_proactor_io_uring_process_cqe(proactor, cqe, &status);
        iree_io_uring_ring_cq_advance(&proactor->ring, 1);
      }
    }
  }
  return status;
}

void iree_async_proactor_io_uring_destroy(
    iree_async_proactor_t* base_proactor) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);
  iree_allocator_t allocator = proactor->base.allocator;

  // Stop signal callback admission before driving terminal native progress.
  if (proactor->signal.initialized) {
    for (int i = 0; i < IREE_ASYNC_SIGNAL_COUNT; ++i) {
      while (proactor->signal.subscriptions[i]) {
        iree_async_signal_subscription_t* subscription =
            proactor->signal.subscriptions[i];
        proactor->signal.subscriptions[i] = subscription->next;
        iree_allocator_free(allocator, subscription);
      }
    }
    iree_allocator_free(allocator, proactor->signal.event_source);
    proactor->signal.event_source = NULL;
    iree_async_linux_signal_deinitialize(&proactor->signal.linux_state);
    proactor->signal.initialized = false;
    iree_async_signal_release_ownership(&proactor->base);
  }

  // Ring close schedules asynchronous kernel teardown; it is not a native
  // retirement receipt. Borrowed observers join their actual completions first.
  // A native progress failure cannot authorize returning those resources.
  iree_status_t observer_status =
      iree_async_proactor_io_uring_drain_observers(proactor);
  if (!iree_status_is_ok(observer_status)) {
    // A failed native retirement may still reach ring and source storage.
    // Diagnose and retain the complete proactor allocation.
    iree_status_fprint(stderr, observer_status);
    iree_status_free(observer_status);
    IREE_TRACE_ZONE_END(z0);
    return;
  }
  iree_io_uring_ring_deinitialize(&proactor->ring);

  iree_async_message_pool_deinitialize(&proactor->message_pool);
  iree_atomic_slist_deinitialize(&proactor->pending_software_operations);
  iree_atomic_slist_deinitialize(&proactor->pending_notifications);
  iree_atomic_slist_deinitialize(&proactor->pending_semaphore_waits);
  iree_async_semaphore_wait_context_deinitialize(
      &proactor->semaphore_wait_context);
  iree_io_uring_sparse_table_free(proactor->buffer_table, allocator);
  if (proactor->wake_eventfd >= 0) {
    close(proactor->wake_eventfd);
  }
  iree_allocator_free(allocator, proactor);
  IREE_TRACE_ZONE_END(z0);
}
