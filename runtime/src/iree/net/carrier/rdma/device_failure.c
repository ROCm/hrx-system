// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/device_failure.h"

#include <fcntl.h>
#include <unistd.h>

#include "iree/async/operations/scheduling.h"

enum iree_net_rdma_device_failure_flag_bits_e {
  IREE_NET_RDMA_DEVICE_FAILURE_FLAG_PROGRESS = 1u << 0,
  IREE_NET_RDMA_DEVICE_FAILURE_FLAG_FAILED = 1u << 1,
  IREE_NET_RDMA_DEVICE_FAILURE_FLAG_STOPPING = 1u << 2,
  IREE_NET_RDMA_DEVICE_FAILURE_FLAG_STOPPED = 1u << 3,
};
typedef uint32_t iree_net_rdma_device_failure_flags_t;

struct iree_net_rdma_device_failure_t {
  // Allocator for this service.
  iree_allocator_t host_allocator;
  // Retained native device/event owner.
  iree_net_rdma_context_t* context;
  // Retained executor of owner callbacks and monitor retirement.
  iree_async_proactor_t* proactor;
  // Duplicated async descriptor permits multiple monitors on one proactor.
  int descriptor;
  // Native readiness monitor, joined before descriptor release.
  iree_async_event_source_t* monitor;
  // Bounded ready-only continuation, never an idle polling loop.
  iree_async_progress_entry_t progress;
  // Maximum native event read attempts in one ready visit.
  uint32_t service_batch_size;
  // Poll-owner-only service lifecycle.
  iree_net_rdma_device_failure_flags_t flags;
  // Context-local subscription; unsubscribe joins cross-thread publication.
  iree_net_rdma_device_event_subscription_t subscription;
  // One-shot publication from whichever proactor consumes the native event.
  struct {
    // Preallocated operation; submission publishes status to the owner thread.
    iree_async_nop_operation_t operation;
    // Owned error until the operation callback transfers it to the owner.
    iree_status_t status;
    // Published before submission; read after submit or unsubscribe joins.
    bool pending;
  } handoff;
  // Borrowed owner callback, invoked only by the owning proactor.
  void (*on_failure)(void* user_data, iree_status_t status);
  // Borrowed callback owner, alive through deactivation completion.
  void* user_data;
  // Native monitor plus already-published handoff joins during deactivation.
  uint32_t pending_joins;
  // Final ownership callback, allowed to destroy this service.
  iree_async_event_source_unregistered_callback_t deactivated_callback;
};

static void iree_net_rdma_device_failure_joined(void* user_data,
                                                iree_status_t status) {
  iree_net_rdma_device_failure_t* failure = user_data;
  if (!iree_status_is_ok(status)) {
    failure->on_failure(failure->user_data, status);
    return;
  }
  if (--failure->pending_joins != 0) {
    return;
  }
  failure->flags |= IREE_NET_RDMA_DEVICE_FAILURE_FLAG_STOPPED;
  if (failure->deactivated_callback.fn) {
    failure->deactivated_callback.fn(failure->deactivated_callback.user_data,
                                     iree_ok_status());
  }
}

static void iree_net_rdma_device_failure_deliver(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_net_rdma_device_failure_t* failure = user_data;
  failure->handoff.pending = false;
  status = iree_status_join(failure->handoff.status, status);
  failure->handoff.status = iree_ok_status();
  failure->flags |= IREE_NET_RDMA_DEVICE_FAILURE_FLAG_FAILED;
  failure->on_failure(failure->user_data, status);
  if (iree_any_bit_set(failure->flags,
                       IREE_NET_RDMA_DEVICE_FAILURE_FLAG_STOPPING)) {
    iree_net_rdma_device_failure_joined(failure, iree_ok_status());
  }
}

static void iree_net_rdma_device_failure_publish(void* user_data,
                                                 iree_status_t status) {
  iree_net_rdma_device_failure_t* failure = user_data;
  failure->handoff.status = status;
  failure->handoff.pending = true;
  IREE_CHECK_OK(iree_async_proactor_submit_one(
      failure->proactor, &failure->handoff.operation.base));
}

static void iree_net_rdma_device_failure_progress_removed(void* user_data) {
  iree_net_rdma_device_failure_t* failure = user_data;
  failure->flags &= ~IREE_NET_RDMA_DEVICE_FAILURE_FLAG_PROGRESS;
}

static iree_status_t iree_net_rdma_device_failure_progress(
    void* user_data, iree_host_size_t* out_completed_count) {
  iree_net_rdma_device_failure_t* failure = user_data;
  uint32_t count = 0;
  bool has_more = iree_net_rdma_device_events_poll(
      iree_net_rdma_context_device_events(failure->context),
      failure->service_batch_size, &count);
  *out_completed_count = count;
  failure->progress.remove_requested = !has_more;
  return iree_ok_status();
}

static void iree_net_rdma_device_failure_ready(
    void* user_data, iree_async_event_source_t* source,
    iree_async_poll_events_t ready) {
  (void)source;
  iree_net_rdma_device_failure_t* failure = user_data;
  if (iree_any_bit_set(failure->flags,
                       IREE_NET_RDMA_DEVICE_FAILURE_FLAG_FAILED)) {
    return;
  }
  iree_net_rdma_device_events_t* events =
      iree_net_rdma_context_device_events(failure->context);
  if (iree_async_poll_has_error(ready)) {
    iree_net_rdma_device_events_fail(
        events, iree_make_status(IREE_STATUS_UNAVAILABLE,
                                 "RDMA async event channel failed"));
    return;
  }
  uint32_t count = 0;
  if (iree_net_rdma_device_events_poll(events, failure->service_batch_size,
                                       &count) &&
      !iree_any_bit_set(failure->flags,
                        IREE_NET_RDMA_DEVICE_FAILURE_FLAG_PROGRESS)) {
    failure->flags |= IREE_NET_RDMA_DEVICE_FAILURE_FLAG_PROGRESS;
    iree_async_proactor_register_progress(failure->proactor,
                                          &failure->progress);
  }
}

static void iree_net_rdma_device_failure_free(
    iree_net_rdma_device_failure_t* failure) {
  if (failure->descriptor >= 0) {
    int result = close(failure->descriptor);
    IREE_ASSERT(result == 0, "owned async descriptor must close exactly once");
    (void)result;
  }
  iree_async_proactor_release(failure->proactor);
  iree_net_rdma_context_release(failure->context);
  iree_allocator_free(failure->host_allocator, failure);
}

iree_status_t iree_net_rdma_device_failure_create(
    iree_net_rdma_context_t* context, iree_async_proactor_t* proactor,
    struct ibv_cq* queue, uint32_t service_batch_size,
    void (*on_failure)(void*, iree_status_t), void* user_data,
    iree_allocator_t host_allocator,
    iree_net_rdma_device_failure_t** out_failure) {
  *out_failure = NULL;
  iree_net_rdma_device_failure_t* failure = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*failure),
                                             (void**)&failure));
  failure->host_allocator = host_allocator;
  failure->context = context;
  iree_net_rdma_context_retain(context);
  failure->proactor = proactor;
  iree_async_proactor_retain(proactor);
  failure->service_batch_size = service_batch_size;
  failure->on_failure = on_failure;
  failure->user_data = user_data;
  failure->progress.fn = iree_net_rdma_device_failure_progress;
  failure->progress.on_remove = iree_net_rdma_device_failure_progress_removed;
  failure->progress.user_data = failure;
  iree_async_operation_initialize(
      &failure->handoff.operation.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_rdma_device_failure_deliver,
      failure);
  failure->descriptor = fcntl(iree_net_rdma_context_device(context)->async_fd,
                              F_DUPFD_CLOEXEC, 0);
  iree_status_t status = iree_ok_status();
  if (failure->descriptor < 0) {
    int error = errno;
    status = iree_make_status(iree_status_code_from_errno(error),
                              "duplicating RDMA async descriptor: %s",
                              strerror(error));
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_register_event_source(
        proactor, iree_async_primitive_from_fd(failure->descriptor),
        (iree_async_event_source_callback_t){
            .fn = iree_net_rdma_device_failure_ready,
            .user_data = failure,
        },
        &failure->monitor);
  }
  if (iree_status_is_ok(status)) {
    // No fallible construction follows publication: the subscription may
    // immediately enqueue an error for a device that has already failed.
    iree_net_rdma_device_events_subscribe(
        iree_net_rdma_context_device_events(context), queue,
        iree_net_rdma_device_failure_publish, failure, &failure->subscription);
    *out_failure = failure;
  } else {
    iree_net_rdma_device_failure_free(failure);
  }
  return status;
}

void iree_net_rdma_device_failure_deactivate(
    iree_net_rdma_device_failure_t* failure,
    iree_async_event_source_unregistered_callback_t callback) {
  iree_net_rdma_device_events_unsubscribe(
      iree_net_rdma_context_device_events(failure->context),
      &failure->subscription);
  failure->flags |= IREE_NET_RDMA_DEVICE_FAILURE_FLAG_STOPPING;
  failure->deactivated_callback = callback;
  // Unsubscribe joined the publisher; the pending bit is now owner-thread-only.
  failure->pending_joins = 1 + failure->handoff.pending;
  if (iree_any_bit_set(failure->flags,
                       IREE_NET_RDMA_DEVICE_FAILURE_FLAG_PROGRESS)) {
    iree_async_proactor_unregister_progress(failure->proactor,
                                            &failure->progress);
    failure->flags &= ~IREE_NET_RDMA_DEVICE_FAILURE_FLAG_PROGRESS;
  }
  iree_async_proactor_unregister_event_source(
      failure->proactor, failure->monitor,
      (iree_async_event_source_unregistered_callback_t){
          .fn = iree_net_rdma_device_failure_joined,
          .user_data = failure,
      });
}

void iree_net_rdma_device_failure_destroy(
    iree_net_rdma_device_failure_t* failure) {
  if (!failure) {
    return;
  }
  IREE_ASSERT(iree_any_bit_set(failure->flags,
                               IREE_NET_RDMA_DEVICE_FAILURE_FLAG_STOPPED));
  iree_net_rdma_device_failure_free(failure);
}
