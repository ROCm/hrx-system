// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/completion_queue.h"

#include <fcntl.h>
#include <limits.h>

typedef enum iree_net_rdma_completion_queue_flag_bits_e {
  IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_PROGRESS = 1u << 0,
  IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_FAILED = 1u << 1,
  IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_STOPPING = 1u << 2,
  IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_STOPPED = 1u << 3,
} iree_net_rdma_completion_queue_flag_bits_t;
typedef uint32_t iree_net_rdma_completion_queue_flags_t;

struct iree_net_rdma_completion_queue_t {
  // Allocator for this object and its trailing completion scratch.
  iree_allocator_t host_allocator;
  // Retained native device/domain/library owner.
  iree_net_rdma_context_t* context;
  // Retained poll owner dispatching all service callbacks.
  iree_async_proactor_t* proactor;
  // Owned native CQ notification channel, or NULL for busy polling.
  struct ibv_comp_channel* channel;
  // Owned CQ shared by the connection's explicitly bounded native QPs.
  struct ibv_cq* handle;
  // Native fd monitor, joined before releasing the channel.
  iree_async_event_source_t* monitor;
  // Bounded service continuation, persistent only for busy polling.
  iree_async_progress_entry_t progress;
  // Poll-owner-only service state.
  iree_net_rdma_completion_queue_flags_t flags;
  // Native poll limit and allocated completion scratch count.
  uint32_t service_batch_size;
  // Borrowed connection callbacks, valid through deactivation.
  iree_net_rdma_completion_queue_callbacks_t callbacks;
  // Published before monitor retirement starts.
  iree_async_event_source_unregistered_callback_t deactivated_callback;
  // Bounded scratch reused on the poll owner without per-transfer allocation.
  struct ibv_wc completions[];
};

static void iree_net_rdma_completion_queue_fail(
    iree_net_rdma_completion_queue_t* queue, iree_status_t status) {
  queue->flags |= IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_FAILED;
  queue->callbacks.on_error(queue->callbacks.user_data, status);
}

// Channel events and CQ entries have independent ownership. Several native
// notifications can name this CQ even after its entries were already drained.
static iree_status_t iree_net_rdma_completion_queue_consume_notifications(
    iree_net_rdma_completion_queue_t* queue, uint32_t limit,
    uint32_t* out_count) {
  const iree_net_rdma_library_t* library =
      iree_net_rdma_context_library(queue->context);
  uint32_t count = 0;
  int error = 0;
  while (count < limit) {
    struct ibv_cq* handle = NULL;
    void* owner = NULL;
    if (library->ibv_get_cq_event(queue->channel, &handle, &owner)) {
      error = errno;
      break;
    }
    IREE_ASSERT(handle == queue->handle && owner == queue,
                "private notification channel received a foreign CQ");
    ++count;
  }
  if (count) {
    library->ibv_ack_cq_events(queue->handle, count);
  }
  *out_count = count;
  if (error && error != EAGAIN) {
    return iree_make_status(iree_status_code_from_errno(error),
                            "ibv_get_cq_event: %s", strerror(error));
  }
  return iree_ok_status();
}

static uint32_t iree_net_rdma_completion_queue_poll_batch(
    iree_net_rdma_completion_queue_t* queue) {
  int count = ibv_poll_cq(queue->handle, (int)queue->service_batch_size,
                          queue->completions);
  if (count < 0) {
    iree_net_rdma_completion_queue_fail(
        queue, iree_make_status(IREE_STATUS_INTERNAL, "ibv_poll_cq failed (%d)",
                                count));
    return 0;
  }
  if (count) {
    queue->callbacks.on_completions(queue->callbacks.user_data, count,
                                    queue->completions);
  }
  return (uint32_t)count;
}

static void iree_net_rdma_completion_queue_progress_removed(void* user_data) {
  iree_net_rdma_completion_queue_t* queue = user_data;
  queue->flags &= ~IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_PROGRESS;
}

static iree_status_t iree_net_rdma_completion_queue_progress(
    void* user_data, iree_host_size_t* out_completed_count) {
  iree_net_rdma_completion_queue_t* queue = user_data;
  uint32_t count = 0;
  if (!iree_any_bit_set(queue->flags,
                        IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_FAILED)) {
    count = iree_net_rdma_completion_queue_poll_batch(queue);
  }
  *out_completed_count = count;
  queue->progress.remove_requested =
      iree_any_bit_set(queue->flags,
                       IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_FAILED) ||
      (queue->channel && count < queue->service_batch_size);
  return iree_ok_status();
}

static void iree_net_rdma_completion_queue_notified(
    void* user_data, iree_async_event_source_t* source,
    iree_async_poll_events_t events) {
  (void)source;
  iree_net_rdma_completion_queue_t* queue = user_data;
  if (iree_any_bit_set(queue->flags,
                       IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_FAILED)) {
    return;
  }
  if (iree_async_poll_has_error(events)) {
    iree_net_rdma_completion_queue_fail(
        queue, iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "RDMA completion channel failed"));
    return;
  }
  uint32_t notification_count = 0;
  iree_status_t status = iree_net_rdma_completion_queue_consume_notifications(
      queue, queue->service_batch_size, &notification_count);
  if (iree_status_is_ok(status)) {
    int error = ibv_req_notify_cq(queue->handle, 0);
    if (error) {
      status = iree_make_status(iree_status_code_from_errno(error),
                                "ibv_req_notify_cq: %s", strerror(error));
    }
  }
  if (!iree_status_is_ok(status)) {
    iree_net_rdma_completion_queue_fail(queue, status);
    return;
  }
  if (iree_net_rdma_completion_queue_poll_batch(queue) ==
          queue->service_batch_size &&
      !iree_any_bit_set(queue->flags,
                        IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_PROGRESS)) {
    queue->flags |= IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_PROGRESS;
    iree_async_proactor_register_progress(queue->proactor, &queue->progress);
  }
}

static void iree_net_rdma_completion_queue_free(
    iree_net_rdma_completion_queue_t* queue) {
  const iree_net_rdma_library_t* library =
      iree_net_rdma_context_library(queue->context);
  if (queue->handle) {
    int error = library->ibv_destroy_cq(queue->handle);
    if (error) {
      iree_status_abort(iree_make_status(iree_status_code_from_errno(error),
                                         "ibv_destroy_cq: %s",
                                         strerror(error)));
    }
  }
  if (queue->channel) {
    int error = library->ibv_destroy_comp_channel(queue->channel);
    if (error) {
      iree_status_abort(iree_make_status(iree_status_code_from_errno(error),
                                         "ibv_destroy_comp_channel: %s",
                                         strerror(error)));
    }
  }
  iree_net_rdma_context_release(queue->context);
  iree_async_proactor_release(queue->proactor);
  iree_allocator_free(queue->host_allocator, queue);
}

iree_status_t iree_net_rdma_completion_queue_create(
    iree_net_rdma_context_t* context, iree_async_proactor_t* proactor,
    iree_net_rdma_completion_queue_options_t options,
    iree_net_rdma_completion_queue_callbacks_t callbacks,
    iree_allocator_t host_allocator,
    iree_net_rdma_completion_queue_t** out_queue) {
  *out_queue = NULL;
  if (!context || !proactor || !callbacks.on_completions ||
      !callbacks.on_error || !options.capacity || options.capacity > INT_MAX ||
      !options.service_batch_size ||
      options.service_batch_size > options.capacity ||
      options.completion_vector > INT_MAX ||
      options.mode > IREE_NET_RDMA_COMPLETION_QUEUE_MODE_BUSY_POLL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid RDMA completion queue configuration");
  }
  iree_host_size_t allocation_size = 0;
  if (!iree_host_size_checked_mul(options.service_batch_size,
                                  sizeof(struct ibv_wc), &allocation_size) ||
      !iree_host_size_checked_add(sizeof(iree_net_rdma_completion_queue_t),
                                  allocation_size, &allocation_size)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "RDMA completion scratch exceeds host size");
  }
  iree_net_rdma_completion_queue_t* queue = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, allocation_size, (void**)&queue));
  queue->host_allocator = host_allocator;
  queue->context = context;
  iree_net_rdma_context_retain(context);
  queue->proactor = proactor;
  iree_async_proactor_retain(proactor);
  queue->service_batch_size = options.service_batch_size;
  queue->callbacks = callbacks;
  queue->progress.fn = iree_net_rdma_completion_queue_progress;
  queue->progress.on_remove = iree_net_rdma_completion_queue_progress_removed;
  queue->progress.user_data = queue;
  const iree_net_rdma_library_t* library =
      iree_net_rdma_context_library(context);
  struct ibv_context* device = iree_net_rdma_context_device(context);
  iree_status_t status = iree_ok_status();
  if (options.mode == IREE_NET_RDMA_COMPLETION_QUEUE_MODE_READINESS) {
    queue->channel = library->ibv_create_comp_channel(device);
    if (!queue->channel) {
      int error = errno;
      status = iree_make_status(iree_status_code_from_errno(error),
                                "ibv_create_comp_channel: %s", strerror(error));
    }
  }
  if (iree_status_is_ok(status) && queue->channel) {
    int flags = fcntl(queue->channel->fd, F_GETFL);
    if (flags < 0 || fcntl(queue->channel->fd, F_SETFL, flags | O_NONBLOCK)) {
      int error = errno;
      status = iree_make_status(iree_status_code_from_errno(error),
                                "RDMA completion channel flags: %s",
                                strerror(error));
    }
  }
  if (iree_status_is_ok(status)) {
    queue->handle =
        library->ibv_create_cq(device, (int)options.capacity, queue,
                               queue->channel, (int)options.completion_vector);
    if (!queue->handle) {
      int error = errno;
      status = iree_make_status(iree_status_code_from_errno(error),
                                "ibv_create_cq: %s", strerror(error));
    }
  }
  if (iree_status_is_ok(status) && queue->channel) {
    int error = ibv_req_notify_cq(queue->handle, 0);
    if (error) {
      status = iree_make_status(iree_status_code_from_errno(error),
                                "ibv_req_notify_cq: %s", strerror(error));
    }
  }
  if (iree_status_is_ok(status) && queue->channel) {
    status = iree_async_proactor_register_event_source(
        proactor, iree_async_primitive_from_fd(queue->channel->fd),
        (iree_async_event_source_callback_t){
            .fn = iree_net_rdma_completion_queue_notified,
            .user_data = queue,
        },
        &queue->monitor);
  }
  if (iree_status_is_ok(status)) {
    if (!queue->channel) {
      queue->flags |= IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_PROGRESS;
      iree_async_proactor_register_progress(proactor, &queue->progress);
    }
    *out_queue = queue;
  } else {
    iree_net_rdma_completion_queue_free(queue);
  }
  return status;
}

struct ibv_cq* iree_net_rdma_completion_queue_handle(
    iree_net_rdma_completion_queue_t* queue) {
  return queue->handle;
}

static void iree_net_rdma_completion_queue_unregistered(void* user_data,
                                                        iree_status_t status) {
  iree_net_rdma_completion_queue_t* queue = user_data;
  if (!iree_status_is_ok(status)) {
    // The monitor still owns this service and its native channel. Publish the
    // cleanup failure through the established owner channel and withhold the
    // deactivation receipt so no containing ownership can be released.
    iree_net_rdma_completion_queue_fail(queue, status);
    return;
  }
  queue->monitor = NULL;
  queue->flags &= ~IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_STOPPING;
  queue->flags |= IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_STOPPED;
  iree_async_event_source_unregistered_callback_t callback =
      queue->deactivated_callback;
  if (callback.fn) {
    callback.fn(callback.user_data, iree_ok_status());
  }
}

void iree_net_rdma_completion_queue_deactivate(
    iree_net_rdma_completion_queue_t* queue,
    iree_async_event_source_unregistered_callback_t callback) {
  IREE_ASSERT(!iree_any_bit_set(
      queue->flags, IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_STOPPING |
                        IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_STOPPED));
  queue->flags |= IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_STOPPING;
  queue->deactivated_callback = callback;
  if (iree_any_bit_set(queue->flags,
                       IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_PROGRESS)) {
    iree_async_proactor_unregister_progress(queue->proactor, &queue->progress);
    queue->flags &= ~IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_PROGRESS;
  }
  if (queue->monitor) {
    iree_async_proactor_unregister_event_source(
        queue->proactor, queue->monitor,
        (iree_async_event_source_unregistered_callback_t){
            .fn = iree_net_rdma_completion_queue_unregistered,
            .user_data = queue,
        });
  } else {
    iree_net_rdma_completion_queue_unregistered(queue, iree_ok_status());
  }
}

void iree_net_rdma_completion_queue_destroy(
    iree_net_rdma_completion_queue_t* queue) {
  if (!queue) {
    return;
  }
  IREE_ASSERT(iree_any_bit_set(queue->flags,
                               IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_STOPPED));
  // Consumed notifications were acknowledged before their callbacks. Native
  // destruction discards unread notifications; it does not require reading a
  // channel that may itself have failed along with the device.
  iree_net_rdma_completion_queue_free(queue);
}
