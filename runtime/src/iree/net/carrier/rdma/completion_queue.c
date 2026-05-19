// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/completion_queue.h"

#include <errno.h>
#include <limits.h>
#include <string.h>

#include "iree/net/carrier/rdma/pollable_fd.h"

#define IREE_NET_RDMA_COMPLETION_QUEUE_DRAIN_BATCH_CAPACITY 32
#define IREE_NET_RDMA_COMPLETION_QUEUE_VALID_FLAGS \
  IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_DEFER_ACTIVATION

struct iree_net_rdma_completion_queue_t {
  // RDMA context retained by the completion queue.
  iree_net_rdma_context_t* context;

  // Borrowed libibverbs symbol table owned by context.
  const iree_net_libverbs_t* libverbs;

  // Proactor monitoring completion_channel->fd. Retained by the queue.
  iree_async_proactor_t* proactor;

  // Native ibverbs completion event channel owned by this wrapper.
  struct ibv_comp_channel* completion_channel;

  // Native ibverbs completion queue owned by this wrapper.
  struct ibv_cq* completion_queue;

  // Proactor event source registered for completion_channel->fd.
  iree_async_event_source_t* event_source;

  // Completion/error callback invoked from proactor poll.
  iree_net_rdma_completion_queue_callback_t callback;

  // Host allocator used for this queue allocation.
  iree_allocator_t host_allocator;
};

static int iree_net_rdma_completion_queue_error_from_result(int result) {
  if (result >= 0) return result;
  if (errno != 0) return errno;
  return result == -1 ? EIO : -result;
}

static iree_status_t iree_net_rdma_completion_queue_status_from_errno_required(
    const char* file, uint32_t line, int error, const char* call) {
  return iree_status_from_errno(file, line, error != 0 ? error : EIO, call);
}

static iree_status_t iree_net_rdma_completion_queue_status_from_verbs_result(
    const char* file, uint32_t line, int result, const char* call) {
  return iree_status_from_errno(
      file, line, iree_net_rdma_completion_queue_error_from_result(result),
      call);
}

static iree_status_t iree_net_rdma_completion_queue_rearm(
    iree_net_rdma_completion_queue_t* queue) {
  errno = 0;
  int result = ibv_req_notify_cq(queue->completion_queue, 0);
  return iree_net_rdma_completion_queue_status_from_verbs_result(
      __FILE__, __LINE__, result, "ibv_req_notify_cq");
}

static void iree_net_rdma_completion_queue_ack_events(
    iree_net_rdma_completion_queue_t* queue, unsigned int event_count) {
  if (event_count != 0) {
    queue->libverbs->ibv_ack_cq_events(queue->completion_queue, event_count);
  }
}

static iree_status_t iree_net_rdma_completion_queue_drain_channel(
    iree_net_rdma_completion_queue_t* queue) {
  iree_status_t status = iree_ok_status();
  unsigned int event_count = 0;

  while (iree_status_is_ok(status)) {
    struct ibv_cq* event_completion_queue = NULL;
    void* event_context = NULL;
    errno = 0;
    int result = queue->libverbs->ibv_get_cq_event(
        queue->completion_channel, &event_completion_queue, &event_context);
    if (result != 0) {
      int error = iree_net_rdma_completion_queue_error_from_result(result);
      if (error == EAGAIN || error == EWOULDBLOCK) break;
      status = iree_net_rdma_pollable_fd_status_from_errno(
          __FILE__, __LINE__, error, "ibv_get_cq_event");
      break;
    }

    if (event_completion_queue == queue->completion_queue &&
        event_context == queue) {
      if (event_count == UINT_MAX) {
        iree_net_rdma_completion_queue_ack_events(queue, event_count);
        event_count = 0;
      }
      ++event_count;
    } else {
      iree_net_rdma_completion_queue_ack_events(queue, event_count);
      event_count = 0;
      if (event_completion_queue) {
        queue->libverbs->ibv_ack_cq_events(event_completion_queue, 1);
      }
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "ibv_get_cq_event returned unexpected CQ event: cq=%p context=%p",
          event_completion_queue, event_context);
    }
  }

  iree_net_rdma_completion_queue_ack_events(queue, event_count);
  return status;
}

static iree_status_t iree_net_rdma_completion_queue_poll(
    iree_net_rdma_completion_queue_t* queue) {
  iree_status_t status = iree_ok_status();
  struct ibv_wc
      completions[IREE_NET_RDMA_COMPLETION_QUEUE_DRAIN_BATCH_CAPACITY];

  while (iree_status_is_ok(status)) {
    errno = 0;
    int completion_count = ibv_poll_cq(
        queue->completion_queue,
        IREE_NET_RDMA_COMPLETION_QUEUE_DRAIN_BATCH_CAPACITY, completions);
    if (completion_count < 0) {
      status = iree_net_rdma_completion_queue_status_from_verbs_result(
          __FILE__, __LINE__, completion_count, "ibv_poll_cq");
      break;
    } else if (completion_count == 0) {
      break;
    }

    queue->callback.fn(queue->callback.user_data, iree_ok_status(), completions,
                       (iree_host_size_t)completion_count);
    if (completion_count <
        IREE_NET_RDMA_COMPLETION_QUEUE_DRAIN_BATCH_CAPACITY) {
      break;
    }
  }

  return status;
}

IREE_API_EXPORT iree_status_t
iree_net_rdma_completion_queue_drain(iree_net_rdma_completion_queue_t* queue) {
  if (!queue) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue must not be NULL");
  }

  iree_status_t status = iree_net_rdma_completion_queue_drain_channel(queue);
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_completion_queue_rearm(queue);
  }
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_completion_queue_poll(queue);
  }
  return status;
}

static void iree_net_rdma_completion_queue_on_event_source(
    void* user_data, iree_async_event_source_t* source,
    iree_async_poll_events_t events) {
  (void)source;
  iree_net_rdma_completion_queue_t* queue =
      (iree_net_rdma_completion_queue_t*)user_data;

  iree_status_t status = iree_ok_status();
  if (events & IREE_ASYNC_POLL_EVENT_IN) {
    status = iree_net_rdma_completion_queue_drain(queue);
  }
  if (iree_status_is_ok(status) && iree_async_poll_has_error(events)) {
    status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "RDMA completion channel closed or failed");
  }

  if (!iree_status_is_ok(status)) {
    queue->callback.fn(queue->callback.user_data, status, NULL, 0);
  }
}

IREE_API_EXPORT iree_status_t iree_net_rdma_completion_queue_activate(
    iree_net_rdma_completion_queue_t* queue) {
  if (!queue) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "queue must not be NULL");
  }
  if (queue->event_source) return iree_ok_status();

  iree_async_event_source_callback_t event_callback = {
      iree_net_rdma_completion_queue_on_event_source,
      queue,
  };
  iree_status_t status = iree_async_proactor_register_event_source(
      queue->proactor,
      iree_async_primitive_from_fd(queue->completion_channel->fd),
      event_callback, &queue->event_source);
  if (iree_status_is_ok(status)) {
    status = iree_net_rdma_completion_queue_drain(queue);
  }
  if (!iree_status_is_ok(status) && queue->event_source) {
    iree_async_proactor_unregister_event_source(queue->proactor,
                                                queue->event_source);
    queue->event_source = NULL;
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_net_rdma_completion_queue_create(
    iree_net_rdma_context_t* context, iree_async_proactor_t* proactor,
    iree_net_rdma_completion_queue_options_t options,
    iree_net_rdma_completion_queue_callback_t callback,
    iree_allocator_t host_allocator,
    iree_net_rdma_completion_queue_t** out_queue) {
  IREE_ASSERT_ARGUMENT(out_queue);
  *out_queue = NULL;

  iree_status_t status = iree_ok_status();
  if (!context) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "context must not be NULL");
  } else if (!proactor) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "proactor must not be NULL");
  } else if (options.completion_capacity <= 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "completion_capacity must be positive");
  } else if (options.completion_vector < 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "completion_vector must be non-negative");
  } else if (options.flags & ~IREE_NET_RDMA_COMPLETION_QUEUE_VALID_FLAGS) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported completion queue flags: 0x%02X",
                              options.flags);
  } else if (!callback.fn) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "callback.fn must not be NULL");
  }

  iree_net_rdma_completion_queue_t* queue = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_allocator_malloc(host_allocator, sizeof(*queue), (void**)&queue);
  }

  if (iree_status_is_ok(status)) {
    memset(queue, 0, sizeof(*queue));
    queue->context = context;
    iree_net_rdma_context_retain(context);
    queue->libverbs = iree_net_rdma_context_libverbs(context);
    queue->proactor = proactor;
    iree_async_proactor_retain(proactor);
    queue->callback = callback;
    queue->host_allocator = host_allocator;
  }

  if (iree_status_is_ok(status)) {
    queue->completion_channel = queue->libverbs->ibv_create_comp_channel(
        iree_net_rdma_context_device(context));
    if (!queue->completion_channel) {
      status = iree_net_rdma_completion_queue_status_from_errno_required(
          __FILE__, __LINE__, errno, "ibv_create_comp_channel");
    }
  }

  if (iree_status_is_ok(status)) {
    status =
        iree_net_rdma_pollable_fd_initialize(queue->completion_channel->fd);
  }

  if (iree_status_is_ok(status)) {
    queue->completion_queue = queue->libverbs->ibv_create_cq(
        iree_net_rdma_context_device(context), options.completion_capacity,
        queue, queue->completion_channel, options.completion_vector);
    if (!queue->completion_queue) {
      status = iree_net_rdma_completion_queue_status_from_errno_required(
          __FILE__, __LINE__, errno, "ibv_create_cq");
    }
  }

  if (iree_status_is_ok(status)) {
    bool defer_activation = iree_all_bits_set(
        options.flags, IREE_NET_RDMA_COMPLETION_QUEUE_FLAG_DEFER_ACTIVATION);
    if (!defer_activation) {
      status = iree_net_rdma_completion_queue_activate(queue);
    }
  }

  if (iree_status_is_ok(status)) {
    *out_queue = queue;
  } else if (queue) {
    iree_net_rdma_completion_queue_release(queue);
  }
  return status;
}

IREE_API_EXPORT void iree_net_rdma_completion_queue_release(
    iree_net_rdma_completion_queue_t* queue) {
  if (!queue) return;

  if (queue->event_source) {
    iree_async_proactor_unregister_event_source(queue->proactor,
                                                queue->event_source);
  }
  if (queue->completion_queue) {
    int result = queue->libverbs->ibv_destroy_cq(queue->completion_queue);
    IREE_ASSERT(result == 0, "ibv_destroy_cq failed during destroy: %d",
                result);
  }
  if (queue->completion_channel) {
    int result =
        queue->libverbs->ibv_destroy_comp_channel(queue->completion_channel);
    IREE_ASSERT(result == 0,
                "ibv_destroy_comp_channel failed during destroy: %d", result);
  }
  iree_async_proactor_release(queue->proactor);
  iree_net_rdma_context_release(queue->context);

  iree_allocator_t host_allocator = queue->host_allocator;
  iree_allocator_free(host_allocator, queue);
}

IREE_API_EXPORT struct ibv_comp_channel*
iree_net_rdma_completion_queue_native_channel(
    const iree_net_rdma_completion_queue_t* queue) {
  return queue ? queue->completion_channel : NULL;
}

IREE_API_EXPORT struct ibv_cq* iree_net_rdma_completion_queue_native_cq(
    const iree_net_rdma_completion_queue_t* queue) {
  return queue ? queue->completion_queue : NULL;
}
