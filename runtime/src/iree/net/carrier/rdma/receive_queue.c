// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/receive_queue.h"

#include <errno.h>
#include <string.h>

#include "iree/net/carrier/rdma/sge.h"

#define IREE_NET_RDMA_RECEIVE_QUEUE_INVALID_INDEX UINT32_MAX

typedef uint8_t iree_net_rdma_receive_slot_flags_t;
enum iree_net_rdma_receive_slot_bits_e {
  IREE_NET_RDMA_RECEIVE_SLOT_POSTED = 1u << 0,
};

struct iree_net_rdma_receive_slot_t {
  // Lease held while the receive WQE is posted.
  iree_async_buffer_lease_t lease;

  // WR table identifier placed into the native receive WQE.
  uint64_t work_request_id;

  // Next slot index in the free list when not posted.
  uint32_t next_free;

  // Bitfield of iree_net_rdma_receive_slot_bits_e values.
  iree_net_rdma_receive_slot_flags_t flags;
};

static int iree_net_rdma_receive_queue_error_from_result(int result) {
  if (result >= 0) return result;
  if (errno != 0) return errno;
  return result == -1 ? EIO : -result;
}

static iree_status_t iree_net_rdma_receive_queue_status_from_result(
    const char* file, uint32_t line, int result, const char* call) {
  if (result == 0) return iree_ok_status();
  return iree_status_from_errno(
      file, line, iree_net_rdma_receive_queue_error_from_result(result), call);
}

static bool iree_net_rdma_receive_slot_is_posted(
    const iree_net_rdma_receive_slot_t* slot) {
  return iree_any_bit_set(slot->flags, IREE_NET_RDMA_RECEIVE_SLOT_POSTED);
}

static void iree_net_rdma_receive_queue_push_free_slot(
    iree_net_rdma_receive_queue_t* queue, uint32_t slot_index) {
  iree_net_rdma_receive_slot_t* slot = &queue->slots[slot_index];
  slot->work_request_id = 0;
  slot->next_free = queue->free_head;
  slot->flags = 0;
  queue->free_head = slot_index;
  ++queue->available_capacity;
}

IREE_API_EXPORT iree_status_t iree_net_rdma_receive_queue_initialize(
    iree_net_rdma_queue_pair_t* queue_pair,
    iree_net_rdma_work_request_table_t* work_request_table,
    iree_async_buffer_pool_t* buffer_pool, uint32_t capacity,
    iree_allocator_t host_allocator, iree_net_rdma_receive_queue_t* out_queue) {
  if (!out_queue) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_queue must not be NULL");
  }
  memset(out_queue, 0, sizeof(*out_queue));
  out_queue->host_allocator = host_allocator;
  out_queue->free_head = IREE_NET_RDMA_RECEIVE_QUEUE_INVALID_INDEX;

  iree_status_t status = iree_ok_status();
  if (!queue_pair || !iree_net_rdma_queue_pair_native_qp(queue_pair)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "queue_pair must contain a native QP");
  }
  if (iree_status_is_ok(status) && !work_request_table) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "work_request_table must not be NULL");
  }
  if (iree_status_is_ok(status) && !buffer_pool) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "buffer_pool must not be NULL");
  }
  if (iree_status_is_ok(status) && capacity == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "capacity is zero");
  }

  iree_net_rdma_receive_slot_t* slots = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, capacity,
                                         sizeof(*slots), (void**)&slots);
  }

  if (iree_status_is_ok(status)) {
    memset(slots, 0, capacity * sizeof(*slots));
    for (uint32_t index = 0; index < capacity; ++index) {
      slots[index].next_free = index + 1 < capacity
                                   ? index + 1
                                   : IREE_NET_RDMA_RECEIVE_QUEUE_INVALID_INDEX;
    }
    out_queue->queue_pair = queue_pair;
    out_queue->work_request_table = work_request_table;
    out_queue->buffer_pool = buffer_pool;
    out_queue->slots = slots;
    out_queue->capacity = capacity;
    out_queue->available_capacity = capacity;
    out_queue->free_head = 0;
  }

  return status;
}

IREE_API_EXPORT void iree_net_rdma_receive_queue_deinitialize(
    iree_net_rdma_receive_queue_t* queue) {
  if (!queue) return;

  if (queue->slots) {
    for (uint32_t index = 0; index < queue->capacity; ++index) {
      iree_net_rdma_receive_slot_t* slot = &queue->slots[index];
      if (iree_net_rdma_receive_slot_is_posted(slot)) {
        iree_net_rdma_work_request_completion_t completion;
        iree_status_t status = iree_net_rdma_work_request_table_complete(
            queue->work_request_table, slot->work_request_id, &completion);
        if (!iree_status_is_ok(status)) iree_status_abort(status);
        iree_async_buffer_lease_release(&slot->lease);
      }
    }
    iree_allocator_free(queue->host_allocator, queue->slots);
  }
  memset(queue, 0, sizeof(*queue));
  queue->free_head = IREE_NET_RDMA_RECEIVE_QUEUE_INVALID_INDEX;
}

IREE_API_EXPORT uint32_t iree_net_rdma_receive_queue_posted_count(
    const iree_net_rdma_receive_queue_t* queue) {
  return queue ? queue->posted_count : 0;
}

IREE_API_EXPORT uint32_t iree_net_rdma_receive_queue_available_capacity(
    const iree_net_rdma_receive_queue_t* queue) {
  return queue ? queue->available_capacity : 0;
}

IREE_API_EXPORT iree_status_t iree_net_rdma_receive_queue_replenish(
    iree_net_rdma_receive_queue_t* queue, uint32_t target_posted_count,
    uint32_t* out_posted_count) {
  if (out_posted_count) *out_posted_count = 0;

  iree_status_t status = iree_ok_status();
  if (!queue || !queue->slots) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "queue must be initialized");
  }
  if (iree_status_is_ok(status) && target_posted_count > queue->capacity) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target_posted_count %u exceeds capacity %u",
                              target_posted_count, queue->capacity);
  }

  uint32_t posted_count = 0;
  while (iree_status_is_ok(status) &&
         queue->posted_count < target_posted_count) {
    uint32_t slot_index = queue->free_head;
    if (slot_index == IREE_NET_RDMA_RECEIVE_QUEUE_INVALID_INDEX) {
      status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "RDMA receive queue is full");
      break;
    }

    iree_net_rdma_receive_slot_t* slot = &queue->slots[slot_index];
    iree_async_buffer_lease_t lease;
    memset(&lease, 0, sizeof(lease));

    status = iree_async_buffer_pool_acquire(queue->buffer_pool, &lease);
    if (iree_status_code(status) == IREE_STATUS_RESOURCE_EXHAUSTED) {
      (void)iree_status_consume_code(status);
      status = iree_ok_status();
      break;
    }

    struct ibv_sge scatter_gather_entry;
    if (iree_status_is_ok(status)) {
      status = iree_net_rdma_sge_from_span(lease.span, &scatter_gather_entry);
    }

    uint64_t work_request_id = 0;
    if (iree_status_is_ok(status)) {
      status = iree_net_rdma_work_request_table_acquire(
          queue->work_request_table, IREE_NET_RDMA_WORK_REQUEST_OPERATION_RECV,
          slot_index, /*byte_length=*/0, /*retained_buffer_lease=*/NULL,
          &work_request_id);
    }

    if (iree_status_is_ok(status)) {
      struct ibv_recv_wr work_request;
      memset(&work_request, 0, sizeof(work_request));
      work_request.wr_id = work_request_id;
      work_request.sg_list = &scatter_gather_entry;
      work_request.num_sge = 1;

      struct ibv_recv_wr* bad_work_request = NULL;
      errno = 0;
      int result =
          ibv_post_recv(iree_net_rdma_queue_pair_native_qp(queue->queue_pair),
                        &work_request, &bad_work_request);
      status = iree_net_rdma_receive_queue_status_from_result(
          __FILE__, __LINE__, result, "ibv_post_recv");
    }

    if (iree_status_is_ok(status)) {
      queue->free_head = slot->next_free;
      --queue->available_capacity;
      ++queue->posted_count;
      ++posted_count;

      slot->lease = lease;
      slot->work_request_id = work_request_id;
      slot->next_free = IREE_NET_RDMA_RECEIVE_QUEUE_INVALID_INDEX;
      slot->flags = IREE_NET_RDMA_RECEIVE_SLOT_POSTED;
    } else {
      if (work_request_id != 0) {
        iree_net_rdma_work_request_completion_t completion;
        status = iree_status_join(
            status,
            iree_net_rdma_work_request_table_complete(
                queue->work_request_table, work_request_id, &completion));
      }
      iree_async_buffer_lease_release(&lease);
    }
  }

  if (out_posted_count) *out_posted_count = posted_count;
  return status;
}

IREE_API_EXPORT iree_status_t iree_net_rdma_receive_queue_complete(
    iree_net_rdma_receive_queue_t* queue,
    iree_net_rdma_work_request_completion_t completion,
    iree_host_size_t byte_length, iree_async_buffer_lease_t* out_lease) {
  if (out_lease) memset(out_lease, 0, sizeof(*out_lease));

  iree_status_t status = iree_ok_status();
  if (!out_lease) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "out_lease must not be NULL");
  }
  if (iree_status_is_ok(status) && (!queue || !queue->slots)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "queue must be initialized");
  }
  if (iree_status_is_ok(status) &&
      completion.operation != IREE_NET_RDMA_WORK_REQUEST_OPERATION_RECV) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "completion operation %u is not RECV",
                              completion.operation);
  }
  if (iree_status_is_ok(status) && completion.user_data >= queue->capacity) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "receive slot index %" PRIu64
                              " is outside capacity %u",
                              completion.user_data, queue->capacity);
  }

  iree_async_buffer_lease_t lease;
  memset(&lease, 0, sizeof(lease));
  if (iree_status_is_ok(status)) {
    uint32_t slot_index = (uint32_t)completion.user_data;
    iree_net_rdma_receive_slot_t* slot = &queue->slots[slot_index];
    if (!iree_net_rdma_receive_slot_is_posted(slot)) {
      status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "receive slot %u is not posted", slot_index);
    } else {
      lease = slot->lease;
      memset(&slot->lease, 0, sizeof(slot->lease));
      --queue->posted_count;
      iree_net_rdma_receive_queue_push_free_slot(queue, slot_index);
    }
  }

  if (iree_status_is_ok(status) && byte_length > lease.span.length) {
    status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "received byte length %" PRIhsz
                              " exceeds buffer length %" PRIhsz,
                              byte_length, lease.span.length);
  }

  if (iree_status_is_ok(status)) {
    lease.span.length = byte_length;
    *out_lease = lease;
  } else {
    iree_async_buffer_lease_release(&lease);
  }
  return status;
}
