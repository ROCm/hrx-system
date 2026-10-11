// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/operation.h"

#include "iree/async/event.h"
#include "iree/async/file.h"
#include "iree/async/notification.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/socket.h"

static uint8_t iree_async_operation_acquire_socket_spans(
    iree_async_span_list_t spans, iree_async_socket_io_platform_t* platform,
    iree_async_region_t** out_regions) {
  IREE_ASSERT_LE(spans.count, IREE_ASYNC_SOCKET_SCATTER_GATHER_MAX_BUFFERS);
  for (iree_host_size_t i = 0; i < spans.count; ++i) {
    iree_async_span_t span = spans.values[i];
    platform->prepared.spans[i].offset = span.offset;
    platform->prepared.spans[i].length = span.length;
    out_regions[i] = span.region;
    iree_async_region_retain(span.region);
  }
  return (uint8_t)spans.count;
}

void iree_async_operation_acquire_resources(iree_async_operation_t* operation) {
  if (operation->resources_acquired) {
    return;
  }

  operation->acquired_span_count = 0;
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT: {
      iree_async_socket_accept_operation_t* accept =
          (iree_async_socket_accept_operation_t*)operation;
      iree_async_socket_retain(accept->listen_socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT: {
      iree_async_socket_connect_operation_t* connect_op =
          (iree_async_socket_connect_operation_t*)operation;
      iree_async_socket_retain(connect_op->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV: {
      iree_async_socket_recv_operation_t* recv =
          (iree_async_socket_recv_operation_t*)operation;
      iree_async_socket_retain(recv->socket);
      operation->acquired_span_count =
          iree_async_operation_acquire_socket_spans(
              recv->buffers, &recv->platform, recv->retained_buffer_regions);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL: {
      iree_async_socket_recv_pool_operation_t* recv_pool =
          (iree_async_socket_recv_pool_operation_t*)operation;
      iree_async_socket_retain(recv_pool->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND: {
      iree_async_socket_send_operation_t* send =
          (iree_async_socket_send_operation_t*)operation;
      iree_async_socket_retain(send->socket);
      operation->acquired_span_count =
          iree_async_operation_acquire_socket_spans(
              send->buffers, &send->platform, send->retained_buffer_regions);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM: {
      iree_async_socket_recvfrom_operation_t* recvfrom =
          (iree_async_socket_recvfrom_operation_t*)operation;
      iree_async_socket_retain(recvfrom->socket);
      operation->acquired_span_count =
          iree_async_operation_acquire_socket_spans(
              recvfrom->buffers, &recvfrom->platform,
              recvfrom->retained_buffer_regions);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO: {
      iree_async_socket_sendto_operation_t* sendto_op =
          (iree_async_socket_sendto_operation_t*)operation;
      iree_async_socket_retain(sendto_op->socket);
      operation->acquired_span_count =
          iree_async_operation_acquire_socket_spans(
              sendto_op->buffers, &sendto_op->platform,
              sendto_op->retained_buffer_regions);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT: {
      iree_async_event_wait_operation_t* event_wait =
          (iree_async_event_wait_operation_t*)operation;
      iree_async_event_retain(event_wait->event);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_HANDLE_POLL:
      // Reset at acceptance, including linked polls cancelled before native
      // admission. The primitive remains borrowed, with no resource to retain.
      ((iree_async_handle_poll_operation_t*)operation)->result_events =
          IREE_ASYNC_POLL_EVENT_NONE;
      return;
    case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT:
      // Establish the association before acceptance permits cancellation,
      // including waits deferred behind linked predecessors. Mark acquisition
      // so later continuation dispatch cannot reset a live association.
      ((iree_async_semaphore_wait_operation_t*)operation)->tracker = NULL;
      break;
    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT: {
      iree_async_notification_wait_operation_t* notification_wait =
          (iree_async_notification_wait_operation_t*)operation;
      iree_async_notification_retain(notification_wait->notification);
      if (iree_all_bits_set(notification_wait->wait_flags,
                            IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN)) {
        iree_async_notification_begin_observe(notification_wait->notification);
      }
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL: {
      iree_async_notification_signal_operation_t* notification_signal =
          (iree_async_notification_signal_operation_t*)operation;
      iree_async_notification_retain(notification_signal->notification);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_READ: {
      iree_async_file_read_operation_t* read_op =
          (iree_async_file_read_operation_t*)operation;
      iree_async_file_retain(read_op->file);
      iree_async_region_retain(read_op->buffer.region);
      operation->acquired_span_count = 1;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE: {
      iree_async_file_write_operation_t* write_op =
          (iree_async_file_write_operation_t*)operation;
      iree_async_file_retain(write_op->file);
      iree_async_region_retain(write_op->buffer.region);
      operation->acquired_span_count = 1;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE:
    case IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE:
      // Accepted close operations transfer their caller-owned reference.
      break;
    default:
      // This operation type has no proactor-owned resources.
      return;
  }
  operation->resources_acquired = true;
}

void iree_async_operation_list_acquire_resources(
    iree_async_operation_list_t operations) {
  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    iree_async_operation_acquire_resources(operations.values[i]);
  }
}

static uint8_t iree_async_operation_take_socket_regions(
    uint8_t retained_count, iree_async_region_t** retained_regions,
    iree_async_region_t** out_retained_regions) {
  for (uint8_t i = 0; i < retained_count; ++i) {
    out_retained_regions[i] = retained_regions[i];
    retained_regions[i] = NULL;
  }
  return retained_count;
}

uint8_t iree_async_operation_release_resources(
    iree_async_operation_t* operation,
    iree_async_region_t** out_retained_regions) {
  if (!operation->resources_acquired) {
    return 0;
  }
  operation->resources_acquired = false;
  const uint8_t retained_count = operation->acquired_span_count;
  operation->acquired_span_count = 0;
  uint8_t out_retained_count = 0;

  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT: {
      iree_async_socket_accept_operation_t* accept =
          (iree_async_socket_accept_operation_t*)operation;
      iree_async_socket_release(accept->listen_socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT: {
      iree_async_socket_connect_operation_t* connect_op =
          (iree_async_socket_connect_operation_t*)operation;
      iree_async_socket_release(connect_op->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV: {
      iree_async_socket_recv_operation_t* recv =
          (iree_async_socket_recv_operation_t*)operation;
      out_retained_count = iree_async_operation_take_socket_regions(
          retained_count, recv->retained_buffer_regions, out_retained_regions);
      iree_async_socket_release(recv->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL: {
      iree_async_socket_recv_pool_operation_t* recv_pool =
          (iree_async_socket_recv_pool_operation_t*)operation;
      iree_async_socket_release(recv_pool->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND: {
      iree_async_socket_send_operation_t* send =
          (iree_async_socket_send_operation_t*)operation;
      out_retained_count = iree_async_operation_take_socket_regions(
          retained_count, send->retained_buffer_regions, out_retained_regions);
      iree_async_socket_release(send->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM: {
      iree_async_socket_recvfrom_operation_t* recvfrom =
          (iree_async_socket_recvfrom_operation_t*)operation;
      out_retained_count = iree_async_operation_take_socket_regions(
          retained_count, recvfrom->retained_buffer_regions,
          out_retained_regions);
      iree_async_socket_release(recvfrom->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO: {
      iree_async_socket_sendto_operation_t* sendto_op =
          (iree_async_socket_sendto_operation_t*)operation;
      out_retained_count = iree_async_operation_take_socket_regions(
          retained_count, sendto_op->retained_buffer_regions,
          out_retained_regions);
      iree_async_socket_release(sendto_op->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE: {
      // Close consumes the caller's reference. This release IS the consumption,
      // with no prior retain to balance it.
      iree_async_socket_close_operation_t* close_op =
          (iree_async_socket_close_operation_t*)operation;
      iree_async_socket_release(close_op->socket);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT: {
      iree_async_event_wait_operation_t* event_wait =
          (iree_async_event_wait_operation_t*)operation;
      iree_async_event_release(event_wait->event);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT: {
      iree_async_notification_wait_operation_t* notification_wait =
          (iree_async_notification_wait_operation_t*)operation;
      if (iree_all_bits_set(notification_wait->wait_flags,
                            IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN)) {
        iree_async_notification_end_observe(notification_wait->notification);
      }
      iree_async_notification_release(notification_wait->notification);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL: {
      iree_async_notification_signal_operation_t* signal =
          (iree_async_notification_signal_operation_t*)operation;
      iree_async_notification_release(signal->notification);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_READ: {
      iree_async_file_read_operation_t* read_op =
          (iree_async_file_read_operation_t*)operation;
      if (retained_count) {
        out_retained_regions[0] = read_op->buffer.region;
        out_retained_count = 1;
      }
      iree_async_file_release(read_op->file);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE: {
      iree_async_file_write_operation_t* write_op =
          (iree_async_file_write_operation_t*)operation;
      if (retained_count) {
        out_retained_regions[0] = write_op->buffer.region;
        out_retained_count = 1;
      }
      iree_async_file_release(write_op->file);
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE: {
      // Close consumes the caller's reference. This release IS the consumption,
      // with no prior retain to balance it.
      iree_async_file_close_operation_t* file_close_op =
          (iree_async_file_close_operation_t*)operation;
      iree_async_file_release(file_close_op->file);
      break;
    }
    default:
      break;
  }
  return out_retained_count;
}
