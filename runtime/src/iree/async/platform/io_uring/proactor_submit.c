// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Submit path for io_uring proactor.
//
// This module handles SQE preparation and submission for all operation types.
// Fill functions prepare SQEs with operation-specific parameters. The submit
// function handles batching, linked operations, and timer chain emulation.

// Enable GNU extensions for O_DIRECT (used in file open flag translation).
// Must be defined before any includes.
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <fcntl.h>
#include <poll.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include "iree/async/event.h"
#include "iree/async/file.h"
#include "iree/async/operation.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/futex.h"
#include "iree/async/operations/message.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/platform/io_uring/defs.h"
#include "iree/async/platform/io_uring/notification.h"
#include "iree/async/platform/io_uring/proactor.h"
#include "iree/async/platform/io_uring/proactor_validation.h"
#include "iree/async/platform/io_uring/socket.h"
#include "iree/async/semaphore.h"
#include "iree/async/util/continuation.h"
#include "iree/async/util/operation_completion.h"
#include "iree/async/util/semaphore_wait.h"

// See proactor.c for the rationale behind the TSAN annotation bridge.
#if defined(IREE_SANITIZER_THREAD)
#define IREE_IO_URING_TSAN_SUBMIT(operation) \
  iree_atomic_fetch_add(&(operation)->tsan_bridge, 1, iree_memory_order_release)
#else
#define IREE_IO_URING_TSAN_SUBMIT(operation) ((void)0)
#endif  // IREE_SANITIZER_THREAD

//===----------------------------------------------------------------------===//
// Submit
//===----------------------------------------------------------------------===//

// Fills two linked SQEs for an EVENT_WAIT operation.
// Uses POLL_ADD linked to READ to wait for the event's eventfd and auto-drain
// it in the kernel when it becomes readable. This eliminates the need for an
// explicit reset syscall on acquire.
//
// The linked pair:
//   SQE 1 (poll_sqe): POLL_ADD on eventfd, IOSQE_IO_LINK
//   SQE 2 (read_sqe): READ to drain the eventfd counter into drain_buffer
//
// On success, POLL_ADD produces a CQE (tagged internal, ignored by
// process_cqe) and then READ fires the user callback. On POLL_ADD failure,
// both POLL_ADD and READ produce CQEs; the POLL_ADD CQE is ignored and the
// READ -ECANCELED CQE fires the user callback with the error status.
static void iree_async_proactor_io_uring_fill_event_wait(
    iree_io_uring_sqe_t* poll_sqe, iree_io_uring_sqe_t* read_sqe,
    iree_async_operation_t* base_operation) {
  iree_async_event_wait_operation_t* event_wait =
      (iree_async_event_wait_operation_t*)base_operation;

  int fd = event_wait->event->native.wait_primitive.value.fd;

  // SQE 1: POLL_ADD with link to next SQE.
  // Tagged internal so process_cqe ignores it — the READ CQE handles
  // resource release and user callback for both success and failure paths.
  memset(poll_sqe, 0, sizeof(*poll_sqe));
  poll_sqe->opcode = IREE_IORING_OP_POLL_ADD;
  poll_sqe->flags = IREE_IOSQE_IO_LINK;
  poll_sqe->fd = fd;
  poll_sqe->poll32_events = POLLIN;
  poll_sqe->user_data = iree_io_uring_internal_encode(
      IREE_IO_URING_TAG_LINKED_POLL, (uintptr_t)base_operation);

  // SQE 2: READ to drain the eventfd counter.
  // The eventfd stores an 8-byte counter; reading resets it to 0.
  memset(read_sqe, 0, sizeof(*read_sqe));
  read_sqe->opcode = IREE_IORING_OP_READ;
  read_sqe->fd = fd;
  read_sqe->addr = (uint64_t)(uintptr_t)&event_wait->event->drain_buffer;
  read_sqe->len = sizeof(event_wait->event->drain_buffer);
  read_sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a HANDLE_POLL operation.
// Uses a single POLL_ADD SQE to detect readiness on the primitive's fd.
// Unlike EVENT_WAIT, there is no linked READ to drain the handle.
static void iree_async_proactor_io_uring_fill_handle_poll(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_handle_poll_operation_t* handle_poll =
      (iree_async_handle_poll_operation_t*)base_operation;
  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_POLL_ADD;
  sqe->fd = handle_poll->primitive.value.fd;
  sqe->poll32_events =
      (iree_any_bit_set(handle_poll->events, IREE_ASYNC_POLL_EVENT_IN) ? POLLIN
                                                                       : 0) |
      (iree_any_bit_set(handle_poll->events, IREE_ASYNC_POLL_EVENT_OUT)
           ? POLLOUT
           : 0);
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a TIMER operation.
// Uses absolute timeout when CAPABILITY_ABSOLUTE_TIMEOUT is set (kernel 5.4+),
// otherwise converts the deadline to a relative duration at submission time.
static void iree_async_proactor_io_uring_fill_timer(
    iree_async_proactor_io_uring_t* proactor, iree_io_uring_sqe_t* sqe,
    iree_async_operation_t* base_operation) {
  iree_async_timer_operation_t* timer =
      (iree_async_timer_operation_t*)base_operation;

  if (iree_any_bit_set(proactor->capabilities,
                       IREE_ASYNC_PROACTOR_CAPABILITY_ABSOLUTE_TIMEOUT)) {
    // Preferred: absolute timeout with CLOCK_MONOTONIC (no drift).
    // iree/async/ uses iree_time_now() which matches CLOCK_MONOTONIC.
    timer->platform.timespec.tv_sec = timer->deadline_ns / 1000000000LL;
    timer->platform.timespec.tv_nsec = timer->deadline_ns % 1000000000LL;
    sqe->timeout_flags = IREE_IORING_TIMEOUT_ABS;
  } else {
    // Fallback: relative timeout. There's a small window for drift between
    // computing this and the kernel processing the SQE.
    iree_duration_t remaining = timer->deadline_ns - iree_time_now();
    if (remaining < 0) {
      remaining = 0;
    }
    timer->platform.timespec.tv_sec = remaining / 1000000000LL;
    timer->platform.timespec.tv_nsec = remaining % 1000000000LL;
    sqe->timeout_flags = 0;
  }

  // Note: We do NOT set ETIME_SUCCESS here even though it's available on 5.16+.
  // Linked timers use userspace emulation (split at timer, submit continuation
  // on completion) rather than kernel LINK chains. Without kernel LINK, there's
  // no chain to break when -ETIME is returned - we convert -ETIME to OK in
  // cqe_to_status anyway.

  sqe->opcode = IREE_IORING_OP_TIMEOUT;
  sqe->fd = -1;
  sqe->addr = (uint64_t)(uintptr_t)&timer->platform.timespec;
  sqe->len = 1;  // One timespec structure. Event count is in sqe->off (0 = pure
                 // timer).
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a SOCKET_CONNECT operation.
// Uses IORING_OP_CONNECT to initiate an outbound connection.
static void iree_async_proactor_io_uring_fill_socket_connect(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_socket_connect_operation_t* connect =
      (iree_async_socket_connect_operation_t*)base_operation;

  sqe->opcode = IREE_IORING_OP_CONNECT;
  sqe->fd = connect->socket->primitive.value.fd;
  sqe->addr = (uint64_t)(uintptr_t)connect->address.storage;
  sqe->off = connect->address.length;  // connect uses off for addr_len.
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a SOCKET_ACCEPT operation.
// Uses IORING_OP_ACCEPT to accept an incoming connection on a listening socket.
static void iree_async_proactor_io_uring_fill_socket_accept(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_socket_accept_operation_t* accept =
      (iree_async_socket_accept_operation_t*)base_operation;

  // Clear output fields.
  accept->accepted_socket = NULL;
  accept->peer_address.length = sizeof(accept->peer_address.storage);

  sqe->opcode = IREE_IORING_OP_ACCEPT;
  sqe->fd = accept->listen_socket->primitive.value.fd;
  sqe->addr = (uint64_t)(uintptr_t)accept->peer_address.storage;
  sqe->off = (uint64_t)(uintptr_t)&accept->peer_address.length;
  sqe->accept_flags = SOCK_NONBLOCK | SOCK_CLOEXEC;
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;

  // Enable multishot mode if requested (kernel 5.19+).
  // In multishot mode, a single SQE produces multiple CQEs - one for each
  // accepted connection - with CQE_F_MORE set on all but the final CQE.
  if (base_operation->flags & IREE_ASYNC_OPERATION_FLAG_MULTISHOT) {
    sqe->ioprio = IREE_IORING_ACCEPT_MULTISHOT;
  }
}

// Converts accepted socket span snapshots to native iovecs in place.
static void iree_async_proactor_io_uring_materialize_iovecs(
    iree_async_socket_io_platform_t* platform,
    iree_async_region_t* const* retained_regions, uint8_t span_count) {
  iree_async_socket_prepared_span_t
      prepared_spans[IREE_ASYNC_SOCKET_SCATTER_GATHER_MAX_BUFFERS];
  memcpy(prepared_spans, platform->prepared.spans,
         span_count * sizeof(prepared_spans[0]));
  struct iovec* iovecs = (struct iovec*)platform->posix.iovecs;
  for (uint8_t i = 0; i < span_count; ++i) {
    iree_async_span_t span = iree_async_socket_prepared_span_resolve(
        prepared_spans[i], retained_regions[i]);
    iovecs[i].iov_base = iree_async_span_ptr(span);
    iovecs[i].iov_len = span.length;
  }
}

// Fills an SQE for a SOCKET_RECV operation.
// Uses IORING_OP_RECV for single-buffer receives and IORING_OP_RECVMSG for
// scatter-gather receives.
static void iree_async_proactor_io_uring_fill_socket_recv(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_socket_recv_operation_t* recv =
      (iree_async_socket_recv_operation_t*)base_operation;

  // Clear output fields.
  recv->bytes_received = 0;

  sqe->fd = recv->socket->primitive.value.fd;
  sqe->msg_flags = 0;
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;

  const uint8_t span_count = base_operation->acquired_span_count;
  if (span_count == 1) {
    // Single buffer: use simple RECV for efficiency.
    iree_async_span_t first_buffer = iree_async_socket_prepared_span_resolve(
        recv->platform.prepared.spans[0], recv->retained_buffer_regions[0]);
    sqe->opcode = IREE_IORING_OP_RECV;
    sqe->addr = (uint64_t)(uintptr_t)iree_async_span_ptr(first_buffer);
    sqe->len = (uint32_t)first_buffer.length;
  } else {
    // Multiple buffers: use RECVMSG with scatter-gather.
    // Convert spans to iovecs. The iovec array is stored in the operation's
    // platform storage to ensure it lives until completion.
    iree_async_proactor_io_uring_materialize_iovecs(
        &recv->platform, recv->retained_buffer_regions, span_count);
    struct iovec* iovecs = (struct iovec*)recv->platform.posix.iovecs;

    struct msghdr* msg = (struct msghdr*)recv->platform.posix.msg_header;
    memset(msg, 0, sizeof(*msg));
    msg->msg_iov = iovecs;
    msg->msg_iovlen = span_count;

    sqe->opcode = IREE_IORING_OP_RECVMSG;
    sqe->addr = (uint64_t)(uintptr_t)msg;
    sqe->len = 1;  // Number of messages (always 1 for RECVMSG).
  }

  // Enable multishot mode if requested (kernel 5.19+).
  // In multishot mode, a single SQE produces multiple CQEs - one for each
  // received message - with CQE_F_MORE set on all but the final CQE.
  // NOTE: Multishot recv with a fixed buffer reuses the same buffer for each
  // completion. The caller must process data before the next completion or
  // risk data being overwritten. For zero-copy multishot recv, use RECV_POOL
  // with a provided buffer ring (kernel 6.0+).
  if (base_operation->flags & IREE_ASYNC_OPERATION_FLAG_MULTISHOT) {
    sqe->ioprio |= IREE_IORING_RECV_MULTISHOT;
  }
}

// Fills an SQE for a SOCKET_RECV_POOL operation.
// Uses kernel buffer selection when available. Otherwise the batch reservation
// owns a userspace pool lease until this one-shot RECV completes.
static void iree_async_proactor_io_uring_fill_socket_recv_pool(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_socket_recv_pool_operation_t* recv_pool =
      (iree_async_socket_recv_pool_operation_t*)base_operation;

  iree_async_region_t* region = iree_async_buffer_pool_region(recv_pool->pool);

  sqe->opcode = IREE_IORING_OP_RECV;
  sqe->fd = recv_pool->socket->primitive.value.fd;
  sqe->msg_flags = 0;
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
  sqe->len = (uint32_t)region->buffer_size;
  if (region->handles.iouring.buffer_group_id < 0) {
    sqe->addr = (uint64_t)(uintptr_t)iree_async_span_ptr(recv_pool->lease.span);
    return;
  }
  sqe->addr = 0;  // Kernel provides buffer from the ring.
  sqe->flags |= IREE_IOSQE_BUFFER_SELECT;
  sqe->buf_group = (uint16_t)region->handles.iouring.buffer_group_id;

  // Enable multishot mode if requested (kernel 5.19+).
  // Multishot recv requires len=0; the kernel gets buffer sizes from the ring.
  // Single-shot can use buffer_size as a cap on receive length.
  if (iree_any_bit_set(base_operation->flags,
                       IREE_ASYNC_OPERATION_FLAG_MULTISHOT)) {
    sqe->ioprio |= IREE_IORING_RECV_MULTISHOT;
    sqe->len = 0;
  }
}

// Result of checking if a span can use fixed-buffer zero-copy send.
typedef enum {
  // Span is not eligible for fixed-buffer path. Use standard zero-copy.
  // This is NOT an error - just means we fall back to page-pinning per send.
  IREE_ASYNC_FIXED_BUFFER_INELIGIBLE = 0,
  // Span is eligible. out_buffer_index contains the kernel buffer table index.
  IREE_ASYNC_FIXED_BUFFER_ELIGIBLE = 1,
} iree_async_fixed_buffer_eligibility_t;

// Queries whether a validated span can use fixed-buffer zero-copy send.
static iree_async_fixed_buffer_eligibility_t
iree_async_span_query_fixed_buffer_send(
    iree_async_proactor_io_uring_t* proactor, iree_async_span_t span,
    uint16_t* out_buffer_index) {
  *out_buffer_index = 0;

  if (!span.region) {
    return IREE_ASYNC_FIXED_BUFFER_INELIGIBLE;
  }
  iree_async_region_t* region = span.region;
  IREE_ASSERT(region->proactor == &proactor->base);

  if (region->type != IREE_ASYNC_REGION_TYPE_IOURING) {
    return IREE_ASYNC_FIXED_BUFFER_INELIGIBLE;
  }

  uint32_t buffer_count = region->buffer_count;
  if (buffer_count == 0) {
    return IREE_ASYNC_FIXED_BUFFER_INELIGIBLE;
  }

  iree_host_size_t buffer_size = region->buffer_size;
  IREE_ASSERT_GT(buffer_size, 0);
  if (IREE_UNLIKELY(buffer_size == 0)) {
    return IREE_ASYNC_FIXED_BUFFER_INELIGIBLE;
  }

  if (!(region->access_flags & IREE_ASYNC_BUFFER_ACCESS_FLAG_READ)) {
    return IREE_ASYNC_FIXED_BUFFER_INELIGIBLE;
  }

  uint64_t buffer_index_offset = span.offset / buffer_size;
  if (buffer_index_offset >= buffer_count) {
    return IREE_ASYNC_FIXED_BUFFER_INELIGIBLE;
  }

  iree_host_size_t offset_in_buffer = span.offset % buffer_size;
  if (offset_in_buffer + span.length > buffer_size) {
    return IREE_ASYNC_FIXED_BUFFER_INELIGIBLE;
  }

  int32_t base_buffer_index = region->handles.iouring.base_buffer_index;
  if (base_buffer_index < 0) {
    return IREE_ASYNC_FIXED_BUFFER_INELIGIBLE;
  }

  uint64_t final_index = (uint64_t)base_buffer_index + buffer_index_offset;
  IREE_ASSERT_LE(final_index, UINT16_MAX);
  if (IREE_UNLIKELY(final_index > UINT16_MAX)) {
    return IREE_ASYNC_FIXED_BUFFER_INELIGIBLE;
  }

  *out_buffer_index = (uint16_t)final_index;
  return IREE_ASYNC_FIXED_BUFFER_ELIGIBLE;
}

// Fills an SQE for a SOCKET_SEND operation.
// Uses IORING_OP_SEND[_ZC] for single-buffer sends, IORING_OP_SENDMSG[_ZC] for
// scatter-gather (multiple buffers). Zero-copy variants are used when the
// socket has IREE_ASYNC_SOCKET_OPTION_ZERO_COPY AND the capability is
// available, unless the send suppresses the hint with NO_ZERO_COPY. Kernels
// without ZC support use regular SEND/SENDMSG.
static void iree_async_proactor_io_uring_fill_socket_send(
    iree_async_proactor_io_uring_t* proactor, iree_io_uring_sqe_t* sqe,
    iree_async_operation_t* base_operation) {
  iree_async_socket_send_operation_t* send =
      (iree_async_socket_send_operation_t*)base_operation;

  bool zero_copy_requested =
      iree_any_bit_set(send->socket->flags, IREE_ASYNC_SOCKET_FLAG_ZERO_COPY) &&
      !iree_any_bit_set(send->send_flags,
                        IREE_ASYNC_SOCKET_SEND_FLAG_NO_ZERO_COPY);
  bool zero_copy_available = iree_any_bit_set(
      proactor->capabilities, IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND);
  bool use_zero_copy = zero_copy_requested && zero_copy_available;

  iree_async_fixed_buffer_eligibility_t eligibility =
      IREE_ASYNC_FIXED_BUFFER_INELIGIBLE;
  uint16_t fixed_buffer_index = 0;
  const uint8_t span_count = base_operation->acquired_span_count;
  if (use_zero_copy && span_count == 1) {
    iree_async_span_t first_buffer = iree_async_socket_prepared_span_resolve(
        send->platform.prepared.spans[0], send->retained_buffer_regions[0]);
    eligibility = iree_async_span_query_fixed_buffer_send(
        proactor, first_buffer, &fixed_buffer_index);
  }

  // Clear output fields.
  send->bytes_sent = 0;

  sqe->fd = send->socket->primitive.value.fd;
  sqe->msg_flags = 0;
  if (iree_any_bit_set(send->send_flags, IREE_ASYNC_SOCKET_SEND_FLAG_MORE)) {
    sqe->msg_flags |= IREE_MSG_MORE;
  }
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;

  if (span_count == 1) {
    // Single buffer: use simple SEND[_ZC] for efficiency.
    iree_async_span_t first_buffer = iree_async_socket_prepared_span_resolve(
        send->platform.prepared.spans[0], send->retained_buffer_regions[0]);

    sqe->opcode = use_zero_copy ? IREE_IORING_OP_SEND_ZC : IREE_IORING_OP_SEND;

    // CRITICAL: sqe->addr is ALWAYS the full pointer, even with FIXED_BUF.
    // The kernel validates that addr falls within the registered buffer.
    sqe->addr = (uint64_t)(uintptr_t)iree_async_span_ptr(first_buffer);
    sqe->len = (uint32_t)first_buffer.length;

    if (use_zero_copy) {
      // Always request usage reporting so we can tell callers if ZC succeeded.
      sqe->ioprio |= IREE_IORING_SEND_ZC_REPORT_USAGE;
      if (eligibility == IREE_ASYNC_FIXED_BUFFER_ELIGIBLE) {
        sqe->ioprio |= IREE_IORING_RECVSEND_FIXED_BUF;
        sqe->buf_index = fixed_buffer_index;
      }
    }
  } else {
    // Multiple buffers: use SENDMSG[_ZC] with scatter-gather.
    // Convert spans to iovecs (spans have region+offset+length, iovecs have
    // base+length). The iovec array is stored in the operation's platform
    // storage to ensure it lives until completion.
    iree_async_proactor_io_uring_materialize_iovecs(
        &send->platform, send->retained_buffer_regions, span_count);
    struct iovec* iovecs = (struct iovec*)send->platform.posix.iovecs;

    struct msghdr* msg = (struct msghdr*)send->platform.posix.msg_header;
    memset(msg, 0, sizeof(*msg));
    msg->msg_iov = iovecs;
    msg->msg_iovlen = span_count;

    sqe->opcode =
        use_zero_copy ? IREE_IORING_OP_SENDMSG_ZC : IREE_IORING_OP_SENDMSG;
    sqe->addr = (uint64_t)(uintptr_t)msg;
    sqe->len = 1;  // Number of messages (always 1 for SENDMSG[_ZC]).

    if (use_zero_copy) {
      // Always request usage reporting so we can tell callers if ZC succeeded.
      sqe->ioprio |= IREE_IORING_SEND_ZC_REPORT_USAGE;
    }
  }
}

// Fills an SQE for a SOCKET_SENDTO operation.
// Always uses IORING_OP_SENDMSG[_ZC] since we need msg_name for the destination
// address. This works for both single-buffer and scatter-gather sends.
static void iree_async_proactor_io_uring_fill_socket_sendto(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation,
    iree_async_proactor_capabilities_t capabilities) {
  iree_async_socket_sendto_operation_t* sendto =
      (iree_async_socket_sendto_operation_t*)base_operation;

  // Clear output fields.
  sendto->bytes_sent = 0;

  bool zero_copy_requested =
      iree_any_bit_set(sendto->socket->flags,
                       IREE_ASYNC_SOCKET_FLAG_ZERO_COPY) &&
      !iree_any_bit_set(sendto->send_flags,
                        IREE_ASYNC_SOCKET_SEND_FLAG_NO_ZERO_COPY);
  bool zero_copy_available = iree_any_bit_set(
      capabilities, IREE_ASYNC_PROACTOR_CAPABILITY_ZERO_COPY_SEND);
  bool use_zero_copy = zero_copy_requested && zero_copy_available;

  // Convert spans to iovecs. The iovec array is stored in the operation's
  // platform storage to ensure it lives until completion.
  const uint8_t span_count = base_operation->acquired_span_count;
  iree_async_proactor_io_uring_materialize_iovecs(
      &sendto->platform, sendto->retained_buffer_regions, span_count);
  struct iovec* iovecs = (struct iovec*)sendto->platform.posix.iovecs;

  // Build msghdr with destination address.
  struct msghdr* msg = (struct msghdr*)sendto->platform.posix.msg_header;
  memset(msg, 0, sizeof(*msg));
  msg->msg_name = sendto->destination.storage;
  msg->msg_namelen = (socklen_t)sendto->destination.length;
  msg->msg_iov = iovecs;
  msg->msg_iovlen = span_count;

  sqe->fd = sendto->socket->primitive.value.fd;
  sqe->opcode =
      use_zero_copy ? IREE_IORING_OP_SENDMSG_ZC : IREE_IORING_OP_SENDMSG;
  sqe->addr = (uint64_t)(uintptr_t)msg;
  sqe->len = 1;  // Number of messages (always 1 for SENDMSG).
  sqe->msg_flags = 0;
  if (iree_any_bit_set(sendto->send_flags, IREE_ASYNC_SOCKET_SEND_FLAG_MORE)) {
    sqe->msg_flags |= IREE_MSG_MORE;
  }
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;

  if (use_zero_copy) {
    sqe->ioprio |= IREE_IORING_SEND_ZC_REPORT_USAGE;
  }
}

// Fills an SQE for a SOCKET_RECVFROM operation.
// Always uses IORING_OP_RECVMSG since we need msg_name for the sender address.
// This works for both single-buffer and scatter-gather receives.
static void iree_async_proactor_io_uring_fill_socket_recvfrom(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_socket_recvfrom_operation_t* recvfrom =
      (iree_async_socket_recvfrom_operation_t*)base_operation;

  // Clear output fields. The sender address length is set to the storage size;
  // the kernel will update it to the actual address length on completion.
  recvfrom->bytes_received = 0;
  recvfrom->sender.length = sizeof(recvfrom->sender.storage);

  // Convert spans to iovecs. The iovec array is stored in the operation's
  // platform storage to ensure it lives until completion.
  const uint8_t span_count = base_operation->acquired_span_count;
  iree_async_proactor_io_uring_materialize_iovecs(
      &recvfrom->platform, recvfrom->retained_buffer_regions, span_count);
  struct iovec* iovecs = (struct iovec*)recvfrom->platform.posix.iovecs;

  // Build msghdr with sender address buffer.
  struct msghdr* msg = (struct msghdr*)recvfrom->platform.posix.msg_header;
  memset(msg, 0, sizeof(*msg));
  msg->msg_name = recvfrom->sender.storage;
  msg->msg_namelen = sizeof(recvfrom->sender.storage);
  msg->msg_iov = iovecs;
  msg->msg_iovlen = span_count;

  sqe->fd = recvfrom->socket->primitive.value.fd;
  sqe->opcode = IREE_IORING_OP_RECVMSG;
  sqe->addr = (uint64_t)(uintptr_t)msg;
  sqe->len = 1;  // Number of messages (always 1 for RECVMSG).
  sqe->msg_flags = 0;
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a SOCKET_CLOSE operation.
// Uses IORING_OP_CLOSE to close the socket asynchronously.
// The socket reference is consumed (not retained) - it will be destroyed
// after completion.
static void iree_async_proactor_io_uring_fill_socket_close(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_socket_close_operation_t* close_op =
      (iree_async_socket_close_operation_t*)base_operation;

  // Note: We do NOT retain the socket here. The close operation consumes
  // the caller's reference. The socket will be destroyed on completion.

  sqe->opcode = IREE_IORING_OP_CLOSE;
  sqe->fd = close_op->socket->primitive.value.fd;
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

//===----------------------------------------------------------------------===//
// File operation SQE fills
//===----------------------------------------------------------------------===//

// Translates iree_async_file_open_flags_t to POSIX open flags for OPENAT.
static int iree_async_proactor_io_uring_translate_open_flags(
    iree_async_file_open_flags_t open_flags) {
  int flags = 0;
  if (iree_any_bit_set(open_flags, IREE_ASYNC_FILE_OPEN_FLAG_READ)) {
    flags |= iree_any_bit_set(open_flags, IREE_ASYNC_FILE_OPEN_FLAG_WRITE)
                 ? O_RDWR
                 : O_RDONLY;
  } else if (iree_any_bit_set(open_flags, IREE_ASYNC_FILE_OPEN_FLAG_WRITE)) {
    flags |= O_WRONLY;
  }
  if (iree_any_bit_set(open_flags, IREE_ASYNC_FILE_OPEN_FLAG_CREATE)) {
    flags |= O_CREAT;
  }
  if (iree_any_bit_set(open_flags, IREE_ASYNC_FILE_OPEN_FLAG_TRUNCATE)) {
    flags |= O_TRUNC;
  }
  if (iree_any_bit_set(open_flags, IREE_ASYNC_FILE_OPEN_FLAG_APPEND)) {
    flags |= O_APPEND;
  }
  if (iree_any_bit_set(open_flags, IREE_ASYNC_FILE_OPEN_FLAG_DIRECT)) {
    flags |= O_DIRECT;
  }
  flags |= O_CLOEXEC;  // Always set close-on-exec for safety.
  return flags;
}

// Fills an SQE for a FILE_OPEN operation.
// Uses IORING_OP_OPENAT with AT_FDCWD to open a file relative to the
// current working directory. The kernel performs the open asynchronously.
//
// SQE layout (io_uring OPENAT):
//   fd         = directory fd (AT_FDCWD for cwd)
//   addr       = pathname
//   len        = mode (permissions for creation)
//   open_flags = POSIX O_* flags
static void iree_async_proactor_io_uring_fill_file_open(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_file_open_operation_t* open_op =
      (iree_async_file_open_operation_t*)base_operation;

  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_OPENAT;
  sqe->fd = AT_FDCWD;
  sqe->addr = (uint64_t)(uintptr_t)open_op->path;
  sqe->len = 0664;  // Default mode for newly created files (rw-rw-r--).
  sqe->open_flags = (uint32_t)iree_async_proactor_io_uring_translate_open_flags(
      open_op->open_flags);
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a FILE_READ operation.
// Uses IORING_OP_READ for positioned file I/O (pread semantics).
//
// SQE layout (io_uring READ):
//   fd   = file descriptor
//   off  = file offset
//   addr = buffer address
//   len  = buffer length
static void iree_async_proactor_io_uring_fill_file_read(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_file_read_operation_t* read_op =
      (iree_async_file_read_operation_t*)base_operation;

  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_READ;
  sqe->fd = read_op->file->primitive.value.fd;
  sqe->off = read_op->offset;
  sqe->addr = (uint64_t)(uintptr_t)iree_async_span_ptr(read_op->buffer);
  sqe->len = (uint32_t)read_op->buffer.length;
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a FILE_WRITE operation.
// Uses IORING_OP_WRITE for positioned file I/O (pwrite semantics).
//
// SQE layout (io_uring WRITE):
//   fd   = file descriptor
//   off  = file offset
//   addr = buffer address
//   len  = buffer length
static void iree_async_proactor_io_uring_fill_file_write(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_file_write_operation_t* write_op =
      (iree_async_file_write_operation_t*)base_operation;

  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_WRITE;
  sqe->fd = write_op->file->primitive.value.fd;
  sqe->off = write_op->offset;
  sqe->addr = (uint64_t)(uintptr_t)iree_async_span_ptr(write_op->buffer);
  sqe->len = (uint32_t)write_op->buffer.length;
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a FILE_CLOSE operation.
// Uses IORING_OP_CLOSE to close the file descriptor asynchronously.
// The file reference is consumed (not retained) — it will be released
// during completion resource cleanup.
static void iree_async_proactor_io_uring_fill_file_close(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_file_close_operation_t* close_op =
      (iree_async_file_close_operation_t*)base_operation;

  sqe->opcode = IREE_IORING_OP_CLOSE;
  sqe->fd = close_op->file->primitive.value.fd;
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a FUTEX_WAIT operation.
// Uses IORING_OP_FUTEX_WAIT to wait on a futex address in the kernel.
// Requires IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS (kernel 6.7+).
//
// SQE layout (from liburing io_uring_prep_futex_wait):
//   fd    = futex2 flags (FUTEX2_SIZE_* | FUTEX2_PRIVATE)
//   addr  = futex address
//   off   = expected value
//   len   = 0
//   futex_flags = io_uring flags (0)
//   addr3 = bitset mask (~0 to match any)
static void iree_async_proactor_io_uring_fill_futex_wait(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_futex_wait_operation_t* futex_wait =
      (iree_async_futex_wait_operation_t*)base_operation;

  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_FUTEX_WAIT;
  sqe->fd = (int32_t)futex_wait->futex_flags;  // FUTEX2_SIZE_* | FUTEX2_PRIVATE
  sqe->addr = (uint64_t)(uintptr_t)futex_wait->futex_address;
  sqe->off = futex_wait->expected_value;
  sqe->len = 0;
  sqe->futex_flags = 0;      // io_uring flags, not futex2 flags.
  sqe->addr3 = 0xffffffffU;  // FUTEX_BITSET_MATCH_ANY (32-bit mask)
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

// Fills an SQE for a FUTEX_WAKE operation.
// Uses IORING_OP_FUTEX_WAKE to wake waiters on a futex address.
// Requires IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS (kernel 6.7+).
//
// SQE layout (from liburing io_uring_prep_futex_wake):
//   fd    = futex2 flags (FUTEX2_SIZE_* | FUTEX2_PRIVATE)
//   addr  = futex address
//   off   = number of waiters to wake
//   len   = 0
//   futex_flags = io_uring flags (0)
//   addr3 = bitset mask (~0 to match any)
static void iree_async_proactor_io_uring_fill_futex_wake(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_futex_wake_operation_t* futex_wake =
      (iree_async_futex_wake_operation_t*)base_operation;

  // Clear output field.
  futex_wake->woken_count = 0;

  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_FUTEX_WAKE;
  sqe->fd = (int32_t)futex_wake->futex_flags;  // FUTEX2_SIZE_* | FUTEX2_PRIVATE
  sqe->addr = (uint64_t)(uintptr_t)futex_wake->futex_address;
  sqe->off = (uint64_t)futex_wake->wake_count;  // Number of waiters to wake.
  sqe->len = 0;
  sqe->futex_flags = 0;      // io_uring flags, not futex2 flags.
  sqe->addr3 = 0xffffffffU;  // FUTEX_BITSET_MATCH_ANY (32-bit mask)
  sqe->user_data = (uint64_t)(uintptr_t)base_operation;
}

//===----------------------------------------------------------------------===//
// Software operation helpers
//===----------------------------------------------------------------------===//

// Returns true for operation types executed inline while dispatching a
// userspace continuation chain.
static inline bool iree_async_proactor_io_uring_is_inline_software_op(
    iree_async_operation_type_t type) {
  return type == IREE_ASYNC_OPERATION_TYPE_NOP ||
         type == IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL ||
         type == IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT ||
         type == IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL ||
         type == IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
}

// Returns true if a planned MESSAGE operation uses the software message pool.
static bool iree_async_proactor_io_uring_message_uses_fallback(
    const iree_async_message_operation_t* message) {
  return message->platform.io_uring.target_ring_fd < 0;
}

// Returns true for active operations that commit without a kernel SQE.
static bool iree_async_proactor_io_uring_is_submission_software_op(
    const iree_async_operation_t* operation) {
  if (iree_async_proactor_io_uring_is_inline_software_op(operation->type) ||
      operation->type == IREE_ASYNC_OPERATION_TYPE_SEQUENCE) {
    return true;
  }
  return operation->type == IREE_ASYNC_OPERATION_TYPE_MESSAGE &&
         iree_async_proactor_io_uring_message_uses_fallback(
             (const iree_async_message_operation_t*)operation);
}

// Returns true when a successful operation can produce a negative CQE result
// or otherwise requires userspace continuation dispatch. IOSQE_IO_LINK treats
// every negative result as failure and would cancel the remaining chain before
// the proactor can apply its operation-specific status mapping.
static inline bool iree_async_proactor_io_uring_requires_userspace_continuation(
    const iree_async_operation_t* operation) {
  return operation->type == IREE_ASYNC_OPERATION_TYPE_TIMER ||
         operation->type == IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT ||
         iree_async_proactor_io_uring_is_submission_software_op(operation);
}

// Returns true when a LINKED edge must be dispatched in userspace instead of
// represented with IOSQE_IO_LINK.
static inline bool iree_async_proactor_io_uring_requires_userspace_link(
    iree_async_operation_t* operation, iree_async_operation_t* successor) {
  return iree_async_proactor_io_uring_requires_userspace_continuation(
             operation) ||
         successor->type == IREE_ASYNC_OPERATION_TYPE_SEQUENCE ||
         iree_async_proactor_io_uring_is_submission_software_op(successor);
}

// Advances an analysis pass over a submission batch and returns true when the
// operation at |index| belongs to the prefix of its chain that can be submitted
// now. Once a userspace LINKED edge is encountered, the remainder of that
// chain is deferred until its predecessor completes. The next independent
// chain begins a new active prefix.
static bool iree_async_proactor_io_uring_should_submit_batch_operation(
    iree_async_operation_list_t operations, iree_host_size_t index,
    bool* defer_chain_tail) {
  if (index == 0 || !iree_any_bit_set(operations.values[index - 1]->flags,
                                      IREE_ASYNC_OPERATION_FLAG_LINKED)) {
    *defer_chain_tail = false;
  }
  if (*defer_chain_tail) {
    return false;
  }

  iree_async_operation_t* operation = operations.values[index];
  if (iree_any_bit_set(operation->flags, IREE_ASYNC_OPERATION_FLAG_LINKED) &&
      iree_async_proactor_io_uring_requires_userspace_link(
          operation, operations.values[index + 1])) {
    *defer_chain_tail = true;
  }
  return true;
}

iree_async_operation_t*
iree_async_proactor_io_uring_build_software_submission_list(
    iree_async_operation_list_t operations) {
  iree_async_operation_t* head = NULL;
  iree_async_operation_t** tail = &head;
  bool defer_chain_tail = false;
  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    if (!iree_async_proactor_io_uring_should_submit_batch_operation(
            operations, i, &defer_chain_tail)) {
      continue;
    }
    iree_async_operation_t* operation = operations.values[i];
    if (!iree_async_proactor_io_uring_is_submission_software_op(operation)) {
      continue;
    }
    operation->next = NULL;
    *tail = operation;
    tail = &operation->next;
  }
  return head;
}

// The epoch publication belongs to execution, not native SQE construction.
// In a linked chain this runs only after the predecessor succeeds.
static void iree_async_proactor_io_uring_execute_notification_signal(
    iree_async_notification_signal_operation_t* signal_op) {
  signal_op->woken_count = -1;
  iree_async_notification_signal(signal_op->notification,
                                 signal_op->wake_count);
}

// Executes a SEMAPHORE_SIGNAL operation synchronously. Returns OK on success,
// or the first signal failure status. The caller delivers the completion
// (via MPSC push to the poll thread for callback dispatch).
static iree_status_t iree_async_proactor_io_uring_execute_semaphore_signal(
    iree_async_semaphore_signal_operation_t* signal_op) {
  for (iree_host_size_t i = 0; i < signal_op->count; ++i) {
    IREE_RETURN_IF_ERROR(iree_async_semaphore_signal(
        signal_op->semaphores[i], signal_op->values[i], signal_op->frontier));
  }
  return iree_ok_status();
}

// Enqueues a terminal semaphore wait tracker and wakes the poll thread.
static void iree_async_proactor_io_uring_enqueue_semaphore_wait(
    void* user_data, iree_atomic_slist_entry_t* entry) {
  iree_async_proactor_io_uring_t* proactor =
      (iree_async_proactor_io_uring_t*)user_data;
  iree_atomic_slist_push(&proactor->pending_semaphore_waits, entry);
  uint64_t wake_value = 1;
  ssize_t result =
      write(proactor->wake_eventfd, &wake_value, sizeof(wake_value));
  IREE_ASSERT(result >= 0 || errno == EAGAIN);
}

// Executes a SEMAPHORE_WAIT operation. Checks for immediate satisfaction
// first; if not immediately satisfied, allocates a tracker, transfers the
// continuation chain, and registers timepoints.
//
// On return:
//   *out_deferred == false: wait completed immediately (OK or error).
//     The caller handles continuation dispatch and completion delivery.
//   *out_deferred == true: timepoints registered, completion arrives later
//     via pending_semaphore_waits. The tracker holds the continuation chain.
static iree_status_t iree_async_proactor_io_uring_execute_semaphore_wait(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_operation_t* base_operation, bool* out_deferred) {
  *out_deferred = false;

  iree_async_semaphore_wait_operation_t* wait_op =
      (iree_async_semaphore_wait_operation_t*)base_operation;

  // Check for immediate satisfaction before allocating a tracker.
  if (iree_async_semaphore_wait_is_satisfied(wait_op)) {
    return iree_ok_status();
  }

  iree_async_semaphore_wait_enqueue_callback_t enqueue_callback = {
      .fn = iree_async_proactor_io_uring_enqueue_semaphore_wait,
      .user_data = proactor,
  };
  iree_async_semaphore_wait_tracker_t* tracker = NULL;
  IREE_RETURN_IF_ERROR(iree_async_semaphore_wait_tracker_allocate(
      &proactor->semaphore_wait_context, wait_op, enqueue_callback,
      proactor->base.allocator, &tracker));
  iree_async_semaphore_wait_tracker_register_timepoints(tracker);

  *out_deferred = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Iterative continuation chain dispatch
//===----------------------------------------------------------------------===//

// Iteratively dispatches a LINKED continuation chain that may contain software
// operations. Walks the chain in order:
//
//   - Software ops: execute side effects inline and push completion to MPSC
//     for poll-thread delivery (NOP, notification signal, semaphore ops).
//   - Kernel ops: submit the remaining chain via submit_continuation_chain
//     (produces CQEs counted by the CQE processing loop).
//   - Deferred WAIT: the tracker takes ownership of the remaining chain.
//
// Iterative (not recursive) dispatch ensures software completions are pushed
// to MPSC in chain order. Recursive dispatch via submit_continuation_chain
// pushes in depth-first (reverse) order.
//
// On error: the failing software op's error completion is pushed, and the
// remaining chain is cancelled with CANCELLED completions via MPSC.
void iree_async_proactor_io_uring_dispatch_continuation_chain(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_operation_t* chain_head) {
  iree_async_operation_t* op = chain_head;
  while (op) {
    if (!iree_async_proactor_io_uring_is_inline_software_op(op->type)) {
      // Kernel op: submit the remaining chain (op + its linked_next tail).
      // Kernel ops produce CQEs; their completions are counted by the CQE
      // processing loop. If the remaining chain contains more software ops
      // after this kernel op, they will be dispatched when the kernel op's
      // CQE triggers another continuation dispatch.
      iree_async_proactor_io_uring_submit_continuation_chain(proactor, op);
      return;
    }

    if (op->type == IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT) {
      iree_async_io_uring_notification_submit_wait(
          (iree_async_notification_wait_operation_t*)op);
      return;
    }

    iree_status_t op_status = iree_ok_status();
    bool deferred = false;
    if (op->type == IREE_ASYNC_OPERATION_TYPE_NOP) {
      // NOP has no side effect.
    } else if (op->type == IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL) {
      iree_async_proactor_io_uring_execute_notification_signal(
          (iree_async_notification_signal_operation_t*)op);
    } else if (op->type == IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL) {
      op_status = iree_async_proactor_io_uring_execute_semaphore_signal(
          (iree_async_semaphore_signal_operation_t*)op);
    } else {
      // SEMAPHORE_WAIT: may transfer linked_next to the tracker if deferred.
      op_status = iree_async_proactor_io_uring_execute_semaphore_wait(
          proactor, op, &deferred);
    }

    // Extract the continuation before push because pending_status aliases
    // linked_next in the operation base.
    iree_async_operation_t* next = op->linked_next;
    op->linked_next = NULL;

    if (!iree_status_is_ok(op_status)) {
      // Software op failed. Cancel remaining chain via MPSC.
      iree_async_proactor_io_uring_push_software_operation(proactor, op,
                                                           op_status);
      if (next) {
        iree_async_proactor_io_uring_cancel_continuation_chain_to_mpsc(proactor,
                                                                       next);
      }
      return;
    }

    if (deferred) {
      // Tracker holds the continuation chain (transferred from linked_next
      // by execute_semaphore_wait). The WAIT completion arrives later via
      // pending_semaphore_waits, and the tracker dispatches its continuation
      // chain when the WAIT completes.
      //
      // Don't push a completion for this op — the tracker pathway handles it.
      return;
    }

    // Push this op's completion to MPSC.
    iree_async_proactor_io_uring_push_software_operation(proactor, op,
                                                         op_status);

    op = next;
  }
}

// Cancels a continuation chain by pushing callback-bearing operations to the
// MPSC queue. Deliberately suppressed completions are consumed immediately so
// their storage can be released from the preceding callback.
void iree_async_proactor_io_uring_cancel_continuation_chain_to_mpsc(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_operation_t* chain_head) {
  iree_async_operation_t* op = chain_head;
  while (op) {
    iree_async_operation_t* next = op->linked_next;
    op->linked_next = NULL;
    if (op->completion_fn) {
      iree_async_proactor_io_uring_push_software_operation(
          proactor, op, iree_status_from_code(IREE_STATUS_CANCELLED));
    } else {
      iree_async_operation_complete(
          op, iree_status_from_code(IREE_STATUS_CANCELLED),
          IREE_ASYNC_COMPLETION_FLAG_NONE);
    }
    op = next;
  }
}

//===----------------------------------------------------------------------===//
// Message operation helpers
//===----------------------------------------------------------------------===//

// Freezes the delivery strategy for one validated MESSAGE operation. A target
// ring can transition from disabled to enabled concurrently; retaining the
// fallback selected here is always valid, while recomputing later could leak a
// reservation or miscount SQEs.
static void iree_async_proactor_io_uring_plan_message(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_message_operation_t* message) {
  iree_async_proactor_io_uring_t* target =
      iree_async_proactor_io_uring_cast(message->target);
  bool can_use_msg_ring =
      iree_any_bit_set(proactor->capabilities,
                       IREE_ASYNC_PROACTOR_CAPABILITY_PROACTOR_MESSAGING) &&
      !iree_io_uring_ring_needs_enable(&target->ring);
  message->platform.io_uring.target_ring_fd =
      can_use_msg_ring ? target->ring.ring_fd : -1;
  message->platform.io_uring.reserved_entry = NULL;
}

// Fills an SQE for a MESSAGE operation (cross-proactor messaging via MSG_RING).
//
// MSG_RING posts a CQE directly to the target ring without any userspace
// involvement on the target. The message_data is delivered via:
//   - sqe->len (32 bits) -> target cqe->res
//   - sqe->off (64 bits) -> target cqe->user_data
//
// We encode the message_data in the target's user_data along with an internal
// marker so the target proactor can identify it as an incoming message rather
// than a normal operation completion.
static void iree_async_proactor_io_uring_fill_message(
    iree_io_uring_sqe_t* sqe, iree_async_operation_t* base_operation) {
  iree_async_message_operation_t* message =
      (iree_async_message_operation_t*)base_operation;

  memset(sqe, 0, sizeof(*sqe));
  sqe->opcode = IREE_IORING_OP_MSG_RING;
  sqe->fd = message->platform.io_uring.target_ring_fd;

  // The target CQE will receive:
  //   cqe->res = sqe->len (lower 32 bits of message_data)
  //   cqe->user_data = sqe->off (internal encoding with upper 32 bits)
  //
  // We split the 64-bit message_data across both fields to preserve all bits:
  //   - Lower 32 bits go in sqe->len → cqe->res
  //   - Upper 32 bits go in the payload of sqe->off → cqe->user_data
  // The decode reconstructs the full 64-bit value from both fields.
  sqe->len = (uint32_t)(message->message_data & 0xFFFFFFFFULL);
  sqe->off = iree_io_uring_internal_encode(IREE_IO_URING_TAG_MESSAGE_RECEIVE,
                                           message->message_data >> 32);

  // Handle source completion suppression.
  if (message->message_flags & IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION) {
    // User doesn't want a source callback. We mark this as internal with
    // MESSAGE_SOURCE tag so we can discard the CQE when it arrives.
    //
    // NOTE: The kernel has IORING_MSG_RING_CQE_SKIP to suppress the source CQE
    // entirely, but there's no reliable way to detect support for it (it's
    // separate from FEAT_CQE_SKIP which is about IOSQE_CQE_SKIP_SUCCESS). We
    // could try setting it and handle EINVAL, but that would require complex
    // retry logic. For now, we always receive and discard the source CQE.
    sqe->user_data =
        iree_io_uring_internal_encode(IREE_IO_URING_TAG_MESSAGE_SOURCE, 0);
  } else {
    // Normal case: source gets a completion via user callback.
    sqe->user_data = (uint64_t)(uintptr_t)base_operation;
  }
}

// Reserves target capacity for a planned fallback message without publishing
// payload data or waking the target.
static iree_status_t iree_async_proactor_io_uring_reserve_fallback_message(
    iree_async_message_operation_t* message) {
  iree_async_proactor_io_uring_t* target =
      iree_async_proactor_io_uring_cast(message->target);
  iree_async_message_pool_entry_t* entry = NULL;
  IREE_RETURN_IF_ERROR(
      iree_async_message_pool_acquire(&target->message_pool, &entry));
  message->platform.io_uring.reserved_entry = entry;
  return iree_ok_status();
}

// Publishes a reserved fallback message after the complete source batch has
// crossed its admission boundary.
static void iree_async_proactor_io_uring_commit_fallback_message(
    iree_async_message_operation_t* message) {
  iree_async_proactor_io_uring_t* target =
      iree_async_proactor_io_uring_cast(message->target);
  iree_async_message_pool_entry_t* entry =
      (iree_async_message_pool_entry_t*)
          message->platform.io_uring.reserved_entry;
  IREE_ASSERT(entry);
  message->platform.io_uring.reserved_entry = NULL;
  iree_async_message_pool_publish(&target->message_pool, entry,
                                  message->message_data);
  iree_async_proactor_wake(&target->base);
}

//===----------------------------------------------------------------------===//
// Submission reservation rollback
//===----------------------------------------------------------------------===//

// Releases unpublished message and receive reservations. No operation resources
// have been retained when this runs, so close ownership remains caller-owned.
static void iree_async_proactor_io_uring_rollback_reservations(
    iree_async_operation_list_t operations) {
  bool defer_chain_tail = false;
  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    if (!iree_async_proactor_io_uring_should_submit_batch_operation(
            operations, i, &defer_chain_tail)) {
      continue;
    }
    iree_async_operation_t* operation = operations.values[i];
    if (operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL) {
      iree_async_socket_recv_pool_operation_t* receive =
          (iree_async_socket_recv_pool_operation_t*)operation;
      iree_async_buffer_lease_release(&receive->lease);
      continue;
    }
    if (operation->type != IREE_ASYNC_OPERATION_TYPE_MESSAGE) {
      continue;
    }
    iree_async_message_operation_t* message =
        (iree_async_message_operation_t*)operation;
    if (!iree_async_proactor_io_uring_message_uses_fallback(message) ||
        !message->platform.io_uring.reserved_entry) {
      continue;
    }
    iree_async_proactor_io_uring_t* target =
        iree_async_proactor_io_uring_cast(message->target);
    iree_async_message_pool_entry_t* entry =
        (iree_async_message_pool_entry_t*)
            message->platform.io_uring.reserved_entry;
    message->platform.io_uring.reserved_entry = NULL;
    iree_async_message_pool_release(&target->message_pool, entry);
  }
}

// Commits one software operation from a fully admitted batch. The caller must
// capture and clear operation->next before calling because this may publish the
// operation to a concurrent poll owner.
static void iree_async_proactor_io_uring_commit_software_operation(
    iree_async_proactor_io_uring_t* proactor,
    iree_async_operation_t* operation) {
  IREE_TRACE({ operation->submit_time_ns = iree_time_now(); });

  if (operation->type == IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT) {
    iree_async_io_uring_notification_submit_wait(
        (iree_async_notification_wait_operation_t*)operation);
    return;
  }

  if (operation->type == IREE_ASYNC_OPERATION_TYPE_SEQUENCE) {
    iree_async_sequence_prepare_for_submission(
        (iree_async_sequence_operation_t*)operation);
    iree_async_proactor_io_uring_push_software_operation(proactor, operation,
                                                         iree_ok_status());
    return;
  }

  if (operation->type == IREE_ASYNC_OPERATION_TYPE_MESSAGE) {
    iree_async_message_operation_t* message =
        (iree_async_message_operation_t*)operation;
    iree_async_operation_t* continuation = operation->linked_next;
    operation->linked_next = NULL;
    iree_async_proactor_io_uring_commit_fallback_message(message);
    if (!iree_any_bit_set(message->message_flags,
                          IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION)) {
      iree_async_proactor_io_uring_push_software_operation(proactor, operation,
                                                           iree_ok_status());
    }
    if (continuation) {
      iree_async_proactor_io_uring_dispatch_continuation_chain(proactor,
                                                               continuation);
    }
    return;
  }

  iree_status_t operation_status = iree_ok_status();
  bool deferred = false;
  if (operation->type == IREE_ASYNC_OPERATION_TYPE_NOP) {
    // NOP has no side effect.
  } else if (operation->type == IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL) {
    iree_async_proactor_io_uring_execute_notification_signal(
        (iree_async_notification_signal_operation_t*)operation);
  } else if (operation->type == IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL) {
    iree_async_semaphore_signal_operation_t* signal_op =
        (iree_async_semaphore_signal_operation_t*)operation;
    operation_status =
        iree_async_proactor_io_uring_execute_semaphore_signal(signal_op);
  } else {
    // SEMAPHORE_WAIT: check immediate satisfaction or register timepoints.
    operation_status = iree_async_proactor_io_uring_execute_semaphore_wait(
        proactor, operation, &deferred);
  }

  // Extract the continuation before push because pending_status aliases
  // linked_next in the operation base.
  iree_async_operation_t* continuation = operation->linked_next;
  operation->linked_next = NULL;

  if (!iree_status_is_ok(operation_status)) {
    iree_async_proactor_io_uring_push_software_operation(proactor, operation,
                                                         operation_status);
    if (continuation) {
      iree_async_proactor_io_uring_cancel_continuation_chain_to_mpsc(
          proactor, continuation);
    }
    return;
  }

  if (deferred) {
    // Timepoints registered. Completion arrives later via
    // pending_semaphore_waits. The tracker holds the continuation chain
    // (transferred from linked_next in execute_semaphore_wait).
    return;
  }

  // Push this op's completion before dispatching its continuation so the
  // trigger callback is observed first.
  iree_async_proactor_io_uring_push_software_operation(proactor, operation,
                                                       operation_status);
  if (continuation) {
    iree_async_proactor_io_uring_dispatch_continuation_chain(proactor,
                                                             continuation);
  }
}

//===----------------------------------------------------------------------===//
// Submit
//===----------------------------------------------------------------------===//

iree_status_t iree_async_proactor_io_uring_submit(
    iree_async_proactor_t* base_proactor,
    iree_async_operation_list_t operations) {
  iree_async_proactor_io_uring_t* proactor =
      iree_async_proactor_io_uring_cast(base_proactor);

  if (operations.count == 0) {
    return iree_ok_status();
  }

  IREE_RETURN_IF_ERROR(iree_async_continuation_prepare_batch(operations));

  //=========================================================================
  // Phase 1: Validate and plan the complete batch.
  //=========================================================================

  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    if (operations.values[i]->resources_acquired) {
      // Accepted continuations were validated with their original batch and
      // may no longer have accessible caller-owned descriptor arrays.
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_async_proactor_io_uring_validate_operation(
        proactor, operations.values[i]));
    if (operations.values[i]->type ==
        IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT) {
      iree_async_operation_clear_internal_flags(operations.values[i]);
    }
  }
  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    iree_async_operation_t* operation = operations.values[i];
    if (operation->type == IREE_ASYNC_OPERATION_TYPE_MESSAGE) {
      iree_async_proactor_io_uring_plan_message(
          proactor, (iree_async_message_operation_t*)operation);
    } else if (operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL) {
      iree_async_socket_recv_pool_operation_t* receive =
          (iree_async_socket_recv_pool_operation_t*)operation;
      receive->bytes_received = 0;
      memset(&receive->lease, 0, sizeof(receive->lease));
    }
  }

  // Count kernel SQEs and software work across the
  // active prefix of each independent chain. A userspace continuation edge
  // defers only the remainder of its own chain; later independent chains are
  // still part of this submission.
  iree_host_size_t sqes_needed = 0;
  bool defer_chain_tail = false;
  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    if (!iree_async_proactor_io_uring_should_submit_batch_operation(
            operations, i, &defer_chain_tail)) {
      continue;
    }
    iree_async_operation_t* operation = operations.values[i];
    operation->next = NULL;
    if (iree_async_proactor_io_uring_is_submission_software_op(operation)) {
      continue;
    }
    if (operation->type == IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT) {
      sqes_needed += 2;
    } else {
      sqes_needed += 1;
    }
  }
  iree_async_operation_t* software_operations = NULL;

  //=========================================================================
  // Phase 2: Reserve bounded userspace and kernel capacity.
  //=========================================================================

  defer_chain_tail = false;
  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    if (!iree_async_proactor_io_uring_should_submit_batch_operation(
            operations, i, &defer_chain_tail)) {
      continue;
    }
    iree_async_operation_t* operation = operations.values[i];
    iree_status_t status = iree_ok_status();
    if (operation->type == IREE_ASYNC_OPERATION_TYPE_MESSAGE &&
        iree_async_proactor_io_uring_message_uses_fallback(
            (iree_async_message_operation_t*)operation)) {
      status = iree_async_proactor_io_uring_reserve_fallback_message(
          (iree_async_message_operation_t*)operation);
    } else if (operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL) {
      iree_async_socket_recv_pool_operation_t* receive =
          (iree_async_socket_recv_pool_operation_t*)operation;
      if (iree_async_buffer_pool_region(receive->pool)
              ->handles.iouring.buffer_group_id < 0) {
        status = iree_async_buffer_pool_acquire(receive->pool, &receive->lease);
      }
    }
    if (!iree_status_is_ok(status)) {
      iree_async_proactor_io_uring_rollback_reservations(operations);
      return status;
    }
  }

  // Freeze software selection while the complete batch is still
  // submitter-owned. Kernel SQEs may become visible to another poll owner as
  // soon as the SQ lock is released below.
  software_operations =
      iree_async_proactor_io_uring_build_software_submission_list(operations);

  if (sqes_needed > 0) {
    // Acquire the SQ lock for the duration of SQE preparation.
    // All sq_local_tail reads/writes must happen under this lock to prevent a
    // concurrent thread from seeing a partially-filled SQE during flush.
    iree_io_uring_ring_sq_lock(&proactor->ring);

    // Check SQ capacity.
    uint32_t available = iree_io_uring_ring_sq_space_left(&proactor->ring);
    if (available < sqes_needed) {
      iree_io_uring_ring_sq_unlock(&proactor->ring);
      iree_async_proactor_io_uring_rollback_reservations(operations);
      while (software_operations) {
        iree_async_operation_t* operation = software_operations;
        software_operations = operation->next;
        operation->next = NULL;
      }
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "SQ has %u slots but %zu SQEs needed for %zu "
                              "operations",
                              available, sqes_needed, operations.count);
    }

    // The complete list is now accepted. The SQ lock keeps the capacity
    // reservation stable while linked successors acquire their resources and
    // submit-scoped descriptors before any SQE is published.
    iree_async_operation_list_acquire_resources(operations);

    //=======================================================================
    // Phase 3: Commit kernel SQEs under the SQ lock.
    //=======================================================================

    // Fill SQEs for the active kernel prefixes. Software ops execute in Phase
    // 4 and deferred chain tails are submitted by their predecessors.
    defer_chain_tail = false;
    for (iree_host_size_t i = 0; i < operations.count; ++i) {
      if (!iree_async_proactor_io_uring_should_submit_batch_operation(
              operations, i, &defer_chain_tail)) {
        continue;
      }
      iree_async_operation_t* operation = operations.values[i];

      if (iree_async_proactor_io_uring_is_submission_software_op(operation)) {
        continue;
      }

      // EVENT_WAIT uses linked POLL_ADD+READ; notification waits are software.
      if (operation->type == IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT) {
        iree_io_uring_sqe_t* poll_sqe =
            iree_io_uring_ring_get_sqe(&proactor->ring);
        iree_io_uring_sqe_t* read_sqe =
            iree_io_uring_ring_get_sqe(&proactor->ring);
        IREE_ASSERT(poll_sqe);
        IREE_ASSERT(read_sqe);

        iree_async_proactor_io_uring_fill_event_wait(poll_sqe, read_sqe,
                                                     operation);

        // Apply kernel LINK to the terminal SQE of the internal pair when the
        // edge does not require userspace status interpretation.
        if (iree_any_bit_set(operation->flags,
                             IREE_ASYNC_OPERATION_FLAG_LINKED) &&
            !iree_async_proactor_io_uring_requires_userspace_link(
                operation, operations.values[i + 1])) {
          read_sqe->flags |= IREE_IOSQE_IO_LINK;
          operation->linked_next = NULL;
        }
      } else {
        // All other kernel operations use 1 SQE.
        iree_io_uring_sqe_t* sqe = iree_io_uring_ring_get_sqe(&proactor->ring);
        IREE_ASSERT(sqe);

        switch (operation->type) {
          case IREE_ASYNC_OPERATION_TYPE_TIMER:
            iree_async_proactor_io_uring_fill_timer(proactor, sqe, operation);
            break;
          case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
            iree_async_proactor_io_uring_fill_socket_connect(sqe, operation);
            break;
          case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
            iree_async_proactor_io_uring_fill_socket_accept(sqe, operation);
            break;
          case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
            iree_async_proactor_io_uring_fill_socket_recv(sqe, operation);
            break;
          case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
            iree_async_proactor_io_uring_fill_socket_recv_pool(sqe, operation);
            break;
          case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
            iree_async_proactor_io_uring_fill_socket_send(proactor, sqe,
                                                          operation);
            break;
          case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO:
            iree_async_proactor_io_uring_fill_socket_sendto(
                sqe, operation, proactor->capabilities);
            break;
          case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
            iree_async_proactor_io_uring_fill_socket_recvfrom(sqe, operation);
            break;
          case IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE:
            iree_async_proactor_io_uring_fill_socket_close(sqe, operation);
            break;
          case IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT:
            iree_async_proactor_io_uring_fill_futex_wait(sqe, operation);
            break;
          case IREE_ASYNC_OPERATION_TYPE_FUTEX_WAKE:
            iree_async_proactor_io_uring_fill_futex_wake(sqe, operation);
            break;
          case IREE_ASYNC_OPERATION_TYPE_MESSAGE:
            iree_async_proactor_io_uring_fill_message(sqe, operation);
            break;
          case IREE_ASYNC_OPERATION_TYPE_HANDLE_POLL:
            iree_async_proactor_io_uring_fill_handle_poll(sqe, operation);
            break;
          case IREE_ASYNC_OPERATION_TYPE_FILE_OPEN:
            iree_async_proactor_io_uring_fill_file_open(sqe, operation);
            break;
          case IREE_ASYNC_OPERATION_TYPE_FILE_READ:
            iree_async_proactor_io_uring_fill_file_read(sqe, operation);
            break;
          case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE:
            iree_async_proactor_io_uring_fill_file_write(sqe, operation);
            break;
          case IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE:
            iree_async_proactor_io_uring_fill_file_close(sqe, operation);
            break;
          default:
            IREE_ASSERT_UNREACHABLE("operation type must be validated");
            IREE_BUILTIN_UNREACHABLE();
        }

        // Apply kernel LINK when the edge does not require userspace status
        // interpretation.
        if (iree_any_bit_set(operation->flags,
                             IREE_ASYNC_OPERATION_FLAG_LINKED) &&
            !iree_async_proactor_io_uring_requires_userspace_link(
                operation, operations.values[i + 1])) {
          sqe->flags |= IREE_IOSQE_IO_LINK;
          operation->linked_next = NULL;
        }
      }

      // Publish all writes to the operation (fill, resource retain, user data)
      // via the TSAN atomic bridge. Pairs with TSAN_COMPLETE in process_cqe.
      IREE_IO_URING_TSAN_SUBMIT(operation);

      IREE_TRACE({ operation->submit_time_ns = iree_time_now(); });
    }

    // Release the SQ lock. All SQEs are fully filled; sq_local_tail is
    // advanced. The SQEs are not yet visible to the kernel.
    iree_io_uring_ring_sq_unlock(&proactor->ring);
  }

  if (sqes_needed == 0) {
    // Pure-software batches cross their acceptance boundary after fallback
    // message reservations complete.
    iree_async_operation_list_acquire_resources(operations);
  }

  //=========================================================================
  // Phase 4: Commit software operations (no lock held).
  //=========================================================================
  //
  // Inline operations execute their side effects after the complete batch has
  // reserved capacity. Fallback messages publish their reserved target entry,
  // and sequences enter the same MPSC queue for startup by the poll owner.
  // Completion entries are queued before continuations are dispatched so user
  // callbacks preserve chain order.

  while (software_operations) {
    iree_async_operation_t* operation = software_operations;
    software_operations = operation->next;
    operation->next = NULL;
    iree_async_proactor_io_uring_commit_software_operation(proactor, operation);
  }

  //=========================================================================
  // Phase 5: Flush or wake.
  //=========================================================================

  int32_t dispatch_tid = iree_atomic_load(&proactor->polling.dispatch_tid,
                                          iree_memory_order_relaxed);
  if (dispatch_tid != 0) {
    if (dispatch_tid == (int32_t)syscall(__NR_gettid)) {
      // Poll thread during CQE processing: flush SQEs immediately for
      // latency. This gets SQEs to the kernel promptly so inline sends
      // (where the socket buffer has room) complete during io_uring_enter.
      // NOTE: This is a latency optimization, not a data lifetime guarantee.
      // Under socket buffer pressure, the kernel defers sends to io-wq
      // worker threads that read buffer data after io_uring_enter returns.
      // Callers must ensure buffer data survives until the completion
      // callback fires.
      //
      // No GETEVENTS: we avoid running task_work mid-processing. The drain
      // loop handles deferred completions with GETEVENTS after the CQE loop.
      iree_status_t flush_status = iree_io_uring_ring_submit(
          &proactor->ring, /*min_complete=*/0, /*flags=*/0);
      if (!iree_status_is_ok(flush_status)) {
        // ring_submit publishes the SQ tail before entering the kernel. The
        // operations are therefore accepted even when io_uring_enter fails;
        // the poll loop's unconditional flush will either submit them or
        // propagate a persistent ring failure. Returning the flush error here
        // would falsely return ownership to the caller and permit a duplicate
        // completion while the published SQEs still reference the operations.
        iree_status_free(flush_status);
      }
      return iree_ok_status();
    }
    // Publish queued work to the active dispatch interval before suppressing
    // the wake. This RMW pairs with the owner's exchange to idle: either its
    // final drain observes our work or we observe idle and wake it below.
    // A load alone can observe active while the final drain misses our work.
    if (iree_atomic_compare_exchange_strong(
            &proactor->polling.dispatch_tid, &dispatch_tid, dispatch_tid,
            iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
      return iree_ok_status();
    }
  }

  // Poll thread idle. Only the poll thread may call io_uring_enter
  // (SINGLE_ISSUER constraint), so wake it to flush pending SQEs.
  // For pure-software batches (sqes_needed == 0), the wake causes poll() to
  // drain pending software operations and fire callbacks.
  iree_async_proactor_wake(&proactor->base);
  return iree_ok_status();
}
