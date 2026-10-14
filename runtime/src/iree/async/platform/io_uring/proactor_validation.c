// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/io_uring/proactor_validation.h"

#include <limits.h>
#include <stdint.h>

#include "iree/async/event.h"
#include "iree/async/file.h"
#include "iree/async/notification.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/futex.h"
#include "iree/async/operations/message.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/platform/io_uring/defs.h"
#include "iree/async/util/sequence_emulation.h"

static iree_status_t iree_async_proactor_io_uring_validate_socket(
    iree_async_proactor_io_uring_t* proactor, const iree_async_socket_t* socket,
    const char* operation_name) {
  if (!socket) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s socket is NULL",
                            operation_name);
  }
  if (socket->proactor != &proactor->base) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s socket belongs to a different proactor",
                            operation_name);
  }
  if (socket->primitive.type != IREE_ASYNC_PRIMITIVE_TYPE_FD ||
      socket->primitive.value.fd < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s socket has an invalid descriptor",
                            operation_name);
  }
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_io_uring_validate_file(
    iree_async_proactor_io_uring_t* proactor, const iree_async_file_t* file,
    const char* operation_name) {
  if (!file) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s file is NULL",
                            operation_name);
  }
  if (file->proactor != &proactor->base) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s file belongs to a different proactor",
                            operation_name);
  }
  if (file->primitive.type != IREE_ASYNC_PRIMITIVE_TYPE_FD ||
      file->primitive.value.fd < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s file has an invalid descriptor",
                            operation_name);
  }
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_io_uring_validate_address(
    const iree_async_address_t* address, const char* operation_name) {
  if (address->length == 0 || address->length > sizeof(address->storage)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "%s address length %" PRIhsz " is outside the valid range [1, %zu]",
        operation_name, address->length, sizeof(address->storage));
  }
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_io_uring_validate_span(
    iree_async_proactor_io_uring_t* proactor, iree_async_span_t span,
    iree_async_buffer_access_flags_t required_access,
    const char* operation_name) {
  if (!span.region) {
    if (span.length > 0 && span.offset == 0) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "%s span has a NULL data pointer",
                              operation_name);
    }
    return iree_ok_status();
  }

  const iree_async_region_t* region = span.region;
  if (region->proactor != &proactor->base) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s span region belongs to a different proactor",
                            operation_name);
  }
  if (!iree_async_span_is_cpu_accessible(span)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "%s requires CPU-accessible memory on the io_uring backend",
        operation_name);
  }
  if (!iree_all_bits_set(region->access_flags, required_access)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "%s span region does not provide required access flags 0x%08X",
        operation_name, required_access);
  }
  if (span.offset > region->length ||
      span.length > region->length - span.offset) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%s span range [%" PRIhsz ", %" PRIhsz
                            ") exceeds region length %" PRIhsz,
                            operation_name, span.offset,
                            span.offset + span.length, region->length);
  }
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_io_uring_validate_span_list(
    iree_async_proactor_io_uring_t* proactor, iree_async_span_list_t spans,
    iree_host_size_t maximum_count,
    iree_async_buffer_access_flags_t required_access,
    const char* operation_name) {
  if (spans.count > maximum_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "%s span count %" PRIhsz
                            " exceeds the maximum of %" PRIhsz,
                            operation_name, spans.count, maximum_count);
  }
  if (spans.count > 0 && !spans.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s span list is NULL", operation_name);
  }
  for (iree_host_size_t i = 0; i < spans.count; ++i) {
    IREE_RETURN_IF_ERROR(iree_async_proactor_io_uring_validate_span(
        proactor, spans.values[i], required_access, operation_name));
  }
  return iree_ok_status();
}

static iree_status_t iree_async_proactor_io_uring_validate_futex(
    iree_async_proactor_io_uring_t* proactor, const void* futex_address,
    iree_async_futex_flags_t futex_flags, const char* operation_name) {
  if (!iree_any_bit_set(proactor->capabilities,
                        IREE_ASYNC_PROACTOR_CAPABILITY_FUTEX_OPERATIONS)) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "%s requires kernel io_uring futex operation support", operation_name);
  }
  if (!futex_address) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s futex address is NULL", operation_name);
  }
  const iree_async_futex_flags_t known_flags =
      IREE_ASYNC_FUTEX_SIZE_U64 | IREE_ASYNC_FUTEX_FLAG_PRIVATE;
  if (futex_flags & ~known_flags) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s has unknown flags 0x%08X", operation_name,
                            futex_flags & ~known_flags);
  }
  iree_host_size_t alignment = (iree_host_size_t)1
                               << (futex_flags & IREE_ASYNC_FUTEX_SIZE_U64);
  if ((uintptr_t)futex_address & (alignment - 1)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "%s address must be %" PRIhsz "-byte aligned",
                            operation_name, alignment);
  }
  return iree_ok_status();
}

iree_status_t iree_async_proactor_io_uring_validate_operation(
    iree_async_proactor_io_uring_t* proactor,
    const iree_async_operation_t* operation) {
  const iree_async_operation_flags_t known_operation_flags =
      IREE_ASYNC_OPERATION_FLAG_MULTISHOT | IREE_ASYNC_OPERATION_FLAG_LINKED |
      IREE_ASYNC_OPERATION_FLAG_CANCELLATION_IS_SUCCESS;
  if (operation->flags & ~known_operation_flags) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "operation has unknown flags 0x%08X",
                            operation->flags & ~known_operation_flags);
  }

  bool skips_completion = false;
  if (operation->type == IREE_ASYNC_OPERATION_TYPE_MESSAGE) {
    const iree_async_message_operation_t* message =
        (const iree_async_message_operation_t*)operation;
    skips_completion = iree_any_bit_set(
        message->message_flags, IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION);
  }
  if (!skips_completion && !operation->completion_fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "operation requires a completion callback");
  }

  if (iree_any_bit_set(operation->flags, IREE_ASYNC_OPERATION_FLAG_MULTISHOT)) {
    if (operation->type != IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT &&
        operation->type != IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV &&
        operation->type != IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "MULTISHOT is unsupported for io_uring operation type %d",
          (int)operation->type);
    }
    if (!iree_any_bit_set(proactor->capabilities,
                          IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT)) {
      return iree_make_status(
          IREE_STATUS_UNAVAILABLE,
          "MULTISHOT requires kernel io_uring multishot support");
    }
  }

  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_NOP:
    case IREE_ASYNC_OPERATION_TYPE_TIMER:
      return iree_ok_status();

    case IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT: {
      const iree_async_event_wait_operation_t* wait =
          (const iree_async_event_wait_operation_t*)operation;
      if (!wait->event) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "EVENT_WAIT event is NULL");
      }
      if (wait->event->proactor != &proactor->base) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "EVENT_WAIT event belongs to a different proactor");
      }
      if (wait->event->native.wait_primitive.type !=
              IREE_ASYNC_PRIMITIVE_TYPE_FD ||
          wait->event->native.wait_primitive.value.fd < 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "EVENT_WAIT event has an invalid descriptor");
      }
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT: {
      const iree_async_semaphore_wait_operation_t* wait =
          (const iree_async_semaphore_wait_operation_t*)operation;
      if (wait->count > INT32_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "SEMAPHORE_WAIT count %" PRIhsz
                                " exceeds the maximum of %d",
                                wait->count, INT32_MAX);
      }
      if (wait->mode != IREE_ASYNC_WAIT_MODE_ALL &&
          wait->mode != IREE_ASYNC_WAIT_MODE_ANY) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "SEMAPHORE_WAIT mode %d is invalid",
                                (int)wait->mode);
      }
      if (wait->count > 0 && (!wait->semaphores || !wait->values)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "SEMAPHORE_WAIT arrays are NULL for a non-empty wait");
      }
      for (iree_host_size_t i = 0; i < wait->count; ++i) {
        if (!wait->semaphores[i]) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "SEMAPHORE_WAIT semaphore %" PRIhsz " is NULL", i);
        }
      }
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL: {
      const iree_async_semaphore_signal_operation_t* signal =
          (const iree_async_semaphore_signal_operation_t*)operation;
      if (signal->count > 0 && (!signal->semaphores || !signal->values)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "SEMAPHORE_SIGNAL arrays are NULL for a non-empty signal");
      }
      for (iree_host_size_t i = 0; i < signal->count; ++i) {
        if (!signal->semaphores[i]) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "SEMAPHORE_SIGNAL semaphore %" PRIhsz " is NULL", i);
        }
      }
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_SEQUENCE:
      return iree_async_sequence_validate(
          (const iree_async_sequence_operation_t*)operation);

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT: {
      const iree_async_socket_accept_operation_t* accept =
          (const iree_async_socket_accept_operation_t*)operation;
      return iree_async_proactor_io_uring_validate_socket(
          proactor, accept->listen_socket, "SOCKET_ACCEPT");
    }

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT: {
      const iree_async_socket_connect_operation_t* connect =
          (const iree_async_socket_connect_operation_t*)operation;
      IREE_RETURN_IF_ERROR(iree_async_proactor_io_uring_validate_socket(
          proactor, connect->socket, "SOCKET_CONNECT"));
      return iree_async_proactor_io_uring_validate_address(&connect->address,
                                                           "SOCKET_CONNECT");
    }

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV: {
      const iree_async_socket_recv_operation_t* recv =
          (const iree_async_socket_recv_operation_t*)operation;
      IREE_RETURN_IF_ERROR(iree_async_proactor_io_uring_validate_socket(
          proactor, recv->socket, "SOCKET_RECV"));
      return iree_async_proactor_io_uring_validate_span_list(
          proactor, recv->buffers, IREE_ASYNC_SOCKET_RECV_MAX_BUFFERS,
          IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, "SOCKET_RECV");
    }

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL: {
      const iree_async_socket_recv_pool_operation_t* recv =
          (const iree_async_socket_recv_pool_operation_t*)operation;
      IREE_RETURN_IF_ERROR(iree_async_proactor_io_uring_validate_socket(
          proactor, recv->socket, "SOCKET_RECV_POOL"));
      if (!recv->pool) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "SOCKET_RECV_POOL pool is NULL");
      }
      const iree_async_region_t* region =
          iree_async_buffer_pool_region(recv->pool);
      if (region->proactor != &proactor->base) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "SOCKET_RECV_POOL region belongs to a different proactor");
      }
      if (region->type != IREE_ASYNC_REGION_TYPE_IOURING) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "SOCKET_RECV_POOL requires an io_uring slab registration");
      }
      if (iree_any_bit_set(operation->flags,
                           IREE_ASYNC_OPERATION_FLAG_MULTISHOT) &&
          region->handles.iouring.buffer_group_id < 0) {
        return iree_make_status(
            IREE_STATUS_UNAVAILABLE,
            "multishot SOCKET_RECV_POOL requires a provided buffer ring");
      }
      if (!iree_any_bit_set(region->access_flags,
                            IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE)) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "SOCKET_RECV_POOL region does not provide WRITE access");
      }
      if (!region->base_ptr) {
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "SOCKET_RECV_POOL requires CPU-accessible memory");
      }
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND: {
      const iree_async_socket_send_operation_t* send =
          (const iree_async_socket_send_operation_t*)operation;
      IREE_RETURN_IF_ERROR(iree_async_proactor_io_uring_validate_socket(
          proactor, send->socket, "SOCKET_SEND"));
      const iree_async_socket_send_flags_t unknown_flags =
          send->send_flags & ~(IREE_ASYNC_SOCKET_SEND_FLAG_MORE |
                               IREE_ASYNC_SOCKET_SEND_FLAG_REPORT_PROGRESS |
                               IREE_ASYNC_SOCKET_SEND_FLAG_NO_ZERO_COPY);
      if (unknown_flags) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "SOCKET_SEND has unknown flags 0x%08X",
                                unknown_flags);
      }
      return iree_async_proactor_io_uring_validate_span_list(
          proactor, send->buffers, IREE_ASYNC_SOCKET_SEND_MAX_BUFFERS,
          IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, "SOCKET_SEND");
    }

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO: {
      const iree_async_socket_sendto_operation_t* send =
          (const iree_async_socket_sendto_operation_t*)operation;
      IREE_RETURN_IF_ERROR(iree_async_proactor_io_uring_validate_socket(
          proactor, send->socket, "SOCKET_SENDTO"));
      IREE_RETURN_IF_ERROR(iree_async_proactor_io_uring_validate_address(
          &send->destination, "SOCKET_SENDTO"));
      const iree_async_socket_send_flags_t unknown_flags =
          send->send_flags & ~(IREE_ASYNC_SOCKET_SEND_FLAG_MORE |
                               IREE_ASYNC_SOCKET_SEND_FLAG_REPORT_PROGRESS |
                               IREE_ASYNC_SOCKET_SEND_FLAG_NO_ZERO_COPY);
      if (unknown_flags) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "SOCKET_SENDTO has unknown flags 0x%08X",
                                unknown_flags);
      }
      return iree_async_proactor_io_uring_validate_span_list(
          proactor, send->buffers, IREE_ASYNC_SOCKET_SENDTO_MAX_BUFFERS,
          IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, "SOCKET_SENDTO");
    }

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM: {
      const iree_async_socket_recvfrom_operation_t* recv =
          (const iree_async_socket_recvfrom_operation_t*)operation;
      IREE_RETURN_IF_ERROR(iree_async_proactor_io_uring_validate_socket(
          proactor, recv->socket, "SOCKET_RECVFROM"));
      return iree_async_proactor_io_uring_validate_span_list(
          proactor, recv->buffers, IREE_ASYNC_SOCKET_RECVFROM_MAX_BUFFERS,
          IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, "SOCKET_RECVFROM");
    }

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE: {
      const iree_async_socket_close_operation_t* close =
          (const iree_async_socket_close_operation_t*)operation;
      return iree_async_proactor_io_uring_validate_socket(
          proactor, close->socket, "SOCKET_CLOSE");
    }

    case IREE_ASYNC_OPERATION_TYPE_FILE_OPEN: {
      const iree_async_file_open_operation_t* open =
          (const iree_async_file_open_operation_t*)operation;
      const iree_async_file_open_flags_t known_open_flags =
          IREE_ASYNC_FILE_OPEN_FLAG_READ | IREE_ASYNC_FILE_OPEN_FLAG_WRITE |
          IREE_ASYNC_FILE_OPEN_FLAG_CREATE |
          IREE_ASYNC_FILE_OPEN_FLAG_TRUNCATE |
          IREE_ASYNC_FILE_OPEN_FLAG_APPEND | IREE_ASYNC_FILE_OPEN_FLAG_DIRECT;
      if (!open->path) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "FILE_OPEN path is NULL");
      }
      if (open->open_flags & ~known_open_flags) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "FILE_OPEN has unknown flags 0x%08X",
                                open->open_flags & ~known_open_flags);
      }
      if (!iree_any_bit_set(open->open_flags,
                            IREE_ASYNC_FILE_OPEN_FLAG_READ |
                                IREE_ASYNC_FILE_OPEN_FLAG_WRITE)) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "FILE_OPEN requires READ, WRITE, or both");
      }
      if (iree_any_bit_set(open->open_flags,
                           IREE_ASYNC_FILE_OPEN_FLAG_CREATE |
                               IREE_ASYNC_FILE_OPEN_FLAG_TRUNCATE |
                               IREE_ASYNC_FILE_OPEN_FLAG_APPEND) &&
          !iree_any_bit_set(open->open_flags,
                            IREE_ASYNC_FILE_OPEN_FLAG_WRITE)) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "FILE_OPEN create, truncate, and append flags require WRITE");
      }
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_FILE_READ: {
      const iree_async_file_read_operation_t* read =
          (const iree_async_file_read_operation_t*)operation;
      IREE_RETURN_IF_ERROR(iree_async_proactor_io_uring_validate_file(
          proactor, read->file, "FILE_READ"));
      return iree_async_proactor_io_uring_validate_span(
          proactor, read->buffer, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
          "FILE_READ");
    }

    case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE: {
      const iree_async_file_write_operation_t* write =
          (const iree_async_file_write_operation_t*)operation;
      IREE_RETURN_IF_ERROR(iree_async_proactor_io_uring_validate_file(
          proactor, write->file, "FILE_WRITE"));
      return iree_async_proactor_io_uring_validate_span(
          proactor, write->buffer, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ,
          "FILE_WRITE");
    }

    case IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE: {
      const iree_async_file_close_operation_t* close =
          (const iree_async_file_close_operation_t*)operation;
      return iree_async_proactor_io_uring_validate_file(proactor, close->file,
                                                        "FILE_CLOSE");
    }

    case IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT: {
      const iree_async_futex_wait_operation_t* wait =
          (const iree_async_futex_wait_operation_t*)operation;
      return iree_async_proactor_io_uring_validate_futex(
          proactor, wait->futex_address, wait->futex_flags, "FUTEX_WAIT");
    }

    case IREE_ASYNC_OPERATION_TYPE_FUTEX_WAKE: {
      const iree_async_futex_wake_operation_t* wake =
          (const iree_async_futex_wake_operation_t*)operation;
      if (wake->wake_count < 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "FUTEX_WAKE wake count must be non-negative");
      }
      return iree_async_proactor_io_uring_validate_futex(
          proactor, wake->futex_address, wake->futex_flags, "FUTEX_WAKE");
    }

    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT: {
      const iree_async_notification_wait_operation_t* wait =
          (const iree_async_notification_wait_operation_t*)operation;
      if (!wait->notification) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "NOTIFICATION_WAIT notification is NULL");
      }
      if (wait->notification->proactor != &proactor->base) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "NOTIFICATION_WAIT notification belongs to a different proactor");
      }
      if (wait->wait_flags &
          ~IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "NOTIFICATION_WAIT has unknown flags 0x%08X",
            wait->wait_flags &
                ~IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN);
      }
      const iree_async_primitive_t primitive =
          wait->notification->platform.io_uring.event.wait_primitive;
      if (primitive.type != IREE_ASYNC_PRIMITIVE_TYPE_FD ||
          primitive.value.fd < 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "notification wait has an invalid descriptor");
      }
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL: {
      const iree_async_notification_signal_operation_t* signal =
          (const iree_async_notification_signal_operation_t*)operation;
      if (!signal->notification) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "NOTIFICATION_SIGNAL notification is NULL");
      }
      if (signal->notification->proactor != &proactor->base) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "NOTIFICATION_SIGNAL notification belongs to a different "
            "proactor");
      }
      if (signal->wake_count < 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "NOTIFICATION_SIGNAL wake count must be non-negative");
      }
      const iree_async_primitive_t primitive =
          signal->notification->platform.io_uring.event.signal_primitive;
      if (primitive.type != IREE_ASYNC_PRIMITIVE_TYPE_FD ||
          primitive.value.fd < 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "notification signal has an invalid descriptor");
      }
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_HANDLE_POLL: {
      const iree_async_handle_poll_operation_t* poll =
          (const iree_async_handle_poll_operation_t*)operation;
      if (poll->primitive.type != IREE_ASYNC_PRIMITIVE_TYPE_FD ||
          poll->primitive.value.fd < 0) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "HANDLE_POLL requires a valid POSIX descriptor");
      }
      if (!poll->events || (poll->events & ~(IREE_ASYNC_POLL_EVENT_IN |
                                             IREE_ASYNC_POLL_EVENT_OUT))) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "HANDLE_POLL requires IN and/or OUT interests");
      }
      return iree_ok_status();
    }

    case IREE_ASYNC_OPERATION_TYPE_MESSAGE: {
      const iree_async_message_operation_t* message =
          (const iree_async_message_operation_t*)operation;
      IREE_RETURN_IF_ERROR(iree_async_message_operation_validate(message));
      if (message->message_flags &
          ~IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT, "MESSAGE has unknown flags 0x%08X",
            message->message_flags &
                ~IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION);
      }
      if (message->target->vtable != &iree_async_proactor_io_uring_vtable) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "MESSAGE target must be an io_uring proactor from the same "
            "backend");
      }
      return iree_ok_status();
    }

    default:
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "io_uring proactor does not support operation type %d",
          (int)operation->type);
  }
}
