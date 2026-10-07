// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/util/local_stream.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

#include "iree/async/operations/scheduling.h"

#if defined(IREE_PLATFORM_WINDOWS)
#include <windows.h>
#elif defined(IREE_ASYNC_HAVE_FD)
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#if defined(IREE_PLATFORM_APPLE)
#include <sys/param.h>
#endif
#endif

typedef enum iree_async_local_stream_state_e {
  IREE_ASYNC_LOCAL_STREAM_STATE_OPEN = 0,
  IREE_ASYNC_LOCAL_STREAM_STATE_FAILED,
  IREE_ASYNC_LOCAL_STREAM_STATE_CLOSING,
  IREE_ASYNC_LOCAL_STREAM_STATE_RETIRING,
  IREE_ASYNC_LOCAL_STREAM_STATE_CLOSED,
} iree_async_local_stream_state_t;

typedef enum iree_async_local_stream_transfer_kind_e {
  IREE_ASYNC_LOCAL_STREAM_TRANSFER_NONE = 0,
  IREE_ASYNC_LOCAL_STREAM_TRANSFER_SEND,
  IREE_ASYNC_LOCAL_STREAM_TRANSFER_RECEIVE,
  IREE_ASYNC_LOCAL_STREAM_TRANSFER_ACCEPT,
} iree_async_local_stream_transfer_kind_t;

enum iree_async_local_stream_flag_bits_e {
  IREE_ASYNC_LOCAL_STREAM_FLAG_SCHEDULED = 1u << 0,
  IREE_ASYNC_LOCAL_STREAM_FLAG_PENDING = 1u << 1,
  IREE_ASYNC_LOCAL_STREAM_FLAG_CANCELLING = 1u << 2,
  IREE_ASYNC_LOCAL_STREAM_FLAG_COMPLETE = 1u << 3,
  IREE_ASYNC_LOCAL_STREAM_FLAG_NEEDS_ACCEPT = 1u << 4,
  IREE_ASYNC_LOCAL_STREAM_FLAG_CANCEL_REQUESTED = 1u << 5,
};
typedef uint32_t iree_async_local_stream_flags_t;

struct iree_async_local_stream_t {
  // Allocator owning this helper and its trailing native scratch.
  iree_allocator_t host_allocator;
  // Retained poll owner; all helper state is serialized on this thread.
  iree_async_proactor_t* proactor;
  // Borrowed exclusively until deactivation and destruction.
  iree_async_primitive_t channel;
  // Maximum resource count accepted by public transfer admission.
  iree_host_size_t handle_capacity;
  // Admission and final ownership state.
  iree_async_local_stream_state_t state;
  // Native/control ownership and terminal notification flags.
  iree_async_local_stream_flags_t flags;
  // One-shot owner work; unlinked while waiting on native readiness.
  iree_async_progress_entry_t progress;
  // Final join notification, borrowed after deactivation admission.
  iree_async_local_stream_deactivated_callback_t deactivated_callback;
  // The single transfer slot, reusable only after its terminal callback begins.
  struct {
    // NONE when no public transfer is owed a callback.
    iree_async_local_stream_transfer_kind_t kind;
    // Borrowed source or destination bytes, selected by kind.
    union {
      // Borrowed immutable send data.
      const uint8_t* source;
      // Borrowed receive destination.
      uint8_t* target;
    } data;
    // Exact application byte extent, excluding native resource encoding.
    iree_host_size_t length;
    // Application bytes transferred so far.
    iree_host_size_t offset;
    // Exact resource extent for this protocol phase.
    iree_host_size_t handle_count;
    // Borrowed source or owned output array, selected by kind.
    union {
      // Borrowed exports kept stable by the caller through its import ACK.
      const iree_async_primitive_t* source;
      // Caller storage containing tentative imports until terminal success.
      iree_async_primitive_t* target;
    } handles;
    // Locally acquired resources, closed on any transfer failure.
    iree_host_size_t received_count;
    // Owned terminal error, transferred to callback only after native
    // retirement.
    iree_status_t status;
    // Borrowed through terminal dispatch.
    iree_async_local_stream_callback_t callback;
  } transfer;
#if defined(IREE_PLATFORM_WINDOWS)
  // One native operation slot, reused only after consuming terminal completion.
  OVERLAPPED overlapped;
  // Persistent event observer registered before issuing any OVERLAPPED I/O.
  iree_async_event_source_t* event_source;
  // Owned PROCESS_DUP_HANDLE reference to the kernel-identified connected peer.
  iree_async_primitive_t peer_process;
  // Trailing little-endian uint64 handle values, independent of application
  // data.
  uint64_t* handle_values;
  // Native resource-prefix bytes already transferred in the current operation.
  iree_host_size_t prefix_offset;
#elif defined(IREE_ASYNC_HAVE_FD)
  // Private one-shot readiness target, joined with its cancellation receipt.
  iree_async_handle_poll_operation_t readiness;
  // Separate ownership of any issued native cancellation key.
  iree_async_cancel_request_t cancellation;
  // Ancillary storage, aligned for cmsghdr and supplied on every recvmsg.
  void* control;
  // Actual native ancillary capacity, including CMSG_SPACE padding.
  iree_host_size_t control_capacity;
#endif
};

static void iree_async_local_stream_schedule(
    iree_async_local_stream_t* stream) {
  if (!iree_any_bit_set(stream->flags,
                        IREE_ASYNC_LOCAL_STREAM_FLAG_SCHEDULED)) {
    stream->flags |= IREE_ASYNC_LOCAL_STREAM_FLAG_SCHEDULED;
    iree_async_proactor_register_progress(stream->proactor, &stream->progress);
  }
}

//===----------------------------------------------------------------------===//
// POSIX ancillary transfer and readiness ownership
//===----------------------------------------------------------------------===//

#if defined(IREE_ASYNC_HAVE_FD) && !defined(IREE_PLATFORM_WINDOWS)

static void iree_async_local_stream_cancel_complete(void* user_data) {
  iree_async_local_stream_t* stream = user_data;
  stream->flags &= ~IREE_ASYNC_LOCAL_STREAM_FLAG_CANCELLING;
  iree_async_local_stream_schedule(stream);
}

static void iree_async_local_stream_ready(void* user_data,
                                          iree_async_operation_t* operation,
                                          iree_status_t status,
                                          iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_async_local_stream_t* stream = user_data;
  stream->flags &= ~IREE_ASYNC_LOCAL_STREAM_FLAG_PENDING;
  stream->transfer.status = iree_status_join(stream->transfer.status, status);
  if (iree_any_bit_set(stream->flags,
                       IREE_ASYNC_LOCAL_STREAM_FLAG_CANCELLING)) {
    iree_async_proactor_cancel_request_target_retired(stream->proactor,
                                                      &stream->cancellation);
  }
  iree_async_local_stream_schedule(stream);
}

static iree_status_t iree_async_local_stream_wait(
    iree_async_local_stream_t* stream) {
  memset(&stream->readiness, 0, sizeof(stream->readiness));
  iree_async_operation_initialize(&stream->readiness.base,
                                  IREE_ASYNC_OPERATION_TYPE_HANDLE_POLL, 0,
                                  iree_async_local_stream_ready, stream);
  stream->readiness.primitive = stream->channel;
  stream->readiness.events =
      stream->transfer.kind == IREE_ASYNC_LOCAL_STREAM_TRANSFER_SEND
          ? IREE_ASYNC_POLL_EVENT_OUT
          : IREE_ASYNC_POLL_EVENT_IN;
  IREE_RETURN_IF_ERROR(iree_async_proactor_submit_one(stream->proactor,
                                                      &stream->readiness.base));
  stream->flags |= IREE_ASYNC_LOCAL_STREAM_FLAG_PENDING;
  return iree_ok_status();
}

// Every installed descriptor is owned or closed before reporting any error.
// Native padding may admit more rights than the logical resource capacity.
static iree_status_t iree_async_local_stream_receive_rights(
    iree_async_local_stream_t* stream, struct msghdr* message) {
  iree_status_t status = iree_ok_status();
  bool overflow = false;
  for (struct cmsghdr* header = CMSG_FIRSTHDR(message); header;
       header = CMSG_NXTHDR(message, header)) {
    if (header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS) {
      continue;
    }
    // Darwin retains the original cmsg_len on truncation. Only complete
    // descriptor values inside returned control storage have been delivered.
    iree_host_size_t available =
        message->msg_controllen -
        (iree_host_size_t)((uint8_t*)header - (uint8_t*)message->msg_control);
    iree_host_size_t length = iree_min(header->cmsg_len, available);
    iree_host_size_t count =
        length >= CMSG_LEN(0) ? (length - CMSG_LEN(0)) / sizeof(int) : 0;
    for (iree_host_size_t i = 0; i < count; ++i) {
      int descriptor = -1;
      memcpy(&descriptor, CMSG_DATA(header) + i * sizeof(int), sizeof(int));
      if (stream->transfer.received_count >= stream->transfer.handle_count) {
        close(descriptor);
        overflow = true;
        continue;
      }
      stream->transfer.handles.target[stream->transfer.received_count++] =
          iree_async_primitive_from_fd(descriptor);
#if !defined(MSG_CMSG_CLOEXEC)
      if (fcntl(descriptor, F_SETFD, FD_CLOEXEC) == -1 &&
          iree_status_is_ok(status)) {
        status = iree_make_status(iree_status_code_from_errno(errno),
                                  "setting received descriptor close-on-exec");
      }
#endif
    }
  }
  if (iree_status_is_ok(status) &&
      (overflow || (message->msg_flags & MSG_CTRUNC))) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "local stream resource bundle exceeds extent");
  }
  return status;
}

static iree_status_t iree_async_local_stream_advance(
    iree_async_local_stream_t* stream) {
  const bool sending =
      stream->transfer.kind == IREE_ASYNC_LOCAL_STREAM_TRANSFER_SEND;
  iree_status_t status = iree_ok_status();
  while (
      iree_status_is_ok(status) &&
      stream->transfer.offset < stream->transfer.length &&
      !iree_any_bit_set(stream->flags, IREE_ASYNC_LOCAL_STREAM_FLAG_PENDING)) {
    struct iovec buffer = {
        .iov_base =
            sending ? (void*)(stream->transfer.data.source +
                              stream->transfer.offset)
                    : stream->transfer.data.target + stream->transfer.offset,
        .iov_len = iree_min(stream->transfer.length - stream->transfer.offset,
                            (iree_host_size_t)SSIZE_MAX),
    };
    struct msghdr message = {.msg_iov = &buffer, .msg_iovlen = 1};
    int flags = 0;
    if (sending) {
#if defined(MSG_NOSIGNAL)
      flags |= MSG_NOSIGNAL;
#endif
      if (stream->transfer.offset == 0 && stream->transfer.handle_count) {
        message.msg_control = stream->control;
        message.msg_controllen =
            CMSG_SPACE(stream->transfer.handle_count * sizeof(int));
        memset(stream->control, 0, message.msg_controllen);
        struct cmsghdr* header = CMSG_FIRSTHDR(&message);
        header->cmsg_level = SOL_SOCKET;
        header->cmsg_type = SCM_RIGHTS;
        header->cmsg_len =
            CMSG_LEN(stream->transfer.handle_count * sizeof(int));
        for (iree_host_size_t i = 0; i < stream->transfer.handle_count; ++i) {
          int descriptor = stream->transfer.handles.source[i].value.fd;
          memcpy(CMSG_DATA(header) + i * sizeof(int), &descriptor, sizeof(int));
        }
      }
    } else {
      message.msg_control = stream->control;
      message.msg_controllen = stream->control_capacity;
#if defined(MSG_CMSG_CLOEXEC)
      flags |= MSG_CMSG_CLOEXEC;
#endif
    }
    ssize_t length = sending
                         ? sendmsg(stream->channel.value.fd, &message, flags)
                         : recvmsg(stream->channel.value.fd, &message, flags);
    if (length < 0) {
      int error = errno;
      if (error == EAGAIN || error == EWOULDBLOCK) {
        status = iree_async_local_stream_wait(stream);
      } else if (error != EINTR) {
        status = iree_make_status(iree_status_code_from_errno(error),
                                  "local stream %s failed",
                                  sending ? "send" : "receive");
      }
    } else {
      if (!sending) {
        status = iree_async_local_stream_receive_rights(stream, &message);
      }
      if (iree_status_is_ok(status) && length == 0) {
        status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "local stream ended before the byte extent");
      }
      stream->transfer.offset += length;
    }
  }
  if (iree_status_is_ok(status) && !sending &&
      stream->transfer.offset == stream->transfer.length &&
      stream->transfer.received_count != stream->transfer.handle_count) {
    status = iree_make_status(IREE_STATUS_DATA_LOSS,
                              "local stream resource bundle is incomplete");
  }
  return status;
}

static iree_status_t iree_async_local_stream_cancel_native(
    iree_async_local_stream_t* stream) {
  if (iree_any_bit_set(stream->flags, IREE_ASYNC_LOCAL_STREAM_FLAG_PENDING) &&
      !iree_any_bit_set(stream->flags,
                        IREE_ASYNC_LOCAL_STREAM_FLAG_CANCEL_REQUESTED)) {
    iree_async_cancel_request_initialize(
        (iree_async_cancel_callback_t){iree_async_local_stream_cancel_complete,
                                       stream},
        &stream->cancellation);
    IREE_RETURN_IF_ERROR(iree_async_proactor_request_cancel(
        stream->proactor, &stream->readiness.base, &stream->cancellation));
    stream->flags |= IREE_ASYNC_LOCAL_STREAM_FLAG_CANCELLING |
                     IREE_ASYNC_LOCAL_STREAM_FLAG_CANCEL_REQUESTED;
  }
  return iree_ok_status();
}

#endif  // IREE_ASYNC_HAVE_FD && !IREE_PLATFORM_WINDOWS

//===----------------------------------------------------------------------===//
// Windows overlapped transfer and persistent observation
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_WINDOWS)

static iree_status_t iree_async_local_stream_open_peer(
    iree_async_local_stream_t* stream, DWORD pipe_flags) {
  ULONG process_id = 0;
  BOOL queried =
      (pipe_flags & PIPE_SERVER_END)
          ? GetNamedPipeClientProcessId(
                (HANDLE)stream->channel.value.win32_handle, &process_id)
          : GetNamedPipeServerProcessId(
                (HANDLE)stream->channel.value.win32_handle, &process_id);
  if (!queried) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "querying local stream peer process");
  }
  HANDLE process = OpenProcess(PROCESS_DUP_HANDLE, FALSE, process_id);
  if (!process) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "opening local stream peer for resource transfer");
  }
  stream->peer_process =
      iree_async_primitive_from_win32_handle((uintptr_t)process);
  return iree_ok_status();
}

static iree_status_t iree_async_local_stream_native_result(
    iree_async_local_stream_t* stream, DWORD error, DWORD length) {
  if (error != ERROR_SUCCESS) {
    if (error == ERROR_OPERATION_ABORTED) {
      return iree_status_from_code(IREE_STATUS_CANCELLED);
    }
    if (error == ERROR_BROKEN_PIPE || error == ERROR_PIPE_NOT_CONNECTED) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "local stream ended before the byte extent");
    }
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "local stream native operation failed (%lu)",
                            error);
  }
  if (stream->transfer.kind == IREE_ASYNC_LOCAL_STREAM_TRANSFER_ACCEPT) {
    IREE_RETURN_IF_ERROR(
        iree_async_local_stream_open_peer(stream, PIPE_SERVER_END));
    stream->flags &= ~IREE_ASYNC_LOCAL_STREAM_FLAG_NEEDS_ACCEPT;
    return iree_ok_status();
  }
  if (length == 0) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "local stream made no byte progress");
  }
  if (stream->prefix_offset <
      stream->transfer.handle_count * sizeof(uint64_t)) {
    stream->prefix_offset += length;
  } else {
    stream->transfer.offset += length;
  }
  return iree_ok_status();
}

static void iree_async_local_stream_observe(void* user_data,
                                            iree_async_event_source_t* source,
                                            iree_async_poll_events_t events) {
  (void)source;
  (void)events;
  iree_async_local_stream_t* stream = user_data;
  // The event belongs to this helper and is still observed. Reset before
  // querying the current operation so a later native completion cannot be lost.
  BOOL reset = ResetEvent(stream->overlapped.hEvent);
  IREE_ASSERT(reset, "live local stream event could not be reset");
  (void)reset;
  if (!iree_any_bit_set(stream->flags, IREE_ASYNC_LOCAL_STREAM_FLAG_PENDING)) {
    return;
  }
  DWORD length = 0;
  BOOL completed =
      GetOverlappedResult((HANDLE)stream->channel.value.win32_handle,
                          &stream->overlapped, &length, FALSE);
  DWORD error = completed ? ERROR_SUCCESS : GetLastError();
  if (error == ERROR_IO_INCOMPLETE) {
    return;
  }
  stream->flags &= ~(IREE_ASYNC_LOCAL_STREAM_FLAG_PENDING |
                     IREE_ASYNC_LOCAL_STREAM_FLAG_CANCELLING);
  stream->transfer.status =
      iree_async_local_stream_native_result(stream, error, length);
  iree_async_local_stream_schedule(stream);
}

static iree_status_t iree_async_local_stream_import_handles(
    iree_async_local_stream_t* stream) {
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status) &&
         stream->transfer.received_count < stream->transfer.handle_count) {
    iree_host_size_t index = stream->transfer.received_count;
    uint64_t value = iree_unaligned_load_le_u64(&stream->handle_values[index]);
    HANDLE handle = NULL;
    // The wire representation canonically zero-extends the significant low
    // 32 bits. Process/thread pseudo-handles are not owned exported resources.
    if (!value || value > UINT32_MAX ||
        value == (uint32_t)(uintptr_t)GetCurrentProcess() ||
        value == (uint32_t)(uintptr_t)GetCurrentThread()) {
      status =
          iree_make_status(IREE_STATUS_DATA_LOSS,
                           "local stream received an invalid source handle");
    } else if (!DuplicateHandle((HANDLE)stream->peer_process.value.win32_handle,
                                (HANDLE)(uintptr_t)value, GetCurrentProcess(),
                                &handle, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
      status =
          iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                           "importing local stream resource");
    } else {
      stream->transfer.handles.target[index] =
          iree_async_primitive_from_win32_handle((uintptr_t)handle);
      ++stream->transfer.received_count;
    }
  }
  return status;
}

static iree_status_t iree_async_local_stream_advance(
    iree_async_local_stream_t* stream) {
  iree_status_t status = iree_ok_status();
  bool accepting =
      stream->transfer.kind == IREE_ASYNC_LOCAL_STREAM_TRANSFER_ACCEPT;
  while (
      iree_status_is_ok(status) &&
      (accepting ? iree_any_bit_set(stream->flags,
                                    IREE_ASYNC_LOCAL_STREAM_FLAG_NEEDS_ACCEPT)
                 : stream->transfer.offset < stream->transfer.length) &&
      !iree_any_bit_set(stream->flags, IREE_ASYNC_LOCAL_STREAM_FLAG_PENDING)) {
    HANDLE event = stream->overlapped.hEvent;
    BOOL reset = ResetEvent(event);
    IREE_ASSERT(reset, "live local stream event could not be reset");
    (void)reset;
    memset(&stream->overlapped, 0, sizeof(stream->overlapped));
    stream->overlapped.hEvent = event;
    HANDLE channel = (HANDLE)stream->channel.value.win32_handle;
    DWORD length = 0;
    BOOL completed = FALSE;
    if (accepting) {
      completed = ConnectNamedPipe(channel, &stream->overlapped);
    } else {
      bool sending =
          stream->transfer.kind == IREE_ASYNC_LOCAL_STREAM_TRANSFER_SEND;
      iree_host_size_t prefix_length =
          stream->transfer.handle_count * sizeof(uint64_t);
      void* data = NULL;
      iree_host_size_t remaining = 0;
      if (stream->prefix_offset < prefix_length) {
        data = (uint8_t*)stream->handle_values + stream->prefix_offset;
        remaining = prefix_length - stream->prefix_offset;
      } else {
        if (!sending) {
          status = iree_async_local_stream_import_handles(stream);
        }
        data = sending ? (void*)(stream->transfer.data.source +
                                 stream->transfer.offset)
                       : stream->transfer.data.target + stream->transfer.offset;
        remaining = stream->transfer.length - stream->transfer.offset;
      }
      if (iree_status_is_ok(status)) {
        DWORD request_length =
            (DWORD)iree_min(remaining, (iree_host_size_t)MAXDWORD);
        completed = sending ? WriteFile(channel, data, request_length, &length,
                                        &stream->overlapped)
                            : ReadFile(channel, data, request_length, &length,
                                       &stream->overlapped);
      }
    }
    if (iree_status_is_ok(status)) {
      DWORD error = completed ? ERROR_SUCCESS : GetLastError();
      if (accepting && error == ERROR_PIPE_CONNECTED) {
        error = ERROR_SUCCESS;
      }
      if (error == ERROR_IO_PENDING) {
        stream->flags |= IREE_ASYNC_LOCAL_STREAM_FLAG_PENDING;
      } else {
        status = iree_async_local_stream_native_result(stream, error, length);
      }
    }
  }
  return status;
}

static iree_status_t iree_async_local_stream_cancel_native(
    iree_async_local_stream_t* stream) {
  if (iree_any_bit_set(stream->flags, IREE_ASYNC_LOCAL_STREAM_FLAG_PENDING) &&
      !iree_any_bit_set(stream->flags,
                        IREE_ASYNC_LOCAL_STREAM_FLAG_CANCEL_REQUESTED)) {
    if (!CancelIoEx((HANDLE)stream->channel.value.win32_handle,
                    &stream->overlapped)) {
      DWORD error = GetLastError();
      if (error != ERROR_NOT_FOUND) {
        return iree_make_status(iree_status_code_from_win32_error(error),
                                "cancelling local stream I/O (%lu)", error);
      }
    }
    // Cancellation admission is not native completion. The persistent observer
    // returns OVERLAPPED and buffer ownership, even when cancellation loses.
    stream->flags |= IREE_ASYNC_LOCAL_STREAM_FLAG_CANCELLING |
                     IREE_ASYNC_LOCAL_STREAM_FLAG_CANCEL_REQUESTED;
  }
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_WINDOWS

//===----------------------------------------------------------------------===//
// Shared transfer admission and final join
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_WINDOWS)
static void iree_async_local_stream_unobserved(void* user_data,
                                               iree_status_t status) {
  iree_async_local_stream_t* stream = user_data;
  if (!iree_status_is_ok(status)) {
    // Native observer reachability remains unproven. Complete any admitted
    // transfer with the cleanup failure, but withhold deactivation so the
    // stream, channel, and proactor remain alive.
    if (stream->transfer.kind != IREE_ASYNC_LOCAL_STREAM_TRANSFER_NONE) {
      stream->transfer.status =
          iree_status_join(stream->transfer.status, status);
      stream->flags |= IREE_ASYNC_LOCAL_STREAM_FLAG_COMPLETE;
      iree_async_local_stream_schedule(stream);
    } else {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
    }
    return;
  }
  stream->state = IREE_ASYNC_LOCAL_STREAM_STATE_CLOSED;
  iree_async_local_stream_schedule(stream);
}
#endif

static void iree_async_local_stream_removed(void* user_data) {
  iree_async_local_stream_t* stream = user_data;
  stream->flags &= ~IREE_ASYNC_LOCAL_STREAM_FLAG_SCHEDULED;
  iree_async_local_stream_callback_t callback = {0};
  iree_status_t status = iree_ok_status();
  if (iree_any_bit_set(stream->flags, IREE_ASYNC_LOCAL_STREAM_FLAG_COMPLETE)) {
    stream->flags &= ~IREE_ASYNC_LOCAL_STREAM_FLAG_COMPLETE;
    callback = stream->transfer.callback;
    status = stream->transfer.status;
    if (!iree_status_is_ok(status)) {
      if (stream->transfer.kind == IREE_ASYNC_LOCAL_STREAM_TRANSFER_RECEIVE) {
        for (iree_host_size_t i = 0; i < stream->transfer.received_count; ++i) {
          iree_async_primitive_close(&stream->transfer.handles.target[i]);
        }
      }
      if (stream->state == IREE_ASYNC_LOCAL_STREAM_STATE_OPEN) {
        stream->state = IREE_ASYNC_LOCAL_STREAM_STATE_FAILED;
      }
    }
    memset(&stream->transfer, 0, sizeof(stream->transfer));
  }
  iree_async_local_stream_deactivated_callback_t deactivated = {0};
  if (stream->state == IREE_ASYNC_LOCAL_STREAM_STATE_CLOSED) {
    deactivated = stream->deactivated_callback;
  }
  // Both callbacks may release their own contexts; the final callback may also
  // free the helper. No helper access follows either callback.
  if (callback.fn) {
    callback.fn(callback.user_data, status);
  }
  if (deactivated.fn) {
    deactivated.fn(deactivated.user_data);
  }
}

static iree_status_t iree_async_local_stream_progress(
    void* user_data, iree_host_size_t* out_completed_count) {
  iree_async_local_stream_t* stream = user_data;
  if (stream->state == IREE_ASYNC_LOCAL_STREAM_STATE_CLOSING) {
#if defined(IREE_PLATFORM_WINDOWS) || defined(IREE_ASYNC_HAVE_FD)
    IREE_RETURN_IF_ERROR(iree_async_local_stream_cancel_native(stream));
#endif
    if (!iree_any_bit_set(stream->flags,
                          IREE_ASYNC_LOCAL_STREAM_FLAG_PENDING |
                              IREE_ASYNC_LOCAL_STREAM_FLAG_CANCELLING)) {
      if (stream->transfer.kind != IREE_ASYNC_LOCAL_STREAM_TRANSFER_NONE) {
        bool complete =
            stream->transfer.kind == IREE_ASYNC_LOCAL_STREAM_TRANSFER_ACCEPT
                ? !iree_any_bit_set(stream->flags,
                                    IREE_ASYNC_LOCAL_STREAM_FLAG_NEEDS_ACCEPT)
                : stream->transfer.offset == stream->transfer.length;
        if (!complete && iree_status_is_ok(stream->transfer.status)) {
          stream->transfer.status =
              iree_status_from_code(IREE_STATUS_CANCELLED);
        }
        stream->flags |= IREE_ASYNC_LOCAL_STREAM_FLAG_COMPLETE;
        ++*out_completed_count;
      }
#if defined(IREE_PLATFORM_WINDOWS)
      // I/O retirement does not imply retirement of its persistent observer.
      // Clear admission before unregister, which may complete inline.
      stream->state = IREE_ASYNC_LOCAL_STREAM_STATE_RETIRING;
      iree_async_event_source_t* event_source = stream->event_source;
      stream->event_source = NULL;
      iree_async_proactor_unregister_event_source(
          stream->proactor, event_source,
          (iree_async_event_source_unregistered_callback_t){
              iree_async_local_stream_unobserved, stream});
#else
      stream->state = IREE_ASYNC_LOCAL_STREAM_STATE_CLOSED;
#endif
      stream->flags &= ~(IREE_ASYNC_LOCAL_STREAM_FLAG_NEEDS_ACCEPT |
                         IREE_ASYNC_LOCAL_STREAM_FLAG_CANCEL_REQUESTED);
    }
  } else if (stream->state < IREE_ASYNC_LOCAL_STREAM_STATE_CLOSING) {
#if defined(IREE_PLATFORM_WINDOWS) || defined(IREE_ASYNC_HAVE_FD)
    if (iree_status_is_ok(stream->transfer.status)) {
      stream->transfer.status = iree_async_local_stream_advance(stream);
    }
#endif
    if (!iree_any_bit_set(stream->flags,
                          IREE_ASYNC_LOCAL_STREAM_FLAG_PENDING)) {
      stream->flags |= IREE_ASYNC_LOCAL_STREAM_FLAG_COMPLETE;
      ++*out_completed_count;
    }
  }
  if (stream->state == IREE_ASYNC_LOCAL_STREAM_STATE_CLOSED) {
    ++*out_completed_count;
  }
  stream->progress.remove_requested = true;
  return iree_ok_status();
}

iree_status_t iree_async_local_stream_create(
    iree_async_proactor_t* proactor, iree_async_primitive_t channel,
    iree_host_size_t handle_capacity, iree_allocator_t host_allocator,
    iree_async_local_stream_t** out_stream) {
  *out_stream = NULL;
  iree_host_size_t scratch_length = 0;
  iree_host_size_t scratch_alignment = sizeof(uint64_t);
  if (handle_capacity > UINT32_MAX / sizeof(uint64_t)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "local stream resource capacity exceeds native limits");
  }
#if defined(IREE_PLATFORM_WINDOWS)
  if (channel.type != IREE_ASYNC_PRIMITIVE_TYPE_WIN32_HANDLE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local stream requires a pipe HANDLE");
  }
  DWORD pipe_flags = 0;
  if (!GetNamedPipeInfo((HANDLE)channel.value.win32_handle, &pipe_flags, NULL,
                        NULL, NULL)) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "querying local pipe");
  }
  if (pipe_flags & PIPE_TYPE_MESSAGE) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local stream requires a byte pipe");
  }
  scratch_length = handle_capacity * sizeof(uint64_t);
#elif defined(IREE_ASYNC_HAVE_FD)
  if (channel.type != IREE_ASYNC_PRIMITIVE_TYPE_FD) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local stream requires a descriptor");
  }
  int descriptor = channel.value.fd;
  int descriptor_flags = fcntl(descriptor, F_GETFL);
  if (descriptor_flags == -1) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "querying local stream descriptor");
  }
  int socket_type = 0;
  socklen_t socket_type_length = sizeof(socket_type);
  struct sockaddr_storage peer_address;
  socklen_t peer_address_length = sizeof(peer_address);
  if (getsockopt(descriptor, SOL_SOCKET, SO_TYPE, &socket_type,
                 &socket_type_length) == -1 ||
      getpeername(descriptor, (struct sockaddr*)&peer_address,
                  &peer_address_length) == -1) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "querying local stream socket");
  }
  if (!(descriptor_flags & O_NONBLOCK) || socket_type != SOCK_STREAM ||
      peer_address.ss_family != AF_UNIX) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local stream requires a connected nonblocking Unix stream socket");
  }
#if defined(SO_NOSIGPIPE)
  int no_sigpipe = 1;
  if (setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe,
                 sizeof(no_sigpipe)) == -1) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "disabling local stream SIGPIPE");
  }
#endif
  scratch_length = CMSG_SPACE(handle_capacity * sizeof(int));
#if defined(IREE_PLATFORM_APPLE)
  // Darwin externalizes the entire rights record before copying control data
  // to userspace; truncation would lose installed descriptor identities. Its
  // sockargs bounds a control record by MCLBYTES, and soreceive stops at the
  // next control record. Capture that native extent even for small bundles.
  scratch_length = iree_max(scratch_length, (iree_host_size_t)MCLBYTES);
#endif
  scratch_alignment = iree_alignof(struct cmsghdr);
#else
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "native local streams are not available");
#endif
  iree_host_size_t total_size = 0;
  iree_host_size_t scratch_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(iree_async_local_stream_t), &total_size,
      IREE_STRUCT_FIELD_ALIGNED(scratch_length, uint8_t, scratch_alignment,
                                &scratch_offset)));
  iree_async_local_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&stream));
  memset(stream, 0, total_size);
  stream->host_allocator = host_allocator;
  stream->proactor = proactor;
  stream->channel = channel;
  stream->handle_capacity = handle_capacity;
  stream->progress.fn = iree_async_local_stream_progress;
  stream->progress.on_remove = iree_async_local_stream_removed;
  stream->progress.user_data = stream;
  iree_status_t status = iree_ok_status();
#if defined(IREE_PLATFORM_WINDOWS)
  stream->handle_values = (uint64_t*)((uint8_t*)stream + scratch_offset);
  if (pipe_flags & PIPE_SERVER_END) {
    stream->flags |= IREE_ASYNC_LOCAL_STREAM_FLAG_NEEDS_ACCEPT;
  } else {
    status = iree_async_local_stream_open_peer(stream, pipe_flags);
  }
  if (iree_status_is_ok(status)) {
    stream->overlapped.hEvent = CreateEventW(NULL, TRUE, FALSE, NULL);
    if (!stream->overlapped.hEvent) {
      status =
          iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                           "creating local stream event");
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_register_event_source(
        proactor,
        iree_async_primitive_from_win32_handle(
            (uintptr_t)stream->overlapped.hEvent),
        (iree_async_event_source_callback_t){iree_async_local_stream_observe,
                                             stream},
        &stream->event_source);
  }
#elif defined(IREE_ASYNC_HAVE_FD)
  stream->control = (uint8_t*)stream + scratch_offset;
  stream->control_capacity = scratch_length;
#endif
  if (iree_status_is_ok(status)) {
    iree_async_proactor_retain(proactor);
    *out_stream = stream;
  } else {
#if defined(IREE_PLATFORM_WINDOWS)
    if (stream->overlapped.hEvent) {
      CloseHandle(stream->overlapped.hEvent);
    }
    iree_async_primitive_close(&stream->peer_process);
#endif
    iree_allocator_free(host_allocator, stream);
  }
  return status;
}

static iree_status_t iree_async_local_stream_validate_transfer(
    iree_async_local_stream_t* stream, iree_host_size_t length,
    const void* data, iree_host_size_t handle_count, const void* handles,
    iree_async_local_stream_callback_t callback) {
  if (stream->state != IREE_ASYNC_LOCAL_STREAM_STATE_OPEN ||
      stream->transfer.kind != IREE_ASYNC_LOCAL_STREAM_TRANSFER_NONE ||
      iree_any_bit_set(stream->flags,
                       IREE_ASYNC_LOCAL_STREAM_FLAG_NEEDS_ACCEPT)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "local stream is not ready for a transfer");
  }
  if (!length || !data || !callback.fn || (handle_count && !handles)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "local stream transfer requires data, resources, and callback");
  }
  if (handle_count > stream->handle_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "local stream resource extent exceeds capacity");
  }
  return iree_ok_status();
}

iree_status_t iree_async_local_stream_send(
    iree_async_local_stream_t* stream, iree_const_byte_span_t data,
    iree_host_size_t handle_count, const iree_async_primitive_t* handles,
    iree_async_local_stream_callback_t callback) {
  IREE_RETURN_IF_ERROR(iree_async_local_stream_validate_transfer(
      stream, data.data_length, data.data, handle_count, handles, callback));
  for (iree_host_size_t i = 0; i < handle_count; ++i) {
    if (handles[i].type != stream->channel.type) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "local stream resource has incompatible primitive type");
    }
  }
  stream->transfer.kind = IREE_ASYNC_LOCAL_STREAM_TRANSFER_SEND;
  stream->transfer.data.source = data.data;
  stream->transfer.length = data.data_length;
  stream->transfer.handles.source = handles;
  stream->transfer.handle_count = handle_count;
  stream->transfer.callback = callback;
#if defined(IREE_PLATFORM_WINDOWS)
  stream->prefix_offset = 0;
  for (iree_host_size_t i = 0; i < handle_count; ++i) {
    iree_unaligned_store_le_u64(&stream->handle_values[i],
                                (uint32_t)handles[i].value.win32_handle);
  }
#endif
  iree_async_local_stream_schedule(stream);
  return iree_ok_status();
}

iree_status_t iree_async_local_stream_receive(
    iree_async_local_stream_t* stream, iree_byte_span_t data,
    iree_host_size_t handle_count, iree_async_primitive_t* out_handles,
    iree_async_local_stream_callback_t callback) {
  IREE_RETURN_IF_ERROR(iree_async_local_stream_validate_transfer(
      stream, data.data_length, data.data, handle_count, out_handles,
      callback));
  for (iree_host_size_t i = 0; i < handle_count; ++i) {
    out_handles[i] = iree_async_primitive_none();
  }
  stream->transfer.kind = IREE_ASYNC_LOCAL_STREAM_TRANSFER_RECEIVE;
  stream->transfer.data.target = data.data;
  stream->transfer.length = data.data_length;
  stream->transfer.handles.target = out_handles;
  stream->transfer.handle_count = handle_count;
  stream->transfer.callback = callback;
#if defined(IREE_PLATFORM_WINDOWS)
  stream->prefix_offset = 0;
#endif
  iree_async_local_stream_schedule(stream);
  return iree_ok_status();
}

iree_status_t iree_async_local_stream_deactivate(
    iree_async_local_stream_t* stream,
    iree_async_local_stream_deactivated_callback_t callback) {
  if (!callback.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local stream deactivation requires a callback");
  }
  if (stream->state >= IREE_ASYNC_LOCAL_STREAM_STATE_CLOSING) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "local stream is already deactivating");
  }
  stream->state = IREE_ASYNC_LOCAL_STREAM_STATE_CLOSING;
  stream->deactivated_callback = callback;
  iree_async_local_stream_schedule(stream);
  return iree_ok_status();
}

void iree_async_local_stream_destroy(iree_async_local_stream_t* stream) {
  if (!stream) {
    return;
  }
  IREE_ASSERT(
      stream->state == IREE_ASYNC_LOCAL_STREAM_STATE_CLOSED && !stream->flags,
      "local stream freed before the deactivation join");
#if defined(IREE_PLATFORM_WINDOWS)
  CloseHandle(stream->overlapped.hEvent);
  iree_async_primitive_close(&stream->peer_process);
#endif
  iree_async_proactor_release(stream->proactor);
  iree_allocator_free(stream->host_allocator, stream);
}

//===----------------------------------------------------------------------===//
// Windows local named-pipe acquisition
//===----------------------------------------------------------------------===//

#if defined(IREE_PLATFORM_WINDOWS)

static iree_status_t iree_async_local_stream_pipe_name(iree_string_view_t name,
                                                       wchar_t out_path[256]) {
  const wchar_t prefix[] = L"\\\\.\\pipe\\";
  const int prefix_length = IREE_ARRAYSIZE(prefix) - 1;
  if (!name.size || name.size > INT_MAX || memchr(name.data, 0, name.size) ||
      memchr(name.data, '\\', name.size) || memchr(name.data, '/', name.size)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "expected a nonempty local pipe name without path separators");
  }
  memcpy(out_path, prefix, sizeof(prefix));
  int length = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, name.data,
                                   (int)name.size, out_path + prefix_length,
                                   255 - prefix_length);
  if (!length) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "converting local pipe name");
  }
  out_path[prefix_length + length] = 0;
  return iree_ok_status();
}

iree_status_t iree_async_local_stream_pipe_create(
    iree_string_view_t name, uint32_t instance_limit,
    iree_async_local_stream_pipe_flags_t flags,
    iree_async_primitive_t* out_channel) {
  *out_channel = iree_async_primitive_none();
  if (!instance_limit || instance_limit > PIPE_UNLIMITED_INSTANCES ||
      (flags & ~IREE_ASYNC_LOCAL_STREAM_PIPE_FLAG_FIRST_INSTANCE)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid local pipe instance options");
  }
  wchar_t path[256];
  IREE_RETURN_IF_ERROR(iree_async_local_stream_pipe_name(name, path));
  DWORD open_mode = PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED;
  if (flags & IREE_ASYNC_LOCAL_STREAM_PIPE_FLAG_FIRST_INSTANCE) {
    open_mode |= FILE_FLAG_FIRST_PIPE_INSTANCE;
  }
  HANDLE pipe = CreateNamedPipeW(path, open_mode,
                                 PIPE_TYPE_BYTE | PIPE_READMODE_BYTE |
                                     PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
                                 instance_limit, 4096, 4096, 0, NULL);
  if (pipe == INVALID_HANDLE_VALUE) {
    return iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                            "creating local pipe instance");
  }
  *out_channel = iree_async_primitive_from_win32_handle((uintptr_t)pipe);
  return iree_ok_status();
}

iree_status_t iree_async_local_stream_pipe_open(
    iree_string_view_t name, iree_async_primitive_t* out_channel) {
  *out_channel = iree_async_primitive_none();
  wchar_t path[256];
  IREE_RETURN_IF_ERROR(iree_async_local_stream_pipe_name(name, path));
  HANDLE pipe = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, NULL,
                            OPEN_EXISTING, FILE_FLAG_OVERLAPPED, NULL);
  if (pipe == INVALID_HANDLE_VALUE) {
    DWORD error = GetLastError();
    return iree_make_status(error == ERROR_PIPE_BUSY
                                ? IREE_STATUS_UNAVAILABLE
                                : iree_status_code_from_win32_error(error),
                            "opening local pipe (%lu)", error);
  }
  *out_channel = iree_async_primitive_from_win32_handle((uintptr_t)pipe);
  return iree_ok_status();
}

iree_status_t iree_async_local_stream_pipe_accept(
    iree_async_local_stream_t* stream,
    iree_async_local_stream_callback_t callback) {
  if (!callback.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "local pipe accept requires a callback");
  }
  if (stream->state != IREE_ASYNC_LOCAL_STREAM_STATE_OPEN ||
      stream->transfer.kind != IREE_ASYNC_LOCAL_STREAM_TRANSFER_NONE ||
      !iree_any_bit_set(stream->flags,
                        IREE_ASYNC_LOCAL_STREAM_FLAG_NEEDS_ACCEPT)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "local stream is not awaiting pipe acceptance");
  }
  stream->transfer.kind = IREE_ASYNC_LOCAL_STREAM_TRANSFER_ACCEPT;
  stream->transfer.callback = callback;
  iree_async_local_stream_schedule(stream);
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_WINDOWS
