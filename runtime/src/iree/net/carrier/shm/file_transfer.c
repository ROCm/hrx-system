// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/file_transfer.h"

#include <inttypes.h>
#include <string.h>

#include "iree/base/threading/mutex.h"

#if IREE_FILE_IO_ENABLE && !defined(IREE_PLATFORM_WINDOWS)
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>
#endif  // IREE_FILE_IO_ENABLE && !IREE_PLATFORM_WINDOWS

#if IREE_FILE_IO_ENABLE && defined(IREE_PLATFORM_WINDOWS)
#include <fcntl.h>
#include <io.h>
#include <windows.h>

#include "iree/net/carrier/shm/pipe_peer_win32.h"
#endif  // IREE_FILE_IO_ENABLE && IREE_PLATFORM_WINDOWS

#define IREE_NET_SHM_FILE_TRANSFER_MAGIC 0x54464853u
#define IREE_NET_SHM_FILE_TRANSFER_VERSION 1u

#if IREE_FILE_IO_ENABLE
#define IREE_NET_SHM_FILE_TRANSFER_INVALID_FD (-1)
#define IREE_NET_SHM_FILE_TRANSFER_SUPPORTED_ACCESS \
  (IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE)
#define IREE_NET_SHM_FILE_TRANSFER_SUPPORTED_MODE \
  (IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE | IREE_IO_FILE_MODE_ASYNC)
#endif  // IREE_FILE_IO_ENABLE

typedef uint8_t iree_net_shm_file_transfer_opcode_t;
enum iree_net_shm_file_transfer_opcode_e {
  IREE_NET_SHM_FILE_TRANSFER_OPCODE_FILE = 1u,
  IREE_NET_SHM_FILE_TRANSFER_OPCODE_CANCEL = 2u,
};

typedef struct iree_net_shm_file_transfer_payload_t {
  // POSIX: opaque sideband transfer ID. Windows: peer-process HANDLE value.
  uint64_t id;
  // iree_io_file_mode_t used when the peer wraps the received handle.
  uint64_t mode;
  // iree_io_file_access_t allowed through the received handle.
  uint32_t access;
  // Reserved for future platform-specific handle metadata.
  uint32_t reserved;
} iree_net_shm_file_transfer_payload_t;
static_assert(sizeof(iree_net_shm_file_transfer_payload_t) == 24, "");

typedef struct iree_net_shm_file_transfer_sideband_header_t {
  // IREE_NET_SHM_FILE_TRANSFER_MAGIC.
  uint32_t magic;
  // IREE_NET_SHM_FILE_TRANSFER_VERSION.
  uint16_t version;
  // iree_net_shm_file_transfer_opcode_t.
  uint8_t opcode;
  // Reserved for future flags.
  uint8_t reserved;
  // Opaque sideband transfer ID.
  uint64_t id;
} iree_net_shm_file_transfer_sideband_header_t;
static_assert(sizeof(iree_net_shm_file_transfer_sideband_header_t) == 16, "");

typedef struct iree_net_shm_file_transfer_pending_t {
  // Next pending transfer in the out-of-order import list.
  struct iree_net_shm_file_transfer_pending_t* next;
  // Opaque sideband transfer ID.
  uint64_t id;
#if IREE_FILE_IO_ENABLE && !defined(IREE_PLATFORM_WINDOWS)
  // POSIX fd received from the peer and owned by this pending entry.
  int fd;
#endif  // IREE_FILE_IO_ENABLE && !IREE_PLATFORM_WINDOWS
} iree_net_shm_file_transfer_pending_t;

struct iree_net_shm_file_transfer_t {
  // Host allocator used for transfer bookkeeping.
  iree_allocator_t host_allocator;
  // Owned bootstrap channel used as the sideband transport.
  iree_async_primitive_t channel;
#if IREE_FILE_IO_ENABLE && !defined(IREE_PLATFORM_WINDOWS)
  // Guards channel send/recv, id allocation, and pending_file_transfers.
  iree_slim_mutex_t mutex;
  // Next nonzero transfer ID to assign.
  uint64_t next_id;
  // Out-of-order file transfers received before their control payload.
  iree_net_shm_file_transfer_pending_t* pending_file_transfers;
  // Partially received sideband header bytes, for stream socket framing.
  uint8_t partial_header[sizeof(iree_net_shm_file_transfer_sideband_header_t)];
  // Number of valid bytes in partial_header.
  iree_host_size_t partial_header_length;
  // FD received with a partial header; -1 if none is pending.
  int partial_fd;
#endif  // IREE_FILE_IO_ENABLE && !IREE_PLATFORM_WINDOWS
#if IREE_FILE_IO_ENABLE && defined(IREE_PLATFORM_WINDOWS)
  // Owned peer process HANDLE used to duplicate file HANDLEs.
  iree_async_primitive_t peer_process;
#endif  // IREE_FILE_IO_ENABLE && IREE_PLATFORM_WINDOWS
};

#if IREE_FILE_IO_ENABLE

static iree_io_file_mode_t iree_net_shm_file_transfer_mode_from_handle(
    iree_io_file_handle_t* file_handle) {
  const iree_io_file_access_t access = iree_io_file_handle_access(file_handle);
  iree_io_file_mode_t mode = IREE_IO_FILE_MODE_NONE;
  if (iree_all_bits_set(access, IREE_IO_FILE_ACCESS_READ)) {
    mode |= IREE_IO_FILE_MODE_READ;
  }
  if (iree_all_bits_set(access, IREE_IO_FILE_ACCESS_WRITE)) {
    mode |= IREE_IO_FILE_MODE_WRITE;
  }
  if (iree_io_file_handle_uses_async_io(file_handle)) {
    mode |= IREE_IO_FILE_MODE_ASYNC;
  }
  return mode;
}

static iree_status_t iree_net_shm_file_transfer_validate_payload(
    iree_const_byte_span_t transfer_payload,
    iree_net_shm_file_transfer_payload_t* out_payload) {
  if (transfer_payload.data_length != sizeof(*out_payload)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SHM file transfer payload length must be %" PRIhsz
                            " bytes",
                            (iree_host_size_t)sizeof(*out_payload));
  }
  memcpy(out_payload, transfer_payload.data, sizeof(*out_payload));
  if (out_payload->id == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SHM file transfer handle/id must be nonzero");
  }
  if (out_payload->reserved != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "SHM file transfer reserved field must be 0");
  }
  if (out_payload->access & ~IREE_NET_SHM_FILE_TRANSFER_SUPPORTED_ACCESS) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SHM file transfer access has unsupported bits 0x%08x",
        out_payload->access & ~IREE_NET_SHM_FILE_TRANSFER_SUPPORTED_ACCESS);
  }
  if (out_payload->mode & ~IREE_NET_SHM_FILE_TRANSFER_SUPPORTED_MODE) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SHM file transfer mode has unsupported bits 0x%016" PRIx64,
        out_payload->mode & ~IREE_NET_SHM_FILE_TRANSFER_SUPPORTED_MODE);
  }
  iree_io_file_access_t mode_access = 0;
  if (iree_all_bits_set(out_payload->mode, IREE_IO_FILE_MODE_READ)) {
    mode_access |= IREE_IO_FILE_ACCESS_READ;
  }
  if (iree_all_bits_set(out_payload->mode, IREE_IO_FILE_MODE_WRITE)) {
    mode_access |= IREE_IO_FILE_ACCESS_WRITE;
  }
  if (mode_access != out_payload->access) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SHM file transfer mode/access mismatch: mode=0x%016" PRIx64
        " access=0x%08x",
        out_payload->mode, out_payload->access);
  }
  return iree_ok_status();
}

#endif  // IREE_FILE_IO_ENABLE

#if IREE_FILE_IO_ENABLE && !defined(IREE_PLATFORM_WINDOWS)

static int iree_net_shm_file_transfer_channel_fd(
    iree_net_shm_file_transfer_t* transfer) {
  if (transfer->channel.type != IREE_ASYNC_PRIMITIVE_TYPE_FD) return -1;
  return transfer->channel.value.fd;
}

static void iree_net_shm_file_transfer_close_fd(int fd) {
  if (fd >= 0) close(fd);
}

static iree_status_t iree_net_shm_file_transfer_set_close_on_exec(int fd) {
#if defined(MSG_CMSG_CLOEXEC)
  (void)fd;
  return iree_ok_status();
#else
  int flags = fcntl(fd, F_GETFD);
  if (flags == -1) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "fcntl(F_GETFD) failed for transferred fd");
  }
  if (fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == -1) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "fcntl(F_SETFD) failed for transferred fd");
  }
  return iree_ok_status();
#endif  // MSG_CMSG_CLOEXEC
}

static iree_status_t iree_net_shm_file_transfer_set_nonblocking(int fd) {
  int flags = fcntl(fd, F_GETFL);
  if (flags == -1) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "fcntl(F_GETFL) failed for transfer channel");
  }
  if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "fcntl(F_SETFL) failed for transfer channel");
  }
  return iree_ok_status();
}

static iree_status_t iree_net_shm_file_transfer_send_sideband(
    iree_net_shm_file_transfer_t* transfer,
    iree_net_shm_file_transfer_opcode_t opcode, uint64_t id, int fd) {
  const int channel_fd = iree_net_shm_file_transfer_channel_fd(transfer);
  if (channel_fd < 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SHM file transfer channel is not a POSIX fd");
  }

  iree_net_shm_file_transfer_sideband_header_t header;
  memset(&header, 0, sizeof(header));
  header.magic = IREE_NET_SHM_FILE_TRANSFER_MAGIC;
  header.version = IREE_NET_SHM_FILE_TRANSFER_VERSION;
  header.opcode = opcode;
  header.id = id;

  struct iovec iov;
  iov.iov_base = &header;
  iov.iov_len = sizeof(header);

  struct msghdr message;
  memset(&message, 0, sizeof(message));
  message.msg_iov = &iov;
  message.msg_iovlen = 1;

  char control_buffer[CMSG_SPACE(sizeof(int))];
  if (fd >= 0) {
    memset(control_buffer, 0, sizeof(control_buffer));
    message.msg_control = control_buffer;
    message.msg_controllen = CMSG_SPACE(sizeof(int));

    struct cmsghdr* control_message = CMSG_FIRSTHDR(&message);
    control_message->cmsg_level = SOL_SOCKET;
    control_message->cmsg_type = SCM_RIGHTS;
    control_message->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(control_message), &fd, sizeof(fd));
  }

  int send_flags = 0;
#if defined(MSG_NOSIGNAL)
  send_flags |= MSG_NOSIGNAL;
#endif  // MSG_NOSIGNAL
#if defined(MSG_DONTWAIT)
  send_flags |= MSG_DONTWAIT;
#endif  // MSG_DONTWAIT

  ssize_t sent = 0;
  do {
    sent = sendmsg(channel_fd, &message, send_flags);
  } while (sent < 0 && errno == EINTR);
  if (sent < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "SHM file transfer sideband socket is backpressured");
    }
    return iree_make_status(iree_status_code_from_errno(errno),
                            "SHM file transfer sendmsg failed");
  }
  if ((size_t)sent != sizeof(header)) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "SHM file transfer sideband short write: %zd/%zu",
                            sent, sizeof(header));
  }

  return iree_ok_status();
}

typedef struct iree_net_shm_file_transfer_message_t {
  // Decoded sideband header.
  iree_net_shm_file_transfer_sideband_header_t header;
  // Received fd for FILE messages, or -1 for CANCEL.
  int fd;
} iree_net_shm_file_transfer_message_t;

static iree_status_t iree_net_shm_file_transfer_recv_sideband(
    iree_net_shm_file_transfer_t* transfer, bool* out_has_message,
    iree_net_shm_file_transfer_message_t* out_message) {
  *out_has_message = false;
  memset(out_message, 0, sizeof(*out_message));
  out_message->fd = IREE_NET_SHM_FILE_TRANSFER_INVALID_FD;

  const int channel_fd = iree_net_shm_file_transfer_channel_fd(transfer);
  if (channel_fd < 0) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "SHM file transfer channel is not a POSIX fd");
  }

  iree_status_t status = iree_ok_status();
  bool preserve_partial_message = false;
  while (iree_status_is_ok(status) &&
         transfer->partial_header_length < sizeof(out_message->header)) {
    iree_host_size_t remaining =
        sizeof(out_message->header) - transfer->partial_header_length;
    struct iovec iov;
    iov.iov_base = transfer->partial_header + transfer->partial_header_length;
    iov.iov_len = remaining;

    char control_buffer[CMSG_SPACE(sizeof(int))];
    memset(control_buffer, 0, sizeof(control_buffer));

    struct msghdr message;
    memset(&message, 0, sizeof(message));
    message.msg_iov = &iov;
    message.msg_iovlen = 1;
    message.msg_control = control_buffer;
    message.msg_controllen = sizeof(control_buffer);

    int recv_flags = 0;
#if defined(MSG_DONTWAIT)
    recv_flags |= MSG_DONTWAIT;
#endif  // MSG_DONTWAIT
#if defined(MSG_CMSG_CLOEXEC)
    recv_flags |= MSG_CMSG_CLOEXEC;
#endif  // MSG_CMSG_CLOEXEC

    ssize_t received = 0;
    do {
      received = recvmsg(channel_fd, &message, recv_flags);
    } while (received < 0 && errno == EINTR);
    if (received < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        if (transfer->partial_header_length == 0) {
          status = iree_ok_status();
        } else {
          preserve_partial_message = true;
          status =
              iree_make_status(IREE_STATUS_UNAVAILABLE,
                               "partial SHM file transfer sideband header");
        }
        break;
      } else {
        status = iree_make_status(iree_status_code_from_errno(errno),
                                  "SHM file transfer recvmsg failed");
        break;
      }
    }
    if (received == 0) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "SHM file transfer peer disconnected");
      break;
    }

    const bool control_truncated =
        iree_all_bits_set(message.msg_flags, MSG_CTRUNC);
    const bool data_truncated = iree_all_bits_set(message.msg_flags, MSG_TRUNC);
    for (struct cmsghdr* control_message = CMSG_FIRSTHDR(&message);
         iree_status_is_ok(status) && control_message != NULL;
         control_message = CMSG_NXTHDR(&message, control_message)) {
      if (control_message->cmsg_level == SOL_SOCKET &&
          control_message->cmsg_type == SCM_RIGHTS) {
        if (control_message->cmsg_len < CMSG_LEN(0)) {
          status = iree_make_status(
              IREE_STATUS_DATA_LOSS,
              "malformed SHM file transfer fd control payload");
          break;
        }
        const iree_host_size_t payload_size =
            (iree_host_size_t)(control_message->cmsg_len - CMSG_LEN(0));
        if ((payload_size % sizeof(int)) != 0) {
          status = iree_make_status(
              IREE_STATUS_DATA_LOSS,
              "malformed SHM file transfer fd control payload");
          break;
        }
        const iree_host_size_t file_descriptor_count =
            payload_size / sizeof(int);
        int* file_descriptors = (int*)CMSG_DATA(control_message);
        for (iree_host_size_t i = 0;
             i < file_descriptor_count && iree_status_is_ok(status); ++i) {
          if (transfer->partial_fd != IREE_NET_SHM_FILE_TRANSFER_INVALID_FD) {
            iree_net_shm_file_transfer_close_fd(file_descriptors[i]);
            status = iree_make_status(
                IREE_STATUS_DATA_LOSS,
                "SHM file transfer message carried multiple fds");
          } else {
            transfer->partial_fd = file_descriptors[i];
            status = iree_net_shm_file_transfer_set_close_on_exec(
                file_descriptors[i]);
            if (!iree_status_is_ok(status)) {
              iree_net_shm_file_transfer_close_fd(file_descriptors[i]);
              transfer->partial_fd = IREE_NET_SHM_FILE_TRANSFER_INVALID_FD;
            }
          }
        }
      }
    }
    if (iree_status_is_ok(status) && control_truncated) {
      status = iree_make_status(
          IREE_STATUS_DATA_LOSS,
          "SHM file transfer fd control payload was truncated");
    }
    if (iree_status_is_ok(status) && data_truncated) {
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "SHM file transfer header was truncated");
    }
    if (iree_status_is_ok(status)) {
      transfer->partial_header_length += (iree_host_size_t)received;
    }
  }
  if (!iree_status_is_ok(status)) {
    if (!preserve_partial_message) {
      iree_net_shm_file_transfer_close_fd(transfer->partial_fd);
      transfer->partial_fd = IREE_NET_SHM_FILE_TRANSFER_INVALID_FD;
      transfer->partial_header_length = 0;
    }
    return status;
  }
  if (transfer->partial_header_length < sizeof(out_message->header)) {
    return iree_ok_status();
  }

  memcpy(&out_message->header, transfer->partial_header,
         sizeof(out_message->header));
  out_message->fd = transfer->partial_fd;
  transfer->partial_header_length = 0;
  transfer->partial_fd = IREE_NET_SHM_FILE_TRANSFER_INVALID_FD;

  if (out_message->header.magic != IREE_NET_SHM_FILE_TRANSFER_MAGIC) {
    iree_net_shm_file_transfer_close_fd(out_message->fd);
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "SHM file transfer sideband magic mismatch");
  }
  if (out_message->header.version != IREE_NET_SHM_FILE_TRANSFER_VERSION) {
    iree_net_shm_file_transfer_close_fd(out_message->fd);
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "unsupported SHM file transfer sideband version %u",
                            (uint32_t)out_message->header.version);
  }
  if (out_message->header.reserved != 0 || out_message->header.id == 0) {
    iree_net_shm_file_transfer_close_fd(out_message->fd);
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "invalid SHM file transfer sideband header");
  }
  if (out_message->header.opcode == IREE_NET_SHM_FILE_TRANSFER_OPCODE_FILE &&
      out_message->fd == IREE_NET_SHM_FILE_TRANSFER_INVALID_FD) {
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "SHM file transfer FILE message missing fd");
  }
  if (out_message->header.opcode == IREE_NET_SHM_FILE_TRANSFER_OPCODE_CANCEL &&
      out_message->fd != IREE_NET_SHM_FILE_TRANSFER_INVALID_FD) {
    iree_net_shm_file_transfer_close_fd(out_message->fd);
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "SHM file transfer CANCEL message carried fd");
  }

  *out_has_message = true;
  return iree_ok_status();
}

static iree_net_shm_file_transfer_pending_t*
iree_net_shm_file_transfer_take_pending_locked(
    iree_net_shm_file_transfer_t* transfer, uint64_t id) {
  iree_net_shm_file_transfer_pending_t** previous_next =
      &transfer->pending_file_transfers;
  iree_net_shm_file_transfer_pending_t* pending =
      transfer->pending_file_transfers;
  while (pending && pending->id != id) {
    previous_next = &pending->next;
    pending = pending->next;
  }
  if (pending) {
    *previous_next = pending->next;
    pending->next = NULL;
  }
  return pending;
}

static iree_status_t iree_net_shm_file_transfer_store_pending_locked(
    iree_net_shm_file_transfer_t* transfer, uint64_t id, int fd) {
  bool duplicate = false;
  for (iree_net_shm_file_transfer_pending_t* pending =
           transfer->pending_file_transfers;
       pending; pending = pending->next) {
    if (pending->id != id) continue;
    duplicate = true;
    break;
  }
  if (duplicate) {
    iree_net_shm_file_transfer_close_fd(fd);
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "duplicate SHM file transfer sideband ID %" PRIu64,
                            id);
  }

  iree_net_shm_file_transfer_pending_t* pending = NULL;
  iree_status_t status = iree_allocator_malloc(
      transfer->host_allocator, sizeof(*pending), (void**)&pending);
  if (iree_status_is_ok(status)) {
    memset(pending, 0, sizeof(*pending));
    pending->id = id;
    pending->fd = fd;
    pending->next = transfer->pending_file_transfers;
    transfer->pending_file_transfers = pending;
  } else {
    iree_net_shm_file_transfer_close_fd(fd);
  }
  return status;
}

static void iree_net_shm_file_transfer_cancel_pending_locked(
    iree_net_shm_file_transfer_t* transfer, uint64_t id) {
  iree_net_shm_file_transfer_pending_t* pending =
      iree_net_shm_file_transfer_take_pending_locked(transfer, id);
  if (pending) {
    iree_net_shm_file_transfer_close_fd(pending->fd);
    iree_allocator_free(transfer->host_allocator, pending);
  }
}

#endif  // IREE_FILE_IO_ENABLE && !IREE_PLATFORM_WINDOWS

#if IREE_FILE_IO_ENABLE && defined(IREE_PLATFORM_WINDOWS)

static void iree_net_shm_file_transfer_close_fd(int fd) {
  if (fd >= 0) _close(fd);
}

static iree_status_t iree_net_shm_file_transfer_win32_handle_from_file(
    iree_io_file_handle_t* file_handle, HANDLE* out_handle) {
  *out_handle = INVALID_HANDLE_VALUE;
  if (iree_io_file_handle_type(file_handle) != IREE_IO_FILE_HANDLE_TYPE_FD) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "SHM carrier can only transfer descriptor-backed file handles");
  }
  intptr_t os_file_handle =
      _get_osfhandle(iree_io_file_handle_value(file_handle).fd);
  if (os_file_handle == -1) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "file descriptor is not backed by a valid Win32 HANDLE");
  }
  *out_handle = (HANDLE)os_file_handle;
  return iree_ok_status();
}

static iree_status_t iree_net_shm_file_transfer_win32_close_peer_handle(
    iree_net_shm_file_transfer_t* transfer, uint64_t handle_value) {
  if (iree_async_primitive_is_none(transfer->peer_process)) {
    return iree_ok_status();
  }
  if (handle_value == 0) return iree_ok_status();

  if (!DuplicateHandle((HANDLE)transfer->peer_process.value.win32_handle,
                       (HANDLE)(uintptr_t)handle_value, NULL, NULL, 0, FALSE,
                       DUPLICATE_CLOSE_SOURCE)) {
    return iree_make_status(
        iree_status_code_from_win32_error(GetLastError()),
        "failed to close transferred Win32 HANDLE in peer process");
  }
  return iree_ok_status();
}

#endif  // IREE_FILE_IO_ENABLE && IREE_PLATFORM_WINDOWS

#if IREE_FILE_IO_ENABLE

static void iree_net_shm_file_transfer_release_primitive(
    void* user_data, iree_io_file_handle_primitive_t handle_primitive) {
  (void)user_data;
  IREE_ASSERT_EQ(handle_primitive.type, IREE_IO_FILE_HANDLE_TYPE_FD);
  iree_net_shm_file_transfer_close_fd(handle_primitive.value.fd);
}

#endif  // IREE_FILE_IO_ENABLE

iree_status_t iree_net_shm_file_transfer_create(
    iree_async_primitive_t channel, iree_allocator_t host_allocator,
    iree_net_shm_file_transfer_t** out_transfer) {
  IREE_ASSERT_ARGUMENT(out_transfer);
  *out_transfer = NULL;

  iree_net_shm_file_transfer_t* transfer = NULL;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*transfer), (void**)&transfer);
  if (iree_status_is_ok(status)) {
    memset(transfer, 0, sizeof(*transfer));
    transfer->host_allocator = host_allocator;
    transfer->channel = channel;
#if IREE_FILE_IO_ENABLE && !defined(IREE_PLATFORM_WINDOWS)
    iree_slim_mutex_initialize(&transfer->mutex);
    transfer->next_id = 1;
    transfer->partial_fd = IREE_NET_SHM_FILE_TRANSFER_INVALID_FD;
    status = iree_net_shm_file_transfer_set_nonblocking(
        iree_net_shm_file_transfer_channel_fd(transfer));
    if (!iree_status_is_ok(status)) {
      iree_slim_mutex_deinitialize(&transfer->mutex);
      iree_async_primitive_close(&transfer->channel);
      iree_allocator_free(host_allocator, transfer);
      transfer = NULL;
    }
#endif  // IREE_FILE_IO_ENABLE && !IREE_PLATFORM_WINDOWS
#if IREE_FILE_IO_ENABLE && defined(IREE_PLATFORM_WINDOWS)
    status = iree_net_shm_win32_pipe_open_peer_process(transfer->channel,
                                                       &transfer->peer_process);
    if (!iree_status_is_ok(status)) {
      iree_async_primitive_close(&transfer->channel);
      iree_allocator_free(host_allocator, transfer);
      transfer = NULL;
    }
#endif  // IREE_FILE_IO_ENABLE && IREE_PLATFORM_WINDOWS
    *out_transfer = transfer;
  } else {
    iree_async_primitive_close(&channel);
  }
  return status;
}

void iree_net_shm_file_transfer_release(
    iree_net_shm_file_transfer_t* transfer) {
  if (!transfer) return;
#if IREE_FILE_IO_ENABLE && !defined(IREE_PLATFORM_WINDOWS)
  while (transfer->pending_file_transfers) {
    iree_net_shm_file_transfer_pending_t* next =
        transfer->pending_file_transfers->next;
    iree_net_shm_file_transfer_close_fd(transfer->pending_file_transfers->fd);
    iree_allocator_free(transfer->host_allocator,
                        transfer->pending_file_transfers);
    transfer->pending_file_transfers = next;
  }
  iree_net_shm_file_transfer_close_fd(transfer->partial_fd);
  iree_slim_mutex_deinitialize(&transfer->mutex);
#endif  // IREE_FILE_IO_ENABLE && !IREE_PLATFORM_WINDOWS
#if IREE_FILE_IO_ENABLE && defined(IREE_PLATFORM_WINDOWS)
  iree_async_primitive_close(&transfer->peer_process);
#endif  // IREE_FILE_IO_ENABLE && IREE_PLATFORM_WINDOWS
  iree_async_primitive_close(&transfer->channel);
  iree_allocator_t host_allocator = transfer->host_allocator;
  iree_allocator_free(host_allocator, transfer);
}

iree_net_file_handle_transfer_type_t iree_net_shm_file_transfer_type(
    const iree_net_shm_file_transfer_t* transfer) {
  if (!transfer) return IREE_NET_FILE_HANDLE_TRANSFER_TYPE_NONE;
#if IREE_FILE_IO_ENABLE && !defined(IREE_PLATFORM_WINDOWS)
  return transfer->channel.type == IREE_ASYNC_PRIMITIVE_TYPE_FD
             ? IREE_NET_FILE_HANDLE_TRANSFER_TYPE_POSIX_FD
             : IREE_NET_FILE_HANDLE_TRANSFER_TYPE_NONE;
#elif IREE_FILE_IO_ENABLE && defined(IREE_PLATFORM_WINDOWS)
  return !iree_async_primitive_is_none(transfer->peer_process)
             ? IREE_NET_FILE_HANDLE_TRANSFER_TYPE_WIN32_HANDLE
             : IREE_NET_FILE_HANDLE_TRANSFER_TYPE_NONE;
#else
  (void)transfer;
  return IREE_NET_FILE_HANDLE_TRANSFER_TYPE_NONE;
#endif  // IREE_FILE_IO_ENABLE && !IREE_PLATFORM_WINDOWS
}

iree_status_t iree_net_shm_file_transfer_query(
    iree_net_shm_file_transfer_t* transfer, iree_io_file_handle_t* file_handle,
    iree_net_file_handle_transfer_type_t* out_transfer_type,
    iree_host_size_t* out_payload_length) {
  IREE_ASSERT_ARGUMENT(out_transfer_type);
  IREE_ASSERT_ARGUMENT(out_payload_length);
  *out_transfer_type = IREE_NET_FILE_HANDLE_TRANSFER_TYPE_NONE;
  *out_payload_length = 0;

#if IREE_FILE_IO_ENABLE
  if (!transfer) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "SHM carrier does not have a file transfer "
                            "sideband");
  }
#if defined(IREE_PLATFORM_WINDOWS)
  const iree_net_file_handle_transfer_type_t expected_transfer_type =
      IREE_NET_FILE_HANDLE_TRANSFER_TYPE_WIN32_HANDLE;
  const char* expected_transfer_name = "Win32 HANDLE";
#else
  const iree_net_file_handle_transfer_type_t expected_transfer_type =
      IREE_NET_FILE_HANDLE_TRANSFER_TYPE_POSIX_FD;
  const char* expected_transfer_name = "POSIX fd";
#endif  // IREE_PLATFORM_WINDOWS
  if (iree_net_shm_file_transfer_type(transfer) != expected_transfer_type) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "SHM carrier cannot transfer %s rights",
                            expected_transfer_name);
  }
  if (iree_io_file_handle_type(file_handle) != IREE_IO_FILE_HANDLE_TYPE_FD) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "SHM carrier can only transfer descriptor-backed file handles");
  }
  *out_transfer_type = expected_transfer_type;
  *out_payload_length = sizeof(iree_net_shm_file_transfer_payload_t);
  return iree_ok_status();
#else
  (void)transfer;
  (void)file_handle;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "SHM file handle transfer is unavailable on this "
                          "platform or file support is disabled");
#endif  // IREE_FILE_IO_ENABLE && !IREE_PLATFORM_WINDOWS
}

iree_status_t iree_net_shm_file_transfer_export(
    iree_net_shm_file_transfer_t* transfer, iree_io_file_handle_t* file_handle,
    iree_net_file_handle_transfer_type_t transfer_type,
    iree_byte_span_t transfer_payload) {
#if IREE_FILE_IO_ENABLE
  if (!transfer) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "SHM carrier does not have a file transfer "
                            "sideband");
  }
#if defined(IREE_PLATFORM_WINDOWS)
  const iree_net_file_handle_transfer_type_t expected_transfer_type =
      IREE_NET_FILE_HANDLE_TRANSFER_TYPE_WIN32_HANDLE;
#else
  const iree_net_file_handle_transfer_type_t expected_transfer_type =
      IREE_NET_FILE_HANDLE_TRANSFER_TYPE_POSIX_FD;
#endif  // IREE_PLATFORM_WINDOWS
  if (transfer_type != expected_transfer_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported SHM file transfer type %u",
                            (uint32_t)transfer_type);
  }
  if (transfer_payload.data_length !=
      sizeof(iree_net_shm_file_transfer_payload_t)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SHM file transfer payload length must be %" PRIhsz " bytes",
        (iree_host_size_t)sizeof(iree_net_shm_file_transfer_payload_t));
  }
  if (iree_io_file_handle_type(file_handle) != IREE_IO_FILE_HANDLE_TYPE_FD) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "SHM carrier can only transfer descriptor-backed file handles");
  }

#if defined(IREE_PLATFORM_WINDOWS)
  if (iree_async_primitive_is_none(transfer->peer_process)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "SHM carrier cannot transfer Win32 HANDLE rights to its peer");
  }

  HANDLE source_handle = INVALID_HANDLE_VALUE;
  iree_status_t status = iree_net_shm_file_transfer_win32_handle_from_file(
      file_handle, &source_handle);
  HANDLE peer_handle = NULL;
  if (iree_status_is_ok(status) &&
      !DuplicateHandle(GetCurrentProcess(), source_handle,
                       (HANDLE)transfer->peer_process.value.win32_handle,
                       &peer_handle, 0, FALSE, DUPLICATE_SAME_ACCESS)) {
    status = iree_make_status(iree_status_code_from_win32_error(GetLastError()),
                              "failed to duplicate Win32 file HANDLE into "
                              "SHM peer process");
  }
  if (iree_status_is_ok(status)) {
    iree_net_shm_file_transfer_payload_t payload = {
        .id = (uint64_t)(uintptr_t)peer_handle,
        .mode = iree_net_shm_file_transfer_mode_from_handle(file_handle),
        .access = iree_io_file_handle_access(file_handle),
        .reserved = 0,
    };
    memcpy(transfer_payload.data, &payload, sizeof(payload));
  }
  return status;
#else
  iree_slim_mutex_lock(&transfer->mutex);
  uint64_t id = transfer->next_id++;
  if (transfer->next_id == 0) transfer->next_id = 1;
  const int fd = iree_io_file_handle_value(file_handle).fd;
  iree_status_t status = iree_net_shm_file_transfer_send_sideband(
      transfer, IREE_NET_SHM_FILE_TRANSFER_OPCODE_FILE, id, fd);
  iree_slim_mutex_unlock(&transfer->mutex);

  if (iree_status_is_ok(status)) {
    iree_net_shm_file_transfer_payload_t payload = {
        .id = id,
        .mode = iree_net_shm_file_transfer_mode_from_handle(file_handle),
        .access = iree_io_file_handle_access(file_handle),
        .reserved = 0,
    };
    memcpy(transfer_payload.data, &payload, sizeof(payload));
  }
  return status;
#endif  // IREE_PLATFORM_WINDOWS
#else
  (void)transfer;
  (void)file_handle;
  (void)transfer_type;
  (void)transfer_payload;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "SHM file handle transfer is unavailable on this "
                          "platform or file support is disabled");
#endif  // IREE_FILE_IO_ENABLE
}

iree_status_t iree_net_shm_file_transfer_import(
    iree_net_shm_file_transfer_t* transfer,
    iree_net_file_handle_transfer_type_t transfer_type,
    iree_const_byte_span_t transfer_payload, iree_allocator_t host_allocator,
    iree_io_file_handle_t** out_file_handle) {
  IREE_ASSERT_ARGUMENT(out_file_handle);
  *out_file_handle = NULL;

#if IREE_FILE_IO_ENABLE
  if (!transfer) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "SHM carrier does not have a file transfer "
                            "sideband");
  }
#if defined(IREE_PLATFORM_WINDOWS)
  const iree_net_file_handle_transfer_type_t expected_transfer_type =
      IREE_NET_FILE_HANDLE_TRANSFER_TYPE_WIN32_HANDLE;
#else
  const iree_net_file_handle_transfer_type_t expected_transfer_type =
      IREE_NET_FILE_HANDLE_TRANSFER_TYPE_POSIX_FD;
#endif  // IREE_PLATFORM_WINDOWS
  if (transfer_type != expected_transfer_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported SHM file transfer type %u",
                            (uint32_t)transfer_type);
  }

  iree_net_shm_file_transfer_payload_t payload;
  iree_status_t status =
      iree_net_shm_file_transfer_validate_payload(transfer_payload, &payload);

  int fd = IREE_NET_SHM_FILE_TRANSFER_INVALID_FD;
#if defined(IREE_PLATFORM_WINDOWS)
  if (iree_status_is_ok(status)) {
    int open_flags = 0;
    if (!iree_all_bits_set(payload.mode, IREE_IO_FILE_MODE_WRITE)) {
      open_flags |= _O_RDONLY;
    }
    fd = _open_osfhandle((intptr_t)(HANDLE)(uintptr_t)payload.id, open_flags);
    if (fd == -1) {
      CloseHandle((HANDLE)(uintptr_t)payload.id);
      status = iree_make_status(
          IREE_STATUS_INTERNAL,
          "unable to transfer Win32 HANDLE to a CRT file descriptor");
    }
  }
#else
  iree_slim_mutex_lock(&transfer->mutex);
  iree_net_shm_file_transfer_pending_t* pending = NULL;
  if (iree_status_is_ok(status)) {
    pending =
        iree_net_shm_file_transfer_take_pending_locked(transfer, payload.id);
    if (pending) {
      fd = pending->fd;
      iree_allocator_free(transfer->host_allocator, pending);
    }
  }
  while (iree_status_is_ok(status) &&
         fd == IREE_NET_SHM_FILE_TRANSFER_INVALID_FD) {
    bool has_message = false;
    iree_net_shm_file_transfer_message_t message;
    status = iree_net_shm_file_transfer_recv_sideband(transfer, &has_message,
                                                      &message);
    if (!iree_status_is_ok(status)) break;
    if (!has_message) {
      status = iree_make_status(
          IREE_STATUS_NOT_FOUND,
          "SHM file transfer sideband ID %" PRIu64 " not found", payload.id);
      break;
    }

    if (message.header.opcode == IREE_NET_SHM_FILE_TRANSFER_OPCODE_FILE) {
      if (message.header.id == payload.id) {
        fd = message.fd;
      } else {
        status = iree_net_shm_file_transfer_store_pending_locked(
            transfer, message.header.id, message.fd);
      }
    } else if (message.header.opcode ==
               IREE_NET_SHM_FILE_TRANSFER_OPCODE_CANCEL) {
      iree_net_shm_file_transfer_cancel_pending_locked(transfer,
                                                       message.header.id);
      if (message.header.id == payload.id) {
        status = iree_make_status(IREE_STATUS_CANCELLED,
                                  "SHM file transfer sideband ID %" PRIu64
                                  " was cancelled",
                                  payload.id);
      }
    } else {
      iree_net_shm_file_transfer_close_fd(message.fd);
      status = iree_make_status(IREE_STATUS_DATA_LOSS,
                                "unknown SHM file transfer sideband opcode %u",
                                (uint32_t)message.header.opcode);
    }
  }
  iree_slim_mutex_unlock(&transfer->mutex);
#endif  // IREE_PLATFORM_WINDOWS

  if (iree_status_is_ok(status)) {
    iree_io_file_handle_primitive_t handle_primitive = {
        .type = IREE_IO_FILE_HANDLE_TYPE_FD,
        .value = {.fd = fd},
    };
    const iree_io_file_handle_release_callback_t release_callback = {
        .fn = iree_net_shm_file_transfer_release_primitive,
        .user_data = NULL,
    };
    status = iree_io_file_handle_wrap(
        payload.access, (iree_io_file_mode_t)payload.mode, handle_primitive,
        release_callback, host_allocator, out_file_handle);
    if (iree_status_is_ok(status)) {
      fd = IREE_NET_SHM_FILE_TRANSFER_INVALID_FD;
    }
  }
  iree_net_shm_file_transfer_close_fd(fd);
  return status;
#else
  (void)transfer;
  (void)transfer_type;
  (void)transfer_payload;
  (void)host_allocator;
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "SHM file handle transfer is unavailable on this "
                          "platform or file support is disabled");
#endif  // IREE_FILE_IO_ENABLE
}

iree_status_t iree_net_shm_file_transfer_release_export(
    iree_net_shm_file_transfer_t* transfer,
    iree_net_file_handle_transfer_type_t transfer_type,
    iree_const_byte_span_t transfer_payload) {
#if IREE_FILE_IO_ENABLE
  if (!transfer) return iree_ok_status();
#if defined(IREE_PLATFORM_WINDOWS)
  const iree_net_file_handle_transfer_type_t expected_transfer_type =
      IREE_NET_FILE_HANDLE_TRANSFER_TYPE_WIN32_HANDLE;
#else
  const iree_net_file_handle_transfer_type_t expected_transfer_type =
      IREE_NET_FILE_HANDLE_TRANSFER_TYPE_POSIX_FD;
#endif  // IREE_PLATFORM_WINDOWS
  if (transfer_type != expected_transfer_type) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported SHM file transfer type %u",
                            (uint32_t)transfer_type);
  }

  iree_net_shm_file_transfer_payload_t payload;
  iree_status_t status =
      iree_net_shm_file_transfer_validate_payload(transfer_payload, &payload);
  if (iree_status_is_ok(status)) {
#if defined(IREE_PLATFORM_WINDOWS)
    status = iree_net_shm_file_transfer_win32_close_peer_handle(transfer,
                                                                payload.id);
#else
    iree_slim_mutex_lock(&transfer->mutex);
    status = iree_net_shm_file_transfer_send_sideband(
        transfer, IREE_NET_SHM_FILE_TRANSFER_OPCODE_CANCEL, payload.id,
        IREE_NET_SHM_FILE_TRANSFER_INVALID_FD);
    iree_slim_mutex_unlock(&transfer->mutex);
#endif  // IREE_PLATFORM_WINDOWS
  }
  return status;
#else
  (void)transfer;
  (void)transfer_type;
  (void)transfer_payload;
  return iree_ok_status();
#endif  // IREE_FILE_IO_ENABLE
}
