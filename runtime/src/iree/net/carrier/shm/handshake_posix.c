// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// POSIX handshake handle exchange: passes fds via SCM_RIGHTS over sendmsg/
// recvmsg on a Unix domain socket. Each handshake message consists of the
// fixed-size header as the iovec payload and up to 3 fds as ancillary data.
// The channel is non-blocking in factory use, so partial stream transfers and
// readiness waits are handled explicitly.

#include "iree/net/carrier/shm/handshake.h"

#if !defined(IREE_PLATFORM_WINDOWS)

#include <errno.h>
#include <poll.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

// Maximum number of fds sent in a single handshake message.
// OFFER sends 3 (shm_region, wake_epoch_shm, signal_primitive).
// ACCEPT sends 2 (wake_epoch_shm, signal_primitive).
#define MAX_HANDSHAKE_FDS 3

// Size of the cmsg buffer for SCM_RIGHTS. Must be large enough for MAX_FDS.
#define CMSG_BUF_SIZE (CMSG_SPACE(MAX_HANDSHAKE_FDS * sizeof(int)))

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Extracts a POSIX fd from an iree_shm_handle_t or iree_async_primitive_t.
// Returns -1 for invalid/NONE handles.
static int iree_shm_handle_to_fd(iree_shm_handle_t handle) {
  if (!iree_shm_handle_is_valid(handle)) return -1;
  return (int)handle.value;
}

static iree_shm_handle_t iree_shm_handle_from_fd(int fd) {
  iree_shm_handle_t handle;
  handle.value = (uint64_t)fd;
  return handle;
}

static int iree_async_primitive_to_fd(iree_async_primitive_t primitive) {
  if (primitive.type != IREE_ASYNC_PRIMITIVE_TYPE_FD) return -1;
  return primitive.value.fd;
}

static iree_status_t iree_net_shm_handshake_cancelled_status(void) {
  return iree_make_status(IREE_STATUS_CANCELLED, "SHM handshake cancelled");
}

static iree_status_t iree_net_shm_handshake_wait_fd(
    int channel_fd, short events,
    const iree_net_shm_handshake_cancellation_t* cancellation) {
  if (iree_net_shm_handshake_cancellation_is_requested(cancellation)) {
    return iree_net_shm_handshake_cancelled_status();
  }

  struct pollfd poll_fds[2];
  memset(poll_fds, 0, sizeof(poll_fds));
  poll_fds[0].fd = channel_fd;
  poll_fds[0].events = events;
  nfds_t poll_fd_count = 1;
  if (cancellation &&
      cancellation->interrupt_primitive.type == IREE_ASYNC_PRIMITIVE_TYPE_FD) {
    poll_fds[1].fd = cancellation->interrupt_primitive.value.fd;
    poll_fds[1].events = POLLIN;
    poll_fd_count = 2;
  }
  int poll_result = 0;
  do {
    poll_result = poll(poll_fds, poll_fd_count, /*timeout=*/-1);
  } while (poll_result < 0 && errno == EINTR &&
           !iree_net_shm_handshake_cancellation_is_requested(cancellation));

  if (iree_net_shm_handshake_cancellation_is_requested(cancellation)) {
    return iree_net_shm_handshake_cancelled_status();
  }
  if (poll_result < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "SHM handshake poll failed");
  }
  if (poll_fd_count == 2 && (poll_fds[1].revents & POLLIN) != 0) {
    return iree_net_shm_handshake_cancelled_status();
  }
  if ((poll_fds[0].revents & events) != 0) return iree_ok_status();
  return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "SHM handshake channel closed while waiting (revents=0x%x)",
      poll_fds[0].revents);
}

static void iree_net_shm_handshake_close_fds(int* fds, int fd_count) {
  for (int i = 0; i < fd_count; ++i) close(fds[i]);
}

//===----------------------------------------------------------------------===//
// Send/recv with SCM_RIGHTS
//===----------------------------------------------------------------------===//

iree_status_t iree_net_shm_handshake_send(
    iree_async_primitive_t channel,
    const iree_net_shm_handshake_cancellation_t* cancellation,
    const iree_net_shm_handshake_header_t* header,
    const iree_net_shm_handshake_handles_t* handles) {
  int channel_fd = iree_async_primitive_to_fd(channel);
  if (channel_fd < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "handshake channel is not a valid POSIX fd");
  }

  // Collect fds to send. Order matters — receiver unpacks in same order.
  int fds[MAX_HANDSHAKE_FDS];
  int fd_count = 0;

  int shm_fd = iree_shm_handle_to_fd(handles->shm_region);
  if (shm_fd >= 0) {
    fds[fd_count++] = shm_fd;
  }
  int epoch_fd = iree_shm_handle_to_fd(handles->wake_epoch_shm);
  if (epoch_fd >= 0) {
    fds[fd_count++] = epoch_fd;
  }
  int signal_fd = iree_async_primitive_to_fd(handles->signal_primitive);
  if (signal_fd >= 0) {
    fds[fd_count++] = signal_fd;
  }

  iree_host_size_t offset = 0;
  iree_status_t status = iree_ok_status();
  while (offset < sizeof(*header) && iree_status_is_ok(status)) {
    if (iree_net_shm_handshake_cancellation_is_requested(cancellation)) {
      status = iree_net_shm_handshake_cancelled_status();
      break;
    }

    struct iovec iov = {
        .iov_base = (uint8_t*)header + offset,
        .iov_len = sizeof(*header) - offset,
    };
    struct msghdr message;
    memset(&message, 0, sizeof(message));
    message.msg_iov = &iov;
    message.msg_iovlen = 1;

    // Descriptor rights are attached only until the first payload byte is
    // accepted. The kernel transfers them with that byte; retransmitting them
    // after a partial write would duplicate the receiver's descriptors.
    char control_buffer[CMSG_BUF_SIZE];
    if (offset == 0 && fd_count > 0) {
      memset(control_buffer, 0, sizeof(control_buffer));
      message.msg_control = control_buffer;
      message.msg_controllen = CMSG_SPACE(fd_count * sizeof(int));
      struct cmsghdr* control_message = CMSG_FIRSTHDR(&message);
      control_message->cmsg_level = SOL_SOCKET;
      control_message->cmsg_type = SCM_RIGHTS;
      control_message->cmsg_len = CMSG_LEN(fd_count * sizeof(int));
      memcpy(CMSG_DATA(control_message), fds, fd_count * sizeof(int));
    }

    ssize_t send_count =
        sendmsg(channel_fd, &message, MSG_NOSIGNAL | MSG_DONTWAIT);
    if (send_count > 0) {
      offset += (iree_host_size_t)send_count;
    } else if (send_count == 0) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "SHM handshake peer disconnected during send");
    } else if (errno == EINTR) {
      continue;
    } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
      status =
          iree_net_shm_handshake_wait_fd(channel_fd, POLLOUT, cancellation);
    } else if (iree_net_shm_handshake_cancellation_is_requested(cancellation)) {
      status = iree_net_shm_handshake_cancelled_status();
    } else {
      status = iree_make_status(iree_status_code_from_errno(errno),
                                "SHM handshake sendmsg failed");
    }
  }
  return status;
}

iree_status_t iree_net_shm_handshake_recv(
    iree_async_primitive_t channel,
    const iree_net_shm_handshake_cancellation_t* cancellation,
    iree_net_shm_handshake_header_t* out_header,
    iree_net_shm_handshake_handles_t* out_handles) {
  memset(out_header, 0, sizeof(*out_header));
  *out_handles = iree_net_shm_handshake_handles_empty();

  int channel_fd = iree_async_primitive_to_fd(channel);
  if (channel_fd < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "handshake channel is not a valid POSIX fd");
  }

  int fds[MAX_HANDSHAKE_FDS];
  int fd_count = 0;
  bool received_control = false;
  iree_host_size_t offset = 0;
  iree_status_t status = iree_ok_status();
  while (offset < sizeof(*out_header) && iree_status_is_ok(status)) {
    if (iree_net_shm_handshake_cancellation_is_requested(cancellation)) {
      status = iree_net_shm_handshake_cancelled_status();
      break;
    }

    struct iovec iov = {
        .iov_base = (uint8_t*)out_header + offset,
        .iov_len = sizeof(*out_header) - offset,
    };
    char control_buffer[CMSG_BUF_SIZE];
    memset(control_buffer, 0, sizeof(control_buffer));
    struct msghdr message;
    memset(&message, 0, sizeof(message));
    message.msg_iov = &iov;
    message.msg_iovlen = 1;
    if (!received_control) {
      message.msg_control = control_buffer;
      message.msg_controllen = sizeof(control_buffer);
    }

    int recv_flags = 0;
#if defined(MSG_CMSG_CLOEXEC)
    recv_flags |= MSG_CMSG_CLOEXEC;
#endif  // MSG_CMSG_CLOEXEC
    ssize_t receive_count =
        recvmsg(channel_fd, &message, recv_flags | MSG_DONTWAIT);
    if (receive_count > 0) {
      offset += (iree_host_size_t)receive_count;
      if (!received_control) {
        received_control = true;
        if ((message.msg_flags & MSG_CTRUNC) != 0) {
          status = iree_make_status(
              IREE_STATUS_RESOURCE_EXHAUSTED,
              "SHM handshake descriptor control data was truncated");
          break;
        }
        for (struct cmsghdr* control_message = CMSG_FIRSTHDR(&message);
             control_message != NULL;
             control_message = CMSG_NXTHDR(&message, control_message)) {
          if (control_message->cmsg_level != SOL_SOCKET ||
              control_message->cmsg_type != SCM_RIGHTS) {
            continue;
          }
          iree_host_size_t payload_size =
              control_message->cmsg_len - CMSG_LEN(0);
          if (payload_size % sizeof(int) != 0) {
            status = iree_make_status(
                IREE_STATUS_DATA_LOSS,
                "SHM handshake descriptor payload is malformed");
            break;
          }
          int received_fd_count = (int)(payload_size / sizeof(int));
          if (fd_count + received_fd_count > MAX_HANDSHAKE_FDS) {
            status =
                iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                 "SHM handshake sent too many descriptors");
            break;
          }
          memcpy(&fds[fd_count], CMSG_DATA(control_message), payload_size);
          fd_count += received_fd_count;
        }
      }
    } else if (receive_count == 0) {
      status = iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "SHM handshake peer disconnected after %" PRIhsz
                                " of %zu bytes",
                                offset, sizeof(*out_header));
    } else if (errno == EINTR) {
      continue;
    } else if (errno == EAGAIN || errno == EWOULDBLOCK) {
      status = iree_net_shm_handshake_wait_fd(channel_fd, POLLIN, cancellation);
    } else if (iree_net_shm_handshake_cancellation_is_requested(cancellation)) {
      status = iree_net_shm_handshake_cancelled_status();
    } else {
      status = iree_make_status(iree_status_code_from_errno(errno),
                                "SHM handshake recvmsg failed");
    }
  }

  if (!iree_status_is_ok(status)) {
    iree_net_shm_handshake_close_fds(fds, fd_count);
    return status;
  }

  // Unpack fds based on message type.
  // OFFER: 3 fds (shm_region, wake_epoch_shm, signal_primitive).
  // ACCEPT: 2 fds (wake_epoch_shm, signal_primitive).
  // READY: 0 fds.
  if (out_header->type == IREE_NET_SHM_HANDSHAKE_MESSAGE_OFFER) {
    if (fd_count != 3) {
      // Close any fds we did receive before failing.
      iree_net_shm_handshake_close_fds(fds, fd_count);
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "OFFER expected 3 fds, got %d", fd_count);
    }
    out_handles->shm_region = iree_shm_handle_from_fd(fds[0]);
    out_handles->wake_epoch_shm = iree_shm_handle_from_fd(fds[1]);
    out_handles->signal_primitive = iree_async_primitive_from_fd(fds[2]);
  } else if (out_header->type == IREE_NET_SHM_HANDSHAKE_MESSAGE_ACCEPT) {
    if (fd_count != 2) {
      iree_net_shm_handshake_close_fds(fds, fd_count);
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "ACCEPT expected 2 fds, got %d", fd_count);
    }
    out_handles->wake_epoch_shm = iree_shm_handle_from_fd(fds[0]);
    out_handles->signal_primitive = iree_async_primitive_from_fd(fds[1]);
  } else if (out_header->type == IREE_NET_SHM_HANDSHAKE_MESSAGE_READY) {
    if (fd_count != 0) {
      iree_net_shm_handshake_close_fds(fds, fd_count);
      return iree_make_status(IREE_STATUS_DATA_LOSS,
                              "READY expected no fds, got %d", fd_count);
    }
  } else {
    // Unknown message type — close any received fds.
    iree_net_shm_handshake_close_fds(fds, fd_count);
  }

  return iree_ok_status();
}

#endif  // !IREE_PLATFORM_WINDOWS
