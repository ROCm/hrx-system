// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Submit path for the IOCP proactor.
//
// This module validates complete batches, reserves all carrier and message
// storage needed by active heads, and only then commits operations. Per-type
// commit helpers issue overlapped I/O calls or route poll-owned work through
// the pending queue. Platform failures after acceptance are delivered through
// ordinary terminal completions instead of escaping from submit().

#include <string.h>

#include "iree/async/buffer_pool.h"
#include "iree/async/event.h"
#include "iree/async/file.h"
#include "iree/async/notification.h"
#include "iree/async/operation.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/message.h"
#include "iree/async/operations/net.h"
#include "iree/async/operations/scheduling.h"
#include "iree/async/operations/semaphore.h"
#include "iree/async/platform/iocp/proactor.h"
#include "iree/async/platform/iocp/proactor_validation.h"
#include "iree/async/proactor.h"
#include "iree/async/semaphore.h"
#include "iree/async/span.h"
#include "iree/async/util/continuation.h"
#include "iree/async/util/message_pool.h"
#include "iree/async/util/semaphore_wait.h"
#include "iree/async/util/sequence_emulation.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/memory.h"
#include "iree/base/internal/path.h"

#if defined(IREE_PLATFORM_WINDOWS)

// Windows headers — winsock2.h must precede windows.h to avoid conflicts.
// clang-format off
#include <winsock2.h>
#include <ws2tcpip.h>
#include <mswsock.h>
#include <windows.h>
// clang-format on

//===----------------------------------------------------------------------===//
// Carrier allocation and direct completion helpers
//===----------------------------------------------------------------------===//

// Acquires and initializes a carrier for an operation. Pops from the carrier
// freelist when available, falling back to heap allocation on demand. The
// carrier is zeroed and configured with the given type, operation, and native
// I/O handle. The caller fills in the type-specific data union members.
iree_status_t iree_async_proactor_iocp_acquire_carrier(
    iree_async_proactor_iocp_t* proactor,
    iree_async_iocp_carrier_type_t carrier_type,
    iree_async_operation_t* operation, uintptr_t io_handle,
    iree_async_iocp_carrier_t** out_carrier) {
  iree_async_iocp_carrier_t* carrier =
      (iree_async_iocp_carrier_t*)iree_atomic_slist_pop(
          &proactor->carrier_freelist);
  if (!carrier) {
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(
        proactor->base.allocator, sizeof(*carrier), (void**)&carrier));
  }
  memset(carrier, 0, sizeof(*carrier));
  carrier->type = carrier_type;
  carrier->proactor = proactor;
  carrier->operation = operation;
  carrier->io_handle = io_handle;
  iree_atomic_fetch_add(&proactor->outstanding_carrier_count, 1,
                        iree_memory_order_relaxed);
  *out_carrier = carrier;
  return iree_ok_status();
}

// Returns a carrier to the freelist for reuse. Decrements the outstanding
// carrier count. The carrier must not be referenced after this call.
void iree_async_proactor_iocp_release_carrier(
    iree_async_proactor_iocp_t* proactor, iree_async_iocp_carrier_t* carrier) {
  IREE_ASSERT(carrier->type != IREE_ASYNC_IOCP_CARRIER_DIRECT ||
              iree_status_is_ok(carrier->data.direct.status));
  iree_atomic_fetch_sub(&proactor->outstanding_carrier_count, 1,
                        iree_memory_order_relaxed);
  iree_atomic_slist_push(&proactor->carrier_freelist,
                         (iree_atomic_slist_entry_t*)carrier);
}

// Builds a WSABUF array from descriptors captured at accepted submission.
static DWORD iree_async_proactor_iocp_build_wsabuf(
    WSABUF* wsabuf, const iree_async_socket_io_platform_t* platform,
    iree_async_region_t* const* retained_regions, uint8_t span_count) {
  DWORD count = (DWORD)span_count;
  for (DWORD i = 0; i < count; ++i) {
    iree_async_span_t span = iree_async_socket_prepared_span_resolve(
        platform->prepared.spans[i], retained_regions[i]);
    wsabuf[i].buf = (char*)iree_async_span_ptr(span);
    wsabuf[i].len = (ULONG)span.length;
  }
  return count;
}

// Publishes an accepted synthetic completion. Posting to the completion port
// is only a notification optimization: the reserved carrier is durable state,
// so a failed post moves it to the fallback queue and wakes the poll owner via
// the completion-port component's APC path.
static void iree_async_proactor_iocp_post_direct_completion(
    iree_async_proactor_iocp_t* proactor, iree_async_iocp_carrier_t* carrier,
    iree_status_t status) {
  carrier->operation->next = NULL;
  carrier->type = IREE_ASYNC_IOCP_CARRIER_DIRECT;
  carrier->data.direct.status = status;

  DWORD error_code = ERROR_SUCCESS;
  if (!iree_async_iocp_completion_port_try_post(&proactor->completion_port, 0,
                                                0, &carrier->overlapped,
                                                &error_code)) {
    iree_async_iocp_fallback_completion_slist_push(
        &proactor->fallback_completion_queue, carrier);
    iree_async_iocp_completion_port_request_fallback_wake(
        &proactor->completion_port);
  }
}

static void iree_async_proactor_iocp_post_win32_error(
    iree_async_proactor_iocp_t* proactor, iree_async_iocp_carrier_t* carrier,
    DWORD error_code, const char* operation_name) {
  iree_async_proactor_iocp_post_direct_completion(
      proactor, carrier,
      iree_make_status(iree_status_code_from_win32_error(error_code),
                       "%s failed (Win32 error %lu)", operation_name,
                       (unsigned long)error_code));
}

//===----------------------------------------------------------------------===//
// Socket submit helpers
//===----------------------------------------------------------------------===//

static void iree_async_proactor_iocp_commit_socket_accept(
    iree_async_proactor_iocp_t* proactor,
    iree_async_socket_accept_operation_t* accept_op,
    iree_async_iocp_carrier_t* carrier) {
  iree_async_socket_t* listen_socket = accept_op->listen_socket;
  SOCKET listen_sock = (SOCKET)listen_socket->primitive.value.win32_handle;
  carrier->io_handle = (uintptr_t)listen_sock;

  accept_op->accepted_socket = NULL;
  memset(&accept_op->peer_address, 0, sizeof(accept_op->peer_address));

  // AcceptEx requires a pre-created accept socket of the same family/type.
  int domain =
      listen_socket->type == IREE_ASYNC_SOCKET_TYPE_TCP6 ? AF_INET6 : AF_INET;

  SOCKET accept_sock = WSASocketW(domain, SOCK_STREAM, IPPROTO_TCP, NULL, 0,
                                  WSA_FLAG_OVERLAPPED);
  if (accept_sock == INVALID_SOCKET) {
    int wsa_error = WSAGetLastError();
    iree_async_proactor_iocp_post_win32_error(
        proactor, carrier, (DWORD)wsa_error, "WSASocket for accept");
    return;
  }

  // Associate accept socket with IOCP port.
  HANDLE result = CreateIoCompletionPort(
      (HANDLE)accept_sock, (HANDLE)proactor->completion_port.handle, 0, 0);
  if (result == NULL) {
    DWORD error = GetLastError();
    closesocket(accept_sock);
    iree_async_proactor_iocp_post_win32_error(
        proactor, carrier, error, "CreateIoCompletionPort for accept socket");
    return;
  }

  carrier->data.accept.accept_socket = accept_sock;
  // Each address slot needs the maximum address size plus 16 bytes of
  // padding (AcceptEx requirement: at least 16 bytes more than the maximum
  // address length for the transport protocol). sockaddr_storage covers both
  // supported address families.
  carrier->data.accept.local_address_length =
      (DWORD)(sizeof(struct sockaddr_storage) + 16);
  carrier->data.accept.remote_address_length =
      (DWORD)(sizeof(struct sockaddr_storage) + 16);

  // Issue AcceptEx. dwReceiveDataLength = 0 means accept-only (no initial
  // data reception).
  DWORD bytes_received = 0;
  BOOL accepted = proactor->wsa_extensions.AcceptEx(
      listen_sock, accept_sock, carrier->data.accept.address_buffer,
      /*dwReceiveDataLength=*/0, carrier->data.accept.local_address_length,
      carrier->data.accept.remote_address_length, &bytes_received,
      &carrier->overlapped);

  if (!accepted) {
    int wsa_error = WSAGetLastError();
    if (wsa_error != WSA_IO_PENDING) {
      closesocket(accept_sock);
      iree_async_proactor_iocp_post_win32_error(proactor, carrier,
                                                (DWORD)wsa_error, "AcceptEx");
      return;
    }
    // WSA_IO_PENDING: I/O is pending, completion will arrive via GQCS.
  }
}

// Binds an unbound socket to INADDR_ANY:0 (or in6addr_any:0 for IPv6).
// ConnectEx requires the socket to be bound before it can be called, and UDP
// connect also needs a bound socket. This is invisible to the caller.
//
// Detection: getsockname() returns WSAEINVAL on an unbound Windows socket.
// Imported sockets have unknown bind state, so the platform remains the source
// of truth until this path or an explicit bind resolves it.
static iree_status_t iree_async_proactor_iocp_auto_bind_if_needed(
    iree_async_socket_t* socket, SOCKET sock) {
  struct sockaddr_storage probe_address;
  int probe_length = sizeof(probe_address);
  if (getsockname(sock, (struct sockaddr*)&probe_address, &probe_length) !=
      SOCKET_ERROR) {
    iree_atomic_store(&socket->bind_state, IREE_ASYNC_SOCKET_BIND_STATE_BOUND,
                      iree_memory_order_release);
    return iree_ok_status();  // Already bound.
  }

  int error = WSAGetLastError();
  if (error != WSAEINVAL) {
    return iree_make_status(iree_status_code_from_win32_error(error),
                            "getsockname before auto-bind failed "
                            "(WSA error %d)",
                            error);
  }

  iree_async_address_t bind_address;
  memset(&bind_address, 0, sizeof(bind_address));

  switch (socket->type) {
    case IREE_ASYNC_SOCKET_TYPE_TCP:
    case IREE_ASYNC_SOCKET_TYPE_UDP: {
      struct sockaddr_in* addr4 = (struct sockaddr_in*)bind_address.storage;
      addr4->sin_family = AF_INET;
      addr4->sin_addr.s_addr = INADDR_ANY;
      addr4->sin_port = 0;
      bind_address.length = sizeof(struct sockaddr_in);
      break;
    }
    case IREE_ASYNC_SOCKET_TYPE_TCP6:
    case IREE_ASYNC_SOCKET_TYPE_UDP6: {
      struct sockaddr_in6* addr6 = (struct sockaddr_in6*)bind_address.storage;
      addr6->sin6_family = AF_INET6;
      addr6->sin6_addr = in6addr_any;
      addr6->sin6_port = 0;
      bind_address.length = sizeof(struct sockaddr_in6);
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "auto-bind not supported for socket type %d",
                              socket->type);
  }

  return iree_async_socket_bind(socket, &bind_address);
}

// Returns true if the socket type is connectionless (UDP/UDP6).
static bool iree_async_socket_type_is_datagram(iree_async_socket_type_t type) {
  return type == IREE_ASYNC_SOCKET_TYPE_UDP ||
         type == IREE_ASYNC_SOCKET_TYPE_UDP6;
}

static void iree_async_proactor_iocp_commit_socket_connect(
    iree_async_proactor_iocp_t* proactor,
    iree_async_socket_connect_operation_t* connect_op,
    iree_async_iocp_carrier_t* carrier) {
  iree_async_socket_t* socket = connect_op->socket;
  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;
  carrier->io_handle = (uintptr_t)sock;

  // Auto-bind if needed (ConnectEx requires a bound socket; UDP connect also
  // needs it when the socket hasn't been explicitly bound yet).
  iree_status_t bind_status =
      iree_async_proactor_iocp_auto_bind_if_needed(socket, sock);
  if (!iree_status_is_ok(bind_status)) {
    iree_async_proactor_iocp_post_direct_completion(proactor, carrier,
                                                    bind_status);
    return;
  }

  // UDP connect is synchronous: it just sets the default destination address
  // in the socket. ConnectEx is TCP-only (connection-oriented). For UDP, use
  // plain connect() and post a direct completion.
  if (iree_async_socket_type_is_datagram(socket->type)) {
    const struct sockaddr* target_addr =
        (const struct sockaddr*)connect_op->address.storage;
    int addr_length = (int)connect_op->address.length;

    int result = connect(sock, target_addr, addr_length);
    if (result == SOCKET_ERROR) {
      int wsa_error = WSAGetLastError();
      iree_async_proactor_iocp_post_win32_error(proactor, carrier,
                                                (DWORD)wsa_error, "connect");
      return;
    }

    socket->state = IREE_ASYNC_SOCKET_STATE_CONNECTED;

    iree_async_proactor_iocp_post_direct_completion(proactor, carrier,
                                                    iree_ok_status());
    return;
  }

  // TCP: use ConnectEx for overlapped (asynchronous) connection.
  const struct sockaddr* target_addr =
      (const struct sockaddr*)connect_op->address.storage;
  int addr_length = (int)connect_op->address.length;

  BOOL connected = proactor->wsa_extensions.ConnectEx(
      sock, target_addr, addr_length, NULL, 0, NULL, &carrier->overlapped);

  if (!connected) {
    int wsa_error = WSAGetLastError();
    if (wsa_error != WSA_IO_PENDING) {
      iree_async_proactor_iocp_post_win32_error(proactor, carrier,
                                                (DWORD)wsa_error, "ConnectEx");
      return;
    }
  }

  socket->state = IREE_ASYNC_SOCKET_STATE_CONNECTING;
}

static void iree_async_proactor_iocp_commit_socket_recv(
    iree_async_proactor_iocp_t* proactor,
    iree_async_socket_recv_operation_t* recv_op,
    iree_async_iocp_carrier_t* carrier) {
  iree_async_socket_t* socket = recv_op->socket;
  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;
  carrier->io_handle = (uintptr_t)sock;

  recv_op->bytes_received = 0;

  carrier->data.socket_io.buffer_count = iree_async_proactor_iocp_build_wsabuf(
      carrier->data.socket_io.wsabuf, &recv_op->platform,
      recv_op->retained_buffer_regions, recv_op->base.acquired_span_count);
  carrier->data.socket_io.flags = 0;

  int result =
      WSARecv(sock, carrier->data.socket_io.wsabuf,
              carrier->data.socket_io.buffer_count, NULL,
              &carrier->data.socket_io.flags, &carrier->overlapped, NULL);
  if (result == SOCKET_ERROR) {
    int wsa_error = WSAGetLastError();
    if (wsa_error != WSA_IO_PENDING) {
      iree_async_proactor_iocp_post_win32_error(proactor, carrier,
                                                (DWORD)wsa_error, "WSARecv");
      return;
    }
  }
}

static void iree_async_proactor_iocp_commit_socket_send(
    iree_async_proactor_iocp_t* proactor,
    iree_async_socket_send_operation_t* send_op,
    iree_async_iocp_carrier_t* carrier) {
  iree_async_socket_t* socket = send_op->socket;
  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;
  carrier->io_handle = (uintptr_t)sock;

  send_op->bytes_sent = 0;

  carrier->data.socket_io.buffer_count = iree_async_proactor_iocp_build_wsabuf(
      carrier->data.socket_io.wsabuf, &send_op->platform,
      send_op->retained_buffer_regions, send_op->base.acquired_span_count);
  // MSG_MORE: silently ignored on Windows (no equivalent; TCP_NODELAY controls
  // coalescing at the socket level).
  DWORD flags = 0;

  int result = WSASend(sock, carrier->data.socket_io.wsabuf,
                       carrier->data.socket_io.buffer_count, NULL, flags,
                       &carrier->overlapped, NULL);
  if (result == SOCKET_ERROR) {
    int wsa_error = WSAGetLastError();
    if (wsa_error != WSA_IO_PENDING) {
      iree_async_proactor_iocp_post_win32_error(proactor, carrier,
                                                (DWORD)wsa_error, "WSASend");
      return;
    }
  }
}

static void iree_async_proactor_iocp_commit_socket_sendto(
    iree_async_proactor_iocp_t* proactor,
    iree_async_socket_sendto_operation_t* sendto_op,
    iree_async_iocp_carrier_t* carrier) {
  iree_async_socket_t* socket = sendto_op->socket;
  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;
  carrier->io_handle = (uintptr_t)sock;

  sendto_op->bytes_sent = 0;

  carrier->data.socket_io.buffer_count = iree_async_proactor_iocp_build_wsabuf(
      carrier->data.socket_io.wsabuf, &sendto_op->platform,
      sendto_op->retained_buffer_regions, sendto_op->base.acquired_span_count);

  const struct sockaddr* dest_addr =
      (const struct sockaddr*)sendto_op->destination.storage;
  int dest_length = (int)sendto_op->destination.length;

  int result = WSASendTo(sock, carrier->data.socket_io.wsabuf,
                         carrier->data.socket_io.buffer_count, NULL, 0,
                         dest_addr, dest_length, &carrier->overlapped, NULL);
  if (result == SOCKET_ERROR) {
    int wsa_error = WSAGetLastError();
    if (wsa_error != WSA_IO_PENDING) {
      iree_async_proactor_iocp_post_win32_error(proactor, carrier,
                                                (DWORD)wsa_error, "WSASendTo");
      return;
    }
  }
}

static void iree_async_proactor_iocp_commit_socket_recvfrom(
    iree_async_proactor_iocp_t* proactor,
    iree_async_socket_recvfrom_operation_t* recvfrom_op,
    iree_async_iocp_carrier_t* carrier) {
  iree_async_socket_t* socket = recvfrom_op->socket;
  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;
  carrier->io_handle = (uintptr_t)sock;

  recvfrom_op->bytes_received = 0;
  memset(&recvfrom_op->sender, 0, sizeof(recvfrom_op->sender));

  carrier->data.socket_io.buffer_count = iree_async_proactor_iocp_build_wsabuf(
      carrier->data.socket_io.wsabuf, &recvfrom_op->platform,
      recvfrom_op->retained_buffer_regions,
      recvfrom_op->base.acquired_span_count);
  carrier->data.socket_io.flags = 0;

  // WSARecvFrom writes the actual sender address length asynchronously at
  // completion time. The length must be stored in the carrier (heap-allocated,
  // persists until completion), not a stack local.
  carrier->data.socket_io.sender_address_length =
      (int)sizeof(recvfrom_op->sender.storage);

  int result = WSARecvFrom(sock, carrier->data.socket_io.wsabuf,
                           carrier->data.socket_io.buffer_count, NULL,
                           &carrier->data.socket_io.flags,
                           (struct sockaddr*)recvfrom_op->sender.storage,
                           &carrier->data.socket_io.sender_address_length,
                           &carrier->overlapped, NULL);
  if (result == SOCKET_ERROR) {
    int wsa_error = WSAGetLastError();
    if (wsa_error != WSA_IO_PENDING) {
      iree_async_proactor_iocp_post_win32_error(
          proactor, carrier, (DWORD)wsa_error, "WSARecvFrom");
      return;
    }
  }
}

static void iree_async_proactor_iocp_commit_socket_recv_pool(
    iree_async_proactor_iocp_t* proactor,
    iree_async_socket_recv_pool_operation_t* recv_pool_op,
    iree_async_iocp_carrier_t* carrier) {
  iree_async_socket_t* socket = recv_pool_op->socket;
  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;
  carrier->io_handle = (uintptr_t)sock;

  recv_pool_op->bytes_received = 0;
  memset(&recv_pool_op->lease, 0, sizeof(recv_pool_op->lease));

  // Buffer availability is dynamic. Once the batch is accepted, exhaustion is
  // an asynchronous operation result rather than a submit rejection.
  iree_status_t status =
      iree_async_buffer_pool_acquire(recv_pool_op->pool, &recv_pool_op->lease);
  if (!iree_status_is_ok(status)) {
    iree_async_proactor_iocp_post_direct_completion(proactor, carrier, status);
    return;
  }

  carrier->data.recv_pool.wsabuf.buf =
      (char*)iree_async_span_ptr(recv_pool_op->lease.span);
  carrier->data.recv_pool.wsabuf.len = (ULONG)recv_pool_op->lease.span.length;
  carrier->data.recv_pool.flags = 0;

  int result =
      WSARecv(sock, &carrier->data.recv_pool.wsabuf, 1, NULL,
              &carrier->data.recv_pool.flags, &carrier->overlapped, NULL);
  if (result == SOCKET_ERROR) {
    int wsa_error = WSAGetLastError();
    if (wsa_error != WSA_IO_PENDING) {
      iree_async_buffer_lease_release(&recv_pool_op->lease);
      iree_async_proactor_iocp_post_win32_error(proactor, carrier,
                                                (DWORD)wsa_error, "WSARecv");
      return;
    }
  }
}

static void iree_async_proactor_iocp_commit_socket_close(
    iree_async_proactor_iocp_t* proactor,
    iree_async_socket_close_operation_t* close_op,
    iree_async_iocp_carrier_t* carrier) {
  iree_async_socket_t* socket = close_op->socket;
  SOCKET sock = (SOCKET)socket->primitive.value.win32_handle;

  // The native close is synchronous; the reserved carrier delivers its
  // terminal result. Preserve the socket handle on failure so its eventual
  // destroy can retry instead of leaking it.
  int close_error = 0;
  if (sock != INVALID_SOCKET) {
    if (closesocket(sock) == SOCKET_ERROR) {
      close_error = WSAGetLastError();
    }
  }
  if (close_error == 0) {
    socket->primitive = iree_async_primitive_none();
    socket->state = IREE_ASYNC_SOCKET_STATE_CLOSED;
  }

  iree_status_t status =
      close_error == 0
          ? iree_ok_status()
          : iree_make_status(iree_status_code_from_win32_error(close_error),
                             "closesocket failed (WSA error %d)", close_error);
  iree_async_proactor_iocp_post_direct_completion(proactor, carrier, status);
}

//===----------------------------------------------------------------------===//
// Semaphore operations
//===----------------------------------------------------------------------===//

static void iree_async_proactor_iocp_commit_semaphore_signal(
    iree_async_proactor_iocp_t* proactor,
    iree_async_semaphore_signal_operation_t* signal_op,
    iree_async_iocp_carrier_t* carrier) {
  // Signal all semaphores synchronously. On first error, break.
  iree_status_t op_status = iree_ok_status();
  for (iree_host_size_t i = 0; i < signal_op->count; ++i) {
    iree_status_t status = iree_async_semaphore_signal(
        signal_op->semaphores[i], signal_op->values[i], signal_op->frontier);
    if (!iree_status_is_ok(status)) {
      op_status = status;
      break;
    }
  }

  iree_async_proactor_iocp_post_direct_completion(proactor, carrier, op_status);
}

// Enqueues a terminal semaphore wait tracker and wakes the poll owner.
static void iree_async_proactor_iocp_enqueue_semaphore_wait(
    void* user_data, iree_atomic_slist_entry_t* entry) {
  iree_async_proactor_iocp_t* proactor = (iree_async_proactor_iocp_t*)user_data;
  iree_atomic_slist_push(&proactor->pending_semaphore_waits, entry);
  iree_async_proactor_iocp_wake(&proactor->base);
}

// Commits a semaphore wait after the batch has been accepted. Immediate waits
// and tracker-allocation failures use the reserved carrier for poll-owned
// completion. Deferred waits release that carrier after the tracker has
// retained every referenced semaphore and taken ownership of the continuation.
static void iree_async_proactor_iocp_commit_semaphore_wait(
    iree_async_proactor_iocp_t* proactor,
    iree_async_semaphore_wait_operation_t* wait_op,
    iree_async_iocp_carrier_t* carrier) {
  if (iree_async_semaphore_wait_is_satisfied(wait_op)) {
    iree_async_proactor_iocp_post_direct_completion(proactor, carrier,
                                                    iree_ok_status());
    return;
  }

  iree_async_semaphore_wait_enqueue_callback_t enqueue_callback = {
      .fn = iree_async_proactor_iocp_enqueue_semaphore_wait,
      .user_data = proactor,
  };
  iree_async_semaphore_wait_tracker_t* tracker = NULL;
  iree_status_t status = iree_async_semaphore_wait_tracker_allocate(
      &proactor->semaphore_wait_context, wait_op, enqueue_callback,
      proactor->base.allocator, &tracker);
  if (!iree_status_is_ok(status)) {
    iree_async_proactor_iocp_post_direct_completion(proactor, carrier, status);
    return;
  }

  iree_async_proactor_iocp_release_carrier(proactor, carrier);
  iree_async_semaphore_wait_tracker_register_timepoints(tracker);
}

//===----------------------------------------------------------------------===//
// Notification operation submit
//===----------------------------------------------------------------------===//

// NOTIFICATION_WAIT: wait-token selection, routed through pending_queue so
// the poll thread can either complete immediately (epoch already advanced)
// or link into the notification's pending_waits list.
static void iree_async_proactor_iocp_commit_notification_wait(
    iree_async_proactor_iocp_t* proactor,
    iree_async_notification_wait_operation_t* wait_op) {
  // Capture the epoch token at submit time unless the caller provided one.
  // Queries shared state (not the local epoch field) because shared
  // notifications have their epoch in SHM.
  if (!iree_all_bits_set(wait_op->wait_flags,
                         IREE_ASYNC_NOTIFICATION_WAIT_FLAG_USE_WAIT_TOKEN)) {
    wait_op->wait_token =
        iree_async_notification_query_epoch(wait_op->notification);
  }
  iree_async_proactor_iocp_push_pending(proactor, &wait_op->base);
}

// NOTIFICATION_SIGNAL: signal synchronously and post direct completion.
static void iree_async_proactor_iocp_commit_notification_signal(
    iree_async_proactor_iocp_t* proactor,
    iree_async_notification_signal_operation_t* signal_op,
    iree_async_iocp_carrier_t* carrier) {
  // Perform the signal synchronously.
  iree_async_notification_signal(signal_op->notification,
                                 signal_op->wake_count);
  // The native wake APIs do not report the number of observers woken.
  signal_op->woken_count = -1;

  iree_async_proactor_iocp_post_direct_completion(proactor, carrier,
                                                  iree_ok_status());
}

//===----------------------------------------------------------------------===//
// Message operation submit
//===----------------------------------------------------------------------===//

static void iree_async_proactor_iocp_commit_message(
    iree_async_proactor_iocp_t* proactor,
    iree_async_message_operation_t* message,
    iree_async_iocp_carrier_t* carrier) {
  const bool skip_source_completion = iree_any_bit_set(
      message->message_flags, IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION);
  iree_async_proactor_t* target = message->target;
  iree_async_proactor_iocp_t* target_iocp =
      iree_async_proactor_iocp_cast(target);

  iree_async_message_pool_entry_t* target_entry =
      (iree_async_message_pool_entry_t*)
          message->platform.software.reserved_entry;
  message->platform.software.reserved_entry = NULL;
  IREE_ASSERT(target_entry);
  iree_async_message_pool_publish(&target_iocp->message_pool, target_entry,
                                  message->message_data);
  target->vtable->wake(target);

  if (skip_source_completion) {
    return;
  }
  iree_async_proactor_iocp_post_direct_completion(proactor, carrier,
                                                  iree_ok_status());
}

//===----------------------------------------------------------------------===//
// File I/O submit handlers
//===----------------------------------------------------------------------===//

static void iree_async_proactor_iocp_commit_file_open(
    iree_async_proactor_iocp_t* proactor,
    iree_async_file_open_operation_t* open_op,
    iree_async_iocp_carrier_t* carrier) {
  IREE_TRACE_ZONE_BEGIN(z0);
  open_op->opened_file = NULL;

  // Translate open flags to Windows CreateFileW parameters.
  DWORD desired_access = 0;
  DWORD share_mode = FILE_SHARE_READ;
  DWORD creation_disposition = OPEN_EXISTING;
  DWORD flags_and_attributes = FILE_FLAG_OVERLAPPED;

  if (iree_any_bit_set(open_op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_READ)) {
    desired_access |= GENERIC_READ;
  }
  if (iree_any_bit_set(open_op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_WRITE)) {
    desired_access |= GENERIC_WRITE;
    share_mode = 0;  // Exclusive write access.
  }
  if (iree_any_bit_set(open_op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_APPEND)) {
    desired_access |= FILE_APPEND_DATA;
    share_mode = 0;
  }

  // Determine creation disposition based on flags.
  bool create =
      iree_any_bit_set(open_op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_CREATE);
  bool truncate =
      iree_any_bit_set(open_op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_TRUNCATE);
  if (create && truncate) {
    creation_disposition = CREATE_ALWAYS;
  } else if (create) {
    creation_disposition = OPEN_ALWAYS;
  } else if (truncate) {
    creation_disposition = TRUNCATE_EXISTING;
  } else {
    creation_disposition = OPEN_EXISTING;
  }

  if (iree_any_bit_set(open_op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_DIRECT)) {
    flags_and_attributes |= FILE_FLAG_NO_BUFFERING;
  }

  // Convert from the runtime's UTF-8 representation only at the Win32 API
  // boundary. The converted path is absolute and extended-length.
  wchar_t* win32_path = NULL;
  iree_status_t status =
      iree_file_path_to_win32(iree_make_cstring_view(open_op->path),
                              proactor->base.allocator, &win32_path);
  if (!iree_status_is_ok(status)) {
    IREE_TRACE_ZONE_END(z0);
    iree_async_proactor_iocp_post_direct_completion(proactor, carrier, status);
    return;
  }

  HANDLE file_handle =
      CreateFileW(win32_path, desired_access, share_mode, NULL,
                  creation_disposition, flags_and_attributes, NULL);
  DWORD open_error =
      file_handle == INVALID_HANDLE_VALUE ? GetLastError() : ERROR_SUCCESS;
  iree_allocator_free(proactor->base.allocator, win32_path);
  if (file_handle == INVALID_HANDLE_VALUE) {
    IREE_TRACE_ZONE_END(z0);
    iree_async_proactor_iocp_post_win32_error(proactor, carrier, open_error,
                                              "CreateFileW");
    return;
  }

  // Import the file handle (associates with IOCP port).
  iree_async_primitive_t primitive =
      iree_async_primitive_from_win32_handle((uintptr_t)file_handle);
  iree_async_file_flags_t flags =
      iree_any_bit_set(open_op->open_flags, IREE_ASYNC_FILE_OPEN_FLAG_DIRECT)
          ? IREE_ASYNC_FILE_FLAG_DIRECT
          : IREE_ASYNC_FILE_FLAG_NONE;
  status = iree_async_file_import(&proactor->base, primitive, flags,
                                  &open_op->opened_file);
  if (!iree_status_is_ok(status)) {
    CloseHandle(file_handle);
    IREE_TRACE_ZONE_END(z0);
    iree_async_proactor_iocp_post_direct_completion(proactor, carrier, status);
    return;
  }

  IREE_TRACE_ZONE_END(z0);
  iree_async_proactor_iocp_post_direct_completion(proactor, carrier,
                                                  iree_ok_status());
}

static void iree_async_proactor_iocp_commit_file_read(
    iree_async_proactor_iocp_t* proactor,
    iree_async_file_read_operation_t* read_op,
    iree_async_iocp_carrier_t* carrier) {
  iree_async_file_t* file = read_op->file;
  HANDLE file_handle = (HANDLE)file->primitive.value.win32_handle;
  carrier->io_handle = (uintptr_t)file_handle;

  read_op->bytes_read = 0;

  // Encode file offset in the OVERLAPPED structure.
  carrier->overlapped.Offset = (DWORD)(read_op->offset & 0xFFFFFFFF);
  carrier->overlapped.OffsetHigh = (DWORD)(read_op->offset >> 32);

  void* buffer_ptr = iree_async_span_ptr(read_op->buffer);
  DWORD buffer_length = (DWORD)read_op->buffer.length;

  BOOL read_ok = ReadFile(file_handle, buffer_ptr, buffer_length, NULL,
                          &carrier->overlapped);
  if (!read_ok) {
    DWORD error = GetLastError();
    if (error != ERROR_IO_PENDING) {
      iree_async_proactor_iocp_post_win32_error(proactor, carrier, error,
                                                "ReadFile");
      return;
    }
  }
}

static void iree_async_proactor_iocp_commit_file_write(
    iree_async_proactor_iocp_t* proactor,
    iree_async_file_write_operation_t* write_op,
    iree_async_iocp_carrier_t* carrier) {
  iree_async_file_t* file = write_op->file;
  HANDLE file_handle = (HANDLE)file->primitive.value.win32_handle;
  carrier->io_handle = (uintptr_t)file_handle;

  write_op->bytes_written = 0;

  // Encode file offset in the OVERLAPPED structure.
  // For APPEND mode, the caller opened the file with FILE_APPEND_DATA.
  // Windows handles append semantics at the kernel level — the offset
  // is still specified but the kernel atomically appends.
  carrier->overlapped.Offset = (DWORD)(write_op->offset & 0xFFFFFFFF);
  carrier->overlapped.OffsetHigh = (DWORD)(write_op->offset >> 32);

  const void* buffer_ptr = iree_async_span_ptr(write_op->buffer);
  DWORD buffer_length = (DWORD)write_op->buffer.length;

  BOOL write_ok = WriteFile(file_handle, buffer_ptr, buffer_length, NULL,
                            &carrier->overlapped);
  if (!write_ok) {
    DWORD error = GetLastError();
    if (error != ERROR_IO_PENDING) {
      iree_async_proactor_iocp_post_win32_error(proactor, carrier, error,
                                                "WriteFile");
      return;
    }
  }
}

static void iree_async_proactor_iocp_commit_file_close(
    iree_async_proactor_iocp_t* proactor,
    iree_async_file_close_operation_t* close_op,
    iree_async_iocp_carrier_t* carrier) {
  iree_async_file_t* file = close_op->file;
  HANDLE file_handle = (HANDLE)file->primitive.value.win32_handle;

  // The native close is synchronous; the reserved carrier delivers its
  // terminal result. Preserve a handle that failed to close so the resource
  // destructor can retry instead of losing ownership.
  DWORD close_error = ERROR_SUCCESS;
  if (file_handle != NULL && file_handle != INVALID_HANDLE_VALUE) {
    if (!CloseHandle(file_handle)) {
      close_error = GetLastError();
    }
  }
  if (close_error == ERROR_SUCCESS) {
    file->primitive.value.win32_handle = 0;
  }

  iree_status_t status =
      close_error == ERROR_SUCCESS
          ? iree_ok_status()
          : iree_make_status(iree_status_code_from_win32_error(close_error),
                             "CloseHandle failed (Win32 error %lu)",
                             (unsigned long)close_error);
  iree_async_proactor_iocp_post_direct_completion(proactor, carrier, status);
}

//===----------------------------------------------------------------------===//
// Batch reservation and commit
//===----------------------------------------------------------------------===//

static bool iree_async_proactor_iocp_carrier_type_for_operation(
    const iree_async_operation_t* operation,
    iree_async_iocp_carrier_type_t* out_carrier_type) {
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
      *out_carrier_type = IREE_ASYNC_IOCP_CARRIER_ACCEPT;
      return true;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
      *out_carrier_type = IREE_ASYNC_IOCP_CARRIER_CONNECT;
      return true;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
      *out_carrier_type = IREE_ASYNC_IOCP_CARRIER_SOCKET_IO;
      return true;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
      *out_carrier_type = IREE_ASYNC_IOCP_CARRIER_RECV_POOL;
      return true;
    case IREE_ASYNC_OPERATION_TYPE_FILE_READ:
    case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE:
      *out_carrier_type = IREE_ASYNC_IOCP_CARRIER_FILE_IO;
      return true;
    case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL:
    case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT:
    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL:
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE:
    case IREE_ASYNC_OPERATION_TYPE_FILE_OPEN:
    case IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE:
      *out_carrier_type = IREE_ASYNC_IOCP_CARRIER_DIRECT;
      return true;
    case IREE_ASYNC_OPERATION_TYPE_MESSAGE: {
      const iree_async_message_operation_t* message =
          (const iree_async_message_operation_t*)operation;
      if (iree_any_bit_set(message->message_flags,
                           IREE_ASYNC_MESSAGE_FLAG_SKIP_SOURCE_COMPLETION)) {
        return false;
      }
      *out_carrier_type = IREE_ASYNC_IOCP_CARRIER_DIRECT;
      return true;
    }
    default:
      return false;
  }
}

static iree_status_t iree_async_proactor_iocp_reserve_operation(
    iree_async_proactor_iocp_t* proactor, iree_async_operation_t* operation) {
  if (operation->type == IREE_ASYNC_OPERATION_TYPE_MESSAGE) {
    iree_async_message_operation_t* message =
        (iree_async_message_operation_t*)operation;
    iree_async_proactor_iocp_t* target =
        iree_async_proactor_iocp_cast(message->target);
    iree_async_message_pool_entry_t* target_entry = NULL;
    IREE_RETURN_IF_ERROR(
        iree_async_message_pool_acquire(&target->message_pool, &target_entry));
    message->platform.software.reserved_entry = target_entry;
  }

  iree_async_iocp_carrier_type_t carrier_type = IREE_ASYNC_IOCP_CARRIER_DIRECT;
  if (!iree_async_proactor_iocp_carrier_type_for_operation(operation,
                                                           &carrier_type)) {
    return iree_ok_status();
  }

  iree_async_iocp_carrier_t* carrier = NULL;
  IREE_RETURN_IF_ERROR(iree_async_proactor_iocp_acquire_carrier(
      proactor, carrier_type, operation, /*io_handle=*/0, &carrier));
  operation->next = (iree_async_operation_t*)carrier;
  return iree_ok_status();
}

static void iree_async_proactor_iocp_rollback_reservations(
    iree_async_proactor_iocp_t* proactor,
    iree_async_operation_list_t operations) {
  iree_async_continuation_chain_iterator_t iterator =
      iree_async_continuation_chain_iterator_make(operations);
  iree_async_operation_t* operation = NULL;
  while ((operation = iree_async_continuation_chain_iterator_next(&iterator)) !=
         NULL) {
    if (operation->next) {
      iree_async_iocp_carrier_t* carrier =
          (iree_async_iocp_carrier_t*)operation->next;
      operation->next = NULL;
      iree_async_proactor_iocp_release_carrier(proactor, carrier);
    }
    if (operation->type == IREE_ASYNC_OPERATION_TYPE_MESSAGE) {
      iree_async_message_operation_t* message =
          (iree_async_message_operation_t*)operation;
      iree_async_message_pool_entry_t* target_entry =
          (iree_async_message_pool_entry_t*)
              message->platform.software.reserved_entry;
      if (target_entry) {
        iree_async_proactor_iocp_t* target =
            iree_async_proactor_iocp_cast(message->target);
        message->platform.software.reserved_entry = NULL;
        iree_async_message_pool_release(&target->message_pool, target_entry);
      }
    }
  }
}

static void iree_async_proactor_iocp_commit_operation(
    iree_async_proactor_iocp_t* proactor, iree_async_operation_t* operation) {
  if (operation->type == IREE_ASYNC_OPERATION_TYPE_SEQUENCE) {
    iree_async_sequence_prepare_for_submission(
        (iree_async_sequence_operation_t*)operation);
  } else {
    iree_async_operation_clear_internal_flags(operation);
  }

  iree_async_iocp_carrier_t* carrier =
      (iree_async_iocp_carrier_t*)operation->next;
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_NOP:
    case IREE_ASYNC_OPERATION_TYPE_TIMER:
    case IREE_ASYNC_OPERATION_TYPE_EVENT_WAIT:
    case IREE_ASYNC_OPERATION_TYPE_SEQUENCE:
    case IREE_ASYNC_OPERATION_TYPE_HANDLE_POLL:
      iree_async_proactor_iocp_push_pending(proactor, operation);
      return;

    case IREE_ASYNC_OPERATION_TYPE_SOCKET_ACCEPT:
      iree_async_proactor_iocp_commit_socket_accept(
          proactor, (iree_async_socket_accept_operation_t*)operation, carrier);
      return;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
      iree_async_proactor_iocp_commit_socket_connect(
          proactor, (iree_async_socket_connect_operation_t*)operation, carrier);
      return;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
      iree_async_proactor_iocp_commit_socket_recv(
          proactor, (iree_async_socket_recv_operation_t*)operation, carrier);
      return;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
      iree_async_proactor_iocp_commit_socket_send(
          proactor, (iree_async_socket_send_operation_t*)operation, carrier);
      return;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO:
      iree_async_proactor_iocp_commit_socket_sendto(
          proactor, (iree_async_socket_sendto_operation_t*)operation, carrier);
      return;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
      iree_async_proactor_iocp_commit_socket_recvfrom(
          proactor, (iree_async_socket_recvfrom_operation_t*)operation,
          carrier);
      return;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
      iree_async_proactor_iocp_commit_socket_recv_pool(
          proactor, (iree_async_socket_recv_pool_operation_t*)operation,
          carrier);
      return;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CLOSE:
      iree_async_proactor_iocp_commit_socket_close(
          proactor, (iree_async_socket_close_operation_t*)operation, carrier);
      return;

    case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_SIGNAL:
      iree_async_proactor_iocp_commit_semaphore_signal(
          proactor, (iree_async_semaphore_signal_operation_t*)operation,
          carrier);
      return;
    case IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT:
      iree_async_proactor_iocp_commit_semaphore_wait(
          proactor, (iree_async_semaphore_wait_operation_t*)operation, carrier);
      return;
    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_WAIT:
      iree_async_proactor_iocp_commit_notification_wait(
          proactor, (iree_async_notification_wait_operation_t*)operation);
      return;
    case IREE_ASYNC_OPERATION_TYPE_NOTIFICATION_SIGNAL:
      iree_async_proactor_iocp_commit_notification_signal(
          proactor, (iree_async_notification_signal_operation_t*)operation,
          carrier);
      return;
    case IREE_ASYNC_OPERATION_TYPE_MESSAGE:
      iree_async_proactor_iocp_commit_message(
          proactor, (iree_async_message_operation_t*)operation, carrier);
      return;

    case IREE_ASYNC_OPERATION_TYPE_FILE_OPEN:
      iree_async_proactor_iocp_commit_file_open(
          proactor, (iree_async_file_open_operation_t*)operation, carrier);
      return;
    case IREE_ASYNC_OPERATION_TYPE_FILE_READ:
      iree_async_proactor_iocp_commit_file_read(
          proactor, (iree_async_file_read_operation_t*)operation, carrier);
      return;
    case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE:
      iree_async_proactor_iocp_commit_file_write(
          proactor, (iree_async_file_write_operation_t*)operation, carrier);
      return;
    case IREE_ASYNC_OPERATION_TYPE_FILE_CLOSE:
      iree_async_proactor_iocp_commit_file_close(
          proactor, (iree_async_file_close_operation_t*)operation, carrier);
      return;

    case IREE_ASYNC_OPERATION_TYPE_FUTEX_WAIT:
    case IREE_ASYNC_OPERATION_TYPE_FUTEX_WAKE:
    default:
      IREE_ASSERT_UNREACHABLE("operation type must be validated");
      IREE_BUILTIN_UNREACHABLE();
  }
}

static iree_status_t iree_async_proactor_iocp_submit_prepared(
    iree_async_proactor_iocp_t* proactor,
    iree_async_operation_list_t operations) {
  for (iree_host_size_t i = 0; i < operations.count; ++i) {
    if (operations.values[i]->resources_acquired) {
      // Accepted continuations were validated with their original batch and
      // may no longer have accessible caller-owned descriptor arrays.
      continue;
    }
    IREE_RETURN_IF_ERROR(iree_async_proactor_iocp_validate_operation(
        proactor, operations.values[i]));
  }

  iree_async_continuation_chain_iterator_t clear_iterator =
      iree_async_continuation_chain_iterator_make(operations);
  iree_async_operation_t* operation = NULL;
  while ((operation = iree_async_continuation_chain_iterator_next(
              &clear_iterator)) != NULL) {
    operation->next = NULL;
    if (operation->type == IREE_ASYNC_OPERATION_TYPE_MESSAGE) {
      ((iree_async_message_operation_t*)operation)
          ->platform.software.reserved_entry = NULL;
    }
  }

  iree_async_continuation_chain_iterator_t reserve_iterator =
      iree_async_continuation_chain_iterator_make(operations);
  while ((operation = iree_async_continuation_chain_iterator_next(
              &reserve_iterator)) != NULL) {
    iree_status_t status =
        iree_async_proactor_iocp_reserve_operation(proactor, operation);
    if (!iree_status_is_ok(status)) {
      iree_async_proactor_iocp_rollback_reservations(proactor, operations);
      return status;
    }
  }

  // The complete list is now accepted. Acquire linked successors before any
  // active carrier can complete and before submit-scoped descriptors expire.
  iree_async_operation_list_acquire_resources(operations);

  iree_async_continuation_chain_iterator_t commit_iterator =
      iree_async_continuation_chain_iterator_make(operations);
  while ((operation = iree_async_continuation_chain_iterator_next(
              &commit_iterator)) != NULL) {
    iree_async_proactor_iocp_commit_operation(proactor, operation);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Submit
//===----------------------------------------------------------------------===//

iree_status_t iree_async_proactor_iocp_submit(
    iree_async_proactor_t* base_proactor,
    iree_async_operation_list_t operations) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_async_proactor_iocp_t* proactor =
      iree_async_proactor_iocp_cast(base_proactor);

  if (iree_atomic_load(&proactor->shutdown_requested,
                       iree_memory_order_acquire)) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_ABORTED, "proactor is shutting down");
  }

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_continuation_prepare_batch(operations));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_async_proactor_iocp_submit_prepared(proactor, operations));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_async_proactor_iocp_submit_continuation(
    void* user_data, iree_async_operation_t* chain_head) {
  iree_async_operation_t* operations[] = {chain_head};
  return iree_async_proactor_iocp_submit_prepared(
      (iree_async_proactor_iocp_t*)user_data,
      iree_async_operation_list_make(operations, IREE_ARRAYSIZE(operations)));
}

#endif  // IREE_PLATFORM_WINDOWS
