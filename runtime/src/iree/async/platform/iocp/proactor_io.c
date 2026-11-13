// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Native I/O results, cancellation identity, and terminal carrier retirement.

#include <string.h>

#include "iree/async/buffer_pool.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/net.h"
#include "iree/async/platform/iocp/proactor.h"
#include "iree/async/platform/iocp/socket.h"
#include "iree/async/util/continuation.h"
#include "iree/async/util/operation_completion.h"

#if defined(IREE_PLATFORM_WINDOWS)

iree_async_iocp_carrier_t* iree_async_proactor_iocp_acquire_io_cancel(
    iree_async_operation_t* operation) {
  // Admission and terminal close share this word. Acquiring first prevents
  // completion from clearing next or recycling its carrier until release.
  int32_t flags =
      iree_atomic_load(&operation->internal_flags, iree_memory_order_acquire);
  for (;;) {
    int32_t desired = flags | IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED;
    const bool retired =
        iree_any_bit_set(flags, IREE_ASYNC_IOCP_INTERNAL_FLAG_IO_RETIRED);
    if (!retired) {
      IREE_ASSERT((flags & IREE_ASYNC_IOCP_INTERNAL_IO_CANCEL_REFERENCE_MASK) !=
                      IREE_ASYNC_IOCP_INTERNAL_IO_CANCEL_REFERENCE_MASK,
                  "native cancellation reference count overflow");
      desired += IREE_ASYNC_IOCP_INTERNAL_IO_CANCEL_REFERENCE;
    }
    if (iree_atomic_compare_exchange_weak(&operation->internal_flags, &flags,
                                          desired, iree_memory_order_acq_rel,
                                          iree_memory_order_acquire)) {
      if (retired) {
        return NULL;
      }
      break;
    }
  }

  iree_async_iocp_carrier_t* carrier =
      (iree_async_iocp_carrier_t*)operation->next;
  if (!carrier) {
    // Final completion cleared backend flags after returning the carrier. The
    // caller still owns operation storage through cancel(), but no native
    // identity remains. Submission cannot reuse that storage during this call.
    iree_atomic_fetch_sub(&operation->internal_flags,
                          IREE_ASYNC_IOCP_INTERNAL_IO_CANCEL_REFERENCE,
                          iree_memory_order_release);
  }
  return carrier;
}

void iree_async_proactor_iocp_release_io_cancel(
    iree_async_proactor_iocp_t* proactor, iree_async_iocp_carrier_t* carrier) {
  int32_t previous = iree_atomic_fetch_sub(
      &carrier->operation->internal_flags,
      IREE_ASYNC_IOCP_INTERNAL_IO_CANCEL_REFERENCE, iree_memory_order_acq_rel);
  if (iree_any_bit_set(previous, IREE_ASYNC_IOCP_INTERNAL_FLAG_IO_RETIRED) &&
      (previous & IREE_ASYNC_IOCP_INTERNAL_IO_CANCEL_REFERENCE_MASK) ==
          IREE_ASYNC_IOCP_INTERNAL_IO_CANCEL_REFERENCE) {
    // Completion published its terminal result before closing the identity.
    // Only the last borrower can transfer this held result to the poll owner.
    iree_async_iocp_carrier_slist_push(&proactor->pending_completions, carrier);
    iree_async_proactor_iocp_wake(&proactor->base);
  }
}

static iree_status_t iree_async_proactor_iocp_cancel_native_io(
    iree_async_iocp_carrier_t* carrier) {
  if (!CancelIoEx((HANDLE)carrier->io_handle, &carrier->overlapped)) {
    DWORD error = GetLastError();
    if (error != ERROR_NOT_FOUND) {
      return iree_make_status(iree_status_code_from_win32_error(error),
                              "CancelIoEx failed: %lu", (unsigned long)error);
    }
  }
  return iree_ok_status();
}

iree_status_t iree_async_proactor_iocp_cancel_io(
    iree_async_proactor_iocp_t* proactor, iree_async_operation_t* operation) {
  iree_async_iocp_carrier_t* carrier =
      iree_async_proactor_iocp_acquire_io_cancel(operation);
  if (!carrier) {
    return iree_ok_status();
  }
  iree_status_t status = iree_async_proactor_iocp_cancel_native_io(carrier);
  iree_async_proactor_iocp_release_io_cancel(proactor, carrier);
  return status;
}

iree_status_t iree_async_proactor_iocp_finish_io_issue(
    iree_async_iocp_carrier_t* carrier) {
  // The RMW pairs with cancellation's intent publication. Either issue sees
  // that intent, or the cancelling thread reaches an already-issued request.
  // The polling owner retains this carrier regardless of cancellation outcome.
  int32_t flags = iree_atomic_fetch_or(&carrier->operation->internal_flags, 0,
                                       iree_memory_order_acq_rel);
  if (iree_any_bit_set(flags, IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED)) {
    return iree_async_proactor_iocp_cancel_native_io(carrier);
  }
  return iree_ok_status();
}

// Returns the socket associated with a socket I/O operation, or NULL for
// non-socket operations and operations where failure doesn't indicate a broken
// socket (accept errors reflect transient resource issues, not a broken
// listener; close errors are moot).
static iree_async_socket_t* iree_async_proactor_iocp_socket_from_io_operation(
    iree_async_operation_t* operation) {
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_CONNECT:
      return ((iree_async_socket_connect_operation_t*)operation)->socket;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV:
      return ((iree_async_socket_recv_operation_t*)operation)->socket;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND:
      return ((iree_async_socket_send_operation_t*)operation)->socket;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL:
      return ((iree_async_socket_recv_pool_operation_t*)operation)->socket;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM:
      return ((iree_async_socket_recvfrom_operation_t*)operation)->socket;
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO:
      return ((iree_async_socket_sendto_operation_t*)operation)->socket;
    default:
      return NULL;
  }
}

// Dispatches completion for a single operation: sticky failure on socket,
// resource release, user callback, then linked continuation callbacks.
//
// For intermediate multishot completions (MORE flag set), resource release and
// linked continuation dispatch are skipped. The operation is still in flight:
// resources must remain retained until the final completion, and linked
// continuations should only fire once at the end of the multishot sequence.
void iree_async_proactor_iocp_dispatch_completion(
    iree_async_proactor_iocp_t* proactor, iree_async_operation_t* operation,
    iree_status_t status, iree_async_completion_flags_t flags,
    iree_host_size_t* completed_count) {
  // Set sticky failure on the socket when I/O completes with an error.
  if (!iree_status_is_ok(status)) {
    iree_async_socket_t* socket =
        iree_async_proactor_iocp_socket_from_io_operation(operation);
    if (socket) {
      iree_async_socket_set_failure(socket, iree_status_code(status));
    }
  }
  iree_async_continuation_t continuation = {0};
  if (!iree_any_bit_set(flags, IREE_ASYNC_COMPLETION_FLAG_MORE)) {
    iree_async_operation_t* chain_head =
        iree_async_continuation_take(operation);
    continuation = iree_async_continuation_begin(
        iree_async_proactor_iocp_submit_continuation, proactor, chain_head,
        iree_status_code(status));
  }
  *completed_count += iree_async_operation_complete(operation, status, flags);
  *completed_count += iree_async_continuation_finish(&continuation);
}

// Completes a resolved result from native I/O, a synthetic packet, or the
// owner-ready queue. Closing the identity joins all earlier cancellation
// borrows before returning the carrier or any accepted resource ownership.
void iree_async_proactor_iocp_complete_direct(
    iree_async_proactor_iocp_t* proactor, iree_async_iocp_carrier_t* carrier,
    iree_host_size_t* completed_count) {
  iree_async_operation_t* operation = carrier->operation;
  if (carrier->type == IREE_ASYNC_IOCP_CARRIER_IO_RESULT) {
    int32_t previous = iree_atomic_fetch_or(
        &operation->internal_flags, IREE_ASYNC_IOCP_INTERNAL_FLAG_IO_RETIRED,
        iree_memory_order_acq_rel);
    if (previous & IREE_ASYNC_IOCP_INTERNAL_IO_CANCEL_REFERENCE_MASK) {
      return;
    }
  }
  iree_status_t status = carrier->data.direct.status;
  carrier->data.direct.status = iree_ok_status();
  operation->next = NULL;
  iree_async_proactor_iocp_release_carrier(proactor, carrier);
  iree_async_proactor_iocp_dispatch_completion(proactor, operation, status,
                                               IREE_ASYNC_COMPLETION_FLAG_NONE,
                                               completed_count);
}

void iree_async_proactor_iocp_complete_io_result(
    iree_async_proactor_iocp_t* proactor, iree_async_iocp_carrier_t* carrier,
    iree_status_t status, iree_host_size_t* completed_count) {
  if (carrier->type != IREE_ASYNC_IOCP_CARRIER_DIRECT) {
    carrier->type = IREE_ASYNC_IOCP_CARRIER_IO_RESULT;
  }
  carrier->data.direct.status = status;
  iree_async_proactor_iocp_complete_direct(proactor, carrier, completed_count);
}

static bool iree_async_proactor_iocp_operation_is_cancelled(
    iree_async_operation_t* operation) {
  return iree_any_bit_set(iree_async_operation_load_internal_flags(operation),
                          IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED);
}

// Retrieves the overlapped I/O result for a socket carrier and checks for
// cancellation. Returns the status to use for dispatch.
static iree_status_t iree_async_proactor_iocp_get_socket_io_result(
    iree_async_iocp_carrier_t* carrier, iree_async_operation_t* operation,
    const char* error_context, DWORD* out_bytes_transferred, DWORD* out_flags) {
  *out_bytes_transferred = 0;
  *out_flags = 0;
  iree_status_t io_status = iree_ok_status();

  BOOL overlapped_ok =
      WSAGetOverlappedResult((SOCKET)carrier->io_handle, &carrier->overlapped,
                             out_bytes_transferred, FALSE, out_flags);
  if (!overlapped_ok) {
    int wsa_error = WSAGetLastError();
    if (wsa_error == WSAENOTSOCK) {
      // Socket was closed while this I/O was pending. The kernel posted
      // ERROR_OPERATION_ABORTED but the handle is now invalid so
      // WSAGetOverlappedResult can't determine the provider.
      io_status = iree_status_from_code(IREE_STATUS_CANCELLED);
    } else {
      io_status =
          iree_make_status(iree_status_code_from_win32_error(wsa_error),
                           "%s (WSA error %d)", error_context, wsa_error);
    }
  }

  // Check for cancellation (may race with natural completion).
  if (iree_async_proactor_iocp_operation_is_cancelled(operation)) {
    io_status = iree_status_join(iree_status_from_code(IREE_STATUS_CANCELLED),
                                 io_status);
  }

  return io_status;
}

// Completes a general socket I/O carrier (WSARecv, WSASend, WSASendTo,
// WSARecvFrom). Writes bytes_transferred to the appropriate operation field
// and handles multishot re-arm for recv operations.
static iree_status_t iree_async_proactor_iocp_complete_socket_io(
    iree_async_proactor_iocp_t* proactor, iree_async_iocp_carrier_t* carrier,
    iree_async_operation_t* operation, iree_host_size_t* completed_count) {
  DWORD bytes_transferred = 0;
  DWORD flags = 0;
  iree_status_t io_status = iree_async_proactor_iocp_get_socket_io_result(
      carrier, operation, "socket I/O failed", &bytes_transferred, &flags);

  // Write results to the operation.
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV: {
      iree_async_socket_recv_operation_t* recv_op =
          (iree_async_socket_recv_operation_t*)operation;
      recv_op->bytes_received = bytes_transferred;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SEND: {
      iree_async_socket_send_operation_t* send_op =
          (iree_async_socket_send_operation_t*)operation;
      send_op->bytes_sent = bytes_transferred;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_SENDTO: {
      iree_async_socket_sendto_operation_t* sendto_op =
          (iree_async_socket_sendto_operation_t*)operation;
      sendto_op->bytes_sent = bytes_transferred;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM: {
      iree_async_socket_recvfrom_operation_t* recvfrom_op =
          (iree_async_socket_recvfrom_operation_t*)operation;
      recvfrom_op->bytes_received = bytes_transferred;
      recvfrom_op->sender.length =
          (iree_host_size_t)carrier->data.socket_io.sender_address_length;
      break;
    }
    default:
      break;
  }

  // Multishot re-arm for recv operations.
  if (iree_status_is_ok(io_status) && bytes_transferred > 0 &&
      iree_any_bit_set(operation->flags, IREE_ASYNC_OPERATION_FLAG_MULTISHOT) &&
      (operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV ||
       operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_RECVFROM)) {
    // Dispatch with MORE flag, then re-arm.
    iree_async_proactor_iocp_dispatch_completion(
        proactor, operation, iree_ok_status(), IREE_ASYNC_COMPLETION_FLAG_MORE,
        completed_count);
    // CancelIoEx cannot cancel the already-completed OVERLAPPED when the MORE
    // callback cancels itself, so observe the flag before emulated re-arm.
    if (iree_async_proactor_iocp_operation_is_cancelled(operation)) {
      io_status = iree_status_from_code(IREE_STATUS_CANCELLED);
    } else {
      // Zero OVERLAPPED and re-issue WSARecv/WSARecvFrom.
      memset(&carrier->overlapped, 0, sizeof(carrier->overlapped));
      carrier->data.socket_io.flags = 0;
      int rearm_result = SOCKET_ERROR;
      if (operation->type == IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV) {
        rearm_result =
            WSARecv((SOCKET)carrier->io_handle, carrier->data.socket_io.wsabuf,
                    carrier->data.socket_io.buffer_count, NULL,
                    &carrier->data.socket_io.flags, &carrier->overlapped, NULL);
      } else {
        iree_async_socket_recvfrom_operation_t* rf =
            (iree_async_socket_recvfrom_operation_t*)operation;
        carrier->data.socket_io.sender_address_length =
            (int)sizeof(rf->sender.storage);
        rearm_result = WSARecvFrom(
            (SOCKET)carrier->io_handle, carrier->data.socket_io.wsabuf,
            carrier->data.socket_io.buffer_count, NULL,
            &carrier->data.socket_io.flags,
            (struct sockaddr*)rf->sender.storage,
            &carrier->data.socket_io.sender_address_length,
            &carrier->overlapped, NULL);
      }

      int rearm_error = (rearm_result == SOCKET_ERROR) ? WSAGetLastError() : 0;
      if (rearm_result != SOCKET_ERROR || rearm_error == WSA_IO_PENDING) {
        return iree_async_proactor_iocp_finish_io_issue(carrier);
      }
      io_status = iree_make_status(
          iree_status_code_from_win32_error(rearm_error),
          "multishot recv re-arm failed (WSA error %d)", rearm_error);
    }
  }

  iree_async_proactor_iocp_complete_io_result(proactor, carrier, io_status,
                                              completed_count);
  return iree_ok_status();
}

// Completes a socket accept carrier. Handles SO_UPDATE_ACCEPT_CONTEXT, peer
// address extraction via GetAcceptExSockaddrs, accepted socket creation, and
// multishot re-arm with a new accept socket.
static iree_status_t iree_async_proactor_iocp_complete_accept(
    iree_async_proactor_iocp_t* proactor, iree_async_iocp_carrier_t* carrier,
    iree_async_operation_t* operation, iree_host_size_t* completed_count) {
  DWORD bytes_transferred = 0;
  DWORD flags = 0;
  iree_status_t io_status = iree_async_proactor_iocp_get_socket_io_result(
      carrier, operation, "AcceptEx completion failed", &bytes_transferred,
      &flags);

  SOCKET listen_sock = (SOCKET)carrier->io_handle;
  iree_async_socket_accept_operation_t* accept_op =
      (iree_async_socket_accept_operation_t*)operation;

  if (iree_status_is_ok(io_status)) {
    SOCKET accept_sock = carrier->data.accept.accept_socket;

    // SO_UPDATE_ACCEPT_CONTEXT transfers the listen socket's properties to
    // the accepted socket. Required after AcceptEx for getpeername,
    // getsockname, shutdown, and inherited socket options to work.
    if (setsockopt(accept_sock, SOL_SOCKET, SO_UPDATE_ACCEPT_CONTEXT,
                   (char*)&listen_sock, sizeof(listen_sock)) == SOCKET_ERROR) {
      int wsa_error = WSAGetLastError();
      io_status = iree_make_status(
          iree_status_code_from_win32_error(wsa_error),
          "SO_UPDATE_ACCEPT_CONTEXT failed (WSA error %d)", wsa_error);
      closesocket(accept_sock);
    }

    if (iree_status_is_ok(io_status)) {
      // Extract peer address from AcceptEx output buffer.
      struct sockaddr* local_addr = NULL;
      struct sockaddr* remote_addr = NULL;
      int local_addr_length = 0;
      int remote_addr_length = 0;
      proactor->wsa_extensions.GetAcceptExSockaddrs(
          carrier->data.accept.address_buffer, 0,
          carrier->data.accept.local_address_length,
          carrier->data.accept.remote_address_length, &local_addr,
          &local_addr_length, &remote_addr, &remote_addr_length);

      if (remote_addr && remote_addr_length > 0) {
        memcpy(accept_op->peer_address.storage, remote_addr,
               (size_t)remote_addr_length);
        accept_op->peer_address.length = (iree_host_size_t)remote_addr_length;
      }

      // Create an iree_async_socket_t for the accepted connection.
      iree_status_t create_status = iree_async_iocp_socket_create_accepted(
          proactor, (uintptr_t)accept_sock, accept_op->listen_socket->type,
          accept_op->listen_socket->flags, &accept_op->accepted_socket);
      if (!iree_status_is_ok(create_status)) {
        closesocket(accept_sock);
        io_status = create_status;
      }
    }
  } else {
    // AcceptEx failed: close the pre-created accept socket.
    closesocket(carrier->data.accept.accept_socket);
  }

  // Multishot accept re-arm.
  if (iree_status_is_ok(io_status) &&
      iree_any_bit_set(operation->flags, IREE_ASYNC_OPERATION_FLAG_MULTISHOT)) {
    // Dispatch with MORE flag.
    iree_async_proactor_iocp_dispatch_completion(
        proactor, operation, iree_ok_status(), IREE_ASYNC_COMPLETION_FLAG_MORE,
        completed_count);
    // CancelIoEx cannot cancel the already-completed OVERLAPPED when the MORE
    // callback cancels itself, so observe the flag before emulated re-arm.
    if (iree_async_proactor_iocp_operation_is_cancelled(operation)) {
      io_status = iree_status_from_code(IREE_STATUS_CANCELLED);
    } else {
      // Create a new accept socket for the next accept.
      int domain = accept_op->listen_socket->type == IREE_ASYNC_SOCKET_TYPE_TCP6
                       ? AF_INET6
                       : AF_INET;

      SOCKET new_accept_sock = WSASocketW(domain, SOCK_STREAM, IPPROTO_TCP,
                                          NULL, 0, WSA_FLAG_OVERLAPPED);
      if (new_accept_sock == INVALID_SOCKET) {
        int wsa_error = WSAGetLastError();
        io_status = iree_make_status(
            iree_status_code_from_win32_error(wsa_error),
            "multishot accept re-arm: WSASocketW failed (WSA error %d)",
            wsa_error);
      } else {
        HANDLE assoc_result = CreateIoCompletionPort(
            (HANDLE)new_accept_sock, (HANDLE)proactor->completion_port.handle,
            0, 0);
        if (assoc_result == NULL) {
          DWORD error = GetLastError();
          closesocket(new_accept_sock);
          io_status = iree_make_status(
              iree_status_code_from_win32_error(error),
              "multishot accept re-arm: IOCP association failed (error %lu)",
              (unsigned long)error);
        } else {
          // Zero OVERLAPPED and re-arm AcceptEx.
          memset(&carrier->overlapped, 0, sizeof(carrier->overlapped));
          carrier->data.accept.accept_socket = new_accept_sock;

          DWORD rearm_bytes = 0;
          BOOL rearm_ok = proactor->wsa_extensions.AcceptEx(
              listen_sock, new_accept_sock, carrier->data.accept.address_buffer,
              0, carrier->data.accept.local_address_length,
              carrier->data.accept.remote_address_length, &rearm_bytes,
              &carrier->overlapped);
          int rearm_error = rearm_ok ? 0 : WSAGetLastError();
          if (rearm_ok || rearm_error == WSA_IO_PENDING) {
            return iree_async_proactor_iocp_finish_io_issue(carrier);
          }
          closesocket(new_accept_sock);
          io_status = iree_make_status(
              iree_status_code_from_win32_error(rearm_error),
              "multishot accept re-arm: AcceptEx failed (WSA error %d)",
              rearm_error);
        }
      }
    }
  }

  iree_async_proactor_iocp_complete_io_result(proactor, carrier, io_status,
                                              completed_count);
  return iree_ok_status();
}

// Completes a socket connect carrier. Applies SO_UPDATE_CONNECT_CONTEXT and
// transitions the socket to CONNECTED state.
static void iree_async_proactor_iocp_complete_connect(
    iree_async_proactor_iocp_t* proactor, iree_async_iocp_carrier_t* carrier,
    iree_async_operation_t* operation, iree_host_size_t* completed_count) {
  DWORD bytes_transferred = 0;
  DWORD flags = 0;
  iree_status_t io_status = iree_async_proactor_iocp_get_socket_io_result(
      carrier, operation, "ConnectEx completion failed", &bytes_transferred,
      &flags);

  if (iree_status_is_ok(io_status)) {
    // SO_UPDATE_CONNECT_CONTEXT transfers connection state to the socket.
    // Required after ConnectEx for getpeername, getsockname, shutdown, and
    // TransmitFile to work correctly.
    if (setsockopt((SOCKET)carrier->io_handle, SOL_SOCKET,
                   SO_UPDATE_CONNECT_CONTEXT, NULL, 0) == SOCKET_ERROR) {
      int wsa_error = WSAGetLastError();
      io_status = iree_make_status(
          iree_status_code_from_win32_error(wsa_error),
          "SO_UPDATE_CONNECT_CONTEXT failed (WSA error %d)", wsa_error);
    } else {
      iree_async_socket_connect_operation_t* connect_op =
          (iree_async_socket_connect_operation_t*)operation;
      connect_op->socket->state = IREE_ASYNC_SOCKET_STATE_CONNECTED;
    }
  }

  iree_async_proactor_iocp_complete_io_result(proactor, carrier, io_status,
                                              completed_count);
}

// Completes a recv_pool carrier. Writes bytes_received and handles multishot
// re-arm by acquiring a new buffer from the pool.
static iree_status_t iree_async_proactor_iocp_complete_recv_pool(
    iree_async_proactor_iocp_t* proactor, iree_async_iocp_carrier_t* carrier,
    iree_async_operation_t* operation, iree_host_size_t* completed_count) {
  DWORD bytes_transferred = 0;
  DWORD flags = 0;
  iree_status_t io_status = iree_async_proactor_iocp_get_socket_io_result(
      carrier, operation, "WSARecv (pool) completion failed",
      &bytes_transferred, &flags);

  iree_async_socket_recv_pool_operation_t* recv_pool_op =
      (iree_async_socket_recv_pool_operation_t*)operation;
  recv_pool_op->bytes_received = bytes_transferred;

  // Multishot recv_pool re-arm.
  if (iree_status_is_ok(io_status) && bytes_transferred > 0 &&
      iree_any_bit_set(operation->flags, IREE_ASYNC_OPERATION_FLAG_MULTISHOT)) {
    // Dispatch with MORE flag.
    iree_async_proactor_iocp_dispatch_completion(
        proactor, operation, iree_ok_status(), IREE_ASYNC_COMPLETION_FLAG_MORE,
        completed_count);
    // CancelIoEx cannot cancel the already-completed OVERLAPPED when the MORE
    // callback cancels itself, so observe the flag before emulated re-arm.
    if (iree_async_proactor_iocp_operation_is_cancelled(operation)) {
      io_status = iree_status_from_code(IREE_STATUS_CANCELLED);
    } else {
      // Acquire new buffer from pool for next receive.
      iree_status_t acquire_status = iree_async_buffer_pool_acquire(
          recv_pool_op->pool, &recv_pool_op->lease);
      if (iree_status_is_ok(acquire_status)) {
        carrier->data.recv_pool.wsabuf.buf =
            (char*)iree_async_span_ptr(recv_pool_op->lease.span);
        carrier->data.recv_pool.wsabuf.len =
            (ULONG)recv_pool_op->lease.span.length;
        carrier->data.recv_pool.flags = 0;
        memset(&carrier->overlapped, 0, sizeof(carrier->overlapped));

        int rearm_result = WSARecv(
            (SOCKET)carrier->io_handle, &carrier->data.recv_pool.wsabuf, 1,
            NULL, &carrier->data.recv_pool.flags, &carrier->overlapped, NULL);
        int rearm_error =
            (rearm_result == SOCKET_ERROR) ? WSAGetLastError() : 0;
        if (rearm_result != SOCKET_ERROR || rearm_error == WSA_IO_PENDING) {
          return iree_async_proactor_iocp_finish_io_issue(carrier);
        }
        iree_async_buffer_lease_release(&recv_pool_op->lease);
        io_status = iree_make_status(
            iree_status_code_from_win32_error(rearm_error),
            "multishot recv_pool re-arm failed (WSA error %d)", rearm_error);
      } else {
        io_status = acquire_status;
      }
    }
  }

  iree_async_proactor_iocp_complete_io_result(proactor, carrier, io_status,
                                              completed_count);
  return iree_ok_status();
}

// Completes a file I/O carrier (ReadFile/WriteFile). Writes bytes_read or
// bytes_written to the appropriate operation field.
static void iree_async_proactor_iocp_complete_file_io(
    iree_async_proactor_iocp_t* proactor, iree_async_iocp_carrier_t* carrier,
    iree_async_operation_t* operation, iree_host_size_t* completed_count) {
  DWORD bytes_transferred = 0;
  iree_status_t io_status = iree_ok_status();

  BOOL overlapped_ok =
      GetOverlappedResult((HANDLE)carrier->io_handle, &carrier->overlapped,
                          &bytes_transferred, FALSE);
  if (!overlapped_ok) {
    DWORD error = GetLastError();
    if (error == ERROR_HANDLE_EOF &&
        operation->type == IREE_ASYNC_OPERATION_TYPE_FILE_READ) {
      // Reading at or past EOF: not an error, just 0 bytes transferred.
      // Matches POSIX read() returning 0 at EOF.
      bytes_transferred = 0;
    } else if (error == ERROR_INVALID_HANDLE) {
      // File was closed while this I/O was pending.
      io_status = iree_status_from_code(IREE_STATUS_CANCELLED);
    } else {
      io_status =
          iree_make_status(iree_status_code_from_win32_error(error),
                           "file I/O failed (error %lu)", (unsigned long)error);
    }
  }

  if (iree_any_bit_set(iree_async_operation_load_internal_flags(operation),
                       IREE_ASYNC_IOCP_INTERNAL_FLAG_CANCELLED)) {
    io_status = iree_status_join(iree_status_from_code(IREE_STATUS_CANCELLED),
                                 io_status);
  }

  // Write results to the operation.
  switch (operation->type) {
    case IREE_ASYNC_OPERATION_TYPE_FILE_READ: {
      iree_async_file_read_operation_t* read_op =
          (iree_async_file_read_operation_t*)operation;
      read_op->bytes_read = bytes_transferred;
      break;
    }
    case IREE_ASYNC_OPERATION_TYPE_FILE_WRITE: {
      iree_async_file_write_operation_t* write_op =
          (iree_async_file_write_operation_t*)operation;
      write_op->bytes_written = bytes_transferred;
      break;
    }
    default:
      break;
  }

  iree_async_proactor_iocp_complete_io_result(proactor, carrier, io_status,
                                              completed_count);
}

iree_status_t iree_async_proactor_iocp_complete_io(
    iree_async_proactor_iocp_t* proactor, iree_async_iocp_carrier_t* carrier,
    iree_host_size_t* completed_count) {
  iree_async_operation_t* operation = carrier->operation;
  switch (carrier->type) {
    case IREE_ASYNC_IOCP_CARRIER_SOCKET_IO:
      return iree_async_proactor_iocp_complete_socket_io(
          proactor, carrier, operation, completed_count);
    case IREE_ASYNC_IOCP_CARRIER_ACCEPT:
      return iree_async_proactor_iocp_complete_accept(
          proactor, carrier, operation, completed_count);
    case IREE_ASYNC_IOCP_CARRIER_CONNECT:
      iree_async_proactor_iocp_complete_connect(proactor, carrier, operation,
                                                completed_count);
      break;
    case IREE_ASYNC_IOCP_CARRIER_RECV_POOL:
      return iree_async_proactor_iocp_complete_recv_pool(
          proactor, carrier, operation, completed_count);
    case IREE_ASYNC_IOCP_CARRIER_FILE_IO:
      iree_async_proactor_iocp_complete_file_io(proactor, carrier, operation,
                                                completed_count);
      break;
    default:
      IREE_CHECK_UNREACHABLE("unexpected native I/O carrier type");
  }
  return iree_ok_status();
}

#endif  // IREE_PLATFORM_WINDOWS
