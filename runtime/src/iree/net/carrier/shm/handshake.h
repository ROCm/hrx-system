// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cross-process SHM handshake: bootstraps carrier pairs over a connected
// channel (Unix domain socket on POSIX, named pipe on Windows).
//
// The handshake is a three-message blocking exchange. Production factory paths
// run it on a dedicated bootstrap worker and observe completion through the
// proactor; direct callers are responsible for providing a peer concurrently.
//
// After the handshake, each side has everything needed to create an
// iree_net_shm_carrier_t: a mapping of the shared ring buffers, a proxy
// notification for waking the peer, and armed flag pointers.
//
// Platform-specific handle exchange (the only divergent code):
//   POSIX:   fd passing via SCM_RIGHTS over sendmsg/recvmsg on Unix sockets.
//   Windows: DuplicateHandle over ReadFile/WriteFile on named pipes.
//
// The public handshake helpers attach the channel to the returned xproc context
// as a descriptor/HANDLE transfer sideband on success and close it on failure.
// The per-endpoint helpers below leave channel ownership with the caller so one
// bootstrap channel can exchange handles for several endpoint carriers before
// being attached to the control endpoint.

#ifndef IREE_NET_CARRIER_SHM_HANDSHAKE_H_
#define IREE_NET_CARRIER_SHM_HANDSHAKE_H_

#include "iree/async/primitive.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/shm.h"
#include "iree/net/carrier/shm/carrier.h"
#include "iree/net/carrier/shm/shared_wake.h"
#include "iree/net/carrier/shm/xproc_context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Protocol constants
//===----------------------------------------------------------------------===//

#define IREE_NET_SHM_HANDSHAKE_MAGIC 0x49524853u  // "IRHS" (IREE Handshake SHM)
#define IREE_NET_SHM_HANDSHAKE_VERSION 2u

enum iree_net_shm_handshake_message_type_e {
  // Server → Client: offering the SHM region and server's wake handles.
  IREE_NET_SHM_HANDSHAKE_MESSAGE_OFFER = 1u,
  // Client → Server: accepting with client's wake handles.
  IREE_NET_SHM_HANDSHAKE_MESSAGE_ACCEPT = 2u,
  // Server → Client: receiver ownership of all handles is established.
  IREE_NET_SHM_HANDSHAKE_MESSAGE_READY = 3u,
};
typedef uint8_t iree_net_shm_handshake_message_type_t;

// Cooperative cancellation checked around each blocking channel operation.
// A NULL pointer or NULL |requested| field represents no cancellation.
typedef struct iree_net_shm_handshake_cancellation_t {
  // Atomic flag set to nonzero when the handshake must terminate.
  const iree_atomic_int32_t* requested;
  // Waitable primitive signaled when |requested| transitions to nonzero.
  iree_async_primitive_t interrupt_primitive;
} iree_net_shm_handshake_cancellation_t;

// Returns true when |cancellation| requests termination.
static inline bool iree_net_shm_handshake_cancellation_is_requested(
    const iree_net_shm_handshake_cancellation_t* cancellation) {
  return cancellation && cancellation->requested &&
         iree_atomic_load(cancellation->requested, iree_memory_order_acquire) !=
             0;
}

// Fixed-size message header. Sent as the primary payload on the socket.
// Handles are sent alongside (POSIX: SCM_RIGHTS ancillary data; Windows:
// named object strings appended after the header).
typedef struct iree_net_shm_handshake_header_t {
  // Protocol magic identifying SHM bootstrap messages.
  uint32_t magic;
  // Protocol version governing message and ownership semantics.
  uint32_t version;
  // Message payload and attached-handle shape.
  iree_net_shm_handshake_message_type_t type;
  // Reserved for future message flags; must be zero.
  uint8_t reserved[3];
  // OFFER only: total size of the SHM region in bytes.
  uint32_t region_size;
  // OFFER only: MPSC queue data capacity in bytes (power of two).
  uint32_t ring_capacity;
  // Size of the wake epoch SHM region (always one page, but sent for
  // validation).
  uint32_t wake_epoch_size;
  uint8_t padding[8];
} iree_net_shm_handshake_header_t;

//===----------------------------------------------------------------------===//
// Internal platform interface
//===----------------------------------------------------------------------===//
// Send and receive are implemented in handshake_posix.c and handshake_win32.c.

// Handles exchanged alongside a handshake message. The OFFER includes the
// SHM region handle plus wake handles, the ACCEPT includes only wake handles,
// and READY includes no handles.
typedef struct iree_net_shm_handshake_handles_t {
  // Sending process ID for platforms where the receiver needs it to interpret
  // handle values. Zero when the platform does not provide a process ID.
  uint32_t sender_process_id;
  // SHM region handle (OFFER only; invalid for ACCEPT and READY).
  iree_shm_handle_t shm_region;
  // Wake epoch SHM handle (OFFER and ACCEPT; invalid for READY).
  iree_shm_handle_t wake_epoch_shm;
  // Signal primitive (OFFER and ACCEPT; NONE for READY).
  iree_async_primitive_t signal_primitive;
} iree_net_shm_handshake_handles_t;

// Returns an empty handle set that is safe to close.
static inline iree_net_shm_handshake_handles_t
iree_net_shm_handshake_handles_empty(void) {
  iree_net_shm_handshake_handles_t handles;
  handles.sender_process_id = 0;
  handles.shm_region = IREE_SHM_HANDLE_INVALID;
  handles.wake_epoch_shm = IREE_SHM_HANDLE_INVALID;
  handles.signal_primitive = iree_async_primitive_none();
  return handles;
}

// Sends a handshake message with attached handles over the channel.
// Platform-specific: POSIX uses SCM_RIGHTS over sendmsg; Windows uses
// DuplicateHandle over WriteFile.
iree_status_t iree_net_shm_handshake_send(
    iree_async_primitive_t channel,
    const iree_net_shm_handshake_cancellation_t* cancellation,
    const iree_net_shm_handshake_header_t* header,
    const iree_net_shm_handshake_handles_t* handles);

// Receives a handshake message with attached handles from the channel.
// Blocks until data is available, cancellation is requested, or the peer
// disconnects. |out_handles| is empty on failure.
iree_status_t iree_net_shm_handshake_recv(
    iree_async_primitive_t channel,
    const iree_net_shm_handshake_cancellation_t* cancellation,
    iree_net_shm_handshake_header_t* out_header,
    iree_net_shm_handshake_handles_t* out_handles);

// Closes all handles and resets the set to empty.
void iree_net_shm_handshake_handles_close(
    iree_net_shm_handshake_handles_t* handles);

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

// Handshake result: everything needed to create one carrier.
typedef struct iree_net_shm_handshake_result_t {
  // Carrier creation parameters. Ready to pass to
  // iree_net_shm_carrier_create().
  iree_net_shm_carrier_create_params_t carrier_params;
  // Cross-process context (set as carrier's release_context). Owns the SHM
  // mapping, peer notification proxy, and peer signal primitive.
  iree_net_shm_xproc_context_t* context;
  // Region 0 info. Stored here so the create_params.regions pointer (which
  // points to this field) remains valid until the result is consumed.
  iree_net_shm_region_info_t region;
} iree_net_shm_handshake_result_t;

// Releases any resources held by |result| before carrier creation.
//
// Carrier creation transfers |result->context| ownership into the carrier; call
// this only for handshake results that have not been consumed by a carrier.
void iree_net_shm_handshake_result_deinitialize(
    iree_net_shm_handshake_result_t* result);

// Attaches |channel| as the file-transfer sideband for |result|.
//
// On return, |channel| is consumed and set to NONE whether the call succeeds or
// fails. On success, |result|'s xproc context owns the transfer sideband and
// the carrier params advertise it. On failure, |result| is deinitialized.
iree_status_t iree_net_shm_handshake_result_attach_file_transfer(
    iree_async_primitive_t* channel, iree_allocator_t host_allocator,
    iree_net_shm_handshake_result_t* result);

// Server side: performs one endpoint handle exchange over |channel|.
//
// The caller owns |channel| on both success and failure. No file-transfer
// sideband is attached; call
// iree_net_shm_handshake_result_attach_file_transfer() on the control endpoint
// after all endpoint handshakes complete.
iree_status_t iree_net_shm_handshake_server_endpoint(
    iree_async_primitive_t channel,
    const iree_net_shm_handshake_cancellation_t* cancellation,
    iree_net_shm_shared_wake_t* shared_wake,
    iree_net_shm_carrier_options_t options, iree_async_proactor_t* proactor,
    iree_allocator_t host_allocator,
    iree_net_shm_handshake_result_t* out_result);

// Client side: performs one endpoint handle exchange over |channel|.
//
// The caller owns |channel| on both success and failure. No file-transfer
// sideband is attached; call
// iree_net_shm_handshake_result_attach_file_transfer() on the control endpoint
// after all endpoint handshakes complete.
iree_status_t iree_net_shm_handshake_client_endpoint(
    iree_async_primitive_t channel,
    const iree_net_shm_handshake_cancellation_t* cancellation,
    iree_net_shm_shared_wake_t* shared_wake, iree_async_proactor_t* proactor,
    iree_allocator_t host_allocator,
    iree_net_shm_handshake_result_t* out_result);

// Server side: creates a SHM region, exchanges OFFER and ACCEPT, establishes
// all local resources, and sends READY before returning carrier params.
//
// |channel| is a connected channel primitive (Unix domain socket fd on POSIX,
// named pipe HANDLE on Windows). The handshake blocks until its peer completes
// the exchange; the channel is transferred to the returned xproc context on
// success and closed on failure.
//
// |shared_wake| must have been created with
// iree_net_shm_shared_wake_create_shared().
//
// On success, |out_result| contains carrier params ready for
// iree_net_shm_carrier_create(), with the xproc context set as release_context.
IREE_API_EXPORT iree_status_t iree_net_shm_handshake_server(
    iree_async_primitive_t channel, iree_net_shm_shared_wake_t* shared_wake,
    iree_net_shm_carrier_options_t options, iree_async_proactor_t* proactor,
    iree_allocator_t host_allocator,
    iree_net_shm_handshake_result_t* out_result);

// Client side: receives OFFER, maps the SHM region, sends ACCEPT, and waits for
// READY before returning carrier params.
//
// Same semantics as server: blocks until the exchange completes, with the
// channel transferred to the returned xproc context on success and closed on
// failure.
IREE_API_EXPORT iree_status_t iree_net_shm_handshake_client(
    iree_async_primitive_t channel, iree_net_shm_shared_wake_t* shared_wake,
    iree_async_proactor_t* proactor, iree_allocator_t host_allocator,
    iree_net_shm_handshake_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_SHM_HANDSHAKE_H_
