// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Session bootstrap wire format.
//
// Defines the message payloads exchanged during session establishment. These
// are carried as DATA frames on the control channel (frame type 0x80). The
// control channel handles transport-level concerns (PING/PONG liveness,
// GOAWAY shutdown, ERROR reporting) natively; bootstrap messages ride as
// application-defined DATA payloads alongside those.
//
// ## Session lifecycle
//
// The bootstrap protocol establishes a session between two peers. After
// bootstrap, the session provides a control channel for application messages
// and an endpoint provisioning API for opening queue/bulk channels.
//
//   Client                                                Server
//     |                                                      |
//     === TRANSPORT ==========================================
//     |                                                      |
//     |  factory.connect(address)                            |
//     |  --------------------------------------------------->|
//     |  <---------------------- connection ready (accept)   |
//     |                                                      |
//     === BOOTSTRAP ==========================================
//     |                                                      |
//     |  connection.open_endpoint()                          |
//     |  --------------------------------------------------->|
//     |  <------------------------- control endpoint ready   |
//     |                                                      |
//     |  Create control_channel(control_endpoint)            |
//     |  Activate control_channel                            |
//     |                                                      |
//     |  DATA: HELLO                                         |
//     |  { protocol_version, capabilities, machine_index,    |
//     |    session_epoch, axes[]: [{axis, epoch}, ...] }     |
//     |  --------------------------------------------------->|
//     |                                                      |
//     |                [Server validates HELLO]              |
//     |                [Allocates session_id]                |
//     |                [Creates proxy semaphores for         |
//     |                 client's axes, registers in tracker] |
//     |                                                      |
//     |  <------------------------------------- DATA: HELLO_ACK
//     |    { session_id, negotiated_capabilities,            |
//     |      machine_index, session_epoch,                   |
//     |      axes[]: [{axis, epoch}, ...] }                  |
//     |                                                      |
//     |  [Create proxy semaphores for server's axes]         |
//     |  [Register in frontier_tracker with current epochs]  |
//     |                                                      |
//     |  Swap control DATA handler -> application (HAL)      |
//     |                                                      |
//     === OPERATIONAL ========================================
//     |                                                      |
//     |  session.open_endpoint() (for queue traffic)         |
//     |  --------------------------------------------------->|
//     |  <--------------------------- queue endpoint ready   |
//     |                                                      |
//     |  Control DATA: HAL control messages (both dirs)      |
//     |  Queue endpoint: COMMAND/ADVANCE (frontier updates)  |
//     |                                                      |
//     === SHUTDOWN ============================================
//     |                                                      |
//     |  GOAWAY -------------------------------------------->|
//     |  [Drain queue endpoints]                             |
//     |  <------------------------------------------ GOAWAY  |
//     |  [Deactivate all endpoints]                          |
//     |  [frontier_tracker_fail_axis() for remote axes]      |
//     |  [Release proxy semaphores]                          |
//     |  [Release connection]                                |
//
// ## Frontier propagation
//
// Frontier updates flow through queue channels, not the control channel.
// Every GPU completion produces an ADVANCE frame on the queue endpoint that
// carries the new (axis, epoch) values. The HAL queue handler calls
// frontier_tracker_advance() to signal proxy semaphores and wake waiters.
// No dedicated frontier endpoint is needed — queue ADVANCE frames are the
// sole frontier propagation mechanism.
//
// ## Multi-connection sessions (future)
//
// The HELLO_ACK includes a session_id that could be used with JOIN/JOIN_ACK
// messages on additional connections (e.g., separate TCP connections for
// queue and bulk traffic, or RDMA connections alongside TCP control). This
// is not implemented in the initial version — each session uses a single
// connection with multiplexed endpoints.
//
// ## Wire format conventions
//
// All multi-byte fields are little-endian. Structs use natural alignment
// and are padded to 8-byte boundaries for consistent layout. Variable-length
// data (axis arrays, reason strings) follows the fixed-size struct.

#ifndef IREE_NET_BOOTSTRAP_H_
#define IREE_NET_BOOTSTRAP_H_

#include <stddef.h>

#include "iree/base/api.h"

#if !defined(IREE_ENDIANNESS_LITTLE) || !IREE_ENDIANNESS_LITTLE
#error "session bootstrap messages require a little-endian target"
#endif  // !IREE_ENDIANNESS_LITTLE

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Current bootstrap protocol version.
#define IREE_NET_BOOTSTRAP_PROTOCOL_VERSION 2

//===----------------------------------------------------------------------===//
// Bootstrap message types
//===----------------------------------------------------------------------===//

// Discriminator for bootstrap messages carried as control channel DATA frames.
typedef enum iree_net_bootstrap_type_e {
  // Client -> server: new session request with topology.
  IREE_NET_BOOTSTRAP_TYPE_HELLO = 1,
  // Server -> client: session created, server topology included.
  IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK = 2,
  // Server -> client: session rejected (auth failure, resource limit, etc.).
  IREE_NET_BOOTSTRAP_TYPE_REJECT = 3,
} iree_net_bootstrap_type_t;

//===----------------------------------------------------------------------===//
// Capabilities
//===----------------------------------------------------------------------===//

// Feature bits exchanged during handshake. The client sends its supported set
// in HELLO; the server responds with the negotiated intersection in HELLO_ACK.
typedef enum iree_net_bootstrap_capability_bits_e {
  IREE_NET_BOOTSTRAP_CAPABILITY_NONE = 0u,

  // Peer supports bulk transfer endpoints for large buffer moves.
  // When set, the session may open dedicated bulk endpoints for transfers
  // that exceed the queue channel's inline data threshold.
  IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER = 1u << 0,

  // Peer supports RDMA operations (one-sided read/write).
  // When set, bulk transfers may use RDMA instead of message-based copy.
  // Requires the transport to advertise IREE_NET_TRANSPORT_CAPABILITY_RDMA.
  IREE_NET_BOOTSTRAP_CAPABILITY_RDMA = 1u << 1,

  // All capability bits recognized by the current protocol version.
  IREE_NET_BOOTSTRAP_CAPABILITY_ALL_RECOGNIZED =
      IREE_NET_BOOTSTRAP_CAPABILITY_BULK_TRANSFER |
      IREE_NET_BOOTSTRAP_CAPABILITY_RDMA,
} iree_net_bootstrap_capability_bits_t;
typedef uint32_t iree_net_bootstrap_capabilities_t;

//===----------------------------------------------------------------------===//
// Bootstrap message header
//===----------------------------------------------------------------------===//

// Common header for all bootstrap messages (first 8 bytes of DATA payload).
// The control channel delivers DATA payloads with the 8-byte control frame
// header already stripped, so this header starts at offset 0 of the payload.
typedef struct iree_net_bootstrap_header_t {
  // Bootstrap message type (iree_net_bootstrap_type_t).
  uint8_t type;
  // Must be zero.
  uint8_t reserved0[3];
  // Must be zero.
  uint32_t reserved1;
} iree_net_bootstrap_header_t;
static_assert(sizeof(iree_net_bootstrap_header_t) == 8, "");
static_assert(offsetof(iree_net_bootstrap_header_t, type) == 0, "");
static_assert(offsetof(iree_net_bootstrap_header_t, reserved0) == 1, "");
static_assert(offsetof(iree_net_bootstrap_header_t, reserved1) == 4, "");

//===----------------------------------------------------------------------===//
// Axis entry
//===----------------------------------------------------------------------===//

// A single axis and its current epoch, exchanged during topology advertisement.
// The receiver creates a proxy semaphore for each remote axis and registers
// (axis, semaphore) in its frontier_tracker. The current_epoch initializes the
// tracker's epoch counter so the receiver doesn't wait for already-completed
// work.
typedef struct iree_net_bootstrap_axis_entry_t {
  // Full 64-bit axis identifier (see iree/async/frontier.h for encoding).
  uint64_t axis;
  // Current epoch on this axis at the time of the handshake.
  uint64_t current_epoch;
} iree_net_bootstrap_axis_entry_t;
static_assert(sizeof(iree_net_bootstrap_axis_entry_t) == 16, "");
static_assert(offsetof(iree_net_bootstrap_axis_entry_t, axis) == 0, "");
static_assert(offsetof(iree_net_bootstrap_axis_entry_t, current_epoch) == 8,
              "");

//===----------------------------------------------------------------------===//
// HELLO (client -> server)
//===----------------------------------------------------------------------===//

// New session request. Sent by the client as the first DATA frame on the
// control channel after activation.
//
// Followed by |axis_count| iree_net_bootstrap_axis_entry_t entries describing
// the client's local axes (device queues, host contexts, etc.) that the server
// needs to track for causal ordering. Opaque application data follows the axis
// entries and is padded to 8-byte alignment.
typedef struct iree_net_bootstrap_hello_t {
  // Must be IREE_NET_BOOTSTRAP_TYPE_HELLO.
  iree_net_bootstrap_header_t header;

  // Protocol version offered by client. Server rejects if unsupported.
  uint32_t protocol_version;

  // Capabilities supported by this client.
  iree_net_bootstrap_capabilities_t capabilities;

  // Machine index for this client (0-255 within the session).
  // Encoded in axes: iree_async_axis_machine(axis) == machine_index.
  uint8_t machine_index;

  // Session epoch for ABA prevention in axis encoding.
  uint8_t session_epoch;

  // Number of axis entries following this struct.
  uint16_t axis_count;

  // Must be zero.
  uint32_t reserved;

  // Number of application-defined bytes following the axis entries.
  uint64_t application_data_length;
} iree_net_bootstrap_hello_t;
static_assert(sizeof(iree_net_bootstrap_hello_t) == 32, "");
static_assert(offsetof(iree_net_bootstrap_hello_t, header) == 0, "");
static_assert(offsetof(iree_net_bootstrap_hello_t, protocol_version) == 8, "");
static_assert(offsetof(iree_net_bootstrap_hello_t, capabilities) == 12, "");
static_assert(offsetof(iree_net_bootstrap_hello_t, machine_index) == 16, "");
static_assert(offsetof(iree_net_bootstrap_hello_t, session_epoch) == 17, "");
static_assert(offsetof(iree_net_bootstrap_hello_t, axis_count) == 18, "");
static_assert(offsetof(iree_net_bootstrap_hello_t, reserved) == 20, "");
static_assert(offsetof(iree_net_bootstrap_hello_t, application_data_length) ==
                  24,
              "");

//===----------------------------------------------------------------------===//
// HELLO_ACK (server -> client)
//===----------------------------------------------------------------------===//

// Session accepted. Sent by the server in response to a valid HELLO.
//
// Followed by |axis_count| iree_net_bootstrap_axis_entry_t entries describing
// the server's axes that the client needs to track. The client creates proxy
// semaphores for each and registers them in its frontier_tracker. Opaque
// application data follows the axis entries and is padded to 8-byte alignment.
typedef struct iree_net_bootstrap_hello_ack_t {
  // Must be IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK.
  iree_net_bootstrap_header_t header;

  // Server-generated session identifier. Opaque to the client. Would be used
  // with JOIN messages for multi-connection sessions (future).
  uint64_t session_id;

  // Number of application-defined bytes following the axis entries.
  uint64_t application_data_length;

  // Intersection of client and server capabilities. The session operates with
  // only these features enabled.
  iree_net_bootstrap_capabilities_t negotiated_capabilities;

  // Server's machine index (0-255 within the session).
  uint8_t machine_index;

  // Session epoch (echoed from HELLO or server-assigned).
  uint8_t session_epoch;

  // Number of axis entries following this struct.
  uint16_t axis_count;
} iree_net_bootstrap_hello_ack_t;
static_assert(sizeof(iree_net_bootstrap_hello_ack_t) == 32, "");
static_assert(offsetof(iree_net_bootstrap_hello_ack_t, header) == 0, "");
static_assert(offsetof(iree_net_bootstrap_hello_ack_t, session_id) == 8, "");
static_assert(offsetof(iree_net_bootstrap_hello_ack_t,
                       application_data_length) == 16,
              "");
static_assert(offsetof(iree_net_bootstrap_hello_ack_t,
                       negotiated_capabilities) == 24,
              "");
static_assert(offsetof(iree_net_bootstrap_hello_ack_t, machine_index) == 28,
              "");
static_assert(offsetof(iree_net_bootstrap_hello_ack_t, session_epoch) == 29,
              "");
static_assert(offsetof(iree_net_bootstrap_hello_ack_t, axis_count) == 30, "");

//===----------------------------------------------------------------------===//
// REJECT (server -> client)
//===----------------------------------------------------------------------===//

// Session rejected. Sent by the server instead of HELLO_ACK when the session
// cannot be established (invalid topology, resource limits exceeded,
// unsupported protocol version, etc.).
//
// Followed by a UTF-8 reason string (not null-terminated). The string length
// is determined by the DATA payload length minus sizeof(this struct).
typedef struct iree_net_bootstrap_reject_t {
  // Must be IREE_NET_BOOTSTRAP_TYPE_REJECT.
  iree_net_bootstrap_header_t header;

  // Error code indicating the rejection reason (iree_status_code_t value).
  uint32_t reason_code;

  // Must be zero.
  uint32_t reserved;
} iree_net_bootstrap_reject_t;
static_assert(sizeof(iree_net_bootstrap_reject_t) == 16, "");
static_assert(offsetof(iree_net_bootstrap_reject_t, header) == 0, "");
static_assert(offsetof(iree_net_bootstrap_reject_t, reason_code) == 8, "");
static_assert(offsetof(iree_net_bootstrap_reject_t, reserved) == 12, "");

//===----------------------------------------------------------------------===//
// Canonical topology layout
//===----------------------------------------------------------------------===//

// Canonical byte layout of a HELLO or HELLO_ACK payload.
typedef struct iree_net_bootstrap_topology_layout_t {
  // Byte offset of the first encoded axis entry.
  iree_host_size_t axis_entries_offset;
  // Byte offset of the unpadded application data.
  iree_host_size_t application_data_offset;
  // Total payload length including zero alignment padding.
  iree_host_size_t payload_length;
} iree_net_bootstrap_topology_layout_t;

// Calculates the canonical layout of a topology-bearing bootstrap message.
// |message_type| must be HELLO or HELLO_ACK. The output is zeroed on failure.
iree_status_t iree_net_bootstrap_topology_layout_calculate(
    iree_net_bootstrap_type_t message_type, uint32_t axis_count,
    iree_host_size_t application_data_length,
    iree_net_bootstrap_topology_layout_t* out_layout);

//===----------------------------------------------------------------------===//
// Parsed message views
//===----------------------------------------------------------------------===//

// Borrowed encoded axis entries from a structurally validated message.
typedef struct iree_net_bootstrap_axis_list_t {
  // Complete encoded entries in their original wire storage.
  iree_const_byte_span_t encoded_entries;
  // Number of fixed-size entries in |encoded_entries|.
  uint32_t count;
} iree_net_bootstrap_axis_list_t;

// Returns an aligned copy of axis entry |index| using an unaligned-safe load.
// |axis_list| must come from a successfully parsed message and |index| must be
// less than |axis_list->count|.
iree_net_bootstrap_axis_entry_t iree_net_bootstrap_axis_list_get(
    const iree_net_bootstrap_axis_list_t* axis_list, uint32_t index);

// Structurally validated borrowed view of a HELLO message.
typedef struct iree_net_bootstrap_hello_view_t {
  // Aligned copy of the fixed wire prefix.
  iree_net_bootstrap_hello_t fixed;
  // Borrowed encoded axis list.
  iree_net_bootstrap_axis_list_t axes;
  // Borrowed unpadded application bytes.
  iree_const_byte_span_t application_data;
} iree_net_bootstrap_hello_view_t;

// Structurally validated borrowed view of a HELLO_ACK message.
typedef struct iree_net_bootstrap_hello_ack_view_t {
  // Aligned copy of the fixed wire prefix.
  iree_net_bootstrap_hello_ack_t fixed;
  // Borrowed encoded axis list.
  iree_net_bootstrap_axis_list_t axes;
  // Borrowed unpadded application bytes.
  iree_const_byte_span_t application_data;
} iree_net_bootstrap_hello_ack_view_t;

// Structurally validated borrowed view of a REJECT message.
typedef struct iree_net_bootstrap_reject_view_t {
  // Aligned copy of the fixed wire prefix.
  iree_net_bootstrap_reject_t fixed;
  // Borrowed diagnostic reason bytes interpreted as an opaque string.
  iree_string_view_t reason;
} iree_net_bootstrap_reject_view_t;

// Structurally validated borrowed view of one complete bootstrap message.
typedef struct iree_net_bootstrap_message_view_t {
  // Parsed message type selecting the active union member.
  iree_net_bootstrap_type_t type;
  // Message-specific validated view.
  union {
    // HELLO view when |type| is IREE_NET_BOOTSTRAP_TYPE_HELLO.
    iree_net_bootstrap_hello_view_t hello;
    // HELLO_ACK view when |type| is IREE_NET_BOOTSTRAP_TYPE_HELLO_ACK.
    iree_net_bootstrap_hello_ack_view_t hello_ack;
    // REJECT view when |type| is IREE_NET_BOOTSTRAP_TYPE_REJECT.
    iree_net_bootstrap_reject_view_t reject;
  } value;
} iree_net_bootstrap_message_view_t;

// Parses and structurally validates one complete bootstrap DATA payload.
//
// Fixed prefixes are copied into aligned view storage. Axis entries,
// application data, and rejection reasons borrow |payload| and remain valid
// for the same lifetime. Validation covers current-version framing, reserved
// fields, recognized capabilities, exact extents, and zero alignment padding.
// The output is zeroed on failure.
iree_status_t iree_net_bootstrap_message_parse(
    iree_const_byte_span_t payload,
    iree_net_bootstrap_message_view_t* out_message);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_BOOTSTRAP_H_
