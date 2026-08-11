// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fixed-message boundary for the cross-process SHM carrier handshake.
//
// Platform handle exchange delivers exactly one 32-byte message plus the
// message-specific handles. Receivers validate this untrusted scalar message
// before mapping or dereferencing any transferred handle.

#ifndef IREE_NET_CARRIER_SHM_HANDSHAKE_MESSAGE_H_
#define IREE_NET_CARRIER_SHM_HANDSHAKE_MESSAGE_H_

#include <stddef.h>

#include "iree/base/api.h"

#if !defined(IREE_ENDIANNESS_LITTLE) || !IREE_ENDIANNESS_LITTLE
#error "SHM handshake messages require a little-endian target"
#endif  // !IREE_ENDIANNESS_LITTLE

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_shm_region_layout_t iree_net_shm_region_layout_t;

// Magic number identifying SHM handshake messages ("IRHS" in LE).
#define IREE_NET_SHM_HANDSHAKE_MAGIC ((uint32_t)0x53485249)

// Current fixed-message and handle-ownership protocol version.
#define IREE_NET_SHM_HANDSHAKE_VERSION ((uint32_t)3)

enum iree_net_shm_handshake_message_type_e {
  // Server to client: transport region and server wake handles.
  IREE_NET_SHM_HANDSHAKE_MESSAGE_OFFER = 1u,
  // Client to server: client wake handles.
  IREE_NET_SHM_HANDSHAKE_MESSAGE_ACCEPT = 2u,
  // Server to client: receiver ownership of ACCEPT handles is established.
  IREE_NET_SHM_HANDSHAKE_MESSAGE_READY = 3u,
};
typedef uint8_t iree_net_shm_handshake_message_type_t;

// Fixed-size scalar message sent alongside platform handles.
//
// OFFER populates all geometry fields. ACCEPT populates only
// |wake_epoch_size|. READY leaves all geometry fields zero.
typedef struct iree_net_shm_handshake_header_t {
  // Protocol magic identifying SHM handshake messages.
  uint32_t magic;
  // Protocol version governing message and ownership semantics.
  uint32_t version;
  // OFFER transport mapping extent in bytes; zero for ACCEPT and READY.
  uint64_t transport_region_size;
  // OFFER MPSC queue data capacity in bytes; zero for ACCEPT and READY.
  uint32_t ring_capacity;
  // Creator wake-epoch mapping extent; zero for READY.
  uint32_t wake_epoch_size;
  // Message payload and attached-handle shape.
  iree_net_shm_handshake_message_type_t type;
  // Reserved bytes that must be zero.
  uint8_t reserved[7];
} iree_net_shm_handshake_header_t;
static_assert(sizeof(iree_net_shm_handshake_header_t) == 32, "");
static_assert(offsetof(iree_net_shm_handshake_header_t, magic) == 0, "");
static_assert(offsetof(iree_net_shm_handshake_header_t, version) == 4, "");
static_assert(offsetof(iree_net_shm_handshake_header_t,
                       transport_region_size) == 8,
              "");
static_assert(offsetof(iree_net_shm_handshake_header_t, ring_capacity) == 16,
              "");
static_assert(offsetof(iree_net_shm_handshake_header_t, wake_epoch_size) == 20,
              "");
static_assert(offsetof(iree_net_shm_handshake_header_t, type) == 24, "");
static_assert(offsetof(iree_net_shm_handshake_header_t, reserved) == 25, "");

// Validates an OFFER and returns its canonical local region layout. The output
// is zeroed on failure. Validation includes exact transport extent and creator
// page geometry.
iree_status_t iree_net_shm_handshake_message_validate_offer(
    const iree_net_shm_handshake_header_t* header,
    iree_net_shm_region_layout_t* out_layout);

// Validates an ACCEPT and its creator wake-epoch extent.
iree_status_t iree_net_shm_handshake_message_validate_accept(
    const iree_net_shm_handshake_header_t* header);

// Validates a READY and requires all geometry fields to be zero.
iree_status_t iree_net_shm_handshake_message_validate_ready(
    const iree_net_shm_handshake_header_t* header);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_SHM_HANDSHAKE_MESSAGE_H_
