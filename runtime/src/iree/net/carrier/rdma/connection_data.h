// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RDMA connection private-data codec.
//
// rdma_cm allows peers to exchange a small private-data payload during
// connect/accept. The RDMA carrier uses that payload to exchange queue
// capacities, posted receive buffer size, and the remote memory location used
// for credit returns. This file owns the wire layout so the connection state
// machine and carrier do not reinterpret ad-hoc structs.

#ifndef IREE_NET_CARRIER_RDMA_CONNECTION_DATA_H_
#define IREE_NET_CARRIER_RDMA_CONNECTION_DATA_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Magic number identifying RDMA connection private data ("IRDM" in LE).
#define IREE_NET_RDMA_CONNECTION_DATA_MAGIC ((uint32_t)0x4D445249u)

// Current RDMA connection private-data ABI version.
#define IREE_NET_RDMA_CONNECTION_DATA_VERSION ((uint16_t)2u)

// Fixed private-data payload length.
#define IREE_NET_RDMA_CONNECTION_DATA_LENGTH ((iree_host_size_t)56)

// Maximum send scatter-gather entries supported by the connection-data ABI.
#define IREE_NET_RDMA_CONNECTION_DATA_MAX_SEND_SGE 16u

// Remote memory used by the peer to return credits without a round trip.
typedef struct iree_net_rdma_remote_credit_memory_t {
  // Remote virtual address of the credit counter memory.
  uint64_t address;

  // Remote key authorizing writes to address.
  uint32_t rkey;

  // Byte length of the remote credit memory.
  uint32_t length;
} iree_net_rdma_remote_credit_memory_t;

// Decoded RDMA connection parameters exchanged through rdma_cm private data.
typedef struct iree_net_rdma_connection_data_t {
  // Protocol feature flags. Reserved; must be 0 for version 1.
  uint16_t flags;

  // Peer's send queue depth.
  uint32_t send_queue_depth;

  // Peer's receive queue depth.
  uint32_t recv_queue_depth;

  // Peer's per-receive buffer size for two-sided SEND messages.
  uint32_t recv_buffer_size;

  // Peer's maximum send scatter-gather entry count.
  uint32_t max_send_sge;

  // Peer's maximum receive scatter-gather entry count.
  uint32_t max_recv_sge;

  // Peer's maximum inline data size.
  uint32_t max_inline_data;

  // Initial receive credits available to the remote sender.
  uint32_t initial_recv_credits;

  // Remote memory the peer exposes for credit returns.
  iree_net_rdma_remote_credit_memory_t credit_memory;
} iree_net_rdma_connection_data_t;

// Validates decoded RDMA connection data before applying it to a carrier.
IREE_API_EXPORT iree_status_t iree_net_rdma_connection_data_validate(
    const iree_net_rdma_connection_data_t* data);

// Serializes RDMA connection data into an rdma_cm private-data payload.
IREE_API_EXPORT iree_status_t iree_net_rdma_connection_data_serialize(
    const iree_net_rdma_connection_data_t* data, iree_byte_span_t target,
    iree_host_size_t* out_length);

// Deserializes RDMA connection data from an rdma_cm private-data payload.
IREE_API_EXPORT iree_status_t iree_net_rdma_connection_data_deserialize(
    iree_const_byte_span_t source, iree_net_rdma_connection_data_t* out_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_CONNECTION_DATA_H_
