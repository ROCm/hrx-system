// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RDMA logical-connection endpoint bootstrap-data codec.
//
// Multi-endpoint RDMA connections are represented as a group of rdma_cm RC
// connections, one QP/carrier per endpoint slot. Each CM connection carries
// only this compact fixed bootstrap payload so the listener can group endpoint
// QPs into one logical iree_net_connection_t and safely exchange full
// connection data with the first post-establish carrier SEND.

#ifndef IREE_NET_CARRIER_RDMA_ENDPOINT_DATA_H_
#define IREE_NET_CARRIER_RDMA_ENDPOINT_DATA_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/net/carrier/rdma/connection_data.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Magic number identifying RDMA endpoint bootstrap data ("IRDE" in LE).
#define IREE_NET_RDMA_ENDPOINT_DATA_MAGIC ((uint32_t)0x45445249u)

// Current RDMA endpoint bootstrap-data ABI version.
#define IREE_NET_RDMA_ENDPOINT_DATA_VERSION ((uint16_t)3u)

// Fixed bootstrap-data payload length. RoCE CM requests only expose a small
// provider-dependent consumer private-data budget, so this layout intentionally
// stays tiny and leaves extensible negotiation to post-establish carrier SENDs.
#define IREE_NET_RDMA_ENDPOINT_DATA_LENGTH ((iree_host_size_t)32)

// Decoded RDMA endpoint bootstrap parameters exchanged through rdma_cm.
typedef struct iree_net_rdma_endpoint_data_t {
  // Protocol feature flags. Reserved; must be 0 for this version.
  uint16_t flags;

  // Logical connection group identifier shared by all endpoint QPs.
  uint64_t group_id;

  // Zero-based endpoint slot index within the logical connection group.
  uint16_t endpoint_index;

  // Total endpoint slots expected in the logical connection group.
  uint16_t endpoint_count;

  // Peer's per-receive buffer size available for the setup SEND.
  uint32_t bootstrap_recv_buffer_size;

  // Peer's posted receive count available for setup SENDs.
  uint32_t bootstrap_recv_credits;
} iree_net_rdma_endpoint_data_t;

// Serializes RDMA endpoint bootstrap data into an rdma_cm private-data payload.
IREE_API_EXPORT iree_status_t iree_net_rdma_endpoint_data_serialize(
    const iree_net_rdma_endpoint_data_t* data, iree_byte_span_t target,
    iree_host_size_t* out_length);

// Deserializes RDMA endpoint bootstrap data from an rdma_cm private-data
// payload.
IREE_API_EXPORT iree_status_t iree_net_rdma_endpoint_data_deserialize(
    iree_const_byte_span_t source, iree_net_rdma_endpoint_data_t* out_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_ENDPOINT_DATA_H_
