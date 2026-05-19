// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RDMA logical-connection endpoint private-data codec.
//
// Multi-endpoint RDMA connections are represented as a group of rdma_cm RC
// connections, one QP/carrier per endpoint slot. Each CM connection exchanges
// this fixed private-data payload so the listener can group endpoint QPs into
// one logical iree_net_connection_t while preserving the existing per-carrier
// RDMA connection data.

#ifndef IREE_NET_CARRIER_RDMA_ENDPOINT_DATA_H_
#define IREE_NET_CARRIER_RDMA_ENDPOINT_DATA_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/net/carrier/rdma/connection_data.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Magic number identifying RDMA endpoint private data ("IRDE" in LE).
#define IREE_NET_RDMA_ENDPOINT_DATA_MAGIC ((uint32_t)0x45445249u)

// Current RDMA endpoint private-data ABI version.
#define IREE_NET_RDMA_ENDPOINT_DATA_VERSION ((uint16_t)1u)

// Fixed header length before the embedded per-carrier connection data.
#define IREE_NET_RDMA_ENDPOINT_DATA_HEADER_LENGTH ((iree_host_size_t)32)

// Fixed private-data payload length.
#define IREE_NET_RDMA_ENDPOINT_DATA_LENGTH     \
  (IREE_NET_RDMA_ENDPOINT_DATA_HEADER_LENGTH + \
   IREE_NET_RDMA_CONNECTION_DATA_LENGTH)

// Decoded RDMA endpoint parameters exchanged through rdma_cm private data.
typedef struct iree_net_rdma_endpoint_data_t {
  // Protocol feature flags. Reserved; must be 0 for version 1.
  uint16_t flags;

  // Logical connection group identifier shared by all endpoint QPs.
  uint64_t group_id;

  // Zero-based endpoint slot index within the logical connection group.
  uint16_t endpoint_index;

  // Total endpoint slots expected in the logical connection group.
  uint16_t endpoint_count;

  // Per-carrier QP capacities and credit-memory exchange payload.
  iree_net_rdma_connection_data_t connection_data;
} iree_net_rdma_endpoint_data_t;

// Serializes RDMA endpoint data into an rdma_cm private-data payload.
IREE_API_EXPORT iree_status_t iree_net_rdma_endpoint_data_serialize(
    const iree_net_rdma_endpoint_data_t* data, iree_byte_span_t target,
    iree_host_size_t* out_length);

// Deserializes RDMA endpoint data from an rdma_cm private-data payload.
IREE_API_EXPORT iree_status_t iree_net_rdma_endpoint_data_deserialize(
    iree_const_byte_span_t source, iree_net_rdma_endpoint_data_t* out_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_ENDPOINT_DATA_H_
