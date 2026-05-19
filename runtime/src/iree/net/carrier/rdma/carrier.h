// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RDMA carrier: reliable connected queue-pair transport.
//
// The RDMA carrier is the host-initiated data plane for RDMA-capable remoting.
// It owns the verbs resources attached to one rdma_cm Reliable Connection (RC)
// ID and exposes them through the generic iree_net_carrier_t interface:
//
//   - send(): two-sided IBV_WR_SEND for ordered message/control traffic.
//   - direct_write(): RDMA WRITE or RDMA WRITE WITH IMMEDIATE for bulk data.
//   - direct_read(): RDMA READ for pull-based bulk data.
//   - register_buffer(): one-time MR registration for host or dma-buf memory.
//
// The transport factory owns rdma_cm listener/connect state and route
// resolution. The carrier owns the QP, CQs, receive postings, and credit
// memory that must exist before rdma_connect/rdma_accept private-data exchange.
// The factory borrows those carrier-owned handshake values, completes the
// rdma_cm state machine, then activates the carrier data plane. Keeping the
// rdma_cm state machine out of carrier send paths makes the hot path just WR
// posting, CQ draining, and credit accounting.
//
// ## Capability model
//
// RC QPs provide reliable in-order execution within a QP, so the carrier
// reports RELIABLE | ORDERED for message traffic. One-sided operations are
// still completion-correlated by user_data/immediate values: callers must not
// infer cross-QP ordering, and future GPU-initiated QPs are separate data
// planes with their own ordering domains.
//
// ## Backpressure
//
// query_send_budget() reports two-sided SEND admission and must account for
// local SQ capacity plus remote receive credits so callers never intentionally
// drive the QP into RNR. query_direct_write_budget() reports one-sided WRITE
// admission: plain WRITEs consume only local SQ capacity, while WRITE WITH
// IMMEDIATE also consumes remote receive credits. Credit return is carrier
// internal; higher layers should not add their own remote wait/round-trip.

#ifndef IREE_NET_CARRIER_RDMA_CARRIER_H_
#define IREE_NET_CARRIER_RDMA_CARRIER_H_

#include <stdint.h>

#include "iree/async/buffer_pool.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/net/carrier.h"
#include "iree/net/carrier/rdma/connection_data.h"
#include "iree/net/carrier/rdma/context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_rdma_carrier_t iree_net_rdma_carrier_t;

// Default send queue depth for host-initiated RDMA work requests.
#define IREE_NET_RDMA_CARRIER_DEFAULT_SEND_QUEUE_DEPTH 256u

// Default receive queue depth for two-sided messages and immediate signals.
#define IREE_NET_RDMA_CARRIER_DEFAULT_RECV_QUEUE_DEPTH 256u

// Maximum scatter-gather entries supported by the current carrier ABI.
#define IREE_NET_RDMA_CARRIER_MAX_SEND_SGE \
  IREE_NET_RDMA_CONNECTION_DATA_MAX_SEND_SGE

// Default maximum scatter-gather entries per send work request.
#define IREE_NET_RDMA_CARRIER_DEFAULT_MAX_SEND_SGE 16u

// Default maximum scatter-gather entries per receive work request.
#define IREE_NET_RDMA_CARRIER_DEFAULT_MAX_RECV_SGE 1u

// Default inline send threshold requested from the provider.
#define IREE_NET_RDMA_CARRIER_DEFAULT_MAX_INLINE_DATA 64u

// Default per-send staging buffer size for CPU spans that are not RDMA regions.
#define IREE_NET_RDMA_CARRIER_DEFAULT_SEND_STAGING_BUFFER_SIZE 4096u

// Options for RDMA carrier creation.
typedef struct iree_net_rdma_carrier_options_t {
  // Send queue depth. 0 selects the default.
  uint32_t send_queue_depth;

  // Receive queue depth. 0 selects the default.
  uint32_t recv_queue_depth;

  // Maximum send scatter-gather entries. 0 selects the default.
  uint32_t max_send_sge;

  // Maximum receive scatter-gather entries. 0 selects the default.
  uint32_t max_recv_sge;

  // Maximum inline data bytes requested for small sends. 0 selects the default.
  uint32_t max_inline_data;

  // Per-buffer size for registered send staging. 0 selects the default.
  uint32_t send_staging_buffer_size;
} iree_net_rdma_carrier_options_t;

// Returns default RDMA carrier options.
static inline iree_net_rdma_carrier_options_t
iree_net_rdma_carrier_options_default(void) {
  iree_net_rdma_carrier_options_t options;
  memset(&options, 0, sizeof(options));
  options.send_queue_depth = IREE_NET_RDMA_CARRIER_DEFAULT_SEND_QUEUE_DEPTH;
  options.recv_queue_depth = IREE_NET_RDMA_CARRIER_DEFAULT_RECV_QUEUE_DEPTH;
  options.max_send_sge = IREE_NET_RDMA_CARRIER_DEFAULT_MAX_SEND_SGE;
  options.max_recv_sge = IREE_NET_RDMA_CARRIER_DEFAULT_MAX_RECV_SGE;
  options.max_inline_data = IREE_NET_RDMA_CARRIER_DEFAULT_MAX_INLINE_DATA;
  options.send_staging_buffer_size =
      IREE_NET_RDMA_CARRIER_DEFAULT_SEND_STAGING_BUFFER_SIZE;
  return options;
}

// Parameters for creating a carrier around an rdma_cm connection ID.
typedef struct iree_net_rdma_carrier_create_params_t {
  // Shared RDMA context. Retained by the carrier.
  iree_net_rdma_context_t* context;

  // Proactor used to monitor RDMA completion channels. Retained by the carrier.
  iree_async_proactor_t* proactor;

  // Receive buffer pool for two-sided messages. Referenced, not owned; the pool
  // must outlive the carrier.
  iree_async_buffer_pool_t* recv_pool;

  // Resolved rdma_cm connection identifier. The carrier creates and owns the QP
  // attached to this ID before the factory calls rdma_connect/rdma_accept.
  // Ownership transfers to the carrier on successful creation; on failure, the
  // caller retains ownership.
  struct rdma_cm_id* connection_id;

  // Completion callback for carrier operations.
  iree_net_carrier_callback_t callback;

  // Carrier queue and inline-data options.
  iree_net_rdma_carrier_options_t options;
} iree_net_rdma_carrier_create_params_t;

// Creates an RDMA carrier around a resolved rdma_cm connection ID.
//
// The connection ID must already be bound to a local RDMA device and must not
// yet have a QP associated with it. The carrier creates the QP and local
// private-data state; the transport factory is responsible for rdma_cm address
// resolution, route resolution, connect/accept, and exchanging the carrier's
// serialized private data.
IREE_API_EXPORT iree_status_t iree_net_rdma_carrier_create(
    iree_net_rdma_carrier_create_params_t params,
    iree_allocator_t host_allocator, iree_net_carrier_t** out_carrier);

// Casts a generic carrier to an RDMA carrier.
IREE_API_EXPORT iree_net_rdma_carrier_t* iree_net_rdma_carrier_cast(
    iree_net_carrier_t* carrier);

// Returns the generic carrier view of an RDMA carrier.
IREE_API_EXPORT iree_net_carrier_t* iree_net_rdma_carrier_as_generic(
    iree_net_rdma_carrier_t* carrier);

// Returns the rdma_cm ID owned by |carrier| for connect/accept operations.
IREE_API_EXPORT struct rdma_cm_id* iree_net_rdma_carrier_connection_id(
    iree_net_rdma_carrier_t* carrier);

// Exports the carrier's local rdma_cm private-data fields.
IREE_API_EXPORT iree_status_t iree_net_rdma_carrier_export_connection_data(
    iree_net_rdma_carrier_t* carrier,
    iree_net_rdma_connection_data_t* out_data);

// Imports peer rdma_cm private-data fields after connect/accept.
IREE_API_EXPORT iree_status_t iree_net_rdma_carrier_import_connection_data(
    iree_net_rdma_carrier_t* carrier,
    const iree_net_rdma_connection_data_t* data);

// Serializes the carrier's local rdma_cm private-data payload.
IREE_API_EXPORT iree_status_t
iree_net_rdma_carrier_serialize_local_connection_data(
    iree_net_rdma_carrier_t* carrier, iree_byte_span_t target,
    iree_host_size_t* out_length);

// Applies the peer private-data payload after rdma_cm connect/accept.
IREE_API_EXPORT iree_status_t
iree_net_rdma_carrier_apply_remote_connection_data(
    iree_net_rdma_carrier_t* carrier, iree_const_byte_span_t source);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_CARRIER_H_
