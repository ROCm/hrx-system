// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RDMA scatter-gather entry construction from async spans.
//
// Host-posted ibverbs work requests need native ibv_sge entries containing a
// registered virtual address, byte length, and lkey. This component owns the
// validation at the async/RDMA boundary so carrier send, receive, read, and
// write paths all enforce the same representation contract.

#ifndef IREE_NET_CARRIER_RDMA_SGE_H_
#define IREE_NET_CARRIER_RDMA_SGE_H_

#include "iree/async/span.h"
#include "iree/base/api.h"
#include "iree/net/carrier/rdma/libverbs.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Converts one async span to an ibverbs SGE.
IREE_API_EXPORT iree_status_t
iree_net_rdma_sge_from_span(iree_async_span_t span, struct ibv_sge* out_sge);

// Converts a span list to ibverbs SGEs.
//
// |sge_capacity| must be at least |spans.count|. |out_sge_count| receives the
// number of initialized entries as an int suitable for ibv_send_wr.num_sge.
IREE_API_EXPORT iree_status_t iree_net_rdma_sge_list_from_span_list(
    iree_async_span_list_t spans, iree_host_size_t sge_capacity,
    struct ibv_sge* out_sges, int* out_sge_count);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_SGE_H_
