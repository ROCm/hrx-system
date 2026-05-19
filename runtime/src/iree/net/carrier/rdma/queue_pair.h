// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// rdma_cm-owned Reliable Connection queue pair setup.
//
// The RDMA transport factory creates and connects rdma_cm IDs, while the
// carrier posts work requests to the resulting QP. This component owns the
// small but error-prone ibv_qp_init_attr construction and rdma_create_qp /
// rdma_destroy_qp pairing so carriers do not duplicate provider cap plumbing.

#ifndef IREE_NET_CARRIER_RDMA_QUEUE_PAIR_H_
#define IREE_NET_CARRIER_RDMA_QUEUE_PAIR_H_

#include "iree/base/api.h"
#include "iree/net/carrier/rdma/completion_queue.h"
#include "iree/net/carrier/rdma/context.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_net_rdma_queue_pair_t {
  // Borrowed librdmacm symbol table owned by context.
  const iree_net_librdmacm_t* librdmacm;

  // rdma_cm ID with the created QP attached.
  struct rdma_cm_id* connection_id;

  // Native QP created on connection_id.
  struct ibv_qp* native_qp;

  // Provider-adjusted QP capabilities returned by rdma_create_qp.
  struct ibv_qp_cap capabilities;
} iree_net_rdma_queue_pair_t;

typedef struct iree_net_rdma_queue_pair_options_t {
  // Maximum outstanding send work requests.
  uint32_t send_queue_depth;

  // Maximum outstanding receive work requests.
  uint32_t recv_queue_depth;

  // Maximum send scatter-gather entries per work request.
  uint32_t max_send_sge;

  // Maximum receive scatter-gather entries per work request.
  uint32_t max_recv_sge;

  // Maximum inline send data requested from the provider.
  uint32_t max_inline_data;

  // True to request send completions for every posted send work request.
  // When false, callers must set IBV_SEND_SIGNALED on any WR needing a CQE.
  bool signal_all_send_work_requests;
} iree_net_rdma_queue_pair_options_t;

typedef struct iree_net_rdma_queue_pair_initialize_params_t {
  // RDMA context providing the protection domain and rdma_cm symbols.
  iree_net_rdma_context_t* context;

  // rdma_cm ID that has been bound/resolved by the transport factory.
  struct rdma_cm_id* connection_id;

  // Completion queue used for send completions.
  iree_net_rdma_completion_queue_t* send_completion_queue;

  // Completion queue used for receive completions.
  iree_net_rdma_completion_queue_t* recv_completion_queue;

  // Opaque native QP context pointer supplied to ibv_qp_init_attr.
  void* qp_context;

  // Queue-pair sizing and signaling options.
  iree_net_rdma_queue_pair_options_t options;
} iree_net_rdma_queue_pair_initialize_params_t;

// Creates an RC QP on |params.connection_id|.
//
// The rdma_cm ID must already be bound to a local RDMA device. The completion
// queues and context are borrowed and must outlive |out_queue_pair|.
IREE_API_EXPORT iree_status_t iree_net_rdma_queue_pair_initialize(
    iree_net_rdma_queue_pair_initialize_params_t params,
    iree_net_rdma_queue_pair_t* out_queue_pair);

// Destroys the QP attached by initialize.
IREE_API_EXPORT void iree_net_rdma_queue_pair_deinitialize(
    iree_net_rdma_queue_pair_t* queue_pair);

// Returns the native QP.
IREE_API_EXPORT struct ibv_qp* iree_net_rdma_queue_pair_native_qp(
    const iree_net_rdma_queue_pair_t* queue_pair);

// Returns the provider-adjusted QP capabilities.
IREE_API_EXPORT const struct ibv_qp_cap* iree_net_rdma_queue_pair_capabilities(
    const iree_net_rdma_queue_pair_t* queue_pair);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_RDMA_QUEUE_PAIR_H_
