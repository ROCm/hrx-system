// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/queue_pair.h"

#include <errno.h>
#include <string.h>

static int iree_net_rdma_queue_pair_error_from_result(int result) {
  if (result >= 0) return result;
  if (errno != 0) return errno;
  return result == -1 ? EIO : -result;
}

static iree_status_t iree_net_rdma_queue_pair_status_from_result(
    const char* file, uint32_t line, int result, const char* call) {
  return iree_status_from_errno(
      file, line, iree_net_rdma_queue_pair_error_from_result(result), call);
}

static iree_status_t iree_net_rdma_queue_pair_validate_options(
    iree_net_rdma_queue_pair_options_t options) {
  if (options.send_queue_depth == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send_queue_depth must be non-zero");
  }
  if (options.recv_queue_depth == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "recv_queue_depth must be non-zero");
  }
  if (options.max_send_sge == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "max_send_sge must be non-zero");
  }
  if (options.max_recv_sge == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "max_recv_sge must be non-zero");
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_rdma_queue_pair_initialize(
    iree_net_rdma_queue_pair_initialize_params_t params,
    iree_net_rdma_queue_pair_t* out_queue_pair) {
  if (!out_queue_pair) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_queue_pair must not be NULL");
  }
  memset(out_queue_pair, 0, sizeof(*out_queue_pair));

  iree_status_t status =
      iree_net_rdma_queue_pair_validate_options(params.options);
  if (iree_status_is_ok(status) && !params.context) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "context must not be NULL");
  }
  if (iree_status_is_ok(status) && !params.connection_id) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "connection_id must not be NULL");
  }
  if (iree_status_is_ok(status) && !params.send_completion_queue) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "send_completion_queue must not be NULL");
  }
  if (iree_status_is_ok(status) && !params.recv_completion_queue) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "recv_completion_queue must not be NULL");
  }

  const iree_net_librdmacm_t* librdmacm = NULL;
  struct ibv_qp_init_attr qp_init_attr;
  if (iree_status_is_ok(status)) {
    librdmacm = iree_net_rdma_context_librdmacm(params.context);
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.qp_context = params.qp_context;
    qp_init_attr.send_cq =
        iree_net_rdma_completion_queue_native_cq(params.send_completion_queue);
    qp_init_attr.recv_cq =
        iree_net_rdma_completion_queue_native_cq(params.recv_completion_queue);
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.sq_sig_all = params.options.signal_all_send_work_requests;
    qp_init_attr.cap.max_send_wr = params.options.send_queue_depth;
    qp_init_attr.cap.max_recv_wr = params.options.recv_queue_depth;
    qp_init_attr.cap.max_send_sge = params.options.max_send_sge;
    qp_init_attr.cap.max_recv_sge = params.options.max_recv_sge;
    qp_init_attr.cap.max_inline_data = params.options.max_inline_data;

    if (!qp_init_attr.send_cq || !qp_init_attr.recv_cq) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "completion queues must contain native CQs");
    }
  }

  if (iree_status_is_ok(status)) {
    errno = 0;
    int result = librdmacm->rdma_create_qp(
        params.connection_id,
        iree_net_rdma_context_protection_domain(params.context), &qp_init_attr);
    status = iree_net_rdma_queue_pair_status_from_result(
        __FILE__, __LINE__, result, "rdma_create_qp");
    if (iree_status_is_ok(status) && !params.connection_id->qp) {
      librdmacm->rdma_destroy_qp(params.connection_id);
      status = iree_make_status(IREE_STATUS_INTERNAL,
                                "rdma_create_qp returned no native QP");
    }
  }

  if (iree_status_is_ok(status)) {
    out_queue_pair->librdmacm = librdmacm;
    out_queue_pair->connection_id = params.connection_id;
    out_queue_pair->native_qp = params.connection_id->qp;
    out_queue_pair->capabilities = qp_init_attr.cap;
  }
  return status;
}

IREE_API_EXPORT void iree_net_rdma_queue_pair_deinitialize(
    iree_net_rdma_queue_pair_t* queue_pair) {
  if (!queue_pair) return;

  if (queue_pair->native_qp) {
    queue_pair->librdmacm->rdma_destroy_qp(queue_pair->connection_id);
  }
  memset(queue_pair, 0, sizeof(*queue_pair));
}

IREE_API_EXPORT struct ibv_qp* iree_net_rdma_queue_pair_native_qp(
    const iree_net_rdma_queue_pair_t* queue_pair) {
  return queue_pair ? queue_pair->native_qp : NULL;
}

IREE_API_EXPORT const struct ibv_qp_cap* iree_net_rdma_queue_pair_capabilities(
    const iree_net_rdma_queue_pair_t* queue_pair) {
  return queue_pair ? &queue_pair->capabilities : NULL;
}
