// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/queue_pair.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

iree_net_rdma_queue_pair_options_t ValidOptions() {
  iree_net_rdma_queue_pair_options_t options = {};
  options.send_queue_depth = 8;
  options.recv_queue_depth = 8;
  options.max_send_sge = 1;
  options.max_recv_sge = 1;
  options.max_inline_data = 0;
  options.signal_all_send_work_requests = true;
  return options;
}

TEST(RdmaQueuePairTest, RejectsInvalidOptionsBeforeDereferencingResources) {
  iree_net_rdma_queue_pair_initialize_params_t params = {};
  params.options = ValidOptions();
  iree_net_rdma_queue_pair_t queue_pair;

  params.options.send_queue_depth = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_queue_pair_initialize(params, &queue_pair));

  params.options = ValidOptions();
  params.options.recv_queue_depth = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_queue_pair_initialize(params, &queue_pair));

  params.options = ValidOptions();
  params.options.max_send_sge = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_queue_pair_initialize(params, &queue_pair));

  params.options = ValidOptions();
  params.options.max_recv_sge = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_queue_pair_initialize(params, &queue_pair));
}

TEST(RdmaQueuePairTest, RejectsMissingResources) {
  iree_net_rdma_queue_pair_initialize_params_t params = {};
  params.options = ValidOptions();
  iree_net_rdma_queue_pair_t queue_pair;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_queue_pair_initialize(params, &queue_pair));
}

TEST(RdmaQueuePairTest, NullAccessorsReturnNull) {
  EXPECT_EQ(nullptr, iree_net_rdma_queue_pair_native_qp(nullptr));
  EXPECT_EQ(nullptr, iree_net_rdma_queue_pair_capabilities(nullptr));
}

TEST(RdmaQueuePairTest, DeinitializeAcceptsNullAndZeroedPair) {
  iree_net_rdma_queue_pair_deinitialize(nullptr);

  iree_net_rdma_queue_pair_t queue_pair = {};
  iree_net_rdma_queue_pair_deinitialize(&queue_pair);
}

}  // namespace
