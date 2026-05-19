// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/carrier.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

void NoopCompletion(void* user_data, iree_net_carrier_completion_kind_t kind,
                    uint64_t operation_user_data, iree_status_t status,
                    iree_host_size_t bytes_transferred,
                    iree_async_buffer_lease_t* recv_lease) {
  (void)user_data;
  (void)kind;
  (void)operation_user_data;
  (void)bytes_transferred;
  (void)recv_lease;
  iree::Status consumed_status = iree::internal::ConsumeForTest(status);
  (void)consumed_status;
}

TEST(RdmaCarrierTest, RejectsInvalidArguments) {
  iree_net_carrier_t* carrier = nullptr;
  iree_net_rdma_carrier_create_params_t params = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_carrier_create(params, iree_allocator_system(), nullptr));

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_carrier_create(params, iree_allocator_system(), &carrier));
  EXPECT_EQ(nullptr, carrier);

  params.callback.fn = NoopCompletion;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_carrier_create(params, iree_allocator_system(), &carrier));
  EXPECT_EQ(nullptr, carrier);
}

TEST(RdmaCarrierTest, NullHelpersAreSafe) {
  EXPECT_EQ(nullptr, iree_net_rdma_carrier_cast(nullptr));
  EXPECT_EQ(nullptr, iree_net_rdma_carrier_as_generic(nullptr));
  EXPECT_EQ(nullptr, iree_net_rdma_carrier_connection_id(nullptr));

  iree_host_size_t length = 1;
  char storage[IREE_NET_RDMA_CONNECTION_DATA_LENGTH] = {0};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_carrier_serialize_local_connection_data(
          nullptr, iree_make_byte_span(storage, sizeof(storage)), &length));
  EXPECT_EQ(1u, length);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_rdma_carrier_apply_remote_connection_data(
          nullptr, iree_make_const_byte_span(storage, sizeof(storage))));
}

}  // namespace
