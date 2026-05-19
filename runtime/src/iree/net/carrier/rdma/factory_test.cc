// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/factory.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

bool ConsumeUnavailableStatus(iree_status_t& status) {
  iree_status_code_t code = iree_status_code(status);
  if (code != IREE_STATUS_NOT_FOUND && code != IREE_STATUS_UNAVAILABLE) {
    return false;
  }
  iree::Status consumed_status = iree::internal::ConsumeForTest(status);
  (void)consumed_status;
  return true;
}

TEST(RdmaFactoryTest, RejectsEndpointCountOutOfRange) {
  iree_net_rdma_factory_options_t options =
      iree_net_rdma_factory_options_default();
  options.max_endpoint_count = UINT16_MAX + 1u;

  iree_net_transport_factory_t* factory = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_net_rdma_factory_create(options, iree_allocator_system(), &factory));
  EXPECT_EQ(nullptr, factory);
}

TEST(RdmaFactoryTest, AcceptsMultipleEndpointCapacity) {
  iree_net_rdma_factory_options_t options =
      iree_net_rdma_factory_options_default();
  options.max_endpoint_count = 2;

  iree_net_transport_factory_t* factory = nullptr;
  iree_status_t status =
      iree_net_rdma_factory_create(options, iree_allocator_system(), &factory);
  if (ConsumeUnavailableStatus(status)) {
    GTEST_SKIP() << "RDMA factory is not available on this machine";
  }
  IREE_ASSERT_OK(status);

  iree_net_transport_factory_release(factory);
}

TEST(RdmaFactoryTest, QueryCapabilities) {
  iree_net_transport_factory_t* factory = nullptr;
  iree_status_t status =
      iree_net_rdma_factory_create(iree_net_rdma_factory_options_default(),
                                   iree_allocator_system(), &factory);
  if (ConsumeUnavailableStatus(status)) {
    GTEST_SKIP() << "RDMA factory is not available on this machine";
  }
  IREE_ASSERT_OK(status);

  iree_net_transport_capabilities_t capabilities =
      iree_net_transport_factory_query_capabilities(factory);
  EXPECT_TRUE(capabilities & IREE_NET_TRANSPORT_CAPABILITY_RELIABLE);
  EXPECT_TRUE(capabilities & IREE_NET_TRANSPORT_CAPABILITY_ORDERED);
  EXPECT_TRUE(capabilities & IREE_NET_TRANSPORT_CAPABILITY_RDMA);
  iree_net_transport_factory_release(factory);
}

}  // namespace
