// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/cts/transfer_trial.h"

#include "iree/net/carrier/loopback/factory.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::net::cts {
namespace {

class TransferTrialFailureTest
    : public ::testing::TestWithParam<iree::async::cts::BackendInfo> {};

TEST_P(TransferTrialFailureTest,
       InsufficientEndpointsDrainsAcceptedConnections) {
  auto transport = GetTransportBackend();
  transport.create_factory = +[](iree_allocator_t host_allocator,
                                 iree_net_transport_factory_t** out_factory) {
    auto options = iree_net_loopback_factory_options_default();
    // One endpoint is valid for the carrier, but the trial requires two.
    options.max_endpoint_count = 1;
    return iree_net_loopback_factory_create(&options, host_allocator,
                                            out_factory);
  };
  TransferTrialOptions options;
  options.connection_count = 16;
  TransferTrialResult result;
  iree_status_t status =
      RunTransferTrial(transport, GetParam().factory, options, {}, &result);
  if (!result.available && iree_status_is_unavailable(status)) {
    const std::string reason = iree::Status::ToString(status);
    iree_status_free(status);
    GTEST_SKIP() << reason;
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, status);
  EXPECT_TRUE(result.available);
  EXPECT_EQ(result.records, 0u);
}

CTS_REGISTER_TEST_SUITE(TransferTrialFailureTest);

}  // namespace
}  // namespace iree::net::cts
