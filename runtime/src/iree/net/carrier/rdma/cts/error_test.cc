// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RDMA-specific raw-carrier error propagation tests.

#include <atomic>
#include <cstring>

#include "iree/net/carrier/cts/util/registry.h"
#include "iree/net/carrier/cts/util/test_base.h"

namespace iree::net::carrier::cts {
namespace {

class RdmaErrorTest : public CarrierTestBase<> {};

struct FailingRecvHandlerState {
  std::atomic<int> call_count{0};
};

static iree_status_t FailingRecvHandler(void* user_data, iree_async_span_t data,
                                        iree_async_buffer_lease_t* lease) {
  (void)data;
  auto* state = static_cast<FailingRecvHandlerState*>(user_data);
  state->call_count.fetch_add(1, std::memory_order_release);
  iree_async_buffer_lease_release(lease);
  return iree_make_status(IREE_STATUS_INTERNAL,
                          "test RDMA receive handler failure");
}

TEST_P(RdmaErrorTest, RawCarrierRecvHandlerFailureIsSticky) {
  FailingRecvHandlerState failing_recv_state;
  iree_net_carrier_recv_handler_t failing_recv_handler = {FailingRecvHandler,
                                                          &failing_recv_state};
  ActivateBoth(MakeNullRecvHandler(), failing_recv_handler);

  const char client_payload[] = "trigger RDMA receive handler failure";
  iree_async_span_t client_span;
  iree_net_send_params_t client_params =
      MakeSendParams(client_payload, strlen(client_payload), &client_span);
  IREE_ASSERT_OK(iree_net_carrier_send(client_, &client_params));
  ASSERT_TRUE(PollUntil([&]() {
    return failing_recv_state.call_count.load(std::memory_order_acquire) > 0;
  }));

  iree_net_carrier_send_budget_t server_budget =
      iree_net_carrier_query_send_budget(server_);
  EXPECT_EQ(0u, server_budget.bytes);
  EXPECT_EQ(0u, server_budget.slots);

  const char server_payload[] = "server observes sticky failure";
  iree_async_span_t server_span;
  iree_net_send_params_t server_params =
      MakeSendParams(server_payload, strlen(server_payload), &server_span);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_net_carrier_send(server_, &server_params));
}

CTS_REGISTER_TEST_SUITE(RdmaErrorTest);

}  // namespace
}  // namespace iree::net::carrier::cts
