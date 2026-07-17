// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/endpoint_lifecycle.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

struct CallbackOrder {
  int next = 0;
  int endpoint = -1;
  int connection = -1;
};

TEST(EndpointLifecycleTest, ActivationIsExclusiveAndRollbackRestoresCreated) {
  iree_net_endpoint_lifecycle_t lifecycle;
  iree_net_endpoint_lifecycle_initialize(&lifecycle);

  IREE_ASSERT_OK(iree_net_endpoint_lifecycle_activate(&lifecycle));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_net_endpoint_lifecycle_activate(&lifecycle));

  iree_net_endpoint_lifecycle_rollback_activation(&lifecycle);
  IREE_ASSERT_OK(iree_net_endpoint_lifecycle_activate(&lifecycle));

  iree_net_endpoint_lifecycle_actions_t actions =
      IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE;
  IREE_ASSERT_OK(iree_net_endpoint_lifecycle_request_deactivation(
      &lifecycle, /*callback=*/nullptr, /*user_data=*/nullptr, &actions));
  EXPECT_TRUE(iree_all_bits_set(
      actions, IREE_NET_ENDPOINT_LIFECYCLE_ACTION_BEGIN_DEACTIVATION));
  iree_net_endpoint_lifecycle_complete_deactivation(&lifecycle);
  iree_net_endpoint_lifecycle_deinitialize(&lifecycle);
}

TEST(EndpointLifecycleTest, ConnectionJoinsEndpointDeactivation) {
  iree_net_endpoint_lifecycle_t lifecycle;
  iree_net_endpoint_lifecycle_initialize(&lifecycle);
  IREE_ASSERT_OK(iree_net_endpoint_lifecycle_activate(&lifecycle));

  CallbackOrder order;
  iree_net_endpoint_lifecycle_actions_t actions =
      IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE;
  IREE_ASSERT_OK(iree_net_endpoint_lifecycle_request_deactivation(
      &lifecycle,
      [](void* user_data) {
        auto* order = static_cast<CallbackOrder*>(user_data);
        order->endpoint = order->next++;
      },
      &order, &actions));
  EXPECT_TRUE(iree_all_bits_set(
      actions, IREE_NET_ENDPOINT_LIFECYCLE_ACTION_BEGIN_DEACTIVATION));

  iree_net_endpoint_deactivation_barrier_t barrier;
  iree_net_endpoint_deactivation_barrier_initialize(
      {[](void* user_data) {
         auto* order = static_cast<CallbackOrder*>(user_data);
         order->connection = order->next++;
       },
       &order},
      &barrier);
  actions = iree_net_endpoint_lifecycle_join_deactivation(&lifecycle, &barrier);
  EXPECT_EQ(actions, IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE);

  iree_net_endpoint_deactivation_barrier_commit(&barrier);
  EXPECT_EQ(order.endpoint, -1);
  EXPECT_EQ(order.connection, -1);

  iree_net_endpoint_lifecycle_complete_deactivation(&lifecycle);
  EXPECT_EQ(order.endpoint, 0);
  EXPECT_EQ(order.connection, 1);
  iree_net_endpoint_lifecycle_deinitialize(&lifecycle);
}

TEST(EndpointLifecycleTest, ConnectionStartsEndpointDeactivation) {
  iree_net_endpoint_lifecycle_t lifecycle;
  iree_net_endpoint_lifecycle_initialize(&lifecycle);
  IREE_ASSERT_OK(iree_net_endpoint_lifecycle_activate(&lifecycle));

  int callback_count = 0;
  iree_net_endpoint_deactivation_barrier_t barrier;
  iree_net_endpoint_deactivation_barrier_initialize(
      {[](void* user_data) { ++*static_cast<int*>(user_data); },
       &callback_count},
      &barrier);
  iree_net_endpoint_lifecycle_actions_t actions =
      iree_net_endpoint_lifecycle_join_deactivation(&lifecycle, &barrier);
  EXPECT_TRUE(iree_all_bits_set(
      actions, IREE_NET_ENDPOINT_LIFECYCLE_ACTION_BEGIN_DEACTIVATION));

  iree_net_endpoint_deactivation_barrier_commit(&barrier);
  EXPECT_EQ(callback_count, 0);
  iree_net_endpoint_lifecycle_complete_deactivation(&lifecycle);
  EXPECT_EQ(callback_count, 1);
  iree_net_endpoint_lifecycle_deinitialize(&lifecycle);
}

TEST(EndpointLifecycleTest, CreatedEndpointDoesNotDelayConnection) {
  iree_net_endpoint_lifecycle_t lifecycle;
  iree_net_endpoint_lifecycle_initialize(&lifecycle);

  int callback_count = 0;
  iree_net_endpoint_deactivation_barrier_t barrier;
  iree_net_endpoint_deactivation_barrier_initialize(
      {[](void* user_data) { ++*static_cast<int*>(user_data); },
       &callback_count},
      &barrier);
  EXPECT_EQ(iree_net_endpoint_lifecycle_join_deactivation(&lifecycle, &barrier),
            IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE);
  iree_net_endpoint_deactivation_barrier_commit(&barrier);
  EXPECT_EQ(callback_count, 1);
  iree_net_endpoint_lifecycle_deinitialize(&lifecycle);
}

}  // namespace
}  // namespace iree
