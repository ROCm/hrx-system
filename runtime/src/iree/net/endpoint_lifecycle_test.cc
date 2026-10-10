// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/endpoint_lifecycle.h"

#include <condition_variable>
#include <mutex>
#include <thread>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree {
namespace {

struct CallbackOrder {
  int next = 0;
  int endpoint = -1;
  int connection = -1;
};

struct CallbackDispatchWitness {
  // Serializes the endpoint and connection callback participants.
  std::mutex mutex;
  // Announces endpoint callback entry and permission to return.
  std::condition_variable condition;
  // True after the endpoint callback begins executing.
  bool endpoint_entered = false;
  // True when the endpoint callback may return.
  bool allow_endpoint_return = false;
  // True after the endpoint callback finishes its work.
  bool endpoint_finished = false;
  // True if connection completion overtook endpoint callback completion.
  bool connection_finished_early = false;
  // Number of delivered connection completion callbacks.
  int connection_completion_count = 0;
};

struct ReentrantDrainWitness {
  // Lifecycle rejoined from its own endpoint callback.
  iree_net_endpoint_lifecycle_t* lifecycle = nullptr;
  // Connection barrier committed from the endpoint callback.
  iree_net_endpoint_deactivation_barrier_t* barrier = nullptr;
  // Endpoint callback progress observed by connection completion.
  int endpoint_stage = 0;
  // Endpoint stage observed when connection completion executes.
  int connection_stage = -1;
  // Actions returned by the reentrant connection join.
  iree_net_endpoint_lifecycle_actions_t join_actions =
      IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE;
};

struct DestroyLifecycleWitness {
  // Standalone lifecycle destroyed by its endpoint callback.
  iree_net_endpoint_lifecycle_t* lifecycle = nullptr;
  // True after the endpoint callback destroys the lifecycle.
  bool callback_invoked = false;
};

TEST(EndpointLifecycleTest, ActivationIsExclusiveAndRollbackRestoresCreated) {
  iree_net_endpoint_lifecycle_t lifecycle;
  iree_net_endpoint_lifecycle_initialize(
      /*connection_barrier=*/nullptr, &lifecycle);

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

TEST(EndpointLifecycleTest, StandaloneCallbackMayDestroyLifecycle) {
  DestroyLifecycleWitness witness = {
      .lifecycle = new iree_net_endpoint_lifecycle_t,
  };
  iree_net_endpoint_lifecycle_initialize(
      /*connection_barrier=*/nullptr, witness.lifecycle);
  IREE_ASSERT_OK(iree_net_endpoint_lifecycle_activate(witness.lifecycle));

  iree_net_endpoint_lifecycle_actions_t actions =
      IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE;
  IREE_ASSERT_OK(iree_net_endpoint_lifecycle_request_deactivation(
      witness.lifecycle,
      [](void* user_data) {
        auto* witness = static_cast<DestroyLifecycleWitness*>(user_data);
        iree_net_endpoint_lifecycle_deinitialize(witness->lifecycle);
        delete witness->lifecycle;
        witness->lifecycle = nullptr;
        witness->callback_invoked = true;
      },
      &witness, &actions));

  iree_net_endpoint_lifecycle_complete_deactivation(witness.lifecycle);
  EXPECT_EQ(witness.lifecycle, nullptr);
  EXPECT_TRUE(witness.callback_invoked);
}

TEST(EndpointLifecycleTest, ConnectionJoinsEndpointDeactivation) {
  iree_net_endpoint_deactivation_barrier_t barrier;
  iree_net_endpoint_deactivation_barrier_initialize(&barrier);
  iree_net_endpoint_lifecycle_t lifecycle;
  iree_net_endpoint_lifecycle_initialize(&barrier, &lifecycle);
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

  actions = iree_net_endpoint_lifecycle_join_deactivation(&lifecycle);
  EXPECT_EQ(actions, IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE);

  iree_net_endpoint_deactivation_barrier_commit(
      &barrier, {[](void* user_data) {
                   auto* order = static_cast<CallbackOrder*>(user_data);
                   order->connection = order->next++;
                 },
                 &order});
  EXPECT_EQ(order.endpoint, -1);
  EXPECT_EQ(order.connection, -1);

  iree_net_endpoint_lifecycle_complete_deactivation(&lifecycle);
  EXPECT_EQ(order.endpoint, 0);
  EXPECT_EQ(order.connection, 1);
  iree_net_endpoint_lifecycle_deinitialize(&lifecycle);
}

TEST(EndpointLifecycleTest, OwnerDrainWaitsForAcceptedOperation) {
  iree_net_endpoint_deactivation_barrier_t barrier;
  iree_net_endpoint_deactivation_barrier_initialize(&barrier);
  iree_net_endpoint_lifecycle_t lifecycle;
  iree_net_endpoint_lifecycle_initialize(&barrier, &lifecycle);
  IREE_ASSERT_OK(iree_net_endpoint_lifecycle_activate(&lifecycle));
  EXPECT_TRUE(iree_net_endpoint_lifecycle_try_begin_operation(&lifecycle));

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
  EXPECT_FALSE(iree_net_endpoint_lifecycle_try_begin_operation(&lifecycle));

  actions = iree_net_endpoint_lifecycle_join_deactivation(&lifecycle);
  EXPECT_EQ(actions, IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE);
  iree_net_endpoint_deactivation_barrier_commit(
      &barrier, {[](void* user_data) {
                   auto* order = static_cast<CallbackOrder*>(user_data);
                   order->connection = order->next++;
                 },
                 &order});

  iree_net_endpoint_lifecycle_complete_deactivation(&lifecycle);
  EXPECT_EQ(order.endpoint, -1);
  EXPECT_EQ(order.connection, -1);

  iree_net_endpoint_lifecycle_end_operation(&lifecycle);
  EXPECT_EQ(order.endpoint, 0);
  EXPECT_EQ(order.connection, 1);
  iree_net_endpoint_lifecycle_deinitialize(&lifecycle);
}

TEST(EndpointLifecycleTest, AcceptedOperationWaitsForOwnerDrain) {
  iree_net_endpoint_deactivation_barrier_t barrier;
  iree_net_endpoint_deactivation_barrier_initialize(&barrier);
  iree_net_endpoint_lifecycle_t lifecycle;
  iree_net_endpoint_lifecycle_initialize(&barrier, &lifecycle);
  IREE_ASSERT_OK(iree_net_endpoint_lifecycle_activate(&lifecycle));
  EXPECT_TRUE(iree_net_endpoint_lifecycle_try_begin_operation(&lifecycle));

  int callback_count = 0;
  iree_net_endpoint_lifecycle_actions_t actions =
      iree_net_endpoint_lifecycle_join_deactivation(&lifecycle);
  EXPECT_TRUE(iree_all_bits_set(
      actions, IREE_NET_ENDPOINT_LIFECYCLE_ACTION_BEGIN_DEACTIVATION));
  iree_net_endpoint_deactivation_barrier_commit(
      &barrier, {[](void* user_data) { ++*static_cast<int*>(user_data); },
                 &callback_count});

  iree_net_endpoint_lifecycle_end_operation(&lifecycle);
  EXPECT_EQ(callback_count, 0);
  iree_net_endpoint_lifecycle_complete_deactivation(&lifecycle);
  EXPECT_EQ(callback_count, 1);
  iree_net_endpoint_lifecycle_deinitialize(&lifecycle);
}

TEST(EndpointLifecycleTest, ConnectionJoinsAfterOwnerDrainCompletes) {
  iree_net_endpoint_deactivation_barrier_t barrier;
  iree_net_endpoint_deactivation_barrier_initialize(&barrier);
  iree_net_endpoint_lifecycle_t lifecycle;
  iree_net_endpoint_lifecycle_initialize(&barrier, &lifecycle);
  IREE_ASSERT_OK(iree_net_endpoint_lifecycle_activate(&lifecycle));
  EXPECT_TRUE(iree_net_endpoint_lifecycle_try_begin_operation(&lifecycle));

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
  iree_net_endpoint_lifecycle_complete_deactivation(&lifecycle);
  EXPECT_EQ(order.endpoint, -1);

  actions = iree_net_endpoint_lifecycle_join_deactivation(&lifecycle);
  EXPECT_EQ(actions, IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE);
  iree_net_endpoint_deactivation_barrier_commit(
      &barrier, {[](void* user_data) {
                   auto* order = static_cast<CallbackOrder*>(user_data);
                   order->connection = order->next++;
                 },
                 &order});
  EXPECT_EQ(order.connection, -1);

  iree_net_endpoint_lifecycle_end_operation(&lifecycle);
  EXPECT_EQ(order.endpoint, 0);
  EXPECT_EQ(order.connection, 1);
  iree_net_endpoint_lifecycle_deinitialize(&lifecycle);
}

TEST(EndpointLifecycleTest, LateConnectionJoinWaitsForEndpointCallback) {
  iree_net_endpoint_deactivation_barrier_t barrier;
  iree_net_endpoint_deactivation_barrier_initialize(&barrier);
  iree_net_endpoint_lifecycle_t lifecycle;
  iree_net_endpoint_lifecycle_initialize(&barrier, &lifecycle);
  IREE_ASSERT_OK(iree_net_endpoint_lifecycle_activate(&lifecycle));

  CallbackDispatchWitness witness;
  iree_net_endpoint_lifecycle_actions_t actions =
      IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE;
  IREE_ASSERT_OK(iree_net_endpoint_lifecycle_request_deactivation(
      &lifecycle,
      [](void* user_data) {
        auto* witness = static_cast<CallbackDispatchWitness*>(user_data);
        std::unique_lock<std::mutex> lock(witness->mutex);
        witness->endpoint_entered = true;
        witness->condition.notify_all();
        witness->condition.wait(lock,
                                [&] { return witness->allow_endpoint_return; });
        witness->endpoint_finished = true;
      },
      &witness, &actions));

  std::thread endpoint_thread(
      [&] { iree_net_endpoint_lifecycle_complete_deactivation(&lifecycle); });
  {
    std::unique_lock<std::mutex> lock(witness.mutex);
    witness.condition.wait(lock, [&] { return witness.endpoint_entered; });
  }

  EXPECT_EQ(iree_net_endpoint_lifecycle_join_deactivation(&lifecycle),
            IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE);
  iree_net_endpoint_deactivation_barrier_commit(
      &barrier,
      {[](void* user_data) {
         auto* witness = static_cast<CallbackDispatchWitness*>(user_data);
         std::lock_guard<std::mutex> lock(witness->mutex);
         witness->connection_finished_early = !witness->endpoint_finished;
         ++witness->connection_completion_count;
       },
       &witness});
  {
    std::lock_guard<std::mutex> lock(witness.mutex);
    EXPECT_EQ(witness.connection_completion_count, 0);
    EXPECT_FALSE(witness.connection_finished_early);
    witness.allow_endpoint_return = true;
  }
  witness.condition.notify_all();
  endpoint_thread.join();

  EXPECT_EQ(witness.connection_completion_count, 1);
  EXPECT_FALSE(witness.connection_finished_early);
  iree_net_endpoint_lifecycle_deinitialize(&lifecycle);
}

TEST(EndpointLifecycleTest,
     ConnectionDeactivationInsideEndpointCallbackWaitsForReturn) {
  iree_net_endpoint_deactivation_barrier_t barrier;
  iree_net_endpoint_deactivation_barrier_initialize(&barrier);
  iree_net_endpoint_lifecycle_t lifecycle;
  iree_net_endpoint_lifecycle_initialize(&barrier, &lifecycle);
  IREE_ASSERT_OK(iree_net_endpoint_lifecycle_activate(&lifecycle));

  ReentrantDrainWitness witness = {
      .lifecycle = &lifecycle,
      .barrier = &barrier,
  };
  iree_net_endpoint_lifecycle_actions_t actions =
      IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE;
  IREE_ASSERT_OK(iree_net_endpoint_lifecycle_request_deactivation(
      &lifecycle,
      [](void* user_data) {
        auto* witness = static_cast<ReentrantDrainWitness*>(user_data);
        witness->endpoint_stage = 1;
        witness->join_actions =
            iree_net_endpoint_lifecycle_join_deactivation(witness->lifecycle);
        iree_net_endpoint_deactivation_barrier_commit(
            witness->barrier,
            {[](void* callback_user_data) {
               auto* witness =
                   static_cast<ReentrantDrainWitness*>(callback_user_data);
               witness->connection_stage = witness->endpoint_stage;
             },
             witness});
        witness->endpoint_stage = 2;
      },
      &witness, &actions));

  iree_net_endpoint_lifecycle_complete_deactivation(&lifecycle);
  EXPECT_EQ(witness.join_actions, IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE);
  EXPECT_EQ(witness.endpoint_stage, 2);
  EXPECT_EQ(witness.connection_stage, 2);
  iree_net_endpoint_lifecycle_deinitialize(&lifecycle);
}

TEST(EndpointLifecycleTest, ConnectionStartsEndpointDeactivation) {
  iree_net_endpoint_deactivation_barrier_t barrier;
  iree_net_endpoint_deactivation_barrier_initialize(&barrier);
  iree_net_endpoint_lifecycle_t lifecycle;
  iree_net_endpoint_lifecycle_initialize(&barrier, &lifecycle);
  IREE_ASSERT_OK(iree_net_endpoint_lifecycle_activate(&lifecycle));

  int callback_count = 0;
  iree_net_endpoint_lifecycle_actions_t actions =
      iree_net_endpoint_lifecycle_join_deactivation(&lifecycle);
  EXPECT_TRUE(iree_all_bits_set(
      actions, IREE_NET_ENDPOINT_LIFECYCLE_ACTION_BEGIN_DEACTIVATION));

  iree_net_endpoint_deactivation_barrier_commit(
      &barrier, {[](void* user_data) { ++*static_cast<int*>(user_data); },
                 &callback_count});
  EXPECT_EQ(callback_count, 0);
  iree_net_endpoint_lifecycle_complete_deactivation(&lifecycle);
  EXPECT_EQ(callback_count, 1);
  iree_net_endpoint_lifecycle_deinitialize(&lifecycle);
}

TEST(EndpointLifecycleTest, CreatedEndpointDoesNotDelayConnection) {
  iree_net_endpoint_deactivation_barrier_t barrier;
  iree_net_endpoint_deactivation_barrier_initialize(&barrier);
  iree_net_endpoint_lifecycle_t lifecycle;
  iree_net_endpoint_lifecycle_initialize(&barrier, &lifecycle);

  int callback_count = 0;
  EXPECT_EQ(iree_net_endpoint_lifecycle_join_deactivation(&lifecycle),
            IREE_NET_ENDPOINT_LIFECYCLE_ACTION_NONE);
  iree_net_endpoint_deactivation_barrier_commit(
      &barrier, {[](void* user_data) { ++*static_cast<int*>(user_data); },
                 &callback_count});
  EXPECT_EQ(callback_count, 1);
  iree_net_endpoint_lifecycle_deinitialize(&lifecycle);
}

}  // namespace
}  // namespace iree
