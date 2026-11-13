// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/server/maintenance.h"

#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(MaintenanceTest, DisabledPolicyNeedsNoGroupOrWakeTarget) {
  loom_serve_maintenance_t maintenance;
  IREE_ASSERT_OK(
      loom_serve_maintenance_initialize(nullptr, 0, nullptr, &maintenance));
  bool progress = true;
  IREE_ASSERT_OK(
      loom_serve_maintenance_advance(&maintenance, false, &progress));
  EXPECT_FALSE(progress);
  EXPECT_FALSE(loom_serve_maintenance_is_pending(&maintenance));
  IREE_ASSERT_OK(loom_serve_maintenance_advance(&maintenance, true, &progress));
  EXPECT_FALSE(progress);
  EXPECT_FALSE(loom_serve_maintenance_is_pending(&maintenance));
  loom_serve_maintenance_deinitialize(&maintenance);
}

TEST(MaintenanceTest, InvalidConfigurationAcceptsNoAsynchronousWork) {
  const loom_runtime_pool_idle_policy_t policies[] = {
      {.workspace_timeout = -1},
      {.parameter_timeout = -1},
      {.workspace_timeout = 1},
      {.parameter_timeout = 1},
  };
  for (const auto policy : policies) {
    loom_serve_maintenance_t maintenance;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_maintenance_initialize(nullptr, 1, &policy, &maintenance));
    EXPECT_FALSE(loom_serve_maintenance_is_pending(&maintenance));
    loom_serve_maintenance_deinitialize(&maintenance);
  }
}

TEST(MaintenanceTest, OneTimerUsesTheShortestEnabledPoolPolicy) {
  loom_runtime_host_t* host = nullptr;
  IREE_ASSERT_OK(
      loom_runtime_host_create(1, nullptr, iree_allocator_system(), &host));
  const loom_runtime_device_options_t options = {
      .driver = loom_runtime_host_task_driver(host),
      .uri = IREE_SV("task"),
  };
  loom_runtime_device_group_t* group = nullptr;
  IREE_ASSERT_OK(loom_runtime_device_group_allocate(
      host, 1, &options, iree_allocator_system(), &group));
  const loom_runtime_pool_options_t pool_options = {};
  loom_runtime_pool_t* first_pool = nullptr;
  IREE_ASSERT_OK(loom_runtime_device_group_create_pool(
      group, {IREE_SV("first"), 1}, &pool_options, &first_pool));
  loom_runtime_pool_t* second_pool = nullptr;
  IREE_ASSERT_OK(loom_runtime_device_group_create_pool(
      group, {IREE_SV("second"), 1}, &pool_options, &second_pool));
  EXPECT_EQ(loom_runtime_device_group_count(group), 1u);
  EXPECT_EQ(loom_runtime_device_group_pool_count(group), 2u);
  EXPECT_EQ(loom_runtime_device_group_pool_at(group, 1), second_pool);
  loom_runtime_pool_idle_policy_t policies[2] = {};
  loom_serve_maintenance_t maintenance;
  IREE_ASSERT_OK(
      loom_serve_maintenance_initialize(group, 2, policies, &maintenance));
  EXPECT_EQ(maintenance.timer.interval, 0);
  EXPECT_EQ(maintenance.timer.proactor, nullptr);
  loom_serve_maintenance_deinitialize(&maintenance);

  policies[1].workspace_timeout = iree_make_duration_ms(2500);
  IREE_ASSERT_OK(
      loom_serve_maintenance_initialize(group, 2, policies, &maintenance));
  EXPECT_EQ(maintenance.timer.interval, iree_make_duration_ms(1000));
  EXPECT_EQ(maintenance.policies, policies);
  EXPECT_FALSE(loom_serve_maintenance_is_pending(&maintenance));
  loom_serve_maintenance_deinitialize(&maintenance);

  policies[1].parameter_timeout = iree_make_duration_ms(100);
  IREE_ASSERT_OK(
      loom_serve_maintenance_initialize(group, 2, policies, &maintenance));
  EXPECT_EQ(maintenance.timer.interval, iree_make_duration_ms(100));
  loom_serve_maintenance_deinitialize(&maintenance);

  for (const auto count : {1u, 3u}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_maintenance_initialize(
                              group, count, policies, &maintenance));
    loom_serve_maintenance_deinitialize(&maintenance);
  }
  policies[1].workspace_timeout = -1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_maintenance_initialize(group, 2, policies, &maintenance));
  loom_serve_maintenance_deinitialize(&maintenance);
  IREE_EXPECT_OK(loom_runtime_device_group_free(group));
  loom_runtime_host_release(host);
}

}  // namespace
