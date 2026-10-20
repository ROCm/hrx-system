// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/tools/loom-serve/execution_options.h"

#include <deque>
#include <initializer_list>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ExecutionOptionsTest : public ::testing::Test {
 protected:
  void TearDown() override {
    for (auto& permit : permits_) {
      loom_serve_execution_permit_deinitialize(&permit);
    }
    loom_serve_execution_scheduler_free(scheduler_);
    loom_serve_execution_options_deinitialize(&options_);
  }

  iree_status_t Parse(size_t devices,
                      std::initializer_list<const char*> limits = {},
                      std::initializer_list<const char*> scopes = {}) {
    loom_serve_execution_options_deinitialize(&options_);
    std::vector<iree_string_view_t> limit_values, scope_values;
    for (const char* value : limits) {
      limit_values.push_back(iree_make_cstring_view(value));
    }
    for (const char* value : scopes) {
      scope_values.push_back(iree_make_cstring_view(value));
    }
    return loom_serve_execution_options_initialize(
        devices, {limit_values.size(), limit_values.data()},
        {scope_values.size(), scope_values.data()}, iree_allocator_system(),
        &options_);
  }

  void CreateScheduler(size_t devices) {
    IREE_ASSERT_OK(loom_serve_execution_scheduler_allocate(
        devices + 1, &options_.policy, iree_allocator_system(), &scheduler_));
    // Construction must own policy independently of the CLI's temporary data.
    loom_serve_execution_options_deinitialize(&options_);
  }

  loom_serve_execution_permit_t* Offer(uint64_t resources,
                                       loom_serve_execution_qos_t qos) {
    auto& permit = permits_.emplace_back();
    loom_serve_execution_permit_initialize(scheduler_, resources, &permit);
    loom_serve_execution_permit_request(&permit, qos);
    return &permit;
  }

  // Cold parsed storage, consumed only when creating the scheduler.
  loom_serve_execution_options_t options_ = {};
  // Actual production scheduler owning the copied policy.
  loom_serve_execution_scheduler_t* scheduler_ = nullptr;
  // Stable operation records retired before scheduler destruction.
  std::deque<loom_serve_execution_permit_t> permits_;
};

TEST_F(ExecutionOptionsTest, LimitsPreserveDefaultCrossDeviceExclusion) {
  IREE_ASSERT_OK(Parse(2, {"host:8", "1:1"}));
  ASSERT_EQ(options_.policy.scope_count, 1u);
  EXPECT_EQ(options_.policy.scopes[0].resources, 7u);
  EXPECT_EQ(options_.policy.max_active[0], 8u);
  EXPECT_EQ(options_.policy.max_active[1], 0u);
  EXPECT_EQ(options_.policy.max_active[2], 1u);
  ASSERT_NO_FATAL_FAILURE(CreateScheduler(2));
  auto* background = Offer(2, LOOM_SERVE_EXECUTION_QOS_BACKGROUND);
  auto* voice = Offer(4, LOOM_SERVE_EXECUTION_QOS_REALTIME);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(background));
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(voice));
  loom_serve_execution_permit_complete(voice);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(background));
}

TEST_F(ExecutionOptionsTest, IndependentScopesAndLimitsHaveSeparateEffects) {
  IREE_ASSERT_OK(Parse(2, {"1:1"}, {"exclude:host,0", "overlap:1"}));
  ASSERT_NO_FATAL_FAILURE(CreateScheduler(2));
  auto* background = Offer(2, LOOM_SERVE_EXECUTION_QOS_BACKGROUND);
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(background));
  auto* voice = Offer(1, LOOM_SERVE_EXECUTION_QOS_REALTIME);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(voice));
  auto* independent = Offer(4, LOOM_SERVE_EXECUTION_QOS_BACKGROUND);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(independent));
  auto* second = Offer(4, LOOM_SERVE_EXECUTION_QOS_BACKGROUND);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(second));
  loom_serve_execution_permit_complete(background);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(voice));
  loom_serve_execution_permit_complete(independent);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(second));
}

TEST_F(ExecutionOptionsTest, OverlapDoesNotEraseAnIntersectingExclusion) {
  IREE_ASSERT_OK(Parse(2, {}, {"overlap:host,0,1", "exclude:host,1"}));
  ASSERT_NO_FATAL_FAILURE(CreateScheduler(2));
  auto* voice = Offer(4, LOOM_SERVE_EXECUTION_QOS_REALTIME);
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(voice));
  auto* host = Offer(1, LOOM_SERVE_EXECUTION_QOS_BACKGROUND);
  auto* device = Offer(2, LOOM_SERVE_EXECUTION_QOS_BACKGROUND);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(host));
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(device));
}

TEST_F(ExecutionOptionsTest, ExplicitOverlapAdmitsBothClasses) {
  IREE_ASSERT_OK(Parse(1, {"host:0", "0:2"}, {"overlap:host,0"}));
  ASSERT_NO_FATAL_FAILURE(CreateScheduler(1));
  auto* voice = Offer(3, LOOM_SERVE_EXECUTION_QOS_REALTIME);
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(voice));
  auto* background = Offer(3, LOOM_SERVE_EXECUTION_QOS_BACKGROUND);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(background));
}

TEST_F(ExecutionOptionsTest, HighestResourceUsesAllSixtyFourBits) {
  IREE_ASSERT_OK(Parse(63, {"62:4294967295"}));
  EXPECT_EQ(options_.policy.scopes[0].resources, UINT64_MAX);
  EXPECT_EQ(options_.policy.max_active[63], UINT32_MAX);
}

TEST_F(ExecutionOptionsTest, RejectsAmbiguousOrInvalidResourceDeclarations) {
  for (const char* value : {"", "0", "0:", "0:-1", "0:1.5", "0:4294967296",
                            "0:1:2", "1:1", "-1:1", "cpu:1"}) {
    SCOPED_TRACE(value);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(1, {value}));
    EXPECT_EQ(options_.scopes, nullptr);
    EXPECT_EQ(options_.policy.max_active, nullptr);
  }
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Parse(1, {"host:0", "host:1"}));
  for (const char* value : {"", "exclude", "exclude:", "exclusive:0",
                            "overlap:1", "overlap:host,host", "exclude:0,",
                            "exclude:host,,0", "exclude:host:0"}) {
    SCOPED_TRACE(value);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(1, {}, {value}));
    EXPECT_EQ(options_.scopes, nullptr);
    EXPECT_EQ(options_.policy.max_active, nullptr);
  }
  for (size_t devices : {0, 64}) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Parse(devices));
    EXPECT_EQ(options_.scopes, nullptr);
  }
}

TEST_F(ExecutionOptionsTest, SchedulerRejectsIncompleteExplicitCoverage) {
  IREE_ASSERT_OK(Parse(2, {}, {"exclude:host,0"}));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_serve_execution_scheduler_allocate(
          3, &options_.policy, iree_allocator_system(), &scheduler_));
  EXPECT_EQ(scheduler_, nullptr);
}

}  // namespace
