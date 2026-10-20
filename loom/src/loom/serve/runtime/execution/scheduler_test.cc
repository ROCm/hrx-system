// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/scheduler.h"

#include <memory>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ExecutionSchedulerTest : public ::testing::Test {
 protected:
  void Initialize(size_t count,
                  const loom_serve_execution_policy_t* policy = nullptr) {
    IREE_ASSERT_OK(loom_serve_execution_scheduler_allocate(
        count, policy, iree_allocator_system(), &scheduler_));
  }

  loom_serve_execution_permit_t* Offer(
      uint64_t resources,
      loom_serve_execution_qos_t qos = LOOM_SERVE_EXECUTION_QOS_BACKGROUND) {
    auto permit = std::make_unique<loom_serve_execution_permit_t>();
    loom_serve_execution_permit_initialize(scheduler_, resources, permit.get());
    loom_serve_execution_permit_request(permit.get(), qos);
    auto* result = permit.get();
    permits_.push_back(std::move(permit));
    return result;
  }

  void TearDown() override {
    for (const auto& permit : permits_) {
      loom_serve_execution_permit_deinitialize(permit.get());
    }
    loom_serve_execution_scheduler_free(scheduler_);
  }

  // Cold resource owner, outliving every accepted test unit.
  loom_serve_execution_scheduler_t* scheduler_ = nullptr;
  // Stable records modelling independent owners and their retirement cuts.
  std::vector<std::unique_ptr<loom_serve_execution_permit_t>> permits_;
};

TEST_F(ExecutionSchedulerTest, DefaultExcludesAcrossDevicesWithoutGlobalCount) {
  ASSERT_NO_FATAL_FAILURE(Initialize(3));
  auto* first = Offer(1);
  auto* second = Offer(1);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(first));
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(second));

  auto* voice = Offer(4, LOOM_SERVE_EXECUTION_QOS_REALTIME);
  auto* background = Offer(2);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(voice));
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(background));
  loom_serve_execution_permit_complete(first);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(voice));
  loom_serve_execution_permit_complete(second);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(voice));

  // Compatible realtime work can overlap; exclusion is not a count of one.
  auto* second_voice = Offer(2, LOOM_SERVE_EXECUTION_QOS_REALTIME);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(second_voice));
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(background));
  loom_serve_execution_permit_complete(voice);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(background));
  loom_serve_execution_permit_complete(second_voice);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(background));
}

TEST_F(ExecutionSchedulerTest, FreshDemandIsCheckedAtAcceptanceAndPlanning) {
  ASSERT_NO_FATAL_FAILURE(Initialize(2));
  uint64_t demand = 0;
  loom_serve_execution_source_t source;
  loom_serve_execution_source_initialize(
      scheduler_, [](void* self) { return *static_cast<uint64_t*>(self); },
      &demand, &source);
  auto* background = Offer(1);
  EXPECT_TRUE(loom_serve_execution_scheduler_is_ready(scheduler_));

  // A new ready input or returned output credit becomes visible before its
  // service runs. Earlier readiness conveys no permission to overtake it.
  demand = 2;
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(background));
  EXPECT_TRUE(loom_serve_execution_is_preempted(
      scheduler_, 1, LOOM_SERVE_EXECUTION_QOS_BACKGROUND));
  EXPECT_FALSE(loom_serve_execution_scheduler_is_ready(scheduler_));

  // An input-starved or output-stalled source withdraws numerical demand.
  demand = 0;
  EXPECT_FALSE(loom_serve_execution_is_preempted(
      scheduler_, 1, LOOM_SERVE_EXECUTION_QOS_BACKGROUND));
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(background));
  loom_serve_execution_source_deinitialize(&source);
}

TEST_F(ExecutionSchedulerTest, ExplicitIndependentScopesDoNotInterfere) {
  const loom_serve_execution_scope_t scopes[] = {
      {1, LOOM_SERVE_EXECUTION_COEXISTENCE_EXCLUDE},
      {6, LOOM_SERVE_EXECUTION_COEXISTENCE_EXCLUDE},
  };
  const loom_serve_execution_policy_t policy = {
      .scope_count = IREE_ARRAYSIZE(scopes),
      .scopes = scopes,
  };
  ASSERT_NO_FATAL_FAILURE(Initialize(3, &policy));
  auto* voice = Offer(2, LOOM_SERVE_EXECUTION_QOS_REALTIME);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(voice));
  auto* independent = Offer(1);
  auto* interfering = Offer(4);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(independent));
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(interfering));
  loom_serve_execution_permit_complete(voice);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(interfering));
}

TEST_F(ExecutionSchedulerTest, ExplicitOverlapStillHonorsIndividualLimits) {
  const uint32_t limits[] = {2, 1};
  const loom_serve_execution_scope_t scope = {
      3, LOOM_SERVE_EXECUTION_COEXISTENCE_OVERLAP};
  const loom_serve_execution_policy_t policy = {
      .max_active = limits,
      .scope_count = 1,
      .scopes = &scope,
  };
  ASSERT_NO_FATAL_FAILURE(Initialize(2, &policy));
  auto* background = Offer(1);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(background));
  auto* voice = Offer(1, LOOM_SERVE_EXECUTION_QOS_REALTIME);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(voice));
  auto* other_device = Offer(2);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(other_device));
  auto* waiting = Offer(1);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(waiting));
  loom_serve_execution_permit_complete(background);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(waiting));
}

TEST_F(ExecutionSchedulerTest, SharedLastSlotGoesToReadyRealtime) {
  const uint32_t limit = 1;
  const loom_serve_execution_scope_t scope = {
      1, LOOM_SERVE_EXECUTION_COEXISTENCE_OVERLAP};
  const loom_serve_execution_policy_t policy = {
      .max_active = &limit,
      .scope_count = 1,
      .scopes = &scope,
  };
  ASSERT_NO_FATAL_FAILURE(Initialize(1, &policy));
  auto* background = Offer(1);
  auto* voice = Offer(1, LOOM_SERVE_EXECUTION_QOS_REALTIME);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(background));
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(voice));
  loom_serve_execution_permit_complete(voice);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(background));
}

TEST_F(ExecutionSchedulerTest, CompoundAcquisitionDoesNotHoldPartialResources) {
  const uint32_t limits[] = {1, 1};
  const loom_serve_execution_scope_t scope = {
      3, LOOM_SERVE_EXECUTION_COEXISTENCE_OVERLAP};
  const loom_serve_execution_policy_t policy = {
      .max_active = limits,
      .scope_count = 1,
      .scopes = &scope,
  };
  ASSERT_NO_FATAL_FAILURE(Initialize(2, &policy));
  auto* occupied = Offer(1);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(occupied));
  auto* compound = Offer(3);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(compound));
  auto* independent = Offer(2, LOOM_SERVE_EXECUTION_QOS_REALTIME);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(independent));
  loom_serve_execution_permit_complete(occupied);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(compound));
  loom_serve_execution_permit_complete(independent);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(compound));
}

TEST_F(ExecutionSchedulerTest, OlderCompoundOfferPreservesOnlyNeededCapacity) {
  const uint32_t limits[] = {1, 2};
  const loom_serve_execution_scope_t scope = {
      3, LOOM_SERVE_EXECUTION_COEXISTENCE_OVERLAP};
  const loom_serve_execution_policy_t policy = {
      .max_active = limits,
      .scope_count = 1,
      .scopes = &scope,
  };
  ASSERT_NO_FATAL_FAILURE(Initialize(2, &policy));
  auto* occupied = Offer(1);
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(occupied));
  auto* compound = Offer(3);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(compound));
  auto* residual = Offer(2);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(residual));
  auto* later = Offer(2);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(later));
  loom_serve_execution_permit_complete(occupied);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(compound));
  loom_serve_execution_permit_complete(compound);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(later));
}

TEST_F(ExecutionSchedulerTest, CancellationNeverRevokesAcceptedClaims) {
  ASSERT_NO_FATAL_FAILURE(Initialize(2));
  auto* voice = Offer(1, LOOM_SERVE_EXECUTION_QOS_REALTIME);
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(voice));
  auto* waiting = Offer(2);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(waiting));
  loom_serve_execution_permit_cancel(voice);
  EXPECT_EQ(voice->phase, LOOM_SERVE_EXECUTION_PERMIT_ACCEPTED);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(waiting));
  loom_serve_execution_permit_cancel(waiting);
  EXPECT_EQ(waiting->phase, LOOM_SERVE_EXECUTION_PERMIT_IDLE);
  loom_serve_execution_permit_request(waiting,
                                      LOOM_SERVE_EXECUTION_QOS_BACKGROUND);
  // Failed readiness and detached service interest leave the model responsible
  // for its actual retirement. Only that owner can discharge the final claim.
  loom_serve_execution_permit_deinitialize(voice);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(waiting));
}

TEST_F(ExecutionSchedulerTest, EveryRepresentableResourceParticipates) {
  ASSERT_NO_FATAL_FAILURE(Initialize(64));
  auto* voice = Offer(UINT64_C(1) << 63, LOOM_SERVE_EXECUTION_QOS_REALTIME);
  auto* background = Offer(1);
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(background));
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(voice));
  loom_serve_execution_permit_complete(voice);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(background));
}

TEST(ExecutionPolicyTest, InvalidCountsAndIncompleteCoverageReject) {
  loom_serve_execution_scheduler_t* scheduler = nullptr;
  for (size_t count : {0, 65}) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        loom_serve_execution_scheduler_allocate(
            count, nullptr, iree_allocator_system(), &scheduler));
    EXPECT_EQ(scheduler, nullptr);
  }
  for (const auto scope : {
           loom_serve_execution_scope_t{
               0, LOOM_SERVE_EXECUTION_COEXISTENCE_EXCLUDE},
           loom_serve_execution_scope_t{
               1, LOOM_SERVE_EXECUTION_COEXISTENCE_EXCLUDE},
           loom_serve_execution_scope_t{
               7, LOOM_SERVE_EXECUTION_COEXISTENCE_EXCLUDE},
           loom_serve_execution_scope_t{
               3, static_cast<loom_serve_execution_coexistence_t>(2)},
       }) {
    const loom_serve_execution_policy_t policy = {
        .scope_count = 1,
        .scopes = &scope,
    };
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_serve_execution_scheduler_allocate(
                              2, &policy, iree_allocator_system(), &scheduler));
    EXPECT_EQ(scheduler, nullptr);
  }
}

}  // namespace
