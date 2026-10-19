// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/scheduler.h"

#include "iree/testing/gtest.h"

namespace {

TEST(ExecutionSchedulerTest, ConcurrentDoesNotWaitForOtherExecutions) {
  loom_serve_execution_scheduler_t scheduler;
  loom_serve_execution_scheduler_initialize(0, &scheduler);
  loom_serve_execution_permit_t first, second;
  loom_serve_execution_permit_initialize(&scheduler, &first);
  loom_serve_execution_permit_initialize(&scheduler, &second);
  loom_serve_execution_permit_request(&first, 1);
  loom_serve_execution_permit_request(&second, 8);
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(&first, 100));
  EXPECT_TRUE(loom_serve_execution_permit_try_acquire(&second, 101));
  EXPECT_FALSE(loom_serve_execution_scheduler_advance(&scheduler, 101));
  loom_serve_execution_permit_complete(&second, 110);
  loom_serve_execution_permit_complete(&first, 150);
  loom_serve_execution_permit_deinitialize(&first);
  loom_serve_execution_permit_deinitialize(&second);
}

TEST(ExecutionSchedulerTest, WeightedServiceAccountsForDifferentUnitLengths) {
  loom_serve_execution_scheduler_t scheduler;
  loom_serve_execution_scheduler_initialize(1, &scheduler);
  loom_serve_execution_permit_t voice, background;
  loom_serve_execution_permit_initialize(&scheduler, &voice);
  loom_serve_execution_permit_initialize(&scheduler, &background);
  loom_serve_execution_permit_request(&voice, 4);
  loom_serve_execution_permit_request(&background, 1);
  iree_time_t now = 0;
  uint64_t voice_time = 0, background_time = 0;
  for (int i = 0; i < 22; ++i) {
    ASSERT_TRUE(loom_serve_execution_scheduler_advance(&scheduler, now));
    const bool selected_voice =
        loom_serve_execution_permit_try_acquire(&voice, now);
    loom_serve_execution_permit_t* selected =
        selected_voice ? &voice : &background;
    if (!selected_voice) {
      ASSERT_TRUE(loom_serve_execution_permit_try_acquire(selected, now));
    }
    EXPECT_FALSE(loom_serve_execution_scheduler_advance(&scheduler, now));
    const iree_duration_t elapsed = selected_voice ? 2000000 : 5000000;
    (selected_voice ? voice_time : background_time) += elapsed;
    now += elapsed;
    loom_serve_execution_permit_complete(selected, now);
    loom_serve_execution_permit_request(selected, selected_voice ? 4 : 1);
  }
  EXPECT_EQ(voice_time, 40000000u);
  EXPECT_EQ(background_time, 10000000u);
  loom_serve_execution_permit_cancel(&voice);
  loom_serve_execution_permit_cancel(&background);
  EXPECT_FALSE(loom_serve_execution_scheduler_advance(&scheduler, now));
}

TEST(ExecutionSchedulerTest, CancellationWithdrawsWaitingAndUntakenGrants) {
  loom_serve_execution_scheduler_t scheduler;
  loom_serve_execution_scheduler_initialize(1, &scheduler);
  loom_serve_execution_permit_t first, second, third;
  loom_serve_execution_permit_initialize(&scheduler, &first);
  loom_serve_execution_permit_initialize(&scheduler, &second);
  loom_serve_execution_permit_initialize(&scheduler, &third);
  loom_serve_execution_permit_request(&first, 1);
  loom_serve_execution_permit_request(&second, 1);
  loom_serve_execution_permit_request(&third, 1);
  ASSERT_TRUE(loom_serve_execution_scheduler_advance(&scheduler, 0));
  loom_serve_execution_permit_cancel(&second);
  loom_serve_execution_permit_cancel(&first);
  ASSERT_TRUE(loom_serve_execution_scheduler_advance(&scheduler, 100));
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&third, 100));
  loom_serve_execution_permit_complete(&third, 110);
  EXPECT_FALSE(loom_serve_execution_scheduler_advance(&scheduler, 110));
  EXPECT_EQ(scheduler.head, nullptr);
  EXPECT_EQ(scheduler.tail, nullptr);
  EXPECT_EQ(scheduler.outstanding_count, 0u);
}

TEST(ExecutionSchedulerTest, LateArrivalDoesNotBankUnusedService) {
  loom_serve_execution_scheduler_t scheduler;
  loom_serve_execution_scheduler_initialize(1, &scheduler);
  loom_serve_execution_permit_t running, late;
  loom_serve_execution_permit_initialize(&scheduler, &running);
  loom_serve_execution_permit_initialize(&scheduler, &late);
  loom_serve_execution_permit_request(&running, 1);
  ASSERT_TRUE(loom_serve_execution_scheduler_advance(&scheduler, 100));
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&running, 100));
  loom_serve_execution_permit_complete(&running, 1100);
  loom_serve_execution_permit_request(&running, 1);
  loom_serve_execution_permit_request(&late, 1);
  ASSERT_TRUE(loom_serve_execution_scheduler_advance(&scheduler, 1100));
  EXPECT_FALSE(loom_serve_execution_permit_try_acquire(&late, 1100));
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&running, 1100));
  loom_serve_execution_permit_complete(&running, 1200);
  ASSERT_TRUE(loom_serve_execution_scheduler_advance(&scheduler, 1200));
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&late, 1200));
  loom_serve_execution_permit_complete(&late, 1300);
  EXPECT_FALSE(loom_serve_execution_scheduler_advance(&scheduler, 1300));
}

TEST(ExecutionSchedulerTest, PreparedWorkJoinsAnAcceptedPeer) {
  loom_serve_execution_scheduler_t scheduler;
  loom_serve_execution_scheduler_initialize(2, &scheduler);
  loom_serve_execution_permit_t text, speech, image;
  loom_serve_execution_permit_initialize(&scheduler, &text);
  loom_serve_execution_permit_initialize(&scheduler, &speech);
  loom_serve_execution_permit_initialize(&scheduler, &image);
  loom_serve_execution_permit_request(&text, 8);
  ASSERT_TRUE(loom_serve_execution_scheduler_advance(&scheduler, 0));
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&text, 0));

  // CPU preparation finishes while text still owns its numerical work.
  loom_serve_execution_permit_request(&speech, 8);
  ASSERT_TRUE(loom_serve_execution_scheduler_advance(&scheduler, 10));
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&speech, 10));
  loom_serve_execution_permit_request(&image, 1);
  EXPECT_FALSE(loom_serve_execution_scheduler_advance(&scheduler, 10));
  EXPECT_EQ(scheduler.outstanding_count, 2u);

  // Speech retirement frees its own slot without joining the text borrower.
  loom_serve_execution_permit_complete(&speech, 25);
  ASSERT_TRUE(loom_serve_execution_scheduler_advance(&scheduler, 25));
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&image, 25));
  loom_serve_execution_permit_request(&speech, 8);
  EXPECT_FALSE(loom_serve_execution_scheduler_advance(&scheduler, 25));
  loom_serve_execution_permit_complete(&image, 35);
  ASSERT_TRUE(loom_serve_execution_scheduler_advance(&scheduler, 35));
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&speech, 35));
  EXPECT_EQ(text.phase, LOOM_SERVE_EXECUTION_PERMIT_ACCEPTED);
  loom_serve_execution_permit_complete(&speech, 50);

  // Withdrawing an untaken grant cannot revoke the accepted peer's slot.
  loom_serve_execution_permit_request(&image, 1);
  ASSERT_TRUE(loom_serve_execution_scheduler_advance(&scheduler, 80));
  loom_serve_execution_permit_cancel(&image);
  EXPECT_EQ(scheduler.outstanding_count, 1u);
  EXPECT_EQ(text.phase, LOOM_SERVE_EXECUTION_PERMIT_ACCEPTED);
  loom_serve_execution_permit_complete(&text, 125);
  EXPECT_EQ(scheduler.outstanding_count, 0u);
  EXPECT_EQ(scheduler.head, nullptr);
  EXPECT_EQ(scheduler.tail, nullptr);
}

TEST(ExecutionSchedulerTest, OverlapPreservesBackgroundService) {
  loom_serve_execution_scheduler_t scheduler;
  loom_serve_execution_scheduler_initialize(2, &scheduler);
  loom_serve_execution_permit_t permits[3];
  const uint32_t weights[] = {8, 8, 1};
  const iree_duration_t durations[] = {125000000, 15000000, 8000000};
  iree_time_t completions[3] = {0};
  uint32_t completed_counts[3] = {0};
  for (size_t i = 0; i < IREE_ARRAYSIZE(permits); ++i) {
    loom_serve_execution_permit_initialize(&scheduler, &permits[i]);
    loom_serve_execution_permit_request(&permits[i], weights[i]);
  }
  iree_time_t now = 0;
  for (size_t iteration = 0; iteration < 64; ++iteration) {
    while (loom_serve_execution_scheduler_advance(&scheduler, now)) {
      for (size_t i = 0; i < IREE_ARRAYSIZE(permits); ++i) {
        if (loom_serve_execution_permit_try_acquire(&permits[i], now)) {
          completions[i] = now + durations[i];
        }
      }
    }
    EXPECT_GE(scheduler.outstanding_count, 1u);
    EXPECT_LE(scheduler.outstanding_count, 2u);
    size_t next = IREE_ARRAYSIZE(permits);
    for (size_t i = 0; i < IREE_ARRAYSIZE(permits); ++i) {
      if (permits[i].phase == LOOM_SERVE_EXECUTION_PERMIT_ACCEPTED &&
          (next == IREE_ARRAYSIZE(permits) ||
           completions[i] < completions[next])) {
        next = i;
      }
    }
    ASSERT_NE(next, IREE_ARRAYSIZE(permits));
    now = completions[next];
    loom_serve_execution_permit_complete(&permits[next], now);
    ++completed_counts[next];
    loom_serve_execution_permit_request(&permits[next], weights[next]);
  }
  uint64_t minimum_service = UINT64_MAX;
  uint64_t maximum_service = 0;
  uint64_t maximum_charge = 0;
  for (size_t i = 0; i < IREE_ARRAYSIZE(permits); ++i) {
    EXPECT_GT(completed_counts[i], 0u);
    const uint64_t charge = (uint64_t)durations[i] / weights[i];
    const uint64_t service = completed_counts[i] * charge;
    minimum_service = iree_min(minimum_service, service);
    maximum_service = iree_max(maximum_service, service);
    maximum_charge = iree_max(maximum_charge, charge);
    if (permits[i].phase == LOOM_SERVE_EXECUTION_PERMIT_ACCEPTED) {
      loom_serve_execution_permit_complete(&permits[i], completions[i]);
    } else {
      loom_serve_execution_permit_cancel(&permits[i]);
    }
  }
  // With continuously ready producers, completed weighted service differs by
  // no more than one longest indivisible unit, despite different unit lengths.
  EXPECT_LE(maximum_service - minimum_service, maximum_charge);
  EXPECT_EQ(scheduler.outstanding_count, 0u);
  EXPECT_EQ(scheduler.head, nullptr);
  EXPECT_EQ(scheduler.tail, nullptr);
}

TEST(ExecutionSchedulerTest, SpareCapacityPreservesAcceptedServiceShare) {
  loom_serve_execution_scheduler_t scheduler;
  loom_serve_execution_scheduler_initialize(2, &scheduler);
  loom_serve_execution_permit_t foreground, background;
  loom_serve_execution_permit_initialize(&scheduler, &foreground);
  loom_serve_execution_permit_initialize(&scheduler, &background);
  loom_serve_execution_permit_request(&foreground, 8);
  ASSERT_TRUE(loom_serve_execution_scheduler_advance(&scheduler, 0));
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&foreground, 0));
  loom_serve_execution_permit_request(&background, 1);
  ASSERT_TRUE(loom_serve_execution_scheduler_advance(&scheduler, 0));
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&background, 0));

  loom_serve_execution_permit_complete(&background, 8000000);
  loom_serve_execution_permit_request(&background, 1);
  EXPECT_FALSE(loom_serve_execution_scheduler_advance(&scheduler, 8000000));
  EXPECT_EQ(scheduler.outstanding_count, 1u);
  EXPECT_EQ(background.phase, LOOM_SERVE_EXECUTION_PERMIT_WAITING);

  // A later owner wake observes equal weighted service. Background may join
  // without waiting for foreground completion; no timer belongs to the grant.
  ASSERT_TRUE(loom_serve_execution_scheduler_advance(&scheduler, 64000000));
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&background, 64000000));
  loom_serve_execution_permit_complete(&background, 72000000);
  EXPECT_EQ(foreground.phase, LOOM_SERVE_EXECUTION_PERMIT_ACCEPTED);
  loom_serve_execution_permit_complete(&foreground, 125000000);
  EXPECT_EQ(scheduler.outstanding_count, 0u);
  EXPECT_FALSE(loom_serve_execution_scheduler_advance(&scheduler, 125000000));
}

TEST(ExecutionSchedulerTest, IndependentDomainsAndFailureDetachment) {
  loom_serve_execution_scheduler_t first_domain, second_domain;
  loom_serve_execution_scheduler_initialize(1, &first_domain);
  loom_serve_execution_scheduler_initialize(1, &second_domain);
  loom_serve_execution_permit_t first, second;
  loom_serve_execution_permit_initialize(&first_domain, &first);
  loom_serve_execution_permit_initialize(&second_domain, &second);
  loom_serve_execution_permit_request(&first, 1);
  loom_serve_execution_permit_request(&second, 1);
  ASSERT_TRUE(loom_serve_execution_scheduler_advance(&first_domain, 100));
  ASSERT_TRUE(loom_serve_execution_scheduler_advance(&second_domain, 100));
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&first, 100));
  ASSERT_TRUE(loom_serve_execution_permit_try_acquire(&second, 100));
  // The owner has stopped new grants after a terminal native failure. Actual
  // device storage retires independently of these final service records.
  loom_serve_execution_permit_deinitialize(&first);
  loom_serve_execution_permit_deinitialize(&second);
  EXPECT_EQ(first_domain.outstanding_count, 0u);
  EXPECT_EQ(second_domain.outstanding_count, 0u);
}

}  // namespace
