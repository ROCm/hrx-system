// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/utils/profile_clock_alignment.h"

#include "iree/testing/gtest.h"

namespace iree::hal {
namespace {

TEST(ProfileClockAlignmentTest, EventBoundedByClockSamplesIsAligned) {
  iree_hal_profile_clock_alignment_t alignment;
  iree_hal_profile_clock_alignment_reset(&alignment);

  EXPECT_FALSE(
      iree_hal_profile_clock_alignment_record_clock_tick(&alignment, 100));
  iree_hal_profile_clock_alignment_record_event_range(&alignment, 120, 180);
  EXPECT_FALSE(
      iree_hal_profile_clock_alignment_record_clock_tick(&alignment, 200));
}

TEST(ProfileClockAlignmentTest, EventWaitsForEndingClockSample) {
  iree_hal_profile_clock_alignment_t alignment;
  iree_hal_profile_clock_alignment_reset(&alignment);

  EXPECT_FALSE(
      iree_hal_profile_clock_alignment_record_clock_tick(&alignment, 100));
  iree_hal_profile_clock_alignment_record_event_range(&alignment, 120, 250);
  EXPECT_FALSE(alignment.has_invalid_alignment);
  EXPECT_FALSE(
      iree_hal_profile_clock_alignment_record_clock_tick(&alignment, 300));
}

TEST(ProfileClockAlignmentTest, ShiftedEventClockIsPermanentlyUnaligned) {
  iree_hal_profile_clock_alignment_t alignment;
  iree_hal_profile_clock_alignment_reset(&alignment);

  EXPECT_FALSE(
      iree_hal_profile_clock_alignment_record_clock_tick(&alignment, 100));
  iree_hal_profile_clock_alignment_record_event_range(&alignment, 1120, 1180);
  EXPECT_TRUE(
      iree_hal_profile_clock_alignment_record_clock_tick(&alignment, 200));
  EXPECT_TRUE(
      iree_hal_profile_clock_alignment_record_clock_tick(&alignment, 1200));
}

TEST(ProfileClockAlignmentTest, EventBeforeInitialSampleIsUnaligned) {
  iree_hal_profile_clock_alignment_t alignment;
  iree_hal_profile_clock_alignment_reset(&alignment);

  EXPECT_FALSE(
      iree_hal_profile_clock_alignment_record_clock_tick(&alignment, 100));
  iree_hal_profile_clock_alignment_record_event_range(&alignment, 80, 120);
  EXPECT_TRUE(
      iree_hal_profile_clock_alignment_record_clock_tick(&alignment, 200));
}

TEST(ProfileClockAlignmentTest, ReversedEventRangeIsUnaligned) {
  iree_hal_profile_clock_alignment_t alignment;
  iree_hal_profile_clock_alignment_reset(&alignment);

  iree_hal_profile_clock_alignment_record_event_range(&alignment, 200, 100);
  EXPECT_TRUE(
      iree_hal_profile_clock_alignment_record_clock_tick(&alignment, 300));
}

TEST(ProfileClockAlignmentTest, ResetStartsIndependentSession) {
  iree_hal_profile_clock_alignment_t alignment;
  iree_hal_profile_clock_alignment_reset(&alignment);
  iree_hal_profile_clock_alignment_record_event_range(&alignment, 200, 100);
  EXPECT_TRUE(
      iree_hal_profile_clock_alignment_record_clock_tick(&alignment, 300));

  iree_hal_profile_clock_alignment_reset(&alignment);
  EXPECT_FALSE(
      iree_hal_profile_clock_alignment_record_clock_tick(&alignment, 400));
  iree_hal_profile_clock_alignment_record_event_range(&alignment, 450, 500);
  EXPECT_FALSE(
      iree_hal_profile_clock_alignment_record_clock_tick(&alignment, 550));
}

}  // namespace
}  // namespace iree::hal
