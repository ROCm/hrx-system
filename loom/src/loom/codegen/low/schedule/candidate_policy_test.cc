// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/schedule/candidate_policy.h"

#include "iree/testing/gtest.h"

namespace loom {
namespace {

static loom_low_schedule_candidate_score_t MakeConstrainedCandidate(
    uint32_t effective_stall_cycles, uint16_t flags) {
  return {
      .effective_stall_cycles = effective_stall_cycles,
      .active_unspillable_completion_domain_id = 1,
      .flags = flags,
  };
}

class CandidatePolicyTest : public ::testing::Test {
 protected:
  CandidatePolicyTest() { state_.options = &options_; }

  loom_low_schedule_options_t options_ = {
      .strategy = LOOM_LOW_SCHEDULE_STRATEGY_RESOURCE_STALL,
  };
  loom_low_schedule_build_state_t state_ = {};
};

TEST_F(CandidatePolicyTest, PackingRecoveryPreservesStorageHandoff) {
  const loom_low_schedule_candidate_score_t storage_handoff =
      MakeConstrainedCandidate(
          3, LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_STORAGE);
  const loom_low_schedule_candidate_score_t ready_peer =
      MakeConstrainedCandidate(1, 0);

  // The handoff retains the selected aggregate completion. Opening the peer
  // first can fragment the full bank before that completion reaches its use.
  EXPECT_FALSE(loom_low_schedule_candidate_policy_uses_ready_progress_order(
      &state_, LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION,
      &storage_handoff, &ready_peer));
  EXPECT_FALSE(loom_low_schedule_candidate_policy_uses_ready_progress_order(
      &state_, LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION,
      &ready_peer, &storage_handoff));
}

TEST_F(CandidatePolicyTest, PressureReliefSchedulesReadyPeer) {
  const loom_low_schedule_candidate_score_t storage_handoff =
      MakeConstrainedCandidate(
          3, LOOM_LOW_SCHEDULE_CANDIDATE_FLAG_ADVANCES_STORAGE);
  const loom_low_schedule_candidate_score_t ready_peer =
      MakeConstrainedCandidate(1, 0);

  EXPECT_TRUE(loom_low_schedule_candidate_policy_uses_ready_progress_order(
      &state_, LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PRESSURE_RELIEF, &ready_peer,
      &storage_handoff));
}

TEST_F(CandidatePolicyTest, PackingRecoverySchedulesReadyPeerWithoutHandoff) {
  const loom_low_schedule_candidate_score_t stalled_peer =
      MakeConstrainedCandidate(3, 0);
  const loom_low_schedule_candidate_score_t ready_peer =
      MakeConstrainedCandidate(1, 0);

  EXPECT_TRUE(loom_low_schedule_candidate_policy_uses_ready_progress_order(
      &state_, LOOM_LOW_SCHEDULE_CANDIDATE_COMPARE_PACKING_COMPLETION,
      &ready_peer, &stalled_peer));
}

}  // namespace
}  // namespace loom
