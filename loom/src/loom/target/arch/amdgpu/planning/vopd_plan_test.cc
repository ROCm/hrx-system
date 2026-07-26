// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/vopd_plan.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

class AmdgpuVopdPlanVerifyTest : public ::testing::Test {
 protected:
  AmdgpuVopdPlanVerifyTest() {
    schedule_.node_count = 3;
    schedule_.scheduled_node_count = 3;
  }

  loom_amdgpu_vopd_pair_t ValidPair() {
    loom_amdgpu_vopd_pair_t pair = {};
    pair.reason = LOOM_AMDGPU_VOPD_PAIR_REASON_MIXED_COMPONENTS;
    pair.first_packet_index = 0;
    pair.second_packet_index = 1;
    pair.first_node_index = 0;
    pair.second_node_index = 1;
    pair.op_x = LOOM_AMDGPU_VOPD_OP_ADD_F32;
    pair.op_y = LOOM_AMDGPU_VOPD_OP_MUL_F32;
    return pair;
  }

  loom_amdgpu_vopd_plan_t Plan(loom_amdgpu_vopd_pair_t* pairs,
                               iree_host_size_t pair_count,
                               loom_amdgpu_vopd_packet_t* packets,
                               iree_host_size_t packet_count) {
    loom_amdgpu_vopd_plan_t plan = {};
    plan.schedule = &schedule_;
    plan.allocation = &allocation_;
    plan.pairs = pairs;
    plan.pair_count = pair_count;
    plan.packets = packets;
    plan.packet_count = packet_count;
    return plan;
  }

  loom_low_schedule_table_t schedule_ = {};
  loom_low_allocation_table_t allocation_ = {};
};

TEST_F(AmdgpuVopdPlanVerifyTest, AcceptsConsistentPacketMembership) {
  loom_amdgpu_vopd_pair_t pairs[1] = {ValidPair()};
  loom_amdgpu_vopd_packet_t packets[3] = {
      {LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST, 0},
      {LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND, 0},
      {LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE, LOOM_AMDGPU_VOPD_PAIR_NONE},
  };
  loom_amdgpu_vopd_plan_t plan = Plan(pairs, 1, packets, 3);

  IREE_ASSERT_OK(loom_amdgpu_vopd_plan_verify(&schedule_, &allocation_, &plan));
}

TEST_F(AmdgpuVopdPlanVerifyTest, RejectsMissingPacketRecords) {
  loom_amdgpu_vopd_pair_t pairs[1] = {ValidPair()};
  loom_amdgpu_vopd_plan_t plan = Plan(pairs, 1, nullptr, 3);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_vopd_plan_verify(&schedule_, &allocation_, &plan));
}

TEST_F(AmdgpuVopdPlanVerifyTest, RejectsPacketPairOutOfRange) {
  loom_amdgpu_vopd_pair_t pairs[1] = {ValidPair()};
  loom_amdgpu_vopd_packet_t packets[3] = {
      {LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST, 1},
      {LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND, 0},
      {LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE, LOOM_AMDGPU_VOPD_PAIR_NONE},
  };
  loom_amdgpu_vopd_plan_t plan = Plan(pairs, 1, packets, 3);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_vopd_plan_verify(&schedule_, &allocation_, &plan));
}

TEST_F(AmdgpuVopdPlanVerifyTest, RejectsUnsupportedPacketRole) {
  loom_amdgpu_vopd_packet_t packets[3] = {
      {static_cast<loom_amdgpu_vopd_packet_role_t>(99),
       LOOM_AMDGPU_VOPD_PAIR_NONE},
      {LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE, LOOM_AMDGPU_VOPD_PAIR_NONE},
      {LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE, LOOM_AMDGPU_VOPD_PAIR_NONE},
  };
  loom_amdgpu_vopd_plan_t plan = Plan(nullptr, 0, packets, 3);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_vopd_plan_verify(&schedule_, &allocation_, &plan));
}

TEST_F(AmdgpuVopdPlanVerifyTest, RejectsUnmatchedPairMembership) {
  loom_amdgpu_vopd_pair_t pairs[1] = {ValidPair()};
  loom_amdgpu_vopd_packet_t packets[3] = {
      {LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST, 0},
      {LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE, LOOM_AMDGPU_VOPD_PAIR_NONE},
      {LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE, LOOM_AMDGPU_VOPD_PAIR_NONE},
  };
  loom_amdgpu_vopd_plan_t plan = Plan(pairs, 1, packets, 3);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_vopd_plan_verify(&schedule_, &allocation_, &plan));
}

TEST_F(AmdgpuVopdPlanVerifyTest, RejectsUnsupportedComponentOps) {
  loom_amdgpu_vopd_pair_t pairs[1] = {ValidPair()};
  pairs[0].op_x = UINT16_MAX;
  loom_amdgpu_vopd_packet_t packets[3] = {
      {LOOM_AMDGPU_VOPD_PACKET_ROLE_FIRST, 0},
      {LOOM_AMDGPU_VOPD_PACKET_ROLE_SECOND, 0},
      {LOOM_AMDGPU_VOPD_PACKET_ROLE_NONE, LOOM_AMDGPU_VOPD_PAIR_NONE},
  };
  loom_amdgpu_vopd_plan_t plan = Plan(pairs, 1, packets, 3);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_amdgpu_vopd_plan_verify(&schedule_, &allocation_, &plan));
}

TEST_F(AmdgpuVopdPlanVerifyTest, RejectsMissingWaitStateRecords) {
  loom_amdgpu_vopd_plan_t plan = Plan(nullptr, 0, nullptr, 0);
  loom_amdgpu_wait_state_plan_t wait_states = {};
  wait_states.schedule = &schedule_;
  wait_states.allocation = &allocation_;
  wait_states.state_count = 1;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_amdgpu_vopd_plan_verify_wait_insertions(
                            &plan, nullptr, &wait_states));
}

TEST_F(AmdgpuVopdPlanVerifyTest, RejectsUnsupportedWaitStateActions) {
  loom_amdgpu_vopd_plan_t plan = Plan(nullptr, 0, nullptr, 0);
  loom_amdgpu_wait_state_t states[1] = {};
  states[0].action = static_cast<loom_amdgpu_wait_state_action_t>(99);
  loom_amdgpu_wait_state_plan_t wait_states = {};
  wait_states.schedule = &schedule_;
  wait_states.allocation = &allocation_;
  wait_states.states = states;
  wait_states.state_count = 1;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_amdgpu_vopd_plan_verify_wait_insertions(
                            &plan, nullptr, &wait_states));
}

}  // namespace
}  // namespace loom
