// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/kernel_report.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/planning/wait_counters.h"

namespace loom {
namespace {

class AmdgpuKernelReportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_target_compile_report_initialize(&report_, iree_allocator_system());
    report_.requested_detail_flags =
        LOOM_TARGET_COMPILE_REPORT_DETAIL_WAIT_PLAN;
    schedule_.nodes = nodes_;
    schedule_.node_count = IREE_ARRAYSIZE(nodes_);
    for (uint32_t i = 0; i < IREE_ARRAYSIZE(nodes_); ++i) {
      nodes_[i].scheduled_ordinal = i;
    }
    plan_.wait_plan.schedule = &schedule_;
    plan_.wait_states.schedule = &schedule_;
  }

  void TearDown() override {
    loom_target_compile_report_deinitialize(&report_);
  }

  // Four already scheduled structural nodes; reporting needs no authored IR.
  loom_low_schedule_node_t nodes_[4] = {};
  // Producer-owned coordinates used by wait endpoint reporting.
  loom_low_schedule_table_t schedule_ = {};
  // Final counter and fixed-state actions, borrowing the test's stack storage.
  loom_amdgpu_packet_plan_t plan_ = {};
  // Ordinary compiler report owning its copied rows.
  loom_target_compile_report_t report_ = {};
};

TEST_F(AmdgpuKernelReportTest, NativeDestinationWaitWithoutCounterActions) {
  const loom_amdgpu_wait_state_t state = {
      .reason = LOOM_AMDGPU_WAIT_STATE_REASON_SGPR_READ_DEPENDENCY,
      .action = LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAIT_ALU,
      .node_index = 2,
      .scheduled_ordinal = 2,
      .producer_node = 1,
      .consumer_node = 2,
      .required_cycle_count = 1,
      .cycle_count = 1,
      .immediate = 0xfffe,
  };
  plan_.wait_states.states = &state;
  plan_.wait_states.state_count = 1;
  IREE_ASSERT_OK(loom_amdgpu_kernel_report_record_plan(&report_, &plan_));
  EXPECT_EQ(report_.wait_plan.action_count, 1u);
  EXPECT_EQ(report_.wait_plan.partial_wait_count, 1u);
  EXPECT_EQ(report_.wait_plan.full_drain_count, 0u);
  ASSERT_EQ(report_.wait_action_rows.count, 1u);
  const auto* row =
      static_cast<const loom_target_compile_report_wait_action_row_t*>(
          loom_target_compile_report_vec_const_rows(
              report_.wait_action_rows.head));
  EXPECT_EQ(row->counter_id, LOOM_AMDGPU_WAIT_COUNTER_ALU);
  EXPECT_TRUE(iree_string_view_equal(row->reason_name,
                                     IREE_SV("amdgpu.sgpr_read_dependency")));
  EXPECT_EQ(row->producer_node, 1u);
  EXPECT_EQ(row->consumer_node, 2u);
  EXPECT_EQ(row->target_count, 0xfffeu);
  EXPECT_EQ(row->outstanding_after, 0u);
}

TEST_F(AmdgpuKernelReportTest, MergesCounterAndNativeActionsInScheduleOrder) {
  const loom_amdgpu_wait_plan_action_t counter = {
      .kind = LOOM_AMDGPU_WAIT_PLAN_ACTION_PLANNED,
      .reason = LOOM_AMDGPU_WAIT_PLAN_REASON_SSA_USE,
      .counter_id = LOOM_AMDGPU_WAIT_COUNTER_VMEM_LOAD,
      .node_index = 2,
      .scheduled_ordinal = 2,
      .producer_node = 0,
      .consumer_node = 2,
      .outstanding_before = 1,
  };
  loom_amdgpu_wait_state_t states[3] = {};
  states[0].reason = LOOM_AMDGPU_WAIT_STATE_REASON_DELAY_ALU_DEPENDENCY;
  states[0].action = LOOM_AMDGPU_WAIT_STATE_ACTION_S_DELAY_ALU;
  states[0].producer_node = LOOM_LOW_SCHEDULE_NODE_NONE;
  states[0].required_cycle_count = states[0].cycle_count = 1;
  states[0].immediate = 1;
  for (unsigned i = 1; i < IREE_ARRAYSIZE(states); ++i) {
    states[i].reason = LOOM_AMDGPU_WAIT_STATE_REASON_SGPR_READ_DEPENDENCY;
    states[i].action = LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAIT_ALU;
    states[i].node_index = i == 1 ? 1 : 3;
    states[i].scheduled_ordinal = states[i].node_index;
    states[i].producer_node = LOOM_LOW_SCHEDULE_NODE_NONE;
    states[i].consumer_node = states[i].node_index;
    states[i].required_cycle_count = states[i].cycle_count = 1;
    states[i].immediate = i == 1 ? 0xf1ff : 0xfffd;
  }
  plan_.wait_plan.actions = &counter;
  plan_.wait_plan.action_count = 1;
  plan_.wait_states.states = states;
  plan_.wait_states.state_count = IREE_ARRAYSIZE(states);
  IREE_ASSERT_OK(loom_amdgpu_kernel_report_record_plan(&report_, &plan_));
  EXPECT_EQ(report_.wait_plan.action_count, 3u);
  EXPECT_EQ(report_.wait_plan.full_drain_count, 1u);
  EXPECT_EQ(report_.wait_plan.partial_wait_count, 2u);
  ASSERT_EQ(report_.wait_action_rows.count, 3u);
  ASSERT_EQ(report_.wait_action_rows.head->count, 3u);
  const auto* rows =
      static_cast<const loom_target_compile_report_wait_action_row_t*>(
          loom_target_compile_report_vec_const_rows(
              report_.wait_action_rows.head));
  for (unsigned i = 0; i < 3; ++i) {
    EXPECT_EQ(rows[i].node_index, i + 1);
  }
  EXPECT_EQ(rows[0].producer_node, LOOM_LOW_SCHEDULE_NODE_NONE);
  EXPECT_EQ(rows[1].drained_count, 1u);
}

TEST_F(AmdgpuKernelReportTest, SummaryDoesNotRetainActionDetails) {
  report_.requested_detail_flags = LOOM_TARGET_COMPILE_REPORT_DETAIL_NONE;
  const loom_amdgpu_wait_state_t state = {
      .reason = LOOM_AMDGPU_WAIT_STATE_REASON_SGPR_READ_DEPENDENCY,
      .action = LOOM_AMDGPU_WAIT_STATE_ACTION_S_WAIT_ALU,
      .producer_node = LOOM_LOW_SCHEDULE_NODE_NONE,
      .required_cycle_count = 1,
      .cycle_count = 1,
      .immediate = 0xf1fe,
  };
  plan_.wait_states.states = &state;
  plan_.wait_states.state_count = 1;
  IREE_ASSERT_OK(loom_amdgpu_kernel_report_record_plan(&report_, &plan_));
  EXPECT_EQ(report_.wait_plan.action_count, 1u);
  EXPECT_EQ(report_.wait_action_rows.count, 0u);
  EXPECT_EQ(report_.wait_counter_rows.count, 0u);
  EXPECT_EQ(report_.wait_reason_summary_rows.count, 1u);
}

}  // namespace
}  // namespace loom
