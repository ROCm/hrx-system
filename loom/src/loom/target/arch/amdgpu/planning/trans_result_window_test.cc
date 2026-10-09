// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/trans_result_window.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"

namespace loom {
namespace {

class AmdgpuTransResultWindowTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(16 * 1024, iree_allocator_system(),
                                     &pool_);
    iree_arena_initialize(&pool_, &arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  static loom_low_allocation_assignment_t PhysicalVgpr(uint32_t base,
                                                       uint32_t count) {
    loom_low_allocation_assignment_t assignment =
        {};  // NOLINT(iree-cpp-designated-initializer) -- Assignment order
             // differs from declaration order.
    assignment.location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER;
    assignment.descriptor_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_VGPR;
    assignment.location_base = base;
    assignment.location_count = count;
    return assignment;
  }

  iree_arena_block_pool_t pool_;
  iree_arena_allocator_t arena_;
};

TEST_F(AmdgpuTransResultWindowTest, TracksOverlappingAssignmentOrigins) {
  loom_amdgpu_trans_result_window_t window = {};
  IREE_ASSERT_OK(loom_amdgpu_trans_result_window_initialize(
      8, LOOM_AMDGPU_TRANS_RESULT_WINDOW_FLAG_TRACK_ORIGINS, &arena_, &window));

  const loom_low_allocation_assignment_t first = PhysicalVgpr(2, 2);
  loom_amdgpu_trans_result_window_record_assignment(&window, &first, 17);
  EXPECT_TRUE(loom_amdgpu_trans_result_window_has_active(&window));
  uint32_t origin = UINT32_MAX;
  EXPECT_TRUE(loom_amdgpu_trans_result_window_query_assignment_origin(
      &window, &first, &origin));
  EXPECT_EQ(origin, 17u);

  const loom_low_allocation_assignment_t overlapping = PhysicalVgpr(3, 2);
  loom_amdgpu_trans_result_window_record_assignment(&window, &overlapping, 29);
  const loom_low_allocation_assignment_t register_two = PhysicalVgpr(2, 1);
  EXPECT_TRUE(loom_amdgpu_trans_result_window_query_assignment_origin(
      &window, &register_two, &origin));
  EXPECT_EQ(origin, 17u);
  const loom_low_allocation_assignment_t register_three = PhysicalVgpr(3, 1);
  EXPECT_TRUE(loom_amdgpu_trans_result_window_query_assignment_origin(
      &window, &register_three, &origin));
  EXPECT_EQ(origin, 29u);

  loom_amdgpu_trans_result_window_clear_assignment(&window, &first);
  EXPECT_FALSE(loom_amdgpu_trans_result_window_query_assignment_origin(
      &window, &register_three, &origin));
  const loom_low_allocation_assignment_t register_four = PhysicalVgpr(4, 1);
  EXPECT_TRUE(loom_amdgpu_trans_result_window_query_assignment_origin(
      &window, &register_four, &origin));
  EXPECT_EQ(origin, 29u);
  loom_amdgpu_trans_result_window_clear_assignment(&window, &register_four);
  EXPECT_FALSE(loom_amdgpu_trans_result_window_has_active(&window));
}

TEST_F(AmdgpuTransResultWindowTest, ExpiresAtArchitectureIntervals) {
  loom_amdgpu_trans_result_window_t window = {};
  IREE_ASSERT_OK(loom_amdgpu_trans_result_window_initialize(
      4, LOOM_AMDGPU_TRANS_RESULT_WINDOW_FLAG_TRACK_ORIGINS, &arena_, &window));
  const loom_low_allocation_assignment_t assignment = PhysicalVgpr(1, 1);

  loom_amdgpu_trans_result_window_record_assignment(&window, &assignment, 3);
  for (uint32_t i = 0; i < LOOM_AMDGPU_VALU_TRANS_USE_DEPCTR_MAX_VALU_INTERVAL;
       ++i) {
    loom_amdgpu_trans_result_window_advance(
        &window, LOOM_AMDGPU_TRANS_RESULT_PACKET_FLAG_VECTOR_ALU);
    EXPECT_TRUE(loom_amdgpu_trans_result_window_has_active(&window));
  }
  loom_amdgpu_trans_result_window_advance(
      &window, LOOM_AMDGPU_TRANS_RESULT_PACKET_FLAG_VECTOR_ALU);
  EXPECT_FALSE(loom_amdgpu_trans_result_window_has_active(&window));

  loom_amdgpu_trans_result_window_record_assignment(&window, &assignment, 5);
  for (uint32_t i = 0; i < LOOM_AMDGPU_VALU_TRANS_USE_DEPCTR_MAX_TRANS_INTERVAL;
       ++i) {
    loom_amdgpu_trans_result_window_advance(
        &window, LOOM_AMDGPU_TRANS_RESULT_PACKET_FLAG_TRANSCENDENTAL);
    EXPECT_TRUE(loom_amdgpu_trans_result_window_has_active(&window));
  }
  loom_amdgpu_trans_result_window_advance(
      &window, LOOM_AMDGPU_TRANS_RESULT_PACKET_FLAG_TRANSCENDENTAL);
  EXPECT_FALSE(loom_amdgpu_trans_result_window_has_active(&window));
}

TEST_F(AmdgpuTransResultWindowTest, OverwriteRestartsAndClearReusesStorage) {
  loom_amdgpu_trans_result_window_t window = {};
  IREE_ASSERT_OK(loom_amdgpu_trans_result_window_initialize(
      2, LOOM_AMDGPU_TRANS_RESULT_WINDOW_FLAG_TRACK_ORIGINS, &arena_, &window));
  const loom_low_allocation_assignment_t assignment = PhysicalVgpr(0, 1);

  loom_amdgpu_trans_result_window_record_assignment(&window, &assignment, 7);
  for (uint32_t i = 0; i < LOOM_AMDGPU_VALU_TRANS_USE_DEPCTR_MAX_VALU_INTERVAL;
       ++i) {
    loom_amdgpu_trans_result_window_advance(
        &window, LOOM_AMDGPU_TRANS_RESULT_PACKET_FLAG_VECTOR_ALU);
  }
  loom_amdgpu_trans_result_window_record_assignment(&window, &assignment, 11);
  loom_amdgpu_trans_result_window_advance(
      &window, LOOM_AMDGPU_TRANS_RESULT_PACKET_FLAG_VECTOR_ALU);
  uint32_t origin = UINT32_MAX;
  EXPECT_TRUE(loom_amdgpu_trans_result_window_query_assignment_origin(
      &window, &assignment, &origin));
  EXPECT_EQ(origin, 11u);

  loom_amdgpu_trans_result_window_clear(&window);
  EXPECT_FALSE(loom_amdgpu_trans_result_window_has_active(&window));
  loom_amdgpu_trans_result_window_record_assignment(&window, &assignment, 13);
  EXPECT_TRUE(loom_amdgpu_trans_result_window_query_assignment_origin(
      &window, &assignment, &origin));
  EXPECT_EQ(origin, 13u);
}

TEST_F(AmdgpuTransResultWindowTest, OriginsAreOptional) {
  loom_amdgpu_trans_result_window_t window = {};
  IREE_ASSERT_OK(loom_amdgpu_trans_result_window_initialize(2, /*flags=*/0,
                                                            &arena_, &window));
  const loom_low_allocation_assignment_t assignment = PhysicalVgpr(0, 1);
  loom_amdgpu_trans_result_window_record_assignment(&window, &assignment, 17);
  EXPECT_TRUE(loom_amdgpu_trans_result_window_has_active(&window));
  loom_amdgpu_trans_result_window_clear(&window);
  EXPECT_FALSE(loom_amdgpu_trans_result_window_has_active(&window));
}

}  // namespace
}  // namespace loom
