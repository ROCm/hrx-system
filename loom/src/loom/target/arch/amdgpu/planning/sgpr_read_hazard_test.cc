// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/planning/sgpr_read_hazard.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amdgpu/refs/target_refs.h"

namespace loom {
namespace {

class AmdgpuSgprReadHazardTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(16 * 1024, iree_allocator_system(),
                                     &pool_);
    iree_arena_initialize(&pool_, &arena_);
    IREE_ASSERT_OK(loom_amdgpu_sgpr_read_hazard_initialize(
        8, /*fixed_register_base=*/0, /*fixed_register_count=*/0, &arena_,
        &hazard_));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  static loom_low_allocation_assignment_t PhysicalSgpr(uint32_t base,
                                                       uint32_t count) {
    loom_low_allocation_assignment_t assignment = {
        .descriptor_reg_class_id = LOOM_AMDGPU_REG_CLASS_ID_SGPR,
        .location_kind = LOOM_LOW_ALLOCATION_LOCATION_PHYSICAL_REGISTER,
        .location_base = base,
        .location_count = count,
    };
    return assignment;
  }

  iree_arena_block_pool_t pool_;
  iree_arena_allocator_t arena_;
  loom_amdgpu_sgpr_read_hazard_t hazard_ = {};
};

TEST_F(AmdgpuSgprReadHazardTest, ScalarWriteBlocksScalarAndVectorReads) {
  const loom_low_allocation_assignment_t odd_pair_read = PhysicalSgpr(1, 1);
  loom_amdgpu_sgpr_read_hazard_track_read(&hazard_, &odd_pair_read);
  const loom_low_allocation_assignment_t even_pair_write = PhysicalSgpr(0, 1);
  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard_, &even_pair_write, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR,
      17);

  uint32_t origin = UINT32_MAX;
  EXPECT_TRUE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard_, &even_pair_write, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR,
      &origin));
  EXPECT_EQ(origin, 17u);
  EXPECT_TRUE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard_, &even_pair_write, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_VECTOR,
      &origin));
  EXPECT_EQ(origin, 17u);
}

TEST_F(AmdgpuSgprReadHazardTest, VectorWriteBlocksOnlyVectorReads) {
  const loom_low_allocation_assignment_t even_pair_read = PhysicalSgpr(2, 1);
  loom_amdgpu_sgpr_read_hazard_track_read(&hazard_, &even_pair_read);
  const loom_low_allocation_assignment_t odd_pair_write = PhysicalSgpr(3, 1);
  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard_, &odd_pair_write, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_VECTOR,
      23);

  uint32_t origin = UINT32_MAX;
  EXPECT_FALSE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard_, &odd_pair_write, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR,
      &origin));
  EXPECT_TRUE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard_, &odd_pair_write, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_VECTOR,
      &origin));
  EXPECT_EQ(origin, 23u);
}

TEST_F(AmdgpuSgprReadHazardTest, TracksEveryOverlappingPair) {
  const loom_low_allocation_assignment_t spanning_read = PhysicalSgpr(1, 4);
  loom_amdgpu_sgpr_read_hazard_track_read(&hazard_, &spanning_read);

  const loom_low_allocation_assignment_t first_pair = PhysicalSgpr(0, 1);
  const loom_low_allocation_assignment_t middle_pair = PhysicalSgpr(3, 1);
  const loom_low_allocation_assignment_t last_pair = PhysicalSgpr(4, 1);
  const loom_low_allocation_assignment_t untracked_pair = PhysicalSgpr(6, 1);
  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard_, &first_pair, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR, 31);
  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard_, &middle_pair, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR, 37);
  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard_, &last_pair, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR, 41);
  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard_, &untracked_pair, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR,
      43);

  uint32_t origin = UINT32_MAX;
  EXPECT_TRUE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard_, &first_pair, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR,
      &origin));
  EXPECT_EQ(origin, 31u);
  EXPECT_TRUE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard_, &middle_pair, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR,
      &origin));
  EXPECT_EQ(origin, 37u);
  EXPECT_TRUE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard_, &last_pair, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR,
      &origin));
  EXPECT_EQ(origin, 41u);
  EXPECT_FALSE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard_, &untracked_pair, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_VECTOR,
      &origin));
}

TEST_F(AmdgpuSgprReadHazardTest, NewWriteReplacesDependencyKindAndOrigin) {
  const loom_low_allocation_assignment_t assignment = PhysicalSgpr(2, 1);
  loom_amdgpu_sgpr_read_hazard_track_read(&hazard_, &assignment);
  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard_, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR, 47);
  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard_, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_VECTOR, 53);

  uint32_t origin = UINT32_MAX;
  EXPECT_FALSE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard_, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR,
      &origin));
  EXPECT_TRUE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard_, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_VECTOR,
      &origin));
  EXPECT_EQ(origin, 53u);

  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard_, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR, 59);
  EXPECT_TRUE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard_, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR,
      &origin));
  EXPECT_EQ(origin, 59u);
}

TEST_F(AmdgpuSgprReadHazardTest, WriteBeforePairTrackingIsNotRetroactive) {
  const loom_low_allocation_assignment_t assignment = PhysicalSgpr(0, 1);
  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard_, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR, 61);
  loom_amdgpu_sgpr_read_hazard_track_read(&hazard_, &assignment);

  uint32_t origin = UINT32_MAX;
  EXPECT_FALSE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard_, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_VECTOR,
      &origin));
}

TEST_F(AmdgpuSgprReadHazardTest, DependencyDrainPreservesTrackedPairs) {
  const loom_low_allocation_assignment_t assignment = PhysicalSgpr(4, 1);
  loom_amdgpu_sgpr_read_hazard_track_read(&hazard_, &assignment);
  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard_, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR, 29);
  loom_amdgpu_sgpr_read_hazard_clear_dependencies(&hazard_);

  uint32_t origin = UINT32_MAX;
  EXPECT_FALSE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard_, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_VECTOR,
      &origin));
  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard_, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR, 31);
  EXPECT_TRUE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard_, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_VECTOR,
      &origin));
  EXPECT_EQ(origin, 31u);
}

TEST_F(AmdgpuSgprReadHazardTest, BlockBoundaryDropsTrackedPairs) {
  const loom_low_allocation_assignment_t assignment = PhysicalSgpr(6, 2);
  loom_amdgpu_sgpr_read_hazard_track_read(&hazard_, &assignment);
  loom_amdgpu_sgpr_read_hazard_begin_block(&hazard_);
  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard_, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR, 37);

  uint32_t origin = UINT32_MAX;
  EXPECT_FALSE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard_, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_VECTOR,
      &origin));
}

TEST_F(AmdgpuSgprReadHazardTest, TracksAnUnpairedTerminalRegister) {
  loom_amdgpu_sgpr_read_hazard_t hazard = {};
  IREE_ASSERT_OK(loom_amdgpu_sgpr_read_hazard_initialize(
      /*allocated_register_count=*/7, /*fixed_register_base=*/0,
      /*fixed_register_count=*/0, &arena_, &hazard));
  const loom_low_allocation_assignment_t assignment = PhysicalSgpr(6, 1);
  loom_amdgpu_sgpr_read_hazard_track_read(&hazard, &assignment);
  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR, 67);

  uint32_t origin = UINT32_MAX;
  EXPECT_TRUE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR,
      &origin));
  EXPECT_EQ(origin, 67u);
}

TEST_F(AmdgpuSgprReadHazardTest, TracksTheDisjointAbiFixedRange) {
  loom_amdgpu_sgpr_read_hazard_t hazard = {};
  IREE_ASSERT_OK(loom_amdgpu_sgpr_read_hazard_initialize(
      /*allocated_register_count=*/4, /*fixed_register_base=*/108,
      /*fixed_register_count=*/16, &arena_, &hazard));
  const loom_low_allocation_assignment_t assignment = PhysicalSgpr(123, 1);
  loom_amdgpu_sgpr_read_hazard_track_read(&hazard, &assignment);
  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR, 71);

  uint32_t origin = UINT32_MAX;
  EXPECT_TRUE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard, &assignment, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_VECTOR,
      &origin));
  EXPECT_EQ(origin, 71u);
}

TEST_F(AmdgpuSgprReadHazardTest, FixedRangePreservesPhysicalPairParity) {
  loom_amdgpu_sgpr_read_hazard_t hazard = {};
  IREE_ASSERT_OK(loom_amdgpu_sgpr_read_hazard_initialize(
      /*allocated_register_count=*/4, /*fixed_register_base=*/109,
      /*fixed_register_count=*/2, &arena_, &hazard));
  const loom_low_allocation_assignment_t first = PhysicalSgpr(109, 1);
  const loom_low_allocation_assignment_t second = PhysicalSgpr(110, 1);
  loom_amdgpu_sgpr_read_hazard_track_read(&hazard, &first);
  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard, &second, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR, 73);

  uint32_t origin = UINT32_MAX;
  EXPECT_FALSE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard, &second, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_VECTOR, &origin));
  loom_amdgpu_sgpr_read_hazard_record_write(
      &hazard, &first, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_SCALAR, 79);
  EXPECT_TRUE(loom_amdgpu_sgpr_read_hazard_query_read(
      &hazard, &first, LOOM_AMDGPU_SGPR_READ_HAZARD_ALU_FLAG_VECTOR, &origin));
  EXPECT_EQ(origin, 79u);
}

}  // namespace
}  // namespace loom
