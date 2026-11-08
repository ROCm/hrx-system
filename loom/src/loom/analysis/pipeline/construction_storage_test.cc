// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/analysis/pipeline/construction_storage.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/analysis/pipeline/invocations.h"
#include "loom/analysis/source_storage_packing.h"

namespace loom {
namespace {

// A parent owns a concurrent wrapper. The wrapper has its own concurrent child
// and two serial children in nested scopes. Only the latter two may share
// storage: all other owners remain live through their descendants.
class PipelineConstructionStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &blocks_);
    iree_arena_initialize(&blocks_, &arena_);
    for (size_t i = 0; i < 5; ++i) {
      entries_[i] = &occurrences_[i];
      occurrences_[i].ordinal = i;
      occurrences_[i].parent = i == 0 ? nullptr : &occurrences_[i == 1 ? 0 : 1];
      occurrences_[i].resources = &resources_[i];
    }
    resources_[0].composition_count = 1;
    occurrences_[0].compositions = &entries_[1];
    resources_[1].composition_count = 1;
    occurrences_[1].compositions = &entries_[2];
    control_.graph.block_count = 1;
    steps_[0][0].kind = LOOM_PIPELINE_SCHEDULE_STEP_COMPOSITION;
    steps_[0][1] = {
        .nested = &schedules_[1],
        .kind = LOOM_PIPELINE_SCHEDULE_STEP_SERIAL,
    };
    steps_[0][2].kind = LOOM_PIPELINE_SCHEDULE_STEP_YIELD;
    steps_[1][0].kind = LOOM_PIPELINE_SCHEDULE_STEP_COMPOSITION;
    steps_[1][1].kind = LOOM_PIPELINE_SCHEDULE_STEP_YIELD;
    bindings_[0][0].child = &occurrences_[3];
    bindings_[0][1].nested = &bound_[1];
    bindings_[1][0].child = &occurrences_[4];
    for (size_t i = 0; i < 2; ++i) {
      schedule_blocks_[i] = {steps_[i], i == 0 ? 3u : 2u};
      schedules_[i].control = &control_;
      schedules_[i].blocks = &schedule_blocks_[i];
      bound_blocks_[i] = bindings_[i];
      bound_[i] = {&schedules_[i], &bound_blocks_[i]};
      occurrences_[i + 3].schedule = &schedules_[i];
    }
    resources_[1].schedule_count = 1;
    occurrences_[1].schedules = bound_;
    invocations_ = {.entries = entries_, .count = 5};
    IREE_ASSERT_OK(loom_pipeline_lifetimes_build(&invocations_, &arena_,
                                                 &invocations_.lifetimes));
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&blocks_);
  }

  // Arena backing for retained plans and placements.
  iree_arena_block_pool_t blocks_;
  // Allocation owner for the complete graph.
  iree_arena_allocator_t arena_;
  // Construction-local counts and allocation requirements.
  loom_pipeline_resources_t resources_[5] = {};
  // Parent-before-child occurrences supplied by the admitting owner.
  loom_pipeline_invocation_t occurrences_[5] = {};
  // Complete inventory indexing those occurrences.
  const loom_pipeline_invocation_t* entries_[5] = {};
  // Canonical single-block CFG used by both serial scopes.
  loom_value_fact_cfg_region_t control_ = {};
  // Classified steps including each scope's yield.
  loom_pipeline_schedule_step_t steps_[2][3] = {};
  // Source block descriptors.
  loom_pipeline_schedule_block_t schedule_blocks_[2] = {};
  // Source serial scopes.
  loom_pipeline_schedule_t schedules_[2] = {};
  // Bound execution references parallel to the classified steps.
  loom_pipeline_invocation_step_t bindings_[2][3] = {};
  // Block pointer for each bound serial scope.
  const loom_pipeline_invocation_step_t* bound_blocks_[2] = {};
  // Scope bindings with distinct outer and nested lifetimes.
  loom_pipeline_invocation_schedule_t bound_[2] = {};
  // Complete retained inventory.
  loom_pipeline_invocations_t invocations_ = {};
};

TEST_F(PipelineConstructionStorageTest,
       StorageRespectsOwnersAndCanonicalBackings) {
  const loom_source_storage_packing_range_t fixed[] = {{0, 64}};
  loom_pipeline_resource_pool_t pools[2] = {};
  IREE_ASSERT_OK(loom_source_storage_packing_allocate({}, fixed, 1, &arena_,
                                                      &pools[0].packing));
  IREE_ASSERT_OK(loom_source_storage_packing_allocate({}, nullptr, 0, &arena_,
                                                      &pools[1].packing));
  // A different inventory ordering must still identify the same two backings.
  const loom_pipeline_resource_pool_t reversed[] = {pools[1], pools[0]};
  const uint64_t lengths[] = {32, 33, 48, 96, 24};
  const uint64_t alignments[] = {16, 32, 16, 128, 64};
  loom_pipeline_resource_allocation_t allocations[5][2] = {};
  for (size_t i = 0; i < 5; ++i) {
    resources_[i].pools = i % 2 ? reversed : pools;
    resources_[i].pool_count = 2;
    resources_[i].allocations = allocations[i];
    resources_[i].allocation_count = 2;
    allocations[i][0] = {
        .pool_index = static_cast<uint32_t>(i % 2),
        .byte_length = lengths[i],
        .byte_alignment = alignments[i],
    };
    allocations[i][1] = {
        .pool_index = static_cast<uint32_t>(1 - i % 2),
        .construction_order = 1,
        .byte_length = i == 3 ? 8u : 0u,
        .byte_alignment = 8,
    };
  }
  const loom_pipeline_storage_allocation_t** placed = nullptr;
  IREE_ASSERT_OK(loom_pipeline_construction_storage_build(&invocations_,
                                                          &arena_, &placed));
  const uint64_t offsets[] = {128, 256, 304, 384, 384};
  for (size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(placed[i][0].source, &allocations[i][0]);
    EXPECT_EQ(placed[i][0].byte_offset, offsets[i]);
    EXPECT_EQ(placed[i][0].byte_offset % alignments[i], 0u);
    occurrences_[i].storage = placed[i];
    for (size_t j = 0; j < i; ++j) {
      if (i == 4 && j == 3) {
        continue;
      }
      EXPECT_TRUE(offsets[i] >= offsets[j] + lengths[j] ||
                  offsets[j] >= offsets[i] + lengths[i]);
    }
  }
  EXPECT_EQ(
      loom_source_storage_packing_requirement(pools[0].packing).byte_length,
      480u);
  EXPECT_EQ(
      loom_source_storage_packing_requirement(pools[1].packing).byte_length,
      8u);
  uint64_t service_offset = UINT64_MAX;
  IREE_ASSERT_OK(loom_source_storage_packing_reserve(
      pools[0].packing, 64, 64, nullptr, 0, &service_offset));
  EXPECT_EQ(service_offset, 64u);
}

TEST_F(PipelineConstructionStorageTest,
       ConcurrentStorageUsesConstructionOrderAndFixedReservationHoles) {
  // Keep only the root's concurrent wrapper and its concurrent child.
  resources_[1].schedule_count = 0;
  invocations_.count = 3;
  IREE_ASSERT_OK(loom_pipeline_lifetimes_build(&invocations_, &arena_,
                                               &invocations_.lifetimes));
  const loom_source_storage_packing_range_t fixed[] = {{64, 64}, {192, 64}};
  loom_pipeline_resource_pool_t pool = {};
  IREE_ASSERT_OK(loom_source_storage_packing_allocate({}, fixed, 2, &arena_,
                                                      &pool.packing));
  loom_pipeline_resource_allocation_t allocations[3] = {};
  for (size_t i = 0; i < 3; ++i) {
    resources_[i].pools = &pool;
    resources_[i].pool_count = 1;
    resources_[i].allocations = &allocations[i];
    resources_[i].allocation_count = 1;
    allocations[i].byte_length = 64;
    allocations[i].byte_alignment = 64;
  }
  // Lookup order differs from construction order. The first allocation fits
  // below the first fixed range, while the second needs the next complete hole.
  const loom_pipeline_resource_allocation_t root_allocations[] = {
      {
          .root_value_id = 1,
          .construction_order = 1,
          .byte_length = 64,
          .byte_alignment = 64,
      },
      {
          .root_value_id = 2,
          .construction_order = 0,
          .byte_length = 32,
          .byte_alignment = 32,
      },
  };
  resources_[0].allocations = root_allocations;
  resources_[0].allocation_count = IREE_ARRAYSIZE(root_allocations);
  const loom_pipeline_storage_allocation_t** placed = nullptr;
  IREE_ASSERT_OK(loom_pipeline_construction_storage_build(&invocations_,
                                                          &arena_, &placed));
  EXPECT_EQ(placed[0][0].byte_offset, 128u);
  EXPECT_EQ(placed[0][1].byte_offset, 0u);
  EXPECT_EQ(placed[1][0].byte_offset, 256u);
  EXPECT_EQ(placed[2][0].byte_offset, 320u);
  EXPECT_EQ(loom_source_storage_packing_requirement(pool.packing).byte_length,
            384u);
}

}  // namespace
}  // namespace loom
