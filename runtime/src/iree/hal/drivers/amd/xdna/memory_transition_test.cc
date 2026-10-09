// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/memory_transition.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

using ::iree::testing::status::StatusIs;

TEST(MemoryTransitionTest, PreservesHostReleaseRecipeAndPairFacts) {
  amdf_memory_pair_info_t source = {};
  source.flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE |
                 AMDF_MEMORY_PAIR_FLAG_MAPPING_SOURCE |
                 AMDF_MEMORY_PAIR_FLAG_FIXED_COST_KNOWN;
  source.release.kind = AMDF_CACHE_TRANSITION_KIND_RANGE;
  source.release.executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT;
  source.release.host_operation = AMDF_HOST_CACHE_OPERATION_FLUSH;
  source.release.host_instruction = AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH;
  source.release.host_fence_after = AMDF_HOST_CACHE_FENCE_X86_MFENCE;
  source.release.range_granularity = 64;
  source.acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
  source.atomic_reach.scope_32 = AMDF_ATOMIC_SCOPE_FABRIC;
  source.atomic_reach.scope_64 = AMDF_ATOMIC_SCOPE_SYSTEM;
  source.estimated_fixed_cost_nanoseconds = 17;

  iree_hal_memory_pair_info_t pair;
  IREE_ASSERT_OK(iree_hal_amd_xdna_memory_translate_pair(&source, &pair));
  EXPECT_EQ(pair.flags, IREE_HAL_MEMORY_PAIR_SHARED_BACKING_REACHABLE |
                            IREE_HAL_MEMORY_PAIR_MAPPING_SOURCE |
                            IREE_HAL_MEMORY_PAIR_FIXED_COST_KNOWN);
  EXPECT_EQ(pair.release.kind, IREE_HAL_MEMORY_TRANSITION_KIND_RANGE);
  EXPECT_EQ(pair.release.executor,
            IREE_HAL_MEMORY_TRANSITION_EXECUTOR_HOST_DIRECT);
  EXPECT_EQ(pair.release.operation,
            IREE_HAL_MEMORY_TRANSITION_OPERATION_HOST_FLUSH);
  EXPECT_EQ(pair.release.range_granularity, 64u);
  EXPECT_EQ(pair.release.host.instruction,
            IREE_HAL_HOST_CACHE_INSTRUCTION_X86_CLFLUSH);
  EXPECT_EQ(pair.release.host.fence_after,
            IREE_HAL_HOST_CACHE_FENCE_X86_MFENCE);
  EXPECT_EQ(pair.acquire.kind, IREE_HAL_MEMORY_TRANSITION_KIND_NONE);
  EXPECT_EQ(pair.atomic_reach.scope_32, IREE_HAL_ATOMIC_REACH_FABRIC);
  EXPECT_EQ(pair.atomic_reach.scope_64, IREE_HAL_ATOMIC_REACH_SYSTEM);
  EXPECT_EQ(pair.estimated_fixed_cost_nanoseconds, 17u);
}

TEST(MemoryTransitionTest, PreservesQueueAcquireRecipe) {
  amdf_memory_pair_info_t source = {};
  source.flags = AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE;
  source.release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
  source.acquire.kind = AMDF_CACHE_TRANSITION_KIND_GLOBAL;
  source.acquire.executor = AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE;
  source.acquire.operation = AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM;

  iree_hal_memory_pair_info_t pair;
  IREE_ASSERT_OK(iree_hal_amd_xdna_memory_translate_pair(&source, &pair));
  EXPECT_EQ(pair.release.kind, IREE_HAL_MEMORY_TRANSITION_KIND_NONE);
  EXPECT_EQ(pair.acquire.kind, IREE_HAL_MEMORY_TRANSITION_KIND_GLOBAL);
  EXPECT_EQ(pair.acquire.executor, IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE);
  EXPECT_EQ(pair.acquire.operation,
            IREE_HAL_MEMORY_TRANSITION_OPERATION_ACQUIRE_FROM_SYSTEM);
}

TEST(MemoryTransitionTest, RejectsMalformedSuccessfulResult) {
  amdf_memory_pair_info_t source = {};
  source.release.kind = AMDF_CACHE_TRANSITION_KIND_RANGE;
  source.release.executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT;
  source.release.host_operation = AMDF_HOST_CACHE_OPERATION_FLUSH;
  source.release.range_granularity = 96;

  iree_hal_memory_pair_info_t pair;
  iree_status_t status =
      iree_hal_amd_xdna_memory_translate_pair(&source, &pair);
  EXPECT_THAT(status, StatusIs(iree::StatusCode::kFailedPrecondition));
  iree_status_free(status);
}

}  // namespace
