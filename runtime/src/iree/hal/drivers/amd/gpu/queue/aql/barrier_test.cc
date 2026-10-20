// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/barrier.h"

#include "iree/hal/device_spec.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

uint32_t Prepare(const iree_hal_barrier_t& value) {
  const iree_hal_barrier_list_t list = {1, &value};
  const iree_hal_queue_family_spec_t spec = {};
  const iree_hal_queue_family_t family = {.spec = &spec};
  IREE_CHECK_OK(iree_hal_barrier_validate(&value, &family));
  return iree_hal_amd_gpu_aql_barrier_prepare(&list);
}

TEST(AqlBarrierTest, DefaultAndExplicitEmptyHaveDifferentSemantics) {
  EXPECT_EQ(iree_hal_amd_gpu_aql_barrier_prepare(nullptr),
            IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
                IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM);
  const iree_hal_barrier_list_t empty = {};
  EXPECT_EQ(iree_hal_amd_gpu_aql_barrier_prepare(&empty), 0u);
  EXPECT_EQ(Prepare({}), 0u);
}

TEST(AqlBarrierTest, ExecutionOrderingDoesNotRequireCacheEffects) {
  EXPECT_EQ(Prepare({
                .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
                .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
            }),
            IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED);
  EXPECT_EQ(Prepare({
                .source_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST,
                .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
            }),
            0u);
}

TEST(AqlBarrierTest, LocalDependenciesStayAtAgentScope) {
  const iree_hal_access_scope_t scopes[] = {
      IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ,
      IREE_HAL_ACCESS_SCOPE_CONSTANT_READ,
      IREE_HAL_ACCESS_SCOPE_DISPATCH_READ,
      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      IREE_HAL_ACCESS_SCOPE_TRANSFER_READ,
      IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE,
      IREE_HAL_ACCESS_SCOPE_ATOMIC_READ,
      IREE_HAL_ACCESS_SCOPE_ATOMIC_WRITE,
  };
  for (auto source : scopes) {
    for (auto target : scopes) {
      const iree_hal_memory_barrier_t memory = {source, target};
      EXPECT_EQ(
          Prepare({.memory_barrier_count = 1, .memory_barriers = &memory}),
          IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
              IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_AGENT |
              IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_AGENT);
    }
  }
}

TEST(AqlBarrierTest, ExternalAccessPreservesDirection) {
  for (auto scope :
       {IREE_HAL_ACCESS_SCOPE_HOST_READ, IREE_HAL_ACCESS_SCOPE_HOST_WRITE,
        IREE_HAL_ACCESS_SCOPE_MEMORY_READ,
        IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE}) {
    const iree_hal_memory_barrier_t acquire = {
        scope, IREE_HAL_ACCESS_SCOPE_DISPATCH_READ};
    const iree_hal_memory_barrier_t release = {
        IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE, scope};
    EXPECT_EQ(Prepare({.memory_barrier_count = 1, .memory_barriers = &acquire}),
              IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                  IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
                  IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_AGENT);
    EXPECT_EQ(Prepare({.memory_barrier_count = 1, .memory_barriers = &release}),
              IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                  IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_AGENT |
                  IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM);
  }
}

TEST(AqlBarrierTest, OneSidedAccessScopesRetainTheirFence) {
  const iree_hal_memory_barrier_t acquire = {
      0, IREE_HAL_ACCESS_SCOPE_DISPATCH_READ};
  const iree_hal_memory_barrier_t release = {
      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE, 0};
  EXPECT_EQ(Prepare({.memory_barrier_count = 1, .memory_barriers = &acquire}),
            IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_AGENT);
  EXPECT_EQ(Prepare({.memory_barrier_count = 1, .memory_barriers = &release}),
            IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_AGENT);
}

TEST(AqlBarrierTest, PreparedRangesPromoteToWholeCacheActions) {
  for (bool acquire : {false, true}) {
    const uint32_t range_effect =
        acquire ? IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM
                : IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM;
    const uint32_t global_effect =
        acquire ? IREE_HAL_MEMORY_EFFECT_GLOBAL_ACQUIRE_FROM_SYSTEM
                : IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM;
    const iree_hal_memory_transition_recipe_info_t operation = {
        .kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE,
        .executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE,
        .operation =
            acquire ? IREE_HAL_MEMORY_TRANSITION_OPERATION_ACQUIRE_FROM_SYSTEM
                    : IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM,
        .range_granularity = 64,
    };
    const iree_hal_memory_transition_recipe_t recipe = {
        .effects = {range_effect},
        .operation_count = 1,
        .operations = &operation,
    };
    const iree_hal_buffer_barrier_t buffer = {.recipe = &recipe};
    EXPECT_EQ(Prepare({
                  .effects = {range_effect},
                  .buffer_barrier_count = 1,
                  .buffer_barriers = &buffer,
              }),
              Prepare({.effects = {global_effect}}));
  }
}

TEST(AqlBarrierTest, CombiningAgentAndSystemUsesScopeMaximum) {
  const iree_hal_memory_barrier_t memory = {
      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      IREE_HAL_ACCESS_SCOPE_DISPATCH_READ};
  const iree_hal_barrier_t records[] = {
      {.flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE},
      {.memory_barrier_count = 1, .memory_barriers = &memory},
      {.flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE},
  };
  const iree_hal_barrier_list_t list = {IREE_ARRAYSIZE(records), records};
  // Bitwise OR would produce reserved scope 3 when agent 1 and system 2 meet.
  EXPECT_EQ(iree_hal_amd_gpu_aql_barrier_prepare(&list),
            IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
                IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM);
}

TEST(AqlBarrierTest, PreparedJoinPreservesValidScopes) {
  for (uint32_t a = 0; a < 3; ++a) {
    for (uint32_t b = 0; b < 3; ++b) {
      const uint32_t first = (a << 9) | (b << 11);
      const uint32_t second =
          IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED | (b << 9) | (a << 11);
      const uint32_t maximum = std::max(a, b);
      EXPECT_EQ(iree_hal_amd_gpu_aql_barrier_join(first, second),
                IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED | (maximum << 9) |
                    (maximum << 11));
    }
  }
}

}  // namespace
