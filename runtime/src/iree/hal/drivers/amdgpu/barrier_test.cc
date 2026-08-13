// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/barrier.h"

#include "iree/testing/gtest.h"

namespace iree::hal::amdgpu {
namespace {

TEST(BarrierTest, ExecutionOnlyHasNoFenceScope) {
  const iree_hal_amdgpu_barrier_scopes_t scopes =
      iree_hal_amdgpu_barrier_resolve_scopes(
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
          IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE,
          IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
          /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
          /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr);
  EXPECT_EQ(scopes.acquire, IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_EQ(scopes.release, IREE_HSA_FENCE_SCOPE_NONE);
}

TEST(BarrierTest, GenericMemoryScopesUseAgentFences) {
  const iree_hal_memory_barrier_t memory_barrier = {
      /*.source_scope=*/IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE,
      /*.target_scope=*/IREE_HAL_ACCESS_SCOPE_MEMORY_READ,
  };
  const iree_hal_amdgpu_barrier_scopes_t scopes =
      iree_hal_amdgpu_barrier_resolve_scopes(
          IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_STAGE_DISPATCH,
          IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
          /*memory_barrier_count=*/1, &memory_barrier,
          /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr);
  EXPECT_EQ(scopes.acquire, IREE_HSA_FENCE_SCOPE_AGENT);
  EXPECT_EQ(scopes.release, IREE_HSA_FENCE_SCOPE_AGENT);
}

TEST(BarrierTest, SystemFlagsWidenScopesIndependently) {
  const iree_hal_memory_barrier_t memory_barrier = {
      /*.source_scope=*/IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      /*.target_scope=*/IREE_HAL_ACCESS_SCOPE_DISPATCH_READ,
  };
  iree_hal_amdgpu_barrier_scopes_t scopes =
      iree_hal_amdgpu_barrier_resolve_scopes(
          IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_STAGE_DISPATCH,
          IREE_HAL_EXECUTION_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
          /*memory_barrier_count=*/1, &memory_barrier,
          /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr);
  EXPECT_EQ(scopes.acquire, IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(scopes.release, IREE_HSA_FENCE_SCOPE_AGENT);

  scopes = iree_hal_amdgpu_barrier_resolve_scopes(
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_EXECUTION_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
      /*memory_barrier_count=*/1, &memory_barrier,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr);
  EXPECT_EQ(scopes.acquire, IREE_HSA_FENCE_SCOPE_AGENT);
  EXPECT_EQ(scopes.release, IREE_HSA_FENCE_SCOPE_SYSTEM);
}

TEST(BarrierTest, HostStagesImplySystemScopes) {
  iree_hal_amdgpu_barrier_scopes_t scopes =
      iree_hal_amdgpu_barrier_resolve_scopes(
          IREE_HAL_EXECUTION_STAGE_HOST, IREE_HAL_EXECUTION_STAGE_DISPATCH,
          IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
          /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
          /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr);
  EXPECT_EQ(scopes.acquire, IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(scopes.release, IREE_HSA_FENCE_SCOPE_NONE);

  scopes = iree_hal_amdgpu_barrier_resolve_scopes(
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_STAGE_HOST,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr);
  EXPECT_EQ(scopes.acquire, IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_EQ(scopes.release, IREE_HSA_FENCE_SCOPE_SYSTEM);
}

TEST(BarrierTest, AtomicOrderingSelectsHandoffScope) {
  EXPECT_EQ(
      iree_hal_amdgpu_barrier_resolve_atomic_handoff_scope(
          IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
          IREE_HAL_ATOMIC_FLAG_ACQUIRE),
      IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_EQ(iree_hal_amdgpu_barrier_resolve_atomic_handoff_scope(
                IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_ATOMIC_FLAG_ACQUIRE,
                IREE_HAL_ATOMIC_FLAG_ACQUIRE),
            IREE_HSA_FENCE_SCOPE_AGENT);
  EXPECT_EQ(
      iree_hal_amdgpu_barrier_resolve_atomic_handoff_scope(
          IREE_HAL_EXECUTION_STAGE_DISPATCH,
          IREE_HAL_ATOMIC_FLAG_ACQUIRE | IREE_HAL_ATOMIC_FLAG_SYSTEM_SCOPE,
          IREE_HAL_ATOMIC_FLAG_ACQUIRE),
      IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(iree_hal_amdgpu_barrier_resolve_atomic_handoff_scope(
                IREE_HAL_EXECUTION_STAGE_HOST, IREE_HAL_ATOMIC_FLAG_RELEASE,
                IREE_HAL_ATOMIC_FLAG_RELEASE),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
}

}  // namespace
}  // namespace iree::hal::amdgpu
