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
          IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE, IREE_HAL_BARRIER_FLAG_NONE,
          /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
          /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr);
  EXPECT_EQ(scopes.acquire, IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_EQ(scopes.release, IREE_HSA_FENCE_SCOPE_NONE);
}

TEST(BarrierTest, GenericMemoryScopesUseAgentFences) {
  const iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_MEMORY_READ,
  };
  const iree_hal_amdgpu_barrier_scopes_t scopes =
      iree_hal_amdgpu_barrier_resolve_scopes(
          IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_STAGE_DISPATCH,
          IREE_HAL_BARRIER_FLAG_NONE,
          /*memory_barrier_count=*/1, &memory_barrier,
          /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr);
  EXPECT_EQ(scopes.acquire, IREE_HSA_FENCE_SCOPE_AGENT);
  EXPECT_EQ(scopes.release, IREE_HSA_FENCE_SCOPE_AGENT);
}

TEST(BarrierTest, SystemFlagsWidenScopesIndependently) {
  const iree_hal_memory_barrier_t memory_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ,
  };
  iree_hal_amdgpu_barrier_scopes_t scopes =
      iree_hal_amdgpu_barrier_resolve_scopes(
          IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_STAGE_DISPATCH,
          IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
          /*memory_barrier_count=*/1, &memory_barrier,
          /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr);
  EXPECT_EQ(scopes.acquire, IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(scopes.release, IREE_HSA_FENCE_SCOPE_AGENT);

  scopes = iree_hal_amdgpu_barrier_resolve_scopes(
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
      /*memory_barrier_count=*/1, &memory_barrier,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr);
  EXPECT_EQ(scopes.acquire, IREE_HSA_FENCE_SCOPE_AGENT);
  EXPECT_EQ(scopes.release, IREE_HSA_FENCE_SCOPE_SYSTEM);
}

TEST(BarrierTest, RangedRecipesWidenSelectedScope) {
  iree_hal_memory_transition_recipe_info_t operation = {
      .kind = IREE_HAL_MEMORY_TRANSITION_KIND_RANGE,
      .executor = IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE,
      .operation = IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM,
      .range_granularity = 64,
  };
  iree_hal_memory_transition_recipe_t recipe = {
      .effects = {IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM},
      .operation_count = 1,
      .operations = &operation,
  };
  iree_hal_buffer_barrier_t buffer_barrier = {
      .source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      .target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ,
      .buffer_ref = {},
      .recipe = &recipe,
  };
  iree_hal_amdgpu_barrier_scopes_t scopes =
      iree_hal_amdgpu_barrier_resolve_scopes(
          IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_STAGE_DISPATCH,
          IREE_HAL_BARRIER_FLAG_NONE,
          /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
          /*buffer_barrier_count=*/1, &buffer_barrier);
  EXPECT_EQ(scopes.acquire, IREE_HSA_FENCE_SCOPE_AGENT);
  EXPECT_EQ(scopes.release, IREE_HSA_FENCE_SCOPE_SYSTEM);

  operation.operation =
      IREE_HAL_MEMORY_TRANSITION_OPERATION_ACQUIRE_FROM_SYSTEM;
  recipe.effects.bits = IREE_HAL_MEMORY_EFFECT_RANGE_ACQUIRE_FROM_SYSTEM;
  scopes = iree_hal_amdgpu_barrier_resolve_scopes(
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_STAGE_DISPATCH,
      IREE_HAL_BARRIER_FLAG_NONE,
      /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
      /*buffer_barrier_count=*/1, &buffer_barrier);
  EXPECT_EQ(scopes.acquire, IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(scopes.release, IREE_HSA_FENCE_SCOPE_AGENT);
}

TEST(BarrierTest, HostStagesImplySystemScopes) {
  iree_hal_amdgpu_barrier_scopes_t scopes =
      iree_hal_amdgpu_barrier_resolve_scopes(
          IREE_HAL_EXECUTION_STAGE_HOST, IREE_HAL_EXECUTION_STAGE_DISPATCH,
          IREE_HAL_BARRIER_FLAG_NONE,
          /*memory_barrier_count=*/0, /*memory_barriers=*/nullptr,
          /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr);
  EXPECT_EQ(scopes.acquire, IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(scopes.release, IREE_HSA_FENCE_SCOPE_NONE);

  scopes = iree_hal_amdgpu_barrier_resolve_scopes(
      IREE_HAL_EXECUTION_STAGE_DISPATCH, IREE_HAL_EXECUTION_STAGE_HOST,
      IREE_HAL_BARRIER_FLAG_NONE,
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

TEST(BarrierTest, QueueBoundaryDefaultsCanBeReplacedIndependently) {
  auto native = iree_hal_amdgpu_queue_barriers_resolve(nullptr);
  EXPECT_EQ(native.before.acquire, IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(native.before.release, IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(native.after.acquire, IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(native.after.release, IREE_HSA_FENCE_SCOPE_SYSTEM);

  const iree_hal_barrier_list_t empty = {};
  iree_hal_queue_barriers_t barriers = {&empty, nullptr};
  native = iree_hal_amdgpu_queue_barriers_resolve(&barriers);
  EXPECT_EQ(native.before.acquire, IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_EQ(native.before.release, IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_EQ(native.after.acquire, IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(native.after.release, IREE_HSA_FENCE_SCOPE_SYSTEM);

  iree_hal_barrier_t release = {};
  release.effects.bits = IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM;
  const iree_hal_barrier_list_t release_list = {1, &release};
  barriers = {&release_list, &empty};
  native = iree_hal_amdgpu_queue_barriers_resolve(&barriers);
  EXPECT_EQ(native.before.acquire, IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_EQ(native.before.release, IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(native.after.acquire, IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_EQ(native.after.release, IREE_HSA_FENCE_SCOPE_NONE);

  iree_hal_barrier_t acquire = {
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  };
  const iree_hal_barrier_list_t acquire_list = {1, &acquire};
  barriers = {&empty, &acquire_list};
  native = iree_hal_amdgpu_queue_barriers_resolve(&barriers);
  EXPECT_EQ(native.before.acquire, IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_EQ(native.before.release, IREE_HSA_FENCE_SCOPE_NONE);
  EXPECT_EQ(native.after.acquire, IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(native.after.release, IREE_HSA_FENCE_SCOPE_NONE);
}

}  // namespace
}  // namespace iree::hal::amdgpu
