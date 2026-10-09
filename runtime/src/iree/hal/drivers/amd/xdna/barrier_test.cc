// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/xdna/barrier.h"

#include "iree/hal/command_buffer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::xdna {
namespace {

TEST(BarrierTest, AcceptsInherentVisibilityBoundaries) {
  IREE_EXPECT_OK(iree_hal_amd_xdna_queue_barriers_validate(nullptr));

  iree_hal_queue_barriers_t barriers = {};
  IREE_EXPECT_OK(iree_hal_amd_xdna_queue_barriers_validate(&barriers));

  const iree_hal_barrier_list_t empty = {};
  barriers = {&empty, &empty};
  IREE_EXPECT_OK(iree_hal_amd_xdna_queue_barriers_validate(&barriers));

  iree_hal_barrier_t global = {};
  global.source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH;
  global.target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE;
  global.flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE;
  global.effects.bits = IREE_HAL_MEMORY_EFFECT_GLOBAL_RELEASE_TO_SYSTEM;
  const iree_hal_barrier_list_t global_list = {1, &global};
  barriers = {&global_list, nullptr};
  IREE_EXPECT_OK(iree_hal_amd_xdna_queue_barriers_validate(&barriers));
}

TEST(BarrierTest, RejectsRangedQueueOperations) {
  const iree_hal_memory_transition_recipe_info_t operation = {
      /*.kind=*/IREE_HAL_MEMORY_TRANSITION_KIND_RANGE,
      /*.executor=*/IREE_HAL_MEMORY_TRANSITION_EXECUTOR_QUEUE,
      /*.operation=*/IREE_HAL_MEMORY_TRANSITION_OPERATION_RELEASE_TO_SYSTEM,
      /*.range_granularity=*/64,
  };
  const iree_hal_memory_transition_recipe_t recipe = {
      /*.effects=*/{IREE_HAL_MEMORY_EFFECT_RANGE_RELEASE_TO_SYSTEM},
      /*.operation_count=*/1,
      /*.operations=*/&operation,
  };
  const iree_hal_buffer_barrier_t range = {
      /*.source_scope=*/IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      /*.target_scope=*/IREE_HAL_ACCESS_SCOPE_MEMORY_READ,
      /*.buffer_ref=*/{},
      /*.recipe=*/&recipe,
  };
  const iree_hal_barrier_t barrier = {
      /*.source_stage_mask=*/IREE_HAL_EXECUTION_STAGE_DISPATCH,
      /*.target_stage_mask=*/IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      /*.flags=*/IREE_HAL_BARRIER_FLAG_NONE,
      /*.effects=*/recipe.effects,
      /*.memory_barrier_count=*/0,
      /*.memory_barriers=*/nullptr,
      /*.buffer_barrier_count=*/1,
      /*.buffer_barriers=*/&range,
  };
  IREE_EXPECT_OK(iree_hal_barrier_validate(&barrier));
  const iree_hal_barrier_list_t list = {1, &barrier};

  iree_hal_queue_barriers_t barriers = {&list, nullptr};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_amd_xdna_queue_barriers_validate(&barriers));
  barriers = {nullptr, &list};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNIMPLEMENTED,
                        iree_hal_amd_xdna_queue_barriers_validate(&barriers));
}

}  // namespace
}  // namespace iree::hal::amd::xdna
