// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <string>
#include <thread>
#include <tuple>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/pool_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/amd/gpu/cts/test_device.h"
#include "iree/hal/driver/amd/gpu/cts/testdata.h"
#include "iree/hal/driver/amd/gpu/image/cooperative_images.h"
#include "iree/hal/driver/amd/gpu/queue/aql/queue.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/queue.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::cts {
namespace {

using iree::hal::cts::Ref;

enum class GridBarrier { kCombined, kSplit };

class GpuCooperativeTest
    : public GpuNativeTest,
      public ::testing::WithParamInterface<
          std::tuple<amdf_queue_command_type_t, GridBarrier>> {
 protected:
  amdf_queue_command_type_t command_type() const override {
    return std::get<0>(GetParam());
  }
};

TEST_P(GpuCooperativeTest, QueueStateSurvivesMixedReplaysAndQueueReplacement) {
  options_.queue.ring_byte_length = 4096;
  options_.queue.retirement_capacity = 4;
  Create();
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }
  const auto* family = iree_hal_queue_family(queues_[0]);
  if (!(iree_hal_queue_family_spec(family)->supported_queue_features &
        IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH)) {
    GTEST_SKIP() << "Native family has no cooperative launch support";
  }

  iree_hal_queue_params_t params;
  iree_hal_queue_params_initialize(&params);
  params.features = IREE_HAL_QUEUE_FEATURE_FLAG_COOPERATIVE_DISPATCH;
  Ref<iree_hal_queue_t> queue;
  IREE_ASSERT_OK(iree_hal_queue_acquire(family, &params, queue.out()));
  {
    Ref<iree_hal_queue_t> alias;
    IREE_ASSERT_OK(iree_hal_device_trim(device_));
    IREE_ASSERT_OK(iree_hal_queue_acquire(family, &params, alias.out()));
    EXPECT_EQ(queue.get(), alias.get());
  }
  const auto alternate_placement =
      iree_hal_queue_placement(queue) == IREE_HAL_QUEUE_PLACEMENT_HOST_LOCAL
          ? IREE_HAL_QUEUE_PLACEMENT_DEVICE_LOCAL
          : IREE_HAL_QUEUE_PLACEMENT_HOST_LOCAL;
  if (iree_hal_queue_family_spec(family)->placement_capabilities &
      (1u << alternate_placement)) {
    params.placement = alternate_placement;
    Ref<iree_hal_queue_t> conflicting;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_UNAVAILABLE,
        iree_hal_queue_acquire(family, &params, conflicting.out()));
  }
  // No work has been submitted: trim must return the sole cached admission,
  // including its immutable placement, without draining unrelated queues.
  queue.reset();
  IREE_ASSERT_OK(iree_hal_device_trim(device_));
  IREE_ASSERT_OK(iree_hal_queue_acquire(family, &params, queue.out()));
  if (params.placement != IREE_HAL_QUEUE_PLACEMENT_DEFAULT) {
    EXPECT_EQ(iree_hal_queue_placement(queue), params.placement);
  }
  const auto* target =
      &iree_hal_device_spec_executables(iree_hal_device_spec(device_))
           ->targets[0];
  iree_hal_executable_load_params_t load;
  iree_hal_executable_load_params_initialize(&load);
  IREE_ASSERT_OK(LookupCodeObject(iree_hal_amd_gpu_cooperative_images_create(),
                                  iree_hal_amd_gpu_cooperative_images_size(),
                                  target->target_key, &load.executable_data));
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(
      iree_hal_executable_load(family, target, &load, executable.out()));
  iree_hal_executable_function_t function;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable,
      std::get<1>(GetParam()) == GridBarrier::kSplit
          ? IREE_SV("hrx_cooperative_grid_sync_split")
          : IREE_SV("hrx_cooperative_grid_sync"),
      &function));

  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(iree::hal::cts::CreateQueueSlabPool(
      group_, family, IREE_HAL_BUFFER_USAGE_STORAGE,
      {IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
       IREE_HAL_MAPPING_MODE_SCOPED},
      iree_allocator_system(), pool.out()));
  constexpr uint32_t kGroups = 4;
  constexpr uint32_t kReplays = 3;
  constexpr uint32_t kBytes = kGroups * sizeof(uint32_t);
  Ref<iree_hal_buffer_t> scratch, output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(kBytes, 4),
      iree_infinite_timeout(), scratch.out()));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(kBytes * kReplays, 4),
      iree_infinite_timeout(), output.out()));
  Ref<iree_hal_semaphore_t> timeline;
  CreateSemaphore(timeline.out());
  if (HasFatalFailure()) {
    return;
  }
  auto* semaphore = timeline.get();
  uint64_t completed = 0;
  const std::array<uint32_t, 2> constants = {17, kGroups};
  const uint32_t sum = 17 * kGroups + kGroups * (kGroups - 1) / 2;
  auto config = iree_hal_make_static_dispatch_config(kGroups, 1, 1);
  config.workgroup_size[0] = 64;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  iree_hal_queue_dispatch_concurrency_t concurrency;
  IREE_ASSERT_OK(iree_hal_queue_query_dispatch_concurrency(
      queue, executable, function, {{64, 1, 1}, 0}, 0, &concurrency));
  auto oversized = config;
  oversized.workgroup_count[0] = (uint32_t)iree_min(
      iree_hal_queue_dispatch_concurrency_total_workgroup_count(concurrency) +
          1,
      uint64_t{UINT16_MAX} + 1);
  const iree_hal_buffer_ref_t direct_bindings[] = {
      iree_hal_make_buffer_ref(scratch, 0, kBytes),
      iree_hal_make_buffer_ref(output, 0, kBytes),
  };
  auto empty = config;
  empty.workgroup_count[0] = 0;
  uint64_t empty_done = 1;
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queue, {}, {1, &semaphore, &empty_done}, executable, function, empty,
      iree_make_const_byte_span(constants.data(), sizeof(constants)),
      {IREE_ARRAYSIZE(direct_bindings), direct_bindings}, nullptr,
      IREE_HAL_DISPATCH_FLAG_COOPERATIVE));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_queue_dispatch(
          queue, {}, {}, executable, function, oversized,
          iree_make_const_byte_span(constants.data(), sizeof(constants)),
          {IREE_ARRAYSIZE(direct_bindings), direct_bindings}, nullptr,
          IREE_HAL_DISPATCH_FLAG_COOPERATIVE));
  std::array<uint32_t, kGroups> direct_values = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(output, 0, direct_values.data(),
                                           sizeof(direct_values)));
  uint64_t direct_done = 2;
  IREE_ASSERT_OK(iree_hal_queue_dispatch(
      queue, {}, {1, &semaphore, &direct_done}, executable, function, config,
      iree_make_const_byte_span(constants.data(), sizeof(constants)),
      {IREE_ARRAYSIZE(direct_bindings), direct_bindings}, nullptr,
      IREE_HAL_DISPATCH_FLAG_COOPERATIVE));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, direct_done,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_buffer_map_read(output, 0, direct_values.data(),
                                          sizeof(direct_values)));
  for (auto value : direct_values) {
    EXPECT_EQ(value, sum);
  }
  completed = direct_done;
  // Queue different grid extents without a host completion between them. The
  // shared direct-dispatch barrier record must be initialized in device order,
  // including when the caller supplies no optional payload cache operations.
  std::array<uint32_t, kGroups * kReplays> queued_values = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(output, 0, queued_values.data(),
                                           sizeof(queued_values)));
  const iree_hal_queue_barriers_t no_payload_barriers = {};
  for (uint32_t replay = 0; replay < kReplays; ++replay) {
    const uint32_t groups = kGroups - replay;
    auto queued_config = config;
    queued_config.workgroup_count[0] = groups;
    const std::array<uint32_t, 2> queued_constants = {17 + replay, groups};
    const iree_hal_buffer_ref_t queued_bindings[] = {
        iree_hal_make_buffer_ref(scratch, 0, kBytes),
        iree_hal_make_buffer_ref(output, replay * kBytes, kBytes),
    };
    uint64_t next = completed + 1;
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        queue, {1, &semaphore, &completed}, {1, &semaphore, &next}, executable,
        function, queued_config,
        iree_make_const_byte_span(queued_constants.data(),
                                  sizeof(queued_constants)),
        {IREE_ARRAYSIZE(queued_bindings), queued_bindings},
        replay == 1 ? &no_payload_barriers : nullptr,
        IREE_HAL_DISPATCH_FLAG_COOPERATIVE));
    completed = next;
  }
  IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, completed,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_buffer_map_read(output, 0, queued_values.data(),
                                          sizeof(queued_values)));
  for (uint32_t replay = 0; replay < kReplays; ++replay) {
    const uint32_t groups = kGroups - replay;
    const uint32_t expected =
        (17 + replay) * groups + groups * (groups - 1) / 2;
    for (uint32_t group = 0; group < kGroups; ++group) {
      EXPECT_EQ(queued_values[replay * kGroups + group],
                group < groups ? expected : 0u);
    }
  }
  const iree_hal_buffer_ref_t bindings[] = {
      iree_hal_make_indirect_buffer_ref(0, 0, kBytes),
      iree_hal_make_indirect_buffer_ref(1, 0, kBytes),
  };
  const iree_hal_barrier_t acquire = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  };
  const iree_hal_memory_barrier_t memory = {
      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
          IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
  };
  const iree_hal_barrier_t dependency = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .memory_barrier_count = 1,
      .memory_barriers = &memory,
  };
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  // Exercise bounded host preparation, resident storage, and multiple native
  // blocks. Direct and recorded launches share state and change geometry on
  // the same queue without an intervening CPU completion wait.
  for (uint32_t dispatches : {1u, 8u, 129u}) {
    SCOPED_TRACE(dispatches);
    std::array<uint32_t, kGroups * kReplays> values = {};
    IREE_ASSERT_OK(
        iree_hal_buffer_map_write(output, 0, values.data(), sizeof(values)));
    Ref<iree_hal_command_buffer_t> recording;
    IREE_ASSERT_OK(iree_hal_command_buffer_create(
        family, 0, IREE_HAL_COMMAND_CATEGORY_DISPATCH, 2, recording.out()));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &acquire));
    for (uint32_t i = 0; i < dispatches; ++i) {
      if (i) {
        IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &dependency));
      }
      IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
          recording, executable, function, config,
          iree_make_const_byte_span(constants.data(), sizeof(constants)),
          {IREE_ARRAYSIZE(bindings), bindings},
          IREE_HAL_DISPATCH_FLAG_COOPERATIVE));
    }
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
    IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
    std::array<iree_hal_buffer_binding_t, 2> table = {
        iree_hal_buffer_binding_t{scratch, 0, kBytes},
        iree_hal_buffer_binding_t{output, 0, kBytes},
    };
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_FAILED_PRECONDITION,
        iree_hal_queue_execute(queues_[0], {}, {}, recording,
                               {table.size(), table.data()}, 0));
    // A deferred invocation must release its captures without touching the
    // shared state or output when its dependency fails. The same recording
    // and queue remain usable for the successful replays below.
    Ref<iree_hal_semaphore_t> dependency_gate, rejected_completion;
    CreateSemaphore(dependency_gate.out());
    CreateSemaphore(rejected_completion.out());
    if (HasFatalFailure()) {
      return;
    }
    auto* dependency_semaphore = dependency_gate.get();
    auto* rejected_semaphore = rejected_completion.get();
    uint64_t rejected_value = 1;
    IREE_ASSERT_OK(iree_hal_queue_execute(
        queue, {1, &dependency_semaphore, &rejected_value},
        {1, &rejected_semaphore, &rejected_value}, recording,
        {table.size(), table.data()}, 0));
    iree_hal_semaphore_fail(dependency_semaphore,
                            iree_status_from_code(IREE_STATUS_ABORTED));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_ABORTED,
        iree_hal_semaphore_wait(rejected_semaphore, rejected_value,
                                iree_infinite_timeout(),
                                IREE_ASYNC_WAIT_FLAG_NONE));
    for (uint32_t replay = 0; replay < kReplays; ++replay) {
      table[1].offset = replay * kBytes;
      uint64_t next = completed + 1;
      IREE_ASSERT_OK(iree_hal_queue_execute(
          queue, {completed ? 1u : 0u, &semaphore, &completed},
          {1, &semaphore, &next}, recording, {table.size(), table.data()}, 0));
      completed = next;
      const uint32_t groups = kGroups - replay;
      auto mixed_config = config;
      mixed_config.workgroup_count[0] = groups;
      const std::array<uint32_t, 2> mixed_constants = {17 + replay, groups};
      const iree_hal_buffer_ref_t mixed_bindings[] = {
          iree_hal_make_buffer_ref(scratch, 0, kBytes),
          iree_hal_make_buffer_ref(output, replay * kBytes, kBytes),
      };
      next = completed + 1;
      IREE_ASSERT_OK(iree_hal_queue_dispatch(
          queue, {1, &semaphore, &completed}, {1, &semaphore, &next},
          executable, function, mixed_config,
          iree_make_const_byte_span(mixed_constants.data(),
                                    sizeof(mixed_constants)),
          {IREE_ARRAYSIZE(mixed_bindings), mixed_bindings}, nullptr,
          IREE_HAL_DISPATCH_FLAG_COOPERATIVE));
      completed = next;
    }
    IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, completed,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
    IREE_ASSERT_OK(
        iree_hal_buffer_map_read(output, 0, values.data(), sizeof(values)));
    for (uint32_t replay = 0; replay < kReplays; ++replay) {
      const uint32_t groups = kGroups - replay;
      const uint32_t extra = (17 + replay) * groups + groups * (groups - 1) / 2;
      for (uint32_t group = 0; group < kGroups; ++group) {
        EXPECT_EQ(values[replay * kGroups + group],
                  dispatches * sum + (group < groups ? extra : 0u));
      }
    }

    // The recording survives native queue teardown. Changing an available
    // placement requires a new cached queue and catches a retained old owner.
    const auto replacement_placement =
        iree_hal_queue_placement(queue) == IREE_HAL_QUEUE_PLACEMENT_HOST_LOCAL
            ? IREE_HAL_QUEUE_PLACEMENT_DEVICE_LOCAL
            : IREE_HAL_QUEUE_PLACEMENT_HOST_LOCAL;
    if (iree_hal_queue_family_spec(family)->placement_capabilities &
        (1u << replacement_placement)) {
      params.placement = replacement_placement;
    }
    // Public completion ends payload borrows before the worker's final
    // retirement bookkeeping. Trim can replace the queue only after that
    // separate idle predicate holds.
    const auto is_idle = command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
                             ? iree_hal_amd_gpu_pm4_queue_is_idle
                             : iree_hal_amd_gpu_aql_queue_is_idle;
    while (!is_idle(queue)) {
      std::this_thread::yield();
    }
    queue.reset();
    IREE_ASSERT_OK(iree_hal_device_trim(device_));
    IREE_ASSERT_OK(iree_hal_queue_acquire(family, &params, queue.out()));
    if (params.placement != IREE_HAL_QUEUE_PLACEMENT_DEFAULT) {
      EXPECT_EQ(iree_hal_queue_placement(queue), params.placement);
    }
    table[1].offset = 0;
    uint64_t next = completed + 1;
    IREE_ASSERT_OK(iree_hal_queue_execute(queue, {1, &semaphore, &completed},
                                          {1, &semaphore, &next}, recording,
                                          {table.size(), table.data()}, 0));
    completed = next;
    IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, completed,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_buffer_map_read(output, 0, direct_values.data(),
                                            sizeof(direct_values)));
    for (uint32_t group = 0; group < kGroups; ++group) {
      EXPECT_EQ(direct_values[group], values[group] + dispatches * sum);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Protocol, GpuCooperativeTest,
    ::testing::Combine(::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                         AMDF_QUEUE_COMMAND_TYPE_GPU_AQL),
                       ::testing::Values(GridBarrier::kCombined,
                                         GridBarrier::kSplit)),
    [](const auto& info) {
      return std::string(std::get<0>(info.param) ==
                                 AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
                             ? "Pm4"
                             : "Aql") +
             (std::get<1>(info.param) == GridBarrier::kSplit ? "Split"
                                                             : "Combined");
    });

}  // namespace
}  // namespace iree::hal::amd::gpu::cts
