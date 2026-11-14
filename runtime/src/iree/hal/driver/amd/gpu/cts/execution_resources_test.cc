// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/pool_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/amd/gpu/cts/test_device.h"
#include "iree/hal/driver/amd/gpu/cts/testdata.h"
#include "iree/hal/driver/amd/gpu/image/hip_images.h"
#include "iree/hal/driver/amd/gpu/queue/aql/command_buffer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::cts {
namespace {

using iree::hal::cts::Ref;

class GpuExecutionResourcesTest
    : public GpuNativeTest,
      public ::testing::WithParamInterface<amdf_queue_command_type_t> {
 protected:
  amdf_queue_command_type_t command_type() const override { return GetParam(); }
};

TEST_P(GpuExecutionResourcesTest, SelectedQueuesExecuteDirectAndResidentWork) {
  Create();
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }
  const auto* family = iree_hal_queue_family(queues_[0]);
  const auto* spec = iree_hal_queue_family_spec(family);
  std::array<std::vector<iree_hal_queue_execution_resource_ordinal_t>, 2>
      selections;
  for (uint32_t group = 0; group < spec->execution_resource_group_count;
       ++group) {
    const uint32_t minimum = std::max(
        1u,
        spec->execution_resource_groups[group].minimum_selected_resource_count);
    uint32_t available = 0;
    for (uint32_t i = 0; i < spec->execution_resource_count; ++i) {
      if (spec->execution_resources[i].group_ordinal != group) {
        continue;
      }
      if (available < 2 * minimum) {
        selections[available / minimum].push_back(i);
      }
      ++available;
    }
    if (available < 2 * minimum) {
      GTEST_SKIP() << "family cannot provide two disjoint resource selections";
    }
  }
  std::array<Ref<iree_hal_queue_t>, 2> selected_queues;
  uint32_t selected_units = 0;
  for (uint32_t slot = 0; slot < selections.size(); ++slot) {
    auto& selection = selections[slot];
    std::sort(selection.begin(), selection.end());
    iree_hal_queue_params_t params;
    iree_hal_queue_params_initialize(&params);
    params.execution_resources = {selection.size(), selection.data()};
    IREE_ASSERT_OK(
        iree_hal_queue_acquire(family, &params, selected_queues[slot].out()));
    const auto retained =
        iree_hal_queue_execution_resources(selected_queues[slot]);
    ASSERT_EQ(retained.count, selection.size());
    EXPECT_TRUE(
        std::equal(selection.begin(), selection.end(), retained.ordinals));
    if (slot == 0) {
      for (auto ordinal : selection) {
        selected_units +=
            spec->execution_resources[ordinal].execution_unit_count;
      }
    }
  }
  EXPECT_NE(selected_queues[0].get(), selected_queues[1].get());

  const auto* target =
      &iree_hal_device_spec_executables(iree_hal_device_spec(device_))
           ->targets[0];
  iree_hal_executable_load_params_t load;
  iree_hal_executable_load_params_initialize(&load);
  IREE_ASSERT_OK(LookupCodeObject(iree_hal_amd_gpu_hip_images_create(),
                                  iree_hal_amd_gpu_hip_images_size(),
                                  target->target_key, &load.executable_data));
  Ref<iree_hal_executable_t> executable;
  IREE_ASSERT_OK(
      iree_hal_executable_load(family, target, &load, executable.out()));
  iree_hal_executable_function_t function;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable, IREE_SV("image_accumulate"), &function));
  const iree_hal_queue_dispatch_concurrency_params_t concurrency_params = {
      .workgroup_size = {64, 1, 1},
  };
  iree_hal_queue_dispatch_concurrency_t full;
  IREE_ASSERT_OK(iree_hal_queue_query_dispatch_concurrency(
      queues_[0], executable, function, concurrency_params, 0, &full));
  for (auto& queue : selected_queues) {
    iree_hal_queue_dispatch_concurrency_t selected;
    IREE_ASSERT_OK(iree_hal_queue_query_dispatch_concurrency(
        queue, executable, function, concurrency_params, 0, &selected));
    EXPECT_GT(selected.scheduling_domain_count, 0u);
    EXPECT_EQ(selected.maximum_concurrent_workgroup_count_per_domain,
              full.maximum_concurrent_workgroup_count_per_domain);
    EXPECT_EQ(
        uint64_t{selected.scheduling_domain_count} * spec->execution_unit_count,
        uint64_t{full.scheduling_domain_count} * selected_units);
  }

  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(iree::hal::cts::CreateQueueSlabPool(
      group_, family,
      IREE_HAL_BUFFER_USAGE_STORAGE |
          IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
      {IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
       IREE_HAL_MAPPING_MODE_SCOPED},
      iree_allocator_system(), pool.out()));
  Ref<iree_hal_buffer_t> output;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(128, 64),
      iree_infinite_timeout(), output.out()));
  std::array<uint32_t, 32> values = {};
  IREE_ASSERT_OK(
      iree_hal_buffer_map_write(output, 0, values.data(), sizeof(values)));
  Ref<iree_hal_buffer_t> counts;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(12, 4),
      iree_infinite_timeout(), counts.out()));
  const std::array<uint32_t, 3> workgroup_count = {1, 1, 1};
  IREE_ASSERT_OK(iree_hal_buffer_map_write(counts, 0, workgroup_count.data(),
                                           sizeof(workgroup_count)));

  Ref<iree_hal_command_buffer_t> recording;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      family, 0, IREE_HAL_COMMAND_CATEGORY_DISPATCH, 1, recording.out()));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording));
  const iree_hal_barrier_t acquire = {
      .source_stage_mask =
          IREE_HAL_EXECUTION_STAGE_HOST | IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &acquire));
  const iree_hal_memory_barrier_t memory = {
      IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE,
      IREE_HAL_ACCESS_SCOPE_DISPATCH_READ |
          IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE};
  const iree_hal_barrier_t dependency = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .memory_barrier_count = 1,
      .memory_barriers = &memory,
  };
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 64;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  const uint32_t increment = 7;
  const auto constants =
      iree_make_const_byte_span(&increment, sizeof(increment));
  const auto indirect =
      iree_hal_make_indirect_buffer_ref(0, 0, sizeof(uint32_t));
  // Device-supplied geometry exercises native preparation and publication on
  // the selected resources independently of the recording's size.
  auto indirect_config = config;
  indirect_config.workgroup_count_ref = iree_hal_make_buffer_ref(counts, 0, 12);
  constexpr uint32_t kDispatchCount = 64;
  for (uint32_t i = 0; i < kDispatchCount; ++i) {
    if (i) {
      IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &dependency));
    }
    IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
        recording, executable, function, indirect_config, constants,
        {1, &indirect}, IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS));
  }
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_DISPATCH,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      .flags = IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording, &release));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording));
  if (command_type() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL) {
    const auto* program =
        iree_hal_amd_gpu_aql_command_buffer_program(recording);
    ASSERT_NE(program->resident.pool, nullptr);
  }

  for (uint32_t slot = 0; slot < selected_queues.size(); ++slot) {
    Ref<iree_hal_semaphore_t> timeline;
    CreateSemaphore(timeline.out());
    if (HasFatalFailure()) {
      return;
    }
    auto* semaphore = timeline.get();
    uint64_t direct_done = 1;
    uint64_t recorded_done = 2;
    const auto binding =
        iree_hal_make_buffer_ref(output, slot * 64, sizeof(uint32_t));
    IREE_ASSERT_OK(iree_hal_queue_dispatch(
        selected_queues[slot], {}, {1, &semaphore, &direct_done}, executable,
        function, config, constants, {1, &binding}, nullptr, 0));
    const iree_hal_buffer_binding_t table = {output, slot * 64,
                                             sizeof(uint32_t)};
    IREE_ASSERT_OK(iree_hal_queue_execute(
        selected_queues[slot], {1, &semaphore, &direct_done},
        {1, &semaphore, &recorded_done}, recording, {1, &table}, 0));
    IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, recorded_done,
                                           iree_infinite_timeout(),
                                           IREE_ASYNC_WAIT_FLAG_NONE));
  }
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(output, 0, values.data(), sizeof(values)));
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(values[i],
              i == 0 || i == 16 ? (kDispatchCount + 1) * increment : 0u);
  }
}

INSTANTIATE_TEST_SUITE_P(Protocol, GpuExecutionResourcesTest,
                         ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                           AMDF_QUEUE_COMMAND_TYPE_GPU_AQL),
                         [](const auto& info) {
                           return info.param == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                                      ? "Aql"
                                      : "Pm4";
                         });

}  // namespace
}  // namespace iree::hal::amd::gpu::cts
