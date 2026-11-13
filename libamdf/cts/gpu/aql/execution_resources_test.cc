// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/kernels/execution_resources.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/execution_resources_kernels.h"

namespace {
namespace protocol = kernels::execution_resources;

TEST_F(AqlDispatchTest, DisjointSelectionsConfineEveryDispatch) {
  if ((family_.user_queue_capabilities &
       AMDF_USER_QUEUE_CAPABILITY_EXECUTION_RESOURCE_SELECTION) == 0) {
    GTEST_SKIP() << "queue family does not support resource selection";
  }
  const auto& geometry = gpu_endpoint_info_.execution_resources;
  ASSERT_GT(geometry.compute_units_per_resource, 0u);
  ASSERT_GT(geometry.group_count, 0u);
  const uint32_t selected_count =
      geometry.group_count *
      std::max(1u, geometry.minimum_selected_resources_per_group);
  const uint32_t resource_count =
      gpu_endpoint_info_.compute.compute_unit_count /
      geometry.compute_units_per_resource;
  if (resource_count < selected_count * 2) {
    GTEST_SKIP() << "endpoint cannot supply two disjoint legal selections";
  }
  const auto* kernel = protocol::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel, nullptr) << "missing compiled execution identity kernel";
  ASSERT_EQ(kernel->workgroup_size(), 1u);
  ASSERT_EQ(kernel->private_segment_byte_length, 0u);
  RecordProperty("execution_identity_target", kernel->target);
  RecordProperty("selected_resources_per_queue", selected_count);
  RecordProperty("execution_resource_groups", geometry.group_count);

  constexpr uint32_t kGroupCount = 1024;
  constexpr uint32_t kEpochCount = 2;
  constexpr uint32_t kArgumentStride = 64;
  std::array<GpuUserQueue*, 2> queues = {};
  std::array<GpuMemory*, 2> outputs = {};
  std::array<GpuMemory*, 2> arguments = {};
  std::array<GpuMemory*, 2> completions = {};
  std::array<uint64_t, 2> positions = {};
  std::array<std::set<uint32_t>, 2> observed;
  std::array<uint64_t, 2> descriptors = {};
  for (uint32_t slot = 0; slot < queues.size(); ++slot) {
    std::vector<uint32_t> ordinals(selected_count);
    for (uint32_t i = 0; i < selected_count; ++i) {
      ordinals[i] = slot * selected_count + i;
    }
    const amdf_gpu_queue_execution_resource_selection_t selection = {
        .type = AMDF_STRUCTURE_TYPE_GPU_QUEUE_EXECUTION_RESOURCE_SELECTION,
        .structure_size = sizeof(selection),
        .resource_count = selected_count,
        .resource_ordinals = ordinals.data(),
    };
    ASSERT_NO_FATAL_FAILURE(GpuCommandTest::CreateQueue(
        family_, GpuUserQueueOptions{.next = &selection}, &queues[slot]));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                     kGroupCount * sizeof(protocol::Record), &outputs[slot]));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments[slot]));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096,
                     &completions[slot]));
    ASSERT_EQ(arguments[slot]->device_address % kernel->arguments.alignment,
              0u);
    ASSERT_LE(kernel->arguments.byte_length, sizeof(protocol::Arguments));
    std::memset(arguments[slot]->host.pointer, 0,
                arguments[slot]->info.byte_length);
    std::memset(completions[slot]->host.pointer, 0,
                completions[slot]->info.byte_length);
    auto& signal = *static_cast<aql::Signal*>(completions[slot]->host.pointer);
    signal = aql::UserSignal();
    ASSERT_NO_FATAL_FAILURE(
        PublishKernel(*queues[slot], *kernel, slot == 0 ? "first" : "second",
                      &positions[slot], &descriptors[slot]));
  }

  for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
    for (uint32_t slot = 0; slot < queues.size(); ++slot) {
      const uint32_t seed = 0x751a0000u + epoch * 2 + slot;
      const protocol::Arguments payload = {outputs[slot]->device_address, seed};
      std::memset(outputs[slot]->host.pointer, 0xff,
                  outputs[slot]->info.byte_length);
      std::memcpy(static_cast<uint8_t*>(arguments[slot]->host.pointer) +
                      epoch * kArgumentStride,
                  &payload, kernel->arguments.byte_length);
      auto& signal =
          *static_cast<aql::Signal*>(completions[slot]->host.pointer);
      signal.value = 1;
      const auto packet = aql::Dispatch(
          aql::HeaderBarrier::kDisabled, {1, {1, 1, 1}, {kGroupCount, 1, 1}},
          kernel->private_segment_byte_length,
          kernel->group_segment_byte_length, descriptors[slot],
          arguments[slot]->device_address + epoch * kArgumentStride,
          completions[slot]->device_address);
      GpuStoreRelease(queues[slot]->host.write_index_address,
                      positions[slot] + 1);
      ASSERT_TRUE(aql::Publish(*queues[slot], positions[slot]++, packet));
    }
    for (uint32_t slot = 0; slot < queues.size(); ++slot) {
      auto& signal =
          *static_cast<aql::Signal*>(completions[slot]->host.pointer);
      ASSERT_TRUE(
          GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0));
      const auto* records =
          static_cast<const protocol::Record*>(outputs[slot]->host.pointer);
      const uint32_t seed = 0x751a0000u + epoch * 2 + slot;
      for (uint32_t group = 0; group < kGroupCount; ++group) {
        EXPECT_EQ(records[group].value, group ^ seed)
            << "queue=" << slot << " epoch=" << epoch << " group=" << group;
        observed[slot].insert(records[group].identity);
      }
      EXPECT_LE(observed[slot].size(), selected_count);
      ASSERT_NO_FATAL_FAILURE(
          queues[slot]->WaitConsumed(api_, positions[slot]));
    }
    if (HasFailure()) {
      return;
    }
  }
  for (uint32_t identity : observed[0]) {
    EXPECT_FALSE(observed[1].contains(identity))
        << "disjoint queue selections executed on physical resource "
        << identity;
  }
  EXPECT_FALSE(observed[0].empty());
  EXPECT_FALSE(observed[1].empty());
  RecordProperty("first_observed_resource_count",
                 std::to_string(observed[0].size()));
  RecordProperty("second_observed_resource_count",
                 std::to_string(observed[1].size()));
}

}  // namespace
