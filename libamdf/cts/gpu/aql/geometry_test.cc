// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/geometry_ids.h"
#include "libamdf/cts/gpu/kernels/geometry_ids_kernels.h"

namespace {

using Arguments = kernels::geometry_ids::Arguments;

class AqlGeometryTest : public AqlDispatchTest {
 protected:
  void RunGeometry(std::array<aql::DispatchGeometry, 2> geometries);
};

void AqlGeometryTest::RunGeometry(
    std::array<aql::DispatchGeometry, 2> geometries) {
  const auto* kernel_product =
      kernels::geometry_ids::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel_product, nullptr)
      << "missing compiled geometry_ids kernel for endpoint";
  const auto& kernel = *kernel_product;
  RecordProperty("geometry_ids_kernel_target", kernel.target);

  constexpr uint32_t kRecordCapacity = 4096;
  constexpr uint32_t kFlatWorkgroupSize = 64;
  constexpr uint32_t kRecordWordCount = 7;
  constexpr uint32_t kGuardWordCount = 16;
  constexpr uint32_t kWordCount =
      kRecordCapacity * kRecordWordCount + 2 * kGuardWordCount;
  constexpr uint32_t kPrefixGuard = 0x619b30d5u;
  constexpr uint32_t kSuffixGuard = 0xe270c84bu;
  constexpr uint32_t kInactiveWord = 0xb73a51c9u;
  constexpr std::array<uint32_t, 2> kEpochTokens = {0x13579bdfu, 0xa5c31f27u};
  constexpr std::array<const char*, 3> kAxes = {"x", "y", "z"};

  amdf_gpu_endpoint_info_t endpoint_info = {};
  endpoint_info.type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO;
  endpoint_info.structure_size = sizeof(endpoint_info);
  ASSERT_EQ(gpu_api_->endpoint_query_info(endpoint_, &endpoint_info),
            AMDF_STATUS_OK);
  ASSERT_EQ(endpoint_info.compute.wavefront_size, kernel.wavefront_size);
  std::array<std::array<uint32_t, 3>, 2> storage_sizes;
  std::array<uint32_t, 2> active_record_counts;
  std::array<uint32_t, 2> inactive_record_counts;
  for (uint32_t epoch = 0; epoch < geometries.size(); ++epoch) {
    const auto& geometry = geometries[epoch];
    ASSERT_EQ(uint64_t{geometry.workgroup_size[0]} *
                  geometry.workgroup_size[1] * geometry.workgroup_size[2],
              kFlatWorkgroupSize);
    for (uint32_t axis = 0; axis < kAxes.size(); ++axis) {
      ASSERT_GE(geometry.grid_size[axis], geometry.workgroup_size[axis]);
      const uint64_t storage_extent = ((uint64_t{geometry.grid_size[axis]} +
                                        geometry.workgroup_size[axis] - 1) /
                                       geometry.workgroup_size[axis]) *
                                      geometry.workgroup_size[axis];
      ASSERT_LE(storage_extent, kRecordCapacity);
      storage_sizes[epoch][axis] = static_cast<uint32_t>(storage_extent);
    }
    const auto& storage_size = storage_sizes[epoch];
    const uint64_t storage_record_count =
        uint64_t{storage_size[0]} * storage_size[1] * storage_size[2];
    ASSERT_LE(storage_record_count, kRecordCapacity);
    const uint64_t record_count = uint64_t{geometry.grid_size[0]} *
                                  geometry.grid_size[1] * geometry.grid_size[2];
    active_record_counts[epoch] = static_cast<uint32_t>(record_count);
    inactive_record_counts[epoch] =
        static_cast<uint32_t>(storage_record_count - record_count);
  }
  RecordProperty("aql_geometry_record_capacity", kRecordCapacity);
  RecordProperty("aql_geometry_record_word_count", kRecordWordCount);
  RecordProperty("aql_geometry_checked_word_count", kWordCount);
  RecordProperty("aql_geometry_kernarg_byte_length", sizeof(Arguments));
  RecordProperty("aql_geometry_flat_workgroup_size", kFlatWorkgroupSize);

  GpuMemory* output = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   uint64_t{kWordCount} * sizeof(uint32_t), &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
  ASSERT_GE(arguments->info.byte_length, sizeof(Arguments));
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
  std::memset(completion->host.pointer, 0, completion->info.byte_length);
  auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
  signal.kind = 1;

  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  uint64_t index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(
      PublishKernel(*queue, kernel, "aql_kernel", &index, &descriptor_address));
  ASSERT_LE(index + geometries.size(),
            queue->host.ring_byte_length / sizeof(aql::Packet));

  std::array<uint32_t, kWordCount> expected;
  std::array<uint32_t, kWordCount> observed;
  for (uint32_t epoch = 0; epoch < geometries.size(); ++epoch) {
    const auto& geometry = geometries[epoch];
    const auto& storage_size = storage_sizes[epoch];
    // The checked tail covers the selected epochs' old/new argument mixtures.
    // Nominal workgroup dimensions match the packet; rounded output pitches
    // keep inactive edge coordinates in separate records.
    expected.fill(kInactiveWord);
    for (uint32_t word = 0; word < kGuardWordCount; ++word) {
      expected[word] = kPrefixGuard;
      expected[kWordCount - kGuardWordCount + word] = kSuffixGuard;
    }
    observed = expected;
    // Recover coordinates and raw IDs from storage indices, independently of
    // the kernel's forward formula. Only exact-grid records must be written.
    for (uint32_t record = 0; record < kRecordCapacity; ++record) {
      const uint32_t x = record % storage_size[0];
      const uint32_t y = (record / storage_size[0]) % storage_size[1];
      const uint32_t z = record / (storage_size[0] * storage_size[1]);
      if (x >= geometry.grid_size[0] || y >= geometry.grid_size[1] ||
          z >= geometry.grid_size[2]) {
        continue;
      }
      const uint32_t position = kGuardWordCount + record * kRecordWordCount;
      expected[position] = x / geometry.workgroup_size[0];
      expected[position + 1] = y / geometry.workgroup_size[1];
      expected[position + 2] = z / geometry.workgroup_size[2];
      expected[position + 3] = x % geometry.workgroup_size[0];
      expected[position + 4] = y % geometry.workgroup_size[1];
      expected[position + 5] = z % geometry.workgroup_size[2];
      expected[position + 6] = kEpochTokens[epoch];
      for (uint32_t word = 0; word < kRecordWordCount; ++word) {
        observed[position + word] = ~expected[position + word];
      }
    }
    std::memcpy(output->host.pointer, observed.data(), sizeof(observed));
    const Arguments payload = {
        output->device_address + kGuardWordCount * sizeof(uint32_t),
        {geometry.workgroup_size[0], geometry.workgroup_size[1],
         geometry.workgroup_size[2]},
        {storage_size[0], storage_size[1]},
        kEpochTokens[epoch],
    };
    std::memcpy(arguments->host.pointer, &payload, sizeof(payload));
    ASSERT_EQ(
        GpuLoadAcquire<int64_t>(reinterpret_cast<uintptr_t>(&signal.value)), 0);
    signal.value = 1;
    const auto packet =
        aql::Dispatch(aql::HeaderBarrier::kDisabled, geometry,
                      kernel.private_segment_byte_length,
                      kernel.group_segment_byte_length, descriptor_address,
                      arguments->device_address, completion->device_address,
                      {aql::FenceScope::kSystem, aql::FenceScope::kSystem});
    GpuStoreRelease(queue->host.write_index_address, index + 1);
    aql::Publish(*queue, index++, packet);
    // Snapshot completion-visible data before diagnostics or ring retirement.
    GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0);
    std::memcpy(observed.data(), output->host.pointer, sizeof(observed));
    for (uint32_t word = 0; word < kWordCount; ++word) {
      EXPECT_EQ(observed[word], expected[word])
          << "epoch=" << epoch << " word=" << word;
    }
    // Retire even after a failed oracle, before any next-epoch storage reuse.
    EXPECT_NO_FATAL_FAILURE(queue->WaitConsumed(api_, index));
    if (HasFailure()) {
      return;
    }
    const std::string prefix =
        "aql_geometry_epoch_" + std::to_string(epoch + 1);
    RecordProperty(prefix + "_token", std::to_string(kEpochTokens[epoch]));
    RecordProperty(prefix + "_dimensions", geometry.dimensions);
    RecordProperty(prefix + "_active_records", active_record_counts[epoch]);
    RecordProperty(prefix + "_inactive_rounded_records",
                   inactive_record_counts[epoch]);
    for (uint32_t axis = 0; axis < kAxes.size(); ++axis) {
      RecordProperty(prefix + "_workgroup_" + kAxes[axis],
                     geometry.workgroup_size[axis]);
      RecordProperty(prefix + "_grid_" + kAxes[axis], geometry.grid_size[axis]);
      RecordProperty(prefix + "_storage_" + kAxes[axis], storage_size[axis]);
    }
  }
  RecordProperty("aql_geometry_completed_epochs", geometries.size());
  RecordProperty("aql_geometry_final_packet_index", std::to_string(index));
}

TEST_F(AqlGeometryTest, TwoDimensionsRefreshWorkgroupAndGridShape) {
  constexpr std::array<aql::DispatchGeometry, 2> kGeometries = {{
      {2, {16, 4, 1}, {48, 8, 1}},
      {2, {4, 16, 1}, {8, 48, 1}},
  }};
  ASSERT_NO_FATAL_FAILURE(RunGeometry(kGeometries));
}

TEST_F(AqlGeometryTest, ThreeDimensionsRefreshWorkgroupAndGridShape) {
  constexpr std::array<aql::DispatchGeometry, 2> kGeometries = {{
      {3, {8, 4, 2}, {24, 8, 4}},
      {3, {4, 2, 8}, {8, 6, 16}},
  }};
  ASSERT_NO_FATAL_FAILURE(RunGeometry(kGeometries));
}

TEST_F(AqlGeometryTest, TwoDimensionsPopulateOnlyExactWorkitems) {
  constexpr std::array<aql::DispatchGeometry, 2> kGeometries = {{
      {2, {16, 4, 1}, {35, 7, 1}},
      {2, {4, 16, 1}, {7, 37, 1}},
  }};
  ASSERT_NO_FATAL_FAILURE(RunGeometry(kGeometries));
}

TEST_F(AqlGeometryTest, ThreeDimensionsPopulateOnlyExactWorkitems) {
  constexpr std::array<aql::DispatchGeometry, 2> kGeometries = {{
      {3, {8, 4, 2}, {19, 7, 3}},
      {3, {4, 2, 8}, {7, 5, 13}},
  }};
  ASSERT_NO_FATAL_FAILURE(RunGeometry(kGeometries));
}

}  // namespace
