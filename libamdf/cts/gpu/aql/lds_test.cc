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

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/lds_exchange.h"
#include "libamdf/cts/gpu/kernels/lds_exchange_kernels.h"

namespace {

using Arguments = kernels::lds_exchange::Arguments;

using AqlLdsTest = AqlDispatchTest;

TEST_F(AqlLdsTest, StaticStorageExchangesBetweenWaves) {
  constexpr uint32_t kGridSize = 512;
  constexpr uint32_t kOutputWordCount = kGridSize * 2;
  constexpr uint32_t kGuardWordCount = 16;
  constexpr uint32_t kWordCount = kOutputWordCount + 2 * kGuardWordCount;
  constexpr uint32_t kPrefixGuard = 0x619b30d5u;
  constexpr uint32_t kSuffixGuard = 0xe270c84bu;
  constexpr std::array<uint32_t, 3> kSeeds = {0x13579bdfu, 0xa5c31f27u,
                                              0x2468ace1u};

  amdf_gpu_endpoint_info_t endpoint_info = {};
  endpoint_info.type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO;
  endpoint_info.structure_size = sizeof(endpoint_info);
  ASSERT_EQ(gpu_api_->endpoint_query_info(endpoint_, &endpoint_info),
            AMDF_STATUS_OK);
  const auto* selected = kernels::lds_exchange::kKernels.Find(endpoint_info);
  ASSERT_NE(selected, nullptr) << "missing compiled LDS kernel for endpoint";
  const auto& kernel = *selected;
  RecordProperty("lds_kernel_target", kernel.target);

  ASSERT_EQ(endpoint_info.compute.wavefront_size, kernel.wavefront_size);
  ASSERT_LE(kernel.group_segment_byte_length,
            endpoint_info.compute.local_data_share_byte_length);
  RecordProperty(
      "aql_lds_capacity_per_compute_unit",
      std::to_string(endpoint_info.compute.local_data_share_byte_length));
  RecordProperty("aql_lds_fixed_group_byte_length",
                 kernel.group_segment_byte_length);
  RecordProperty("aql_lds_kernarg_semantic_byte_length",
                 kernels::lds_exchange::kArgumentByteLength);
  RecordProperty("aql_lds_kernarg_slot_byte_length", sizeof(Arguments));
  RecordProperty("aql_lds_workgroup_size", kernel.workgroup_size());
  RecordProperty("aql_lds_grid_size", kGridSize);
  RecordProperty("aql_lds_output_words_per_epoch", kOutputWordCount);

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

  std::array<uint32_t, kWordCount> expected;
  std::array<uint32_t, kWordCount> observed;
  for (uint32_t epoch = 0; epoch < kSeeds.size(); ++epoch) {
    expected.fill(kSuffixGuard);
    for (uint32_t word = 0; word < kGuardWordCount; ++word) {
      expected[word] = kPrefixGuard;
    }
    for (uint32_t workitem = 0; workitem < kGridSize; ++workitem) {
      const auto record =
          kernels::lds_exchange::ExpectedRecord(workitem, kSeeds[epoch]);
      const uint32_t position = kGuardWordCount + workitem * 2;
      expected[position] = record[0];
      expected[position + 1] = record[1];
    }

    observed = expected;
    for (uint32_t word = 0; word < kOutputWordCount; ++word) {
      observed[kGuardWordCount + word] = ~expected[kGuardWordCount + word];
    }
    std::memcpy(output->host.pointer, observed.data(), sizeof(observed));
    const Arguments payload = {
        output->device_address + kGuardWordCount * sizeof(uint32_t),
        kSeeds[epoch],
    };
    // Zero the full fetch span before copying the semantic argument fields.
    std::memset(arguments->host.pointer, 0, sizeof(Arguments));
    std::memcpy(arguments->host.pointer, &payload,
                kernels::lds_exchange::kArgumentByteLength);
    ASSERT_EQ(
        GpuLoadAcquire<int64_t>(reinterpret_cast<uintptr_t>(&signal.value)), 0);
    signal.value = 1;
    const auto packet =
        aql::Dispatch(aql::HeaderBarrier::kDisabled,
                      {1,
                       {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
                       {kGridSize, 1, 1}},
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
    const std::string prefix = "aql_lds_epoch_" + std::to_string(epoch + 1);
    RecordProperty(prefix + "_seed", std::to_string(kSeeds[epoch]));
    RecordProperty(prefix + "_group_byte_length",
                   kernel.group_segment_byte_length);
  }
  RecordProperty("aql_lds_completed_epochs", kSeeds.size());
}

}  // namespace
