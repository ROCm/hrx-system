// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/private_roundtrip.h"
#include "libamdf/cts/gpu/kernels/private_roundtrip_kernels.h"

namespace {

using Arguments = kernels::private_roundtrip::Arguments;

TEST_F(AqlDispatchTest, CallerOwnedFixedScratchChangesAcrossEpochs) {
  const auto* kernel_product =
      kernels::private_roundtrip::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel_product, nullptr)
      << "missing compiled private_roundtrip kernel for endpoint";
  const auto& kernel = *kernel_product;
  RecordProperty("private_roundtrip_kernel_target", kernel.target);

  constexpr uint32_t kGridSize = 512;
  constexpr uint32_t kPrivateWordCount = 9;
  constexpr uint32_t kOutputWordCount = kGridSize * kPrivateWordCount;
  constexpr uint32_t kGuardWordCount = 16;
  constexpr uint32_t kWordCount = kOutputWordCount + 2 * kGuardWordCount;
  constexpr uint32_t kPrefixGuard = 0x619b30d5u;
  constexpr uint32_t kSuffixGuard = 0xe270c84bu;
  constexpr std::array<uint32_t, 2> kSeeds = {0x13579bdfu, 0xa5c31f27u};
  constexpr std::array<uint32_t, 2> kRotations = {1, 7};

  RecordProperty("aql_private_segment_byte_length",
                 kernel.private_segment_byte_length);
  RecordProperty("aql_private_output_words_per_epoch", kOutputWordCount);

  GpuMemory* output = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   uint64_t{kWordCount} * sizeof(uint32_t), &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
  std::memset(completion->host.pointer, 0, completion->info.byte_length);
  auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
  signal.kind = 1;

  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateFixedScratchQueue(kernel.private_segment_byte_length, &queue));
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
      const auto record = kernels::private_roundtrip::ExpectedRecord(
          workitem, kSeeds[epoch], kRotations[epoch]);
      for (uint32_t word = 0; word < kPrivateWordCount; ++word) {
        expected[kGuardWordCount + workitem * kPrivateWordCount + word] =
            record[word];
      }
    }

    observed = expected;
    for (uint32_t word = 0; word < kOutputWordCount; ++word) {
      observed[kGuardWordCount + word] = ~expected[kGuardWordCount + word];
    }
    std::memcpy(output->host.pointer, observed.data(), sizeof(observed));
    const Arguments payload = {
        output->device_address + kGuardWordCount * sizeof(uint32_t),
        kSeeds[epoch],
        kRotations[epoch],
    };
    std::memset(arguments->host.pointer, 0, arguments->info.byte_length);
    std::memcpy(arguments->host.pointer, &payload, sizeof(payload));
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
    const std::string prefix = "aql_private_epoch_" + std::to_string(epoch + 1);
    RecordProperty(prefix + "_seed", std::to_string(kSeeds[epoch]));
    RecordProperty(prefix + "_rotation", kRotations[epoch]);
  }
  RecordProperty("aql_private_completed_epochs", kSeeds.size());
}

}  // namespace
