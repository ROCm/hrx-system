// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstring>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/argument_payloads_kernels.h"

namespace {

TEST_F(AqlDispatchTest, NarrowArgumentPayloadsIgnoreSlotPadding) {
  const auto* product =
      kernels::argument_payloads::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(product, nullptr);
  const auto& kernel = *product;
  RecordProperty("argument_payloads_target", kernel.target);
  constexpr std::array<uint32_t, 13> kPayloadSizes = {8, 1, 1, 2, 1, 1, 2,
                                                      2, 3, 5, 6, 4, 1};
  ASSERT_EQ(kernel.arguments.byte_lengths.size(), kPayloadSizes.size());
  for (size_t i = 0; i < kPayloadSizes.size(); ++i) {
    ASSERT_EQ(kernel.arguments.byte_lengths[i], kPayloadSizes[i])
        << "argument=" << i;
  }

  constexpr uint32_t kLaneCount = 64;
  constexpr uint32_t kColumnCount = 19;
  constexpr uint32_t kGuard = 0xdeadbeefu;
  GpuMemory* output = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      (kLaneCount * kColumnCount + 2) * sizeof(uint32_t), &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
  std::memset(completion->host.pointer, 0, completion->info.byte_length);
  auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
  signal = aql::UserSignal();
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  uint64_t index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*queue, kernel, "argument_payloads",
                                        &index, &descriptor_address));

  constexpr std::array<uint8_t, 4> kPadding = {0, 0xa5, 0xff, 0x5a};
  constexpr std::array<uint8_t, 4> kBytes = {0, 0x7f, 0x80, 0xff};
  constexpr std::array<uint16_t, 4> kHalves = {0, 0x7fff, 0x8000, 0xffff};
  auto* words = static_cast<uint32_t*>(output->host.pointer);
  for (size_t epoch = 0; epoch < kPadding.size(); ++epoch) {
    SCOPED_TRACE(epoch);
    const uint8_t flag = static_cast<uint8_t>(epoch % 2);
    const uint8_t tail = static_cast<uint8_t>(1u - flag);
    const uint8_t byte = kBytes[epoch];
    const uint16_t half = kHalves[epoch];
    const uint8_t tiny4 = 0x3c;      // 1.5 in f8E4M3.
    const uint8_t tiny5 = 0x3e;      // 1.5 in f8E5M2.
    const uint16_t narrow = 0x3e00;  // 1.5 in f16.
    const uint16_t brain = 0x3fc0;   // 1.5 in bf16.
    const std::array<uint8_t, 3> predicates = {flag, tail, flag};
    const std::array<uint8_t, 5> bytes = {1, 2, 3, 4, byte};
    const std::array<uint16_t, 3> halves = {3, 4, half};
    const uint32_t wide = 0x87654321u;
    const uint64_t output_address = output->device_address + sizeof(uint32_t);
    const std::array<const void*, kPayloadSizes.size()> payloads = {
        &output_address,
        &flag,
        &byte,
        &half,
        &tiny4,
        &tiny5,
        &narrow,
        &brain,
        predicates.data(),
        bytes.data(),
        halves.data(),
        &wide,
        &tail};
    // Copy precisely the advertised payload bytes into deliberately dirty
    // native slots, as a by-value AMDHSA launcher is entitled to do.
    std::memset(arguments->host.pointer, kPadding[epoch],
                arguments->info.byte_length);
    auto* argument_bytes = static_cast<uint8_t*>(arguments->host.pointer);
    for (size_t i = 0; i < payloads.size(); ++i) {
      std::memcpy(argument_bytes + kernel.arguments.byte_offsets[i],
                  payloads[i], kPayloadSizes[i]);
    }
    for (uint32_t i = 0; i < kLaneCount * kColumnCount + 2; ++i) {
      words[i] = kGuard;
    }
    signal.value = 1;
    const auto packet =
        aql::Dispatch(aql::HeaderBarrier::kDisabled,
                      {1, {kLaneCount, 1, 1}, {kLaneCount, 1, 1}},
                      kernel.private_segment_byte_length,
                      kernel.group_segment_byte_length, descriptor_address,
                      arguments->device_address, completion->device_address);
    GpuStoreRelease(queue->host.write_index_address, index + 1);
    ASSERT_TRUE(aql::Publish(*queue, index++, packet));
    ASSERT_TRUE(
        GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0));
    // Widen in the CPU oracle before subtraction to define signed extension
    // without relying on implementation-defined unsigned-to-signed casts.
    const uint32_t signed_byte = byte < 128 ? byte : uint32_t{byte} - 256;
    const uint32_t signed_half = half < 32768 ? half : uint32_t{half} - 65536;
    for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
      const std::array<uint32_t, kColumnCount> expected = {
          flag,        tail,        tail ? lane : uint32_t{0} - 17,
          signed_byte, byte,        signed_half,
          half,        0x3fc00000,  0x3fc00000,
          0x3fc00000,  0x3fc00000,  flag,
          tail,        flag,        signed_byte,
          byte,        signed_half, half,
          wide};
      for (uint32_t column = 0; column < kColumnCount; ++column) {
        EXPECT_EQ(words[1 + lane * kColumnCount + column], expected[column])
            << "lane=" << lane << " column=" << column;
      }
      if (HasFailure()) {
        break;
      }
    }
    EXPECT_EQ(words[0], kGuard);
    EXPECT_EQ(words[kLaneCount * kColumnCount + 1], kGuard);
    EXPECT_NO_FATAL_FAILURE(queue->WaitConsumed(api_, index));
    if (HasFailure()) {
      return;
    }
  }
}

}  // namespace
