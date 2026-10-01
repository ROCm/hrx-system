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
#include "libamdf/cts/gpu/kernels/byte_copy_unaligned.h"
#include "libamdf/cts/gpu/kernels/byte_copy_unaligned_kernels.h"

namespace {

using Arguments = kernels::byte_copy_unaligned::Arguments;

constexpr uint32_t kArgumentSemanticByteLength =
    offsetof(Arguments, workgroup_size_x) + sizeof(uint32_t);

TEST_F(AqlDispatchTest, CoherentSystemByteCopyPreservesBoundsAcrossEpochs) {
  const auto* kernel_product =
      kernels::byte_copy_unaligned::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel_product, nullptr)
      << "missing compiled byte_copy_unaligned kernel for endpoint";
  const auto& kernel = *kernel_product;
  RecordProperty("byte_copy_unaligned_kernel_target", kernel.target);

  constexpr uint32_t kPageByteLength = 4096;
  constexpr uint32_t kSourceByteOffset = 65;
  constexpr uint32_t kTargetByteOffset = 131;
  constexpr uint32_t kGridSize = 64;
  constexpr uint32_t kVectorBodyByteLength = 1024;
  constexpr uint32_t kArgumentSlotByteLength = 64;
  constexpr uint8_t kArgumentGuard = 0xa7;
  constexpr uint8_t kCompletionGuard = 0xd3;
  constexpr std::array<uint32_t, 2> kByteLengths = {1039, 1025};
  constexpr aql::FenceScopes kScopes = {aql::FenceScope::kSystem,
                                        aql::FenceScope::kSystem};

  static_assert(kSourceByteOffset + kByteLengths[0] <= kPageByteLength);
  static_assert(kTargetByteOffset + kByteLengths[0] <= kPageByteLength);

  GpuMemory* source = nullptr;
  GpuMemory* target = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kPageByteLength, &source));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kPageByteLength, &target));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kPageByteLength, &arguments));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kPageByteLength, &completion));
  ASSERT_EQ(source->device_address % 16, 0u);
  ASSERT_EQ(target->device_address % 16, 0u);
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
  ASSERT_EQ(completion->device_address % alignof(aql::Signal), 0u);

  std::array<uint8_t, kPageByteLength> expected_source;
  std::array<uint8_t, kPageByteLength> observed_source;
  std::array<uint8_t, kPageByteLength> expected_target;
  std::array<uint8_t, kPageByteLength> observed_target;
  std::array<uint8_t, kPageByteLength> expected_arguments;
  std::array<uint8_t, kPageByteLength> observed_arguments;
  std::array<uint8_t, kPageByteLength> observed_completion;
  std::memset(completion->host.pointer, 0, kPageByteLength);
  auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
  signal.kind = 1;
  // Native reserved fields start zero and remain native-owned. Arbitrary
  // guards begin after the complete USER signal, never inside its ABI block.
  std::memset(
      static_cast<uint8_t*>(completion->host.pointer) + sizeof(aql::Signal),
      kCompletionGuard, kPageByteLength - sizeof(aql::Signal));

  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  uint64_t index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*queue, kernel, "aql_byte_copy_kernel",
                                        &index, &descriptor_address));
  const uint64_t first_work_packet_index = index;
  const uint64_t capacity = queue->host.ring_byte_length / sizeof(aql::Packet);
  ASSERT_GE(capacity, index + kByteLengths.size());

  RecordProperty("aql_byte_copy_kernel_entry_byte_offset",
                 kernel.entry_byte_offset);
  RecordProperty("aql_byte_copy_page_byte_length", kPageByteLength);
  RecordProperty("aql_byte_copy_source_byte_offset", kSourceByteOffset);
  RecordProperty("aql_byte_copy_target_byte_offset", kTargetByteOffset);
  RecordProperty("aql_byte_copy_vector_body_byte_length",
                 kVectorBodyByteLength);
  RecordProperty("aql_byte_copy_grid_size", kGridSize);
  RecordProperty("aql_byte_copy_workgroup_size", kernel.workgroup_size());
  RecordProperty("aql_byte_copy_kernarg_byte_length",
                 kernel.arguments.byte_length);
  RecordProperty("aql_byte_copy_kernarg_semantic_byte_length",
                 kArgumentSemanticByteLength);
  RecordProperty("aql_byte_copy_kernarg_slot_byte_length",
                 kArgumentSlotByteLength);
  RecordProperty("aql_byte_copy_completion_guard_byte_offset",
                 sizeof(aql::Signal));
  RecordProperty("aql_byte_copy_completion_guard_byte_length",
                 kPageByteLength - sizeof(aql::Signal));
  RecordProperty("aql_byte_copy_acquire_scope",
                 static_cast<uint32_t>(kScopes.acquire));
  RecordProperty("aql_byte_copy_release_scope",
                 static_cast<uint32_t>(kScopes.release));
  RecordProperty("aql_byte_copy_first_work_packet_index",
                 std::to_string(first_work_packet_index));
  RecordProperty("aql_byte_copy_ring_capacity_packets",
                 std::to_string(capacity));

  for (uint32_t epoch = 0; epoch < kByteLengths.size(); ++epoch) {
    SCOPED_TRACE(epoch);
    for (uint32_t byte = 0; byte < kPageByteLength; ++byte) {
      expected_source[byte] = static_cast<uint8_t>(
          uint64_t{37} * byte + 17 * (byte / 256) + 83 * epoch + 11);
      expected_target[byte] =
          static_cast<uint8_t>(uint64_t{29} * byte + 19 * epoch + 0x5d);
    }
    observed_target = expected_target;
    for (uint32_t byte = 0; byte < kByteLengths[0]; ++byte) {
      const uint64_t source_byte = uint64_t{kSourceByteOffset} + byte;
      // The oracle derives each byte independently of device-observed data.
      const uint8_t result = static_cast<uint8_t>(
          37 * source_byte + 17 * (source_byte / 256) + 83 * epoch + 11);
      const uint8_t poison = result ^ 0x80;
      observed_target[kTargetByteOffset + byte] = poison;
      // After shrink, the final 14 bytes retain their new-epoch poison.
      expected_target[kTargetByteOffset + byte] =
          byte < kByteLengths[epoch] ? result : poison;
    }
    std::memcpy(source->host.pointer, expected_source.data(),
                sizeof(expected_source));
    std::memcpy(target->host.pointer, observed_target.data(),
                sizeof(observed_target));
    const Arguments payload = {
        source->device_address + kSourceByteOffset,
        target->device_address + kTargetByteOffset,
        kByteLengths[epoch],
        kGridSize,
        1,
        kernel.workgroup_size(),
    };
    expected_arguments.fill(kArgumentGuard);
    std::memset(expected_arguments.data(), 0, kArgumentSlotByteLength);
    // Copy only semantic fields. The compiler's rounded segment and any
    // argument-fetch padding remain inside the initialized slot.
    std::memcpy(expected_arguments.data(), &payload,
                kArgumentSemanticByteLength);
    std::memcpy(arguments->host.pointer, expected_arguments.data(),
                sizeof(expected_arguments));
    ASSERT_EQ(
        GpuLoadAcquire<int64_t>(reinterpret_cast<uintptr_t>(&signal.value)), 0);
    signal.value = 1;
    const auto packet = aql::Dispatch(
        aql::HeaderBarrier::kDisabled,
        {1,
         {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
         {kGridSize, 1, 1}},
        kernel.private_segment_byte_length, kernel.group_segment_byte_length,
        descriptor_address, arguments->device_address,
        completion->device_address, kScopes);
    GpuStoreRelease(queue->host.write_index_address, index + 1);
    aql::Publish(*queue, index++, packet);

    // All four pages are captured before diagnostics or ring consumption can
    // contribute a second observation boundary. Completion bounds code use.
    GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0);
    std::memcpy(observed_target.data(), target->host.pointer,
                sizeof(observed_target));
    std::memcpy(observed_source.data(), source->host.pointer,
                sizeof(observed_source));
    std::memcpy(observed_arguments.data(), arguments->host.pointer,
                sizeof(observed_arguments));
    std::memcpy(observed_completion.data(), completion->host.pointer,
                sizeof(observed_completion));
    aql::Signal observed_signal;
    std::memcpy(&observed_signal, observed_completion.data(),
                sizeof(observed_signal));
    for (uint32_t byte = 0; byte < kPageByteLength; ++byte) {
      EXPECT_EQ(observed_target[byte], expected_target[byte])
          << "target byte=" << byte;
      EXPECT_EQ(observed_source[byte], expected_source[byte])
          << "source byte=" << byte;
      EXPECT_EQ(observed_arguments[byte], expected_arguments[byte])
          << "argument byte=" << byte;
    }
    EXPECT_EQ(observed_signal.kind, 1);
    EXPECT_EQ(observed_signal.value, 0);
    for (uint32_t byte = sizeof(aql::Signal); byte < kPageByteLength; ++byte) {
      EXPECT_EQ(observed_completion[byte], kCompletionGuard)
          << "completion guard byte=" << byte;
    }
    // Every nonfatal mismatch still reaches retirement. No page or signal is
    // rearmed after an observation or retirement failure.
    EXPECT_NO_FATAL_FAILURE(queue->WaitConsumed(api_, index));
    if (HasFailure()) {
      return;
    }
    const std::string prefix = "aql_byte_copy_epoch_" + std::to_string(epoch);
    RecordProperty(prefix + "_byte_length", kByteLengths[epoch]);
    RecordProperty(prefix + "_tail_byte_length",
                   kByteLengths[epoch] - kVectorBodyByteLength);
    RecordProperty(prefix + "_packet_index", std::to_string(index - 1));
  }
  RecordProperty("aql_byte_copy_completed_epochs", kByteLengths.size());
  RecordProperty("aql_byte_copy_work_packet_count",
                 std::to_string(index - first_work_packet_index));
  RecordProperty("aql_byte_copy_final_packet_index", std::to_string(index));
}

}  // namespace
