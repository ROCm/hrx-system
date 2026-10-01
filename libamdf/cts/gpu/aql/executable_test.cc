// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_alternate_kernels.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"

namespace {

TEST_F(AqlDispatchTest, ReplacesCompletedExecutableAtSameAddress) {
  const auto* first_kernel_product =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(first_kernel_product, nullptr)
      << "missing compiled transform kernel for endpoint";
  const auto& first_kernel = *first_kernel_product;
  RecordProperty("transform_kernel_target", first_kernel.target);

  const auto* alternate_kernel_product =
      kernels::transform_alternate::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(alternate_kernel_product, nullptr)
      << "missing compiled transform_alternate kernel for endpoint";
  const auto& alternate_kernel = *alternate_kernel_product;
  RecordProperty("transform_alternate_kernel_target", alternate_kernel.target);

  constexpr uint32_t kPageByteLength = 4096;
  constexpr uint32_t kWordCount = 2048;
  constexpr uint32_t kPayloadWordOffset = 16;
  constexpr uint32_t kGridSize = 1024;
  constexpr uint32_t kCount = 1003;
  constexpr uint32_t kAddend = 7;
  constexpr uint32_t kInputGuard = 0x759bf13du;
  constexpr uint32_t kOutputGuard = 0xa36cf197u;
  constexpr uint8_t kArgumentGuard = 0x94;
  constexpr uint8_t kControlGuard = 0x6b;
  constexpr uint32_t kSignalGuardByteOffset = 2 * sizeof(aql::Signal);
  const uint32_t kCacheRangeByteLength =
      (std::max(first_kernel.executable.byte_length,
                alternate_kernel.executable.byte_length) +
       255u) &
      ~255u;
  const uint64_t kCodeByteLength =
      (std::max(
           aql::ExecutableByteLength(gpu_endpoint_info_, first_kernel),
           aql::ExecutableByteLength(gpu_endpoint_info_, alternate_kernel)) +
       kPageByteLength - 1) &
      ~(uint64_t{kPageByteLength} - 1);
  const std::array<const kernels::Kernel*, 3> kKernels = {
      &first_kernel, &alternate_kernel, &first_kernel};
  constexpr std::array<uint32_t, 3> kMultipliers = {3, 5, 3};
  constexpr uint64_t kPacketCount = 2 * kKernels.size();

  GpuMemory* code = nullptr;
  GpuMemory* commands = nullptr;
  GpuMemory* input = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* control = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
                   kCodeByteLength, &code));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
                   kPageByteLength, &commands));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ,
                                       kWordCount * sizeof(uint32_t), &input));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kWordCount * sizeof(uint32_t), &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kPageByteLength, &arguments));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kPageByteLength, &control));
  ASSERT_EQ(code->device_address % 256, 0u);
  ASSERT_LT(code->device_address, UINT64_C(1) << 48);
  ASSERT_LE(kCodeByteLength, (UINT64_C(1) << 48) - code->device_address);
  ASSERT_EQ(commands->device_address % 4, 0u);
  ASSERT_LT(commands->device_address, UINT64_C(1) << 48);
  ASSERT_LE(kPageByteLength, (UINT64_C(1) << 48) - commands->device_address);
  ASSERT_EQ(arguments->device_address % first_kernel.arguments.alignment, 0u);
  ASSERT_EQ(arguments->device_address % alternate_kernel.arguments.alignment,
            0u);
  ASSERT_EQ(control->device_address % alignof(aql::Signal), 0u);

  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  const uint64_t capacity = queue->host.ring_byte_length / sizeof(aql::Packet);
  ASSERT_GE(capacity, kPacketCount);
  uint64_t next_packet_index = 0;

  std::array<uint32_t, kWordCount> expected_input;
  std::array<uint32_t, kWordCount> expected_output;
  std::array<uint32_t, kWordCount> observed_input;
  std::array<uint32_t, kWordCount> observed_output;
  std::vector<uint8_t> expected_code(kCodeByteLength);
  std::vector<uint8_t> observed_code(kCodeByteLength);
  std::array<uint8_t, kPageByteLength> expected_commands = {};
  std::array<uint8_t, kPageByteLength> observed_commands;
  std::array<uint8_t, kPageByteLength> expected_arguments;
  std::array<uint8_t, kPageByteLength> observed_arguments;
  std::array<uint8_t, kPageByteLength> observed_control;
  expected_input.fill(kInputGuard);
  for (uint32_t i = 0; i < kGridSize; ++i) {
    // Odd inputs distinguish both programs in every active lane modulo 2^32.
    expected_input[kPayloadWordOffset + i] = 0x13579bdfu + 2u * i * 0x01030507u;
  }
  std::memcpy(input->host.pointer, expected_input.data(),
              sizeof(expected_input));
  expected_arguments.fill(kArgumentGuard);
  std::fill_n(expected_arguments.begin(), 64, 0);
  const kernels::transform::Arguments payload = {
      input->device_address + kPayloadWordOffset * sizeof(uint32_t),
      output->device_address + kPayloadWordOffset * sizeof(uint32_t),
      kCount,
      kAddend,
  };
  // Both images fetch 24 semantic bytes. The host object's alignment padding
  // is not copied into the initialized argument slot.
  std::memcpy(expected_arguments.data(), &payload,
              first_kernel.arguments.byte_length);
  std::memcpy(arguments->host.pointer, expected_arguments.data(),
              sizeof(expected_arguments));
  const auto cache_commands = aql::CodeCacheInvalidate(
      gpu_endpoint_info_, code->device_address, kCacheRangeByteLength);
  std::memcpy(expected_commands.data(), cache_commands.data(),
              cache_commands.size() * sizeof(uint32_t));
  std::memcpy(commands->host.pointer, expected_commands.data(),
              sizeof(expected_commands));
  std::memset(control->host.pointer, kControlGuard, kPageByteLength);
  std::memset(control->host.pointer, 0, kSignalGuardByteOffset);
  auto* signals = static_cast<aql::Signal*>(control->host.pointer);
  signals[0].kind = 1;
  signals[1].kind = 1;
  const auto cache_packet = aql::IndirectBuffer(
      aql::HeaderBarrier::kDisabled, commands->device_address,
      cache_commands.size(), control->device_address,
      {aql::FenceScope::kNone, aql::FenceScope::kNone});

  RecordProperty("aql_replacement_first_image_sha256",
                 first_kernel.executable.sha256);
  RecordProperty("aql_replacement_alternate_image_sha256",
                 alternate_kernel.executable.sha256);
  RecordProperty("aql_replacement_image_byte_length",
                 first_kernel.executable.byte_length);
  RecordProperty("aql_replacement_descriptor_byte_offset",
                 first_kernel.executable.descriptor_byte_offset);
  RecordProperty("aql_replacement_entry_byte_offset",
                 first_kernel.entry_byte_offset);
  RecordProperty("aql_replacement_kernarg_byte_length",
                 first_kernel.arguments.byte_length);
  RecordProperty("aql_replacement_code_allocation_count", 1);
  RecordProperty("aql_replacement_code_backing_byte_length",
                 std::to_string(kCodeByteLength));
  RecordProperty("aql_replacement_cache_range_byte_length",
                 kCacheRangeByteLength);
  RecordProperty("aql_replacement_cache_ib_word_count", cache_commands.size());
  RecordProperty("aql_replacement_checked_data_bytes_each",
                 sizeof(expected_input));
  RecordProperty("aql_replacement_checked_page_bytes_each", kPageByteLength);
  RecordProperty("aql_replacement_signal_guard_byte_offset",
                 kSignalGuardByteOffset);
  RecordProperty("aql_replacement_grid_workitems", kGridSize);
  RecordProperty("aql_replacement_workgroup_workitems",
                 first_kernel.workgroup_size());
  RecordProperty("aql_replacement_active_workitems", kCount);
  RecordProperty("aql_replacement_addend", kAddend);
  RecordProperty("aql_replacement_multipliers", "3,5,3");
  RecordProperty("aql_replacement_ring_capacity_packets",
                 std::to_string(capacity));
  RecordProperty("aql_replacement_completed_generations", 0);

  for (uint32_t generation = 0; generation < kKernels.size(); ++generation) {
    SCOPED_TRACE(generation);
    const kernels::Kernel& kernel = *kKernels[generation];
    const auto& image = kernel.executable;
    std::fill(expected_code.begin(), expected_code.end(), 0);
    std::memcpy(expected_code.data(), image.words, image.byte_length);
    // This queue is the only borrower. The preceding iteration completed and
    // retired every old use before reaching either of these host writes.
    std::memcpy(code->host.pointer, expected_code.data(), expected_code.size());
    expected_output.fill(kOutputGuard);
    for (uint32_t i = 0; i < kCount; ++i) {
      expected_output[kPayloadWordOffset + i] = static_cast<uint32_t>(
          uint64_t{expected_input[kPayloadWordOffset + i]} *
              kMultipliers[generation] +
          kAddend);
    }
    observed_output = expected_output;
    for (uint32_t i = 0; i < kCount; ++i) {
      observed_output[kPayloadWordOffset + i] =
          ~expected_output[kPayloadWordOffset + i];
    }
    std::memcpy(output->host.pointer, observed_output.data(),
                sizeof(observed_output));
    signals[0].value = 1;
    signals[1].value = 1;

    GpuStoreRelease(queue->host.write_index_address, next_packet_index + 1);
    aql::Publish(*queue, next_packet_index++, cache_packet);
    // Ring consumption alone is not code publication. Complete the explicit
    // unpredicated cache command before making the dispatch reachable.
    ASSERT_NO_FATAL_FAILURE(
        WaitCompletionAndConsumption(*queue, signals[0], next_packet_index));
    const auto dispatch_packet = aql::Dispatch(
        aql::HeaderBarrier::kDisabled,
        {1,
         {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
         {kGridSize, 1, 1}},
        kernel.private_segment_byte_length, kernel.group_segment_byte_length,
        code->device_address + image.descriptor_byte_offset,
        arguments->device_address,
        control->device_address + sizeof(aql::Signal),
        {aql::FenceScope::kSystem, aql::FenceScope::kSystem});
    GpuStoreRelease(queue->host.write_index_address, next_packet_index + 1);
    aql::Publish(*queue, next_packet_index++, dispatch_packet);
    GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signals[1].value), 0);

    // Capture every initialized extent before diagnostics or ring retirement.
    std::memcpy(observed_output.data(), output->host.pointer,
                sizeof(observed_output));
    std::memcpy(observed_input.data(), input->host.pointer,
                sizeof(observed_input));
    std::memcpy(observed_code.data(), code->host.pointer, observed_code.size());
    std::memcpy(observed_commands.data(), commands->host.pointer,
                sizeof(observed_commands));
    std::memcpy(observed_arguments.data(), arguments->host.pointer,
                sizeof(observed_arguments));
    std::memcpy(observed_control.data(), control->host.pointer,
                sizeof(observed_control));
    for (uint32_t i = 0; i < kWordCount; ++i) {
      EXPECT_EQ(observed_output[i], expected_output[i]) << "output word=" << i;
      EXPECT_EQ(observed_input[i], expected_input[i]) << "input word=" << i;
    }
    for (uint64_t i = 0; i < kCodeByteLength; ++i) {
      EXPECT_EQ(observed_code[i], expected_code[i]) << "code byte=" << i;
    }
    for (uint32_t i = 0; i < kPageByteLength; ++i) {
      EXPECT_EQ(observed_commands[i], expected_commands[i]) << "IB byte=" << i;
      EXPECT_EQ(observed_arguments[i], expected_arguments[i])
          << "argument byte=" << i;
    }
    for (uint32_t i = 0; i < 2; ++i) {
      aql::Signal observed_signal;
      std::memcpy(&observed_signal,
                  observed_control.data() + i * sizeof(aql::Signal),
                  sizeof(observed_signal));
      EXPECT_EQ(observed_signal.kind, 1) << "signal=" << i;
      EXPECT_EQ(observed_signal.value, 0) << "signal=" << i;
    }
    // Native reserved fields keep their ABI meaning; guards start after both
    // complete signal blocks, never inside firmware-owned control storage.
    for (uint32_t i = kSignalGuardByteOffset; i < kPageByteLength; ++i) {
      EXPECT_EQ(observed_control[i], kControlGuard) << "control byte=" << i;
    }
    // A failed oracle must still retire the command borrow and prevent the
    // next host overwrite. All allocation owners survive queue destruction.
    EXPECT_NO_FATAL_FAILURE(queue->WaitConsumed(api_, next_packet_index));
    if (HasFailure()) {
      return;
    }
    RecordProperty("aql_replacement_completed_generations", generation + 1);
  }

  amdf_user_queue_status_t status = {};
  status.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS;
  status.structure_size = sizeof(status);
  ASSERT_EQ(api_->user_queue_query_status(queue->queue, &status),
            AMDF_STATUS_OK);
  ASSERT_EQ(status.terminal_status, AMDF_STATUS_OK);
  ASSERT_EQ(status.producer_index, kPacketCount);
  ASSERT_EQ(status.consumed_index, kPacketCount);
  RecordProperty("aql_replacement_image_upload_count", kKernels.size());
  RecordProperty("aql_replacement_count", kKernels.size() - 1);
  RecordProperty("aql_replacement_cache_publication_count", kKernels.size());
  RecordProperty("aql_replacement_dispatch_count", kKernels.size());
  RecordProperty("aql_replacement_final_producer_index",
                 std::to_string(status.producer_index));
  RecordProperty("aql_replacement_final_consumed_index",
                 std::to_string(status.consumed_index));
}

}  // namespace
