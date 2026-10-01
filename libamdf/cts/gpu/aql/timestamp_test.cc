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
#include <utility>

#include "libamdf/cts/gpu/aql/queue_fixture.h"
#include "libamdf/cts/gpu/pm4/encoding/memory_commands.h"

namespace {

class AqlTimestampTest
    : public AqlQueueTest,
      public ::testing::WithParamInterface<pm4::CopyDataPolicy> {
 protected:
  AqlTimestampTest()
      : AqlQueueTest(AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL) {
  }
};

TEST_P(AqlTimestampTest, ConfirmedClockSamplesAreVisibleBeforeReuse) {
  constexpr size_t kByteLength = 4096;
  constexpr size_t kWordCount = kByteLength / sizeof(uint32_t);
  constexpr size_t kOutputWordCount = kByteLength / sizeof(uint64_t);
  constexpr size_t kCompletionGuardWordCount =
      (kByteLength - sizeof(aql::Signal)) / sizeof(uint32_t);
  constexpr size_t kEpochCount = 2;
  constexpr std::array<size_t, 2> kTimestampByteOffsets = {64, 128};
  constexpr std::array<size_t, 2> kTimestampIndices = {
      kTimestampByteOffsets[0] / sizeof(uint64_t),
      kTimestampByteOffsets[1] / sizeof(uint64_t)};
  constexpr uint32_t kBodyWordCount = 12;
  constexpr aql::FenceScopes kScopes = {aql::FenceScope::kSystem,
                                        aql::FenceScope::kSystem};
  GpuMemory* output = nullptr;
  GpuMemory* commands = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kByteLength, &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
                   kByteLength, &commands));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kByteLength, &completion));
  for (const auto& [name, memory] :
       {std::pair{"output", output}, std::pair{"ib", commands},
        std::pair{"completion", completion}}) {
    SCOPED_TRACE(name);
    ASSERT_EQ(memory->info.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
    ASSERT_NE(memory->info.flags & AMDF_MEMORY_FLAG_HOST_VISIBLE, 0u);
    ASSERT_NE(memory->access_info.flags & AMDF_MEMORY_FLAG_HOST_COHERENT, 0u);
    ASSERT_EQ(memory->host.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
    constexpr amdf_memory_map_flags_t kHostAccess =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    ASSERT_EQ(memory->host.flags & kHostAccess, kHostAccess);
    const std::string prefix = std::string("aql_timestamp_") + name;
    RecordProperty(prefix + "_memory_class", memory->info.memory_class);
    RecordProperty(prefix + "_memory_profile_ordinal",
                   memory->info.memory_profile_ordinal);
    RecordProperty(prefix + "_backing_flags",
                   std::to_string(memory->info.flags));
    RecordProperty(prefix + "_device_access", memory->access_info.access);
    RecordProperty(prefix + "_access_flags",
                   std::to_string(memory->access_info.flags));
    RecordProperty(prefix + "_host_cacheability", memory->host.cacheability);
    RecordProperty(prefix + "_host_access", memory->host.flags);
  }
  ASSERT_EQ(output->device_address % sizeof(uint64_t), 0u);
  ASSERT_EQ(commands->device_address % sizeof(uint32_t), 0u);
  ASSERT_EQ(completion->device_address % alignof(aql::Signal), 0u);
  constexpr uint64_t kIbAddressLimit = UINT64_C(1) << 48;
  ASSERT_LT(commands->device_address, kIbAddressLimit);
  ASSERT_LE(commands->info.byte_length,
            kIbAddressLimit - commands->device_address);

  // This same-backing query supplies visibility. Family transfer admission
  // separately permits the clock-source COPY_DATA form.
  const amdf_memory_site_t output_device = output->DeviceSite(family_.ordinal);
  const amdf_memory_site_t output_host = output->HostSite();
  amdf_memory_pair_info_t egress = {};
  egress.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
  egress.structure_size = sizeof(egress);
  ASSERT_EQ(api_->memory_query_pair_info(&output_device, &output_host, &egress),
            AMDF_STATUS_OK);
  ASSERT_NE(egress.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
  ASSERT_EQ(egress.release.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
  ASSERT_EQ(egress.release.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
  ASSERT_EQ(egress.release.operation, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM);
  ASSERT_EQ(egress.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
  ASSERT_EQ(egress.acquire.executor, AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
  ASSERT_EQ(egress.acquire.operation, AMDF_CACHE_OPERATION_NONE);
  for (const auto* transition : {&egress.release, &egress.acquire}) {
    ASSERT_EQ(transition->host_operation, AMDF_HOST_CACHE_OPERATION_NONE);
    ASSERT_EQ(transition->host_instruction, AMDF_HOST_CACHE_INSTRUCTION_NONE);
    ASSERT_EQ(transition->host_fence_before, AMDF_HOST_CACHE_FENCE_NONE);
    ASSERT_EQ(transition->host_fence_after, AMDF_HOST_CACHE_FENCE_NONE);
    ASSERT_EQ(transition->range_granularity, 0u);
  }
  RecordProperty("aql_timestamp_egress_flags", std::to_string(egress.flags));
  RecordProperty("aql_timestamp_egress_release",
                 "kind=" + std::to_string(egress.release.kind) +
                     ",executor=" + std::to_string(egress.release.executor) +
                     ",operation=" + std::to_string(egress.release.operation));
  RecordProperty("aql_timestamp_egress_acquire",
                 "kind=" + std::to_string(egress.acquire.kind) +
                     ",executor=" + std::to_string(egress.acquire.executor) +
                     ",operation=" + std::to_string(egress.acquire.operation));

  std::array<uint32_t, kWordCount> expected_commands;
  expected_commands.fill(0x53b79d21u);
  const uint32_t prefix_word_count = aql::SingleExecutorPrefix(
      expected_commands.data(), gpu_endpoint_info_.topology.xcc_count,
      kBodyWordCount);
  const uint32_t ib_word_count = prefix_word_count + kBodyWordCount;
  size_t word_count = prefix_word_count;
  for (size_t byte_offset : kTimestampByteOffsets) {
    word_count +=
        pm4::CopyGpuClock64(expected_commands.data() + word_count,
                            output->device_address + byte_offset, GetParam());
  }
  ASSERT_EQ(word_count, ib_word_count);
  // Upload the entire initialized page once. Predication selects the clock
  // executor; native completion still protects every XCC's borrowed IB use.
  std::memcpy(commands->host.pointer, expected_commands.data(),
              sizeof(expected_commands));
  const aql::Packet packet = aql::IndirectBuffer(
      aql::HeaderBarrier::kEnabled, commands->device_address, ib_word_count,
      completion->device_address, kScopes);
  auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
  auto* completion_guard_address =
      static_cast<uint8_t*>(completion->host.pointer) + sizeof(aql::Signal);
  std::array<uint64_t, kOutputWordCount> initial_output;
  std::array<uint64_t, kOutputWordCount> observed_output;
  std::array<uint32_t, kWordCount> observed_commands;
  std::array<uint32_t, kCompletionGuardWordCount> initial_completion_guards;
  std::array<uint32_t, kCompletionGuardWordCount> observed_completion_guards;
  aql::Signal observed_signal = {};
  uint64_t previous_end = 0;
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  const uint64_t capacity = queue->host.ring_byte_length / sizeof(aql::Packet);
  ASSERT_GE(capacity, kEpochCount);
  const uint64_t first_index =
      GpuLoadAcquire<uint64_t>(queue->host.write_index_address);
  ASSERT_LE(first_index, capacity - kEpochCount);
  uint64_t index = first_index;
  RecordProperty("aql_timestamp_byte_length", kByteLength);
  RecordProperty("aql_timestamp_width_bits", 64);
  RecordProperty("aql_timestamp_control",
                 std::to_string(expected_commands[prefix_word_count + 1]));
  RecordProperty("aql_timestamp_body_word_count", kBodyWordCount);
  RecordProperty("aql_timestamp_ib_word_count", ib_word_count);
  RecordProperty("aql_timestamp_prefix_word_count", prefix_word_count);
  RecordProperty("aql_timestamp_sample_byte_offsets", "64,128");
  RecordProperty("aql_timestamp_acquire_scope",
                 static_cast<uint32_t>(kScopes.acquire));
  RecordProperty("aql_timestamp_release_scope",
                 static_cast<uint32_t>(kScopes.release));
  RecordProperty("aql_timestamp_first_packet_index",
                 std::to_string(first_index));
  RecordProperty("aql_timestamp_ring_capacity_packets",
                 std::to_string(capacity));
  for (size_t epoch = 0; epoch < kEpochCount; ++epoch) {
    SCOPED_TRACE(epoch);
    for (size_t i = 0; i < kOutputWordCount; ++i) {
      initial_output[i] = UINT64_C(0x17395b7d2468ace0) +
                          epoch * UINT64_C(0x0011001100110011) +
                          i * UINT64_C(0x0102030405060708);
    }
    // Opposite poisons check that both sample locations change without an
    // absolute clock oracle. Every nonsample word is independently guarded.
    initial_output[kTimestampIndices[0]] = UINT64_MAX;
    initial_output[kTimestampIndices[1]] = 0;
    std::memcpy(output->host.pointer, initial_output.data(),
                sizeof(initial_output));
    aql::Signal initial_signal = {};
    initial_signal.kind = 1;
    initial_signal.value = 1;
    std::memcpy(completion->host.pointer, &initial_signal,
                sizeof(initial_signal));
    initial_completion_guards.fill(0x68d329b7u + static_cast<uint32_t>(epoch));
    // All native fields start zero except USER kind/value. Arbitrary guards
    // occupy only bytes after the complete native 64-byte signal block.
    std::memcpy(completion_guard_address, initial_completion_guards.data(),
                sizeof(initial_completion_guards));
    GpuStoreRelease(queue->host.write_index_address, index + 1);
    aql::Publish(*queue, index++, packet);

    // Capture every byte before diagnostics or consumption can contribute
    // another observation boundary. Clock samples do not complete shader work.
    GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0);
    std::memcpy(observed_output.data(), output->host.pointer,
                sizeof(observed_output));
    std::memcpy(observed_commands.data(), commands->host.pointer,
                sizeof(observed_commands));
    std::memcpy(&observed_signal, completion->host.pointer,
                sizeof(observed_signal));
    std::memcpy(observed_completion_guards.data(), completion_guard_address,
                sizeof(observed_completion_guards));
    for (size_t i = 0; i < kOutputWordCount; ++i) {
      if (i == kTimestampIndices[0] || i == kTimestampIndices[1]) {
        EXPECT_NE(observed_output[i], initial_output[i]) << "sample word=" << i;
      } else {
        EXPECT_EQ(observed_output[i], initial_output[i])
            << "output guard=" << i;
      }
    }
    for (size_t i = 0; i < kWordCount; ++i) {
      EXPECT_EQ(observed_commands[i], expected_commands[i]) << "IB word=" << i;
    }
    EXPECT_EQ(observed_signal.kind, 1);
    EXPECT_EQ(observed_signal.value, 0);
    for (size_t i = 0; i < kCompletionGuardWordCount; ++i) {
      EXPECT_EQ(observed_completion_guards[i], initial_completion_guards[i])
          << "completion guard=" << i;
    }
    const uint64_t begin = observed_output[kTimestampIndices[0]];
    const uint64_t end = observed_output[kTimestampIndices[1]];
    // This finite same-XCC observation assumes no reset or counter wrap.
    // Equality is legal; raw ordering establishes neither rate nor duration.
    EXPECT_LE(begin, end);
    if (epoch != 0) {
      EXPECT_LE(previous_end, begin);
    }
    const std::string prefix = "aql_timestamp_epoch_" + std::to_string(epoch);
    RecordProperty(prefix + "_begin_ticks", std::to_string(begin));
    RecordProperty(prefix + "_end_ticks", std::to_string(end));
    // Even failed nonfatal observations reach retirement. Neither signal nor
    // output is rearmed after any observation or retirement failure.
    EXPECT_NO_FATAL_FAILURE(queue->WaitConsumed(api_, index));
    if (HasFailure()) {
      return;
    }
    previous_end = end;
  }
  RecordProperty("aql_timestamp_completed_epochs", kEpochCount);
  RecordProperty("aql_timestamp_immutable_ib_checks", kEpochCount);
  RecordProperty("aql_timestamp_work_packet_count", kEpochCount);
  RecordProperty("aql_timestamp_final_packet_index", std::to_string(index));
}

INSTANTIATE_TEST_SUITE_P(
    Policy, AqlTimestampTest,
    ::testing::Values(pm4::CopyDataPolicy::kDefault,
                      pm4::CopyDataPolicy::kStreaming),
    [](const ::testing::TestParamInfo<pm4::CopyDataPolicy>& info) {
      return info.param == pm4::CopyDataPolicy::kDefault ? "Default"
                                                         : "Streaming";
    });

}  // namespace
