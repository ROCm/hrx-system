// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdint>
#include <cstring>
#include <string>

#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"
#include "libamdf/cts/gpu/pm4/dispatch_fixture.h"

namespace {

void CheckSystemTransition(const amdf_cache_transition_t& transition,
                           amdf_cache_operation_t operation) {
  const bool global = operation != AMDF_CACHE_OPERATION_NONE;
  ASSERT_EQ(transition.kind, global ? AMDF_CACHE_TRANSITION_KIND_GLOBAL
                                    : AMDF_CACHE_TRANSITION_KIND_NONE);
  ASSERT_EQ(transition.executor, global ? AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE
                                        : AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
  ASSERT_EQ(transition.operation, operation);
  ASSERT_EQ(transition.host_operation, AMDF_HOST_CACHE_OPERATION_NONE);
  ASSERT_EQ(transition.host_instruction, AMDF_HOST_CACHE_INSTRUCTION_NONE);
  ASSERT_EQ(transition.host_fence_before, AMDF_HOST_CACHE_FENCE_NONE);
  ASSERT_EQ(transition.host_fence_after, AMDF_HOST_CACHE_FENCE_NONE);
  ASSERT_EQ(transition.range_granularity, 0u);
}

void CheckHostTransition(const amdf_cache_transition_t& transition,
                         const GpuMemory& memory,
                         amdf_host_cache_operation_t operation) {
  if ((memory.access_info.flags & AMDF_MEMORY_FLAG_HOST_COHERENT) != 0) {
    CheckSystemTransition(transition, AMDF_CACHE_OPERATION_NONE);
  } else {
    const auto& expected = operation == AMDF_HOST_CACHE_OPERATION_FLUSH
                               ? memory.host.flush
                               : memory.host.invalidate;
    EXPECT_EQ(transition.kind, expected.kind);
    EXPECT_EQ(transition.executor, expected.executor);
    EXPECT_EQ(transition.host_operation, operation);
    EXPECT_EQ(transition.host_instruction, expected.host_instruction);
    EXPECT_EQ(transition.host_fence_before, expected.host_fence_before);
    EXPECT_EQ(transition.host_fence_after, expected.host_fence_after);
    EXPECT_EQ(transition.range_granularity, expected.range_granularity);
  }
}

void CheckSystemPair(const amdf_api_t* api, const GpuMemory& memory,
                     const amdf_memory_site_t& producer,
                     const amdf_memory_site_t& consumer) {
  amdf_memory_pair_info_t pair = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(pair),
  };
  ASSERT_EQ(api->memory_query_pair_info(&producer, &consumer, &pair),
            AMDF_STATUS_OK);
  ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
  if (producer.kind == AMDF_MEMORY_SITE_KIND_DEVICE) {
    ASSERT_NO_FATAL_FAILURE(CheckSystemTransition(
        pair.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM));
  } else {
    ASSERT_NO_FATAL_FAILURE(CheckHostTransition(
        pair.release, memory, AMDF_HOST_CACHE_OPERATION_FLUSH));
  }
  if (consumer.kind == AMDF_MEMORY_SITE_KIND_DEVICE) {
    ASSERT_NO_FATAL_FAILURE(CheckSystemTransition(
        pair.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM));
  } else {
    ASSERT_NO_FATAL_FAILURE(CheckHostTransition(
        pair.acquire, memory, AMDF_HOST_CACHE_OPERATION_INVALIDATE));
  }
}

class Pm4SystemHandoffTest
    : public Pm4DispatchTest,
      public ::testing::WithParamInterface<amdf_memory_flags_t> {
 protected:
  void TearDown() override {
    // A failed readiness check or producer publication must not leave the
    // consumer waiting for work that the test will never submit. The backing
    // stays alive through the base fixture's native queue teardown.
    if (pending_gate_.host_address != 0) {
      GpuStoreRelease<uint32_t>(pending_gate_.host_address,
                                pending_gate_.value);
    }
    Pm4DispatchTest::TearDown();
  }

  void CreatePayload(amdf_memory_access_t access, uint64_t byte_length,
                     GpuMemory** out_memory) {
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(access, byte_length, out_memory, GetParam()));
  }

  // Accepted consumer work whose producer may not yet have been submitted.
  struct {
    // Host address in fixture-owned coherent control storage, or zero.
    uint64_t host_address = 0;
    // Exact value satisfying the current consumer's wait.
    uint32_t value = 0;
  } pending_gate_;
};

TEST_P(Pm4SystemHandoffTest, ShaderHandoffAcrossQueues) {
  const auto* kernel_product =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel_product, nullptr)
      << "missing compiled transform kernel for endpoint";
  const auto& kernel = *kernel_product;
  RecordProperty("transform_kernel_target", kernel.target);

  constexpr uint32_t kGridSize = 1024;
  constexpr uint32_t kPayloadWordCount = 2048;
  constexpr uint32_t kPageWordCount = 1024;
  constexpr uint32_t kPayloadOffset = 16;
  constexpr uint32_t kArgumentStride = 64;
  constexpr uint32_t kProducerWordsPerEpoch = 64;
  constexpr uint32_t kConsumerWordsPerEpoch = 80;
  constexpr uint32_t kControlGuard = 0x68d329b7u;
  constexpr uint32_t kGateWord = 0;
  constexpr uint32_t kReadyWord = 16;
  constexpr uint32_t kProducerCompletionWord = 32;
  constexpr uint32_t kConsumerCompletionWord = 48;
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  constexpr std::array<uint32_t, 2> kCounts = {1003, 997};
  constexpr std::array<uint32_t, 2> kProducerAddends = {7, 0x80000023u};
  constexpr std::array<uint32_t, 2> kConsumerAddends = {11, 0x10203045u};

  GpuMemory* input = nullptr;
  GpuMemory* intermediate = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* control = nullptr;
  GpuMemory* code = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreatePayload(
      AMDF_MEMORY_ACCESS_READ, kPayloadWordCount * sizeof(uint32_t), &input));
  ASSERT_NO_FATAL_FAILURE(CreatePayload(
      kReadWrite, kPayloadWordCount * sizeof(uint32_t), &intermediate));
  ASSERT_NO_FATAL_FAILURE(
      CreatePayload(kReadWrite, kPayloadWordCount * sizeof(uint32_t), &output));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ, kPageWordCount * sizeof(uint32_t), &arguments));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(kReadWrite, kPageWordCount * sizeof(uint32_t), &control));
  Pm4ComputeProgram program = {
      0,
      kernel.program.resource1,
      kernel.program.resource2,
      kernel.program.resource3,
      kernel.group_segment_byte_length,
      kernel.wavefront_size,
      {kernel.workgroup_size(), 1, 1},
  };
  ASSERT_NO_FATAL_FAILURE(PrepareProgram(kernel.executable,
                                         kernel.entry_byte_offset, &program,
                                         "pm4_handoff", &code));
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
  ASSERT_EQ(control->device_address % 64, 0u);
  RecordProperty(
      "payload_is_host_coherent",
      (input->access_info.flags & AMDF_MEMORY_FLAG_HOST_COHERENT) != 0);
  const std::array<GpuMemory*, 6> backings = {input,     intermediate, output,
                                              arguments, control,      code};
  for (const auto* memory : backings) {
    SCOPED_TRACE(memory->device_address);
    ASSERT_NO_FATAL_FAILURE(
        CheckSystemPair(api_, *memory, memory->HostSite(),
                        memory->DeviceSite(family_.ordinal)));
  }
  ASSERT_NO_FATAL_FAILURE(CheckSystemPair(
      api_, *intermediate, intermediate->DeviceSite(family_.ordinal),
      intermediate->DeviceSite(family_.ordinal)));
  for (const auto* memory : {intermediate, output}) {
    ASSERT_NO_FATAL_FAILURE(CheckSystemPair(api_, *memory,
                                            memory->DeviceSite(family_.ordinal),
                                            memory->HostSite()));
  }

  std::array<uint32_t, kPageWordCount> expected_control;
  expected_control.fill(kControlGuard);
  for (const auto word : {kGateWord, kReadyWord, kProducerCompletionWord,
                          kConsumerCompletionWord}) {
    expected_control[word] = 0;
  }
  // Only GPU writers advance the four control cells after initialization.
  std::memcpy(control->host.pointer, expected_control.data(),
              sizeof(expected_control));
  auto* control_words = static_cast<uint32_t*>(control->host.pointer);
  std::array<uint32_t, kPageWordCount> expected_code = {};
  std::memcpy(expected_code.data(), kernel.executable.words,
              kernel.executable.byte_length);

  GpuCommandQueue* producer = nullptr;
  GpuCommandQueue* consumer = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&producer));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&consumer));
  ASSERT_TRUE(
      amdf_device_id_is_equal(&producer->device_id(), &consumer->device_id()));
  ASSERT_NE(producer->native_handle(), consumer->native_handle());
  ASSERT_EQ(producer->words().size_bytes(), kPageWordCount * sizeof(uint32_t));
  ASSERT_EQ(consumer->words().size_bytes(), kPageWordCount * sizeof(uint32_t));
  std::memset(producer->words().data(), 0, producer->words().size_bytes());
  std::memset(consumer->words().data(), 0, consumer->words().size_bytes());
  Pm4CommandWriter produce(producer->words().data(), *pm4_profile_);
  Pm4CommandWriter consume(consumer->words().data(), *pm4_profile_);
  for (uint32_t epoch = 0; epoch < kCounts.size(); ++epoch) {
    produce.SystemBarrier();
    produce.BindCompute(program, arguments->device_address,
                        family_.format_features);
    produce.Dispatch(program, kGridSize, 1, 1);
    produce.SystemBarrier();
    produce.WriteData32(control->device_address + kGateWord * 4, epoch + 1);
    produce.WriteData32(control->device_address + kProducerCompletionWord * 4,
                        epoch + 1);
    produce.PadToEightWords();
    ASSERT_EQ(produce.word_count(), (epoch + 1) * kProducerWordsPerEpoch);

    consume.SystemBarrier();
    consume.WriteData32(control->device_address + kReadyWord * 4, epoch + 1);
    consume.WaitMemory32(control->device_address + kGateWord * 4, epoch + 1);
    // The wait orders execution. This separate acquire makes shader payload
    // visible; it is not the control cell's fresh-polling mechanism.
    consume.SystemBarrier();
    consume.BindCompute(program, arguments->device_address + kArgumentStride,
                        family_.format_features);
    consume.Dispatch(program, kGridSize, 1, 1);
    consume.SystemBarrier();
    consume.WriteData32(control->device_address + kConsumerCompletionWord * 4,
                        epoch + 1);
    consume.PadToEightWords();
    ASSERT_EQ(consume.word_count(), (epoch + 1) * kConsumerWordsPerEpoch);
  }
  std::array<uint32_t, kPageWordCount> expected_producer_commands;
  std::array<uint32_t, kPageWordCount> expected_consumer_commands;
  std::memcpy(expected_producer_commands.data(), producer->words().data(),
              sizeof(expected_producer_commands));
  std::memcpy(expected_consumer_commands.data(), consumer->words().data(),
              sizeof(expected_consumer_commands));

  for (uint32_t epoch = 0; epoch < kCounts.size(); ++epoch) {
    SCOPED_TRACE(epoch);
    std::array<uint32_t, kPayloadWordCount> expected_input;
    std::array<uint32_t, kPayloadWordCount> expected_intermediate;
    std::array<uint32_t, kPayloadWordCount> expected_output;
    std::array<uint32_t, kPayloadWordCount> observed_input;
    std::array<uint32_t, kPayloadWordCount> observed_intermediate;
    std::array<uint32_t, kPayloadWordCount> observed_output;
    expected_input.fill(0x759bf13du ^ (epoch * 0x01010101u));
    expected_intermediate.fill(0xa36cf197u ^ (epoch * 0x01010101u));
    expected_output.fill(0x4e90b725u ^ (epoch * 0x01010101u));
    observed_intermediate = expected_intermediate;
    observed_output = expected_output;
    for (uint32_t i = 0; i < kGridSize; ++i) {
      const uint32_t value =
          0xfffffff0u + i * 0x01030507u + epoch * 0x11111111u;
      expected_input[kPayloadOffset + i] = value;
      if (i < kCounts[epoch]) {
        const uint32_t middle = static_cast<uint32_t>(uint64_t{value} * 3 +
                                                      kProducerAddends[epoch]);
        // The final oracle is independent of observed intermediate values.
        const uint32_t final = static_cast<uint32_t>(
            uint64_t{value} * 9 + uint64_t{kProducerAddends[epoch]} * 3 +
            kConsumerAddends[epoch]);
        expected_intermediate[kPayloadOffset + i] = middle;
        expected_output[kPayloadOffset + i] = final;
        observed_intermediate[kPayloadOffset + i] = ~middle;
        observed_output[kPayloadOffset + i] = ~final;
      }
    }
    std::memcpy(input->host.pointer, expected_input.data(),
                sizeof(expected_input));
    std::memcpy(intermediate->host.pointer, observed_intermediate.data(),
                sizeof(observed_intermediate));
    std::memcpy(output->host.pointer, observed_output.data(),
                sizeof(observed_output));
    const kernels::transform::Arguments producer_arguments = {
        input->device_address + kPayloadOffset * sizeof(uint32_t),
        intermediate->device_address + kPayloadOffset * sizeof(uint32_t),
        kCounts[epoch], kProducerAddends[epoch]};
    const kernels::transform::Arguments consumer_arguments = {
        intermediate->device_address + kPayloadOffset * sizeof(uint32_t),
        output->device_address + kPayloadOffset * sizeof(uint32_t),
        kCounts[epoch], kConsumerAddends[epoch]};
    std::array<uint8_t, kPageWordCount * sizeof(uint32_t)> expected_arguments =
        {};
    std::memcpy(expected_arguments.data(), &producer_arguments,
                kernel.arguments.byte_length);
    std::memcpy(expected_arguments.data() + kArgumentStride,
                &consumer_arguments, kernel.arguments.byte_length);
    std::memcpy(arguments->host.pointer, expected_arguments.data(),
                sizeof(expected_arguments));
    std::array<uint8_t, sizeof(expected_arguments)> observed_arguments;
    std::array<uint32_t, kPageWordCount> observed_control;
    std::array<uint32_t, kPageWordCount> observed_code;
    std::array<uint32_t, kPageWordCount> observed_producer_commands;
    std::array<uint32_t, kPageWordCount> observed_consumer_commands;

    for (const auto* memory : {input, intermediate, output}) {
      ASSERT_NO_FATAL_FAILURE(memory->TransferHostCache(
          api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
    }

    const uint64_t producer_frontier = (epoch + 1) * kProducerWordsPerEpoch;
    const uint64_t consumer_frontier = (epoch + 1) * kConsumerWordsPerEpoch;
    pending_gate_.host_address =
        reinterpret_cast<uintptr_t>(control_words + kGateWord);
    pending_gate_.value = epoch + 1;
    ASSERT_NO_FATAL_FAILURE(
        consumer->Publish(api_, gpu_api_, consumer_frontier));
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        reinterpret_cast<uintptr_t>(control_words + kReadyWord), epoch + 1));
    const std::array<uint32_t, 3> pending = {
        GpuLoadAcquire<uint32_t>(
            reinterpret_cast<uintptr_t>(control_words + kGateWord)),
        GpuLoadAcquire<uint32_t>(reinterpret_cast<uintptr_t>(
            control_words + kProducerCompletionWord)),
        GpuLoadAcquire<uint32_t>(reinterpret_cast<uintptr_t>(
            control_words + kConsumerCompletionWord)),
    };
    // Prefix readiness is not proof that the wait has polled unsuccessfully.
    // The finite producer runs without an intervening host completion join.
    ASSERT_NO_FATAL_FAILURE(
        producer->Publish(api_, gpu_api_, producer_frontier));
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        reinterpret_cast<uintptr_t>(control_words + kConsumerCompletionWord),
        epoch + 1));
    ASSERT_NO_FATAL_FAILURE(output->TransferHostCache(
        api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_INVALIDATE));
    std::memcpy(observed_output.data(), output->host.pointer,
                sizeof(observed_output));
    // Output is already captured. Independently join the producer's remaining
    // control writes before observing other owners or attempting retirement.
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        reinterpret_cast<uintptr_t>(control_words + kProducerCompletionWord),
        epoch + 1));
    ASSERT_NO_FATAL_FAILURE(intermediate->TransferHostCache(
        api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_INVALIDATE));
    std::memcpy(observed_intermediate.data(), intermediate->host.pointer,
                sizeof(observed_intermediate));
    std::memcpy(observed_input.data(), input->host.pointer,
                sizeof(observed_input));
    std::memcpy(observed_arguments.data(), arguments->host.pointer,
                sizeof(observed_arguments));
    std::memcpy(observed_control.data(), control->host.pointer,
                sizeof(observed_control));
    std::memcpy(observed_code.data(), code->host.pointer,
                sizeof(observed_code));
    std::memcpy(observed_producer_commands.data(), producer->words().data(),
                sizeof(observed_producer_commands));
    std::memcpy(observed_consumer_commands.data(), consumer->words().data(),
                sizeof(observed_consumer_commands));
    for (const auto word : {kGateWord, kReadyWord, kProducerCompletionWord,
                            kConsumerCompletionWord}) {
      expected_control[word] = epoch + 1;
    }
    for (const auto value : pending) {
      EXPECT_EQ(value, epoch);
    }
    for (uint32_t i = 0; i < kPayloadWordCount; ++i) {
      EXPECT_EQ(observed_output[i], expected_output[i]) << "output word=" << i;
      EXPECT_EQ(observed_intermediate[i], expected_intermediate[i])
          << "intermediate word=" << i;
      EXPECT_EQ(observed_input[i], expected_input[i]) << "input word=" << i;
    }
    for (uint32_t i = 0; i < expected_arguments.size(); ++i) {
      EXPECT_EQ(observed_arguments[i], expected_arguments[i])
          << "argument byte=" << i;
    }
    for (uint32_t i = 0; i < kPageWordCount; ++i) {
      EXPECT_EQ(observed_control[i], expected_control[i])
          << "control word=" << i;
      EXPECT_EQ(observed_code[i], expected_code[i]) << "code word=" << i;
      EXPECT_EQ(observed_producer_commands[i], expected_producer_commands[i])
          << "producer command storage word=" << i;
      EXPECT_EQ(observed_consumer_commands[i], expected_consumer_commands[i])
          << "consumer command storage word=" << i;
    }
    EXPECT_NO_FATAL_FAILURE(producer->WaitRetired(api_));
    EXPECT_NO_FATAL_FAILURE(consumer->WaitRetired(api_));
    if (HasFailure()) {
      return;
    }
    const std::string prefix = "pm4_handoff_epoch_" + std::to_string(epoch);
    RecordProperty(prefix + "_active_count", kCounts[epoch]);
    RecordProperty(prefix + "_producer_addend",
                   std::to_string(kProducerAddends[epoch]));
    RecordProperty(prefix + "_consumer_addend",
                   std::to_string(kConsumerAddends[epoch]));
    RecordProperty(prefix + "_producer_frontier",
                   std::to_string(producer_frontier));
    RecordProperty(prefix + "_consumer_frontier",
                   std::to_string(consumer_frontier));
  }
  RecordProperty("pm4_handoff_completed_epochs", kCounts.size());
  RecordProperty("pm4_handoff_pair_query_count", 9);
  RecordProperty("pm4_handoff_grid_size", kGridSize);
  RecordProperty("pm4_handoff_payload_word_offset", kPayloadOffset);
  RecordProperty("pm4_handoff_payload_word_count", kPayloadWordCount);
  RecordProperty("pm4_handoff_argument_stride", kArgumentStride);
  RecordProperty("pm4_handoff_observed_backing_bytes", 9 * 4096);
  RecordProperty("pm4_handoff_observed_commands_bytes", 2 * 4096);
  RecordProperty("pm4_handoff_producer_command_word_count",
                 std::to_string(produce.word_count()));
  RecordProperty("pm4_handoff_consumer_command_word_count",
                 std::to_string(consume.word_count()));
}

INSTANTIATE_TEST_SUITE_P(
    PayloadPolicy, Pm4SystemHandoffTest,
    ::testing::Values(amdf_memory_flags_t{0},
                      amdf_memory_flags_t{AMDF_MEMORY_FLAG_HOST_COHERENT},
                      amdf_memory_flags_t{AMDF_MEMORY_FLAG_HOST_COHERENT |
                                          AMDF_MEMORY_FLAG_QUEUE_STORAGE}));

}  // namespace
