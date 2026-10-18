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
#include <tuple>

#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"
#include "libamdf/cts/gpu/pm4/dispatch_fixture.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/util/wait.h"

namespace {

// The command processor transports the payload without interpreting it. Only
// the final generation establishes when the host may consume the record and
// the shader output it describes.
struct CompletionRecord {
  // Opaque host action identity or caller-defined completion payload.
  uint64_t payload;
  // Device-written generation, never cleared by the consumer between uses.
  uint32_t generation;
  // Untouched surrounding storage, checking the exact native write extent.
  uint32_t guard;
};
static_assert(sizeof(CompletionRecord) == 16);
static_assert(offsetof(CompletionRecord, generation) == 8);

class Pm4RetirementTest
    : public Pm4DispatchTest,
      public ::testing::WithParamInterface<std::tuple<uint32_t, uint32_t>> {};

TEST_P(Pm4RetirementTest, PublishesShaderResultsBeforeReusingCompletionSlots) {
  const uint32_t slot_count = std::get<0>(GetParam());
  const uint32_t initial_generation = std::get<1>(GetParam());
  const auto* kernel_product =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel_product, nullptr);
  const auto& kernel = *kernel_product;
  RecordProperty("transform_kernel_target", kernel.target);

  constexpr uint32_t kMaximumSlotCount = 16;
  constexpr uint32_t kEpochCount = 8;
  constexpr uint32_t kGridSize = 128;
  constexpr uint32_t kPayloadWordCount = kMaximumSlotCount * kGridSize;
  constexpr uint32_t kArgumentStride = 64;
  constexpr uint32_t kRecordByteOffset = 64;
  constexpr uint32_t kControlWordCount = 1024;
  constexpr uint32_t kGuard = 0x68d329b7u;
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  GpuMemory* input = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* control = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ, kPayloadWordCount * sizeof(uint32_t), &input));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(kReadWrite, kPayloadWordCount * sizeof(uint32_t), &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &control));
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
  ASSERT_EQ(kArgumentStride % kernel.arguments.alignment, 0u);
  ASSERT_LE(kernel.arguments.byte_length, kArgumentStride);
  ASSERT_EQ(control->host.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);

  Pm4ComputeProgram program = {
      0,
      kernel.program.resource1,
      kernel.program.resource2,
      kernel.program.resource3,
      kernel.group_segment_byte_length,
      kernel.wavefront_size,
      {kernel.workgroup_size(), 1, 1},
  };
  ASSERT_NO_FATAL_FAILURE(PrepareProgram(
      kernel.executable, kernel.entry_byte_offset, &program, "pm4_retirement"));

  std::array<uint32_t, kControlWordCount> expected_control;
  expected_control.fill(kGuard);
  for (uint32_t slot = 0; slot < slot_count; ++slot) {
    const size_t record_word =
        (kRecordByteOffset + slot * sizeof(CompletionRecord)) /
        sizeof(uint32_t);
    expected_control[record_word + 2] = initial_generation;
  }
  // The CPU initializes generations once. Subsequent reuse is distinguished
  // solely by each slot's next device-written generation, including wrap.
  std::memcpy(control->host.pointer, expected_control.data(),
              sizeof(expected_control));
  std::memset(arguments->host.pointer, 0, arguments->host.byte_length);
  const uint64_t host_records =
      reinterpret_cast<uintptr_t>(control->host.pointer) + kRecordByteOffset;
  const uint64_t device_records = control->device_address + kRecordByteOffset;

  GpuCommandQueue* queue = nullptr;
  // Every epoch uses a distinct finite command range. Completion slots and
  // operands are reused; this case does not claim primary-ring wrap coverage.
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue, 65536));
  ASSERT_GT(queue->words().size(), kEpochCount * kMaximumSlotCount * 64u);
  Pm4CommandWriter commands(queue->words().data(), *pm4_profile_);
  for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
    SCOPED_TRACE(epoch);
    const uint32_t generation = initial_generation + epoch + 1;
    std::array<uint32_t, kPayloadWordCount> expected_input;
    std::array<uint32_t, kPayloadWordCount> expected_output;
    std::array<uint32_t, kPayloadWordCount> observed_output;
    std::array<uint64_t, kMaximumSlotCount> payloads = {};
    expected_input.fill(kGuard);
    expected_output.fill(kGuard);
    observed_output.fill(kGuard);
    commands.SystemBarrier();
    for (uint32_t slot = 0; slot < slot_count; ++slot) {
      const uint32_t count = kGridSize - 1 - slot;
      const uint32_t addend = 0x81234567u + epoch * 31u + slot;
      const uint64_t offset = slot * kGridSize * sizeof(uint32_t);
      for (uint32_t i = 0; i < kGridSize; ++i) {
        const uint32_t value =
            0xfffffff0u + epoch * 0x1010101u + slot * 0x10001u + i;
        expected_input[slot * kGridSize + i] = value;
        if (i < count) {
          const uint32_t result =
              static_cast<uint32_t>(uint64_t{value} * 3 + addend);
          expected_output[slot * kGridSize + i] = result;
          observed_output[slot * kGridSize + i] = ~result;
        }
      }
      const kernels::transform::Arguments payload = {
          input->device_address + offset, output->device_address + offset,
          count, addend};
      std::memcpy(static_cast<uint8_t*>(arguments->host.pointer) +
                      slot * kArgumentStride,
                  &payload, kernel.arguments.byte_length);
      commands.BindCompute(program,
                           arguments->device_address + slot * kArgumentStride,
                           family_.format_features);
      commands.Dispatch(program, kGridSize, 1, 1);
      // Exercise both halves of the opaque payload across epochs and slots.
      // A real client may carry pointer bits; the device never dereferences it.
      payloads[slot] = UINT64_C(0xa5b6789012345678) ^
                       (uint64_t{generation} << 32) ^
                       (uint64_t{slot} * UINT64_C(0x1020304050607081));
      const uint32_t words[] = {static_cast<uint32_t>(payloads[slot]),
                                static_cast<uint32_t>(payloads[slot] >> 32)};
      const uint64_t record_address =
          device_records + slot * sizeof(CompletionRecord);
      commands.WriteData(record_address, words, 2);
      // The release joins shader writes and publishes the confirmed CP payload
      // through the profile's system release before committing this DWORD.
      // There is no intervening CPU wait or extra SystemBarrier per dispatch.
      commands.ReleaseSystem32(
          record_address + offsetof(CompletionRecord, generation), generation);
    }
    commands.PadToEightWords();
    std::memcpy(input->host.pointer, expected_input.data(),
                sizeof(expected_input));
    std::memcpy(output->host.pointer, observed_output.data(),
                sizeof(observed_output));
    ASSERT_NO_FATAL_FAILURE(
        queue->Publish(api_, gpu_api_, commands.word_count()));

    // Let the GPU fill the batch before draining. Observing the last slot does
    // not substitute for checking every other slot's own generation.
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        host_records + (slot_count - 1) * sizeof(CompletionRecord) +
            offsetof(CompletionRecord, generation),
        generation));
    for (uint32_t slot = 0; slot < slot_count; ++slot) {
      const uint64_t record_address =
          host_records + slot * sizeof(CompletionRecord);
      ASSERT_TRUE(GpuWaitEqual<uint32_t>(
          record_address + offsetof(CompletionRecord, generation), generation));
      CompletionRecord observed;
      std::memcpy(&observed, reinterpret_cast<const void*>(record_address),
                  sizeof(observed));
      EXPECT_EQ(observed.payload, payloads[slot]) << "slot=" << slot;
      EXPECT_EQ(observed.generation, generation) << "slot=" << slot;
      EXPECT_EQ(observed.guard, kGuard) << "slot=" << slot;
      const size_t record_word =
          (kRecordByteOffset + slot * sizeof(CompletionRecord)) /
          sizeof(uint32_t);
      expected_control[record_word] = static_cast<uint32_t>(payloads[slot]);
      expected_control[record_word + 1] =
          static_cast<uint32_t>(payloads[slot] >> 32);
      expected_control[record_word + 2] = generation;
    }
    // Capture results before native consumption/retirement can add ordering.
    std::memcpy(observed_output.data(), output->host.pointer,
                sizeof(observed_output));
    EXPECT_EQ(observed_output, expected_output);
    EXPECT_EQ(std::memcmp(input->host.pointer, expected_input.data(),
                          sizeof(expected_input)),
              0);
    EXPECT_EQ(std::memcmp(control->host.pointer, expected_control.data(),
                          sizeof(expected_control)),
              0);
    ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
    if (HasFailure()) {
      return;
    }
  }
  RecordProperty("retirement_record_bytes", sizeof(CompletionRecord));
  RecordProperty("retirement_slot_count", slot_count);
  RecordProperty("retirement_epoch_count", kEpochCount);
  RecordProperty("retirement_initial_generation",
                 std::to_string(initial_generation));
}

INSTANTIATE_TEST_SUITE_P(
    Batches, Pm4RetirementTest,
    ::testing::Combine(::testing::Values(1u, 4u, 16u),
                       ::testing::Values(0u, UINT32_MAX - 3)),
    [](const ::testing::TestParamInfo<Pm4RetirementTest::ParamType>& info) {
      return "Slots" + std::to_string(std::get<0>(info.param)) +
             (std::get<1>(info.param) == 0 ? "Initial" : "GenerationWrap");
    });

// USER-only lifecycle: completion records reclaim primary storage while native
// events provide unattended capacity progress. Scheduled queues own that ring.
class Pm4RetirementReuseTest : public Pm4DispatchTest,
                               public ::testing::WithParamInterface<uint32_t> {
 protected:
  Pm4RetirementReuseTest()
      : Pm4DispatchTest(AMDF_QUEUE_PUBLICATION_MODE_USER) {}

  void SetUp() override {
    ASSERT_NO_FATAL_FAILURE(Pm4DispatchTest::SetUp());
    if (IsSkipped()) {
      return;
    }
    if ((features_ & AMDF_GPU_DEVICE_FEATURE_NATIVE_EVENTS) == 0) {
      GTEST_SKIP() << "device has no native GPU event producer";
    }
    const amdf_gpu_event_create_info_t create_info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_CREATE_INFO,
        .structure_size = sizeof(create_info),
    };
    ASSERT_EQ(gpu_api_->event_create(device_, &create_info, &event_),
              AMDF_STATUS_OK);
    ASSERT_EQ(gpu_api_->event_query_info(event_, &event_info_), AMDF_STATUS_OK);
    if (event_info_.format != AMDF_GPU_EVENT_FORMAT_KFD_SIGNAL ||
        event_info_.format_version != 1 ||
        (event_info_.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER) ==
            0) {
      GTEST_SKIP() << "PM4 case has no recipe for this event producer format";
    }
  }

  void TearDown() override {
    ASSERT_NO_FATAL_FAILURE(Pm4DispatchTest::TearDown());
    if (event_) {
      ASSERT_EQ(gpu_api_->event_destroy(event_), AMDF_STATUS_OK);
    }
  }

  // Native event retained beyond every queue that may write its mailbox.
  amdf_gpu_event_t* event_ = nullptr;
  // Immutable native operands selected before command publication.
  amdf_gpu_event_info_t event_info_ = {
      .type = AMDF_STRUCTURE_TYPE_GPU_EVENT_INFO,
      .structure_size = sizeof(event_info_),
  };
};

TEST_P(Pm4RetirementReuseTest,
       RecyclesCompletedPrefixesWithoutReadIndexPolling) {
  const uint32_t record_count = GetParam();
  constexpr uint32_t kMaximumRecordCount = 32;
  constexpr uint32_t kSubmissionCount = 512;
  constexpr uint32_t kGridSize = 128;
  constexpr uint32_t kArgumentStride = 64;
  constexpr uint32_t kInitialGeneration = UINT32_MAX - 3;
  constexpr uint32_t kGuard = 0x47b3e695;
  const auto* kernel = kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel, nullptr);
  GpuMemory* input = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* control = nullptr;
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  const uint32_t payload_bytes = record_count * kGridSize * sizeof(uint32_t);
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, payload_bytes, &input));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, payload_bytes, &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &control));
  ASSERT_EQ(control->host.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
  ASSERT_LE(kernel->arguments.byte_length, kArgumentStride);
  ASSERT_EQ(arguments->device_address % kernel->arguments.alignment, 0u);
  ASSERT_EQ(kArgumentStride % kernel->arguments.alignment, 0u);
  Pm4ComputeProgram program = {
      0,
      kernel->program.resource1,
      kernel->program.resource2,
      kernel->program.resource3,
      kernel->group_segment_byte_length,
      kernel->wavefront_size,
      {kernel->workgroup_size(), 1, 1},
  };
  ASSERT_NO_FATAL_FAILURE(PrepareProgram(kernel->executable,
                                         kernel->entry_byte_offset, &program,
                                         "pm4_ring_reuse"));

  std::array<CompletionRecord, kMaximumRecordCount> initial_records;
  initial_records.fill({0, kInitialGeneration, kGuard});
  std::memcpy(control->host.pointer, initial_records.data(),
              sizeof(initial_records));
  std::memset(arguments->host.pointer, 0, arguments->host.byte_length);
  auto* records = static_cast<CompletionRecord*>(control->host.pointer);
  auto* input_words = static_cast<uint32_t*>(input->host.pointer);
  auto* output_words = static_cast<uint32_t*>(output->host.pointer);

  // Eight records constrain retirement capacity; thirty-two let the small
  // primary ring fill first. Both use the same event-driven progress protocol.
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateQueue(family_, &queue, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
                  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER, 4096));
  auto* ring = reinterpret_cast<uint32_t*>(queue->host.ring_address);
  const uint64_t capacity = queue->host.ring_byte_length / sizeof(uint32_t);
  std::array<uint64_t, kMaximumRecordCount> release_positions = {};
  uint64_t published_words = 0;
  uint64_t consumed_words = 0;
  uint32_t submitted = 0;
  uint32_t retired = 0;
  uint32_t capacity_stalls = 0;
  uint32_t ring_wraps = 0;
  const amdf::cts::WaitDeadline deadline;
  while (retired < kSubmissionCount && !HasFailure()) {
    while (submitted < kSubmissionCount && submitted - retired < record_count &&
           !HasFailure()) {
      const uint32_t slot = submitted % record_count;
      const uint32_t generation =
          kInitialGeneration + submitted / record_count + 1;
      const uint32_t addend = 0x81234567u + submitted;
      const uint64_t payload = UINT64_C(0xb5e700008acd0000) ^
                               (uint64_t{generation} << 32) ^ submitted;
      const uint64_t offset = slot * kGridSize * sizeof(uint32_t);
      const kernels::transform::Arguments values = {
          input->device_address + offset, output->device_address + offset,
          kGridSize - 1, addend};
      const uint64_t record_address =
          control->device_address + slot * sizeof(CompletionRecord);
      std::array<uint32_t, 128> encoded;
      Pm4CommandWriter commands(encoded.data(), *pm4_profile_);
      commands.SystemBarrier();
      commands.BindCompute(program,
                           arguments->device_address + slot * kArgumentStride,
                           family_.format_features);
      commands.Dispatch(program, kGridSize, 1, 1);
      const uint32_t payload_words[] = {static_cast<uint32_t>(payload),
                                        static_cast<uint32_t>(payload >> 32)};
      // Both payload packetizations produce the same record. Their different
      // spans exercise ring-tail padding without inserting artificial delay.
      if (submitted % 2 == 0) {
        commands.WriteData(record_address, payload_words, 2);
      } else {
        commands.WriteData(record_address, payload_words, 1);
        commands.WriteData(record_address + sizeof(uint32_t), payload_words + 1,
                           1);
      }
      commands.ReleaseSystem32(
          record_address + offsetof(CompletionRecord, generation), generation);
      const uint64_t release_end = commands.word_count();
      commands.ReleaseSystemInterrupt32(event_info_.mailbox_address,
                                        event_info_.trigger_data,
                                        event_info_.interrupt_context);
      commands.PadToEightWords();
      const uint64_t span = commands.word_count();
      ASSERT_LE(span, encoded.size());
      ASSERT_LT(span * 2, capacity);
      const uint64_t position = published_words & (capacity - 1);
      const uint64_t tail =
          position + span > capacity ? capacity - position : 0;
      if (published_words - consumed_words + tail + span >= capacity) {
        ++capacity_stalls;
        break;
      }

      // The record credit establishes that these operands' prior use retired.
      for (uint32_t i = 0; i < kGridSize; ++i) {
        input_words[slot * kGridSize + i] = 0xfffffff0u + submitted * 31u + i;
        output_words[slot * kGridSize + i] = kGuard;
      }
      std::memcpy(static_cast<uint8_t*>(arguments->host.pointer) +
                      slot * kArgumentStride,
                  &values, kernel->arguments.byte_length);
      if (tail) {
        Pm4CommandWriter padding(ring + position, *pm4_profile_);
        padding.Noop(tail);
        ++ring_wraps;
      }
      std::memcpy(ring + (tail ? 0 : position), encoded.data(),
                  span * sizeof(uint32_t));
      release_positions[slot] = published_words + tail + release_end;
      published_words += tail + span;
      ASSERT_NO_FATAL_FAILURE(queue->PublishStream(published_words));
      ++submitted;
    }

    // Capacity is reclaimed from each exact completed record, without reading
    // native RPTR or waiting for queue consumption between submissions.
    const uint32_t before_retirement = retired;
    while (retired < submitted && !HasFailure()) {
      const uint32_t slot = retired % record_count;
      const uint32_t generation =
          kInitialGeneration + retired / record_count + 1;
      if (GpuLoadAcquire<uint32_t>(reinterpret_cast<uintptr_t>(
              &records[slot].generation)) != generation) {
        break;
      }
      const uint64_t payload =
          UINT64_C(0xb5e700008acd0000) ^ (uint64_t{generation} << 32) ^ retired;
      EXPECT_EQ(records[slot].payload, payload);
      EXPECT_EQ(records[slot].guard, kGuard);
      for (uint32_t i = 0; i < kGridSize; ++i) {
        const uint32_t value = 0xfffffff0u + retired * 31u + i;
        const uint32_t expected =
            i + 1 < kGridSize ? value * 3u + 0x81234567u + retired : kGuard;
        EXPECT_EQ(input_words[slot * kGridSize + i], value);
        EXPECT_EQ(output_words[slot * kGridSize + i], expected);
      }
      consumed_words = release_positions[slot];
      ++retired;
    }
    if (retired == before_retirement && !HasFailure()) {
      const uint64_t remaining_ns = deadline.remaining_ns();
      ASSERT_NE(remaining_ns, 0u) << "native event did not advance retirement";
      ASSERT_EQ(gpu_api_->event_wait(event_, remaining_ns), AMDF_STATUS_OK);
    }
  }
  // The last record does not release its following event command. Join that
  // tail only after all user-visible outputs and capacity progress are proven.
  ASSERT_NO_FATAL_FAILURE(queue->WaitConsumed(api_, published_words));
  EXPECT_EQ(retired, kSubmissionCount);
  EXPECT_GT(published_words, capacity);
  EXPECT_GT(ring_wraps, 0u);
  if (record_count == kMaximumRecordCount) {
    EXPECT_GT(capacity_stalls, 0u);
  }
  RecordProperty("retirement_ring_records", record_count);
  RecordProperty("primary_ring_wraps", ring_wraps);
  RecordProperty("primary_capacity_stalls", capacity_stalls);
}

INSTANTIATE_TEST_SUITE_P(Pressure, Pm4RetirementReuseTest,
                         ::testing::Values(8u, 32u));

}  // namespace
