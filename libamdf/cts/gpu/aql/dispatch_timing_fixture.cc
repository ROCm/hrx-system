// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/aql/dispatch_timing_fixture.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"
#include "util/observation.h"

namespace {

struct alignas(64) TimingSlot {
  // One exclusive signal reused only after its result reaches the sink.
  aql::Signal signal;
  // Neighboring storage never named by a dispatch.
  std::array<uint64_t, 8> guard;
};
static_assert(sizeof(TimingSlot) == 128);

}  // namespace

AqlDispatchTimingTest::AqlDispatchTimingTest()
    : AqlDispatchTest(AMDF_GPU_AQL_FORMAT_FEATURE_DISPATCH_TIMESTAMPS) {}

void AqlDispatchTimingTest::Exercise(bool enabled, uint32_t dispatch_count) {
  ASSERT_GE(gpu_api_->extension_version, AMDF_GPU_EXTENSION_VERSION_2);
  ObservationHostClock host_clock;
  ASSERT_NO_FATAL_FAILURE(host_clock.Initialize(api_));
  amdf_observation_clock_info_t clock = {};
  ASSERT_EQ(
      gpu_api_->device_query_queue_clock_info(device_, family_.ordinal, &clock),
      AMDF_STATUS_OK);
  ASSERT_GT(clock.frequency_hz, 0u);
  const auto* kernel = kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->private_segment_byte_length, 0u);
  ASSERT_LE(kernel->arguments.byte_length,
            sizeof(kernels::transform::Arguments));
  ASSERT_EQ(256u % kernel->arguments.alignment, 0u);
  constexpr uint32_t kSlotCount = 4;
  constexpr uint32_t kWords = 2048;
  constexpr uint32_t kPayload = 16;
  constexpr uint32_t kGrid = 1024;
  constexpr uint32_t kGuard = 0x5b3e27c1;
  constexpr uint64_t kSignalGuard = UINT64_C(0xc719a53e82df640b);
  GpuMemory* input = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* signals = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kWords * sizeof(uint32_t), &input));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kSlotCount * kWords * sizeof(uint32_t), &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &signals));
  for (auto* memory : {input, output, arguments, signals}) {
    ASSERT_NE(memory->access_info.flags & AMDF_MEMORY_FLAG_HOST_COHERENT, 0u);
  }
  for (auto* memory : {output, signals}) {
    const auto device_site = memory->DeviceSite(family_.ordinal);
    const auto host_site = memory->HostSite();
    amdf_memory_pair_info_t pair = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
        .structure_size = sizeof(pair),
    };
    ASSERT_EQ(api_->memory_query_pair_info(&device_site, &host_site, &pair),
              AMDF_STATUS_OK);
    ASSERT_EQ(pair.release.operation, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM);
    ASSERT_EQ(pair.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
  }
  auto* input_words = static_cast<uint32_t*>(input->host.pointer);
  auto* output_words = static_cast<uint32_t*>(output->host.pointer);
  for (uint32_t i = 0; i < kWords; ++i) {
    input_words[i] = 0xfffffff0u + i * 17u;
  }
  std::memset(arguments->host.pointer, 0, arguments->info.byte_length);
  std::fill_n(static_cast<uint64_t*>(signals->host.pointer),
              signals->info.byte_length / sizeof(uint64_t), kSignalGuard);
  auto* slots = reinterpret_cast<TimingSlot*>(
      static_cast<uint8_t*>(signals->host.pointer) + 128);
  ASSERT_EQ(arguments->device_address % kernel->arguments.alignment, 0u);
  GpuUserQueue* queue = nullptr;
  const amdf_gpu_user_queue_flags_t flags =
      enabled ? AMDF_GPU_USER_QUEUE_FLAG_DISPATCH_TIMESTAMPS : 0u;
  ASSERT_NO_FATAL_FAILURE(
      CreateQueue(family_, &queue, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
                  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER, 0, flags));
  uint64_t packet_index = 0;
  uint64_t descriptor = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*queue, *kernel, "timing_kernel",
                                        &packet_index, &descriptor));
  amdf_observation_clock_sample_t native_start = {};
  ASSERT_EQ(gpu_api_->device_query_clock_sample(device_, family_.ordinal,
                                                &native_start),
            AMDF_STATUS_OK);

  std::array<uint64_t, kSlotCount> host_before = {};
  std::array<uint64_t, kSlotCount> previous_end = {};
  uint64_t first_tick = 0;
  uint64_t first_host = 0;
  uint64_t sample_count = 0;
  uint64_t numeric_sum = 0;
  long double minimum_offset = -std::numeric_limits<long double>::infinity();
  long double maximum_offset = std::numeric_limits<long double>::infinity();
  long double total_duration_ns = 0;

  auto submit = [&](uint32_t sequence) {
    const uint32_t slot = sequence % kSlotCount;
    std::fill_n(output_words + slot * kWords, kWords, kGuard);
    slots[slot].signal = aql::UserSignal(1);
    const kernels::transform::Arguments payload = {
        input->device_address + kPayload * sizeof(uint32_t),
        output->device_address + (slot * kWords + kPayload) * sizeof(uint32_t),
        1003u - sequence % 17u,
        sequence + 7u,
    };
    std::memcpy(static_cast<uint8_t*>(arguments->host.pointer) + slot * 256,
                &payload, kernel->arguments.byte_length);
    const auto packet = aql::Dispatch(
        aql::HeaderBarrier::kDisabled,
        {1,
         {static_cast<uint16_t>(kernel->workgroup_size()), 1, 1},
         {kGrid, 1, 1}},
        0, kernel->group_segment_byte_length, descriptor,
        arguments->device_address + slot * 256,
        signals->device_address + 128 + slot * sizeof(TimingSlot));
    ASSERT_EQ(host_clock.Read(&host_before[slot]), AMDF_STATUS_OK);
    GpuStoreRelease(queue->host.write_index_address, packet_index + 1);
    ASSERT_TRUE(aql::Publish(*queue, packet_index++, packet));
  };

  auto collect = [&](uint32_t sequence) {
    const uint32_t slot = sequence % kSlotCount;
    const auto& signal = slots[slot].signal;
    ASSERT_TRUE(
        GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0));
    uint64_t host_after = 0;
    ASSERT_EQ(host_clock.Read(&host_after), AMDF_STATUS_OK);
    const aql::Signal snapshot = signal;
    EXPECT_EQ(snapshot.kind, 1);
    EXPECT_EQ(snapshot.value, 0);
    EXPECT_EQ(snapshot.event.mailbox_address, 0u);
    EXPECT_EQ(snapshot.event.id, 0u);
    EXPECT_EQ(snapshot.event.reserved, 0u);
    EXPECT_EQ(snapshot.queue_address, 0u);
    EXPECT_EQ(snapshot.reserved, 0u);
    for (uint64_t word : slots[slot].guard) {
      EXPECT_EQ(word, kSignalGuard);
    }
    const uint32_t count = 1003u - sequence % 17u;
    for (uint32_t i = 0; i < kWords; ++i) {
      const uint32_t source = 0xfffffff0u + i * 17u;
      const uint32_t expected =
          i >= kPayload && i < kPayload + count
              ? static_cast<uint32_t>(uint64_t{source} * 3 + sequence + 7)
              : kGuard;
      EXPECT_EQ(output_words[slot * kWords + i], expected)
          << "sequence=" << sequence << " word=" << i;
      numeric_sum += output_words[slot * kWords + i];
    }
    if (enabled) {
      EXPECT_GT(snapshot.timestamps.start, 0u);
      // Reusing this slot follows its previous dispatch's acquired
      // completion. Different live slots have no imposed execution or
      // completion order.
      EXPECT_GE(snapshot.timestamps.start, previous_end[slot]);
      previous_end[slot] = snapshot.timestamps.end;
      EXPECT_GE(snapshot.timestamps.end, snapshot.timestamps.start);
      const long double duration_ns =
          (static_cast<long double>(snapshot.timestamps.end) -
           snapshot.timestamps.start) *
          1e9L / clock.frequency_hz;
      EXPECT_LE(duration_ns,
                host_clock.Nanoseconds(host_after - host_before[slot]));
      if (sample_count == 0) {
        first_tick = snapshot.timestamps.start;
        first_host = host_before[slot];
        RecordProperty("first_start_ticks", std::to_string(first_tick));
        RecordProperty("first_host_before_ticks", std::to_string(first_host));
        RecordProperty("first_host_after_ticks", std::to_string(host_after));
      }
      if (sequence + 1 == dispatch_count) {
        RecordProperty("last_start_ticks",
                       std::to_string(snapshot.timestamps.start));
        RecordProperty("last_host_before_ticks",
                       std::to_string(host_before[slot]));
        RecordProperty("last_host_after_ticks", std::to_string(host_after));
      }
      const long double start_ns =
          (static_cast<long double>(snapshot.timestamps.start) - first_tick) *
          1e9L / clock.frequency_hz;
      const long double end_ns =
          (static_cast<long double>(snapshot.timestamps.end) - first_tick) *
          1e9L / clock.frequency_hz;
      minimum_offset = std::max(
          minimum_offset,
          host_clock.Nanoseconds(host_before[slot] - first_host) - start_ns);
      maximum_offset =
          std::min(maximum_offset,
                   host_clock.Nanoseconds(host_after - first_host) - end_ns);
      total_duration_ns += duration_ns;
    } else {
      EXPECT_EQ(snapshot.timestamps.start, 0u);
      EXPECT_EQ(snapshot.timestamps.end, 0u);
    }
    ++sample_count;
  };

  const uint32_t initial_count = std::min(dispatch_count, kSlotCount);
  for (uint32_t i = 0; i < initial_count; ++i) {
    submit(i);
  }
  uint32_t published = initial_count;
  for (uint32_t completed = 0; completed < published; ++completed) {
    collect(completed);
    // Drain every already-published dispatch on an oracle failure, retaining
    // its resources. No new work is published after the first failed check.
    if (!HasFailure() && published < dispatch_count) {
      submit(published++);
    }
  }
  ASSERT_NO_FATAL_FAILURE(queue->WaitConsumed(api_, packet_index));
  amdf_observation_clock_sample_t native_end = {};
  ASSERT_EQ(gpu_api_->device_query_clock_sample(device_, family_.ordinal,
                                                &native_end),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_start.clock.frequency_hz, clock.frequency_hz);
  EXPECT_EQ(native_end.clock.frequency_hz, clock.frequency_hz);
  EXPECT_LE(native_start.timestamp, native_end.timestamp);
  EXPECT_LE(native_start.host.after, native_end.host.before);
  EXPECT_LE(native_start.host.timestamp, native_start.host.after);
  EXPECT_LE(native_end.host.timestamp, native_end.host.after);
  EXPECT_EQ(sample_count, dispatch_count);
  for (uint32_t i = 0; i < kWords; ++i) {
    EXPECT_EQ(input_words[i], 0xfffffff0u + i * 17u);
  }
  const auto* guard_words = static_cast<const uint64_t*>(signals->host.pointer);
  for (size_t i = 0; i < 128 / sizeof(uint64_t); ++i) {
    EXPECT_EQ(guard_words[i], kSignalGuard);
  }
  for (size_t i = (128 + kSlotCount * sizeof(TimingSlot)) / sizeof(uint64_t);
       i < signals->info.byte_length / sizeof(uint64_t); ++i) {
    EXPECT_EQ(guard_words[i], kSignalGuard);
  }
  RecordProperty("dispatch_count", std::to_string(sample_count));
  RecordProperty("signal_slot_count", kSlotCount);
  RecordProperty("numeric_checksum", std::to_string(numeric_sum));
  RecordProperty("frequency_hz", std::to_string(clock.frequency_hz));
  RecordProperty("native_clock_before", std::to_string(native_start.timestamp));
  RecordProperty("native_clock_after", std::to_string(native_end.timestamp));
  if (enabled) {
    const uint64_t last_tick =
        *std::max_element(previous_end.begin(), previous_end.end());
    RecordProperty("native_clock_encloses_dispatches",
                   native_start.timestamp <= first_tick &&
                       last_tick <= native_end.timestamp);
    RecordProperty("duration_sum_ns", std::to_string(total_duration_ns));
    // Relative clock drift can reject a capture-wide fixed offset even when
    // every dispatch publishes its interval correctly.
    RecordProperty("host_constant_offset_consistent",
                   minimum_offset <= maximum_offset);
    RecordProperty("host_offset_width_ns",
                   std::to_string(maximum_offset - minimum_offset));
  }
}
