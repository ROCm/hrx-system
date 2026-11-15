// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/kernels/shader_event.h"

#include <array>
#include <cstring>

#include "libamdf/cts/gpu/kernels/shader_event_kernels.h"
#include "libamdf/cts/gpu/kernels/shader_event_resident_kernels.h"
#include "libamdf/cts/gpu/pm4/dispatch_fixture.h"
#include "libamdf/cts/gpu/util/event_fixture.h"

namespace {

class Pm4ShaderEventTest : public GpuEventTest<Pm4DispatchTest> {
 protected:
  void TearDown() override {
    // Release the resident producer even after a fatal host assertion. Native
    // queue teardown then joins execution before the event or backing retires.
    if (acknowledgement_address_ != 0) {
      GpuStoreRelease<uint64_t>(acknowledgement_address_, UINT64_MAX);
    }
    GpuEventTest<Pm4DispatchTest>::TearDown();
  }

  void RunResidentPublisher(uint32_t acknowledged_epochs);

  // Host acknowledgement word, retained with the fixture's coherent backing.
  uint64_t acknowledgement_address_ = 0;
};

TEST_F(Pm4ShaderEventTest, PublishesAndWakesWithoutCommandProcessorInterrupts) {
  namespace protocol = kernels::shader_event;
  const auto* kernel = protocol::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel, nullptr) << "missing compiled shader-event kernel";
  RecordProperty("shader_event_target", kernel->target);
  ASSERT_EQ(kernel->workgroup_size(), 1u);
  ASSERT_EQ(kernel->private_segment_byte_length, 0u);
  ASSERT_EQ(kernel->group_segment_byte_length, 0u);

  GpuMemory* records = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096,
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE,
      &records));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096,
                                       AMDF_MEMORY_FLAG_HOST_COHERENT,
                                       &arguments));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096,
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE,
      &completion));
  constexpr uint64_t kGuard = UINT64_C(0xfedcba9876543210);
  std::array<protocol::Record, 3> expected;
  expected.fill({kGuard, kGuard});
  std::memcpy(records->host.pointer, expected.data(), sizeof(expected));
  std::memset(completion->host.pointer, 0, completion->info.byte_length);
  const uint64_t record_address =
      records->device_address + sizeof(protocol::Record);
  const uint64_t host_generation =
      reinterpret_cast<uintptr_t>(records->host.pointer) +
      sizeof(protocol::Record) + offsetof(protocol::Record, generation);

  Pm4ComputeProgram program = {
      0,
      kernel->program.resource1,
      kernel->program.resource2,
      kernel->program.resource3,
      kernel->group_segment_byte_length,
      kernel->wavefront_size,
      {1, 1, 1},
  };
  ASSERT_NO_FATAL_FAILURE(PrepareProgram(
      kernel->executable, kernel->entry_byte_offset, &program, "shader_event"));
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(65536, &queue));
  Pm4CommandWriter commands(queue->words().data(), *pm4_profile_);
  constexpr uint32_t kEpochCount = 8;
  for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
    SCOPED_TRACE(epoch);
    const uint64_t generation = UINT64_C(0x100000000) + epoch;
    const protocol::Arguments payload = {
        record_address, event_info_.mailbox_address, generation,
        event_info_.trigger_data, event_info_.interrupt_context};
    std::memcpy(arguments->host.pointer, &payload, sizeof(payload));
    commands.SystemBarrier();
    commands.BindCompute(program, arguments->device_address,
                         family_.format_features);
    commands.Dispatch(program, 1, 1, 1);
    commands.PadToEightWords();
    ASSERT_NO_FATAL_FAILURE(
        queue->Publish(api_, gpu_api_, commands.word_count()));

    // No CP release, interrupt, completion marker, or queue wait can rescue
    // this publication. The native shader interrupt is the sole wake source.
    const auto observed = WaitForRecord(host_generation, generation);
    EXPECT_TRUE(observed);
    if (observed) {
      expected[1] = {generation ^ protocol::kPayloadMask, generation};
      EXPECT_EQ(
          std::memcmp(records->host.pointer, expected.data(), sizeof(expected)),
          0);
    }
    // The shader publication oracle precedes any native join. Retire this
    // publication before reusing the fixture's single pending KERNEL slot.
    ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));

    // Join the finite shader after observation, including on an oracle failure.
    // Ring consumption alone would not establish shader execution completion.
    commands.SystemBarrier();
    commands.WriteData32(completion->device_address, epoch + 1);
    commands.PadToEightWords();
    ASSERT_NO_FATAL_FAILURE(
        queue->Publish(api_, gpu_api_, commands.word_count()));
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        reinterpret_cast<uintptr_t>(completion->host.pointer), epoch + 1));
    ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
    if (HasFailure()) {
      return;
    }
  }
  RecordProperty("shader_event_epochs", kEpochCount);
}

void Pm4ShaderEventTest::RunResidentPublisher(uint32_t acknowledged_epochs) {
  namespace protocol = kernels::shader_event;
  const auto* kernel =
      kernels::shader_event_resident::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel, nullptr) << "missing compiled resident shader-event kernel";
  RecordProperty("shader_event_target", kernel->target);
  ASSERT_EQ(kernel->workgroup_size(), 1u);
  ASSERT_EQ(kernel->private_segment_byte_length, 0u);
  ASSERT_EQ(kernel->group_segment_byte_length, 0u);

  GpuMemory* records = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096,
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE,
      &records));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096,
                                       AMDF_MEMORY_FLAG_HOST_COHERENT,
                                       &arguments));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096,
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE,
      &completion));
  constexpr uint64_t kGuard = UINT64_C(0xfedcba9876543210);
  std::array<protocol::ResidentRecord, 3> expected;
  expected.fill({kGuard, kGuard, kGuard, kGuard});
  expected[1] = {};
  std::memcpy(records->host.pointer, expected.data(), sizeof(expected));
  std::memset(completion->host.pointer, 0, completion->info.byte_length);
  const uint64_t record_address =
      records->device_address + sizeof(protocol::ResidentRecord);
  const uint64_t host_record =
      reinterpret_cast<uintptr_t>(records->host.pointer) +
      sizeof(protocol::ResidentRecord);
  const uint64_t host_generation =
      host_record + offsetof(protocol::ResidentRecord, generation);
  acknowledgement_address_ =
      host_record + offsetof(protocol::ResidentRecord, acknowledgement);
  constexpr uint64_t kFirstGeneration = UINT64_C(0x100000000);
  const protocol::Arguments payload = {
      record_address, event_info_.mailbox_address, kFirstGeneration,
      event_info_.trigger_data, event_info_.interrupt_context};
  std::memcpy(arguments->host.pointer, &payload, sizeof(payload));

  Pm4ComputeProgram program = {
      0,
      kernel->program.resource1,
      kernel->program.resource2,
      kernel->program.resource3,
      kernel->group_segment_byte_length,
      kernel->wavefront_size,
      {1, 1, 1},
  };
  ASSERT_NO_FATAL_FAILURE(PrepareProgram(kernel->executable,
                                         kernel->entry_byte_offset, &program,
                                         "shader_event_resident"));
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(65536, &queue));
  Pm4CommandWriter commands(queue->words().data(), *pm4_profile_);
  commands.SystemBarrier();
  commands.BindCompute(program, arguments->device_address,
                       family_.format_features);
  commands.Dispatch(program, 1, 1, 1);
  commands.PadToEightWords();
  ASSERT_NO_FATAL_FAILURE(
      queue->Publish(api_, gpu_api_, commands.word_count()));

  // The shader cannot return or publish another generation until the host
  // consumes this one. Neither a new dispatch nor CP completion can wake us.
  const uint32_t publications =
      acknowledged_epochs < protocol::kResidentEpochCount
          ? acknowledged_epochs + 1
          : acknowledged_epochs;
  for (uint32_t epoch = 0; epoch < publications; ++epoch) {
    SCOPED_TRACE(epoch);
    const uint64_t generation = kFirstGeneration + epoch;
    ASSERT_TRUE(WaitForRecord(host_generation, generation));
    expected[1].payload = generation ^ protocol::kPayloadMask;
    expected[1].generation = generation;
    ASSERT_EQ(
        std::memcmp(records->host.pointer, expected.data(), sizeof(expected)),
        0);
    const uint64_t acknowledgement =
        epoch < acknowledged_epochs ? generation : UINT64_MAX;
    expected[1].acknowledgement = acknowledgement;
    GpuStoreRelease<uint64_t>(acknowledgement_address_, acknowledgement);
  }

  // Observation and host acknowledgement precede the native execution join.
  // Stop also joins the last shader interrupt before the event can be released.
  ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
  commands.SystemBarrier();
  commands.WriteData32(completion->device_address, 1);
  commands.PadToEightWords();
  ASSERT_NO_FATAL_FAILURE(
      queue->Publish(api_, gpu_api_, commands.word_count()));
  ASSERT_TRUE(GpuWaitEqual<uint32_t>(
      reinterpret_cast<uintptr_t>(completion->host.pointer), 1));
  ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
  expected[1].completed_epochs = acknowledged_epochs;
  EXPECT_EQ(
      std::memcmp(records->host.pointer, expected.data(), sizeof(expected)), 0);
  EXPECT_EQ(std::memcmp(arguments->host.pointer, &payload, sizeof(payload)), 0);
  RecordProperty("shader_event_epochs", acknowledged_epochs);
  RecordProperty("shader_event_dispatches", 1);
}

TEST_F(Pm4ShaderEventTest, PublishesAndWakesBeforeResidentPublisherReturns) {
  RunResidentPublisher(kernels::shader_event::kResidentEpochCount);
}

TEST_F(Pm4ShaderEventTest, StopsResidentPublisherAfterHostObservation) {
  RunResidentPublisher(3);
}

}  // namespace
