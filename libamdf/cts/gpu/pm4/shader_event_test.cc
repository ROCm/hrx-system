// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/kernels/shader_event.h"

#include <array>
#include <cstring>

#include "libamdf/cts/gpu/kernels/shader_event_kernels.h"
#include "libamdf/cts/gpu/pm4/dispatch_fixture.h"
#include "libamdf/cts/gpu/util/shader_event_fixture.h"

namespace {

using Pm4ShaderEventTest = GpuShaderEventTest<Pm4DispatchTest>;

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
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &records,
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096,
                                       &arguments,
                                       AMDF_MEMORY_FLAG_HOST_COHERENT));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion,
      AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_QUEUE_STORAGE));
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
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue, 65536));
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

}  // namespace
