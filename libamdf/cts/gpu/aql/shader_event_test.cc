// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/kernels/shader_event.h"

#include <array>
#include <cstring>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/shader_event_kernels.h"
#include "libamdf/cts/gpu/util/shader_event_fixture.h"

namespace {

using AqlShaderEventTest = GpuShaderEventTest<AqlDispatchTest>;

TEST_F(AqlShaderEventTest, PublishesAndWakesWithoutCompletionSignalEvents) {
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
  const uint64_t record_address =
      records->device_address + sizeof(protocol::Record);
  const uint64_t host_generation =
      reinterpret_cast<uintptr_t>(records->host.pointer) +
      sizeof(protocol::Record) + offsetof(protocol::Record, generation);
  // Firmware completion has no event mailbox and cannot wake the shader event.
  auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
  signal = aql::UserSignal();
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  uint64_t index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*queue, *kernel, "shader_event", &index,
                                        &descriptor_address));

  constexpr uint32_t kEpochCount = 8;
  for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
    SCOPED_TRACE(epoch);
    const uint64_t generation = UINT64_C(0x100000000) + epoch;
    const protocol::Arguments payload = {
        record_address, event_info_.mailbox_address, generation,
        event_info_.trigger_data, event_info_.interrupt_context};
    std::memcpy(arguments->host.pointer, &payload, sizeof(payload));
    signal.value = 1;
    // Release NONE keeps packet completion from supplying the shader's release.
    const auto packet =
        aql::Dispatch(aql::HeaderBarrier::kDisabled, {1, {1, 1, 1}, {1, 1, 1}},
                      0, 0, descriptor_address, arguments->device_address,
                      completion->device_address,
                      {aql::FenceScope::kSystem, aql::FenceScope::kNone});
    GpuStoreRelease(queue->host.write_index_address, index + 1);
    ASSERT_TRUE(aql::Publish(*queue, index++, packet));

    const auto observed = WaitForRecord(host_generation, generation);
    EXPECT_TRUE(observed);
    if (observed) {
      expected[1] = {generation ^ protocol::kPayloadMask, generation};
      EXPECT_EQ(
          std::memcmp(records->host.pointer, expected.data(), sizeof(expected)),
          0);
    }
    // Completion and consumption protect reuse, after the shader-only oracle.
    ASSERT_NO_FATAL_FAILURE(
        WaitCompletionAndConsumption(*queue, signal, index));
    if (HasFailure()) {
      return;
    }
  }
  RecordProperty("shader_event_epochs", kEpochCount);
}

}  // namespace
