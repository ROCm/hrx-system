// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/private_roundtrip.h"
#include "libamdf/cts/gpu/kernels/private_roundtrip_kernels.h"
#include "libamdf/cts/gpu/util/event_fixture.h"

namespace {

class AqlScratchServiceTest : public GpuEventTest<AqlDispatchTest>,
                              public ::testing::WithParamInterface<bool> {
 protected:
  void CreateScratch(uint64_t byte_length, GpuMemory** out_memory) {
    const amdf_memory_device_access_t attachment = {
        device_,
        {
            .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
        }};
    const uint32_t profile = FindGpuMemoryProfileOrdinal(
        api_, local_scope_, device_, AMDF_MEMORY_PROFILE_ROLE_CREATE,
        AMDF_MEMORY_FLAG_DEVICE_LOCAL, attachment.requirements);
    ASSERT_NE(profile, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    const amdf_memory_create_info_t creation = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(creation),
        .memory_profile_ordinal = profile,
        .access_count = 1,
        .required_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL,
        .byte_length = byte_length,
        .minimum_alignment = 4096,
        .accesses = &attachment,
    };
    ASSERT_NO_FATAL_FAILURE(CreateMemory(local_scope_, creation, out_memory));
  }
};

TEST_P(AqlScratchServiceTest, FirmwareRequestsBackingAndReleasesPreviousRange) {
  if ((family_.format_features & AMDF_GPU_AQL_FORMAT_FEATURE_SCRATCH_SERVICE) ==
      0) {
    GTEST_SKIP() << "queue family has no caller-serviced scratch";
  }
  if (local_scope_ == nullptr) {
    GTEST_SKIP() << "GPU has no local-memory scope";
  }
  const auto* kernel =
      kernels::private_roundtrip::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel, nullptr);
  ASSERT_GT(kernel->private_segment_byte_length, 16u);
  constexpr uint32_t kGridSize = 512;
  constexpr uint32_t kOutputWords = kGridSize * 9;
  constexpr uint32_t kGuardWords = 16;
  constexpr uint32_t kGuard = 0xa59e6703;
  constexpr uint32_t kWords = kOutputWords + 2 * kGuardWords;

  GpuMemory* output = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, kWords * 4, &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
  auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
  signal = aql::UserSignal();
  signal.event.mailbox_address = event_info_.mailbox_address;
  signal.event.id = event_info_.interrupt_context;

  // The optional initial allocation fits a small private kernel. The actual
  // published kernel legally asks firmware for a larger private segment.
  GpuMemory* scratch_memory = nullptr;
  amdf_gpu_queue_scratch_t initial = {};
  if (GetParam()) {
    const auto& topology = gpu_endpoint_info_.topology;
    const auto& compute = gpu_endpoint_info_.compute;
    const uint32_t engines =
        topology.xcc_count * topology.shader_engine_count_per_xcc;
    const uint32_t slots =
        ((compute.compute_unit_count + engines - 1) / engines) * engines *
        compute.maximum_scratch_wave_count_per_compute_unit;
    const uint64_t bytes = uint64_t{slots} * 1024;
    ASSERT_NO_FATAL_FAILURE(CreateScratch(bytes, &scratch_memory));
    initial = {
        .memory = scratch_memory->memory,
        .byte_length = bytes,
        .maximum_private_segment_byte_length = 16,
        .maximum_wave_count = slots,
    };
  }
  const amdf_gpu_aql_scratch_service_info_t service = {
      .type = AMDF_STRUCTURE_TYPE_GPU_AQL_SCRATCH_SERVICE_INFO,
      .structure_size = sizeof(service),
      .event = event_,
  };
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateQueue(family_, &queue, AMDF_QUEUE_PRODUCER_MODE_SINGLE, initial,
                  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER, 4096, 0, &service));
  uint64_t index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*queue, *kernel, "private_kernel",
                                        &index, &descriptor_address));

  amdf_gpu_aql_scratch_request_t request = {
      .type = AMDF_STRUCTURE_TYPE_GPU_AQL_SCRATCH_REQUEST,
      .structure_size = sizeof(request),
  };
  ASSERT_EQ(gpu_api_->user_queue_query_scratch(queue->queue, &request),
            AMDF_STATUS_OK);
  ASSERT_EQ(request.byte_length, 0u);
  for (uint32_t round = 0; round < 5; ++round) {
    SCOPED_TRACE(round);
    const uint32_t seed = 0x13579bdf + round;
    const uint32_t rotation = 1 + round;
    const uint32_t private_bytes =
        kernel->private_segment_byte_length * (round == 0 ? 1 : 3);
    auto* words = static_cast<uint32_t*>(output->host.pointer);
    for (uint32_t i = 0; i < kWords; ++i) {
      words[i] = kGuard;
    }
    const kernels::private_roundtrip::Arguments payload = {
        output->device_address + kGuardWords * 4, seed, rotation};
    std::memcpy(arguments->host.pointer, &payload, sizeof(payload));
    signal.value = 1;
    const auto packet = aql::Dispatch(
        aql::HeaderBarrier::kDisabled,
        {1,
         {static_cast<uint16_t>(kernel->workgroup_size()), 1, 1},
         {kGridSize, 1, 1}},
        private_bytes, kernel->group_segment_byte_length, descriptor_address,
        arguments->device_address, completion->device_address,
        {aql::FenceScope::kSystem, aql::FenceScope::kSystem});
    const uint64_t dispatch_index = index++;
    GpuStoreRelease(queue->host.write_index_address, index);
    ASSERT_TRUE(aql::Publish(*queue, dispatch_index, packet));

    uint32_t requests = 0;
    do {
      // Both request and completion use the event. Only acquiring their
      // respective predicates determines which ownership transition occurred.
      ASSERT_EQ(gpu_api_->event_wait(event_, AMDF_TIMEOUT_INFINITE),
                AMDF_STATUS_OK);
      ASSERT_EQ(gpu_api_->user_queue_query_scratch(queue->queue, &request),
                AMDF_STATUS_OK);
      if (request.byte_length == 0) {
        continue;
      }
      ++requests;
      ASSERT_EQ(request.dispatch_index, dispatch_index);
      ASSERT_EQ(request.private_segment_byte_length, private_bytes);
      ASSERT_EQ(request.wavefront_size, kernel->wavefront_size);
      if (scratch_memory != nullptr) {
        // Firmware has surrendered the old range. Release before allocating
        // replacement to exercise this boundary without double backing.
        ASSERT_TRUE(scratch_memory->Release(api_));
      }
      ASSERT_NO_FATAL_FAILURE(
          CreateScratch(request.byte_length, &scratch_memory));
      const amdf_gpu_queue_scratch_t scratch = {
          .memory = scratch_memory->memory,
          .byte_length = request.byte_length,
          .maximum_private_segment_byte_length = private_bytes,
          .maximum_wave_count = request.wave_count,
      };
      ASSERT_EQ(gpu_api_->user_queue_resume_scratch(queue->queue, &scratch),
                AMDF_STATUS_OK);
    } while (GpuLoadAcquire<int64_t>(
                 reinterpret_cast<uintptr_t>(&signal.value)) != 0);

    EXPECT_EQ(requests, round < 2 || round == 3 ? 1u : 0u);
    for (uint32_t workitem = 0; workitem < kGridSize; ++workitem) {
      const auto expected =
          kernels::private_roundtrip::ExpectedRecord(workitem, seed, rotation);
      for (uint32_t word = 0; word < 9; ++word) {
        EXPECT_EQ(words[kGuardWords + workitem * 9 + word], expected[word]);
      }
    }
    for (uint32_t i = 0; i < kGuardWords; ++i) {
      EXPECT_EQ(words[i], kGuard);
      EXPECT_EQ(words[kGuardWords + kOutputWords + i], kGuard);
    }
    ASSERT_NO_FATAL_FAILURE(queue->WaitConsumed(api_, index));
    if (HasFailure()) {
      return;
    }
    if (round == 2) {
      // Completion and consumption are both joined, and this fixture excludes
      // publication. Detach firmware's cached binding before freeing it.
      ASSERT_EQ(gpu_api_->user_queue_trim_scratch(queue->queue),
                AMDF_STATUS_OK);
      ASSERT_TRUE(scratch_memory->Release(api_));
      scratch_memory = nullptr;
      ASSERT_EQ(gpu_api_->user_queue_trim_scratch(queue->queue),
                AMDF_STATUS_OK);
    }
  }
  RecordProperty("scratch_kernel_target", kernel->target);
  RecordProperty("scratch_wavefront_size", kernel->wavefront_size);
  RecordProperty("scratch_growth_count", 3);
  RecordProperty("scratch_trim_count", 1);
}

INSTANTIATE_TEST_SUITE_P(InitialBacking, AqlScratchServiceTest,
                         ::testing::Bool(), [](const auto& info) {
                           return info.param ? "Smaller" : "Empty";
                         });

}  // namespace
