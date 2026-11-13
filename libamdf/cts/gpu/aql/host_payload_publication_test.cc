// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"

namespace {

// Cache bypass is part of the protocol. Exact instructions keep optimization
// from replacing advisory non-temporal metadata with ordinary stores.
void StoreNonTemporal(uint32_t* target, uint32_t value) {
#if defined(__GNUC__) || defined(__clang__)
  __asm__ __volatile__("movnti %1, %0" : "=m"(*target) : "r"(value));
#else
  _mm_stream_si32(reinterpret_cast<int*>(target), static_cast<int>(value));
#endif
}

TEST_F(AqlDispatchTest, NonTemporalLocalPayloadPublicationAcrossReuse) {
  if ((family_.user_queue_capabilities &
       AMDF_USER_QUEUE_CAPABILITY_HOST_NON_TEMPORAL_PUBLICATION) == 0) {
    GTEST_SKIP() << "queue family has no non-temporal payload protocol";
  }
  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(
      GpuUserQueueOptions{
          .required_capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER,
          .ring_byte_length = 4096,
          .placement = AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL,
      },
      &queue));
  if ((queue->info.capabilities &
       AMDF_USER_QUEUE_CAPABILITY_HOST_NON_TEMPORAL_PUBLICATION) == 0) {
    GTEST_SKIP() << "live queue has no LOCAL payload publication route";
  }
  ASSERT_EQ(queue->info.placement, AMDF_QUEUE_PLACEMENT_DEVICE_LOCAL);
  ASSERT_NE(local_scope_, nullptr);
  const auto* kernel = kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel, nullptr);
  constexpr size_t kArgumentByteLength = 24;
  ASSERT_EQ(kernel->arguments.byte_length, kArgumentByteLength);
  constexpr uint32_t kDispatchCount = 4097;
  constexpr uint32_t kBatchCount = 16;
  constexpr uint32_t kArgumentStride = 64;
  constexpr uint32_t kLaneCount = 64;
  constexpr uint32_t kGuard = 0x759bf13du;
  const amdf_memory_device_access_t attachment = {
      device_,
      {
          .access = AMDF_MEMORY_ACCESS_READ,
          .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      },
  };
  const amdf_memory_flags_t flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_LOCAL;
  const uint32_t ordinal = FindMemoryProfileOrdinal(
      local_scope_,
      AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      flags, attachment.requirements);
  ASSERT_NE(ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  const amdf_memory_create_info_t creation = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(creation),
      .memory_profile_ordinal = ordinal,
      .access_count = 1,
      .required_flags = flags,
      .byte_length = 4096,
      .minimum_alignment = 4096,
      .accesses = &attachment,
  };
  // Independent backing generations exercise the conditional payload contract;
  // neither their CPU addresses nor residency are inferred from the ring.
  std::array<GpuMemory*, 2> arguments = {};
  for (auto*& memory : arguments) {
    ASSERT_NO_FATAL_FAILURE(CreateMemory(local_scope_, creation, &memory));
    ASSERT_EQ(memory->info.memory_class, AMDF_MEMORY_CLASS_LOCAL);
    ASSERT_EQ(memory->device_address % kernel->arguments.alignment, 0u);
  }
  GpuMemory* input = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ, 2 * kLaneCount * sizeof(uint32_t), &input));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      (kDispatchCount * kLaneCount + 2) * sizeof(uint32_t), &output));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
  auto* input_words = static_cast<uint32_t*>(input->host.pointer);
  auto* output_words = static_cast<uint32_t*>(output->host.pointer);
  for (uint32_t i = 0; i < 2 * kLaneCount; ++i) {
    input_words[i] = 0xfffffff0u + i * 0x01030507u;
  }
  std::fill_n(output_words, kDispatchCount * kLaneCount + 2, kGuard);
  auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
  signal = aql::UserSignal();

  const uint64_t capacity = queue->host.ring_byte_length / sizeof(aql::Packet);
  ASSERT_EQ(capacity, 64u);
  uint64_t index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*queue, *kernel, "payload_kernel",
                                        &index, &descriptor_address));
  const uint64_t first_index = index;
  auto* ring = reinterpret_cast<uint32_t*>(queue->host.ring_address);
  for (uint32_t first = 0; first < kDispatchCount; first += kBatchCount) {
    const uint32_t count = std::min(kBatchCount, kDispatchCount - first);
    signal.value = 1;
    GpuStoreRelease(queue->host.write_index_address, index + count);
    uint32_t first_header = 0;
    for (uint32_t slot = 0; slot < count; ++slot) {
      const uint32_t launch = first + slot;
      const auto* memory = arguments[(first / kBatchCount + slot) % 2];
      const uint32_t offset = slot * kArgumentStride;
      const kernels::transform::Arguments payload = {
          input->device_address + (launch % 2) * kLaneCount * sizeof(uint32_t),
          output->device_address + (1 + launch * kLaneCount) * sizeof(uint32_t),
          33 + launch % 32,
          0x80000023u + launch * 0x110101u,
      };
      std::array<uint32_t, kArgumentByteLength / sizeof(uint32_t)> words;
      std::memcpy(words.data(), &payload, kArgumentByteLength);
      auto* target = reinterpret_cast<uint32_t*>(
          static_cast<uint8_t*>(memory->host.pointer) + offset);
      for (size_t i = 0; i < words.size(); ++i) {
        StoreNonTemporal(target + i, words[i]);
      }
      // The first SYSTEM acquire covers all payloads prepared behind its
      // INVALID header. Header barriers order subsequent consumers, and the
      // final completion retires every payload before any CPU reuse.
      const auto packet = aql::Dispatch(
          aql::HeaderBarrier::kEnabled,
          {1,
           {static_cast<uint16_t>(kernel->workgroup_size()), 1, 1},
           {kLaneCount, 1, 1}},
          kernel->private_segment_byte_length,
          kernel->group_segment_byte_length, descriptor_address,
          memory->device_address + offset,
          slot + 1 == count ? completion->device_address : 0,
          {slot == 0 ? aql::FenceScope::kSystem : aql::FenceScope::kAgent,
           slot + 1 == count ? aql::FenceScope::kSystem
                             : aql::FenceScope::kAgent});
      auto* packet_words = ring + ((index + slot) & (capacity - 1)) * 16;
      for (uint32_t i = slot == 0 ? 1 : 0; i < packet.size(); ++i) {
        StoreNonTemporal(packet_words + i, packet[i]);
      }
      if (slot == 0) {
        first_header = packet[0];
      }
    }
    _mm_sfence();
    StoreNonTemporal(ring + (index & (capacity - 1)) * 16, first_header);
    index += count;
    GpuRingDoorbell(queue->host.doorbell_address, index - 1);
    ASSERT_TRUE(
        GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0));
    for (uint32_t slot = 0; slot < count; ++slot) {
      const uint32_t launch = first + slot;
      for (uint32_t lane = 0; lane < kLaneCount; ++lane) {
        const uint32_t input_index = (launch % 2) * kLaneCount + lane;
        const uint32_t value = 0xfffffff0u + input_index * 0x01030507u;
        const uint32_t expected =
            lane < 33 + launch % 32
                ? static_cast<uint32_t>(uint64_t{value} * 3 + 0x80000023u +
                                        launch * 0x110101u)
                : kGuard;
        EXPECT_EQ(output_words[1 + launch * kLaneCount + lane], expected)
            << "launch=" << launch << " lane=" << lane;
      }
    }
    EXPECT_EQ(output_words[0], kGuard);
    EXPECT_EQ(output_words[kDispatchCount * kLaneCount + 1], kGuard);
    EXPECT_NO_FATAL_FAILURE(queue->WaitConsumed(api_, index));
    if (HasFailure()) {
      return;
    }
  }
  for (uint32_t i = 0; i < 2 * kLaneCount; ++i) {
    EXPECT_EQ(input_words[i], 0xfffffff0u + i * 0x01030507u);
  }
  RecordProperty("payload_dispatch_count", kDispatchCount);
  RecordProperty("payload_backing_count", arguments.size());
  RecordProperty("payload_argument_byte_length", kernel->arguments.byte_length);
  RecordProperty("payload_packet_ring_wraps", (index - first_index) / capacity);
}

}  // namespace
