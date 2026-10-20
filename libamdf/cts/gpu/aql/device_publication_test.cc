// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/aql_publish.h"
#include "libamdf/cts/gpu/kernels/aql_publish_kernels.h"
#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"

namespace {

enum class Publication {
  kReservedSuccessor,
  kDeviceDoorbell,
};

class AqlDevicePublicationTest
    : public AqlDispatchTest,
      public ::testing::WithParamInterface<Publication> {};

TEST_P(AqlDevicePublicationTest,
       DevicePublishesIndirectDispatchAcrossRingWrap) {
  if ((family_.user_queue_capabilities &
       AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER) == 0) {
    GTEST_SKIP() << "AQL family has no device producer mapping";
  }
  if (local_scope_ == nullptr) {
    GTEST_SKIP() << "GPU has no local memory for the immutable program";
  }
  const auto* publish_kernel =
      kernels::aql_publish::kKernels.Find(gpu_endpoint_info_);
  const auto* transform_kernel =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(publish_kernel, nullptr);
  ASSERT_NE(transform_kernel, nullptr);
  ASSERT_EQ(publish_kernel->private_segment_byte_length, 0u);
  ASSERT_EQ(publish_kernel->workgroup_size(), 1u);
  const bool device_doorbell = GetParam() == Publication::kDeviceDoorbell;

  // The program is uploaded once into local memory. Neither the host nor the
  // device reads the native ring to recover packet fields on any invocation.
  const amdf_memory_device_access_t attachment = {
      device_,
      {
          .access = AMDF_MEMORY_ACCESS_READ,
          .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      }};
  amdf_memory_create_info_t creation = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(creation),
      .access_count = 1,
      .required_flags =
          AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .byte_length = 4096,
      .minimum_alignment = 4096,
      .accesses = &attachment,
  };
  creation.memory_profile_ordinal = FindMemoryProfileOrdinal(
      local_scope_,
      AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      creation.required_flags, attachment.requirements);
  if (creation.memory_profile_ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
    GTEST_SKIP() << "GPU has no host-visible local program profile";
  }
  GpuMemory* program = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(local_scope_, creation, &program));
  ASSERT_EQ(program->info.memory_class, AMDF_MEMORY_CLASS_LOCAL);

  GpuMemory* input = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* control = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &input));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &output));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &control));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
  auto* signals = static_cast<aql::Signal*>(completion->host.pointer);
  signals[0] = aql::UserSignal();
  signals[1] = aql::UserSignal();
  std::memset(control->host.pointer, 0, 4096);
  std::memset(arguments->host.pointer, 0, 4096);

  GpuUserQueue* consumer = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateQueue(family_, &consumer, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
                  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                      AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER,
                  4096));
  GpuUserQueue* producer = consumer;
  if (device_doorbell) {
    ASSERT_NO_FATAL_FAILURE(CreateQueue(&producer));
  }
  uint64_t consumer_index = 0;
  uint64_t producer_index = 0;
  uint64_t target_descriptor = 0;
  uint64_t counts_descriptor = 0;
  uint64_t publish_descriptor = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*consumer, *transform_kernel, "target",
                                        &consumer_index, &target_descriptor));
  counts_descriptor = target_descriptor;
  if (device_doorbell) {
    ASSERT_NO_FATAL_FAILURE(PublishKernel(*producer, *transform_kernel,
                                          "counts", &producer_index,
                                          &counts_descriptor));
  } else {
    producer_index = consumer_index;
  }
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*producer, *publish_kernel, "publisher",
                                        &producer_index, &publish_descriptor));
  if (!device_doorbell) {
    consumer_index = producer_index;
  }

  constexpr uint32_t kWordCount = 1024;
  constexpr uint32_t kGuard = 0xa59c710fu;
  constexpr uint32_t kAddend = 37;
  std::array<uint32_t, kWordCount> source;
  for (uint32_t word = 0; word < kWordCount; ++word) {
    source[word] = 0xf13b7245u + word * 0x10203u;
  }
  std::memcpy(input->host.pointer, source.data(), sizeof(source));
  const kernels::transform::Arguments counts_arguments = {
      control->device_address, control->device_address + 64, 3, 1};
  const kernels::transform::Arguments target_arguments = {
      input->device_address, output->device_address, kWordCount, kAddend};
  auto* argument_bytes = static_cast<uint8_t*>(arguments->host.pointer);
  std::memcpy(argument_bytes, &counts_arguments, sizeof(counts_arguments));
  std::memcpy(argument_bytes + 64, &target_arguments, sizeof(target_arguments));
  const uint16_t workgroup_size = transform_kernel->workgroup_size();
  ASSERT_LE(13u * workgroup_size, kWordCount);
  const auto target = aql::Dispatch(
      aql::HeaderBarrier::kDisabled, {1, {workgroup_size, 1, 1}, {0, 0, 0}},
      transform_kernel->private_segment_byte_length,
      transform_kernel->group_segment_byte_length, target_descriptor,
      arguments->device_address + 64,
      completion->device_address + sizeof(aql::Signal));
  std::memcpy(program->host.pointer, target.data(), sizeof(target));
  ASSERT_NO_FATAL_FAILURE(program->TransferHostCache(
      api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));

  const uint64_t capacity =
      consumer->host.ring_byte_length / sizeof(aql::Packet);
  const uint32_t round_count = static_cast<uint32_t>(capacity * 2 + 3);
  for (uint32_t round = 0; round < round_count; ++round) {
    SCOPED_TRACE(round);
    const uint32_t seed = round % 5;
    std::memcpy(control->host.pointer, &seed, sizeof(seed));
    auto* output_words = static_cast<uint32_t*>(output->host.pointer);
    std::fill(output_words, output_words + kWordCount, kGuard);
    signals[0].value = 2;
    signals[1].value = 1;
    const uint64_t target_index =
        device_doorbell ? consumer_index : producer_index + 2;
    const auto& mapping = consumer->producer.info;
    const kernels::aql_publish::Arguments publish_arguments = {
        mapping.ring_address + (target_index & (capacity - 1)) * 64,
        program->device_address,
        control->device_address + 64,
        mapping.write_index_address,
        mapping.doorbell_address,
        target_index,
        device_doorbell ? 1u : 0u,
    };
    std::memcpy(argument_bytes + 128, &publish_arguments,
                sizeof(publish_arguments));
    const auto counts = aql::Dispatch(
        aql::HeaderBarrier::kDisabled,
        {1, {workgroup_size, 1, 1}, {workgroup_size, 1, 1}},
        transform_kernel->private_segment_byte_length,
        transform_kernel->group_segment_byte_length, counts_descriptor,
        arguments->device_address, completion->device_address);
    const auto publish = aql::Dispatch(
        aql::HeaderBarrier::kEnabled, {1, {1, 1, 1}, {1, 1, 1}}, 0,
        publish_kernel->group_segment_byte_length, publish_descriptor,
        arguments->device_address + 128, completion->device_address);
    const uint64_t producer_end = producer_index + (device_doorbell ? 2 : 3);
    GpuStoreRelease(producer->host.write_index_address, producer_end);
    ASSERT_TRUE(aql::Publish(*producer, producer_index++, counts));
    ASSERT_TRUE(aql::Publish(*producer, producer_index++, publish));
    if (!device_doorbell) {
      // Only the GPU writes this slot, including its valid header. Publishing
      // the tail doorbell may expose the INVALID slot before the shader runs.
      GpuRingDoorbell(producer->host.doorbell_address, producer_index++);
    }
    consumer_index = target_index + 1;
    ASSERT_NO_FATAL_FAILURE(
        WaitCompletionAndConsumption(*consumer, signals[1], consumer_index));
    ASSERT_NO_FATAL_FAILURE(
        WaitCompletionAndConsumption(*producer, signals[0], producer_index));
    const uint32_t active_words = (seed * 3 + 1) * workgroup_size;
    for (uint32_t word = 0; word < kWordCount; ++word) {
      const uint32_t expected =
          word < active_words ? source[word] * 3 + kAddend : kGuard;
      EXPECT_EQ(output_words[word], expected) << "word=" << word;
    }
    if (HasFailure()) {
      return;
    }
  }
  RecordProperty("aql_device_publication_rounds", round_count);
  RecordProperty("aql_ring_capacity", std::to_string(capacity));
}

INSTANTIATE_TEST_SUITE_P(Publication, AqlDevicePublicationTest,
                         ::testing::Values(Publication::kReservedSuccessor,
                                           Publication::kDeviceDoorbell),
                         [](const ::testing::TestParamInfo<Publication>& info) {
                           return info.param == Publication::kReservedSuccessor
                                      ? "ReservedSuccessor"
                                      : "DeviceDoorbell";
                         });

}  // namespace
