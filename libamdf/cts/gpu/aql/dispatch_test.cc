// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <string>

#include "libamdf/cts/gpu/aql/dispatch_fixture.h"
#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"

namespace {

void CheckDeviceLocalIngressPair(const amdf_memory_pair_info_t& pair) {
  ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
  ASSERT_EQ(pair.release.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
  ASSERT_EQ(pair.release.executor, AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API);
  ASSERT_EQ(pair.release.host_operation, AMDF_HOST_CACHE_OPERATION_FLUSH);
  ASSERT_EQ(pair.release.host_instruction, AMDF_HOST_CACHE_INSTRUCTION_NONE);
  ASSERT_EQ(pair.release.host_fence_before, AMDF_HOST_CACHE_FENCE_NONE);
  ASSERT_EQ(pair.release.host_fence_after, AMDF_HOST_CACHE_FENCE_NONE);
  ASSERT_EQ(pair.acquire.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
  ASSERT_EQ(pair.acquire.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
  ASSERT_EQ(pair.acquire.operation, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM);
}

TEST_F(AqlDispatchTest, CoherentSystemPayloadChangesAcrossEpochs) {
  const auto* kernel_product =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel_product, nullptr)
      << "missing compiled transform kernel for endpoint";
  const auto& kernel = *kernel_product;
  RecordProperty("transform_kernel_target", kernel.target);

  constexpr uint32_t kGridSize = 1024;
  constexpr uint32_t kWordCount = 2048;
  constexpr uint32_t kPayloadOffset = 16;
  constexpr uint32_t kGuard = 0x759bf13du;
  constexpr std::array<uint32_t, 2> kCounts = {1003, 997};
  constexpr std::array<uint32_t, 2> kAddends = {7, 0x80000023u};
  GpuMemory* input = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ,
                                       kWordCount * sizeof(uint32_t), &input));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kWordCount * sizeof(uint32_t), &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
  std::memset(completion->host.pointer, 0, completion->info.byte_length);
  auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
  signal = aql::UserSignal();

  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  uint64_t index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(
      PublishKernel(*queue, kernel, "aql_kernel", &index, &descriptor_address));

  for (uint32_t epoch = 0; epoch < kCounts.size(); ++epoch) {
    std::array<uint32_t, kWordCount> upload;
    std::array<uint32_t, kWordCount> expected;
    std::array<uint32_t, kWordCount> download;
    std::array<uint32_t, kWordCount> unchanged_input;
    upload.fill(kGuard);
    expected.fill(kGuard);
    download.fill(kGuard);
    for (uint32_t i = 0; i < kGridSize; ++i) {
      const uint32_t value =
          0xfffffff0u + i * 0x01030507u + epoch * 0x11111111u;
      upload[kPayloadOffset + i] = value;
      if (i < kCounts[epoch]) {
        // The CPU oracle uses wider arithmetic, then applies uint32 wrapping.
        const uint32_t result =
            static_cast<uint32_t>(uint64_t{value} * 3 + kAddends[epoch]);
        expected[kPayloadOffset + i] = result;
        download[kPayloadOffset + i] = ~result;
      }
    }
    std::memcpy(input->host.pointer, upload.data(), sizeof(upload));
    std::memcpy(output->host.pointer, download.data(), sizeof(download));
    const kernels::transform::Arguments payload = {
        input->device_address + kPayloadOffset * sizeof(uint32_t),
        output->device_address + kPayloadOffset * sizeof(uint32_t),
        kCounts[epoch],
        kAddends[epoch],
    };
    std::memset(arguments->host.pointer, 0, arguments->info.byte_length);
    std::memcpy(arguments->host.pointer, &payload,
                kernel.arguments.byte_length);
    signal.value = 1;
    const auto packet =
        aql::Dispatch(aql::HeaderBarrier::kDisabled,
                      {1,
                       {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
                       {kGridSize, 1, 1}},
                      kernel.private_segment_byte_length,
                      kernel.group_segment_byte_length, descriptor_address,
                      arguments->device_address, completion->device_address);
    GpuStoreRelease(queue->host.write_index_address, index + 1);
    ASSERT_TRUE(aql::Publish(*queue, index++, packet));
    // Snapshot completion-visible data before diagnostics or ring retirement.
    ASSERT_TRUE(
        GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0));
    std::memcpy(download.data(), output->host.pointer, sizeof(download));
    std::memcpy(unchanged_input.data(), input->host.pointer,
                sizeof(unchanged_input));
    for (uint32_t i = 0; i < kWordCount; ++i) {
      EXPECT_EQ(download[i], expected[i]) << "epoch=" << epoch << " word=" << i;
      EXPECT_EQ(unchanged_input[i], upload[i])
          << "epoch=" << epoch << " word=" << i;
    }
    // Retire even after a failed oracle, before any next-epoch storage reuse.
    EXPECT_NO_FATAL_FAILURE(queue->WaitConsumed(api_, index));
    if (HasFailure()) {
      return;
    }
  }
  RecordProperty("aql_payload_completed_epochs", kCounts.size());
}

TEST_F(AqlDispatchTest, HostPublishesDeviceLocalKernargsAcrossEpochs) {
  if (local_scope_ == nullptr) {
    GTEST_SKIP() << "GPU has no local-memory scope";
  }
  const auto* kernel_product =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel_product, nullptr)
      << "missing compiled transform kernel for endpoint";
  const auto& kernel = *kernel_product;
  RecordProperty("transform_kernel_target", kernel.target);

  const amdf_memory_device_access_t attachment = {
      device_,
      {
          .access = AMDF_MEMORY_ACCESS_READ,
          .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
      }};
  const amdf_memory_flags_t required_flags =
      AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_LOCAL;
  amdf_memory_scope_info_t scope_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
      .structure_size = sizeof(scope_info),
  };
  ASSERT_EQ(api_->memory_scope_query_info(local_scope_, &scope_info),
            AMDF_STATUS_OK);
  uint32_t profile_ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
  for (uint32_t ordinal = 0; ordinal < scope_info.memory_profile_count;
       ++ordinal) {
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    const amdf_status_t status =
        QueryMemoryProfile(local_scope_, ordinal, attachment.requirements,
                           &profile, &capabilities);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      continue;
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    if ((profile.roles & (AMDF_MEMORY_PROFILE_ROLE_CREATE |
                          AMDF_MEMORY_PROFILE_ROLE_HOST_MAP)) ==
            (AMDF_MEMORY_PROFILE_ROLE_CREATE |
             AMDF_MEMORY_PROFILE_ROLE_HOST_MAP) &&
        (required_flags & ~profile.supported_flags) == 0) {
      profile_ordinal = ordinal;
      break;
    }
  }
  if (profile_ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
    GTEST_SKIP() << "GPU has no CPU-visible local-memory profile";
  }

  amdf_memory_create_info_t creation = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(creation),
      .memory_profile_ordinal = profile_ordinal,
      .access_count = 1,
      .required_flags = required_flags,
      .byte_length = 4096,
      .minimum_alignment = 4096,
      .accesses = &attachment,
  };
  const amdf_memory_profile_pair_query_t profile_query = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
      .structure_size = sizeof(profile_query),
      .memory_profile_ordinal = profile_ordinal,
      .access_count = 1,
      .required_flags = required_flags,
      .accesses = &attachment,
      .producer =
          {
              .kind = AMDF_MEMORY_SITE_KIND_HOST,
              .value = {.host_access = AMDF_MEMORY_MAP_FLAG_WRITE},
          },
      .consumer =
          {
              .kind = AMDF_MEMORY_SITE_KIND_DEVICE,
              .value =
                  {
                      .device =
                          {
                              .access_ordinal = 0,
                              .queue_family_ordinal = family_.ordinal,
                          },
                  },
          },
  };
  amdf_memory_pair_info_t prospective = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(prospective),
  };
  const amdf_status_t prospective_status = api_->memory_scope_query_pair_info(
      local_scope_, &profile_query, &prospective);
  if (prospective_status ==
      amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    GTEST_SKIP() << "GPU has no qualified host-to-local publication path";
  }
  ASSERT_EQ(prospective_status, AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(CheckDeviceLocalIngressPair(prospective));

  GpuMemory* arguments = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(local_scope_, creation, &arguments));
  ASSERT_EQ(arguments->info.memory_class, AMDF_MEMORY_CLASS_LOCAL);
  ASSERT_EQ(arguments->host.cacheability,
            AMDF_HOST_CACHEABILITY_WRITE_COMBINED);
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
  const auto argument_host = arguments->HostSite();
  const auto argument_device = arguments->DeviceSite(family_.ordinal);
  amdf_memory_pair_info_t concrete = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(concrete),
  };
  ASSERT_EQ(
      api_->memory_query_pair_info(&argument_host, &argument_device, &concrete),
      AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(CheckDeviceLocalIngressPair(concrete));

  constexpr uint32_t kGridSize = 1024;
  constexpr uint32_t kWordCount = 2048;
  constexpr uint32_t kPayloadOffset = 16;
  constexpr uint32_t kGuard = 0x759bf13du;
  constexpr std::array<uint32_t, 2> kCounts = {1003, 997};
  constexpr std::array<uint32_t, 2> kAddends = {7, 0x80000023u};
  GpuMemory* input = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ,
                                       kWordCount * sizeof(uint32_t), &input));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kWordCount * sizeof(uint32_t), &output));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
  std::memset(completion->host.pointer, 0, completion->info.byte_length);
  auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
  signal = aql::UserSignal();

  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  uint64_t index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(
      PublishKernel(*queue, kernel, "aql_kernel", &index, &descriptor_address));

  for (uint32_t epoch = 0; epoch < kCounts.size(); ++epoch) {
    std::array<uint32_t, kWordCount> upload;
    std::array<uint32_t, kWordCount> expected;
    std::array<uint32_t, kWordCount> download;
    upload.fill(kGuard);
    expected.fill(kGuard);
    download.fill(kGuard);
    for (uint32_t i = 0; i < kGridSize; ++i) {
      const uint32_t value =
          0xfffffff0u + i * 0x01030507u + epoch * 0x11111111u;
      upload[kPayloadOffset + i] = value;
      if (i < kCounts[epoch]) {
        const uint32_t result =
            static_cast<uint32_t>(uint64_t{value} * 3 + kAddends[epoch]);
        expected[kPayloadOffset + i] = result;
        download[kPayloadOffset + i] = ~result;
      }
    }
    std::memcpy(input->host.pointer, upload.data(), sizeof(upload));
    std::memcpy(output->host.pointer, download.data(), sizeof(download));
    const kernels::transform::Arguments payload = {
        input->device_address + kPayloadOffset * sizeof(uint32_t),
        output->device_address + kPayloadOffset * sizeof(uint32_t),
        kCounts[epoch],
        kAddends[epoch],
    };
    std::memset(arguments->host.pointer, 0, arguments->info.byte_length);
    std::memcpy(arguments->host.pointer, &payload,
                kernel.arguments.byte_length);
    ASSERT_EQ(api_->host_mapping_cache_control(arguments->mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, kernel.arguments.byte_length),
              AMDF_STATUS_OK);
    signal.value = 1;
    const auto packet =
        aql::Dispatch(aql::HeaderBarrier::kDisabled,
                      {1,
                       {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
                       {kGridSize, 1, 1}},
                      kernel.private_segment_byte_length,
                      kernel.group_segment_byte_length, descriptor_address,
                      arguments->device_address, completion->device_address);
    GpuStoreRelease(queue->host.write_index_address, index + 1);
    ASSERT_TRUE(aql::Publish(*queue, index++, packet));
    ASSERT_TRUE(
        GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0));
    std::memcpy(download.data(), output->host.pointer, sizeof(download));
    for (uint32_t i = 0; i < kWordCount; ++i) {
      EXPECT_EQ(download[i], expected[i]) << "epoch=" << epoch << " word=" << i;
    }
    EXPECT_NO_FATAL_FAILURE(queue->WaitConsumed(api_, index));
    if (HasFailure()) {
      return;
    }
  }
  RecordProperty("aql_device_local_kernarg_completed_epochs", kCounts.size());
}

TEST_F(AqlDispatchTest,
       HeaderBarrierOrdersProducerConsumerBeforeTrailingCompletion) {
  const auto* kernel_product =
      kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel_product, nullptr)
      << "missing compiled transform kernel for endpoint";
  const auto& kernel = *kernel_product;
  RecordProperty("transform_kernel_target", kernel.target);

  constexpr uint32_t kGridSize = 1024;
  constexpr uint32_t kWordCount = 2048;
  constexpr uint32_t kPayloadOffset = 16;
  constexpr uint32_t kInputGuard = 0x759bf13du;
  constexpr uint32_t kIntermediateGuard = 0xa36cf197u;
  constexpr uint32_t kOutputGuard = 0x4e90b725u;
  constexpr uint32_t kCompletionGuard = 0x68d329b7u;
  constexpr uint32_t kCompletionByteLength = 4096;
  constexpr uint32_t kCompletionGuardWordCount =
      (kCompletionByteLength - sizeof(aql::Signal)) / sizeof(uint32_t);
  constexpr uint32_t kArgumentStride = 64;
  constexpr uint64_t kPacketsPerEpoch = 3;
  constexpr std::array<uint32_t, 2> kCounts = {1003, 997};
  constexpr std::array<uint32_t, 2> kProducerAddends = {7, 0x80000023u};
  constexpr std::array<uint32_t, 2> kConsumerAddends = {0x101u, 0x2468ace1u};
  const aql::DispatchGeometry kGeometry = {
      1,
      {static_cast<uint16_t>(kernel.workgroup_size()), 1, 1},
      {kGridSize, 1, 1}};
  constexpr aql::FenceScopes kScopes = {aql::FenceScope::kSystem,
                                        aql::FenceScope::kSystem};

  GpuMemory* input = nullptr;
  GpuMemory* intermediate = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ,
                                       kWordCount * sizeof(uint32_t), &input));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kWordCount * sizeof(uint32_t), &intermediate));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kWordCount * sizeof(uint32_t), &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kCompletionByteLength, &completion));
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
  std::memset(completion->host.pointer, 0, completion->info.byte_length);
  auto& signal = *static_cast<aql::Signal*>(completion->host.pointer);
  signal = aql::UserSignal();
  // Native signal fields retain their ABI meaning. Guards begin after the
  // complete 64-byte block, leaving its unused fields zero-initialized.
  auto* completion_guard_address =
      static_cast<uint8_t*>(completion->host.pointer) + sizeof(aql::Signal);
  std::array<uint32_t, kCompletionGuardWordCount> completion_words;
  completion_words.fill(kCompletionGuard);
  std::memcpy(completion_guard_address, completion_words.data(),
              sizeof(completion_words));

  GpuUserQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  uint64_t index = 0;
  uint64_t descriptor_address = 0;
  ASSERT_NO_FATAL_FAILURE(
      PublishKernel(*queue, kernel, "aql_kernel", &index, &descriptor_address));
  const uint64_t first_work_packet_index = index;
  // Keep both epochs in distinct resident slots after the completed cold
  // publication. The operational frontier comes from that helper.
  ASSERT_GE(queue->host.ring_byte_length / sizeof(aql::Packet),
            index + kPacketsPerEpoch * kCounts.size());

  for (uint32_t epoch = 0; epoch < kCounts.size(); ++epoch) {
    SCOPED_TRACE(epoch);
    std::array<uint32_t, kWordCount> upload;
    std::array<uint32_t, kWordCount> expected_intermediate;
    std::array<uint32_t, kWordCount> expected_output;
    std::array<uint32_t, kWordCount> input_words;
    std::array<uint32_t, kWordCount> intermediate_words;
    std::array<uint32_t, kWordCount> output_words;
    upload.fill(kInputGuard);
    expected_intermediate.fill(kIntermediateGuard);
    expected_output.fill(kOutputGuard);
    intermediate_words.fill(kIntermediateGuard);
    output_words.fill(kOutputGuard);
    for (uint32_t i = 0; i < kGridSize; ++i) {
      const uint32_t value =
          0xfffffff0u + i * 0x01030507u + epoch * 0x11111111u;
      upload[kPayloadOffset + i] = value;
      if (i < kCounts[epoch]) {
        const uint32_t intermediate_result = static_cast<uint32_t>(
            uint64_t{value} * 3 + kProducerAddends[epoch]);
        // Flatten both transforms with wide CPU arithmetic. The observed
        // intermediate never participates in the final output oracle.
        const uint32_t output_result = static_cast<uint32_t>(
            uint64_t{value} * 9 + uint64_t{kProducerAddends[epoch]} * 3 +
            kConsumerAddends[epoch]);
        expected_intermediate[kPayloadOffset + i] = intermediate_result;
        expected_output[kPayloadOffset + i] = output_result;
        intermediate_words[kPayloadOffset + i] = ~intermediate_result;
        output_words[kPayloadOffset + i] = ~output_result;
      }
    }
    std::memcpy(input->host.pointer, upload.data(), sizeof(upload));
    std::memcpy(intermediate->host.pointer, intermediate_words.data(),
                sizeof(intermediate_words));
    std::memcpy(output->host.pointer, output_words.data(),
                sizeof(output_words));
    const kernels::transform::Arguments producer_payload = {
        input->device_address + kPayloadOffset * sizeof(uint32_t),
        intermediate->device_address + kPayloadOffset * sizeof(uint32_t),
        kCounts[epoch],
        kProducerAddends[epoch],
    };
    const kernels::transform::Arguments consumer_payload = {
        intermediate->device_address + kPayloadOffset * sizeof(uint32_t),
        output->device_address + kPayloadOffset * sizeof(uint32_t),
        kCounts[epoch],
        kConsumerAddends[epoch],
    };
    // Both disjoint records keep zero padding and remain immutable through
    // terminal completion and consumption. The compiler consumes 24 bytes.
    std::memset(arguments->host.pointer, 0, arguments->info.byte_length);
    std::memcpy(arguments->host.pointer, &producer_payload,
                kernel.arguments.byte_length);
    std::memcpy(
        static_cast<uint8_t*>(arguments->host.pointer) + kArgumentStride,
        &consumer_payload, kernel.arguments.byte_length);
    signal.value = 1;
    const auto produce = aql::Dispatch(
        aql::HeaderBarrier::kDisabled, kGeometry,
        kernel.private_segment_byte_length, kernel.group_segment_byte_length,
        descriptor_address, arguments->device_address, 0, kScopes);
    const auto consume =
        aql::Dispatch(aql::HeaderBarrier::kEnabled, kGeometry,
                      kernel.private_segment_byte_length,
                      kernel.group_segment_byte_length, descriptor_address,
                      arguments->device_address + kArgumentStride, 0, kScopes);
    const auto complete =
        aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kEnabled,
                     completion->device_address, {}, kScopes);

    // Reserve only after constructing every packet. The consumer's header
    // joins producer completion; null dispatch completions keep their fences.
    GpuStoreRelease(queue->host.write_index_address, index + kPacketsPerEpoch);
    ASSERT_TRUE(aql::Publish(*queue, index++, produce));
    ASSERT_TRUE(aql::Publish(*queue, index++, consume));
    ASSERT_TRUE(aql::Publish(*queue, index++, complete));

    // The terminal header independently joins both dispatches, even if the
    // middle edge produced wrong data. Snapshot before any diagnostic or
    // consumption operation can add synchronization to these observations.
    ASSERT_TRUE(
        GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0));
    std::memcpy(output_words.data(), output->host.pointer,
                sizeof(output_words));
    std::memcpy(intermediate_words.data(), intermediate->host.pointer,
                sizeof(intermediate_words));
    std::memcpy(input_words.data(), input->host.pointer, sizeof(input_words));
    aql::Signal completed_signal = {};
    std::memcpy(&completed_signal, completion->host.pointer,
                sizeof(completed_signal));
    std::memcpy(completion_words.data(), completion_guard_address,
                sizeof(completion_words));
    for (uint32_t i = 0; i < kWordCount; ++i) {
      EXPECT_EQ(output_words[i], expected_output[i]) << "output word=" << i;
      EXPECT_EQ(intermediate_words[i], expected_intermediate[i])
          << "intermediate word=" << i;
      EXPECT_EQ(input_words[i], upload[i]) << "input word=" << i;
    }
    EXPECT_EQ(completed_signal.kind, 1);
    EXPECT_EQ(completed_signal.value, 0);
    for (uint32_t i = 0; i < completion_words.size(); ++i) {
      EXPECT_EQ(completion_words[i], kCompletionGuard)
          << "completion guard word=" << i;
    }
    // Retire even after an oracle failure. Signal rearming and all backing
    // reuse require successful observations and a consumed frontier.
    EXPECT_NO_FATAL_FAILURE(queue->WaitConsumed(api_, index));
    if (HasFailure()) {
      return;
    }
  }
  RecordProperty("aql_chain_completed_epochs", kCounts.size());
  RecordProperty("aql_chain_work_packet_count",
                 std::to_string(index - first_work_packet_index));
  RecordProperty("aql_chain_final_packet_index", std::to_string(index));
}

}  // namespace
