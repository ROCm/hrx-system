// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <string>

#include "libamdf/cts/gpu/sdma/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"

namespace {

class SdmaCacheTest : public GpuCommandTest {
 protected:
  SdmaCacheTest()
      : GpuCommandTest({
            .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
            .roles = AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL,
            .format_features = AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR,
            .cache_operations = AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM |
                                AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM,
            .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
        }) {}
};

TEST_F(SdmaCacheTest, UserGcrBracketsCopiesAcrossEpochs) {
  constexpr size_t kDataLength = 8192;
  constexpr size_t kControlLength = 4096;
  constexpr size_t kSourceWord = 16;
  constexpr size_t kTargetWord = 64;
  constexpr size_t kCompletionWord = 16;
  constexpr size_t kCopyWordCount = 1027;
  constexpr size_t kWordsPerEpoch = 21;
  constexpr std::array<uint32_t, 2> kSeeds = {0x176da309, 0xc8542bef};
  constexpr size_t kCommandWordCount = kSeeds.size() * kWordsPerEpoch;
  GpuMemory* source = nullptr;
  GpuMemory* target = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kDataLength, &source));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kDataLength, &target));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kControlLength, &completion));
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  ASSERT_GT(queue->words().size(), kCommandWordCount);

  // Pre-encode both publications; no submitted command is rewritten between
  // epochs. The completion value advances without a host-side rearm.
  SdmaCommandWriter commands(queue->words().data(), family_.format_features);
  for (uint32_t marker = 1; marker <= kSeeds.size(); ++marker) {
    commands.AcquireFromSystem();
    commands.CopyLinear(source->device_address + kSourceWord * sizeof(uint32_t),
                        target->device_address + kTargetWord * sizeof(uint32_t),
                        kCopyWordCount * sizeof(uint32_t));
    commands.ReleaseToSystem();
    commands.Fence32(
        completion->device_address + kCompletionWord * sizeof(uint32_t),
        marker);
  }
  ASSERT_EQ(commands.word_count(), kCommandWordCount);
  queue->words()[kCommandWordCount] = 0x692bde45;
  std::array<uint32_t, kCommandWordCount + 1> expected_commands;
  std::memcpy(expected_commands.data(), queue->words().data(),
              sizeof(expected_commands));

  // Allocate all observation storage before publication. Whole allocations,
  // including the immutable source and control guards, have CPU-owned oracles.
  std::array<uint32_t, kDataLength / sizeof(uint32_t)> expected_source;
  std::array<uint32_t, kDataLength / sizeof(uint32_t)> expected_target;
  std::array<uint32_t, kControlLength / sizeof(uint32_t)> expected_completion;
  std::array<uint32_t, kDataLength / sizeof(uint32_t)> observed_source;
  std::array<uint32_t, kDataLength / sizeof(uint32_t)> observed_target;
  std::array<uint32_t, kControlLength / sizeof(uint32_t)> observed_completion;
  std::array<uint32_t, kCommandWordCount + 1> observed_commands;
  for (size_t i = 0; i < expected_completion.size(); ++i) {
    expected_completion[i] =
        0x5431cd8bu ^ (static_cast<uint32_t>(i) * 0x03050709u);
  }
  expected_completion[kCompletionWord] = 0;
  std::memcpy(completion->host.pointer, expected_completion.data(),
              kControlLength);
  RecordProperty("sdma_gcr_format_features",
                 std::to_string(family_.format_features));
  RecordProperty("sdma_gcr_words_per_epoch", kWordsPerEpoch);
  RecordProperty("sdma_gcr_copy_byte_length",
                 kCopyWordCount * sizeof(uint32_t));
  RecordProperty("sdma_gcr_data_checked_byte_length", kDataLength);
  RecordProperty("sdma_gcr_control_checked_byte_length", kControlLength);
  RecordProperty("sdma_gcr_completed_epochs", 0);

  for (size_t epoch = 0; epoch < kSeeds.size(); ++epoch) {
    const uint32_t seed = kSeeds[epoch];
    const uint32_t marker = static_cast<uint32_t>(epoch + 1);
    for (size_t i = 0; i < expected_source.size(); ++i) {
      const uint32_t word = static_cast<uint32_t>(i);
      expected_source[i] = seed ^ (0x17a63d91u + word * 0x01030709u);
      expected_target[i] = seed ^ (0x69d28543u + word * 0x05070b0du);
    }
    std::memcpy(source->host.pointer, expected_source.data(), kDataLength);
    std::memcpy(target->host.pointer, expected_target.data(), kDataLength);
    for (size_t i = 0; i < kCopyWordCount; ++i) {
      expected_target[kTargetWord + i] = expected_source[kSourceWord + i];
    }
    expected_completion[kCompletionWord] = marker;
    ASSERT_NO_FATAL_FAILURE(
        queue->Publish(api_, gpu_api_, (epoch + 1) * kWordsPerEpoch));
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        reinterpret_cast<uintptr_t>(completion->host.pointer) +
            kCompletionWord * sizeof(uint32_t),
        marker));

    // Capture payloads and guards before diagnostics or consumed-frontier
    // polling can supply any additional synchronization.
    std::memcpy(observed_source.data(), source->host.pointer, kDataLength);
    std::memcpy(observed_target.data(), target->host.pointer, kDataLength);
    std::memcpy(observed_completion.data(), completion->host.pointer,
                kControlLength);
    std::memcpy(observed_commands.data(), queue->words().data(),
                sizeof(observed_commands));
    for (size_t i = 0; i < expected_source.size(); ++i) {
      EXPECT_EQ(observed_source[i], expected_source[i])
          << "epoch=" << epoch << " source word=" << i;
      EXPECT_EQ(observed_target[i], expected_target[i])
          << "epoch=" << epoch << " target word=" << i;
    }
    for (size_t i = 0; i < expected_completion.size(); ++i) {
      EXPECT_EQ(observed_completion[i], expected_completion[i])
          << "epoch=" << epoch << " control word=" << i;
    }
    EXPECT_EQ(observed_commands, expected_commands) << "epoch=" << epoch;
    EXPECT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
    if (HasFailure()) {
      return;
    }
    RecordProperty("sdma_gcr_completed_epochs", marker);
  }
}

class SdmaSubmissionCacheTest
    : public GpuCommandTest,
      public ::testing::WithParamInterface<amdf_memory_flags_t> {
 protected:
  SdmaSubmissionCacheTest()
      : GpuCommandTest({
            .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
            .roles = AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL,
            .format_features = AMDF_GPU_SDMA_FORMAT_FEATURE_SUBMISSION_GCR,
            .cache_operations = AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM |
                                AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM,
            .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
            .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
        }) {}
};

TEST_P(SdmaSubmissionCacheTest, NativeBoundariesPublishReusedPrivatePayloads) {
  constexpr uint32_t kByteLength = 256 * 1024;
  constexpr auto kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  GpuMemory* source = nullptr;
  GpuMemory* target = nullptr;
  GpuMemory* local = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(kReadWrite, kByteLength, &source, GetParam()));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(kReadWrite, kByteLength, &target, GetParam()));
  ASSERT_NE(local_scope_, nullptr);
  const amdf_memory_device_access_t access = {
      device_,
      {.access = kReadWrite, .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
  const uint32_t ordinal = FindGpuMemoryProfileOrdinal(
      api_, local_scope_, device_, AMDF_MEMORY_PROFILE_ROLE_CREATE,
      AMDF_MEMORY_FLAG_DEVICE_LOCAL, access.requirements);
  ASSERT_NE(ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  const amdf_memory_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create),
      .memory_profile_ordinal = ordinal,
      .access_count = 1,
      .required_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL,
      .byte_length = kByteLength,
      .minimum_alignment = 4096,
      .accesses = &access,
  };
  ASSERT_NO_FATAL_FAILURE(CreateMemory(local_scope_, create, &local));
  GpuCommandQueue* upload = nullptr;
  GpuCommandQueue* download = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&upload));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&download));
  const auto local_site = local->DeviceSite(family_.ordinal);
  amdf_memory_pair_info_t handoff = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(handoff),
  };
  ASSERT_EQ(api_->memory_query_pair_info(&local_site, &local_site, &handoff),
            AMDF_STATUS_OK);
  const bool scoped = (family_.format_features &
                       AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) != 0;
  for (const auto* transition : {&handoff.release, &handoff.acquire}) {
    ASSERT_EQ(transition->kind, scoped ? AMDF_CACHE_TRANSITION_KIND_NONE
                                       : AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    ASSERT_EQ(transition->executor, scoped
                                        ? AMDF_CACHE_TRANSITION_EXECUTOR_NONE
                                        : AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
  }

  auto* input = static_cast<uint32_t*>(source->host.pointer);
  auto* output = static_cast<uint32_t*>(target->host.pointer);
  constexpr size_t kWordsPerSubmission = 8;
  for (uint32_t epoch = 1; epoch <= 3; ++epoch) {
    for (uint32_t i = 0; i < kByteLength / sizeof(uint32_t); ++i) {
      input[i] = epoch * 0x127390abu + i * 0x13579bdu;
      output[i] = ~input[i];
    }
    ASSERT_NO_FATAL_FAILURE(source->TransferHostCache(
        api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
    ASSERT_NO_FATAL_FAILURE(target->TransferHostCache(
        api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
    SdmaCommandWriter produce(
        upload->words().data() + (epoch - 1) * kWordsPerSubmission,
        family_.format_features);
    produce.CopyLinear(source->device_address, local->device_address,
                       kByteLength);
    produce.Noop();
    ASSERT_EQ(produce.word_count(), kWordsPerSubmission);
    ASSERT_NO_FATAL_FAILURE(
        upload->Publish(api_, gpu_api_, epoch * kWordsPerSubmission));
    // The native epilogue, not a fence packet inside the stream, owns release.
    // The second queue's submission starts only after that producer boundary.
    ASSERT_NO_FATAL_FAILURE(upload->WaitRetired(api_));
    SdmaCommandWriter consume(
        download->words().data() + (epoch - 1) * kWordsPerSubmission,
        family_.format_features);
    consume.CopyLinear(local->device_address, target->device_address,
                       kByteLength);
    consume.Noop();
    ASSERT_EQ(consume.word_count(), kWordsPerSubmission);
    ASSERT_NO_FATAL_FAILURE(
        download->Publish(api_, gpu_api_, epoch * kWordsPerSubmission));
    ASSERT_NO_FATAL_FAILURE(download->WaitRetired(api_));
    ASSERT_NO_FATAL_FAILURE(target->TransferHostCache(
        api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_INVALIDATE));
    for (uint32_t i = 0; i < kByteLength / sizeof(uint32_t); ++i) {
      ASSERT_EQ(output[i], input[i]) << "epoch=" << epoch << " word=" << i;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    HostPolicies, SdmaSubmissionCacheTest,
    ::testing::Values(amdf_memory_flags_t{0},
                      amdf_memory_flags_t{AMDF_MEMORY_FLAG_HOST_COHERENT}));

}  // namespace
