// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/sdma/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"

namespace {

class SdmaCopyTest : public GpuCommandTest {
 protected:
  SdmaCopyTest()
      : GpuCommandTest({
            .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
            .roles = AMDF_QUEUE_ROLE_TRANSFER,
            .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER |
                                 AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
        }) {}
};

class SdmaPayloadCopyTest
    : public SdmaCopyTest,
      public ::testing::WithParamInterface<amdf_memory_flags_t> {
 protected:
  void SetUp() override {
    SdmaCopyTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    scope_ = (GetParam() & AMDF_MEMORY_FLAG_DEVICE_LOCAL) ? local_scope_
                                                          : system_scope_;
    if (!scope_) {
      GTEST_SKIP() << "endpoint has no local memory scope";
    }
    profile_ordinal_ = FindMemoryProfileOrdinal(
        scope_,
        AMDF_MEMORY_PROFILE_ROLE_CREATE | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
        AMDF_MEMORY_FLAG_HOST_VISIBLE |
            (GetParam() & AMDF_MEMORY_FLAG_DEVICE_LOCAL),
        {
            .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
            .flags = (GetParam() & ~AMDF_MEMORY_FLAG_DEVICE_LOCAL) |
                     AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
        });
    if (profile_ordinal_ == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN) {
      GTEST_SKIP() << "requested host-visible payload policy is unavailable";
    }
  }

  void CreatePayload(amdf_memory_access_t access, uint64_t byte_length,
                     GpuMemory** out_memory) {
    const amdf_memory_device_access_t attachment = {
        device_,
        {
            .access = access,
            .flags = (GetParam() & ~AMDF_MEMORY_FLAG_DEVICE_LOCAL) |
                     AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
        }};
    const amdf_memory_create_info_t creation = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(creation),
        .memory_profile_ordinal = profile_ordinal_,
        .access_count = 1,
        .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE |
                          (GetParam() & AMDF_MEMORY_FLAG_DEVICE_LOCAL),
        .byte_length = byte_length,
        .minimum_alignment = 4096,
        .accesses = &attachment,
    };
    ASSERT_NO_FATAL_FAILURE(CreateMemory(scope_, creation, out_memory));
    RecordProperty("payload_memory_class", (*out_memory)->info.memory_class);
    if (GetParam() & AMDF_MEMORY_FLAG_DEVICE_LOCAL) {
      EXPECT_EQ((*out_memory)->info.memory_class, AMDF_MEMORY_CLASS_LOCAL);
      EXPECT_NE((*out_memory)->info.flags & AMDF_MEMORY_FLAG_HOST_VISIBLE, 0u);
    }
  }

 private:
  // Borrowed placement scope selected before any workload is created.
  amdf_memory_scope_t* scope_ = nullptr;
  // Available host-visible profile for this case's complete payload policy.
  uint32_t profile_ordinal_ = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
};

TEST_P(SdmaPayloadCopyTest, LinearCopyCompletesBeforeFence) {
  GpuMemory* source = nullptr;
  GpuMemory* target = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreatePayload(AMDF_MEMORY_ACCESS_READ, 4096, &source));
  ASSERT_NO_FATAL_FAILURE(CreatePayload(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &target));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
  RecordProperty(
      "payload_is_host_coherent",
      (source->access_info.flags & AMDF_MEMORY_FLAG_HOST_COHERENT) != 0);
  auto* input = static_cast<uint32_t*>(source->host.pointer);
  auto* output = static_cast<uint32_t*>(target->host.pointer);
  *static_cast<uint32_t*>(completion->host.pointer) = 0;
  constexpr size_t kWordCount = 257;
  for (size_t i = 0; i < kWordCount; ++i) {
    input[i] = 0x2ac40000u + static_cast<uint32_t>(i) * 0x00010301u;
    output[i] = ~input[i];
  }
  // Guard words make an incorrect byte count observable.
  output[kWordCount] = 0x725ae191;
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  ASSERT_GE(queue->words().size_bytes(), 21u * sizeof(uint32_t));
  SdmaCommandWriter commands(queue->words().data(), family_.format_features);
  const bool user_gcr =
      (family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) != 0;
  if (user_gcr) {
    commands.AcquireFromSystem();
  }
  commands.CopyLinear(source->device_address, target->device_address,
                      kWordCount * sizeof(uint32_t));
  if (user_gcr) {
    commands.ReleaseToSystem();
  }
  commands.Fence32(completion->device_address, 1);
  for (const auto* memory : {source, target}) {
    ASSERT_NO_FATAL_FAILURE(memory->TransferHostCache(
        api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
  }
  ASSERT_NO_FATAL_FAILURE(
      queue->Publish(api_, gpu_api_, commands.word_count()));
  ASSERT_TRUE(GpuWaitEqual<uint32_t>(
      reinterpret_cast<uintptr_t>(completion->host.pointer), 1));
  ASSERT_NO_FATAL_FAILURE(target->TransferHostCache(
      api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_INVALIDATE));
  // Capture every observation before diagnostics or retirement can intervene.
  std::array<uint32_t, kWordCount + 1> observed_output;
  std::array<uint32_t, kWordCount> observed_input;
  std::memcpy(observed_output.data(), output, sizeof(observed_output));
  std::memcpy(observed_input.data(), input, sizeof(observed_input));
  for (size_t i = 0; i < kWordCount; ++i) {
    const uint32_t expected =
        0x2ac40000u + static_cast<uint32_t>(i) * 0x00010301u;
    EXPECT_EQ(observed_output[i], expected) << i;
    EXPECT_EQ(observed_input[i], expected) << "source word " << i;
  }
  EXPECT_EQ(observed_output[kWordCount], 0x725ae191u);
  // Nonfatal oracle failures still reach normal retirement.
  ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
}

TEST_P(SdmaPayloadCopyTest, ByteTailsAndPageCrossingsPreserveSurroundingBytes) {
  constexpr std::array<uint32_t, 6> kByteLengths = {1, 2, 3, 4, 31, 4101};
  constexpr uint64_t kSourceLength = 12288;
  constexpr uint64_t kTargetStride = 12288;
  constexpr uint64_t kTargetLength = kTargetStride * kByteLengths.size();
  GpuMemory* source = nullptr;
  GpuMemory* target = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreatePayload(AMDF_MEMORY_ACCESS_READ, kSourceLength, &source));
  ASSERT_NO_FATAL_FAILURE(
      CreatePayload(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                    kTargetLength, &target));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
  RecordProperty(
      "payload_is_host_coherent",
      (source->access_info.flags & AMDF_MEMORY_FLAG_HOST_COHERENT) != 0);
  auto* input = static_cast<uint8_t*>(source->host.pointer);
  auto* output = static_cast<uint8_t*>(target->host.pointer);
  for (uint64_t i = 0; i < kSourceLength; ++i) {
    input[i] = static_cast<uint8_t>(i * 73 + (i >> 8) * 19 + 7);
  }
  std::fill_n(output, kTargetLength, 0xa5);
  std::vector<uint8_t> expected(kTargetLength, 0xa5);
  std::vector<uint8_t> observed_output(kTargetLength);
  std::vector<uint8_t> observed_input(kSourceLength);
  *static_cast<uint32_t*>(completion->host.pointer) = 0;
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  ASSERT_GE(queue->words().size_bytes(), 256u);
  SdmaCommandWriter commands(queue->words().data(), family_.format_features);
  const bool user_gcr =
      (family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) != 0;
  if (user_gcr) {
    commands.AcquireFromSystem();
  }
  for (size_t i = 0; i < kByteLengths.size(); ++i) {
    // Differently aligned ranges cross the first source and target page.
    const uint64_t source_offset = i == 3 ? 4092 : 4095;
    const uint64_t target_offset = i * kTargetStride + (i == 3 ? 4092 : 4091);
    commands.CopyLinear(source->device_address + source_offset,
                        target->device_address + target_offset,
                        kByteLengths[i]);
    std::copy_n(input + source_offset, kByteLengths[i],
                expected.data() + target_offset);
  }
  if (user_gcr) {
    commands.ReleaseToSystem();
  }
  commands.Fence32(completion->device_address, 1);
  for (const auto* memory : {source, target}) {
    ASSERT_NO_FATAL_FAILURE(memory->TransferHostCache(
        api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
  }
  ASSERT_NO_FATAL_FAILURE(
      queue->Publish(api_, gpu_api_, commands.word_count()));
  ASSERT_TRUE(GpuWaitEqual<uint32_t>(
      reinterpret_cast<uintptr_t>(completion->host.pointer), 1));
  ASSERT_NO_FATAL_FAILURE(target->TransferHostCache(
      api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_INVALIDATE));
  // Capture every observation before diagnostics or retirement can intervene.
  std::memcpy(observed_output.data(), output, observed_output.size());
  std::memcpy(observed_input.data(), input, observed_input.size());
  for (uint64_t i = 0; i < kTargetLength; ++i) {
    EXPECT_EQ(observed_output[i], expected[i]) << "byte " << i;
  }
  for (uint64_t i = 0; i < kSourceLength; ++i) {
    EXPECT_EQ(observed_input[i],
              static_cast<uint8_t>(i * 73 + (i >> 8) * 19 + 7))
        << "source byte " << i;
  }
  // Nonfatal oracle failures still reach normal retirement.
  ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
}

INSTANTIATE_TEST_SUITE_P(
    PayloadPolicy, SdmaPayloadCopyTest,
    ::testing::Values(amdf_memory_flags_t{0},
                      amdf_memory_flags_t{AMDF_MEMORY_FLAG_HOST_COHERENT},
                      amdf_memory_flags_t{AMDF_MEMORY_FLAG_HOST_COHERENT |
                                          AMDF_MEMORY_FLAG_QUEUE_STORAGE},
                      amdf_memory_flags_t{AMDF_MEMORY_FLAG_DEVICE_LOCAL}));

class SdmaDependencyTest : public SdmaCopyTest {};

TEST_F(SdmaDependencyTest, NopOrdersDependentCopiesAcrossEpochs) {
  constexpr size_t kDataLength = 8192;
  constexpr size_t kControlLength = 4096;
  constexpr uint32_t kCopyLength = 4096;
  constexpr uint32_t kSourceOffset = 128;
  constexpr uint32_t kIntermediateOffset = 256;
  constexpr uint32_t kOutputOffset = 384;
  constexpr uint32_t kCompletionOffset = 64;
  const bool user_gcr =
      (family_.format_features & AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) != 0;
  const size_t words_per_epoch = 19 + (user_gcr ? 10 : 0);
  constexpr std::array<uint32_t, 2> kSeeds = {0x13579bdfu, 0xa5c31f27u};
  GpuMemory* source = nullptr;
  GpuMemory* intermediate = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, kDataLength, &source));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kDataLength, &intermediate));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kDataLength, &output));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   kControlLength, &completion));
  auto* intermediate_words = static_cast<uint32_t*>(intermediate->host.pointer);
  auto* output_words = static_cast<uint32_t*>(output->host.pointer);
  auto* completion_words = static_cast<uint32_t*>(completion->host.pointer);

  // All observation storage exists before any command is published.
  std::array<uint32_t, kDataLength / sizeof(uint32_t)> expected_source;
  std::array<uint32_t, kDataLength / sizeof(uint32_t)> expected_intermediate;
  std::array<uint32_t, kDataLength / sizeof(uint32_t)> expected_output;
  std::array<uint32_t, kControlLength / sizeof(uint32_t)> expected_completion;
  std::array<uint32_t, kDataLength / sizeof(uint32_t)> observed_source;
  std::array<uint32_t, kDataLength / sizeof(uint32_t)> observed_intermediate;
  std::array<uint32_t, kDataLength / sizeof(uint32_t)> observed_output;
  std::array<uint32_t, kControlLength / sizeof(uint32_t)> observed_completion;
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));
  ASSERT_GE(queue->words().size_bytes(),
            kSeeds.size() * words_per_epoch * sizeof(uint32_t));
  SdmaCommandWriter commands(queue->words().data(), family_.format_features);

  RecordProperty("sdma_nop_family_ordinal", family_.ordinal);
  RecordProperty("sdma_nop_format_version", family_.format_version);
  RecordProperty("sdma_nop_format_features",
                 std::to_string(family_.format_features));
  RecordProperty("sdma_nop_seed_0", std::to_string(kSeeds[0]));
  RecordProperty("sdma_nop_seed_1", std::to_string(kSeeds[1]));
  RecordProperty("sdma_nop_source_offset", kSourceOffset);
  RecordProperty("sdma_nop_intermediate_offset", kIntermediateOffset);
  RecordProperty("sdma_nop_output_offset", kOutputOffset);
  RecordProperty("sdma_nop_completion_offset", kCompletionOffset);
  RecordProperty("sdma_nop_copy_byte_length", kCopyLength);
  RecordProperty("sdma_nop_source_checked_byte_length", kDataLength);
  RecordProperty("sdma_nop_intermediate_checked_byte_length", kDataLength);
  RecordProperty("sdma_nop_output_checked_byte_length", kDataLength);
  RecordProperty("sdma_nop_completion_checked_byte_length", kControlLength);
  RecordProperty("sdma_nop_words_per_epoch", words_per_epoch);
  RecordProperty("sdma_nop_first_byte_frontier",
                 words_per_epoch * sizeof(uint32_t));
  RecordProperty("sdma_nop_final_byte_frontier",
                 kSeeds.size() * words_per_epoch * sizeof(uint32_t));
  RecordProperty("sdma_nop_completed_epochs", 0);

  for (size_t epoch = 0; epoch < kSeeds.size(); ++epoch) {
    const uint32_t seed = kSeeds[epoch];
    const uint32_t marker = static_cast<uint32_t>(epoch + 1);
    for (size_t i = 0; i < expected_source.size(); ++i) {
      const uint32_t word = static_cast<uint32_t>(i);
      expected_source[i] = seed ^ (0x179b3de1u + word * 0x01030507u);
      expected_intermediate[i] = seed ^ (0x25a64bc3u + word * 0x03050709u);
      expected_output[i] = seed ^ (0x4962d5e7u + word * 0x0507090bu);
    }
    for (size_t i = 0; i < expected_completion.size(); ++i) {
      expected_completion[i] =
          seed ^ (0x6de8912fu + static_cast<uint32_t>(i) * 0x07090b0du);
    }
    std::memcpy(source->host.pointer, expected_source.data(), kDataLength);
    std::memcpy(intermediate->host.pointer, expected_intermediate.data(),
                kDataLength);
    std::memcpy(output->host.pointer, expected_output.data(), kDataLength);
    std::memcpy(completion->host.pointer, expected_completion.data(),
                kControlLength);
    for (size_t i = 0; i < kCopyLength / sizeof(uint32_t); ++i) {
      // Derive the oracle from the CPU formula, never from a device result.
      const uint32_t source_word =
          static_cast<uint32_t>(kSourceOffset / sizeof(uint32_t) + i);
      const uint32_t expected =
          seed ^ (0x179b3de1u + source_word * 0x01030507u);
      expected_intermediate[kIntermediateOffset / sizeof(uint32_t) + i] =
          expected;
      expected_output[kOutputOffset / sizeof(uint32_t) + i] = expected;
      intermediate_words[kIntermediateOffset / sizeof(uint32_t) + i] =
          expected ^ 0x55555555u;
      output_words[kOutputOffset / sizeof(uint32_t) + i] =
          expected ^ 0xaaaaaaaau;
    }
    completion_words[kCompletionOffset / sizeof(uint32_t)] = 0;
    expected_completion[kCompletionOffset / sizeof(uint32_t)] = marker;

    if (user_gcr) {
      commands.AcquireFromSystem();
    }
    commands.CopyLinear(source->device_address + kSourceOffset,
                        intermediate->device_address + kIntermediateOffset,
                        kCopyLength);
    commands.Noop();
    commands.CopyLinear(intermediate->device_address + kIntermediateOffset,
                        output->device_address + kOutputOffset, kCopyLength);
    if (user_gcr) {
      commands.ReleaseToSystem();
    }
    commands.Fence32(completion->device_address + kCompletionOffset, marker);
    ASSERT_EQ(commands.word_count(), (epoch + 1) * words_per_epoch);
    ASSERT_NO_FATAL_FAILURE(
        queue->Publish(api_, gpu_api_, commands.word_count()));
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        reinterpret_cast<uintptr_t>(completion->host.pointer) +
            kCompletionOffset,
        marker));

    // Capture every complete allocation before diagnostics or consumed-frontier
    // polling can intervene. The marker is the only execution observation.
    std::memcpy(observed_source.data(), source->host.pointer, kDataLength);
    std::memcpy(observed_intermediate.data(), intermediate->host.pointer,
                kDataLength);
    std::memcpy(observed_output.data(), output->host.pointer, kDataLength);
    std::memcpy(observed_completion.data(), completion->host.pointer,
                kControlLength);
    for (size_t i = 0; i < expected_source.size(); ++i) {
      EXPECT_EQ(observed_source[i], expected_source[i])
          << "epoch=" << epoch << " source word=" << i;
      EXPECT_EQ(observed_intermediate[i], expected_intermediate[i])
          << "epoch=" << epoch << " intermediate word=" << i;
      EXPECT_EQ(observed_output[i], expected_output[i])
          << "epoch=" << epoch << " output word=" << i;
    }
    for (size_t i = 0; i < expected_completion.size(); ++i) {
      EXPECT_EQ(observed_completion[i], expected_completion[i])
          << "epoch=" << epoch << " completion word=" << i;
    }
    // An oracle failure still retires the published stream. A failed epoch
    // leaves every allocation untouched until queue-first teardown.
    EXPECT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
    if (HasFailure()) {
      return;
    }
    RecordProperty("sdma_nop_completed_epochs", marker);
  }
}

}  // namespace
