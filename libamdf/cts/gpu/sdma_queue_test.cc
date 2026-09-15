// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/user_queue_memory.h"

namespace {

void AppendSdmaCacheTransition(uint32_t* words, size_t* ordinal,
                               uint32_t control) {
  words[(*ordinal)++] = 17;
  words[(*ordinal)++] = 0;
  words[(*ordinal)++] = (control & UINT32_C(0xffff)) << 16;
  words[(*ordinal)++] = control >> 16;
  words[(*ordinal)++] = 0;
}

EncodedUserQueueStream EncodeCopyStream(amdf_queue_format_features_t features,
                                        uint32_t* words,
                                        uint64_t source_address,
                                        uint64_t target_address) {
  enum : uint32_t {
    kAcquireControl = 0x043a1,
    kReleaseControl = 0x0c3a1,
    kCopyByteLength = kUserQueueMemoryElementCount * sizeof(uint32_t),
    kUncachedFenceHeader = 5 | (3 << 16),
  };
  const bool has_scope =
      (features & AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE) != 0;
  const uint32_t copy_scope = has_scope ? (3u << 18) | (3u << 26) : 0;
  uint32_t fence_header = kUncachedFenceHeader;
  if ((features & AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM) != 0) {
    fence_header |= 1u << 20;
  }
  if (has_scope) fence_header |= 3u << 24;
  size_t ordinal = 0;
  AppendSdmaCacheTransition(words, &ordinal, kAcquireControl);
  words[ordinal++] = 1 | (has_scope ? 1u << 28 : 0);
  words[ordinal++] = kCopyByteLength - 1;
  words[ordinal++] = copy_scope;
  words[ordinal++] = static_cast<uint32_t>(source_address);
  words[ordinal++] = static_cast<uint32_t>(source_address >> 32);
  words[ordinal++] = static_cast<uint32_t>(target_address);
  words[ordinal++] = static_cast<uint32_t>(target_address >> 32);
  AppendSdmaCacheTransition(words, &ordinal, kReleaseControl);
  const uint64_t completion_address =
      target_address + kUserQueueMemoryCompletionByteOffset;
  words[ordinal++] = fence_header;
  words[ordinal++] = static_cast<uint32_t>(completion_address);
  words[ordinal++] = static_cast<uint32_t>(completion_address >> 32);
  words[ordinal++] = kUserQueueMemoryCompletionValue;
  words[ordinal++] = 0;
  const size_t byte_length = ordinal * sizeof(uint32_t);
  return {byte_length, byte_length};
}

constexpr UserQueueMemoryCommands kCommands = {
    .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
    .format_version = AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1,
    .required_format_features = AMDF_GPU_SDMA_FORMAT_FEATURE_GCR,
    .encode = EncodeCopyStream,
};

class SdmaQueueTest : public UserQueueMemoryTest {
 protected:
  SdmaQueueTest() : UserQueueMemoryTest(kCommands) {}
};

TEST_F(SdmaQueueTest, CopiesBetweenExactAccessAttachments) {
  RunCopiesBetweenExactAccessAttachments();
}

TEST_F(SdmaQueueTest, ConcurrentDeviceCreationAndRecreation) {
  RunConcurrentDeviceCreationAndRecreation();
}

}  // namespace
