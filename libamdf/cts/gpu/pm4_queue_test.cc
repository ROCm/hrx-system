// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstring>

#include "libamdf/cts/gpu/user_queue_memory.h"

namespace {

uint32_t MakePm4Header(uint32_t opcode, uint32_t dword_count) {
  return (UINT32_C(3) << 30) | (opcode << 8) | ((dword_count - 2) << 16);
}

void AppendSystemBarrier(uint32_t* words, size_t* ordinal) {
  enum : uint32_t {
    kEventWriteOpcode = 0x46,
    kAcquireMemoryOpcode = 0x58,
    kEventWriteDwordCount = 2,
    kAcquireMemoryDwordCount = 8,
    kComputeShaderPartialFlush = 7 | (4 << 8),
    kConservativeGcrControl = (3 << 0) | (1 << 4) | (1 << 5) | (1 << 7) |
                              (1 << 8) | (1 << 9) | (1 << 14) | (1 << 15),
  };
  words[(*ordinal)++] = MakePm4Header(kEventWriteOpcode, kEventWriteDwordCount);
  words[(*ordinal)++] = kComputeShaderPartialFlush;
  words[(*ordinal)++] =
      MakePm4Header(kAcquireMemoryOpcode, kAcquireMemoryDwordCount);
  words[(*ordinal)++] = 0;
  words[(*ordinal)++] = UINT32_MAX;
  words[(*ordinal)++] = 0xff;
  words[(*ordinal)++] = 0;
  words[(*ordinal)++] = 0;
  words[(*ordinal)++] = 0x0a;
  words[(*ordinal)++] = kConservativeGcrControl;
}

void AppendCopyData32(uint32_t* words, size_t* ordinal, uint64_t source_address,
                      uint64_t target_address) {
  enum : uint32_t {
    kCopyDataOpcode = 0x40,
    kCopyDataDwordCount = 6,
    kSourceTcL2 = 2 << 0,
    kTargetTcL2 = 2 << 8,
    kWaitForConfirmation = 1 << 20,
  };
  words[(*ordinal)++] = MakePm4Header(kCopyDataOpcode, kCopyDataDwordCount);
  words[(*ordinal)++] = kSourceTcL2 | kTargetTcL2 | kWaitForConfirmation;
  words[(*ordinal)++] =
      static_cast<uint32_t>(source_address) & UINT32_C(0xfffffffc);
  words[(*ordinal)++] = static_cast<uint32_t>(source_address >> 32);
  words[(*ordinal)++] =
      static_cast<uint32_t>(target_address) & UINT32_C(0xfffffffc);
  words[(*ordinal)++] = static_cast<uint32_t>(target_address >> 32);
}

void AppendWriteData32(uint32_t* words, size_t* ordinal,
                       uint64_t target_address, uint32_t value) {
  enum : uint32_t {
    kWriteDataOpcode = 0x37,
    kWriteDataDwordCount = 5,
    kTargetTcL2 = 2 << 8,
    kWaitForConfirmation = 1 << 20,
  };
  words[(*ordinal)++] = MakePm4Header(kWriteDataOpcode, kWriteDataDwordCount);
  words[(*ordinal)++] = kTargetTcL2 | kWaitForConfirmation;
  words[(*ordinal)++] =
      static_cast<uint32_t>(target_address) & UINT32_C(0xfffffffc);
  words[(*ordinal)++] = static_cast<uint32_t>(target_address >> 32);
  words[(*ordinal)++] = value;
}

EncodedUserQueueStream EncodeCopyStream(
    amdf_queue_format_features_t /*features*/, uint32_t* words,
    uint64_t source_address, uint64_t target_address) {
  size_t ordinal = 0;
  AppendSystemBarrier(words, &ordinal);
  for (size_t i = 0; i < kUserQueueMemoryElementCount; ++i) {
    AppendCopyData32(words, &ordinal, source_address + i * sizeof(uint32_t),
                     target_address + i * sizeof(uint32_t));
  }
  AppendSystemBarrier(words, &ordinal);
  AppendWriteData32(words, &ordinal,
                    target_address + kUserQueueMemoryCompletionByteOffset,
                    kUserQueueMemoryCompletionValue);

  size_t padding_dword_count = 8 - ordinal % 8;
  if (padding_dword_count == 1) padding_dword_count += 8;
  words[ordinal] = MakePm4Header(0x10, padding_dword_count);
  std::memset(words + ordinal + 1, 0,
              (padding_dword_count - 1) * sizeof(*words));
  const size_t dword_count = ordinal + padding_dword_count;
  return {dword_count * sizeof(uint32_t), dword_count};
}

constexpr UserQueueMemoryCommands kCommands = {
    .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
    .format_version = AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1,
    .required_format_features = AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR,
    .encode = EncodeCopyStream,
};

class Pm4QueueTest : public UserQueueMemoryTest {
 protected:
  Pm4QueueTest() : UserQueueMemoryTest(kCommands) {}
};

TEST_F(Pm4QueueTest, CopiesBetweenExactAccessAttachments) {
  RunCopiesBetweenExactAccessAttachments();
}

TEST_F(Pm4QueueTest, ConcurrentDeviceCreationAndRecreation) {
  RunConcurrentDeviceCreationAndRecreation();
}

}  // namespace
