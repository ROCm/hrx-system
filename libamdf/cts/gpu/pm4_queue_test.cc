// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/pm4_commands.h"
#include "libamdf/cts/gpu/user_queue_memory.h"

namespace {

EncodedUserQueueStream EncodeCopyStream(
    amdf_queue_format_features_t /*features*/, uint32_t* words,
    uint64_t source_address, uint64_t target_address) {
  Pm4CommandWriter commands(words);
  commands.SystemBarrier();
  for (size_t i = 0; i < kUserQueueMemoryElementCount; ++i) {
    commands.CopyData32(source_address + i * sizeof(uint32_t),
                        target_address + i * sizeof(uint32_t));
  }
  commands.SystemBarrier();
  commands.WriteData32(target_address + kUserQueueMemoryCompletionByteOffset,
                       kUserQueueMemoryCompletionValue);
  commands.PadToEightWords();
  return {commands.word_count() * sizeof(uint32_t), commands.word_count()};
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
