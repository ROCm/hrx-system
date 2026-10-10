// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/lifecycle/user_queue_memory.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"

namespace {

EncodedUserQueueStream EncodeCopyStream(
    const amdf_gpu_endpoint_info_t& target,
    amdf_queue_format_features_t /*features*/, uint32_t* words,
    uint64_t source_address, uint64_t target_address) {
  Pm4CommandWriter commands(words, *Pm4CommandProfile::Find(target));
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
    .required_roles = AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL,
    .required_cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                                 AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
    .required_cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
    .encode = EncodeCopyStream,
};

class Pm4DeviceLifetimeTest : public UserQueueMemoryTest {
 protected:
  Pm4DeviceLifetimeTest() : UserQueueMemoryTest(kCommands) {}

  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override {
    amdf_gpu_endpoint_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
        .structure_size = sizeof(info),
    };
    const amdf_status_t status = gpu_api_->endpoint_query_info(endpoint, &info);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (!Pm4CommandProfile::Find(info)) {
      *out_matches = false;
      return AMDF_STATUS_OK;
    }
    return UserQueueMemoryTest::MatchGpuEndpoint(endpoint, out_matches);
  }
};

TEST_F(Pm4DeviceLifetimeTest, CopiesBetweenExactAccessAttachments) {
  RunCopiesBetweenExactAccessAttachments();
}

TEST_F(Pm4DeviceLifetimeTest, DISABLED_ConcurrentDeviceCreationAndRecreation) {
  RunConcurrentDeviceCreationAndRecreation();
}

}  // namespace
