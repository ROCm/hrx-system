// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_GPU_USER_QUEUE_MEMORY_H_
#define AMDF_CTS_GPU_USER_QUEUE_MEMORY_H_

#include <cstddef>
#include <cstdint>

#include "libamdf/cts/gpu/gpu_device_fixture.h"

// Shared payload contract for the PM4 and SDMA memory scenarios.
constexpr size_t kUserQueueMemoryElementCount = 16;
constexpr uint64_t kUserQueueMemoryCompletionByteOffset = 256;
constexpr uint32_t kUserQueueMemoryCompletionValue = UINT32_C(0x71c04a5e);
// Ring space sufficient for either encoder's complete copy and completion.
constexpr uint64_t kUserQueueMemoryCommandByteCapacity = 512;

struct EncodedUserQueueStream {
  // Number of initialized command bytes in the ring.
  size_t byte_length;
  // Producer frontier in the units defined by the queue format.
  uint64_t published_index;
};

// Caller-owned command encoding used by one conformance scenario.
struct UserQueueMemoryCommands {
  // Native packet representation emitted by the encoder.
  amdf_queue_command_type_t command_type;
  // Publication protocol and baseline native packet version.
  uint32_t format_version;
  // Packet encodings the scenario requires from the selected family.
  amdf_queue_format_features_t required_format_features;
  // Writes the shared payload copy and completion marker into at least 512
  // bytes of ring storage, using the selected family's encoding features.
  EncodedUserQueueStream (*encode)(amdf_queue_format_features_t features,
                                   uint32_t* words, uint64_t source_address,
                                   uint64_t target_address);
};

// Selects a host-published transfer/cache-control family before activation.
// The selected facts drive command encoding and all later queue operations.
class UserQueueMemoryTest : public GpuDeviceFixture {
 protected:
  explicit UserQueueMemoryTest(const UserQueueMemoryCommands& commands)
      : commands_(commands) {}

  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override;
  void RunCopiesBetweenExactAccessAttachments();
  void RunConcurrentDeviceCreationAndRecreation();

 private:
  // Immutable caller encoding description, copied from the test definition.
  UserQueueMemoryCommands commands_;
  // Complete passive family selected before borrowing the cached device.
  amdf_queue_family_info_t family_ = {};
};

#endif  // AMDF_CTS_GPU_USER_QUEUE_MEMORY_H_
