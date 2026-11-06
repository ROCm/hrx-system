// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>
#include <utility>

#include "iree/hal/driver/amd/gpu/cts/native_transfer_test_base.h"

namespace {

class GpuNativeTransferStressTest
    : public iree::hal::amd::gpu::cts::GpuNativeTransferTest {};

TEST_P(GpuNativeTransferStressTest, SustainedConcurrentStagingPressure) {
  RunConcurrentTransfers(8, 16);
}

INSTANTIATE_TEST_SUITE_P(
    Boundaries, GpuNativeTransferStressTest,
    ::testing::Values(std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, false),
                      std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, true),
                      std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, false),
                      std::make_pair(AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, true)),
    [](const auto& info) {
      return std::string(info.param.first == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                             ? "Aql"
                             : "Pm4") +
             (info.param.second ? "Explicit" : "Default");
    });

}  // namespace
