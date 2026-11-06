// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/transfer.h"

#include <array>

#include "iree/testing/gtest.h"

namespace {

TEST(DownloadTest, PartialEndpointsPreserveEveryDestinationAndGuard) {
  std::array<uint8_t, 24> staging;
  for (size_t i = 0; i < staging.size(); ++i) {
    staging[i] = static_cast<uint8_t>(i * 7 + 1);
  }
  std::array<std::array<uint8_t, 10>, 3> targets;
  std::array<iree_hal_amd_gpu_transfer_operation_t, 3> operations = {};
  for (size_t i = 0; i < operations.size(); ++i) {
    operations[i] = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .target = {.host = targets[i].data() + 1},
        .length = 8,
    };
  }
  // Every start/end combination includes empty, partial, and complete rows.
  for (size_t offset = 0; offset < 8; ++offset) {
    for (size_t length = 0; length <= staging.size() - offset; ++length) {
      SCOPED_TRACE(::testing::Message() << offset << '+' << length);
      for (auto& target : targets) {
        target.fill(0xCC);
      }
      const iree_hal_amd_gpu_download_t download = {
          .source = staging.data(),
          .operation = operations.data(),
          .offset = offset,
          .length = length,
      };
      iree_hal_amd_gpu_download_complete(&download);
      for (size_t row = 0; row < targets.size(); ++row) {
        EXPECT_EQ(targets[row].front(), 0xCC);
        EXPECT_EQ(targets[row].back(), 0xCC);
        for (size_t i = 0; i < 8; ++i) {
          const size_t position = row * 8 + i;
          EXPECT_EQ(targets[row][i + 1],
                    position >= offset && position < offset + length
                        ? staging[position - offset]
                        : 0xCC);
        }
      }
    }
  }
}

}  // namespace
