// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"
#include "loom/runtime/testing/device.h"
#include "loom/runtime/testing/stalled_stderr.h"
#include "loom/serve/service/finite/embedding/source_service_test_fixture.h"
#include "loom/serve/testing/model.h"
#include "loom/serve/testing/package.h"

namespace loom_serve::testing::embedding {

using loom_runtime::testing::StalledStderr;

class GroupEmbeddingServiceTest : public SourceEmbeddingServiceTest {
 protected:
  iree_host_size_t DeviceCount() const override { return 2; }

  void TearDown() override {
    // Release a held dependency even when an assertion interrupts the witness.
    if (gate_) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate_, 2, nullptr));
    }
    iree_hal_buffer_release(workspace_);
    if (execution_) {
      IREE_EXPECT_OK(loom_runtime_execution_drain(execution_));
    }
    loom_runtime_execution_release(execution_);
    iree_hal_semaphore_release(gate_);
    SourceEmbeddingServiceTest::TearDown();
  }

  // Work on the group's other device, independent of the embedding timeline.
  loom_runtime_execution_t* execution_ = nullptr;
  // External predecessor held until the other device returns its HTTP result.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Queue-allocated root retained through submission of its final reader.
  iree_hal_buffer_t* workspace_ = nullptr;
  // Feedback storage outliving accepted work, including assertion cleanup.
  std::array<uint32_t, 16> output_ = {};
};

TEST_F(GroupEmbeddingServiceTest, EmbeddingsCompleteBeforeOtherDeviceRetires) {
  auto* first = loom_runtime_device_group_at(group_, 0);
  auto* first_pool = loom_runtime_device_group_pool_at(group_, 0);
  EXPECT_EQ(loom_runtime_device_group_count(group_), 2u);
  EXPECT_EQ(loom_runtime_device_group(first),
            loom_runtime_device_group(device_));
  EXPECT_EQ(
      iree_hal_device_group_device_count(loom_runtime_device_group(first)), 2u);
  EXPECT_NE(loom_runtime_pool_residency_cache(first_pool),
            loom_runtime_pool_residency_cache(storage_owner_));
  IREE_ASSERT_OK(
      loom_runtime_device_create_execution(first, first_pool, &execution_));
  IREE_ASSERT_OK(loom_runtime_execution_reserve_workspace(execution_, 64, 16));
  const iree_hal_pool_reservation_request_t request = {
      .allocation_size = 64,
      .min_alignment = 16,
  };
  IREE_ASSERT_OK(
      loom_runtime_execution_alloca(execution_, 1, &request, &workspace_));
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_runtime_device_handle(first), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate_));
  uint64_t value = 1;
  IREE_ASSERT_OK(
      loom_runtime_execution_barrier(execution_, {1, &gate_, &value}));
  const uint32_t pattern = 0x13579BDF;
  const iree_hal_transfer_operation_t fill = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
      .fill =
          {
              .target_buffer = workspace_,
              .length = sizeof(output_),
              .pattern = &pattern,
              .pattern_length = sizeof(pattern),
          },
  };
  IREE_ASSERT_OK(loom_runtime_execution_transfer(execution_, 1, &fill, &value));
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = workspace_,
              .target = output_.data(),
              .length = sizeof(output_),
          },
  };
  IREE_ASSERT_OK(
      loom_runtime_execution_feedback(execution_, 1, &download, &value));
  auto frontier = loom_runtime_execution_frontier(execution_);
  // The feedback branch is a final reader, not an implicit FIFO dependency.
  IREE_ASSERT_OK(loom_runtime_execution_barrier(
      execution_, {1, &frontier.semaphores[1], &frontier.values[1]}));
  IREE_ASSERT_OK(
      loom_runtime_execution_dealloca(execution_, workspace_, &value));
  iree_hal_buffer_release(std::exchange(workspace_, nullptr));

  StartRouter();
  const auto response =
      Read(Send(R"({"model":"fixture","input":["a","bbb"]})"));
  EXPECT_EQ(response.find("HTTP/1.1 200"), 0u);
  EXPECT_NE(response.find(R"("index":0,"embedding":[1,1,1,2])"),
            std::string::npos);
  EXPECT_NE(response.find(R"("index":1,"embedding":[6,3,2,2])"),
            std::string::npos);
  uint64_t observed = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(frontier.semaphores[1], &observed));
  EXPECT_LT(observed, frontier.values[1]);
  IREE_ASSERT_OK(loom_runtime_pool_trim(first_pool, 0));
  EXPECT_EQ(
      loom_runtime_pool_statistics(first_pool).workspace.reservation_count, 1u);
  const auto held = loom_runtime_pool_statistics(first_pool);
  EXPECT_GT(held.workspace.bytes_committed, 0u);
  if (!loom_runtime_pool_memory(first_pool)) {
    // Fixed workspace is backed by the ordinary storage source, not a second
    // physical commitment to add to it.
    EXPECT_GE(held.storage.bytes_committed, held.workspace.bytes_committed);
    EXPECT_EQ(held.retained.committed_bytes, 0u);
  }
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  IREE_ASSERT_OK(loom_runtime_execution_drain(execution_));
  for (uint32_t word : output_) {
    EXPECT_EQ(word, pattern);
  }
  IREE_ASSERT_OK(loom_runtime_test_pool_trim_and_drain(first_pool, 0));
  EXPECT_EQ(loom_runtime_pool_statistics(first_pool).workspace.bytes_committed,
            0u);
  EXPECT_EQ(loom_runtime_pool_statistics(first_pool).storage.bytes_committed,
            0u);
}

}  // namespace loom_serve::testing::embedding
