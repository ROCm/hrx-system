// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/finite/result.h"

#include <array>
#include <condition_variable>
#include <mutex>
#include <utility>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/runtime/device/device.h"
#include "loom/runtime/execution/retirement.h"
#include "loom/runtime/execution/workspace.h"

namespace {

class ResultTest : public ::testing::Test {
 protected:
  void TearDown() override {
    loom_serve_finite_result_release(result_);
    iree_vm_buffer_release(plan_);
  }

  iree_status_t CreatePlan() {
    IREE_RETURN_IF_ERROR(iree_vm_buffer_create(
        sizeof(words_), 8, iree_allocator_system(), &plan_));
    iree_byte_span_t bytes;
    IREE_RETURN_IF_ERROR(
        iree_vm_buffer_map_write(plan_, 0, sizeof(words_), &bytes));
    for (size_t i = 0; i < words_.size(); ++i) {
      iree_unaligned_store_le_u64(bytes.data + i * 8, words_[i]);
    }
    return iree_ok_status();
  }

  iree_status_t CreateResult(iree_allocator_t allocator) {
    return loom_serve_finite_result_create(plan_, 32, 20, rows_.size(),
                                           rows_.data(), allocator, &result_);
  }

  // Two original requests with ragged row counts and reversed physical rows.
  std::array<uint64_t, 16> words_ = {32, 20, 2, 3, 0, 2, 7,  2,
                                     1,  11, 8, 2, 0, 2, 16, 1};
  // Protocol-derived row counts, independent of the external source plan.
  const std::array<iree_host_size_t, 2> rows_ = {2, 1};
  // Source-owned metadata, allowed to be reused after result creation.
  iree_vm_buffer_t* plan_ = nullptr;
  // Independent checked result, including before device storage is bound.
  loom_serve_finite_result_t* result_ = nullptr;
};

TEST_F(ResultTest, SourceBufferReuseCannotMutateCheckedRows) {
  IREE_ASSERT_OK(CreatePlan());
  IREE_ASSERT_OK(CreateResult(iree_allocator_system()));
  iree_byte_span_t source;
  IREE_ASSERT_OK(iree_vm_buffer_map_write(plan_, 0, sizeof(words_), &source));
  memset(source.data, 0xFF, source.data_length);
  iree_vm_buffer_release(std::exchange(plan_, nullptr));
  EXPECT_EQ(loom_serve_finite_result_output_length(result_), 32u);
  EXPECT_EQ(loom_serve_finite_result_length(result_), 20u);
  EXPECT_EQ(loom_serve_finite_result_input_tokens(result_, 0), 7u);
  EXPECT_EQ(loom_serve_finite_result_input_tokens(result_, 1), 11u);
  const auto first = loom_serve_finite_result_row(result_, 0, 0);
  const auto second = loom_serve_finite_result_row(result_, 0, 1);
  const auto third = loom_serve_finite_result_row(result_, 1, 0);
  EXPECT_EQ(first.offset, 8u);
  EXPECT_EQ(first.length, 8u);
  EXPECT_EQ(second.offset, 0u);
  EXPECT_EQ(second.length, 8u);
  EXPECT_EQ(third.offset, 16u);
  EXPECT_EQ(third.length, 4u);
}

TEST_F(ResultTest, RejectsMalformedSourcePlanBeforePublication) {
  const auto valid = words_;
  const std::array<std::pair<size_t, uint64_t>, 12> invalid = {
      {{0, 0},
       {0, 33},
       {1, 0},
       {1, 21},
       {2, 3},
       {3, 4},
       {4, 1},
       {5, 1},
       {7, 1},
       {10, 1},
       {11, 0},
       {15, UINT64_MAX}}};
  for (const auto& [word, value] : invalid) {
    SCOPED_TRACE(word);
    words_ = valid;
    words_[word] = value;
    IREE_ASSERT_OK(CreatePlan());
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          CreateResult(iree_allocator_system()));
    EXPECT_EQ(result_, nullptr);
    iree_vm_buffer_release(std::exchange(plan_, nullptr));
  }
}

TEST_F(ResultTest, AllocationFailureDoesNotPublishPartialResult) {
  IREE_ASSERT_OK(CreatePlan());
  const iree_allocator_t failing = {
      nullptr, [](void*, iree_allocator_command_t, const void*, void**) {
        return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
      }};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, CreateResult(failing));
  EXPECT_EQ(result_, nullptr);
}

// Observes actual allocation retirement while leaving allocator, queues, pool,
// copies, and result ownership on their production paths.
class LifetimeAllocator {
 public:
  iree_allocator_t allocator() { return {this, Control}; }

  size_t live() {
    std::lock_guard<std::mutex> lock(mutex_);
    return live_;
  }

  void AwaitEmpty() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return !live_; });
  }

 private:
  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* parameters, void** pointer) {
    auto* allocator = static_cast<LifetimeAllocator*>(self);
    const bool present = *pointer != nullptr;
    const auto system = iree_allocator_system();
    const auto status = system.ctl(system.self, command, parameters, pointer);
    if (iree_status_is_ok(status)) {
      std::lock_guard<std::mutex> lock(allocator->mutex_);
      if (command == IREE_ALLOCATOR_COMMAND_FREE) {
        allocator->live_ -= present;
      } else if (!present) {
        ++allocator->live_;
      }
      allocator->condition_.notify_all();
    }
    return status;
  }

  // Protects allocation observations from asynchronous queue destruction.
  std::mutex mutex_;
  // Publishes final release without depending on semaphore success.
  std::condition_variable condition_;
  // Exact number of owned result, snapshot, and capture-view allocations.
  size_t live_ = 0;
};

class SnapshotLifetimeTest : public ResultTest,
                             public ::testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        loom_runtime_host_create(4, nullptr, iree_allocator_system(), &host_));
    const loom_runtime_device_options_t options = {
        .driver = loom_runtime_host_task_driver(host_),
        .uri = IREE_SV("task"),
    };
    IREE_ASSERT_OK(loom_runtime_device_group_allocate(
        host_, 1, &options, iree_allocator_system(), &group_));
    device_ = loom_runtime_device_group_at(group_, 0);
    const loom_runtime_pool_options_t pool_options = {};
    IREE_ASSERT_OK(loom_runtime_device_group_create_pool(
        group_, {IREE_SV("pool"), 1}, &pool_options, &storage_owner_));
    IREE_ASSERT_OK(loom_runtime_device_create_execution(device_, storage_owner_,
                                                        &execution_));
    IREE_ASSERT_OK(
        loom_runtime_execution_reserve_workspace(execution_, 32, 16));
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        loom_runtime_device_handle(device_), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate_));
    uint64_t value = 1;
    IREE_ASSERT_OK(
        loom_runtime_execution_barrier(execution_, {1, &gate_, &value}));
    const auto request = iree_hal_make_pool_reservation_request(32, 16);
    IREE_ASSERT_OK(
        loom_runtime_execution_alloca(execution_, 1, &request, &source_));
    const uint32_t pattern = 0x3F800000;
    const iree_hal_transfer_operation_t fill = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
        .fill =
            {
                .target_buffer = source_,
                .length = 32,
                .pattern = &pattern,
                .pattern_length = sizeof(pattern),
            },
    };
    IREE_ASSERT_OK(
        loom_runtime_execution_transfer(execution_, 1, &fill, &value));
    IREE_ASSERT_OK(CreatePlan());
    IREE_ASSERT_OK(CreateResult(allocator_.allocator()));
    loom_serve_finite_result_bind(result_, execution_,
                                  iree_hal_buffer_allocated_buffer(source_));
  }

  void TearDown() override {
    if (gate_) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate_, 2, nullptr));
    }
    loom_serve_finite_snapshot_release(snapshot_);
    iree_hal_buffer_release(source_);
    ResultTest::TearDown();
    if (execution_) {
      auto status = loom_runtime_execution_drain(execution_);
      if (failed_) {
        IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, status);
      } else {
        IREE_EXPECT_OK(status);
      }
      loom_runtime_execution_release(execution_);
    }
    allocator_.AwaitEmpty();
    iree_hal_semaphore_release(gate_);
    IREE_EXPECT_OK(loom_runtime_device_group_free(group_));
    loom_runtime_host_release(host_);
  }

  // Result allocation evidence, outliving every asynchronous release callback.
  LifetimeAllocator allocator_;
  // Production task runtime and device ownership.
  loom_runtime_host_t* host_ = nullptr;
  // Application group enclosing the execution and all raw root captures.
  loom_runtime_device_group_t* group_ = nullptr;
  // Borrowed group device used by the producer and explicit snapshot.
  loom_runtime_device_t* device_ = nullptr;
  // Fixed pool retained through the final result and snapshot references.
  loom_runtime_pool_t* storage_owner_ = nullptr;
  // Producer timeline with an independent download branch.
  loom_runtime_execution_t* execution_ = nullptr;
  // Unsignaled input dependency, independent of failed output readiness.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Producer's tracked view, distinct from the result allocation root.
  iree_hal_buffer_t* source_ = nullptr;
  // Caller reference, relinquished before actual download retirement.
  loom_serve_finite_snapshot_t* snapshot_ = nullptr;
  // Whether the test deliberately failed the feedback semaphore.
  bool failed_ = false;
};

TEST_P(SnapshotLifetimeTest, EarlyReleasePreservesAcceptedDownloadStorage) {
  loom_runtime_execution_frontier_t frontier;
  IREE_ASSERT_OK(
      loom_serve_finite_result_download(result_, &snapshot_, &frontier));
  if (GetParam()) {
    iree_hal_semaphore_fail(frontier.semaphores[1],
                            iree_status_from_code(IREE_STATUS_CANCELLED));
    failed_ = true;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_CANCELLED,
        iree_hal_semaphore_wait(frontier.semaphores[1], frontier.values[1],
                                iree_infinite_timeout(),
                                IREE_ASYNC_WAIT_FLAG_NONE));
  }
  loom_serve_finite_snapshot_release(std::exchange(snapshot_, nullptr));
  loom_serve_finite_result_release(std::exchange(result_, nullptr));
  iree_hal_buffer_release(std::exchange(source_, nullptr));
  // Queue capture, not success/failure readiness, owns the host snapshot and
  // source view with its callback. Result metadata is already independent.
  EXPECT_EQ(allocator_.live(), 2u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  allocator_.AwaitEmpty();
  EXPECT_EQ(allocator_.live(), 0u);
}

INSTANTIATE_TEST_SUITE_P(FailedReadiness, SnapshotLifetimeTest,
                         ::testing::Bool());

class SuballocationLifetimeTest : public ResultTest {
 protected:
  void SetUp() override {
    loom_runtime_retirement_initialize(&producer_);
    IREE_ASSERT_OK(
        loom_runtime_host_create(2, nullptr, iree_allocator_system(), &host_));
    const loom_runtime_device_options_t options = {
        .driver = loom_runtime_host_task_driver(host_),
        .uri = IREE_SV("task"),
    };
    IREE_ASSERT_OK(loom_runtime_device_group_allocate(
        host_, 1, &options, iree_allocator_system(), &group_));
    auto* device = loom_runtime_device_group_at(group_, 0);
    const loom_runtime_pool_options_t pool_options = {};
    loom_runtime_pool_t* storage_owner = nullptr;
    IREE_ASSERT_OK(loom_runtime_device_group_create_pool(
        group_, {IREE_SV("pool"), 1}, &pool_options, &storage_owner));
    IREE_ASSERT_OK(loom_runtime_device_create_execution(device, storage_owner,
                                                        &execution_));
    auto* workspace = loom_runtime_pool_workspace(storage_owner);
    IREE_ASSERT_OK(loom_runtime_workspace_reserve(workspace, 2048, 256));
    IREE_ASSERT_OK(loom_runtime_workspace_acquire_pool(workspace, &pool_));
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(256, 256),
        iree_infinite_timeout(), &prefix_));
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        pool_, iree_hal_make_pool_reservation_request(32, 256),
        iree_infinite_timeout(), &output_));
  }

  void TearDown() override {
    iree_hal_buffer_release(view_);
    loom_serve_finite_snapshot_release(snapshot_);
    ResultTest::TearDown();
    iree_hal_buffer_release(output_);
    iree_hal_buffer_release(prefix_);
    if (execution_) {
      IREE_EXPECT_OK(loom_runtime_execution_drain(execution_));
    }
    loom_runtime_retirement_await(&producer_);
    loom_runtime_retirement_deinitialize(&producer_);
    loom_runtime_execution_release(execution_);
    iree_hal_pool_release(pool_);
    IREE_EXPECT_OK(loom_runtime_device_group_free(group_));
    loom_runtime_host_release(host_);
  }

  // Runtime owner kept through every pool and queue consumer.
  loom_runtime_host_t* host_ = nullptr;
  // One task device with its application-owned persistent workspace.
  loom_runtime_device_group_t* group_ = nullptr;
  // Producer and download timelines, also retained by the result.
  loom_runtime_execution_t* execution_ = nullptr;
  // Shared logical allocator outliving its ordinary suballocated buffers.
  iree_hal_pool_t* pool_ = nullptr;
  // Occupied prefix forcing the output to a nonzero physical offset.
  iree_hal_buffer_t* prefix_ = nullptr;
  // Independent output storage with its own reservation-release obligation.
  iree_hal_buffer_t* output_ = nullptr;
  // Producer tracking, which must retire while the result remains usable.
  loom_runtime_retirement_t producer_;
  // Producer-only view captured by the upload operation.
  iree_hal_buffer_t* view_ = nullptr;
  // Explicit host result, independent of the device output after download.
  loom_serve_finite_snapshot_t* snapshot_ = nullptr;
  // Upload payload outliving accepted queue work, including failed checks.
  const std::array<float, 5> values_ = {11, 12, 21, 22, 31};
};

TEST_F(SuballocationLifetimeTest, RetainsLogicalStorageAfterProducerRetires) {
  ASSERT_NE(iree_hal_buffer_byte_offset(output_), 0u);
  ASSERT_NE(output_, iree_hal_buffer_allocated_buffer(output_));
  IREE_ASSERT_OK(CreatePlan());
  IREE_ASSERT_OK(CreateResult(iree_allocator_system()));
  view_ = output_;
  iree_hal_buffer_retain(view_);
  IREE_ASSERT_OK(loom_runtime_retirement_track(&producer_, &view_,
                                               iree_allocator_system()));
  const iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload =
          {
              .source = values_.data(),
              .target_buffer = view_,
              .length = sizeof(values_),
          },
  };
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_runtime_execution_transfer(execution_, 1, &upload, &completion));
  loom_serve_finite_result_bind(result_, execution_, output_);
  const auto row = loom_serve_finite_result_row(result_, 0, 0);
  EXPECT_EQ(row.buffer, output_);
  EXPECT_EQ(row.offset, 8u);
  iree_hal_buffer_release(std::exchange(output_, nullptr));
  iree_hal_buffer_release(std::exchange(view_, nullptr));
  IREE_ASSERT_OK(loom_runtime_execution_wait(execution_, completion));
  loom_runtime_retirement_await(&producer_);
  iree_hal_pool_stats_t statistics;
  iree_hal_pool_query_stats(pool_, &statistics);
  EXPECT_EQ(statistics.reservation_count, 2u);

  loom_runtime_execution_frontier_t ready;
  IREE_ASSERT_OK(
      loom_serve_finite_result_download(result_, &snapshot_, &ready));
  loom_serve_finite_result_release(std::exchange(result_, nullptr));
  IREE_ASSERT_OK(
      loom_runtime_execution_feedback_wait(execution_, ready.values[1]));
  const auto first = loom_serve_finite_snapshot_row(snapshot_, 0, 0);
  const auto second = loom_serve_finite_snapshot_row(snapshot_, 0, 1);
  const auto third = loom_serve_finite_snapshot_row(snapshot_, 1, 0);
  EXPECT_EQ(first.data_length, 2 * sizeof(float));
  EXPECT_EQ(second.data_length, 2 * sizeof(float));
  EXPECT_EQ(third.data_length, sizeof(float));
  EXPECT_EQ(iree_unaligned_load_le_f32(first.data), 21);
  EXPECT_EQ(iree_unaligned_load_le_f32(first.data + 4), 22);
  EXPECT_EQ(iree_unaligned_load_le_f32(second.data), 11);
  EXPECT_EQ(iree_unaligned_load_le_f32(second.data + 4), 12);
  EXPECT_EQ(iree_unaligned_load_le_f32(third.data), 31);
}

}  // namespace
