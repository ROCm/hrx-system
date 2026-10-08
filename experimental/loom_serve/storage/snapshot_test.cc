// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/storage/snapshot.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#include "experimental/loom_serve/runtime/device.h"
#include "experimental/loom_serve/runtime/retirement.h"
#include "experimental/loom_serve/storage/memory.h"
#include "iree/async/operations/semaphore.h"
#include "iree/task/executor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// More logical ranges than one native transfer batch can hold.
constexpr size_t kRangeCount = 70;

class SnapshotTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_serve_retirement_initialize(&retirement);
    const loom_serve_device_options_t options = {
        .uri = IREE_SV("amdgpu"),
        .backing = LOOM_SERVE_DEVICE_BACKING_ELASTIC,
        .slab_size = kSlabSize,
        .memory_limit = 256 * kSlabSize};
    loom_serve_host_t* host = nullptr;
    IREE_ASSERT_OK(loom_serve_host_create(1, &host, iree_allocator_system()));
    auto status = loom_serve_device_group_create(host, 1, &options, &group,
                                                 iree_allocator_system());
    loom_serve_host_release(host);
    IREE_ASSERT_OK(status);
    device = loom_serve_device_group_at(group, 0);
    IREE_ASSERT_OK(loom_serve_device_create_execution(device, &execution));
    pool = loom_serve_device_memory_pool(device);
  }

  void TearDown() override {
    if (execution) {
      IREE_EXPECT_STATUS_IS(terminal_code,
                            loom_serve_execution_drain(execution));
    }
    loom_serve_snapshot_release(snapshot);
    for (auto* root : roots) {
      iree_hal_buffer_release(root);
    }
    loom_serve_retirement_deinitialize(&retirement);
    for (auto* reservation : reservations) {
      IREE_EXPECT_OK(loom_serve_virtual_buffer_destroy(reservation));
    }
    loom_serve_execution_release(execution);
    IREE_EXPECT_OK(loom_serve_device_group_destroy(group));
  }

  // Capacity granularity, independent of the packed transfer lengths.
  static constexpr uint64_t kSlabSize = 2 * 1024 * 1024;
  // Source contents deliberately use wide addresses.
  static constexpr uint64_t kHighOffset = (uint64_t{1} << 32) + 512;
  // Application group enclosing all device resources.
  loom_serve_device_group_t* group = nullptr;
  // Borrowed device record within the application group.
  loom_serve_device_t* device = nullptr;
  // Owned model submission timeline.
  loom_serve_execution_t* execution = nullptr;
  // Physical budget enclosing both independent roots.
  loom_serve_memory_pool_t* pool = nullptr;
  // Accounting survives destruction and replacement of the roots.
  loom_serve_memory_statistics_t statistics = {};
  // Counter enclosing all exported root references before address release.
  loom_serve_retirement_t retirement = {};
  // Owned tracked views retained through successful or failed transfers.
  std::array<iree_hal_buffer_t*, 2> roots = {};
  // Source roots are destroyed before replacement roots are created here.
  std::array<loom_serve_virtual_buffer_t*, 2> reservations = {};
  // Host image remains valid without any device root from capture.
  loom_serve_snapshot_t* snapshot = nullptr;
  // Expected domain result, including intentionally failed feedback readiness.
  iree_status_code_t terminal_code = IREE_STATUS_OK;
};

TEST_F(SnapshotTest, LogicalRangesSurviveSourceDestructionAndFreshPlacement) {
  for (size_t i = 0; i < roots.size(); ++i) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_create(
        pool, kHighOffset + kSlabSize, 256, &statistics, &reservations[i]));
    roots[i] = loom_serve_virtual_buffer_handle(reservations[i]);
    iree_hal_buffer_retain(roots[i]);
    IREE_ASSERT_OK(loom_serve_retirement_track(&retirement, &roots[i],
                                               iree_allocator_system()));
    IREE_ASSERT_OK(
        loom_serve_virtual_buffer_commit(reservations[i], kHighOffset, 32768));
  }
  std::array<loom_serve_snapshot_range_t, kRangeCount> ranges;
  std::array<uint32_t, kRangeCount> patterns;
  std::array<iree_hal_transfer_operation_t, kRangeCount> fills = {};
  for (size_t i = 0; i < ranges.size(); ++i) {
    // Logical order intentionally opposes physical offset order.
    ranges[i] = {i % 2, kHighOffset + (kRangeCount - i - 1) * 256, 16};
    patterns[i] = 0xABC00000u + static_cast<uint32_t>(i * 17);
    fills[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
    fills[i].fill = {roots[ranges[i].buffer_index], ranges[i].offset,
                     ranges[i].length, &patterns[i], sizeof(patterns[i])};
  }
  uint64_t completion = 0;
  IREE_ASSERT_OK(loom_serve_execution_transfer(execution, fills.size(),
                                               fills.data(), &completion));
  IREE_ASSERT_OK(loom_serve_snapshot_capture(
      execution, roots.size(), roots.data(), ranges.size(), ranges.data(),
      &snapshot, iree_allocator_system()));
  EXPECT_EQ(loom_serve_snapshot_size(snapshot), ranges.size() * 16);
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  for (auto*& root : roots) {
    iree_hal_buffer_release(std::exchange(root, nullptr));
  }
  loom_serve_retirement_await(&retirement);
  for (auto*& reservation : reservations) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_destroy(reservation));
    reservation = nullptr;
  }
  EXPECT_EQ(statistics.committed_bytes, 0u);
  EXPECT_EQ(statistics.reserved_bytes, 0u);
  for (size_t i = 0; i < roots.size(); ++i) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_create(
        pool, kSlabSize, 256, &statistics, &reservations[i]));
    roots[i] = loom_serve_virtual_buffer_handle(reservations[i]);
    iree_hal_buffer_retain(roots[i]);
    IREE_ASSERT_OK(loom_serve_retirement_track(&retirement, &roots[i],
                                               iree_allocator_system()));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(reservations[i], 0, 32768));
  }
  for (size_t i = 0; i < ranges.size(); ++i) {
    ranges[i].buffer_index = 1 - ranges[i].buffer_index;
    ranges[i].offset = 1024 + i * 128;
  }
  IREE_ASSERT_OK(loom_serve_snapshot_restore(snapshot, execution, roots.size(),
                                             roots.data(), ranges.size(),
                                             ranges.data()));
  loom_serve_snapshot_release(std::exchange(snapshot, nullptr));
  std::array<std::array<uint32_t, 4>, kRangeCount> observed = {};
  std::array<iree_hal_transfer_operation_t, kRangeCount> downloads = {};
  for (size_t i = 0; i < ranges.size(); ++i) {
    downloads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    downloads[i].download = {roots[ranges[i].buffer_index], ranges[i].offset,
                             observed[i].data(), ranges[i].length};
  }
  IREE_ASSERT_OK(loom_serve_execution_transfer(execution, downloads.size(),
                                               downloads.data(), &completion));
  IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
  for (size_t i = 0; i < observed.size(); ++i) {
    for (const auto value : observed[i]) {
      EXPECT_EQ(value, patterns[i]);
    }
  }
}

TEST_F(SnapshotTest, LargeSparseStateSurvivesBackingReuseAndFreshPlacement) {
  constexpr uint64_t kStateBytes = 156893184;
  constexpr uint64_t kStateStride = kStateBytes + 256 + 2048 + 256;
  constexpr uint64_t kCacheOrigin = kHighOffset + kStateStride * 17;
  constexpr uint64_t kPlaneStride = 2048 * 2048;
  constexpr uint64_t kPageBytes = 131072;
  constexpr size_t kPageCount = 136;
  constexpr uint64_t kPayloadBytes =
      kStateBytes + kPageBytes * kPageCount + 12 + 2048 + 32 + 20480;
  for (size_t i = 0; i < roots.size(); ++i) {
    IREE_ASSERT_OK(
        loom_serve_virtual_buffer_create(pool, kCacheOrigin + kPlaneStride * 32,
                                         256, &statistics, &reservations[i]));
    roots[i] = loom_serve_virtual_buffer_handle(reservations[i]);
    iree_hal_buffer_retain(roots[i]);
    IREE_ASSERT_OK(loom_serve_retirement_track(&retirement, &roots[i],
                                               iree_allocator_system()));
  }
  const uint64_t private_origin = kHighOffset + kStateStride * 7;
  std::array<loom_serve_snapshot_range_t, kPageCount + 5> ranges = {};
  ranges[0] = {0, private_origin, 12};
  ranges[1] = {0, private_origin + 256, kStateBytes};
  ranges[2] = {0, private_origin + 256 + kStateBytes, 2048};
  ranges[3] = {0, private_origin + 256 + kStateBytes + 2048, 32};
  ranges[4] = {1, kHighOffset, 20480};
  for (size_t i = 0; i < kPageCount; ++i) {
    ranges[i + 5] = {
        i < 128 ? 0u : 1u,
        kCacheOrigin + ((i / 4) % 32) * kPlaneStride + (i % 4) * kPageBytes,
        kPageBytes};
  }
  std::vector<uint32_t> expected(kPayloadBytes / sizeof(uint32_t));
  for (size_t i = 0; i < expected.size(); ++i) {
    // Every word differs, including between chunks of the same large range.
    expected[i] = (static_cast<uint32_t>(i) * 2654435761u) ^ 0x12345678u;
  }
  std::array<iree_hal_transfer_operation_t, kPageCount + 5> uploads = {};
  size_t offset = 0;
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto& range = ranges[i];
    IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(
        reservations[range.buffer_index], range.offset, range.length));
    uploads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
    uploads[i].upload = {expected.data() + offset, roots[range.buffer_index],
                         range.offset, range.length};
    offset += range.length / sizeof(uint32_t);
  }
  uint64_t completion = 0;
  IREE_ASSERT_OK(loom_serve_execution_transfer(execution, uploads.size(),
                                               uploads.data(), &completion));
  std::vector<uint32_t> observed(kPayloadBytes / sizeof(uint32_t));
  std::array<iree_hal_transfer_operation_t, kPageCount + 5> downloads = {};
  offset = 0;
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto& range = ranges[i];
    downloads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    downloads[i].download = {roots[range.buffer_index], range.offset,
                             observed.data() + offset, range.length};
    offset += range.length / sizeof(uint32_t);
  }
  IREE_ASSERT_OK(loom_serve_execution_transfer(execution, downloads.size(),
                                               downloads.data(), &completion));
  IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
  for (size_t i = 0; i < observed.size(); ++i) {
    ASSERT_EQ(observed[i], expected[i]) << "before capture, word " << i;
  }
  IREE_ASSERT_OK(loom_serve_snapshot_capture(
      execution, roots.size(), roots.data(), ranges.size(), ranges.data(),
      &snapshot, iree_allocator_system()));
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  for (auto* reservation : reservations) {
    loom_serve_virtual_buffer_begin_trim(reservation);
    IREE_ASSERT_OK(loom_serve_virtual_buffer_trim(reservation));
  }
  EXPECT_EQ(statistics.committed_bytes, 0u);
  ranges[1].offset = kHighOffset + kStateStride * 5 + 256;
  constexpr uint32_t fresh_blocks[] = {16, 2, 31, 10};
  for (size_t i = 0; i < kPageCount; ++i) {
    ranges[i + 5].offset = kCacheOrigin + ((i / 4) % 32) * kPlaneStride +
                           fresh_blocks[i % 4] * kPageBytes;
  }
  for (const auto& range : ranges) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(
        reservations[range.buffer_index], range.offset, range.length));
  }
  IREE_ASSERT_OK(loom_serve_snapshot_restore(snapshot, execution, roots.size(),
                                             roots.data(), ranges.size(),
                                             ranges.data()));
  loom_serve_snapshot_release(std::exchange(snapshot, nullptr));
  std::fill(observed.begin(), observed.end(), 0);
  offset = 0;
  for (size_t i = 0; i < ranges.size(); ++i) {
    const auto& range = ranges[i];
    downloads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    downloads[i].download = {roots[range.buffer_index], range.offset,
                             observed.data() + offset, range.length};
    offset += range.length / sizeof(uint32_t);
  }
  IREE_ASSERT_OK(loom_serve_execution_transfer(execution, downloads.size(),
                                               downloads.data(), &completion));
  IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
  offset = 0;
  for (size_t i = 0; i < ranges.size(); ++i) {
    for (size_t j = 0; j < ranges[i].length / sizeof(uint32_t); ++j) {
      ASSERT_EQ(observed[offset + j], expected[offset + j])
          << "range " << i << ", word " << j;
    }
    offset += ranges[i].length / sizeof(uint32_t);
  }
}

// Submission must return the only worker before its predecessor can resolve.
// The caller relinquishes both its snapshot and tracked root references while
// native transfers are held. Failed feedback readiness also leaves actual work
// held until its completion callback schedules the independent unblock task.
struct SnapshotTransfer {
  static iree_status_t Submit(iree_task_process_t* process,
                              const iree_task_worker_context_t*,
                              iree_task_process_drain_result_t* result) {
    auto* self = static_cast<SnapshotTransfer*>(process->user_data);
    auto local_ranges = self->ranges;
    auto status =
        self->snapshot
            ? loom_serve_snapshot_restore(
                  self->snapshot, self->execution, self->roots.size(),
                  self->roots.data(), local_ranges.size(), local_ranges.data())
            : loom_serve_snapshot_capture(
                  self->execution, self->roots.size(), self->roots.data(),
                  local_ranges.size(), local_ranges.data(), &self->snapshot,
                  iree_allocator_system());
    loom_serve_snapshot_release(std::exchange(self->snapshot, nullptr));
    local_ranges.fill({});
    std::array<iree_hal_transfer_operation_t, kRangeCount> downloads = {};
    for (size_t i = 0; i < downloads.size(); ++i) {
      const auto& range = self->ranges[i];
      downloads[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
      downloads[i].download = {self->roots[range.buffer_index], range.offset,
                               self->observed[i].data(), range.length};
    }
    uint64_t completion = 0;
    if (iree_status_is_ok(status)) {
      status = loom_serve_execution_feedback(self->execution, downloads.size(),
                                             downloads.data(), &completion);
    }
    for (auto*& root : self->roots) {
      iree_hal_buffer_release(std::exchange(root, nullptr));
    }
    if (iree_status_is_ok(status)) {
      self->frontier = loom_serve_execution_frontier(self->execution);
      for (size_t i = 0; i < self->semaphores.size(); ++i) {
        self->semaphores[i] = reinterpret_cast<iree_async_semaphore_t*>(
            self->frontier.semaphores[i]);
      }
      self->wait.base.type = IREE_ASYNC_OPERATION_TYPE_SEMAPHORE_WAIT;
      self->wait.base.completion_fn = Ready;
      self->wait.base.user_data = self;
      self->wait.semaphores = self->semaphores.data();
      self->wait.values = self->frontier.values;
      self->wait.count = self->semaphores.size();
      self->wait.mode = IREE_ASYNC_WAIT_MODE_ALL;
      status = iree_async_proactor_submit_one(self->proactor, &self->wait.base);
    }
    if (iree_status_is_ok(status) && self->fail_readiness) {
      // Fail the feedback fork without cancelling the held work branch: its
      // snapshot uploads/downloads must still access their host payload.
      iree_hal_semaphore_fail(self->frontier.semaphores[1],
                              iree_status_from_code(IREE_STATUS_CANCELLED));
    } else if (!iree_status_is_ok(status)) {
      Ready(self, nullptr, status, IREE_ASYNC_COMPLETION_FLAG_NONE);
    }
    if (!self->fail_readiness) {
      iree_task_executor_schedule_process(self->executor, &self->unblock);
    }
    result->did_work = true;
    result->completed = true;
    return iree_ok_status();
  }

  static iree_status_t Unblock(iree_task_process_t* process,
                               const iree_task_worker_context_t*,
                               iree_task_process_drain_result_t* result) {
    auto* self = static_cast<SnapshotTransfer*>(process->user_data);
    iree_slim_mutex_lock(&self->retirement->mutex);
    self->held_roots = self->retirement->count;
    iree_slim_mutex_unlock(&self->retirement->mutex);
    auto status = iree_hal_semaphore_signal(self->gate, 1, nullptr);
    if (!iree_status_is_ok(status)) {
      iree_hal_semaphore_fail(self->gate, iree_status_clone(status));
    }
    {
      std::lock_guard<std::mutex> lock(self->mutex);
      self->status = iree_status_join(self->status, status);
    }
    result->did_work = true;
    result->completed = true;
    return iree_ok_status();
  }

  static void Ready(void* user_data, iree_async_operation_t*,
                    iree_status_t status, iree_async_completion_flags_t) {
    auto* self = static_cast<SnapshotTransfer*>(user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    self->status = iree_status_join(self->status, status);
    self->ready = true;
    if (self->fail_readiness) {
      iree_task_executor_schedule_process(self->executor, &self->unblock);
    }
    self->condition.notify_one();
  }

  static void Release(iree_task_process_t* process) {
    auto* self = static_cast<SnapshotTransfer*>(process->user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    ++self->released;
    self->condition.notify_one();
  }

  iree_status_t Run() {
    iree_task_process_initialize(Submit, 0, 1, &submit);
    iree_task_process_initialize(Unblock, 0, 1, &unblock);
    for (auto* process : {&submit, &unblock}) {
      process->user_data = this;
      process->release_fn = Release;
    }
    iree_task_executor_schedule_process(executor, &submit);
    std::unique_lock<std::mutex> lock(mutex);
    condition.wait(lock, [&] { return ready && released == 2; });
    return std::exchange(status, iree_ok_status());
  }

  // Borrowed shared executor with exactly one worker.
  iree_task_executor_t* executor = nullptr;
  // Borrowed host proactor driven independently of the task worker.
  iree_async_proactor_t* proactor = nullptr;
  // Borrowed model work/feedback timelines.
  loom_serve_execution_t* execution = nullptr;
  // Borrowed counter for actual final root use, independent of readiness.
  loom_serve_retirement_t* retirement = nullptr;
  // Borrowed predecessor resolved by a separate turn on the only worker.
  iree_hal_semaphore_t* gate = nullptr;
  // Owned completed snapshot for restore, or null to capture a new one.
  loom_serve_snapshot_t* snapshot = nullptr;
  // Owned views consumed after queue submission, not after completion.
  std::array<iree_hal_buffer_t*, 2> roots = {};
  // Logical plan copied into the submitting task's stack before use.
  std::array<loom_serve_snapshot_range_t, kRangeCount> ranges = {};
  // Feedback bytes retained by the outer owner through actual root retirement.
  std::array<std::array<uint32_t, 4>, kRangeCount> observed = {};
  // Submitting service turn, which must not wait on device completion.
  iree_task_process_t submit = {};
  // Independent task needed to release the transfer predecessor.
  iree_task_process_t unblock = {};
  // Multi-wait storage retained through the terminal callback.
  iree_async_semaphore_wait_operation_t wait = {};
  // Borrowed timelines and values accepted before registering readiness.
  loom_serve_execution_frontier_t frontier = {};
  // Toll-free async identities for the captured HAL semaphores.
  std::array<iree_async_semaphore_t*, 2> semaphores = {};
  // Injects failed feedback readiness before resolving held work.
  bool fail_readiness = false;
  // Serializes terminal status with final callback/process retirement.
  std::mutex mutex;
  // Joins only the outer test thread, never a task worker.
  std::condition_variable condition;
  // Number of roots still queue-owned immediately before resolving the gate.
  size_t held_roots = 0;
  // Terminal readiness has been published by the proactor callback.
  bool ready = false;
  // Number of executor-owned processes that have retired.
  unsigned released = 0;
  // Owned result transferred to the outer caller after task retirement.
  iree_status_t status = iree_ok_status();
};

class AsyncSnapshotTest
    : public SnapshotTest,
      public ::testing::WithParamInterface<std::tuple<bool, bool>> {
 protected:
  void TearDown() override {
    // Failed assertions must still release any accepted native work.
    if (transfer.gate) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(transfer.gate, 2, nullptr));
    }
    loom_serve_snapshot_release(transfer.snapshot);
    for (auto* root : transfer.roots) {
      iree_hal_buffer_release(root);
    }
    iree_hal_semaphore_release(transfer.gate);
    SnapshotTest::TearDown();
  }

  // Task and I/O storage whose lifetime includes terminal retirement.
  SnapshotTransfer transfer;
};

TEST_P(AsyncSnapshotTest, OneWorkerSubmitsAndReleasesBeforeNativeRetirement) {
  const auto [restore, fail_readiness] = GetParam();
  transfer.execution = execution;
  transfer.retirement = &retirement;
  transfer.fail_readiness = fail_readiness;
  auto* host = loom_serve_device_host(device);
  transfer.executor = loom_serve_host_executor(host);
  IREE_ASSERT_OK(iree_async_proactor_pool_get(
      loom_serve_host_proactor_pool(host), 0, &transfer.proactor));
  for (size_t i = 0; i < transfer.roots.size(); ++i) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_create(
        pool, kHighOffset + kSlabSize, 256, &statistics, &reservations[i]));
    IREE_ASSERT_OK(
        loom_serve_virtual_buffer_commit(reservations[i], kHighOffset, 32768));
    transfer.roots[i] = loom_serve_virtual_buffer_handle(reservations[i]);
    iree_hal_buffer_retain(transfer.roots[i]);
    IREE_ASSERT_OK(loom_serve_retirement_track(&retirement, &transfer.roots[i],
                                               iree_allocator_system()));
  }
  std::array<uint32_t, kRangeCount> patterns;
  std::array<iree_hal_transfer_operation_t, kRangeCount> fills = {};
  for (size_t i = 0; i < transfer.ranges.size(); ++i) {
    transfer.ranges[i] = {i % 2, kHighOffset + (kRangeCount - i - 1) * 256, 16};
    const auto& range = transfer.ranges[i];
    patterns[i] = 0xFED00000u + static_cast<uint32_t>(i * 31);
    fills[i].type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL;
    fills[i].fill = {transfer.roots[range.buffer_index], range.offset,
                     range.length, &patterns[i], sizeof(patterns[i])};
  }
  uint64_t completion = 0;
  IREE_ASSERT_OK(loom_serve_execution_transfer(execution, fills.size(),
                                               fills.data(), &completion));
  if (restore) {
    IREE_ASSERT_OK(loom_serve_snapshot_capture(
        execution, transfer.roots.size(), transfer.roots.data(),
        transfer.ranges.size(), transfer.ranges.data(), &transfer.snapshot,
        iree_allocator_system()));
  }
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &transfer.gate));
  uint64_t held = 1;
  IREE_ASSERT_OK(
      loom_serve_execution_barrier(execution, {1, &transfer.gate, &held}));
  if (restore) {
    // Restore targets different physical ranges, not the original input.
    for (auto& range : transfer.ranges) {
      range.buffer_index = 1 - range.buffer_index;
      range.offset += 32768;
    }
  }
  terminal_code = fail_readiness ? IREE_STATUS_CANCELLED : IREE_STATUS_OK;
  IREE_ASSERT_STATUS_IS(terminal_code, transfer.Run());
  EXPECT_EQ(transfer.held_roots, transfer.roots.size());
  EXPECT_EQ(transfer.snapshot, nullptr);
  loom_serve_retirement_await(&retirement);
  // Feedback may have failed, but native work itself is required to finish.
  IREE_ASSERT_OK(
      loom_serve_execution_wait(execution, transfer.frontier.values[0]));
  if (!fail_readiness) {
    for (size_t i = 0; i < patterns.size(); ++i) {
      for (const auto value : transfer.observed[i]) {
        EXPECT_EQ(value, patterns[i]);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(DirectionAndReadiness, AsyncSnapshotTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

}  // namespace
