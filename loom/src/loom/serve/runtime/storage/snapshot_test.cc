// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/storage/snapshot.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#include "iree/async/operations/semaphore.h"
#include "iree/task/executor.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/retirement.h"
#include "loom/serve/runtime/storage/memory.h"
#include "loom/serve/testing/device.h"

namespace {

// More logical ranges than one native transfer batch can hold.
constexpr size_t kRangeCount = 70;
// Capacity granularity, independent of the packed transfer lengths.
constexpr uint64_t kSlabSize = 2 * 1024 * 1024;

std::array<loom_serve_snapshot_buffer_t, 2> SnapshotBuffers(
    loom_serve_execution_t* execution,
    const std::array<iree_hal_buffer_t*, 2>& roots) {
  return {{{roots[0], execution}, {roots[1], execution}}};
}

class SnapshotEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    IREE_ASSERT_OK(
        loom_serve_host_create(1, nullptr, iree_allocator_system(), &host));
    IREE_ASSERT_OK(loom_serve_test_driver_acquire(host, &driver));
    const loom_serve_device_options_t options = {
        .driver = driver,
        .uri = loom_serve_test_device()->device_uri,
    };
    const loom_serve_pool_options_t pool_options = {
        .elastic = true,
        .weight_slab_size = kSlabSize,
        .state_slab_size = kSlabSize,
        .memory_limit = 256 * kSlabSize,
    };
    IREE_ASSERT_OK(loom_serve_device_group_allocate(
        host, 1, &options, iree_allocator_system(), &group));
    IREE_ASSERT_OK(loom_serve_device_group_create_pool(
        group, {IREE_SV("pool"), 1}, &pool_options, &pool));
  }

  void TearDown() override {
    IREE_EXPECT_OK(loom_serve_device_group_free(group));
    iree_hal_driver_release(driver);
    loom_serve_host_release(host);
  }

  // Application infrastructure enclosing both synchronous and async suites.
  loom_serve_host_t* host = nullptr;
  // One native connection enclosing the shared device group.
  iree_hal_driver_t* driver = nullptr;
  // One native device group shared by every case in the process.
  loom_serve_device_group_t* group = nullptr;
  // Shared physical backing; each case owns and retires its reservations.
  loom_serve_pool_t* pool = nullptr;
};

SnapshotEnvironment* const environment = static_cast<SnapshotEnvironment*>(
    ::testing::AddGlobalTestEnvironment(new SnapshotEnvironment));

class SnapshotTest : public ::testing::Test {
 protected:
  void SetUp() override {
    loom_serve_retirement_initialize(&retirement);
    storage_owner = environment->pool;
    device = loom_serve_device_group_at(environment->group, 0);
    IREE_ASSERT_OK(
        loom_serve_device_create_execution(device, storage_owner, &execution));
    pool = loom_serve_pool_memory(storage_owner);
  }

  void TearDown() override {
    if (gate) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate, 2, nullptr));
    }
    if (execution) {
      loom_serve_execution_t* executions[] = {execution, peer_execution};
      IREE_EXPECT_STATUS_IS(terminal_code,
                            loom_serve_execution_drain_all(
                                IREE_ARRAYSIZE(executions), executions));
    }
    loom_serve_snapshot_release(snapshot);
    for (auto* root : roots) {
      iree_hal_buffer_release(root);
    }
    loom_serve_retirement_await(&retirement);
    loom_serve_retirement_deinitialize(&retirement);
    for (auto* reservation : reservations) {
      IREE_EXPECT_OK(loom_serve_virtual_buffer_free(reservation));
    }
    loom_serve_execution_release(peer_execution);
    loom_serve_execution_release(execution);
    iree_hal_semaphore_release(gate);
    EXPECT_EQ(statistics.reserved_bytes, 0u);
    EXPECT_EQ(statistics.committed_bytes, 0u);
    EXPECT_EQ(statistics.pending_bytes, 0u);
  }

  void AllocateRoots() {
    for (size_t i = 0; i < roots.size(); ++i) {
      IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
          pool, loom_serve_pool_state_geometry(storage_owner), kSlabSize, 256,
          &statistics, &reservations[i]));
      roots[i] = loom_serve_virtual_buffer_handle(reservations[i]);
      iree_hal_buffer_retain(roots[i]);
      IREE_ASSERT_OK(loom_serve_retirement_track(&retirement, &roots[i],
                                                 iree_allocator_system()));
      IREE_ASSERT_OK(
          loom_serve_virtual_buffer_commit(reservations[i], 0, 32768));
    }
  }

  void Hold(loom_serve_execution_t* owner) {
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
        IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &gate));
    uint64_t value = 1;
    IREE_ASSERT_OK(loom_serve_execution_barrier(owner, {1, &gate, &value}));
  }

  // Source contents deliberately use wide addresses.
  static constexpr uint64_t kHighOffset = (uint64_t{1} << 32) + 512;
  // Borrowed process-owned device, shared by every case.
  loom_serve_device_t* device = nullptr;
  // Borrowed shared storage owner for case-owned reservations.
  loom_serve_pool_t* storage_owner = nullptr;
  // Owned model submission timeline.
  loom_serve_execution_t* execution = nullptr;
  // Optional independent ordering domain sharing the physical pool.
  loom_serve_execution_t* peer_execution = nullptr;
  // Explicit predecessor, released by teardown after any failed assertion.
  iree_hal_semaphore_t* gate = nullptr;
  // Physical budget enclosing both independent roots.
  loom_serve_memory_pool_t* pool = nullptr;
  // Accounting survives destruction and replacement of the roots.
  loom_runtime_storage_statistics_t statistics = {};
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
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        pool, loom_serve_pool_state_geometry(storage_owner),
        kHighOffset + kSlabSize, 256, &statistics, &reservations[i]));
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
      roots.size(), SnapshotBuffers(execution, roots).data(), ranges.size(),
      ranges.data(), iree_allocator_system(), &snapshot));
  EXPECT_EQ(loom_serve_snapshot_size(snapshot), ranges.size() * 16);
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  for (auto*& root : roots) {
    iree_hal_buffer_release(std::exchange(root, nullptr));
  }
  loom_serve_retirement_await(&retirement);
  for (auto*& reservation : reservations) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_free(reservation));
    reservation = nullptr;
  }
  EXPECT_EQ(statistics.committed_bytes, 0u);
  EXPECT_EQ(statistics.reserved_bytes, 0u);
  for (size_t i = 0; i < roots.size(); ++i) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        pool, loom_serve_pool_state_geometry(storage_owner), kSlabSize, 256,
        &statistics, &reservations[i]));
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
  IREE_ASSERT_OK(loom_serve_snapshot_restore(
      snapshot, roots.size(), SnapshotBuffers(execution, roots).data(),
      ranges.size(), ranges.data()));
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

TEST_F(SnapshotTest, CaptureAndRestoreMayGroupTheSameLogicalBytesDifferently) {
  for (size_t i = 0; i < roots.size(); ++i) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        pool, loom_serve_pool_state_geometry(storage_owner), kSlabSize, 256,
        &statistics, &reservations[i]));
    roots[i] = loom_serve_virtual_buffer_handle(reservations[i]);
    iree_hal_buffer_retain(roots[i]);
    IREE_ASSERT_OK(loom_serve_retirement_track(&retirement, &roots[i],
                                               iree_allocator_system()));
    IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(reservations[i], 0, 512));
  }
  const std::array<uint32_t, 6> expected = {17, 31, 53, 79, 113, 151};
  const iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload = {expected.data(), roots[0], 256, sizeof(expected)},
  };
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution, 1, &upload, &completion));
  const loom_serve_snapshot_range_t contiguous = {0, 256, sizeof(expected)};
  const loom_serve_snapshot_range_t fragmented[] = {
      {1, 64, 4}, {1, 128, 12}, {1, 16, 8}};
  IREE_ASSERT_OK(loom_serve_snapshot_capture(
      roots.size(), SnapshotBuffers(execution, roots).data(), 1, &contiguous,
      iree_allocator_system(), &snapshot));
  IREE_ASSERT_OK(loom_serve_snapshot_restore(
      snapshot, roots.size(), SnapshotBuffers(execution, roots).data(),
      IREE_ARRAYSIZE(fragmented), fragmented));
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  loom_serve_snapshot_release(std::exchange(snapshot, nullptr));

  IREE_ASSERT_OK(loom_serve_snapshot_capture(
      roots.size(), SnapshotBuffers(execution, roots).data(),
      IREE_ARRAYSIZE(fragmented), fragmented, iree_allocator_system(),
      &snapshot));
  const loom_serve_snapshot_range_t restored = {0, 0, sizeof(expected)};
  IREE_ASSERT_OK(loom_serve_snapshot_restore(
      snapshot, roots.size(), SnapshotBuffers(execution, roots).data(), 1,
      &restored));
  std::array<uint32_t, 6> observed = {};
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download = {roots[0], 0, observed.data(), sizeof(observed)},
  };
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_wait(execution, completion));
  EXPECT_EQ(observed, expected);
}

TEST_F(SnapshotTest, DistinctOwnersPreserveLogicalOrderAndIndependentProgress) {
  IREE_ASSERT_OK(loom_serve_device_create_execution(device, storage_owner,
                                                    &peer_execution));
  ASSERT_NO_FATAL_FAILURE(AllocateRoots());
  loom_serve_execution_t* executions[] = {execution, peer_execution};
  loom_serve_snapshot_buffer_t buffers[] = {{roots[0], execution},
                                            {roots[1], peer_execution}};
  ASSERT_NO_FATAL_FAILURE(Hold(peer_execution));
  std::array<loom_serve_snapshot_range_t, kRangeCount> ranges;
  std::array<uint32_t, kRangeCount> expected;
  for (size_t owner = 0; owner < roots.size(); ++owner) {
    std::array<iree_hal_transfer_operation_t, kRangeCount / 2> fills;
    for (size_t j = 0; j < fills.size(); ++j) {
      const size_t i = j * 2 + owner;
      expected[i] = 0xABC00000u + static_cast<uint32_t>(i * 17);
      ranges[i] = {owner, (kRangeCount - i - 1) * 256, sizeof(uint32_t)};
      fills[j] = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
          .fill = {roots[owner], ranges[i].offset, sizeof(uint32_t),
                   &expected[i], sizeof(expected[i])},
      };
    }
    uint64_t completion = 0;
    IREE_ASSERT_OK(loom_serve_execution_transfer(
        executions[owner], fills.size(), fills.data(), &completion));
  }
  const auto before = loom_serve_execution_frontier(peer_execution);
  IREE_ASSERT_OK(loom_serve_snapshot_capture(
      IREE_ARRAYSIZE(buffers), buffers, ranges.size(), ranges.data(),
      iree_allocator_system(), &snapshot));
  EXPECT_EQ(loom_serve_snapshot_size(snapshot), sizeof(expected));
  const auto after = loom_serve_execution_frontier(peer_execution);
  ASSERT_GT(after.values[0], before.values[0]);
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  uint64_t completed = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(after.semaphores[0], &completed));
  EXPECT_LT(completed, after.values[0]);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  IREE_ASSERT_OK(
      loom_serve_execution_drain_all(IREE_ARRAYSIZE(executions), executions));

  // No source root or original placement survives into restoration. Swap the
  // execution owners too; the packed host payload has no physical identities.
  for (auto*& root : roots) {
    iree_hal_buffer_release(std::exchange(root, nullptr));
  }
  loom_serve_retirement_await(&retirement);
  for (auto*& reservation : reservations) {
    IREE_ASSERT_OK(loom_serve_virtual_buffer_free(reservation));
    reservation = nullptr;
  }
  ASSERT_NO_FATAL_FAILURE(AllocateRoots());
  for (size_t i = 0; i < IREE_ARRAYSIZE(buffers); ++i) {
    buffers[i] = {roots[i], executions[1 - i]};
  }
  for (size_t i = 0; i < ranges.size(); ++i) {
    ranges[i].buffer_index = 1 - ranges[i].buffer_index;
    ranges[i].offset = 1024 + i * 64;
  }
  const auto restore_before = loom_serve_execution_frontier(peer_execution);
  IREE_ASSERT_OK(loom_serve_snapshot_restore(snapshot, IREE_ARRAYSIZE(buffers),
                                             buffers, ranges.size(),
                                             ranges.data()));
  ASSERT_GT(loom_serve_execution_frontier(peer_execution).values[0],
            restore_before.values[0]);
  loom_serve_snapshot_release(std::exchange(snapshot, nullptr));
  std::array<uint32_t, kRangeCount> observed = {};
  for (size_t root = 0; root < roots.size(); ++root) {
    std::array<iree_hal_transfer_operation_t, kRangeCount / 2> downloads;
    for (size_t j = 0; j < downloads.size(); ++j) {
      const size_t i = j * 2 + 1 - root;
      downloads[j] = {
          .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
          .download = {roots[root], ranges[i].offset, &observed[i],
                       sizeof(observed[i])},
      };
    }
    uint64_t completion = 0;
    IREE_ASSERT_OK(
        loom_serve_execution_feedback(buffers[root].execution, downloads.size(),
                                      downloads.data(), &completion));
  }
  IREE_ASSERT_OK(
      loom_serve_execution_drain_all(IREE_ARRAYSIZE(executions), executions));
  EXPECT_EQ(observed, expected);
}

TEST_F(SnapshotTest, LaterOwnerFailureRetainsEarlierAcceptedPayload) {
  IREE_ASSERT_OK(loom_serve_device_create_execution(device, storage_owner,
                                                    &peer_execution));
  ASSERT_NO_FATAL_FAILURE(AllocateRoots());
  const uint32_t expected = 0x12345678;
  const iree_hal_transfer_operation_t fill = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
      .fill = {roots[0], 0, sizeof(expected), &expected, sizeof(expected)},
  };
  uint64_t completion = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(execution, 1, &fill, &completion));
  const loom_serve_snapshot_range_t capture[] = {{0, 0, sizeof(expected)},
                                                 {0, 0, sizeof(expected)}};
  IREE_ASSERT_OK(loom_serve_snapshot_capture(
      roots.size(), SnapshotBuffers(execution, roots).data(),
      IREE_ARRAYSIZE(capture), capture, iree_allocator_system(), &snapshot));
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  ASSERT_NO_FATAL_FAILURE(Hold(execution));
  const auto peer = loom_serve_execution_frontier(peer_execution);
  terminal_code = IREE_STATUS_CANCELLED;
  iree_hal_semaphore_fail(peer.semaphores[0],
                          iree_status_from_code(terminal_code));
  const loom_serve_snapshot_buffer_t buffers[] = {{roots[0], execution},
                                                  {roots[1], peer_execution}};
  const loom_serve_snapshot_range_t restore[] = {{0, 256, sizeof(expected)},
                                                 {1, 512, sizeof(expected)}};
  auto status =
      loom_serve_snapshot_restore(snapshot, IREE_ARRAYSIZE(buffers), buffers,
                                  IREE_ARRAYSIZE(restore), restore);
  // Drivers may observe failed prerequisites during enqueue or afterward.
  // Neither outcome releases the payload borrowed by the held earlier owner.
  if (!iree_status_is_ok(status)) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, status);
  }
  loom_serve_snapshot_release(std::exchange(snapshot, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate, 1, nullptr));
  uint32_t observed = 0;
  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download = {roots[0], 256, &observed, sizeof(observed)},
  };
  IREE_ASSERT_OK(
      loom_serve_execution_feedback(execution, 1, &download, &completion));
  IREE_ASSERT_OK(loom_serve_execution_feedback_wait(execution, completion));
  EXPECT_EQ(observed, expected);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        loom_serve_execution_drain(peer_execution));
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
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        pool, loom_serve_pool_state_geometry(storage_owner),
        kCacheOrigin + kPlaneStride * 32, 256, &statistics, &reservations[i]));
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
  if (std::memcmp(observed.data(), expected.data(), kPayloadBytes) != 0) {
    for (size_t i = 0; i < observed.size(); ++i) {
      ASSERT_EQ(observed[i], expected[i]) << "before capture, word " << i;
    }
  }
  IREE_ASSERT_OK(loom_serve_snapshot_capture(
      roots.size(), SnapshotBuffers(execution, roots).data(), ranges.size(),
      ranges.data(), iree_allocator_system(), &snapshot));
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
  IREE_ASSERT_OK(loom_serve_snapshot_restore(
      snapshot, roots.size(), SnapshotBuffers(execution, roots).data(),
      ranges.size(), ranges.data()));
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
    if (std::memcmp(observed.data() + offset, expected.data() + offset,
                    ranges[i].length) != 0) {
      for (size_t j = 0; j < ranges[i].length / sizeof(uint32_t); ++j) {
        ASSERT_EQ(observed[offset + j], expected[offset + j])
            << "range " << i << ", word " << j;
      }
    }
    offset += ranges[i].length / sizeof(uint32_t);
  }
}

// Submission must return the only worker before its predecessor can resolve.
// The caller relinquishes both its snapshot and tracked root references while
// native transfers are held. Failed feedback readiness also leaves actual work
// held until its completion callback schedules the independent unblock task.
struct loom_serve_test_snapshot_transfer_t {
  static iree_status_t Submit(iree_task_process_t* process,
                              const iree_task_worker_context_t* context,
                              iree_task_process_drain_result_t* result) {
    auto* self =
        static_cast<loom_serve_test_snapshot_transfer_t*>(process->user_data);
    auto local_ranges = self->ranges;
    auto status =
        self->snapshot
            ? loom_serve_snapshot_restore(
                  self->snapshot, self->roots.size(),
                  SnapshotBuffers(self->execution, self->roots).data(),
                  local_ranges.size(), local_ranges.data())
            : loom_serve_snapshot_capture(
                  self->roots.size(),
                  SnapshotBuffers(self->execution, self->roots).data(),
                  local_ranges.size(), local_ranges.data(),
                  iree_allocator_system(), &self->snapshot);
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
                               const iree_task_worker_context_t* context,
                               iree_task_process_drain_result_t* result) {
    auto* self =
        static_cast<loom_serve_test_snapshot_transfer_t*>(process->user_data);
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

  static void Ready(void* user_data, iree_async_operation_t* operation,
                    iree_status_t status, iree_async_completion_flags_t flags) {
    auto* self = static_cast<loom_serve_test_snapshot_transfer_t*>(user_data);
    std::lock_guard<std::mutex> lock(self->mutex);
    self->status = iree_status_join(self->status, status);
    self->ready = true;
    if (self->fail_readiness) {
      iree_task_executor_schedule_process(self->executor, &self->unblock);
    }
    self->condition.notify_one();
  }

  static void Release(iree_task_process_t* process) {
    auto* self =
        static_cast<loom_serve_test_snapshot_transfer_t*>(process->user_data);
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
  loom_serve_test_snapshot_transfer_t transfer;
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
    IREE_ASSERT_OK(loom_serve_virtual_buffer_allocate(
        pool, loom_serve_pool_state_geometry(storage_owner),
        kHighOffset + kSlabSize, 256, &statistics, &reservations[i]));
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
        transfer.roots.size(),
        SnapshotBuffers(execution, transfer.roots).data(),
        transfer.ranges.size(), transfer.ranges.data(), iree_allocator_system(),
        &transfer.snapshot));
  }
  IREE_ASSERT_OK(loom_serve_execution_drain(execution));
  IREE_ASSERT_OK(iree_hal_semaphore_create(
      loom_serve_device_handle(device), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0,
      IREE_HAL_SEMAPHORE_FLAG_DEFAULT, &transfer.gate));
  uint64_t held = 1;
  IREE_ASSERT_OK(
      loom_serve_execution_barrier(execution, {1, &transfer.gate, &held}));
  if (restore) {
    // Restore targets separately admitted physical ranges, not the original
    // input. A small mapping granularity must not make this test depend on
    // incidental backing for adjacent virtual addresses.
    for (auto* reservation : reservations) {
      IREE_ASSERT_OK(loom_serve_virtual_buffer_commit(
          reservation, kHighOffset + 32768, 32768));
    }
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
