// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/runtime/execution/workspace.h"

#include <array>
#include <memory>
#include <thread>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/execution/queues.h"

namespace {

TEST(WorkspaceLifetimeTest, NullLifetimeOperations) {
  loom_serve_workspace_retain(nullptr);
  loom_serve_workspace_release(nullptr);
}

class WorkspaceTest : public ::testing::TestWithParam<uint32_t> {
 protected:
  static constexpr iree_device_size_t kSlabBytes = 4096;

  struct loom_serve_test_workspace_domain_t {
    // Owned timeline; queues may be shared with another independent timeline.
    loom_serve_execution_t* execution = nullptr;
    // Caller-owned allocation view, released before joining consumers.
    iree_hal_buffer_t* buffer = nullptr;
    // Download destination kept alive through every accepted transfer.
    std::array<uint32_t, 16> output = {};
  };

  void SetUp() override {
    IREE_ASSERT_OK(loom_serve_host_create(GetParam(), nullptr,
                                          iree_allocator_system(), &host_));
    const loom_serve_device_options_t options[] = {
        {.driver = loom_serve_host_task_driver(host_), .uri = IREE_SV("task")},
        {.driver = loom_serve_host_task_driver(host_), .uri = IREE_SV("task")}};
    IREE_ASSERT_OK(loom_serve_device_group_allocate(
        host_, IREE_ARRAYSIZE(options), options, iree_allocator_system(),
        &group_));
    std::array<iree_hal_pool_family_access_t, 2> families = {};
    for (iree_host_size_t i = 0; i < families.size(); ++i) {
      auto* device = loom_serve_device_group_at(group_, i);
      families[i] = {
          .family =
              iree_hal_queue_family(loom_serve_device_dispatch_queue(device)),
          .usage =
              IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
      };
      ASSERT_EQ(
          families[i].family,
          iree_hal_queue_family(loom_serve_device_transfer_queue(device)));
      IREE_ASSERT_OK(loom_serve_execution_queues_create(
          families[i].family, families[i].family, iree_allocator_system(),
          &execution_queues_[i]));
    }
    const iree_hal_pool_scope_t scope = {families.size(), families.data()};
    const iree_hal_slab_pool_options_t pool_options = {};
    auto* hal_group =
        loom_serve_device_group(loom_serve_device_group_at(group_, 0));
    IREE_ASSERT_OK(iree_hal_slab_pool_create(hal_group, scope, &pool_options,
                                             iree_allocator_system(),
                                             &backing_pool_));

    // This test's CPU queues have coherent access. Establish that through the
    // captured contract; sharing a group alone never licenses eliding barriers.
    iree_hal_memory_scope_t sites[2];
    for (iree_host_size_t i = 0; i < families.size(); ++i) {
      IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
          hal_group, {IREE_HAL_MEMORY_SITE_QUEUE, families[i].family},
          &sites[i]));
    }
    iree_hal_memory_transition_pair_t pair;
    const auto table = iree_hal_pool_transition_table(backing_pool_);
    IREE_ASSERT_OK(iree_hal_memory_transition_prepare_pair(
        table, sites[0], sites[1], IREE_HAL_MEMORY_TRANSITION_ACQUIRE, &pair));
    const auto transition = iree_hal_memory_transition_query(table, pair);
    ASSERT_TRUE(iree_hal_memory_effects_is_empty(transition.release));
    ASSERT_TRUE(iree_hal_memory_effects_is_empty(transition.acquire));

    const iree_hal_asan_pool_options_t asan_options = {};
    IREE_ASSERT_OK(
        loom_serve_workspace_create(hal_group, backing_pool_, &asan_options,
                                    iree_allocator_system(), &workspace_));
    // Only the workspace keeps the supplied source alive from this point on.
    iree_hal_pool_release(backing_pool_);
    backing_pool_ = nullptr;
    IREE_ASSERT_OK(loom_serve_workspace_reserve(workspace_, kSlabBytes, 16));
    loom_serve_test_workspace_domain_t* domains[] = {&producer_, &consumer_,
                                                     &independent_};
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(domains); ++i) {
      IREE_ASSERT_OK(loom_serve_execution_create(
          execution_queues_[i % 2], workspace_, iree_allocator_system(),
          &domains[i]->execution));
    }
    IREE_ASSERT_OK(iree_hal_semaphore_create(
        loom_serve_device_handle(loom_serve_device_group_at(group_, 1)),
        IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, 0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT,
        &gate_));
  }

  void TearDown() override {
    if (gate_) {
      IREE_EXPECT_OK(iree_hal_semaphore_signal(gate_, 2, nullptr));
    }
    // An assertion may leave a caller view alive or miss queue_dealloca.
    // Release views before joining; accepted consumers still own their actual
    // borrows.
    for (auto* domain : {&producer_, &consumer_, &independent_}) {
      iree_hal_buffer_release(domain->buffer);
      domain->buffer = nullptr;
    }
    for (auto* domain : {&producer_, &consumer_, &independent_}) {
      if (domain->execution) {
        IREE_EXPECT_OK(loom_serve_execution_drain(domain->execution));
        loom_serve_execution_release(domain->execution);
      }
    }
    for (auto* queues : execution_queues_) {
      loom_serve_execution_queues_release(queues);
    }
    iree_hal_semaphore_release(gate_);
    loom_serve_workspace_release(workspace_);
    iree_hal_pool_release(backing_pool_);
    IREE_EXPECT_OK(loom_serve_device_group_free(group_));
    loom_serve_host_release(host_);
  }

  iree_status_t Allocate(loom_serve_test_workspace_domain_t& domain) {
    const auto request =
        iree_hal_make_pool_reservation_request(sizeof(domain.output), 16);
    return loom_serve_execution_alloca(domain.execution, &request,
                                       &domain.buffer);
  }

  iree_status_t DownloadAndRelease(loom_serve_test_workspace_domain_t& domain) {
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = domain.buffer,
                .target = domain.output.data(),
                .length = sizeof(domain.output),
            },
    };
    uint64_t value = 0;
    IREE_RETURN_IF_ERROR(
        loom_serve_execution_feedback(domain.execution, 1, &download, &value));
    auto frontier = loom_serve_execution_frontier(domain.execution);
    IREE_RETURN_IF_ERROR(loom_serve_execution_barrier(
        domain.execution, {1, &frontier.semaphores[1], &frontier.values[1]}));
    IREE_RETURN_IF_ERROR(
        loom_serve_execution_dealloca(domain.execution, domain.buffer, &value));
    iree_hal_buffer_release(domain.buffer);
    domain.buffer = nullptr;
    return iree_ok_status();
  }

  iree_status_t CompleteIndependentWork() {
    IREE_RETURN_IF_ERROR(Allocate(independent_));
    const uint32_t pattern = 0x13579BDF;
    const iree_hal_transfer_operation_t fill = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
        .fill =
            {
                .target_buffer = independent_.buffer,
                .length = sizeof(independent_.output),
                .pattern = &pattern,
                .pattern_length = sizeof(pattern),
            },
    };
    uint64_t value = 0;
    IREE_RETURN_IF_ERROR(loom_serve_execution_transfer(independent_.execution,
                                                       1, &fill, &value));
    IREE_RETURN_IF_ERROR(DownloadAndRelease(independent_));
    return loom_serve_execution_drain(independent_.execution);
  }

  void Trim() { loom_serve_workspace_trim(workspace_); }

  // Application workers shared by both native CPU devices.
  loom_serve_host_t* host_ = nullptr;
  // One topology/frontier namespace enclosing all pools and execution domains.
  loom_serve_device_group_t* group_ = nullptr;
  // Owned only until its reference transfers to the shared workspace.
  iree_hal_pool_t* backing_pool_ = nullptr;
  // Separate device inventories sharing the same physical workspace scope.
  std::array<loom_serve_execution_queues_t*, 2> execution_queues_ = {};
  // Shared geometry/accounting owner, independent of device records.
  loom_serve_workspace_t* workspace_ = nullptr;
  // First-device producer of an escaping allocation.
  loom_serve_test_workspace_domain_t producer_;
  // Second-device final consumer of the producer's allocation.
  loom_serve_test_workspace_domain_t consumer_;
  // Independent first-device work using the same backing and worker budget.
  loom_serve_test_workspace_domain_t independent_;
  // Test-controlled final reader readiness, opened before all teardown joins.
  iree_hal_semaphore_t* gate_ = nullptr;
  // Upload payload owned until all producer work has retired.
  std::array<uint32_t, 16> input_ = {};
};

TEST_P(WorkspaceTest, TrimReturnsUnusedSlabsAndPreservesLiveContents) {
  const auto request = iree_hal_make_pool_reservation_request(kSlabBytes, 16);
  for (auto* domain : {&producer_, &consumer_, &independent_}) {
    IREE_ASSERT_OK(loom_serve_execution_alloca(domain->execution, &request,
                                               &domain->buffer));
    const auto frontier = loom_serve_execution_frontier(domain->execution);
    IREE_ASSERT_OK(
        loom_serve_execution_wait(domain->execution, frontier.values[0]));
  }
  const uint32_t pattern = 0x13579BDF;
  const iree_hal_transfer_operation_t fill = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
      .fill =
          {
              .target_buffer = producer_.buffer,
              .length = sizeof(producer_.output),
              .pattern = &pattern,
              .pattern_length = sizeof(pattern),
          },
  };
  uint64_t value = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(producer_.execution, 1, &fill, &value));
  IREE_ASSERT_OK(loom_serve_execution_wait(producer_.execution, value));

  for (auto* domain : {&consumer_, &independent_}) {
    IREE_ASSERT_OK(loom_serve_execution_dealloca(domain->execution,
                                                 domain->buffer, &value));
    iree_hal_buffer_release(domain->buffer);
    domain->buffer = nullptr;
    IREE_ASSERT_OK(loom_serve_execution_drain(domain->execution));
  }

  // Logical returns propagate through TLSF and the shared cache immediately.
  // The one live slab remains available with its contents unchanged; detached
  // native returns are no longer part of workspace retention accounting.
  EXPECT_EQ(loom_serve_workspace_trim(workspace_), kSlabBytes);
  const auto retained = loom_serve_workspace_statistics(workspace_);
  EXPECT_EQ(retained.reservation_count, 1u);
  EXPECT_EQ(retained.bytes_committed, kSlabBytes);
  IREE_ASSERT_OK(DownloadAndRelease(producer_));
  IREE_ASSERT_OK(loom_serve_execution_drain(producer_.execution));
  for (uint32_t value : producer_.output) {
    EXPECT_EQ(value, pattern);
  }
  Trim();
  EXPECT_EQ(loom_serve_workspace_statistics(workspace_).bytes_committed, 0u);
}

TEST_P(WorkspaceTest,
       IdleExpiryWaitsForRetirementAndDoesNotRefreshOnObservation) {
  const auto timeout = iree_make_duration_ms(3600000);
  EXPECT_FALSE(
      loom_serve_workspace_trim_idle(workspace_, iree_time_now(), timeout));
  IREE_ASSERT_OK(Allocate(producer_));
  const auto allocated = loom_serve_execution_frontier(producer_.execution);
  IREE_ASSERT_OK(
      loom_serve_execution_wait(producer_.execution, allocated.values[0]));
  const auto distant = iree_time_now() + 2 * timeout;
  // A caller-owned view excludes expiry even after the model timeline retires.
  EXPECT_TRUE(loom_serve_workspace_trim_idle(workspace_, distant, timeout));
  EXPECT_EQ(loom_serve_workspace_statistics(workspace_).reservation_count, 1u);
  const iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload =
          {
              .source = input_.data(),
              .target_buffer = producer_.buffer,
              .length = sizeof(input_),
          },
  };
  uint64_t value = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(producer_.execution, 1, &upload, &value));
  IREE_ASSERT_OK(DownloadAndRelease(producer_));
  IREE_ASSERT_OK(loom_serve_execution_drain(producer_.execution));
  // Final raw queue ownership can outlive semaphore readiness. Collection is
  // the production retirement predicate, not a delay or a larger timeout.
  while (loom_serve_workspace_statistics(workspace_).reservation_count) {
    std::this_thread::yield();
  }
  const auto retired = iree_time_now();
  EXPECT_TRUE(loom_serve_workspace_trim_idle(workspace_, retired, timeout));
  EXPECT_TRUE(loom_serve_workspace_trim_idle(workspace_, retired + timeout / 2,
                                             timeout));
  EXPECT_GT(loom_serve_workspace_statistics(workspace_).bytes_committed, 0u);
  // Expiry returns logical ownership without waiting for physical freeing.
  EXPECT_FALSE(
      loom_serve_workspace_trim_idle(workspace_, retired + timeout, timeout));
  EXPECT_FALSE(loom_serve_workspace_trim_idle(workspace_, retired + 2 * timeout,
                                              timeout));
  Trim();
  EXPECT_EQ(loom_serve_workspace_statistics(workspace_).bytes_committed, 0u);
  IREE_ASSERT_OK(CompleteIndependentWork());
  for (uint32_t value : independent_.output) {
    EXPECT_EQ(value, 0x13579BDFu);
  }
}

TEST_P(WorkspaceTest, PersistentStateDoesNotKeepTransientWorkspaceActive) {
  const auto timeout = iree_make_duration_ms(3600000);
  auto* storage =
      loom_serve_device_storage_pool(loom_serve_device_group_at(group_, 0));
  const auto request =
      iree_hal_make_pool_reservation_request(sizeof(consumer_.output), 16);
  IREE_ASSERT_OK(loom_serve_execution_alloca_from_pool(
      producer_.execution, storage, 1, &request, &consumer_.buffer));
  const uint32_t pattern = 0x2468ACE0;
  iree_hal_transfer_operation_t fill = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
      .fill =
          {
              .target_buffer = consumer_.buffer,
              .length = sizeof(consumer_.output),
              .pattern = &pattern,
              .pattern_length = sizeof(pattern),
          },
  };
  uint64_t value = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(producer_.execution, 1, &fill, &value));
  IREE_ASSERT_OK(loom_serve_execution_wait(producer_.execution, value));
  // Storage is live, but no command workspace has been acquired yet.
  EXPECT_FALSE(
      loom_serve_workspace_trim_idle(workspace_, iree_time_now(), timeout));

  // Repeated transient allocations on the same execution must retire behind
  // independently retained model state without keeping their slabs hot.
  for (int iteration = 0; iteration < 4; ++iteration) {
    IREE_ASSERT_OK(Allocate(producer_));
    fill.fill.target_buffer = producer_.buffer;
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(producer_.execution, 1, &fill, &value));
    IREE_ASSERT_OK(DownloadAndRelease(producer_));
    IREE_ASSERT_OK(loom_serve_execution_drain(producer_.execution));
    for (uint32_t actual : producer_.output) {
      EXPECT_EQ(actual, pattern);
    }
  }
  // Queue root retirement can trail semaphore readiness. Advance the idle
  // policy until it observes that actual boundary; the outer harness catches
  // a lost retirement, without a timing assumption in this witness.
  while (loom_serve_workspace_trim_idle(
      workspace_, iree_time_now() + 2 * timeout, timeout)) {
    std::this_thread::yield();
  }
  EXPECT_EQ(loom_serve_workspace_statistics(workspace_).bytes_committed, 0u);

  const iree_hal_transfer_operation_t download = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = consumer_.buffer,
              .target = consumer_.output.data(),
              .length = sizeof(consumer_.output),
          },
  };
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(producer_.execution, 1, &download, &value));
  IREE_ASSERT_OK(loom_serve_execution_wait(producer_.execution, value));
  for (uint32_t actual : consumer_.output) {
    EXPECT_EQ(actual, pattern);
  }
  IREE_ASSERT_OK(loom_serve_execution_dealloca(producer_.execution,
                                               consumer_.buffer, &value));
  iree_hal_buffer_release(consumer_.buffer);
  consumer_.buffer = nullptr;
  IREE_ASSERT_OK(loom_serve_execution_drain(producer_.execution));
  EXPECT_FALSE(
      loom_serve_workspace_trim_idle(workspace_, iree_time_now(), timeout));
}

TEST_P(WorkspaceTest,
       SharedEscapeRetiresAfterLastReaderWithoutBlockingOtherWork) {
  // Establish both reservations in one slab before holding the last reader.
  // Submission alone can leave native backing acquisition pending.
  for (auto* domain : {&producer_, &consumer_}) {
    IREE_ASSERT_OK(Allocate(*domain));
    const auto frontier = loom_serve_execution_frontier(domain->execution);
    IREE_ASSERT_OK(
        loom_serve_execution_wait(domain->execution, frontier.values[0]));
  }
  for (size_t i = 0; i < input_.size(); ++i) {
    input_[i] = 17 + i * i;
  }
  const iree_hal_transfer_operation_t upload = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .upload =
          {
              .source = input_.data(),
              .target_buffer = producer_.buffer,
              .length = sizeof(input_),
          },
  };
  uint64_t value = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(producer_.execution, 1, &upload, &value));
  const auto produced = loom_serve_execution_frontier(producer_.execution);
  iree_hal_semaphore_t* waits[] = {produced.semaphores[0], gate_};
  uint64_t wait_values[] = {produced.values[0], 1};
  IREE_ASSERT_OK(loom_serve_execution_barrier(consumer_.execution,
                                              {2, waits, wait_values}));
  const iree_hal_transfer_operation_t copy = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
      .copy =
          {
              .source_buffer = producer_.buffer,
              .target_buffer = consumer_.buffer,
              .length = sizeof(input_),
          },
  };
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(consumer_.execution, 1, &copy, &value));
  auto consumed = loom_serve_execution_frontier(consumer_.execution);
  // The allocation owner explicitly joins the other domain's last read.
  // No queue order, shared pool, or device-group membership supplies this edge.
  IREE_ASSERT_OK(loom_serve_execution_barrier(
      producer_.execution, {1, &consumed.semaphores[0], &consumed.values[0]}));
  IREE_ASSERT_OK(loom_serve_execution_dealloca(producer_.execution,
                                               producer_.buffer, &value));
  iree_hal_buffer_release(producer_.buffer);
  producer_.buffer = nullptr;
  IREE_ASSERT_OK(DownloadAndRelease(consumer_));

  IREE_ASSERT_OK(CompleteIndependentWork());
  for (uint32_t actual : independent_.output) {
    EXPECT_EQ(actual, 0x13579BDFu);
  }
  uint64_t observed = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(consumed.semaphores[0], &observed));
  EXPECT_LT(observed, consumed.values[0]);
  EXPECT_FALSE(loom_serve_execution_is_retired(producer_.execution));
  Trim();
  const auto live = loom_serve_workspace_statistics(workspace_);
  EXPECT_EQ(live.reservation_count, 2u);
  EXPECT_EQ(live.slab_count, 1u);
  EXPECT_EQ(live.bytes_committed, kSlabBytes);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  IREE_ASSERT_OK(loom_serve_execution_drain(consumer_.execution));
  IREE_ASSERT_OK(loom_serve_execution_drain(producer_.execution));
  EXPECT_EQ(consumer_.output, input_);
  EXPECT_TRUE(loom_serve_execution_is_retired(producer_.execution));
  Trim();
  const auto retired = loom_serve_workspace_statistics(workspace_);
  EXPECT_EQ(retired.reservation_count, 0u);
  EXPECT_EQ(retired.bytes_committed, 0u);
  EXPECT_EQ(retired.reserve_count, 3u);
  EXPECT_EQ(retired.release_count, 3u);
}

TEST_P(WorkspaceTest, ReleasedExecutionPreservesPendingRawAllocationOwnership) {
  uint64_t gate_value = 1;
  IREE_ASSERT_OK(loom_serve_execution_barrier(producer_.execution,
                                              {1, &gate_, &gate_value}));
  IREE_ASSERT_OK(Allocate(producer_));
  uint64_t value = 0;
  IREE_ASSERT_OK(loom_serve_execution_dealloca(producer_.execution,
                                               producer_.buffer, &value));
  iree_hal_buffer_release(producer_.buffer);
  producer_.buffer = nullptr;
  EXPECT_TRUE(loom_serve_execution_is_retired(producer_.execution));
  auto pending = loom_serve_execution_frontier(producer_.execution);
  IREE_ASSERT_OK(loom_serve_execution_barrier(
      consumer_.execution, {1, &pending.semaphores[0], &pending.values[0]}));
  loom_serve_execution_release(producer_.execution);
  producer_.execution = nullptr;

  // Accepted raw allocation records survive their model. Releasing that model
  // must not join the held queue or reclaim backing still borrowed by HAL.
  IREE_ASSERT_OK(CompleteIndependentWork());
  for (uint32_t actual : independent_.output) {
    EXPECT_EQ(actual, 0x13579BDFu);
  }
  auto joined = loom_serve_execution_frontier(consumer_.execution);
  uint64_t observed = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(joined.semaphores[0], &observed));
  EXPECT_LT(observed, joined.values[0]);
  Trim();
  // The CPU alloca waits before reserving physical storage. Its pending buffer
  // still borrows the pool even though only independent work has allocated.
  const auto held = loom_serve_workspace_statistics(workspace_);
  EXPECT_EQ(held.reservation_count, 0u);
  EXPECT_EQ(held.reserve_count, 1u);
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_, 1, nullptr));
  IREE_ASSERT_OK(loom_serve_execution_drain(consumer_.execution));
  Trim();
  const auto retired = loom_serve_workspace_statistics(workspace_);
  EXPECT_EQ(retired.reservation_count, 0u);
  EXPECT_EQ(retired.bytes_committed, 0u);
  EXPECT_EQ(retired.reserve_count, 2u);
  EXPECT_EQ(retired.release_count, 2u);
}

TEST_P(WorkspaceTest, LiveEscapeDoesNotRetainRetiredGeneration) {
  IREE_ASSERT_OK(Allocate(producer_));
  const uint32_t pattern = 0x2468ACE0;
  const iree_hal_transfer_operation_t fill = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
      .fill =
          {
              .target_buffer = producer_.buffer,
              .length = sizeof(producer_.output),
              .pattern = &pattern,
              .pattern_length = sizeof(pattern),
          },
  };
  uint64_t value = 0;
  IREE_ASSERT_OK(
      loom_serve_execution_transfer(producer_.execution, 1, &fill, &value));

  // A model's escaping input/output is allocated before its private command
  // transients. Geometry may grow while that escape remains live.
  IREE_ASSERT_OK(loom_serve_workspace_reserve(workspace_, 2 * kSlabBytes, 16));
  const auto request = iree_hal_make_pool_reservation_request(64, 16);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      loom_serve_execution_alloca(producer_.execution, &request, &buffer));
  std::unique_ptr<iree_hal_buffer_t, decltype(&iree_hal_buffer_release)>
      transient(buffer, &iree_hal_buffer_release);
  IREE_ASSERT_OK(loom_serve_execution_dealloca(producer_.execution,
                                               transient.get(), &value));
  transient.reset();
  IREE_ASSERT_OK(loom_serve_execution_wait(producer_.execution, value));

  // Moving to a third geometry makes the middle generation entirely idle.
  // Its raw captures must retire independently of the earlier live escape.
  IREE_ASSERT_OK(loom_serve_workspace_reserve(workspace_, 4 * kSlabBytes, 16));
  IREE_ASSERT_OK(CompleteIndependentWork());
  const auto live = loom_serve_workspace_statistics(workspace_);
  EXPECT_EQ(live.reservation_count, 1u);
  EXPECT_EQ(live.slab_count, 2u);
  EXPECT_EQ(live.bytes_committed, 5 * kSlabBytes);

  IREE_ASSERT_OK(DownloadAndRelease(producer_));
  IREE_ASSERT_OK(loom_serve_execution_drain(producer_.execution));
  for (uint32_t actual : producer_.output) {
    EXPECT_EQ(actual, pattern);
  }
  Trim();
  const auto retired = loom_serve_workspace_statistics(workspace_);
  EXPECT_EQ(retired.reservation_count, 0u);
  EXPECT_EQ(retired.bytes_committed, 0u);
  EXPECT_EQ(retired.reserve_count, 3u);
  EXPECT_EQ(retired.release_count, 3u);
}

TEST_P(WorkspaceTest, CaptureCompactionPreservesLiveRootsAcrossRingWrap) {
  // Retire a prefix so the following full-ring compaction wraps physically.
  IREE_ASSERT_OK(Allocate(producer_));
  uint64_t value = 0;
  IREE_ASSERT_OK(loom_serve_execution_dealloca(producer_.execution,
                                               producer_.buffer, &value));
  iree_hal_buffer_release(producer_.buffer);
  producer_.buffer = nullptr;
  IREE_ASSERT_OK(loom_serve_execution_wait(producer_.execution, value));
  IREE_ASSERT_OK(Allocate(producer_));

  // Interleave long-lived values with retired command-private allocations on
  // the same execution. The oldest live root cannot hide the completed ones.
  using Buffer =
      std::unique_ptr<iree_hal_buffer_t, decltype(&iree_hal_buffer_release)>;
  std::vector<Buffer> buffers;
  const auto request = iree_hal_make_pool_reservation_request(64, 16);
  for (uint32_t i = 0; i < 32; ++i) {
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(
        loom_serve_execution_alloca(producer_.execution, &request, &buffer));
    buffers.emplace_back(buffer, &iree_hal_buffer_release);
    const uint32_t pattern = i + 1;
    const iree_hal_transfer_operation_t fill = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_FILL,
        .fill =
            {
                .target_buffer = buffer,
                .length = 64,
                .pattern = &pattern,
                .pattern_length = sizeof(pattern),
            },
    };
    IREE_ASSERT_OK(
        loom_serve_execution_transfer(producer_.execution, 1, &fill, &value));
    if (!(i % 2)) {
      IREE_ASSERT_OK(
          loom_serve_execution_dealloca(producer_.execution, buffer, &value));
      buffers.back().reset();
    }
    IREE_ASSERT_OK(loom_serve_execution_wait(producer_.execution, value));
  }
  EXPECT_EQ(loom_serve_workspace_statistics(workspace_).reservation_count, 17u);
  for (size_t i = 1; i < buffers.size(); i += 2) {
    const iree_hal_transfer_operation_t download = {
        .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
        .download =
            {
                .source_buffer = buffers[i].get(),
                .target = &consumer_.output[i / 2],
                .length = sizeof(uint32_t),
            },
    };
    IREE_ASSERT_OK(loom_serve_execution_transfer(producer_.execution, 1,
                                                 &download, &value));
    IREE_ASSERT_OK(loom_serve_execution_dealloca(producer_.execution,
                                                 buffers[i].get(), &value));
    buffers[i].reset();
  }
  IREE_ASSERT_OK(loom_serve_execution_dealloca(producer_.execution,
                                               producer_.buffer, &value));
  iree_hal_buffer_release(producer_.buffer);
  producer_.buffer = nullptr;
  IREE_ASSERT_OK(loom_serve_execution_drain(producer_.execution));
  for (size_t i = 0; i < consumer_.output.size(); ++i) {
    EXPECT_EQ(consumer_.output[i], 2 * (i + 1));
  }
  Trim();
  const auto retired = loom_serve_workspace_statistics(workspace_);
  EXPECT_EQ(retired.reservation_count, 0u);
  EXPECT_EQ(retired.bytes_committed, 0u);
  EXPECT_EQ(retired.reserve_count, 34u);
  EXPECT_EQ(retired.release_count, 34u);
}

INSTANTIATE_TEST_SUITE_P(Workers, WorkspaceTest, ::testing::Values(1, 4));

}  // namespace
