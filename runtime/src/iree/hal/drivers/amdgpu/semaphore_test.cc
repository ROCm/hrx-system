// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/semaphore.h"

#include <string.h>

#include <atomic>
#include <thread>
#include <vector>

#include "iree/async/frontier.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/drivers/amdgpu/host_queue_policy.h"
#include "iree/hal/drivers/amdgpu/host_queue_waits.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_async_proactor_t* test_proactor() {
  static iree_async_proactor_t* proactor = nullptr;
  if (!proactor) {
    IREE_CHECK_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor));
    atexit([] {
      iree_async_proactor_release(proactor);
      proactor = nullptr;
    });
  }
  return proactor;
}

static iree_async_axis_t test_queue_axis(uint8_t queue_index) {
  return iree_async_axis_make_queue(/*session_epoch=*/1, /*machine_index=*/0,
                                    /*device_index=*/0, queue_index,
                                    /*queue_incarnation=*/0);
}

TEST(LastSignalTest, ConcurrentPublicationReturnsOneCompleteGeneration) {
  // Keep every published snapshot valid so returning the caller's initialized
  // outputs while a writer is active is distinguishable from an empty cache.
  constexpr uint64_t kGenerationCount = 65536;
  iree_hal_amdgpu_last_signal_t cache = {};
  auto publish = [&](uint64_t generation) {
    iree_hal_amdgpu_last_signal_store(
        &cache, IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_VALID,
        test_queue_axis(static_cast<uint8_t>(generation % 16)), generation,
        generation * 17 + 5);
  };
  publish(1);

  std::atomic<bool> reader_ready{false};
  std::atomic<bool> writer_ready{false};
  std::thread writer([&] {
    writer_ready.store(true, std::memory_order_release);
    while (!reader_ready.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (uint64_t generation = 2; generation <= kGenerationCount;
         ++generation) {
      publish(generation);
    }
  });
  reader_ready.store(true, std::memory_order_release);
  while (!writer_ready.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  uint64_t invalid_snapshots = 0;
  uint64_t last_epoch = 0;
  for (uint64_t i = 0; i < kGenerationCount; ++i) {
    iree_hal_amdgpu_last_signal_flags_t flags =
        IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_NONE;
    iree_async_axis_t axis = 0;
    uint64_t epoch = 0;
    uint64_t value = 0;
    bool valid =
        iree_hal_amdgpu_last_signal_load(&cache, &flags, &axis, &epoch, &value);
    if (!valid || epoch == 0 || epoch > kGenerationCount ||
        epoch < last_epoch ||
        axis != test_queue_axis(static_cast<uint8_t>(epoch % 16)) ||
        value != epoch * 17 + 5) {
      ++invalid_snapshots;
    }
    last_epoch = epoch;
  }
  writer.join();
  EXPECT_EQ(invalid_snapshots, 0u);
}

class FrontierBuilder {
 public:
  iree_async_frontier_t* Build(
      std::initializer_list<iree_async_frontier_entry_t> entries) {
    storage_.resize(sizeof(iree_async_frontier_t) +
                    entries.size() * sizeof(iree_async_frontier_entry_t));
    auto* frontier = reinterpret_cast<iree_async_frontier_t*>(storage_.data());
    iree_async_frontier_initialize(frontier,
                                   static_cast<uint8_t>(entries.size()));
    iree_host_size_t i = 0;
    for (const auto& entry : entries) {
      frontier->entries[i++] = entry;
    }
    return frontier;
  }

 private:
  std::vector<uint8_t> storage_;
};

class SemaphoreTest : public ::testing::Test {
 protected:
  void SetUp() override {
    static uintptr_t fake_device_storage = 0;
    fake_device_ = reinterpret_cast<iree_hal_amdgpu_logical_device_t*>(
        &fake_device_storage);
    IREE_ASSERT_OK(CreateSemaphore(IREE_HAL_SEMAPHORE_FLAG_NONE, &semaphore_));
  }

  void TearDown() override { iree_hal_semaphore_release(semaphore_); }

  iree_status_t CreateSemaphore(iree_hal_semaphore_flags_t flags,
                                iree_hal_semaphore_t** out_semaphore) {
    return iree_hal_amdgpu_semaphore_create(
        fake_device_, test_proactor(), IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        /*initial_value=*/0, flags, iree_allocator_system(), out_semaphore);
  }

  iree_hal_amdgpu_wait_resolution_t ResolveWait(
      uint8_t queue_index, uint64_t value,
      const iree_async_frontier_t* frontier = nullptr) {
    alignas(iree_hal_amdgpu_epoch_signal_table_t)
        uint8_t table_storage[sizeof(iree_hal_amdgpu_epoch_signal_table_t) +
                              3 * sizeof(hsa_signal_t)];
    auto* table =
        reinterpret_cast<iree_hal_amdgpu_epoch_signal_table_t*>(table_storage);
    iree_hal_amdgpu_epoch_signal_table_initialize(table, 1, 0, 0, 3);
    // Only wait resolution runs: these handles identify native operands but
    // are never passed to ROCr or submitted to hardware.
    for (uint8_t i = 0; i < 3; ++i) {
      iree_hal_amdgpu_epoch_signal_table_register(
          table, i, hsa_signal_t{uint64_t(i + 1)});
    }
    iree_hal_amdgpu_host_queue_t queue = {
        .logical_device = reinterpret_cast<iree_hal_device_t*>(fake_device_),
        .axis = test_queue_axis(queue_index),
        .epoch_table = table,
    };
    queue.wait_barrier_strategy =
        IREE_HAL_AMDGPU_WAIT_BARRIER_STRATEGY_AQL_BARRIER_VALUE;
    iree_slim_mutex_initialize(&queue.locks.submission_mutex);
    auto* queue_frontier = iree_hal_amdgpu_host_queue_frontier(&queue);
    iree_async_frontier_initialize(queue_frontier, 0);
    if (frontier) {
      EXPECT_TRUE(iree_async_frontier_merge(
          queue_frontier, IREE_HAL_AMDGPU_QUEUE_FRONTIER_CAPACITY, frontier));
    }
    iree_hal_amdgpu_wait_resolution_t resolution = {};
    iree_slim_mutex_lock(&queue.locks.submission_mutex);
    iree_hal_amdgpu_host_queue_resolve_waits(
        &queue, iree_hal_semaphore_list_t{1, &semaphore_, &value}, &resolution);
    iree_slim_mutex_unlock(&queue.locks.submission_mutex);
    iree_slim_mutex_deinitialize(&queue.locks.submission_mutex);
    return resolution;
  }

  iree_hal_amdgpu_logical_device_t* fake_device_ = nullptr;
  iree_hal_semaphore_t* semaphore_ = nullptr;
};

TEST_F(SemaphoreTest, PrivateStreamSemanticsRequireStrictFlags) {
  iree_hal_semaphore_t* private_semaphore = nullptr;
  IREE_ASSERT_OK(CreateSemaphore(IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL |
                                     IREE_HAL_SEMAPHORE_FLAG_SINGLE_PRODUCER,
                                 &private_semaphore));
  EXPECT_TRUE(iree_hal_amdgpu_semaphore_has_private_stream_semantics(
      private_semaphore, fake_device_));
  iree_hal_semaphore_release(private_semaphore);

  iree_hal_semaphore_t* public_local_semaphore = nullptr;
  IREE_ASSERT_OK(CreateSemaphore(IREE_HAL_SEMAPHORE_FLAG_DEFAULT |
                                     IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL |
                                     IREE_HAL_SEMAPHORE_FLAG_SINGLE_PRODUCER,
                                 &public_local_semaphore));
  EXPECT_FALSE(iree_hal_amdgpu_semaphore_has_private_stream_semantics(
      public_local_semaphore, fake_device_));
  iree_hal_semaphore_release(public_local_semaphore);

  iree_hal_semaphore_t* multi_producer_semaphore = nullptr;
  IREE_ASSERT_OK(CreateSemaphore(IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL,
                                 &multi_producer_semaphore));
  EXPECT_FALSE(iree_hal_amdgpu_semaphore_has_private_stream_semantics(
      multi_producer_semaphore, fake_device_));
  iree_hal_semaphore_release(multi_producer_semaphore);
}

TEST_F(SemaphoreTest, QueuePolicyUsesAgentScopeOnlyForSamePhysicalDevice) {
  iree_hal_amdgpu_logical_device_t logical_device;
  memset(&logical_device, 0, sizeof(logical_device));
  logical_device.physical_device_count = 2;

  iree_hal_amdgpu_host_queue_t queue;
  memset(&queue, 0, sizeof(queue));
  queue.logical_device = (iree_hal_device_t*)&logical_device;
  queue.device_ordinal = 0;

  iree_hal_semaphore_t* same_agent_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_create(
      &logical_device, test_proactor(),
      /*queue_family_affinity=*/iree_hal_make_queue_family_affinity(0),
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL,
      iree_allocator_system(), &same_agent_semaphore));
  EXPECT_EQ(iree_hal_amdgpu_host_queue_wait_acquire_scope(&queue,
                                                          same_agent_semaphore),
            IREE_HSA_FENCE_SCOPE_AGENT);
  EXPECT_EQ(iree_hal_amdgpu_host_queue_signal_release_scope(
                &queue, same_agent_semaphore),
            IREE_HSA_FENCE_SCOPE_AGENT);
  iree_hal_semaphore_release(same_agent_semaphore);

  iree_hal_semaphore_t* cross_agent_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_create(
      &logical_device, test_proactor(),
      /*queue_family_affinity=*/iree_hal_make_queue_family_affinity(1),
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL,
      iree_allocator_system(), &cross_agent_semaphore));
  EXPECT_EQ(iree_hal_amdgpu_host_queue_wait_acquire_scope(
                &queue, cross_agent_semaphore),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(iree_hal_amdgpu_host_queue_signal_release_scope(
                &queue, cross_agent_semaphore),
            IREE_HSA_FENCE_SCOPE_SYSTEM);
  iree_hal_semaphore_release(cross_agent_semaphore);

  iree_hal_semaphore_t* public_semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_semaphore_create(
      &logical_device, test_proactor(),
      /*queue_family_affinity=*/iree_hal_make_queue_family_affinity(0),
      /*initial_value=*/0,
      IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL |
          IREE_HAL_SEMAPHORE_FLAG_HOST_INTERRUPT,
      iree_allocator_system(), &public_semaphore));
  EXPECT_EQ(
      iree_hal_amdgpu_host_queue_wait_acquire_scope(&queue, public_semaphore),
      IREE_HSA_FENCE_SCOPE_SYSTEM);
  EXPECT_EQ(
      iree_hal_amdgpu_host_queue_signal_release_scope(&queue, public_semaphore),
      IREE_HSA_FENCE_SCOPE_SYSTEM);
  iree_hal_semaphore_release(public_semaphore);
}

TEST_F(SemaphoreTest, PrivateStreamSignalPublishesExactProducerEpoch) {
  iree_hal_semaphore_t* private_semaphore = nullptr;
  IREE_ASSERT_OK(CreateSemaphore(IREE_HAL_SEMAPHORE_FLAG_DEVICE_LOCAL |
                                     IREE_HAL_SEMAPHORE_FLAG_SINGLE_PRODUCER,
                                 &private_semaphore));

  const iree_async_axis_t producer_axis = test_queue_axis(2);
  iree_hal_amdgpu_semaphore_publish_private_stream_signal(
      private_semaphore, producer_axis, /*producer_epoch=*/7,
      /*producer_value=*/3);

  iree_hal_amdgpu_last_signal_flags_t flags =
      IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_NONE;
  iree_async_axis_t cached_axis = 0;
  uint64_t cached_epoch = 0;
  uint64_t cached_value = 0;
  EXPECT_TRUE(iree_hal_amdgpu_last_signal_load(
      iree_hal_amdgpu_semaphore_last_signal(private_semaphore), &flags,
      &cached_axis, &cached_epoch, &cached_value));
  EXPECT_EQ(cached_axis, producer_axis);
  EXPECT_EQ(cached_epoch, 7u);
  EXPECT_EQ(cached_value, 3u);
  EXPECT_EQ(flags,
            IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_VALID |
                IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT);

  iree_hal_semaphore_release(private_semaphore);
}

TEST_F(SemaphoreTest,
       PublishSignalMarksExactWhenProducerFrontierCoversTransitiveDeps) {
  const iree_async_axis_t producer_axis = test_queue_axis(2);
  const iree_async_axis_t peer_axis = test_queue_axis(1);

  FrontierBuilder frontier_builder;
  iree_async_frontier_t* initial_frontier =
      frontier_builder.Build({iree_async_frontier_entry_t{peer_axis, 4}});
  EXPECT_TRUE(iree_hal_amdgpu_semaphore_publish_signal(
      semaphore_, peer_axis, initial_frontier, /*producer_epoch=*/4,
      /*producer_value=*/1));

  iree_async_frontier_t* transitive_frontier =
      frontier_builder.Build({iree_async_frontier_entry_t{peer_axis, 4},
                              iree_async_frontier_entry_t{producer_axis, 7}});
  EXPECT_TRUE(iree_hal_amdgpu_semaphore_publish_signal(
      semaphore_, producer_axis, transitive_frontier, /*producer_epoch=*/7,
      /*producer_value=*/2));

  iree_hal_amdgpu_last_signal_flags_t flags =
      IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_NONE;
  iree_async_axis_t cached_axis = 0;
  uint64_t cached_epoch = 0;
  uint64_t cached_value = 0;
  EXPECT_TRUE(iree_hal_amdgpu_last_signal_load(
      iree_hal_amdgpu_semaphore_last_signal(semaphore_), &flags, &cached_axis,
      &cached_epoch, &cached_value));
  EXPECT_EQ(cached_axis, producer_axis);
  EXPECT_EQ(cached_epoch, 7u);
  EXPECT_EQ(cached_value, 2u);
  EXPECT_EQ(flags,
            IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_VALID |
                IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_PRODUCER_FRONTIER_EXACT);
}

TEST_F(SemaphoreTest, PublishSignalClearsExactForIndependentFanIn) {
  const iree_async_axis_t first_axis = test_queue_axis(1);
  const iree_async_axis_t second_axis = test_queue_axis(2);

  FrontierBuilder frontier_builder;
  iree_async_frontier_t* first_frontier =
      frontier_builder.Build({iree_async_frontier_entry_t{first_axis, 5}});
  EXPECT_TRUE(iree_hal_amdgpu_semaphore_publish_signal(
      semaphore_, first_axis, first_frontier, /*producer_epoch=*/5,
      /*producer_value=*/1));

  iree_async_frontier_t* second_frontier =
      frontier_builder.Build({iree_async_frontier_entry_t{second_axis, 9}});
  EXPECT_TRUE(iree_hal_amdgpu_semaphore_publish_signal(
      semaphore_, second_axis, second_frontier, /*producer_epoch=*/9,
      /*producer_value=*/2));

  iree_hal_amdgpu_last_signal_flags_t flags =
      IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_NONE;
  iree_async_axis_t cached_axis = 0;
  uint64_t cached_epoch = 0;
  uint64_t cached_value = 0;
  EXPECT_TRUE(iree_hal_amdgpu_last_signal_load(
      iree_hal_amdgpu_semaphore_last_signal(semaphore_), &flags, &cached_axis,
      &cached_epoch, &cached_value));
  EXPECT_EQ(cached_axis, second_axis);
  EXPECT_EQ(cached_epoch, 9u);
  EXPECT_EQ(cached_value, 2u);
  EXPECT_EQ(flags, IREE_HAL_AMDGPU_LAST_SIGNAL_FLAG_VALID);
}

TEST_F(SemaphoreTest, OlderWaitDoesNotUseLaterProducerEpoch) {
  FrontierBuilder builder;
  const auto producer_axis = test_queue_axis(1);
  ASSERT_TRUE(iree_hal_amdgpu_semaphore_publish_signal(
      semaphore_, producer_axis, builder.Build({{producer_axis, 5}}), 5, 1));
  ASSERT_TRUE(iree_hal_amdgpu_semaphore_publish_signal(
      semaphore_, producer_axis, builder.Build({{producer_axis, 9}}), 9, 2));

  // A consumer of S=1 can itself be a prerequisite of S=2. Waiting on the
  // latter's epoch invents a cycle, both within and across native queues.
  for (uint8_t queue_index = 0; queue_index < 2; ++queue_index) {
    auto resolution = ResolveWait(queue_index, 1);
    EXPECT_TRUE(resolution.needs_deferral);
    EXPECT_EQ(resolution.barrier_count, 0);
  }

  // Completion of the actual requested value releases the software path even
  // while the cached later producer has not completed.
  IREE_ASSERT_OK(iree_hal_semaphore_signal(semaphore_, 1, nullptr));
  auto resolution = ResolveWait(0, 1);
  EXPECT_FALSE(resolution.needs_deferral);
  EXPECT_EQ(resolution.barrier_count, 0);
}

TEST_F(SemaphoreTest, ExactWaitAndCoveredLaterProofRemainNative) {
  FrontierBuilder builder;
  const auto producer_axis = test_queue_axis(1);
  const auto* frontier = builder.Build({{producer_axis, 9}});
  ASSERT_TRUE(iree_hal_amdgpu_semaphore_publish_signal(
      semaphore_, producer_axis, frontier, 9, 2));

  auto exact = ResolveWait(0, 2);
  ASSERT_FALSE(exact.needs_deferral);
  ASSERT_EQ(exact.barrier_count, 1);
  EXPECT_EQ(exact.barriers[0].axis, producer_axis);
  EXPECT_EQ(exact.barriers[0].target_epoch, 9u);

  // A later proof already covered by the consumer is sufficient for elision.
  // It is never substituted as a new blocking dependency for the older wait.
  auto covered = ResolveWait(0, 1, frontier);
  EXPECT_FALSE(covered.needs_deferral);
  EXPECT_EQ(covered.barrier_count, 0);
  auto future = ResolveWait(0, 3, frontier);
  EXPECT_TRUE(future.needs_deferral);
  EXPECT_EQ(future.barrier_count, 0);
}

TEST_F(SemaphoreTest, FanInWaitUsesOnlyItsExactSignalFrontier) {
  FrontierBuilder builder;
  const auto first_axis = test_queue_axis(1);
  const auto second_axis = test_queue_axis(2);
  ASSERT_TRUE(iree_hal_amdgpu_semaphore_publish_signal(
      semaphore_, first_axis, builder.Build({{first_axis, 4}}), 4, 1));
  ASSERT_TRUE(iree_hal_amdgpu_semaphore_publish_signal(
      semaphore_, second_axis, builder.Build({{second_axis, 9}}), 9, 2));

  auto exact = ResolveWait(0, 2);
  ASSERT_FALSE(exact.needs_deferral);
  ASSERT_EQ(exact.barrier_count, 2);
  EXPECT_EQ(exact.barriers[0].axis, first_axis);
  EXPECT_EQ(exact.barriers[0].target_epoch, 4u);
  EXPECT_EQ(exact.barriers[1].axis, second_axis);
  EXPECT_EQ(exact.barriers[1].target_epoch, 9u);

  auto older = ResolveWait(0, 1);
  EXPECT_TRUE(older.needs_deferral);
  EXPECT_EQ(older.barrier_count, 0);
}

TEST_F(SemaphoreTest, ConcurrentPublicationDoesNotUpgradeWaitFrontier) {
  constexpr uint64_t kPublicationCount = 8192;
  FrontierBuilder builder;
  const auto first_axis = test_queue_axis(1);
  const auto second_axis = test_queue_axis(2);
  ASSERT_TRUE(iree_hal_amdgpu_semaphore_publish_signal(
      semaphore_, first_axis, builder.Build({{first_axis, 4}}), 4, 1));
  ASSERT_TRUE(iree_hal_amdgpu_semaphore_publish_signal(
      semaphore_, second_axis, builder.Build({{second_axis, 9}}), 9, 2));

  std::atomic<bool> reader_ready{false};
  std::thread writer([&] {
    FrontierBuilder producer;
    while (!reader_ready.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
    for (uint64_t value = 3; value <= kPublicationCount; ++value) {
      EXPECT_TRUE(iree_hal_amdgpu_semaphore_publish_signal(
          semaphore_, second_axis, producer.Build({{second_axis, value + 7}}),
          value + 7, value));
    }
  });
  reader_ready.store(true, std::memory_order_release);

  uint64_t upgraded_waits = 0;
  for (uint64_t i = 0; i < kPublicationCount; ++i) {
    iree_hal_amdgpu_last_signal_flags_t flags = 0;
    iree_async_axis_t axis = 0;
    uint64_t epoch = 0;
    uint64_t value = 0;
    EXPECT_TRUE(iree_hal_amdgpu_last_signal_load(
        iree_hal_amdgpu_semaphore_last_signal(semaphore_), &flags, &axis,
        &epoch, &value));
    auto resolution = ResolveWait(0, value);
    // A racing publication can make the exact metadata unavailable. It cannot
    // turn the requested wait into a barrier on that publication's epoch.
    if (!resolution.needs_deferral &&
        (resolution.barrier_count != 2 ||
         resolution.barriers[0].axis != first_axis ||
         resolution.barriers[0].target_epoch != 4 ||
         resolution.barriers[1].axis != second_axis ||
         resolution.barriers[1].target_epoch != value + 7)) {
      ++upgraded_waits;
    }
  }
  writer.join();
  EXPECT_EQ(upgraded_waits, 0u);
}

}  // namespace
