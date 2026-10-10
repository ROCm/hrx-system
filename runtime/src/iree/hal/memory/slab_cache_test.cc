// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/slab_cache.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/maintenance_thread.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Controls only the underlying host allocation dependency. All successful
// backing uses the real CPU provider and system allocator.
class ControlledAllocator {
 public:
  iree_allocator_t allocator() { return {this, Control}; }
  void FailNext() { fail_next_.store(true); }
  void BlockNext() {
    std::lock_guard<std::mutex> lock(mutex_);
    armed_ = true;
    entered_ = false;
    resumed_ = false;
  }
  void AwaitEntry() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [&] { return entered_; });
  }
  void Resume() {
    std::lock_guard<std::mutex> lock(mutex_);
    resumed_ = true;
    condition_.notify_all();
  }

 private:
  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_pointer) {
    auto* allocator = static_cast<ControlledAllocator*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      if (allocator->fail_next_.exchange(false)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "native allocation unavailable");
      }
      std::unique_lock<std::mutex> lock(allocator->mutex_);
      if (allocator->armed_) {
        allocator->armed_ = false;
        allocator->entered_ = true;
        allocator->condition_.notify_all();
        allocator->condition_.wait(lock, [&] { return allocator->resumed_; });
      }
    }
    const auto system = iree_allocator_system();
    return system.ctl(system.self, command, params, inout_pointer);
  }
  // One explicit native allocation failure, consumed by the next allocation.
  std::atomic<bool> fail_next_{false};
  // Protects the allocation gate's handshake.
  std::mutex mutex_;
  // Signals entry and resumption without timing assumptions.
  std::condition_variable condition_;
  // Whether the next allocation must enter the gate.
  bool armed_ = false;
  // Whether the worker reached the gate.
  bool entered_ = false;
  // Whether the test has released the worker.
  bool resumed_ = false;
};

class SlabCacheTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), allocator_, &proactor_));
    IREE_ASSERT_OK(iree_async_notification_create(
        proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), allocator_, &tracker_));
    for (uint8_t i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(iree_async_frontier_tracker_register_axis(
          tracker_, Axis(i), nullptr));
    }
    IREE_ASSERT_OK(iree_hal_memory_maintenance_thread_create({}, allocator_,
                                                             &maintenance_));
    IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
        256, native_allocator_.allocator(), &provider_));
    IREE_ASSERT_OK(
        iree_hal_passthrough_pool_create({}, provider_, notification_, tracker_,
                                         maintenance_, allocator_, &native_));
    params_.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    params_.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params_.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
    params_.queue_family_affinity = iree_hal_make_queue_family_affinity(0);
    params_.min_alignment = 16;
    IREE_ASSERT_OK(CreateCache(native_, 4096, 0, 4, &cache_));
  }

  void TearDown() override {
    for (auto i = children_.rbegin(); i != children_.rend(); ++i) {
      iree_hal_pool_release(*i);
    }
    iree_hal_pool_release(cache_);
    iree_hal_pool_release(native_);
    iree_hal_slab_provider_release(provider_);
    iree_hal_memory_maintenance_release(maintenance_);
    iree_async_frontier_tracker_release(tracker_);
    iree_async_notification_release(notification_);
    iree_async_proactor_release(proactor_);
  }

  static iree_async_axis_t Axis(uint8_t i) {
    return iree_async_axis_make_queue(1, 0, 0, i, 0);
  }

  iree_status_t CreateCache(iree_hal_pool_t* parent, iree_device_size_t length,
                            uint32_t target, uint32_t maximum,
                            iree_hal_pool_t** out_pool) {
    iree_hal_slab_cache_options_t options;
    iree_hal_slab_cache_options_initialize(&options);
    options.slab = {params_, length};
    options.target_count = target;
    options.max_count = maximum;
    return iree_hal_slab_cache_create(parent, &options, allocator_, out_pool);
  }

  iree_status_t CreateChild(iree_hal_pool_t* parent, iree_device_size_t length,
                            iree_hal_pool_t** out_pool) {
    iree_hal_tlsf_pool_options_t options = {};
    options.tlsf_options.range_length = length;
    options.tlsf_options.alignment = 16;
    options.tlsf_options.frontier_capacity = 2;
    iree_status_t status =
        iree_hal_tlsf_pool_create(parent, &options, allocator_, out_pool);
    if (iree_status_is_ok(status)) {
      children_.push_back(*out_pool);
    }
    return status;
  }

  void Drain() {
    iree_hal_memory_maintenance_call(
        maintenance_,
        [](void* user_data) {
          auto* owner = static_cast<iree_hal_memory_maintenance_t*>(user_data);
          while (iree_hal_memory_maintenance_run_one(owner)) {
          }
        },
        maintenance_);
  }

  iree_status_t Acquire(iree_hal_pool_t* pool, iree_device_size_t length,
                        const iree_async_frontier_t* requester,
                        iree_hal_pool_reserve_flags_t flags,
                        iree_hal_pool_reservation_t* reservation,
                        iree_hal_pool_acquire_info_t* info,
                        iree_hal_pool_acquire_result_t* result) {
    const iree_hal_pool_reservation_request_t request = {params_, length};
    return iree_hal_pool_acquire_reservations(pool, 1, &request, requester,
                                              flags, reservation, info, result);
  }

  iree_status_t CheckBytes(iree_hal_pool_t* pool,
                           const iree_hal_pool_reservation_t& reservation,
                           uint8_t pattern, uintptr_t* out_address) {
    const iree_hal_pool_reservation_request_t request = {
        params_, reservation.byte_length};
    iree_hal_buffer_t* buffer = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_pool_materialize_reservations(
        pool, 1, &request, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
        &buffer));
    iree_hal_buffer_mapping_t mapping;
    iree_status_t status = iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, IREE_HAL_WHOLE_BUFFER, &mapping);
    if (iree_status_is_ok(status)) {
      *out_address = reinterpret_cast<uintptr_t>(mapping.contents.data);
      memset(mapping.contents.data, pattern, mapping.contents.data_length);
      status = iree_hal_buffer_unmap_range(&mapping);
    }
    std::vector<uint8_t> actual(reservation.byte_length);
    if (iree_status_is_ok(status)) {
      status =
          iree_hal_buffer_map_read(buffer, 0, actual.data(), actual.size());
    }
    if (iree_status_is_ok(status)) {
      for (uint8_t value : actual) {
        EXPECT_EQ(value, pattern);
      }
    }
    iree_hal_buffer_release(buffer);
    return status;
  }

  // Controllable dependency for real native backing and wrappers.
  ControlledAllocator native_allocator_;
  // Allocator for pool metadata.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Sealed progress owner retained until every pool has retired.
  iree_async_proactor_t* proactor_ = nullptr;
  // Capacity publication shared by this native memory domain.
  iree_async_notification_t* notification_ = nullptr;
  // Two explicit causal axes used by reuse tests.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Placement-local native owner shared by all pools.
  iree_hal_memory_maintenance_t* maintenance_ = nullptr;
  // Real CPU native storage implementation.
  iree_hal_slab_provider_t* provider_ = nullptr;
  // Leaf reservation owner.
  iree_hal_pool_t* native_ = nullptr;
  // Explicit shared retention policy.
  iree_hal_pool_t* cache_ = nullptr;
  // Child allocator policies released in reverse creation order.
  std::vector<iree_hal_pool_t*> children_;
  // User-visible geometry and permissions.
  iree_hal_buffer_params_t params_ = {};
};

TEST_F(SlabCacheTest, IndependentChildrenReuseOneNativeAllocation) {
  iree_hal_pool_t* first = nullptr;
  IREE_ASSERT_OK(CreateChild(cache_, 4096, &first));
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(Acquire(first, 256, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &reservation, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  uintptr_t first_address = 0;
  IREE_ASSERT_OK(CheckBytes(first, reservation, 0x49, &first_address));
  iree_hal_pool_release_reservations(first, 1, &reservation, nullptr);
  iree_hal_pool_t* second = nullptr;
  IREE_ASSERT_OK(CreateChild(cache_, 4096, &second));
  IREE_ASSERT_OK(Acquire(second, 256, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &reservation, &info, &result));
  uintptr_t second_address = 0;
  IREE_ASSERT_OK(CheckBytes(second, reservation, 0x93, &second_address));
  EXPECT_EQ(first_address, second_address);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(native_, &stats);
  EXPECT_EQ(stats.reserve_count, 1u);
  iree_hal_slab_cache_stats_t cache_stats;
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache_, &cache_stats));
  EXPECT_EQ(cache_stats.hit_count, 1u);
  iree_hal_pool_release_reservations(second, 1, &reservation, nullptr);
}

TEST_F(SlabCacheTest, PendingReturnPreservesExactHistoryAndPrefersReady) {
  std::array<iree_hal_pool_reservation_request_t, 2> requests = {
      {{params_, 4096}, {params_, 4096}}};
  std::array<iree_hal_pool_reservation_t, 2> reservations;
  std::array<iree_hal_pool_acquire_info_t, 2> infos;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      cache_, 2, requests.data(), nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      reservations.data(), infos.data(), &result));
  iree_hal_pool_release_reservations(cache_, 1, &reservations[0], nullptr);
  iree_async_single_frontier_t frontier = {.entry_count = 1};
  frontier.entries[0] = {Axis(0), 7};
  const auto* history = iree_async_fixed_frontier_as_const_frontier(&frontier);
  iree_hal_pool_release_reservations(cache_, 1, &reservations[1], history);
  iree_hal_pool_reservation_t ready;
  iree_hal_pool_acquire_info_t info;
  const auto flags = IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH |
                     IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER;
  IREE_ASSERT_OK(Acquire(cache_, 4096, nullptr, flags, &ready, &info, &result));
  EXPECT_EQ(ready.block_handle, reservations[0].block_handle);
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_pool_reservation_t pending;
  IREE_ASSERT_OK(
      Acquire(cache_, 4096, nullptr, flags, &pending, &info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  ASSERT_NE(info.reuse_frontier, nullptr);
  EXPECT_EQ(info.reuse_frontier->entries[0].epoch, 7u);
  // Unused return may alias the cache's own exact-history storage.
  iree_hal_pool_release_reservations(cache_, 1, &pending, info.reuse_frontier);
  IREE_ASSERT_OK(
      Acquire(cache_, 4096, history, flags, &pending, &info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  EXPECT_EQ(info.reuse_frontier->entries[0].epoch, 7u);
  iree_hal_pool_release_reservations(cache_, 1, &pending, info.reuse_frontier);
  iree_hal_slab_cache_stats_t stats;
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache_, &stats));
  EXPECT_EQ(stats.pending_count, 1u);
  iree_async_frontier_tracker_advance(tracker_, Axis(0), 7);
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache_, &stats));
  EXPECT_EQ(stats.ready_count, 1u);
  EXPECT_EQ(stats.pending_count, 0u);
  iree_hal_pool_release_reservations(cache_, 1, &ready, nullptr);
}

TEST_F(SlabCacheTest, FailedBatchRestoresIdleCapacityWithoutNotification) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(Acquire(cache_, 4096, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &reservation, &info, &result));
  iree_hal_pool_release_reservations(cache_, 1, &reservation, nullptr);
  const std::array<iree_hal_pool_reservation_request_t, 2> requests = {
      {{params_, 4096}, {params_, 4096}}};
  std::array<iree_hal_pool_reservation_t, 2> reservations;
  memset(reservations.data(), 0xA5, sizeof(reservations));
  const auto original = reservations;
  std::array<iree_hal_pool_acquire_info_t, 2> infos;
  const uint32_t token =
      iree_async_notification_begin_observe(iree_hal_pool_notification(cache_));
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      cache_, 2, requests.data(), nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH, reservations.data(),
      infos.data(), &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  EXPECT_EQ(memcmp(reservations.data(), original.data(), sizeof(reservations)),
            0);
  EXPECT_EQ(infos[0].result, IREE_HAL_POOL_ACQUIRE_NONE);
  EXPECT_EQ(infos[1].flags, IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED);
  EXPECT_EQ(
      iree_async_notification_query_epoch(iree_hal_pool_notification(cache_)),
      token);
  iree_async_notification_end_observe(iree_hal_pool_notification(cache_));
  IREE_ASSERT_OK(Acquire(cache_, 4096, nullptr,
                         IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH,
                         &reservation, &info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_pool_release_reservations(cache_, 1, &reservation, nullptr);
}

TEST_F(SlabCacheTest, ExplicitPrefillAndTrimRetainPolicyAndByteFloor) {
  IREE_ASSERT_OK(iree_hal_slab_cache_set_target(cache_, 2));
  Drain();
  iree_hal_slab_cache_stats_t stats;
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache_, &stats));
  EXPECT_EQ(stats.ready_count, 2u);
  iree_hal_pool_trim(cache_, IREE_HAL_POOL_TRIM_FLAG_EXCESS, 0);
  Drain();
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache_, &stats));
  EXPECT_EQ(stats.ready_count, 2u);
  iree_hal_pool_trim(cache_, IREE_HAL_POOL_TRIM_FLAG_ALL, 4096);
  Drain();
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache_, &stats));
  EXPECT_EQ(stats.ready_count, 1u);
  iree_hal_pool_trim(cache_, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  Drain();
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache_, &stats));
  EXPECT_EQ(stats.ready_count, 0u);
  // ALL is one-shot; the configured target still applies on the next trigger.
  IREE_ASSERT_OK(iree_hal_slab_cache_set_target(cache_, 2));
  Drain();
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache_, &stats));
  EXPECT_EQ(stats.ready_count, 2u);
  IREE_ASSERT_OK(iree_hal_slab_cache_set_target(cache_, 0));
}

TEST_F(SlabCacheTest, OversizedAndAlignmentBypassUseSameParent) {
  for (const auto& request : std::array<iree_hal_pool_reservation_request_t, 2>{
           {{params_, 8192}, {params_, 256}}}) {
    auto aligned_request = request;
    if (aligned_request.allocation_size == 256) {
      aligned_request.params.min_alignment = 256;
    }
    iree_hal_pool_reservation_t reservation;
    iree_hal_pool_acquire_info_t info;
    iree_hal_pool_acquire_result_t result;
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
        cache_, 1, &aligned_request, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
        &reservation, &info, &result));
    uintptr_t address = 0;
    IREE_ASSERT_OK(CheckBytes(cache_, reservation, 0x2A, &address));
    EXPECT_EQ(address % aligned_request.params.min_alignment, 0u);
    iree_hal_pool_release_reservations(cache_, 1, &reservation, nullptr);
  }
  Drain();
  iree_hal_slab_cache_stats_t stats;
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache_, &stats));
  EXPECT_EQ(stats.ready_count, 0u);
  EXPECT_EQ(stats.bypass_count, 2u);
  iree_hal_pool_stats_t native_stats;
  iree_hal_pool_query_stats(native_, &native_stats);
  EXPECT_EQ(native_stats.reserve_count, 2u);
}

TEST_F(SlabCacheTest, OwningMaterializationReturnsWholeEntry) {
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      cache_, params_, 256, iree_infinite_timeout(), &buffer));
  EXPECT_EQ(iree_hal_buffer_byte_length(buffer), 4096u);
  const uint8_t pattern = 0x3B;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_fill(buffer, 0, IREE_HAL_WHOLE_BUFFER, &pattern, 1));
  iree_hal_buffer_release(buffer);
  iree_hal_slab_cache_stats_t stats;
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache_, &stats));
  EXPECT_EQ(stats.ready_count, 1u);
}

TEST_F(SlabCacheTest, ChildReturnPreservesUntouchedAlignmentTailHistory) {
  iree_hal_pool_release(cache_);
  cache_ = nullptr;
  IREE_ASSERT_OK(CreateCache(native_, 4097, 0, 1, &cache_));
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(Acquire(cache_, 4097, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &reservation, &info, &result));
  iree_async_single_frontier_t original = {.entry_count = 1};
  original.entries[0] = {Axis(0), 7};
  const auto* original_history =
      iree_async_fixed_frontier_as_const_frontier(&original);
  iree_hal_pool_release_reservations(cache_, 1, &reservation, original_history);
  iree_hal_pool_t* child = nullptr;
  IREE_ASSERT_OK(CreateChild(cache_, 4096, &child));
  IREE_ASSERT_OK(Acquire(child, 4096, original_history,
                         IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &info,
                         &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  // The child owns only the aligned 4096-byte interval. Return that interval
  // under its later epoch; the untouched final byte still carries axis 0.
  iree_async_single_frontier_t later = {.entry_count = 1};
  later.entries[0] = {Axis(1), 11};
  iree_hal_pool_release_reservations(
      child, 1, &reservation,
      iree_async_fixed_frontier_as_const_frontier(&later));
  iree_hal_pool_trim(child, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  IREE_ASSERT_OK(Acquire(cache_, 4097, nullptr,
                         IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER |
                             IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH,
                         &reservation, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  ASSERT_NE(info.reuse_frontier, nullptr);
  ASSERT_EQ(info.reuse_frontier->entry_count, 2u);
  EXPECT_EQ(info.reuse_frontier->entries[0].axis, Axis(0));
  EXPECT_EQ(info.reuse_frontier->entries[0].epoch, 7u);
  EXPECT_EQ(info.reuse_frontier->entries[1].axis, Axis(1));
  EXPECT_EQ(info.reuse_frontier->entries[1].epoch, 11u);
  iree_async_frontier_tracker_advance(tracker_, Axis(0), 7);
  iree_async_frontier_tracker_advance(tracker_, Axis(1), 11);
  iree_hal_pool_release_reservations(cache_, 1, &reservation, nullptr);
}

TEST_F(SlabCacheTest, HotReuseAndTrimDoNotWaitForNativePreparation) {
  IREE_ASSERT_OK(iree_hal_slab_cache_set_target(cache_, 1));
  Drain();
  native_allocator_.BlockNext();
  IREE_ASSERT_OK(iree_hal_slab_cache_set_target(cache_, 2));
  native_allocator_.AwaitEntry();
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_EXPECT_OK(Acquire(cache_, 256, nullptr,
                         IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH,
                         &reservation, &info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  uintptr_t address = 0;
  IREE_EXPECT_OK(CheckBytes(cache_, reservation, 0x9C, &address));
  iree_hal_pool_release_reservations(cache_, 1, &reservation, nullptr);
  iree_hal_pool_trim(cache_, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  native_allocator_.Resume();
  Drain();
  iree_hal_slab_cache_stats_t stats;
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache_, &stats));
  EXPECT_EQ(stats.ready_count, 0u);
  // Preparation admitted before ALL cannot repopulate the cache afterward.
  IREE_ASSERT_OK(iree_hal_slab_cache_set_target(cache_, 2));
  Drain();
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache_, &stats));
  EXPECT_EQ(stats.ready_count, 2u);
}

TEST_F(SlabCacheTest, TrimIncludesQueuedChildReturnsDuringNativePreparation) {
  iree_hal_pool_t* children[2];
  iree_hal_pool_reservation_t reservations[2];
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  for (size_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(CreateChild(cache_, 4096, &children[i]));
    IREE_ASSERT_OK(Acquire(children[i], 4096, nullptr,
                           IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservations[i],
                           &info, &result));
    ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  }
  Drain();
  native_allocator_.BlockNext();
  IREE_ASSERT_OK(iree_hal_slab_cache_set_target(cache_, 1));
  native_allocator_.AwaitEntry();

  // Both returns precede their respective trim, but the second return follows
  // the first trim's queued sweep. Neither trim may join the paused owner.
  iree_hal_pool_release_reservations(children[0], 1, &reservations[0], nullptr);
  // A child trim floor does not turn the child into an idle retention cache.
  iree_hal_pool_trim(children[0], IREE_HAL_POOL_TRIM_FLAG_ALL, 4096);
  iree_hal_pool_trim(cache_, IREE_HAL_POOL_TRIM_FLAG_EXCESS, 4096);
  iree_hal_pool_release_reservations(children[1], 1, &reservations[1], nullptr);
  iree_hal_pool_trim(cache_, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  native_allocator_.Resume();
  // The first sweep moves behind the second generation's child return.
  // Draining joins that requeued sweep and all resulting parent returns.
  Drain();
  iree_hal_slab_cache_stats_t cache_stats;
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache_, &cache_stats));
  EXPECT_EQ(cache_stats.ready_count, 0u);
  EXPECT_EQ(cache_stats.pending_count, 0u);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(cache_, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
  for (auto* child : children) {
    iree_hal_pool_query_stats(child, &stats);
    EXPECT_EQ(stats.slab_count, 0u);
  }
}

TEST_F(SlabCacheTest, AsyncPreparationFailurePropagatesAndCanBeRetried) {
  native_allocator_.FailNext();
  IREE_ASSERT_OK(iree_hal_slab_cache_set_target(cache_, 1));
  Drain();
  iree_hal_pool_reservation_t reservation;
  memset(&reservation, 0xA5, sizeof(reservation));
  const auto original = reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result = IREE_HAL_POOL_ACQUIRE_NONE;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      Acquire(cache_, 256, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
              &reservation, &info, &result));
  EXPECT_EQ(memcmp(&reservation, &original, sizeof(reservation)), 0);
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_NONE);
  IREE_ASSERT_OK(Acquire(cache_, 256, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &reservation, &info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_pool_release_reservations(cache_, 1, &reservation, nullptr);
}

TEST_F(SlabCacheTest, FailedOwningMaterializationKeepsEveryReservation) {
  std::array<iree_hal_pool_reservation_request_t, 2> requests = {
      {{params_, 256}, {params_, 256}}};
  std::array<iree_hal_pool_reservation_t, 2> reservations;
  std::array<iree_hal_pool_acquire_info_t, 2> infos;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      cache_, 2, requests.data(), nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      reservations.data(), infos.data(), &result));
  // The first staged view succeeds. The second requests a usage bit that
  // the prepared native view does not grant.
  requests[1].params.usage = static_cast<iree_hal_buffer_usage_t>(1u << 31);
  std::array<iree_hal_buffer_t*, 2> buffers = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_pool_materialize_reservations(
          cache_, 2, requests.data(), reservations.data(),
          IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP,
          buffers.data()));
  EXPECT_EQ(buffers[0], nullptr);
  EXPECT_EQ(buffers[1], nullptr);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(cache_, &stats);
  EXPECT_EQ(stats.reservation_count, 2u);
  requests[1] = requests[0];
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      cache_, 2, requests.data(), reservations.data(),
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP,
      buffers.data()));
  for (auto* buffer : buffers) {
    iree_hal_buffer_release(buffer);
  }
  iree_hal_pool_query_stats(cache_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
}

TEST_F(SlabCacheTest, ConcurrentOwningAllocationsAndReturns) {
  std::array<std::thread, 4> threads;
  for (size_t thread_index = 0; thread_index < threads.size(); ++thread_index) {
    threads[thread_index] = std::thread([&, thread_index] {
      for (size_t iteration = 0; iteration < 128; ++iteration) {
        iree_hal_buffer_t* buffer = nullptr;
        IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
            cache_, params_, 256, iree_infinite_timeout(), &buffer));
        const uint8_t pattern = static_cast<uint8_t>(thread_index + iteration);
        IREE_EXPECT_OK(iree_hal_buffer_map_fill(
            buffer, 0, IREE_HAL_WHOLE_BUFFER, &pattern, 1));
        uint8_t actual = 0;
        IREE_EXPECT_OK(iree_hal_buffer_map_read(buffer, 255, &actual, 1));
        EXPECT_EQ(actual, pattern);
        iree_hal_buffer_release(buffer);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  Drain();
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(cache_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.reserve_count, 512u);
  EXPECT_EQ(stats.release_count, 512u);
  iree_hal_slab_cache_stats_t cache_stats;
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache_, &cache_stats));
  EXPECT_LE(cache_stats.ready_count, 4u);
}

TEST_F(SlabCacheTest, InteriorRangePreparationPreservesNeighbors) {
  iree_hal_buffer_t* backing = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      native_, params_, 8192, iree_infinite_timeout(), &backing));
  const uint8_t pattern = 0x6D;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_fill(backing, 0, IREE_HAL_WHOLE_BUFFER, &pattern, 1));
  iree_hal_fixed_block_pool_options_t options = {.block_size = 1024};
  iree_hal_pool_t* interior = nullptr;
  IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
      backing, 16, 1024, &options, allocator_, &interior));
  iree_hal_pool_t* cache = nullptr;
  IREE_ASSERT_OK(CreateCache(interior, 1024, 1, 1, &cache));
  Drain();
  iree_hal_slab_cache_stats_t stats;
  IREE_ASSERT_OK(iree_hal_slab_cache_query_stats(cache, &stats));
  EXPECT_EQ(stats.ready_count, 1u);
  std::array<uint8_t, 8192> actual;
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(backing, 0, actual.data(), actual.size()));
  for (size_t i = 0; i < actual.size(); ++i) {
    if (i < 16 || i >= 1040) {
      EXPECT_EQ(actual[i], pattern) << i;
    }
  }
  iree_hal_pool_release(cache);
  iree_hal_pool_release(interior);
  iree_hal_buffer_release(backing);
}

}  // namespace
