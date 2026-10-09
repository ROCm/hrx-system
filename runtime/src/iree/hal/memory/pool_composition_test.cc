// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <thread>
#include <tuple>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/maintenance_thread.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

enum class AllocatorKind { kTLSF, kFixedBlock };

class PoolCompositionTest : public ::testing::TestWithParam<
                                std::tuple<AllocatorKind, AllocatorKind>> {
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
    IREE_ASSERT_OK(
        iree_hal_cpu_slab_provider_create(0, allocator_, &provider_));
    IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
        {}, provider_, notification_, tracker_, maintenance_, allocator_,
        &native_pool_));
    params_.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
    params_.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params_.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
    params_.queue_family_affinity = iree_hal_make_queue_family_affinity(0) |
                                    iree_hal_make_queue_family_affinity(3);
    params_.min_alignment = 16;
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        native_pool_, params_, 4096, iree_infinite_timeout(), &backing_));
  }

  void TearDown() override {
    for (auto i = pools_.rbegin(); i != pools_.rend(); ++i) {
      iree_hal_pool_release(*i);
    }
    iree_hal_buffer_release(backing_);
    iree_hal_pool_release(native_pool_);
    iree_hal_slab_provider_release(provider_);
    iree_hal_memory_maintenance_release(maintenance_);
    iree_async_frontier_tracker_release(tracker_);
    iree_async_notification_release(notification_);
    iree_async_proactor_release(proactor_);
  }

  static iree_async_axis_t Axis(uint8_t index) {
    return iree_async_axis_make_queue(1, 0, 0, index, 0);
  }

  iree_status_t CreateSource(iree_device_size_t offset,
                             iree_device_size_t length,
                             iree_device_size_t block_size,
                             iree_hal_pool_t** out_pool) {
    iree_status_t status = iree_ok_status();
    if (std::get<0>(GetParam()) == AllocatorKind::kTLSF) {
      iree_hal_tlsf_pool_options_t options = {};
      options.tlsf_options.frontier_capacity = 2;
      status = iree_hal_tlsf_pool_create_from_buffer(
          backing_, offset, length, &options, allocator_, out_pool);
    } else {
      iree_hal_fixed_block_pool_options_t options = {.block_size = block_size,
                                                     .frontier_capacity = 2};
      status = iree_hal_fixed_block_pool_create_from_buffer(
          backing_, offset, length, &options, allocator_, out_pool);
    }
    if (iree_status_is_ok(status)) {
      pools_.push_back(*out_pool);
    }
    return status;
  }

  iree_status_t CreateChild(iree_hal_pool_t* parent,
                            iree_device_size_t slab_size,
                            uint8_t frontier_capacity,
                            iree_hal_pool_t** out_pool,
                            iree_device_size_t block_size = 256) {
    iree_status_t status;
    if (std::get<1>(GetParam()) == AllocatorKind::kTLSF) {
      iree_hal_tlsf_pool_options_t options = {};
      options.tlsf_options.range_length = slab_size;
      options.tlsf_options.alignment = 16;
      options.tlsf_options.frontier_capacity = frontier_capacity;
      status =
          iree_hal_tlsf_pool_create(parent, &options, allocator_, out_pool);
    } else {
      iree_hal_fixed_block_pool_options_t options = {};
      options.block_size = block_size;
      options.blocks_per_slab = slab_size / block_size;
      options.alignment = 16;
      options.frontier_capacity = frontier_capacity;
      status = iree_hal_fixed_block_pool_create(parent, &options, allocator_,
                                                out_pool);
    }
    if (iree_status_is_ok(status)) {
      pools_.push_back(*out_pool);
    }
    return status;
  }

  void WaitForMaintenance() {
    iree_hal_memory_maintenance_call(
        maintenance_,
        [](void* user_data) {
          auto* owner = static_cast<iree_hal_memory_maintenance_t*>(user_data);
          while (iree_hal_memory_maintenance_run_one(owner)) {
          }
        },
        maintenance_);
  }

  static iree_hal_pool_stats_t Stats(iree_hal_pool_t* pool) {
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool, &stats);
    return stats;
  }

  iree_status_t Acquire(iree_hal_pool_t* pool, iree_device_size_t length,
                        const iree_async_frontier_t* frontier,
                        iree_hal_pool_reserve_flags_t flags,
                        iree_hal_pool_reservation_t* out_reservation,
                        iree_hal_pool_acquire_info_t* out_info,
                        iree_hal_pool_acquire_result_t* out_result) {
    const iree_hal_pool_reservation_request_t request = {params_, length};
    return iree_hal_pool_acquire_reservations(pool, 1, &request, frontier,
                                              flags, out_reservation, out_info,
                                              out_result);
  }

  // Writes and reads through the actual native mapping, returning its address
  // only as an identity for comparing two independently owned epochs.
  iree_status_t CheckBytes(iree_hal_pool_t* pool,
                           const iree_hal_pool_reservation_t& reservation,
                           uint8_t pattern, uintptr_t* out_address) {
    const iree_hal_pool_reservation_request_t request = {
        params_, reservation.byte_length};
    iree_hal_buffer_t* buffer = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_pool_materialize_reservations(
        pool, 1, &request, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
        &buffer));
    EXPECT_EQ(
        iree_hal_buffer_allocation_placement(buffer).queue_family_affinity,
        params_.queue_family_affinity);
    iree_hal_buffer_mapping_t mapping;
    iree_status_t status = iree_hal_buffer_map_range(
        buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_ALL,
        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, reservation.byte_length, &mapping);
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

  // Host metadata allocator, independent of native memory ownership.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Progress owner retained for the fixture's notifications.
  iree_async_proactor_t* proactor_ = nullptr;
  // Capacity notification captured by the native leaf.
  iree_async_notification_t* notification_ = nullptr;
  // Shared tracker with two exact queue axes.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Independent cold owner of native retirement.
  iree_hal_memory_maintenance_t* maintenance_ = nullptr;
  // Production CPU backing provider.
  iree_hal_slab_provider_t* provider_ = nullptr;
  // Production native leaf whose allocation count establishes physical reuse.
  iree_hal_pool_t* native_pool_ = nullptr;
  // One owned native epoch subdivided by the parent strategies.
  iree_hal_buffer_t* backing_ = nullptr;
  // Pools in creation order; teardown releases children before parents.
  std::vector<iree_hal_pool_t*> pools_;
  // Permissions and family scope carried from native backing to child views.
  iree_hal_buffer_params_t params_ = {};
};

TEST_P(PoolCompositionTest, ReusesBackingAfterFirstChildIsDestroyed) {
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(0, 4096, 4096, &source));
  iree_hal_pool_t* first = nullptr;
  IREE_ASSERT_OK(CreateChild(source, 4096, 2, &first));
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(first, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(Acquire(first, 256, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &reservation, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  uintptr_t first_address = 0;
  IREE_ASSERT_OK(CheckBytes(first, reservation, 0x49, &first_address));
  iree_hal_pool_release_reservations(first, 1, &reservation, nullptr);
  iree_hal_pool_trim(first, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  iree_hal_pool_release(first);
  pools_.pop_back();

  iree_hal_pool_t* second = nullptr;
  IREE_ASSERT_OK(CreateChild(source, 4096, 2, &second));
  IREE_ASSERT_OK(Acquire(second, 256, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &reservation, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  uintptr_t second_address = 0;
  IREE_ASSERT_OK(CheckBytes(second, reservation, 0xA7, &second_address));
  EXPECT_EQ(second_address, first_address);
  iree_hal_pool_query_stats(native_pool_, &stats);
  EXPECT_EQ(stats.reserve_count, 1u);
  EXPECT_EQ(stats.release_count, 0u);
  EXPECT_EQ(stats.bytes_committed, 4096u);
  iree_hal_pool_release_reservations(second, 1, &reservation, nullptr);
}

TEST_P(PoolCompositionTest, UntouchedRangesAndReturnsPreserveHistory) {
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(0, 4096, 4096, &source));
  alignas(iree_async_frontier_entry_t)
      uint8_t storage[sizeof(iree_async_frontier_t) +
                      2 * sizeof(iree_async_frontier_entry_t)];
  auto* history = reinterpret_cast<iree_async_frontier_t*>(storage);
  iree_async_frontier_initialize(history, 2);
  history->entries[0] = {Axis(0), 7};
  history->entries[1] = {Axis(1), 11};
  iree_hal_pool_reservation_t seed;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(Acquire(source, 4096, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &seed, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_pool_release_reservations(source, 1, &seed, history);
  iree_hal_pool_t* child = nullptr;
  IREE_ASSERT_OK(CreateChild(source, 4096, 2, &child));
  std::array<iree_hal_pool_reservation_t, 2> held;
  IREE_ASSERT_OK(Acquire(child, 256, history, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &held[0], &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  ASSERT_NE(info.reuse_frontier, nullptr);
  EXPECT_EQ(iree_async_frontier_compare(info.reuse_frontier, history),
            IREE_ASYNC_FRONTIER_EQUAL);
  IREE_ASSERT_OK(Acquire(child, 256, nullptr,
                         IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH, &held[1],
                         &info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  IREE_ASSERT_OK(Acquire(child, 256, nullptr,
                         IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
                         &held[1], &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  EXPECT_EQ(iree_async_frontier_compare(info.reuse_frontier, history),
            IREE_ASYNC_FRONTIER_EQUAL);
  iree_hal_pool_release_reservations(child, held.size(), held.data(), history);
  // Destruction itself returns the exact history, including unused bytes.
  iree_hal_pool_release(child);
  pools_.pop_back();
  IREE_ASSERT_OK(Acquire(source, 4096, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &seed, &info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  IREE_ASSERT_OK(Acquire(source, 4096, nullptr,
                         IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER, &seed,
                         &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  EXPECT_EQ(iree_async_frontier_compare(info.reuse_frontier, history),
            IREE_ASYNC_FRONTIER_EQUAL);
  iree_hal_pool_release_reservations(source, 1, &seed, info.reuse_frontier);

  // A new child can also inherit an unresolved parent reservation directly.
  IREE_ASSERT_OK(CreateChild(source, 4096, 2, &child));
  IREE_ASSERT_OK(Acquire(child, 256, nullptr,
                         IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
                         &held[0], &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  EXPECT_EQ(iree_async_frontier_compare(info.reuse_frontier, history),
            IREE_ASYNC_FRONTIER_EQUAL);
  iree_hal_pool_release_reservations(child, 1, &held[0], info.reuse_frontier);
  iree_hal_pool_trim(child, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
}

TEST_P(PoolCompositionTest, FailedHistoryAdoptionReturnsParentReservation) {
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(0, 4096, 4096, &source));
  alignas(iree_async_frontier_entry_t)
      uint8_t storage[sizeof(iree_async_frontier_t) +
                      2 * sizeof(iree_async_frontier_entry_t)];
  auto* history = reinterpret_cast<iree_async_frontier_t*>(storage);
  iree_async_frontier_initialize(history, 2);
  history->entries[0] = {Axis(0), 7};
  history->entries[1] = {Axis(1), 11};
  iree_hal_pool_reservation_t seed;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(Acquire(source, 4096, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &seed, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_pool_release_reservations(source, 1, &seed, history);
  iree_hal_pool_t* child = nullptr;
  IREE_ASSERT_OK(CreateChild(source, 4096, 1, &child));
  iree_hal_pool_reservation_t unchanged;
  memset(&unchanged, 0xA5, sizeof(unchanged));
  auto reservation = unchanged;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      Acquire(child, 256, history, IREE_HAL_POOL_RESERVE_FLAG_NONE,
              &reservation, &info, &result));
  EXPECT_EQ(memcmp(&reservation, &unchanged, sizeof(reservation)), 0);
  IREE_ASSERT_OK(Acquire(source, 4096, nullptr,
                         IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER, &seed,
                         &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  EXPECT_EQ(iree_async_frontier_compare(info.reuse_frontier, history),
            IREE_ASYNC_FRONTIER_EQUAL);
  iree_hal_pool_release_reservations(source, 1, &seed, info.reuse_frontier);
}

TEST_P(PoolCompositionTest, ExhaustedParentLeavesOutputsUntouched) {
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(0, 4096, 4096, &source));
  iree_hal_pool_reservation_t held;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(Acquire(source, 4096, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &held, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_pool_t* child = nullptr;
  IREE_ASSERT_OK(CreateChild(source, 4096, 2, &child));
  iree_hal_pool_reservation_t unchanged;
  memset(&unchanged, 0xA5, sizeof(unchanged));
  auto reservation = unchanged;
  IREE_ASSERT_OK(Acquire(child, 256, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &reservation, &info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  EXPECT_EQ(info.result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  EXPECT_EQ(memcmp(&reservation, &unchanged, sizeof(reservation)), 0);
  iree_hal_pool_release_reservations(source, 1, &held, nullptr);
  IREE_ASSERT_OK(Acquire(child, 256, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &reservation, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_pool_release_reservations(child, 1, &reservation, nullptr);
}

TEST_P(PoolCompositionTest, ParentBudgetIsReportedWithoutOwningBacking) {
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(0, 4096, 4096, &source));
  iree_hal_tlsf_pool_options_t options = {};
  options.tlsf_options.range_length = 4096;
  options.budget_limit = 1024;
  iree_hal_pool_t* limited = nullptr;
  IREE_ASSERT_OK(
      iree_hal_tlsf_pool_create(source, &options, allocator_, &limited));
  pools_.push_back(limited);
  iree_hal_pool_t* child = nullptr;
  IREE_ASSERT_OK(CreateChild(limited, 4096, 2, &child));
  iree_hal_pool_reservation_t unchanged;
  memset(&unchanged, 0xA5, sizeof(unchanged));
  auto reservation = unchanged;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(Acquire(child, 256, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &reservation, &info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OVER_BUDGET);
  EXPECT_EQ(info.result, IREE_HAL_POOL_ACQUIRE_OVER_BUDGET);
  EXPECT_EQ(memcmp(&reservation, &unchanged, sizeof(reservation)), 0);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(child, &stats);
  EXPECT_EQ(stats.over_budget_count, 1u);
  EXPECT_EQ(stats.bytes_committed, 0u);
  iree_hal_pool_query_stats(limited, &stats);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.bytes_committed, 0u);
}

TEST_P(PoolCompositionTest, InteriorBackingPreservesNeighborBytes) {
  std::array<uint8_t, 4096> bytes;
  bytes.fill(0x37);
  IREE_ASSERT_OK(
      iree_hal_buffer_map_write(backing_, 0, bytes.data(), bytes.size()));
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(128, 2048, 1024, &source));
  std::array<iree_hal_pool_t*, 2> children;
  std::array<iree_hal_pool_reservation_t, 2> reservations;
  for (size_t i = 0; i < children.size(); ++i) {
    IREE_ASSERT_OK(CreateChild(source, 1024, 2, &children[i], 1024));
    iree_hal_pool_acquire_info_t info;
    iree_hal_pool_acquire_result_t result;
    IREE_ASSERT_OK(Acquire(children[i], 1024, nullptr,
                           IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservations[i],
                           &info, &result));
    ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
    uintptr_t address = 0;
    IREE_ASSERT_OK(
        CheckBytes(children[i], reservations[i], 0x80 + i, &address));
  }
  IREE_ASSERT_OK(
      iree_hal_buffer_map_read(backing_, 0, bytes.data(), bytes.size()));
  for (size_t i = 0; i < bytes.size(); ++i) {
    const uint8_t expected = i < 128 || i >= 2176 ? 0x37
                             : i < 1152           ? 0x80
                                                  : 0x81;
    EXPECT_EQ(bytes[i], expected) << i;
  }
  for (size_t i = 0; i < children.size(); ++i) {
    iree_hal_pool_release_reservations(children[i], 1, &reservations[i],
                                       nullptr);
  }
}

TEST_P(PoolCompositionTest, ParentPreparesAdvertisedChildAlignment) {
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(0, 4096, 512, &source));
  iree_hal_pool_reservation_t anchor;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(Acquire(source, 16, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &anchor, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_pool_t* child = nullptr;
  IREE_ASSERT_OK(CreateChild(source, 512, 2, &child, 64));
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(child, &capabilities);
  iree_hal_pool_reservation_request_t request = {params_, 64};
  request.params.min_alignment =
      iree_min(4096, capabilities.max_allocation_alignment);
  iree_hal_pool_reservation_t reservation;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      child, 1, &request, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      &reservation, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      child, 1, &request, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
      &buffer));
  const auto memory = iree_hal_buffer_memory_view(buffer);
  EXPECT_EQ(memory.offset % request.params.min_alignment, 0u);
  iree_hal_buffer_release(buffer);
  iree_hal_pool_release_reservations(child, 1, &reservation, nullptr);
  iree_hal_pool_release_reservations(source, 1, &anchor, nullptr);
}

TEST_P(PoolCompositionTest, BatchGrowthRollsBackAndReturnsWholeRanges) {
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(128, 1536, 512, &source));
  iree_hal_pool_t* child = nullptr;
  IREE_ASSERT_OK(CreateChild(source, 512, 2, &child));
  std::array<iree_hal_pool_reservation_request_t, 7> requests;
  requests.fill({params_, 256});
  std::array<iree_hal_pool_reservation_t, 7> reservations;
  memset(reservations.data(), 0xA5, sizeof(reservations));
  const auto unchanged = reservations;
  std::array<iree_hal_pool_acquire_info_t, 7> infos;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      child, requests.size(), requests.data(), nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, reservations.data(), infos.data(),
      &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  EXPECT_EQ(memcmp(reservations.data(), unchanged.data(), sizeof(reservations)),
            0);
  EXPECT_EQ(Stats(child).reservation_count, 0u);
  WaitForMaintenance();
  EXPECT_EQ(Stats(child).bytes_committed, 0u);
  EXPECT_EQ(Stats(source).reservation_count, 0u);

  // The same capacity is available to a fitting batch after rollback. Each
  // parent extent contains two blocks, so this transaction spans three slabs.
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      child, 6, requests.data(), nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      reservations.data(), infos.data(), &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(Stats(child).slab_count, 3u);
  EXPECT_EQ(Stats(source).reservation_count, 3u);
  std::array<iree_hal_buffer_t*, 6> buffers = {};
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      child, buffers.size(), requests.data(), reservations.data(),
      IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, buffers.data()));
  for (size_t i = 0; i < buffers.size(); ++i) {
    const uint8_t pattern = 0x50 + i;
    IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffers[i], 0, 256, &pattern, 1));
  }
  for (size_t i = 0; i < buffers.size(); ++i) {
    std::array<uint8_t, 256> bytes;
    IREE_ASSERT_OK(
        iree_hal_buffer_map_read(buffers[i], 0, bytes.data(), bytes.size()));
    for (uint8_t value : bytes) {
      EXPECT_EQ(value, 0x50 + i);
    }
    iree_hal_buffer_release(buffers[i]);
  }
  iree_hal_pool_release_reservations(child, buffers.size(), reservations.data(),
                                     nullptr);
  WaitForMaintenance();
  EXPECT_EQ(Stats(child).bytes_committed, 0u);
  EXPECT_EQ(Stats(source).reservation_count, 0u);
  EXPECT_EQ(Stats(native_pool_).reserve_count, 1u);
}

TEST_P(PoolCompositionTest, ConcurrentGrowthAndReturnPreserveLiveBytes) {
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(0, 4096, 512, &source));
  iree_hal_pool_t* child = nullptr;
  IREE_ASSERT_OK(CreateChild(source, 512, 2, &child));
  std::array<std::thread, 4> threads;
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i] {
      for (uint32_t iteration = 0; iteration < 100; ++iteration) {
        iree_hal_buffer_t* buffer = nullptr;
        IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
            child, params_, 256, iree_infinite_timeout(), &buffer));
        const uint8_t pattern = i * 100 + iteration;
        IREE_EXPECT_OK(iree_hal_buffer_map_fill(buffer, 0, 256, &pattern, 1));
        std::this_thread::yield();
        std::array<uint8_t, 256> bytes;
        IREE_EXPECT_OK(
            iree_hal_buffer_map_read(buffer, 0, bytes.data(), bytes.size()));
        for (uint8_t value : bytes) {
          EXPECT_EQ(value, pattern);
        }
        iree_hal_buffer_release(buffer);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  WaitForMaintenance();
  EXPECT_EQ(Stats(child).reservation_count, 0u);
  EXPECT_EQ(Stats(child).bytes_committed, 0u);
  EXPECT_EQ(Stats(source).reservation_count, 0u);
  EXPECT_EQ(Stats(native_pool_).reserve_count, 1u);
}

class FixedBlockPoolCompositionTest : public PoolCompositionTest {};

TEST_P(FixedBlockPoolCompositionTest,
       WholeReturnPreservesUntouchedTailHistory) {
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(0, 1024, 1024, &source));
  iree_async_single_frontier_t previous;
  iree_async_single_frontier_initialize(&previous, Axis(0), 7);
  const auto* previous_frontier =
      iree_async_single_frontier_as_const_frontier(&previous);
  iree_hal_pool_reservation_t seed;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(Acquire(source, 1024, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &seed, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_pool_release_reservations(source, 1, &seed, previous_frontier);

  iree_hal_slab_cache_options_t cache_options;
  iree_hal_slab_cache_options_initialize(&cache_options);
  cache_options.slab = {params_, 1024};
  iree_hal_pool_capabilities_t source_capabilities;
  iree_hal_pool_query_capabilities(source, &source_capabilities);
  cache_options.slab.params.min_alignment =
      iree_min(4096, source_capabilities.max_allocation_alignment);
  iree_hal_pool_t* cache = nullptr;
  IREE_ASSERT_OK(
      iree_hal_slab_cache_create(source, &cache_options, allocator_, &cache));
  pools_.push_back(cache);

  // Five 192-byte blocks leave a 64-byte tail whose original history must
  // survive even when every managed block acquires a different history.
  iree_hal_pool_t* child = nullptr;
  IREE_ASSERT_OK(CreateChild(cache, 1024, 2, &child, 192));
  std::array<iree_hal_pool_reservation_request_t, 5> requests;
  requests.fill({params_, 192});
  std::array<iree_hal_pool_reservation_t, 5> reservations;
  std::array<iree_hal_pool_acquire_info_t, 5> infos;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      child, requests.size(), requests.data(), previous_frontier,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, reservations.data(), infos.data(),
      &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  for (const auto& acquired : infos) {
    ASSERT_NE(acquired.reuse_frontier, nullptr);
    EXPECT_EQ(
        iree_async_frontier_compare(acquired.reuse_frontier, previous_frontier),
        IREE_ASYNC_FRONTIER_EQUAL);
  }
  iree_async_frontier_tracker_advance(tracker_, Axis(0), 7);
  iree_async_single_frontier_t next;
  iree_async_single_frontier_initialize(&next, Axis(1), 11);
  iree_hal_pool_release_reservations(
      child, reservations.size(), reservations.data(),
      iree_async_single_frontier_as_const_frontier(&next));
  WaitForMaintenance();
  EXPECT_EQ(Stats(child).bytes_committed, 0u);
  IREE_ASSERT_OK(Acquire(cache, 1024, nullptr,
                         IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER, &seed,
                         &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  ASSERT_EQ(info.reuse_frontier->entry_count, 2);
  EXPECT_EQ(info.reuse_frontier->entries[0].axis, Axis(0));
  EXPECT_EQ(info.reuse_frontier->entries[0].epoch, 7u);
  EXPECT_EQ(info.reuse_frontier->entries[1].axis, Axis(1));
  EXPECT_EQ(info.reuse_frontier->entries[1].epoch, 11u);
  iree_async_frontier_tracker_advance(tracker_, Axis(1), 11);
  iree_hal_pool_release_reservations(cache, 1, &seed, nullptr);
}

TEST_P(FixedBlockPoolCompositionTest,
       WideUnionRetainsIndividualBlockHistories) {
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(0, 512, 512, &source));
  iree_hal_pool_t* child = nullptr;
  IREE_ASSERT_OK(CreateChild(source, 512, 1, &child));
  std::array<iree_hal_pool_reservation_request_t, 2> requests;
  requests.fill({params_, 256});
  std::array<iree_hal_pool_reservation_t, 2> reservations;
  std::array<iree_hal_pool_acquire_info_t, 2> infos;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      child, requests.size(), requests.data(), nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, reservations.data(), infos.data(),
      &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  for (uint8_t i = 0; i < 2; ++i) {
    iree_async_single_frontier_t frontier;
    iree_async_single_frontier_initialize(&frontier, Axis(i), 7 + i);
    iree_hal_pool_release_reservations(
        child, 1, &reservations[i],
        iree_async_single_frontier_as_const_frontier(&frontier));
  }
  WaitForMaintenance();
  EXPECT_EQ(Stats(child).bytes_committed, 512u);
  EXPECT_EQ(Stats(source).reservation_count, 1u);
  // The union is wider than one entry, but both constituent blocks remain
  // reusable with their exact prerequisite. Neither becomes tainted or fresh.
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      child, requests.size(), requests.data(), nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH |
          IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
      reservations.data(), infos.data(), &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  uint32_t seen = 0;
  for (const auto& info : infos) {
    ASSERT_EQ(info.reuse_frontier->entry_count, 1);
    const auto entry = info.reuse_frontier->entries[0];
    const uint8_t index = entry.axis == Axis(0) ? 0 : 1;
    EXPECT_EQ(entry.axis, Axis(index));
    EXPECT_EQ(entry.epoch, 7u + index);
    seen |= 1u << index;
    iree_async_frontier_tracker_advance(tracker_, entry.axis, entry.epoch);
  }
  EXPECT_EQ(seen, 3u);
  iree_hal_pool_release_reservations(child, reservations.size(),
                                     reservations.data(), nullptr);
  WaitForMaintenance();
  EXPECT_EQ(Stats(child).bytes_committed, 0u);
  EXPECT_EQ(Stats(source).reservation_count, 0u);
}

INSTANTIATE_TEST_SUITE_P(
    Sources, FixedBlockPoolCompositionTest,
    ::testing::Combine(::testing::Values(AllocatorKind::kTLSF,
                                         AllocatorKind::kFixedBlock),
                       ::testing::Values(AllocatorKind::kFixedBlock)));

INSTANTIATE_TEST_SUITE_P(
    Policies, PoolCompositionTest,
    ::testing::Combine(
        ::testing::Values(AllocatorKind::kTLSF, AllocatorKind::kFixedBlock),
        ::testing::Values(AllocatorKind::kTLSF, AllocatorKind::kFixedBlock)));

}  // namespace
