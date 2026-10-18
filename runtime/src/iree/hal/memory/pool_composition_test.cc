// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <atomic>
#include <thread>
#include <tuple>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/fixed_block_pool.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/memory/slab_cache.h"
#include "iree/hal/memory/tlsf_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct MetadataAllocator {
  // Calls that would obtain or resize host storage.
  std::atomic<uint64_t> allocations{0};
  // Successfully allocated host objects that have not been freed.
  std::atomic<int64_t> live{0};
  // Rejects allocation while preserving infallible release.
  std::atomic<bool> fail{false};

  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** pointer) {
    auto* state = static_cast<MetadataAllocator*>(self);
    const bool had_storage = (command == IREE_ALLOCATOR_COMMAND_FREE ||
                              command == IREE_ALLOCATOR_COMMAND_REALLOC) &&
                             *pointer != nullptr;
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      ++state->allocations;
      if (state->fail) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "metadata allocation rejected");
      }
    }
    const auto system = iree_allocator_system();
    iree_status_t status = system.ctl(system.self, command, params, pointer);
    if (iree_status_is_ok(status)) {
      if (command == IREE_ALLOCATOR_COMMAND_FREE && had_storage) {
        --state->live;
      } else if (command != IREE_ALLOCATOR_COMMAND_FREE && !had_storage) {
        ++state->live;
      }
    }
    return status;
  }

  iree_allocator_t allocator() { return {this, Control}; }
};

enum class AllocatorKind { kTLSF, kFixedBlock };

class PoolCompositionTest : public ::testing::TestWithParam<
                                std::tuple<AllocatorKind, AllocatorKind>> {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), allocator_, &proactor_));
    IREE_ASSERT_OK(iree_async_notification_create(
        proactor_, IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    auto tracker_options = iree_async_frontier_tracker_options_default();
    tracker_options.axis_table_capacity = 260;
    IREE_ASSERT_OK(iree_async_frontier_tracker_create(
        tracker_options, progress_metadata_.allocator(), &tracker_));
    for (uint8_t i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(iree_async_frontier_tracker_register_axis(
          tracker_, Axis(i), nullptr));
    }
    IREE_ASSERT_OK(
        iree_hal_cpu_slab_provider_create(0, allocator_, &provider_));
    IREE_ASSERT_OK(iree_hal_passthrough_pool_create(
        {}, provider_, notification_, tracker_, allocator_, &native_pool_));
    IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
        native_pool_,
        iree_hal_make_pool_reservation_request(4096, kMinAlignment),
        iree_infinite_timeout(), &backing_));
  }

  void TearDown() override {
    for (auto i = pools_.rbegin(); i != pools_.rend(); ++i) {
      iree_hal_pool_release(*i);
    }
    iree_hal_buffer_release(backing_);
    iree_hal_pool_release(native_pool_);
    iree_hal_slab_provider_release(provider_);
    iree_async_frontier_tracker_release(tracker_);
    iree_async_notification_release(notification_);
    iree_async_proactor_release(proactor_);
    EXPECT_EQ(metadata_.live, 0);
    EXPECT_EQ(progress_metadata_.live, 0);
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
      iree_hal_fixed_block_pool_options_t options = {
          .block_size = block_size,
          .frontier_capacity = 2,
      };
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
                            iree_device_size_t block_size = 256,
                            iree_allocator_t host_allocator = {}) {
    if (!host_allocator.ctl) {
      host_allocator = allocator_;
    }
    iree_status_t status;
    if (std::get<1>(GetParam()) == AllocatorKind::kTLSF) {
      iree_hal_tlsf_pool_options_t options = {};
      options.tlsf_options.range_length = slab_size;
      options.tlsf_options.alignment = 16;
      options.tlsf_options.frontier_capacity = frontier_capacity;
      status =
          iree_hal_tlsf_pool_create(parent, &options, host_allocator, out_pool);
    } else {
      iree_hal_fixed_block_pool_options_t options = {
          .block_size = block_size,
      };
      options.blocks_per_slab = slab_size / block_size;
      options.alignment = 16;
      options.frontier_capacity = frontier_capacity;
      status = iree_hal_fixed_block_pool_create(parent, &options,
                                                host_allocator, out_pool);
    }
    if (iree_status_is_ok(status)) {
      pools_.push_back(*out_pool);
    }
    return status;
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
    const iree_hal_pool_reservation_request_t request =
        iree_hal_make_pool_reservation_request(length, kMinAlignment);
    return iree_hal_pool_acquire_reservations_sync(pool, 1, &request, frontier,
                                                   out_reservation, out_info,
                                                   out_result, flags);
  }

  // Writes and reads through the actual native mapping, returning its address
  // only as an identity for comparing two independently owned epochs.
  iree_status_t CheckBytes(iree_hal_pool_t* pool,
                           const iree_hal_pool_reservation_t& reservation,
                           uint8_t pattern, uintptr_t* out_address) {
    const iree_hal_pool_reservation_request_t request =
        iree_hal_make_pool_reservation_request(reservation.byte_length,
                                               kMinAlignment);
    iree_hal_buffer_t* buffer = nullptr;
    IREE_RETURN_IF_ERROR(iree_hal_pool_materialize_reservations(
        pool, 1, &request, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
        &buffer));
    EXPECT_EQ(
        iree_hal_buffer_allocation_placement(buffer).queue_family_affinity,
        iree_hal_buffer_allocation_placement(backing_).queue_family_affinity);
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
  // Optional accounting and allocation failures for child metadata only.
  MetadataAllocator metadata_;
  // Tracker-owned dependency captures outlive the originating child metadata.
  MetadataAllocator progress_metadata_;
  // Progress owner retained for the fixture's notifications.
  iree_async_proactor_t* proactor_ = nullptr;
  // Capacity notification captured by the native leaf.
  iree_async_notification_t* notification_ = nullptr;
  // Shared tracker with two exact queue axes.
  iree_async_frontier_tracker_t* tracker_ = nullptr;
  // Production CPU backing provider.
  iree_hal_slab_provider_t* provider_ = nullptr;
  // Production native leaf whose allocation count establishes physical reuse.
  iree_hal_pool_t* native_pool_ = nullptr;
  // One owned native epoch subdivided by the parent strategies.
  iree_hal_buffer_t* backing_ = nullptr;
  // Pools in creation order; teardown releases children before parents.
  std::vector<iree_hal_pool_t*> pools_;
  static constexpr iree_device_size_t kMinAlignment = 16;
};

TEST_P(PoolCompositionTest, SiblingReuseNeedsNoMetadataAllocation) {
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(0, 4096, 4096, &source));
  std::array<iree_hal_pool_t*, 2> children;
  for (auto& child : children) {
    IREE_ASSERT_OK(
        CreateChild(source, 4096, 2, &child, 64, metadata_.allocator()));
  }
  const int64_t initial_live = metadata_.live;
  std::array<iree_hal_pool_reservation_request_t, 32> requests;
  requests.fill(iree_hal_make_pool_reservation_request(64, 16));
  auto cycle = [&](iree_hal_pool_t* child, size_t count, uint8_t pattern) {
    std::array<iree_hal_pool_reservation_t, 32> reservations;
    std::array<iree_hal_pool_acquire_info_t, 32> infos;
    iree_hal_pool_acquire_result_t result;
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
        child, count, requests.data(), nullptr, reservations.data(),
        infos.data(), &result, IREE_HAL_POOL_RESERVE_FLAG_NONE));
    ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
    std::array<iree_hal_pool_reservation_view_t, 32> views;
    ASSERT_TRUE(iree_hal_pool_query_reservation_views(
        child, count, reservations.data(), views.data()));
    std::array<uint8_t, 64> bytes;
    for (size_t i = 0; i < count; ++i) {
      bytes.fill(pattern + i);
      const auto offset =
          views[i].byte_offset - iree_hal_buffer_byte_offset(views[i].buffer);
      IREE_ASSERT_OK(iree_hal_buffer_map_write(views[i].buffer, offset,
                                               bytes.data(), bytes.size()));
    }
    for (size_t i = 0; i < count; ++i) {
      bytes.fill(0);
      const auto offset =
          views[i].byte_offset - iree_hal_buffer_byte_offset(views[i].buffer);
      IREE_ASSERT_OK(iree_hal_buffer_map_read(views[i].buffer, offset,
                                              bytes.data(), bytes.size()));
      for (uint8_t value : bytes) {
        EXPECT_EQ(value, static_cast<uint8_t>(pattern + i));
      }
    }
    iree_hal_pool_release_reservations(child, count, reservations.data(),
                                       nullptr);
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(child, &stats);
    EXPECT_EQ(stats.bytes_committed, 0u);
  };
  for (auto* child : children) {
    for (size_t count : {1, 8, 9, 32}) {
      cycle(child, count, 0x51);
    }
  }
  const uint64_t allocations = metadata_.allocations;
  const uint64_t progress_allocations = progress_metadata_.allocations;
  metadata_.fail = true;
  progress_metadata_.fail = true;
  for (int i = 0; i < 32; ++i) {
    for (size_t count : {32, 9, 8, 1}) {
      cycle(children[i % 2], count, 0x80 + i);
    }
  }
  EXPECT_EQ(metadata_.allocations, allocations);
  EXPECT_EQ(progress_metadata_.allocations, progress_allocations);
  iree_hal_pool_stats_t native_stats;
  iree_hal_pool_query_stats(native_pool_, &native_stats);
  EXPECT_EQ(native_stats.reserve_count, 1u);

  for (auto* child : children) {
    iree_hal_pool_trim(child, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
  }
  EXPECT_EQ(metadata_.live, initial_live);
  metadata_.fail = false;
  progress_metadata_.fail = false;
  for (auto* child : children) {
    cycle(child, 32, 0x73);
  }
  EXPECT_GT(metadata_.allocations, allocations);
}

TEST_P(PoolCompositionTest, ReusedMetadataAdoptsSiblingHistory) {
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(0, 4096, 4096, &source));
  std::array<iree_hal_pool_t*, 2> children;
  for (auto& child : children) {
    IREE_ASSERT_OK(CreateChild(source, 4096, 2, &child, 512));
  }
  const auto request = iree_hal_make_pool_reservation_request(512, 16);
  for (uint64_t epoch = 0; epoch < 6; ++epoch) {
    auto* child = children[epoch % 2];
    iree_async_single_frontier_t frontier;
    iree_async_single_frontier_initialize(&frontier, Axis(0), epoch);
    const auto* history =
        iree_async_single_frontier_as_const_frontier(&frontier);
    iree_hal_pool_reservation_t reservation;
    iree_hal_pool_acquire_info_t info;
    iree_hal_pool_acquire_result_t result;
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
        child, 1, &request, history, &reservation, &info, &result,
        IREE_HAL_POOL_RESERVE_FLAG_NONE));
    if (epoch) {
      ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
      ASSERT_NE(info.reuse_frontier, nullptr);
      EXPECT_EQ(iree_async_frontier_compare(info.reuse_frontier, history),
                IREE_ASYNC_FRONTIER_EQUAL);
    } else {
      ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
    }
    iree_async_single_frontier_initialize(&frontier, Axis(0), epoch + 1);
    iree_hal_pool_release_reservations(child, 1, &reservation, history);
  }
  iree_async_frontier_tracker_advance(tracker_, Axis(0), 6);
}

TEST_P(PoolCompositionTest, MetadataTrimRacesAcquisitionAndReturn) {
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(0, 4096, 4096, &source));
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(
      CreateChild(source, 4096, 2, &pool, 512, metadata_.allocator()));
  auto exercise = [&](size_t i) {
    for (int j = 0; j < 64; ++j) {
      if (i == 0) {
        iree_hal_pool_trim(pool, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
      } else {
        iree_hal_buffer_t* buffer = nullptr;
        IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
            pool, iree_hal_make_pool_reservation_request(512, 16),
            iree_infinite_timeout(), &buffer));
        std::array<uint8_t, 512> bytes;
        const uint8_t pattern = static_cast<uint8_t>(i + j);
        bytes.fill(pattern);
        IREE_ASSERT_OK(
            iree_hal_buffer_map_write(buffer, 0, bytes.data(), bytes.size()));
        bytes.fill(0);
        IREE_ASSERT_OK(
            iree_hal_buffer_map_read(buffer, 0, bytes.data(), bytes.size()));
        for (uint8_t value : bytes) {
          EXPECT_EQ(value, pattern);
        }
        iree_hal_buffer_release(buffer);
      }
    }
  };
  std::atomic<uint32_t> completed{0};
  std::array<std::thread, 3> threads;
  for (size_t i = 0; i < threads.size(); ++i) {
    threads[i] = std::thread([&, i] {
      exercise(i);
      ++completed;
      iree_async_proactor_wake(proactor_);
    });
  }
  while (completed.load() != threads.size()) {
    IREE_ASSERT_OK(
        iree_async_proactor_poll(proactor_, iree_infinite_timeout(), nullptr));
  }
  for (auto& thread : threads) {
    thread.join();
  }
  iree_hal_pool_trim(pool, IREE_HAL_POOL_TRIM_FLAG_ALL, 0);
}

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
  IREE_ASSERT_OK(Acquire(child, 256, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &held[1], &info, &result));
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

TEST_P(PoolCompositionTest, WideInheritedHistorySurvivesChildDestruction) {
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
  iree_hal_pool_reservation_t reservation;
  IREE_ASSERT_OK(Acquire(child, 256, nullptr,
                         IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
                         &reservation, &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  ASSERT_NE(info.reuse_frontier, nullptr);
  EXPECT_EQ(info.reuse_frontier->entry_count, 1u);
  iree_hal_pool_release_reservations(child, 1, &reservation,
                                     info.reuse_frontier);
  EXPECT_EQ(Stats(child).bytes_committed, 0u);
  iree_hal_pool_release(child);
  pools_.pop_back();
  IREE_ASSERT_OK(Acquire(source, 4096, nullptr,
                         IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER, &seed,
                         &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  bool satisfied = true;
  IREE_ASSERT_OK(iree_async_frontier_tracker_query(
      tracker_, info.reuse_frontier, &satisfied));
  EXPECT_FALSE(satisfied);
  iree_async_frontier_tracker_advance(tracker_, Axis(0), 7);
  IREE_ASSERT_OK(iree_async_frontier_tracker_query(
      tracker_, info.reuse_frontier, &satisfied));
  EXPECT_FALSE(satisfied);
  iree_async_frontier_tracker_advance(tracker_, Axis(1), 11);
  IREE_ASSERT_OK(iree_async_frontier_tracker_query(
      tracker_, info.reuse_frontier, &satisfied));
  EXPECT_TRUE(satisfied);
  iree_hal_pool_release_reservations(source, 1, &seed, info.reuse_frontier);
}

TEST_P(PoolCompositionTest, FailedOverflowAcquisitionPreservesImmutableProofs) {
  IREE_ASSERT_OK(
      iree_async_frontier_tracker_register_axis(tracker_, Axis(2), nullptr));
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(0, 4096, 4096, &source));
  iree_hal_pool_t* child = nullptr;
  IREE_ASSERT_OK(CreateChild(source, 4096, 1, &child, 2048));
  std::array<iree_hal_pool_reservation_t, 2> reservations;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  for (auto& reservation : reservations) {
    IREE_ASSERT_OK(Acquire(child, 2048, nullptr,
                           IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &info,
                           &result));
    ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  }
  alignas(iree_async_frontier_entry_t)
      uint8_t storage[sizeof(iree_async_frontier_t) +
                      2 * sizeof(iree_async_frontier_entry_t)];
  auto* history = reinterpret_cast<iree_async_frontier_t*>(storage);
  iree_async_frontier_initialize(history, 2);
  history->entries[0] = {Axis(0), 7};
  history->entries[1] = {Axis(1), 11};
  iree_hal_pool_release_reservations(child, 1, &reservations[0], history);

  // Acquiring a partial backing must secure its next infallible return before
  // exposing the current proof. Failed preparation leaves the block reusable.
  iree_hal_pool_reservation_t unchanged;
  memset(&unchanged, 0xA5, sizeof(unchanged));
  reservations[0] = unchanged;
  progress_metadata_.fail = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        Acquire(child, 2048, nullptr,
                                IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
                                &reservations[0], &info, &result));
  progress_metadata_.fail = false;
  EXPECT_EQ(memcmp(&reservations[0], &unchanged, sizeof(unchanged)), 0);
  EXPECT_EQ(Stats(child).reservation_count, 1u);
  IREE_ASSERT_OK(Acquire(child, 2048, nullptr,
                         IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
                         &reservations[0], &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  ASSERT_EQ(info.reuse_frontier->entry_count, 1u);
  iree_async_single_frontier_t first;
  iree_async_single_frontier_initialize(&first,
                                        info.reuse_frontier->entries[0].axis,
                                        info.reuse_frontier->entries[0].epoch);

  // A later use includes the first proof but cannot add its dependency to
  // that already published point, even while the original uses are pending.
  history->entries[0] = {Axis(2), 13};
  history->entries[1] = info.reuse_frontier->entries[0];
  iree_hal_pool_release_reservations(child, 1, &reservations[0], history);
  IREE_ASSERT_OK(Acquire(child, 2048, nullptr,
                         IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
                         &reservations[0], &info, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  iree_async_frontier_tracker_advance(tracker_, Axis(0), 7);
  iree_async_frontier_tracker_advance(tracker_, Axis(1), 11);
  bool satisfied = false;
  IREE_ASSERT_OK(iree_async_frontier_tracker_query(
      tracker_, iree_async_single_frontier_as_const_frontier(&first),
      &satisfied));
  EXPECT_TRUE(satisfied);
  IREE_ASSERT_OK(iree_async_frontier_tracker_query(
      tracker_, info.reuse_frontier, &satisfied));
  EXPECT_FALSE(satisfied);
  iree_async_frontier_tracker_advance(tracker_, Axis(2), 13);
  IREE_ASSERT_OK(iree_async_frontier_tracker_query(
      tracker_, info.reuse_frontier, &satisfied));
  EXPECT_TRUE(satisfied);
  iree_hal_pool_release_reservations(child, reservations.size(),
                                     reservations.data(), nullptr);
  EXPECT_EQ(Stats(child).bytes_committed, 0u);
}

TEST_P(PoolCompositionTest,
       OverflowReturnSurvivesChildDestructionAndReusesExactBytes) {
  for (uint32_t i = 2; i < 256; ++i) {
    IREE_ASSERT_OK(iree_async_frontier_tracker_register_axis(
        tracker_, Axis(static_cast<uint8_t>(i)), nullptr));
  }
  uint64_t epoch = 0;
  for (uint32_t axis_count : {9u, 256u}) {
    SCOPED_TRACE(axis_count);
    ++epoch;
    iree_hal_pool_t* source = nullptr;
    IREE_ASSERT_OK(CreateSource(0, 4096, 4096, &source));
    const uint8_t capacity = axis_count == 9 ? 8 : 1;
    iree_hal_pool_t* child = nullptr;
    IREE_ASSERT_OK(CreateChild(source, 4096, capacity, &child, 2048,
                               metadata_.allocator()));
    std::array<iree_hal_pool_reservation_t, 2> reservations;
    iree_hal_pool_acquire_info_t info;
    iree_hal_pool_acquire_result_t result;
    std::vector<uint8_t> expected(4096);
    for (uint32_t i = 0; i < 2; ++i) {
      IREE_ASSERT_OK(Acquire(child, 2048, nullptr,
                             IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservations[i],
                             &info, &result));
      ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
      uintptr_t address;
      IREE_ASSERT_OK(
          CheckBytes(child, reservations[i], i ? 0xA5 : 0x5A, &address));
      memset(expected.data() + i * 2048, i ? 0xA5 : 0x5A, 2048);
    }
    alignas(iree_async_frontier_entry_t)
        uint8_t storage[sizeof(iree_async_frontier_t) +
                        255 * sizeof(iree_async_frontier_entry_t)];
    auto* first = reinterpret_cast<iree_async_frontier_t*>(storage);
    iree_async_frontier_initialize(first, static_cast<uint8_t>(axis_count - 1));
    for (uint32_t i = 0; i < axis_count - 1; ++i) {
      first->entries[i] = {Axis(static_cast<uint8_t>(i)), epoch};
    }
    iree_async_single_frontier_t last;
    iree_async_single_frontier_initialize(
        &last, Axis(static_cast<uint8_t>(axis_count - 1)), epoch);
    const auto child_allocations = metadata_.allocations.load();
    const auto progress_allocations = progress_metadata_.allocations.load();
    metadata_.fail = true;
    progress_metadata_.fail = true;
    iree_hal_pool_release_reservations(child, 1, &reservations[0], first);
    iree_hal_pool_release_reservations(
        child, 1, &reservations[1],
        iree_async_single_frontier_as_const_frontier(&last));
    EXPECT_EQ(Stats(child).bytes_committed, 0u);
    iree_hal_pool_release(child);
    pools_.pop_back();
    EXPECT_EQ(metadata_.live, 0);
    EXPECT_EQ(metadata_.allocations, child_allocations);
    EXPECT_EQ(progress_metadata_.allocations, progress_allocations);
    progress_metadata_.fail = false;
    metadata_.fail = false;

    iree_hal_pool_t* sibling = nullptr;
    IREE_ASSERT_OK(CreateChild(source, 4096, capacity, &sibling, 4096));
    iree_hal_pool_reservation_t reused;
    IREE_ASSERT_OK(Acquire(sibling, 4096, nullptr,
                           IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
                           &reused, &info, &result));
    ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
    ASSERT_NE(info.reuse_frontier, nullptr);
    EXPECT_EQ(info.reuse_frontier->entry_count, 1u);
    bool satisfied = true;
    for (uint32_t i = 0; i < axis_count - 1; ++i) {
      iree_async_frontier_tracker_advance(tracker_,
                                          Axis(static_cast<uint8_t>(i)), epoch);
    }
    IREE_ASSERT_OK(iree_async_frontier_tracker_query(
        tracker_, info.reuse_frontier, &satisfied));
    EXPECT_FALSE(satisfied);
    iree_async_frontier_tracker_advance(
        tracker_, Axis(static_cast<uint8_t>(axis_count - 1)), epoch);
    IREE_ASSERT_OK(iree_async_frontier_tracker_query(
        tracker_, info.reuse_frontier, &satisfied));
    EXPECT_TRUE(satisfied);
    const auto request = iree_hal_make_pool_reservation_request(4096, 16);
    iree_hal_buffer_t* view = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
        sibling, 1, &request, &reused, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
        &view));
    std::vector<uint8_t> actual(4096);
    IREE_ASSERT_OK(
        iree_hal_buffer_map_read(view, 0, actual.data(), actual.size()));
    EXPECT_EQ(actual, expected);
    iree_hal_buffer_release(view);
    iree_hal_pool_release_reservations(sibling, 1, &reused, nullptr);
    iree_hal_pool_release(sibling);
    pools_.pop_back();
    iree_hal_pool_release(source);
    pools_.pop_back();
  }
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
    iree_hal_pool_reservation_view_t view;
    ASSERT_TRUE(iree_hal_pool_query_reservation_views(children[i], 1,
                                                      &reservations[i], &view));
    EXPECT_EQ(view.buffer, backing_);
    EXPECT_EQ(view.byte_offset, 128 + i * 1024);
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
  iree_hal_pool_reservation_request_t request =
      iree_hal_make_pool_reservation_request(64, kMinAlignment);
  request.min_alignment = iree_min(4096, capabilities.max_allocation_alignment);
  iree_hal_pool_reservation_t reservation;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      child, 1, &request, nullptr, &reservation, &info, &result,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      child, 1, &request, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
      &buffer));
  const auto memory = iree_hal_buffer_memory_view(buffer);
  EXPECT_EQ(memory.offset % request.min_alignment, 0u);
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
  requests.fill(iree_hal_make_pool_reservation_request(256, kMinAlignment));
  std::array<iree_hal_pool_reservation_t, 7> reservations;
  memset(reservations.data(), 0xA5, sizeof(reservations));
  const auto unchanged = reservations;
  std::array<iree_hal_pool_acquire_info_t, 7> infos;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      child, requests.size(), requests.data(), nullptr, reservations.data(),
      infos.data(), &result, IREE_HAL_POOL_RESERVE_FLAG_NONE));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  EXPECT_EQ(memcmp(reservations.data(), unchanged.data(), sizeof(reservations)),
            0);
  EXPECT_EQ(Stats(child).reservation_count, 0u);
  EXPECT_EQ(Stats(child).bytes_committed, 0u);
  EXPECT_EQ(Stats(source).reservation_count, 0u);

  // The same capacity is available to a fitting batch after rollback. Each
  // parent extent contains two blocks, so this transaction spans three slabs.
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      child, 6, requests.data(), nullptr, reservations.data(), infos.data(),
      &result, IREE_HAL_POOL_RESERVE_FLAG_NONE));
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
            child, iree_hal_make_pool_reservation_request(256, kMinAlignment),
            iree_infinite_timeout(), &buffer));
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
  EXPECT_EQ(Stats(child).reservation_count, 0u);
  EXPECT_EQ(Stats(child).bytes_committed, 0u);
  EXPECT_EQ(Stats(source).reservation_count, 0u);
  EXPECT_EQ(Stats(native_pool_).reserve_count, 1u);
}

TEST_P(PoolCompositionTest, ReservationViewPreservesPreparedRangeAndHistory) {
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(0, 4096, 4096, &source));
  iree_hal_pool_t* child = nullptr;
  IREE_ASSERT_OK(CreateChild(source, 4096, 2, &child));

  iree_hal_pool_reservation_t anchor;
  iree_hal_pool_acquire_info_t anchor_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(Acquire(child, 64, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &anchor, &anchor_info, &result));
  ASSERT_TRUE(result == IREE_HAL_POOL_ACQUIRE_OK_FRESH ||
              result == IREE_HAL_POOL_ACQUIRE_OK);

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t info;
  IREE_ASSERT_OK(Acquire(child, 64, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
                         &reservation, &info, &result));
  ASSERT_TRUE(result == IREE_HAL_POOL_ACQUIRE_OK_FRESH ||
              result == IREE_HAL_POOL_ACQUIRE_OK);
  iree_async_single_frontier_t prior_use;
  iree_async_single_frontier_initialize(&prior_use, Axis(0), 7);
  const iree_async_frontier_t* prior_use_frontier =
      iree_async_single_frontier_as_const_frontier(&prior_use);
  iree_hal_pool_release_reservations(child, 1, &reservation,
                                     prior_use_frontier);

  IREE_ASSERT_OK(Acquire(child, 64, prior_use_frontier,
                         IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &info,
                         &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  ASSERT_NE(info.reuse_frontier, nullptr);
  EXPECT_EQ(
      iree_async_frontier_compare(info.reuse_frontier, prior_use_frontier),
      IREE_ASYNC_FRONTIER_EQUAL);

  iree_hal_pool_reservation_view_t view = {};
  ASSERT_TRUE(
      iree_hal_pool_query_reservation_views(child, 1, &reservation, &view));
  EXPECT_EQ(view.buffer, backing_);
  EXPECT_EQ(view.byte_length, reservation.byte_length);
  EXPECT_EQ(view.memory.reuse_frontier, info.reuse_frontier);

  const iree_hal_pool_reservation_request_t request =
      iree_hal_make_pool_reservation_request(64, kMinAlignment);
  iree_hal_buffer_t* materialized = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      child, 1, &request, &reservation, IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
      &materialized));
  const iree_hal_buffer_memory_view_t materialized_memory =
      iree_hal_buffer_memory_view(materialized);
  EXPECT_EQ(view.byte_offset, iree_hal_buffer_byte_offset(materialized));
  EXPECT_EQ(view.byte_length, iree_hal_buffer_byte_length(materialized));
  EXPECT_EQ(view.memory.contract, materialized_memory.contract);
  EXPECT_EQ(view.memory.bindings, materialized_memory.bindings);
  EXPECT_EQ(view.memory.binding_offset, materialized_memory.binding_offset);
  EXPECT_EQ(view.memory.backing, materialized_memory.backing);
  EXPECT_EQ(view.memory.offset, materialized_memory.offset);
  EXPECT_EQ(view.memory.reuse_frontier, materialized_memory.reuse_frontier);

  iree_hal_buffer_release(materialized);
  iree_hal_pool_release_reservations(child, 1, &reservation, nullptr);
  iree_hal_pool_release_reservations(child, 1, &anchor, nullptr);
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
  cache_options.slab =
      iree_hal_make_pool_reservation_request(1024, kMinAlignment);
  iree_hal_pool_capabilities_t source_capabilities;
  iree_hal_pool_query_capabilities(source, &source_capabilities);
  cache_options.slab.min_alignment =
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
  requests.fill(iree_hal_make_pool_reservation_request(192, kMinAlignment));
  std::array<iree_hal_pool_reservation_t, 5> reservations;
  std::array<iree_hal_pool_acquire_info_t, 5> infos;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      child, requests.size(), requests.data(), previous_frontier,
      reservations.data(), infos.data(), &result,
      IREE_HAL_POOL_RESERVE_FLAG_NONE));
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
       LiveBackingRetainsIndividualBlockHistories) {
  iree_hal_pool_t* source = nullptr;
  IREE_ASSERT_OK(CreateSource(0, 768, 768, &source));
  iree_hal_pool_t* child = nullptr;
  IREE_ASSERT_OK(CreateChild(source, 768, 1, &child));
  std::array<iree_hal_pool_reservation_request_t, 3> requests;
  requests.fill(iree_hal_make_pool_reservation_request(256, kMinAlignment));
  std::array<iree_hal_pool_reservation_t, 3> reservations;
  std::array<iree_hal_pool_acquire_info_t, 3> infos;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      child, requests.size(), requests.data(), nullptr, reservations.data(),
      infos.data(), &result, IREE_HAL_POOL_RESERVE_FLAG_NONE));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  for (uint8_t i = 0; i < 2; ++i) {
    iree_async_single_frontier_t frontier;
    iree_async_single_frontier_initialize(&frontier, Axis(i), 7 + i);
    iree_hal_pool_release_reservations(
        child, 1, &reservations[i],
        iree_async_single_frontier_as_const_frontier(&frontier));
  }
  EXPECT_EQ(Stats(child).bytes_committed, 768u);
  EXPECT_EQ(Stats(source).reservation_count, 1u);
  // The third block keeps this backing live. The two free blocks retain their
  // individual histories until a whole-backing return requires their union.
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations_sync(
      child, 2, requests.data(), nullptr, reservations.data(), infos.data(),
      &result, IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  uint32_t seen = 0;
  for (size_t i = 0; i < 2; ++i) {
    const auto& info = infos[i];
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
