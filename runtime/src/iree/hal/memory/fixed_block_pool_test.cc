// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/memory/fixed_block_pool.h"

#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/async/notification.h"
#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/api.h"
#include "iree/hal/memory/cpu_slab_provider.h"
#include "iree/hal/memory/maintenance_thread.h"
#include "iree/hal/memory/passthrough_pool.h"
#include "iree/hal/pool_set.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static iree_async_frontier_tracker_t* test_frontier_tracker() {
  static iree_async_frontier_tracker_t* tracker = nullptr;
  if (!tracker) {
    IREE_CHECK_OK(iree_async_frontier_tracker_create(
        iree_async_frontier_tracker_options_default(), iree_allocator_system(),
        &tracker));
    for (uint8_t queue_index = 0; queue_index < 6; ++queue_index) {
      IREE_CHECK_OK(iree_async_frontier_tracker_register_axis(
          tracker, iree_async_axis_make_queue(1, 0, 0, queue_index, 0),
          nullptr));
    }
    atexit([] {
      iree_async_frontier_tracker_release(tracker);
      tracker = nullptr;
    });
  }
  return tracker;
}

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

static iree_async_axis_t TestQueueAxis(uint8_t queue_index) {
  return iree_async_axis_make_queue(1, 0, 0, queue_index, 0);
}

static iree_async_frontier_entry_t E(iree_async_axis_t axis, uint64_t epoch) {
  return {axis, epoch};
}

static iree_async_frontier_t* BuildFrontier(
    uint8_t* storage, iree_host_size_t storage_size,
    std::initializer_list<iree_async_frontier_entry_t> entries) {
  iree_async_frontier_t* frontier =
      reinterpret_cast<iree_async_frontier_t*>(storage);
  iree_async_frontier_initialize(frontier,
                                 static_cast<uint8_t>(entries.size()));
  uint8_t i = 0;
  for (const auto& entry : entries) {
    frontier->entries[i++] = entry;
  }
  return frontier;
}

#define MAKE_FRONTIER(name, capacity, ...)                              \
  alignas(16) uint8_t                                                   \
      name##_storage[sizeof(iree_async_frontier_t) +                    \
                     (capacity) * sizeof(iree_async_frontier_entry_t)]; \
  iree_async_frontier_t* name =                                         \
      BuildFrontier(name##_storage, sizeof(name##_storage), {__VA_ARGS__})

typedef struct iree_hal_test_opaque_slab_provider_t {
  // Base slab provider interface.
  iree_hal_slab_provider_t base;

  // Host allocator used for provider metadata and slab allocations.
  iree_allocator_t host_allocator;

  // Number of trim calls received by the provider.
  uint32_t trim_count;

  // Retention policy from the most recent trim request.
  iree_hal_pool_trim_flags_t last_trim_flags;

  // Number of wrap_buffer calls received by the provider.
  iree_atomic_int32_t wrap_count;

  // Number of ASAN advice calls received by the provider.
  iree_atomic_int32_t asan_advice_count;

  // Number of allocated ASAN advice calls received by the provider.
  iree_atomic_int32_t asan_allocated_count;

  // Number of released ASAN advice calls received by the provider.
  iree_atomic_int32_t asan_released_count;

  // Last ASAN backing offset received by the provider.
  iree_device_size_t last_asan_backing_offset;

  // Last ASAN layout received by the provider.
  iree_hal_asan_allocation_layout_t last_asan_layout;

  // Sequence number of the last allocated ASAN advice call.
  int32_t last_asan_allocated_sequence;

  // Sequence number of the last released ASAN advice call.
  int32_t last_asan_released_sequence;
} iree_hal_test_opaque_slab_provider_t;

extern const iree_hal_slab_provider_vtable_t
    iree_hal_test_opaque_slab_provider_vtable;

static iree_status_t iree_hal_test_opaque_slab_provider_create(
    iree_allocator_t host_allocator, iree_hal_slab_provider_t** out_provider) {
  *out_provider = NULL;
  iree_hal_test_opaque_slab_provider_t* provider = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*provider),
                                             (void**)&provider));
  memset(provider, 0, sizeof(*provider));
  iree_hal_slab_provider_initialize(&iree_hal_test_opaque_slab_provider_vtable,
                                    &provider->base);
  provider->host_allocator = host_allocator;
  *out_provider = &provider->base;
  return iree_ok_status();
}

static void iree_hal_test_opaque_slab_provider_destroy(
    iree_hal_slab_provider_t* base_provider) {
  iree_hal_test_opaque_slab_provider_t* provider =
      (iree_hal_test_opaque_slab_provider_t*)base_provider;
  iree_allocator_free(provider->host_allocator, provider);
}

static iree_status_t iree_hal_test_opaque_slab_provider_acquire_slab(
    iree_hal_slab_provider_t* base_provider, iree_device_size_t min_length,
    iree_device_size_t alignment, iree_hal_slab_t* out_slab) {
  (void)alignment;
  iree_hal_test_opaque_slab_provider_t* provider =
      (iree_hal_test_opaque_slab_provider_t*)base_provider;
  memset(out_slab, 0, sizeof(*out_slab));
  void* backing = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
      provider->host_allocator, min_length, IREE_HAL_HEAP_BUFFER_ALIGNMENT,
      /*offset=*/0, &backing));
  out_slab->base_ptr = (uint8_t*)(uintptr_t)1;
  out_slab->length = min_length;
  out_slab->provider_handle = (uint64_t)(uintptr_t)backing;
  return iree_ok_status();
}

static void iree_hal_test_opaque_slab_provider_release_slab(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab) {
  iree_hal_test_opaque_slab_provider_t* provider =
      (iree_hal_test_opaque_slab_provider_t*)base_provider;
  iree_allocator_free_aligned(provider->host_allocator,
                              (void*)(uintptr_t)slab->provider_handle);
}

static iree_status_t iree_hal_test_opaque_slab_provider_wrap_buffer(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t slab_offset, iree_device_size_t allocation_size,
    iree_hal_buffer_params_t params,
    iree_hal_buffer_release_callback_t release_callback,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_test_opaque_slab_provider_t* provider =
      (iree_hal_test_opaque_slab_provider_t*)base_provider;
  iree_atomic_fetch_add(&provider->wrap_count, 1, iree_memory_order_relaxed);
  iree_byte_span_t data = iree_make_byte_span(
      (uint8_t*)(uintptr_t)slab->provider_handle + slab_offset,
      (iree_host_size_t)allocation_size);
  return iree_hal_heap_buffer_wrap(iree_hal_buffer_placement_undefined(),
                                   params.type, params.access, params.usage,
                                   allocation_size, data, release_callback,
                                   provider->host_allocator, out_buffer);
}

static iree_status_t iree_hal_test_opaque_slab_provider_validate_asan_options(
    const iree_hal_slab_provider_t* base_provider,
    const iree_hal_asan_pool_options_t* options) {
  (void)base_provider;
  (void)options;
  return iree_ok_status();
}

static void iree_hal_test_opaque_slab_provider_advise_asan_range(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t backing_offset,
    iree_hal_asan_range_advice_flags_t advice_flags,
    const iree_hal_asan_allocation_layout_t* layout) {
  iree_hal_test_opaque_slab_provider_t* provider =
      (iree_hal_test_opaque_slab_provider_t*)base_provider;
  (void)slab;
  provider->last_asan_backing_offset = backing_offset;
  provider->last_asan_layout = *layout;
  const int32_t advice_sequence =
      iree_atomic_fetch_add(&provider->asan_advice_count, 1,
                            iree_memory_order_relaxed) +
      1;
  if (advice_flags == IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED) {
    provider->last_asan_allocated_sequence = advice_sequence;
    iree_atomic_fetch_add(&provider->asan_allocated_count, 1,
                          iree_memory_order_relaxed);
  } else if (advice_flags == IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED) {
    provider->last_asan_released_sequence = advice_sequence;
    iree_atomic_fetch_add(&provider->asan_released_count, 1,
                          iree_memory_order_relaxed);
  } else {
    IREE_ASSERT(false, "unsupported ASAN range advice flags 0x%x",
                advice_flags);
  }
}

static void iree_hal_test_opaque_slab_provider_prefault(
    iree_hal_slab_provider_t* base_provider, const iree_hal_slab_t* slab,
    iree_device_size_t offset, iree_device_size_t length) {}

static void iree_hal_test_opaque_slab_provider_trim(
    iree_hal_slab_provider_t* base_provider, iree_hal_pool_trim_flags_t flags) {
  auto* provider = (iree_hal_test_opaque_slab_provider_t*)base_provider;
  ++provider->trim_count;
  provider->last_trim_flags = flags;
}

static void iree_hal_test_opaque_slab_provider_query_stats(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_visited_set_t* visited,
    iree_hal_slab_provider_stats_t* out_stats) {
  iree_hal_slab_provider_visited(visited, base_provider);
}

static void iree_hal_test_opaque_slab_provider_query_properties(
    const iree_hal_slab_provider_t* base_provider,
    iree_hal_slab_provider_properties_t* out_properties) {
  out_properties->allocation_alignment = IREE_HAL_HEAP_BUFFER_ALIGNMENT;
  out_properties->max_allocation_alignment = IREE_HAL_HEAP_BUFFER_ALIGNMENT;
  out_properties->memory_type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE |
      IREE_HAL_MEMORY_TYPE_HOST_COHERENT | IREE_HAL_MEMORY_TYPE_HOST_CACHED;
  out_properties->supported_usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                                    IREE_HAL_BUFFER_USAGE_STORAGE |
                                    IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED |
                                    IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;
  out_properties->queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY;
  out_properties->atomic_operations.device_scope_32 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_STORE;
  out_properties->atomic_operations.system_scope_64 =
      IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_OR;
}

const iree_hal_slab_provider_vtable_t
    iree_hal_test_opaque_slab_provider_vtable = {
        .destroy = iree_hal_test_opaque_slab_provider_destroy,
        .acquire_slab = iree_hal_test_opaque_slab_provider_acquire_slab,
        .release_slab = iree_hal_test_opaque_slab_provider_release_slab,
        .wrap_buffer = iree_hal_test_opaque_slab_provider_wrap_buffer,
        .validate_asan_options =
            iree_hal_test_opaque_slab_provider_validate_asan_options,
        .advise_asan_range =
            iree_hal_test_opaque_slab_provider_advise_asan_range,
        .prefault = iree_hal_test_opaque_slab_provider_prefault,
        .trim = iree_hal_test_opaque_slab_provider_trim,
        .query_stats = iree_hal_test_opaque_slab_provider_query_stats,
        .query_properties = iree_hal_test_opaque_slab_provider_query_properties,
};

static iree_hal_pool_reservation_request_t MakeReservationRequest(
    iree_device_size_t allocation_size, iree_device_size_t min_alignment) {
  iree_hal_pool_reservation_request_t request = {};
  request.params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  request.params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  request.params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  request.params.min_alignment = min_alignment;
  request.allocation_size = allocation_size;
  return request;
}

static iree_status_t AcquireOneReservation(
    iree_hal_pool_t* pool, iree_device_size_t allocation_size,
    iree_device_size_t min_alignment,
    const iree_async_frontier_t* requester_frontier,
    iree_hal_pool_reserve_flags_t flags,
    iree_hal_pool_reservation_t* out_reservation,
    iree_hal_pool_acquire_info_t* out_info,
    iree_hal_pool_acquire_result_t* out_result) {
  const iree_hal_pool_reservation_request_t request =
      MakeReservationRequest(allocation_size, min_alignment);
  return iree_hal_pool_acquire_reservations(
      pool, 1, &request, requester_frontier, flags, out_reservation, out_info,
      out_result);
}

static void ReleaseOneReservation(
    iree_hal_pool_t* pool, const iree_hal_pool_reservation_t* reservation,
    const iree_async_frontier_t* death_frontier) {
  iree_hal_pool_release_reservations(pool, 1, reservation, death_frontier);
}

static iree_status_t MaterializeOneReservation(
    iree_hal_pool_t* pool, iree_hal_buffer_params_t params,
    const iree_hal_pool_reservation_t* reservation,
    iree_hal_pool_materialize_flags_t flags, iree_hal_buffer_t** out_buffer) {
  const iree_hal_pool_reservation_request_t request = {
      .params = params,
      .allocation_size = reservation->byte_length,
  };
  return iree_hal_pool_materialize_reservations(pool, 1, &request, reservation,
                                                flags, out_buffer);
}

static iree_hal_memory_maintenance_t* test_maintenance() {
  static iree_hal_memory_maintenance_t* owner = nullptr;
  if (!owner) {
    IREE_CHECK_OK(iree_hal_memory_maintenance_thread_create(
        {}, iree_allocator_system(), &owner));
    atexit([] { iree_hal_memory_maintenance_release(owner); });
  }
  return owner;
}

static void WaitForMaintenance() {
  iree_hal_memory_maintenance_call(
      test_maintenance(),
      [](void* user_data) {
        auto* owner = static_cast<iree_hal_memory_maintenance_t*>(user_data);
        while (iree_hal_memory_maintenance_run_one(owner)) {
        }
      },
      test_maintenance());
}

static iree_status_t CreateFinitePool(
    iree_hal_fixed_block_pool_options_t options,
    iree_hal_slab_provider_t* provider, iree_async_notification_t* notification,
    iree_async_frontier_tracker_t* tracker, iree_allocator_t allocator,
    iree_hal_pool_t** out_backing_pool, iree_hal_pool_t** out_pool) {
  *out_backing_pool = nullptr;
  *out_pool = nullptr;
  iree_hal_pool_t* backing_pool = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_passthrough_pool_create(
      {}, provider, notification, tracker, test_maintenance(), allocator,
      &backing_pool));
  iree_hal_pool_reservation_request_t request;
  iree_status_t status = iree_hal_fixed_block_pool_query_backing_request(
      backing_pool, &options, &request);
  iree_hal_buffer_t* buffer = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_pool_allocate_buffer(backing_pool, request.params,
                                           request.allocation_size,
                                           iree_infinite_timeout(), &buffer);
  }
  if (iree_status_is_ok(status)) {
    options.blocks_per_slab = 0;
    status = iree_hal_fixed_block_pool_create_from_buffer(
        buffer, 0, IREE_HAL_WHOLE_BUFFER, &options, allocator, out_pool);
  }
  iree_hal_buffer_release(buffer);
  if (iree_status_is_ok(status)) {
    *out_backing_pool = backing_pool;
  } else {
    iree_hal_pool_release(backing_pool);
  }
  return status;
}

static iree_status_t CreateGrowingPool(
    const iree_hal_fixed_block_pool_options_t& options,
    iree_hal_slab_provider_t* provider, iree_async_notification_t* notification,
    iree_async_frontier_tracker_t* tracker, iree_hal_pool_epoch_query_t query,
    iree_allocator_t allocator, iree_hal_pool_t** out_pool) {
  iree_hal_passthrough_pool_options_t source_options = {.epoch_query = query};
  iree_hal_pool_t* source = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_passthrough_pool_create(
      source_options, provider, notification, tracker, test_maintenance(),
      allocator, &source));
  iree_status_t status =
      iree_hal_fixed_block_pool_create(source, &options, allocator, out_pool);
  iree_hal_pool_release(source);
  return status;
}

static iree_hal_fixed_block_pool_options_t DefaultOptions() {
  iree_hal_fixed_block_pool_options_t options = {
      .block_size = 256, .blocks_per_slab = 4, .frontier_capacity = 2};
  return options;
}

static iree_hal_asan_pool_options_t ShadowOptions() {
  iree_hal_asan_pool_options_t options = {};
  options.mode = IREE_HAL_ASAN_POOL_MODE_SHADOW;
  options.shadow_granule_size = 8;
  options.redzone_size = 16;
  options.backing_alignment = IREE_HAL_HEAP_BUFFER_ALIGNMENT;
  return options;
}

class FixedBlockPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
        /*min_alignment=*/0, allocator_, &slab_provider_));
    IREE_ASSERT_OK(iree_async_notification_create(
        test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification_));
    IREE_ASSERT_OK(CreateFinitePool(DefaultOptions(), slab_provider_,
                                    notification_, test_frontier_tracker(),
                                    allocator_, &backing_pool_, &pool_));
  }

  void TearDown() override {
    iree_hal_pool_release(pool_);
    iree_hal_pool_release(backing_pool_);
    backing_pool_ = nullptr;
    iree_async_notification_release(notification_);
    iree_hal_slab_provider_release(slab_provider_);
  }

  // Host allocator for fixture and pool metadata.
  iree_allocator_t allocator_ = iree_allocator_system();
  // Real CPU storage provider retained through native pool destruction.
  iree_hal_slab_provider_t* slab_provider_ = nullptr;
  // Capacity observation shared by the native owner and child.
  iree_async_notification_t* notification_ = nullptr;
  // Native owner of a finite arena; growable pools retain their source
  // directly.
  iree_hal_pool_t* backing_pool_ = nullptr;
  // Allocation policy under test, destroyed before its native owner.
  iree_hal_pool_t* pool_ = nullptr;
};

// Counts calls to the real allocator used by the pool and its native backing.
class CountingAllocator {
 public:
  iree_allocator_t allocator() { return {this, Control}; }
  size_t allocation_calls() const { return allocation_calls_; }

 private:
  static iree_status_t Control(void* self, iree_allocator_command_t command,
                               const void* params, void** inout_ptr) {
    auto* allocator = static_cast<CountingAllocator*>(self);
    if (command != IREE_ALLOCATOR_COMMAND_FREE) {
      ++allocator->allocation_calls_;
    }
    iree_allocator_t system_allocator = iree_allocator_system();
    return system_allocator.ctl(system_allocator.self, command, params,
                                inout_ptr);
  }

  // Number of allocation-like calls, including unsuccessful attempts.
  size_t allocation_calls_ = 0;
};

TEST(FixedBlockPool, GrowableProbesAndBudgetAdmissionDoNotAllocate) {
  CountingAllocator allocator;
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(
      iree_hal_cpu_slab_provider_create(0, allocator.allocator(), &provider));
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  auto options = DefaultOptions();
  options.blocks_per_slab = 2;
  options.budget_limit = 512;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreateGrowingPool(
      options, provider, notification, test_frontier_tracker(),
      iree_hal_pool_epoch_query_null(), allocator.allocator(), &pool));
  iree_hal_pool_reservation_request_t requests[3];
  for (auto& request : requests) {
    request = MakeReservationRequest(256, 16);
  }
  iree_hal_pool_reservation_t reservations[3];
  iree_hal_pool_acquire_info_t infos[3];
  iree_hal_pool_acquire_result_t result;
  const size_t initial_calls = allocator.allocation_calls();
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, 1, requests, nullptr, IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH,
      reservations, infos, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  EXPECT_EQ(infos[0].flags, IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED);
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, 3, requests, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE, reservations,
      infos, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OVER_BUDGET);
  EXPECT_EQ(allocator.allocation_calls(), initial_calls);

  // Cold preparation acquires one source range. Its live first block keeps
  // that range available for an allocation-free probe of the second block.
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, 1, requests, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE, reservations,
      infos, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  const size_t prepared_calls = allocator.allocation_calls();
  EXPECT_GT(prepared_calls, initial_calls);
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, 1, requests, nullptr, IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH,
      &reservations[1], infos, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(allocator.allocation_calls(), prepared_calls);
  iree_hal_pool_release_reservations(pool, 2, reservations, nullptr);
  WaitForMaintenance();
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.bytes_committed, 0u);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.over_budget_count, 1u);
  iree_hal_pool_release(pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(provider);
}

TEST(FixedBlockPool, NoGrowthDefersLargeBatchStaging) {
  CountingAllocator allocator;
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(
      /*min_alignment=*/0, allocator.allocator(), &provider));
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  constexpr size_t kCount = 12;
  auto options = DefaultOptions();
  options.blocks_per_slab = kCount;
  iree_hal_pool_t* backing_pool = nullptr;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(CreateFinitePool(options, provider, notification,
                                  test_frontier_tracker(),
                                  allocator.allocator(), &backing_pool, &pool));

  iree_hal_pool_reservation_request_t requests[kCount];
  for (auto& request : requests) {
    request = MakeReservationRequest(256, 16);
    request.params.usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  }
  iree_hal_pool_reservation_t reservations[kCount];
  memset(reservations, 0xA5, sizeof(reservations));
  iree_hal_pool_reservation_t originals[kCount];
  memcpy(originals, reservations, sizeof(originals));
  iree_hal_pool_acquire_info_t infos[kCount];
  iree_hal_pool_acquire_result_t result;
  const size_t allocation_calls = allocator.allocation_calls();
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, kCount, requests, nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH, reservations, infos,
      &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  EXPECT_EQ(allocator.allocation_calls(), allocation_calls);
  EXPECT_EQ(memcmp(reservations, originals, sizeof(originals)), 0);
  for (const auto& info : infos) {
    EXPECT_EQ(info.result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
    EXPECT_EQ(info.flags, IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED);
    EXPECT_EQ(info.reuse_frontier, nullptr);
  }

  // An inline-sized batch uses the same backing with no preparation.
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, 1, requests, nullptr, IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH,
      reservations, infos, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(allocator.allocation_calls(), allocation_calls);
  iree_hal_pool_release_reservations(pool, 1, reservations, nullptr);

  // The ordinary cold retry stages and acquires the complete batch.
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool, kCount, requests, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      reservations, infos, &result));
  ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(allocator.allocation_calls() - allocation_calls, 1u);
  for (size_t i = 0; i < kCount; ++i) {
    iree_hal_buffer_t* buffer = nullptr;
    IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
        pool, 1, &requests[i], &reservations[i],
        IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &buffer));
    const uint8_t pattern = static_cast<uint8_t>(i + 1);
    IREE_EXPECT_OK(iree_hal_buffer_map_fill(buffer, 0, 256, &pattern, 1));
    uint8_t actual[256] = {};
    IREE_EXPECT_OK(iree_hal_buffer_map_read(buffer, 0, actual, sizeof(actual)));
    for (uint8_t value : actual) {
      EXPECT_EQ(value, pattern);
    }
    iree_hal_buffer_release(buffer);
  }
  iree_hal_pool_release_reservations(pool, kCount, reservations, nullptr);
  iree_hal_pool_release(pool);
  iree_hal_pool_release(backing_pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(provider);
}

TEST(FixedBlockPool, SixQueueHistorySurvivesInlineAndColdStaging) {
  CountingAllocator allocator;
  iree_hal_slab_provider_t* provider = nullptr;
  IREE_ASSERT_OK(
      iree_hal_cpu_slab_provider_create(0, allocator.allocator(), &provider));
  iree_async_notification_t* notification = nullptr;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));
  MAKE_FRONTIER(death, 6, E(TestQueueAxis(0), 10), E(TestQueueAxis(1), 20),
                E(TestQueueAxis(2), 30), E(TestQueueAxis(3), 40),
                E(TestQueueAxis(4), 50), E(TestQueueAxis(5), 60));
  for (uint16_t capacity : {6, 16}) {
    SCOPED_TRACE(capacity);
    auto options = DefaultOptions();
    options.blocks_per_slab = 2;
    options.frontier_capacity = capacity;
    iree_hal_pool_t* backing_pool = nullptr;
    iree_hal_pool_t* pool = nullptr;
    IREE_ASSERT_OK(CreateFinitePool(
        options, provider, notification, test_frontier_tracker(),
        allocator.allocator(), &backing_pool, &pool));
    auto request = MakeReservationRequest(64, 16);
    request.params.usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
    const iree_hal_pool_reservation_request_t requests[2] = {request, request};
    iree_hal_pool_reservation_t reservations[2];
    iree_hal_pool_acquire_info_t infos[2];
    iree_hal_pool_acquire_result_t result;
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
        pool, 2, requests, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
        reservations, infos, &result));
    ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
    iree_hal_pool_release_reservations(pool, 2, reservations, death);

    const size_t allocation_calls = allocator.allocation_calls();
    IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
        pool, 2, requests, death, IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH,
        reservations, infos, &result));
    EXPECT_EQ(allocator.allocation_calls(), allocation_calls);
    if (capacity == 16) {
      EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
      EXPECT_EQ(infos[0].flags, IREE_HAL_POOL_ACQUIRE_FLAG_GROWTH_REQUIRED);
      IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
          pool, 2, requests, death, IREE_HAL_POOL_RESERVE_FLAG_NONE,
          reservations, infos, &result));
      EXPECT_EQ(allocator.allocation_calls(), allocation_calls + 1);
    }
    ASSERT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
    for (iree_host_size_t i = 0; i < 2; ++i) {
      ASSERT_NE(infos[i].reuse_frontier, nullptr);
      EXPECT_EQ(infos[i].reuse_frontier->entry_count, 6u);
      EXPECT_EQ(memcmp(infos[i].reuse_frontier->entries, death->entries,
                       6 * sizeof(death->entries[0])),
                0);
      iree_hal_buffer_t* buffer = nullptr;
      IREE_ASSERT_OK(MaterializeOneReservation(
          pool, request.params, &reservations[i],
          IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &buffer));
      const uint8_t pattern = static_cast<uint8_t>(capacity + i);
      IREE_ASSERT_OK(iree_hal_buffer_map_fill(buffer, 0, 64, &pattern, 1));
      uint8_t actual[64] = {};
      IREE_ASSERT_OK(
          iree_hal_buffer_map_read(buffer, 0, actual, sizeof(actual)));
      for (uint8_t value : actual) {
        EXPECT_EQ(value, pattern);
      }
      iree_hal_buffer_release(buffer);
    }
    iree_hal_pool_release_reservations(pool, 2, reservations, death);
    iree_hal_pool_release(pool);
    iree_hal_pool_release(backing_pool);
  }
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(provider);
}

TEST_F(FixedBlockPoolTest, ReserveReleaseFresh) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 128, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));

  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(reservation.offset, 0u);
  EXPECT_EQ(reservation.byte_length, 128u);
  EXPECT_NE(reservation.block_handle, 0u);
  EXPECT_EQ(reserve_info.reuse_frontier, nullptr);
  EXPECT_EQ(reserve_info.flags, IREE_HAL_POOL_ACQUIRE_FLAG_NONE);

  ReleaseOneReservation(pool_, &reservation, NULL);
}

TEST_F(FixedBlockPoolTest, ReservationTransactionIsAllOrNone) {
  const uint32_t wait_token =
      iree_async_notification_begin_observe(iree_hal_pool_notification(pool_));
  const iree_hal_pool_reservation_request_t oversized_requests[5] = {
      MakeReservationRequest(64, 16), MakeReservationRequest(64, 16),
      MakeReservationRequest(64, 16), MakeReservationRequest(64, 16),
      MakeReservationRequest(64, 16),
  };
  iree_hal_pool_reservation_t failed_reservations[5];
  memset(failed_reservations, 0xA5, sizeof(failed_reservations));
  iree_hal_pool_reservation_t original_reservations[5];
  memcpy(original_reservations, failed_reservations,
         sizeof(original_reservations));
  iree_hal_pool_acquire_info_t failed_infos[5];
  iree_hal_pool_acquire_result_t result = IREE_HAL_POOL_ACQUIRE_NONE;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, IREE_ARRAYSIZE(oversized_requests), oversized_requests,
      /*requester_frontier=*/NULL, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      failed_reservations, failed_infos, &result));

  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);
  EXPECT_EQ(memcmp(failed_reservations, original_reservations,
                   sizeof(failed_reservations)),
            0);
  for (iree_host_size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(failed_infos[i].result, IREE_HAL_POOL_ACQUIRE_NONE);
  }
  EXPECT_EQ(failed_infos[4].result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.reserve_count, 0u);
  EXPECT_EQ(stats.release_count, 0u);
  EXPECT_FALSE(iree_async_notification_wait_for_token(
      iree_hal_pool_notification(pool_), wait_token, iree_make_timeout_ms(0)));
  iree_async_notification_end_observe(iree_hal_pool_notification(pool_));

  const iree_hal_pool_reservation_request_t fitting_requests[4] = {
      MakeReservationRequest(64, 16),
      MakeReservationRequest(64, 16),
      MakeReservationRequest(64, 16),
      MakeReservationRequest(64, 16),
  };
  iree_hal_pool_reservation_t reservations[4];
  iree_hal_pool_acquire_info_t infos[4];
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, IREE_ARRAYSIZE(fitting_requests), fitting_requests,
      /*requester_frontier=*/NULL, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      reservations, infos, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  for (const auto& info : infos) {
    EXPECT_EQ(info.result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  }

  iree_hal_pool_release_reservations(pool_, IREE_ARRAYSIZE(reservations),
                                     reservations,
                                     /*death_frontier=*/NULL);
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.reserve_count, 4u);
  EXPECT_EQ(stats.release_count, 4u);
}

TEST_F(FixedBlockPoolTest, BatchChecksEachAvailableFrontierOnce) {
  iree_hal_pool_release(pool_);
  iree_hal_pool_release(backing_pool_);
  backing_pool_ = nullptr;
  auto options = DefaultOptions();
  constexpr uint32_t kBlockCount = 4095;
  constexpr iree_host_size_t kRequestCount = 8;
  options.blocks_per_slab = kBlockCount + 1;
  iree_host_size_t query_count = 0;
  const iree_hal_pool_epoch_query_t query = {
      [](void* user_data, iree_async_axis_t axis, uint64_t epoch) {
        ++*static_cast<iree_host_size_t*>(user_data);
        return false;
      },
      &query_count,
  };
  IREE_ASSERT_OK(CreateGrowingPool(options, slab_provider_, notification_,
                                   test_frontier_tracker(), query, allocator_,
                                   &pool_));
  const auto request = MakeReservationRequest(64, 16);
  std::vector<iree_hal_pool_reservation_request_t> requests(kBlockCount,
                                                            request);
  std::vector<iree_hal_pool_reservation_t> reservations(kBlockCount);
  std::vector<iree_hal_pool_acquire_info_t> infos(kBlockCount);
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, kBlockCount, requests.data(), nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, reservations.data(), infos.data(),
      &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_pool_reservation_t anchor;
  iree_hal_pool_acquire_info_t anchor_info;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, 1, &request, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE, &anchor,
      &anchor_info, &result));
  MAKE_FRONTIER(death, 1, E(TestQueueAxis(0), 10));
  iree_hal_pool_release_reservations(pool_, kBlockCount, reservations.data(),
                                     death);

  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, kRequestCount, requests.data(), nullptr,
      IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER |
          IREE_HAL_POOL_RESERVE_FLAG_DISALLOW_GROWTH,
      reservations.data(), infos.data(), &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  EXPECT_EQ(query_count, kBlockCount);
  for (iree_host_size_t i = 0; i < kRequestCount; ++i) {
    ASSERT_NE(infos[i].reuse_frontier, nullptr);
    EXPECT_EQ(infos[i].reuse_frontier->entries[0].epoch, 10u);
  }
  iree_hal_pool_release_reservations(pool_, kRequestCount, reservations.data(),
                                     death);
  ReleaseOneReservation(pool_, &anchor, nullptr);
}

// Holds one completion query while another caller uses the same real backing.
// Subsequent queries return the deliberately stale completed epoch of 10.
class PausedEpochQuery {
 public:
  iree_hal_pool_epoch_query_t query() { return {Query, this}; }
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
  static bool Query(void* user_data, iree_async_axis_t axis, uint64_t epoch) {
    auto* self = static_cast<PausedEpochQuery*>(user_data);
    std::unique_lock<std::mutex> lock(self->mutex_);
    if (!self->entered_) {
      self->entered_ = true;
      self->condition_.notify_all();
      self->condition_.wait(lock, [&] { return self->resumed_; });
    }
    return epoch <= 10;
  }
  // Protects the query-entry and continuation handoff.
  std::mutex mutex_;
  // Publishes the two explicit handoff events.
  std::condition_variable condition_;
  // True once the selecting thread has entered its first completion query.
  bool entered_ = false;
  // True once the competing allocation has returned its changed frontier.
  bool resumed_ = false;
};

TEST_F(FixedBlockPoolTest, ConcurrentUseRevalidatesTheEntireBatch) {
  iree_hal_pool_release(pool_);
  iree_hal_pool_release(backing_pool_);
  backing_pool_ = nullptr;
  auto options = DefaultOptions();
  options.blocks_per_slab = 3;
  PausedEpochQuery query;
  IREE_ASSERT_OK(CreateGrowingPool(options, slab_provider_, notification_,
                                   test_frontier_tracker(), query.query(),
                                   allocator_, &pool_));
  auto request = MakeReservationRequest(64, 16);
  request.params.usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  const iree_hal_pool_reservation_request_t requests[2] = {request, request};
  iree_hal_pool_reservation_t reservations[2];
  iree_hal_pool_acquire_info_t infos[2];
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, 2, requests, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      reservations, infos, &result));
  iree_hal_pool_reservation_t anchor;
  iree_hal_pool_acquire_info_t anchor_info;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, 1, &request, nullptr, IREE_HAL_POOL_RESERVE_FLAG_NONE, &anchor,
      &anchor_info, &result));
  MAKE_FRONTIER(old_death, 1, E(TestQueueAxis(0), 10));
  iree_hal_pool_release_reservations(pool_, 2, reservations, old_death);

  iree::Status selection_status;
  std::thread selector([&] {
    selection_status = iree_hal_pool_acquire_reservations(
        pool_, 2, requests, nullptr,
        IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER, reservations, infos,
        &result);
  });
  query.AwaitEntry();

  // This caller covers the old prerequisite, so it can run while the other
  // caller is paused. Its returned history invalidates the earlier snapshot.
  auto use_while_paused = [&]() -> iree_status_t {
    iree_hal_pool_reservation_t competing;
    iree_hal_pool_acquire_info_t competing_info;
    iree_hal_pool_acquire_result_t competing_result;
    IREE_RETURN_IF_ERROR(iree_hal_pool_acquire_reservations(
        pool_, 1, &request, old_death, IREE_HAL_POOL_RESERVE_FLAG_NONE,
        &competing, &competing_info, &competing_result));
    EXPECT_EQ(competing_result, IREE_HAL_POOL_ACQUIRE_OK);
    EXPECT_EQ(competing.offset, 0u);
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool_, &stats);
    EXPECT_EQ(stats.bytes_reserved, 2 * options.block_size);
    iree_hal_buffer_t* buffer = nullptr;
    iree_status_t status =
        MaterializeOneReservation(pool_, request.params, &competing,
                                  IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &buffer);
    const uint32_t expected[16] = {0x12345678};
    uint32_t actual[16] = {};
    if (iree_status_is_ok(status)) {
      status = iree_hal_buffer_map_write(buffer, 0, expected, sizeof(expected));
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_buffer_map_read(buffer, 0, actual, sizeof(actual));
    }
    if (iree_status_is_ok(status)) {
      EXPECT_EQ(memcmp(expected, actual, sizeof(actual)), 0);
    }
    iree_hal_buffer_release(buffer);
    MAKE_FRONTIER(new_death, 1, E(TestQueueAxis(0), 20));
    ReleaseOneReservation(pool_, &competing, new_death);
    return status;
  };
  IREE_EXPECT_OK(use_while_paused());
  query.Resume();
  selector.join();
  IREE_ASSERT_OK(selection_status);
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  ASSERT_NE(infos[0].reuse_frontier, nullptr);
  ASSERT_NE(infos[1].reuse_frontier, nullptr);
  EXPECT_EQ(infos[0].reuse_frontier->entries[0].epoch, 10u);
  EXPECT_EQ(infos[1].reuse_frontier->entries[0].epoch, 20u);
  EXPECT_EQ(infos[0].result, IREE_HAL_POOL_ACQUIRE_OK);
  EXPECT_EQ(infos[1].result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);

  // The selected pending block is the block written by the competing caller.
  // All CPU accesses above have completed; inspect its bytes through the new
  // reservation and return each block with its own exact history.
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(
      MaterializeOneReservation(pool_, request.params, &reservations[1],
                                IREE_HAL_POOL_MATERIALIZE_FLAG_NONE, &buffer));
  uint32_t actual[16] = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(buffer, 0, actual, sizeof(actual)));
  EXPECT_EQ(actual[0], 0x12345678u);
  for (iree_host_size_t i = 1; i < IREE_ARRAYSIZE(actual); ++i) {
    EXPECT_EQ(actual[i], 0u);
  }
  iree_hal_buffer_release(buffer);
  for (iree_host_size_t i = 0; i < 2; ++i) {
    ReleaseOneReservation(pool_, &reservations[i], infos[i].reuse_frontier);
  }
  ReleaseOneReservation(pool_, &anchor, nullptr);
}

TEST_F(FixedBlockPoolTest, AlignmentStrengthensBackingWithinBlockStride) {
  // Native CPU storage can be aligned on demand. A fixed-block pool must still
  // reject alignment beyond its 256-byte block stride.
  iree_hal_pool_reservation_t reservation;
  memset(&reservation, 0xA5, sizeof(reservation));
  const auto original = reservation;
  iree_hal_pool_acquire_info_t info;
  iree_hal_pool_acquire_result_t result;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        AcquireOneReservation(pool_, 128, 512, nullptr,
                                              IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                              &reservation, &info, &result));
  EXPECT_EQ(memcmp(&reservation, &original, sizeof(original)), 0);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reserve_count, 0u);
  EXPECT_EQ(stats.bytes_reserved, 0u);

  auto request = MakeReservationRequest(128, 256);
  request.params.usage |= IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, request.params, request.allocation_size, iree_infinite_timeout(),
      &buffer));
  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 128, &mapping));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(mapping.contents.data) % 256, 0u);
  memset(mapping.contents.data, 0x6B, 128);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  uint8_t actual[128] = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_read(buffer, 0, actual, sizeof(actual)));
  for (uint8_t value : actual) {
    EXPECT_EQ(value, 0x6B);
  }
  iree_hal_buffer_release(buffer);
}

TEST_F(FixedBlockPoolTest, MaterializationTransactionTransfersAllReservations) {
  const iree_hal_pool_reservation_request_t requests[2] = {
      MakeReservationRequest(64, 16),
      MakeReservationRequest(128, 16),
  };
  iree_hal_pool_reservation_t reservations[2];
  iree_hal_pool_acquire_info_t infos[2];
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(iree_hal_pool_acquire_reservations(
      pool_, IREE_ARRAYSIZE(requests), requests,
      /*requester_frontier=*/NULL, IREE_HAL_POOL_RESERVE_FLAG_NONE,
      reservations, infos, &result));

  iree_hal_buffer_t* buffers[2];
  IREE_ASSERT_OK(iree_hal_pool_materialize_reservations(
      pool_, IREE_ARRAYSIZE(reservations), requests, reservations,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, buffers));
  EXPECT_EQ(iree_hal_buffer_byte_length(buffers[0]), 64u);
  EXPECT_EQ(iree_hal_buffer_byte_length(buffers[1]), 128u);

  iree_hal_buffer_release(buffers[0]);
  iree_hal_buffer_release(buffers[1]);
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.release_count, 2u);
}

TEST_F(FixedBlockPoolTest, ReserveReusesDominatedFrontier) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 128, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  MAKE_FRONTIER(death, 1, E(TestQueueAxis(0), 10));
  ReleaseOneReservation(pool_, &reservation, death);

  MAKE_FRONTIER(requester, 1, E(TestQueueAxis(0), 10));
  IREE_ASSERT_OK(AcquireOneReservation(pool_, 64, 16, requester,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  EXPECT_EQ(reservation.offset, 0u);
  ASSERT_NE(reserve_info.reuse_frontier, nullptr);
  EXPECT_EQ(reserve_info.reuse_frontier->entry_count, 1u);
  EXPECT_EQ(reserve_info.reuse_frontier->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(reserve_info.reuse_frontier->entries[0].epoch, 10u);
  EXPECT_EQ(reserve_info.flags, IREE_HAL_POOL_ACQUIRE_FLAG_NONE);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reuse_count, 1u);
  EXPECT_EQ(stats.fresh_count, 1u);

  // Returning an unused reservation preserves the prerequisite even though
  // this requester's frontier covered it.
  ReleaseOneReservation(pool_, &reservation, reserve_info.reuse_frontier);
  IREE_ASSERT_OK(AcquireOneReservation(pool_, 64, 16, requester,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  ASSERT_NE(reserve_info.reuse_frontier, nullptr);
  EXPECT_EQ(reserve_info.reuse_frontier->entries[0].epoch, 10u);
  ReleaseOneReservation(pool_, &reservation, NULL);
}

TEST_F(FixedBlockPoolTest, ReserveSkipsStaleBlockAndReturnsLaterFreshBlock) {
  iree_hal_pool_reservation_t stale_block;
  iree_hal_pool_reservation_t fresh_block;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 64, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &stale_block, &reserve_info, &result));
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 64, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &fresh_block, &reserve_info, &result));
  EXPECT_EQ(stale_block.offset, 0u);
  EXPECT_EQ(fresh_block.offset, 256u);

  MAKE_FRONTIER(death, 1, E(TestQueueAxis(0), 20));
  ReleaseOneReservation(pool_, &stale_block, death);
  ReleaseOneReservation(pool_, &fresh_block, NULL);

  MAKE_FRONTIER(requester, 1, E(TestQueueAxis(0), 10));
  iree_hal_pool_reservation_t reservation;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 64, 16, requester, IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
      &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(reservation.offset, 256u);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reuse_miss_count, 1u);
  EXPECT_EQ(stats.wait_count, 0u);

  ReleaseOneReservation(pool_, &reservation, NULL);

  MAKE_FRONTIER(dominating_requester, 1, E(TestQueueAxis(0), 20));
  IREE_ASSERT_OK(AcquireOneReservation(pool_, 64, 16, dominating_requester,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK);
  EXPECT_EQ(reservation.offset, 0u);

  ReleaseOneReservation(pool_, &reservation, NULL);
}

TEST_F(FixedBlockPoolTest, ReserveCanReturnStaleBlockWhenWaitAllowed) {
  iree_hal_pool_release(pool_);
  iree_hal_pool_release(backing_pool_);
  backing_pool_ = nullptr;
  pool_ = NULL;
  iree_hal_fixed_block_pool_options_t options = DefaultOptions();
  options.blocks_per_slab = 1;
  IREE_ASSERT_OK(CreateFinitePool(options, slab_provider_, notification_,
                                  test_frontier_tracker(), allocator_,
                                  &backing_pool_, &pool_));

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 64, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  MAKE_FRONTIER(death, 1, E(TestQueueAxis(0), 20));
  ReleaseOneReservation(pool_, &reservation, death);

  MAKE_FRONTIER(requester, 1, E(TestQueueAxis(0), 10));
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 64, 16, requester, IREE_HAL_POOL_RESERVE_FLAG_ALLOW_WAIT_FRONTIER,
      &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_NEEDS_WAIT);
  ASSERT_NE(reserve_info.reuse_frontier, nullptr);
  EXPECT_EQ(reserve_info.reuse_frontier->entry_count, 1u);
  EXPECT_EQ(reserve_info.reuse_frontier->entries[0].axis, TestQueueAxis(0));
  EXPECT_EQ(reserve_info.reuse_frontier->entries[0].epoch, 20u);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reuse_miss_count, 1u);
  EXPECT_EQ(stats.wait_count, 1u);

  ReleaseOneReservation(pool_, &reservation, reserve_info.reuse_frontier);
}

TEST(FixedBlockPool, ReserveRejectedTaintRemainsRejected) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(/*min_alignment=*/0,
                                                   allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_fixed_block_pool_options_t options = DefaultOptions();
  options.blocks_per_slab = 1;
  options.frontier_capacity = 1;

  iree_hal_pool_t* backing_pool = nullptr;
  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(CreateFinitePool(options, slab_provider, notification,
                                  test_frontier_tracker(), allocator,
                                  &backing_pool, &pool));

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 64, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);

  MAKE_FRONTIER(oversized_death, 2, E(TestQueueAxis(0), 10),
                E(TestQueueAxis(1), 20));
  ReleaseOneReservation(pool, &reservation, oversized_death);

  MAKE_FRONTIER(requester, 2, E(TestQueueAxis(0), 100),
                E(TestQueueAxis(1), 100));
  IREE_ASSERT_OK(AcquireOneReservation(pool, 64, 16, requester,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);

  IREE_ASSERT_OK(AcquireOneReservation(pool, 64, 16, requester,
                                       IREE_HAL_POOL_RESERVE_FLAG_NONE,
                                       &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_EXHAUSTED);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reuse_miss_count, 2u);
  EXPECT_EQ(stats.exhausted_count, 2u);

  iree_hal_pool_release(pool);
  iree_hal_pool_release(backing_pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST_F(FixedBlockPoolTest, WrapReservationCreatesBuffer) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 128, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(MaterializeOneReservation(
      pool_, params, &reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(iree_hal_buffer_allocation_size(buffer), 1024u);
  EXPECT_EQ(iree_hal_buffer_byte_length(buffer), 128u);

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 128, &mapping));
  memset(mapping.contents.data, 0x5A, 128);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 1u);
  EXPECT_EQ(stats.bytes_reserved, 256u);

  iree_hal_buffer_release(buffer);

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.release_count, 1u);
}

TEST_F(FixedBlockPoolTest, PoolSetRoutesByUserVisibleRange) {
  iree_hal_pool_set_t pool_set;
  IREE_ASSERT_OK(iree_hal_pool_set_initialize(/*initial_capacity=*/1,
                                              allocator_, &pool_set));
  IREE_ASSERT_OK(iree_hal_pool_set_register(&pool_set, 10, pool_));

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  EXPECT_EQ(pool_, iree_hal_pool_set_select(&pool_set, params, 128));
  EXPECT_EQ(nullptr, iree_hal_pool_set_select(&pool_set, params, 257));

  iree_hal_pool_set_deinitialize(&pool_set);
}

TEST_F(FixedBlockPoolTest, BorrowedMaterializationDoesNotReleaseReservation) {
  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t acquire_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 128, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &acquire_info, &result));

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(MaterializeOneReservation(pool_, params, &reservation,
                                           IREE_HAL_POOL_MATERIALIZE_FLAG_NONE,
                                           &buffer));

  iree_hal_buffer_release(buffer);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 1u);
  EXPECT_EQ(stats.release_count, 0u);

  ReleaseOneReservation(pool_, &reservation,
                        /*death_frontier=*/NULL);

  iree_hal_pool_query_stats(pool_, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);
  EXPECT_EQ(stats.release_count, 1u);
}

TEST(FixedBlockPool, UsesProviderHooks) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_pool_t* backing_pool = nullptr;
  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(CreateFinitePool(DefaultOptions(), slab_provider, notification,
                                  test_frontier_tracker(), allocator,
                                  &backing_pool, &pool));

  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool, &capabilities);
  EXPECT_EQ(capabilities.atomic_operations.device_scope_32,
            IREE_HAL_ATOMIC_OPERATION_FLAG_STORE);
  EXPECT_EQ(capabilities.atomic_operations.system_scope_64,
            IREE_HAL_ATOMIC_OPERATION_FLAG_RMW_OR);

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 128, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(MaterializeOneReservation(
      pool, params, &reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &buffer));

  // Policy trim preserves live buffers; native trim is an explicit source
  // operation.
  auto* provider = (iree_hal_test_opaque_slab_provider_t*)slab_provider;
  const iree_hal_pool_trim_flags_t trim_flags[] = {
      IREE_HAL_POOL_TRIM_FLAG_NONE,
      IREE_HAL_POOL_TRIM_FLAG_EXCESS,
      IREE_HAL_POOL_TRIM_FLAG_ALL,
      IREE_HAL_POOL_TRIM_FLAG_ALL | IREE_HAL_POOL_TRIM_FLAG_EXCESS,
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(trim_flags); ++i) {
    iree_hal_pool_trim(pool, trim_flags[i], /*min_bytes_to_keep=*/0);
    EXPECT_EQ(provider->trim_count, i);
    iree_hal_pool_trim(backing_pool, trim_flags[i], 0);
    WaitForMaintenance();
    EXPECT_EQ(provider->trim_count, i + 1);
    EXPECT_EQ(provider->last_trim_flags, trim_flags[i]);
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool, &stats);
    EXPECT_EQ(stats.reservation_count, 1u);
  }

  iree_hal_buffer_mapping_t mapping;
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      buffer, IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 0, 128, &mapping));
  memset(mapping.contents.data, 0x3C, 128);
  EXPECT_EQ(((uint8_t*)mapping.contents.data)[127], 0x3C);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));

  EXPECT_EQ(
      1,
      iree_atomic_load(
          &((iree_hal_test_opaque_slab_provider_t*)slab_provider)->wrap_count,
          iree_memory_order_relaxed));

  iree_hal_buffer_release(buffer);
  iree_hal_pool_release(pool);
  iree_hal_pool_release(backing_pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(FixedBlockPool, CreateRejectsASANWhenProviderCannotAdviseRanges) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(iree_hal_cpu_slab_provider_create(/*min_alignment=*/0,
                                                   allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_fixed_block_pool_options_t options = DefaultOptions();
  options.asan = ShadowOptions();

  iree_hal_pool_t* backing_pool = nullptr;
  iree_hal_pool_t* pool = NULL;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        CreateFinitePool(options, slab_provider, notification,
                                         test_frontier_tracker(), allocator,
                                         &backing_pool, &pool));
  EXPECT_EQ(pool, nullptr);
  iree_hal_pool_release(backing_pool);

  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST(FixedBlockPool, ASANAdvisesBackingBlockAndExposesUserRange) {
  iree_allocator_t allocator = iree_allocator_system();
  iree_hal_slab_provider_t* slab_provider = NULL;
  IREE_ASSERT_OK(
      iree_hal_test_opaque_slab_provider_create(allocator, &slab_provider));
  iree_async_notification_t* notification = NULL;
  IREE_ASSERT_OK(iree_async_notification_create(
      test_proactor(), IREE_ASYNC_NOTIFICATION_FLAG_NONE, &notification));

  iree_hal_fixed_block_pool_options_t options = DefaultOptions();
  options.block_size = 64;
  options.blocks_per_slab = 2;
  options.asan = ShadowOptions();

  iree_hal_pool_t* backing_pool = nullptr;
  iree_hal_pool_t* pool = NULL;
  IREE_ASSERT_OK(CreateFinitePool(options, slab_provider, notification,
                                  test_frontier_tracker(), allocator,
                                  &backing_pool, &pool));

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 13, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));

  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  EXPECT_EQ(reservation.offset, 64u);
  EXPECT_EQ(reservation.byte_length, 13u);
  EXPECT_NE(reservation.block_handle, 0u);

  iree_hal_test_opaque_slab_provider_t* provider =
      (iree_hal_test_opaque_slab_provider_t*)slab_provider;
  EXPECT_TRUE(iree_hal_pool_requires_asan_advice(pool));
  EXPECT_EQ(0, iree_atomic_load(&provider->asan_advice_count,
                                iree_memory_order_relaxed));
  iree_hal_pool_advise_asan_reservations(
      pool, 1, &reservation, IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED);
  EXPECT_EQ(iree_atomic_load(&provider->asan_allocated_count,
                             iree_memory_order_relaxed),
            1);
  EXPECT_EQ(provider->last_asan_backing_offset, 0u);
  EXPECT_EQ(provider->last_asan_layout.backing_length, 192u);
  EXPECT_EQ(provider->last_asan_layout.user_offset, 64u);
  EXPECT_EQ(provider->last_asan_layout.user_length, 13u);
  EXPECT_EQ(provider->last_asan_layout.right_redzone_length, 115u);

  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.bytes_reserved, 192u);
  EXPECT_EQ(stats.bytes_free, 192u);

  iree_hal_buffer_params_t params = {0};
  params.usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
  params.type = IREE_HAL_MEMORY_TYPE_HOST_LOCAL;
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;

  iree_hal_buffer_t* buffer = NULL;
  IREE_ASSERT_OK(MaterializeOneReservation(
      pool, params, &reservation,
      IREE_HAL_POOL_MATERIALIZE_FLAG_TRANSFER_RESERVATION_OWNERSHIP, &buffer));
  ASSERT_NE(buffer, nullptr);
  EXPECT_EQ(iree_hal_buffer_allocation_size(buffer), 384u);
  EXPECT_EQ(iree_hal_buffer_byte_length(buffer), 13u);

  iree_hal_buffer_release(buffer);

  EXPECT_EQ(iree_atomic_load(&provider->asan_released_count,
                             iree_memory_order_relaxed),
            1);
  const int32_t first_release_sequence = provider->last_asan_released_sequence;
  EXPECT_LT(provider->last_asan_allocated_sequence, first_release_sequence);
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.bytes_reserved, 0u);
  EXPECT_EQ(stats.reservation_count, 0u);

  IREE_ASSERT_OK(AcquireOneReservation(
      pool, 13, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OK_FRESH);
  iree_hal_pool_advise_asan_reservations(
      pool, 1, &reservation, IREE_HAL_ASAN_RANGE_ADVICE_FLAG_ALLOCATED);
  EXPECT_EQ(iree_atomic_load(&provider->asan_allocated_count,
                             iree_memory_order_relaxed),
            2);
  EXPECT_LT(first_release_sequence, provider->last_asan_allocated_sequence);

  iree_hal_pool_advise_asan_reservations(
      pool, 1, &reservation, IREE_HAL_ASAN_RANGE_ADVICE_FLAG_RELEASED);
  ReleaseOneReservation(pool, &reservation,
                        /*death_frontier=*/NULL);
  EXPECT_EQ(iree_atomic_load(&provider->asan_released_count,
                             iree_memory_order_relaxed),
            2);
  EXPECT_LT(provider->last_asan_allocated_sequence,
            provider->last_asan_released_sequence);

  iree_hal_pool_release(pool);
  iree_hal_pool_release(backing_pool);
  iree_async_notification_release(notification);
  iree_hal_slab_provider_release(slab_provider);
}

TEST_F(FixedBlockPoolTest, QueryCapabilitiesAndBudget) {
  iree_hal_pool_capabilities_t capabilities;
  iree_hal_pool_query_capabilities(pool_, &capabilities);
  EXPECT_TRUE(iree_all_bits_set(capabilities.memory_type,
                                IREE_HAL_MEMORY_TYPE_HOST_LOCAL));
  EXPECT_TRUE(iree_all_bits_set(capabilities.supported_usage,
                                IREE_HAL_BUFFER_USAGE_TRANSFER));
  EXPECT_EQ(capabilities.min_allocation_size, 1u);
  EXPECT_EQ(capabilities.max_allocation_size, 256u);

  iree_hal_pool_release(pool_);
  iree_hal_pool_release(backing_pool_);
  backing_pool_ = nullptr;
  pool_ = NULL;

  iree_hal_fixed_block_pool_options_t options = DefaultOptions();
  options.budget_limit = 255;
  IREE_ASSERT_OK(CreateFinitePool(options, slab_provider_, notification_,
                                  test_frontier_tracker(), allocator_,
                                  &backing_pool_, &pool_));

  iree_hal_pool_reservation_t reservation;
  iree_hal_pool_acquire_info_t reserve_info;
  iree_hal_pool_acquire_result_t result;
  IREE_ASSERT_OK(AcquireOneReservation(
      pool_, 64, 16, /*requester_frontier=*/NULL,
      IREE_HAL_POOL_RESERVE_FLAG_NONE, &reservation, &reserve_info, &result));
  EXPECT_EQ(result, IREE_HAL_POOL_ACQUIRE_OVER_BUDGET);
}

}  // namespace
